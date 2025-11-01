# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
#
# Code modified to follow the design patterns of EMO for better integration as a backbone.
# The forward method is now stabilized to always return a list of feature maps.

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
import sys
import os

# =================================================================================================
# Helper Functions and Classes (Unaltered)
# =================================================================================================

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        # This is a custom implementation path, ensure it's available in your environment
        try:
            sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
        except ImportError:
            print("Warning: Custom large kernel implementation not found. Falling back to nn.Conv2d.")
            return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                             padding=padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

use_sync_bn = False

def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True

def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel, small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2, groups=groups, dilation=1)
    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out
    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b
    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels, out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')

class ConvFFN(nn.Module):
    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = get_bn(in_channels)
        self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.nonlinear = nn.GELU()
    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)

class RepLKBlock(nn.Module):
    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel, drop_path, small_kernel_merged=False):
        super().__init__()
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                                  stride=1, groups=dw_channels, small_kernel=small_kernel, small_kernel_merged=small_kernel_merged)
        self.lk_nonlinear = nn.ReLU()
        self.prelkb_bn = get_bn(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)

class RepLKNetStage(nn.Module):
    def __init__(self, channels, num_blocks, stage_lk_size, drop_path, small_kernel, dw_ratio=1, ffn_ratio=4,
                 use_checkpoint=False, small_kernel_merged=False, norm_intermediate_features=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        blks = []
        for i in range(num_blocks):
            block_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            replk_block = RepLKBlock(in_channels=channels, dw_channels=int(channels * dw_ratio), block_lk_size=stage_lk_size,
                                     small_kernel=small_kernel, drop_path=block_drop_path, small_kernel_merged=small_kernel_merged)
            convffn_block = ConvFFN(in_channels=channels, internal_channels=int(channels * ffn_ratio), out_channels=channels,
                                    drop_path=block_drop_path)
            blks.append(replk_block)
            blks.append(convffn_block)
        self.blocks = nn.ModuleList(blks)
        if norm_intermediate_features:
            self.norm = get_bn(channels)
        else:
            self.norm = nn.Identity()
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

# =================================================================================================
# Main RepLKNet Class (Modified for Robust Backbone Usage)
# =================================================================================================

class RepLKNet(nn.Module):
    def __init__(self, large_kernel_sizes, layers, channels, drop_path_rate, small_kernel,
                 dw_ratio=1, ffn_ratio=4, in_channels=3,
                 # `num_classes` and `out_indices` are now primarily for clarity and `width_list` calculation.
                 # The forward pass is fixed to feature extraction mode.
                 num_classes=None, out_indices=(1, 2, 3),
                 use_checkpoint=False, small_kernel_merged=False, use_sync_bn=True,
                 norm_intermediate_features=False):
        super().__init__()

        if use_sync_bn:
            enable_sync_bn()

        self.out_indices = out_indices
        base_width = channels[0]
        self.use_checkpoint = use_checkpoint
        self.num_stages = len(layers)

        # Stem layers
        self.stem = nn.ModuleList([
            conv_bn_relu(in_channels=in_channels, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=1, padding=1, groups=base_width),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=1, stride=1, padding=0, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=base_width)])
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        
        # Build stages and transitions
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            layer = RepLKNetStage(channels=channels[stage_idx], num_blocks=layers[stage_idx],
                                  stage_lk_size=large_kernel_sizes[stage_idx],
                                  drop_path=dpr[sum(layers[:stage_idx]):sum(layers[:stage_idx + 1])],
                                  small_kernel=small_kernel, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
                                  use_checkpoint=use_checkpoint, small_kernel_merged=small_kernel_merged,
                                  norm_intermediate_features=norm_intermediate_features)
            self.stages.append(layer)
            if stage_idx < len(layers) - 1:
                transition = nn.Sequential(
                    conv_bn_relu(channels[stage_idx], channels[stage_idx + 1], 1, 1, 0, groups=1),
                    conv_bn_relu(channels[stage_idx + 1], channels[stage_idx + 1], 3, stride=2, padding=1, groups=channels[stage_idx + 1]))
                self.transitions.append(transition)

        # Classification head is removed from the main build process to avoid conflicts.
        # The model is now solely a feature extractor.
        
        # Add self.width_list, inspired by EMO backbone
        # This part dynamically determines the output channel dimensions.
        self.width_list = []
        # Temporarily switch to eval mode
        self_training = self.training
        self.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 224, 224)
            try:
                # The forward pass will return a list of features
                features = self.forward(dummy_input)
                # Store the channel dimension of each feature map that we intend to use
                if self.out_indices:
                    self.width_list = [features[i].size(1) for i in self.out_indices]
                else: # if out_indices is not specified, take the last 3 stages by default
                    self.width_list = [f.size(1) for f in features[1:]]
            except Exception as e:
                print(f"Warning: Could not compute width_list due to: {e}")
        # Restore original training mode
        self.train(self_training)

    def forward(self, x):
        # This forward pass is now designed specifically for feature extraction.
        # It will ALWAYS return a list of feature maps, one from each stage.
        
        # Stem
        x = self.stem[0](x)
        for stem_layer in self.stem[1:]:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(stem_layer, x)
            else:
                x = stem_layer(x)
        
        # Stages
        feature_maps = []
        for stage_idx in range(self.num_stages):
            x = self.stages[stage_idx](x)
            
            # Append the output of the current stage
            feature_maps.append(x)
            
            if stage_idx < self.num_stages - 1:
                x = self.transitions[stage_idx](x)
                
        # The output is always a list of 4 feature maps.
        # The downstream consumer (e.g., YOLO neck) will select the ones it needs.
        return feature_maps

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

# =================================================================================================
# Factory Functions (Using a consistent style)
# =================================================================================================

def RepLKNet31B(drop_path_rate=0.3, **kwargs):
    return RepLKNet(large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], channels=[128,256,512,1024],
                    drop_path_rate=drop_path_rate, small_kernel=5, **kwargs)

def RepLKNet31L(drop_path_rate=0.3, **kwargs):
    return RepLKNet(large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], channels=[192,384,768,1536],
                    drop_path_rate=drop_path_rate, small_kernel=5, **kwargs)

def RepLKNetXL(drop_path_rate=0.3, **kwargs):
    return RepLKNet(large_kernel_sizes=[27,27,27,13], layers=[2,2,18,2], channels=[256,512,1024,2048],
                    drop_path_rate=drop_path_rate, small_kernel=None, dw_ratio=1.5, **kwargs)


# =================================================================================================
# Test Script
# =================================================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing RepLKNet in Feature Extraction (Backbone) Mode")
    print("-" * 60)
    
    # We now instantiate the model without num_classes, as it's designed as a backbone.
    # We can still pass `out_indices` to control which features are used to build `width_list`.
    # Common for YOLO-like models is to use the last 3 stages.
    output_indices_for_yolo = (1, 2, 3) 
    model_backbone = RepLKNet_31B(out_indices=output_indices_for_yolo, use_checkpoint=False)
    model_backbone.eval()
    
    print(f"Model Class: {model_backbone.__class__.__name__}")
    x = torch.randn(2, 3, 224, 224)
    
    # Test the forward pass
    output_features = model_backbone(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output type (backbone): {type(output_features)}")
    print(f"Number of output feature maps: {len(output_features)}")
    print("Shapes of ALL output feature maps (forward now always returns 4):")
    for i, out_feat in enumerate(output_features):
        print(f"  - Feature from Stage {i}: {out_feat.shape}")
    
    # Check the automatically generated width_list. It should match the channels of the selected out_indices.
    print(f"\nSpecified out_indices for width_list: {model_backbone.out_indices}")
    print(f"Automatically generated width_list: {model_backbone.width_list}")
    expected_width_list = [output_features[i].shape[1] for i in output_indices_for_yolo]
    print(f"Expected width_list based on out_indices: {expected_width_list}")
    assert model_backbone.width_list == expected_width_list

    print("\nTest passed: The model consistently returns a list of features.")