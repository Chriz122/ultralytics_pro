# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
# Add WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension into your PYTHONPATH by the following commands:
# sys.path.append('/home/lili/CC/SLaK/cutlass/examples/19_large_depthwise_conv2d_torch_extension')

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
# from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
from typing import List, Tuple, Union # Added List, Tuple, Union

use_sync_bn = True # You might want to set this to False for simpler local testing if not using DDP

# def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
#     return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    calculated_paddings = None
    if isinstance(kernel_size, tuple):
        calculated_paddings = (kernel_size[0] // 2, kernel_size[1] // 2)
    elif isinstance(kernel_size, int):
        calculated_paddings = kernel_size // 2
    
    # Use explicitly provided padding if it's different from the default calculation or if calculated_paddings is None
    if padding is not None:
        # Check if padding was explicitly set to something different than standard 'same' padding
        is_standard_padding = False
        if calculated_paddings is not None:
            if isinstance(padding, int) and isinstance(kernel_size, int):
                is_standard_padding = (padding == kernel_size // 2)
            elif isinstance(padding, tuple) and isinstance(kernel_size, tuple):
                is_standard_padding = (padding == (kernel_size[0] // 2, kernel_size[1] // 2))
        
        if not is_standard_padding or calculated_paddings is None: # If padding is custom or couldn't be calculated
            actual_padding = padding
        else: # Padding was provided but it matches the standard calculation
            actual_padding = calculated_paddings
    elif calculated_paddings is not None: # Padding not provided, use calculated
        actual_padding = calculated_paddings
    else:
        raise ValueError(f"Cannot determine padding for kernel_size {kernel_size} and padding {padding}")

    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, actual_padding, dilation, groups, bias)


def get_bn(channels):
    if use_sync_bn:
        try:
            # Check if distributed process group is initialized
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return nn.SyncBatchNorm(channels)
            else:
                # print("Warning: Distributed process group not initialized. Falling back to BatchNorm2d from SyncBatchNorm.")
                return nn.BatchNorm2d(channels)
        except Exception: # Catch any other SyncBatchNorm error
            # print("Warning: SyncBatchNorm failed to initialize. Falling back to BatchNorm2d.")
            return nn.BatchNorm2d(channels)
    else:
        return nn.BatchNorm2d(channels)

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True):
    # Determine padding if None
    if padding is None:
        if isinstance(kernel_size, tuple):
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif isinstance(kernel_size, int):
            padding = kernel_size // 2
        else: # Should not happen if kernel_size is int or tuple
            padding = 0 

    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation, bn=bn)
    result.add_module('nonlinear', nn.ReLU())
    return result

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True):
    # Determine padding if None
    if padding is None:
        if isinstance(kernel_size, tuple):
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif isinstance(kernel_size, int):
            padding = kernel_size // 2
        else: # Should not happen
            padding = 0
             
    result = nn.Sequential()
    # Bias is False if BN is used, True if no BN
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=not bn))

    if bn:
        result.add_module('bn', get_bn(out_channels))
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

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel, # This is an int or None
                 small_kernel_merged=False, Decom=False, bn=True):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size # This is the large kernel size (int)
        self.small_kernel = small_kernel # This is the small kernel size (int) or None
        self.Decom = Decom
        self.bn = bn # Store bn flag

        # Padding for the main large kernel (if not Decom) or the merged reparam kernel
        padding = self.kernel_size // 2
        
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            if self.Decom:
                # In Decom mode, 'kernel_size' is the large dimension, 'small_kernel' is the rank (smaller dimension)
                # Kernel shapes are (kernel_size, small_kernel) and (small_kernel, kernel_size)
                # These are passed as tuples to conv_bn
                if self.small_kernel is None or not isinstance(self.small_kernel, int):
                    raise ValueError("For Decom=True, small_kernel must be an integer representing the decomposition rank.")

                lora_k1_shape = (self.kernel_size, self.small_kernel)
                lora_k2_shape = (self.small_kernel, self.kernel_size)
                
                # Padding for decomposed convs should also be tuples
                padding_lora1 = (lora_k1_shape[0]//2, lora_k1_shape[1]//2)
                padding_lora2 = (lora_k2_shape[0]//2, lora_k2_shape[1]//2)

                self.LoRA1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=lora_k1_shape,
                                      stride=stride, padding=padding_lora1, dilation=1, groups=groups, bn=self.bn)
                self.LoRA2 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=lora_k2_shape,
                                     stride=stride, padding=padding_lora2, dilation=1, groups=groups, bn=self.bn)
                # Note: The original SLaK code does not seem to add small_conv path if Decom=True,
                # but if it did, the small_conv would be a regular conv, not part of decomposition.
                # The current forward path for Decom only sums LoRA1 and LoRA2, and then adds small_conv if it exists.
                # This implies small_kernel for Decom is the rank, and a separate small_conv (if any) is handled by `self.small_kernel`
                # when constructing the Block. This needs careful handling if small_kernel has dual meaning.
                # For now, assume if Decom, small_kernel is rank. A separate actual small kernel branch is not typical with Decom.
                # The current Block structure passes kernel_size=(large_k, small_k_for_branch)
                # and ReparamLargeKernelConv takes kernel_size=large_k, small_kernel=small_k_for_branch.
                # If Decom is true, small_k_for_branch is used as rank for LoRA.
                # Let's stick to the original SLaK paper's interpretation of Decom as two sequential convs (depthwise separable style)
                # or LoRA-style parallel branches. The provided code uses parallel branches.

            else: # Not Decom and not small_kernel_merged
                self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups, bn=self.bn)

            # This small_conv is a parallel branch, added to lkb_origin or LoRA output
            if self.small_kernel is not None and isinstance(self.small_kernel, int) and self.small_kernel > 0:
                if self.Decom and self.small_kernel >= self.kernel_size :
                    # If Decom, small_kernel is rank, should be smaller than kernel_size typically.
                    # If we intend a separate small_conv branch even with Decom, its kernel size should be < self.kernel_size.
                    # The original code's condition: `small_kernel < kernel_size` for adding small_conv.
                    # Let's assume if Decom=True, self.small_kernel is the rank for LoRA and not for a separate small_conv branch unless explicitly handled.
                    # The current code will add a small_conv if self.small_kernel < self.kernel_size even if Decom.
                    # This might be intended: Decomposed Large Kernel + Standard Small Kernel.
                    pass # Current structure allows this.

                if not self.Decom and self.small_kernel >= self.kernel_size:
                    # print(f"Warning: small_kernel ({self.small_kernel}) >= kernel_size ({self.kernel_size}). Not creating small_conv branch.")
                    pass # Do not create small_conv if it's not smaller
                else:
                     self.small_conv_branch = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=self.small_kernel,
                                                 stride=stride, padding=self.small_kernel//2, groups=groups, dilation=1, bn=self.bn)


    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        elif self.Decom: # Decom means LoRA1 and LoRA2 exist
            out = self.LoRA1(inputs) + self.LoRA2(inputs)
            if hasattr(self, 'small_conv_branch'): # Additive small kernel if it exists
                out += self.small_conv_branch(inputs)
        else: # Not Decom, so lkb_origin must exist
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv_branch'): # Additive small kernel if it exists
                out += self.small_conv_branch(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        if self.Decom: 
            # Merging for Decom=True (LoRA like branches) is more complex and not simply fusing lkb_origin
            # It would involve summing the weights of LoRA1 and LoRA2 appropriately (if possible, depends on padding etc.)
            # and then potentially adding the small_conv_branch.
            # For now, following original behavior of not merging Decom.
            raise NotImplementedError("get_equivalent_kernel_bias is not implemented for Decom=True mode.")

        # This part is for non-Decom, non-merged cases
        if not hasattr(self, 'lkb_origin'):
             raise RuntimeError("lkb_origin not found in get_equivalent_kernel_bias for non-Decom. Object may be in an unexpected state (e.g. already merged or Decom=True).")

        if self.bn :
            if not hasattr(self.lkb_origin, 'bn'):
                raise AttributeError(f"lkb_origin is missing 'bn' submodule when self.bn is True. lkb_origin: {list(self.lkb_origin.named_children())}")
            eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
            
            if hasattr(self, 'small_conv_branch'):
                if not hasattr(self.small_conv_branch, 'bn'):
                     raise AttributeError("small_conv_branch is missing 'bn' submodule when self.bn is True.")
                small_k, small_b = fuse_bn(self.small_conv_branch.conv, self.small_conv_branch.bn)
                eq_b += small_b
                # Ensure self.small_kernel is an int for padding calculation
                if not isinstance(self.small_kernel, int):
                    raise TypeError(f"self.small_kernel must be an int for padding, got {type(self.small_kernel)}")
                eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        else: # No BN
            eq_k = self.lkb_origin.conv.weight
            if self.lkb_origin.conv.bias is None:
                 raise AttributeError("lkb_origin.conv.bias is None when bn=False. It should exist as conv_bn sets bias=True.")
            eq_b = self.lkb_origin.conv.bias

            if hasattr(self, 'small_conv_branch'):
                if self.small_conv_branch.conv.bias is None:
                    raise AttributeError("small_conv_branch.conv.bias is None when bn=False. It should exist.")
                small_k = self.small_conv_branch.conv.weight
                small_b = self.small_conv_branch.conv.bias
                eq_b += small_b
                if not isinstance(self.small_kernel, int):
                    raise TypeError(f"self.small_kernel must be an int for padding, got {type(self.small_kernel)}")
                eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b


    def merge_kernel(self):
        if self.Decom:
            # print("Warning: merge_kernel for Decom=True is a no-op as it's not directly equivalent to a single conv in the same way.")
            return 
        if hasattr(self, 'lkb_reparam'):
            # print("Info: Kernel already merged or in merged mode (lkb_reparam exists).")
            return
        if not hasattr(self, 'lkb_origin'):
            # This case should ideally not be hit if SLaK.merge_reparam_kernels checks correctly.
            print(f"Warning: Attempted to merge kernel but lkb_origin does not exist and not Decom/already_merged. State: Decom={self.Decom}, has_lkb_reparam={hasattr(self,'lkb_reparam')}")
            return


        eq_k, eq_b = self.get_equivalent_kernel_bias()
        
        # Ensure self.lkb_origin.conv exists before accessing its attributes
        if not hasattr(self.lkb_origin, 'conv'):
            raise AttributeError("lkb_origin does not have a 'conv' submodule.")

        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True) # Merged kernel always has bias
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        
        # Delete original components
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv_branch'): # Changed from 'small_conv'
            self.__delattr__('small_conv_branch')
        
        # The problematic line was here, accessing self.lkb_origin after deletion.
        # It was: if self.bn and hasattr(self.lkb_origin, 'bn'): pass
        # This is now removed as lkb_origin is gone, and its bn (if any) is also gone.
        # No further action related to lkb_origin.bn is needed here.


class Block(nn.Module):
    r""" SLaK Block.
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        kernel_config (tuple): Tuple of (large_kernel_size (int), small_kernel_spec (int or None), Decom_this_block (bool)).
                               small_kernel_spec is for the additive branch if not Decom, or rank if Decom.
        bn (bool): Whether to use BatchNorm in ReparamLargeKernelConv.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, 
                 kernel_config: Tuple[int, Union[int, None], bool] = (7, None, False), 
                 bn=True):
        super().__init__()

        large_k_size, small_k_spec, decom_this_block = kernel_config
        
        # In ReparamLargeKernelConv:
        # - kernel_size is always the large kernel dimension.
        # - small_kernel is the spec for the *additive* small kernel branch OR the rank for Decom.
        # The meaning of small_k_spec depends on decom_this_block.
        # If Decom, small_k_spec is treated as the rank.
        # If not Decom, small_k_spec is for the parallel small_conv_branch.
        
        self.large_kernel = ReparamLargeKernelConv(in_channels=dim, out_channels=dim,
                                                   kernel_size=large_k_size, # large kernel dimension
                                                   stride=1, groups=dim, 
                                                   small_kernel=small_k_spec, # rank if Decom, else size of additive small kernel
                                                   small_kernel_merged=False, 
                                                   Decom=decom_this_block, 
                                                   bn=bn)

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1) 
        
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_tensor = x 
        x = self.large_kernel(x)
        x = self.norm(x) 
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x 
        x = input_tensor + self.drop_path(x)
        return x

class SLaK(nn.Module):
    r""" SLaK
    Args:
        kernel_configs (list): List of kernel configurations for each stage. 
                           Each element is a tuple: (large_k_size, small_k_spec, Decom_bool).
                           Example: [(51, 5, False), (49, 5, False), (47, 7, True), (13, 3, False)]
                           small_k_spec: size of additive small kernel, or rank if Decom_bool is True.
        bn_stages (list/bool): Whether to use BatchNorm. List of bools for each stage, or single bool.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1., 
                 # kernel_sizes replaced by kernel_configs
                 kernel_configs: List[Tuple[int, Union[int, None], bool]] = None,
                 width_factor=1.0, 
                 # Decom is now part of kernel_configs
                 bn_stages:Union[bool, List[bool]]=True, # Renamed from bn to bn_stages
                 **kwargs 
                 ):
        super().__init__()
        
        self.num_classes = num_classes 
        self.dims = [int(x*width_factor) for x in dims]
        self.in_chans = in_chans 

        # Default kernel_configs if not provided
        if kernel_configs is None:
            print("Using default kernel_configs for SLaK.")
            # (large_k, small_k_additive_or_rank, Decom_flag)
            kernel_configs = [
                (51, 5, False), # Stage 0: 51x51 large, 5x5 additive, no Decom
                (49, 5, False), # Stage 1: 49x49 large, 5x5 additive, no Decom
                (47, 7, True),  # Stage 2: 47x47 large, rank 7 Decom (no separate additive small kernel implied here by default)
                (13, 3, False)  # Stage 3: 13x13 large, 3x3 additive, no Decom
            ]
        if len(kernel_configs) != 4:
            raise ValueError(f"kernel_configs must be a list of 4 tuples, got {len(kernel_configs)}")
        self.kernel_configs = kernel_configs

        if not isinstance(bn_stages, list): 
            self.bn_stages = [bn_stages] * 4
        else:
            if len(bn_stages) != 4:
                raise ValueError(f"bn_stages list must have 4 elements, got {len(bn_stages)}")
            self.bn_stages = bn_stages


        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(self.in_chans, self.dims[0], kernel_size=4, stride=4),
            LayerNorm(self.dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(self.dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4): # 4 stages
            stage_kernel_config = self.kernel_configs[i] # Tuple: (large_k, small_k_spec, Decom_flag)
            stage_bn = self.bn_stages[i]

            stage = nn.Sequential(
                *[Block(dim=self.dims[i], drop_path=dp_rates[cur + j], 
                        layer_scale_init_value=layer_scale_init_value, 
                        kernel_config=stage_kernel_config, 
                        bn=stage_bn) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(self.dims[-1], eps=1e-6) 
        self.head = nn.Linear(self.dims[-1], num_classes)

        self.apply(self._init_weights)
        if hasattr(self.head, 'weight'):
             self.head.weight.data.mul_(head_init_scale)
        if hasattr(self.head, 'bias') and self.head.bias is not None:
             self.head.bias.data.mul_(head_init_scale)

        self.width_list: List[int] = []
        try:
            first_param = next(self.parameters(), None)
            device = first_param.device if first_param is not None else torch.device("cpu")
            original_mode = self.training
            self.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, self.in_chans, 224, 224)
                # Ensure dummy input is on the same device as the model
                if first_param is not None and first_param.is_cuda:
                    if not dummy_input.is_cuda: dummy_input = dummy_input.to(device)
                elif dummy_input.is_cuda and (first_param is None or not first_param.is_cuda): # model on cpu, input on cuda
                    dummy_input = dummy_input.cpu()

                features = self._forward_features_extract(dummy_input)
                self.width_list = [f.size(1) for f in features]
            self.train(original_mode) 
        except Exception as e:
            print(f"Warning: Could not compute width_list during SLaK init: {e}")
            import traceback
            traceback.print_exc()
            self.width_list = self.dims if hasattr(self, 'dims') else []


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _forward_features_extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self._forward_features_extract(x)

    def get_classification_output(self, x: torch.Tensor) -> torch.Tensor:
        feature_list = self._forward_features_extract(x)
        last_feature_map = feature_list[-1]
        pooled_output = last_feature_map.mean([-2, -1]) 
        output = self.norm(pooled_output) 
        output = self.head(output)
        return output
    
    def merge_reparam_kernels(self):
        print("Attempting to merge kernels for all ReparamLargeKernelConv modules...")
        for stage_idx, stage in enumerate(self.stages):
            for block_idx, block_module in enumerate(stage): # Renamed block to block_module
                # Ensure block_module is a Block instance, not just any nn.Module
                if isinstance(block_module, Block) and hasattr(block_module, 'large_kernel') and \
                   isinstance(block_module.large_kernel, ReparamLargeKernelConv):
                    
                    large_kernel_module = block_module.large_kernel
                    
                    if not large_kernel_module.Decom and not hasattr(large_kernel_module, 'lkb_reparam'):
                        print(f"  Merging kernel in Stage {stage_idx}, Block {block_idx} (bn={large_kernel_module.bn}, decom={large_kernel_module.Decom})...")
                        large_kernel_module.merge_kernel()
                    elif large_kernel_module.Decom:
                        print(f"  Skipping merge for Stage {stage_idx}, Block {block_idx}: Decom=True.")
                    elif hasattr(large_kernel_module, 'lkb_reparam'):
                         print(f"  Skipping merge for Stage {stage_idx}, Block {block_idx}: Kernel already merged (has lkb_reparam).")
                # else:
                #     print(f"  Skipping module in Stage {stage_idx}, item {block_idx}: Not a valid Block or no ReparamLargeKernelConv found.")

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # normalized_shape is expected to be an int (number of channels) for convnext-style LayerNorm
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        # For F.layer_norm, normalized_shape needs to be a tuple/list of the shape of the last dimensions to normalize over
        self.normalized_shape_for_fn = (normalized_shape, ) if isinstance(normalized_shape, int) else normalized_shape
    
    def forward(self, x):
        if self.data_format == "channels_last":
            # Input (N, H, W, C), normalized_shape_for_fn = (C,)
            return F.layer_norm(x, self.normalized_shape_for_fn, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # Input (N, C, H, W), norm over C. weight/bias are (C,).
            # This is the ConvNeXt style LayerNorm.
            # Permute to (N, H, W, C), apply layer_norm, permute back.
            # Or implement manually for (N,C,H,W) where normalization is over C dimension:
            u = x.mean(1, keepdim=True) # Mean over C dimension
            s = (x - u).pow(2).mean(1, keepdim=True) # Variance over C dimension
            x = (x - u) / torch.sqrt(s + self.eps)
            # Weight and bias are of shape (C), need to be reshaped to (1, C, 1, 1) for broadcasting
            x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
            return x

# Factory functions with **kwargs
@register_model
def slak_tiny(pretrained=False, **kwargs): 
    # Default kernel_configs for tiny if not overridden by kwargs
    # (large_k, small_k_additive_or_rank, Decom_flag)
    if 'kernel_configs' not in kwargs:
        kwargs['kernel_configs'] = [ (31, 5, False), (29, 5, False), (27, 5, False), (13, 3, False) ]
    model = SLaK(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

@register_model
def slak_small(pretrained=False, **kwargs):
    if 'kernel_configs' not in kwargs:
        kwargs['kernel_configs'] = [ (41, 5, False), (39, 5, False), (37, 5, False), (13, 3, False) ]
    model = SLaK(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

@register_model
def slak_base(pretrained=False, **kwargs): 
    if 'kernel_configs' not in kwargs:
        # (large_k, small_k_additive_or_rank, Decom_flag)
        kwargs['kernel_configs'] = [ (51, 5, False), (49, 5, False), (47, 7, True), (13, 3, False) ]
    model = SLaK(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

@register_model
def slak_large(pretrained=False, **kwargs): 
    if 'kernel_configs' not in kwargs:
        kwargs['kernel_configs'] = [ (61, 7, False), (59, 7, False), (57, 9, True), (13, 3, False) ]
    model = SLaK(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


# --- Example Usage ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # To disable SyncBatchNorm for local testing if not using DDP
    use_sync_bn = False 
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        use_sync_bn = False
        # print("Setting use_sync_bn to False as distributed environment is not set up.")


    # --- Test SLaK_tiny ---
    print("\n--- Testing SLaK Tiny ---")
    # Using default kernel_configs from slak_tiny factory
    model_tiny = slak_tiny(num_classes=100, bn_stages=True).to(device)
    print(f"SLaK Tiny width_list: {model_tiny.width_list}") 
    print(f"SLaK Tiny kernel_configs: {model_tiny.kernel_configs}")
    
    image_size = (2, 3, 224, 224) 
    image = torch.rand(*image_size).to(device)

    try:
        print(f"\nModel is on: {next(model_tiny.parameters()).device}")
        print(f"Input image is on: {image.device}")

        output_features_tiny = model_tiny(image) 
        print("\nSLaK Tiny Output Feature Shapes (from model.forward):")
        for i, feat in enumerate(output_features_tiny):
            print(f"Stage {i} output shape: {feat.shape}, Device: {feat.device}")
        
        class_output_tiny = model_tiny.get_classification_output(image)
        print(f"\nSLaK Tiny Classification output shape: {class_output_tiny.shape}, Device: {class_output_tiny.device}")

        print("\nTesting kernel merging for SLaK Tiny...")
        model_tiny.merge_reparam_kernels()
        print("Kernel merging process completed for SLaK Tiny.")
        
        class_output_tiny_merged = model_tiny.get_classification_output(image)
        print(f"SLaK Tiny Classification output shape (after merge): {class_output_tiny_merged.shape}")
        
        # Test that merged output is close to original (requires saving original output or model)
        # For simplicity, just check if it runs

    except Exception as e:
        print(f"Error during SLaK Tiny test: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*30 + "\n")

    # --- Test SLaK_base with different kernel config and Decom ---
    print("--- Testing SLaK Base with custom bn_stages and some Decom ---")
    # kernel_configs: [(large_k_s0, small_k_s0_or_rank, Decom_s0), ...]
    custom_kc_base = [
        (31, 7, False),      # Stage 0: Large 31, Additive 7, No Decom
        (29, 5, True),       # Stage 1: Large 29, Rank 5 Decom
        (27, None, False),   # Stage 2: Large 27, No Additive, No Decom
        (13, 3, True)        # Stage 3: Large 13, Rank 3 Decom
    ]
    custom_bn_base = [True, True, False, False] 
    
    model_base = slak_base(
        num_classes=200, 
        kernel_configs=custom_kc_base, 
        bn_stages=custom_bn_base
    ).to(device)
    print(f"SLaK Base width_list: {model_base.width_list}")
    print(f"SLaK Base kernel_configs: {model_base.kernel_configs}")
    print(f"SLaK Base bn_stages: {model_base.bn_stages}")
    
    try:
        output_features_base = model_base(image)
        print("\nSLaK Base Output Feature Shapes:")
        for i, feat in enumerate(output_features_base):
            print(f"Stage {i} output shape: {feat.shape}")

        class_output_base = model_base.get_classification_output(image)
        print(f"\nSLaK Base Classification output shape: {class_output_base.shape}")

        print("\nTesting kernel merging for SLaK Base (should skip Decom=True blocks)...")
        model_base.merge_reparam_kernels() # Will only merge non-Decom blocks
        print("Kernel merging process completed for SLaK_base.")
        class_output_base_merged = model_base.get_classification_output(image)
        print(f"SLaK Base Classification output shape (after attempted merge): {class_output_base_merged.shape}")

    except Exception as e:
        print(f"Error during SLaK Base test: {e}")
        import traceback
        traceback.print_exc()