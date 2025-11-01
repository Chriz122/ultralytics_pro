import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import DropPath
from typing import List
import sys
import os
import copy

#====================================================================================
# Helper Functions 和 Modules (get_conv2d, FASA, ConvFFN, etc.) 保持不變
#====================================================================================

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        # This part is for a custom kernel implementation, keeping it as is.
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

class FASA(nn.Module):
    def __init__(self, dim: int, kernel_size: int, num_heads: int, window_size: int):
        super().__init__()
        self.q = nn.Conv2d(dim, dim, 1, 1, 0)
        self.kv = nn.Conv2d(dim, dim*2, 1, 1, 0)
        self.local_mixer = get_conv2d(dim, dim, kernel_size, 1, kernel_size//2, 1, dim, True)
        self.mixer = nn.Conv2d(dim, dim, 1, 1, 0)
        self.dim_head = dim // num_heads
        self.pool = self.refined_downsample(dim, window_size, 5)
        self.window_size = window_size
        self.scalor = self.dim_head ** -0.5

    def refined_downsample(self, dim, window_size, kernel_size):
        if window_size==1:
            return nn.Identity()
        for i in range(4):
            if 2**i == window_size:
                break
        block = nn.Sequential()
        for num in range(i):
            block.add_module('conv{}'.format(num), get_conv2d(dim, dim, kernel_size, 2, kernel_size//2, 1, dim, True))
            block.add_module('bn{}'.format(num), nn.SyncBatchNorm(dim))
            if num != i-1:
                block.add_module('linear{}'.format(num), nn.Conv2d(dim, dim, 1, 1, 0))
        return block

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size()
        H = math.ceil(h/self.window_size)
        W = math.ceil(w/self.window_size)
        q_local = self.q(x)
        q = q_local.reshape(b, -1, self.dim_head, h*w).transpose(-1, -2).contiguous()
        k, v = self.kv(self.pool(x)).reshape(b, 2, -1, self.dim_head, H*W).permute(1, 0, 2, 4, 3).contiguous()
        attn = torch.softmax(self.scalor * q @ k.transpose(-1, -2), -1)
        global_feat = attn @ v
        global_feat = global_feat.transpose(-1, -2).reshape(b, c, h, w)
        local_feat = self.local_mixer(q_local)
        local_weight = torch.sigmoid(local_feat)
        local_feat = local_feat * local_weight
        local2global = torch.sigmoid(global_feat)
        global2local = torch.sigmoid(local_feat)
        local_feat = local_feat * local2global
        return self.mixer(local_feat * global_feat)

class ConvFFN(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, out_channels):
        super().__init__()
        self.stride = stride
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.act = nn.GELU()
        self.dwconv = get_conv2d(hidden_channels, hidden_channels, kernel_size, stride, kernel_size//2, 1, hidden_channels, True)
        self.bn = nn.SyncBatchNorm(hidden_channels)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)
    
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        if self.stride == 1:
            x = x + self.dwconv(x)
        else:
            x = self.dwconv(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x

class FATBlock(nn.Module):
    def __init__(self, dim: int, out_dim: int, kernel_size: int, num_heads: int, window_size: int, 
                 mlp_kernel_size: int, mlp_ratio: float, stride: int, drop_path=0.):
        super().__init__()
        self.dim = dim
        self.cpe = get_conv2d(dim, dim, kernel_size, 1, kernel_size//2, 1, dim, True)
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = FASA(dim, kernel_size, num_heads, window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.GroupNorm(1, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                get_conv2d(dim, dim, mlp_kernel_size, stride, mlp_kernel_size//2, 1, dim, True),
                nn.SyncBatchNorm(dim),
                nn.Conv2d(dim, out_dim, 1, 1, 0)
            )
        self.ffn = ConvFFN(dim, mlp_hidden_dim, mlp_kernel_size, stride, out_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.cpe(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.downsample(x) + self.drop_path(self.ffn(self.norm2(x)))
        return x
    
class FATlayer(nn.Module):
    def __init__(self, depth: int, dim: int, out_dim: int, kernel_size: int, num_heads: int, 
                 window_size: int, mlp_kernel_size: int, mlp_ratio: float, drop_paths=[0., 0.],
                 downsample=True, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
            FATBlock(dim, dim, kernel_size, num_heads, window_size, mlp_kernel_size,
                  mlp_ratio, 1, drop_paths[i]) for i in range(depth-1)
            ]
        )
        if downsample:
            self.blocks.append(FATBlock(dim, out_dim, kernel_size, num_heads, window_size,
                                     mlp_kernel_size, mlp_ratio, 2, drop_paths[-1]))
        else:
            self.blocks.append(FATBlock(dim, out_dim, kernel_size, num_heads, window_size,
                                     mlp_kernel_size, mlp_ratio, 1, drop_paths[-1]))
    
    def forward(self, x: torch.Tensor):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2),
                        nn.SyncBatchNorm(out_channels),
                        nn.ReLU()
                    )
    
    def forward(self, x: torch.Tensor):
        return self.conv(x)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        self.conv1 = BasicBlock(in_channels, out_channels//2, 3, 2)
        self.conv2 = BasicBlock(out_channels//2, out_channels, 3, 2)
        self.conv3 = BasicBlock(out_channels, out_channels, 3, 1)
        self.conv4 = BasicBlock(out_channels, out_channels, 3, 1)
        self.conv5 = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        self.layernorm = nn.GroupNorm(1, out_channels)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.layernorm(x)
    
#====================================================================================
# 以下是主要修改區域
#====================================================================================

class FAT(nn.Module):
    """
    Focal Attention and Transformer (FAT) model.
    """
    def __init__(self, in_chans=3, num_classes=1000, img_size=224, embed_dims: List[int]=None, 
                 depths: List[int]=None, kernel_sizes: List[int]=None, num_heads: List[int]=None, 
                 window_sizes: List[int]=None, mlp_kernel_sizes: List[int]=None, 
                 mlp_ratios: List[float]=None, drop_path_rate=0.1, use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.mlp_ratios = mlp_ratios
        self.embed_dims = embed_dims
        self.img_size = img_size
        self.in_chans = in_chans
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(in_chans, embed_dims[0])
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build FAT stages
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            is_last_layer = (i_layer == self.num_layers - 1)
            layer = FATlayer(
                depths[i_layer], 
                embed_dims[i_layer], 
                embed_dims[i_layer + 1] if not is_last_layer else embed_dims[i_layer], 
                kernel_sizes[i_layer],
                num_heads[i_layer],
                window_sizes[i_layer], 
                mlp_kernel_sizes[i_layer], 
                mlp_ratios[i_layer],  
                dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], 
                downsample=not is_last_layer, 
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)
            
        # Classification head (kept for standalone classification tasks)
        self.norm = nn.GroupNorm(1, embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self.apply(self._init_weights)
        
        # --- Add width_list calculation ---
        # This part performs a dummy forward pass to get the channel dimensions of each stage's output
        self.width_list = []
        try:
            self.eval() 
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size)
            with torch.no_grad():
                features = self.forward(dummy_input)
            self.width_list = [f.size(1) for f in features]
            self.train() 
        except Exception as e:
            print(f"Error during dummy forward pass for width_list calculation: {e}")
            print("Setting width_list to embed_dims as fallback.")
            self.width_list = self.embed_dims 
            self.train()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    #================================================================
    # MODIFIED: forward_features is changed to produce the correct output channels.
    #================================================================
    def forward_features(self, x):
        """
        Forward pass that returns a list of feature maps from each stage.
        This is required by frameworks like ultralytics for detection/segmentation.
        This modified version produces the desired channel list: [48, 96, 192, 384] for fat_b1.
        """
        outputs = []
        
        # Stage 1: Output from PatchEmbedding (e.g., 48 channels for fat_b1)
        x = self.patch_embed(x)
        outputs.append(x)
        
        # Stage 2: Output from layers[0] (e.g., 96 channels for fat_b1)
        x = self.layers[0](x)
        outputs.append(x)
        
        # Stage 3: Output from layers[1] (e.g., 192 channels for fat_b1)
        x = self.layers[1](x)
        outputs.append(x)
        
        # Stage 4: Get final feature map from the last two layers.
        # This ensures all layers are used and the final output is correct for the classifier.
        x = self.layers[2](x) # Output has embed_dims[3] channels (e.g., 384)
        x = self.layers[3](x) # Final refinement stage, output also has embed_dims[3] channels
        outputs.append(x)
            
        return outputs

    def forward_cls(self, x):
        """
        Forward pass for classification.
        """
        # Note: This uses forward_features but only takes the last feature map
        features = self.forward_features(x)
        x = self.avgpool(self.norm(features[-1]))
        x = x.flatten(1)
        return self.head(x)

    def forward(self, x):
        """
        The main forward pass. Returns a list of feature maps.
        """
        return self.forward_features(x)

#================================================================
# MODIFIED: Removed build_model and config strings
# NEW: Factory functions for different FAT model sizes
#================================================================

def fat_b0(pretrained=False, **kwargs):
    #pretrained is a placeholder, no weights are loaded yet
    model = FAT(
        embed_dims=[32, 80, 160, 256],
        depths=[2, 2, 6, 2],
        kernel_sizes=[3, 5, 7, 9],
        num_heads=[2, 5, 10, 16],
        window_sizes=[8, 4, 2, 1],
        mlp_kernel_sizes=[5, 5, 5, 5],
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.05,
        **kwargs
    )
    return model

def fat_b1(pretrained=False, **kwargs):
    model = FAT(
        embed_dims=[48, 96, 192, 384],
        depths=[2, 2, 6, 2],
        kernel_sizes=[3, 5, 7, 9],
        num_heads=[3, 6, 12, 24],
        window_sizes=[8, 4, 2, 1],
        mlp_kernel_sizes=[5, 5, 5, 5],
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.1,
        **kwargs
    )
    return model

def fat_b2(pretrained=False, **kwargs):
    model = FAT(
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 6, 2],
        kernel_sizes=[3, 5, 7, 9],
        num_heads=[2, 4, 8, 16],
        window_sizes=[8, 4, 2, 1],
        mlp_kernel_sizes=[5, 5, 5, 5],
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.1,
        **kwargs
    )
    return model

def fat_b3(pretrained=False, **kwargs):
    model = FAT(
        embed_dims=[64, 128, 256, 512],
        depths=[4, 4, 16, 4],
        kernel_sizes=[3, 5, 7, 9],
        num_heads=[2, 4, 8, 16],
        window_sizes=[8, 4, 2, 1],
        mlp_kernel_sizes=[5, 5, 5, 5],
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.15,
        **kwargs
    )
    return model


if __name__ == '__main__':
    # 在 SyncBatchNorm 可用時，需要初始化 process group
    # 在單機單卡上運行時，可以註解掉這部分
    # torch.distributed.init_process_group(backend='gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

    img_h, img_w = 512, 512
    print("--- Creating FAT-b1 model ---")
    # We pass img_size to the constructor for the dummy forward pass
    model = fat_b1(img_size=img_h) 
    print("Model created successfully.")
    
    # 1. Check the calculated width_list
    print("Calculated width_list:", model.width_list)

    # 2. Test the main forward pass
    input_tensor = torch.rand(2, 3, img_h, img_w)
    print(f"\n--- Testing FAT-b1 forward pass (Input: {input_tensor.shape}) ---")

    model.eval()
    try:
        with torch.no_grad():
            # The output should be a list of tensors
            output_features = model(input_tensor)
        
        print("Forward pass successful.")
        print(f"Output type: {type(output_features)}")
        print("Output feature shapes:")
        assert isinstance(output_features, list), "Output is not a list!"
        
        # 3. Verify output shapes and width_list
        runtime_widths = []
        for i, features in enumerate(output_features):
            print(f"Stage {i+1}: {features.shape}")
            runtime_widths.append(features.size(1))

        print("\nRuntime output feature channels:", runtime_widths)
        assert model.width_list == runtime_widths, "Width list mismatch!"
        print("Width list verified successfully.")

        # 4. Test deepcopy
        print("\n--- Testing deepcopy ---")
        copied_model = copy.deepcopy(model)
        print("Deepcopy successful.")
        
        with torch.no_grad():
             output_copied = copied_model(input_tensor)
        print("Copied model forward pass successful.")
        assert len(output_copied) == len(output_features)
        print("Copied model output shapes verified.")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()