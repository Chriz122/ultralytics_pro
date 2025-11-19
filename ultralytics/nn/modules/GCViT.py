#!/usr/bin/env python3

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# written by Ali Hatamizadeh and Pavlo Molchanov from NVResearch


import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models import register_model
from timm.models.helpers import build_model_with_cfg
# For pretrained weight loading if not using build_model_with_cfg directly in user functions
from timm.models.helpers import load_pretrained
import logging

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'pool_size': None,
            'crop_pct': 0.875,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            **kwargs
            }


default_cfgs = {
    'gc_vit_xxtiny': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_xxtiny.pth.tar',
                          crop_pct=1.0,
                          input_size=(3, 224, 224),
                          crop_mode= 'center'),
    'gc_vit_xtiny': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_xtiny.pth.tar',
                         crop_pct=0.875,
                         input_size=(3, 224, 224),
                         crop_mode= 'center'),
    'gc_vit_tiny': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_tiny.pth.tar',
                        crop_pct=1.0,
                        input_size=(3, 224, 224),
                        crop_mode= 'center'),
    'gc_vit_tiny2': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_tiny2.pth.tar',
                         crop_pct=0.875,
                         input_size=(3, 224, 224),
                         crop_mode= 'center'),
    'gc_vit_small': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_small.pth.tar',
                         crop_pct=0.875,
                         input_size=(3, 224, 224),
                         crop_mode= 'center'),
    'gc_vit_small2': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_small2.pth.tar',
                          crop_pct=0.875,
                          input_size=(3, 224, 224),
                          crop_mode= 'center'),
    'gc_vit_base': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_base.pth.tar',
                        crop_pct=1.0,
                        input_size=(3, 224, 224),
                        crop_mode= 'center'),
    'gc_vit_large': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_large.pth.tar',
                         crop_pct=1.0,
                         input_size=(3, 224, 224),
                         crop_mode= 'center'),
    'gc_vit_large_224_21k': _cfg(url='https://drive.google.com/uc?export=download&id=1maGDr6mJkLyRTUkspMzCgSlhDzNRFGEf',
                                 crop_pct=1.0,
                                 input_size=(3, 224, 224),
                                 crop_mode= 'center',
                                 num_classes=21841), # Specify num_classes for 21k
    'gc_vit_large_384_21k': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_21k_large_384.pth.tar',
                                 crop_pct=1.0,
                                 input_size=(3, 384, 384),
                                 crop_mode='squash',
                                 num_classes=21841),
    'gc_vit_large_512_21k': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_21k_large_512.pth.tar',
                                 crop_pct=1.0,
                                 input_size=(3, 512, 512),
                                 crop_mode='squash',
                                 num_classes=21841),
}


def _to_channel_last(x):
    """
    Args:
        x: (B, C, H, W)

    Returns:
        x: (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x):
    """
    Args:
        x: (B, H, W, C)

    Returns:
        x: (B, C, H, W)
    """
    return x.permute(0, 3, 1, 2)


def window_partition(x, window_size, h_w, w_w):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size

    Returns:
        local window features (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, h_w, window_size, w_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, h_w, w_w, B):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image

    Returns:
        x: (B, H, W, C)
    """
    # B = int(windows.shape[0] // (H * W // window_size // window_size))
    x = windows.view(B, h_w, w_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SE(nn.Module):
    """
    Squeeze and excitation block
    """

    def __init__(self,
                 inp,
                 oup,
                 expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ReduceSize(nn.Module):
    """
    Down-sampling block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 norm_layer=nn.LayerNorm,
                 keep_dim=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False)
        self.norm2 = norm_layer(dim_out)
        self.norm1 = norm_layer(dim)

    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        x_conv = _to_channel_first(x) # Prepare for Conv2d
        x_conv = x_conv + self.conv(x_conv)
        x_conv = self.reduction(x_conv)
        x_reduced = _to_channel_last(x_conv) # Back to (B, H, W, C) for norm
        x = self.norm2(x_reduced)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, in_chans=3, dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim, 3, 2, 1) # First 2x downsample
        self.conv_down = ReduceSize(dim=dim, keep_dim=True) # Does not change dim, but applies norm. Internally it has another 2x, but keep_dim=True means output dim is same as input 'dim' to ReduceSize.
                                                          # Actually, PatchEmbed's self.conv_down's ReduceSize has keep_dim=True, so its 'reduction' layer will be Conv2d(dim, dim, 3, 2, 1).
                                                          # This means PatchEmbed does two 2x downsamplings.
                                                          # The ReduceSize(dim=dim, keep_dim=True) -> self.reduction = nn.Conv2d(dim, dim, 3, 2, 1). This is effectively a 2x downsampler.

    def forward(self, x): # x: (B, C_in, H_in, W_in)
        x = self.proj(x) # x: (B, dim, H_in/2, W_in/2)
        x = _to_channel_last(x) # x: (B, H_in/2, W_in/2, dim)
        x = self.conv_down(x) # x: (B, H_in/4, W_in/4, dim) because conv_down's ReduceSize will apply its self.reduction
        return x


class FeatExtract(nn.Module):
    """
    Feature extraction block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x): # x: (B, C, H, W)
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        super().__init__()
        window_size = to_2tuple(window_size) # Ensure tuple
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) # Explicitly use 'ij'
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global): # q_global is not used here, but kept for API consistency with WindowAttentionGlobal
        B_, N, C = x.shape
        head_dim = torch.div(C, self.num_heads, rounding_mode='floor')
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttentionGlobal(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        super().__init__()
        window_size = to_2tuple(window_size) # Ensure tuple
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) # Explicitly use 'ij'
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias) # Q is global, so only K, V from local
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        B_, N, C = x.shape
        B_global = q_global.shape[0] # This B is the original batch_size
        head_dim = torch.div(C, self.num_heads, rounding_mode='floor')
        
        # q_global is (B_global, 1, num_heads, N_global_tokens, head_dim)
        # x is (B_windowed, N_local_tokens, C) where B_windowed = B_global * num_windows
        # N_local_tokens = window_size*window_size. N_global_tokens should match N_local_tokens.
        
        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # k,v: (B_windowed, num_heads, N_local_tokens, head_dim)
        
        # Reshape/repeat q_global to match k and v's batch dimension B_windowed
        # q_global: (B_global, 1, self.num_heads, N, head_dim)
        # Assuming N from q_global is same as N from x (window_size*window_size)
        if B_ != B_global: # If x is windowed (B_ = B_global * num_windows)
            num_windows_b = torch.div(B_, B_global, rounding_mode='floor')
            q_global_reshaped = q_global.repeat_interleave(num_windows_b, dim=0) # (B_windowed, 1, num_heads, N, head_dim)
        else: # This case might occur if num_windows is 1
            q_global_reshaped = q_global

        q = q_global_reshaped.squeeze(1) # (B_windowed, num_heads, N, head_dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GCViTBlock(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution, # Tuple (H, W)
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 attention=WindowAttentionGlobal,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 ):
        super().__init__()
        self.window_size = to_2tuple(window_size)[0] # Keep as int for calculations
        self.norm1 = norm_layer(dim)

        self.attn = attention(dim,
                              num_heads=num_heads,
                              window_size=self.window_size, # Pass int
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
        
        # input_resolution is a tuple (H, W)
        # For non-square features, window partitioning might need care.
        # Assuming square features for simplicity based on original code's int input_resolution.
        # If input_resolution is truly (H,W), then num_windows might not be int(inp_w*inp_w)
        # For now, assume input_resolution refers to one dimension of a square feature map.
        # inp_w = torch.div(input_resolution[0], self.window_size, rounding_mode='floor')
        # self.num_windows = int(inp_w * inp_w) # This calculation is not directly used in forward.

    def forward(self, x, q_global):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        
        # Calculate h_w, w_w based on actual H, W
        h_w = torch.div(H, self.window_size, rounding_mode='floor')
        w_w = torch.div(W, self.window_size, rounding_mode='floor')
        
        # Pad if H or W are not multiples of window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x_padded = nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b)) # Pad C, W, H
            h_w_padded = torch.div(x_padded.shape[1], self.window_size, rounding_mode='floor')
            w_w_padded = torch.div(x_padded.shape[2], self.window_size, rounding_mode='floor')
        else:
            x_padded = x
            h_w_padded, w_w_padded = h_w, w_w

        x_windows = window_partition(x_padded, self.window_size, h_w_padded, w_w_padded)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, q_global)
        
        x_reversed = window_reverse(attn_windows, self.window_size, x_padded.shape[1], x_padded.shape[2], h_w_padded, w_w_padded, B)

        if pad_b > 0 or pad_r > 0:
            x_reversed = x_reversed[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(self.gamma1 * x_reversed)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class GlobalQueryGen(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution, # tuple (H,W) for feature map
                 image_resolution, # int, original image H (assume H=W)
                 window_size,      # int
                 num_heads):
        super().__init__()
        ws = to_2tuple(window_size)[0] # Keep as int

        # Determine downsampling based on feature map resolution relative to image resolution
        # Example: image_resolution=224.
        # PatchEmbed output res: image_resolution / 4 = 56
        # Level 0 input res: 56. Level 1 input res: 28. Level 2 input res: 14. Level 3 input res: 7.

        feat_map_res_h = input_resolution[0] # Current feature map height

        if feat_map_res_h == image_resolution // 4: # e.g., 56 for 224 input
            # Target: 56 -> 28 -> 14 -> 7. Global query from 7x7.
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False), # 56->28
                FeatExtract(dim, keep_dim=False), # 28->14
                FeatExtract(dim, keep_dim=False), # 14->7
            )
        elif feat_map_res_h == image_resolution // 8: # e.g., 28
            # Target: 28 -> 14 -> 7
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False), # 28->14
                FeatExtract(dim, keep_dim=False), # 14->7
            )
        elif feat_map_res_h == image_resolution // 16: # e.g., 14
            # Target: 14 -> 7.
            # If window_size (e.g. 14) is already the target resolution for global query (e.g. 7),
            # this logic might need adjustment. The original paper implies global query is from a fixed small feature map (e.g., 7x7).
            # If ws (window_size) == feat_map_res_h, then global query is from feat_map_res_h.
            # If ws (local window) is 7, and feat_map_res_h is 14, global query should be from 7.
            if ws == feat_map_res_h: # e.g. window_size=14, feat_map_res_h=14. Query from 14x14.
                 self.to_q_global = nn.Sequential(FeatExtract(dim, keep_dim=True))
            else: # e.g. window_size=7 (local), feat_map_res_h=14. Global query from 7x7.
                 self.to_q_global = nn.Sequential(FeatExtract(dim, keep_dim=False)) # 14->7
        elif feat_map_res_h == image_resolution // 32: # e.g., 7
            # Target: 7 -> 7
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=True) # 7->7
            )
        else:
            _logger.warning(f"GlobalQueryGen: Unexpected input_resolution {input_resolution} for image_resolution {image_resolution}. Using single FeatExtract(keep_dim=True).")
            self.to_q_global = nn.Sequential(FeatExtract(dim, keep_dim=True))


        self.num_heads = num_heads
        self.N = ws * ws # Number of tokens in a window, should match global query tokens
        self.dim_head = torch.div(dim, self.num_heads, rounding_mode='floor')

    def forward(self, x): # x is (B,C,H,W)
        # x should be channel_first for FeatExtract
        x_global = self.to_q_global(x) # (B, C, H_global, W_global)
        x_global = _to_channel_last(x_global) # (B, H_global, W_global, C)
        
        B = x_global.shape[0]
        # H_global, W_global should be equal to self.window_size after FeatExtract
        # Reshape to (B, 1, N_tokens_global, num_heads, dim_head) then permute
        # N_tokens_global = H_global * W_global, should be self.N
        try:
            q_global = x_global.reshape(B, 1, self.N, self.num_heads, self.dim_head).permute(0, 1, 3, 2, 4)
        except RuntimeError as e:
            _logger.error(f"Error in GlobalQueryGen reshape: {e}. x_global shape: {x_global.shape}, target N: {self.N}, num_heads: {self.num_heads}, dim_head: {self.dim_head}")
            # Fallback or raise: For now, let it raise to highlight the issue.
            # This usually means H_global*W_global from to_q_global does not match self.N (window_size*window_size)
            # Check FeatExtract logic based on input_resolution vs image_resolution.
            # The target size for global query features should be window_size x window_size.
            raise e
        return q_global


class GCViTLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 input_resolution, # Tuple (H,W)
                 image_resolution, # Int
                 num_heads,
                 window_size,      # Int
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            GCViTBlock(dim=dim,
                       num_heads=num_heads,
                       window_size=window_size,
                       mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias,
                       qk_scale=qk_scale,
                       attention=WindowAttention if (i % 2 == 0) else WindowAttentionGlobal,
                       drop=drop,
                       attn_drop=attn_drop,
                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                       norm_layer=norm_layer,
                       layer_scale=layer_scale,
                       input_resolution=input_resolution) # Pass tuple
            for i in range(depth)])
        self.downsample = None
        if downsample:
             self.downsample = ReduceSize(dim=dim, norm_layer=norm_layer, keep_dim=False) # keep_dim=False makes it 2x dim

        self.q_global_gen = GlobalQueryGen(dim, input_resolution, image_resolution, window_size, num_heads)

    def forward(self, x): # x: (B, H, W, C)
        q_global = self.q_global_gen(_to_channel_first(x)) # q_global_gen expects (B,C,H,W)
        for blk in self.blocks:
            x = blk(x, q_global)
        if self.downsample is None:
            return x
        return self.downsample(x)


class GCViT(nn.Module):
    def __init__(self,
                 dim,
                 depths,
                 window_size, # list of ints
                 mlp_ratio,
                 num_heads,   # list of ints
                 resolution=224, # int, input image resolution (H or W, assumes square)
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 out_indices=None, # Tuple like (0, 1, 2, 3) for feature extraction
                 **kwargs): # Consume other kwargs from build_model_with_cfg
        super().__init__()

        self.num_classes = num_classes
        self.out_indices = out_indices
        self.depths = depths
        self.num_layers = len(depths) # Number of stages

        self.patch_embed = PatchEmbed(in_chans=in_chans, dim=dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Calculate feature map resolution at each stage input
        # PatchEmbed outputs features at resolution / 4
        current_resolution_h = resolution // 4
        current_resolution_w = resolution // 4 # Assuming square for simplicity

        self.levels = nn.ModuleList()
        current_dim = dim
        for i in range(self.num_layers):
            level = GCViTLayer(dim=current_dim, # Dim for this stage
                               depth=depths[i],
                               num_heads=num_heads[i],
                               window_size=window_size[i],
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               norm_layer=norm_layer,
                               downsample=(i < self.num_layers - 1),
                               layer_scale=layer_scale,
                               input_resolution=(current_resolution_h, current_resolution_w),
                               image_resolution=resolution)
            self.levels.append(level)
            
            if i < self.num_layers - 1:
                current_resolution_h //= 2
                current_resolution_w //= 2
                current_dim *= 2 # Dimension doubles after ReduceSize in GCViTLayer

        # Norm layers for feature extraction if out_indices is specified
        if self.out_indices is not None:
            self.stage_output_channels = []
            _ch = dim # Channel dim after PatchEmbed
            for i in range(self.num_layers):
                if i < self.num_layers - 1: # Downsampling happens in GCViTLayer
                    _ch_out_stage = _ch * 2
                else: # No downsample in the last GCViTLayer's self.downsample
                    _ch_out_stage = _ch
                self.stage_output_channels.append(_ch_out_stage)
                _ch = _ch_out_stage
            
            for i_layer_idx in self.out_indices:
                if i_layer_idx < 0 or i_layer_idx >= len(self.stage_output_channels):
                    raise ValueError(f"out_index {i_layer_idx} is out of range for {len(self.stage_output_channels)} stages.")
                out_c = self.stage_output_channels[i_layer_idx]
                layer = norm_layer(out_c)
                self.add_module(f'norm{i_layer_idx}', layer)
            
            # For feature extraction, head is identity
            self.head = nn.Identity()
            self.norm_final = nn.Identity() # Not used
            self.avgpool = nn.Identity()    # Not used
            self.width_list = [self.stage_output_channels[i] for i in self.out_indices]

        else: # Original classification setup
            # The dim for the final norm is the dim of the last stage's output
            final_stage_dim = int(dim * (2 ** (self.num_layers - 1)))
            self.norm_final = norm_layer(final_stage_dim)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(final_stage_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.width_list = [] # Not applicable for classification mode in this context

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb', 'relative_position_bias_table'} # Added relative_position_bias_table

    def forward(self, x): # Input x: (B, C_in, H_in, W_in)
        x = self.patch_embed(x) # x: (B, H_feat, W_feat, C_feat)
        x = self.pos_drop(x)

        if self.out_indices is not None:
            outs = []
            for i, level in enumerate(self.levels):
                x = level(x) # Output of level is (B, H', W', C')
                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    x_out_normed = norm_layer(x) # Still (B, H', W', C')
                    outs.append(_to_channel_first(x_out_normed)) # (B, C', H', W')
            return outs # List of tensors
        else:
            # Original classification path
            for level in self.levels:
                x = level(x)
            x = self.norm_final(x)    # (B, H, W, C_final)
            x = _to_channel_first(x) # (B, C_final, H, W)
            x = self.avgpool(x)      # (B, C_final, 1, 1)
            x = torch.flatten(x, 1)  # (B, C_final)
            x = self.head(x)         # (B, num_classes)
            return x


def _create_gc_vit(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', False): # timm convention
        out_indices = kwargs.pop('out_indices', (0, 1, 2, 3))
        kwargs['out_indices'] = out_indices
        kwargs['num_classes'] = 0 # Ensure no classification head conflict

    model = build_model_with_cfg(
        GCViT,
        variant,
        pretrained,
        default_cfg=default_cfgs[variant], # Pass the default_cfg
        **kwargs,
    )
    return model

# --- Model Instantiation Functions ---
# These functions now directly call GCViT, allowing `out_indices` to be passed via kwargs
# For pretrained, we'll rely on _create_gc_vit which uses timm's infrastructure.

@register_model
def gc_vit_xxtiny(pretrained=False, **kwargs) -> GCViT:
    model_args = dict(depths=[2, 2, 6, 2], num_heads=[2, 4, 8, 16], window_size=[7, 7, 14, 7],
                      dim=64, mlp_ratio=3, drop_path_rate=kwargs.pop('drop_path_rate', 0.2))
    model_args.update(kwargs)
    return _create_gc_vit('gc_vit_xxtiny', pretrained=pretrained, **model_args)

@register_model
def gc_vit_xtiny(pretrained=False, **kwargs) -> GCViT:
    model_args = dict(depths=[3, 4, 6, 5], num_heads=[2, 4, 8, 16], window_size=[7, 7, 14, 7],
                      dim=64, mlp_ratio=3, drop_path_rate=kwargs.pop('drop_path_rate', 0.2))
    model_args.update(kwargs)
    return _create_gc_vit('gc_vit_xtiny', pretrained=pretrained, **model_args)

@register_model
def gc_vit_tiny(pretrained=False, **kwargs) -> GCViT:
    model_args = dict(depths=[3, 4, 19, 5], num_heads=[2, 4, 8, 16], window_size=[7, 7, 14, 7],
                      dim=64, mlp_ratio=3, drop_path_rate=kwargs.pop('drop_path_rate', 0.2))
    model_args.update(kwargs)
    return _create_gc_vit('gc_vit_tiny', pretrained=pretrained, **model_args)

@register_model
def gc_vit_tiny2(pretrained=False, **kwargs) -> GCViT: # Actual name from config seems to be gc_vit_tiny
    model_args = dict(depths=[3, 4, 29, 5], num_heads=[2, 4, 8, 16], window_size=[7, 7, 14, 7],
                      dim=64, mlp_ratio=3, drop_path_rate=kwargs.pop('drop_path_rate', 0.25))
    model_args.update(kwargs)
    return _create_gc_vit('gc_vit_tiny2', pretrained=pretrained, **model_args)

@register_model
def gc_vit_small(pretrained=False, **kwargs) -> GCViT:
    model_args = dict(depths=[3, 4, 19, 5], num_heads=[3, 6, 12, 24], window_size=[7, 7, 14, 7],
                      dim=96, mlp_ratio=2, drop_path_rate=kwargs.pop('drop_path_rate', 0.3), layer_scale=1e-5)
    model_args.update(kwargs)
    return _create_gc_vit('gc_vit_small', pretrained=pretrained, **model_args)

@register_model
def gc_vit_small2(pretrained=False, **kwargs) -> GCViT:
    model_args = dict(depths=[3, 4, 23, 5], num_heads=[3, 6, 12, 24], window_size=[7, 7, 14, 7],
                      dim=96, mlp_ratio=3, drop_path_rate=kwargs.pop('drop_path_rate', 0.35), layer_scale=1e-5)
    model_args.update(kwargs)
    return _create_gc_vit('gc_vit_small2', pretrained=pretrained, **model_args)


@register_model
def gc_vit_base(pretrained=False, **kwargs) -> GCViT:
    model_args = dict(depths=[3, 4, 19, 5], num_heads=[4, 8, 16, 32], window_size=[7, 7, 14, 7],
                      dim=128, mlp_ratio=2, drop_path_rate=kwargs.pop('drop_path_rate', 0.5), layer_scale=1e-5)
    model_args.update(kwargs)
    return _create_gc_vit('gc_vit_base', pretrained=pretrained, **model_args)

@register_model
def gc_vit_large(pretrained=False, **kwargs) -> GCViT:
    model_args = dict(depths=[3, 4, 19, 5], num_heads=[6, 12, 24, 48], window_size=[7, 7, 14, 7],
                      dim=192, mlp_ratio=2, drop_path_rate=kwargs.pop('drop_path_rate', 0.5), layer_scale=1e-5)
    model_args.update(kwargs)
    return _create_gc_vit('gc_vit_large', pretrained=pretrained, **model_args)

@register_model
def gc_vit_large_224_21k(pretrained=False, **kwargs) -> GCViT:
    model_args = dict(depths=[3, 4, 19, 5], num_heads=[6, 12, 24, 48], window_size=[7, 7, 14, 7],
                      dim=192, mlp_ratio=2, drop_path_rate=kwargs.pop('drop_path_rate', 0.5), layer_scale=1e-5)
    model_args.update(kwargs)
    # Ensure num_classes is correctly set for 21k pretrained weights if not overridden
    if pretrained and 'num_classes' not in kwargs:
        model_args['num_classes'] = default_cfgs['gc_vit_large_224_21k']['num_classes']
    return _create_gc_vit('gc_vit_large_224_21k', pretrained=pretrained, **model_args)

@register_model
def gc_vit_large_384_21k(pretrained=False, **kwargs) -> GCViT:
    model_args = dict(depths=[3, 4, 19, 5], num_heads=[6, 12, 24, 48], window_size=[12, 12, 24, 12], # Note window size change
                      dim=192, mlp_ratio=2, drop_path_rate=kwargs.pop('drop_path_rate', 0.1), layer_scale=1e-5,
                      resolution=384) # Pass resolution
    model_args.update(kwargs)
    if pretrained and 'num_classes' not in kwargs:
        model_args['num_classes'] = default_cfgs['gc_vit_large_384_21k']['num_classes']
    return _create_gc_vit('gc_vit_large_384_21k', pretrained=pretrained, **model_args)

@register_model
def gc_vit_large_512_21k(pretrained=False, **kwargs) -> GCViT:
    model_args = dict(depths=[3, 4, 19, 5], num_heads=[6, 12, 24, 48], window_size=[16, 16, 32, 16], # Note window size change
                      dim=192, mlp_ratio=2, drop_path_rate=kwargs.pop('drop_path_rate', 0.1), layer_scale=1e-5,
                      resolution=512) # Pass resolution
    model_args.update(kwargs)
    if pretrained and 'num_classes' not in kwargs:
        model_args['num_classes'] = default_cfgs['gc_vit_large_512_21k']['num_classes']
    return _create_gc_vit('gc_vit_large_512_21k', pretrained=pretrained, **model_args)

if __name__ == '__main__':
    # Example Usage for feature extraction (like in YOLO)
    print("Testing GCViT for feature extraction:")
    model_feat = gc_vit_tiny(pretrained=False, out_indices=(0, 1, 2, 3), num_classes=0, resolution=224)
    # model_feat.eval() # Set to eval mode if using pretrained weights

    print(f"Model out_indices: {model_feat.out_indices}")
    print(f"Model width_list: {model_feat.width_list}")

    dummy_input = torch.randn(2, 3, 224, 224)
    features = model_feat(dummy_input)

    print(f"Number of feature maps: {len(features)}")
    for i, f_map in enumerate(features):
        print(f"Shape of feature map {i}: {f_map.shape}")

    # Example Usage for classification
    print("\nTesting GCViT for classification:")
    model_cls = gc_vit_tiny(pretrained=False, num_classes=100, resolution=224) # 100 classes
    # model_cls.eval()

    dummy_input_cls = torch.randn(2, 3, 224, 224)
    predictions = model_cls(dummy_input_cls)
    print(f"Shape of classification output: {predictions.shape}")

    # Test a larger model with different resolution
    print("\nTesting GCViT Large 384 for feature extraction:")
    # For large models, pretrained=True might download weights. Set to False if not needed.
    # Make sure resolution matches the model variant if using pretrained.
    model_large_feat = gc_vit_large_384_21k(pretrained=False, out_indices=(0,1,2,3), num_classes=0, resolution=384)
    print(f"Model Large out_indices: {model_large_feat.out_indices}")
    print(f"Model Large width_list: {model_large_feat.width_list}")
    dummy_input_384 = torch.randn(1, 3, 384, 384)
    large_features = model_large_feat(dummy_input_384)
    print(f"Number of feature maps (large): {len(large_features)}")
    for i, f_map in enumerate(large_features):
        print(f"Shape of feature map {i} (large): {f_map.shape}")