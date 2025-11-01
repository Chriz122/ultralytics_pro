# --------------------------------------------------------
# Swin Transformer (Modified from Code 1 and Code 2)
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei (for original Swin)
# Modifications by AI for SlabSwinTransformer adaptation
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from functools import partial
import numpy as np

# import seaborn as sns; sns.set()
# import matplotlib.pyplot as plt

class RepBN(nn.Module):
    def __init__(self, channels):
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        # x shape: (B, N, C)
        if x.ndim == 3:
            x = x.transpose(1, 2) # (B, C, N)
            x = self.bn(x) + self.alpha * x
            x = x.transpose(1, 2) # (B, N, C)
        elif x.ndim == 2: # (N, C) - for LayerNorm like behavior in PatchEmbed
            # This case might not be ideal for RepBN which expects Batch.
            # For simplicity, let's assume it's (1, N, C)
            original_shape = x.shape
            x = x.unsqueeze(0).transpose(1,2) # (1, C, N)
            x = self.bn(x) + self.alpha * x
            x = x.transpose(1,2).squeeze(0) # (N,C)
        else:
            raise ValueError(f"RepBN expects 2D or 3D input, got {x.ndim}D")
        return x


class LinearNorm(nn.Module):
    def __init__(self, dim, norm1, norm2, warm=0, step=300000, r0=1.0):
        super(LinearNorm, self).__init__()
        self.register_buffer('warm', torch.tensor(warm))
        self.register_buffer('iter', torch.tensor(step))
        self.register_buffer('total_step', torch.tensor(step))
        self.r0 = r0
        self.norm1 = norm1(dim)
        self.norm2 = norm2(dim)

    def forward(self, x):
        if self.training:
            if self.warm > 0:
                self.warm.copy_(self.warm - 1)
                x = self.norm1(x)
            else:
                lamda = self.r0 * self.iter / self.total_step
                if self.iter > 0:
                    self.iter.copy_(self.iter - 1)
                x1 = self.norm1(x)
                x2 = self.norm2(x)
                x = lamda * x1 + (1 - lamda) * x2
        else:
            # During evaluation, typically one norm is chosen.
            # Original RepBN paper suggests using the BN branch or a fused version.
            # Here, using norm2 (RepBN) during eval if it's the target.
            # Or, more simply, use norm1 (e.g., LayerNorm) for eval stability.
            # For this adaptation, let's assume norm1 is the stabler one for eval,
            # or make it configurable. Let's stick to norm2 like original code.
            x = self.norm2(x)
        return x


ln = nn.LayerNorm
linearnorm = partial(LinearNorm, norm1=ln, norm2=RepBN, step=60000)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SimplifiedLinearAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.focusing_factor = focusing_factor # Not explicitly used in this simplified version
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Softmax is not typically used in linear attention in this way, often replaced by kernel functions
        # self.softmax = nn.Softmax(dim=-1) # Let's comment this out as per typical linear attention

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        
        # Positional encoding for K, as in the original snippet
        # The size of positional_encoding should match K's dimension after head splitting
        # K has shape (B_ * num_heads, N, head_dim)
        # So positional_encoding should be (1, N, num_heads * head_dim) or broadcastable
        # Original was (1, N, dim) which is fine as it's added before head splitting.
        self.positional_encoding = nn.Parameter(torch.zeros(1, window_size[0] * window_size[1], dim))
        trunc_normal_(self.positional_encoding, std=.02)


        # print('Linear Attention window{} f{} kernel{}'.
        #       format(window_size, focusing_factor, kernel_size))

    def forward(self, x, mask=None): # Mask is not typically used in global linear attention
        B_, N, C = x.shape # num_windows*B, N, C
        
        qkv = self.qkv(x) # (B_, N, 3*C)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # Each (B_, N, C)

        k = k + self.positional_encoding # Add positional encoding to K

        # Apply kernel function (e.g., ReLU or elu+1)
        # Original paper "Transformers are RNNs" uses elu(x) + 1
        # Swin-Lite used ReLU. Let's stick to ReLU as per the provided code.
        q = F.relu(q)
        k = F.relu(k)

        # Split heads
        q = rearrange(q, "b n (h d) -> (b h) n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> (b h) n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> (b h) n d", h=self.num_heads)
        # q, k, v are now (B_*num_heads, N, head_dim)

        # Linear attention: Q (K^T V) / (Q K^T 1)
        # For numerical stability, often computed as Q (K^T V) / ( (Q K^T) sum )
        # Or using the efficient (associative) way: (Q K^T) V
        # The provided code uses a specific formulation:
        # z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        # kv = torch.einsum("b j c, b j d -> b c d", k, v)
        # x_attn = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        # This is one way:  Q * ( (K^T V) / (Q @ sum(K, dim=1) ) )
        
        # Let's use the standard efficient linear attention for clarity if possible,
        # or stick to the provided one. The einsum formulation can be efficient.
        # The formulation sum_j K_j / (Q_i . sum_j K_j) * (Q_i . sum_j K_j V_j)
        # is equivalent to: Q (K^T V) / (Q @ sum_k(K_k) ) if Q,K are row vectors
        
        # Sticking to the provided code's linear attention logic
        # Required for disabling AMP for this part if using mixed precision
        # with torch.cuda.amp.autocast(enabled=False):
        #     q_fp32 = q.float()
        #     k_fp32 = k.float()
        #     v_fp32 = v.float()

        #     # Denominator: sum_j k_j (effectively)
        #     # For each query q_i, the denominator is q_i . (sum_j k_j)
        #     k_sum = torch.sum(k_fp32, dim=1) # (B*h, d)
        #     # einsum for q_i . k_sum: (B*h, N, d) * (B*h, d) -> (B*h, N)
        #     denominator = torch.einsum('bnd,bd->bn', q_fp32, k_sum).unsqueeze(2) # (B*h, N, 1)
            
        #     # Numerator: (K^T V) part
        #     # k: (B*h, N, d), v: (B*h, N, d_v) (here d_v = d)
        #     # kv = K^T V : (B*h, d, d_v)
        #     kv = torch.einsum('bnd,bnm->bdm', k_fp32, v_fp32) # (B*h, d, d)
            
        #     # q @ kv : (B*h, N, d) @ (B*h, d, d) -> (B*h, N, d)
        #     numerator = torch.einsum('bnd,bdm->bnm', q_fp32, kv) # (B*h, N, d)
            
        #     x_attn = numerator / (denominator + 1e-6)
        #     x_attn = x_attn.to(x.dtype)
        
        # The provided code's einsum way:
        # i, j are sequence length indices. c, d are feature dimension indices.
        # q: (bh) i c, k: (bh) j c, v: (bh) j d
        # z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        # if i * j * (c + d) > c * d * (i + j): # This is a heuristic for choosing computation path
        #     kv = torch.einsum("b j c, b j d -> b c d", k, v)
        #     x_attn = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        # else:
        #     qk = torch.einsum("b i c, b j c -> b i j", q, k)
        #     x_attn = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)
        
        # Simpler FAVOR+ style (or Performer if kernels are orthogonal random features):
        # Q (K^T V)
        # We need to be careful with dimensions for K^T V if N is large.
        # (K^T V) is d x d_v. Then Q @ (K^T V) is N x d_v.
        # context_matrix = torch.einsum('bsd,bsm->bdm', k, v) # (bh, d_k, d_v)
        # x_attn = torch.einsum('bsd,bdm->bsm', q, context_matrix) # (bh, N, d_v)
        
        # Normalization:
        # For FAVOR+, typically a softmax is applied to Q, and K is positive.
        # Or, Q K^T is computed, scaled, and then multiplied by V.
        # For linear attention, often it's phi(Q) (phi(K)^T V)
        # where phi is some kernel.
        # Let's use the common Q(K^T V) / (Q K^T 1) with elu kernel
        q_prime = torch.nn.functional.elu(q) + 1.0
        k_prime = torch.nn.functional.elu(k) + 1.0

        # (K'^T V)
        kv_prime = torch.einsum('bnd,bnm->bdm', k_prime, v) # (bh, d, d_v)
        # Q' @ (K'^T V)
        x_attn = torch.einsum('bnd,bdm->bnm', q_prime, kv_prime) # (bh, N, d_v)

        # Denominator: Q' @ (K'^T @ 1s)
        k_prime_sum = k_prime.sum(dim=1) # (bh, d)
        normalizer = torch.einsum('bnd,bd->bn', q_prime, k_prime_sum).unsqueeze(-1) # (bh, N, 1)
        x_attn = x_attn / (normalizer + 1e-6)


        # DWC part
        # v is (B_*num_heads, N, head_dim), N = window_h * window_w
        # We need to know window_h, window_w. Assuming square window from self.window_size
        win_h, win_w = self.window_size
        if not (N == win_h * win_w): # Should not happen if N is derived from window
             # Fallback if N is not window_size_h * window_size_w, e.g. dynamic shape
            patch_num_sqrt = int(N**0.5)
            if patch_num_sqrt * patch_num_sqrt != N: # Not a perfect square
                # Cannot do DWC easily if not a square grid and Wh, Ww unknown
                # This part might need to be skipped or handled differently
                # For now, we assume N is always window_h * window_w from Swin blocks
                pass # skip DWC if not a perfect square and shape mismatch
            else:
                win_h, win_w = patch_num_sqrt, patch_num_sqrt

        # Reshape v for DWC: (B*h, N, d) -> (B*h, d, win_h, win_w)
        feature_map_v = rearrange(v, "bh (wh ww) d -> bh d wh ww", wh=win_h, ww=win_w)
        feature_map_v = self.dwc(feature_map_v) # (bh, d, win_h, win_w)
        feature_map_v = rearrange(feature_map_v, "bh d wh ww -> bh (wh ww) d") # (bh, N, d)
        
        x_out = x_attn + feature_map_v # Add DWC features

        # Merge heads
        x_out = rearrange(x_out, "(b h) n d -> b n (h d)", h=self.num_heads)
        
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion (used for init, H,W passed to forward).
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focusing_factor, kernel_size, attn_type: For SimplifiedLinearAttention or choosing WindowAttention
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 focusing_factor=3, kernel_size=5, attn_type='L'):
        super().__init__()
        self.dim = dim
        # self.input_resolution = input_resolution # H,W are now passed to forward
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # If window size is larger than input resolution, we don't partition windows
        # This check should happen in BasicLayer or be dynamic based on H,W in forward
        # For now, keep init-time check based on passed input_resolution for this stage
        if min(input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        assert attn_type in ['L', 'S']
        if attn_type == 'L':
            self.attn = SimplifiedLinearAttention(
                dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                focusing_factor=focusing_factor, kernel_size=kernel_size)
        else: # 'S'
            self.attn = WindowAttention(
                dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, mask_matrix): # H, W of the input feature map for this block
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, L={L}, H={H}, W={W}, H*W={H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature maps to multiples of window size
        # (As in Code 2 SwinTransformerBlock)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b)) # (B, H_pad, W_pad, C)
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix # Use mask from BasicLayer
        else:
            shifted_x = x
            attn_mask = None # No mask needed for non-shifted attention

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA or SimplifiedLinearAttention
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B, H_pad, W_pad, C

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x) # In original Swin, this is x = shortcut + self.drop_path(self.attn(self.norm1(x)))
                                        # The current structure here is norm -> attn -> residual -> FFN(norm -> mlp -> residual)
                                        # Let's adjust to match standard Swin:
                                        # x = self.norm1(x) ... attn ...
                                        # x = shortcut + self.drop_path(x)
                                        # x = x + self.drop_path(self.mlp(self.norm2(x)))
                                        # This seems correct. The input 'x' to attn is from self.norm1(x).
                                        # The output of attn logic is 'x'. This 'x' is then added to 'shortcut'.
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm): # input_resolution removed
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W): # H, W of the input x
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even." # Padding handles this

        x = x.view(B, H, W, C)

        # Padding (from Code 2 PatchMerging)
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2)) # Pad C, W, H dims

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        
        # Update H, W after merging for view
        # H_merged, W_merged = x.shape[1], x.shape[2] # H/2, W/2 (approx)
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x # Returns merged x. New H, W will be calculated in BasicLayer
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution for this stage.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        attn_type can be 'L', 'S', or 'M<k>' (e.g., 'M2' means first 2 Linear, rest Standard)
        ... other args
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 focusing_factor=3, kernel_size=5, attn_type='L'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution # For block initialization
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size # Base window size for this layer
        self.shift_size = window_size // 2

        # Determine attn_type for each block
        if isinstance(attn_type, str) and attn_type.startswith('M'):
            try:
                linear_block_count = int(attn_type[1:])
            except ValueError:
                raise ValueError(f"Invalid M-type attn_type: {attn_type}. Should be M<num_linear_blocks>.")
            _attn_types = ['L'] * linear_block_count + ['S'] * (depth - linear_block_count)
            if len(_attn_types) > depth: _attn_types = _attn_types[:depth]
        elif isinstance(attn_type, str) and attn_type in ['L', 'S']:
            _attn_types = [attn_type] * depth
        elif isinstance(attn_type, list) and len(attn_type) == depth:
            _attn_types = attn_type
        else:
            raise ValueError(f"Invalid attn_type: {attn_type}")

        # Determine window_size for each block based on its attn_type
        # Original SlabSwin had:
        # window_sizes = [(window_size if attn_types[i] == 'L' else (7 if window_size <= 56 else 12)) for i in range(depth)]
        # Let's adapt this slightly: Use self.window_size for 'L', and a fixed or adaptive for 'S'
        # For simplicity, let's use self.window_size for all now, as attn_type 'S' will use WindowAttention which takes window_size.
        # If specific window sizes for 'S' are needed, this logic can be refined.
        _window_sizes = []
        for i in range(depth):
            current_block_input_res = input_resolution # This is the resolution for the whole BasicLayer
                                                       # SwinTransformerBlock caps its window_size if too large
            ws = self.window_size
            if _attn_types[i] == 'S': # Potentially different window for standard attention
                # ws = 7 # or some other logic like in original SlabSwin
                pass # Sticking to self.window_size for now
            _window_sizes.append(ws)


        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution, # Pass stage's input_res
                                 num_heads=num_heads, window_size=_window_sizes[i],
                                 shift_size=0 if (i % 2 == 0) else _window_sizes[i] // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 focusing_factor=focusing_factor,
                                 kernel_size=kernel_size,
                                 attn_type=_attn_types[i])
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W): # Current H, W of feature map x
        # Calculate attention mask for SW-MSA (Shifted Window Multi-head Self-Attention)
        # This mask is shared by all blocks in this layer that use shifting.
        # (Logic from Code 2 BasicLayer)
        # Effective window size for mask calculation should be based on the blocks that *use* SW-MSA.
        # Typically, all blocks in a layer use the same window_size or it's adapted.
        # We use self.window_size (from BasicLayer's init) for mask calculation.
        
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        
        # Slices for mask generation need to use self.window_size and self.shift_size
        # (which were derived from self.window_size)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size) # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # (nW, 1, N) - (nW, N, 1) -> (nW, N, N)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn_mask = attn_mask.to(x.dtype) # Ensure dtype match

        for blk in self.blocks:
            # Pass H, W of current feature map `x` to the block
            # The block itself will handle padding if its internal window_size requires it
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W, attn_mask)
            else:
                x = blk(x, H, W, attn_mask)
        
        x_out_before_downsample = x # This is the output of this stage, before downsampling

        if self.downsample is not None:
            x_down = self.downsample(x, H, W) # Pass H,W for PatchMerging
            # Calculate new H, W after downsampling
            # PatchMerging pads for odd H/W, so (H+1)//2 is robust
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_out_before_downsample, H, W, x_down, Wh, Ww
        else:
            return x_out_before_downsample, H, W, x, H, W # No downsampling, pass x and original H,W

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size. Default: 224. (Not strictly needed if H,W passed to forward)
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # img_size = to_2tuple(img_size) # Not strictly needed with dynamic H,W
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        # self.img_size = img_size # Stored for reference, but forward uses actual input H,W
        # self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        # self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Padding (from Code 2 PatchEmbed)
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1])) # Pad last dim (W)
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0])) # Pad second to last dim (H)
        
        x = self.proj(x)  # B, embed_dim, Wh, Ww
        Wh, Ww = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)  # B, Wh*Ww, embed_dim
        if self.norm is not None:
            x = self.norm(x)
        # Return x, and new H, W of the patch grid
        return x, Wh, Ww


class SlabSwinTransformer(nn.Module):
    r""" Slab Swin Transformer (adapted for backbone usage)
    Args:
        img_size (int | tuple(int)): Input image size. Default 224. (Used for APE if enabled)
        pretrain_img_size (int | tuple(int)): Pretrained image size for APE. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        ... other Swin args ...
        out_indices (tuple[int]): Output from which stages (0, 1, 2, 3).
    """
    def __init__(self, img_size=224, pretrain_img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=linearnorm, ape=False, patch_norm=True,
                 use_checkpoint=False,
                 focusing_factor=3, kernel_size=5, attn_type='LLLL', # attn_type can be string or list
                 out_indices=(0, 1, 2, 3), frozen_stages=-1, # Added from Code 2 Swin
                 **kwargs):
        super().__init__()

        self.pretrain_img_size = to_2tuple(pretrain_img_size)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) # This is for final output
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages # Not implemented in this version, but kept for compatibility

        # Ensure attn_type is a list of length num_layers
        if isinstance(attn_type, str) and len(attn_type) == self.num_layers:
            self.attn_types = list(attn_type)
        elif isinstance(attn_type, str) and len(attn_type) == 1: # e.g. 'L' -> ['L','L','L','L']
            self.attn_types = [attn_type] * self.num_layers
        elif isinstance(attn_type, list) and len(attn_type) == self.num_layers:
            self.attn_types = attn_type
        else: # Handles M-type like M2LL -> ['M2','L','L'] needs BasicLayer to parse 'M2'
              # Or a single 'M2' for the first layer
            self.attn_types = []
            temp_attn_type_str = attn_type if isinstance(attn_type, str) else "".join(attn_type)
            
            #This logic is for parsing complex strings like 'M2LS'
            #For simplicity, assume attn_type is either a list of types per layer,
            #or a single character ('L', 'S') repeated for all layers,
            #or a M-type string for *each* layer e.g. ['M2', 'M1', 'S', 'S']
            #The current BasicLayer init handles single char 'L', 'S' or 'M<k>' per layer.
            if isinstance(attn_type, str) and attn_type.startswith('M'): # 'M2LL' style for whole model
                idx = 0
                parsed_types = []
                while idx < len(attn_type):
                    if attn_type[idx] == 'M':
                        num_str = ""
                        idx += 1
                        while idx < len(attn_type) and attn_type[idx].isdigit():
                            num_str += attn_type[idx]
                            idx += 1
                        parsed_types.append(f"M{num_str}")
                    else:
                        parsed_types.append(attn_type[idx])
                        idx +=1
                self.attn_types = parsed_types
            elif isinstance(attn_type, str): # 'LLLL' or 'L'
                 self.attn_types = [c for c in attn_type] if len(attn_type) == self.num_layers else [attn_type[0]] * self.num_layers
            elif isinstance(attn_type, list): # ['L', 'L', 'S', 'S'] or [['M2'], ['L'], ['S'], ['S']]
                self.attn_types = attn_type
            else:
                raise ValueError(f"attn_type format is not recognized: {attn_type}")

            if len(self.attn_types) != self.num_layers:
                 print(f"Warning: Length of parsed attn_types ({len(self.attn_types)}) does not match num_layers ({self.num_layers}). Using first char for all or check format.")
                 # Fallback or specific error, for now, let's assume BasicLayer handles its own attn_type string.
                 # The BasicLayer expects attn_type to define behavior *within* that layer.
                 # So self.attn_types should be a list of such definitions, one for each BasicLayer.
                 # e.g. attn_type = ['M2', 'L', 'S', 'S'] if depths = [4,2,2,2], then BasicLayer0 gets 'M2' for its 4 blocks.


        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        # After patch_embed, H, W are patch_resolution
        # Wh_init = img_size[0] // patch_size if isinstance(img_size, (tuple,list)) else img_size // patch_size
        # Ww_init = img_size[1] // patch_size if isinstance(img_size, (tuple,list)) else img_size // patch_size
        # self.patches_resolution = (Wh_init, Ww_init) # Initial patch resolution

        # Absolute position embedding
        if self.ape:
            # APE should be (1, embed_dim, pretrain_Hp, pretrain_Wp) as in Code 2 for interpolation
            # Or (1, num_patches, embed_dim) as in Code 1 for direct addition
            # Let's follow Code 2 for interpolation if img_size varies.
            pretrain_patch_res_h = self.pretrain_img_size[0] // to_2tuple(patch_size)[0]
            pretrain_patch_res_w = self.pretrain_img_size[1] // to_2tuple(patch_size)[1]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, pretrain_patch_res_h, pretrain_patch_res_w))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        else:
            self.absolute_pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # Calculate input_resolution for this specific layer
            # This is the H, W of patches for this stage
            current_patches_resolution_h = to_2tuple(img_size)[0] // (to_2tuple(patch_size)[0] * (2 ** i_layer))
            current_patches_resolution_w = to_2tuple(img_size)[1] // (to_2tuple(patch_size)[1] * (2 ** i_layer))
            
            layer_attn_type = self.attn_types[i_layer]

            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(current_patches_resolution_h, current_patches_resolution_w),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size, # Pass base window_size
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               focusing_factor=focusing_factor,
                               kernel_size=kernel_size,
                               attn_type=layer_attn_type) # Pass specific attn_type for this layer
            self.layers.append(layer)

        # Feature dimensions for each output layer
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # Add norm layer for each output feature level (as in Code 2)
        for i_idx in out_indices:
            if i_idx < self.num_layers : # Make sure index is valid
                p_norm_layer = norm_layer(self.num_features[i_idx])
                layer_name = f'norm{i_idx}'
                self.add_module(layer_name, p_norm_layer)
            else: # Should not happen if out_indices are correct
                 print(f"Warning: out_indices contains {i_idx} which is out of bounds for num_layers {self.num_layers}")


        self.apply(self._init_weights)
        # self._freeze_stages() # If frozen_stages > -1

        # Calculate width_list for YOLO integration
        # Requires a dummy forward pass. Make sure model is on a device if using CUDA in dummy forward.
        # This needs to be done carefully, as it runs a forward pass.
        # For now, let's compute it on CPU.
        self_device = next(self.parameters()).device
        try:
            with torch.no_grad():
                dummy_input_size = to_2tuple(img_size) # Use img_size for dummy input
                # For tasks like detection, image size can vary, so using a common one like 640.
                # Let's use a placeholder size if img_size is small.
                # Or, make width_list computation optional / post-init.
                # For now, use a representative size.
                # Using a size that will not cause issues with patch_size and downsampling.
                s = 256 
                if dummy_input_size[0] < 64 or dummy_input_size[1] < 64 : # if img_size is too small
                     s_h = max(64, to_2tuple(patch_size)[0] * (2**(self.num_layers-1)))
                     s_w = max(64, to_2tuple(patch_size)[1] * (2**(self.num_layers-1)))
                     s = max(s_h, s_w)


                # If model has parameters, it should be on a device.
                # Create dummy on the same device as model if possible.
                dummy_input = torch.randn(1, in_chans, s, s).to(self_device)
                # Temporarily set to eval mode for dummy pass if norm layers behave differently
                original_mode = self.training
                self.eval()
                dummy_outputs = self.forward(dummy_input)
                self.width_list = [out.shape[1] for out in dummy_outputs]
                self.train(original_mode) # Restore original mode
        except Exception as e:
            print(f"Failed to compute width_list during init: {e}")
            print("Please ensure img_size, patch_size, and depths allow for a valid forward pass.")
            print("Setting width_list to num_features for selected out_indices as a fallback.")
            self.width_list = [self.num_features[i] for i in self.out_indices if i < len(self.num_features)]


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm): # Also for RepBN's bn part if it were LayerNorm-like
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, RepBN): # Initialize RepBN components
            if hasattr(m.bn, 'weight') and m.bn.weight is not None:
                 nn.init.constant_(m.bn.weight, 1.0)
            if hasattr(m.bn, 'bias') and m.bn.bias is not None:
                 nn.init.constant_(m.bn.bias, 0)
            if hasattr(m, 'alpha') and m.alpha is not None:
                 nn.init.constant_(m.alpha, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x): # This is the main forward now
        x, Wh, Ww = self.patch_embed(x) # Wh, Ww are patch resolution H, W

        if self.absolute_pos_embed is not None:
            # Interpolate APE (Code 2 style)
            # x is (B, N, C), APE is (1, C, Hp, Wp)
            # Need to reshape x to (B, C, Wh, Ww) then add APE, then reshape back
            B, N, C = x.shape
            x = x.view(B, Wh, Ww, C).permute(0, 3, 1, 2).contiguous() # B, C, Wh, Ww
            
            ape_interp = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic', align_corners=False)
            x = x + ape_interp
            x = x.flatten(2).transpose(1, 2) # B, Wh*Ww, C

        x = self.pos_drop(x)

        outs = []
        current_H, current_W = Wh, Ww # These are dimensions of the patch grid

        for i in range(self.num_layers):
            layer = self.layers[i]
            # x_out_before_downsample is the feature map for this stage's output
            # x_after_downsample is input to next stage
            # H_out, W_out are dims for x_out_before_downsample
            # H_next, W_next are dims for x_after_downsample
            x_out_stage, H_out, W_out, x_after_downsample, H_next, W_next = \
                layer(x, current_H, current_W)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_normed = norm_layer(x_out_stage) # Input (B, L, C)
                # Reshape to (B, C, H, W) for detection heads
                # H_out, W_out are the dimensions of x_out_stage's feature map
                out = x_normed.view(-1, H_out, W_out, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
            
            x = x_after_downsample
            current_H, current_W = H_next, W_next
            
        return outs # Return list of feature maps


# Factory functions
def _update_weight_keys(model_dict, weight_dict):
    """
    Handles potential key mismatches, e.g. 'layers.0.blocks.0.attn.relative_position_bias_table'
    vs 'layers.0.blocks.0.attn.relative_position_bias_table.relative_position_bias_table'
    This is a placeholder; more sophisticated key mapping might be needed if using official Swin weights.
    For SlabSwin trained weights, keys should match.
    """
    new_weight_dict = {}
    for k_w, v_w in weight_dict.items():
        if k_w in model_dict:
            new_weight_dict[k_w] = v_w
        else:
            # Try common modifications if official Swin weights are used
            # e.g., some frameworks add module names
            # This is highly dependent on the source of weights
            pass # For now, only exact matches
    return new_weight_dict

def update_weight(model_dict, weight_dict):
    # Placeholder for more complex weight loading if needed
    # For now, using a simple key check
    idx, temp_dict = 0, {}
    processed_weight_dict = _update_weight_keys(model_dict, weight_dict)

    for k, v in processed_weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'Loading weights... {idx}/{len(model_dict)} items matched and loaded.')
    return model_dict

# Example Factory Functions for SlabSwinTransformer
# Users should define these based on their SlabSwin configurations

def SlabSwinTransformer_T(weights='', **kwargs): # Tiny
    model_kwargs = dict(
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=56,
        norm_layer=linearnorm, attn_type='LLSS', # Example: all linear attention
        **kwargs
    )
    model = SlabSwinTransformer(**model_kwargs)
    if weights:
        # model.load_state_dict(torch.load(weights)['model']) # Adjust based on weight file format
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights, map_location='cpu').get('model', torch.load(weights, map_location='cpu'))))
    return model

def SlabSwinTransformer_S(weights='', **kwargs): # Small
    model_kwargs = dict(
        embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=56,
        norm_layer=linearnorm, attn_type='LLSS', # Example: all standard attention
        **kwargs
    )
    model = SlabSwinTransformer(**model_kwargs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights, map_location='cpu').get('model', torch.load(weights, map_location='cpu'))))
    return model

def SlabSwinTransformer_B(weights='', **kwargs): # Small
    model_kwargs = dict(
        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=56,
        norm_layer=linearnorm, attn_type='LLSS', # Example: all standard attention
        **kwargs
    )
    model = SlabSwinTransformer(**model_kwargs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights, map_location='cpu').get('model', torch.load(weights, map_location='cpu'))))
    return model


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("Testing SlabSwinTransformer_T (all Linear Attention)")
    # Ensure img_size is large enough for patch_size and num_layers
    # min_dim = patch_size * (2**(num_layers-1))
    # For patch_size=4, num_layers=4 -> 4 * 2^3 = 4 * 8 = 32. So img_size >= 32
    model_t = SlabSwinTransformer_T(img_size=224, patch_size=4, out_indices=(0,1,2,3)).to(device)
    # print(model_t)
    print(f"T Model width_list: {model_t.width_list}")

    # Test with a common detection input size
    inputs = torch.randn((1, 3, 320, 320)).to(device) # Using 320x320
    
    model_t.eval() # Important for norms and dropout
    with torch.no_grad():
        res_t = model_t(inputs)
    
    print(f"Output for Tiny (LLLL) with input {inputs.shape}:")
    for i, r_t in enumerate(res_t):
        print(f"Stage {model_t.out_indices[i]}: {r_t.shape}")
    
    print("\nTesting SlabSwinTransformer_S (all Standard Attention)")
    model_s = SlabSwinTransformer_S(img_size=256, patch_size=4, out_indices=(0,1,2,3)).to(device)
    print(f"S Model width_list: {model_s.width_list}")
    model_s.eval()
    with torch.no_grad():
        res_s = model_s(inputs) # Using same input
    print(f"Output for Small (SSSS) with input {inputs.shape}:")
    for i, r_s in enumerate(res_s):
        print(f"Stage {model_s.out_indices[i]}: {r_s.shape}")

    print("\nTesting SlabSwinTransformer_M2LL (Mixed Attention)")
    # attn_type=['M1','L','S','S'] means:
    # Layer 0 (depth 2): attn_type 'M1' -> 1 Linear block, 1 Standard block
    # Layer 1 (depth 2): attn_type 'L'  -> 2 Linear blocks
    # Layer 2 (depth 6): attn_type 'S'  -> 6 Standard blocks
    # Layer 3 (depth 2): attn_type 'S'  -> 2 Standard blocks
    model_m = SlabSwinTransformer_M2LL(img_size=224, patch_size=4, out_indices=(0,1,2,3),
                                       depths=[2,2,6,2], # Must match attn_type list length
                                       attn_type=['M1','L','S','S']).to(device)
    print(f"Mixed Model width_list: {model_m.width_list}")
    model_m.eval()
    with torch.no_grad():
        res_m = model_m(inputs) # Using same input
    print(f"Output for Mixed with input {inputs.shape}:")
    for i, r_m in enumerate(res_m):
        print(f"Stage {model_m.out_indices[i]}: {r_m.shape}")

    # Check if model can handle dynamic input sizes if APE is disabled or interpolated well
    print("\nTesting with a different input size (480x640):")
    inputs_diff = torch.randn((1, 3, 480, 640)).to(device)
    model_t.eval()
    with torch.no_grad():
        res_t_diff = model_t(inputs_diff)
    print(f"Output for Tiny (LLLL) with input {inputs_diff.shape}:")
    for i, r_t_d in enumerate(res_t_diff):
        print(f"Stage {model_t.out_indices[i]}: {r_t_d.shape}")