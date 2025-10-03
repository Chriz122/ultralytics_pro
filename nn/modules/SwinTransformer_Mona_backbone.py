# --------------------------------------------------------
# Swin Transformer with Mona Adapter (mmcv-free)
# Based on original Swin Transformer code:
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# Modified for Mona Adapter and mmcv removal
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init

import math
import logging # Using standard logging instead of mmcls logger
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# Set up basic logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Helper function similar to Code 2's update_weight
# More robust loading using load_state_dict(strict=False) is generally preferred
def update_weight(model_dict, weight_dict):
    """
    Loads weights from weight_dict into model_dict, matching keys and shapes.
    """
    temp_dict = {}
    loaded_count = 0
    total_count = len(model_dict)

    # Filter pretrained weights
    for k, v in weight_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                temp_dict[k] = v
                loaded_count += 1
            else:
                logger.warning(f"Shape mismatch for key {k}: model {model_dict[k].shape}, weights {v.shape}. Skipping.")
        # else:
        #     logger.warning(f"Key {k} not found in model state_dict. Skipping.") # Optional: report unexpected keys

    # Check for missing keys in the model that were not in the pretrained weights
    missing_keys = [k for k in model_dict.keys() if k not in temp_dict]
    if missing_keys:
        # Note: Keys related to 'my_module' are expected to be missing if loading standard Swin weights
        # Filter out expected missing keys if necessary
        adapter_keys = {name for name, _ in model_dict.items() if 'my_module' in name}
        reportable_missing = [k for k in missing_keys if k not in adapter_keys]
        if reportable_missing:
           logger.warning(f"Missing keys in loaded weights (potentially non-adapter): {reportable_missing}")
        pass # Suppress warnings about missing adapter keys

    unexpected_keys = [k for k in weight_dict.keys() if k not in model_dict]
    # if unexpected_keys:
    #    logger.warning(f"Unexpected keys in pretrained weights: {unexpected_keys}") # Optional: report unexpected keys

    logger.info(f'Loading weights: successfully loaded {loaded_count}/{total_count} items.')
    model_dict.update(temp_dict)
    return model_dict, missing_keys, unexpected_keys


class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # Using integer division // for padding consistency
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        # Combine convolutions and add identity
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x # Store intermediate result for residual connection

        x = self.projector(x)

        return identity + x

# Changed BaseModule to nn.Module
class Mona(nn.Module):
    def __init__(self,
                 in_dim,
                 factor=4): # factor is not used in the current implementation? Kept for signature consistency.
        super().__init__()

        # Assuming hidden_dim calculation was intended, like factor*some_base or fixed like 64
        hidden_dim = 64 # Using fixed hidden dim as in the original code
        self.project1 = nn.Linear(in_dim, hidden_dim)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(hidden_dim, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(hidden_dim) # Use hidden_dim here

        self.norm = nn.LayerNorm(in_dim)
        # Initialize gamma close to zero for minimal initial impact
        self.gamma = nn.Parameter(torch.zeros(in_dim)) # Changed init to zeros
        self.gammax = nn.Parameter(torch.ones(in_dim)) # This parameter seems unused in the forward?

    def forward(self, x, hw_shapes):
        """
        Args:
            x (Tensor): Input tensor shape (B, N, C)
            hw_shapes (tuple): Tuple containing (H, W) of the feature map
        """
        identity = x

        # Apply LayerNorm before the adapter branch
        x_norm = self.norm(x)

        project1 = self.project1(x_norm) # Apply projection to normalized input

        b, n, c_hidden = project1.shape
        if hw_shapes is None:
             # Attempt to infer H, W if not provided (requires N to be a perfect square)
             h = w = int(math.sqrt(n))
             if h * w != n:
                 raise ValueError(f"Cannot infer H, W from N={n}. Please provide hw_shapes.")
             logger.warning(f"hw_shapes not provided, inferred H=W={h} from N={n}")
        else:
            h, w = hw_shapes
            # Add check for consistency
            if n != h * w:
                logger.error(f"Shape inconsistency in Mona: N={n}, H={h}, W={w}")
                # Attempt to recover if possible, otherwise raise error
                if b * h * w * x.shape[-1] == identity.numel(): # Check if H, W match identity shape if flattened
                     n = h*w # Correct n based on hw_shapes if B, C dimensions match
                     x = x.view(b, n, -1) # Reshape x assuming H,W are correct
                     x_norm = self.norm(x)
                     project1 = self.project1(x_norm)
                else:
                    raise ValueError(f"Shape inconsistency in Mona: N={n} != H*W ({h}*{w}). Input shape {identity.shape}")


        # Reshape for Conv Adapter
        # B, N, C_hidden -> B, H, W, C_hidden -> B, C_hidden, H, W
        try:
            project1_reshaped = project1.view(b, h, w, c_hidden).permute(0, 3, 1, 2).contiguous()
        except RuntimeError as e:
            logger.error(f"Error reshaping in Mona: project1 shape {project1.shape}, target B={b}, H={h}, W={w}, C_hidden={c_hidden}. Error: {e}")
            raise e
        project1_conv = self.adapter_conv(project1_reshaped)
        # B, C_hidden, H, W -> B, H, W, C_hidden -> B, N, C_hidden
        project1_restored = project1_conv.permute(0, 2, 3, 1).contiguous().view(b, n, c_hidden)

        nonlinear = self.nonlinear(project1_restored)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        # Apply learnable scaling (gamma) to the adapter output before adding to identity
        # Original: self.norm(x) * self.gamma + x * self.gammax
        # Let's apply gamma to the adapter branch output, scaled by adapter factor concept.
        # Using gamma initialized near zero means the adapter branch starts small.
        return identity + project2 * self.gamma


class Mlp(nn.Module):
    """ Multilayer perceptron."""
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
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    B = int(windows.shape[0] / (num_windows_h * num_windows_w))
    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias."""
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
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) # Use indexing='ij' for torch >= 1.10
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

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            # Ensure mask dtype matches attention scores dtype
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0).to(attn.dtype)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block with Mona Adapters. """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, adapter_factor=8): # Added adapter_factor
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # Ensure window_size is tuple
        self.window_size = window_size if isinstance(window_size, tuple) else to_2tuple(window_size)
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size < self.window_size[1], "shift_size must in 0-window_size"


        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads, # Use tuple window_size
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

        # Initialize Mona adapters here
        self.my_module_1 = Mona(dim, factor=adapter_factor) # Pass factor if Mona uses it
        self.my_module_2 = Mona(dim, factor=adapter_factor)

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        # H and W are now set externally by BasicLayer before calling forward
        if self.H is None or self.W is None:
             # Fallback: Try to infer H, W if not set (e.g., during direct block testing)
             inferred_size = int(math.sqrt(L))
             if inferred_size * inferred_size == L:
                  self.H = self.W = inferred_size
                  logger.warning(f"H and W not set for SwinTransformerBlock, inferred H=W={self.H} from L={L}")
             else:
                 raise ValueError(f"H and W not set for SwinTransformerBlock and cannot be inferred from L={L}.")

        H, W = self.H, self.W # Use the H, W set by BasicLayer
        # Crucial check: Ensure L matches H*W from BasicLayer
        if L != H * W:
             raise ValueError(f"Shape mismatch in SwinTransformerBlock: Input L={L}, but BasicLayer provided H={H}, W={W}")


        shortcut = x
        x_norm1 = self.norm1(x) # Normalize first (B, L, C)

        # Reshape before padding/shifting
        try:
             x_view = x_norm1.view(B, H, W, C)
        except RuntimeError as e:
             logger.error(f"Error in SwinTransformerBlock view: x_norm1 shape {x_norm1.shape}, target B={B}, H={H}, W={W}, C={C}. Error: {e}")
             raise e


        # Pad feature maps to multiples of window size
        # Use tuple window_size here
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        if pad_r > 0 or pad_b > 0:
             x_pad = F.pad(x_view, (0, 0, pad_l, pad_r, pad_t, pad_b)) # Pad the normalized tensor
        else:
             x_pad = x_view
        _, Hp, Wp, _ = x_pad.shape

        # Cyclic shift
        if self.shift_size > 0:
            # Shift using tuple shifts if needed, assuming square shift for now
            shifted_x = torch.roll(x_pad, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x_pad
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size[0]) # Assuming square window for partition? No, should handle tuple.
        # window_partition expects int, timm version uses int. Let's assume square for now or fix partition/reverse.
        # Revisit this if non-square windows are used. For now, assume square window_size[0]==window_size[1].
        win_h, win_w = self.window_size
        x_windows = window_partition(shifted_x, win_h) # Use height for partitioning logic? Check impl. Needs testing for non-square.
                                                       # Let's assume square window_size = int(ws[0]) if ws[0]==ws[1] else ws
        window_size_int = self.window_size[0] # Assuming square for partition/reverse

        x_windows = x_windows.view(-1, window_size_int * window_size_int, C)  # nW*B, Wh*Ww, C


        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, Wh*Ww, C

        # Merge windows
        attn_windows = attn_windows.view(-1, window_size_int, window_size_int, C)
        # Use Hp, Wp calculated after padding
        shifted_x = window_reverse(attn_windows, window_size_int, Hp, Wp)  # B H' W' C

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_attn_out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_attn_out = shifted_x

        # Remove padding - Use original H, W
        if pad_r > 0 or pad_b > 0:
            x_attn_out = x_attn_out[:, :H, :W, :].contiguous()

        x_attn_out = x_attn_out.view(B, H * W, C)

        # First residual connection + DropPath + First Adapter
        x = shortcut + self.drop_path(x_attn_out)
        x = self.my_module_1(x, (H,W)) # Pass original H, W to adapter

        # Store identity for second residual connection
        identity = x
        x_norm2 = self.norm2(x) # Normalize before FFN

        # FFN + DropPath
        x_ffn = self.mlp(x_norm2)
        x = identity + self.drop_path(x_ffn)

        # Second Adapter
        x = self.my_module_2(x, (H,W)) # Pass original H, W to adapter

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        if L != H * W:
            raise ValueError(f"Input feature has wrong size L={L}, H={H}, W={W}")
        # Swin expects even dimensions for merging without padding issues
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # Pad input if dimensions are odd
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # Pad (right, bottom)
             x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
             H_pad, W_pad = x.shape[1], x.shape[2] # Store padded dimensions
        else:
             H_pad, W_pad = H, W # No padding needed

        # Extract patches
        x0 = x[:, 0::2, 0::2, :]  # B H_pad/2 W_pad/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H_pad/2 W_pad/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H_pad/2 W_pad/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H_pad/2 W_pad/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H_pad/2 W_pad/2 4*C

        # Calculate output dimensions based on original H, W (ceiling division)
        # Wh_out, Ww_out = (H + 1) // 2, (W + 1) // 2
        # Reshape using padded dimensions floor division
        x = x.view(B, (H_pad // 2) * (W_pad // 2) , 4 * C) # B (H_pad/2)*(W_pad/2) 4*C


        x = self.norm(x)
        x = self.reduction(x)

        # The returned x has dimensions corresponding to Wh_out, Ww_out
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage. """
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        # Ensure window_size is tuple
        self.window_size = window_size if isinstance(window_size, tuple) else to_2tuple(window_size)
        # Shift size usually int based on height/width, assume square for now
        self.shift_size = self.window_size[0] // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=self.window_size, # Pass tuple
                shift_size=0 if (i % 2 == 0) else self.shift_size, # Use self.shift_size
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        # calculate attention mask for SW-MSA
        # Use tuple window_size
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        # Use tuple shift_size if non-square needed, assume square shift for now
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size), # Assuming square shift
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Assume square window partition/reverse logic for mask generation
        window_size_int = self.window_size[0] # Assuming square for mask generation
        mask_windows = window_partition(img_mask, window_size_int) # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, window_size_int * window_size_int)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # Use -inf for masking in softmax
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float(0.0))

        # Store output of blocks before potential downsampling
        x_out = x
        for blk in self.blocks:
            blk.H, blk.W = H, W # Set H, W for the block before calling it
            if self.use_checkpoint:
                x_out = checkpoint.checkpoint(blk, x_out, attn_mask)
            else:
                x_out = blk(x_out, attn_mask)

        # Now x_out is the final output of the blocks for this layer
        # H, W are still the input dimensions for this layer

        if self.downsample is not None:
            # Perform downsampling on the output of the blocks (x_out)
            # Pass the correct input H, W to the downsample layer
            x_down = self.downsample(x_out, H, W)

            # Calculate output dimensions after downsampling using ceiling division on input H, W
            Wh_next, Ww_next = (H + 1) // 2, (W + 1) // 2
            # Return:
            # 1. Output of blocks (before downsample) - x_out
            # 2. Input H for this layer - H
            # 3. Input W for this layer - W
            # 4. Output after downsample (input for next layer) - x_down
            # 5. Output H for next layer - Wh_next
            # 6. Output W for next layer - Ww_next
            return x_out, H, W, x_down, Wh_next, Ww_next
        else:
            # No downsampling, output of blocks is input for next layer
            # Output dimensions are same as input dimensions
            # Return: x_out, H, W, x_out, H, W
            return x_out, H, W, x_out, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        B, C, H, W = x.size()
        # Pad input to be divisible by patch size
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        if pad_w > 0 or pad_h > 0:
             # Pad (left, right, top, bottom) - F.pad uses last dim first
             x = F.pad(x, (pad_l, pad_r, pad_t, pad_b)) # Correct padding format
             x = F.pad(x, (0, pad_w, 0, pad_h))


        x = self.proj(x)  # B C Ph Pw (Ph = Hp/patch_size, Pw = Wp/patch_size)
        Ph, Pw = x.size(2), x.size(3) # Patch resolution

        x = x.flatten(2).transpose(1, 2) # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)

        return x, Ph, Pw # Return flattened patches and patch resolution


# Changed BaseModule to nn.Module, removed BACKBONES decorator
class SwinTransformer_mona(nn.Module):
    """ Swin Transformer backbone with Mona adapters. (mmcv-free) """
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]


        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_size = self.patch_embed.patch_size # Store patch size tuple


        # absolute position embedding
        if self.ape:
            pretrain_img_size_tuple = to_2tuple(pretrain_img_size)
            patch_size_tuple = to_2tuple(patch_size) # Use tuple
            patches_resolution = [pretrain_img_size_tuple[0] // patch_size_tuple[0], pretrain_img_size_tuple[1] // patch_size_tuple[1]]

            num_patches = patches_resolution[0] * patches_resolution[1]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        else:
            self.absolute_pos_embed = None


        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size, # Pass original window_size arg
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)


        # add a norm layer for each output
        for i_layer in out_indices:
            # Norm layer should correspond to the output feature dimension of that layer
            norm_feature_dim = self.num_features[i_layer]
            layer = norm_layer(norm_feature_dim)
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # --- Add width_list calculation ---
        # Needs a dummy forward pass, ensure model is on correct device if needed
        # Use try-except to handle potential issues during init pass
        self.width_list = []
        try:
            # Use a size related to pretrain_img_size for the dummy input
            dummy_input_size = to_2tuple(self.pretrain_img_size)
            # Make sure size is divisible by patch_size * (2**(num_layers-1)) for downsampling
            # Calculate total downsampling factor based on patch size and number of PatchMerging layers
            total_stride_h = self.patch_size[0] * (2**(self.num_layers - 1))
            total_stride_w = self.patch_size[1] * (2**(self.num_layers - 1))

            dummy_h = int(math.ceil(dummy_input_size[0] / total_stride_h) * total_stride_h)
            dummy_w = int(math.ceil(dummy_input_size[1] / total_stride_w) * total_stride_w)

            dummy_input = torch.randn(1, in_chans, dummy_h, dummy_w)
            logger.info(f"Calculating width_list with dummy input size: {dummy_input.shape}")

            # Run dummy forward (ensure requires_grad=False if not needed)
            self.eval() # Set to eval mode for dummy pass (disables dropout etc.)
            with torch.no_grad():
                 outputs = self.forward(dummy_input)
                 self.width_list = [out.size(1) for out in outputs] # C is dim 1 after permute
            self.train() # Set back to train mode by default
            logger.info(f"Calculated output widths: {self.width_list}")
        except Exception as e:
             logger.error(f"Failed to calculate width_list during initialization: {e}", exc_info=True)
             # Set default or keep empty based on requirements
             # Fallback: use num_features *if* out_indices are standard 0,1,2,3
             if list(out_indices) == list(range(self.num_layers)):
                  self.width_list = self.num_features
                  logger.warning(f"Using num_features as fallback for width_list: {self.width_list}")
             else: # Cannot easily infer if out_indices are sparse
                  self.width_list = [self.num_features[i] for i in out_indices]
                  logger.warning(f"Using num_features for specified out_indices as fallback for width_list: {self.width_list}")



        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if hasattr(self.patch_embed, 'proj'): # Check if proj exists
                self.patch_embed.eval()
                for param in self.patch_embed.parameters():
                    param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            if self.absolute_pos_embed is not None:
                 self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            # Layer indices are 0-based, frozen_stages is 1-based count?
            # Original logic freezes layers[0] when frozen_stages=2
            # Freeze layers up to (frozen_stages - 1) -> Correct logic is range(0, frozen_stages-1)
            # Example: frozen_stages=2 freezes layer 0. frozen_stages=3 freezes layers 0, 1.
            # frozen_stages=1 should only freeze patch_embed/APE.
            for i in range(self.frozen_stages - 1): # Correct range
                if i < len(self.layers): # Ensure index is valid
                    m = self.layers[i]
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

    def _init_weights(self, m):
        """ Initialize weights for linear and layernorm layers. """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # Renamed from init_weights to match usage pattern of applying after init
    def load_pretrained(self, pretrained_path=None):
        """
        Initialize weights from scratch or load pretrained weights.
        Also handles freezing non-adapter parameters after loading/initialization.
        """
        if isinstance(pretrained_path, str):
            logger.info(f"Initializing weights using _init_weights before loading pretrained.")
            self.apply(self._init_weights) # Initialize all weights first

            logger.info(f"Loading pretrained weights from: {pretrained_path}")
            try:
                # Load checkpoint
                # Use map_location='cpu' to avoid GPU memory issues if loading on different device
                checkpoint = torch.load(pretrained_path, map_location='cpu')

                # Check if weights are nested (common formats)
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Load weights using strict=False
                missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
                logger.info(f"Weights loaded from {pretrained_path}")
                if missing_keys:
                    # Filter out expected missing adapter keys before reporting
                    adapter_keys = {name for name, _ in self.named_parameters() if 'my_module' in name}
                    reportable_missing = [k for k in missing_keys if k not in adapter_keys]
                    if reportable_missing:
                        logger.warning(f"Missing keys NOT related to adapters: {reportable_missing}")
                    # Log missing adapter keys at info level if desired
                    # missing_adapter_keys = [k for k in missing_keys if k in adapter_keys]
                    # if missing_adapter_keys:
                    #     logger.info(f"Missing adapter keys (expected): {missing_adapter_keys}")

                if unexpected_keys:
                    # Filter out known differences if needed (e.g., head vs no head)
                    reportable_unexpected = [k for k in unexpected_keys if 'head' not in k] # Example filter
                    if reportable_unexpected:
                         logger.warning(f"Unexpected keys in pretrained weights: {reportable_unexpected}")

            except FileNotFoundError:
                logger.error(f"Pretrained weights file not found: {pretrained_path}")
                raise
            except Exception as e:
                logger.error(f"Error loading pretrained weights: {e}", exc_info=True)
                raise

        elif pretrained_path is None:
            logger.info("Initializing weights from scratch using _init_weights.")
            self.apply(self._init_weights) # Apply standard initialization
        else:
            raise TypeError('pretrained must be a str or None')

        # --- Freeze non-adapter parameters ---
        # This should happen *after* weights are loaded/initialized
        logger.info("Freezing non-adapter parameters ('my_module' not in name).")
        frozen_count = 0
        trainable_count = 0
        total_count = 0
        for name, param in self.named_parameters():
            total_count += 1
            if 'my_module' not in name:
                param.requires_grad = False
                frozen_count += 1
            else:
                 param.requires_grad = True # Ensure adapter parameters are trainable
                 trainable_count += 1
        logger.info(f"Total params: {total_count}. Froze {frozen_count}. Trainable (adapter): {trainable_count}.")

        # Re-apply stage freezing AFTER loading weights and freezing non-adapters
        self._freeze_stages()


    def forward(self, x):
        """Forward function."""
        x, Wh, Ww = self.patch_embed(x) # Get patch resolution (Wh, Ww) from embedding
        B, L, C = x.shape

        if self.ape:
             if self.absolute_pos_embed is not None:
                 # ... (APE logic remains the same) ...
                 if L != self.absolute_pos_embed.shape[1]:
                      # ... (interpolation/padding logic) ...
                      # Corrected APE handling
                      num_patches_ape = self.absolute_pos_embed.shape[1]
                      embed_dim_ape = self.absolute_pos_embed.shape[2]
                      pretrain_h = pretrain_w = int(math.sqrt(num_patches_ape))
                      if pretrain_h * pretrain_w == num_patches_ape:
                           ape_grid = self.absolute_pos_embed.view(1, pretrain_h, pretrain_w, embed_dim_ape).permute(0, 3, 1, 2)
                           absolute_pos_embed_resized = F.interpolate(ape_grid, size=(Wh, Ww), mode='bicubic', align_corners=False)
                           absolute_pos_embed_final = absolute_pos_embed_resized.permute(0, 2, 3, 1).flatten(1, 2)
                           x = x + absolute_pos_embed_final
                      else:
                           logger.warning(f"Cannot interpolate APE: Pretrained num_patches {num_patches_ape} not square. Using truncation/padding.")
                           if L < num_patches_ape:
                               x = x + self.absolute_pos_embed[:, :L, :]
                           else:
                               ape_padded = torch.cat([self.absolute_pos_embed, torch.zeros(1, L - num_patches_ape, C, device=x.device)], dim=1)
                               x = x + ape_padded
                 else:
                      x = x + self.absolute_pos_embed
             else:
                 logger.warning("APE is enabled but self.absolute_pos_embed is None.")

        x = self.pos_drop(x)

        outs = []
        # Initialize H, W for the first layer's input using dimensions from patch_embed
        H, W = Wh, Ww
        for i in range(self.num_layers):
            layer = self.layers[i]
            # Pass the correct H, W for the input of this layer
            # layer returns: x_out, H_out, W_out, x_next, Wh_next, Ww_next
            x_out, H_out, W_out, x_next, Wh_next, Ww_next = layer(x, H, W)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                # Apply norm to the output of the last block (x_out)
                x_out_norm = norm_layer(x_out)
                # Reshape using the dimensions corresponding to x_out (H_out, W_out)
                out = x_out_norm.view(B, H_out, W_out, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

            # Prepare for the *next* layer:
            x = x_next # Use the potentially downsampled output as input
            H, W = Wh_next, Ww_next # Update H, W to the dimensions of x_next

        # --- CHANGE THIS LINE ---
        # return tuple(outs)  # Old line - Returns a tuple
        return outs          # New line - Returns the list directly
        # --- END CHANGE ---


    def train(self, mode=True):
        """Convert the model into training mode while keeping layers freezed."""
        super(SwinTransformer_mona, self).train(mode)
        # Ensure frozen stages remain frozen during training
        self._freeze_stages()
        # Ensure non-adapter params remain frozen
        # logger.debug("Re-applying non-adapter freeze during train() call.") # Optional debug log
        for name, param in self.named_parameters():
            if 'my_module' not in name:
                 # Check if it wasn't already frozen by _freeze_stages
                 if param.requires_grad:
                      param.requires_grad = False


# --------------------------------------------------------------------------
# Factory Functions for different SwinTransformer_mona sizes
# --------------------------------------------------------------------------

def SwinTransformer_mona_Tiny(pretrained=None, **kwargs):
    """ Swin-Tiny model with Mona adapters """
    # Default Tiny parameters
    model_kwargs = dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        **kwargs # Allow overriding defaults
    )
    model = SwinTransformer_mona(**model_kwargs)

    # Load pretrained and freeze backbone AFTER model initialization
    model.load_pretrained(pretrained_path=pretrained)
    return model

def SwinTransformer_mona_Small(pretrained=None, **kwargs):
    """ Swin-Small model with Mona adapters """
    model_kwargs = dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        **kwargs
    )
    model = SwinTransformer_mona(**model_kwargs)
    model.load_pretrained(pretrained_path=pretrained)
    return model

def SwinTransformer_mona_Base(pretrained=None, **kwargs):
    """ Swin-Base model with Mona adapters """
    model_kwargs = dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7, # Often 7 or 12 for Base/Large
        **kwargs
    )
    model = SwinTransformer_mona(**model_kwargs)
    model.load_pretrained(pretrained_path=pretrained)
    return model

def SwinTransformer_mona_Large(pretrained=None, **kwargs):
    """ Swin-Large model with Mona adapters """
    model_kwargs = dict(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7, # Often 7 or 12 for Base/Large
        **kwargs
    )
    model = SwinTransformer_mona(**model_kwargs)
    model.load_pretrained(pretrained_path=pretrained)
    return model


# Example Usage
if __name__ == '__main__':
    # Example: Create a Tiny model and load pretrained weights (if available)
    # Replace 'path/to/swin_tiny_patch4_window7_224.pth' with your actual checkpoint path
    pretrained_path = None # Set to path like 'swin_tiny_patch4_window7_224.pth' if you have it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- Instantiate using factory function ---
    try:
        logger.info("\n--- Testing Tiny model ---")
        model = SwinTransformer_mona_Tiny(
            pretrained=pretrained_path,
            pretrain_img_size=224,
            ape=False, # Absolute Position Embedding
            patch_norm=True,
            frozen_stages=-1, # Set frozen stages if needed (e.g., 2 to freeze patch_embed and layer 0)
            out_indices=(0, 1, 2, 3) # Which layer outputs to return
        ).to(device)

        # --- Check trainable parameters ---
        print("\nTrainable parameters (Tiny):")
        total_params = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                # print(f"  {name}: {param.shape}") # Uncomment to see all trainable params
                trainable_params += param.numel()
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters (adapters): {trainable_params:,}")
        if total_params > 0:
            print(f"Percentage trainable: {100 * trainable_params / total_params:.4f}%")


        # --- Test forward pass ---
        # Input size should match pretrain_img_size or be compatible
        test_input = torch.randn(2, 3, 224, 224).to(device) # Example batch size 2
        model.eval() # Set to evaluation mode
        with torch.no_grad():
            outputs = model(test_input)

        print("\nOutput shapes (Tiny):")
        for i, out in enumerate(outputs):
            print(f"  Output {i}: {out.shape}")

        print("\nCalculated width list (Tiny):", model.width_list)

    except Exception as e:
        print(f"\nError during Tiny model test: {e}")
        import traceback
        traceback.print_exc()


    # --- Example with different size and APE ---
    try:
        logger.info("\n--- Testing Base model with APE ---")
        pretrained_path_base = None # Path to Swin-Base weights if available
        model_base = SwinTransformer_mona_Base(
            pretrained=pretrained_path_base,
            pretrain_img_size=224, # Or 384 if using weights pretrained on that size
            # window_size=7,      # REMOVED - set inside factory function
            ape=True,             # Enable Absolute Position Embedding
            frozen_stages=-1,
            out_indices=(3,)      # Only get the last layer output
        ).to(device)

        # Check trainable params for Base model
        print("\nTrainable parameters (Base):")
        total_params_base = sum(p.numel() for p in model_base.parameters())
        trainable_params_base = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params_base:,}")
        print(f"Trainable parameters (adapters): {trainable_params_base:,}")
        if total_params_base > 0:
             print(f"Percentage trainable: {100 * trainable_params_base / total_params_base:.4f}%")


        # Test forward pass for Base model
        test_input_base = torch.randn(1, 3, 224, 224).to(device) # Match pretrain_img_size
        model_base.eval()
        with torch.no_grad():
            outputs_base = model_base(test_input_base)
        print("\nBase model output shapes:")
        for i, out in enumerate(outputs_base):
            print(f"  Output {i}: {out.shape}")
        print("\nBase model calculated width list:", model_base.width_list)

    except Exception as e:
        print(f"\nError during Base model test: {e}")
        import traceback
        traceback.print_exc()