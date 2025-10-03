"""
EfficientFormer_v2
"""
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict
import itertools
import numpy as np
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

__all__ = ['efficientformerv2_s0', 'efficientformerv2_s1', 'efficientformerv2_s2', 'efficientformerv2_l']

EfficientFormer_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}

EfficientFormer_depth = {
    'L': [5, 5, 15, 10],  # 26m 83.3%
    'S2': [4, 4, 12, 8],  # 12m
    'S1': [3, 3, 9, 6],  # 79.0
    'S0': [2, 2, 6, 4],  # 75.7
}

# 26m
expansion_ratios_L = {
    '0': [4, 4, 4, 4, 4],
    '1': [4, 4, 4, 4, 4],
    '2': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
}

# 12m
expansion_ratios_S2 = {
    '0': [4, 4, 4, 4],
    '1': [4, 4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 3, 3, 3, 3, 4, 4],
}

# 6.1m
expansion_ratios_S1 = {
    '0': [4, 4, 4],
    '1': [4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 4, 4, 4],
    '3': [4, 4, 3, 3, 4, 4],
}

# 3.5m
expansion_ratios_S0 = {
    '0': [4, 4],
    '1': [4, 4],
    '2': [4, 3, 3, 3, 4, 4],
    '3': [4, 3, 3, 4],
}


class Attention4D(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 act_layer=nn.ReLU, # Changed default to nn.ReLU like in EfficientFormerV1
                 stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads # 8 * 32 = 256

        self.attn_ratio = attn_ratio # default 4
        self.d = int(attn_ratio * key_dim) # 4 * 32 = 128
        self.dh = self.d * num_heads # 128 * 8 = 1024

        self.stride = stride
        # Note: initialization resolution values (resolution, N) are used for RPE size ONLY
        # The actual N for reshape is calculated dynamically in forward
        if stride is not None:
            # This resolution is used to initialize self.attention_biases
            self.resolution = math.ceil(resolution / stride)
            # LayerNorm and Conv2d instead of BN and Conv2d + BN
            # Follows original EfficientFormerV2 more closely
            # self.norm = nn.LayerNorm(dim, eps=1e-6) # V1 uses LayerNorm here
            # self.norm = nn.BatchNorm2d(dim) # V2 uses BN
            self.stride_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                nn.BatchNorm2d(dim),
            )
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False)
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        self.N = self.resolution ** 2 # Used for attention_biases size

        # Removed: self.N2 = self.N (unused typo)

        # Use Conv2d + BN for q, k, v projections
        self.q = nn.Sequential(
            nn.Conv2d(dim, nh_kd, 1),
            nn.BatchNorm2d(nh_kd),
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, nh_kd, 1),
            nn.BatchNorm2d(nh_kd),
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.dh, 1),
            nn.BatchNorm2d(self.dh),
        )
        self.v_local = nn.Sequential(
            nn.Conv2d(self.dh, self.dh, kernel_size=3, stride=1, padding=1, groups=self.dh),
            nn.BatchNorm2d(self.dh),
        )

        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)

        self.proj = nn.Sequential(
            # act_layer(), # Original V2 doesn't have activation here
            nn.Conv2d(self.dh, dim, 1),
            nn.BatchNorm2d(dim),
        )

        # Setup RPE table
        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        N_points = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        # Ensure attention_bias_idxs is registered as a buffer
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_points, N_points), persistent=False)

        # Initialize attention biases buffer for inference
        self.ab = None
        if not self.training:
             self.ab = self.attention_biases[:, self.attention_bias_idxs]

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
             # Clear ab buffer during training
             self.ab = None
        elif not mode and (self.ab is None):
             # Create ab buffer for inference if it doesn't exist
             self.ab = self.attention_biases[:, self.attention_bias_idxs]


    def forward(self, x):  # x (B, C, H, W)
        B, C, H, W = x.shape
        # Pass input through LayerNorm/BatchNorm first? V1 does, V2 doesn't explicitly show it here
        # x = self.norm(x) # If using LayerNorm like V1

        if self.stride_conv is not None:
            # Apply stride conv
            x = self.stride_conv(x)
            # Get the new H, W after stride
            H_s, W_s = x.shape[-2:]
        else:
            H_s, W_s = H, W

        # Calculate N dynamically based on the actual feature map size after striding
        N_actual = H_s * W_s

        # --- Process Q, K, V ---
        q = self.q(x) # (B, nh_kd, H_s, W_s)
        k = self.k(x) # (B, nh_kd, H_s, W_s)
        v = self.v(x) # (B, dh, H_s, W_s)
        v_local = self.v_local(v) # (B, dh, H_s, W_s)

        # Reshape for Attention - Use N_actual
        q = q.flatten(2).reshape(B, self.num_heads, self.key_dim, N_actual).permute(0, 1, 3, 2) # (B, num_heads, N_actual, key_dim)
        k = k.flatten(2).reshape(B, self.num_heads, self.key_dim, N_actual).permute(0, 1, 2, 3) # (B, num_heads, key_dim, N_actual)
        v = v.flatten(2).reshape(B, self.num_heads, self.d, N_actual).permute(0, 1, 3, 2)       # (B, num_heads, N_actual, d)

        # --- Calculate Attention ---
        attn = (q @ k) * self.scale # (B, num_heads, N_actual, N_actual)

        # --- Add Relative Position Encoding ---
        # Check if N_actual matches the size expected by attention_bias_idxs (self.N)
        if N_actual == self.N:
            if self.training:
                attn = attn + self.attention_biases[:, self.attention_bias_idxs]
            else:
                # Ensure ab is initialized and has correct shape
                if self.ab is None or self.ab.shape[-1] != N_actual:
                     self.ab = self.attention_biases[:, self.attention_bias_idxs]
                     # print(f"Warning: Recomputing RPE buffer (ab) in Attention4D inference. N_actual={N_actual}, self.N={self.N}") # Optional warning
                attn = attn + self.ab
        # else:
        #     # If N_actual doesn't match self.N, RPE cannot be applied as configured.
        #     # This might happen if input resolution is different from init resolution.
        #     # Options: 1) Skip RPE, 2) Interpolate RPE, 3) Error out.
        #     # Current behavior: Skips RPE if sizes mismatch.
        #     print(f"Warning: Skipping RPE in Attention4D due to shape mismatch. N_actual={N_actual}, self.N={self.N}") # Optional warning
        #     pass # Skip RPE

        # Talking Heads
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)

        # --- Apply Attention to V ---
        x_attn = (attn @ v) # (B, num_heads, N_actual, d)

        # Reshape back to spatial dimensions H_s, W_s
        out = x_attn.permute(0, 1, 3, 2).reshape(B, self.dh, H_s, W_s) # (B, dh, H_s, W_s)

        # Add local context projection
        out = out + v_local # (B, dh, H_s, W_s)

        # Final projection
        out = self.proj(out) # (B, dim, H_s, W_s)

        # Upsample if strided
        if self.upsample is not None:
            out = self.upsample(out) # (B, dim, H, W)

        return out


def stem(in_chs, out_chs, act_layer=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        act_layer(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        # V2 doesn't show activation after the second stem conv+BN
        # act_layer(),
    )


class LGQuery(torch.nn.Module):
    """ Lightweight Global Query used in Attention4DDownsample """
    def __init__(self, in_dim, out_dim, resolution1, resolution2): # resolution args aren't strictly needed
        super().__init__()
        # self.resolution1 = resolution1 # Not used
        # self.resolution2 = resolution2 # Not used
        # AvgPool2d kernel_size should be stride to halve spatial dim, padding=0
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # Corrected pooling
        self.local = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim),
            # BN after local conv? V2 diagram suggests it's part of the block
            # nn.BatchNorm2d(in_dim) # Adding BN here if needed
            )
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        local_q = self.local(x)
        pool_q = self.pool(x)
        # Pad pooled query if spatial dimensions are odd before pooling
        if local_q.shape[-2:] != pool_q.shape[-2:]:
             pad_h = local_q.shape[-2] - pool_q.shape[-2]
             pad_w = local_q.shape[-1] - pool_q.shape[-1]
             pool_q = F.pad(pool_q, (0, pad_w, 0, pad_h))

        q = local_q + pool_q
        q = self.proj(q)
        return q


class Attention4DDownsample(torch.nn.Module):
    def __init__(self, dim=384, key_dim=16, num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 out_dim=None,
                 act_layer=nn.ReLU, # Changed default
                 ):
        super().__init__()

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads # e.g., 8 * 16 = 128

        self.d = int(attn_ratio * key_dim) # e.g., 4 * 16 = 64
        self.dh = self.d * num_heads # e.g., 64 * 8 = 512
        self.attn_ratio = attn_ratio

        self.out_dim = out_dim if out_dim is not None else dim

        # --- Resolutions for RPE setup ---
        # Resolution of the input feature map (K, V)
        self.resolution = resolution
        # Resolution of the output feature map (Q) - halved
        self.resolution2 = math.ceil(self.resolution / 2)
        self.N = self.resolution ** 2
        self.N2 = self.resolution2 ** 2 # Query sequence length (downsampled)

        # --- Layers ---
        # Query projection includes downsampling
        self.q = LGQuery(dim, nh_kd, self.resolution, self.resolution2)

        # Key and Value projections maintain input resolution
        self.k = nn.Sequential(
            nn.Conv2d(dim, nh_kd, 1),
            nn.BatchNorm2d(nh_kd),
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.dh, 1),
            nn.BatchNorm2d(self.dh),
        )
        # Local context projection includes downsampling (stride=2)
        self.v_local = nn.Sequential(
            nn.Conv2d(self.dh, self.dh, kernel_size=3, stride=2, padding=1, groups=self.dh),
            nn.BatchNorm2d(self.dh),
        )

        # Final projection to output dimension
        self.proj = nn.Sequential(
            act_layer(), # Activation before final projection
            nn.Conv2d(self.dh, self.out_dim, 1),
            nn.BatchNorm2d(self.out_dim),
        )

        # --- Relative Position Encoding Setup ---
        points = list(itertools.product(range(self.resolution), range(self.resolution))) # K, V points
        points_ = list(itertools.product(range(self.resolution2), range(self.resolution2))) # Q points
        N_kv_points = len(points)
        N_q_points = len(points_)
        attention_offsets = {}
        idxs = []
        # Calculate stride factor for coordinate mapping
        stride = self.resolution / self.resolution2 # Approximate stride

        for i, p1 in enumerate(points_): # Query points (downsampled grid)
            for j, p2 in enumerate(points): # Key points (original grid)
                # Map query coordinate back to approximate original grid coordinate
                p1_mapped_y = p1[0] * stride
                p1_mapped_x = p1[1] * stride
                # Calculate offset
                offset_y = abs(p1_mapped_y - p2[0])
                offset_x = abs(p1_mapped_x - p2[1])
                # Use floor/round/ceil for offsets? Paper doesn't specify, use floor for simplicity
                offset = (math.floor(offset_y), math.floor(offset_x))

                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        # Register buffer with correct shape N_q_points x N_kv_points
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_q_points, N_kv_points), persistent=False)

        # Initialize attention biases buffer for inference
        self.ab = None
        if not self.training:
             self.ab = self.attention_biases[:, self.attention_bias_idxs]

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
             # Clear ab buffer during training
             self.ab = None
        elif not mode and (self.ab is None):
             # Create ab buffer for inference if it doesn't exist
             self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B, C, H, W)
        B, C, H, W = x.shape

        # --- Process Q (includes downsampling) ---
        q_out = self.q(x) # (B, nh_kd, H/2, W/2)
        B_q, C_q, H_q, W_q = q_out.shape
        Nq_actual = H_q * W_q
        q = q_out.flatten(2).reshape(B, self.num_heads, self.key_dim, Nq_actual).permute(0, 1, 3, 2) # (B, nH, Nq, d)

        # --- Process K (maintains resolution) ---
        k_out = self.k(x) # (B, nh_kd, H, W)
        B_k, C_k, H_k, W_k = k_out.shape
        Nk_actual = H_k * W_k
        k = k_out.flatten(2).reshape(B, self.num_heads, self.key_dim, Nk_actual).permute(0, 1, 2, 3) # (B, nH, d, Nk)

        # --- Process V (maintains resolution for attn, downsamples for v_local) ---
        v_out = self.v(x) # (B, dh, H, W)
        B_v, C_v, H_v, W_v = v_out.shape
        Nv_actual = H_v * W_v
        v = v_out.flatten(2).reshape(B, self.num_heads, self.d, Nv_actual).permute(0, 1, 3, 2) # (B, nH, Nv, d_h)
        v_local = self.v_local(v_out) # (B, dh, H/2, W/2) - apply local conv to original v

        # --- Calculate Attention ---
        attn = (q @ k) * self.scale # (B, nH, Nq, Nk)

        # --- Add Relative Position Encoding ---
        # Check if dynamic Nq_actual, Nk_actual match sizes expected by RPE table (N2, N)
        if Nq_actual == self.N2 and Nk_actual == self.N:
            if self.training:
                attn = attn + self.attention_biases[:, self.attention_bias_idxs]
            else:
                # Ensure ab is initialized and has correct shape
                if self.ab is None or self.ab.shape[-2] != Nq_actual or self.ab.shape[-1] != Nk_actual:
                     self.ab = self.attention_biases[:, self.attention_bias_idxs]
                     # print(f"Warning: Recomputing RPE buffer (ab) in Attention4DDownsample inference. Nq={Nq_actual}, Nk={Nk_actual}, self.N2={self.N2}, self.N={self.N}") # Optional warning
                attn = attn + self.ab
        # else:
            # print(f"Warning: Skipping RPE in Attention4DDownsample due to shape mismatch. Nq={Nq_actual}, Nk={Nk_actual}, self.N2={self.N2}, self.N={self.N}") # Optional warning
            # pass # Skip RPE

        attn = attn.softmax(dim=-1)

        # --- Apply Attention to V ---
        x_attn = (attn @ v) # (B, nH, Nq, d_h)

        # Reshape back to spatial dimensions H_q, W_q
        out = x_attn.permute(0, 1, 3, 2).reshape(B, self.dh, H_q, W_q) # (B, dh, H/2, W/2)

        # Pad v_local if needed due to rounding differences in conv vs. LGQuery pooling
        if out.shape[-2:] != v_local.shape[-2:]:
             pad_h = out.shape[-2] - v_local.shape[-2]
             pad_w = out.shape[-1] - v_local.shape[-1]
             v_local = F.pad(v_local, (0, pad_w, 0, pad_h))

        # Add local context projection
        out = out + v_local # (B, dh, H/2, W/2)

        # Final projection
        out = self.proj(out) # (B, out_dim, H/2, W/2)

        return out


class Embedding(nn.Module):
    """ Patch Embedding - Uses different strategies based on flags """
    def __init__(self, patch_size=3, stride=2, padding=1,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d,
                 light=False, asub=False, resolution=None, act_layer=nn.ReLU, attn_block=Attention4DDownsample):
        super().__init__()
        self.light = light
        self.asub = asub # "Attention Substitution" for downsampling in later stages

        if self.light:
            # Light embedding (not typically used in standard S0-L models)
            self.new_proj = nn.Sequential(
                nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=2, padding=1, groups=in_chans),
                nn.BatchNorm2d(in_chans),
                nn.Hardswish(), # Note: V2 uses Hardswish here
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0),
                norm_layer(embed_dim), # Uses provided norm_layer
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=2, padding=0),
                norm_layer(embed_dim)
            )
        elif self.asub:
            # Attention Substitution Block (for Stages 2->3, 3->4)
            # Uses Attention4DDownsample + Conv2d
            self.attn = attn_block(dim=in_chans, out_dim=embed_dim,
                                   resolution=resolution, act_layer=act_layer,
                                   # Key dim 16 for downsampling attn, as per paper diagram
                                   key_dim=16, attn_ratio=4)
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            # Conv branch (parallel to attention)
            self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
            self.bn = norm_layer(embed_dim) if norm_layer else nn.Identity()
            # No activation shown after BN in V2 diagram for this branch
        else:
            # Standard Conv Embedding (for Stages 0->1, 1->2)
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
            # No activation shown after Norm in V2 diagram

    def forward(self, x):
        if self.light:
            out = self.new_proj(x) + self.skip(x)
        elif self.asub:
            # Parallel Attention and Conv branches
            out_attn = self.attn(x)
            out_conv = self.conv(x)
            out_conv = self.bn(out_conv)
            out = out_attn + out_conv # Summing the parallel paths
        else:
            # Standard Conv Embedding
            x = self.proj(x)
            out = self.norm(x)
        return out


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions, optionally with a middle 3x3 DWConv (MetaBlock).
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.norm1 = nn.BatchNorm2d(hidden_features) # BN before Act
        self.act = act_layer()

        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features) # BN before Act
            # self.mid_act = act_layer() # V2 diagram shows Act *after* mid_norm

        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.norm2 = nn.BatchNorm2d(out_features) # BN after fc2 (before residual)

        self.drop = nn.Dropout(drop) # Dropout usually applied after activations or final output

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        # Input projection
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        # x = self.drop(x) # Dropout pos 1

        # Middle DWConv block
        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid) # Activation after mid_norm
            # x = self.drop(x) # Dropout pos 2

        # Output projection
        x = self.fc2(x)
        x = self.norm2(x)
        # x = self.drop(x) # Dropout pos 3 (most common for MLP output before residual)
        return x


class AttnFFN(nn.Module):
    """ Attention + FFN Block (used in stages 2, 3) """
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.ReLU, # Changed default
                 norm_layer=nn.BatchNorm2d, # V2 uses BN in MLP
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 resolution=7, stride=None):

        super().__init__()

        # --- Attention Branch ---
        # Note: V2 doesn't show normalization before the attention block itself
        self.token_mixer = Attention4D(dim, resolution=resolution, act_layer=act_layer, stride=stride,
                                       key_dim=32, attn_ratio=4) # Defaults for stage 2/3 attn

        # --- MLP Branch ---
        mlp_hidden_dim = int(dim * mlp_ratio)
        # Note: V2 doesn't show normalization before the MLP block itself
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True) # Always use mid_conv (MetaBlock)

        # --- DropPath and LayerScale ---
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            # Ensure dimensions match the input tensor (B, C, H, W) -> need (1, C, 1, 1)
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(1, dim, 1, 1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(1, dim, 1, 1), requires_grad=True)

    def forward(self, x):
        # Apply branches with residual, drop_path, and layer_scale
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


class FFN(nn.Module):
    """ FFN Block (used in stages 0, 1) - Essentially just the MLP part """
    def __init__(self, dim, pool_size=None, mlp_ratio=4., # pool_size not used in V2 FFN
                 act_layer=nn.ReLU, # Changed default
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        # --- MLP Branch ---
        mlp_hidden_dim = int(dim * mlp_ratio)
        # Note: V2 doesn't show normalization before the MLP block itself
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True) # Always use mid_conv (MetaBlock)

        # --- DropPath and LayerScale ---
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            # Ensure dimensions match the input tensor (B, C, H, W) -> need (1, C, 1, 1)
            self.layer_scale_2 = nn.Parameter( # Only one scale factor needed for FFN-only block
                layer_scale_init_value * torch.ones(1, dim, 1, 1), requires_grad=True)

    def forward(self, x):
        # Apply MLP branch with residual, drop_path, and layer_scale
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x


def eformer_block(dim, index, layers,
                  pool_size=None, mlp_ratio=4., # pool_size not used
                  act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, # Changed defaults
                  drop_rate=.0, drop_path_rate=0.,
                  use_layer_scale=True, layer_scale_init_value=1e-5,
                  vit_num=1, resolution=7, e_ratios=None):
    """ Creates a stage of EfficientFormerV2 blocks. """
    blocks = []
    # Calculate the starting block index for drop path rate calculation
    start_block_idx = sum(layers[:index])
    total_blocks = sum(layers)

    for block_idx in range(layers[index]):
        # Calculate drop path rate for the current block
        block_dpr = drop_path_rate * (start_block_idx + block_idx) / (total_blocks - 1)

        # Get expansion ratio for the current block
        current_mlp_ratio = e_ratios[str(index)][block_idx]

        # Determine block type: AttnFFN or FFN
        # AttnFFN used in later blocks of stages 2 and 3
        is_attn_block = (index >= 2 and block_idx >= layers[index] - vit_num)

        if is_attn_block:
            # Determine stride for Attention4D within AttnFFN
            # Stride=2 only for the *first* AttnFFN block in Stage 2 (index=2)
            attn_stride = 2 if index == 2 and block_idx == layers[index] - vit_num else None
            # print(f"Stage {index}, Block {block_idx}: AttnFFN block, stride={attn_stride}, resolution={resolution}")
            blocks.append(AttnFFN(
                dim, mlp_ratio=current_mlp_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                resolution=resolution,
                stride=attn_stride,
            ))
        else:
            # print(f"Stage {index}, Block {block_idx}: FFN block, resolution={resolution}")
            blocks.append(FFN(
                dim, mlp_ratio=current_mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
    blocks = nn.Sequential(*blocks)
    return blocks


class EfficientFormerV2(nn.Module):
    """ EfficientFormerV2 model """
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, # Not used directly, passed via e_ratios
                 downsamples=None, # Indicates if downsampling occurs *after* the stage
                 pool_size=None, # Not used in V2 blocks
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, # Changed defaults
                 num_classes=1000, # Only used if not fork_feat
                 down_patch_size=3, down_stride=2, down_pad=1, # For Embedding layers
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=True, # Return features at different stages
                 vit_num=0, # Number of AttnFFN blocks at the end of stages 2, 3
                 resolution=640, # Expected input resolution (affects RPE init)
                 e_ratios=None, # Expansion ratios dict
                 **kwargs): # Absorb extra kwargs
        super().__init__()

        self.fork_feat = fork_feat
        self.num_classes = num_classes # Store for potential future use

        # --- Stem ---
        self.patch_embed = stem(3, embed_dims[0], act_layer=act_layer)
        current_resolution = resolution // 4 # After stem (stride 2x2)

        # --- Build Stages ---
        network = []
        self.network = nn.ModuleList() # Use ModuleList for proper registration
        for i in range(len(layers)):
            # Create stage blocks
            stage_resolution = math.ceil(resolution / (2 ** (i + 2))) # Resolution entering this stage
            stage = eformer_block(embed_dims[i], i, layers,
                                  # pool_size=pool_size, # Not needed
                                  # mlp_ratio=mlp_ratios, # Use e_ratios instead
                                  act_layer=act_layer, norm_layer=norm_layer,
                                  drop_rate=drop_rate,
                                  drop_path_rate=drop_path_rate,
                                  use_layer_scale=use_layer_scale,
                                  layer_scale_init_value=layer_scale_init_value,
                                  resolution=stage_resolution,
                                  vit_num=vit_num,
                                  e_ratios=e_ratios)
            self.network.append(stage) # Add stage to ModuleList

            # Add downsampling layer (Embedding) if not the last stage
            if i < len(layers) - 1:
                # Check if downsampling is needed based on config or dim change
                needs_downsample = (downsamples is None or downsamples[i]) or (embed_dims[i] != embed_dims[i+1])

                if needs_downsample:
                    # Determine if using Attention Substitution (ASub)
                    # ASub used for transitions 2->3 (i=1) and 3->4 (i=2) in paper diagram
                    # Note: Index i here is *before* the downsampling layer
                    # Stage 0 -> Stage 1 (i=0): Standard Conv Embedding
                    # Stage 1 -> Stage 2 (i=1): Standard Conv Embedding
                    # Stage 2 -> Stage 3 (i=2): Attention Sub Embedding
                    # Stage 3 -> Stage 4 (i=3): Attention Sub Embedding
                    use_asub = (i >= 2)

                    down_layer = Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1],
                        resolution=stage_resolution, # Resolution *before* downsampling
                        asub=use_asub,
                        act_layer=act_layer, norm_layer=norm_layer,
                    )
                    self.network.append(down_layer) # Add downsampling layer
                    current_resolution = math.ceil(current_resolution / down_stride) # Update resolution tracker


        # --- Feature Forking / Classification Head ---
        if self.fork_feat:
            # Add norm layers for feature outputs
            # Output indices correspond to *after* stage blocks: 0, 2, 4, 6
            # These indices point into self.network ModuleList
            # Output after Stage 0: network[0]
            # Output after Stage 1: network[2] (network[1] is downsampler)
            # Output after Stage 2: network[4] (network[3] is downsampler)
            # Output after Stage 3: network[6] (network[5] is downsampler)
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_layer < len(self.network): # Ensure layer exists
                    layer = norm_layer(embed_dims[i_emb])
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)
                # else: # This case shouldn't happen with standard configs
                #     print(f"Warning: Out index {i_layer} >= network length {len(self.network)}")
        else:
            # Add classification head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
            self.apply(self._init_weights_head) # Initialize head weights

        # Apply generic weight initialization
        self.apply(self._init_weights)

        # Store feature dimensions for potential use by detection heads etc.
        # Run a dummy forward pass to determine output widths dynamically
        # Use a smaller resolution for faster init if possible
        # try:
        #     test_res = resolution // 2 if resolution >= 64 else resolution
        #     self.forward(torch.randn(1, 3, test_res, test_res))
        # except Exception: # Fallback to default resolution if smaller fails
        self.forward(torch.randn(1, 3, resolution, resolution))

    def _init_weights_head(self, m):
         # Initialize classification head similar to Timm VisionTransformer
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_weights(self, m):
        # Generic initialization (already handled in submodules like Mlp)
        # Can add specific initializations for Embedding etc. if needed
        pass


    def forward_tokens(self, x):
        """ Helper function to forward through the network stages """
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                # Apply the corresponding norm layer
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)

        # Store output widths after the forward pass completes
        if self.fork_feat and hasattr(self, 'out_indices'):
            self.width_list = [o.size(1) for o in outs]

        # Return features if fork_feat, otherwise return final output before head
        if self.fork_feat:
            return outs
        else:
            return x # Return final feature map for classification head

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)

        if self.fork_feat:
            # Return list of feature maps
             # Ensure width_list is set after the first forward pass
            if not hasattr(self, 'width_list') or self.width_list is None:
                 self.width_list = [o.size(1) for o in x]
            return x
        else:
            # Apply classification head
            x = self.norm(x)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1) # Global avg pooling
            x = self.head(x)
            return x


def update_weight(model_dict, weight_dict):
    """ Helper to load pretrained weights, matching keys and shapes """
    idx, mismatched_keys = 0, []
    temp_dict = {}
    for k, v in weight_dict.items():
        if k in model_dict:
            if np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                idx += 1
            else:
                mismatched_keys.append(f"[Size mismatch] Pretrained: {k} {v.shape}, Model: {k} {model_dict[k].shape}")
        # else: # Optional: Log keys not found in the current model
        #     mismatched_keys.append(f"[Key not found] Pretrained: {k}")

    if mismatched_keys:
        print("Mismatched keys during weight loading:")
        for key_info in mismatched_keys:
            print(f"  {key_info}")

    model_dict.update(temp_dict)
    print(f'Successully loaded weights for {idx}/{len(model_dict)} items')
    return model_dict

def _load_pretrained(model, weights_path):
    """ Loads pretrained weights from a path. """
    if weights_path and os.path.exists(weights_path):
        try:
            # Common structures for saved checkpoints
            checkpoint = torch.load(weights_path, map_location='cpu')
            if 'model' in checkpoint:
                pretrained_weight = checkpoint['model']
            elif 'state_dict' in checkpoint:
                 pretrained_weight = checkpoint['state_dict']
            else:
                 pretrained_weight = checkpoint

            # Adapt keys if necessary (e.g., remove 'module.' prefix)
            adapted_weights = {}
            for k, v in pretrained_weight.items():
                new_k = k.replace('module.', '') if k.startswith('module.') else k
                adapted_weights[new_k] = v

            model.load_state_dict(update_weight(model.state_dict(), adapted_weights), strict=False)
            print(f"Loaded pretrained weights from: {weights_path}")
        except Exception as e:
            print(f"Error loading pretrained weights from {weights_path}: {e}")
    elif weights_path:
        print(f"Warning: Pretrained weights path specified but not found: {weights_path}")
    return model

# --- Model Instantiation Functions ---

def efficientformerv2_s0(weights='', **kwargs):
    """ EfficientFormerV2-S0 model """
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S0'],
        embed_dims=EfficientFormer_width['S0'],
        downsamples=[True, True, True, True], # Default downsampling after each stage
        vit_num=2, # Last 2 blocks in stages 2, 3 are AttnFFN
        drop_path_rate=0.0, # Default from paper
        e_ratios=expansion_ratios_S0,
        act_layer=nn.ReLU, # Paper uses ReLU for S0, S1, S2
        **kwargs)
    model = _load_pretrained(model, weights)
    return model

def efficientformerv2_s1(weights='', **kwargs):
    """ EfficientFormerV2-S1 model """
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S1'],
        embed_dims=EfficientFormer_width['S1'],
        downsamples=[True, True, True, True],
        vit_num=2, # Last 2 blocks in stages 2, 3 are AttnFFN
        drop_path_rate=0.0, # Default from paper
        e_ratios=expansion_ratios_S1,
        act_layer=nn.ReLU, # Paper uses ReLU for S0, S1, S2
        **kwargs)
    model = _load_pretrained(model, weights)
    return model

def efficientformerv2_s2(weights='', **kwargs):
    """ EfficientFormerV2-S2 model """
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S2'],
        embed_dims=EfficientFormer_width['S2'],
        downsamples=[True, True, True, True],
        vit_num=4, # Last 4 blocks in stages 2, 3 are AttnFFN
        drop_path_rate=0.02, # Default from paper
        e_ratios=expansion_ratios_S2,
        act_layer=nn.ReLU, # Paper uses ReLU for S0, S1, S2
        **kwargs)
    model = _load_pretrained(model, weights)
    return model

def efficientformerv2_l(weights='', **kwargs):
    """ EfficientFormerV2-L model """
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['L'],
        embed_dims=EfficientFormer_width['L'],
        downsamples=[True, True, True, True],
        vit_num=6, # Last 6 blocks in stages 2, 3 are AttnFFN
        drop_path_rate=0.1, # Default from paper
        e_ratios=expansion_ratios_L,
        act_layer=nn.GELU, # Paper uses GELU for L
        **kwargs)
    model = _load_pretrained(model, weights)
    return model


if __name__ == '__main__':
    input_res = 640
    print(f"--- Testing EfficientFormerV2 with input resolution {input_res}x{input_res} ---")
    inputs = torch.randn((1, 3, input_res, input_res))

    # Set fork_feat=True to get feature maps (common for detection/segmentation)
    fork_feat_flag = True
    print(f"Fork Features: {fork_feat_flag}\n")

    print("--- efficientformerv2_s0 ---")
    model_s0 = efficientformerv2_s0(fork_feat=fork_feat_flag, resolution=input_res)
    # model_s0.eval() # Set to eval mode if testing inference
    res_s0 = model_s0(inputs)
    if fork_feat_flag:
        print("Output feature map shapes:")
        for i, out in enumerate(res_s0):
            print(f"  Stage {i}: {out.shape}, Width: {model_s0.width_list[i]}")
    else:
        print(f"Output classification shape: {res_s0.shape}")
    # print(f"Width list: {getattr(model_s0, 'width_list', 'N/A')}")
    print("-" * 30)


    print("--- efficientformerv2_s1 ---")
    model_s1 = efficientformerv2_s1(fork_feat=fork_feat_flag, resolution=input_res)
    res_s1 = model_s1(inputs)
    if fork_feat_flag:
        print("Output feature map shapes:")
        for i, out in enumerate(res_s1):
            print(f"  Stage {i}: {out.shape}, Width: {model_s1.width_list[i]}")
    else:
        print(f"Output classification shape: {res_s1.shape}")
    # print(f"Width list: {getattr(model_s1, 'width_list', 'N/A')}")
    print("-" * 30)


    print("--- efficientformerv2_s2 ---")
    model_s2 = efficientformerv2_s2(fork_feat=fork_feat_flag, resolution=input_res)
    res_s2 = model_s2(inputs)
    if fork_feat_flag:
        print("Output feature map shapes:")
        for i, out in enumerate(res_s2):
            print(f"  Stage {i}: {out.shape}, Width: {model_s2.width_list[i]}")
    else:
        print(f"Output classification shape: {res_s2.shape}")
    # print(f"Width list: {getattr(model_s2, 'width_list', 'N/A')}")
    print("-" * 30)


    print("--- efficientformerv2_l ---")
    model_l = efficientformerv2_l(fork_feat=fork_feat_flag, resolution=input_res)
    res_l = model_l(inputs)
    if fork_feat_flag:
        print("Output feature map shapes:")
        for i, out in enumerate(res_l):
            print(f"  Stage {i}: {out.shape}, Width: {model_l.width_list[i]}")
    else:
        print(f"Output classification shape: {res_l.shape}")
    # print(f"Width list: {getattr(model_l, 'width_list', 'N/A')}")
    print("-" * 30)

    # --- Test Stride Calculation Compatibility ---
    # Simulate how Ultralytics might calculate stride
    print("\n--- Simulating Stride Calculation ---")
    try:
        from ultralytics.nn.tasks import DetectionModel # Need ultralytics installed
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG_PATH

        # Minimal config to test stride calculation on a backbone
        args = {'model': 'yolov8n.yaml', 'task': 'detect'}
        cfg = get_cfg(overrides=args)
        # Replace backbone definition with EfficientFormerV2 (requires custom YAML or modification)
        # For simplicity, just instantiate the backbone and check forward with small input
        stride_test_model = efficientformerv2_s0(fork_feat=True, resolution=64) # Use smaller resolution for test
        stride_test_input = torch.zeros(1, 3, 64, 64)
        stride_test_output = stride_test_model(stride_test_input)
        print("Stride test (64x64 input) output shapes:")
        for i, out in enumerate(stride_test_output):
             print(f"  Stage {i}: {out.shape}")
        print("Stride calculation simulation successful.")

    except ImportError:
        print("Ultralytics not found, skipping stride calculation simulation.")
    except Exception as e:
         print(f"Error during stride calculation simulation: {e}")
         import traceback
         traceback.print_exc()