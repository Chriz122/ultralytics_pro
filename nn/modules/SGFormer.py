import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model # Removed
# from timm.models.vision_transformer import _cfg # Removed
import math
import numpy as np

# --- Positional Encoding Helpers (from original SGFormer) ---
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32) # Changed to float32 for consistency
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# --- DWConv (from original SGFormer) ---
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# --- Mlp (from original SGFormer, uses DWConv) ---
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        # self.apply(self._init_weights) # Initialization will be handled by SGFormer's main apply

    def forward(self, x, H, W):
        x_fc1 = self.fc1(x)
        # The SGFormer MLP structure: x->fc1 (->x1) -> act(x1 + dwconv(x1,H,W)) -> drop -> fc2 -> drop
        x_dw = self.dwconv(x_fc1, H, W) # Apply DWConv on the output of fc1
        x = self.act(x_fc1 + x_dw) 
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# --- Local Convolution Helper (from original SGFormer) ---
def local_conv(dim):
    return nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

# --- Window Partitioning Helpers (from original SGFormer) ---
# NOTE: These functions assume H and W are perfectly divisible by window_size.
# This might be an issue for arbitrary input sizes if the "local" attention path is taken.
def window_partition(x_in, window_size, H, W):
    # x_in shape: (B_num_heads, N, C_head) or (B, num_heads, N, C_head)
    # Ensure x is (B_num_heads, N, C_head)
    if len(x_in.shape) == 4: # (B, num_heads, N, C_head)
        B, num_heads, N_feat, C_head_feat = x_in.shape
        x = x_in.contiguous().view(B * num_heads, N_feat, C_head_feat)
    else: # Already (B_num_heads, N, C_head)
        x = x_in
    
    B_h, N, C_h = x.shape # B_h is B * num_heads
    
    if H * W != N:
        raise ValueError(f"H ({H}) * W ({W}) = {H*W} does not match N ({N}) in window_partition for input shape {x.shape}")

    # Pad if H or W are not divisible by window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = x.view(B_h, H, W, C_h).permute(0, 3, 1, 2) # -> B_h, C_h, H, W
        x = F.pad(x, (0, pad_w, 0, pad_h)) # Pad H, W dims
        x = x.permute(0, 2, 3, 1).contiguous().view(B_h, (H + pad_h) * (W + pad_w), C_h) # -> B_h, N_padded, C_h
        H_pad, W_pad = H + pad_h, W + pad_w
    else:
        H_pad, W_pad = H, W
        
    x = x.view(B_h, H_pad, W_pad, C_h)
    
    num_windows_h = H_pad // window_size
    num_windows_w = W_pad // window_size

    windows = x.view(B_h, num_windows_h, window_size, num_windows_w, window_size, C_h)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C_h)
    return windows, (H_pad, W_pad) # Return padded H,W for reverse


def window_reverse(windows, window_size, H_pad, W_pad, H_orig, W_orig, num_heads_partitioned):
    # num_heads_partitioned is the number of heads that 'windows' tensor corresponds to (e.g., self.num_heads // 2)
    # B_actual_times_heads_partitioned_times_num_windows = windows.shape[0]
    # num_elements_in_window = windows.shape[1] # window_size * window_size
    # C_head = windows.shape[2]

    num_windows_h = H_pad // window_size
    num_windows_w = W_pad // window_size
    
    B_actual_times_heads_partitioned = windows.shape[0] // (num_windows_h * num_windows_w)

    x = windows.view(B_actual_times_heads_partitioned, num_windows_h, num_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B_actual_times_heads_partitioned, H_pad, W_pad, -1) # B_h_part, H_pad, W_pad, C_h

    # Unpad if necessary
    if H_pad > H_orig or W_pad > W_orig:
        x = x.permute(0,3,1,2) # -> B_h_part, C_h, H_pad, W_pad
        x = x[:, :, :H_orig, :W_orig]
        x = x.permute(0,2,3,1).contiguous() # -> B_h_part, H_orig, W_orig, C_h
    
    # Reshape to B, N, C_total
    # B_actual_times_heads_partitioned, H_orig, W_orig, C_head
    # num_heads_partitioned is the number of heads this 'x' belongs to (e.g., num_heads//2)
    B_actual = B_actual_times_heads_partitioned // num_heads_partitioned
    C_total_for_partition = x.shape[-1] * num_heads_partitioned # This should be C/2 or C

    x = x.view(B_actual, num_heads_partitioned, H_orig, W_orig, x.shape[-1]) # B, num_heads_part, H_orig, W_orig, C_h
    x = x.permute(0, 2, 3, 1, 4).contiguous().view(B_actual, H_orig * W_orig, -1) # B, N_orig, C_part_total
    return x


# --- Attention (from original SGFormer) ---
class Attention(nn.Module):
    def __init__(self, dim, mask_init_config, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.mask_init_config = mask_init_config 

        if sr_ratio > 1:
            if mask_init_config: 
                self.q_cfg_true = nn.Linear(dim, dim, bias=qkv_bias) # Renamed to avoid clash
                self.kv1_cfg_true = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv2_cfg_true = nn.Linear(dim, dim, bias=qkv_bias)
                
                # These f1,f2,f3 layers are highly dependent on fixed sequence lengths after grouping.
                # This part is fragile for arbitrary H,W inputs.
                # Values are sequence lengths of token groups, not feature map sizes.
                if self.sr_ratio == 8: fixed_f1, fixed_f2, fixed_f3 = 196, 56, 28 
                elif self.sr_ratio == 4: fixed_f1, fixed_f2, fixed_f3 = 49, 14, 7
                elif self.sr_ratio == 2: fixed_f1, fixed_f2, fixed_f3 = 4, 2, None 
                else: fixed_f1, fixed_f2, fixed_f3 = None,None,None 

                if fixed_f1: self.f1 = nn.Linear(fixed_f1, 1) 
                if fixed_f2: self.f2 = nn.Linear(fixed_f2, 1)
                if fixed_f3: self.f3 = nn.Linear(fixed_f3, 1)

            else: # mask_init_config is False
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
                self.act = nn.GELU()

                self.q1 = nn.Linear(dim, dim // 2, bias=qkv_bias)
                self.kv1_cfg_false = nn.Linear(dim, dim, bias=qkv_bias) # for global
                self.q2 = nn.Linear(dim, dim // 2, bias=qkv_bias)
                self.kv2_cfg_false = nn.Linear(dim, dim, bias=qkv_bias) # for local
        else: # sr_ratio == 1
            self.q_sr1 = nn.Linear(dim, dim, bias=qkv_bias) # Renamed
            self.kv_sr1 = nn.Linear(dim, dim * 2, bias=qkv_bias) # Renamed

        self.lepe_linear = nn.Linear(dim, dim)
        self.lepe_conv = local_conv(dim) 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W, mask_runtime_tensor): 
        B, N, C = x.shape
        lepe = self.lepe_conv(
            self.lepe_linear(x).transpose(1, 2).view(B, C, H, W)
        ).view(B, C, -1).transpose(-1, -2)

        if self.sr_ratio > 1:
            # Path selection based on how Attention was configured AND current runtime mask
            if not self.mask_init_config and mask_runtime_tensor is None:
                # Global-Local path for blocks configured with mask_init_config=False, on their first pass
                # This path generates the initial mask_runtime_tensor.
                
                # Global attention (num_heads // 2 for this part)
                q1_val = self.q1(x).reshape(B, N, self.num_heads // 2, self.head_dim).permute(0, 2, 1, 3)
                
                x_for_sr = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_1_sr = self.sr(x_for_sr).reshape(B, C, -1).permute(0, 2, 1) 
                x_1_sr_norm = self.act(self.norm(x_1_sr))
                
                kv1_pairs = self.kv1_cfg_false(x_1_sr_norm).reshape(B, -1, 2, self.num_heads // 2, self.head_dim).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1_pairs[0], kv1_pairs[1]

                attn1 = (q1_val @ k1.transpose(-2, -1)) * self.scale 
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)

                global_mask_value = torch.mean(attn1.detach().mean(1), dim=1) 
                H_sr, W_sr = H // self.sr_ratio, W // self.sr_ratio
                global_mask_value = F.interpolate(global_mask_value.view(B, 1, H_sr, W_sr),
                                                  size=(H, W), mode='nearest')[:, 0, :, :].contiguous()
                
                # Local attention (num_heads // 2 for this part)
                q2_val = self.q2(x).reshape(B, N, self.num_heads // 2, self.head_dim).permute(0, 2, 1, 3) 
                kv2_pairs = self.kv2_cfg_false(x).reshape(B, N, 2, self.num_heads // 2, self.head_dim).permute(2, 0, 3, 1, 4)
                k2, v2 = kv2_pairs[0], kv2_pairs[1]
                
                q_window_size, kv_window_size = 7, 7 # Hardcoded window size
                
                q2_win, (H_pad_q, W_pad_q) = window_partition(q2_val.reshape(B * (self.num_heads // 2), N, self.head_dim), q_window_size, H, W)
                k2_win, _ = window_partition(k2.reshape(B * (self.num_heads // 2), N, self.head_dim), kv_window_size, H, W)
                v2_win, _ = window_partition(v2.reshape(B * (self.num_heads // 2), N, self.head_dim), kv_window_size, H, W)

                attn2 = (q2_win @ k2_win.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                x2_win = (attn2 @ v2_win)
                x2 = window_reverse(x2_win, q_window_size, H_pad_q, W_pad_q, H, W, self.num_heads // 2)
                
                # Simplified local_mask_value generation (original is complex and tied to exact shapes)
                # This creates a B,H,W mask. A simple proxy: mean of attention scores in windows, upscaled.
                num_windows_H = H_pad_q // kv_window_size
                num_windows_W = W_pad_q // kv_window_size
                attn2_for_mask = attn2.view(B, self.num_heads // 2, num_windows_H * num_windows_W,
                                            kv_window_size * kv_window_size, kv_window_size * kv_window_size)
                
                avg_attn_score_per_window = attn2_for_mask.mean(dim=(1,3,4)) # B, num_windows_total
                avg_attn_score_map = avg_attn_score_per_window.view(B, 1, num_windows_H, num_windows_W)
                local_mask_value = F.interpolate(avg_attn_score_map, size=(H,W), mode='nearest').squeeze(1) # B, H, W


                x = torch.cat([x1, x2], dim=-1)
                x = self.proj(x + lepe) 
                x = self.proj_drop(x)

                mask_from_attention = local_mask_value + global_mask_value 
                mask_1 = mask_from_attention.view(B, H * W)
                mask_2 = mask_from_attention.permute(0, 2, 1).reshape(B, H * W)
                mask_runtime_tensor = [mask_1, mask_2] # Generated mask for next block

            else: # Token sparsification path (mask_init_config=True OR mask_runtime_tensor is available)
                if not hasattr(self, 'q_cfg_true'):
                    # This implies mask_init_config was False, but mask_runtime_tensor is present.
                    # This path should ideally not be hit if logic is consistent.
                    # Fallback to a simple sr_ratio=1 like attention if layers are missing.
                    # This is an unlikely scenario if block logic is correct.
                    q_full = self.q_sr1(x) 
                    kv_full = self.kv_sr1(x)
                    
                    q_val = q_full.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                    kv_pairs = kv_full.reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                    k, v = kv_pairs[0], kv_pairs[1]
                else: # Expected path for mask_init_config=True
                    q_val = self.q_cfg_true(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                    
                    if mask_runtime_tensor is None: 
                        # This means mask_init_config=True, but no runtime mask was provided (e.g. first block in stage)
                        # This scenario is not well-defined in original logic for this path.
                        # Default to using all tokens for K,V or a random subset.
                        # For simplicity, use all tokens if mask is missing here.
                        kv_tokens_seq1 = x
                        kv_tokens_seq2 = x.view(B,H,W,C).permute(0,2,1,3).reshape(B,N,C) # W-first view
                    else:
                        mask_1, mask_2 = mask_runtime_tensor
                        _, mask_sort_index1 = torch.sort(mask_1, dim=1, descending=True)
                        _, mask_sort_index2 = torch.sort(mask_2, dim=1, descending=True)

                        # Simplified token selection for K,V due to fragility of original f1,f2,f3 logic
                        # Select top K tokens based on sr_ratio.
                        num_kv_tokens = N // (self.sr_ratio**2) if self.sr_ratio > 0 else N 
                        num_kv_tokens = max(1, num_kv_tokens) # Ensure at least 1 token

                        kv_tokens_seq1 = torch.gather(x, 1, mask_sort_index1[:, :num_kv_tokens].unsqueeze(-1).expand(-1,-1,C))
                        x_perm = x.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B, N, C)
                        kv_tokens_seq2 = torch.gather(x_perm, 1, mask_sort_index2[:, :num_kv_tokens].unsqueeze(-1).expand(-1,-1,C))


                    kv1_pairs = self.kv1_cfg_true(kv_tokens_seq1).reshape(B, -1, 2, self.num_heads // 2, self.head_dim).permute(2, 0, 3, 1, 4)
                    kv2_pairs = self.kv2_cfg_true(kv_tokens_seq2).reshape(B, -1, 2, self.num_heads // 2, self.head_dim).permute(2, 0, 3, 1, 4)
                    
                    kv = torch.cat([kv1_pairs, kv2_pairs], dim=2) 
                    k, v = kv[0], kv[1]

                attn = (q_val @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x + lepe)
                x = self.proj_drop(x)
                # Mask is typically consumed in this path and not propagated further as a runtime tensor
                mask_runtime_tensor = None 

        else: # sr_ratio == 1, standard self-attention
            q_val = self.q_sr1(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv_pairs = self.kv_sr1(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv_pairs[0], kv_pairs[1]

            attn = (q_val @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x + lepe) 
            x = self.proj_drop(x)
            mask_runtime_tensor = None 

        return x, mask_runtime_tensor


# --- Block (from original SGFormer) ---
class Block(nn.Module):
    def __init__(self, dim, mask_init_config, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, mask_init_config, 
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, mask_runtime_tensor): 
        x_attn, mask_runtime_tensor_out = self.attn(self.norm1(x), H, W, mask_runtime_tensor)
        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x, mask_runtime_tensor_out

# --- Conv2d_BN Helper (from original SGFormer) ---
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        # Original used GroupNorm(1,b) which is like LayerNorm on channels if C=1 per group.
        # For more standard BN behavior, use nn.BatchNorm2d.
        # Let's stick to GroupNorm as it was in provided code.
        bn = nn.GroupNorm(num_groups=max(1, b // min(b,32)), num_channels=b) # More typical GroupNorm
        # bn = nn.GroupNorm(num_groups=1, num_channels=b) # Original's GroupNorm setup
        # bn = nn.BatchNorm2d(b) # Alternative
        if hasattr(bn, 'weight') and bn.weight is not None:
             torch.nn.init.constant_(bn.weight, bn_weight_init)
        if hasattr(bn, 'bias') and bn.bias is not None:
             torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

# --- Head (Patch Embedding Stage 1 for SGFormer) ---
class Head(nn.Module):
    def __init__(self, in_chans, embed_dim): 
        super(Head, self).__init__()
        n = embed_dim 
        self.conv = nn.Sequential(
            Conv2d_BN(in_chans, n, ks=3, stride=2, pad=1), 
            nn.GELU(),
            Conv2d_BN(n, n, ks=3, stride=1, pad=1),
            nn.GELU(),
            Conv2d_BN(n, n, ks=3, stride=2, pad=1), 
        )
        self.norm = nn.LayerNorm(n)

    def forward(self, x):
        x = self.conv(x) 
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) 
        x = self.norm(x)
        return x, H, W

# --- PatchMerging (Subsequent Patch Embedding Stages for SGFormer) ---
class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim): 
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.act = nn.GELU()
        self.conv1 = Conv2d_BN(dim, out_dim, ks=1, stride=1, pad=0) 
        self.conv2 = Conv2d_BN(out_dim, out_dim, ks=3, stride=2, pad=1, groups=out_dim) 
        self.conv3 = Conv2d_BN(out_dim, out_dim, ks=1, stride=1, pad=0) 
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x_in_spatial): # Expect B, C_in, H_in, W_in
        x = self.conv1(x_in_spatial)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) 
        x = self.norm(x) 
        return x, H, W


# --- Main SGFormer Model ---
class SGFormer(nn.Module):
    arch_zoo = {
        's': {'embed_dims': [64, 128, 256, 512], 'num_heads': [2, 4, 8, 16], 'depths': [2, 4, 16, 1]},
        'm': {'embed_dims': [64, 128, 256, 512], 'num_heads': [2, 4, 8, 16], 'depths': [2, 6, 28, 2]},
        'b': {'embed_dims': [96, 192, 384, 768], 'num_heads': [4, 6, 12, 24], 'depths': [4, 6, 24, 2]},
    }
    # Default parameters, can be overridden by arch_settings
    default_mlp_ratios = [4, 4, 4, 4] # Per stage
    default_sr_ratios = [8, 4, 2, 1]  # Per stage

    def __init__(self,
                 c1=3, 
                 arch='s', 
                 img_size=224, 
                 num_classes=1000, 
                 qkv_bias=True, 
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 linear=False, 
                 fork_feat=True, 
                 **kwargs):
        super().__init__()

        self.in_channels = c1
        self.num_classes = num_classes
        self.fork_feat = fork_feat

        if isinstance(arch, str):
            arch_key = arch.lower()
            if arch_key not in self.arch_zoo:
                raise KeyError(f"Architecture '{arch_key}' not found in SGFormer arch_zoo.")
            self.arch_settings = self.arch_zoo[arch_key]
        elif isinstance(arch, dict): 
            self.arch_settings = arch
        else:
            raise TypeError("arch must be a string or a dict.")

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_heads = self.arch_settings['num_heads']
        self.depths = self.arch_settings['depths']
        self.mlp_ratios = self.arch_settings.get('mlp_ratios', self.default_mlp_ratios)
        self.sr_ratios = self.arch_settings.get('sr_ratios', self.default_sr_ratios)
        
        self.num_stages = len(self.depths)

        self.img_size_for_pe = to_2tuple(img_size)
        self.num_patches_for_pe_h = self.img_size_for_pe[0] // 4 
        self.num_patches_for_pe_w = self.img_size_for_pe[1] // 4

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0

        for i in range(self.num_stages):
            if i == 0:
                patch_embed = Head(in_chans=self.in_channels, embed_dim=self.embed_dims[0])
            else:
                patch_embed = PatchMerging(dim=self.embed_dims[i-1], out_dim=self.embed_dims[i])
            
            blocks_for_stage = nn.ModuleList()
            for j in range(self.depths[i]):
                use_mask_specific_attn_layers = bool(j % 2 == 1 and i < self.num_stages - 1)
                blocks_for_stage.append(Block(
                    dim=self.embed_dims[i],
                    mask_init_config=use_mask_specific_attn_layers,
                    num_heads=self.num_heads[i],
                    mlp_ratio=self.mlp_ratios[i],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j],
                    norm_layer=norm_layer,
                    sr_ratio=self.sr_ratios[i],
                    linear=linear
                ))
            
            norm = norm_layer(self.embed_dims[i])
            cur += self.depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blocks_for_stage)
            setattr(self, f"norm{i + 1}", norm)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_for_pe_h * self.num_patches_for_pe_w, self.embed_dims[0]))
        # trunc_normal_ will be applied in _init_weights or here
        trunc_normal_(self.pos_embed, std=.02)

        if self.fork_feat:
            self.head = nn.Identity()
            self.width_list = list(self.embed_dims) 
        else:
            self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
            self.width_list = [self.embed_dims[-1]] if self.embed_dims else []

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None: nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if hasattr(m, 'groups') and m.groups is not None and m.groups > 0:
                 fan_out //= m.groups
            if fan_out > 0:
                 m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            else:
                 m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        
        if hasattr(self, 'pos_embed') and isinstance(self.pos_embed, nn.Parameter):
             grid_dim_for_pe_gen_h = self.num_patches_for_pe_h
             grid_dim_for_pe_gen_w = self.num_patches_for_pe_w
             
             # For get_2d_sincos_pos_embed, a single grid_size is expected. Use H.
             # If H and W for PE are different, sincos might need adjustment or use only H.
             pos_embed_data = get_2d_sincos_pos_embed(
                 self.pos_embed.shape[-1],
                 grid_dim_for_pe_gen_h, # Using H dimension for PE grid size
                 cls_token=False
             )
             # Ensure target PE length matches generated one if W != H for PE source grid
             if pos_embed_data.shape[0] == self.pos_embed.shape[1]:
                 self.pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float().unsqueeze(0))
             else: # Fallback to trunc_normal_ if sincos PE shape doesn't match (e.g. non-square PE grid)
                 # This case should be handled by ensuring num_patches_for_pe_h * num_patches_for_pe_w
                 # is used for PE generation or interpolation.
                 # print(f"Warning: SinCos PE shape mismatch. Generated {pos_embed_data.shape}, target {self.pos_embed.shape}. Using trunc_normal_.")
                 pass # trunc_normal_ already applied during parameter creation

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'} 

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if self.fork_feat: 
            self.head = nn.Identity()
        else:
            self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def _interpolate_pos_embed(self, x_feat_tokens, H_feat, W_feat):
        B, N_feat, C_feat = x_feat_tokens.shape
        # Check if current feature map dimensions match the PE's original dimensions
        if self.num_patches_for_pe_h == H_feat and self.num_patches_for_pe_w == W_feat \
           and N_feat == self.pos_embed.shape[1]:
            return self.pos_embed

        # Reshape PE to 2D grid: (1, N_pe, C) -> (1, H_pe, W_pe, C) -> (1, C, H_pe, W_pe)
        pos_embed_grid = self.pos_embed.reshape(
            1, self.num_patches_for_pe_h, self.num_patches_for_pe_w, C_feat
        ).permute(0, 3, 1, 2)

        # Interpolate to current feature map H_feat, W_feat
        pos_embed_interp = F.interpolate(pos_embed_grid, size=(H_feat, W_feat), mode='bicubic', align_corners=False)
        
        # Reshape interpolated PE back to token sequence: (1, C, H_feat, W_feat) -> (1, N_feat, C)
        pos_embed_interp = pos_embed_interp.permute(0, 2, 3, 1).reshape(1, H_feat * W_feat, C_feat)
        return pos_embed_interp

    def forward_features(self, x_img):
        B = x_img.shape[0]
        outputs_list = [] 
        runtime_mask_tensor = None 
        current_x = x_img # Will be B,C,H,W or B,N,C depending on stage

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            blocks = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            # patch_embed for stage 0 (Head) expects B,C,H,W image
            # patch_embed for later stages (PatchMerging) expects B,C,H,W feature map
            current_x, H, W = patch_embed(current_x) # Output: B, N, C_stage; H,W are feature map H,W
            
            if i == 0: 
                interpolated_pe = self._interpolate_pos_embed(current_x, H, W)
                current_x = current_x + interpolated_pe

            for blk in blocks:
                current_x, runtime_mask_tensor = blk(current_x, H, W, runtime_mask_tensor)
            current_x = norm(current_x) # current_x is B, N, C_stage

            if self.fork_feat:
                out_feat_spatial = current_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                outputs_list.append(out_feat_spatial)
            elif i == self.num_stages - 1: 
                 outputs_list.append(current_x) # Save last B,N,C feature map for classification
            
            # For next stage's PatchMerging, input needs to be B,C,H,W
            if i < self.num_stages - 1: # Prepare for next PatchMerging or if forking all
                current_x = current_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


        if self.fork_feat:
            return outputs_list 
        else:
            return outputs_list[-1] 

    def forward(self, x):
        features = self.forward_features(x)
        if self.fork_feat:
            if not isinstance(features, list): # Should be a list from forward_features
                 return [features] if features is not None else [] # Defensive
            return features
        else:
            # For classification: features is the B,N,C tensor from the last stage
            final_feature_map_tokens = features 
            final_feature_vector = final_feature_map_tokens.mean(dim=1) 
            return self.head(final_feature_vector)


# --- Factory functions similar to MogaNet for different SGFormer sizes ---
def sgformer_s(pretrained=False, **kwargs): 
    model = SGFormer(arch='s', **kwargs)
    return model

def sgformer_m(pretrained=False, **kwargs):
    model = SGFormer(arch='m', **kwargs)
    return model

def sgformer_b(pretrained=False, **kwargs):
    model = SGFormer(arch='b', **kwargs)
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n--- Testing SGFormer with fork_feat=True (Backbone Mode) ---")
    try:
        img_h, img_w = 640, 640
        
        model_s_backbone = SGFormer(c1=3, arch='s', img_size=(img_h,img_w), fork_feat=True).to(device)
        model_s_backbone.eval()

        print(f"SGFormer-S (Backbone) Initialized. width_list: {model_s_backbone.width_list}")
        assert model_s_backbone.width_list == model_s_backbone.embed_dims

        dummy_input = torch.randn(2, 3, img_h, img_w).to(device)
        with torch.no_grad():
            features = model_s_backbone(dummy_input)
        
        print(f"Output is a list of {len(features)} tensors:")
        expected_strides = [4, 8, 16, 32] # Based on Head (stride 4) and PatchMerging (stride 2 each)
        for i, feat in enumerate(features):
            print(f"  Stage {i+1} feature shape: {feat.shape}, "
                  f"Expected C: {model_s_backbone.embed_dims[i]}, "
                  f"Expected H,W: {img_h//expected_strides[i]},{img_w//expected_strides[i]}")
            assert feat.shape[1] == model_s_backbone.embed_dims[i]
            assert feat.shape[2] == img_h // expected_strides[i]
            assert feat.shape[3] == img_w // expected_strides[i]
        
        print("SGFormer-S (Backbone Mode) test PASSED.")

    except Exception as e:
        print(f"Error during SGFormer-S (Backbone Mode) test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing SGFormer with fork_feat=False (Classification Mode) ---")
    try:
        img_h_cls, img_w_cls = 640, 640 

        model_m_cls = SGFormer(c1=3, arch='m', img_size=(img_h_cls, img_w_cls), num_classes=50, fork_feat=False).to(device)
        model_m_cls.eval()
        
        print(f"SGFormer-M (Classification) Initialized. Output classes: 50. width_list: {model_m_cls.width_list}")
        assert model_m_cls.width_list == [model_m_cls.embed_dims[-1]]

        dummy_input_cls = torch.randn(2, 3, img_h_cls, img_w_cls).to(device)
        with torch.no_grad():
            predictions = model_m_cls(dummy_input_cls)
        
        print(f"Output predictions shape: {predictions.shape}")
        assert predictions.shape == (2, 50)
        print("SGFormer-M (Classification Mode) test PASSED.")

    except Exception as e:
        print(f"Error during SGFormer-M (Classification Mode) test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing factory function sgformer_b ---")
    try:
        model_b_factory = sgformer_b(c1=3, img_size=512, fork_feat=True, num_classes=10).to(device)
        model_b_factory.eval()
        print(f"SGFormer-B (Factory) Initialized. width_list: {model_b_factory.width_list}")
        dummy_input_factory = torch.randn(1, 3, 512, 512).to(device)
        with torch.no_grad():
            features_factory = model_b_factory(dummy_input_factory)
        print(f"Factory model output: list of {len(features_factory)} tensors.")
        assert len(features_factory) == len(model_b_factory.embed_dims)
        print("SGFormer factory function test PASSED.")
    except Exception as e:
        print(f"Error during SGFormer factory function test: {e}")
        import traceback
        traceback.print_exc()