# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Demystify Mamba in Vision: A Linear Attention Perspective
# Modified by Dongchen Han
# Further modified for backbone usage and arbitrary input size by AI assistant
# -----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math # Not strictly used in the final version, but good to have for math ops


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


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    Generates rotations dynamically based on H, W passed in forward.
    Handles AMP by promoting ComplexHalf to ComplexFloat for internal computations.
    """
    def __init__(self, feature_dim, base=10000):
        super(RoPE, self).__init__()
        self.feature_dim = feature_dim
        self.base = base
        self._cached_rotations = None
        self._cached_hw_dtype_device = (0, 0, None, None) 

        if self.feature_dim % 2 != 0:
             raise ValueError(f"RoPE feature_dim must be even. Got {self.feature_dim}")
        if (self.feature_dim // 2) % 2 != 0 and self.feature_dim > 0 : # For 2D application with H and W components
             print(f"Warning: RoPE's feature_dim ({self.feature_dim}) // 2 is not even. "
                   f"The 2D meshgrid-style application might behave unusually or be suboptimal "
                   f"if the two concatenated angle components (for H and W) are not of equal dimension (k_max).")


    def _create_rotations(self, H, W, dtype, device):
        k_max_calculated = self.feature_dim // 4 
        
        if self.feature_dim < 4: # Not enough dimensions for separate H and W components in this scheme
            k_max = 0
        else:
            k_max = max(1, k_max_calculated)

        if k_max == 0: # Handle cases where RoPE cannot be meaningfully applied as 2D
            # This results in identity-like rotation (cos=1, sin=0 for the complex parts)
            # Effective shape (H, W, D/2, 2)
            rot_re = torch.ones(H, W, self.feature_dim // 2, 1, dtype=dtype, device=device)
            rot_im = torch.zeros(H, W, self.feature_dim // 2, 1, dtype=dtype, device=device)
            return torch.cat([rot_re, rot_im], dim=-1)

        theta_ks = 1.0 / (self.base ** (torch.arange(0, k_max, dtype=torch.float32, device=device) / k_max)) # Use float32 for precision in theta
        theta_ks = theta_ks.to(dtype) # Cast to target dtype for rotations

        pos_h_vec = torch.arange(H, dtype=dtype, device=device)
        pos_w_vec = torch.arange(W, dtype=dtype, device=device)
        
        grid_h, grid_w = torch.meshgrid([pos_h_vec, pos_w_vec], indexing='ij')
        angles_h_part = grid_h.unsqueeze(-1) * theta_ks 
        angles_w_part = grid_w.unsqueeze(-1) * theta_ks
        angles = torch.cat([angles_h_part, angles_w_part], dim=-1) 
        
        target_angle_dim = self.feature_dim // 2
        if angles.shape[-1] != target_angle_dim:
            if angles.shape[-1] < target_angle_dim:
                padding_size = target_angle_dim - angles.shape[-1]
                padding = torch.zeros(H, W, padding_size, dtype=dtype, device=device)
                angles = torch.cat([angles, padding], dim=-1)
            elif angles.shape[-1] > target_angle_dim:
                angles = angles[..., :target_angle_dim]

        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        return rotations

    def forward(self, x, H, W):
        B, N, C = x.shape
        if C != self.feature_dim:
            raise ValueError(f"Input feature dim {C} does not match RoPE configured dim {self.feature_dim}")
        if N != H * W:
            raise ValueError(f"Input N={N} does not match H*W={H*W}")

        current_key = (H, W, x.dtype, x.device)
        if current_key != self._cached_hw_dtype_device:
            self._cached_rotations = self._create_rotations(H, W, dtype=x.dtype, device=x.device)
            self._cached_hw_dtype_device = current_key
        
        # self._cached_rotations has the same dtype as x (e.g. float16 if x is float16)
        
        x_reshaped = x.view(B, H, W, C)
        x_pairs = x_reshaped.reshape(B, H, W, C // 2, 2) # dtype is x.dtype

        # --- AMP Stability: Use float32 for complex math if input is float16 ---
        original_input_dtype = x.dtype
        if original_input_dtype == torch.float16:
            compute_real_dtype = torch.float32
        else:
            compute_real_dtype = original_input_dtype

        x_pairs_for_complex = x_pairs.to(compute_real_dtype)
        # Ensure cached rotations are also in the compute_real_dtype for complex operations
        rotations_for_complex = self._cached_rotations.to(compute_real_dtype)
        # --- End AMP Stability ---
        
        x_complex = torch.view_as_complex(x_pairs_for_complex) 
        rot_complex = torch.view_as_complex(rotations_for_complex)
        
        rot_complex = rot_complex.to(x_complex.device)
        rotated_complex = x_complex * rot_complex.unsqueeze(0) # Broadcasting happens here
        
        output_pairs_compute_dtype = torch.view_as_real(rotated_complex) # dtype is compute_real_dtype

        # Cast back to original input dtype if necessary
        if output_pairs_compute_dtype.dtype != original_input_dtype:
            output_pairs = output_pairs_compute_dtype.to(original_input_dtype)
        else:
            output_pairs = output_pairs_compute_dtype
        
        output_reshaped = output_pairs.reshape(B, H, W, C)
        output = output_reshaped.reshape(B, N, C)
        
        return output


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim * self.num_heads != self.dim:
            raise ValueError("dim must be divisible by num_heads")

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(feature_dim=dim)

    def forward(self, x, H, W): 
        b, n, c = x.shape
        
        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v_in = qk[0], qk[1], x

        q = self.elu(q) + 1.0 
        k = self.elu(k) + 1.0 

        q_rope_applied = self.rope(q, H, W) 
        k_rope_applied = self.rope(k, H, W) 
        
        q_att = q_rope_applied.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_att = k_rope_applied.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_att = v_in.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        z_denom = q_att @ k_att.mean(dim=-2, keepdim=True).transpose(-2, -1) 
        z = 1 / (z_denom + 1e-6)
        
        scale = n ** -0.5 
        kv = (k_att.transpose(-2, -1) * scale) @ (v_att * scale) 
        
        x_out_attn = q_att @ kv * z
        x_out_attn = x_out_attn.transpose(1, 2).reshape(b, n, c)

        lepe_input = v_in.reshape(b, H, W, c).permute(0, 3, 1, 2)
        lepe_out = self.lepe(lepe_input).permute(0, 2, 3, 1).reshape(b, n, c)
        x_final = x_out_attn + lepe_out

        return x_final

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'


class MLLABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU() 
        self.attn = LinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H, W): 
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x_cpe1 = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.cpe1(x_cpe1).flatten(2).permute(0, 2, 1)
        
        shortcut = x
        x_norm1 = self.norm1(x)
        act_res = self.act(self.act_proj(x_norm1))
        
        x_in_proj = self.in_proj(x_norm1).view(B, H, W, C)
        x_dwc = self.dwc(x_in_proj.permute(0, 3, 1, 2))
        x_act_dwc = self.act(x_dwc).permute(0, 2, 3, 1).view(B, L, C)

        x_attn = self.attn(x_act_dwc, H, W)

        x = self.out_proj(x_attn * act_res)
        x = shortcut + self.drop_path(x)

        x_cpe2 = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.cpe2(x_cpe2).flatten(2).permute(0, 2, 1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    def __init__(self, dim, ratio=4.0): # ratio is for MLLA's conv-based PatchMerging
        super().__init__()
        self.dim = dim
        in_channels = dim
        out_channels = 2 * dim
        
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None, act_func=nn.GELU),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), 
                      kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio),
                      norm=None, act_func=nn.GELU),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None, norm=None)
        )

    def forward(self, x, H, W): 
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x_reshaped = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_merged = self.conv(x_reshaped)
        
        Hp, Wp = x_merged.shape[-2:]
        x_out = x_merged.flatten(2).permute(0, 2, 1)
        return x_out, Hp, Wp

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            MLLABlock(dim=dim, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                      norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim) # Pass current dim
        else:
            self.downsample = None

    def forward(self, x, H, W): 
        for blk in self.blocks:
            if self.use_checkpoint and self.training: 
                x = checkpoint.checkpoint(blk, x, H, W, use_reentrant=False) # use_reentrant=False is recommended for new PyTorch versions
            else:
                x = blk(x, H, W)
        
        x_pre_downsample = x
        H_pre_downsample, W_pre_downsample = H, W

        if self.downsample is not None:
            x_downsampled, H_new, W_new = self.downsample(x, H, W)
            return x_pre_downsample, H_pre_downsample, W_pre_downsample, x_downsampled, H_new, W_new
        else:
            return x_pre_downsample, H_pre_downsample, W_pre_downsample, x, H, W

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


class Stem(nn.Module):
    def __init__(self, patch_size_multiplier=4, in_chans=3, embed_dim=96): # patch_size_multiplier is effective stride
        super().__init__()
        self.patch_size_multiplier = patch_size_multiplier 
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2_res = ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_main = ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        self.conv3_s2 = ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3_pw = ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)

    def forward(self, x):
        x = self.conv1(x)
        x_res = x 
        x = self.conv2_res(x)
        x = self.conv2_main(x) + x_res
        x = F.relu(x, inplace=True) # Apply activation after residual add

        x = self.conv3_s2(x)
        x = self.conv3_pw(x)
        
        H_out, W_out = x.shape[-2:]
        x_flat = x.flatten(2).transpose(1, 2)
        return x_flat, H_out, W_out


class MLLA(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, use_checkpoint=False, 
                 out_indices=(0, 1, 2, 3)):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices
        self.patch_size = patch_size # Effective stride of the stem

        # Check RoPE compatibility (applied on full dim at each stage)
        for i in range(self.num_layers):
            stage_dim = embed_dim * (2**i)
            if stage_dim % 2 != 0 : # RoPE expects even feature_dim
                 raise ValueError(f"MLLA stage {i} dim {stage_dim} is not even, RoPE will fail.")
            # Optional: check for stage_dim % 4 for 2D meshgrid style RoPE
            if stage_dim % 4 != 0:
                 print(f"Warning: MLLA stage {i} dim {stage_dim} is not a multiple of 4. "
                       f"2D RoPE's H/W component splitting might be slightly imbalanced.")
            if stage_dim % num_heads[i] != 0:
                 raise ValueError(f"Stage {i} dim {stage_dim} not divisible by num_heads {num_heads[i]}")

        self.patch_embed = Stem(patch_size_multiplier=patch_size, 
                                in_chans=in_chans, embed_dim=embed_dim)
        
        if self.ape:
            # Initialize APE based on img_size for reference, will be interpolated
            img_h, img_w = to_2tuple(img_size)
            init_patches_h = img_h // patch_size
            init_patches_w = img_w // patch_size
            init_num_patches = init_patches_h * init_patches_w
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, init_num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            # Store init H, W for interpolation reference
            self.ape_init_H = init_patches_h 
            self.ape_init_W = init_patches_w

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        self.num_features_per_stage = [] 
        current_dim = embed_dim
        for i_layer in range(self.num_layers):
            self.num_features_per_stage.append(current_dim)
            layer = BasicLayer(dim=current_dim,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
            if i_layer < self.num_layers - 1: # If downsample is applied for next stage
                current_dim *= 2
        
        for i_idx in self.out_indices:
            if not (0 <= i_idx < len(self.num_features_per_stage)): # Check bounds
                 raise ValueError(f"out_indice {i_idx} is out of range for {len(self.num_features_per_stage)} stages.")
            norm_out_layer = norm_layer(self.num_features_per_stage[i_idx])
            self.add_module(f'norm{i_idx}', norm_out_layer)

        self.apply(self._init_weights)

        # Initialize width_list for compatibility with some frameworks
        # It is based on the `img_size` the model is configured with at initialization.
        try:
            dummy_input_size = to_2tuple(img_size) # Use configured img_size for this
            dummy_input = torch.randn(1, in_chans, dummy_input_size[0], dummy_input_size[1])
            
            # For width_list calculation, ensure model is in eval mode to avoid issues with dropout/batchnorm if they affect shapes.
            original_training_state = self.training
            self.eval() 
            with torch.no_grad():
                dummy_outputs = self.forward(dummy_input)
            if original_training_state: # Restore original state
                self.train()

            self.width_list = [o.shape[1] for o in dummy_outputs] # Channel is at index 1 for (B,C,H,W)
        except Exception as e:
            print(f"Warning: Could not compute self.width_list due to: {e}. Setting to empty list.")
            self.width_list = []


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'} if self.ape else {}

    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(x) # x: (B,L,C), H, W are patch map dimensions
        
        if self.ape:
            # Check if current H, W match the initialized APE dimensions
            if self.ape_init_H == H and self.ape_init_W == W and self.absolute_pos_embed.shape[1] == H*W :
                 abs_pos_embed_interp = self.absolute_pos_embed
            else: # Interpolate APE
                # Reshape APE to (1, C, ape_init_H, ape_init_W) for interpolation
                abs_pos_embed_hw = self.absolute_pos_embed.reshape(1, self.ape_init_H, self.ape_init_W, self.embed_dim).permute(0,3,1,2)
                # Interpolate to current H, W
                abs_pos_embed_interp_hw = F.interpolate(abs_pos_embed_hw, size=(H, W), mode='bicubic', align_corners=False)
                # Reshape back to (1, L, C)
                abs_pos_embed_interp = abs_pos_embed_interp_hw.permute(0,2,3,1).reshape(1, H*W, self.embed_dim)
            x = x + abs_pos_embed_interp
        x = self.pos_drop(x)

        outs = []
        current_H, current_W = H, W 

        for i_layer in range(self.num_layers):
            layer = self.layers[i_layer]
            # layer.forward returns: x_pre_down, H_pre_down, W_pre_down, x_post_down, H_post_down, W_post_down
            x_pre_down, H_pre_down, W_pre_down, x, current_H, current_W = layer(x, current_H, current_W)

            if i_layer in self.out_indices:
                norm_out_layer = getattr(self, f'norm{i_layer}')
                x_out_normed = norm_out_layer(x_pre_down) # x_pre_down is (B, L_pre, C_stage)
                
                C_stage = self.num_features_per_stage[i_layer]
                # Use H_pre_down, W_pre_down for reshaping this output
                out_reshaped = x_out_normed.reshape(B, H_pre_down, W_pre_down, C_stage).permute(0, 3, 1, 2).contiguous()
                outs.append(out_reshaped)
        
        return outs

# --- Factory Functions ---
def MLLA_Tiny(pretrained=False, img_size=224, in_chans=3, num_classes=1000, drop_path_rate=0.2, out_indices=(0,1,2,3), ape=False, use_checkpoint=False, **kwargs):
    model = MLLA(
        img_size=img_size, in_chans=in_chans,
        num_classes=num_classes if not out_indices else 0,
        embed_dim=64, depths=[2, 4, 8, 4], num_heads=[2, 4, 8, 16], 
        drop_path_rate=drop_path_rate, out_indices=out_indices, ape=ape, use_checkpoint=use_checkpoint,
        **kwargs
    )
    return model

def MLLA_Small(pretrained=False, img_size=224, in_chans=3, num_classes=1000, drop_path_rate=0.3, out_indices=(0,1,2,3), ape=False, use_checkpoint=False, **kwargs):
    model = MLLA(
        img_size=img_size, in_chans=in_chans,
        num_classes=num_classes if not out_indices else 0,
        embed_dim=64, depths=[3, 6, 21, 6], num_heads=[2, 4, 8, 16], 
        drop_path_rate=drop_path_rate, out_indices=out_indices, ape=ape, use_checkpoint=use_checkpoint,
        **kwargs
    )
    return model

def MLLA_Base(pretrained=False, img_size=224, in_chans=3, num_classes=1000, drop_path_rate=0.5, out_indices=(0,1,2,3), ape=False, use_checkpoint=False, **kwargs):
    model = MLLA(
        img_size=img_size, in_chans=in_chans,
        num_classes=num_classes if not out_indices else 0,
        embed_dim=96, depths=[3, 6, 21, 6], num_heads=[3, 6, 12, 24],
        drop_path_rate=drop_path_rate, out_indices=out_indices, ape=ape, use_checkpoint=use_checkpoint,
        **kwargs
    )
    return model

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("Testing MLLA_Tiny (configured with img_size=224, APE=True)")
    # Test with use_checkpoint=True to ensure checkpointing compatibility with AMP if used later
    model_tiny = MLLA_Tiny(img_size=224, out_indices=(0, 1, 2, 3), ape=True, use_checkpoint=True).to(device)
    
    test_sizes = [(224, 224), (256, 256), (384, 512), (640,640)] 

    for H_test, W_test in test_sizes:
        print(f"\n--- Testing MLLA_Tiny with input size: {H_test}x{W_test} ---")
        inputs = torch.randn((2, 3, H_test, W_test)).to(device) # Batch size 2
        
        # Test eval mode
        model_tiny.eval()
        with torch.no_grad():
            print("Eval mode outputs:")
            features_list_eval = model_tiny(inputs)
            for i, features in enumerate(features_list_eval):
                print(f"Output from stage {model_tiny.out_indices[i]}: shape {features.shape}")
        
        # Test train mode (if use_checkpoint=True, it's active here)
        # To test backward, we need a dummy loss
        model_tiny.train()
        print("Train mode (with checkpointing if enabled) outputs & backward pass:")
        
        # Simulating AMP context for testing the RoPE fix
        # In a real training loop, scaler would be torch.cuda.amp.GradScaler()
        # For this test, just enabling autocast is enough to trigger float16 paths.
        if device.type == 'cuda':
            with torch.cuda.amp.autocast(enabled=True):
                features_list_train = model_tiny(inputs)
                dummy_loss = sum([features.mean() for features in features_list_train])
            # Scaler would be used here in full training: scaler.scale(dummy_loss).backward()
            # For this simple test, direct backward is fine if not scaling.
            # dummy_loss.backward() will fail if autocast is not also on backward or if scaler isn't used.
            # To fully test backward with AMP:
            # scaler = torch.cuda.amp.GradScaler(enabled=True)
            # with torch.cuda.amp.autocast(enabled=True):
            #    features_list_train = model_tiny(inputs)
            #    dummy_loss = sum([features.mean() for features in features_list_train])
            # scaler.scale(dummy_loss).backward()
            # scaler.step(optimizer) # dummy optimizer
            # scaler.update()
            # print(f"Dummy loss: {dummy_loss.item()}, backward pass successful with AMP.")
            print(f"Train forward pass with AMP successful. Output shapes:")
            for i, features in enumerate(features_list_train): # features_list_train is from autocast context
                print(f"Output from stage {model_tiny.out_indices[i]}: shape {features.shape}, dtype {features.dtype}")

        else: # CPU
            features_list_train = model_tiny(inputs)
            dummy_loss = sum([features.mean() for features in features_list_train])
            dummy_loss.backward()
            print(f"Dummy loss: {dummy_loss.item()}, backward pass successful on CPU.")


    print(f"\nmodel_tiny.width_list (based on init_img_size=224): {model_tiny.width_list}")