# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math # 引入 math 來計算 H, W

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model # 如果不用timm註冊，可以註釋掉
# from timm.models.vision_transformer import _cfg # 如果不用timm配置，可以註釋掉
from einops import rearrange

# __all__ = [
#     'pola_pvt_tiny', 'pola_pvt_small', 'pola_pvt_medium', 'pola_pvt_large'
# ]

# --- Configuration Dictionary for PVT Variants ---
PVT_MODEL_SPECS = {
    "pola_pvt_tiny": {
        "patch_size": 4,
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "mlp_ratios": [8, 8, 4, 4],
        "depths": [2, 2, 2, 2],
        "sr_ratios": [8, 4, 2, 1],
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "la_sr_ratios": '8421', # Example, adjust if needed per model
        "alpha": 4,             # Example, adjust if needed per model
        "kernel_size": 5,       # Example, adjust if needed per model
        "attn_type": 'LLLL'     # Example, adjust if needed per model
    },
    "pola_pvt_small": {
        "patch_size": 4,
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "mlp_ratios": [8, 8, 4, 4],
        "depths": [3, 4, 6, 3],
        "sr_ratios": [8, 4, 2, 1],
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "la_sr_ratios": '8421',
        "alpha": 4,
        "kernel_size": 5,
        "attn_type": 'LLLL'
    },
    "pola_pvt_medium": {
        "patch_size": 4,
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "mlp_ratios": [8, 8, 4, 4],
        "depths": [3, 4, 18, 3],
        "sr_ratios": [8, 4, 2, 1],
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "la_sr_ratios": '8421',
        "alpha": 4,
        "kernel_size": 5,
        "attn_type": 'LLLL'
    },
    "pola_pvt_large": {
        "patch_size": 4,
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "mlp_ratios": [8, 8, 4, 4],
        "depths": [3, 8, 27, 3],
        "sr_ratios": [8, 4, 2, 1],
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "la_sr_ratios": '8421',
        "alpha": 4,
        "kernel_size": 5,
        "attn_type": 'LLLL'
    }
    # Add other variants if needed
}


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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PolaLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        # Calculate the number of patches after spatial reduction
        num_reduced_patches = num_patches // (sr_ratio * sr_ratio) if sr_ratio > 0 else num_patches
        # Ensure num_reduced_patches is at least 1
        num_reduced_patches = max(1, num_reduced_patches)

        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_reduced_patches, dim)))

        # print('Linear Attention sr_ratio{} f{} kernel{}'.
        #       format(sr_ratio, alpha, kernel_size)) # Keep if useful for debugging

    def forward(self, x, H, W):
        B, N, C = x.shape
        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
            # Adjust H, W after spatial reduction for DWC interpolation target size
            H_reduced, W_reduced = H // self.sr_ratio, W // self.sr_ratio
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
            H_reduced, W_reduced = H, W # No reduction
        k, v = kv[0], kv[1]
        n_reduced = k.shape[1] # Number of k/v tokens after reduction

        # Apply positional encoding to K
        # Ensure positional encoding matches the reduced sequence length
        if n_reduced != self.positional_encoding.shape[1]:
             # Simple fallback or error handling if sizes don't match unexpectedly
             # A more robust solution might involve resizing the pos_enc if needed
             # For now, let's assume n_reduced will match self.positional_encoding.shape[1]
             # Or slice/interpolate if necessary based on actual usage pattern
             pos_enc = F.interpolate(self.positional_encoding.transpose(1, 2), size=n_reduced, mode='linear', align_corners=False).transpose(1, 2)
             k = k + pos_enc
        else:
            k = k + self.positional_encoding


        kernel_function = nn.ReLU()

        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * nn.functional.sigmoid(self.power)

        q = q / scale
        k = k / scale
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous() # B, heads, N, head_dim
        k = k.reshape(B, n_reduced, self.num_heads, -1).permute(0, 2, 1, 3).contiguous() # B, heads, n_reduced, head_dim
        v = v.reshape(B, n_reduced, self.num_heads, -1).permute(0, 2, 1, 3).contiguous() # B, heads, n_reduced, head_dim

        q_pos = kernel_function(q) ** power
        q_neg = kernel_function(-q) ** power
        k_pos = kernel_function(k) ** power
        k_neg = kernel_function(-k) ** power

        q_sim = torch.cat([q_pos, q_neg],dim=-1)
        q_opp = torch.cat([q_neg, q_pos],dim=-1)
        k_cat = torch.cat([k_pos, k_neg],dim=-1) # Renamed from k to k_cat

        v1,v2 = torch.chunk(v,2,dim=-1) # Split v based on head_dim

        z_sim = 1 / (q_sim @ k_cat.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6) # Use k_cat here
        kv_sim = (k_cat.transpose(-2, -1) * (n_reduced ** -0.5)) @ (v1 * (n_reduced ** -0.5))
        x_sim = q_sim @ kv_sim * z_sim

        z_opp = 1 / (q_opp @ k_cat.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6) # Use k_cat here
        kv_opp = (k_cat.transpose(-2, -1) * (n_reduced ** -0.5)) @ (v2 * (n_reduced ** -0.5))
        x_opp = q_opp @ kv_opp * z_opp

        # Concatenate along the head_dim dimension
        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C) # B, N, C

        # --- DWC Path ---
        # Reshape v for DWC: B, heads, n_reduced, head_dim -> B*heads, head_dim, H_reduced, W_reduced
        v_dwc = v.reshape(B * self.num_heads, n_reduced, self.head_dim).transpose(1, 2).reshape(B * self.num_heads, self.head_dim, H_reduced, W_reduced)

        # Apply DWC
        v_dwc = self.dwc(v_dwc) # B*heads, head_dim, H_reduced, W_reduced

        # Interpolate if needed (usually if sr_ratio > 1, we need to upsample back to original H, W)
        if H_reduced != H or W_reduced != W:
             v_dwc = F.interpolate(v_dwc, size=(H, W), mode='bilinear', align_corners=False)

        # Reshape back: B*heads, head_dim, H, W -> B, C, N
        v_dwc = v_dwc.reshape(B, self.num_heads * self.head_dim, H * W).permute(0, 2, 1) # B, N, C

        # Add DWC path and apply gating
        x = x + v_dwc
        x = x * g

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_patches, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,
                 alpha=4, kernel_size=5, attn_type='L'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        assert attn_type in ['L', 'S'], f"attn_type must be 'L' or 'S', got {attn_type}"
        if attn_type == 'L':
            # Pass num_patches to PolaLinearAttention
            self.attn = PolaLinearAttention(
                dim, num_patches=num_patches,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
                alpha=alpha, kernel_size=kernel_size)
        else:
            # Standard Attention does not need num_patches directly
            self.attn = Attention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # Calculate H, W based on input image size and patch size
        self.H = img_size[0] // patch_size[0]
        self.W = img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H_in, W_in = x.shape
        # Dynamically calculate H, W based on actual input size
        H, W = H_in // self.patch_size[0], W_in // self.patch_size[1]

        x = self.proj(x).flatten(2).transpose(1, 2) # B, N, C
        x = self.norm(x)

        # Return the calculated H, W for this specific input
        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, model_name='pola_pvt_small', img_size=224, in_chans=3, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        assert model_name in PVT_MODEL_SPECS, f"Model name '{model_name}' not found in PVT_MODEL_SPECS"
        specs = PVT_MODEL_SPECS[model_name]

        self.patch_size = specs['patch_size']
        self.embed_dims = specs['embed_dims']
        self.num_heads = specs['num_heads']
        self.mlp_ratios = specs['mlp_ratios']
        self.depths = specs['depths']
        self.sr_ratios = specs['sr_ratios'] # SR ratios for standard Attention ('S')
        self.la_sr_ratios = [int(r) for r in specs['la_sr_ratios']] # SR ratios for Linear Attention ('L')
        self.qkv_bias = specs['qkv_bias']
        self.norm_layer = specs['norm_layer']
        self.alpha = specs['alpha']
        self.kernel_size = specs['kernel_size']
        self.attn_type = specs['attn_type'] # String like 'LLLL' or 'LSLS'

        self.num_stages = len(self.depths)
        self.img_size = to_2tuple(img_size) # Store image size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]  # stochastic depth decay rule
        cur = 0

        # Input dimensions for patch_embed stages
        current_img_size = self.img_size
        current_in_chans = in_chans

        for i in range(self.num_stages):
            # Determine patch size for the current stage
            stage_patch_size = self.patch_size if i == 0 else 2

            patch_embed = PatchEmbed(img_size=current_img_size,
                                     patch_size=stage_patch_size,
                                     in_chans=current_in_chans,
                                     embed_dim=self.embed_dims[i])

            # Calculate num_patches based on the patch_embed's calculation for this stage
            # We need the *initial* H,W from the patch_embed definition for pos_embed size
            stage_H = current_img_size[0] // stage_patch_size
            stage_W = current_img_size[1] // stage_patch_size
            num_patches = stage_H * stage_W

            pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            # Determine sr_ratio based on attn_type for the stage
            attn_type_stage = self.attn_type[i]
            if attn_type_stage == 'S':
                stage_sr_ratio = self.sr_ratios[i]
            elif attn_type_stage == 'L':
                stage_sr_ratio = self.la_sr_ratios[i]
            else:
                raise ValueError(f"Invalid attn_type '{attn_type_stage}' at stage {i}")


            block = nn.ModuleList([Block(
                dim=self.embed_dims[i],
                num_patches=num_patches, # Pass num_patches here
                num_heads=self.num_heads[i],
                mlp_ratio=self.mlp_ratios[i],
                qkv_bias=self.qkv_bias,
                qk_scale=None, # qk_scale typically None for PVT
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + j],
                norm_layer=self.norm_layer,
                sr_ratio=stage_sr_ratio, # Use stage-specific sr_ratio
                alpha=self.alpha,
                kernel_size=self.kernel_size,
                attn_type=attn_type_stage) # Use stage-specific attn_type
                for j in range(self.depths[i])])

            # Layer norm after each stage's blocks
            norm = self.norm_layer(self.embed_dims[i])

            cur += self.depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm) # Store norm layer for each stage

            # Update input dimensions for the next stage
            current_in_chans = self.embed_dims[i]
            # Image size effectively halves at each stage after the first one
            if i > 0:
                 current_img_size = (current_img_size[0] // 2, current_img_size[1] // 2)


        # Initialize weights
        self._apply_init_weights()

        # Calculate width_list (similar to MobileNetV4)
        # Use a dummy input matching the expected configuration
        # Ensure model parameters are on a device before dummy forward
        # Use default image size for this calculation
        try:
            # Create dummy input on CPU first
            dummy_input = torch.randn(1, in_chans, self.img_size[0], self.img_size[1])
            # If parameters exist and are on a device, move input there
            if next(self.parameters(), None) is not None:
                 device = next(self.parameters()).device
                 dummy_input = dummy_input.to(device)

            self.eval() # Set model to evaluation mode for the dummy forward
            with torch.no_grad():
                 # Perform dummy forward pass
                 features = self.forward(dummy_input)
                 # Extract channel dimensions (dim=1 for BCHW format)
                 self.width_list = [f.size(1) for f in features]
            self.train() # Set model back to train mode
        except Exception as e:
            print(f"Warning: Could not compute width_list during init: {e}")
            # Provide default or expected widths based on embed_dims if calculation fails
            self.width_list = self.embed_dims # Fallback

    def _apply_init_weights(self):
        """ Applies weight initialization """
        for i in range(self.num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
             # Initialize convolutions, e.g., kaiming normal
             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             if m.bias is not None:
                 nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
             if m.weight is not None:
                 nn.init.constant_(m.weight, 1.0)
             if m.bias is not None:
                 nn.init.constant_(m.bias, 0)


    @torch.jit.ignore
    def no_weight_decay(self):
        # Return param names that should not have weight decay
        # Typically includes positional embeddings and norm layer biases/scales
        no_decay = set()
        for i in range(self.num_stages):
            no_decay.add(f"pos_embed{i + 1}")

        for name, param in self.named_parameters():
            if 'norm' in name and ('weight' in name or 'bias' in name):
                no_decay.add(name)
            if 'bias' in name: # Often biases are excluded too
                 no_decay.add(name)

        # Add PolaLinearAttention specific parameters if desired
        for name, param in self.named_parameters():
            if 'pola' in name.lower() and ('scale' in name or 'power' in name):
                 no_decay.add(name)

        return no_decay


    def _get_pos_embed(self, pos_embed, H, W):
        """ Resizes positional embedding if input resolution changes. """
        # Original pos_embed shape: [1, num_patches, embed_dim]
        # Target grid size: H, W
        num_patches_target = H * W
        if pos_embed.shape[1] == num_patches_target:
            return pos_embed
        else:
            # Infer original grid size (H0, W0) from pos_embed shape
            embed_dim = pos_embed.shape[-1]
            num_patches_orig = pos_embed.shape[1]
            # Attempt to find H0, W0 - requires knowing aspect ratio or assuming square
            H0 = W0 = int(math.sqrt(num_patches_orig))
            if H0 * W0 != num_patches_orig:
                 # Fallback if not square - requires patch_embed info or assumptions
                 # This part might need adjustment based on how H0, W0 are determined
                 # Let's assume patch_embed stored H, W corresponding to pos_embed
                 # print(f"Warning: Pos embed resizing for non-square grid ({num_patches_orig} patches) might be inaccurate.")
                 # Try to find factors, simplest case: W0 = H0 or W0 = H0 * ratio
                 # This needs a more robust way if used with non-square inputs/patches often.
                 # Using sqrt as an approximation for now.
                 pass # Use H0=W0=sqrt approximation

            # Reshape to 2D grid format, interpolate, and reshape back
            # [1, num_patches, C] -> [1, H0, W0, C] -> [1, C, H0, W0]
            pos_embed_grid = pos_embed.reshape(1, H0, W0, embed_dim).permute(0, 3, 1, 2)
            # Interpolate: [1, C, H0, W0] -> [1, C, H, W]
            pos_embed_resized = F.interpolate(pos_embed_grid, size=(H, W), mode="bilinear", align_corners=False)
            # Reshape back: [1, C, H, W] -> [1, H, W, C] -> [1, H*W, C]
            pos_embed_final = pos_embed_resized.permute(0, 2, 3, 1).reshape(1, H * W, embed_dim)
            return pos_embed_final

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block_list = getattr(self, f"block{i + 1}") # Renamed 'block' -> 'block_list'
            norm = getattr(self, f"norm{i + 1}")

            x, (H, W) = patch_embed(x) # x: B, N, C

            # Resize pos_embed if needed and add it
            pos_embed_resized = self._get_pos_embed(pos_embed, H, W)
            x = pos_drop(x + pos_embed_resized)

            # Pass through blocks
            for blk in block_list:
                x = blk(x, H, W) # x: B, N, C

            # Apply norm
            x = norm(x)

            # Reshape to BCHW format for the output list
            # x: B, N, C -> B, H, W, C -> B, C, H, W
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        # The loop naturally handles the input 'x' for the next stage,
        # as the patch_embed in the next iteration takes the BCHW output
        # from the previous stage.

        return outs # Return list of features [B, C_i, H_i, W_i]

# --- Wrapper functions for specific model sizes ---

# @register_model # Uncomment if using timm registration
def pola_pvt_tiny(pretrained=False, img_size=224, **kwargs):
    model = PyramidVisionTransformer(model_name='pola_pvt_tiny', img_size=img_size, **kwargs)
    # model.default_cfg = _cfg() # Uncomment if using timm config
    # Load pretrained weights logic here if needed
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(...)
    #     model.load_state_dict(checkpoint)
    return model

# @register_model # Uncomment if using timm registration
def pola_pvt_small(pretrained=False, img_size=224, **kwargs):
    model = PyramidVisionTransformer(model_name='pola_pvt_small', img_size=img_size, **kwargs)
    # model.default_cfg = _cfg() # Uncomment if using timm config
    return model

# @register_model # Uncomment if using timm registration
def pola_pvt_medium(pretrained=False, img_size=224, **kwargs):
    model = PyramidVisionTransformer(model_name='pola_pvt_medium', img_size=img_size, **kwargs)
    # model.default_cfg = _cfg() # Uncomment if using timm config
    return model

# @register_model # Uncomment if using timm registration
def pola_pvt_large(pretrained=False, img_size=224, **kwargs):
    model = PyramidVisionTransformer(model_name='pola_pvt_large', img_size=img_size, **kwargs)
    # model.default_cfg = _cfg() # Uncomment if using timm config
    return model

# --- Example Usage ---
if __name__ == "__main__":
    # Choose input size
    input_h, input_w = 224, 224 # Or 640, 640 or other

    # Generating Sample image
    image_size = (1, 3, input_h, input_w)
    image = torch.rand(*image_size)

    print(f"Input image shape: {image.shape}")

    # --- Test different models ---
    model_names = ['pola_pvt_tiny', 'pola_pvt_small', 'pola_pvt_medium', 'pola_pvt_large']
    model_funcs = [pola_pvt_tiny, pola_pvt_small, pola_pvt_medium, pola_pvt_large]

    for name, func in zip(model_names, model_funcs):
        print(f"\n--- Testing {name} ---")
        try:
            # Instantiate the model using the wrapper function
            model = func(img_size=input_h) # Pass img_size if different from default 224
            model.eval()

            # Perform forward pass
            with torch.no_grad():
                features = model(image)

            print(f"Number of output feature maps: {len(features)}")
            print("Output feature map shapes:")
            for i, feat in enumerate(features):
                print(f"  Stage {i+1}: {feat.shape}")

            # Print the calculated width list
            print(f"Calculated width_list: {model.width_list}")

            # Verify width_list matches output channels
            output_channels = [f.size(1) for f in features]
            print(f"Actual output channels: {output_channels}")
            assert model.width_list == output_channels, "Mismatch between width_list and actual output channels!"
            print("Width list matches actual output channels.")

        except Exception as e:
            print(f"Error testing {name}: {e}")
            import traceback
            traceback.print_exc()