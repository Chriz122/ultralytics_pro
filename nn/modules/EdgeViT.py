from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
# from timm.models.vision_transformer import _cfg # No longer needed for this structure
# from timm.models.registry import register_model # No longer needed for this structure
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import warnings

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


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,  sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = sr_ratio
        if self.sr > 1:
            # sampler can remain AdaptiveAvgPool2d but needs size in forward pass
            # Or use AvgPool2d if stride is fixed
            # Using AvgPool2d might be simpler if sr_ratio is fixed
            self.sampler = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)
            # --- FIX: Replace lambda with nn.Upsample ---
            self.LocalProp = nn.Upsample(scale_factor=sr_ratio, mode='nearest')
            # --- End Fix ---
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.LocalProp = nn.Identity()
            self.norm = nn.Identity()


    def forward(self, x, H:int, W:int):
        B, N, C = x.shape
        x_spatial = x.transpose(1, 2).reshape(B, C, H, W) # Keep spatial form for sampling

        if self.sr > 1.:
            # Apply sampler
            x_sampled = self.sampler(x_spatial)
            sampled_H, sampled_W = x_sampled.shape[2:] # Get actual sampled H, W
            x_sampled = x_sampled.flatten(2).transpose(1, 2) # Shape: B, N_sampled, C
        else:
            x_sampled = x # Use original x if no sampling
            sampled_H, sampled_W = H, W # Use original H, W if no sampling

        # Ensure N_sampled matches the flattened sampled tensor's sequence length
        N_sampled = sampled_H * sampled_W
        qkv = self.qkv(x_sampled).reshape(B, N_sampled, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # qkv shape: [3, B, num_heads, N_sampled, C // num_heads]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Result shape: [B, num_heads, N_sampled, C // num_heads]
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N_sampled, C) # Shape: B, N_sampled, C

        if self.sr > 1:
            # Reshape back to spatial form before upsampling/LocalProp
            x_attn_spatial = x_attn.permute(0, 2, 1).reshape(B, C, sampled_H, sampled_W)
            # Use the nn.Upsample module directly
            x_upsampled = self.LocalProp(x_attn_spatial)

            # Optional: Interpolate to exact original H, W if sampler/upsampler caused slight size mismatch
            # This can happen if H or W are not perfectly divisible by sr_ratio
            if x_upsampled.shape[2] != H or x_upsampled.shape[3] != W:
                 x_upsampled = F.interpolate(x_upsampled, size=(H, W), mode='nearest')

            x_upsampled_flat = x_upsampled.reshape(B, C, -1).permute(0, 2, 1) # Shape: B, N, C
            x = self.norm(x_upsampled_flat) # Apply norm
        else:
             x = x_attn # Use attention output directly if sr=1

        x = self.proj(x)
        x = self.proj_drop(x)
        # The SelfAttn block expects B, N, C output, which this provides.
        # The reshape back to B, C, H, W happens at the end of SelfAttn.forward
        return x


class LocalAgg(nn.Module):
    def __init__(self, dim, num_heads=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d): # Changed norm layer default
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # Use BatchNorm2d for convolutional paths
        self.norm1 = norm_layer(dim) if norm_layer else nn.Identity()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # Add activation after attn? Often helps. GELU is common in transformers.
        self.attn_act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim) if norm_layer else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Input x is expected in B, C, H, W format
        x = x + self.pos_embed(x)
        # Applying norm -> conv1 -> attn -> conv2 -> drop_path
        shortcut = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.attn(x)
        x = self.attn_act(x) # Added activation
        x = self.conv2(x)
        x = shortcut + self.drop_path(x) # Residual connection for the conv path

        # Applying norm -> mlp -> drop_path
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x) # Residual connection for the MLP path
        return x


class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalSparseAttn(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Input x is expected in B, C, H, W format
        x = x + self.pos_embed(x) # Apply pos_embed spatially

        # Reshape for Attention + MLP
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # Shape: B, N, C

        # Attention path
        shortcut = x_flat
        x_norm1 = self.norm1(x_flat)
        x_attn = self.attn(x_norm1, H, W) # Pass H, W
        x_flat = shortcut + self.drop_path(x_attn) # Residual for attention

        # MLP path
        shortcut = x_flat
        x_norm2 = self.norm2(x_flat)
        x_mlp = self.mlp(x_norm2)
        x_flat = shortcut + self.drop_path(x_mlp) # Residual for MLP

        # Reshape back to B, C, H, W for the next block (if any) or output
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x


class LGLBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer_sa=nn.LayerNorm, norm_layer_la=nn.BatchNorm2d, sr_ratio=1.):
        super().__init__()

        # Local Aggregation uses Conv + BatchNorm
        self.LocalAgg = LocalAgg(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                 drop_path, act_layer, norm_layer_la)

        # Self Attention uses Linear + LayerNorm
        self.SelfAttn = SelfAttn(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                 drop_path, act_layer, norm_layer_sa, sr_ratio)

    def forward(self, x):
        # LGL structure: Apply LocalAgg first, then SelfAttn
        # Both expect B, C, H, W input format
        x = self.LocalAgg(x)
        x = self.SelfAttn(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=nn.LayerNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # Calculate patch resolution dynamically
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # Check if the input size matches the configured img_size
        # Allow for dynamic input sizes if needed, but warn if different from init
        if H != self.img_size[0] or W != self.img_size[1]:
            #  warnings.warn(
            #      f"Input image size ({H}*{W}) doesn't match model's configured image size ({self.img_size[0]}*{self.img_size[1]})."
            #      "Patch embeddings may mismatch if not designed for dynamic input sizes.", UserWarning
            #  )
             # Recalculate grid_size based on actual input H, W for this forward pass
             current_grid_size = (H // self.patch_size[0], W // self.patch_size[1])
        else:
             current_grid_size = self.grid_size

        x = self.proj(x) # Output shape: B, embed_dim, grid_H, grid_W
        B, E, H_grid, W_grid = x.shape

        # Flatten and apply norm (if used) - keep spatial format for EdgeVit blocks
        # x = x.flatten(2).transpose(1, 2) # B, N, C
        # x = self.norm(x)
        # Reshape back to spatial format: B, C, H, W (for conv blocks)
        # x = x.transpose(1, 2).reshape(B, E, H_grid, W_grid)
        # EdgeVit blocks expect B, C, H, W, so just return projection output
        return x


class EdgeVit(nn.Module):
    """ EdgeVit Feature Extractor based on the original implementation """
    def __init__(self, depth, embed_dim, head_dim, mlp_ratio, qkv_bias, qk_scale, sr_ratios,
                 img_size=224, in_chans=3, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer_sa=None, norm_layer_la=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = to_2tuple(img_size)
        self.in_chans = in_chans
        norm_layer_sa = norm_layer_sa or partial(nn.LayerNorm, eps=1e-6)
        norm_layer_la = norm_layer_la or nn.BatchNorm2d # Default for LocalAgg

        # Patch Embeddings for each stage
        self.patch_embed1 = PatchEmbed(
            img_size=self.img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        current_size = (self.img_size[0] // 4, self.img_size[1] // 4)
        self.patch_embed2 = PatchEmbed(
            img_size=current_size, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        current_size = (current_size[0] // 2, current_size[1] // 2)
        self.patch_embed3 = PatchEmbed(
            img_size=current_size, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        current_size = (current_size[0] // 2, current_size[1] // 2)
        self.patch_embed4 = PatchEmbed(
            img_size=current_size, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // h_dim for dim, h_dim in zip(embed_dim, head_dim)]

        # Build blocks for each stage
        self.blocks1 = nn.ModuleList([
            LGLBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer_sa=norm_layer_sa, norm_layer_la=norm_layer_la, sr_ratio=sr_ratios[0])
            for i in range(depth[0])])

        self.blocks2 = nn.ModuleList([
            LGLBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]],
                norm_layer_sa=norm_layer_sa, norm_layer_la=norm_layer_la, sr_ratio=sr_ratios[1])
            for i in range(depth[1])])

        self.blocks3 = nn.ModuleList([
            LGLBlock(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]],
                norm_layer_sa=norm_layer_sa, norm_layer_la=norm_layer_la, sr_ratio=sr_ratios[2])
            for i in range(depth[2])])

        self.blocks4 = nn.ModuleList([
            LGLBlock(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]],
                norm_layer_sa=norm_layer_sa, norm_layer_la=norm_layer_la, sr_ratio=sr_ratios[3])
            for i in range(depth[3])])

        # Final norm layer - use BatchNorm as it's common after conv blocks
        # self.norm = norm_layer_la(embed_dim[-1]) # Optional: Apply norm after last stage

        self.apply(self._init_weights)

        # Calculate width_list using a dummy forward pass
        # Ensure the model is in eval mode to disable dropout etc. for this pass
        self.eval()
        try:
            # Use configured img_size and in_chans for the dummy input
            dummy_input = torch.randn(1, self.in_chans, self.img_size[0], self.img_size[1])
            with torch.no_grad():
                 outputs = self.forward(dummy_input)
            self.width_list = [o.size(1) for o in outputs] # Get channel dimension (dim=1 for B,C,H,W)
        except Exception as e:
            print(f"Warning: Failed to compute width_list during init: {e}")
            self.width_list = embed_dim # Fallback
        self.train() # Set back to train mode

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
             # He initialization for Conv2d
             fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
             nn.init.normal_(m.weight, std=math.sqrt(2. / fan_in))
             if m.bias is not None:
                 nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # Typically LayerNorm/BatchNorm weights/biases and pos_embed are excluded
        # Add parameters here if needed
        return {'pos_embed'}

    def forward(self, x):
        outputs = []

        # Stage 1
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        outputs.append(x) # Output after stage 1 blocks

        # Stage 2
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        outputs.append(x) # Output after stage 2 blocks

        # Stage 3
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        outputs.append(x) # Output after stage 3 blocks

        # Stage 4
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        # Optional: Apply final norm if needed
        # x = self.norm(x)
        outputs.append(x) # Output after stage 4 blocks

        return outputs # Return list of features [x1, x2, x3, x4]

# --- Model Instantiation Classes ---

class EdgeVitXXS(EdgeVit):
    def __init__(self, img_size=224, in_chans=3, **kwargs):
        super().__init__(
            depth=[1, 1, 3, 2],
            embed_dim=[36, 72, 144, 288],
            head_dim=[36, 36, 36, 36], # Assuming head_dim matches embed_dim/num_heads, adjust if needed
            mlp_ratio=[4]*4,
            qkv_bias=True,
            qk_scale=None,
            sr_ratios=[4, 2, 2, 1],
            img_size=img_size,
            in_chans=in_chans,
            norm_layer_sa=partial(nn.LayerNorm, eps=1e-6),
            norm_layer_la=nn.BatchNorm2d,
            **kwargs)

class EdgeVitXS(EdgeVit):
    def __init__(self, img_size=224, in_chans=3, **kwargs):
        super().__init__(
            depth=[1, 1, 3, 1],
            embed_dim=[48, 96, 240, 384],
            head_dim=[48, 48, 48, 48], # Adjust head_dim if needed
            mlp_ratio=[4]*4,
            qkv_bias=True,
            qk_scale=None,
            sr_ratios=[4, 2, 2, 1],
            img_size=img_size,
            in_chans=in_chans,
            norm_layer_sa=partial(nn.LayerNorm, eps=1e-6),
            norm_layer_la=nn.BatchNorm2d,
            **kwargs)

class EdgeVitS(EdgeVit):
     def __init__(self, img_size=224, in_chans=3, **kwargs):
        super().__init__(
            depth=[1, 2, 5, 3],
            embed_dim=[48, 96, 240, 384],
            head_dim=[48, 48, 48, 48], # Leads to num_heads = [1, 2, 5, 8]
            mlp_ratio=[4]*4,
            qkv_bias=True,
            qk_scale=None,
            sr_ratios=[4, 2, 2, 1],
            img_size=img_size,
            in_chans=in_chans,
            norm_layer_sa=partial(nn.LayerNorm, eps=1e-6),
            norm_layer_la=nn.BatchNorm2d,
            **kwargs)

# --- Example Usage ---
if __name__ == "__main__":
    # Generating Sample image
    image_size_tuple = (640, 640) # Use tuple for img_size
    image = torch.rand(1, 3, *image_size_tuple)

    # Model Instantiation (using the new classes)
    # model = EdgeVitXXS(img_size=image_size_tuple)
    # model = EdgeVitXS(img_size=image_size_tuple)
    model = EdgeVitS(img_size=image_size_tuple)

    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {image.shape}")

    # Forward pass
    model.eval() # Set to eval mode for inference
    with torch.no_grad():
        out_features = model(image)

    print("Output features:")
    for i, features in enumerate(out_features):
        print(f"  Stage {i+1} output shape: {features.shape}")

    print(f"Calculated width_list: {model.width_list}")

    # Verify width_list matches output channels
    output_channels = [o.size(1) for o in out_features]
    print(f"Actual output channels: {output_channels}")
    assert model.width_list == output_channels, "Mismatch between width_list and actual output channels!"
    print("Width list verification successful.")