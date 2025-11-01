import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

# Imports from timm
try:
    from timm.models.efficientnet import EfficientNet
    from timm.models.vision_transformer import _cfg
    from timm.models.registry import register_model
    from timm.models.helpers import load_pretrained
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    print("timm library not found. Some functionalities like pretrained model loading might not work.")
    def _cfg(url='', **kwargs): return {'url': url, **kwargs}
    def register_model(fn): return fn
    class DropPath(nn.Module):
        def __init__(self, drop_prob=None): super(DropPath, self).__init__(); self.drop_prob = drop_prob
        def forward(self, x):
            if self.drop_prob == 0. or not self.training: return x
            keep_prob = 1 - self.drop_prob; shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device); random_tensor.floor_()
            return x.div(keep_prob) * random_tensor
    def to_2tuple(x): return x if isinstance(x, tuple) else (x, x)
    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        with torch.no_grad(): return tensor.normal_(mean, std).clamp_(min=a*std+mean, max=b*std+mean)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class GPSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 locality_strength=1., use_local_init=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        
        self.rel_indices = None 
        self._last_N_for_rel_indices = -1 

        if use_local_init:
            self.local_init(locality_strength=locality_strength)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        if self.rel_indices is None or self._last_N_for_rel_indices != N:
            self.get_rel_indices(N, x.device)
            self._last_N_for_rel_indices = N
        
        if self.rel_indices.device != x.device:
            self.rel_indices = self.rel_indices.to(x.device)

        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        
        pos_score = self.rel_indices.expand(B, -1, -1, -1) 
        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn_sum = attn.sum(dim=-1, keepdim=True)
        attn = attn / (attn_sum + 1e-8)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map=False):
        B, N, C = x.shape 
        if self.rel_indices is None or self._last_N_for_rel_indices != N:
             self.get_rel_indices(N, x.device)
             self._last_N_for_rel_indices = N
        if self.rel_indices.device != x.device:
            self.rel_indices = self.rel_indices.to(x.device)

        attn_map = self.get_attention(x).mean(0)
        distances = self.rel_indices.squeeze(0)[:, :, -1] ** .5 
        dist = torch.einsum('nm,hnm->h', (distances, attn_map)) 
        dist /= N 
        if return_map: return dist, attn_map
        else: return dist

    def local_init(self, locality_strength=1.):
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1.0
        kernel_size = int(self.num_heads ** .5)
        center = (kernel_size - 1) / 2.0 if kernel_size % 2 != 0 else (kernel_size / 2.0 - 0.5)
        
        for h2_idx in range(kernel_size): 
            for h1_idx in range(kernel_size): 
                position = h1_idx + h2_idx * kernel_size 
                if position < self.num_heads:
                    self.pos_proj.weight.data[position, 2] = -1.0 * locality_strength
                    self.pos_proj.weight.data[position, 0] = -2.0 * (h1_idx - center) * locality_distance * locality_strength
                    self.pos_proj.weight.data[position, 1] = -2.0 * (h2_idx - center) * locality_distance * locality_strength

    def get_rel_indices(self, num_patches, device):
        img_size_float = num_patches ** 0.5
        img_size = int(img_size_float)

        if abs(img_size_float - img_size) > 1e-6 : # Check if not a perfect square more robustly
             raise ValueError(f"GPSA.get_rel_indices: num_patches ({num_patches}) must be a perfect square "
                              f"to compute relative spatial indices. Got img_size_float={img_size_float}.")

        rel_indices_tensor = torch.zeros(1, num_patches, num_patches, 3, device=device)
        patch_row_coords = torch.arange(img_size, device=device).repeat_interleave(img_size) 
        patch_col_coords = torch.arange(img_size, device=device).repeat(img_size)            
        rel_idx_x = patch_col_coords.view(-1, 1) - patch_col_coords.view(1, -1) 
        rel_idx_y = patch_row_coords.view(-1, 1) - patch_row_coords.view(1, -1) 
        rel_idx_dist_sq = rel_idx_x.pow(2) + rel_idx_y.pow(2) 
        rel_indices_tensor[0, :, :, 0] = rel_idx_x
        rel_indices_tensor[0, :, :, 1] = rel_idx_y
        rel_indices_tensor[0, :, :, 2] = rel_idx_dist_sq
        self.rel_indices = rel_indices_tensor

class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0) 

        img_size_approx = int(N ** .5) 
        
        if abs(N**0.5 - img_size_approx) < 1e-6 : 
            patch_row_coords = torch.arange(img_size_approx, device=attn_map.device).repeat_interleave(img_size_approx)
            patch_col_coords = torch.arange(img_size_approx, device=attn_map.device).repeat(img_size_approx)
            rel_idx_x = patch_col_coords.view(-1, 1) - patch_col_coords.view(1, -1)
            rel_idx_y = patch_row_coords.view(-1, 1) - patch_row_coords.view(1, -1)
            distances = (rel_idx_x.pow(2) + rel_idx_y.pow(2)).float().sqrt() 
            dist = torch.einsum('nm,hnm->h', (distances, attn_map))
            dist /= N
        else: 
            distances = torch.abs(torch.arange(N, device=attn_map.device).view(1,-1) - torch.arange(N, device=attn_map.device).view(-1,1)).float()
            dist = torch.einsum('nm,hnm->h', (distances, attn_map))
            dist /= N

        if return_map: return dist, attn_map
        else: return dist

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1); attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x); x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_gpsa=True, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_gpsa = use_gpsa
        if self.use_gpsa:
            self.attn = GPSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             attn_drop=attn_drop, proj_drop=drop, 
                             locality_strength=kwargs.get('locality_strength', 1.0), 
                             use_local_init=kwargs.get('use_local_init', True))
        else:
            self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_norm1 = self.norm1(x)
        x = x + self.drop_path(self.attn(x_norm1))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = to_2tuple(img_size) # Stored configured img_size
        self.patch_size = to_2tuple(patch_size)
        # grid_size and num_patches will be determined by actual input H, W in forward for dynamic support
        # For fixed size, self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
             trunc_normal_(m.weight, std=.02)
             if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        # For dynamic input sizes, don't assert H, W against self.img_size here.
        # The VisionTransformer's pos_embed interpolation will handle size differences.
        # Calculate grid_size based on actual H, W
        self.current_grid_size = (H // self.patch_size[0], W // self.patch_size[1])
        self.current_num_patches = self.current_grid_size[0] * self.current_grid_size[1]

        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            # This can happen if input image size is not a multiple of patch size.
            # ViT typically expects this. Padding or error.
            # For YOLO, input sizes are often multiples of strides, which usually align with patches.
            # print(f"Warning: PatchEmbed input H({H}) or W({W}) not divisible by patch_size ({self.patch_size}). "
            #       "Feature map dimensions might be fractional if not handled by proj.")
            # nn.Conv2d will handle this by truncating if stride > 1.
            pass

        x = self.proj(x).flatten(2).transpose(1, 2) # B, N_patches, C
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 local_up_to_layer=10, locality_strength=1., use_pos_embed=True,
                 out_indices=None):
        super().__init__()
        self.num_classes = num_classes
        self.local_up_to_layer = local_up_to_layer 
        self.embed_dim = embed_dim
        self.depth = depth
        self.locality_strength = locality_strength 
        self.use_pos_embed = use_pos_embed
        self.img_size_tuple = to_2tuple(img_size) # Store configured img_size
        self.patch_size = to_2tuple(patch_size) # Store patch_size

        if hybrid_backbone is not None: 
            raise NotImplementedError("Hybrid backbone not fully verified with current ConViT setup.")
        else:
            # Pass configured img_size to PatchEmbed for its internal self.img_size,
            # but PatchEmbed.forward will use actual input H,W for its grid_size.
            self.patch_embed = PatchEmbed(
                img_size=self.img_size_tuple, patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        # num_patches and patch_HW based on configured img_size, used for pos_embed definition
        self.config_patch_HW = (self.img_size_tuple[0] // self.patch_size[0], self.img_size_tuple[1] // self.patch_size[1])
        self.config_num_patches = self.config_patch_HW[0] * self.config_patch_HW[1]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.config_num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        else:
            self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_gpsa_in_block = (i < local_up_to_layer)
            self.blocks.append(
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    use_gpsa=use_gpsa_in_block,
                    locality_strength=locality_strength 
                )
            )
        self.norm = norm_layer(embed_dim) 
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.out_indices = []
        if out_indices is not None:
            if not all(idx < depth for idx in out_indices):
                raise ValueError(f"All out_indices must be less than depth ({depth}). Got {out_indices}")
            self.out_indices = sorted(list(set(out_indices))) 

        self.feature_channels = [] 
        self.num_features_out = [] 
        self.width_list = [] # ADDED FOR YOLO COMPATIBILITY

        if self.out_indices:
            self.num_features_out = [embed_dim] * len(self.out_indices)
            try:
                is_training = self.training
                self.eval()
                with torch.no_grad():
                    # Use configured img_size for dummy forward
                    dummy_h, dummy_w = self.img_size_tuple
                    # Ensure dummy input dimensions are divisible by patch_size for PatchEmbed proj
                    if dummy_h % self.patch_size[0] != 0:
                        dummy_h = (dummy_h // self.patch_size[0] + 1) * self.patch_size[0]
                    if dummy_w % self.patch_size[1] != 0:
                        dummy_w = (dummy_w // self.patch_size[1] + 1) * self.patch_size[1]
                    dummy_input = torch.randn(1, in_chans, dummy_h, dummy_w)
                    features = self.forward(dummy_input) 
                self.train(is_training)

                if isinstance(features, list):
                    self.feature_channels = [f.size(1) for f in features]
                    self.width_list = self.feature_channels # Assign to width_list
                else:
                    self.feature_channels = [self.embed_dim] * len(self.out_indices)
                    self.width_list = self.feature_channels # Assign to width_list
            except Exception as e:
                # print(f"Warning: Could not perform dummy forward pass to get feature_channels/width_list: {e}")
                # import traceback; traceback.print_exc()
                self.feature_channels = [self.embed_dim] * len(self.out_indices)
                self.width_list = self.feature_channels # Assign to width_list
        # print(f"Initialized VisionTransformer. width_list: {self.width_list}")


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
             trunc_normal_(m.weight, std=.02)
             if m.bias is not None: nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'gating_param'}

    def _interpolate_pos_embed(self, x_patch_tokens, current_H_patch, current_W_patch):
        if self.pos_embed is None:
            return x_patch_tokens
        
        num_current_patches = current_H_patch * current_W_patch
        
        # Check if interpolation is needed
        if num_current_patches == self.config_num_patches and \
           current_H_patch == self.config_patch_HW[0] and \
           current_W_patch == self.config_patch_HW[1] and \
           self.pos_embed.shape[1] == self.config_num_patches:
            return x_patch_tokens + self.pos_embed

        # print(f"Interpolating pos_embed: from {self.config_num_patches} patches ({self.config_patch_HW}) "
        #       f"to {num_current_patches} patches ({current_H_patch, current_W_patch})")
        
        C = self.embed_dim
        # Reshape original pos_embed to 2D grid: (1, H_config_patch, W_config_patch, C) -> (1, C, H_config_patch, W_config_patch)
        pos_embed_2d = self.pos_embed.reshape(1, self.config_patch_HW[0], self.config_patch_HW[1], C).permute(0, 3, 1, 2)
        
        # Interpolate
        pos_embed_interp_2d = F.interpolate(pos_embed_2d, size=(current_H_patch, current_W_patch), mode='bicubic', align_corners=False)
        
        # Reshape back to (1, num_current_patches, C)
        pos_embed_interp_flat = pos_embed_interp_2d.permute(0, 2, 3, 1).reshape(1, num_current_patches, C)
        
        return x_patch_tokens + pos_embed_interp_flat

    def forward(self, x_input):
        B, _, H_in, W_in = x_input.shape 
        
        x_patch_tokens = self.patch_embed(x_input) # B, N_current_patches, C
        # Get current patch grid dimensions from PatchEmbed (which calculated it based on H_in, W_in)
        current_H_patch, current_W_patch = self.patch_embed.current_grid_size
        num_current_patches = self.patch_embed.current_num_patches
        
        if self.use_pos_embed:
            x_patch_tokens = self._interpolate_pos_embed(x_patch_tokens, current_H_patch, current_W_patch)

        x_patch_tokens = self.pos_drop(x_patch_tokens)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        current_x = x_patch_tokens 
        outs = []

        for i, blk in enumerate(self.blocks):
            if i == self.local_up_to_layer:
                current_x = torch.cat((cls_tokens, current_x), dim=1)
            current_x = blk(current_x)

            if i in self.out_indices:
                feature_map_tokens = current_x[:, 1:, :] if i >= self.local_up_to_layer and current_x.shape[1] == (num_current_patches + 1) else current_x
                # Ensure feature_map_tokens has num_current_patches tokens
                if feature_map_tokens.shape[1] != num_current_patches:
                    # This could happen if CLS token logic is misaligned with out_indices selection
                    # Example: out_indices requests a layer before CLS token, but CLS token was already added due to local_up_to_layer
                    # Or, out_indices requests a layer after CLS, but CLS was not added or sliced incorrectly.
                    # For safety, if shape is num_current_patches + 1, assume CLS is first and slice.
                    if feature_map_tokens.shape[1] == num_current_patches + 1:
                         feature_map_tokens = feature_map_tokens[:,1:,:]
                    else:
                         # This is an unexpected state.
                         raise RuntimeError(f"Token mismatch for feature extraction at layer {i}. "
                                            f"Expected {num_current_patches} patch tokens, got {feature_map_tokens.shape[1]}. "
                                            f"current_x shape: {current_x.shape}, i: {i}, local_up_to_layer: {self.local_up_to_layer}")

                out_tensor = feature_map_tokens.transpose(1, 2).reshape(B, self.embed_dim, current_H_patch, current_W_patch)
                outs.append(out_tensor)
        
        if self.out_indices:
            return outs 

        current_x = self.norm(current_x) 
        if self.num_classes > 0:
            is_cls_present_final = (self.local_up_to_layer < self.depth and current_x.shape[1] == (num_current_patches + 1))
            if is_cls_present_final:
                return self.head(current_x[:, 0]) 
            else: # CLS not present or not added, average pool patch tokens
                return self.head(current_x.mean(dim=1))
        else: 
            is_cls_present_final = (self.local_up_to_layer < self.depth and current_x.shape[1] == (num_current_patches + 1))
            if is_cls_present_final:
                 return current_x[:, 0] 
            else:
                 return current_x

# --- Factory functions (unchanged from previous, ensure they pass img_size correctly) ---

def _load_pretrained(model, url, filter_keys=None, strict=True):
    if not url:
        # print("No pretrained URL specified.")
        return model
    try:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True
        )
        if 'model' in checkpoint: checkpoint = checkpoint['model']
        if filter_keys:
            checkpoint_filtered = {k: v for k, v in checkpoint.items() if all(fk not in k for fk in filter_keys)}
            model.load_state_dict(checkpoint_filtered, strict=strict)
        else:
            model.load_state_dict(checkpoint, strict=strict)
        # print(f"Loaded pretrained weights from {url}")
    except Exception as e:
        print(f"Failed to load pretrained weights from {url}: {e}")
    return model

@register_model
def convit_tiny(pretrained=False, img_size=224, **kwargs): # Added img_size
    model_kwargs = dict(
        img_size=img_size, patch_size=16, embed_dim=(48 * 4), depth=12, num_heads=4,
        local_up_to_layer=10, locality_strength=1.0,mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model = VisionTransformer(**model_kwargs)
    model.default_cfg = _cfg(url="https://dl.fbaipublicfiles.com/convit/convit_tiny.pth")
    if pretrained: _load_pretrained(model, model.default_cfg['url'], 
                                    filter_keys=['head.'] if model.num_classes != kwargs.get('num_classes',1000) else None, 
                                    strict=False if model.num_classes != kwargs.get('num_classes',1000) else True)
    return model

@register_model
def convit_small(pretrained=False, img_size=224, **kwargs): # Added img_size
    model_kwargs = dict(
        img_size=img_size, patch_size=16, embed_dim=(48 * 9), depth=12, num_heads=9,
        local_up_to_layer=10, locality_strength=1.0, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model = VisionTransformer(**model_kwargs)
    model.default_cfg = _cfg(url="https://dl.fbaipublicfiles.com/convit/convit_small.pth")
    if pretrained: _load_pretrained(model, model.default_cfg['url'],
                                    filter_keys=['head.'] if model.num_classes != kwargs.get('num_classes',1000) else None, 
                                    strict=False if model.num_classes != kwargs.get('num_classes',1000) else True)
    return model

@register_model
def convit_base(pretrained=False, img_size=224, **kwargs): # Added img_size
    model_kwargs = dict(
        img_size=img_size, patch_size=16, embed_dim=(48 * 16), depth=12, num_heads=16,
        local_up_to_layer=10, locality_strength=1.0, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model = VisionTransformer(**model_kwargs)
    model.default_cfg = _cfg(url="https://dl.fbaipublicfiles.com/convit/convit_base.pth")
    if pretrained: _load_pretrained(model, model.default_cfg['url'],
                                    filter_keys=['head.'] if model.num_classes != kwargs.get('num_classes',1000) else None, 
                                    strict=False if model.num_classes != kwargs.get('num_classes',1000) else True)
    return model

@register_model
def convit_tiny_backbone(pretrained=False, img_size=224, in_chans=3, **kwargs):
    depth = 12
    out_indices = kwargs.pop('out_indices', [i for i in [depth-3, depth-2, depth-1] if i >=0] if depth >=3 else [depth-1])
    model_kwargs = dict(
        img_size=img_size, patch_size=16, in_chans=in_chans,
        embed_dim=(48 * 4), depth=depth, num_heads=4,
        local_up_to_layer=10, locality_strength=1.0, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0, out_indices=out_indices, **kwargs
    )
    model = VisionTransformer(**model_kwargs)
    url = "https://dl.fbaipublicfiles.com/convit/convit_tiny.pth"
    if pretrained: _load_pretrained(model, url, filter_keys=['head.'], strict=False)
    return model

@register_model
def convit_small_backbone(pretrained=False, img_size=224, in_chans=3, **kwargs):
    depth = 12
    out_indices = kwargs.pop('out_indices', [i for i in [depth-3, depth-2, depth-1] if i >=0] if depth >=3 else [depth-1])
    model_kwargs = dict(
        img_size=img_size, patch_size=16, in_chans=in_chans,
        embed_dim=(48 * 9), depth=depth, num_heads=9,
        local_up_to_layer=10, locality_strength=1.0, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0, out_indices=out_indices, **kwargs
    )
    model = VisionTransformer(**model_kwargs)
    url = "https://dl.fbaipublicfiles.com/convit/convit_small.pth"
    if pretrained: _load_pretrained(model, url, filter_keys=['head.'], strict=False)
    return model

@register_model
def convit_base_backbone(pretrained=False, img_size=224, in_chans=3, **kwargs):
    depth = 12
    out_indices = kwargs.pop('out_indices', [i for i in [depth-3, depth-2, depth-1] if i >=0] if depth >=3 else [depth-1])
    model_kwargs = dict(
        img_size=img_size, patch_size=16, in_chans=in_chans,
        embed_dim=(48 * 16), depth=depth, num_heads=16,
        local_up_to_layer=10, locality_strength=1.0, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0, out_indices=out_indices, **kwargs
    )
    model = VisionTransformer(**model_kwargs)
    url = "https://dl.fbaipublicfiles.com/convit/convit_base.pth"
    if pretrained: _load_pretrained(model, url, filter_keys=['head.'], strict=False)
    return model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test classification model
    print("\n--- Testing ConViT Tiny (Classification) ---")
    cls_model = convit_tiny(pretrained=False, num_classes=10, img_size=224).to(device) # Specify img_size
    dummy_input_cls = torch.randn(2, 3, 224, 224).to(device)
    try:
        output_cls = cls_model(dummy_input_cls)
        print("Classification output shape:", output_cls.shape) 
    except Exception as e:
        print(f"Error during classification model test: {e}")
        import traceback; traceback.print_exc()

    # Test backbone model
    print("\n--- Testing ConViT Tiny Backbone (img_size=224) ---")
    backbone_model_224 = convit_tiny_backbone(pretrained=False, img_size=224).to(device)
    print(f"Backbone model (224) feature_channels: {backbone_model_224.feature_channels}")
    dummy_input_backbone_224 = torch.randn(2, 3, 224, 224).to(device)
    try:
        features_list_224 = backbone_model_224(dummy_input_backbone_224)
        if isinstance(features_list_224, list):
            print(f"Backbone (224) output type: list (Correct)")
            for i, feat in enumerate(features_list_224):
                print(f"Feature map {i} shape: {feat.shape}, Channels: {feat.size(1)}")
                # For 224x224, patch_size=16 -> 14x14 patches
                assert feat.shape[2] == 224 // 16 
                assert feat.size(1) == backbone_model_224.feature_channels[i]
        else:
            print(f"Backbone (224) output type: {type(features_list_224)} (Incorrect, expected list)")
    except Exception as e:
        print(f"Error during backbone model (224) test: {e}")
        import traceback; traceback.print_exc()

    print("\n--- Testing ConViT Small Backbone (img_size=384, dynamic size test) ---")
    # Intentionally create with one img_size, then test with another if pos_embed interpolation works
    backbone_model_s_384 = convit_small_backbone(pretrained=False, img_size=224).to(device) # Create with 224
    # Now test with 384x384 input
    print(f"Backbone model (small, created for 224) feature_channels: {backbone_model_s_384.feature_channels}") # Will be based on 224
    
    dummy_input_backbone_s_384 = torch.randn(1, 3, 384, 384).to(device)
    try:
        features_list_s_384 = backbone_model_s_384(dummy_input_backbone_s_384)
        if isinstance(features_list_s_384, list):
            print(f"Backbone (small, 384 input) output type: list")
            for i, feat in enumerate(features_list_s_384):
                print(f"Feature map {i} shape: {feat.shape}")
                # img_size=384, patch_size=16 -> 384/16 = 24 patches per dim
                assert feat.shape[2] == 384 // 16 
                # Channel dim should be consistent with model's embed_dim
                assert feat.size(1) == backbone_model_s_384.embed_dim
        else:
            print(f"Backbone (small, 384 input) output type: {type(features_list_s_384)}")
    except Exception as e:
        print(f"Error during small backbone model (384 input) test: {e}")
        import traceback; traceback.print_exc()

    print("\n--- Testing ConViT Tiny Backbone (img_size=256, different out_indices) ---")
    custom_out_indices = [0, 5, 11] # Example custom indices
    backbone_model_custom = convit_tiny_backbone(pretrained=False, img_size=256, out_indices=custom_out_indices).to(device)
    print(f"Backbone model (custom idx) feature_channels: {backbone_model_custom.feature_channels}")
    print(f"Backbone model (custom idx) out_indices: {backbone_model_custom.out_indices}")

    assert len(backbone_model_custom.feature_channels) == len(custom_out_indices)

    dummy_input_backbone_custom = torch.randn(1, 3, 256, 256).to(device)
    try:
        features_list_custom = backbone_model_custom(dummy_input_backbone_custom)
        if isinstance(features_list_custom, list):
            print(f"Backbone (custom idx) output type: list")
            assert len(features_list_custom) == len(custom_out_indices)
            for i, feat in enumerate(features_list_custom):
                print(f"Feature map {i} (from block {backbone_model_custom.out_indices[i]}) shape: {feat.shape}")
                assert feat.shape[2] == 256 // 16 
                assert feat.size(1) == backbone_model_custom.embed_dim
        else:
            print(f"Backbone (custom idx) output type: {type(features_list_custom)}")
    except Exception as e:
        print(f"Error during custom backbone model test: {e}")
        import traceback; traceback.print_exc()