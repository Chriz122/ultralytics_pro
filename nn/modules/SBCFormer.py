import torch
import torch.nn as nn
import math
import itertools

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

# --- Helper Classes from Code 1 (SBCFormer) ---

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class Conv2d_BN(nn.Module):
    def __init__(self, in_features, out_features=None, kernel_size=3, stride=1, padding=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        # Initialization of BN is handled by the main _init_weights or here specifically.
        # Let's keep specific init here if desired, but general _init_weights will also cover nn.BatchNorm2d.
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

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

class InvertResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=3, act_layer=nn.GELU, drop_path=0.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.pwconv1_bn = Conv2d_BN(self.in_features, self.hidden_features, kernel_size=1,  stride=1, padding=0)
        self.dwconv_bn = Conv2d_BN(self.hidden_features, self.hidden_features, kernel_size=3,  stride=1, padding=1, groups= self.hidden_features)
        self.pwconv2_bn = Conv2d_BN(self.hidden_features, self.out_features, kernel_size=1,  stride=1, padding=0) # Use self.out_features

        self.act = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Residual connection if in_features == out_features
        self.use_res_connect = self.in_features == self.out_features


    def forward(self, x):
        identity = x
        x1 = self.pwconv1_bn(x)
        x1 = self.act(x1)
        x1 = self.dwconv_bn(x1)
        x1 = self.act(x1)
        x1 = self.pwconv2_bn(x1)

        if self.use_res_connect:
            return identity + self.drop_path(x1)
        else:
            return self.drop_path(x1)


class Attention(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8, attn_ratio=2, resolution=7):
        super().__init__()
        self.resolution = resolution # H and W of the input feature map *before* flatten
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.attn_ratio = attn_ratio
        self.scale = key_dim ** -0.5
        
        self.nh_kd = key_dim * num_heads
        self.qk_dim = 2 * self.nh_kd
        self.v_dim = int(attn_ratio * key_dim) * num_heads # This is the output dim of attention block before projection
        dim_h = self.v_dim + self.qk_dim

        # resolution here is the spatial dimension (H or W) of the feature map *token sequence*
        # e.g., if input to SBCFormerBlock is (B,C,14,14) and pool_ratio=1, then mixer output is (B,C,14,14)
        # then it's flattened to (B, 196, C). Here, self.resolution should be 14.
        # N is the number of tokens
        self.N_tokens = self.resolution ** 2 # This is num_tokens after flatten

        self.pwconv = nn.Conv2d(dim, dim_h, kernel_size=1,  stride=1, padding=0)
        self.dwconv = Conv2d_BN(self.v_dim, self.v_dim, kernel_size=3,  stride=1, padding=1, groups=self.v_dim)
        self.proj_out = nn.Linear(self.v_dim, dim) # Projects back to original embedding dim
        self.act = nn.GELU()

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        N_points = len(points) # Should be self.resolution**2
        attention_offsets = {}
        idxs = []
        for p1_idx, p1 in enumerate(points):
            for p2_idx, p2 in enumerate(points):
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_points, N_points)) # N_points x N_points

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            # Ensure attention_bias_idxs matches the expected N_tokens x N_tokens
            if self.attention_bias_idxs.shape[0] != self.N_tokens or self.attention_bias_idxs.shape[1] != self.N_tokens:
                 # This can happen if resolution passed to __init__ was for a different stage
                 # Recompute if necessary, though ideally resolution is fixed per instance
                 print(f"Warning: Mismatch in Attention bias indices. Expected {self.N_tokens}x{self.N_tokens}, got {self.attention_bias_idxs.shape}. This might indicate an issue with dynamic resolution handling.")
            self.ab = self.attention_biases[:, self.attention_bias_idxs]


    def forward(self, x_tokenized): # Expects x in (B, N_tokens, C)
        B, N, C = x_tokenized.shape # N should be self.N_tokens (resolution**2)
        
        # Reshape tokenized input to 2D feature map for pwconv
        # H, W for the feature map before pwconv should match self.resolution
        # N = H * W, so H = W = sqrt(N)
        # This assumes N_tokens is a perfect square and corresponds to self.resolution * self.resolution
        # This H, W is the spatial dimension of the feature map *IF* it were 2D before tokenization
        # For pwconv, we need to reshape it to (B, C, H, W)
        H_feat = W_feat = self.resolution
        x_spatial = x_tokenized.transpose(1, 2).reshape(B, C, H_feat, W_feat)

        x_processed = self.pwconv(x_spatial) # Output has dim_h channels
        qk, v1 = x_processed.split([self.qk_dim, self.v_dim], dim=1)
        
        # Reshape qk for attention
        # qk shape: (B, qk_dim, H_feat, W_feat)
        # N_tokens = H_feat * W_feat
        qk = qk.reshape(B, 2, self.num_heads, self.key_dim, N).permute(1, 0, 2, 4, 3) # (2, B, nH, N, key_dim)
        q, k = qk[0], qk[1] # (B, nH, N, key_dim)

        # Process v1
        v1 = v1 + self.act(self.dwconv(v1)) # v1 shape (B, v_dim, H_feat, W_feat)
        # Reshape v for attention
        # v1 shape: (B, v_dim, H_feat, W_feat) --> (B, num_heads, v_dim/num_heads, N_tokens)
        v_head_dim = self.v_dim // self.num_heads
        v = v1.reshape(B, self.num_heads, v_head_dim, N).permute(0, 1, 3, 2) # (B, nH, N, v_head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add biases
        bias_to_add = self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        # Bias shape: (num_heads, N, N)
        # Attn shape: (B, num_heads, N, N)
        # Need to unsqueeze bias: bias_to_add.unsqueeze(0)
        attn = attn + bias_to_add.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        
        # (B, nH, N, N) @ (B, nH, N, v_head_dim) -> (B, nH, N, v_head_dim)
        x_out_tokenized = (attn @ v).transpose(1, 2).reshape(B, N, self.v_dim) # (B, N, v_dim total)
        x_out_tokenized = self.proj_out(x_out_tokenized) # (B, N, C_original)
        return x_out_tokenized


class ModifiedTransformer(nn.Module):  
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio= 2, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, resolution=7):
        super().__init__()
        self.resolution = resolution # Resolution for the Attention module
        self.dim = dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn = Attention(dim=self.dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, resolution=resolution)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.mlp = Mlp(in_features=self.dim, hidden_features=int(self.dim*mlp_ratio), out_features=self.dim, act_layer=act_layer, drop=drop) # Ensure hidden_features is int
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x): # Expects x in (B, N_tokens, C)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SBCFormerBlock(nn.Module):
    def __init__(self, depth_invres, depth_mattn, depth_mixer, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2, drop=0., attn_drop=0.,
                 drop_paths=[], act_layer=nn.GELU, pool_ratio=1, invres_ratio=1, resolution=7): # resolution of input x (H or W)
        super().__init__()
        self.input_resolution = resolution # Spatial resolution of input feature map x (H or W)
        self.dim = dim
        self.depth_invres = depth_invres
        self.depth_mattn = depth_mattn
        self.depth_mixer = depth_mixer
        self.act = h_sigmoid()

        self.invres_blocks = nn.Sequential()
        for k in range(self.depth_invres):
            self.invres_blocks.add_module(f"InvRes_{k}", InvertResidualBlock(in_features=dim, hidden_features=int(dim*invres_ratio), out_features=dim, kernel_size=3, drop_path=0.)) # drop_path was 0.

        self.pool_ratio = pool_ratio
        self.pooled_resolution = self.input_resolution // self.pool_ratio # Resolution after pooling, for Attention

        if self.pool_ratio > 1:
            self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
            # ConvTranspose2d output size: (Hin-1)*S - 2P + K + OP = (Hin-1)*PR - 0 + PR + 0 = Hin*PR - PR + PR = Hin*PR
            self.convTrans = nn.ConvTranspose2d(dim, dim, kernel_size=pool_ratio, stride=pool_ratio, groups=dim)
            self.norm_after_convtrans = nn.BatchNorm2d(dim) # Renamed self.norm to avoid clash
        else:
            self.pool = nn.Identity()
            self.convTrans = nn.Identity()
            self.norm_after_convtrans = nn.Identity()
        
        self.mixer = nn.Sequential()
        for k in range(self.depth_mixer): # Mixer operates on pooled resolution
            self.mixer.add_module(f"Mixer_{k}", InvertResidualBlock(in_features=dim, hidden_features=int(dim*invres_ratio), out_features=dim, kernel_size=3, drop_path=0.))
        
        self.trans_blocks = nn.Sequential()
        for k in range(self.depth_mattn): # Transformer operates on pooled_resolution
            self.trans_blocks.add_module(f"MAttn_{k}", ModifiedTransformer(dim=dim, key_dim=key_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
             drop=drop, attn_drop=attn_drop, drop_path=drop_paths[k], resolution=self.pooled_resolution))
        
        self.proj = Conv2d_BN(self.dim, self.dim, kernel_size=1,  stride=1, padding=0)
        self.proj_fuse = Conv2d_BN(self.dim*2, self.dim, kernel_size=1,  stride=1, padding=0)
        
    def forward(self, x): # x is (B, C, H, W) where H,W = self.input_resolution
        B, C, H_in, W_in = x.shape
        
        x_invres = self.invres_blocks(x)
        local_fea = x_invres # (B, C, H_in, W_in)

        # Global path
        x_global = x_invres
        if self.pool_ratio > 1:
            x_global = self.pool(x_global) # (B, C, H_in/pool_ratio, W_in/pool_ratio)
        
        # H_pooled, W_pooled = H_in // self.pool_ratio, W_in // self.pool_ratio
        H_pooled, W_pooled = self.pooled_resolution, self.pooled_resolution # Use precalculated

        x_global = self.mixer(x_global) # (B, C, H_pooled, W_pooled)
        
        # Reshape for Transformer: (B, C, H_pooled, W_pooled) -> (B, N_tokens, C)
        N_tokens = H_pooled * W_pooled
        x_global_tokenized = x_global.flatten(2).transpose(1, 2) # (B, N_tokens, C)
        x_global_tokenized = self.trans_blocks(x_global_tokenized) # (B, N_tokens, C)
        
        # Reshape back to spatial: (B, N_tokens, C) -> (B, C, H_pooled, W_pooled)
        x_global = x_global_tokenized.transpose(1, 2).reshape(B, C, H_pooled, W_pooled)

        if self.pool_ratio > 1:
            x_global = self.convTrans(x_global) # Upsamples to H_in, W_in
            x_global = self.norm_after_convtrans(x_global)
        
        global_act = self.act(self.proj(x_global)) # (B, C, H_in, W_in)
        
        x_fused = local_fea * global_act # Element-wise product
        x_cat = torch.cat((x_global, x_fused), dim=1) # Concatenate original global and fused local
        out = self.proj_fuse(x_cat)

        return out


class PatchMerging(nn.Module):
    def __init__(self,  in_features, out_features):
        super().__init__()
        # Norm applied after conv in timm's PatchMerging, but here it seems BN.
        # For consistency with Conv2d_BN, let's use BN. If LayerNorm is needed, input needs to be (B, N, C)
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_features)


    def forward(self, x): # x is (B, C, H, W)
        x = self.conv(x)
        x = self.norm(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dim=64): # embed_dim is the output channel dim
        super().__init__()
        self.embed_dim = embed_dim        
        # This sequence of 3 convs with stride 2 reduces H,W by 2*2*2 = 8
        self.stem = nn.Sequential(
                            Conv2d_BN(in_features=in_chans, out_features=self.embed_dim//4, kernel_size=3, stride=2, padding=1),
                            nn.ReLU(inplace=True),
                            Conv2d_BN(in_features=self.embed_dim//4, out_features=self.embed_dim//2, kernel_size=3, stride=2, padding=1),
                            nn.ReLU(inplace=True),
                            Conv2d_BN(in_features=self.embed_dim//2, out_features=self.embed_dim, kernel_size=3, stride=2, padding=1),
                            nn.ReLU(inplace=True))

    def forward(self, x): # x is (B, C, H, W)
        x = self.stem(x) # Output is (B, embed_dim, H/8, W/8)
        return x

class SBCFormer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, 
                 embed_dims=[128,320,512], key_dim=32, num_heads=[2,4,8], 
                 attn_ratio=2, mlp_ratio=4, invres_ratio=2,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 depths_invres=[2,2,1], depths_mattn=[1,4,3], depths_mixer=[2,2,2], 
                 pool_ratios=[4,2,1]):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.depths_invres = depths_invres
        self.depths_mattn = depths_mattn
        self.depths_mixer = depths_mixer
        self.num_stages = len(self.embed_dims)

        # Store for width_list calculation
        self.img_size = img_size
        self.in_chans = in_chans

        self.merging_blocks = nn.ModuleList()
        self.sbcformer_blocks = nn.ModuleList()

        # Initial patch embedding
        # Output H, W of patch_embed is img_size / 8
        current_resolution = img_size // 8
        self.patch_embed = PatchEmbed(img_size=img_size, in_chans=in_chans, embed_dim=self.embed_dims[0])
        # self.merging_blocks.append(self.patch_embed) # This will be handled in forward_features loop

        # Stochastic depth decay rule
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths_mattn))]  
        
        cur_dpr_idx = 0
        for i in range(self.num_stages):
            # Patch merging layer for all stages except the first (handled by patch_embed)
            if i == 0:
                # For the first stage, the "merging" is the patch_embed itself.
                # The input to sbcformer_blocks[0] comes from patch_embed.
                # Channel dimension for the first stage is embed_dims[0]
                merger = self.patch_embed # Placeholder, actual merging handled in loop
            else:
                # For subsequent stages, add a PatchMerging layer
                merger = PatchMerging(in_features=self.embed_dims[i-1], out_features=self.embed_dims[i])
                current_resolution //= 2 # PatchMerging halves resolution
            self.merging_blocks.append(merger)

            sbc_block = SBCFormerBlock(
                depth_invres=self.depths_invres[i], 
                depth_mattn=self.depths_mattn[i], 
                depth_mixer=self.depths_mixer[i], 
                dim=self.embed_dims[i], 
                key_dim=key_dim, # key_dim seems global in original, or needs to be a list
                num_heads=self.num_heads[i],
                mlp_ratio=mlp_ratio, # mlp_ratio seems global
                attn_ratio=attn_ratio, # attn_ratio seems global
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_paths=self.dpr[cur_dpr_idx : cur_dpr_idx + self.depths_mattn[i]], 
                pool_ratio=pool_ratios[i], 
                invres_ratio=invres_ratio, # invres_ratio seems global
                resolution=current_resolution # Pass calculated H, W for this stage
            )
            self.sbcformer_blocks.append(sbc_block)
            cur_dpr_idx += self.depths_mattn[i]
        
        # Classification head (kept for potential standalone use)
        self.norm_head = nn.LayerNorm(embed_dims[-1], eps=1e-6)
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self.apply(self._init_weights)

        # --- Add width_list calculation ---
        self.width_list = []
        try:
            self.eval() # Set to eval mode
            # Create a dummy input tensor
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size)
            # Pass dummy input through the forward method using torch.no_grad()
            with torch.no_grad():
                 features = self.forward(dummy_input) # This will call forward_features

            # Extract channel dimension (dim=1) from each feature map in the list
            self.width_list = [f.size(1) for f in features]
            self.train() # Set back to train mode
        except Exception as e:
            print(f"Error during dummy forward pass for SBCFormer width_list calculation: {e}")
            print("Setting width_list to embed_dims as fallback.")
            self.width_list = self.embed_dims # Fallback
            self.train() # Ensure model is back in train mode

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            # Check if it's from Conv2d_BN with specific init
            # Conv2d_BN already initializes its BN. If we re-init here, it might override.
            # The current Conv2d_BN init (weight=1, bias=0) is standard.
            # This general BatchNorm2d init will apply to BNs not in Conv2d_BN, or re-init them if not careful.
            # For simplicity, let's assume the specific init in Conv2d_BN is desired and this one is for other BNs.
            # However, SBCFormerBlock.norm_after_convtrans is a direct nn.BatchNorm2d.
            if not hasattr(m, '_already_initialized_by_conv2d_bn'): # Hypothetical flag
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()


    def forward_features(self, x):
        feature_outputs = []
        
        # Stage 0: Patch Embedding
        x = self.patch_embed(x) # self.merging_blocks[0] is self.patch_embed
        x = self.sbcformer_blocks[0](x)
        feature_outputs.append(x)

        # Stages 1 to N-1
        for i in range(1, self.num_stages):
            x = self.merging_blocks[i](x) # PatchMerging layer
            x = self.sbcformer_blocks[i](x)
            feature_outputs.append(x)
        
        return feature_outputs # List of (B, C_stage, H_stage, W_stage)

    def forward(self, x):
        # This forward method now returns a list of feature maps,
        # which is often expected by detection/segmentation heads (e.g., in YOLO).
        features = self.forward_features(x)
        return features

    # Optional: if you need to run classification using this model standalone
    def forward_classification_head(self, x_list):
        # Typically uses the last feature map from the list
        x = x_list[-1] 
        # Global average pooling: (N, C, H, W) -> (N, C)
        x = x.mean(dim=[-2, -1]) 
        x = self.norm_head(x)
        x = self.head(x)
        return x

# --- Factory Functions (similar to SMT style) ---

@register_model
def sbcformer_xs(img_size=224, in_chans=3, num_classes=1000, pretrained=False, **kwargs):
    # Kwargs can override defaults passed to SBCFormer
    # For key_dim, num_heads, attn_ratio, mlp_ratio, invres_ratio, pool_ratios,
    # if they are not lists, they are global. If they are lists, they should match num_stages.
    # Original code suggests key_dim, attn_ratio, mlp_ratio, invres_ratio are global.
    # num_heads, pool_ratios are lists.
    model = SBCFormer(
        img_size=img_size, in_chans=in_chans, num_classes=num_classes,
        embed_dims=[96, 160, 288], 
        key_dim=16, # Global
        num_heads=[3, 5, 6], # Per stage
        invres_ratio=2, # Global
        attn_ratio=2, # Global
        mlp_ratio=4, # Global, ModifiedTransformer uses it. InvertResidualBlock uses invres_ratio
        pool_ratios = [4, 2, 1], # Per stage
        depths_invres=[2, 2, 1], 
        depths_mattn=[2, 3, 2], 
        depths_mixer=[2, 2, 2], 
        **kwargs)
    model.default_cfg = _cfg()
    # Add pretrained loading logic here if pretrained=True
    return model
 
@register_model
def sbcformer_s(img_size=224, in_chans=3, num_classes=1000, pretrained=False, **kwargs): 
    model = SBCFormer(
        img_size=img_size, in_chans=in_chans, num_classes=num_classes,
        embed_dims=[96, 192, 320], 
        key_dim=16, 
        num_heads=[3, 5, 7], # Adjusted num_heads for stage 2 (was 5 in paper, but 320/16=20 not div by 5, maybe head_dim not key_dim?) Assuming key_dim is head_dim for qk.
        invres_ratio=2, 
        attn_ratio=2, 
        mlp_ratio=4,
        pool_ratios = [4, 2, 1],
        depths_invres=[2, 2, 1], 
        depths_mattn=[2, 4, 3], 
        depths_mixer=[2, 2, 2], 
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def sbcformer_b(img_size=224, in_chans=3, num_classes=1000, pretrained=False, **kwargs):
    model = SBCFormer(
        img_size=img_size, in_chans=in_chans, num_classes=num_classes,
        embed_dims=[128, 256, 384], 
        key_dim=24, # key_dim=24 means qk head_dim is 24. num_heads must make embed_dim divisible.
        num_heads=[4, 6, 8], # Adjusted: 128/4=32 (ok); 256/8=32 (ok); 384/8=48 (ok) assuming key_dim is qk_head_dim, and num_heads*key_dim can be != embed_dim
                               # In ViT, embed_dim = num_heads * head_dim. Here Attention splits embed_dim into num_heads * key_dim for qk.
                               # The original Attention class has `self.nh_kd = key_dim * num_heads` and then `self.qk_dim = 2 * self.nh_kd`.
                               # `pwconv` projects `dim` to `v_dim + qk_dim`. This implies `nh_kd` does not need to be `dim`.
                               # So original num_heads=[4,6,8] for B should be fine.
        invres_ratio=2, 
        attn_ratio=2, 
        mlp_ratio=4,
        pool_ratios = [4, 2, 1],
        depths_invres=[2, 2, 1], 
        depths_mattn=[2, 4, 3], 
        depths_mixer=[2, 2, 2], 
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def sbcformer_l(img_size=224, in_chans=3, num_classes=1000, pretrained=False, **kwargs):
    model = SBCFormer(
        img_size=img_size, in_chans=in_chans, num_classes=num_classes,
        embed_dims=[192, 288, 384], 
        key_dim=32, 
        num_heads=[4, 6, 8], # embed_dims must be divisible by num_heads if qk_head_dim is also embed_dim/num_heads
                               # Using original logic for Attention, num_heads doesn't strictly need to divide embed_dim for qk.
                               # Original: [4,6,8]. Let's stick to it and verify Attention logic.
                               # Attention.py: pwconv(dim, dim_h) where dim_h = v_dim + qk_dim. qk_dim = 2*num_heads*key_dim.
                               # So, this is fine.
        invres_ratio=4, 
        attn_ratio=2, 
        mlp_ratio=4,
        pool_ratios = [4, 2, 1],
        depths_invres=[2, 2, 1], 
        depths_mattn=[2, 4, 3], 
        depths_mixer=[2, 2, 2], 
        **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    img_h, img_w = 224, 224
    # Test with SBCFormer XS
    print("--- Creating SBCFormer XS model ---")
    model_xs = sbcformer_xs(img_size=img_h, num_classes=100) # Example num_classes
    print("Model created successfully.")
    print("Calculated width_list for XS:", model_xs.width_list)
    assert model_xs.width_list == model_xs.embed_dims, \
        f"Width list {model_xs.width_list} does not match embed_dims {model_xs.embed_dims}"


    # Test forward pass
    input_tensor = torch.rand(2, 3, img_h, img_w) # Batch size 2, 3 channels
    print(f"\n--- Testing SBCFormer XS forward pass (Input: {input_tensor.shape}) ---")

    model_xs.eval() # Set to evaluation mode
    try:
        with torch.no_grad(): # Disable gradient calculation for inference
            output_features_list = model_xs(input_tensor)
        
        print("Forward pass successful.")
        print("Output is a list of feature maps from each stage:")
        for i, features in enumerate(output_features_list):
            print(f"Stage {i+1} feature map shape: {features.shape}") # Should be [B, C_stage, H_stage, W_stage]
            assert features.size(0) == input_tensor.size(0)
            assert features.size(1) == model_xs.embed_dims[i]
            # You can add more specific checks for H_stage, W_stage if needed

        # Verify width_list matches runtime output channels
        runtime_widths = [f.size(1) for f in output_features_list]
        print("\nRuntime output feature channels:", runtime_widths)
        assert model_xs.width_list == runtime_widths, "Calculated width_list mismatch with runtime!"
        print("Width list verified successfully against runtime output.")

        # Test classification head if needed (using the new method)
        # class_output = model_xs.forward_classification_head(output_features_list)
        # print(f"\nClassification head output shape: {class_output.shape}")
        # assert class_output.shape == (input_tensor.size(0), model_xs.num_classes)


        # --- Test deepcopy ---
        print("\n--- Testing deepcopy of SBCFormer XS ---")
        import copy
        copied_model_xs = copy.deepcopy(model_xs)
        print("Deepcopy successful.")

        # Optional: Test copied model forward pass
        copied_model_xs.eval()
        with torch.no_grad():
             output_copied = copied_model_xs(input_tensor)
        print("Copied model forward pass successful.")
        assert len(output_copied) == len(output_features_list)
        for i in range(len(output_features_list)):
             assert output_copied[i].shape == output_features_list[i].shape
        print("Copied model output shapes verified.")


    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

    # You can uncomment to test other sizes
    # print("\n--- Creating SBCFormer S model ---")
    # model_s = sbcformer_s(img_size=img_h)
    # print("Calculated width_list for S:", model_s.width_list)
    # model_s.eval()
    # with torch.no_grad():
    #     output_s = model_s(input_tensor)
    # for i, features in enumerate(output_s):
    #     print(f"Stage {i+1} (S) feature map shape: {features.shape}")