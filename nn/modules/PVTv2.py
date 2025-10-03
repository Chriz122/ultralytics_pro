import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import copy # Added for deepcopy in factory functions if needed, though not strictly used here yet

# --- Imports from timm ---
# Keep necessary imports from timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# Removed timm registry, MMLab dependencies

# --- Helper Functions/Classes (from Code 1, kept as is) ---

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        # Avoid creating Conv2d with 0 dim
        if dim == 0:
            self.dwconv = nn.Identity()
        else:
            self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        if isinstance(self.dwconv, nn.Identity):
            return x
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear = linear # Store linear flag

        # Avoid creating layers with 0 features
        if in_features == 0 or hidden_features == 0 or out_features == 0:
            self.fc1 = nn.Identity()
            self.dwconv = nn.Identity()
            self.act = nn.Identity()
            self.fc2 = nn.Identity()
            self.drop = nn.Identity()
            if self.linear:
                 self.relu = nn.Identity()
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.dwconv = DWConv(hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)
            if self.linear:
                self.relu = nn.ReLU(inplace=True)

        self.apply(self._init_weights) # Apply weights after defining all layers

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if m.groups > 0: # Check if groups attribute exists and is > 0
                fan_out //= m.groups
            # Ensure fan_out is not zero before calculating math.sqrt
            if fan_out > 0:
                 m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            else:
                 m.weight.data.normal_(0, 0.02) # Fallback initialization
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        if isinstance(self.fc1, nn.Identity):
             return x
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        # Pass H, W to dwconv only if it's not Identity
        if isinstance(self.dwconv, DWConv):
            x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        self.linear = linear # Store linear flag
        self.sr_ratio = sr_ratio

        # Handle dim=0 case
        if dim == 0 or num_heads == 0:
             self.dim = 0
             self.num_heads = 0
             self.head_dim = 0
             self.scale = 1.0
             self.q = nn.Identity()
             self.kv = nn.Identity()
             self.attn_drop = nn.Identity()
             self.proj = nn.Identity()
             self.proj_drop = nn.Identity()
             if not linear:
                 if sr_ratio > 1:
                     self.sr = nn.Identity()
                     self.norm = nn.Identity()
             else:
                 self.pool = nn.Identity()
                 self.sr = nn.Identity()
                 self.norm = nn.Identity()
                 self.act = nn.Identity()
             return # Exit early if dim is 0

        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
            else: # Avoid creating sr/norm if sr_ratio is 1
                 self.sr = nn.Identity()
                 self.norm = nn.Identity()
        else:
            # Check if AdaptiveAvgPool2d supports 0 output size, it doesn't. Use 1.
            pool_size = 7 if dim >= 7 else 1 # Ensure pool size is valid
            self.pool = nn.AdaptiveAvgPool2d(pool_size)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
             if m.bias is not None: nn.init.constant_(m.bias, 0)
             if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if m.groups > 0: # Check if groups attribute exists and is > 0
                fan_out //= m.groups
            # Ensure fan_out is not zero before calculating math.sqrt
            if fan_out > 0:
                 m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            else:
                 m.weight.data.normal_(0, 0.02) # Fallback initialization
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        if self.dim == 0: # Handle dim=0 case
            return x

        B, N, C = x.shape
        # Ensure C matches self.dim
        if C != self.dim:
            raise ValueError(f"Input feature dim {C} does not match Attention dim {self.dim}")

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1 and isinstance(self.sr, nn.Conv2d):
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else: # sr_ratio is 1 or sr is Identity
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else: # linear attention
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # Pool only if pool is not Identity (i.e., dim > 0)
            if isinstance(self.pool, nn.AdaptiveAvgPool2d):
                 x_ = self.pool(x_)
            # Apply sr, norm, act only if they are not Identity
            if isinstance(self.sr, nn.Conv2d):
                 x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            if isinstance(self.norm, nn.LayerNorm):
                 x_ = self.norm(x_)
            if isinstance(self.act, nn.GELU):
                 x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        # Handle dim=0 case
        if dim == 0:
            self.norm1 = nn.Identity()
            self.attn = nn.Identity()
            self.drop_path = nn.Identity()
            self.norm2 = nn.Identity()
            self.mlp = nn.Identity()
        else:
            self.norm1 = norm_layer(dim)
            self.attn = Attention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            # Ensure mlp_hidden_dim is not zero if dim is not zero
            if dim > 0 and mlp_hidden_dim == 0: mlp_hidden_dim = dim # Avoid zero hidden dim
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights) # Apply after defining layers

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
             if m.bias is not None: nn.init.constant_(m.bias, 0)
             if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if m.groups > 0: # Check if groups attribute exists and is > 0
                 fan_out //= m.groups
             # Ensure fan_out is not zero before calculating math.sqrt
            if fan_out > 0:
                  m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            else:
                  m.weight.data.normal_(0, 0.02) # Fallback initialization
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Skip block logic if it's just Identity layers
        if isinstance(self.norm1, nn.Identity):
            return x

        # Pass H, W to attention and mlp only if they are not Identity
        attn_out = self.attn(self.norm1(x), H, W) if isinstance(self.attn, Attention) else self.norm1(x)
        x = x + self.drop_path(attn_out)

        mlp_out = self.mlp(self.norm2(x), H, W) if isinstance(self.mlp, Mlp) else self.norm2(x)
        x = x + self.drop_path(mlp_out)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding with Overlapping Patches """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        # Handle embed_dim=0 case
        if embed_dim == 0 or in_chans == 0:
            self.img_size = img_size
            self.patch_size = patch_size
            self.H, self.W = 0, 0 # Or calculate based on stride? Let's calculate.
            self.H = img_size[0] // stride if stride > 0 else 0
            self.W = img_size[1] // stride if stride > 0 else 0
            self.num_patches = self.H * self.W
            self.proj = nn.Identity()
            self.norm = nn.Identity()
            return

        # Original logic
        assert max(patch_size) > stride, "Set larger patch_size than stride for overlapping"
        self.img_size = img_size
        self.patch_size = patch_size
        self.H = img_size[0] // stride
        self.W = img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights) # Apply after defining layers

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
             if m.bias is not None: nn.init.constant_(m.bias, 0)
             if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if m.groups > 0: # Check if groups attribute exists and is > 0
                 fan_out //= m.groups
             # Ensure fan_out is not zero before calculating math.sqrt
            if fan_out > 0:
                  m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            else:
                  m.weight.data.normal_(0, 0.02) # Fallback initialization
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # Handle Identity case
        if isinstance(self.proj, nn.Identity):
             # Need to return dummy H, W consistent with calculation if stride > 0
             B, C, H_in, W_in = x.shape
             H_out = H_in // getattr(self, 'stride', 1) if hasattr(self, 'stride') and self.stride > 0 else 0
             W_out = W_in // getattr(self, 'stride', 1) if hasattr(self, 'stride') and self.stride > 0 else 0
             # Return x reshaped? Or just x and calculated H, W? Let's return x and H, W.
             # Reshape to match expected output format (B, N, C) if needed downstream
             # For Identity, maybe return (B, H*W, 0) ? Let's return x and H, W.
             return x, H_out, W_out # Downstream needs to handle C=0

        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerV2(nn.Module):
    """ Pyramid Vision Transformer V2 (PVTv2)

        Refactored to remove MMLab dependencies and follow MogaNet structure.
        Includes `width_list` for compatibility with frameworks like Ultralytics.
    """
    arch_zoo = {
        'b0': {'embed_dims': [32, 64, 160, 256], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [8, 8, 4, 4],
               'depths': [2, 2, 2, 2], 'sr_ratios': [8, 4, 2, 1], 'linear': False},
        'b1': {'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [8, 8, 4, 4],
               'depths': [2, 2, 2, 2], 'sr_ratios': [8, 4, 2, 1], 'linear': False},
        'b2': {'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [8, 8, 4, 4],
               'depths': [3, 4, 6, 3], 'sr_ratios': [8, 4, 2, 1], 'linear': False},
        'b2_li': {'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [8, 8, 4, 4],
                  'depths': [3, 4, 6, 3], 'sr_ratios': [8, 4, 2, 1], 'linear': True}, # linear=True
        'b3': {'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [8, 8, 4, 4],
               'depths': [3, 4, 18, 3], 'sr_ratios': [8, 4, 2, 1], 'linear': False},
        'b4': {'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [8, 8, 4, 4],
               'depths': [3, 8, 27, 3], 'sr_ratios': [8, 4, 2, 1], 'linear': False},
        'b5': {'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [4, 4, 4, 4],
               'depths': [3, 6, 40, 3], 'sr_ratios': [8, 4, 2, 1], 'linear': False},
    }

    def __init__(self,
                 c1=3,                # Input channels (like MogaNet)
                 arch='b2',           # Architecture variant string or dict
                 img_size=224,        # Base image size (can be dynamic in forward)
                 patch_size=4,        # Initial patch size (not used directly, OverlapPatchEmbed handles)
                 num_classes=0,       # Number of classes (set to 0 for backbone)
                 qkv_bias=True,       # Use bias in QKV linear layers
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 num_stages=4,        # Number of stages (usually 4)
                 fork_feat=True,      # Return features from each stage (for backbone)
                 **kwargs):           # Allow downstream args
        super().__init__()

        self.num_classes = num_classes # Even if 0
        self.fork_feat = fork_feat

        # --- Determine architecture settings ---
        if isinstance(arch, str):
            arch_key = arch.lower()
            if arch_key not in self.arch_zoo:
                raise KeyError(f"Arch '{arch}' is not in default PVTv2 archs {set(self.arch_zoo.keys())}")
            self.arch_settings = self.arch_zoo[arch_key]
        elif isinstance(arch, dict):
            essential_keys = {'embed_dims', 'num_heads', 'mlp_ratios', 'depths', 'sr_ratios', 'linear'}
            # Check if all essential keys are present
            missing_keys = essential_keys - set(arch.keys())
            if missing_keys:
                 raise ValueError(f"Custom arch dict is missing keys: {missing_keys}")
            # Check if provided keys are valid (optional, helps catch typos)
            unknown_keys = set(arch.keys()) - essential_keys
            if unknown_keys:
                 print(f"Warning: Custom arch dict has unknown keys: {unknown_keys}")
            self.arch_settings = arch
        else:
            raise TypeError("arch must be a string or a dict")

        # --- Extract arch parameters ---
        self.embed_dims = self.arch_settings['embed_dims']
        self.num_heads = self.arch_settings['num_heads']
        self.mlp_ratios = self.arch_settings['mlp_ratios']
        self.depths = self.arch_settings['depths']
        self.sr_ratios = self.arch_settings['sr_ratios']
        self.linear = self.arch_settings.get('linear', False) # Default linear to False if missing
        self.num_stages = len(self.depths) # Ensure num_stages matches arch spec

        # --- Set width_list (Crucial for Ultralytics/YOLO) ---
        if self.fork_feat:
            # width_list should contain the output channels of the stages intended for feature extraction
            # For PVTv2, this is typically the embed_dims of each stage
            self.width_list = list(self.embed_dims)
        else:
            # If not forking features, width_list might contain only the last dim or be None
            self.width_list = [self.embed_dims[-1]] if self.embed_dims else []
            # Or set to None if your framework handles it:
            # self.width_list = None

        # --- Stochastic depth ---
        total_depth = sum(self.depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)] if total_depth > 0 else []
        cur = 0

        # --- Build Stages ---
        current_img_size = to_2tuple(img_size) # Keep track of spatial size if needed
        current_in_chans = c1 # Use c1 from args

        for i in range(self.num_stages):
            # Calculate expected input size for patch embed (PVTv2 reduces H/W by 4 then 2, 2, 2)
            # Note: This assumes img_size input. Real input size is handled in OverlapPatchEmbed forward
            stage_img_size = (current_img_size[0] // (4 if i == 0 else 2),
                              current_img_size[1] // (4 if i == 0 else 2))

            patch_embed = OverlapPatchEmbed(img_size=stage_img_size[0], # Pass expected H for this stage
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=current_in_chans,
                                            embed_dim=self.embed_dims[i])

            stage_blocks = nn.ModuleList([Block(
                dim=self.embed_dims[i],
                num_heads=self.num_heads[i],
                mlp_ratio=self.mlp_ratios[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + j] if dpr else 0.0,
                norm_layer=norm_layer,
                sr_ratio=self.sr_ratios[i],
                linear=self.linear) # Pass linear flag here
                for j in range(self.depths[i])])

            # Layer norm after each stage
            # Avoid norm if embed_dim is 0
            stage_norm = norm_layer(self.embed_dims[i]) if self.embed_dims[i] > 0 else nn.Identity()

            # --- Store modules ---
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", stage_blocks)
            setattr(self, f"norm{i + 1}", stage_norm)

            # Update for next stage
            current_in_chans = self.embed_dims[i]
            cur += self.depths[i]
            current_img_size = stage_img_size # Update expected size (less important now)

        # --- Classification head (Optional, not used if fork_feat=True) ---
        self.head = nn.Identity() # Default to Identity for backbone
        if self.num_classes > 0 and not self.fork_feat:
             head_in_dim = self.embed_dims[-1] if self.embed_dims else 0
             if head_in_dim > 0:
                 self.head = nn.Linear(head_in_dim, self.num_classes)

        # --- Initialize Weights ---
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # This method initializes Linear, LayerNorm, Conv2d layers defined directly within *this* module
        # (e.g., the self.head if it were used).
        # Initialization of submodules (Mlp, Attention, etc.) is handled by *their* apply call.
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # This might not be strictly needed here if all Conv2d are in submodules,
            # but keep for safety/consistency if direct Conv2d layers are added later.
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if m.groups > 0: fan_out //= m.groups
            if fan_out > 0:
                 m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            else:
                 m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

    # Removed init_weights(self, pretrained) - Use external loading logic if needed
    # Removed freeze_patch_emb() - Can be added back if required
    # Removed get_classifier(), reset_classifier() - Not standard for backbone usage

    @torch.jit.ignore
    def no_weight_decay(self):
        # Simplified: Exclude bias and norm weights/biases from weight decay
        # If specific pos_embeds existed, they would be added here.
        nwd = set()
        for n, p in self.named_parameters():
             if '.bias' in n or 'norm' in n.lower() or 'dwconv.dwconv.weight' in n: # Also exclude DWConv weights? Often done.
                 nwd.add(n)
        print(f"PVTv2 No Weight Decay keys ({len(nwd)}): {list(nwd)[:5]}...") # Debug: print first few keys
        # Filter out parameters that don't require grad
        no_decay = {name: p for name, p in self.named_parameters() if name in nwd and p.requires_grad}
        return no_decay

    def forward_features(self, x):
        """ Forward pass returning features based on fork_feat. """
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            blocks = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            # print(f"Stage {i+1} PatchEmbed Out: x={x.shape}, H={H}, W={W}")
            if x.shape[2] == 0: # Handle embed_dim=0 case from patch_embed
                 if self.fork_feat:
                     # Need to append something with correct B, C=0, H, W shape
                     # Create a tensor of shape (B, 0, H, W)
                     zero_channel_feat = torch.zeros((B, 0, H, W), device=x.device, dtype=x.dtype)
                     outs.append(zero_channel_feat)
                 continue # Skip blocks and norm if channels are zero

            for blk in blocks:
                x = blk(x, H, W)

            # Apply norm only if it's not Identity (i.e., dim > 0)
            if isinstance(norm, nn.LayerNorm):
                x = norm(x)

            # Reshape to (B, C, H, W) format expected by detection/segmentation heads
            # Ensure embed_dim matches C dimension after norm
            expected_c = self.embed_dims[i]
            if x.shape[2] != expected_c:
                 raise ValueError(f"Stage {i+1} output C dimension {x.shape[2]} != expected {expected_c}")
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # print(f"Stage {i+1} Final Out: {x.shape}")

            if self.fork_feat:
                outs.append(x)

        if self.fork_feat:
             # Verify output length matches width_list length if width_list exists
             if self.width_list is not None and len(outs) != len(self.width_list):
                 print(f"Warning: PVTv2 forward_features output count ({len(outs)}) "
                       f"mismatches stored width_list length ({len(self.width_list)}).")
                 # Check if the issue is due to zero-dim stages not being added
                 non_zero_dims = sum(d > 0 for d in self.embed_dims)
                 if len(outs) == non_zero_dims and len(self.width_list) == len(self.embed_dims):
                     print("  (Mismatch might be due to zero-dimension stages being skipped in output)")
                 # else: # Potentially raise error or handle differently
             return outs # Return list of features [P1, P2, P3, P4] etc. (relative to model stages)
        else:
            # If not forking, apply global average pooling and head (if exists)
            # This part is usually handled *outside* the backbone in detection models
            if x.shape[2] == 0: return x # Return empty tensor if last stage had 0 dim
            x = F.adaptive_avg_pool2d(x, 1).flatten(1) # Global avg pool
            x = self.head(x) # Apply classifier head
            return x

    def forward(self, x):
        """ Default forward pass. Calls forward_features for backbone usage. """
        return self.forward_features(x)


# --- Factory Functions (similar to MogaNet) ---

def pvtv2_b0(c1=3, fork_feat=True, **kwargs):
    """ PVTv2-B0 """
    model = PyramidVisionTransformerV2(c1=c1, arch='b0', fork_feat=fork_feat, **kwargs)
    return model

def pvtv2_b1(c1=3, fork_feat=True, **kwargs):
    """ PVTv2-B1 """
    model = PyramidVisionTransformerV2(c1=c1, arch='b1', fork_feat=fork_feat, **kwargs)
    return model

def pvtv2_b2(c1=3, fork_feat=True, **kwargs):
    """ PVTv2-B2 """
    model = PyramidVisionTransformerV2(c1=c1, arch='b2', fork_feat=fork_feat, **kwargs)
    return model

def pvtv2_b2_li(c1=3, fork_feat=True, **kwargs):
    """ PVTv2-B2 with Linear Attention"""
    model = PyramidVisionTransformerV2(c1=c1, arch='b2_li', fork_feat=fork_feat, **kwargs)
    return model

def pvtv2_b3(c1=3, fork_feat=True, **kwargs):
    """ PVTv2-B3 """
    model = PyramidVisionTransformerV2(c1=c1, arch='b3', fork_feat=fork_feat, **kwargs)
    return model

def pvtv2_b4(c1=3, fork_feat=True, **kwargs):
    """ PVTv2-B4 """
    model = PyramidVisionTransformerV2(c1=c1, arch='b4', fork_feat=fork_feat, **kwargs)
    return model

def pvtv2_b5(c1=3, fork_feat=True, **kwargs):
    """ PVTv2-B5 """
    model = PyramidVisionTransformerV2(c1=c1, arch='b5', fork_feat=fork_feat, **kwargs)
    return model


# --- Example Usage (for standalone testing) ---
if __name__ == "__main__":
    print("--- Testing Refactored PVTv2 Class ---")
    # Test with a size common in detection/segmentation
    # Note: PVTv2 expects input size divisible by 32 (4 * 2 * 2 * 2)
    test_h, test_w = 640, 640
    image_size = (1, 3, test_h, test_w)
    image = torch.rand(*image_size)
    print(f"Input image shape: {image.shape}")

    # --- Test B2 variant ---
    try:
        print("\n--- Testing pvt_v2_b2 (fork_feat=True) ---")
        # Instantiate using factory function
        model_b2 = pvtv2_b2(c1=3, fork_feat=True, img_size=test_h) # Pass img_size hint
        model_b2.eval()

        print(f"PVTv2-B2 Initialized.")
        # Check width_list
        print(f"  Architecture embed_dims: {model_b2.embed_dims}")
        print(f"  Stored width_list: {model_b2.width_list}")
        assert hasattr(model_b2, 'width_list')
        assert model_b2.width_list == model_b2.embed_dims

        with torch.no_grad():
            out_feat_b2 = model_b2(image) # Calls forward -> forward_features

        print(f"\n  Output: List of {len(out_feat_b2)} tensors")
        output_channels_b2 = []
        expected_strides = [4, 8, 16, 32] # PVTv2 strides
        for i, feat in enumerate(out_feat_b2):
            print(f"    Stage {i+1} Feature Shape: {feat.shape}")
            output_channels_b2.append(feat.shape[1])
            # Check spatial dimensions based on expected strides
            expected_h = test_h // expected_strides[i]
            expected_w = test_w // expected_strides[i]
            print(f"      Expected HxW: {expected_h}x{expected_w}, Got: {feat.shape[2]}x{feat.shape[3]}")
            assert feat.shape[2] == expected_h
            assert feat.shape[3] == expected_w


        # Verify output channels match width_list
        print(f"  Output channels: {output_channels_b2}")
        assert output_channels_b2 == model_b2.width_list
        print("  PVTv2-B2 Test Successful!")

    except Exception as e:
        print(f"Error during pvt_v2_b2 test: {e}")
        import traceback
        traceback.print_exc()

    # --- Test B2_li variant ---
    try:
        print("\n--- Testing pvt_v2_b2_li (fork_feat=True) ---")
        model_b2_li = pvtv2_b2_li(c1=3, fork_feat=True, img_size=test_h)
        model_b2_li.eval()
        print(f"PVTv2-B2_li Initialized.")
        print(f"  Stored width_list: {model_b2_li.width_list}")
        assert model_b2_li.width_list == model_b2_li.embed_dims

        with torch.no_grad():
            out_feat_b2_li = model_b2_li(image)

        print(f"\n  Output: List of {len(out_feat_b2_li)} tensors")
        output_channels_b2_li = []
        for i, feat in enumerate(out_feat_b2_li):
            print(f"    Stage {i+1} Feature Shape: {feat.shape}")
            output_channels_b2_li.append(feat.shape[1])
            expected_h = test_h // expected_strides[i]
            expected_w = test_w // expected_strides[i]
            assert feat.shape[2] == expected_h
            assert feat.shape[3] == expected_w

        print(f"  Output channels: {output_channels_b2_li}")
        assert output_channels_b2_li == model_b2_li.width_list
        print("  PVTv2-B2_li Test Successful!")

    except Exception as e:
        print(f"Error during pvt_v2_b2_li test: {e}")
        import traceback
        traceback.print_exc()

    # --- Test Classification Mode (fork_feat=False) ---
    # try:
    #     print("\n--- Testing pvt_v2_b0 (fork_feat=False, num_classes=50) ---")
    #     model_cls = pvt_v2_b0(c1=3, fork_feat=False, num_classes=50, img_size=test_h)
    #     model_cls.eval()
    #     print(f"PVTv2-B0 (classification) Initialized.")
    #     print(f"  Stored width_list: {model_cls.width_list}")
    #     # Should contain only last embed_dim if fork_feat=False and width_list defined that way
    #     # assert model_cls.width_list == [model_cls.embed_dims[-1]]

    #     with torch.no_grad():
    #         out_cls = model_cls(image) # Calls forward -> forward_features -> head
    #     print(f"  Classification Output Shape: {out_cls.shape}")
    #     assert out_cls.shape == (1, 50)
    #     print("  PVTv2-B0 Classification Test Successful!")

    # except Exception as e:
    #      print(f"Error during classification test: {e}")
    #      import traceback
    #      traceback.print_exc()