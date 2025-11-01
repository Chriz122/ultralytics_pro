import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

# Note: The following imports related to data transforms and complexity calculation
# are kept for context but not strictly necessary for the core model definition changes.
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.data import create_transform # Removed unused import
from timm.data.transforms import str_to_pil_interp
# from ptflops import get_model_complexity_info # Removed from __main__ for clarity
# from thop import profile # Removed from __main__ for clarity


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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W)) # dwconv handles reshape internally
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None,
                       attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads

        # Ensure dim is divisible by head counts where applicable
        if ca_attention == 1 and ca_num_heads > 0:
             assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        if ca_attention == 0 and sa_num_heads > 0:
             assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if ca_attention == 1:
            assert ca_num_heads > 0, "ca_num_heads must be positive when ca_attention=1"
            self.split_groups=self.dim//ca_num_heads
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.s = nn.Linear(dim, dim, bias=qkv_bias)
            for i in range(self.ca_num_heads):
                local_conv = nn.Conv2d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//self.ca_num_heads)
                setattr(self, f"local_conv_{i + 1}", local_conv)
            self.proj0 = nn.Conv2d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups)
            self.bn = nn.BatchNorm2d(dim*expand_ratio)
            self.proj1 = nn.Conv2d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1)

        else: # sa_attention
            assert sa_num_heads > 0, "sa_num_heads must be positive when ca_attention=0"
            head_dim = dim // sa_num_heads
            self.scale = qk_scale or head_dim ** -0.5
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            # Assuming local_conv is always a depthwise conv here for SA
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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


    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.ca_attention == 1:
            v = self.v(x)
            s = self.s(x).reshape(B, H, W, self.ca_num_heads, C//self.ca_num_heads).permute(3, 0, 4, 1, 2)
            s_out_list = [] # Collect outputs from different heads
            for i in range(self.ca_num_heads):
                local_conv = getattr(self, f"local_conv_{i + 1}")
                s_i= s[i] # [B, C//ca_num_heads, H, W]
                s_i = local_conv(s_i)
                s_out_list.append(s_i)

            # Concatenate along the channel dimension
            s_out = torch.cat(s_out_list, dim=1) # Shape should be [B, C, H, W]
            if s_out.shape[1] != C: # Added safety check
                 print(f"Warning: Concatenated s_out channels {s_out.shape[1]} != C {C} in Attention")
                 # Handle potential dimension mismatch if necessary, e.g., adjust proj0/proj1 or raise error

            s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
            # self.modulator = s_out # <--- REMOVED THIS LINE
            s_out_reshaped = s_out.reshape(B, C, N).permute(0, 2, 1)
            x = s_out_reshaped * v # Use the reshaped s_out

        else: # sa_attention
            q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
            k, v_heads = kv[0], kv[1] # Renamed v to v_heads to avoid conflict

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # Standard attention result
            attn_x = (attn @ v_heads).transpose(1, 2).reshape(B, N, C)

            # Local convolution path
            # Reshape v_heads from (B, sa_num_heads, N, C//sa_num_heads) -> (B, N, C) -> (B, C, H, W)
            v_spatial = v_heads.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W)
            local_conv_x = self.local_conv(v_spatial).view(B, C, N).transpose(1, 2)

            x = attn_x + local_conv_x

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, ca_num_heads, sa_num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                    use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ca_attention=1,expand_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # Ensure correct head count is passed based on ca_attention flag
        current_ca_heads = ca_num_heads if ca_attention == 1 else -1
        current_sa_heads = sa_num_heads if ca_attention == 0 else -1

        self.attn = Attention(
            dim,
            ca_num_heads=current_ca_heads, sa_num_heads=current_sa_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, ca_attention=ca_attention,
            expand_ratio=expand_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

        # Removed apply(_init_weights) from Block, let SMT handle it once
        # self.apply(self._init_weights)

    # Removed _init_weights from Block, SMT handles it
    # def _init_weights(self, m): ...

    def forward(self, x, H, W):
        # Apply LayerScale if parameters exist, otherwise use gamma=1.0
        gamma_1 = self.gamma_1 if isinstance(self.gamma_1, nn.Parameter) else 1.0
        gamma_2 = self.gamma_2 if isinstance(self.gamma_2, nn.Parameter) else 1.0

        x = x + self.drop_path(gamma_1 * self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(gamma_2 * self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding with overlapping patches"""
    def __init__(self, img_size=224, patch_size=3, stride=2, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        # img_size = to_2tuple(img_size) # Not directly used

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        # Removed apply(_init_weights) from OverlapPatchEmbed
        # self.apply(self._init_weights)

    # Removed _init_weights from OverlapPatchEmbed
    # def _init_weights(self, m): ...

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # B, N, C
        x = self.norm(x)
        return x, H, W


class Head(nn.Module):
    """ Initial Patch Embedding using Conv layers"""
    def __init__(self, head_conv, dim, in_chans=3): # Added in_chans
        super(Head, self).__init__()
        padding = head_conv // 2
        # Use in_chans parameter
        stem = [nn.Conv2d(in_chans, dim, kernel_size=head_conv, stride=2, padding=padding, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(True)]
        # Add the second conv layer with stride 2
        stem.append(nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False))
        stem.append(nn.BatchNorm2d(dim))
        stem.append(nn.ReLU(True))
        self.conv = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(dim)
        # Removed apply(_init_weights) from Head
        # self.apply(self._init_weights)

    # Removed _init_weights from Head
    # def _init_weights(self, m): ...

    def forward(self, x):
        x = self.conv(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # B, N, C
        x = self.norm(x)
        return x, H, W


class SMT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[8, 6, 4, 2],
                 qkv_bias=False, qk_scale=None, use_layerscale=False, layerscale_value=1e-4, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 2, 8, 1], ca_attentions=[1, 1, 1, 0], num_stages=4, head_conv=3, expand_ratio=2, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.embed_dims = embed_dims
        self.img_size = img_size
        self.in_chans = in_chans
        self.use_layerscale = use_layerscale # Store for Block init

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        # Make norm_layer a class attribute for easier access if needed later
        self.norm_layer = norm_layer

        for i in range(num_stages):
            if i == 0:
                patch_embed = Head(head_conv=head_conv, dim=embed_dims[i], in_chans=in_chans) # Pass in_chans
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size // (2 ** (i + 1)), # Theoretical input size
                                            patch_size=3,
                                            stride=2,
                                            in_chans=embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i],
                # Pass correct head counts for this stage based on ca_attentions[i]
                ca_num_heads=ca_num_heads[i],
                sa_num_heads=sa_num_heads[i],
                mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                use_layerscale=self.use_layerscale, # Use stored value
                layerscale_value=layerscale_value,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=self.norm_layer, # Use stored value
                ca_attention=ca_attentions[i], # Use stage-specific attention type
                expand_ratio=expand_ratio)
                for j in range(depths[i])])
            norm = self.norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head - kept for potential use but not applied in forward
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # Apply weights initialization once at the end
        self.apply(self._init_weights)

        # --- Add width_list calculation (can run dummy forward safely now) ---
        self.width_list = []
        try:
            # Perform a dummy forward pass to determine intermediate feature channel sizes
            self.eval() # Set to eval mode to disable dropout etc.
            # Create a dummy input tensor matching expected dimensions
            # Ensure tensor is on the same device parameters will be on (cpu initially)
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size)

            # Pass dummy input through the forward method
            # Use torch.no_grad() to avoid graph construction during this pass
            with torch.no_grad():
                 features = self.forward(dummy_input)

            # Extract channel dimension (dim=1) from each feature map in the list
            self.width_list = [f.size(1) for f in features]
            self.train() # Set back to train mode
        except Exception as e:
            print(f"Error during dummy forward pass for width_list calculation: {e}")
            print("Setting width_list to embed_dims as fallback.")
            self.width_list = self.embed_dims # Fallback or handle error appropriately
            self.train() # Ensure model is back in train mode even if error occurs


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): # Added ConvTranspose2d just in case
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if m.groups > 0: # Check for groups > 0
                fan_out //= m.groups
            if fan_out > 0: # Avoid division by zero for unusual convs
                 m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            else:
                 m.weight.data.normal_(0, 0.02) # Fallback initialization
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d): # Added BatchNorm initialization
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


    def freeze_patch_emb(self):
        patch_embed1 = getattr(self, "patch_embed1")
        if patch_embed1:
            for param in patch_embed1.parameters():
                 param.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        # Refined based on common practice: parameters of Norm layers and biases
        no_decay = set()
        for name, param in self.named_parameters():
            if 'norm' in name or 'bias' in name:
                 no_decay.add(name)
            # Add other specific parameters if needed, e.g., pos embeds if they existed
            # if 'pos_embed' in name: no_decay.add(name)
        return no_decay

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        feature_outputs = [] # List to store features from each stage

        current_h, current_w = x.shape[2], x.shape[3] # Track spatial dims if needed

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            # Input 'x' should be in (B, C, H, W) format for patch_embed
            x, H, W = patch_embed(x) # Output x shape: B, N, C

            # Update spatial dimensions tracker
            current_h, current_w = H, W

            # Apply transformer blocks for this stage
            for blk in block:
                x = blk(x, H, W)

            # Apply normalization
            x = norm(x) # x shape: B, N, C

            # --- Reshape to spatial format (B, C, H, W) and store ---
            C = self.embed_dims[i] # Get correct channel dim
            if H * W != x.shape[1]:
                 print(f"Warning: H*W ({H*W}) != N ({x.shape[1]}) in stage {i+1}. Reshaping might fail.")
                 # Add fallback or error handling if needed
            x_spatial = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            feature_outputs.append(x_spatial)
            # ---

            # Prepare 'x' for the next stage's patch_embed (needs B, C, H, W)
            x = x_spatial

        return feature_outputs # Return the list of spatial feature maps

    def forward(self, x):
        features = self.forward_features(x)
        return features


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # Reshape from (B, N, C) to (B, C, H, W)
        x_reshaped = x.transpose(1, 2).view(B, C, H, W)
        x_conv = self.dwconv(x_reshaped)
        # Reshape back to (B, N, C)
        x_out = x_conv.flatten(2).transpose(1, 2)
        return x_out


# --- Data Transforms (kept for context) ---
def build_transforms(img_size, center_crop=False):
    t = []
    interp_mode = str_to_pil_interp('bicubic')
    if center_crop:
        size = int((256 / 224) * img_size)
        t.append(transforms.Resize(size, interpolation=interp_mode))
        t.append(transforms.CenterCrop(img_size))
    else:
        t.append(transforms.Resize((img_size, img_size), interpolation=interp_mode)) # Ensure square resize
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_transforms4display(img_size, center_crop=False):
    t = []
    interp_mode = str_to_pil_interp('bicubic')
    if center_crop:
        size = int((256 / 224) * img_size)
        t.append(transforms.Resize(size, interpolation=interp_mode))
        t.append(transforms.CenterCrop(img_size))
    else:
        t.append(transforms.Resize((img_size, img_size), interpolation=interp_mode)) # Ensure square resize
    t.append(transforms.ToTensor())
    return transforms.Compose(t)

# --- Factory Functions ---

def smt_t(pretrained=False, img_size=224, **kwargs):
    model = SMT(
        img_size=img_size,
        embed_dims=[64, 128, 256, 512], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[4, 4, 4, 2], qkv_bias=True, depths=[2, 2, 8, 1],
        ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()
    # Add pretrained loading logic here if required
    return model

def smt_s(pretrained=False, img_size=224, **kwargs):
    model = SMT(
        img_size=img_size,
        embed_dims=[64, 128, 256, 512], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[4, 4, 4, 2], qkv_bias=True, depths=[3, 4, 18, 2],
        ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()
    return model

def smt_b(pretrained=False, img_size=224, **kwargs):
    model = SMT(
        img_size=img_size,
        embed_dims=[64, 128, 256, 512], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[8, 6, 4, 2], qkv_bias=True, depths=[4, 6, 28, 2],
        ca_attentions=[1, 1, 1, 0], head_conv=7, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()
    return model

def smt_l(pretrained=False, img_size=224, **kwargs):
    model = SMT(
        img_size=img_size,
        embed_dims=[96, 192, 384, 768], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[8, 6, 4, 2], qkv_bias=True, depths=[4, 6, 28, 4],
        ca_attentions=[1, 1, 1, 0], head_conv=7, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    import torch
    img_h, img_w = 224, 224
    print("--- Creating SMT Tiny model ---")
    model = smt_t(img_size=img_h)
    print("Model created successfully.")
    print("Calculated width_list:", model.width_list)

    # Test forward pass
    input_tensor = torch.rand(2, 3, img_h, img_w)
    print(f"\n--- Testing SMT Tiny forward pass (Input: {input_tensor.shape}) ---")

    model.eval()
    try:
        with torch.no_grad():
            output_features = model(input_tensor)
        print("Forward pass successful.")
        print("Output feature shapes:")
        for i, features in enumerate(output_features):
            print(f"Stage {i+1}: {features.shape}") # Should be [B, C, H_i, W_i]

        # Verify width_list matches runtime output
        runtime_widths = [f.size(1) for f in output_features]
        print("\nRuntime output feature channels:", runtime_widths)
        assert model.width_list == runtime_widths, "Width list mismatch!"
        print("Width list verified successfully.")

        # --- Test deepcopy ---
        print("\n--- Testing deepcopy ---")
        import copy
        copied_model = copy.deepcopy(model)
        print("Deepcopy successful.")

        # Optional: Test copied model forward pass
        with torch.no_grad():
             output_copied = copied_model(input_tensor)
        print("Copied model forward pass successful.")
        assert len(output_copied) == len(output_features)
        for i in range(len(output_features)):
             assert output_copied[i].shape == output_features[i].shape
             # assert torch.allclose(output_copied[i], output_features[i]) # Check values if needed
        print("Copied model output shapes verified.")


    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

    # Example for another model size
    # print("\n--- Creating SMT Small model ---")
    # model_s = smt_s(img_size=img_h)
    # print("Model created successfully.")
    # print("Calculated width_list (Small):", model_s.width_list)