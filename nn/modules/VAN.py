import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model # Removed: Not using timm registration directly for this modification
# from timm.models.vision_transformer import _cfg # Removed: Related to pretrained cfgs
import math

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
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

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
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

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)
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

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W


class VAN(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4): # Removed flag, assuming num_classes behavior is standard
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.embed_dims = embed_dims # Store for later use, e.g. in reset_classifier

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            # Calculate img_size for current patch_embed, this is fine
            current_stage_img_size = img_size if i == 0 else img_size // (2 ** i) # Adjusted for clarity
            # Stride logic means effective downsampling by 2 at each stage after first
            # patch_size=7 if i == 0 else 3,
            # stride=4 if i == 0 else 2,
            # This means overall downsampling by 4, 2, 2, 2 for stages 1, 2, 3, 4
            # So feature map sizes relative to input H,W: H/4, H/8, H/16, H/32

            patch_embed = OverlapPatchEmbed(img_size=current_stage_img_size,
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block_list = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            
            # This norm is LayerNorm, applied after flattening in original ViT-style VAN
            # For backbone use, features are typically (B,C,H,W) and BatchNorm2d is common within blocks
            stage_norm = norm_layer(embed_dims[i]) 
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block_list)
            setattr(self, f"norm{i + 1}", stage_norm) # This is the LayerNorm for inter-stage connection

        # Classification head
        # Input to head is embed_dims[3] (or embed_dims[num_stages-1])
        self.head = nn.Linear(embed_dims[num_stages-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        # Initialize self.width_list
        # Use torch.no_grad to prevent tracking history for this dummy forward pass
        if in_chans > 0 and img_size > 0 : # Make sure valid inputs for dummy pass
            try:
                with torch.no_grad():
                    dummy_input = torch.randn(1, in_chans, img_size, img_size)
                    features = self.forward_features(dummy_input)
                self.width_list = [f.size(1) for f in features]
            except Exception as e:
                print(f"Warning: Could not initialize self.width_list during VAN init: {e}")
                print(f"Ensure img_size ({img_size}) and in_chans ({in_chans}) are appropriate.")
                self.width_list = [] # Default to empty list on failure
        else:
            self.width_list = []


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d): # Added BatchNorm2d specific init
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        # The input dim to the head is the channel dim of the last stage.
        self.head = nn.Linear(self.embed_dims[self.num_stages-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outputs = [] # List to store feature maps from selected stages

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block_module = getattr(self, f"block{i + 1}")
            norm_module = getattr(self, f"norm{i + 1}") # This is the LayerNorm for ViT-style processing

            x, H, W = patch_embed(x) # x is (B, C, H, W)
            for blk in block_module:
                x = blk(x) # x is (B, C, H, W)
            
            # At this point, x is the (B,C,H,W) feature map from the current stage's blocks.
            # This is what we want to output for detection backbones.
            outputs.append(x) 

            # If not the last stage, prepare x for the next stage using original VAN logic
            # (flatten, LayerNorm, reshape)
            if i < self.num_stages - 1:
                # Original VAN flattens, applies LayerNorm, then reshapes for the next stage
                x_for_next_stage = x.flatten(2).transpose(1, 2) # (B, N, C) where N=H*W
                x_for_next_stage = norm_module(x_for_next_stage)
                x = x_for_next_stage.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
            # For the last stage, x (which is outputs[-1]) is already in (B,C,H,W)
            # and will be returned as part of the 'outputs' list.
            # The original VAN's final global average pooling and head application
            # are handled outside this function if classification is needed.
            
        return outputs # Return a list of (B, C, H, W) tensors

    def forward(self, x):
        # For backbone usage, forward_features returns a list of feature maps
        x = self.forward_features(x)
        return x
    
    def forward_for_classification(self, x):
        """
        Performs forward pass for classification, similar to original VAN.
        This is separate from the main `forward` to keep it clean for backbone use.
        """
        list_features = self.forward_features(x)
        
        # Get the last feature map from the list
        x_last_stage = list_features[-1] # (B, C_last, H_last, W_last)
        
        # Apply the final norm (LayerNorm) and pooling for the head
        # The norm for the last stage's output before head
        final_norm = getattr(self, f"norm{self.num_stages}") 
        B, C, H, W = x_last_stage.shape
        x_for_head = x_last_stage.flatten(2).transpose(1, 2) # (B, N, C)
        x_for_head = final_norm(x_for_head) # Apply LayerNorm
        x_for_head = x_for_head.mean(dim=1) # Global Average Pooling -> (B, C)
        
        return self.head(x_for_head)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

# Removed _conv_filter as it's not used after removing pretrained weight loading logic

# Removed model_urls and load_model_weights function

# --- Model Instantiation Functions (EMO-like style) ---

# @register_model # Removed timm registration
def van_b0(num_classes=0, **kwargs): # Default num_classes=0 for backbone usage
    model = VAN(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        num_classes=num_classes, **kwargs)
    # model.default_cfg = _cfg() # Removed
    return model

# @register_model # Removed timm registration
def van_b1(num_classes=0, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 4, 2],
        num_classes=num_classes, **kwargs)
    # model.default_cfg = _cfg() # Removed
    return model

# @register_model # Removed timm registration
def van_b2(num_classes=0, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 12, 3],
        num_classes=num_classes, **kwargs)
    # model.default_cfg = _cfg() # Removed
    return model

# @register_model # Removed timm registration
def van_b3(num_classes=0, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 5, 27, 3],
        num_classes=num_classes, **kwargs)
    # model.default_cfg = _cfg() # Removed
    return model

# @register_model # Removed timm registration
def van_b4(num_classes=0, **kwargs): # Assuming van_b4 definition, not in original provided urls
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4], # Example, adjust as per actual van_b4
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], # Example
        num_classes=num_classes, **kwargs)
    # model.default_cfg = _cfg() # Removed
    return model

# @register_model # Removed timm registration
def van_b5(num_classes=0, **kwargs): # Assuming van_b5 definition
    model = VAN(
        embed_dims=[96, 192, 480, 768], mlp_ratios=[8, 8, 4, 4], # Example
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 24, 3], # Example
        num_classes=num_classes, **kwargs)
    # model.default_cfg = _cfg() # Removed
    return model

# @register_model # Removed timm registration
def van_b6(num_classes=0, **kwargs): # Assuming van_b6 definition
    model = VAN(
        embed_dims=[96, 192, 384, 768], mlp_ratios=[8, 8, 4, 4], # Example
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[6, 6, 90, 6], # Example
        num_classes=num_classes, **kwargs)
    # model.default_cfg = _cfg() # Removed
    return model

if __name__ == '__main__':
    # Test VAN model creation and forward pass
    print("Testing VAN-B0 as backbone:")
    # img_size is important for patch_embed and width_list calculation
    model_b0 = van_b0(img_size=224, in_chans=3, num_classes=0) 
    dummy_tensor = torch.randn(2, 3, 224, 224)
    
    print(f"Model: {model_b0.__class__.__name__}")
    print(f"Width list: {model_b0.width_list}")

    # Test backbone output
    features_list = model_b0(dummy_tensor)
    print(f"Number of feature maps returned: {len(features_list)}")
    for i, feat in enumerate(features_list):
        print(f"Shape of feature map {i+1}: {feat.shape}")

    # Test VAN-B1 with a classification head (though Ultralytics won't use this directly)
    print("\nTesting VAN-B1 with classification head (example):")
    model_b1_cls = van_b1(img_size=256, in_chans=3, num_classes=100) # Example num_classes
    dummy_tensor_cls = torch.randn(2, 3, 256, 256)
    
    print(f"Model: {model_b1_cls.__class__.__name__}")
    print(f"Width list: {model_b1_cls.width_list}") # Should still be populated

    # To get classification output:
    # class_output = model_b1_cls.forward_for_classification(dummy_tensor_cls)
    # print(f"Shape of classification output: {class_output.shape}")
    
    # The main forward will return features list for backbone usage
    features_list_b1 = model_b1_cls(dummy_tensor_cls)
    print(f"Number of feature maps returned by VAN-B1 forward(): {len(features_list_b1)}")
    for i, feat in enumerate(features_list_b1):
        print(f"Shape of VAN-B1 feature map {i+1}: {feat.shape}")

    # Example: Simulate Ultralytics accessing specific features
    if len(features_list_b1) >= 3: # Assuming Ultralytics might pick features like P3, P4, P5
        p3_equivalent = features_list_b1[1] # Example: second feature map in the list
        p4_equivalent = features_list_b1[2] # Example: third feature map
        p5_equivalent = features_list_b1[3] # Example: fourth feature map (if num_stages is 4)
        print(f"\nSimulated access for neck input:")
        print(f"P3 shape: {p3_equivalent.shape}")
        print(f"P4 shape: {p4_equivalent.shape}")
        print(f"P5 shape: {p5_equivalent.shape}")

    # Test width_list with different input size
    print("\nTesting VAN-B0 with img_size=640:")
    model_b0_large = van_b0(img_size=640, in_chans=3, num_classes=0)
    dummy_tensor_large = torch.randn(1, 3, 640, 640)
    print(f"Width list for VAN-B0 (640): {model_b0_large.width_list}")
    features_large = model_b0_large(dummy_tensor_large)
    for i, feat in enumerate(features_large):
        print(f"Shape of feature map {i+1} (640 input): {feat.shape}")