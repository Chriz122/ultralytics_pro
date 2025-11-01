# ------------------------------------------------------------
# Copyright (c) VCU, Nanjing University.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# Written by Qing-Long Zhang
# ------------------------------------------------------------
# Code modified based on SMT (Code 2) for integration and functionality enhancement.
# ------------------------------------------------------------

import torch
import torch.nn as nn
import copy
import math

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

__all__ = ['rest_lite', 'rest_small', 'rest_base', 'rest_large']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
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
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 apply_transform=False):
        super().__init__()
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
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        self.apply_transform = apply_transform and num_heads > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1)
            self.transform_norm = nn.InstanceNorm2d(self.num_heads)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, apply_transform=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, apply_transform=apply_transform)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(self, patch_size=16, in_ch=3, out_ch=768, with_pos=False):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size + 1, stride=patch_size, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(out_ch)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.norm(x)
        if self.with_pos:
            x = self.pos(x)
        # H, W in output feature map
        H_out, W_out = H // self.patch_size[0], W // self.patch_size[1]
        x = x.flatten(2).transpose(1, 2)
        return x, (H_out, W_out)


class BasicStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, with_pos=False):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        if self.with_pos:
            x = self.pos(x)
        return x


class ResT(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
                 norm_layer=nn.LayerNorm, apply_transform=False, img_size=224, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.apply_transform = apply_transform
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.img_size = img_size

        self.stem = BasicStem(in_ch=in_chans, out_ch=embed_dims[0], with_pos=True)

        self.patch_embed_2 = PatchEmbed(patch_size=2, in_ch=embed_dims[0], out_ch=embed_dims[1], with_pos=True)
        self.patch_embed_3 = PatchEmbed(patch_size=2, in_ch=embed_dims[1], out_ch=embed_dims[2], with_pos=True)
        self.patch_embed_4 = PatchEmbed(patch_size=2, in_ch=embed_dims[2], out_ch=embed_dims[3], with_pos=True)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0], apply_transform=apply_transform)
            for i in range(self.depths[0])])
        self.norm1 = norm_layer(embed_dims[0]) # Add norm layer for stage1 output consistency
        cur += depths[0]

        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1], apply_transform=apply_transform)
            for i in range(self.depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2], apply_transform=apply_transform)
            for i in range(self.depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3], apply_transform=apply_transform)
            for i in range(self.depths[3])])
        self.norm4 = norm_layer(embed_dims[3]) # Changed name from self.norm to self.norm4

        # classification head (kept for standalone classification tasks)
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)
        
        # --- Add width_list calculation ---
        self.width_list = []
        try:
            self.eval() 
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size)
            with torch.no_grad():
                 features = self.forward_features(dummy_input)
            self.width_list = [f.size(1) for f in features]
            self.train()
        except Exception as e:
            print(f"Error during dummy forward pass for width_list calculation: {e}")
            print("Setting width_list to embed_dims as fallback.")
            self.width_list = self.embed_dims
            self.train()

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        B = x.shape[0]
        feature_outputs = []

        # Stage 1
        x = self.stem(x)
        _, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        for blk in self.stage1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x_out = x.permute(0, 2, 1).reshape(B, -1, H, W)
        feature_outputs.append(x_out)

        # Stage 2
        x, (H, W) = self.patch_embed_2(x_out)
        for blk in self.stage2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x_out = x.permute(0, 2, 1).reshape(B, -1, H, W)
        feature_outputs.append(x_out)

        # Stage 3
        x, (H, W) = self.patch_embed_3(x_out)
        for blk in self.stage3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x_out = x.permute(0, 2, 1).reshape(B, -1, H, W)
        feature_outputs.append(x_out)

        # Stage 4
        x, (H, W) = self.patch_embed_4(x_out)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x_out = x.permute(0, 2, 1).reshape(B, -1, H, W)
        feature_outputs.append(x_out)
        
        return feature_outputs # Return list of feature maps

    def forward(self, x):
        # The primary forward now returns features for downstream tasks
        return self.forward_features(x)


@register_model
def rest_lite(pretrained=False, **kwargs):
    model = ResT(embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True, **kwargs)
    model.default_cfg = _cfg(url='')
    return model


@register_model
def rest_small(pretrained=False, **kwargs):
    model = ResT(embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                 depths=[2, 2, 6, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True, **kwargs)
    model.default_cfg = _cfg(url='')
    return model


@register_model
def rest_base(pretrained=False, **kwargs):
    model = ResT(embed_dims=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                 depths=[2, 2, 6, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True, **kwargs)
    model.default_cfg = _cfg(url='')
    return model


@register_model
def rest_large(pretrained=False, **kwargs):
    model = ResT(embed_dims=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                 depths=[2, 2, 18, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True, **kwargs)
    model.default_cfg = _cfg(url='')
    return model

if __name__ == '__main__':
    img_h, img_w = 640, 640
    print("--- Creating ResT-Lite model ---")
    # You can pass img_size for the dummy forward pass
    model = rest_lite(img_size=img_h)
    print("Model created successfully.")
    print("Calculated width_list:", model.width_list)

    # Test forward pass
    input_tensor = torch.rand(2, 3, img_h, img_w)
    print(f"\n--- Testing ResT-Lite forward pass (Input: {input_tensor.shape}) ---")

    model.eval()
    try:
        with torch.no_grad():
            output_features = model(input_tensor)
        print("Forward pass successful.")
        # The output should be a list of tensors
        print("Output is a list: ", isinstance(output_features, list))
        print("Number of feature maps: ", len(output_features))
        print("Output feature shapes:")
        for i, features in enumerate(output_features):
            print(f"  Stage {i+1}: {features.shape}")

        # Verify width_list matches runtime output
        runtime_widths = [f.size(1) for f in output_features]
        print("\nRuntime output feature channels:", runtime_widths)
        assert model.width_list == runtime_widths, "Width list mismatch!"
        print("Width list verified successfully.")

        # --- Test deepcopy ---
        print("\n--- Testing deepcopy ---")
        copied_model = copy.deepcopy(model)
        print("Deepcopy successful.")

        # Optional: Test copied model forward pass
        with torch.no_grad():
             output_copied = copied_model(input_tensor)
        print("Copied model forward pass successful.")
        assert len(output_copied) == len(output_features)
        for i in range(len(output_features)):
             assert output_copied[i].shape == output_features[i].shape
        print("Copied model output shapes verified.")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()