# ------------------------------------------------------------
# Copyright (c) VCU, Nanjing University.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# Written by Qing-Long Zhang
# ------------------------------------------------------------

import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim, eps=1e-6)

        self.up = nn.Sequential(
            nn.Conv2d(dim, sr_ratio * sr_ratio * dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.PixelShuffle(upscale_factor=sr_ratio)
        )
        self.up_norm = nn.LayerNorm(dim, eps=1e-6)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_kv = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_kv = self.sr(x_kv).reshape(B, C, -1).permute(0, 2, 1)
            x_kv = self.sr_norm(x_kv)
        else:
            x_kv = x

        kv = self.kv(x_kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if self.sr_ratio > 1:
            identity = v.transpose(-1, -2).reshape(B, C, H // self.sr_ratio, W // self.sr_ratio)
            identity = self.up(identity).flatten(2).transpose(1, 2)
            x = self.proj(x + self.up_norm(identity))
        else:
            x = self.proj(x)
            
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class ConvStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=96, patch_size=4, with_pos=True):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        stem = []
        in_dim, out_dim = in_ch, out_ch // 2
        for i in range(2):
            stem.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(out_dim))
            stem.append(nn.ReLU(inplace=True))
            in_dim, out_dim = out_dim, out_dim * 2

        stem.append(nn.Conv2d(in_dim, out_ch, kernel_size=1, stride=1))
        self.proj = nn.Sequential(*stem)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, out_ch=96, patch_size=2, with_pos=True):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=self.patch_size[0] + 1, stride=self.patch_size, padding=self.patch_size[0] // 2)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)
        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class ResTV2(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[96, 192, 384, 768],
                 num_heads=[1, 2, 4, 8], drop_path_rate=0.,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], img_size=224, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.img_size = img_size

        self.stem = ConvStem(in_chans, embed_dims[0], patch_size=4)
        self.patch_2 = PatchEmbed(embed_dims[0], embed_dims[1], patch_size=2)
        self.patch_3 = PatchEmbed(embed_dims[1], embed_dims[2], patch_size=2)
        self.patch_4 = PatchEmbed(embed_dims[2], embed_dims[3], patch_size=2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], sr_ratios[0], dpr[cur + i])
            for i in range(depths[0])
        ])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], sr_ratios[1], dpr[cur + i])
            for i in range(depths[1])
        ])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], sr_ratios[2], dpr[cur + i])
            for i in range(depths[2])
        ])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], sr_ratios[3], dpr[cur + i])
            for i in range(depths[3])
        ])

        self.norm = nn.LayerNorm(embed_dims[-1], eps=1e-6)
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
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
            self.width_list = self.embed_dims

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward_features(self, x):
        outputs = []
        B, _, _, _ = x.shape
        
        # stage 1
        x, (H, W) = self.stem(x)
        for blk in self.stage1:
            x = blk(x, H, W)
        x_s1 = x.permute(0, 2, 1).reshape(B, self.embed_dims[0], H, W)
        outputs.append(x_s1)

        # stage 2
        x, (H, W) = self.patch_2(x_s1)
        for blk in self.stage2:
            x = blk(x, H, W)
        x_s2 = x.permute(0, 2, 1).reshape(B, self.embed_dims[1], H, W)
        outputs.append(x_s2)

        # stage 3
        x, (H, W) = self.patch_3(x_s2)
        for blk in self.stage3:
            x = blk(x, H, W)
        x_s3 = x.permute(0, 2, 1).reshape(B, self.embed_dims[2], H, W)
        outputs.append(x_s3)

        # stage 4
        x, (H, W) = self.patch_4(x_s3)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = self.norm(x)
        x_s4 = x.permute(0, 2, 1).reshape(B, self.embed_dims[3], H, W)
        outputs.append(x_s4)

        return outputs

    def forward(self, x):
        x = self.forward_features(x)
        return x

@register_model
def restv2_tiny(pretrained=False, img_size=224, **kwargs):
    model = ResTV2(img_size=img_size, embed_dims=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], depths=[1, 2, 6, 2], sr_ratios=[8, 4, 2, 1], **kwargs)
    return model

@register_model
def restv2_small(pretrained=False, img_size=224, **kwargs):
    model = ResTV2(img_size=img_size, embed_dims=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], depths=[1, 2, 12, 2], sr_ratios=[8, 4, 2, 1], **kwargs)
    return model

@register_model
def restv2_base(pretrained=False, img_size=224, **kwargs):
    model = ResTV2(img_size=img_size, embed_dims=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], depths=[1, 3, 16, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    return model

@register_model
def restv2_large(pretrained=False, img_size=224, **kwargs):
    model = ResTV2(img_size=img_size, embed_dims=[128, 256, 512, 1024], num_heads=[2, 4, 8, 16], depths=[2, 3, 16, 2], sr_ratios=[8, 4, 2, 1], **kwargs)
    return model


if __name__ == '__main__':
    img_h, img_w = 640, 640
    print("--- Creating ResTV2 Tiny model ---")
    # 測試模型創建
    model = restv2_tiny(img_size=img_h)
    print("Model created successfully.")
    
    # 檢查 width_list 是否被正確計算
    print("Calculated width_list:", model.width_list)
    assert model.width_list == [96, 192, 384, 768]
    print("Width list verified successfully.")

    # 測試前向傳播
    input_tensor = torch.rand(2, 3, img_h, img_w)
    print(f"\n--- Testing ResTV2 Tiny forward pass (Input: {input_tensor.shape}) ---")
    
    model.eval()
    try:
        with torch.no_grad():
            output_features = model(input_tensor)
        print("Forward pass successful.")
        
        # 檢查輸出是否為列表且包含四個元素
        print(f"Output type: {type(output_features)}")
        print(f"Number of feature maps in output: {len(output_features)}")
        assert isinstance(output_features, list) and len(output_features) == 4

        # 打印每個輸出特徵圖的形狀
        print("Output feature shapes:")
        for i, features in enumerate(output_features):
            print(f"Stage {i+1}: {features.shape}")

        # 驗證輸出通道數是否與 width_list 一致
        runtime_widths = [f.size(1) for f in output_features]
        print("\nRuntime output feature channels:", runtime_widths)
        assert model.width_list == runtime_widths
        print("Runtime channels verified successfully.")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()