import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# ==============================================================================
# 基礎組件
# ==============================================================================

class LayerNorm2d(nn.Module):
    """(B, C, H, W) 的 LayerNorm"""
    def __init__(self, num_channels, eps=1e-6, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels)) if bias else None
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = x * self.weight.view(1, -1, 1, 1)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        return x

class LayerNormGeneral(nn.Module):
    """(B, ..., C) 的 LayerNorm"""
    def __init__(self, normalized_shape, scale=True, bias=True, eps=1e-6):
        super().__init__()
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(normalized_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(-1, keepdim=True)
        s = c.pow(2).mean(-1, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x

class OversizeConv2d(nn.Module):
    def __init__(self, dim, kernel_size, bias=False):
        super().__init__()
        # 確保 Kernel 是奇數，並計算正確的 Padding
        if kernel_size % 2 == 0:
            kernel_size += 1 # 強制轉為奇數以保證對稱 padding
        
        padding = kernel_size // 2
        
        # 分解卷積：Vertical (H) and Horizontal (W)
        # Padding 格式: (Left, Right, Top, Bottom)
        # 對於 conv_h (Kx1): 需要上下 Padding (Top, Bottom)，即 (0, 0, padding, padding)
        # 對於 conv_w (1xK): 需要左右 Padding (Left, Right)，即 (padding, padding, 0, 0)
        
        # 在 PyTorch Conv2d 中，padding參數如果是 tuple (padH, padW):
        # conv_h (K, 1): padding應為 (pad, 0)
        # conv_w (1, K): padding應為 (0, pad)
        
        self.conv_h = nn.Conv2d(dim, dim, (kernel_size, 1), padding=(padding, 0), groups=dim, bias=bias)
        self.conv_w = nn.Conv2d(dim, dim, (1, kernel_size), padding=(0, padding), groups=dim, bias=bias)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.conv_h(x)
        x = self.conv_w(x)
        return x

class ParC_V2_TokenMixer(nn.Module):
    def __init__(self, dim, expansion_ratio=2, act_layer=nn.GELU, bias=False, kernel_size=7, global_kernel_size=13, padding=None):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, 1, bias=True)
        self.act = act_layer()
        
        # Oversize Conv (Branch 1)
        self.dwconv1 = OversizeConv2d(med_channels // 2, global_kernel_size, bias)
        
        # Standard Conv (Branch 2)
        # 如果沒有指定 padding，自動計算 k//2
        if padding is None:
            padding = kernel_size // 2
            
        self.dwconv2 = nn.Conv2d(
            med_channels // 2, 
            med_channels // 2, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=med_channels // 2, 
            bias=bias
        )
        self.pwconv2 = nn.Conv2d(med_channels // 2, dim, 1, bias=bias)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.pwconv1(x)
        x1, x2 = x.chunk(2, 1)
        x2 = self.act(x2)
        
        # 分別計算兩路分支
        out_oversize = self.dwconv1(x2)
        out_standard = self.dwconv2(x2)
        
        # [關鍵修復] 尺寸安全檢查
        # 如果因為 padding 計算誤差導致尺寸不一致 (e.g. 17 vs 16)，強制對齊
        if out_oversize.shape != out_standard.shape:
            out_oversize = F.interpolate(out_oversize, size=out_standard.shape[-2:], mode='bilinear', align_corners=False)
            
        x2 = out_oversize + out_standard
        x = x1 * x2
        x = self.pwconv2(x)
        return x

class BGU(nn.Module):
    def __init__(self, dim, mlp_ratio=4, act_layer=nn.GELU, drop=0.0, bias=False):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features // 2, dim, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        # x: [B, H, W, C]
        x = self.fc1(x)
        x1, x2 = x.chunk(2, -1)
        x = x1 * self.act(x2)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# ==============================================================================
# YOLO 兼容模塊 (Wrappers)
# ==============================================================================

class ParCDown(nn.Module):
    """ParCNet 下採樣層"""
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.is_stem = (c1 == 3)
        
        # 根據是否為 Stem 決定 Padding
        # Stem (k=7, s=4) 原論文使用 p=2
        # Stage (k=3, s=2) 原論文使用 p=1
        if self.is_stem:
            p = 2 if k == 7 else k // 2
        else:
            p = k // 2

        if self.is_stem:
            self.pre_norm = nn.Identity()
            self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p)
            self.post_norm = LayerNormGeneral(c2, bias=False, eps=1e-6)
        else:
            self.pre_norm = LayerNorm2d(c1, bias=False, eps=1e-6)
            self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p)
            self.post_norm = nn.Identity()

    def forward(self, x):
        if self.is_stem:
            x = self.conv(x)
            x = x.permute(0, 2, 3, 1) 
            x = self.post_norm(x)
            x = x.permute(0, 3, 1, 2)
        else:
            x = self.pre_norm(x)
            x = self.conv(x)
        return x

class ParCBlock(nn.Module):
    """ParCNetV2 基本 Block"""
    def __init__(self, c1, c2, k_global=13, mlp_ratio=4):
        super().__init__()
        assert c1 == c2, f"ParCBlock input {c1} must match output {c2}"
        dim = c1
        
        self.norm1 = LayerNorm2d(dim)
        self.token_mixer = ParC_V2_TokenMixer(dim, global_kernel_size=k_global)
        
        self.norm2 = LayerNorm2d(dim)
        self.mlp = BGU(dim, mlp_ratio=mlp_ratio)
        
        self.layer_scale1 = nn.Parameter(torch.ones(dim) * 1e-6)
        self.layer_scale2 = nn.Parameter(torch.ones(dim) * 1e-6)
        self.drop_path = nn.Identity()

    def forward(self, x):
        # Part 1: Token Mixer
        shortcut = x
        x_norm = self.norm1(x)
        x_mix = self.token_mixer(x_norm)
        x_mix = x_mix * self.layer_scale1.view(1, -1, 1, 1)
        x = shortcut + self.drop_path(x_mix)
        
        # Part 2: MLP
        shortcut = x
        x_norm = self.norm2(x)
        x_norm_nhwc = x_norm.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x_mlp = self.mlp(x_norm_nhwc)
        x_mlp = x_mlp.permute(0, 3, 1, 2)        # [B, H, W, C] -> [B, C, H, W]
        x_mlp = x_mlp * self.layer_scale2.view(1, -1, 1, 1)
        x = shortcut + self.drop_path(x_mlp)
        
        return x