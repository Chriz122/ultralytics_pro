import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# ---------------------------------------------------
# 基礎組件 (保持不變或微調)
# ---------------------------------------------------

class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)

class StripNetMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class StripAttentionBlock(nn.Module):
    def __init__(self, dim, k1, k2):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim, dim, kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)     
        self.conv_spatial2 = nn.Conv2d(dim, dim, kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):   
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)
        return x * attn

class StripAttention(nn.Module):
    def __init__(self, d_model, k1, k2):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripAttentionBlock(d_model, k1, k2)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x

# ---------------------------------------------------
# YOLO 模塊化組件 (核心修改部分)
# ---------------------------------------------------

class StripDownsample(nn.Module):
    """
    對應原始的 OverlapPatchEmbed。
    在 YAML 中使用: [c2, patch_size, stride]
    """
    def __init__(self, c1, c2, patch_size=7, stride=4):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(c1, c2, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        # 為了適配 YOLO 的 CNN 特性，這裡預設使用 BN，
        # 如果需要嚴格復現 StripNet 的 LayerNorm 行為，需要處理 permute
        self.norm = nn.BatchNorm2d(c2) 

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class StripBlock(nn.Module):
    # 參數順序非常重要！YOLO 是按位置傳參的。
    # 對應 YAML: [c2, k1, k2, mlp_ratio]
    # 注意: c1 是 YOLO 自動傳入的，不算在 YAML args 裡
    def __init__(self, c1, c2, k1=1, k2=19, mlp_ratio=4.0, drop=0., drop_path=0.):
        super().__init__()
        assert c1 == c2, f"StripBlock input channel {c1} must equal output channel {c2}"
        
        # --- 下面代碼保持不變 ---
        self.dim = c1
        self.norm1 = nn.LayerNorm(c1, eps=1e-6)
        self.attn = StripAttention(c1, k1, k2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(c1, eps=1e-6)
        
        # 這裡會用到 mlp_ratio (傳入的 8) 和 drop (預設的 0.)
        mlp_hidden_dim = int(c1 * mlp_ratio)
        self.mlp = StripNetMlp(in_features=c1, hidden_features=mlp_hidden_dim, drop=drop)
        
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((c1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((c1)), requires_grad=True)

    def forward(self, x):
        # ... (保持原本的 forward 代碼) ...
        B, C, H, W = x.shape
        x_in = x.permute(0, 2, 3, 1)
        norm1 = self.norm1(x_in).permute(0, 3, 1, 2)
        attn_out = self.attn(norm1)
        scale1 = self.layer_scale_1.view(1, -1, 1, 1)
        x = x + self.drop_path(scale1 * attn_out)
        
        x_in_2 = x.permute(0, 2, 3, 1)
        norm2 = self.norm2(x_in_2).permute(0, 3, 1, 2)
        mlp_out = self.mlp(norm2)
        scale2 = self.layer_scale_2.view(1, -1, 1, 1)
        x = x + self.drop_path(scale2 * mlp_out)
        return x