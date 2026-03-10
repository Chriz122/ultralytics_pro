import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple

# --------------------------------------------------------------------------
# 基礎組件 (Basic Components)
# --------------------------------------------------------------------------

class Partial_conv3(nn.Module):
    """
    FasterNet 的核心卷積組件 PConv
    """
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # 僅用於推理，保留原始輸入用於殘差連接
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # 用於訓練/推理
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class FasterNetBlock(nn.Module):
    """
    單個 FasterNet Block (對應原代碼中的 MLPBlock)
    """
    def __init__(self, dim, n_div=4, mlp_ratio=2., drop_path=0., act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, pconv_fw_type='split_cat'):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        # 定義 MLP 層
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(inplace=True),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        )

        # 空間混合層 PConv
        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)

    def forward(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x


# --------------------------------------------------------------------------
# YOLOv8 適配模塊 (YOLOv8 Compatible Modules)
# --------------------------------------------------------------------------

class PatchEmbed_Faster(nn.Module):
    """
    對應 FasterNet 的 PatchEmbed
    在 YOLO yaml 中使用: [-1, 1, PatchEmbed_Faster, [96, 4, 4]]  # [out_channels, patch_size, stride]
    """
    def __init__(self, c1, c2, patch_size=4, patch_stride=4, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(c2)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.norm(self.proj(x))
        return x


class PatchMerging_Faster(nn.Module):
    """
    對應 FasterNet 的 PatchMerging (下採樣層)
    在 YOLO yaml 中使用: [-1, 1, PatchMerging_Faster, [out_channels]]
    """
    def __init__(self, c1, c2, patch_size2=2, patch_stride2=2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # 注意: FasterNet 的 PatchMerging 通常將通道數翻倍 (c2 = 2 * c1)，但在 YOLO 中 c2 由 yaml 指定
        self.reduction = nn.Conv2d(c1, c2, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(c2)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.norm(self.reduction(x))
        return x


class FasterNetLayer(nn.Module):
    """
    FasterNet 的 Stage 層 (包含多個 FasterNetBlock)
    YAML args: [out_channels, number_of_blocks]
    例如: [-1, 1, FasterNetLayer, [96, 2]] 
    意思: 輸入自動獲取, 輸出96通道, 重複2次 Block
    """
    def __init__(self, c1, c2, n=1, mlp_ratio=2., n_div=4, drop_path=0.):
        super().__init__()
        # 為了保險，雖然 FasterNetBlock 不改變通道，但我們強制 c2 等於 c1
        # 如果 YAML 寫錯導致 c1 != c2，這裡可以加個 assert 或者用 1x1 卷積調整
        # 這裡假設 FasterNetLayer 不做通道變換 (通道變換由 PatchMerging 負責)
        if c1 != c2:
            # 如果使用者硬要變通道，這是一個防呆機制 (雖然標準 FasterNet 不這麼做)
            self.channel_adjust = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        else:
            self.channel_adjust = nn.Identity()

        self.blocks = nn.Sequential(*[
            FasterNetBlock(
                dim=c2, # 使用輸出通道數構建 Block
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                act_layer=nn.ReLU,
                norm_layer=nn.BatchNorm2d,
                pconv_fw_type='split_cat'
            ) for _ in range(n)
        ])

    def forward(self, x):
        x = self.channel_adjust(x)
        x = self.blocks(x)
        return x
