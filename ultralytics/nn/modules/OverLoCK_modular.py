import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 基礎工具 (Utils)
# ==========================================

def autopad(k, p=None, d=1):
    """
    Ultralytics 標準 SAME padding:
    對 stride=1 時輸出 H,W 與輸入相同。
    """
    if p is not None:
        return p

    if isinstance(k, int):
        return d * (k - 1) // 2
    else:
        return [d * (x - 1) // 2 for x in k]


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for Channel First (N, C, H, W)"""
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class GRN(nn.Module):
    """Global Response Normalization"""
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1)) if use_bias else None

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class SEModule(nn.Module):
    """Squeeze-and-Excitation Module"""
    def __init__(self, dim, red=8):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.proj(x)


class LayerScale(nn.Module):
    """
    LayerScale：
    learnable 深度 1x1 scaling，不改變 H,W。
    """
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])


# ==========================================
# 2. 重參數化模塊 (Dilated Reparam)
# ==========================================

def fuse_bn(conv, bn):
    """Fuses a Conv2d and a BatchNorm2d into a single Conv2d layer."""
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    fused_weight = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
    fused_bias = bn.bias + (conv_bias - bn.running_mean) * bn.weight / std
    return fused_weight, fused_bias


def convert_dilated_to_nondilated(kernel, dilate_rate, device):
    """
    把 dilation=r 的 kernel 展成 non-dilated 大 kernel。
    使用 conv_transpose2d 實現。
    """
    identity_kernel = torch.ones((1, 1, 1, 1), device=device)
    if kernel.size(1) == 1:
        return F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
    else:
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(
                kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate
            )
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def center_crop_to(tensor, target_k):
    """
    將 (out_channels, in_channels/groups, H, W) 的 kernel
    置中裁切到 target_k x target_k。
    """
    k = tensor.size(2)
    if k == target_k:
        return tensor
    assert k > target_k, "center_crop_to 只在 k > target_k 時使用"
    start = (k - target_k) // 2
    end = start + target_k
    return tensor[:, :, start:end, start:end]


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    """
    把不同 dilation 的 kernel 轉成等效大 kernel，合併到 large_kernel。
    等效 kernel 可能比 large_kernel 大，需中心裁切。
    """
    device = large_kernel.device
    large_k = large_kernel.size(2)

    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r, device)
    equivalent_k = equivalent_kernel.size(2)

    if equivalent_k > large_k:
        equivalent_kernel = center_crop_to(equivalent_kernel, large_k)
        merged_kernel = large_kernel + equivalent_kernel
    else:
        rows_to_pad = (large_k - equivalent_k) // 2
        merged_kernel = large_kernel + F.pad(
            equivalent_kernel, [rows_to_pad] * 4
        )

    return merged_kernel


def _match_size_to(tensor, target_h, target_w):
    """
    將 tensor 的空間尺寸置中對齊到 (target_h, target_w)，
    需要時會先中心裁切，再對稱 padding。
    """
    _, _, h, w = tensor.shape

    # 1) 先裁切超出的部份（中心裁切）
    if h > target_h:
        dh = h - target_h
        top = dh // 2
        tensor = tensor[..., top:top + target_h, :]
        h = target_h
    if w > target_w:
        dw = w - target_w
        left = dw // 2
        tensor = tensor[..., :, left:left + target_w]
        w = target_w

    # 2) 不足的部份用對稱 padding 補齊
    if h < target_h or w < target_w:
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))

    return tensor


class DilatedReparamBlock_modular(nn.Module):
    """
    OverLoCK 核心 block：
    - 一個大 kernel depthwise 主分支
    - 多個不同 kernel size / dilation 的 depthwise 分支
    - 訓練時多分支；switch_to_deploy 後 fuse 成單一 Conv2d

    ✔ 這一版保證：forward 後的 H,W 一定等於輸入 H,W。
    """
    def __init__(self, channels, kernel_size, deploy=False):
        super().__init__()
        self.channels = channels
        self.kernel_size = int(kernel_size)
        self.deploy = deploy

        # 主分支：大 kernel depthwise conv
        lk_padding = autopad(self.kernel_size, None, 1)
        self.lk_origin = nn.Conv2d(
            channels, channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=lk_padding,
            dilation=1,
            groups=channels,
            bias=deploy,
        )

        if not deploy:
            # 分支設定
            if self.kernel_size == 17:
                self.kernel_sizes = [5, 7, 9, 3, 3, 3]
                self.dilates = [1, 1, 2, 4, 5, 7]
            elif self.kernel_size == 15:
                self.kernel_sizes = [5, 7, 7, 3, 3, 3]
                self.dilates = [1, 1, 2, 3, 5, 7]
            elif self.kernel_size == 13:
                self.kernel_sizes = [5, 7, 7, 3, 3, 3]
                self.dilates = [1, 1, 2, 3, 4, 5]
            elif self.kernel_size == 7:
                self.kernel_sizes = [5, 3, 3, 3]
                self.dilates = [1, 1, 2, 3]
            else:
                self.kernel_sizes = [3] * 3
                self.dilates = [1, 2, 3]

            self.origin_bn = nn.BatchNorm2d(channels)

            for k, r in zip(self.kernel_sizes, self.dilates):
                k, r = int(k), int(r)
                padding = autopad(k, None, r)
                self.__setattr__(
                    f'dil_conv_k{k}_{r}',
                    nn.Conv2d(
                        channels, channels,
                        kernel_size=k,
                        stride=1,
                        padding=padding,
                        dilation=r,
                        groups=channels,
                        bias=False,
                    ),
                )
                self.__setattr__(f'dil_bn_k{k}_{r}', nn.BatchNorm2d(channels))

    def forward(self, x):
        if self.deploy:
            return self.lk_origin(x)

        # 記住輸入尺寸，最後一定要 match 回來
        H_in, W_in = x.shape[-2:]

        # 主分支
        out = self.origin_bn(self.lk_origin(x))

        # 多個 dilation 分支
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__(f'dil_conv_k{k}_{r}')
            bn = self.__getattr__(f'dil_bn_k{k}_{r}')
            branch = bn(conv(x))

            # 先對齊到目前 out 的尺寸，避免相加時 shape mismatch
            branch = _match_size_to(branch, out.shape[-2], out.shape[-1])
            out = out + branch

        # 最後強制輸出尺寸與輸入 x 相同
        if out.shape[-2:] != (H_in, W_in):
            out = _match_size_to(out, H_in, W_in)

        return out

    def switch_to_deploy(self):
        """把多分支 + BN fuse 成單一 depthwise 大 kernel Conv2d。"""
        if self.deploy:
            return

        # fuse 主分支
        origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)

        # fuse 各 dilated 分支到主 kernel
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__(f'dil_conv_k{k}_{r}')
            bn = self.__getattr__(f'dil_bn_k{k}_{r}')
            branch_k, branch_b = fuse_bn(conv, bn)

            origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
            origin_b += branch_b

        # 建立最終 deploy 版 Conv2d
        self.deploy = True
        self.lk_origin = nn.Conv2d(
            self.channels,
            self.channels,
            self.kernel_size,
            stride=1,
            padding=autopad(self.kernel_size, None, 1),
            dilation=1,
            groups=self.channels,
            bias=True,
        )
        self.lk_origin.weight.data = origin_k
        self.lk_origin.bias.data = origin_b

        # 移除訓練時才需要的模組
        self.__delattr__('origin_bn')
        for k, r in zip(self.kernel_sizes, self.dilates):
            self.__delattr__(f'dil_conv_k{k}_{r}')
            self.__delattr__(f'dil_bn_k{k}_{r}')


# ==========================================
# 3. YOLO Block Wrappers
# ==========================================

class ResDWConv(nn.Conv2d):
    """Depthwise convolution with residual connection"""
    def __init__(self, dim, kernel_size=3):
        super().__init__(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=autopad(kernel_size),
            groups=dim,
            bias=True,
        )

    def forward(self, x):
        return x + super().forward(x)


class OverLoCKStem(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c1, c2 // 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c2 // 2),
            nn.GELU(),
            nn.Conv2d(c2 // 2, c2 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c2 // 2),
            nn.GELU(),
            nn.Conv2d(c2 // 2, c2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        return self.stem(x)


class OverLoCKDownsample(nn.Module):
    """
    下採樣模塊：
    - YAML 仍用 (c1, c2) 初始化
    - 但在 forward 會檢查真實輸入 channel，必要時自動重建 Conv2d，
      避免 concat 之後 channel 變多導致 in_channels mismatch。
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.c2 = c2
        # 先用 YAML 提供的 c1 建立一個初始 conv/bn，之後如有不符會在 forward 裡重建
        self.conv = nn.Conv2d(c1, c2, 3, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        in_ch = x.shape[1]
        if in_ch != self.conv.in_channels:
            # 如果實際輸入的 channel 跟原本設計不同，動態重建一個匹配的 Conv2d + BN
            new_conv = nn.Conv2d(in_ch, self.c2, 3, 2, 1, bias=False).to(x.device)
            new_bn = nn.BatchNorm2d(self.c2).to(x.device)
            self.conv = new_conv
            self.bn = new_bn
        return self.bn(self.conv(x))


class RepConvBlock(nn.Module):
    """
    OverLoCK-style block：
    - ResDWConv 作為 local depthwise 殘差
    - LayerNorm2d + DilatedReparamBlock_modular + SE + MLP
    - LayerScale + DropPath
    """
    def __init__(self, c1, kernel_size=7, mlp_ratio=4, drop_path=0.0, deploy=False):
        super().__init__()
        dim = c1
        mlp_dim = int(dim * mlp_ratio)
        self.res_scale = True

        self.dwconv = ResDWConv(dim, kernel_size=3)

        self.proj = nn.Sequential(
            LayerNorm2d(dim),
            DilatedReparamBlock_modular(dim, kernel_size=kernel_size, deploy=deploy),
            nn.BatchNorm2d(dim),
            SEModule(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            DropPath(drop_path) if drop_path > 0. else nn.Identity(),
        )

        self.ls = LayerScale(dim, init_value=1e-6)

    def forward(self, x):
        x = self.dwconv(x)
        if self.res_scale:
            x = self.ls(x) + self.proj(x)
        else:
            x = x + self.ls(self.proj(x))
        return x

    def switch_to_deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
