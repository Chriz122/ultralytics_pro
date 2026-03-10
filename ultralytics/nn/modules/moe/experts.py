# 🐧Please note that this file has been modified by Tencent on 2026/01/16. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Expert modules for Mixture-of-Experts models"""
import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath
from .utils import FlopsUtils, get_safe_groups


# ==========================================
# Optimized expert modules
# ==========================================
class OptimizedSimpleExpert(nn.Module):
    """Use GroupNorm instead of BatchNorm to improve stability for small batches."""

    def __init__(self, in_channels, out_channels, expand_ratio=2, num_groups=8):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(get_safe_groups(hidden_dim, num_groups), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels)
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.conv(x)

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.conv[0], (1, C, H, W))
        flops += FlopsUtils.count_conv2d(self.conv[3], (1, self.hidden_dim, H, W))
        return flops


class FusedGhostExpert(nn.Module):
    """Fused Ghost expert that reduces memory traffic by combining operations."""

    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, num_groups=8):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # Use GroupNorm to improve stability
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(min(num_groups, init_channels), init_channels),
            nn.SiLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=init_channels, bias=False),
            nn.GroupNorm(min(num_groups, new_channels), new_channels),
            nn.SiLU(inplace=True)
        )
        self.init_channels = init_channels

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.primary_conv[0], (1, C, H, W))
        flops += FlopsUtils.count_conv2d(self.cheap_operation[0], (1, self.init_channels, H, W))
        return flops


class SimpleExpert(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x): return self.conv(x)

    def compute_flops(self, input_shape): return FlopsUtils.count_conv2d(self.conv, input_shape)


class SpatialExpert(nn.Module):
    """Expert network with 3x3 spatial convolution, enabling experts to learn spatial patterns."""
    def __init__(self, in_ch, out_ch, expand_ratio=2):
        super().__init__()
        hid = int(in_ch * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hid, 1, bias=False),
            nn.BatchNorm2d(hid),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid, hid, 3, padding=1, groups=hid, bias=False),  # DW spatial conv
            nn.BatchNorm2d(hid),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.conv(x)

    def compute_flops(self, input_shape):
        return FlopsUtils.count_conv2d(self.conv, input_shape)


class GhostExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.out_channels, :, :]

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.primary_conv, input_shape)
        # Compute input shape to cheap op (output of primary conv)
        p_out = self.primary_conv[0].out_channels
        flops += FlopsUtils.count_conv2d(self.cheap_operation, (B, p_out, H, W))
        return flops


class InvertedResidualExpert(nn.Module):
    """
    Highly efficient expert module: Uses Inverted Residual structure (MobileNetV2 style).
    2-3x faster than standard convolution experts, fewer parameters, stronger non-linearity.
    """
    def __init__(self, in_channels, out_channels, expand_ratio=2, kernel_size=3):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            # 1. Pointwise Expand
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            # 2. Depthwise Spatial
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2, 
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            # 3. Pointwise Project
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

    def compute_flops(self, input_shape):
        return FlopsUtils.count_conv2d(self.conv, input_shape)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class EfficientExpertGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(EfficientExpertGroup, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if not hasattr(self, "conv"):
            out_c = x.shape[1]
            self.conv = DepthwiseSeparableConv(x.shape[1], out_c, 3, 1)
        return self.conv(x)
    
    
# -----------------------------------------------------------------------------
# Helper Functions & Basic Layers
# -----------------------------------------------------------------------------

def make_divisible(v, divisor=8, min_value=None):
    """
    Ensure that all layers have a channel number that is divisible by 8
    (It is common for MobileNet optimization).
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvNormAct(nn.Module):
    """
    Standard block: Conv2d -> BatchNorm2d -> SiLU (or Identity)
    """
    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, 
                 padding=None, groups=1, bias=False, apply_act=True):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride, padding, 
            groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_chs)
        self.act = nn.SiLU(inplace=True) if apply_act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer
    """
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channel = make_divisible(channel // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, reduced_channel, 1, bias=True),
            nn.SiLU(inplace=True), # Using SiLU as requested
            nn.Conv2d(reduced_channel, channel, 1, bias=True),
            nn.Hardsigmoid() # Standard for MobileNets, or can be Sigmoid
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

# -----------------------------------------------------------------------------
# Universal Inverted Residual Expert
# -----------------------------------------------------------------------------

class UniversalInvertedResidualExpert(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 dw_kernel_size_start=0, 
                 dw_kernel_size_mid=3, 
                 dw_kernel_size_end=0, 
                 stride=1, 
                 expand_ratio=2, 
                 use_se=True):
        """
        Universal Inverted Bottleneck (UIB) Block for MobileNetV4.
        Combines flexible Depthwise Convolutions (Start, Mid, End).
        """
        super().__init__()
        
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_res_connect = (self.stride == 1 and in_channels == out_channels)
        
        # Calculate hidden dimension (expansion)
        hidden_dim = make_divisible(in_channels * expand_ratio)

        # ---------------------------------------------------------
        # 1. Start Depthwise Conv (Optional)
        # ---------------------------------------------------------
        self.dw_start = nn.Identity()
        if dw_kernel_size_start > 0:
            # If mid DW exists, start DW usually has stride 1.
            # If mid DW is missing, start DW handles the stride.
            dw_start_stride = stride if dw_kernel_size_mid == 0 else 1
            self.dw_start = ConvNormAct(
                in_channels, in_channels, dw_kernel_size_start, 
                stride=dw_start_stride, groups=in_channels, apply_act=False
            )

        # ---------------------------------------------------------
        # 2. Pointwise Expansion
        # ---------------------------------------------------------
        # If expand_ratio is 1, strictly speaking we might skip this, 
        # but UIB usually keeps it for 1x1 interaction unless optimized out.
        self.pw_exp = ConvNormAct(
            in_channels, hidden_dim, kernel_size=1, apply_act=True
        )

        # ---------------------------------------------------------
        # 3. Mid Depthwise Conv (Standard MBConv part)
        # ---------------------------------------------------------
        self.dw_mid = nn.Identity()
        if dw_kernel_size_mid > 0:
            self.dw_mid = ConvNormAct(
                hidden_dim, hidden_dim, dw_kernel_size_mid, 
                stride=stride, groups=hidden_dim, apply_act=True
            )

        # ---------------------------------------------------------
        # 4. Squeeze-and-Excitation (Optional)
        # ---------------------------------------------------------
        self.se = SELayer(hidden_dim) if use_se else nn.Identity()

        # ---------------------------------------------------------
        # 5. Pointwise Projection (Linear)
        # ---------------------------------------------------------
        self.pw_proj = ConvNormAct(
            hidden_dim, out_channels, kernel_size=1, apply_act=False
        )

        # ---------------------------------------------------------
        # 6. End Depthwise Conv (Optional)
        # ---------------------------------------------------------
        self.dw_end = nn.Identity()
        if dw_kernel_size_end > 0:
            self.dw_end = ConvNormAct(
                out_channels, out_channels, dw_kernel_size_end, 
                stride=1, groups=out_channels, apply_act=False
            )

    def forward(self, x):
        identity = x
        
        # UIB Pipeline
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.dw_end(x)
        
        if self.use_res_connect:
            return x + identity
        else:
            return x

    def compute_flops(self, input_shape):
        """
        Calculates approximate FLOPs for this block.
        Mimics the interface of Code 1's FlopsUtils.
        input_shape: (C, H, W) or (B, C, H, W)
        """
        if len(input_shape) == 4:
            c, h, w = input_shape[1], input_shape[2], input_shape[3]
        else:
            c, h, w = input_shape
            
        flops = 0.0
        
        # Helper to calc conv flops: H * W * Cin * Cout * K * K / groups
        def get_conv_flops(module, in_c, in_h, in_w):
            if isinstance(module, nn.Identity):
                return 0.0, in_h, in_w
            
            conv = module.conv
            k = conv.kernel_size[0]
            g = conv.groups
            out_c = conv.out_channels
            s = conv.stride[0]
            
            out_h = in_h // s
            out_w = in_w // s
            
            # FLOPs = Output_H * Output_W * Weights
            # Weights = Cin * Cout * K * K / Groups
            layer_flops = (out_h * out_w) * (in_c * out_c * k * k) // g
            return layer_flops, out_h, out_w

        # Track shapes and accumulate FLOPs
        current_c = self.in_channels
        current_h = h
        current_w = w
        
        # 1. DW Start
        f, current_h, current_w = get_conv_flops(self.dw_start, current_c, current_h, current_w)
        flops += f
        
        # 2. PW Exp
        f, current_h, current_w = get_conv_flops(self.pw_exp, current_c, current_h, current_w)
        flops += f
        current_c = self.pw_exp.conv.out_channels
        
        # 3. DW Mid
        f, current_h, current_w = get_conv_flops(self.dw_mid, current_c, current_h, current_w)
        flops += f
        
        # 4. SE (Approximate: GAP + FC1 + FC2)
        if not isinstance(self.se, nn.Identity):
            # GAP: H*W*C
            flops += current_h * current_w * current_c
            # FCs (simplified 1x1 convs on 1x1 spatial)
            se_mid_c = self.se.fc[0].out_channels
            flops += current_c * se_mid_c # FC1
            flops += se_mid_c * current_c # FC2
            # Scale
            flops += current_h * current_w * current_c

        # 5. PW Proj
        f, current_h, current_w = get_conv_flops(self.pw_proj, current_c, current_h, current_w)
        flops += f
        current_c = self.pw_proj.conv.out_channels

        # 6. DW End
        f, current_h, current_w = get_conv_flops(self.dw_end, current_c, current_h, current_w)
        flops += f

        return flops
    
     
def channel_shuffle(x, groups=2):
    """
    Code 3 中的 Channel Shuffle 函數
    """
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x


class ShuffleExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.stride = stride
        
        # 依據 ShuffleNetV2 邏輯，輸出通道由兩個分支拼接而成，故每個分支輸出為 out_channels // 2
        half_channels = out_channels // 2
        
        # 判斷是否需要 Downsample 結構 (參考 Code 3)
        # 如果 stride > 1 或者輸入輸出通道不相等，則不能使用 Channel Split
        self.use_downsample_branch = stride > 1 or in_channels != out_channels
        
        if self.use_downsample_branch:
            # Branch 1: 僅在 Downsample 模式下存在
            # 結構: 3x3 DW -> BN -> 1x1 PW -> BN -> SiLU
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                          padding=(kernel_size - 1) // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.SiLU(inplace=True)
            )
            
            # Branch 2 (Downsample 模式): 輸入為完整 in_channels
            # 結構: 1x1 PW -> BN -> SiLU -> 3x3 DW -> BN -> 1x1 PW -> BN -> SiLU
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(half_channels, half_channels, kernel_size, stride, 
                          padding=(kernel_size - 1) // 2, groups=half_channels, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.Conv2d(half_channels, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.SiLU(inplace=True)
            )
        else:
            # Basic 模式: 輸入輸出通道必須相等
            assert in_channels == out_channels
            
            # Channel Split 後，Branch 2 的輸入通道減半
            c_in = in_channels // 2
            
            # Branch 2 (Basic 模式)
            # 結構: 1x1 PW -> BN -> SiLU -> 3x3 DW -> BN -> 1x1 PW -> BN -> SiLU
            self.branch2 = nn.Sequential(
                nn.Conv2d(c_in, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(half_channels, half_channels, kernel_size, 1, 
                          padding=(kernel_size - 1) // 2, groups=half_channels, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.Conv2d(half_channels, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.SiLU(inplace=True)
            )

    def forward(self, x):
        if self.use_downsample_branch:
            # 對應 Code 3: out = cat(branch1(x), branch2(x))
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            # 對應 Code 3: Channel Split
            c = x.shape[1] // 2
            x1 = x[:, :c, :, :]
            x2 = x[:, c:, :, :]
            # out = cat(x1, branch2(x2))
            out = torch.cat((x1, self.branch2(x2)), 1)
            
        return channel_shuffle(out, 2)

    def compute_flops(self, input_shape):
        # 為了計算 FLOPs，我們需要將兩個分支的 FLOPs 加總
        # 這裡建立一個臨時的 Sequential 來模擬完整的計算圖供 FlopsUtils 使用
        if self.use_downsample_branch:
            # 近似計算：建立一個包含 branch1 和 branch2 的列表
            # 注意：這裡僅是為了調用 FlopsUtils，實際執行不會走這裡
            dummy_layer = nn.Sequential(self.branch1, self.branch2)
            return FlopsUtils.count_conv2d(dummy_layer, input_shape)
        else:
            # Basic 模式下，只有一半的通道通過 branch2
            # 我們需要調整 input_shape 的通道數來計算 branch2 的 FLOPs
            c, h, w = input_shape
            half_input_shape = (c // 2, h, w)
            return FlopsUtils.count_conv2d(self.branch2, half_input_shape)
        
        
class Partial_conv3(nn.Module):
    """
    FasterNet 中的 PConv (Partial Convolution)
    """
    def __init__(self, dim, n_div, forward='split_cat'):
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
        # only for inference
        x = x.clone()   
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x
    
    
class FasterExpert(nn.Module):
    """
    結合 SimpleExpert 接口與 FasterBlock 邏輯的 Expert 模塊
    ** 已移除殘差邏輯 (No Residual) **
    """
    def __init__(self, in_channels, out_channels, expand_ratio=2, n_div=4, drop_path=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1. 通道對齊 (Adapter)
        if in_channels != out_channels:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.adapter = nn.Identity()

        # 2. 空間混合 (Spatial Mixing) - PConv
        self.spatial_mixing = Partial_conv3(out_channels, n_div, forward='split_cat')

        # 3. 通道混合 (Channel Mixing) - MLP
        hidden_dim = int(out_channels * expand_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        )

        # 注意：移除殘差連接後，DropPath (Stochastic Depth) 必須移除或設為 Identity。
        # 因為 DropPath 會隨機將輸出歸零，沒有殘差相加會導致訊號丟失。
        self.drop_path = nn.Identity()

    def forward(self, x):
        # 1. 適配輸入通道
        x = self.adapter(x)
        
        # 2. 空間混合 (PConv)
        x = self.spatial_mixing(x)
        
        # 3. 通道混合 (MLP)
        # 修改：移除 shortcut 與 drop_path，直接順序執行
        x = self.mlp(x)
        
        return x

    def compute_flops(self, input_shape):
        """
        計算 FLOPs
        """
        if self.in_channels != self.out_channels:
            flops = FlopsUtils.count_conv2d(self.adapter, input_shape)
            c, h, w = input_shape
            mid_shape = (self.out_channels, h, w)
        else:
            flops = 0
            mid_shape = input_shape

        # PConv FLOPs
        flops += FlopsUtils.count_conv2d(self.spatial_mixing.partial_conv3, 
                                         (self.spatial_mixing.dim_conv3, mid_shape[1], mid_shape[2]))

        # MLP FLOPs
        flops += FlopsUtils.count_conv2d(self.mlp, mid_shape)

        return flops
    

# class FasterExpert(nn.Module):
#     """
#     結合 SimpleExpert 接口與 FasterBlock 邏輯的 Expert 模塊
#     激活函數已替換為 SiLU
#     """
#     def __init__(self, in_channels, out_channels, expand_ratio=2, n_div=4, drop_path=0.0):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
        
#         # 1. 通道對齊 (Adapter)
#         # 如果輸入輸出通道不同，先進行投影 (參考 FasterBlock 的 firstConv)
#         if in_channels != out_channels:
#             self.adapter = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#         else:
#             self.adapter = nn.Identity()

#         # 2. 空間混合 (Spatial Mixing) - PConv
#         self.spatial_mixing = Partial_conv3(out_channels, n_div, forward='split_cat')

#         # 3. 通道混合 (Channel Mixing) - MLP
#         # 依據 FasterBlock 邏輯：Conv1x1 -> BN -> Act -> Conv1x1
#         # 但將 ReLU 替換為 SiLU
#         hidden_dim = int(out_channels * expand_ratio)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(out_channels, hidden_dim, 1, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.SiLU(inplace=True),  # 替換為 SiLU
#             nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
#             # FasterNet 原文中 MLP 最後一層通常沒有 BN 和 Act，此處保持一致
#         )

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         # 適配輸入通道
#         x = self.adapter(x)
        
#         # FasterBlock 殘差邏輯: 
#         # Shortcut 是經過 adapter 後的 x
#         shortcut = x
        
#         # 空間混合 (PConv)
#         x = self.spatial_mixing(x)
        
#         # 通道混合 (MLP) + 殘差連接
#         x = shortcut + self.drop_path(self.mlp(x))
        
#         return x

#     def compute_flops(self, input_shape):
#         """
#         計算 FLOPs，需依賴 FlopsUtils。
#         這裡將各部分的 FLOPs 累加。
#         """
#         # 預先計算 adapter 後的 shape
#         if self.in_channels != self.out_channels:
#             flops = FlopsUtils.count_conv2d(self.adapter, input_shape)
#             # 更新 shape 為 out_channels
#             c, h, w = input_shape
#             mid_shape = (self.out_channels, h, w)
#         else:
#             flops = 0
#             mid_shape = input_shape

#         # PConv FLOPs (Partial_conv3 內部包含一個卷積)
#         # 這裡假設 FlopsUtils 可以計算 nn.Conv2d
#         flops += FlopsUtils.count_conv2d(self.spatial_mixing.partial_conv3, 
#                                          (self.spatial_mixing.dim_conv3, mid_shape[1], mid_shape[2]))

#         # MLP FLOPs
#         flops += FlopsUtils.count_conv2d(self.mlp, mid_shape)

#         return flops


class BottConv(nn.Module):
    """
    基於 SCSegamba 的 BottConv 結構
    Pointwise -> Depthwise -> Pointwise
    """
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x
    
    
class GBCExpert(nn.Module):
    """
    已移除殘差路徑 (No Residual) 的 GBCExpert
    """
    def __init__(self, in_channels, out_channels, expand_ratio=2):
        super().__init__()
        
        # 中間層通道數邏輯
        mid_channels = max(8, in_channels // 8)

        # 構建 Block 1: 3x3, in -> in
        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=max(1, in_channels // 16), num_channels=in_channels),
            nn.SiLU(inplace=True)
        )

        # 構建 Block 2: 3x3, in -> in
        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=max(1, in_channels // 16), num_channels=in_channels),
            nn.SiLU(inplace=True)
        )

        # 構建 Block 3: 1x1, in -> in (並行分支)
        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=max(1, in_channels // 16), num_channels=in_channels),
            nn.SiLU(inplace=True)
        )

        # 構建 Block 4: 1x1, in -> out (融合後輸出)
        # 注意：BottConv 輸出為 out_channels，因此 GroupNorm 的通道數必須設為 out_channels
        self.block4 = nn.Sequential(
            BottConv(in_channels, out_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=16, num_channels=out_channels), 
            nn.SiLU(inplace=True)
        )

        # [移除] Residual layer 定義已被刪除
        # self.residual_layer = ...

    def forward(self, x):
        # [移除] residual = self.residual_layer(x)

        # 深度分支 (Depth Branch): Block 1 -> Block 2
        x1 = self.block1(x)
        x1 = self.block2(x1)

        # 廣度分支 (Width Branch): Block 3
        x2 = self.block3(x)

        # 融合: 元素級乘法 (Gating 機制)
        x = x1 * x2

        # 輸出分支: Block 4
        x = self.block4(x)

        # [移除] return x + residual，改為直接返回 x
        return x

    def compute_flops(self, input_shape):
        """
        計算 FLOPs
        """
        # 確保 input_shape 是 list 或 tuple
        c, h, w = input_shape[-3:] 
        
        flops = 0
        # Block 1, 2, 3 的輸入形狀皆為 input_shape
        flops += FlopsUtils.count_conv2d(self.block1, input_shape)
        flops += FlopsUtils.count_conv2d(self.block2, input_shape)
        flops += FlopsUtils.count_conv2d(self.block3, input_shape)
        
        # Block 4 的輸入形狀也是 (C, H, W) (因為 x1*x2 不改變形狀)
        flops += FlopsUtils.count_conv2d(self.block4, input_shape)
        
        # [移除] 殘差層 FLOPs 計算已被刪除
            
        return flops
    

# class GBCExpert(nn.Module):
#     def __init__(self, in_channels, out_channels, expand_ratio=2):
#         """
#         Args:
#             in_channels (int): 輸入通道數
#             out_channels (int): 輸出通道數
#             expand_ratio (float): 用於兼容介面，但在 GBC 原始邏輯中，
#                                   中間通道數通常被固定為 input // 8。
#                                   此處我們保留原始 GBC 的 1/8 壓縮特性以維持其設計哲學。
#         """
#         super().__init__()
        
#         # 依據 GBC 原始邏輯，中間層通道數為輸入的 1/8 (防止過小設為至少 8)
#         # 如果希望利用 expand_ratio 也可以改為: int(in_channels * expand_ratio)
#         mid_channels = max(8, in_channels // 8)

#         # 構建 Block 1: 3x3, in -> in
#         self.block1 = nn.Sequential(
#             BottConv(in_channels, in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.SiLU(inplace=True)
#         )

#         # 構建 Block 2: 3x3, in -> in
#         self.block2 = nn.Sequential(
#             BottConv(in_channels, in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.SiLU(inplace=True)
#         )

#         # 構建 Block 3: 1x1, in -> in (並行分支)
#         self.block3 = nn.Sequential(
#             BottConv(in_channels, in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(in_channels),
#             nn.SiLU(inplace=True)
#         )

#         # 構建 Block 4: 1x1, in -> out (融合後輸出)
#         # 這裡是改變通道數的地方，以適配 Expert 的 out_channels
#         self.block4 = nn.Sequential(
#             BottConv(in_channels, out_channels, mid_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(out_channels),
#             nn.SiLU(inplace=True)
#         )

#         # Residual 處理: 如果輸入輸出通道不一致，需要一個投影層
#         if in_channels != out_channels:
#             self.residual_layer = nn.Conv2d(in_channels, out_channels, 1, bias=False)
#         else:
#             self.residual_layer = nn.Identity()

#     def forward(self, x):
#         # 處理殘差路徑
#         residual = self.residual_layer(x)

#         # 深度分支 (Depth Branch): Block 1 -> Block 2
#         x1 = self.block1(x)
#         x1 = self.block2(x1)

#         # 廣度分支 (Width Branch): Block 3
#         x2 = self.block3(x)

#         # 融合: 元素級乘法 (Gating 機制)
#         x = x1 * x2

#         # 輸出分支: Block 4
#         x = self.block4(x)

#         return x + residual

#     def compute_flops(self, input_shape):
#         """
#         計算 FLOPs，分別計算各個子模組的消耗並加總。
#         input_shape 格式應為 (1, C, H, W) 或 (C, H, W)
#         """
#         # 確保 input_shape 是 list 或 tuple
#         c, h, w = input_shape[-3:] 
        
#         flops = 0
#         # Block 1, 2, 3 的輸入形狀皆為 input_shape
#         flops += FlopsUtils.count_conv2d(self.block1, input_shape)
#         flops += FlopsUtils.count_conv2d(self.block2, input_shape)
#         flops += FlopsUtils.count_conv2d(self.block3, input_shape)
        
#         # Block 4 的輸入形狀也是 (C, H, W) (因為 x1*x2 不改變形狀)
#         flops += FlopsUtils.count_conv2d(self.block4, input_shape)
        
#         # 如果有殘差投影層 (Conv2d)，也需計算
#         if not isinstance(self.residual_layer, nn.Identity):
#             flops += FlopsUtils.count_conv2d(self.residual_layer, input_shape)
            
#         return flops