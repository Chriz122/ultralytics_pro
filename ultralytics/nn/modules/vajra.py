import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import einops

from ultralytics.utils.torch_utils import fuse_conv_and_bn
from .conv import Conv, DWConv, RepConv
from .block import Bottleneck, RepBottleneck, RepConvN, RepNBottleneck, RepNCSP, SPPF, SPPELAN

import logging
logger = logging.getLogger(__name__)

USE_FLASH_ATTN = False
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func
        USE_FLASH_ATTN = True
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")
        
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")

 
class BottleneckV3(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, k: tuple[int, int, int] = (3, 3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.dw_cv = Conv(c_, c_, k[1], 1, g=c_)
        self.cv2 = Conv(c_, c2, k[2], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        cv1 = self.cv1(x)
        dw_cv1 = self.dw_cv(cv1)
        cv2 = self.cv2(dw_cv1)
        return x + self.cv2(self.dw_cv(cv1)) if self.add else self.cv2(self.dw_cv(cv1))
    
    
class AreaAttention(nn.Module):

    def __init__(self, dim, num_heads=8, kernel_size=3, area=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.area = area
        self.scale = self.head_dim ** -0.5
        self.qk = Conv(dim, 2 * dim, 1, act=False)
        self.v = Conv(dim, dim, 1, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.positional_encoding = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        #qkv = self.qkv(x).flatten(2).transpose(1, 2)

        #if self.area > 1:
            #qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            #B, N, _ = qkv.shape
        
        #q, k, v = qkv.view(B, N, self.num_heads, self.head_dim * 3).split([self.head_dim, self.head_dim, self.head_dim], dim=-1)

        if x.is_cuda and USE_FLASH_ATTN:
            qk = self.qk(x).flatten(2).transpose(1, 2)
            v = self.v(x)
            positional_encoding = self.positional_encoding(v)
            v = v.flatten(2).transpose(1, 2)

            if self.area > 1:
                qk = qk.reshape(B * self.area, N // self.area, C*2)
                v = v.reshape(B * self.area, N // self.area, C)
                B, N, _ = qk.shape
            q, k = qk.split([C, C], dim=2)
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            out = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)

            if self.area > 1:
                out = out.reshape(B // self.area, N * self.area, C)
                B, N, _ = out.shape
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        #elif x.is_cuda and not USE_FLASH_ATTN:
            #out = sdpa(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), attn_mask=None, dropout_p=0.0, is_causal=False)
            #out = out.permute(0, 2, 1, 3)
        else:
            qk = self.qk(x).flatten(2)
            v = self.v(x)
            positional_encoding = self.positional_encoding(v)
            v = v.flatten(2)

            if self.area > 1:
                qk = qk.reshape(B * self.area, C*2, N // self.area)
                v = v.reshape(B * self.area, C, N // self.area)
                B, _, N = qk.shape
            
            q, k = qk.split([C, C], dim=1)
            q = q.view(B, self.num_heads, self.head_dim, N)
            k = k.view(B, self.num_heads, self.head_dim, N)
            v = v.view(B, self.num_heads, self.head_dim, N)
            attn = (q.transpose(-2, -1) @ k) * self.scale
            attn = F.softmax(attn, dim=-1)
            #max_attn = attn.max(dim=-1, keepdim=True).values
            #exp_attn = torch.exp(attn - max_attn)
            #attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            out = (v @ attn.transpose(-2, -1))

            if self.area > 1:
                out = out.reshape(B // self.area, C, N * self.area)
                B, _, N = out.shape
            out = out.reshape(B, C, H, W)
            #q = q.permute(0, 2, 3, 1)
            #k = k.permute(0, 2, 3, 1)
            #v = v.permute(0, 2, 3, 1)
            #attn = (q.transpose(-2, -1) @ k) * self.scale
            #attn = F.softmax(attn, dim=-1)
            #out = (v @ attn.transpose(-2, -1))
            #out = out.permute(0, 3, 1, 2) # (B, N, self.num_heads, self.head_dim)
            #v = v.permute(0, 3, 1, 2)

        #if self.area > 1:
            #out = out.reshape(B // self.area, N * self.area, C)
            #v = v.reshape(B // self.area, N * self.area, C)
            #B, N, _ = out.shape
        
        #out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        #v = v.reshape(B, H, W, C).permute(0, 3, 1, 2)
        #out = out + self.positional_encoding(v)
        #out = self.proj(out)
        return self.proj(out + positional_encoding)
    
    
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        square = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(square + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]
    
    
class SwiGLUFFN(nn.Module):
    def __init__(self, in_c, out_c, mlp_ratio=4.0):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        mlp_hidden_dim = int(in_c * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.norm = LayerNorm2d(in_c)
        self.fc1 = nn.Linear(in_c, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim // 2, out_c)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        a, b = self.fc1(x).split(self.mlp_hidden_dim // 2, dim=2)
        hidden = F.silu(a) * b
        out = self.fc2(hidden)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out
    

class AreaAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c, num_heads=8, kernel_size=7, shortcut=True, area=4, mlp_ratio=1.2, swiglu_ffn=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.attn = AreaAttention(in_c, num_heads=num_heads, kernel_size=kernel_size, area=area)
        mlp_dim = int(mlp_ratio * in_c)
        self.conv2 = nn.Sequential(Conv(in_c, mlp_dim, 1, 1), Conv(mlp_dim, out_c, 1, act=False)) if not swiglu_ffn else SwiGLUFFN(in_c, out_c, mlp_ratio)
        self.add = shortcut and in_c == out_c
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.conv2(x) if self.add else self.conv2(x)
        return x


class VajraV2InnerBlock(nn.Module):
    def __init__(self, in_c, out_c, use_attn = False, use_bottleneck_v3=False, shortcut=False, grid_size=1, num_bottleneck_blocks=2, bottleneck_expansion=1.0, rep_bottleneck=True) -> None:
        super().__init__()
        #block = AreaAttentionBlock if use_attn else RepBottleneck
        if use_attn:
            block = AreaAttentionBlock
        else:
            if rep_bottleneck:
                block = RepBottleneck
            else:
                block = Bottleneck
        hidden_c = int(out_c * 0.5)
        self.in_c = in_c
        self.out_c = out_c
        self.shortcut = shortcut
        self.use_attn = use_attn
        self.use_bottleneck_v3 = use_bottleneck_v3
        self.hidden_c = hidden_c
        self.conv1 = Conv(in_c, hidden_c, 1, 1)
        self.conv2 = Conv(in_c, hidden_c, 1, 1)
        if use_attn:
            self.gamma = nn.Parameter(0.01 * torch.ones(out_c), requires_grad=True)
            self.bottleneck_blocks = nn.Sequential(*([block(hidden_c, hidden_c, hidden_c // 32, 7, True, area=grid_size)] * 2))
        elif use_bottleneck_v3:
            self.bottleneck_blocks = nn.Sequential(*([BottleneckV3(hidden_c, hidden_c, True, (5, 5, 5), 1.0), BottleneckV3(hidden_c, hidden_c, True, (5, 5, 5), 1.0)]))
        else:
            self.bottleneck_blocks = nn.Sequential(*(block(hidden_c, hidden_c, shortcut=shortcut, expansion_ratio=bottleneck_expansion) for _ in range(num_bottleneck_blocks)))
        self.conv3 = Conv(2 * hidden_c, out_c, kernel_size=1, stride=1)

    def forward(self, x):
        a = self.conv1(x)
        b = self.bottleneck_blocks(a)
        out = self.conv3(torch.cat((b, self.conv2(x)), 1))
        return x + self.gamma.view(-1, len(self.gamma), 1, 1) * out if self.use_attn else out


class VajraV1MerudandaX(nn.Module):
    def __init__(self, in_c, out_c, mid_c1, mid_c2, num_blocks=1, rep_csp=False):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.mid_c1 = mid_c1
        self.mid_c2 = mid_c2
        self.num_blocks = num_blocks
        self.hidden_c = mid_c1 // 2
        self.conv1 = Conv(in_c, mid_c1, 1, 1)
        self.conv2 = nn.Sequential(VajraV2InnerBlock(mid_c1//2, mid_c2, num_bottleneck_blocks=num_blocks, rep_bottleneck=False, shortcut=True), Conv(mid_c2, mid_c2, 3, 1)) if not rep_csp else nn.Sequential(RepNCSP(mid_c1 // 2, mid_c2, num_blocks), Conv(mid_c2, mid_c2, 3, 1))
        self.conv3 = nn.Sequential(VajraV2InnerBlock(mid_c2, mid_c2, num_bottleneck_blocks=num_blocks, rep_bottleneck=False, shortcut=True), Conv(mid_c2, mid_c2, 3, 1)) if not rep_csp else nn.Sequential(RepNCSP(mid_c2, mid_c2, num_blocks), Conv(mid_c2, mid_c2, 3, 1))
        self.conv4 = Conv(mid_c1 + (2 * mid_c2), out_c, 1, 1)

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(conv(y[-1]) for conv in [self.conv2, self.conv3])
        return self.conv4(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = list(self.conv1(x).split((self.hidden_c, self.hidden_c), 1))
        y.extend(conv(y[-1]) for conv in [self.conv2, self.conv3])
        return self.conv4(torch.cat(y, 1))

    
class AttentionV2(nn.Module):
    def __init__(self, dim, num_heads=8, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * 0.5)
        self.scale = self.head_dim ** -0.5

        self.qk = Conv(dim, 2 * dim, 1, act=False)
        self.v = Conv(dim, dim, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.positional_encoding = Conv(dim, dim, kernel_size, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qk = self.qk(x).view(B, self.num_heads, self.head_dim * 2, N)
        v = self.v(x)
        positional_encoding = self.positional_encoding(v)
        v = v.view(B, self.num_heads, self.head_dim, N)
        q, k = qk.split([self.head_dim, self.head_dim], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.permute(0, 1, 3, 2)
            k = k.permute(0, 1, 3, 2)
            v = v.permute(0, 1, 3, 2)
            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            attn = (q.transpose(-2, -1) @ k) * self.scale
            attn = F.softmax(attn, dim=-1)
            x = (v @ attn.transpose(-2, -1)).view(B, C, H, W)

        return self.proj(x + positional_encoding)
    
    
class AttentionBlockV2(nn.Module):
    def __init__(self, in_c, out_c, num_heads=8, kernel_size=7, shortcut=True, mlp_ratio=2., swiglu_ffn=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.attn = AttentionV2(in_c, num_heads=num_heads, kernel_size=kernel_size)
        mlp_hidden_dim = int(mlp_ratio * in_c)
        self.conv2 = nn.Sequential(Conv(in_c, mlp_hidden_dim, 1, 1), Conv(mlp_hidden_dim, out_c, 1, act=False)) if not swiglu_ffn else SwiGLUFFN(in_c, out_c, mlp_ratio)
        self.add = shortcut and in_c == out_c
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.conv2(x) if self.add else self.conv2(x)
        return x
    
    
class VajraV1AttentionBhag6(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, expansion_ratio=0.5, lite=False, mlp_ratio=2.0, elan=False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lite = lite
        self.num_blocks = num_blocks
        self.expansion_ratio = expansion_ratio
        self.hidden_c = int(expansion_ratio * out_c)
        self.out_c = out_c
        self.lite = lite
        self.mlp_ratio = mlp_ratio
        self.sppf = SPPF(self.in_c, self.in_c, 5) if not elan else SPPELAN(self.in_c, self.in_c, self.hidden_c, 5) #VajraSPPModule(self.in_c, self.in_c, 5)
        self.conv1 = Conv(self.in_c, 2 * self.hidden_c, 1, 1)
        self.attn = nn.Sequential(*(AttentionBlockV2(self.hidden_c, self.hidden_c, num_heads=self.hidden_c // 64 if not lite else self.hidden_c // 8, kernel_size=3, mlp_ratio=mlp_ratio) for _ in range(num_blocks)))
        self.conv2 = Conv(2 * self.hidden_c, out_c, 1, 1)

    def forward(self, x):
        x = self.sppf(x)
        fm1, fm2 = self.conv1(x).chunk(2, 1)
        fm3 = self.attn(fm2)
        return self.conv2(torch.cat((fm3, fm1), 1))
    
    def forward_split(self, x):
        x = self.sppf(x)
        fm1, fm2 = self.conv1(x).split(self.hidden_c, 1)
        fm3 = self.attn(fm2)
        return self.conv2(torch.cat((fm3, fm1), 1))
    
    
class RepVGGDW(nn.Module):
    def __init__(self, dim, kernel_size=7, stride=1) -> None:
        super().__init__()
        self.kernel_size=kernel_size
        self.stride = stride
        self.conv = DWConv(dim, dim, kernel_size, stride, act=False)
        self.conv1 = DWConv(dim, dim, 3, stride, act=False)
        self.dim = dim
        self.act = nn.SiLU()
        self.fused_conv = None  # Placeholder for fused layer

    def forward(self, x):
        if self.fused_conv is not None:
            return self.act(self.fused_conv(x))
        return self.act(self.conv(x) + self.conv1(x))
    
    def _calculate_padding(self, kernel, target_size):
        """Calculate padding required to resize the kernel to the target size."""
        current_size = kernel.shape[2]
        pad_total = target_size - current_size
        pad = pad_total // 2
        return [pad, pad + (pad_total % 2), pad, pad + (pad_total % 2)]
    
    @torch.no_grad()
    def fuse(self):
        # Fuse conv and batchnorm layers for self.conv and self.conv1
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        # Dynamically adjust padding for any kernel size
        if conv_w.shape[2:] != conv1_w.shape[2:]:
            larger_size = max(conv_w.shape[2], conv1_w.shape[2])
            conv_w = torch.nn.functional.pad(conv_w, self._calculate_padding(conv_w, larger_size))
            conv1_w = torch.nn.functional.pad(conv1_w, self._calculate_padding(conv1_w, larger_size))

        # Combine the two weights and biases
        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        # Create a new fused_conv layer with combined weights and biases
        device = next(self.parameters()).device
        self.fused_conv = nn.Conv2d(
            in_channels=self.conv.conv.in_channels,
            out_channels=self.conv.conv.out_channels,
            kernel_size=self.conv.conv.kernel_size,
            stride=self.conv.conv.stride,
            padding=self.conv.conv.padding,
            groups=self.conv.conv.groups,
            bias=True
        ).to(device)
        
        self.fused_conv.weight.data.copy_(final_conv_w.to(device))
        self.fused_conv.bias.data.copy_(final_conv_b.to(device))

    def forward_fuse(self, x):
        return self.act(self.fused_conv(x)) if self.fused_conv is not None else self.act(self.conv(x) + self.conv1(x))
    
    
class MerudandaDW(nn.Module):
    def __init__(self, in_c, out_c, shortcut=True, expansion_ratio=0.5, use_rep_vgg_dw=False, kernel_size=5, stride=1):
        super().__init__()
        hidden_c = int(out_c * expansion_ratio)
        self.in_c = in_c
        self.out_c = out_c
        self.shortcut=shortcut
        self.expansion_ratio = expansion_ratio
        self.use_rep_vgg_dw = use_rep_vgg_dw
        self.kernel_size = kernel_size
        self.stride = stride
        self.add = shortcut and in_c == out_c and stride == 1
        self.block = nn.Sequential(
            DWConv(in_c, in_c, 3),
            Conv(in_c, 2 * hidden_c, 1, 1),
            DWConv(2 * hidden_c, 2 * hidden_c, kernel_size, stride) if not use_rep_vgg_dw else RepVGGDW(2 * hidden_c, kernel_size, stride),
            Conv(2 * hidden_c, out_c, 1, 1),
            DWConv(out_c, out_c, 3, 1),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if self.add else y
    
    
class Squeeze_Excite_Layer(nn.Module):
    def __init__(self, channels, reduction=4):
        super(Squeeze_Excite_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        y = self.avg_pool(x)  # [batch_size, channels, 1, 1]
        y = self.fc(y)       # [batch_size, channels, 1, 1]
        return x * y         # Broadcasting: multiply across spatial dimensions
    
    
class VajraRepViTBlock(nn.Module):
    def __init__(self, in_c, out_c, use_se=False, stride=1, use_rep_vgg_dw = False, kernel_size=7, mlp_ratio=2.0, swiglu_ffn = True):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.use_se = use_se
        hidden_c = int(mlp_ratio * in_c)
        self.add = stride == 2 or in_c == out_c

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv(in_c, in_c, stride=2, kernel_size=3, groups=in_c, act=False),
                Squeeze_Excite_Layer(in_c, reduction=4) if use_se else nn.Identity(),
                Conv(in_c, out_c, 1, 1, act=False)
            )
            self.channel_mixer = nn.Sequential(Conv(out_c, 2 * out_c, 1, 1), Conv(2 * out_c, out_c, 1, 1, act=False))

        else:
            self.token_mixer = nn.Sequential(
                MerudandaDW(in_c, in_c, True, 0.5, use_rep_vgg_dw, kernel_size=kernel_size), #RepVGGDW(dim=in_c, kernel_size=7, stride=1),
                Squeeze_Excite_Layer(in_c, reduction=4) if use_se else nn.Identity(),
            )
            self.channel_mixer = SwiGLUFFN(in_c, out_c, 2.0) if swiglu_ffn else nn.Sequential(Conv(in_c, hidden_c, 1, 1), Conv(hidden_c, out_c, 1, 1, act=False))
        
        self.apply(self._init_weights)

    def forward(self, x):
        token_mixer_out = self.token_mixer(x)
        out = token_mixer_out + self.channel_mixer(token_mixer_out) if self.add else self.channel_mixer(token_mixer_out)
        return out
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    
class VajraV1MerudandaBhag15(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, shortcut=False, expansion_ratio = 0.5, use_mlp=False, kernel_size=7, use_repvgg_dw = False) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.expansion_ratio = expansion_ratio
        hidden_c = int(out_c * expansion_ratio)
        self.hidden_c = hidden_c
        self.conv1 = Conv(in_c, 2 * hidden_c, 1, 1)
        if not use_mlp:
            self.bottleneck_blocks = nn.ModuleList(
                nn.Sequential(MerudandaDW(hidden_c, hidden_c, True, 1.0, use_rep_vgg_dw=use_repvgg_dw, kernel_size=kernel_size, stride=1)) for i in range(num_blocks)
            )
        else:
            self.bottleneck_blocks = nn.ModuleList(
                VajraRepViTBlock(hidden_c, hidden_c, use_se = False, stride=1, use_rep_vgg_dw=use_repvgg_dw, kernel_size=kernel_size, swiglu_ffn=False, mlp_ratio=2.0) for i in range(num_blocks)
            )
        self.conv2 = Conv((num_blocks + 2) * hidden_c, out_c, 1, 1)
    
    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        y = [a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        a, b = self.conv1(x).split(self.hidden_c, 1)
        y = [a, b]
        y.extend(block(y[-1]) for block in self.bottleneck_blocks)
        return self.conv2(torch.cat(y, 1))
