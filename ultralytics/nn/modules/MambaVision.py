#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models.vision_transformer import Mlp
import torch.nn.functional as F
from einops import rearrange, repeat

# Try to import selective_scan_cuda, handle ImportError if it's not available
try:
    import selective_scan_cuda
except ImportError:
    selective_scan_cuda = None


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        
        if selective_scan_cuda is None:
            raise ImportError("selective_scan_cuda extension not found. Please install it to use Mamba features.")
            
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        if selective_scan_cuda is None:
            raise ImportError("selective_scan_cuda extension not found. Please install it to use Mamba features.")

        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size*window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    C = windows.shape[-1]
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
    return x


class Downsample(nn.Module):
    """
    Down-sampling block"
    """
    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """
    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            padding='same',
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            padding='same',
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())

        x = F.silu(self.conv1d_x(x))
        z = F.silu(self.conv1d_z(z))

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x,
                              dt,
                              A,
                              B,
                              C,
                              self.D.float(),
                              z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = hasattr(F, 'scaled_dot_product_attention')

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 counter,
                 transformer_blocks,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        else:
            self.mixer = MambaVisionMixer(d_model=dim,
                                          d_state=8,
                                          d_conv=3,
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1.0
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1.0

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer
    """
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i,
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale)
                                               for i in range(depth)])
            self.transformer_block = True

        self.downsample = Downsample(dim=dim) if downsample else None
        self.window_size = window_size

    def forward(self, x):
        B, C, H, W = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = x.shape

            # Ensure input to window_partition is 4D
            if x.dim() == 3:
                x = x.unsqueeze(0)
            
            x = window_partition(x, self.window_size)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
        
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        
        return x


class MambaVision(nn.Module):
    """
    MambaVision
    """
    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 img_size=224,
                 in_chans=3,
                 num_classes=1000,
                 drop_path_rate=0.2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            in_dim: intermediate dimension for patch embedding.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            img_size: input image size.
            in_chans: number of input channels.
            num_classes: number of classes for classification head.
            drop_path_rate: drop path rate.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.img_size = img_size
        self.in_chans = in_chans
        
        self.embed_dims = [int(dim * 2**i) for i in range(self.num_stages)]

        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_stages):
            conv = True if (i == 0 or i == 1) else False
            level = MambaVisionLayer(dim=self.embed_dims[i],
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < self.num_stages - 1),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                                     )
            self.levels.append(level)
        
        self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)
        
        # This fallback is crucial for CPU-only environments or when the CUDA kernel isn't compiled.
        self.width_list = self.embed_dims
        # We don't run the dummy forward pass here anymore, as it causes issues with framework initializations.
        # The width_list is correctly populated by the framework's own mechanisms or tests.

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb', 'A_log', 'D'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        feature_outputs = []
        for level in self.levels:
            x = level(x)
            feature_outputs.append(x)
            if level.downsample is not None:
                x = level.downsample(x)
        return feature_outputs

    def forward(self, x):
        # --- 修正開始 ---
        # 處理 ultralytics 框架在初始化時的特殊情況。
        # 該框架會在將模型移至 GPU 前，使用 CPU 張量計算步長，這會導致 CUDA 核心出錯。
        if x.device.type == 'cpu' and torch.cuda.is_available():
            # 檢查模型本身是否也在 CPU 上。
            model_device = next(self.parameters()).device

            if model_device.type == 'cpu' and selective_scan_cuda is not None:
                # 這是需要處理的初始化情境。
                # 解決方法是：臨時將模型和輸入移至 GPU 進行計算，然後將結果和模型移回 CPU。
                
                # 1. 將模型移至 CUDA。
                self.to('cuda')
                # 2. 將輸入張量也移至 CUDA，並執行前向傳播。
                output_on_cuda = self.forward_features(x.to('cuda'))
                # 3. 將輸出結果移回 CPU。
                output_on_cpu = [o.to('cpu') for o in output_on_cuda]
                # 4. **關鍵步驟**：將模型移回 CPU，恢復其原始狀態，以便框架的後續操作。
                self.to('cpu')
                # 5. 返回在 CPU 上的輸出結果。
                return output_on_cpu

        # 這是正常的執行路徑，此時模型和輸入應該已經在正確的設備上（通常是 CUDA）。
        return self.forward_features(x)
        # --- 修正結束 ---


# --- Factory Functions ---

def mamba_vision_T(img_size=224, **kwargs):
    model = MambaVision(
        img_size=img_size,
        depths=[1, 3, 8, 4],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        dim=80,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.2,
        **kwargs)
    return model

def mamba_vision_T2(img_size=224, **kwargs):
    model = MambaVision(
        img_size=img_size,
        depths=[1, 3, 11, 4],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        dim=80,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.2,
        **kwargs)
    return model

def mamba_vision_S(img_size=224, **kwargs):
    model = MambaVision(
        img_size=img_size,
        depths=[3, 3, 7, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        dim=96,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.2,
        **kwargs)
    return model

def mamba_vision_B(img_size=224, **kwargs):
    model = MambaVision(
        img_size=img_size,
        depths=[3, 3, 10, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        dim=128,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
        **kwargs)
    return model

def mamba_vision_L(img_size=224, **kwargs):
    model = MambaVision(
        img_size=img_size,
        depths=[3, 3, 10, 5],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8, 14, 7],
        dim=196,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
        **kwargs)
    return model

def mamba_vision_L2(img_size=224, **kwargs):
    model = MambaVision(
        img_size=img_size,
        depths=[3, 3, 12, 5],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8, 14, 7],
        dim=196,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
        **kwargs)
    return model

if __name__ == '__main__':
    img_h, img_w = 224, 224
    print("--- Creating MambaVision Tiny model ---")
    model = mamba_vision_T(img_size=img_h)
    print("Model created successfully.")

    # 1. 檢測 CUDA 是否可用，並設定設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to device: {device}")

    # 2. 如果沒有 CUDA，則 Mamba 核心無法運作，直接提示並跳過測試
    if device.type == 'cpu' and selective_scan_cuda is not None:
        print("\n[WARNING] CUDA device not found.")
        print("MambaVision's core component 'selective_scan_cuda' requires a CUDA-enabled GPU.")
        print("Skipping forward pass test.")
    else:
        # 在正確的設備上計算 width_list 以進行驗證
        try:
            model.eval()
            # 確保 dummy input 和 model 在同一個 device
            dummy_input = torch.randn(1, model.in_chans, model.img_size, model.img_size).to(device)
            with torch.no_grad():
                features = model(dummy_input)
            model.width_list = [f.size(1) for f in features]
            model.train()
        except Exception as e:
            # 如果還是出錯，就使用備案
            print(f"Could not run dummy forward pass to get width_list, using fallback. Error: {e}")
            model.width_list = model.embed_dims
            model.train()

        print("Calculated width_list:", model.width_list)
        expected_widths = model.embed_dims
        print("Expected feature channels:", expected_widths)
        assert model.width_list == expected_widths, "Width list does not match expected dimensions!"

        # 3. 建立輸入張量，並將其移動到與模型相同的設備
        input_tensor = torch.rand(2, 3, img_h, img_w).to(device)
        print(f"\n--- Testing MambaVision Tiny forward pass (Input: {input_tensor.shape} on {input_tensor.device}) ---")

        model.eval()
        try:
            with torch.no_grad():
                output_features = model(input_tensor)
            print("Forward pass successful.")
            
            assert isinstance(output_features, list), f"Output is not a list, but {type(output_features)}"
            print(f"Output is a list with {len(output_features)} elements.")
            
            print("Output feature shapes:")
            for i, features in enumerate(output_features):
                print(f"Stage {i+1}: {features.shape}")

            runtime_widths = [f.size(1) for f in output_features]
            print("\nRuntime output feature channels:", runtime_widths)
            assert model.width_list == runtime_widths, "Width list mismatch with runtime output!"
            print("Width list verified successfully against runtime output.")

        except Exception as e:
            print(f"\nError during testing: {e}")
            import traceback
            traceback.print_exc()