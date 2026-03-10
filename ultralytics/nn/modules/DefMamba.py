import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import einops
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

# ===========================================================================
# CUDA Import Handling (VMamba specific)
# ===========================================================================
try:
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    import selective_scan_cuda_core
except Exception as e:
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    try:
        import selective_scan_cuda
    except ImportError:
        # print("Warning: selective_scan_cuda not found. Model will not run on GPU without it.")
        selective_scan_cuda = None

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# ===========================================================================
# Core VMamba / Selective Scan Components
# ===========================================================================

class SelectiveScan(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        # 參數檢查
        assert nrows in [1, 2, 3, 4], f"{nrows}" 
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        
        # 確保內存連續
        if u.stride(-1) != 1: u = u.contiguous()
        if delta.stride(-1) != 1: delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1: D = D.contiguous()
        if B.stride(-1) != 1: B = B.contiguous()
        if C.stride(-1) != 1: C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        # ============================================================
        # FIX: CPU Fallback for YOLO initialization / Stride check
        # ============================================================
        if not u.is_cuda:
            # 如果輸入在 CPU 上 (通常是 YOLO 的 stride 計算階段)
            # 我們返回一個形狀正確的全零張量。
            # 這不是真正的 SSM 運算，但足以讓 YOLO 完成初始化而不崩潰。
            # 注意：請勿嘗試在 CPU 上進行真正的推論。
            x_shape = (u.shape[0], A.shape[0], A.shape[1]) # B, D, N
            x = u.new_zeros(x_shape)
            out = u.new_zeros(u.shape)
            
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out
        # ============================================================

        # 正常的 CUDA 路徑
        if SSMODE == "mamba_ssm":
            if selective_scan_cuda is None:
                 raise ImportError("mamba_ssm kernels not installed.")
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        # 如果是在 CPU fallback 模式下記錄的 tensor，backward 不支援
        if not dout.is_cuda:
             return (None, None, None, None, None, None, None, None, None)

        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False 
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)

class DeformablePathTrans(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, de_index):
        B, C, N = x.shape
        _, indices = torch.topk(de_index, k=N, dim=-1, largest=False)
        x_gathered = torch.gather(x, 2, indices.unsqueeze(1).expand(-1, C, -1)).contiguous()
        x_out = x_gathered.permute(0, 2, 1).contiguous()
        ctx.save_for_backward(x, de_index, indices)
        return x_out, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        x, de_index, indices = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        grad_x.scatter_add_(2, indices.unsqueeze(1).expand(-1, x.shape[1], -1), grad_output.permute(0, 2, 1).contiguous()).contiguous()
        grad_de_index = (grad_output.permute(0, 2, 1).contiguous()-grad_x).mean(dim=1)
        grad_de_index = grad_de_index.view_as(de_index)
        return grad_x, grad_de_index

class ConvOffset(nn.Module):
    def __init__(self, embed_dim, kk, pad_size):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kk, 1, pad_size, groups=embed_dim)
        self.ca = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//16),
                nn.GELU(),
                nn.Linear(embed_dim//16, embed_dim),
                nn.Sigmoid()
                )
        self.ln = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(embed_dim, 3, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x_c = F.adaptive_avg_pool2d(x, (1, 1))
        x_c = self.ca(x_c.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x1 * x_c.expand_as(x)
        x = self.gelu(self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = self.conv2(x)
        return x

class DeformableLayer(nn.Module):
    def __init__(self, index=0, embed_dim=192, debug=False, h=0, w=0):
        super().__init__()
        self.ksize = [9, 7, 5, 3]
        self.stride = 1
        kk = self.ksize[index] if index < len(self.ksize) else 3
        pad_size = kk // 2 if kk != 1 else 0
        self.debug = debug
        self.conv_offset = ConvOffset(embed_dim, kk, pad_size)
        self.rpe_table = nn.Parameter(torch.zeros(embed_dim, 7, 7))
        trunc_normal_(self.rpe_table, std=0.01)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B, -1, -1, -1) 
        return ref

    @torch.no_grad()
    def _get_key_ref_points(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0, H, H, dtype=dtype, device=device),
            torch.linspace(0, W, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B, -1, -1, -1)
        return ref

    @torch.no_grad()
    def _get_path_ref_points(self, N, B, dtype, device):
        ref_path = torch.linspace(0.5, N - 0.5, N, dtype=dtype, device=device),
        ref_path[0].div_(N - 1.0).mul_(2.0).sub_(1.0)
        ref = ref_path[0][None, ...].expand(B, -1) 
        return ref

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}

    def forward(self, x):
        dtype, device = x.dtype, x.device
        B, C, H, W = x.size()
        N = H * W

        offset = self.conv_offset(x).contiguous() 
        offset, de_index = torch.split(offset, [2, 1], dim=1)
        Hk, Wk = offset.size(2), offset.size(3)

        offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
        offset = offset.tanh().mul(offset_range)

        offset = einops.rearrange(offset, 'b p h w -> b h w p').contiguous()
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        de_index = de_index.tanh().flatten(1)
        path_reference = self._get_path_ref_points(N, B, dtype, device)

        pos = offset + reference
        path_pos = de_index + path_reference

        x_sampled = F.grid_sample(
            input=x,
            grid=pos[..., (1, 0)],
            mode='bilinear', align_corners=True) 

        rpe_table = self.rpe_table
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        rpe_bias = F.interpolate(rpe_bias, size=(H, W), mode='bilinear', align_corners=False)
        key_grid = self._get_key_ref_points(H, W, B, dtype, device)
        displacement = (key_grid - pos) * 0.5
        pos_bias = F.grid_sample(
            input=rpe_bias,
            grid=displacement[..., (1, 0)],
            mode='bilinear', align_corners=True
        )
        x = x_sampled + pos_bias
        x = x.flatten(2)
        x, indices = DeformablePathTrans.apply(x, path_pos) 
        return x, indices

class DeformableLayerReverse(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, indices=None):
        x = x.flatten(2)
        B, C, N = x.size()
        index_re = torch.zeros_like(indices, device=x.device)
        index_re.scatter_add_(1, indices, torch.arange(indices.size(-1), device=x.device).unsqueeze(0).expand(indices.size(0), -1))
        x = torch.gather(x, 2, index_re.unsqueeze(1).expand(-1, C, -1))
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def x_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        nrows=-1,
        delta_softplus=True,
        to_dtype=True,
        force_fp32=True,
        stage=0,
        DS=None,
        DR=None,
        **kwargs,
):
    K, D, R = dt_projs_weight.shape
    B, D, H, W = x.shape
    L = H * W
    _, N = A_logs.shape

    xs = x.new_empty((B, 3, D, H * W))
    xs[:, 0] = x.flatten(2, 3)
    xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
    temp, indices = DS(x)
    xs[:, 2] = temp.permute(0, 2, 1)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float)) 
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    )

    ys = ys.view(B, K, -1, H, W)
    ys = ys.view(B, K, D, -1)
    y = (ys[:, 0] + ys[:, 1].flip(dims=[-1]) + DR(ys[:, 2], indices)) / 3.

    y = y.transpose(dim0=1, dim1=2).contiguous() 
    if K!=1: y = y.view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y), {'ys':ys, 'xs':xs, 'dts':dts, 'As':A_logs, 'Bs':Bs, 'Cs':Cs, 'Ds':Ds, 'delta_bias':delta_bias}

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  
        x1 = x[..., 1::2, 0::2, :] 
        x2 = x[..., 0::2, 1::2, :] 
        x3 = x[..., 1::2, 1::2, :]  
        x = torch.cat([x0, x1, x2, x3], -1) 
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class DSSM(nn.Module):
    def __init__(
        self,
        d_model=96, d_state=16, ssm_ratio=2.0, dt_rank="auto", act_layer=nn.SiLU,
        d_conv=3, conv_bias=True, dropout=0.0, bias=False,
        dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, simple_init=False,
        forward_type="v2", stage = 0, **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = d_expand
        self.d_inner = d_inner
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.stage = stage
        self.K = 3
        self.K2 = self.K

        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        if self.d_conv > 1:
            stride = 1
            self.conv2d = nn.Conv2d(
                in_channels=d_expand, out_channels=d_expand, groups=d_expand, bias=conv_bias,
                kernel_size=d_conv, padding=(d_conv - 1) // 2, stride=stride, **factory_kwargs,
            )

        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) 
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) 
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)

        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.DS = DeformableLayer(index=stage, embed_dim=d_inner, debug=False)
        self.DR = DeformableLayerReverse()
        self.kwargs = kwargs
        
        if simple_init:
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

        self.debug = False
        self.outnorm = None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A) 
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D) 
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = x_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, self.outnorm,
            nrows=nrows, delta_softplus=True, force_fp32=self.training, stage=self.stage, DS=self.DS, DR=self.DR,
            **self.kwargs,
        )
        x = x[0]
        return x

    def forward(self, x: torch.Tensor,h_tokens=None,w_tokens=None, **kwargs):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # =====================================================
        # FIX: Out-of-place activation for z to avoid inplace modification of split view
        z = self.act(z.clone())
        # =====================================================
        
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_simple_init=False,
        forward_type="v2",
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
        stage = 0,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = DSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                simple_init=ssm_simple_init,
                forward_type=forward_type,
                stage=stage,
                **kwargs,
            )
        self.drop_path = DropPath(drop_path)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)
        self.kwargs = kwargs

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            x = x + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

class VSSM(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768], 
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_simple_init=False,
        forward_type="v2",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN",
        downsample_version: str = "v2", 
        patchembed_version: str = "v1", 
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
        norm_layer = nn.LayerNorm
        if isinstance(ssm_act_layer, str):
            ssm_act_layer = _ACTLAYERS[ssm_act_layer.lower()]
        if isinstance(mlp_act_layer, str):
            mlp_act_layer = _ACTLAYERS[mlp_act_layer.lower()]

        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer)

        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                stage=i_layer,
                **kwargs,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), 
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()), 
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_simple_init=False,
        forward_type="v2",
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        stage = 0,
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
                stage=stage,
                **kwargs,
            ))
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for layer in self.layers:
            for block in layer.blocks:
                x = block(x)
            x = layer.downsample(x)
        x = self.classifier(x)
        return x

# ===========================================================================
# MODIFIED Backbone_VSSM with width_list and list output
# ===========================================================================

class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer=nn.LayerNorm, **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

        # --- Added width_list logic from SMT ---
        self.width_list = []
        try:
            self.eval() 
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                 features = self.forward(dummy_input)

            self.width_list = [f.size(1) for f in features]
            self.train() 
        except Exception as e:
            print(f"Error during dummy forward pass for width_list calculation: {e}")
            print("Setting width_list based on dims config as fallback.")
            self.width_list = [self.dims[i] for i in self.out_indices]
            self.train() 

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y
            
        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) 
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        
        return outs

# ===========================================================================
# Factory Functions (mimicking SMT style)
# ===========================================================================

def defm_tiny(pretrained=False, **kwargs):
    model = Backbone_VSSM(
        dims=48,
        depths=[2, 2, 5, 2],
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=1.0,
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.2,
        **kwargs
    )
    return model

def defm_small(pretrained=False, **kwargs):
    model = Backbone_VSSM(
        dims=96,
        depths=[2, 2, 6, 2],
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=1.0,
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.2,
        **kwargs
    )
    return model

def defm_base(pretrained=False, **kwargs):
    model = Backbone_VSSM(
        dims=96,
        depths=[2, 3, 16, 2],
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=1.0,
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.4,
        **kwargs
    )
    return model


if __name__ == '__main__':
    # Test Code
    print("--- Testing DEFM (VSSM Backbone) with CPU Fallback ---")
    
    # 1. Create Model (Tiny) - This will trigger CPU dummy pass for width_list
    try:
        model = defm_tiny()
        print("Model (Tiny) created successfully.")
        
        # 2. Check width_list
        print(f"Calculated width_list: {model.width_list}")
        
        # 3. Test Forward Pass (CPU to verify fallback)
        print("\nTesting CPU Forward Pass (Stride Check Simulation):")
        img_size = 224
        input_tensor_cpu = torch.rand(1, 3, img_size, img_size)
        model.eval()
        with torch.no_grad():
            output_cpu = model(input_tensor_cpu)
        print(f"CPU Forward successful. Output type: {type(output_cpu)}")
        
        # 4. Test Forward Pass (GPU if available)
        if torch.cuda.is_available():
            print("\nTesting GPU Forward Pass (Real Execution):")
            model = model.cuda()
            input_tensor_gpu = input_tensor_cpu.cuda()
            with torch.no_grad():
                output_gpu = model(input_tensor_gpu)
            print("GPU Forward pass successful.")
            print(f"GPU Output shapes: {[o.shape for o in output_gpu]}")
        else:
            print("\nCUDA not available, skipping GPU test.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Runtime Error: {e}")