import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

# =========================================================================
#  Triton Kernels & Helpers (保持原樣，確保核心運算功能)
# =========================================================================
try:
    import selective_scan_cuda_oflex
    HAS_CUDA_KERNEL = True
except ImportError:
    HAS_CUDA_KERNEL = False

import triton
import triton.language as tl

@triton.jit
def triton_cross_scan(
    x, # (B, C, H, W)
    y, # (B, 4, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = _tmp0 + i_h * BH * DW  + tl.arange(0, BH)[:, None] * DW + i_w * BW + tl.arange(0, BW)[None, :]
    p_x = x + i_b * _tmp1 + _tmp2
    p_y1 = y + i_b * 4 * _tmp1 + _tmp2 # same
    p_y2 = y + i_b * 4 * _tmp1 + _tmp1 + _tmp0 + i_w * BW * DH + tl.arange(0, BW)[None, :] * DH + i_h * BH + tl.arange(0, BH)[:, None]  # trans
    p_y3 = y + i_b * 4 * _tmp1 + 2 * _tmp1 + _tmp0 + (NH - i_h - 1) * BH * DW  + (BH - 1 - tl.arange(0, BH)[:, None]) * DW + (NW - i_w - 1) * BW + (BW - 1 - tl.arange(0, BW)[None, :]) + (DH - NH * BH) * DW + (DW - NW * BW) # flip
    p_y4 = y + i_b * 4 * _tmp1 + 3 * _tmp1 + _tmp0 + (NW - i_w - 1) * BW * DH  + (BW - 1 - tl.arange(0, BW)[None, :]) * DH + (NH - i_h - 1) * BH + (BH - 1 - tl.arange(0, BH)[:, None]) + (DH - NH * BH) + (DW - NW * BW) * DH  # trans + flip

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        tl.store(p_y1 + _idx, _x, mask=_mask_hw)
        tl.store(p_y2 + _idx, _x, mask=_mask_hw)
        tl.store(p_y3 + _idx, _x, mask=_mask_hw)
        tl.store(p_y4 + _idx, _x, mask=_mask_hw)

@triton.jit
def triton_cross_merge(
    x, # (B, C, H, W)
    y, # (B, 4, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = _tmp0 + i_h * BH * DW  + tl.arange(0, BH)[:, None] * DW + i_w * BW + tl.arange(0, BW)[None, :]
    p_x = x + i_b * _tmp1 + _tmp2
    p_y1 = y + i_b * 4 * _tmp1 + _tmp2 # same
    p_y2 = y + i_b * 4 * _tmp1 + _tmp1 + _tmp0 + i_w * BW * DH + tl.arange(0, BW)[None, :] * DH + i_h * BH + tl.arange(0, BH)[:, None]  # trans
    p_y3 = y + i_b * 4 * _tmp1 + 2 * _tmp1 + _tmp0 + (NH - i_h - 1) * BH * DW  + (BH - 1 - tl.arange(0, BH)[:, None]) * DW + (NW - i_w - 1) * BW + (BW - 1 - tl.arange(0, BW)[None, :]) + (DH - NH * BH) * DW + (DW - NW * BW) # flip
    p_y4 = y + i_b * 4 * _tmp1 + 3 * _tmp1 + _tmp0 + (NW - i_w - 1) * BW * DH  + (BW - 1 - tl.arange(0, BW)[None, :]) * DH + (NH - i_h - 1) * BH + (BH - 1 - tl.arange(0, BH)[:, None]) + (DH - NH * BH) + (DW - NW * BW) * DH  # trans + flip

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _y1 = tl.load(p_y1 + _idx, mask=_mask_hw)
        _y2 = tl.load(p_y2 + _idx, mask=_mask_hw)
        _y3 = tl.load(p_y3 + _idx, mask=_mask_hw)
        _y4 = tl.load(p_y4 + _idx, mask=_mask_hw)
        tl.store(p_x + _idx, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)

# 為了完整性，這裡提供簡化的 Torch Fallback (CPU模式) 供形狀推斷使用
def torch_cross_scan(x):
    B, C, H, W = x.shape
    L = H * W
    y1 = x.reshape(B, -1, L)
    y2 = x.transpose(2, 3).contiguous().reshape(B, -1, L)
    y3 = torch.flip(x, [2, 3]).reshape(B, -1, L)
    y4 = torch.flip(x.transpose(2, 3), [2, 3]).contiguous().reshape(B, -1, L)
    y = torch.stack([y1, y2, y3, y4], dim=1)
    return y

def torch_cross_merge(y, H, W):
    B, K, C, L = y.shape
    y1, y2, y3, y4 = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    x1 = y1.reshape(B, C, H, W)
    x2 = y2.reshape(B, C, W, H).transpose(2, 3)
    x3 = torch.flip(y3.reshape(B, C, H, W), [2, 3])
    x4 = torch.flip(y4.reshape(B, C, W, H), [2, 3]).transpose(2, 3)
    return x1 + x2 + x3 + x4

# Triton Wrappers
class CrossScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        
        # Output: (B, 4, C, L) CPU Fallback or CUDA
        if x.is_cuda:
            BC, BH, BW = min(triton.next_power_of_2(C), 1), min(triton.next_power_of_2(H), 64), min(triton.next_power_of_2(W), 64)
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
            x = x.contiguous()
            y = x.new_empty((B, 4, C, H, W))
            triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
            return y.view(B, 4, C, -1)
        else:
            return torch_cross_scan(x)
    
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        B, C, H, W = ctx.shape
        if y.is_cuda:
            BC, BH, BW, NC, NH, NW = ctx.triton_shape
            y = y.contiguous().view(B, 4, C, H, W)
            x = y.new_empty((B, C, H, W))
            triton_cross_merge[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
            return x
        else:
             return torch_cross_merge(y, H, W)

class CrossMergeTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor):
        B, K, C, H, W = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        ctx.shape = (B, C, H, W)
        
        if y.is_cuda:
            BC, BH, BW = min(triton.next_power_of_2(C), 1), min(triton.next_power_of_2(H), 64), min(triton.next_power_of_2(W), 64)
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
            y = y.contiguous().view(B, 4, C, H, W)
            x = y.new_empty((B, C, H, W))
            triton_cross_merge[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
            return x.view(B, C, -1)
        else:
            # Flattened y comes in as (B, 4, C, L) -> reshape to H, W needed inside helper
            # But here y is (B, 4, C, H, W) in signature? 
            # In forward_ssm: y = self._cross_merge(ys.reshape(B, K, -1, H, W))
            return torch_cross_merge(y.view(B, 4, C, H*W), H, W).view(B, C, -1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, C, H, W = ctx.shape
        if x.is_cuda:
            BC, BH, BW, NC, NH, NW = ctx.triton_shape
            x = x.contiguous()
            y = x.new_empty((B, 4, C, H, W))
            triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
            return y
        else:
            return torch_cross_scan(x)

class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        if u.is_cuda and HAS_CUDA_KERNEL:
            out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            ctx.delta_softplus = delta_softplus
            return out
        else:
            ctx.is_cpu_fallback = True
            return torch.zeros_like(u) # Dummy for CPU/Shape inference

    @staticmethod
    def backward(ctx, dout, *args):
        if getattr(ctx, 'is_cpu_fallback', False):
             return (torch.zeros_like(dout),) * 10
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

# =========================================================================
#  Basic Layers (LayerNorm, Mlp, etc.)
# =========================================================================
class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
    def forward(self, x):
        return F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)     
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, in_features, kernel_size=1)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = x + self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# =========================================================================
#  SparXSS2D & SparXVSSBlock (Core Logic)
# =========================================================================
class SparXSS2D(nn.Module):

    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=1,
        dt_rank="auto",
        norm_layer=LayerNorm2d,
        act_layer=nn.SiLU,
        d_conv=3,
        dropout=0.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        stage_dense_idx=None,
        layer_idx=None,
        max_dense_depth=None,
        dense_step=None,
        is_cross_layer=None,
        dense_layer_idx=None,
        sr_ratio=1,
        cross_stage=False,   
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        d_inner = int(d_model * ssm_ratio)
        
        self.layer_idx = layer_idx
        self.d_inner = d_inner
        
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        k_group = 4

        # in proj =======================================
        self.in_proj = nn.Conv2d(d_model, d_inner, kernel_size=1)
        self.act = act_layer()
        
        self.is_cross_layer = is_cross_layer
        
        if is_cross_layer:
            if dense_layer_idx == stage_dense_idx[0]:
                intra_dim = d_model
                inner_dim = d_model * dense_layer_idx
            else:
                count = 1 if cross_stage else 0
                for item in stage_dense_idx:
                    if item == dense_layer_idx:
                        break
                    count += 1
                count = min(max_dense_depth, count)
                intra_dim = d_model * count
                inner_dim = d_model * (dense_step - 1)
                
            current_d_dim = int(intra_dim + inner_dim)

            self.d_norm = norm_layer(current_d_dim)

            del self.in_proj
            self.d_proj = nn.Sequential(
                nn.Conv2d(current_d_dim, current_d_dim, kernel_size=3, padding=1, groups=current_d_dim, bias=False),
                nn.BatchNorm2d(current_d_dim),
                nn.GELU(),
                nn.Conv2d(current_d_dim, d_model*2, kernel_size=1),
                nn.GELU(),
            )
            
            self.channel_padding = d_inner - d_inner//3*3

            self.q = nn.Sequential(
                self.get_sr(d_model, sr_ratio),
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
                nn.BatchNorm2d(d_model),
            )
            
            self.k = nn.Sequential(
                self.get_sr(d_model, sr_ratio),
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
                nn.BatchNorm2d(d_model),
            )
            
            self.v = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
                nn.BatchNorm2d(d_model),
            )
            
            self.in_proj = nn.Sequential(
                nn.Conv2d(d_model*3, d_model*3, kernel_size=3, padding=1, groups=d_model*3),
                nn.GELU(),
                nn.Conv2d(d_model*3, int(d_inner//3*3), kernel_size=1, groups=3),     
            )

        # conv =======================================
        if d_conv > 1:
            self.conv2d = nn.Conv2d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=(d_conv-1)//2)
            
        # out proj =======================================
        self.out_norm = norm_layer(d_inner)
        self.out_proj = nn.Conv2d(d_inner, d_model, kernel_size=1)

        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0).view(-1, d_inner, 1))
        del self.x_proj
        
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))    
            
    @staticmethod
    def get_sr(dim, sr_ratio):
        if sr_ratio > 1:
            sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio+1, stride=sr_ratio, padding=(sr_ratio+1)//2, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
            )
        else:
            sr = nn.Identity()
        return sr
      
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def _selective_scan(self, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=None, backnrows=None, ssoflex=False):
        return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
    def _cross_scan(self, x):
        return CrossScanTriton.apply(x)
    
    def _cross_merge(self, x):
        return CrossMergeTriton.apply(x)

    def forward_ssm(self, x, to_dtype=False, force_fp32=False):

        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
  
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W
        
        xs = self._cross_scan(x)
        
        x_dbl = F.conv1d(xs.reshape(B, -1, L), self.x_proj_weight, bias=None, groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), dt_projs_weight.reshape(K * D, -1, 1), groups=K)
        
        xs = xs.reshape(B, -1, L)
        dts = dts.contiguous().reshape(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().reshape(B, K, N, L)
        Cs = Cs.contiguous().reshape(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.reshape(-1).to(torch.float)
              
        if force_fp32:
            xs = xs.to(torch.float32)
            dts = dts.to(torch.float32)
            Bs = Bs.to(torch.float32)
            Cs = Cs.to(torch.float32)
                  
        ys = self._selective_scan(xs, dts, As, Bs, 
                                  Cs, Ds, delta_bias,
                                  delta_softplus=True,
                                  ssoflex=True)

        y = self._cross_merge(ys.reshape(B, K, -1, H, W))
        y = self.out_norm(y.reshape(B, -1, H, W))
        
        if to_dtype:
            y = y.to(x.dtype)

        return y
    

    def forward(self, x, shortcut):
        
        if self.is_cross_layer:
            
            s = self.d_proj(self.d_norm(shortcut))
            k, v = torch.chunk(s, 2, dim=1)
            
            q = self.q(x)
            k = self.k(k)
            v = self.v(v)
            
            B, C, H, W = x.shape
            sr_H, sr_W = q.shape[2:]
            g_dim = q.shape[1] // 4
                        
            q = q.reshape(-1, g_dim, sr_H*sr_W)
            k = k.reshape(-1, g_dim, sr_H*sr_W)
            v = v.reshape(-1, g_dim, H*W)
            
            attn = F.scaled_dot_product_attention(q, k, v).reshape(B, -1, H, W)
            
            if (H, W) == (sr_H, sr_W):
                x = q.reshape(B, -1, H, W)
                
            x = torch.cat([x, attn, v.reshape(B, -1, H, W)], dim=1)
            x = rearrange(x, 'b (c g) h w -> b (g c) h w', g=3).contiguous() ### channel shuffle for efficiency
            x = self.in_proj(x)
            
            if self.channel_padding > 0:
                if self.channel_padding == 1:
                    pad = torch.mean(x, dim=1, keepdim=True)
                    x = torch.cat([x, pad], dim=1)
                else:
                    pad = rearrange(x, 'b c h w -> b (h w) c')
                    pad = F.adaptive_avg_pool1d(pad, self.channel_padding)
                    pad = rearrange(pad, 'b (h w) c -> b c h w', h=H, w=W)
                    x = torch.cat([x, pad], dim=1)
        else:
            x = self.in_proj(x)
            
        if self.d_conv > 1:
            x = self.conv2d(x)
            
        x = self.act(x)
        x = self.forward_ssm(x)   
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x


class SparXVSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=0,
        drop_path=0,
        norm_layer=LayerNorm2d,
        ssm_d_state=16,
        ssm_ratio=1,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_drop_rate=0,
        ssm_init="v0",
        mlp_ratio=4,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0,
        use_checkpoint=False,
        post_norm=False,
        layer_idx=None,
        stage_dense_idx=None,
        max_dense_depth=None,
        dense_step=None,
        is_cross_layer=None,
        dense_layer_idx=None,
        sr_ratio=1,
        cross_stage=False,
        ls_init_value=1e-5,
        **kwargs,
    ):

        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.pos_embed = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)

        self.norm = norm_layer(hidden_dim)
        self.op = SparXSS2D(
            d_model=hidden_dim, 
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            norm_layer=norm_layer,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            layer_idx=layer_idx,
            stage_dense_idx=stage_dense_idx,
            max_dense_depth=max_dense_depth,
            dense_step=dense_step,
            is_cross_layer=is_cross_layer,
            dense_layer_idx=dense_layer_idx,
            sr_ratio=sr_ratio,
            cross_stage=cross_stage,
            **kwargs,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate)

        if ls_init_value is not None:
            self.layerscale_1 = LayerScale(hidden_dim, init_value=ls_init_value)
            self.layerscale_2 = LayerScale(hidden_dim, init_value=ls_init_value)
        else:
            self.layerscale_1 = nn.Identity()
            self.layerscale_2 = nn.Identity()
    
    def _forward(self, x, shortcut):
        
        x = x + self.pos_embed(x)
        x = self.layerscale_1(x) + self.drop_path(self.op(self.norm(x), shortcut)) # Token Mixer
        x = self.layerscale_2(x) + self.drop_path(self.mlp(self.norm2(x))) # FFN
        shortcut = x

        return (x, shortcut)

    def forward(self, x):

        input, shortcut = x
        
        if isinstance(shortcut, (list, tuple)):
            shortcut = torch.cat(shortcut, dim=1)
        
        if self.use_checkpoint and input.requires_grad:
            return checkpoint.checkpoint(self._forward, input, shortcut, use_reentrant=True)
        else:
            return self._forward(input, shortcut)

# =========================================================================
#  YOLO Modular Wrappers (關鍵修改部分)
# =========================================================================

class SparXStem(nn.Module):
    """
    對應原本 SparXMamba 的 patch_embed
    Args:
        c1 (int): 輸入通道 (通常是 3)
        c2 (int): 輸出通道
        k (int): kernel size
        s (int): stride
        stem_type (str): 'v1' 或 'v2'
    """
    def __init__(self, c1, c2, k=3, s=2, stem_type='v1'):
        super().__init__()
        if stem_type == 'v1':
            self.stem = nn.Sequential(
                nn.Conv2d(c1, c2//2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c2//2),
                nn.GELU(),
                nn.Conv2d(c2//2, c2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c2)
            )
        else: # v2
            self.stem = nn.Sequential(
                nn.Conv2d(c1, c2//2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c2//2),
                nn.GELU(),        
                nn.Conv2d(c2//2, c2//2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c2//2),
                nn.GELU(),
                nn.Conv2d(c2//2, c2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        return self.stem(x)

class SparXDownsample(nn.Module):
    """
    對應原本 SparXMamba 的 downsample 層
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
        )
    
    def forward(self, x):
        return self.down(x)

class SparXStage(nn.Module):
    """
    對應原本 SparXMamba 中的一層 (包含多個 Blocks 和 Dense Logic)
    """
    def __init__(self, c1, c2, depth=1, dense_step=1, dense_start=0, max_dense_depth=100, cross_stage=False, sr_ratio=1):
        super().__init__()
        # 在 YOLO 中，Stage 的輸入輸出通道通常保持一致，改變通道由 Downsample 層負責
        assert c1 == c2, f"SparXStage expects c1==c2, but got {c1} and {c2}. Use Downsample to change channels."
        dim = c1
        
        # === 防呆機制 ===
        if dense_step <= 0:
            dense_step = 1
        # ===============

        # 計算 Dense Index (邏輯來自原代碼)
        # 注意：原代碼是 [::-1] 反轉的，這裡保持一致
        d_idx = [i for i in range(depth - 1, dense_start - 1, -dense_step)][::-1]
        
        self.blocks = nn.ModuleList()
        
        # 參數預設值 (可根據需要調整，或從 kwargs 傳入)
        ssm_d_state = 16
        ssm_dt_rank = "auto"
        ssm_ratio = 2.0
        mlp_ratio = 4.0
        
        for d in range(depth):
            if d in d_idx:
                is_cross_layer = True
                dense_layer_idx = d
            else:
                is_cross_layer = False
                dense_layer_idx = None
            
            self.blocks.append(SparXVSSBlock(
                hidden_dim=dim,
                drop_path=0.0, 
                norm_layer=LayerNorm2d,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                mlp_ratio=mlp_ratio,
                layer_idx=d,
                max_dense_depth=max_dense_depth,
                dense_step=dense_step,
                is_cross_layer=is_cross_layer,
                stage_dense_idx=d_idx,
                dense_layer_idx=dense_layer_idx,
                sr_ratio=sr_ratio,
                cross_stage=cross_stage,
                ls_init_value=1e-5
            ))
            
        self.dense_cfg = (dense_step, dense_start, max_dense_depth, d_idx, cross_stage)

    def forward(self, x):
        dense_step, dense_start, max_dense_depth, dense_idx, cross_stage = self.dense_cfg
        
        # === 關鍵修正部分 ===
        # 在 SparXMamba 原邏輯中，'s' 是來自上一層的輸出。
        # 在 YOLO 模組化後，當前 Stage 的輸入 'x' 就是我們的初始狀態 's'。
        # 如果不把 x 加入 inner_list，Dense Layer 拼接時就會少一個 Tensor，導致維度錯誤。
        
        s = x 
        inner_list = [s] # 初始化放入輸入特徵
        cross_list = []  # 暫時不支援跨 Stage 的 Dense 連接 (模組化限制)，設為空
        
        for idx, layer in enumerate(self.blocks):
            if idx in dense_idx:
                # 準備 shortcut: 將 inner 和 cross 列表合併
                # SparXVSSBlock 內部會執行 torch.cat(shortcut, dim=1)
                shortcut = inner_list + cross_list
                
                # 執行 Block
                # 注意：這裡傳入 tuple (x, shortcut)
                x, s = layer((x, shortcut))
                
                # Dense Layer 執行後，將新的狀態 s 加入 cross_list (作為長期記憶)
                cross_list.append(s)
                # 清空 inner_list (因為已經被 Dense Layer 消化並轉化為新的 s)
                inner_list = []
            else:
                # 非 Dense Layer，不需要 Shortcut (或傳入 None)
                x, s = layer((x, None))
                # 普通 Layer 的輸出 s 加入 inner_list (作為短期記憶積累)
                inner_list.append(s)
            
            # 控制最大深度，避免顯存爆炸
            if (max_dense_depth is not None) and len(cross_list) > max_dense_depth:
                cross_list = cross_list[-max_dense_depth:]
        
        return x