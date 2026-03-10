import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import itertools
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite, DropPath, trunc_normal_
from functools import partial
import pywt
import pywt.data

import math
import copy
from typing import Optional, Callable, Any
from collections import OrderedDict
import numpy as np
import warnings

# =========================================================================
# Triton & CUDA Imports and Definitions
# =========================================================================

WITH_TRITON = True
try:
    import triton
    import triton.language as tl
except:
    WITH_TRITON = False
    # warnings.warn("Triton not installed, fall back to pytorch implements.")

if WITH_TRITON:
    try:
        from functools import cached_property
    except:
        warnings.warn("if you are using py37, add this line to functools.py: "
            "cached_property = lambda func: property(lru_cache()(func))")

WITH_SELECTIVESCAN_OFLEX = True
WITH_SELECTIVESCAN_CORE = False
WITH_SELECTIVESCAN_MAMBA = True
try:
    import selective_scan_cuda_oflex
except ImportError:
    WITH_SELECTIVESCAN_OFLEX = False
try:
    import selective_scan_cuda_core
except ImportError:
    WITH_SELECTIVESCAN_CORE = False
try:
    import selective_scan_cuda
except ImportError:
    WITH_SELECTIVESCAN_MAMBA = False

# =========================================================================
# Torch Fallback Implementations
# =========================================================================

def cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=2):
    if in_channel_first:
        B, C, H, W = x.shape
        if scans == 0:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
        elif scans == 1:
            y = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        elif scans == 2:
            y = x.view(B, 1, C, H * W)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
    else:
        B, H, W, C = x.shape
        if scans == 0:
            y = x.new_empty((B, H * W, 4, C))
            y[:, :, 0, :] = x.flatten(1, 2)
            y[:, :, 1, :] = x.transpose(dim0=1, dim1=2).flatten(1, 2)
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])
        elif scans == 1:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 2, 1)
        elif scans == 2:
            y = x.view(B, H * W, 1, C)
            y = torch.cat([y, y.flip(dims=[1])], dim=2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=2):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y = y[:, 0] + y[:, 1].flip(dims=[-1]).view(B, 1, D, -1)
            y = y.sum(1)
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y[:, :, 0] + y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).contiguous().view(B, -1, D)
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y = y[:, :, 0] + y[:, :, 1].flip(dims=[1]).view(B, -1, 1, D)
            y = y.sum(2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()

    return y

def cross_scan1b1_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=2):
    if in_channel_first:
        B, _, C, H, W = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, 0].flatten(2, 3),
                x[:, 1].transpose(dim0=2, dim1=3).flatten(2, 3),
                torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                torch.flip(x[:, 3].transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = x.flatten(2, 3)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(2, 3),
                x[:, 1].flatten(2, 3),
                torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                torch.flip(x[:, 3].flatten(2, 3), dims=[-1]),
            ], dim=1)
    else:
        B, H, W, _, C = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, :, :, 0].flatten(1, 2),
                x[:, :, :, 1].transpose(dim0=1, dim1=2).flatten(1, 2),
                torch.flip(x[:, :, :, 2].flatten(1, 2), dims=[1]),
                torch.flip(x[:, :, :, 3].transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = x.flatten(1, 2)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(1, 2),
                x[:, 1].flatten(1, 2),
                torch.flip(x[:, 2].flatten(1, 2), dims=[-1]),
                torch.flip(x[:, 3].flatten(1, 2), dims=[-1]),
            ], dim=2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge1b1_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=2):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = torch.stack([
                y[:, 0],
                y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).flatten(2, 3),
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3].view(B, -1, W, H).transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, 0],
                y[:, 1],
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3], dims=[-1]),
            ], dim=1)
    else:
        B, H, W, _, D = y.shape
        y = y.view(B, -1, 2, D)
        if scans == 0:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).flatten(1, 2),
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3].view(B, W, H, -1).transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1],
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3], dims=[1]),
            ], dim=2)

    if out_channel_first and (not in_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not out_channel_first) and in_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y

class CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=2):
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        if one_by_one:
            B, K, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, K, C = x.shape
        else:
            B, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, C = x.shape
        ctx.shape = (B, C, H, W)

        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        y = _fn(x, in_channel_first, out_channel_first, scans)
        return y

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape

        ys = ys.view(B, -1, C, H, W) if out_channel_first else ys.view(B, H, W, -1, C)
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)

        if one_by_one:
            y = y.view(B, 2, -1, H, W) if in_channel_first else y.view(B, H, W, 2, -1)
        else:
            y = y.view(B, -1, H, W) if in_channel_first else y.view(B, H, W, -1)

        return y, None, None, None, None

class CrossMergeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=2):
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        B, K, C, H, W = ys.shape
        if not out_channel_first:
            B, H, W, K, C = ys.shape
        ctx.shape = (B, C, H, W)
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape

        if not one_by_one:
            if in_channel_first:
                x = x.view(B, C, H, W)
            else:
                x = x.view(B, H, W, C)
        else:
            if in_channel_first:
                x = x.view(B, 2, C, H, W)
            else:
                x = x.view(B, H, W, 2, C)

        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        x = _fn(x, in_channel_first, out_channel_first, scans)
        x = x.view(B, 2, C, H, W) if out_channel_first else x.view(B, H, W, 2, C)
        return x, None, None, None, None

# =========================================================================
# Triton Kernel Definitions (k2)
# =========================================================================
if WITH_TRITON:
    @triton.jit
    def triton_cross_scan_flex_k2(
        x, y, x_layout: tl.constexpr, y_layout: tl.constexpr, operation: tl.constexpr, onebyone: tl.constexpr,
        scans: tl.constexpr, BC: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr, DC: tl.constexpr,
        DH: tl.constexpr, DW: tl.constexpr, NH: tl.constexpr, NW: tl.constexpr,
    ):
        i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_h, i_w = (i_hw // NW), (i_hw % NW)
        _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
        _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
        _mask_hw = _mask_h[:, None] & _mask_w[None, :]
        _for_C = min(DC - i_c * BC, BC)

        HWRoute0 = i_h * BH * DW  + tl.arange(0, BH)[:, None] * DW + i_w * BW + tl.arange(0, BW)[None, :]
        HWRoute2 = (NH - i_h - 1) * BH * DW  + (BH - 1 - tl.arange(0, BH)[:, None]) * DW + (NW - i_w - 1) * BW + (BW - 1 - tl.arange(0, BW)[None, :]) + (DH - NH * BH) * DW + (DW - NW * BW) # flip

        if scans == 1:
            HWRoute2 = HWRoute0

        _tmp1 = DC * DH * DW

        y_ptr_base = y + i_b * 2 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0 else i_c * BC)
        if y_layout == 0:
            p_y1 = y_ptr_base + HWRoute0
            p_y2 = y_ptr_base + 1 * _tmp1 + HWRoute2
        else:
            p_y1 = y_ptr_base + HWRoute0 * 4 * DC
            p_y2 = y_ptr_base + 1 * DC + HWRoute2 * 4 * DC

        if onebyone == 0:
            x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
            if x_layout == 0:
                p_x = x_ptr_base + HWRoute0
            else:
                p_x = x_ptr_base + HWRoute0 * DC

            if operation == 0:
                for idxc in range(_for_C):
                    _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                    _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                    _x = tl.load(p_x + _idx_x, mask=_mask_hw)
                    tl.store(p_y1 + _idx_y, _x, mask=_mask_hw)
                    tl.store(p_y2 + _idx_y, _x, mask=_mask_hw)
            elif operation == 1:
                for idxc in range(_for_C):
                    _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                    _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                    _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hw)
                    _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hw)
                    tl.store(p_x + _idx_x, _y1 + _y2, mask=_mask_hw)

        else:
            x_ptr_base = x + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
            if x_layout == 0:
                p_x1 = x_ptr_base + HWRoute0
                p_x2 = p_x1 + 2 * _tmp1
            else:
                p_x1 = x_ptr_base + HWRoute0 * 4 * DC
                p_x2 = p_x1 + 2 * DC

            if operation == 0:
                for idxc in range(_for_C):
                    _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                    _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                    _x1 = tl.load(p_x1 + _idx_x, mask=_mask_hw)
                    _x2 = tl.load(p_x2 + _idx_x, mask=_mask_hw)
                    tl.store(p_y1 + _idx_y, _x1, mask=_mask_hw)
                    tl.store(p_y2 + _idx_y, _x2, mask=_mask_hw)
            else:
                for idxc in range(_for_C):
                    _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                    _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                    _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hw)
                    _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hw)
                    tl.store(p_x1 + _idx_x, _y1, mask=_mask_hw)
                    tl.store(p_x2 + _idx_x, _y2, mask=_mask_hw)

    class CrossScanTritonFk2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=2):
            if one_by_one:
                if in_channel_first:
                    B, _, C, H, W = x.shape
                else:
                    B, H, W, _, C = x.shape
            else:
                if in_channel_first:
                    B, C, H, W = x.shape
                else:
                    B, H, W, C = x.shape
            B, C, H, W = int(B), int(C), int(H), int(W)
            BC, BH, BW = 1, 32, 32
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            
            ctx.in_channel_first = in_channel_first
            ctx.out_channel_first = out_channel_first
            ctx.one_by_one = one_by_one
            ctx.scans = scans
            ctx.shape = (B, C, H, W)
            ctx.triton_shape = (BC, BH, BW, NC, NH, NW)

            y = x.new_empty((B, 2, C, H * W)) if out_channel_first else x.new_empty((B, H * W, 2, C))
            triton_cross_scan_flex_k2[(NH * NW, NC, B)](
                x.contiguous(), y, 
                (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans, 
                BC, BH, BW, C, H, W, NH, NW
            )
            return y
            
        @staticmethod
        def backward(ctx, y: torch.Tensor):
            in_channel_first = ctx.in_channel_first
            out_channel_first = ctx.out_channel_first
            one_by_one = ctx.one_by_one
            scans = ctx.scans
            B, C, H, W = ctx.shape
            BC, BH, BW, NC, NH, NW = ctx.triton_shape
            if one_by_one:
                x = y.new_empty((B, 2, C, H, W)) if in_channel_first else y.new_empty((B, H, W, 2, C))
            else:
                x = y.new_empty((B, C, H, W)) if in_channel_first else y.new_empty((B, H, W, C))
            
            triton_cross_scan_flex_k2[(NH * NW, NC, B)](
                x, y.contiguous(), 
                (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
                BC, BH, BW, C, H, W, NH, NW
            )
            return x, None, None, None, None


    class CrossMergeTritonFk2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=2):
            if out_channel_first:
                B, _, C, H, W = y.shape
            else:
                B, H, W, _, C = y.shape
            B, C, H, W = int(B), int(C), int(H), int(W)
            BC, BH, BW = 1, 32, 32
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            ctx.in_channel_first = in_channel_first
            ctx.out_channel_first = out_channel_first
            ctx.one_by_one = one_by_one
            ctx.scans = scans
            ctx.shape = (B, C, H, W)
            ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
            if one_by_one:
                x = y.new_empty((B, 2, C, H * W)) if in_channel_first else y.new_empty((B, H * W, 2, C))
            else:
                x = y.new_empty((B, C, H * W)) if in_channel_first else y.new_empty((B, H * W, C))
            triton_cross_scan_flex_k2[(NH * NW, NC, B)](
                x, y.contiguous(), 
                (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
                BC, BH, BW, C, H, W, NH, NW
            )
            return x
            
        @staticmethod
        def backward(ctx, x: torch.Tensor):
            in_channel_first = ctx.in_channel_first
            out_channel_first = ctx.out_channel_first
            one_by_one = ctx.one_by_one
            scans = ctx.scans
            B, C, H, W = ctx.shape
            BC, BH, BW, NC, NH, NW = ctx.triton_shape
            y = x.new_empty((B, 2, C, H, W)) if out_channel_first else x.new_empty((B, H, W, 2, C))
            triton_cross_scan_flex_k2[(NH * NW, NC, B)](
                x.contiguous(), y, 
                (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans,
                BC, BH, BW, C, H, W, NH, NW
            )
            return y, None, None, None, None, None

def cross_scan_fn_k2(x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=2, force_torch=False):
    CSF = CrossScanTritonFk2 if WITH_TRITON and x.is_cuda and (not force_torch) else CrossScanF
    return CSF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)

def cross_merge_fn_k2(y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=2, force_torch=False):
    CMF = CrossMergeTritonFk2 if WITH_TRITON and y.is_cuda and (not force_torch) else CrossMergeF
    return CMF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)

# =========================================================================
# Selective Scan
# =========================================================================

def selective_scan_torch(
    u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
    D: torch.Tensor = None, delta_bias: torch.Tensor = None, delta_softplus=True, oflex=True, *args, **kwargs
):
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = int(KCdim / K)
    
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
            
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    
    if True:
        x = A.new_zeros((Batch, KCdim, N))
        ys = []
        for i in range(L):
            x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
            y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            ys.append(y)
        y = torch.stack(ys, dim=2) 
    
    out = y if D is None else y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


class SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
        ctx.delta_softplus = delta_softplus
        backend = "oflex" if WITH_SELECTIVESCAN_OFLEX and (backend is None) else backend
        backend = "core" if WITH_SELECTIVESCAN_CORE and (backend is None) else backend
        backend = "mamba" if WITH_SELECTIVESCAN_MAMBA and (backend is None) else backend
        ctx.backend = backend
        if backend == "oflex":
            out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        elif backend == "core":
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        elif backend == "mamba":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
             return selective_scan_torch(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        backend = ctx.backend
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if backend == "oflex":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "core":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "mamba":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False
            )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, oflex=True, backend=None):
    WITH_CUDA = (WITH_SELECTIVESCAN_OFLEX or WITH_SELECTIVESCAN_CORE or WITH_SELECTIVESCAN_MAMBA)
    fn = selective_scan_torch if backend == "torch" or (not WITH_CUDA) else SelectiveScanCuda.apply
    return fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, backend)

# =========================================================================
# Standard CrossScan/Merge (4-way / k=4)
# =========================================================================
if WITH_TRITON:
    @triton.jit
    def triton_cross_scan_flex(
        x, y, x_layout: tl.constexpr, y_layout: tl.constexpr, operation: tl.constexpr, onebyone: tl.constexpr,
        scans: tl.constexpr, BC: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr, DC: tl.constexpr,
        DH: tl.constexpr, DW: tl.constexpr, NH: tl.constexpr, NW: tl.constexpr,
    ):
        i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_h, i_w = (i_hw // NW), (i_hw % NW)
        _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
        _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
        _mask_hw = _mask_h[:, None] & _mask_w[None, :]
        _for_C = min(DC - i_c * BC, BC)

        HWRoute0 = i_h * BH * DW  + tl.arange(0, BH)[:, None] * DW + i_w * BW + tl.arange(0, BW)[None, :]
        HWRoute1 = i_w * BW * DH + tl.arange(0, BW)[None, :] * DH + i_h * BH + tl.arange(0, BH)[:, None] 
        HWRoute2 = (NH - i_h - 1) * BH * DW  + (BH - 1 - tl.arange(0, BH)[:, None]) * DW + (NW - i_w - 1) * BW + (BW - 1 - tl.arange(0, BW)[None, :]) + (DH - NH * BH) * DW + (DW - NW * BW) 
        HWRoute3 = (NW - i_w - 1) * BW * DH  + (BW - 1 - tl.arange(0, BW)[None, :]) * DH + (NH - i_h - 1) * BH + (BH - 1 - tl.arange(0, BH)[:, None]) + (DH - NH * BH) + (DW - NW * BW) * DH 

        if scans == 1:
            HWRoute1 = HWRoute0
            HWRoute2 = HWRoute0
            HWRoute3 = HWRoute0
        elif scans == 2:
            HWRoute1 = HWRoute0
            HWRoute3 = HWRoute2        

        _tmp1 = DC * DH * DW

        y_ptr_base = y + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0 else i_c * BC)
        if y_layout == 0:
            p_y1 = y_ptr_base + HWRoute0
            p_y2 = y_ptr_base + _tmp1 + HWRoute1
            p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2
            p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3
        else:
            p_y1 = y_ptr_base + HWRoute0 * 4 * DC
            p_y2 = y_ptr_base + DC + HWRoute1 * 4 * DC
            p_y3 = y_ptr_base + 2 * DC + HWRoute2 * 4 * DC
            p_y4 = y_ptr_base + 3 * DC + HWRoute3 * 4 * DC       
        
        if onebyone == 0:
            x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
            if x_layout == 0:
                p_x = x_ptr_base + HWRoute0
            else:
                p_x = x_ptr_base + HWRoute0 * DC

            if operation == 0:
                for idxc in range(_for_C):
                    _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                    _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                    _x = tl.load(p_x + _idx_x, mask=_mask_hw)
                    tl.store(p_y1 + _idx_y, _x, mask=_mask_hw)
                    tl.store(p_y2 + _idx_y, _x, mask=_mask_hw)
                    tl.store(p_y3 + _idx_y, _x, mask=_mask_hw)
                    tl.store(p_y4 + _idx_y, _x, mask=_mask_hw)
            elif operation == 1:
                for idxc in range(_for_C):
                    _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                    _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                    _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hw)
                    _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hw)
                    _y3 = tl.load(p_y3 + _idx_y, mask=_mask_hw)
                    _y4 = tl.load(p_y4 + _idx_y, mask=_mask_hw)
                    tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)

        else:
            x_ptr_base = x + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
            if x_layout == 0:
                p_x1 = x_ptr_base + HWRoute0
                p_x2 = p_x1 + _tmp1
                p_x3 = p_x2 + _tmp1
                p_x4 = p_x3 + _tmp1  
            else:
                p_x1 = x_ptr_base + HWRoute0 * 4 * DC
                p_x2 = p_x1 + DC
                p_x3 = p_x2 + DC
                p_x4 = p_x3 + DC        
        
            if operation == 0:
                for idxc in range(_for_C):
                    _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                    _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                    tl.store(p_y1 + _idx_y, tl.load(p_x1 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                    tl.store(p_y2 + _idx_y, tl.load(p_x2 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                    tl.store(p_y3 + _idx_y, tl.load(p_x3 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                    tl.store(p_y4 + _idx_y, tl.load(p_x4 + _idx_x, mask=_mask_hw), mask=_mask_hw)
            else:
                for idxc in range(_for_C):
                    _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                    _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                    tl.store(p_x1 + _idx_x, tl.load(p_y1 + _idx_y), mask=_mask_hw)
                    tl.store(p_x2 + _idx_x, tl.load(p_y2 + _idx_y), mask=_mask_hw)
                    tl.store(p_x3 + _idx_x, tl.load(p_y3 + _idx_y), mask=_mask_hw)
                    tl.store(p_x4 + _idx_x, tl.load(p_y4 + _idx_y), mask=_mask_hw)

    class CrossScanTritonF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
            if one_by_one:
                if in_channel_first:
                    B, _, C, H, W = x.shape
                else:
                    B, H, W, _, C = x.shape
            else:
                if in_channel_first:
                    B, C, H, W = x.shape
                else:
                    B, H, W, C = x.shape
            B, C, H, W = int(B), int(C), int(H), int(W)
            BC, BH, BW = 1, 32, 32
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            
            ctx.in_channel_first = in_channel_first
            ctx.out_channel_first = out_channel_first
            ctx.one_by_one = one_by_one
            ctx.scans = scans
            ctx.shape = (B, C, H, W)
            ctx.triton_shape = (BC, BH, BW, NC, NH, NW)

            y = x.new_empty((B, 4, C, H * W)) if out_channel_first else x.new_empty((B, H * W, 4, C))
            triton_cross_scan_flex[(NH * NW, NC, B)](
                x.contiguous(), y, 
                (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans, 
                BC, BH, BW, C, H, W, NH, NW
            )
            return y
            
        @staticmethod
        def backward(ctx, y: torch.Tensor):
            in_channel_first = ctx.in_channel_first
            out_channel_first = ctx.out_channel_first
            one_by_one = ctx.one_by_one
            scans = ctx.scans
            B, C, H, W = ctx.shape
            BC, BH, BW, NC, NH, NW = ctx.triton_shape
            if one_by_one:
                x = y.new_empty((B, 4, C, H, W)) if in_channel_first else y.new_empty((B, H, W, 4, C))
            else:
                x = y.new_empty((B, C, H, W)) if in_channel_first else y.new_empty((B, H, W, C))
            
            triton_cross_scan_flex[(NH * NW, NC, B)](
                x, y.contiguous(), 
                (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
                BC, BH, BW, C, H, W, NH, NW
            )
            return x, None, None, None, None

    class CrossMergeTritonF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
            if out_channel_first:
                B, _, C, H, W = y.shape
            else:
                B, H, W, _, C = y.shape
            B, C, H, W = int(B), int(C), int(H), int(W)
            BC, BH, BW = 1, 32, 32
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            ctx.in_channel_first = in_channel_first
            ctx.out_channel_first = out_channel_first
            ctx.one_by_one = one_by_one
            ctx.scans = scans
            ctx.shape = (B, C, H, W)
            ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
            if one_by_one:
                x = y.new_empty((B, 4, C, H * W)) if in_channel_first else y.new_empty((B, H * W, 4, C))
            else:
                x = y.new_empty((B, C, H * W)) if in_channel_first else y.new_empty((B, H * W, C))
            triton_cross_scan_flex[(NH * NW, NC, B)](
                x, y.contiguous(), 
                (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
                BC, BH, BW, C, H, W, NH, NW
            )
            return x
            
        @staticmethod
        def backward(ctx, x: torch.Tensor):
            in_channel_first = ctx.in_channel_first
            out_channel_first = ctx.out_channel_first
            one_by_one = ctx.one_by_one
            scans = ctx.scans
            B, C, H, W = ctx.shape
            BC, BH, BW, NC, NH, NW = ctx.triton_shape
            y = x.new_empty((B, 4, C, H, W)) if out_channel_first else x.new_empty((B, H, W, 4, C))
            triton_cross_scan_flex[(NH * NW, NC, B)](
                x.contiguous(), y, 
                (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans,
                BC, BH, BW, C, H, W, NH, NW
            )
            return y, None, None, None, None, None

def cross_scan_fn(x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    CSF = CrossScanTritonF if WITH_TRITON and x.is_cuda and (not force_torch) else CrossScanF
    return CSF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)

def cross_merge_fn(y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    CMF = CrossMergeTritonF if WITH_TRITON and y.is_cuda and (not force_torch) else CrossMergeF
    return CMF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)

# =========================================================================
# Helper Classes
# =========================================================================

class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1).contiguous()).view(B, C, H, W).contiguous()
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C).contiguous()).view(B, H, W, C).contiguous()
        else:
            raise NotImplementedError

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, )
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim,)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0,)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

# =========================================================================
# Mamba Initialization & SS2Dv2 Class
# =========================================================================

class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

class SS2Dv2:
    def __initv2__(
            self,
            d_model=96, d_state=16, ssm_ratio=2.0, dt_rank="auto", act_layer=nn.SiLU,
            d_conv=3, conv_bias=True, dropout=0.0, bias=False,
            dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
            initialize="v0", forward_type="v05", channel_first=False, k_group=4, **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, d_inner, channel_first)

        FORWARD_TYPES = dict(
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        self.k_group = k_group

        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Conv2d_BN(d_model, d_proj)
        
        # Try to initialize activation without inplace to be safe
        try:
             self.act: nn.Module = act_layer(inplace=False)
        except TypeError:
             self.act: nn.Module = act_layer()

        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner, out_channels=d_inner, groups=d_inner, bias=conv_bias,
                kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs,
            )

        self.x_proj = [nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False) for _ in range(k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Conv2d_BN(d_inner, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=k_group,
            )
        elif initialize in ["v1", "v2"]:
             self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
             self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state)) if initialize=="v1" else torch.zeros((k_group * d_inner, d_state)))
             self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((k_group, d_inner, dt_rank)) if initialize=="v1" else 0.1 * torch.rand((k_group, d_inner, dt_rank)))
             self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, d_inner)) if initialize=="v1" else 0.1 * torch.rand((k_group, d_inner)))

    def forward_corev2(
            self, x: torch.Tensor = None, force_fp32=False, ssoflex=True, no_einsum=False,
            selective_scan_backend=None, scan_mode="cross2d", scan_force_torch=False, **kwargs,
    ):
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            if u.device == torch.device("cpu"):
                selective_scan_backend = "torch"
            else:
                selective_scan_backend = "oflex"
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        x_proj_bias = getattr(self, "x_proj_bias", None)
        if self.k_group == 4:
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        else:
            xs = cross_scan_fn_k2(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        
        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L).contiguous(), self.x_proj_weight.view(-1, D, 1).contiguous(),
                                bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L).contiguous(), [R, N, N], dim=2)
            dts = F.conv1d(dts.contiguous().view(B, -1, L).contiguous(), self.dt_projs_weight.view(K * D, -1, 1).contiguous(), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1).contiguous()
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L).contiguous()
        dts = dts.contiguous().view(B, -1, L).contiguous()
        As = -self.A_logs.to(torch.float).exp()
        Ds = self.Ds.to(torch.float)
        Bs = Bs.contiguous().view(B, K, N, L).contiguous()
        Cs = Cs.contiguous().view(B, K, N, L).contiguous()
        delta_bias = self.dt_projs_bias.view(-1).contiguous().to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W).contiguous()

        if self.k_group == 4:
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        else:
            y: torch.Tensor = cross_merge_fn_k2(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        y = y.view(B, -1, H, W).contiguous()
        if not channel_first:
            y = y.view(B, -1, H * W).contiguous().transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1).contiguous()
        y = out_norm(y)

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        
        # [Fix]: RuntimeError: Output 1 of SplitBackward0 is a view and is being modified inplace
        # Break the view dependency using contiguous()
        z = self.act(z.contiguous()) 
        
        x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        out_norm = LayerNorm(d_inner)
        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value

class SS2D(nn.Module, SS2Dv2):
    def __init__(
            self, d_model=96, d_state=16, ssm_ratio=2.0, dt_rank="auto", act_layer=nn.SiLU,
            d_conv=3, conv_bias=True, dropout=0.0, bias=False, dt_min=0.001, dt_max=0.1,
            dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, initialize="v0",
            forward_type="v5", channel_first=False, k_group=4, **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first, k_group=k_group,
        )
        self.__initv2__(**kwargs)

# =========================================================================
# Wavelet Modules
# =========================================================================

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class MBWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',ssm_ratio=1,forward_type="v05",):
        super(MBWTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.global_atten =SS2D(d_model=in_channels, d_state=1,
             ssm_ratio=ssm_ratio, initialize="v2", forward_type=forward_type, channel_first=True, k_group=2)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )

        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.global_atten(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        self.add_module('dwconv3x3',
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels,
                                  bias=False))
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('dwconv1x1',
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels,
                                  bias=False))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))

        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        dwconv3x3, bn1, relu, dwconv1x1, bn2 = self._modules.values()
        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        w1 = dwconv3x3.weight * w1[:, None, None, None]
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5

        fused_dwconv3x3 = nn.Conv2d(w1.size(1) * dwconv3x3.groups, w1.size(0), w1.shape[2:], stride=dwconv3x3.stride,
                                    padding=dwconv3x3.padding, dilation=dwconv3x3.dilation, groups=dwconv3x3.groups,
                                    device=dwconv3x3.weight.device)
        fused_dwconv3x3.weight.data.copy_(w1)
        fused_dwconv3x3.bias.data.copy_(b1)

        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1.weight * w2[:, None, None, None]
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5

        fused_dwconv1x1 = nn.Conv2d(w2.size(1) * dwconv1x1.groups, w2.size(0), w2.shape[2:], stride=dwconv1x1.stride,
                                    padding=dwconv1x1.padding, dilation=dwconv1x1.dilation, groups=dwconv1x1.groups,
                                    device=dwconv1x1.weight.device)
        fused_dwconv1x1.weight.data.copy_(w2)
        fused_dwconv1x1.bias.data.copy_(b2)

        fused_model = nn.Sequential(fused_dwconv3x3, relu, fused_dwconv1x1)
        return fused_model

# =========================================================================
# MobileMamba Components
# =========================================================================

def nearest_multiple_of_16(n):
    if n % 16 == 0:
        return n
    else:
        lower_multiple = (n // 16) * 16
        upper_multiple = lower_multiple + 16
        if (n - lower_multiple) < (upper_multiple - n):
            return lower_multiple
        else:
            return upper_multiple

class MobileMambaModule(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=3, ssm_ratio=1, forward_type="v052d",):
        super().__init__()
        self.dim = dim
        self.global_channels = nearest_multiple_of_16(int(global_ratio * dim))
        if self.global_channels + int(local_ratio * dim) > dim:
            self.local_channels = dim - self.global_channels
        else:
            self.local_channels = int(local_ratio * dim)
        self.identity_channels = self.dim - self.global_channels - self.local_channels
        if self.local_channels != 0:
            self.local_op = DWConv2d_BN_ReLU(self.local_channels, self.local_channels, kernels)
        else:
            self.local_op = nn.Identity()
        if self.global_channels != 0:
            self.global_op = MBWTConv2d(self.global_channels, self.global_channels, kernels, wt_levels=1, ssm_ratio=ssm_ratio, forward_type=forward_type,)
        else:
            self.global_op = nn.Identity()

        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim, dim, bn_weight_init=0,))

    def forward(self, x):
        x1, x2, x3 = torch.split(x, [self.global_channels, self.local_channels, self.identity_channels], dim=1)
        x1 = self.global_op(x1)
        x2 = self.local_op(x2)
        x = self.proj(torch.cat([x1, x2, x3], dim=1))
        return x

class MobileMambaBlockWindow(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=5, ssm_ratio=1, forward_type="v052d",):
        super().__init__()
        self.dim = dim
        self.attn = MobileMambaModule(dim, global_ratio=global_ratio, local_ratio=local_ratio,
                                           kernels=kernels, ssm_ratio=ssm_ratio, forward_type=forward_type,)
    def forward(self, x):
        x = self.attn(x)
        return x

class MobileMambaBlock(torch.nn.Module):
    def __init__(self, type,
                 ed, global_ratio=0.25, local_ratio=0.25,
                 kernels=5,  drop_path=0., has_skip=True, ssm_ratio=1, forward_type="v052d"):
        super().__init__()

        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn0 = Residual(FFN(ed, int(ed * 2)))

        if type == 's':
            self.mixer = Residual(MobileMambaBlockWindow(ed, global_ratio=global_ratio, local_ratio=local_ratio,
                                                       kernels=kernels, ssm_ratio=ssm_ratio,forward_type=forward_type))

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.,))
        self.ffn1 = Residual(FFN(ed, int(ed * 2)))

        self.has_skip = has_skip
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x

# 請確保 MobileMambaBlock, MobileMambaModule, MBWTConv2d, SS2D 等類別已在上下文中定義
# 這裡僅展示將其封裝為 YOLO 模塊的關鍵代碼

# -------------------------------------------------------------------------
# 2. YOLO 專用模塊 (解決 IndexError 與通道傳遞問題)
# -------------------------------------------------------------------------

class MobileMamba_Stem(nn.Module):
    """
    MobileMamba 的 Patch Embedding 層 (Stem)。
    原始設計進行了 4 次 stride=2 的卷積，總縮放倍率為 16。
    為了適配 YOLO，我們將其封裝，輸入為 c1，輸出為 c2。
    """
    def __init__(self, c1, c2):
        super().__init__()
        # 根據原始 MobileMamba 設計：
        # Input -> c2/8 -> c2/4 -> c2/2 -> c2
        # 注意：這要求 c2 必須能被 8 整除
        assert c2 % 8 == 0, f"Stem output channels {c2} must be divisible by 8"
        
        self.stem = torch.nn.Sequential(
            Conv2d_BN(c1, c2 // 8, 3, 2, 1), 
            torch.nn.ReLU(),
            Conv2d_BN(c2 // 8, c2 // 4, 3, 2, 1), 
            torch.nn.ReLU(),
            Conv2d_BN(c2 // 4, c2 // 2, 3, 2, 1), 
            torch.nn.ReLU(),
            Conv2d_BN(c2 // 2, c2, 3, 2, 1)
        )

    def forward(self, x):
        return self.stem(x)

class MobileMamba_Block(nn.Module):
    """
    單個 MobileMamba Block 的封裝。
    YOLO 在解析時會傳入 [c1, c2, *args]。
    """
    def __init__(self, c1, c2, global_ratio=0.8, local_ratio=0.2, kernel=7, ssm_ratio=2, forward_type="v052d"):
        super().__init__()
        # MobileMambaBlock 通常保持通道數不變 (Residual)
        # 如果 c1 != c2，理論上需要 projection，但標準設計中 Block 內部不改變通道
        assert c1 == c2, f"MobileMambaBlock input({c1}) and output({c2}) channels must be same."
        
        # 這裡引用您原始代碼中的 MobileMambaBlock
        # type='s' 是原始代碼中的默認配置
        self.block = MobileMambaBlock(
            type='s', 
            ed=c1, 
            global_ratio=global_ratio, 
            local_ratio=local_ratio,
            kernels=kernel, 
            drop_path=0., 
            has_skip=True, 
            ssm_ratio=ssm_ratio, 
            forward_type=forward_type
        )

    def forward(self, x):
        return self.block(x)

class MobileMamba_Downsample(nn.Module):
    """
    MobileMamba 的下採樣模塊 (Transition Block)。
    包含：ResConv -> ResFFN -> PatchMerging -> ResConv -> ResFFN
    """
    def __init__(self, c1, c2):
        super().__init__()
        # 對應原始代碼中的 down_ops 處理邏輯
        self.pre_process = torch.nn.Sequential(
            Residual(Conv2d_BN(c1, c1, 3, 1, 1, groups=c1)),
            Residual(FFN(c1, int(c1 * 2)))
        )
        self.merging = PatchMerging(c1, c2)
        self.post_process = torch.nn.Sequential(
            Residual(Conv2d_BN(c2, c2, 3, 1, 1, groups=c2)),
            Residual(FFN(c2, int(c2 * 2)))
        )

    def forward(self, x):
        x = self.pre_process(x)
        x = self.merging(x)
        x = self.post_process(x)
        return x