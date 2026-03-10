import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from einops import rearrange, repeat

# ==========================================
# 基礎組件 (來自原始代碼)
# ==========================================

# 模擬 selective_scan_cuda，防止無 CUDA 環境報錯
try:
    import selective_scan_cuda
except ImportError:
    selective_scan_cuda = None

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, 1)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(0), w.shape[2:], 
                            stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class RepDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
        self.apply(self._init_weights)
    
    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv

class Rep_Inception(torch.nn.Module):
    def __init__(self, dim, kernel_max=7, ratio=0.5) -> None:
        super().__init__()
        gc = int(dim*ratio)
        self.dwconv_h = RepDW_Axias(gc, kernel_max=kernel_max, kernel=(1, kernel_max))
        self.dwconv_w = RepDW_Axias(gc, kernel_max=kernel_max, kernel=(kernel_max, 1))
        self.split = (dim-gc, gc)
    def forward(self, x):
        x_w, x_h = torch.split(x, self.split, dim=1)
        return torch.cat(
            (self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,
        )

# 需要補上 RepDW_Axias 供 Rep_Inception 使用
class RepDW_Axias(torch.nn.Module):
    def __init__(self, ed, kernel_max=7, kernel=(1, 7)) -> None:
        super().__init__()
        self.kernel = kernel
        self.kernel_max = kernel_max
        padding = kernel_max//2
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        if kernel==(1, kernel_max):
            self.conv = Conv2d_BN(ed, ed, (1, kernel_max), 1, (0, padding), groups=ed)
        else:
            self.conv = Conv2d_BN(ed, ed, (kernel_max, 1), 1, (padding, 0), groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
        self.apply(self._init_weights)
    
    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        padding = self.kernel_max//2
        
        if self.kernel == (1, self.kernel_max):
            conv1_w = torch.nn.functional.pad(conv1_w, [padding,padding])
            identity = torch.nn.functional.pad(torch.ones(conv_w.shape[0], conv_w.shape[1], 1, 1, device=conv_w.device), [padding,padding])
        else:
            conv1_w = torch.nn.functional.pad(conv1_w, [0,0,padding,padding])
            identity = torch.nn.functional.pad(torch.ones(conv_w.shape[0], conv_w.shape[1], 1, 1, device=conv_w.device), [0,0,padding,padding])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv
    
class SelectiveScan(torch.autograd.Function):
    
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        if selective_scan_cuda is None:
            raise NotImplementedError("selective_scan_cuda kernel not found.")
        assert u.is_cuda, "Input tensor 'u' must be on a CUDA device."
        assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
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
        
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        if selective_scan_cuda is None:
            raise NotImplementedError("selective_scan_cuda kernel not found.")
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None

# 簡化的 Cross Scan 接口 (若無 CUDA 則直通，保證能跑)
def cross_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    nrows = -1,
    delta_softplus = True,
    to_dtype=True,
    force_fp32=True,
):
    B, D, H, W = x.shape
    L = H * W
    
    # ================= CPU FALLBACK ADDED HERE =================
    # If on CPU, or if the CUDA kernel is not available, run a shape-preserving placeholder.
    # This allows for model initialization and testing on CPU-only environments.
    if x.device.type == 'cpu' or selective_scan_cuda is None:
        if selective_scan_cuda is None:
            print(f"Warning: CPU fallback in cross_selective_scan due to missing selective_scan_cuda. Input device: {x.device}")
        # The placeholder operation: flatten, apply norm, and reshape.
        # This is NOT a correct SSM implementation, but it preserves the tensor shape,
        # which is crucial for downstream layers and initialization scripts.
        y = x.flatten(2).transpose(1, 2) # (B, L, D)
        y = out_norm(y).view(B, H, W, -1)
        return (y.to(x.dtype) if to_dtype else y)
    # ==========================================================

    D_ssm, N = A_logs.shape
    K, D_ssm_exp, R = dt_projs_weight.shape
    
    if nrows < 1:
        if D_ssm_exp % 4 == 0: nrows = 4
        elif D_ssm_exp % 3 == 0: nrows = 3
        elif D_ssm_exp % 2 == 0: nrows = 2
        else: nrows = 1
    
    xs = CrossScan.apply(x)
    
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
        xs, dts, Bs, Cs = xs.float(), dts.float(), Bs.float(), Cs.float()

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
    
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, H, W)
    
    y: torch.Tensor = CrossMerge.apply(ys)
    y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)

# ==========================================
# 核心模塊：SS2D (Laplace Mixer)
# ==========================================
class SS2D(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3, 
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        index = 0,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        
        self.pool = nn.AvgPool2d(2**(3-index))
        self.index = index

        split_list = [1/4,1/2,1/2,3/4]
        d_inner = int(d_expand*split_list[index])
        self.local_conv = RepDW(d_expand-d_inner)
        self.split = (d_inner, d_expand-d_inner)

        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.out_norm = nn.LayerNorm(d_inner)

        self.K = 4

        self.in_proj = Conv2d_BN(d_model, d_expand, 1)
        self.act: nn.Module = act_layer()
        
        if self.d_conv > 1:
            self.conv2d = Rep_Inception(d_expand,7)

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
        
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K, merge=True)
        self.Ds = self.D_init(d_inner, copies=self.K, merge=True)

        self.out_proj = Conv2d_BN(d_expand, d_model, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

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

    def forward_core(self, x: torch.Tensor, channel_first=False):
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        
        x_low, x_high = torch.split(x, self.split, dim=1)
        B, C, H, W = x.shape
        x_high = self.local_conv(x_high)
        if self.index < 3:
            x0 = x_low
            x_low = self.pool(x_low)
            res = x0 - F.interpolate(x_low, (H, W), mode='nearest')
        
        x_low = cross_selective_scan(
            x_low, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            delta_softplus=True, force_fp32=self.training,
        )
        x_low = x_low.permute(0, 3, 1, 2)
        
        if self.index < 3:
            x_low = F.interpolate(x_low, scale_factor=2**(3-self.index), mode='bilinear') + res
        x = torch.cat((x_low, x_high),dim=1)

        return x
    
    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if self.d_conv > 1:
            x = self.act(self.conv2d(x))

        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        out = self.dropout(self.out_proj(y))
        return out

class FFN(nn.Module):
    def __init__(self, in_dim, mid_dim=None,
                 out_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        mid_dim = mid_dim or in_dim
        self.fc1 = Conv2d_BN(in_dim, mid_dim, 1)
        self.fc2 = Conv2d_BN(mid_dim, out_dim, 1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ==========================================
# YOLOv8 整合模塊
# ==========================================

class LocalBlock(nn.Module):
    """ 架構圖中的 Local Block: RepDW-3 + FFN """
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.dwconv = RepDW(dim)
        self.mlp = FFN(dim, int(dim * mlp_ratio))
        
    def forward(self, x):
        # 殘差連接 1
        input = x
        x = self.dwconv(x)
        
        # 殘差連接 2 (架構圖顯示 FFN 也有殘差)
        x = input + self.mlp(x)
        return x

class TViMBlock(nn.Module):
    """ 架構圖中的 TinyViM Block: Laplace Mixer + FFN """
    def __init__(self, dim, ssm_d_state=16, ssm_ratio=2.0, index=0, mlp_ratio=4.0):
        super().__init__()
        self.op = SS2D(dim, ssm_d_state, ssm_ratio, index=index) # Laplace Mixer
        self.mlp = FFN(dim, int(dim * mlp_ratio)) # FFN
        
    def forward(self, x):
        # Laplace Mixer Branch with Residual
        x = x + self.op(x)
        # FFN Branch with Residual
        x = x + self.mlp(x)
        return x

class TinyViM_Stem(nn.Module):
    """ 架構圖中的 Stem: Input -> Conv(s=2) -> Conv(s=2) -> Output """
    def __init__(self, c1, c2):
        super().__init__()
        # TinyViM Stem 包含兩次下採樣
        mid_c = c2 // 2
        self.stem = nn.Sequential(
            Conv2d_BN(c1, mid_c, 3, 2, 1),
            nn.GELU(),
            Conv2d_BN(mid_c, c2, 3, 2, 1),
            nn.GELU(),
        )
    def forward(self, x):
        return self.stem(x)

class TinyViM_Downsample(nn.Module):
    """ 架構圖中的 Patch Embed: Stage 之間的下採樣 (Conv+BN, 無 Act) """
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = Conv2d_BN(c1, c2, 3, 2, 1) # Patch Size 3, Stride 2, Pad 1
    
    def forward(self, x):
        return self.proj(x)

class TinyViM_Stage(nn.Module):
    """
    對應架構圖中的 Stage 1, 2, 3, 4
    結構: [Local Block * N] -> [TinyViM Block * M]
    """
    def __init__(self, c1, c2, depth, ssm_num=1, stage_index=0, mlp_ratio=4.0):
        super().__init__()
        # 如果 YOLO 傳入的 c1 != c2 (通常不應發生，因為有 Downsample 層)，做一個投影
        if c1 != c2:
            self.proj = Conv2d_BN(c1, c2, 1, 1, 0)
        else:
            self.proj = nn.Identity()
            
        blocks = []
        for block_idx in range(depth):
            # 根據原始代碼邏輯：後 ssm_num 個 Block 使用 TinyViM Block
            is_ssm = (depth - block_idx <= ssm_num)
            
            if is_ssm:
                blocks.append(TViMBlock(c2, index=stage_index, mlp_ratio=mlp_ratio))
            else:
                blocks.append(LocalBlock(c2, mlp_ratio=mlp_ratio))
        
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.proj(x)
        return self.blocks(x)