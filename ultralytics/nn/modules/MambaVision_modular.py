import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange, repeat

# --- 核心組件 (保留原邏輯) ---
try:
    import selective_scan_cuda
except ImportError:
    selective_scan_cuda = None

class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        if u.stride(-1) != 1: u = u.contiguous()
        if delta.stride(-1) != 1: delta = delta.contiguous()
        if D is not None: D = D.contiguous()
        if B.stride(-1) != 1: B = B.contiguous()
        if C.stride(-1) != 1: C = C.contiguous()
        if z is not None and z.stride(-1) != 1: z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        
        if selective_scan_cuda is None:
            raise ImportError("selective_scan_cuda extension not found.")
            
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]
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
            z, out = None, None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1: dout = dout.contiguous()
        
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus, False
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD if D is not None else None, dz, ddelta_bias if delta_bias is not None else None, None, None)

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)

class MambaVisionMixer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant": nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random": nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        dt = torch.exp(torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        with torch.no_grad(): self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner//2).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(self.d_inner//2, self.d_inner//2, bias=conv_bias, kernel_size=d_conv, groups=self.d_inner//2, padding='same', **factory_kwargs)
        self.conv1d_z = nn.Conv1d(self.d_inner//2, self.d_inner//2, bias=conv_bias, kernel_size=d_conv, groups=self.d_inner//2, padding='same', **factory_kwargs)

    def forward(self, hidden_states):
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        
        # 處理 Ultralytics 的 CPU init -> GPU move 問題
        if x.device.type == 'cpu' and torch.cuda.is_available() and selective_scan_cuda is not None:
             # 如果目前在 CPU 但有 CUDA 環境，可能是初始化階段，避免調用 selective_scan_cuda
             # 或者簡單地跳過計算 (如果是 dummy pass)
             pass 

        x = F.silu(self.conv1d_x(x))
        z = F.silu(self.conv1d_z(z))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, dt, -torch.exp(self.A_log.float()), B, C, self.D.float(), z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        y = torch.cat([y, z], dim=1)
        return self.out_proj(rearrange(y, "b d l -> b l d"))

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class Block(nn.Module):
    def __init__(self, dim, num_heads, counter, transformer_blocks, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(dim, num_heads=num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.)
        else:
            self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(0)
        )

    def forward(self, x):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(dim)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.norm2(self.conv2(self.act1(self.norm1(self.conv1(x)))))
        return input + self.drop_path(x)

def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)

def window_reverse(windows, window_size, H, W):
    C = windows.shape[-1]
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    return x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)

# --- YOLO 模塊化封裝 ---

class MV_Stem(nn.Module):
    """ MambaVision 的 Stem 層 """
    def __init__(self, c1, c2, in_dim=32):
        super().__init__()
        # c1: input channels (3), c2: output channels (embedding dim)
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(c1, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, c2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c2, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_down(self.proj(x))

class MV_Downsample(nn.Module):
    """ MambaVision 的 Downsample 層 """
    def __init__(self, c1, c2):
        super().__init__()
        self.reduction = nn.Conv2d(c1, c2, 3, 2, 1, bias=False)

    def forward(self, x):
        return self.reduction(x)

class MV_Stage(nn.Module):
    """ 
    MambaVision 的 Stage 
    參數順序必須與 YAML args 對應:
    [c1, c2, n, window_size, num_heads, is_conv]
    """
    def __init__(self, c1, c2, n=1, window_size=0, num_heads=0, is_conv=False):
        super().__init__()
        self.c2 = c2
        self.is_conv = is_conv
        self.window_size = window_size
        
        # MambaVision 的邏輯：後半部分是 Transformer Block
        transformer_blocks = []
        if not is_conv:
             # 計算哪些層是 Transformer (根據官方代碼邏輯)
             start_idx = n // 2 + 1 if n % 2 != 0 else n // 2
             transformer_blocks = list(range(start_idx, n))

        if is_conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=c2) for _ in range(n)])
        else:
            self.blocks = nn.ModuleList([
                Block(dim=c2, num_heads=num_heads, counter=i, transformer_blocks=transformer_blocks)
                for i in range(n)
            ])

    def forward(self, x):
        # 處理 Ultralytics 初始化 CPU->GPU 的兼容性問題
        if x.device.type == 'cpu' and torch.cuda.is_available() and selective_scan_cuda is not None:
             try:
                self.to('cuda')
                x = x.to('cuda')
                res = self._forward_impl(x)
                self.to('cpu')
                return res.to('cpu')
             except Exception:
                 pass

        return self._forward_impl(x)

    def _forward_impl(self, x):
        if self.is_conv:
            for blk in self.blocks:
                x = blk(x)
        else:
            B, C, H, W = x.shape
            # Window Partition
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
            
            _, _, Hp, Wp = x.shape
            x = window_partition(x, self.window_size)
            
            for blk in self.blocks:
                x = blk(x)
            
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        return x