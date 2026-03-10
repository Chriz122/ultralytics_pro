import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat

# 嘗試導入 mamba，如果沒有安裝則報錯提示
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    print("Error: mamba_ssm is not installed. Please install it to use VSSD.")
    mamba_chunk_scan_combined = None
    
    
class tTensor(torch.Tensor):
    @property
    def shape(self):
        shape = super().shape
        return tuple([int(s) for s in shape])


to_ttensor = lambda *args: tuple([tTensor(x) for x in args]) if len(args) > 1 else tTensor(args[0])

# --- 基礎組件 (保持原樣或微調) ---

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class StandardAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, H, W):
        # x: B, L, C
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_conv=3, #default to 3 for 2D
        conv_init=None,
        expand=2,
        headdim=64, #default to 64
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="silu", #default to silu
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=False, #default to False, for custom implementation
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        linear_attn_duality=False,
        d_state = 64,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        if ngroups == -1:
            ngroups = self.d_inner // self.headdim #equivalent to multi-head attention
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        #convert chunk_size to triton.language.int32
        self.chunk_size = chunk_size#torch.tensor(chunk_size,dtype=torch.int32)
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.ssd_positve_dA = kwargs.get('ssd_positve_dA', True) #default to False, ablation for linear attn duality
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias, **factory_kwargs) #

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state


        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        #linear attention duality
        self.linear_attn_duality = linear_attn_duality
        self.kwargs = kwargs

    def non_casual_linear_attn(self, x, dt, A, B, C, D, H=None, W=None):
        '''
        non-casual attention duality of mamba v2
        x: (B, L, H, D), equivalent to V in attention
        dt: (B, L, nheads)
        A: (nheads) or (d_inner, d_state)
        B: (B, L, d_state), equivalent to K in attention
        C: (B, L, d_state), equivalent to Q in attention
        D: (nheads), equivalent to the skip connection
        '''

        batch, seqlen, head, dim = x.shape
        dstate = B.shape[2]
        V = x.permute(0, 2, 1, 3) # (B, H, L, D)
        dt = dt.permute(0, 2, 1) # (B, H, L)
        dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
        if self.ssd_positve_dA: dA = -dA

        V_scaled = V * dA
        K = B.view(batch, 1, seqlen, dstate)# (B, 1, L, D)

        if self.ngroups == 1:
            ## get kv via transpose K and V
            KV = K.transpose(-2, -1) @ V_scaled # (B, H, dstate, D)
            Q = C.view(batch, 1, seqlen, dstate)#.repeat(1, head, 1, 1)
            x = Q @ KV # (B, H, L, D)
            x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
        else:
            assert head % self.ngroups == 0
            dstate = dstate // self.ngroups
            K = K.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)
            V_scaled = V_scaled.view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
            Q = C.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)

            KV = K.transpose(-2, -1) @ V_scaled # (B, H//g, g, dstate, D)
            x = Q @ KV # (B, H//g, g, L, D)
            V_skip = (V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)).view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
            x = x + V_skip # (B, H//g, g, L, D)
            x = x.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seqlen, head, dim) # (B, L, H, D)
            x = x.contiguous()

        return x


    def forward(self, u, H, W, seq_idx=None):
        """
        u: (B,C,H,W)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)


        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        assert self.activation in ["silu", "swish"]


        #2D Convolution
        xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x, dt, A, B, C = to_ttensor(x, dt, A, B, C)
        if self.linear_attn_duality:
            y = self.non_casual_linear_attn(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt, A, B, C, self.D, H, W
            )
        else:
            if self.kwargs.get('bidirection', False):
                x = to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)).chunk(2, dim=-2)
                B = to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
                C = to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
                dt = dt.chunk(2, dim=-1)
                A, D = A.chunk(2, dim=-1), self.D.chunk(2,dim=-1)
                y_forward = mamba_chunk_scan_combined(
                    x[0], dt[0], A[0], B[0], C[0], chunk_size=self.chunk_size, D=D[0], z=None, seq_idx=seq_idx,
                    initial_states=initial_states, **dt_limit_kwargs
                )
                y_backward = mamba_chunk_scan_combined(
                    x[1].flip(1), dt[1].flip(1), A[1], B[1].flip(1), C[1].flip(1), chunk_size=self.chunk_size, D=D[1], z=None, seq_idx=seq_idx,
                    initial_states=initial_states, **dt_limit_kwargs
                )
                y = torch.cat([y_forward, y_backward.flip(1)], dim=-2)
            else:
                y = mamba_chunk_scan_combined(
                    to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)),
                    to_ttensor(dt),
                    to_ttensor(A),
                    to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)),
                    to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)),
                    chunk_size=self.chunk_size,
                    D=to_ttensor(self.D),
                    z=None,
                    seq_idx=seq_idx,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )
        y = rearrange(y, "b l h p -> b l (h p)")

        y = self.norm(y)
        y = y*z
        out = self.out_proj(y)
        return out


class VMAMBA2Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type='mamba2', 
                 ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256, linear_attn_duality=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        
        if attn_type == 'standard':
            self.attn = StandardAttention(dim=dim, heads=num_heads, dim_head=dim // num_heads, dropout=drop)
        else:
            self.attn = Mamba2(d_model=dim, expand=ssd_expansion, headdim=dim*ssd_expansion // num_heads,
                               ngroups=ssd_ngroups, chunk_size=ssd_chunk_size,
                               linear_attn_duality=linear_attn_duality, **kwargs)
                               
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        # x: B, L, C
        B, L, C = x.shape
        # CPE1
        feat = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.cpe1(feat).flatten(2).permute(0, 2, 1)
        
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, H, W)
        x = shortcut + self.drop_path(x)
        
        # CPE2
        feat = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.cpe2(feat).flatten(2).permute(0, 2, 1)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# --- YOLO Compatible Modules (Wrapper) ---

class Stem_VSSD(nn.Module):
    """
    VSSD Stem layer adapter for YOLO.
    Input: (B, C_in, H, W)
    Output: (B, C_out, H/4, W/4)
    """
    def __init__(self, c1, c2, k=4, s=4): # k=patch_size
        super().__init__()
        self.in_chans = c1
        self.embed_dim = c2
        self.conv1 = nn.Conv2d(c1, c2 // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c2 // 2, c2 // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(c2 // 2, c2 // 2, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2 // 2, c2 * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(c2 * 4, c2, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # input x: [B, 3, H, W]
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        # output x: [B, Embed_Dim, H/4, W/4]
        return x

class PatchMerging_VSSD(nn.Module):
    """
    Patch Merging Layer for YOLO.
    Input: (B, C1, H, W)
    Output: (B, 2*C1, H/2, W/2)
    Note: c2 argument is required by YOLO parser but usually c2 = 2*c1. 
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.dim = c1
        # VSSD PatchMerging implementation
        self.conv = nn.Sequential(
            nn.Conv2d(c1, int(c2 * 4), kernel_size=1, bias=False),
            nn.Conv2d(int(c2 * 4), int(c2 * 4), kernel_size=3, stride=2, padding=1, groups=int(c2 * 4), bias=False),
            nn.Conv2d(int(c2 * 4), c2, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.conv(x)

class VSSLayer(nn.Module):
    """
    A stage of VSSD blocks (VMAMBA2Block).
    Arguments:
        c1 (int): Input channels
        c2 (int): Output channels (should be same as c1)
        n (int): Number of blocks (depth) - Passed from YOLO 'repeats'
        num_heads (int): Number of attention heads
        attn_type (str): 'mamba2' or 'standard'
    """
    def __init__(self, c1, c2, n=1, num_heads=8, attn_type='mamba2', 
                 drop_path=0., mlp_ratio=4.0, d_state=64):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        assert c1 == c2, "Input and Output channels must be same for VSSLayer (use PatchMerging to change)"
        
        self.blocks = nn.ModuleList([
            VMAMBA2Block(
                dim=c1, 
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=0.0,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                attn_type=attn_type,
                ssd_expansion=2,
                d_state=d_state
            ) for i in range(n)
        ])

    def forward(self, x):
        # YOLO inputs: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Flatten for Mamba: (B, L, C)
        x = x.flatten(2).transpose(1, 2)
        
        for blk in self.blocks:
            x = blk(x, H, W)
            
        # Reshape back to YOLO format: (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x