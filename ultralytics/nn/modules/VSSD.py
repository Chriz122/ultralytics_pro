import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from einops import rearrange, repeat
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg


import math
import copy
from functools import partial


class tTensor(torch.Tensor):
    @property
    def shape(self):
        shape = super().shape
        return tuple([int(s) for s in shape])


to_ttensor = lambda *args: tuple([tTensor(x) for x in args]) if len(args) > 1 else tTensor(args[0])


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

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class Stem(nn.Module):
    r""" Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x)
        _, _, H, W = x.shape
        x = self.conv2(x) + x
        x = self.conv3(x)
        _, _, H_out, W_out = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H_out, W_out

class SimpleStem(nn.Module):
    r'''
    Simple Stem

    '''

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.conv1 = ConvLayer(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x)
        _, _, H_out, W_out = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, H_out, W_out


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """

    def __init__(self, input_resolution, dim, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        in_channels = dim
        out_channels = 2 * dim
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2))
        _, _, H_out, W_out = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        return x, H_out, W_out

class SimplePatchMerging(nn.Module):
    r""" Simple Patch Merging Layer.

        Args:
            input_resolution (tuple[int]): Resolution of input feature.
            dim (int): Number of input channels.
        """

    def __init__(self, input_resolution, dim, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        in_channels = dim
        out_channels = 2 * dim
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels), kernel_size=3,stride=2, padding=1, norm=None),
        )
        self.norm = nn.LayerNorm(out_channels)
    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2))
        _, _, H_out, W_out = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm(x)
        return x, H_out, W_out


class StandardAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.inner_dim = inner_dim


    def forward(self, x, H, W):
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
    r""" VMAMBA2 Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256,
                 linear_attn_duality=False, d_state = 64, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        if kwargs.get('attn_type', 'mamba2') == 'standard':
            self.attn = StandardAttention(dim=dim, heads=num_heads, dim_head=dim // num_heads, dropout=drop)
        elif kwargs.get('attn_type', 'mamba2') == 'mamba2':
            self.attn = Mamba2(d_model=dim, expand=ssd_expansion, headdim= dim*ssd_expansion // num_heads,
                                ngroups=ssd_ngroups, chunk_size=ssd_chunk_size,
                                linear_attn_duality=linear_attn_duality, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)

        x = self.attn(x, H, W)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    """ A basic VMAMBA2 layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256, linear_attn_duality=False, d_state=64, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            VMAMBA2Block(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer,
                      ssd_expansion=ssd_expansion, ssd_ngroups=ssd_ngroups, ssd_chunk_size=ssd_chunk_size,
                      linear_attn_duality=linear_attn_duality, d_state=d_state, **kwargs)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)
        
        # We return the feature map before downsampling, and the downsampled output separately
        if self.downsample is not None:
            x_down, H_down, W_down = self.downsample(x, H, W)
            return x, H, W, x_down, H_down, W_down
        else:
            return x, H, W, x, H, W

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class VMAMBA2(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=64, depths=[2, 4, 12, 4], num_heads=[2, 4, 8, 16],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.2,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), use_checkpoint=False,
                 ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256,
                 linear_attn_duality= True, d_state=64, out_indices=(0, 1, 2, 3), **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.in_chans = in_chans
        self.out_indices = out_indices

        self.simple_downsample = kwargs.get('simple_downsample', False)
        self.simple_patch_embed = kwargs.get('simple_patch_embed', False)
        self.attn_types = kwargs.get('attn_types', ['mamba2', 'mamba2', 'mamba2', 'standard'])

        if self.simple_patch_embed:
            self.patch_embed = SimpleStem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        if self.simple_downsample:
            PatchMergingBlock = SimplePatchMerging
        else:
            PatchMergingBlock = PatchMerging

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs['attn_type'] = self.attn_types[i_layer]
            layer_dim = int(embed_dim * 2 ** i_layer)
            input_res = (patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer))
            
            layer = BasicLayer(dim=layer_dim,
                               input_resolution=input_res,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMergingBlock if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               ssd_expansion=ssd_expansion,
                               ssd_ngroups=ssd_ngroups,
                               ssd_chunk_size=ssd_chunk_size,
                               linear_attn_duality = linear_attn_duality,
                               d_state = d_state,
                               **kwargs)
            self.layers.append(layer)

            if i_layer in out_indices:
                out_norm = norm_layer(layer_dim)
                self.add_module(f'outnorm{i_layer}', out_norm)
        
        # Classification head (kept for versatility, but not used in the backbone-style forward pass)
        self.head_norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        
        # --- Add width_list calculation ---
        self.width_list = []
        try:
            self.eval() 
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size)

            with torch.no_grad():
                 features = self.forward(dummy_input)

            self.width_list = [f.size(1) for f in features]
            self.train()
        except Exception as e:
            print(f"Error during dummy forward pass for width_list calculation: {e}")
            print("Setting width_list to embed_dims as fallback.")
            self.width_list = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
            self.train()


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        no_decay = set()
        for name, param in self.named_parameters():
            if 'norm' in name or 'bias' in name or 'A_log' in name or 'D' in name:
                 no_decay.add(name)
        return no_decay

    def forward(self, x):
        outs = []
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            # Pass through the layer, get pre-downsample and post-downsample outputs
            x_out, H_out, W_out, x, H, W = layer(x, H, W)

            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(x_out)
                B, L, C = out.shape
                out = out.view(B, H_out, W_out, C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

# --- Factory Functions ---

def vssd_micro_e300(pretrained=False, **kwargs):
    model = VMAMBA2(
        embed_dim=48,
        depths=[2, 2, 8, 4],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.1,
        simple_downsample=False,
        simple_patch_embed=False,
        ssd_expansion=2,
        ssd_chunk_size=256,
        linear_attn_duality=True,
        ssd_positve_dA=True,
        attn_types=['mamba2', 'mamba2', 'mamba2', 'standard'],
        d_state=48,
        **kwargs)
    model.default_cfg = _cfg()
    return model

def vssd_tiny_e300(pretrained=False, **kwargs):
    model = VMAMBA2(
        embed_dim=64,
        depths=[2, 4, 8, 4],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.2,
        simple_downsample=False,
        simple_patch_embed=False,
        ssd_expansion=2,
        ssd_chunk_size=256,
        linear_attn_duality=True,
        ssd_positve_dA=True,
        attn_types=['mamba2', 'mamba2', 'mamba2', 'standard'],
        **kwargs)
    model.default_cfg = _cfg()
    return model

def vssd_small_e300(pretrained=False, **kwargs):
    model = VMAMBA2(
        embed_dim=64,
        depths=[3, 4, 21, 5],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.4,
        simple_downsample=False,
        simple_patch_embed=False,
        ssd_expansion=2,
        ssd_chunk_size=256,
        linear_attn_duality=True,
        ssd_positve_dA=True,
        attn_types=['mamba2', 'mamba2', 'mamba2', 'standard'],
        **kwargs)
    model.default_cfg = _cfg()
    return model

def vssd_base_e300(pretrained=False, **kwargs):
    model = VMAMBA2(
        embed_dim=96,
        depths=[3, 4, 21, 5],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.6,
        simple_downsample=False,
        simple_patch_embed=False,
        ssd_expansion=2,
        ssd_chunk_size=256,
        linear_attn_duality=True,
        ssd_positve_dA=True,
        attn_types=['mamba2', 'mamba2', 'mamba2', 'standard'],
        **kwargs)
    model.default_cfg = _cfg()
    return model

if __name__ == '__main__':
    img_h, img_w = 640, 640
    print("--- Creating V-MAMBA-V2 Tiny model ---")
    model = vssd_tiny_e300(img_size=img_h)
    print("Model created successfully.")
    print("Calculated width_list:", model.width_list)

    # Test forward pass
    input_tensor = torch.rand(2, 3, img_h, img_w)
    print(f"\n--- Testing V-MAMBA-V2 Tiny forward pass (Input: {input_tensor.shape}) ---")

    model.eval()
    try:
        with torch.no_grad():
            output_features = model(input_tensor)
        print("Forward pass successful.")
        print("Output is a list of feature maps as expected.")
        print("Output feature shapes:")
        for i, features in enumerate(output_features):
            print(f"Stage {i+1}: {features.shape}")

        # Verify width_list matches runtime output
        runtime_widths = [f.size(1) for f in output_features]
        print("\nRuntime output feature channels:", runtime_widths)
        assert model.width_list == runtime_widths, "Width list mismatch!"
        print("Width list verified successfully.")

        # --- Test deepcopy ---
        print("\n--- Testing deepcopy ---")
        copied_model = copy.deepcopy(model)
        print("Deepcopy successful.")

        # Optional: Test copied model forward pass
        with torch.no_grad():
             output_copied = copied_model(input_tensor)
        print("Copied model forward pass successful.")
        assert len(output_copied) == len(output_features)
        for i in range(len(output_features)):
             assert output_copied[i].shape == output_features[i].shape
        print("Copied model output shapes verified.")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()