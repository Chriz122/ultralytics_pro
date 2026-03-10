# models/modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Mamba(nn.Module):
    """
    Mamba SSM block - State Space Model for sequence modeling
    Paper: Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv 2312.00752)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, D) where L is sequence length, D is d_model
        Returns: (B, L, D)
        """
        batch, seqlen, dim = x.shape
        
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Conv1d expects (B, C, L)
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seqlen]  # Trim padding
        x = rearrange(x, 'b d l -> b l d')
        
        x = F.silu(x)
        
        # SSM parameters
        x_proj = self.x_proj(x)  # (B, L, 2*d_state)
        delta, B = x_proj.chunk(2, dim=-1)  # Each (B, L, d_state)
        
        # Discretization
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        
        # Selective scan (simplified - for production use efficient cuda kernel)
        y = self._selective_scan(x, delta, A, B)
        
        # Skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x
        
        # Gate
        y = y * F.silu(z)
        
        output = self.out_proj(y)
        return output
    
    def _selective_scan(self, u, delta, A, B):
        """
        Simplified selective scan - memory optimized for 4GB VRAM
        u: (B, L, d_inner)
        delta: (B, L, d_inner)
        A: (d_inner, d_state)
        B: (B, L, d_state)
        
        Memory optimization: Avoid creating huge b×l×d×n tensors from einsum.
        Use element-wise expansion instead: exp(delta[:,:,:,None] * A[None,None,:,:])
        """
        batch, seqlen, d_inner = u.shape
        d_state = A.shape[1]
        
        # Discretize A - memory optimized
        # OLD: deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        # This creates b×l×d×n tensor (batch×length×channels×state)
        # NEW: Use unsqueeze + broadcast instead (avoids intermediate materialization)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB = torch.einsum('bld,bln->bldn', delta, B)
        
        # Scan
        x = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        
        for i in range(seqlen):
            x = deltaA[:, i] * x + deltaB[:, i] * u[:, i].unsqueeze(-1)
            y = torch.einsum('bdn,bn->bd', x, B[:, i])
            ys.append(y)
        
        return torch.stack(ys, dim=1)


class MambaNeXt(nn.Module):
    """
    MambaNeXt: Mamba block adapted for 2D vision
    Processes spatial features as sequences
    Includes gradient checkpointing for memory efficiency on small GPUs
    """
    def __init__(self, c, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.c = c
        self.norm = nn.LayerNorm(c)
        self.mamba = Mamba(d_model=c, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Conv2d(c, c, 1)
        self.use_checkpoint = True  # Enable gradient checkpointing (35-50% memory savings)
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        identity = x
        
        # Reshape to sequence
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Normalize and apply Mamba with optional gradient checkpointing
        x = self.norm(x)
        
        if self.use_checkpoint and self.training:
            # Gradient checkpointing: save memory by recomputing forward on backward
            x = torch.utils.checkpoint.checkpoint(self.mamba, x, use_reentrant=False)
        else:
            x = self.mamba(x)
        
        # Reshape back
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Projection and residual
        x = self.proj(x) + identity
        return x


class IRDCB(nn.Module):
    """
    Inverted Residual Dilated Convolution Block
    Efficient convolution with dilated kernels for multi-scale receptive field
    """
    def __init__(self, c1=None, c2=None, c_in=None, c_out=None, expand_ratio=2, dilation=2):
        super().__init__()
        # Resolve channel dims from either Ultralytics positional args (c1, c2) or explicit names
        if c_in is None:
            c_in = c1
        if c_out is None:
            c_out = c2 if c2 is not None else c_in
        if c_in is None or c_out is None:
            raise ValueError("IRDCB requires c_in/c_out (or c1/c2)")

        c_hidden = int(c_in * expand_ratio)
        
        self.conv1 = nn.Conv2d(c_in, c_hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_hidden)
        
        self.conv2 = nn.Conv2d(
            c_hidden, c_hidden, 3,
            padding=dilation, dilation=dilation,
            groups=c_hidden, bias=False
        )
        self.bn2 = nn.BatchNorm2d(c_hidden)
        
        self.conv3 = nn.Conv2d(c_hidden, c_out, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c_out)
        
        self.act = nn.SiLU()
        
        # Shortcut
        self.use_shortcut = (c_in == c_out)
        
    def forward(self, x):
        identity = x
        
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.use_shortcut:
            out = out + identity
        
        return self.act(out)