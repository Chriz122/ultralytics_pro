import torch 
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from .block import C2f, C3k

__all__ = ("DiTBlock", "C2PSA_DiTBlock", "C3k2_DiTBlock")


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads=16, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        x = x.view(batch, channels, height * width).permute(0, 2, 1).clone()  # Reshape to (batch, height*width, channels)

        # Split modulation parameters
        modulation = self.adaLN_modulation(x)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        # Apply modulations and gating
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        gate_msa = torch.sigmoid(gate_msa.mean(dim=-1, keepdim=True))
        x = x + gate_msa * self.attn(attn_input)

        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        gate_mlp = torch.sigmoid(gate_mlp.mean(dim=-1, keepdim=True))
        x = x + gate_mlp * self.mlp(mlp_input)

        x = x.permute(0, 2, 1).view(batch, channels, height, width)  # Restore spatial dimensions
        return x


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
 
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    

class BottleneckDiT(nn.Module):
    # DiT bottleneck
    # expansion = 1
 
    def __init__(self, c1, c2, stride=1, heads=4, dit=True, expansion=1):
        super(BottleneckDiT, self).__init__()
        c_ = int(c2 * expansion)
        self.cv1 = Conv(c1, c_, 1, 1)
        # self.bn1 = nn.BatchNorm2d(c2)
        if not dit:
            self.cv2 = Conv(c_, c2, 3, 1)
        else:
            self.cv2 = nn.ModuleList()
            self.cv2.append(DiTBlock(c2, num_heads=heads))
            if stride == 2:
                self.cv2.append(nn.AvgPool2d(2, 2))
            self.cv2 = nn.Sequential(*self.cv2)
        self.shortcut = c1 == c2
        if stride != 1 or c1 != expansion * c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, expansion * c2, kernel_size=1, stride=stride),
                nn.BatchNorm2d(expansion * c2)
            )
        self.fc1 = nn.Linear(c2, c2)
 
    def forward(self, x):
        out = x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))
        return out
    

# class C3k2_DiTBlock(C2f):

#     def __init__(self, c1, c2, n=1, c3k=False, e=0.5, e2=1, g=1, shortcut=True):
#         """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(
#             C3k(self.c, self.c, 2, shortcut, g) if c3k else BottleneckDiT(self.c, self.c, stride=1, heads=4, dit=True, expansion=e2) for _ in range(n)
#         )


class C3k2_DiTBlock(C2f):

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, e2=1, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else DiTBlock(self.c, num_heads=4) for _ in range(n)
        )


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.
    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.
    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.
    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.
    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """
 
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()
 
        self.attn = DiTBlock(c, num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut
 
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_DiTBlock(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.
    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.
    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.
    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.
    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """
 
    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
 
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
 
    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
    

#################################################################################
#                                   DiT Configs                                  #
#################################################################################

# def DiT_XL_2(**kwargs):
#     return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

# def DiT_XL_4(**kwargs):
#     return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

# def DiT_XL_8(**kwargs):
#     return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

# def DiT_L_2(**kwargs):
#     return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

# def DiT_L_4(**kwargs):
#     return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

# def DiT_L_8(**kwargs):
#     return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

# def DiT_B_2(**kwargs):
#     return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

# def DiT_B_4(**kwargs):
#     return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

# def DiT_B_8(**kwargs):
#     return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

# def DiT_S_2(**kwargs):
#     return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

# def DiT_S_4(**kwargs):
#     return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

# def DiT_S_8(**kwargs):
#     return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


# if __name__ == "__main__":
#     # Test the implementation
#     input_tensor = torch.randn(50, 512, 7, 7)
#     # dit_block = DiTBlock(hidden_size=512, num_heads=4, mlp_ratio=4.0)
#     dit_block = DiT(input_size=512, patch_size=8, in_channels=3, hidden_size=1024, num_heads=8, mlp_ratio=4.0, depth=2)
#     output = dit_block(input_tensor)
#     print(output.shape)
