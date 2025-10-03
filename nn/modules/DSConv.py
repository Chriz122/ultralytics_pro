from .conv import Conv
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import math
import torch
import torch.nn as nn
import torch.nn.functional as F 

__all__ = ("C2f_DSConv")


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
class DSConv(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, dilation=1, groups=1, padding_mode='zeros', bias=False, block_size=32, KDSBias=False,
                 CDS=False):
        padding = _pair(autopad(kernel_size, padding, dilation))
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
 
        blck_numb = math.ceil((in_channels / (block_size * groups)))
        super(DSConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
 
        # KDS weight From Paper
        self.intweight = torch.Tensor(out_channels, in_channels, *kernel_size)
        self.alpha = torch.Tensor(out_channels, blck_numb, *kernel_size)
 
        # KDS bias From Paper
        self.KDSBias = KDSBias
        self.CDS = CDS
 
        if KDSBias:
            self.KDSb = torch.Tensor(out_channels, blck_numb, *kernel_size)
        if CDS:
            self.CDSw = torch.Tensor(out_channels)
            self.CDSb = torch.Tensor(out_channels)
 
        self.reset_parameters()
 
    def get_weight_res(self):
        # Include expansion of alpha and multiplication with weights to include in the convolution layer here
        alpha_res = torch.zeros(self.weight.shape).to(self.alpha.device)
 
        # Include KDSBias
        if self.KDSBias:
            KDSBias_res = torch.zeros(self.weight.shape).to(self.alpha.device)
 
        # Handy definitions:
        nmb_blocks = self.alpha.shape[1]
        total_depth = self.weight.shape[1]
        bs = total_depth // nmb_blocks
 
        llb = total_depth - (nmb_blocks - 1) * bs
 
        # Casting the Alpha values as same tensor shape as weight
        for i in range(nmb_blocks):
            length_blk = llb if i == nmb_blocks - 1 else bs
 
            shp = self.alpha.shape  # Notice this is the same shape for the bias as well
            to_repeat = self.alpha[:, i, ...].view(shp[0], 1, shp[2], shp[3]).clone()
            repeated = to_repeat.expand(shp[0], length_blk, shp[2], shp[3]).clone()
            alpha_res[:, i * bs:(i * bs + length_blk), ...] = repeated.clone()
 
            if self.KDSBias:
                to_repeat = self.KDSb[:, i, ...].view(shp[0], 1, shp[2], shp[3]).clone()
                repeated = to_repeat.expand(shp[0], length_blk, shp[2], shp[3]).clone()
                KDSBias_res[:, i * bs:(i * bs + length_blk), ...] = repeated.clone()
 
        if self.CDS:
            to_repeat = self.CDSw.view(-1, 1, 1, 1)
            repeated = to_repeat.expand_as(self.weight)
            print(repeated.shape)
 
        # Element-wise multiplication of alpha and weight
        weight_res = torch.mul(alpha_res, self.weight)
        if self.KDSBias:
            weight_res = torch.add(weight_res, KDSBias_res)
        return weight_res
 
    def forward(self, input):
        # Get resulting weight
        # weight_res = self.get_weight_res()
 
        # Returning convolution
        return F.conv2d(input, self.weight, self.bias,
                        self.stride, self.padding, self.dilation,
                        self.groups)
 
 
class DSConv2D(Conv):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(inc, ouc, k, s, p, g, d, act)
        self.conv = DSConv(inc, ouc, k, s, p, g, d)
 
 
class Bottleneck_DSconv(nn.Module):
    """Standard bottleneck."""
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DSConv2D(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
 
class C2f_DSConv(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
 
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = DSConv2D((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_DSconv(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
 
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))