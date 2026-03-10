import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import StochasticDepth

__all__ = ("mn_conv", "InvertedBottleneck", "MobileNetV3_BLOCK")


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def activation_function(act="RE"):
    res = nn.Hardswish()
    if act == "RE":
        res = nn.ReLU6(inplace=True)
    elif act == "GE":
        res = nn.GELU()
    elif act == "SI":
        res = nn.SiLU()
    elif act == "EL":
        res = nn.ELU()
    else:
        res = nn.Hardswish()
    return res


class mn_conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act="RE", p=None, g=1, d=1):
        super().__init__()
        padding = 0 if k==s else autopad(k,p,d)
        self.c = nn.Conv2d(c1, c2, k, s, padding, groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = activation_function(act)#nn.ReLU6(inplace=True) if act=="RE" else nn.Hardswish()
    
    def forward(self, x):
        return self.act(self.bn(self.c(x)))


class InvertedBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, e=None, sa="None", act="RE", stride=1, pw=True):
        #input_channels, output_channels, repetition, stride, expension ratio
        super().__init__()
        #act = nn.ReLU6(inplace=True) if NL=="RE" else nn.Hardswish()
        c_mid = e if e != None else c1
        self.residual = c1 == c2 and stride == 1

        features = [mn_conv(c1, c_mid, act=act)] if pw else [] #if c_mid != c1 else []
        features.extend([mn_conv(c_mid, c_mid, k, stride, g=c_mid, act=act),
                         #attn,
                         nn.Conv2d(c_mid, c2, 1),
                         nn.BatchNorm2d(c2),
                         #nn.SiLU(),
                         ])
        self.layers = nn.Sequential(*features)
    def forward(self, x):
        #print(x.shape)
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)


class MobileNetV3_BLOCK(nn.Module):
    def __init__(self, c1, c2, k=3, e=None, sa="None", act="RE", stride=1, pw=True):
        #input_channels, output_channels, repetition, stride, expension ratio
        super().__init__()
        #act = nn.ReLU6(inplace=True) if NL=="RE" else nn.Hardswish()
        c_mid = e if e != None else c1
        self.residual = c1 == c2 and stride == 1

        features = [mn_conv(c1, c_mid, act=act)] if pw else [] #if c_mid != c1 else []
        features.extend([mn_conv(c_mid, c_mid, k, stride, g=c_mid, act=act),
                         #attn,
                         nn.Conv2d(c_mid, c2, 1),
                         nn.BatchNorm2d(c2),
                         #nn.SiLU(),
                         ])
        self.layers = nn.Sequential(*features)
    def forward(self, x):
        #print(x.shape)
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)

