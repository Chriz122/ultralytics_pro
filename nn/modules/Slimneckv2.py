import math
import warnings

import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SNI(nn.Module):
    '''
    https://github.com/AlanLi1997/rethinking-fpn
    soft nearest neighbor interpolation for up-sampling
    secondary features aligned
    '''
    def __init__(self, c1=0, c2=0, up_f=2):
        super(SNI, self).__init__()
        self.us = nn.Upsample(None, up_f, 'nearest')
        self.alpha = 1/(up_f**2)

    def forward(self, x):
        return self.alpha*self.us(x)


class GSConvE(nn.Module):
    '''
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    https://github.com/AlanLi1997/slim-neck-by-gsconv
    '''
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 1, bias=False),
            nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        # shuffle
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


class GSConvE2(nn.Module):
    # enhancement lightweight conv

    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 4
        self.cv1 = Conv(c1, c_, k, s, k // 2, g, d, act)
        self.cv2 = Conv(c_, c_, 9, 1, k // 2, c_, d, act)
        self.cv3 = Conv(c_, c_, 13, 1, k // 2, c_, d, act)
        self.cv4 = Conv(c_, c_, 17, 1, k // 2, c_, d, act)

    def forward(self, x):
        y = torch.cat((self.cv1(x), self.cv2(self.cv1(x)), self.cv3(self.cv1(x)), self.cv4(self.cv1(x))), dim=1)
        # shuffle
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        output = y.permute(0, 2, 1, 3, 4)

        return output.reshape(output.shape[0], -1, output.shape[3], output.shape[4])


class GSConv(nn.Module):
    '''
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    https://github.com/AlanLi1997/slim-neck-by-gsconv
    '''
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, d, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        # shuffle
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


class ESD(nn.Module):
    '''
    https://github.com/AlanLi1997/rethinking-fpn
    Extended spatial window for down-sampling
    lightweight fusion
    '''
    def __init__(self, c1, c2, k=3, s=2, g=1, d=1, act=True):
        super().__init__()
        self.out_c = c2
        self.dense_feature = Conv(c1, c2, k, s, k // 2, g, d, act)  # window_dense_f
        self.global_feature = nn.AvgPool2d(4, 2, 1)  # window_global_f
        self.local_feature = nn.MaxPool2d(4, 2, 1)  # window_local_f

    def forward(self, x):
        if self.out_c == x.shape[1]:
            return self.global_feature(x) + self.local_feature(x) + self.dense_feature(x)
        else:
            return torch.cat((self.global_feature(x), self.local_feature(x)), dim=1) + self.dense_feature(x)


class ESD2(nn.Module):
    '''
    https://github.com/AlanLi1997/rethinking-fpn
    Extended spatial window for down-sampling
    learnable linearly fusion
    '''
    def __init__(self, c1, c2, k=3, s=2, g=1, d=1, act=True):
        super().__init__()
        self.dense_feature = Conv(c1, c2, k, s, None, g, d, act)  # window_dense_f
        self.global_feature = nn.AvgPool2d(4, 2, 1)  # window_global_f
        self.local_feature = nn.MaxPool2d(4, 2, 1)  # window_local_f
        self.fuse = nn.Conv2d(2*c2, c2, 1, 1, bias=False)

    def forward(self, x):
        return self.fuse(torch.cat((self.global_feature(x), self.local_feature(x), self.dense_feature(x)), dim=1))