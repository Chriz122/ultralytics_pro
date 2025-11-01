import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.checkpoint import checkpoint
from timm.models.layers import trunc_normal_


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(torch.tensor(scale_value), requires_grad=scale_learnable)
        self.bias = nn.Parameter(torch.tensor(bias_value), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x).pow(2) + self.bias


def get_fft2freq(d1, d2, use_rfft=False):
    freq_h = torch.fft.fftfreq(d1)
    freq_w = torch.fft.rfftfreq(d2) if use_rfft else torch.fft.fftfreq(d2)
    grid = torch.stack(torch.meshgrid(freq_h, freq_w), dim=-1)
    dist = torch.norm(grid, dim=-1)
    flat = dist.view(-1)
    _, idx = torch.sort(flat)
    if use_rfft:
        d2_ = d2 // 2 + 1
    else:
        d2_ = d2
    coords = torch.stack([idx // d2_, idx % d2_], dim=1)
    return coords.t(), grid


class KernelSpatialModulation_Global(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 reduction=0.0625, kernel_num=4, min_channel=16,
                 temp=1.0, kernel_temp=None, att_multi=2.0,
                 ksm_only_kernel_att=False, stride=1,
                 spatial_freq_decompose=False, act_type='sigmoid'):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.temperature = temp
        self.kernel_temp = kernel_temp
        self.att_multi = att_multi
        self.ksm_only_kernel_att = ksm_only_kernel_att
        self.act_type = act_type

        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = StarReLU()

        # Channel
        if ksm_only_kernel_att:
            self.func_channel = lambda x: 1.0
        else:
            out_c = in_planes * 2 if spatial_freq_decompose and kernel_size>1 else in_planes
            self.channel_fc = nn.Conv2d(attention_channel, out_c, 1)
            self.func_channel = self._get_channel
        # Filter
        if in_planes==out_planes or ksm_only_kernel_att:
            self.func_filter = lambda x: 1.0
        else:
            out_f = out_planes * 2 if spatial_freq_decompose and stride>1 else out_planes
            self.filter_fc = nn.Conv2d(attention_channel, out_f, 1, stride=stride)
            self.func_filter = self._get_filter
        # Spatial
        if kernel_size==1 or ksm_only_kernel_att:
            self.func_spatial = lambda x: 1.0
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size*kernel_size,1)
            self.func_spatial = self._get_spatial
        # Kernel mixing
        if kernel_num==1:
            self.func_kernel = lambda x: 1.0
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num,1)
            self.func_kernel = self._get_kernel

    def _get_channel(self, x):
        att = self.channel_fc(x).view(x.size(0),1,1,-1,1,1)
        if self.act_type=='sigmoid': return torch.sigmoid(att/self.temperature)*self.att_multi
        elif self.act_type=='tanh': return 1+torch.tanh(att/self.temperature)
        raise
    def _get_filter(self,x):
        att = self.filter_fc(x).view(x.size(0),1,-1,1,1,1)
        if self.act_type=='sigmoid': return torch.sigmoid(att/self.temperature)*self.att_multi
        elif self.act_type=='tanh': return 1+torch.tanh(att/self.temperature)
        raise
    def _get_spatial(self,x):
        att = self.spatial_fc(x).view(x.size(0),1,1,1,self.kernel_size,self.kernel_size)
        if self.act_type=='sigmoid': return torch.sigmoid(att/self.temperature)*self.att_multi
        elif self.act_type=='tanh': return 1+torch.tanh(att/self.temperature)
        raise
    def _get_kernel(self,x):
        att = self.kernel_fc(x).view(x.size(0),-1,1,1,1,1)
        return F.softmax(att/self.kernel_temp,dim=1)

    def forward(self,x):
        g = self.relu(self.bn(self.fc(x)))
        return self.func_channel(g), self.func_filter(g), self.func_spatial(g), self.func_kernel(g)


class KernelSpatialModulation_Local(nn.Module):
    def __init__(self, channel, kernel_num=1, out_n=1, k_size=3, use_global=False):
        super().__init__()
        self.kn, self.out_n, self.use_global = kernel_num, out_n, use_global
        self.conv1d = nn.Conv1d(1, kernel_num*out_n, k_size, padding=(k_size-1)//2)
        self.norm = nn.LayerNorm(channel)
        if use_global:
            self.complex_w = nn.Parameter(torch.randn(1,channel//2+1,2)*1e-6)
    def forward(self,x):
        y = x.mean(-1,keepdim=True).transpose(-1,-2)
        if self.use_global:
            fft = torch.fft.rfft(y.float(),dim=-1)
            real = fft.real*self.complex_w[...,0]
            imag = fft.imag*self.complex_w[...,1]
            y = y+torch.fft.irfft(torch.complex(real,imag),dim=-1)
        y = self.norm(y)
        att = self.conv1d(y).reshape(y.size(0),self.kn,self.out_n,-1)
        return att.permute(0,1,3,2)


class FrequencyBandModulation(nn.Module):
    def __init__(self,in_channels,k_list=[2,4,8],lowfreq_att=False,fs_feat='feat',act='sigmoid',spatial='conv',spatial_group=1,spatial_kernel=3,init='zero'):
        super().__init__()
        self.k_list,self.lowfreq_att,self.act= k_list,lowfreq_att,act
        # n_freq_groups used here
        self.spatial_group = spatial_group
        self.convs = nn.ModuleList()
        for _ in range(len(k_list)+int(lowfreq_att)):
            conv=nn.Conv2d(in_channels, self.spatial_group, spatial_kernel, padding=spatial_kernel//2, groups=self.spatial_group)
            if init=='zero': nn.init.normal_(conv.weight, std=1e-6); nn.init.zeros_(conv.bias)
            self.convs.append(conv)
    def forward(self,x):
        b,c,h,w=x.shape
        x_fft=torch.fft.rfft2(x,norm='ortho')
        out=0
        for i,f in enumerate(self.k_list):
            mask=torch.zeros_like(x_fft[:,:1])
            coords,_=get_fft2freq(h,w,use_rfft=True)
            mask[:,:,coords.max(-1)[0]<0.5/f]=1
            low=torch.fft.irfft2(x_fft*mask,s=(h,w),norm='ortho')
            high=x-low; x=low
            w_att=self.convs[i](x)
            w_att = (w_att.sigmoid()*2) if self.act=='sigmoid' else (1+w_att.tanh())
            out+= (w_att.reshape(b,self.spatial_group,-1,h,w)*high.reshape(b,self.spatial_group,-1,h,w)).reshape(b,-1,h,w)
        if self.lowfreq_att:
            w_att=self.convs[-1](x)
            w_att=(w_att.sigmoid()*2) if self.act=='sigmoid' else (1+w_att.tanh())
            out+= (w_att.reshape(b,self.spatial_group,-1,h,w)*x.reshape(b,self.spatial_group,-1,h,w)).reshape(b,-1,h,w)
        else:
            out+=x
        return out


class FDConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1, bias=True,
                 reduction=0.0625, kernel_num=4, n_freq_groups=1,
                 kernel_temp=1.0, temp=None, att_multi=2.0,
                 param_ratio=1, param_reduction=1.0,
                 ksm_only_kernel_att=False, spatial_freq_decompose=False,
                 use_ksm_local=True, ksm_local_act='sigmoid', ksm_global_act='sigmoid',
                 fbm_cfg=None):
        padding = (kernel_size-1)//2 if padding is None else padding
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias)
        self.kernel_num = kernel_num
        self.att_multi = att_multi
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.n_freq_groups = n_freq_groups

        # 全域調製模組
        if temp is None:
            temp = kernel_temp
        self.KSMG = KernelSpatialModulation_Global(
            in_planes=in_channels,
            out_planes=out_channels,
            kernel_size=kernel_size,
            reduction=reduction,
            kernel_num=kernel_num,
            temp=temp,
            kernel_temp=kernel_temp,
            att_multi=att_multi,
            ksm_only_kernel_att=ksm_only_kernel_att,
            spatial_freq_decompose=spatial_freq_decompose,
            act_type=ksm_global_act
        )

        # 局部調製模組
        self.use_local = use_ksm_local
        if use_ksm_local:
            self.KSML = KernelSpatialModulation_Local(
                channel=in_channels,
                kernel_num=1,
                out_n=out_channels * kernel_size * kernel_size
            )

        # 頻帶調製模組
        if fbm_cfg is None:
            fbm_cfg = {}
        fbm_cfg['spatial_group'] = n_freq_groups
        self.FBM = FrequencyBandModulation(in_channels, **fbm_cfg)

        # 準備 DFT 權重
        self._prepare_dft()

    def _prepare_dft(self):
        # 1) 計算 out×in 大矩陣的 DFT
        d1, d2 = self.out_channels, self.in_channels
        k = self.kernel_size[0]
        flat = self.weight.permute(0,2,1,3).reshape(d1*k, d2*k)
        freq = torch.fft.rfft2(flat, dim=(0,1))
        w = torch.stack([freq.real, freq.imag], dim=-1)  # shape = (d1*k, d2*k, 2)

        # 2) 將它做成可訓練參數
        #    shape → (param_ratio, d1*k, d2*k, 2)
        self.dft_weight = nn.Parameter(
            w[None].repeat(self.param_ratio, 1, 1, 1) // (min(d1, d2)//2)
        )

        # **不要再刪除 self.weight**
        # if self.bias is not None:
        #     self.bias = nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, x):
        # fallback to plain conv if小channel 或 非1/3kernel
        if min(self.in_channels, self.out_channels) <= 16 or self.kernel_size[0] not in [1,3]:
            # 這裡 super().forward 會呼叫 nn.Conv2d.forward，必須有 self.weight
            return super().forward(x)

        b, _, h, w = x.shape
        # 計算各種 attention map
        ch, fl, sp, kn = self.KSMG(F.adaptive_avg_pool2d(x, 1))
        hr = 1
        if self.use_local:
            local = self.KSML(F.adaptive_avg_pool2d(x, 1))
            hr = (local.sigmoid() * self.att_multi) if self.KSML.use_global else (1 + local.tanh())

        # 建 DFT map 並還原到 spatial domain
        coords, _ = get_fft2freq(
            self.out_channels * self.kernel_size[0],
            self.in_channels * self.kernel_size[1],
            use_rfft=True
        )
        DFTmap = torch.zeros(
            (b, self.out_channels * self.kernel_size[0], self.in_channels * self.kernel_size[1]//2+1, 2),
            device=x.device
        )
        for i in range(self.param_ratio):
            w = self.dft_weight[i][coords[0], coords[1]][None]
            DFTmap[...,0] += w[...,0] * kn[:, i]
            DFTmap[...,1] += w[...,1] * kn[:, i]

        adapt = torch.fft.irfft2(torch.view_as_complex(DFTmap), dim=(1,2))
        adapt = adapt.reshape(
            b, 1, self.out_channels, self.kernel_size[0],
            self.in_channels, self.kernel_size[1]
        ).permute(0,1,2,4,3,5)

        # 頻帶模組
        x_fbm = self.FBM(x)
        x_in = x_fbm if hasattr(self, 'FBM') else x

        # 聚合所有 attention
        agg = sp * ch * fl * adapt * hr
        agg = agg.sum(1).view(
            -1,
            self.in_channels // self.groups,
            self.kernel_size[0],
            self.kernel_size[1]
        )

        # Depthwise batch conv trick
        xcat = x_in.reshape(1, -1, h, w)
        out = F.conv2d(
            xcat, agg, None,
            self.stride, self.padding,
            self.dilation, self.groups * b
        )
        out = out.view(b, self.out_channels, out.size(-2), out.size(-1))

        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)

        return out


if __name__ == "__main__":
    # Usage example:
    fd = FDConv(in_channels=64, out_channels=64, kernel_size=3, n_freq_groups=64)
    print(fd)