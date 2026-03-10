# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import timm
import random
import math
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
import os

# 替代 easydict，減少依賴
class RPEConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# 定義一個全域函數來解決 Pickle 錯誤
def _no_op_initializer(x):
    return None

# fixed_list=[3,8,8,4,32,8,4,8,8,16,64,256,256]
# div_index = 0

@torch.no_grad()
def piecewise_index(relative_position, alpha, beta, gamma, dtype):
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (alpha +
                                   torch.log(rp_abs_out / alpha) /
                                   math.log(gamma / alpha) *
                                   (beta - alpha)).round().clip(max=beta)).to(dtype)

    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        idx = idx.round().to(dtype)

    idx[not_mask] = y_out
    return idx


def get_absolute_positions(height, width, dtype, device):
    rows = torch.arange(height, dtype=dtype, device=device).view(
        height, 1).repeat(1, width)
    cols = torch.arange(width, dtype=dtype, device=device).view(
        1, width).repeat(height, 1)
    return torch.stack([rows, cols], 2)


@torch.no_grad()
def quantize_values(values):
    res = torch.empty_like(values)
    uq = values.unique()
    cnt = 0
    for (tid, v) in enumerate(uq):
        mask = (values == v)
        cnt += torch.count_nonzero(mask)
        res[mask] = tid
    assert cnt == values.numel()
    return res, uq.numel()


class METHOD:
    EUCLIDEAN = 0
    QUANT = 1
    PRODUCT = 3
    CROSS = 4
    CROSS_ROWS = 41
    CROSS_COLS = 42


@torch.no_grad()
def _rp_2d_euclidean(diff, **kwargs):
    dis = diff.square().sum(2).float().sqrt().round()
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_quant(diff, **kwargs):
    dis = diff.square().sum(2)
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_product(diff, **kwargs):
    beta_int = int(kwargs['beta'])
    S = 2 * beta_int + 1
    r = piecewise_index(diff[:, :, 0], **kwargs) + beta_int
    c = piecewise_index(diff[:, :, 1], **kwargs) + beta_int
    pid = r * S + c
    return pid


@torch.no_grad()
def _rp_2d_cross_rows(diff, **kwargs):
    dis = diff[:, :, 0]
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_cross_cols(diff, **kwargs):
    dis = diff[:, :, 1]
    return piecewise_index(dis, **kwargs)


_METHOD_FUNC = {
    METHOD.EUCLIDEAN: _rp_2d_euclidean,
    METHOD.QUANT: _rp_2d_quant,
    METHOD.PRODUCT: _rp_2d_product,
    METHOD.CROSS_ROWS: _rp_2d_cross_rows,
    METHOD.CROSS_COLS: _rp_2d_cross_cols,
}


def get_num_buckets(method, alpha, beta, gamma):
    beta_int = int(beta)
    if method == METHOD.PRODUCT:
        num_buckets = (2 * beta_int + 1) ** 2
    else:
        num_buckets = 2 * beta_int + 1
    return num_buckets


BUCKET_IDS_BUF = dict()


@torch.no_grad()
def get_bucket_ids_2d_without_skip(method, height, width,
                                   alpha, beta, gamma,
                                   dtype=torch.long, device=torch.device('cpu')):
    key = (method, alpha, beta, gamma, dtype, device)
    value = BUCKET_IDS_BUF.get(key, None)
    if value is None or value[-2] < height or value[-1] < width:
        if value is None:
            max_height, max_width = height, width
        else:
            max_height = max(value[-2], height)
            max_width = max(value[-1], width)
        
        func = _METHOD_FUNC.get(method, None)
        if func is None:
            raise NotImplementedError(
                f"[Error] The method ID {method} does not exist.")
        pos = get_absolute_positions(max_height, max_width, dtype, device)

        max_L = max_height * max_width
        pos1 = pos.view((max_L, 1, 2))
        pos2 = pos.view((1, max_L, 2))
        diff = pos1 - pos2

        bucket_ids = func(diff, alpha=alpha, beta=beta,
                          gamma=gamma, dtype=dtype)
        beta_int = int(beta)
        if method != METHOD.PRODUCT:
            bucket_ids += beta_int
        bucket_ids = bucket_ids.view(
            max_height, max_width, max_height, max_width)

        num_buckets = get_num_buckets(method, alpha, beta, gamma)
        value = (bucket_ids, num_buckets, height, width)
        BUCKET_IDS_BUF[key] = value
    L = height * width
    bucket_ids = value[0][:height, :width, :height, :width].reshape(L, L)
    num_buckets = value[1]

    return bucket_ids, num_buckets, L


@torch.no_grad()
def get_bucket_ids_2d(method, height, width,
                      skip, alpha, beta, gamma,
                      dtype=torch.long, device=torch.device('cpu')):
    bucket_ids, num_buckets, L = get_bucket_ids_2d_without_skip(method, height, width,
                                                                alpha, beta, gamma,
                                                                dtype, device)
    if skip > 0:
        new_bids = bucket_ids.new_empty(size=(skip + L, skip + L))
        extra_bucket_id = num_buckets
        num_buckets += 1

        new_bids[:skip] = extra_bucket_id
        new_bids[:, :skip] = extra_bucket_id
        new_bids[skip:, skip:] = bucket_ids

        bucket_ids = new_bids
    bucket_ids = bucket_ids.contiguous()
    return bucket_ids, num_buckets


class iRPE(nn.Module):
    _rp_bucket_buf = (None, None, None)

    def __init__(self, head_dim, num_heads=8,
                 mode=None, method=None,
                 transposed=True, num_buckets=None,
                 initializer=None, rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mode = mode
        self.method = method
        self.transposed = transposed
        self.num_buckets = num_buckets

        if initializer is None:
            # 修改：使用模組層級函數，避免 pickle 錯誤
            self.initializer = _no_op_initializer
        else:
            self.initializer = initializer
            
        self.rpe_config = rpe_config
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.transposed:
            if self.mode == 'bias':
                self.lookup_table_bias = nn.Parameter(
                    torch.zeros(self.num_heads, self.num_buckets))
                self.initializer(self.lookup_table_bias)
            elif self.mode == 'contextual':
                self.lookup_table_weight = nn.Parameter(
                    torch.zeros(self.num_heads,self.head_dim, self.num_buckets))
                self.initializer(self.lookup_table_weight)
        else:
            if self.mode == 'bias':
                raise NotImplementedError
            elif self.mode == 'contextual':
                self.lookup_table_weight = nn.Parameter(
                    torch.zeros(self.num_heads,
                                self.num_buckets, self.head_dim))
                self.initializer(self.lookup_table_weight)

    def forward(self, x, height=None, width=None):
        rp_bucket, self._ctx_rp_bucket_flatten = \
            self._get_rp_bucket(x, height=height, width=width)
        if self.transposed:
            return self.forward_rpe_transpose(x, rp_bucket)
        return self.forward_rpe_no_transpose(x, rp_bucket)

    def _get_rp_bucket(self, x, height=None, width=None):
        B, H, L, D = x.shape
        device = x.device
        if height is None:
            E = int(math.sqrt(L))
            height = width = E
        key = (height, width, device)

        if self._rp_bucket_buf[0] == key:
            return self._rp_bucket_buf[1:3]

        skip = L - height * width
        config = self.rpe_config
        
        # Assume RPEIndexFunction is None for standard usage
        dtype = torch.long
        
        rp_bucket, num_buckets = get_bucket_ids_2d(method=self.method,
                                                   height=height, width=width,
                                                   skip=skip, alpha=config.alpha,
                                                   beta=config.beta, gamma=config.gamma,
                                                   dtype=dtype, device=device)
        assert num_buckets == self.num_buckets

        _ctx_rp_bucket_flatten = None
        if self.mode == 'contextual' and self.transposed:
            offset = torch.arange(0, L * self.num_buckets, self.num_buckets,
                                  dtype=rp_bucket.dtype, device=rp_bucket.device).view(-1, 1)
            _ctx_rp_bucket_flatten = (rp_bucket + offset).flatten()
        self._rp_bucket_buf = (key, rp_bucket, _ctx_rp_bucket_flatten)
        return rp_bucket, _ctx_rp_bucket_flatten

    def forward_rpe_transpose(self, x, rp_bucket):
        B = len(x)
        L_query, L_key = rp_bucket.shape
        if self.mode == 'bias':
            return self.lookup_table_bias[:, rp_bucket.flatten()].\
                view(1, self.num_heads, L_query, L_key)

        elif self.mode == 'contextual':
            lookup_table = torch.matmul(
                x.transpose(0, 1).reshape(-1, B * L_query, self.head_dim),
                self.lookup_table_weight).\
                view(-1, B, L_query, self.num_buckets).transpose(0, 1)
            
            return lookup_table.flatten(2)[:, :, self._ctx_rp_bucket_flatten].\
                view(B, -1, L_query, L_key)

    def forward_rpe_no_transpose(self, x, rp_bucket):
        B = len(x)
        L_query, L_key = rp_bucket.shape
        assert self.mode == 'contextual'
        weight = self.lookup_table_weight[:, rp_bucket.flatten()].\
            view(self.num_heads, L_query, L_key, self.head_dim)
        return torch.matmul(x.permute(1, 2, 0, 3), weight).permute(2, 0, 1, 3)
    
    def __repr__(self):
        return 'iRPE(head_dim={rpe.head_dim}, num_heads={rpe.num_heads}, \
mode="{rpe.mode}", method={rpe.method}, transposed={rpe.transposed}, \
num_buckets={rpe.num_buckets})'.format(rpe=self)


class iRPE_Cross(nn.Module):
    def __init__(self, method, **kwargs):
        super().__init__()
        assert method == METHOD.CROSS
        self.rp_rows = iRPE(**kwargs, method=METHOD.CROSS_ROWS)
        self.rp_cols = iRPE(**kwargs, method=METHOD.CROSS_COLS)

    def forward(self, x, height=None, width=None):
        rows = self.rp_rows(x, height=height, width=width)
        cols = self.rp_cols(x, height=height, width=width)
        return rows + cols


def get_single_rpe_config(ratio=1.9,
                          method=METHOD.PRODUCT,
                          mode='contextual',
                          shared_head=True,
                          skip=0):
    config = RPEConfig()
    config.shared_head = shared_head
    config.mode = mode
    config.method = method
    config.alpha = 1 * ratio
    config.beta = 2 * ratio
    config.gamma = 8 * ratio

    config.num_buckets = get_num_buckets(method,
                                         config.alpha,
                                         config.beta,
                                         config.gamma)
    if skip > 0:
        config.num_buckets += 1
    return config


def get_rpe_config(ratio=1.9,
                   method=METHOD.PRODUCT,
                   mode='contextual',
                   shared_head=True,
                   skip=0,
                   rpe_on='k'):
    if isinstance(method, str):
        method_mapping = dict(
            euc=METHOD.EUCLIDEAN,
            quant=METHOD.QUANT,
            cross=METHOD.CROSS,
            product=METHOD.PRODUCT,
        )
        method = method_mapping[method.lower()]
    if mode == 'ctx':
        mode = 'contextual'
    config = RPEConfig()
    kwargs = dict(
        ratio=ratio,
        method=method,
        mode=mode,
        shared_head=shared_head,
        skip=skip,
    )
    config.rpe_q = get_single_rpe_config(**kwargs) if 'q' in rpe_on else None
    config.rpe_k = get_single_rpe_config(**kwargs) if 'k' in rpe_on else None
    config.rpe_v = get_single_rpe_config(**kwargs) if 'v' in rpe_on else None
    return config


def build_rpe(config, head_dim, num_heads):
    if config is None:
        return None, None, None
    rpes = [config.rpe_q, config.rpe_k, config.rpe_v]
    transposeds = [True, True, False]

    def _build_single_rpe(rpe, transposed):
        if rpe is None:
            return None

        rpe_cls = iRPE if rpe.method != METHOD.CROSS else iRPE_Cross
        return rpe_cls(
            head_dim=head_dim,
            num_heads=1 if rpe.shared_head else num_heads,
            mode=rpe.mode,
            method=rpe.method,
            transposed=transposed,
            num_buckets=rpe.num_buckets,
            rpe_config=rpe,
        )
    return [_build_single_rpe(rpe, transposed)
            for rpe, transposed in zip(rpes, transposeds)]


def aggregate(gate_k, I, one, K, sort=False):
    if sort:
        _, ind = gate_k.sort(descending=True)
        gate_k = gate_k[:, ind[0, :]]

    U = [(gate_k[0, i] * one + gate_k[1, i] * I) for i in range(K)]
    while len(U) != 1:
        temp = []
        for i in range(0, len(U) - 1, 2):
            temp.append(torch.kron(U[i], U[i + 1]))
        if len(U) % 2 != 0:
            temp.append(U[-1])
        del U
        U = temp

    return U[0], gate_k

def check01(arr):
    index = -1
    flag = False
    for i in range(len(arr)):
        if arr[i] == 0:
            flag = True
        elif flag and arr[i] == 1:
            index = i
            break

    if index != -1:
        return index
    else:
        return True

class DcSign1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        torch.where(input < 0, torch.tensor(0), torch.tensor(1))
        return input
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input.ge(1) | input.le(0), 0)
        return grad_input
    
class DcSign2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)
        return grad_input

# 模擬 model_api 模組的功能，以避免導入錯誤
class MockModelAPI:
    global_gconv = []
    global_epoch = 0
    global_loss_gate = []
    global_n_div = []
    global_regularizer = []

model_api = MockModelAPI()

class DGConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, 
                 sort=False, u_regular=False, l_gate=False, index_div=False, n_div=4, print_n_div=False, pre_epoch=100):
        super(DGConv2d, self).__init__()
        self.register_buffer('I', torch.eye(2)) 
        self.register_buffer('one', torch.ones(2, 2))
        self.register_buffer('c_div', torch.tensor(n_div, dtype=torch.int32))
        self.register_buffer('one_channel', torch.tensor(in_channels//n_div, dtype=torch.int32))
        self.K = int(math.log2(in_channels))
        eps = 1e-8 
        gate_init = [eps * random.choice([-1, 1]) for _ in range(self.K)]
        self.register_parameter('gate', nn.Parameter(torch.Tensor(gate_init)))
        self.consistent_train = False
        if self.consistent_train:
            self.start_traing = False
            self.U_M = torch.zeros((in_channels, out_channels))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sort = sort
        self.u_regular=u_regular
        self.l_gate=l_gate
        self.index_div = index_div
        self.n_div=torch.tensor(n_div)  
        self.pre_epoch = pre_epoch 
        self.print_n_div=print_n_div
        self.print_n_div_eval = True
        if self.u_regular:
            model_api.global_gconv.append(in_channels*out_channels)

    def forward(self, x):
        if self.training:
            global_epoch = model_api.global_epoch
            if self.pre_epoch == 0 or global_epoch >= self.pre_epoch:
                direct_STE = False
                if direct_STE:
                    sign_g = DcSign1.apply(self.gate)
                else:
                    sign_g = DcSign2.apply(self.gate)
                    sign_g = (sign_g + 1)/2

                gate_k = torch.stack((sign_g, 1-sign_g)) 

                index_bool = check01(1-sign_g)  
                if self.l_gate:
                    if isinstance(index_bool, bool):
                        loss_gate = torch.tensor(0).to(self.gate.device)
                    else:
                        loss_gate = sum(self.gate[index_bool:].abs())

                    model_api.global_loss_gate.append(loss_gate)

                self.c_div = 2**(sum(1-sign_g)).int()

                if self.print_n_div:
                    model_api.global_n_div.append(self.c_div.data)

                if self.u_regular:
                    U_regularizer =  2 ** (self.K  + torch.sum(sign_g))
                    model_api.global_regularizer.append(U_regularizer)

                self.one_channel = (self.in_channels / self.c_div).int()

                U, gate_k = aggregate(gate_k, self.I, self.one, self.K, sort=self.sort)
                
                if self.consistent_train:
                    self.start_traing = True
                    self.U_M.data = U.data
                masked_weight = self.conv.weight * U.view(self.out_channels, self.in_channels, 1, 1)
                x_out = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
            else:
                self.c_div = self.n_div.to(self.gate.device)
                self.one_channel = (self.in_channels / self.c_div).int()
                U = torch.zeros(self.in_channels, self.in_channels, dtype=torch.int32).to(self.gate.device)
                U[:self.one_channel, :self.one_channel] = 1
                masked_weight = self.conv.weight * U.view(self.out_channels, self.in_channels, 1, 1)
                x_out = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
                x_out[:, self.one_channel:, :, :] = x[:, self.one_channel:, :, :]
        else:
            if self.consistent_train and self.start_traing:
                masked_weight = self.conv.weight * self.U_M.view(self.out_channels, self.in_channels, 1, 1)
                x_out = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
            else:
                if self.print_n_div_eval:
                    # print(self.c_div)
                    self.print_n_div_eval = False
                pconv_enval = True
                if pconv_enval:
                    x1, x2 = torch.split(x, [self.one_channel, self.in_channels-self.one_channel], dim=1)
                    pconv_weight = self.conv.weight[:self.one_channel, :self.one_channel, ...]
                    x1 = F.conv2d(x1, pconv_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
                    x_out = torch.cat((x1, x2), 1)
                else: 
                    # Default implementation for non-pconv eval
                    x_out = F.conv2d(x, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
            
        return x_out


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x  
    
class RPEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config, head_dim=head_dim, num_heads=num_heads)

    def forward(self, x):
        B, C, h, w = x.shape
        x = x.view(B, C, h*w).transpose(1,2)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q *= self.scale

        attn = (q @ k.transpose(-2, -1))

        if self.rpe_k is not None:
            attn += self.rpe_k(q, h, w)
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1,2).view(B, C, h, w)
        return x

class SRM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.cfc1 = nn.Conv2d(channel, channel, kernel_size=(1,2), bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        mean = x.reshape(b, c, -1).mean(-1).view(b,c,1,1)
        std = x.reshape(b, c, -1).std(-1).view(b,c,1,1)
        u = torch.cat([mean, std], dim=-1)
        z = self.cfc1(u)
        z = self.bn(z)
        g = self.sigmoid(z)
        g = g.reshape(b, c, 1, 1)
        return x * g.expand_as(x)

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward_type, use_attn='', channel_type='', patnet_t0=False):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim = dim
        self.n_div = n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.use_attn = use_attn
        self.channel_type = channel_type

        if use_attn:
            if channel_type == 'self':
                self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
                rpe_config = get_rpe_config(
                        ratio=20,
                        method="euc",
                        mode='bias',
                        shared_head=False,
                        skip=0,
                        rpe_on='k',
                    )
                if patnet_t0:
                    num_heads = 4
                else:
                    num_heads = 6
                self.attn = RPEAttention(self.dim_untouched, num_heads=num_heads, attn_drop=0.1, proj_drop=0.1, rpe_config=rpe_config)
                self.norm = timm.models.layers.LayerNorm2d(self.dim_untouched)
                self.forward = self.forward_atten
            elif channel_type == 'se':
                self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
                self.attn = SRM(self.dim_untouched)
                self.norm = nn.BatchNorm2d(self.dim_untouched)
                self.forward = self.forward_atten
        else:
            if forward_type == 'slicing':
                self.forward = self.forward_slicing
            elif forward_type == 'split_cat':
                self.forward = self.forward_split_cat
            else:
                raise NotImplementedError
    
    def forward_atten(self, x: Tensor) -> Tensor:
        if self.channel_type:
            if self.channel_type == 'se':
                x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
                x1 = self.partial_conv3(x1)
                x2 = self.attn(x2)
                x2 = self.norm(x2)
                x = torch.cat((x1, x2), 1)
            else:
                x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
                x1 = self.partial_conv3(x1)
                x2 = self.norm(x2)
                x2 = self.attn(x2)
                x = torch.cat((x1, x2), 1)
        return x
    
    def forward_slicing(self, x: Tensor) -> Tensor:
        x1 = x.clone() 
        x1[:, :self.dim_conv3, :, :] = self.partial_conv3(x1[:, :self.dim_conv3, :, :])
        return x1

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class partial_spatial_attn_layer_reverse(nn.Module):
    def __init__(self, dim, n_head, partial=0.5):
        super().__init__()
        self.dim = dim
        self.dim_conv = int(partial * dim)
        self.dim_untouched = dim - self.dim_conv
        self.nhead = n_head
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 1, bias=False)
        self.conv_attn = nn.Conv2d(self.dim_untouched, n_head, 1, bias=False)
        self.norm = nn.BatchNorm2d(self.dim_untouched)
        self.norm2 = nn.BatchNorm2d(self.dim_conv)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x1, x2 = torch.split(x, [self.dim_untouched, self.dim_conv], 1)
        weight =self.act(self.conv_attn(x1))
        x1 = x1 * weight
        x1 = self.norm(x1)
        x2 = self.norm2(x2)
        x2 = self.conv(x2)
        x = torch.cat((x1, x2), 1)
        return x


class MLPBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_div,
                 print_n_div,
                 auto_div,
                 u_regular,
                 l_gate,
                 index_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type,
                 pre_epoch,
                 use_channel,
                 use_spatial,
                 channel_type,
                 patnet_t0 = True
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        self.n_div = n_div
        self.print_n_div=print_n_div
        self.auto_div = auto_div
        self.u_regular = u_regular
        self.l_gate = l_gate
        self.index_div=index_div
        self.pre_epoch=pre_epoch
        self.split_shortcut = True if channel_type == "self" else False

        if self.auto_div:
            self.spatial_mixing = DGConv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, sort=False, 
                        u_regular=self.u_regular, l_gate=self.l_gate, index_div=self.index_div, n_div=self.n_div, 
                        print_n_div=self.print_n_div, pre_epoch=self.pre_epoch )
        else:
            self.spatial_mixing = Partial_conv3(
                dim, 
                n_div, 
                pconv_fw_type, 
                use_channel,
                channel_type,
                patnet_t0,
                )
            
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_spatial:
            mlp_layer: List[nn.Module] = [
                nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
                norm_layer(mlp_hidden_dim),
                act_layer(),
                nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),
                partial_spatial_attn_layer_reverse(dim, 1)]
        else:
            mlp_layer: List[nn.Module] = [
                nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),  
                norm_layer(mlp_hidden_dim),
                act_layer(),
                nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)]

        self.mlp = nn.Sequential(*mlp_layer)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward_impl = self.forward_layer_scale
        else:
            self.forward_impl = self.forward_base

    def forward_base(self, x: Tensor) -> Tensor:
        if self.split_shortcut:
            x = x + self.spatial_mixing(x)
            x = x + self.drop_path(self.mlp(x))
        else:
            shortcut = x 
            x = self.spatial_mixing(x)
            x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        if self.split_shortcut:
            x = x + self.spatial_mixing(x)
            x = x + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        else:
            shortcut = x
            x = self.spatial_mixing(x)
            x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x
    
    def forward(self, x):
        return self.forward_impl(x)

class BasicStage(nn.Module):
    def __init__(self,
                dim,
                depth,
                n_div,
                print_n_div,
                auto_div,
                u_regular,
                l_gate,
                index_div,
                mlp_ratio,
                drop_path,
                layer_scale_init_value,
                norm_layer,
                act_layer,
                pconv_fw_type,
                pre_epoch,
                use_channel,
                use_spatial,
                channel_type='',
                patnet_t0 =True
                ):
        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                print_n_div=print_n_div,
                auto_div=auto_div,
                u_regular=u_regular,
                l_gate=l_gate,
                index_div=index_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type,
                pre_epoch=pre_epoch,
                use_channel=use_channel,
                use_spatial=use_spatial,
                channel_type=channel_type,
                patnet_t0= patnet_t0,
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x


class PartialNet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(1, 2, 8, 2),
                 mlp_ratio=2.,
                 n_div=4,
                 print_n_div=False,
                 auto_div=False,
                 u_regular=False,
                 l_gate=False,
                 index_div=False,
                 patch_size=4,
                 patch_stride=4,
                 patch_size2=2,
                 patch_stride2=2,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='BN',
                 act_layer='RELU',
                 fork_feat=True,  
                 pconv_fw_type='split_cat',
                 pre_epoch=100,
                 use_channel_attn=True,
                 use_spatial_attn=True,
                 patnet_t0=True,
                 img_size=224,
                 **kwargs):
        super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            act_layer = partial(nn.ReLU, inplace=True)

        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.fork_feat = fork_feat
        self.in_chans = in_chans
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
                               n_div=n_div,
                               print_n_div=print_n_div,
                               auto_div=auto_div,
                               u_regular=u_regular,
                               l_gate=l_gate,
                               index_div=index_div,
                               depth=depths[i_stage],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type,
                               pre_epoch=pre_epoch,
                               use_channel=use_channel_attn,
                               use_spatial=use_spatial_attn,
                               channel_type='se' if i_stage<=2 else 'self',
                               patnet_t0=patnet_t0,
                               )
            stages_list.append(stage)

            if i_stage < self.num_stages - 1:
                stages_list.append(PatchMerging(
                                        patch_size2=patch_size2,
                                        patch_stride2=patch_stride2,
                                        dim=int(embed_dim * 2 ** i_stage),
                                        norm_layer=norm_layer))

        self.stages = nn.Sequential(*stages_list)

        # Output indices for FPN
        self.out_indices = [0, 2, 4, 6] 

        for i_emb, i_layer in enumerate(self.out_indices):
            layer = norm_layer(int(embed_dim * 2 ** i_emb))
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        
        if not self.fork_feat:
            self.avgpool_pre_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
                act_layer()
            )
            self.head = nn.Linear(feature_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        
        # --- Add width_list calculation ---
        self.width_list = []
        try:
            # 暫時切換到 eval 模式以避免 batchnorm 統計更新
            training_state = self.training
            self.eval()
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size)
            # 強制使用 _forward_det_impl 獲取特徵維度
            with torch.no_grad():
                features = self._forward_det_impl(dummy_input)
            self.width_list = [f.size(1) for f in features]
            if training_state:
                self.train()
        except Exception as e:
            print(f"Error during dummy forward pass for width_list: {e}")
            # Fallback calculation
            self.width_list = [int(embed_dim * 2 ** i) for i in range(self.num_stages)]
            self.train()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_cls(self, x):
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def _forward_det_impl(self, x: Tensor) -> List[Tensor]:
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                if hasattr(self, f'norm{idx}'):
                    norm_layer = getattr(self, f'norm{idx}')
                    x_out = norm_layer(x)
                    outs.append(x_out)
                else:
                    outs.append(x)
        return outs

    def forward_det(self, x: Tensor) -> List[Tensor]:
        return self._forward_det_impl(x)
    
    def forward(self, x):
        if self.fork_feat:
            return self.forward_det(x)
        else:
            return self.forward_cls(x)


# --- Factory Functions ---

def partialnet_s(pretrained=False, img_size=224, **kwargs):
    # 強制設定 fork_feat=True 用於 Object Detection Backbone
    kwargs.setdefault('fork_feat', True)
    model = PartialNet(
        img_size=img_size,
        embed_dim=96, 
        depths=[2, 2, 9, 4], 
        mlp_ratio=2.,
        n_div=4,
        patnet_t0=False,
        **kwargs
    )
    return model

def partialnet_m(pretrained=False, img_size=224, **kwargs):
    kwargs.setdefault('fork_feat', True)
    model = PartialNet(
        img_size=img_size,
        embed_dim=128, 
        depths=[2, 3, 16, 4], 
        mlp_ratio=2.,
        n_div=4,
        drop_path_rate=0.2,
        patnet_t0=False,
        **kwargs
    )
    return model

def partialnet_l(pretrained=False, img_size=224, **kwargs):
    kwargs.setdefault('fork_feat', True)
    model = PartialNet(
        img_size=img_size,
        embed_dim=160,
        depths=[2, 3, 20, 4],
        mlp_ratio=2.,
        n_div=4,
        drop_path_rate=0.3,
        patnet_t0=False,
        **kwargs
    )
    return model


if __name__ == '__main__':
    img_h, img_w = 640, 640
    print("--- Creating PartialNet Tiny model ---")
    # Simulate backbone usage (fork_feat=True)
    model = partialnet_s(img_size=img_h) 
    print("Model created successfully.")
    print("Calculated width_list:", model.width_list)

    input_tensor = torch.rand(1, 3, img_h, img_w)
    print(f"\n--- Testing PartialNet Tiny forward pass (fork_feat={model.fork_feat}) ---")

    model.eval()
    with torch.no_grad():
        output_features = model(input_tensor)
    
    print("Forward pass successful.")
    if isinstance(output_features, list):
        print("Output is a list (Correct for backbone).")
        for i, features in enumerate(output_features):
            print(f"Stage output {i}: {features.shape}")
    else:
        print("Output is not a list!")

    # Verify width_list matches
    runtime_widths = [f.size(1) for f in output_features]
    print("\nRuntime feature channels:", runtime_widths)
    assert model.width_list == runtime_widths, "Width list mismatch!"
    print("Width list verified.")