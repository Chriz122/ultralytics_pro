# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MetaFormer baselines including IdentityFormer, RandFormer, PoolFormerV2,
ConvFormer and CAFormer.
Some implementations are modified from timm (https://github.com/rwightman/pytorch-image-models).

Modified to follow a structure similar to MobileNetV4 definition.
"""
import os
from functools import partial
from typing import List, Optional, Tuple, Union, Type, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers.helpers import to_2tuple


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'dfformer_s18': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/dfformer_s18.pth"),
    'dfformer_s36': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/dfformer_s36.pth"),
    'dfformer_m36': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/dfformer_m36.pth"),
    'dfformer_b36': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/dfformer_b36.pth"),
    'gfformer_s18': _cfg(), # No pretrained weights provided in original code for this
    'cdfformer_s18': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/cdfformer_s18.pth"),
    'cdfformer_s36': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/cdfformer_s36.pth"),
    'cdfformer_m36': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/cdfformer_m36.pth"),
    'cdfformer_b36': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/cdfformer_b36.pth"),
    'dfformer_s18_k2': _cfg(),
    'dfformer_s18_d8': _cfg(),
    'dfformer_s18_gelu': _cfg(),
    'dfformer_s18_relu': _cfg(),
    'dfformer_s18_afno': _cfg(), # Shares cfg with k2 in original, might need adjustment
}

# --- Helper Modules (Copied from original Code 1, no changes needed) ---

class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class SquaredReLU(nn.Module):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0: self.num_heads = 1
        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RandomMixing(nn.Module):
    def __init__(self, num_tokens=196, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1),
            requires_grad=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x


class LayerNormGeneral(nn.Module):
    # ... (LayerNormGeneral definition remains the same)
    def __init__(self, affine_shape=None, normalized_dim=(-1,), scale=True,
                 bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class GlobalFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, size=14, # Size passed here determines init shape of complex_weights
                 **kwargs, ):
        super().__init__()
        size = to_2tuple(size)
        # These store the *initialization* size
        self.init_h = size[0]
        self.init_filter_size = size[1] // 2 + 1 # W_fft at init

        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        # Initialize weights based on the size passed during construction
        self.complex_weights = nn.Parameter(
            torch.randn(self.init_h, self.init_filter_size, self.med_channels, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        B, H, W, _ = x.shape # Actual runtime dimensions
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)

        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        # x_fft shape: [B, H, W//2 + 1, C']

        # --- Resize complex_weights to match runtime x_fft dimensions ---
        runtime_h, runtime_w_fft = x_fft.shape[1], x_fft.shape[2]
        if self.complex_weights.shape[0] != runtime_h or self.complex_weights.shape[1] != runtime_w_fft:
             # Use detach() if complex_weights should remain non-trainable during resize op itself,
             # but the resized result should still track gradients back to original weights.
             # Or just resize directly if nn.Parameter handles requires_grad correctly.
            resized_complex_weights_realimag = resize_complex_weight(
                self.complex_weights, runtime_h, runtime_w_fft
            )
            complex_weights = torch.view_as_complex(resized_complex_weights_realimag)
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)
        # complex_weights shape is now [H, W//2 + 1, C'] (matching x_fft)

        # Element-wise multiplication in frequency domain
        x_filtered = x_fft * complex_weights # Shape: [B, H, W//2+1, C']

        # Inverse FFT
        x_out = torch.fft.irfft2(x_filtered, s=(H, W), dim=(1, 2), norm='ortho')
        x_out = self.act2(x_out)
        x_out = self.pwconv2(x_out)
        return x_out


class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, # Size passed here determines init shape
                 weight_resize=True, # Keep the arg, but maybe force True internally? Or remove. Let's keep default True.
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        # Store init size
        self.init_h = size[0]
        self.init_filter_size = size[1] // 2 + 1 # W_fft at init

        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        # Force resize behavior if needed, ignore weight_resize arg?
        # self.weight_resize = True # Force it
        self.weight_resize = weight_resize # Or keep it flexible if needed elsewhere

        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        # Reweight MLP input dim is the original dim C, output is num_filters * med_channels
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        # Initialize weights based on the size passed during construction
        self.complex_weights = nn.Parameter(
            torch.randn(self.init_h, self.init_filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        B, H, W, C_in = x.shape # Actual runtime dimensions

        # Calculate routing weights based on global average pooling of input
        # Input to reweight is [B, C_in], output is [B, num_filters * med_channels]
        routeing_logits = self.reweight(x.mean(dim=(1, 2)))
        # Reshape to [B, num_filters, med_channels] and apply softmax over filters
        routeing = routeing_logits.view(B, self.num_filters, self.med_channels).softmax(dim=1)

        # Process input feature map
        x_proc = self.pwconv1(x)    # Shape: [B, H, W, med_channels]
        x_proc = self.act1(x_proc)
        x_proc = x_proc.to(torch.float32)

        # FFT
        x_fft = torch.fft.rfft2(x_proc, dim=(1, 2), norm='ortho')
        # x_fft shape: [B, H, W//2 + 1, med_channels]

        # --- Prepare complex weights ---
        runtime_h, runtime_w_fft = x_fft.shape[1], x_fft.shape[2]
        current_complex_weights = self.complex_weights # Shape: [init_h, init_filter_size, num_filters, 2]

        # Resize if necessary OR if self.weight_resize is True
        if self.weight_resize or current_complex_weights.shape[0] != runtime_h or current_complex_weights.shape[1] != runtime_w_fft:
            resized_complex_weights_realimag = resize_complex_weight(
                current_complex_weights, runtime_h, runtime_w_fft
            )
            # resized shape: [runtime_h, runtime_w_fft, num_filters, 2]
        else:
            resized_complex_weights_realimag = current_complex_weights

        # Convert to complex tensor
        complex_weights_resized = torch.view_as_complex(resized_complex_weights_realimag)
        # complex_weights_resized shape: [runtime_h, runtime_w_fft, num_filters] (complex)

        # Combine routing weights with filter kernels
        # routeing shape: [B, num_filters, med_channels] (real)
        # complex_weights_resized shape: [runtime_h, runtime_w_fft, num_filters] (complex)
        # Target weight shape: [B, H, W_fft, med_channels] (complex)

        # Cast routeing to complex
        routeing_c = routeing.to(torch.complex64) # Shape: [B, num_filters, med_channels]

        # Einsum: Combine filter kernels based on routing weights for each channel
        # 'bfc,hwf->bhwc' : b=batch, f=filter, c=channel, h=height, w=width_fft
        # Needs routeing: [B, F, C], complex_weights: [H, W_fft, F]
        # Output: [B, H, W_fft, C]
        weight = torch.einsum('bfc,hwf->bhwc', routeing_c, complex_weights_resized)
        # weight shape: [B, runtime_h, runtime_w_fft, med_channels] (complex)

        # Apply filters: Element-wise multiplication in frequency domain
        # x_fft shape: [B, H, W//2 + 1, med_channels]
        # weight shape: [B, H, W//2 + 1, med_channels]
        x_filtered = x_fft * weight # THIS IS THE OPERATION THAT FAILED BEFORE

        # Inverse FFT
        x_out = torch.fft.irfft2(x_filtered, s=(H, W), dim=(1, 2), norm='ortho')
        # x_out shape: [B, H, W, med_channels]

        # Final layers
        x_out = self.act2(x_out)
        x_out = self.pwconv2(x_out) # Shape: [B, H, W, dim]
        return x_out



class SepConv(nn.Module):
    # ... (SepConv definition remains the same)
    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, kernel_size=7, padding=3,
                 **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class AFNO2D(nn.Module):
    # ... (AFNO2D definition remains the same)
    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, size=14,
                 num_blocks=8, sparsity_threshold=0.01,
                 hard_thresholding_fraction=1, hidden_size_factor=1,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        assert self.med_channels % num_blocks == 0, f"hidden_size {self.med_channels} should be divisble by num_blocks {num_blocks}"
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.med_channels // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        bias_ = x # Renamed bias to bias_ to avoid potential conflict
        B, H, W, C = x.shape
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, self.size, self.filter_size, self.num_blocks, self.block_size)
        o1_real = torch.zeros([B, self.size, self.filter_size, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, self.size, self.filter_size, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)
        kept_modes = int(self.filter_size * self.hard_thresholding_fraction)
        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )
        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )
        o2_real[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )
        o2_imag[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, self.size, self.filter_size, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x + bias_ # Use renamed bias_
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Pooling(nn.Module):
    # ... (Pooling definition remains the same)
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x


class Mlp(nn.Module): # Make sure Mlp is defined before DynamicFilter uses it
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpHead(nn.Module):
    # ... (MlpHead definition remains the same)
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU,
                 norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(nn.Module):
    # ... (MetaFormerBlock definition remains the same)
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None,
                 size=14,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop, size=size)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x

# --- Utilities (Copied from original Code 1) ---

DOWNSAMPLE_LAYERS_FOUR_STAGES = [partial(Downsampling,
                                         kernel_size=7, stride=4, padding=2,
                                         post_norm=partial(LayerNormGeneral, bias=False,
                                                           eps=1e-6)
                                         )] + \
                                [partial(Downsampling,
                                         kernel_size=3, stride=2, padding=1,
                                         pre_norm=partial(LayerNormGeneral, bias=False,
                                                          eps=1e-6), pre_permute=True
                                         )] * 3


def resize_complex_weight(origin_weight, new_h, new_w):
    """ Resize complex weights for Dynamic/Global Filter """
    # origin_weight shape: [H, W_fft, Components..., 2]
    # Components could be num_filters or med_channels
    h, w_fft = origin_weight.shape[0:2]
    num_components = origin_weight.shape[2:-1] # Capture potential extra dimensions like num_filters
    origin_weight_reshaped = origin_weight.reshape(h, w_fft, -1, 2) # Flatten components
    total_components = origin_weight_reshaped.shape[2]

    # Permute to B, C, H, W format for interpolate (B=1, C=total_components*2)
    origin_weight_permuted = origin_weight_reshaped.permute(2, 3, 0, 1).reshape(1, total_components * 2, h, w_fft)

    new_weight = torch.nn.functional.interpolate(
        origin_weight_permuted.float(),  # Interpolate requires float
        size=(new_h, new_w),             # Target spatial dimensions (H, W_fft)
        mode='bicubic',
        align_corners=True               # Use align_corners=True for frequency domain interpolation
    )

    # Reshape back: B, C, new_h, new_w -> new_h, new_w, Components..., 2
    new_weight_reshaped = new_weight.reshape(total_components, 2, new_h, new_w).permute(2, 3, 0, 1)
    final_shape = (new_h, new_w) + num_components + (2,)
    return new_weight_reshaped.reshape(final_shape).contiguous()


def load_weights(model: nn.Module, input_size: Tuple[int, int, int], url: Optional[str] = None):
    """ Loads pretrained weights, adjusting complex weights if necessary. """
    if not url:
        print("Warning: No URL provided for pretrained weights.")
        return

    state_dict = torch.hub.load_state_dict_from_url(
        url=url, map_location="cpu", check_hash=True
    )
    out_dict = {}
    current_input_size = input_size # Use the provided input size

    for k, v in state_dict.items():
        adjusted_v = v
        if 'complex_weights' in k:
            # Determine the stage based on the key name
            stage_index = -1
            if 'stages.0.' in k: stage_index = 0
            elif 'stages.1.' in k: stage_index = 1
            elif 'stages.2.' in k: stage_index = 2
            elif 'stages.3.' in k: stage_index = 3

            if stage_index != -1:
                # Calculate expected H, W for this stage's feature map
                # Input: H, W -> Stage 0: H/4, W/4 -> Stage 1: H/8, W/8 -> etc.
                expected_h = current_input_size[1] // (2**(stage_index + 2))
                # Filter size is W/2 + 1 for rfft2
                expected_w = current_input_size[2] // (2**(stage_index + 3)) + 1 # // (stride * 2) + 1

                # Check if loaded weights match expected size
                if v.shape[0] != expected_h or v.shape[1] != expected_w:
                    print(f"Resizing complex_weights {k} from {v.shape[:2]} to ({expected_h}, {expected_w})")
                    adjusted_v = resize_complex_weight(v, expected_h, expected_w)
            else:
                 print(f"Warning: Could not determine stage for complex_weights key: {k}")

        out_dict[k] = adjusted_v

    # Load weights with strict=False to ignore missing/unexpected keys (like head if num_classes differs)
    load_result = model.load_state_dict(out_dict, strict=False)
    print(f"Pretrained weights load result: {load_result}")


# --- Configuration Dictionary for MetaFormer Variants ---
METAFORMER_SPECS: Dict[str, Dict[str, Any]] = {
    'dfformer_s18': {
        'depths': [3, 3, 9, 3],
        'dims': [64, 128, 320, 512],
        'token_mixers': DynamicFilter,
        'mlps': Mlp,
        'norm_layers': partial(LayerNormGeneral, eps=1e-6, bias=False),
        'head_fn': MlpHead,
    },
    'dfformer_s36': {
        'depths': [3, 12, 18, 3],
        'dims': [64, 128, 320, 512],
        'token_mixers': DynamicFilter,
        'mlps': Mlp,
        'norm_layers': partial(LayerNormGeneral, eps=1e-6, bias=False),
        'head_fn': MlpHead,
    },
    'dfformer_m36': {
        'depths': [3, 12, 18, 3],
        'dims': [96, 192, 384, 576],
        'token_mixers': DynamicFilter,
        'mlps': Mlp,
        'norm_layers': partial(LayerNormGeneral, eps=1e-6, bias=False),
        'head_fn': MlpHead,
    },
    'dfformer_b36': {
        'depths': [3, 12, 18, 3],
        'dims': [128, 256, 512, 768],
        'token_mixers': DynamicFilter,
        'mlps': Mlp,
        'norm_layers': partial(LayerNormGeneral, eps=1e-6, bias=False),
        'head_fn': MlpHead,
    },
    'gfformer_s18': {
        'depths': [3, 3, 9, 3],
        'dims': [64, 128, 320, 512],
        'token_mixers': GlobalFilter,
        'mlps': Mlp,
        'norm_layers': partial(LayerNormGeneral, eps=1e-6, bias=False),
        'head_fn': MlpHead,
    },
    'cdfformer_s18': {
        'depths': [3, 3, 9, 3],
        'dims': [64, 128, 320, 512],
        'token_mixers': [SepConv, SepConv, DynamicFilter, DynamicFilter],
        'mlps': Mlp,
        'norm_layers': partial(LayerNormGeneral, eps=1e-6, bias=False),
        'head_fn': MlpHead,
    },
    'cdfformer_s36': {
        'depths': [3, 12, 18, 3],
        'dims': [64, 128, 320, 512],
        'token_mixers': [SepConv, SepConv, DynamicFilter, DynamicFilter],
        'mlps': Mlp,
        'norm_layers': partial(LayerNormGeneral, eps=1e-6, bias=False),
        'head_fn': MlpHead,
    },
    'cdfformer_m36': {
        'depths': [3, 12, 18, 3],
        'dims': [96, 192, 384, 576],
        'token_mixers': [SepConv, SepConv, DynamicFilter, DynamicFilter],
        'mlps': Mlp,
        'norm_layers': partial(LayerNormGeneral, eps=1e-6, bias=False),
        'head_fn': MlpHead,
    },
    'cdfformer_b36': {
        'depths': [3, 12, 18, 3],
        'dims': [128, 256, 512, 768],
        'token_mixers': [SepConv, SepConv, DynamicFilter, DynamicFilter],
        'mlps': Mlp,
        'norm_layers': partial(LayerNormGeneral, eps=1e-6, bias=False),
        'head_fn': MlpHead,
    },
    # Ablation models
     'dfformer_s18_gelu': {
        'depths': [3, 3, 9, 3],
        'dims': [64, 128, 320, 512],
        'token_mixers': partial(DynamicFilter, act1_layer=nn.GELU),
        'mlps': partial(Mlp, act_layer=nn.GELU),
        'norm_layers': partial(LayerNormGeneral, eps=1e-6, bias=False),
        'head_fn': MlpHead,
     },
    'dfformer_s18_relu': {
        'depths': [3, 3, 9, 3],
        'dims': [64, 128, 320, 512],
        'token_mixers': partial(DynamicFilter, act1_layer=nn.ReLU),
        'mlps': partial(Mlp, act_layer=nn.ReLU),
        'norm_layers': partial(LayerNormGeneral, eps=1e-6, bias=False),
        'head_fn': MlpHead,
    },
    'dfformer_s18_k2': {
        'depths': [3, 3, 9, 3],
        'dims': [64, 128, 320, 512],
        'token_mixers': partial(DynamicFilter, num_filters=2),
        'mlps': Mlp,
        'norm_layers': partial(LayerNormGeneral, eps=1e-6, bias=False),
        'head_fn': MlpHead,
    },
    'dfformer_s18_d8': {
        'depths': [3, 3, 9, 3],
        'dims': [64, 128, 320, 512],
        'token_mixers': partial(DynamicFilter, reweight_expansion_ratio=.125),
        'mlps': Mlp,
        'norm_layers': partial(LayerNormGeneral, eps=1e-6, bias=False),
        'head_fn': MlpHead,
    },
    'dfformer_s18_afno': {
        'depths': [3, 3, 9, 3],
        'dims': [64, 128, 320, 512],
        'token_mixers': AFNO2D,
        'mlps': Mlp,
        'norm_layers': partial(LayerNormGeneral, eps=1e-6, bias=False),
        'head_fn': MlpHead,
    },
}


# --- Main MetaFormer Class (Modified) ---
class MetaFormer(nn.Module):
    r""" MetaFormer based on the configuration dictionary approach.
        Designed to output a list of feature maps for downstream tasks like detection,
        while optionally providing classification via a separate `forward_head` method.
    """
    def __init__(self, model_name: str,
                 in_chans: int = 3,
                 num_classes: int = 1000, # Used for optional classification head
                 downsample_layers: List[Type[nn.Module]] = DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 drop_path_rate: float = 0.,
                 head_dropout: float = 0.0,
                 layer_scale_init_values: Optional[Union[List, float]] = None,
                 res_scale_init_values: Optional[Union[List, float]] = [None, None, 1.0, 1.0],
                 # fork_feat determines if intermediate norms are applied to features
                 # Default should be True for typical detection/segmentation usage
                 fork_feat: bool = True,
                 output_norm: Type[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 input_size: Tuple[int, int, int] = (3, 224, 224),
                 **kwargs):
        super().__init__()

        if model_name not in METAFORMER_SPECS:
            raise ValueError(f"Unknown model_name: {model_name}. Available models: {list(METAFORMER_SPECS.keys())}")

        self.model_name = model_name
        self.in_chans = in_chans
        self.input_size = input_size # Store input_size for width_list and filter init
        self.model_spec = METAFORMER_SPECS[model_name] # Store spec

        # Get architecture specifics from the spec
        depths = self.model_spec['depths']
        dims = self.model_spec['dims']
        token_mixers = self.model_spec['token_mixers']
        mlps = self.model_spec.get('mlps', Mlp) # Default to Mlp if not specified
        norm_layers = self.model_spec.get('norm_layers', partial(LayerNormGeneral, eps=1e-6, bias=False))
        self.dims = dims # Store dims

        if not isinstance(depths, (list, tuple)): depths = [depths]
        if not isinstance(dims, (list, tuple)): dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        # Downsample Layers
        if not isinstance(downsample_layers, (list, tuple)): downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i + 1]) for i in range(num_stage)]
        )

        # Per-stage components
        if not isinstance(token_mixers, (list, tuple)): token_mixers = [token_mixers] * num_stage
        if not isinstance(mlps, (list, tuple)): mlps = [mlps] * num_stage
        if not isinstance(norm_layers, (list, tuple)): norm_layers = [norm_layers] * num_stage

        # Stochastic depth
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # LayerScale / ResScale values
        if layer_scale_init_values is not None and not isinstance(layer_scale_init_values, (list, tuple)):
             layer_scale_init_values = [layer_scale_init_values] * num_stage
        if res_scale_init_values is not None and not isinstance(res_scale_init_values, (list, tuple)):
             res_scale_init_values = [res_scale_init_values] * num_stage

        # Build Stages (MetaFormer Blocks)
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(num_stage):
            # Calculate feature map size for this stage based on INITIAL input_size
            # This size is passed to token mixers like DynamicFilter for INITIALIZATION
            current_h = input_size[1] // (2**(i + 2)) # After stem (//4) and i downsamples (//2 each)
            current_w = input_size[2] // (2**(i + 2))
            stage_size = (current_h, current_w)

            stage = nn.Sequential(
                *[MetaFormerBlock(dim=dims[i],
                                  token_mixer=token_mixers[i],
                                  mlp=mlps[i],
                                  norm_layer=norm_layers[i],
                                  drop_path=dp_rates[cur + j],
                                  layer_scale_init_value=layer_scale_init_values[i] if layer_scale_init_values else None,
                                  res_scale_init_value=res_scale_init_values[i] if res_scale_init_values else None,
                                  size=stage_size, # Pass calculated size based on init input_size
                                  ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Intermediate Norm layers (used only if fork_feat is True in forward_features)
        self._fork_feat = fork_feat # Store the intended setting for applying norms
        for i in range(self.num_stage):
            # Check for environment variable to potentially skip first norm (e.g., RetinaNet)
            # Note: Using env vars inside model definition isn't always ideal
            if i == 0 and os.environ.get('FORK_LAST3', None):
                layer = nn.Identity()
            else:
                layer = output_norm(dims[i])
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        # Classification Head (Optional, used by forward_head)
        _head_fn = self.model_spec.get('head_fn', nn.Linear) # Get head type from spec
        self.num_classes = num_classes
        if num_classes > 0:
             final_dim = self.dims[-1]
             if head_dropout > 0.0 and _head_fn != nn.Linear and issubclass(_head_fn, nn.Module): # Check if MlpHead etc.
                 # Assume MlpHead or similar handles dropout internally via its own args
                 self._head = _head_fn(final_dim, num_classes, head_dropout=head_dropout)
             else:
                 # Standard Linear head or MlpHead without explicit dropout arg here
                 self._head = _head_fn(final_dim, num_classes)
                 # Add dropout separately if it's just a Linear head
                 if head_dropout > 0.0 and _head_fn == nn.Linear:
                     self._head = nn.Sequential(self._head, nn.Dropout(head_dropout))

        else:
             self._head = nn.Identity() # No head if num_classes is 0 or less

        # Final Norm (used by forward_head before classification)
        self._norm_final = output_norm(self.dims[-1])

        # Calculate width_list (useful for debugging/verification)
        self.width_list = []
        try:
            self.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, self.in_chans, *self.input_size[1:])
                features = self.forward(dummy_input) # Use main forward (returns list)
                if isinstance(features, list):
                     # Features are already B, C, H, W from forward_features
                     self.width_list = [f.size(1) for f in features]
                else:
                     print("Warning: Could not compute width_list. Forward pass did not return a list.")
            self.train()
            # Verification check
            if self.width_list and len(self.width_list) == len(self.dims):
                 assert self.width_list == self.dims, \
                     f"Calculated width_list {self.width_list} does not match dims {self.dims}"
            elif self.width_list:
                 print(f"Warning: Length mismatch between width_list ({len(self.width_list)}) "
                       f"and dims ({len(self.dims)})")
        except Exception as e:
            print(f"Warning: Failed to compute width_list during initialization: {e}")
            self.train()

        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ Initialize weights for Conv2d and Linear layers. """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # LayerNorm/LayerNormGeneral/Scale weights/biases are typically handled in their own __init__

    @torch.jit.ignore
    def no_weight_decay(self):
        """ Parameters that should not have weight decay applied. """
        # Include norms, biases, scale parameters, and complex weights used in filters
        return {'norm', 'norm0', 'norm1', 'norm2', 'norm3', 'bias', 'scale', 'complex_weights'}

    def forward_features(self, x):
        """
        Extracts features from the stages.
        Applies intermediate norms (norm0, norm1, ...) ONLY if self._fork_feat is True.
        Returns a list of feature maps in [B, C, H, W] format.
        """
        outs = []
        current_x = x
        for i in range(self.num_stage):
            # Apply downsampling
            current_x = self.downsample_layers[i](current_x)
            # Pass through stage blocks
            current_x = self.stages[i](current_x) # Output shape [B, H, W, C]

            # Decide whether to normalize before appending
            feature_to_append = current_x
            if self._fork_feat:
                try:
                    norm_layer = getattr(self, f'norm{i}')
                    feature_to_append = norm_layer(current_x) # Still [B, H, W, C]
                except AttributeError:
                    # This should ideally not happen if norms are created correctly
                    print(f"Warning: Norm layer norm{i} not found. Using unnormalized output for stage {i}.")

            # Permute to [B, C, H, W] and append
            outs.append(feature_to_append.permute(0, 3, 1, 2))

        # outs contains a list of tensors, one for each stage, in B C H W format
        return outs

    def forward(self, x):
        """
        Main forward pass for using MetaFormer as a backbone.
        Returns a list of feature maps [B, C, H, W] suitable for detection/segmentation heads.
        The classification head (_head) is NOT used here.
        """
        # Directly return the list of features produced by forward_features
        return self.forward_features(x)

    # --- Optional Classification Method ---
    def forward_head(self, x, pre_logits: bool = False):
        """
        Provides a way to get classification output.
        Runs the input 'x' through the backbone, takes the last feature map,
        pools it, applies the final norm, and passes it through the classification head.

        Args:
            x (Tensor): Input tensor [B, C, H, W].
            pre_logits (bool): If True, return the feature vector before the
                               final classification layer. Default: False.

        Returns:
            Tensor: Classification logits [B, num_classes] or pre-logits [B, final_dim].
        """
        # 1. Get feature maps from the backbone
        # We need the output just before permutation from the last stage
        # Let's re-run forward_features carefully or extract from the list
        features_chw = self.forward_features(x) # Gets list [B, C, H, W]

        # 2. Get the last feature map and permute back to B, H, W, C
        last_feature_chw = features_chw[-1]
        last_feature_hwc = last_feature_chw.permute(0, 2, 3, 1).contiguous() # B, H, W, C

        # 3. Apply final norm and global average pooling
        # Note: Original MetaFormer pools first, then norms. Timm models often norm first.
        # Let's follow the original: Pool then Norm
        pooled_feature = last_feature_hwc.mean([1, 2]) # Global avg pool -> [B, C]
        normed_pooled_feature = self._norm_final(pooled_feature)

        # 4. Return pre-logits or final classification output
        if pre_logits:
            return normed_pooled_feature
        else:
            if not hasattr(self, '_head'):
                 raise RuntimeError("Model does not have a classification head (_head). "
                                    "Initialize with num_classes > 0.")
            return self._head(normed_pooled_feature)


# --- Model Registration Functions (Modified) ---

def _create_metaformer(model_name, pretrained=False, **kwargs):
    """ Helper function to create MetaFormer models """
    # print(f"Creating MetaFormer model: {model_name}")
    model = MetaFormer(model_name=model_name, **kwargs)

    # Attempt to get default_cfg, handle if model_name not in default_cfgs
    model.default_cfg = default_cfgs.get(model_name, _cfg()) # Use generic _cfg if specific one isn't found

    if pretrained:
        url = model.default_cfg.get('url', None)
        if url:
            input_size = kwargs.get('input_size', model.default_cfg['input_size'])
            print(f"Loading pretrained weights from {url} for input size {input_size}")
            load_weights(model, input_size, url)
        else:
            print(f"Warning: Pretrained weights requested for {model_name} but no URL found in default_cfg.")
    return model

# Register models using the helper
@register_model
def dfformer_s18(pretrained=False, **kwargs):
    return _create_metaformer('dfformer_s18', pretrained=pretrained, **kwargs)

@register_model
def dfformer_s36(pretrained=False, **kwargs):
    return _create_metaformer('dfformer_s36', pretrained=pretrained, **kwargs)

@register_model
def dfformer_m36(pretrained=False, **kwargs):
    return _create_metaformer('dfformer_m36', pretrained=pretrained, **kwargs)

@register_model
def dfformer_b36(pretrained=False, **kwargs):
    return _create_metaformer('dfformer_b36', pretrained=pretrained, **kwargs)

@register_model
def gfformer_s18(pretrained=False, **kwargs):
    return _create_metaformer('gfformer_s18', pretrained=pretrained, **kwargs)

@register_model
def cdfformer_s18(pretrained=False, **kwargs):
    return _create_metaformer('cdfformer_s18', pretrained=pretrained, **kwargs)

@register_model
def cdfformer_s36(pretrained=False, **kwargs):
    return _create_metaformer('cdfformer_s36', pretrained=pretrained, **kwargs)

@register_model
def cdfformer_m36(pretrained=False, **kwargs):
    return _create_metaformer('cdfformer_m36', pretrained=pretrained, **kwargs)

@register_model
def cdfformer_b36(pretrained=False, **kwargs):
    return _create_metaformer('cdfformer_b36', pretrained=pretrained, **kwargs)

# Ablation models
@register_model
def dfformer_s18_gelu(pretrained=False, **kwargs):
    return _create_metaformer('dfformer_s18_gelu', pretrained=pretrained, **kwargs)

@register_model
def dfformer_s18_relu(pretrained=False, **kwargs):
    return _create_metaformer('dfformer_s18_relu', pretrained=pretrained, **kwargs)

@register_model
def dfformer_s18_k2(pretrained=False, **kwargs):
    return _create_metaformer('dfformer_s18_k2', pretrained=pretrained, **kwargs)

@register_model
def dfformer_s18_d8(pretrained=False, **kwargs):
    return _create_metaformer('dfformer_s18_d8', pretrained=pretrained, **kwargs)

@register_model
def dfformer_s18_afno(pretrained=False, **kwargs):
    # Note: default_cfg for afno was pointing to k2 in original.
    # Adjust METAFORMER_SPECS or default_cfgs if specific cfg/weights exist.
    return _create_metaformer('dfformer_s18_afno', pretrained=pretrained, **kwargs)


# --- Example Usage (Similar to MobileNetV4 example) ---
if __name__ == "__main__":
    # Generating Sample image
    image_size_h = 224 # Example standard size
    image_size_w = 224
    image_size = (1, 3, image_size_h, image_size_w)
    image = torch.rand(*image_size)
    print(f"Input image shape: {image.shape}")

    # --- Test Classification Mode ---
    print("\n--- Testing Classification Mode (fork_feat=False) ---")
    model_cls = dfformer_s18(pretrained=False, num_classes=10, input_size=image_size[1:]) # Example: 10 classes
    model_cls.eval()
    out_cls = model_cls(image)
    print(f"Classification model: {model_cls.model_name}")
    print(f"Output (classification) shape: {out_cls.shape}")
    print(f"Calculated width list: {model_cls.width_list}")

    # --- Test Feature Extraction Mode ---
    print("\n--- Testing Feature Extraction Mode (fork_feat=True) ---")
    # input_size is crucial here for width_list and potentially weight loading
    model_feat = cdfformer_s18(pretrained=False, fork_feat=True, input_size=image_size[1:])
    model_feat.eval()
    out_feat = model_feat(image)
    print(f"Feature extraction model: {model_feat.model_name}")
    print(f"Output (features) is a list of {len(out_feat)} tensors:")
    for i, f in enumerate(out_feat):
        print(f"  Feature {i} shape: {f.shape}") # Should be B, C, H, W
    print(f"Calculated width list: {model_feat.width_list}") # Should match channel dims

    # --- Test Pretrained Loading (if URL exists) ---
    print("\n--- Testing Pretrained Loading (dfformer_s18) ---")
    try:
        # Use a different input size to test weight resizing
        large_input_size = (3, 384, 384)
        large_image = torch.rand(1, *large_input_size)
        model_pretrained = dfformer_s18(pretrained=True, num_classes=1000, input_size=large_input_size)
        model_pretrained.eval()
        out_pretrained = model_pretrained(large_image)
        print(f"Output (pretrained) shape: {out_pretrained.shape}")
        print(f"Pretrained model width list: {model_pretrained.width_list}")
    except Exception as e:
        print(f"Could not load or run pretrained dfformer_s18: {e}")
        print("(This might happen if the weights URL is invalid or download fails)")