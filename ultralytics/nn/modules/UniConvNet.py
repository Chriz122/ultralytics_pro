# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

try:
    import DCNv3
    import pkg_resources
    # 檢查 DCNv3 版本
    try:
        dcn_version = float(pkg_resources.get_distribution('DCNv3').version)
    except pkg_resources.DistributionNotFound:
        dcn_version = 0.0
        warnings.warn("DCNv3 package not found. Assuming version 0.0.")
except ImportError:
    DCNv3 = None
    dcn_version = 0.0
    warnings.warn("DCNv3 package is not installed. DCNv3_cuda module will not be available.")


class DCNv3Function(Function):
    @staticmethod
    @custom_fwd
    def forward(
            ctx, input, offset, mask,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w,
            group, group_channels, offset_scale, im2col_step, remove_center):
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center

        args = [
            input, offset, mask, kernel_h,
            kernel_w, stride_h, stride_w, pad_h,
            pad_w, dilation_h, dilation_w, group,
            group_channels, offset_scale, ctx.im2col_step
        ]
        if remove_center or dcn_version > 1.0:
            args.append(remove_center)

        output = DCNv3.dcnv3_forward(*args)
        ctx.save_for_backward(input, offset, mask)

        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        input, offset, mask = ctx.saved_tensors

        args = [
            input, offset, mask, ctx.kernel_h,
            ctx.kernel_w, ctx.stride_h, ctx.stride_w, ctx.pad_h,
            ctx.pad_w, ctx.dilation_h, ctx.dilation_w, ctx.group,
            ctx.group_channels, ctx.offset_scale, grad_output.contiguous(), ctx.im2col_step
        ]
        if ctx.remove_center or dcn_version > 1.0:
            args.append(ctx.remove_center)

        grad_input, grad_offset, grad_mask = \
            DCNv3.dcnv3_backward(*args)

        # ================================== FIX STARTS HERE ==================================
        # 修正：返回的梯度數量必須與 forward 的輸入參數數量一致 (16個)
        return grad_input, grad_offset, grad_mask, \
            None, None, None, None, None, None, None, None, None, None, None, None, None
        # =================================== FIX ENDS HERE ===================================

    @staticmethod
    def symbolic(g, input, offset, mask, kernel_h, kernel_w, stride_h,
                 stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
                 group_channels, offset_scale, im2col_step, remove_center):
        """Symbolic function for mmdeploy::DCNv3.

        Returns:
            DCNv3 op for onnx.
        """
        return g.op(
            'mmdeploy::TRTDCNv3',
            input,
            offset,
            mask,
            kernel_h_i=int(kernel_h),
            kernel_w_i=int(kernel_w),
            stride_h_i=int(stride_h),
            stride_w_i=int(stride_w),
            pad_h_i=int(pad_h),
            pad_w_i=int(pad_w),
            dilation_h_i=int(dilation_h),
            dilation_w_i=int(dilation_w),
            group_i=int(group),
            group_channels_i=int(group_channels),
            offset_scale_f=float(offset_scale),
            im2col_step_i=int(im2col_step),
            remove_center=int(remove_center),
        )


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))

    return (n & (n - 1) == 0) and n != 0


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


class DCNv3_cuda(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        dw_kernel_size=None,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer='GELU',
        norm_layer='LN',
        center_feature_scale=False,
        remove_center=False,
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if DCNv3 is None:
            raise ImportError("DCNv3 package is not installed, which is required for DCNv3_cuda.")
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels)
        )
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center) * 2)
        self.mask = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        # 檢查輸入張量是否在CPU上
        if input.device.type == 'cpu':
            # DCNv3 的核心操作並未在CPU上實現。
            # 為了允許在CPU上進行的啞元前向傳播（例如計算步長）能夠成功，
            # 我們繞過CUDA核心，直接返回一個形狀正確的張量。
            # 這對於形狀推斷是足夠的，但計算結果本身是不正確的。
            # 在實際的訓練和推論中，模型和數據必須位於GPU上。
            warnings.warn(
                "DCNv3_cuda received a CPU tensor. Bypassing the CUDA kernel for initialization. "
                "Make sure to move the model and data to a CUDA device for actual training or inference."
            )
            # 經過輸入和輸出投影層以確保輸出的形狀和類型正確
            return self.output_proj(self.input_proj(input))

        N, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1).permute(0, 2, 3, 1)
        offset = self.offset(x1)
        mask = self.mask(x1)
        
        x = DCNv3Function.apply(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center)
        
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x


class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a1 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 4, 7, padding=3, groups=dim // 4)
        )
        self.v1 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v11 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v12 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.conv3_1 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.a2 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, 9, padding=4, groups=dim // 2)
        )
        self.v2 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.v21 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.v22 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj2 = nn.Conv2d(dim // 2, dim // 4, 1)
        self.conv3_2 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.norm3 = LayerNorm(dim * 3 // 4, eps=1e-6, data_format="channels_first")
        self.a3 = nn.Sequential(
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 11, padding=5, groups=dim * 3 // 4)
        )
        self.v3 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v31 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v32 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj3 = nn.Conv2d(dim * 3 // 4, dim // 4, 1)
        self.conv3_3 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

    def forward(self, x):
        x = self.norm1(x)
        x_split = torch.split(x, self.dim // 4, dim=1)
        a = self.a1(x_split[0])
        mul = a * self.v1(x_split[0])
        mul = self.v11(mul)
        x1 = self.conv3_1(self.v12(x_split[1]))
        x1 = x1 + a
        x1 = torch.cat((x1, mul), dim=1)

        x1 = self.norm2(x1)
        a = self.a2(x1)
        mul = a * self.v2(x1)
        mul = self.v21(mul)
        x2 = self.conv3_2(self.v22(x_split[2]))
        x2 = x2 + self.proj2(a)
        x2 = torch.cat((x2, mul), dim=1)

        x2 = self.norm3(x2)
        a = self.a3(x2)
        mul = a * self.v3(x2)
        mul = self.v31(mul)
        x3 = self.conv3_3(self.v32(x_split[3]))
        x3 = x3 + self.proj3(a)
        x = torch.cat((x3, mul), dim=1)

        return x


class Block(nn.Module):
    def __init__(self, dim,
                 drop=0.,
                 drop_path=0.,
                 mlp_ratio=4,
                 layer_scale_init_value=1e-5,
                 ):
        super().__init__()
        self.attn = ConvMod(dim)
        self.mlp = MLPLayer(in_features=dim,
                            hidden_features=int(dim * mlp_ratio),
                            drop=drop)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = build_norm_layer(dim, 'LN')
        self.norm2 = build_norm_layer(dim, 'LN')
        self.dcn = DCNv3_cuda(
            channels=dim,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=dim // 16 if dim % 16 == 0 and dim > 16 else dim // 4, # 確保 group 可整除
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
        )

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x_permuted = x.permute(0, 2, 3, 1)
        x_res = self.drop_path(self.gamma1 * self.dcn(self.norm1(x_permuted)))
        x_permuted = x_permuted + x_res
        x_permuted = x_permuted + self.drop_path(self.gamma2 * self.mlp(self.norm2(x_permuted)))
        return x_permuted.permute(0, 3, 1, 2)


class UniConvNet(nn.Module):
    r""" UniConvNet
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[2, 2, 8, 2], dims=[64, 128, 256, 512], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., drop=0., img_size=224
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.in_chans = in_chans
        self.dims = dims
        self.img_size = img_size

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Dropout(drop)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        # _init_deform_weights is deprecated as DCNv3 has its own reset.
        # self.apply(self._init_deform_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

        # --- Add width_list calculation ---
        self.width_list = []
        try:
            device = next(self.parameters()).device
            self.eval()
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size).to(device)
            with torch.no_grad():
                features = self.forward_features(dummy_input)
            self.width_list = [f.size(1) for f in features]
            self.train()
        except Exception as e:
            warnings.warn(f"Error during dummy forward pass for width_list: {e}")
            self.width_list = self.dims
            self.train()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_deform_weights(self, m):
        if isinstance(m, DCNv3_cuda):
            m._reset_parameters()

    def forward_features(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features

    def forward(self, x):
        x = self.forward_features(x)
        return x
    
@register_model
def UniConvNet_A(**kwargs):
    model = UniConvNet(depths=[2, 3, 9, 2], dims=[24, 48, 96, 192], **kwargs)
    return model

@register_model
def UniConvNet_P0(**kwargs):
    model = UniConvNet(depths=[2, 2, 7, 2], dims=[32, 64, 128, 256], **kwargs)
    return model

@register_model
def UniConvNet_P1(**kwargs):
    model = UniConvNet(depths=[2, 3, 6, 3], dims=[32, 64, 128, 256], **kwargs)
    return model

@register_model
def UniConvNet_P2(**kwargs):
    model = UniConvNet(depths=[3, 3, 11, 3], dims=[32, 64, 128, 256], **kwargs)
    return model

@register_model
def UniConvNet_N0(**kwargs):
    model = UniConvNet(depths=[2, 2, 7, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

@register_model
def UniConvNet_N1(**kwargs):
    model = UniConvNet(depths=[2, 2, 8, 3], dims=[48, 96, 192, 384], **kwargs)
    return model

@register_model
def UniConvNet_N2(**kwargs):
    model = UniConvNet(depths=[3, 3, 11, 3], dims=[48, 96, 192, 384], **kwargs)
    return model

@register_model
def UniConvNet_N3(**kwargs):
    model = UniConvNet(depths=[3, 3, 19, 3], dims=[48, 96, 192, 384], **kwargs)
    return model

@register_model
def UniConvNet_T(**kwargs):
    model = UniConvNet(depths=[3, 3, 15, 3], dims=[64, 128, 256, 512], **kwargs)
    return model

@register_model
def UniConvNet_S(**kwargs):
    model = UniConvNet(depths=[3, 3, 17, 3], dims=[80, 160, 320, 640], **kwargs)
    return model

@register_model
def UniConvNet_B(**kwargs):
    model = UniConvNet(depths=[4, 4, 13, 4], dims=[112, 224, 448, 896], **kwargs)
    return model

@register_model
def UniConvNet_L(**kwargs):
    model = UniConvNet(depths=[3, 3, 18, 3], dims=[160, 320, 640, 1280], **kwargs)
    return model

@register_model
def UniConvNet_XL(**kwargs):
    model = UniConvNet(depths=[3, 3, 22, 3], dims=[160, 320, 640, 1280], **kwargs)
    return model


if __name__ == '__main__':
    # 檢查是否有可用的 CUDA 設備
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping DCNv3 CUDA module test.")
    elif DCNv3 is None:
        print("DCNv3 package not found. Skipping DCNv3 CUDA module test.")
    else:
        device = torch.device("cuda")
        # 測試程式碼
        img_h, img_w = 640, 640
        print("--- Creating UniConvNet_T model ---")
        # 傳入 img_size 以便 dummy forward pass
        # 將模型移動到 GPU
        model = UniConvNet_T(img_size=img_h).to(device)
        print("Model created successfully and moved to GPU.")
        # 在 `__init__` 中 width_list 已經在 GPU 上計算，這裡可以選擇重新驗證
        print("Calculated width_list:", model.width_list)

        # 測試前向傳播
        # 創建一個在 GPU 上的輸入張量
        input_tensor = torch.rand(2, 3, img_h, img_w).to(device)
        print(f"\n--- Testing UniConvNet_T forward pass (Input: {input_tensor.shape} on {input_tensor.device}) ---")

        model.eval()
        try:
            with torch.no_grad():
                output_features = model(input_tensor)
            print("Forward pass successful.")
            print("Output feature shapes:")
            for i, features in enumerate(output_features):
                # 輸出應為 [B, C, H_i, W_i]
                print(f"Stage {i + 1}: {features.shape} on {features.device}")

            # 驗證 width_list 是否與執行時的輸出匹配
            runtime_widths = [f.size(1) for f in output_features]
            print("\nRuntime output feature channels:", runtime_widths)
            assert model.width_list == runtime_widths, "Width list mismatch!"
            print("Width list verified successfully.")

            # --- 測試 deepcopy ---
            print("\n--- Testing deepcopy ---")
            import copy
            # 深拷貝會創建一個在相同設備上的新模型
            copied_model = copy.deepcopy(model)
            print("Deepcopy successful.")

            # 測試複製後模型的前向傳播
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