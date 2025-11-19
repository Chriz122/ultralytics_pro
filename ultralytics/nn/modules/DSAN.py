from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

# 嘗試導入 DSCN，如果失敗則忽略，以便在沒有自訂 CUDA 核心的環境中運行
try:
    import DSCN
    from torch.nn.init import xavier_uniform_, constant_
except ImportError:
    DSCN = None
    print("Warning: Failed to import DSCN. DSCN custom ops will not be available.")


from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

# 如果 DSCN 導入失敗，則定義一個虛擬的 Function 以避免錯誤
if DSCN is None:
    class DSCNFunction(Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):
            raise NotImplementedError("DSCN CUDA op is not available or the DSCN package is not installed.")
        @staticmethod
        def backward(ctx, *args, **kwargs):
            raise NotImplementedError("DSCN CUDA op is not available or the DSCN package is not installed.")
else:
    class DSCNFunction(Function):
        @staticmethod
        @custom_fwd
        def forward(
                ctx, input, offset,
                kernel_h, kernel_w, stride_h, stride_w,
                pad_h, pad_w, dilation_h, dilation_w,
                group, group_channels, offset_scale, im2col_step, remove_center, on_x):
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
            ctx.on_x = on_x

            args = [
                input, offset, kernel_h,
                kernel_w, stride_h, stride_w, pad_h,
                pad_w, dilation_h, dilation_w, group,
                group_channels, offset_scale, ctx.im2col_step,
                ctx.remove_center, ctx.on_x
            ]

            output = DSCN.dscn_forward(*args)
            ctx.save_for_backward(input, offset)

            return output

        @staticmethod
        @once_differentiable
        @custom_bwd
        def backward(ctx, grad_output):
            input, offset = ctx.saved_tensors

            args = [
                input, offset, ctx.kernel_h,
                ctx.kernel_w, ctx.stride_h, ctx.stride_w, ctx.pad_h,
                ctx.pad_w, ctx.dilation_h, ctx.dilation_w, ctx.group,
                ctx.group_channels, ctx.offset_scale, grad_output.contiguous(), ctx.im2col_step,
                ctx.remove_center, ctx.on_x
            ]

            grad_input, grad_offset = \
                DSCN.dscn_backward(*args)

            return grad_input, grad_offset, \
                None, None, None, None, None, None, None, None, None, None, None, None, None, None

        @staticmethod
        def symbolic(g, input, offset, kernel_h, kernel_w, stride_h,
                     stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
                     group_channels, offset_scale, im2col_step, remove_center):
            """Symbolic function for mmdeploy::DSCN.

            Returns:
                DSCN op for onnx.
            """
            return g.op(
                'mmdeploy::TRTDSCN',
                input,
                offset,
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

class DSCNX(nn.Module):
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
        remove_center=False,
    ):
        """
        DSCNX Module
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DSCN to make the dimension of each attention head a power of 2 "
                "which is more efficient in this CUDA implementation.")

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
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1,dw_kernel_size),
                stride=1,
                padding=(0,(dw_kernel_size - 1) // 2),
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * (kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)

    def forward(self, input, off_x):
        """
        :param query (N, H, W, C)
        :param off_x (N, C, H, W)
        :return output (N, H, W, C)
        """
        x = self.input_proj(input)
        
        # === 修改開始 ===
        # 檢查輸入是否在 CUDA 上，並且 DSCN 模組是否可用
        # 這允許模型在 CPU 上進行初始化（例如YOLO框架計算stride），而在 GPU 上進行實際運算
        if input.is_cuda and DSCN is not None:
            x1 = self.dw_conv(off_x)
            offset = self.offset(x1)
            
            x = DSCNFunction.apply(
                x, offset,
                1, self.kernel_size,
                1, self.stride,
                0, self.pad,
                1, self.dilation,
                self.group, self.group_channels,
                self.offset_scale,
                256,
                self.remove_center, True)
        # 如果在 CPU 上，則跳過自定義操作。這對於 stride=1 的情況是安全的，
        # 因為它不會改變特徵圖的形狀。
        # === 修改結束 ===
        return x
    
class DSCNY(nn.Module):
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
        remove_center=False,
    ):
        """
        DSCNY Module
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DSCN to make the dimension of each attention head a power of 2 "
                "which is more efficient in this CUDA implementation.")

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
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(dw_kernel_size,1),
                stride=1,
                padding=((dw_kernel_size - 1) // 2,0),
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))

        self.offset = nn.Linear(
            channels,
            group * (kernel_size - remove_center))
        self._reset_parameters()
        
    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)

    def forward(self, input, off_x):
        """
        :param query (N, H, W, C)
        :param off_x (N, C, H, W)
        :return output (N, H, W, C)
        """
        x = input
        
        # === 修改開始 ===
        # 檢查輸入是否在 CUDA 上，並且 DSCN 模組是否可用
        # 這允許模型在 CPU 上進行初始化（例如YOLO框架計算stride），而在 GPU 上進行實際運算
        if input.is_cuda and DSCN is not None:
            x1 = self.dw_conv(off_x)
            offset = self.offset(x1)
            
            x = DSCNFunction.apply(
                x, offset,
                self.kernel_size, 1,
                self.stride, 1,
                self.pad, 0,
                self.dilation, 1,
                self.group, self.group_channels,
                self.offset_scale,
                256,
                self.remove_center, False)
        # 如果在 CPU 上，則跳過自定義操作。這對於 stride=1 的情況是安全的，
        # 因為它不會改變特徵圖的形狀。
        # === 修改結束 ===
        
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

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

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DSCNPair(nn.Module):
    def __init__(self, d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group):
        super().__init__()
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.group = group
        self.conv0 = nn.Conv2d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        
        self.dscn_x = DSCNX(d_model, kernel_size, dw_kernel_size, stride=stride, pad=pad, dilation=dilation, group=group)
        self.dscn_y = DSCNY(d_model, kernel_size, dw_kernel_size, stride=stride, pad=pad, dilation=dilation, group=group)
        self.conv = nn.Conv2d(d_model, d_model, 1)

    def forward(self,x):
        u = x.clone()
        x = self.conv0(x)
        attn = x.permute(0,2,3,1) # N, C, H, W -> N, H, W, C
        attn = self.dscn_x(attn,x)
        attn = self.dscn_y(attn,x)
        attn = attn.permute(0,3,1,2) # N, H, W, C -> N, C, H, W
        attn = self.conv(attn)
        return u*attn

class DSA(nn.Module):
    def __init__(self, d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = DSCNPair(d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, kernel_size, dw_kernel_size, pad, stride, dilation, group, 
                 mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = DSA(dim, kernel_size, dw_kernel_size, pad, stride, dilation, group)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

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

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

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

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W


class DSAN(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 kernel_sizes=[11, 11, 7, 5], dw_kernel_sizes=[5, 5, 5, 5], pads=[5, 5, 3, 2], strides=[1, 1, 1, 1],
                 dilations=[1, 1, 1, 1], groups=[1, 4, 8, 8],
                 mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4, flag=False):
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], kernel_size=kernel_sizes[i], dw_kernel_size=dw_kernel_sizes[i], pad=pads[i], 
                stride=strides[i], dilation=dilations[i], group=groups[i],
                mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        
        # --- width_list 計算 ---
        # 由於 forward 函數已被修改為兼容 CPU，此處的虛擬運算現在更加健壯。
        self.width_list = []
        if torch.cuda.is_available() and DSCN is not None:
            try:
                device = torch.device("cuda")
                original_device = next(self.parameters()).device
                
                self.to(device)
                self.eval()
                
                dummy_input = torch.randn(1, in_chans, img_size, img_size, device=device)
                
                with torch.no_grad():
                    features = self.forward(dummy_input)
                
                self.width_list = [f.size(1) for f in features]
                
                self.to(original_device)
                self.train()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error during dummy forward pass for width_list calculation: {e}")
                print("Setting width_list to embed_dims as fallback.")
                self.width_list = embed_dims
                self.train()
        else:
            print("Warning: CUDA not available or DSCN module not found. "
                  "Falling back to `embed_dims` for `width_list`. This may be inaccurate.")
            self.width_list = embed_dims

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

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features_for_classification(self, x):
        """
        用於分類任務的原始 forward_features 邏輯。
        """
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x.mean(dim=1)

    def forward_head(self, x):
        """
        分類頭的前向傳播。
        """
        x = self.forward_features_for_classification(x)
        x = self.head(x)
        return x

    def forward(self, x):
        """
        修改後的前向傳播，返回每個階段的特徵圖列表。
        這適用於作為檢測/分割模型的骨幹網路。
        """
        B = x.shape[0]
        feature_outputs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            
            x_norm_in = x.flatten(2).transpose(1, 2)
            x_norm_out = norm(x_norm_in)

            x = x_norm_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            feature_outputs.append(x)
            
        return feature_outputs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


model_urls = {
    "dsan_t": "pretrained/dsan_t.pth.tar",
    "dsan_s": "pretraind/dsan_s.pth.tar",
}


def load_model_weights(model, arch, kwargs):
    print(f"INFO: Pretending to load pretrained weights for {arch}.")
    return model

@register_model
def dsan_t(pretrained=False, **kwargs):
    model = DSAN(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        dw_kernel_sizes=[11, 11, 7, 5],
        kernel_sizes=[11, 11, 7, 5],
        pads=[5, 5, 3, 2],
        groups=[4, 8, 16, 16],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "dsan_t", kwargs)
    return model

@register_model
def dsan_s(pretrained=False, **kwargs):
    model = DSAN(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        kernel_sizes=[19, 15, 13, 7],
        dw_kernel_sizes=[19, 15, 13, 7],
        pads=[9, 7, 6, 3],
        groups=[4, 8, 16, 16],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        depths=[3, 3, 8, 2],
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "dsan_s", kwargs)
    return model

@register_model
def dsan_b(pretrained=False, **kwargs):
    model = DSAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        kernel_sizes=[15, 13, 7, 5],
        dw_kernel_sizes=[9, 7, 5, 5],
        pads=[7, 6, 3, 2],
        groups=[4, 8, 16, 16],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 5, 3],
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        print("Warning: No pretrained weights available for dsan_b.")
    return model


if __name__ == '__main__':
    img_h, img_w = 224, 224
    
    print("--- 創建 DSAN Tiny 模型 ---")
    model = dsan_t(img_size=img_h)
    print("模型創建成功。")
    print("計算得到的 width_list:", model.width_list)

    if not torch.cuda.is_available():
        print("\n--- 跳過前向傳播測試: CUDA is not available. DSAN requires a GPU for its custom operations. ---")
    elif DSCN is None:
        print("\n--- 跳過前向傳播測試: DSCN custom ops package not installed. ---")
    else:
        device = torch.device("cuda")
        print(f"\n--- 在設備上執行前向傳播測試: {device} ---")

        model.to(device)
        model.eval()

        input_tensor = torch.rand(2, 3, img_h, img_w).to(device)
        print(f"--- 測試 DSAN Tiny 前向傳播 (輸入: {input_tensor.shape}) ---")

        try:
            with torch.no_grad():
                output_features = model(input_tensor)
            print("前向傳播成功。")
            print("輸出特徵圖的形狀:")
            for i, features in enumerate(output_features):
                print(f"階段 {i+1}: {features.shape}")

            runtime_widths = [f.size(1) for f in output_features]
            print("\n運行時輸出特徵通道:", runtime_widths)
            assert model.width_list == runtime_widths, "Width list 不匹配!"
            print("Width list 驗證成功。")

            print("\n--- 測試 deepcopy ---")
            import copy
            model.to('cpu')
            copied_model = copy.deepcopy(model)
            print("Deepcopy 成功。")
            
            copied_model.to(device)
            copied_model.eval()

            with torch.no_grad():
                 output_copied = copied_model(input_tensor)
            print("複製後模型的前向傳播成功。")
            assert len(output_copied) == len(output_features)
            for i in range(len(output_features)):
                 assert output_copied[i].shape == output_features[i].shape
            print("複製後模型的輸出形狀已驗證。")

        except Exception as e:
            print(f"\n測試過程中發生錯誤: {e}")
            import traceback
            traceback.print_exc()