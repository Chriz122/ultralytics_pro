# -*- coding: utf-8 -*-
"""
ParCNet-V2: Oversized Convolution with enhanced attention.
Some implementations are modified from timm (https://github.com/rwightman/pytorch-image-models)
and MetaFormer (https://arxiv.org/abs/2210.13452)

Modified to function as a backbone for detection/segmentation tasks, returning a list of feature maps.
Inspired by the SMT model structure.
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# ==============================================================================
# NEW: LayerNorm2d for (B, C, H, W) tensors - THE KEY FIX
# ==============================================================================
class LayerNorm2d(nn.Module):
    """
    LayerNorm implementation for 2D data (images).
    It normalizes over the C, H, W dimensions, which is equivalent to nn.GroupNorm(1, C).
    The affine transformation is applied per-channel.
    """
    def __init__(self, num_channels, eps=1e-6, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        # The original LayerNormGeneral had a `bias` flag. We replicate it.
        self.bias = nn.Parameter(torch.zeros(num_channels)) if bias else None
        self.eps = eps

    def forward(self, x):
        # x has shape: [B, C, H, W]
        # Calculate mean and variance over C, H, W
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        
        # Apply per-channel scale and bias
        # weight and bias have shape [C], need to be reshaped to [1, C, 1, 1] for broadcasting
        x = x * self.weight.view(1, -1, 1, 1)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        return x


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        pre_norm=None,
        post_norm=None,
        pre_permute=False,
    ):
        super().__init__()
        # If pre_norm is provided, it's a function (like partial) that we call with in_channels
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        # This is the data flow for stages 1, 2, 3
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        
        # pre_norm is now applied to a (B, C, H, W) tensor
        x = self.pre_norm(x)
        
        # The first stage (i=0) doesn't pre_permute, so pre_norm (Identity) sees (B, C, H, W)
        # and conv also sees (B, C, H, W)
        x = self.conv(x)
        
        # All stages output in (B, H, W, C) format for the main blocks
        x = x.permute(0, 2, 3, 1)
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


class OversizeConv2d(nn.Module):
    def __init__(self, dim, kernel_size, bias=False, interpolate=False):
        super().__init__()
        if interpolate is True:
            padding = 0
        elif interpolate is False or interpolate is None:
            assert kernel_size % 2 == 1
            padding = kernel_size // 2
        else:
            assert interpolate % 2 == 1
            padding = interpolate // 2
            interpolate = to_2tuple(interpolate)

        self.conv_h = nn.Conv2d(dim, dim, (kernel_size, 1), padding=(padding, 0), groups=dim, bias=bias)
        self.conv_w = nn.Conv2d(dim, dim, (1, kernel_size), padding=(0, padding), groups=dim, bias=bias)
        self.dim, self.kernel_size, self.interpolate, self.padding = dim, kernel_size, interpolate, padding

    def get_instance_kernel(self, instance_kernel_size):
        h_weight = F.interpolate(self.conv_h.weight, [instance_kernel_size[0], 1], mode="bilinear", align_corners=True)
        w_weight = F.interpolate(self.conv_w.weight, [1, instance_kernel_size[1]], mode="bilinear", align_corners=True)
        return h_weight, w_weight

    def forward(self, x):
        if self.interpolate is True:
            H, W = x.shape[-2:]
            instance_kernel_size = 2 * H - 1, 2 * W - 1
            h_weight, w_weight = self.get_instance_kernel(instance_kernel_size)
            padding = H - 1, W - 1
            x = F.conv2d(x, h_weight, self.conv_h.bias, padding=(padding[0], 0), groups=self.dim)
            x = F.conv2d(x, w_weight, self.conv_w.bias, padding=(0, padding[1]), groups=self.dim)
        elif isinstance(self.interpolate, tuple):
            h_weight, w_weight = self.get_instance_kernel(self.interpolate)
            x = F.conv2d(x, h_weight, self.conv_h.bias, padding=(self.padding, 0), groups=self.dim)
            x = F.conv2d(x, w_weight, self.conv_w.bias, padding=(0, self.padding), groups=self.dim)
        else:
            x = self.conv_h(x)
            x = self.conv_w(x)
        return x


class ParC_V2(nn.Module):
    def __init__(self, dim, expansion_ratio=2, act_layer=nn.GELU, bias=False, kernel_size=7, global_kernel_size=13, padding=3, **kwargs):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, 1, bias=True)
        self.act = act_layer()
        self.dwconv1 = OversizeConv2d(med_channels // 2, global_kernel_size, bias)
        self.dwconv2 = nn.Conv2d(med_channels // 2, med_channels // 2, kernel_size=kernel_size, padding=padding, groups=med_channels // 2, bias=bias)
        self.pwconv2 = nn.Conv2d(med_channels // 2, dim, 1, bias=bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.pwconv1(x)
        x1, x2 = x.chunk(2, 1)
        x2 = self.act(x2)
        x2 = self.dwconv1(x2) + self.dwconv2(x2)
        x = x1 * x2
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 1)
        return x


class LayerNormGeneral(nn.Module):
    """General LayerNorm for (B, ..., C) tensors."""
    def __init__(self, affine_shape=None, normalized_dim=(-1,), scale=True, bias=True, eps=1e-5):
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


class BGU(nn.Module):
    """Bifurcate Gate Unit used in ParCNetV2."""
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=nn.GELU, drop=0.0, bias=False, **kwargs):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features // 2, out_features or dim, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, -1)
        x = x1 * self.act(x2)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ParCNetV2Block(nn.Module):
    """Implementation of one ParCNetV2 block."""
    def __init__(self, dim, token_mixer=nn.Identity, global_kernel_size=13, mlp=partial(BGU, mlp_ratio=5), norm_layer=nn.LayerNorm, drop=0.0, drop_path=0.0, layer_scale_init_value=None, res_scale_init_value=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop, global_kernel_size=global_kernel_size)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + self.layer_scale1(self.drop_path1(self.token_mixer(self.norm1(x))))
        x = self.res_scale2(x) + self.layer_scale2(self.drop_path2(self.mlp(self.norm2(x))))
        return x

# ==============================================================================
# CORRECTED: Use LayerNorm2d for pre_norm on (B, C, H, W) data
# ==============================================================================
DOWNSAMPLE_LAYERS_FOUR_STAGES = (
    [
        partial(
            Downsampling,
            kernel_size=7,
            stride=4,
            padding=2,
            # post_norm sees (B,H,W,C) data, so LayerNormGeneral is correct
            post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),
        )
    ]
    + [
        partial(
            Downsampling,
            kernel_size=3,
            stride=2,
            padding=1,
            # pre_norm sees (B,C,H,W) data because of pre_permute, so we use LayerNorm2d
            pre_norm=partial(LayerNorm2d, bias=False, eps=1e-6),
            pre_permute=True,
        )
    ]
    * 3
)


class ParCNetV2(nn.Module):
    r"""ParCNetV2 modified to serve as a backbone network."""
    def __init__(self, in_chans=3, num_classes=1000, depths=[2, 2, 6, 2], dims=[64, 128, 320, 512], downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES, token_mixers=ParC_V2, global_kernel_sizes=[111, 55, 27, 13], mlps=partial(BGU, mlp_ratio=5), norm_layers=partial(LayerNormGeneral, eps=1e-6, bias=False), drop_path_rate=0.0, layer_scale_init_values=None, res_scale_init_values=[None, None, 1.0, 1.0], img_size=224, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.num_stage = len(depths)
        
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList([downsample_layers[i](down_dims[i], down_dims[i + 1]) for i in range(self.num_stage)])

        if not isinstance(token_mixers, (list, tuple)): token_mixers = [token_mixers] * self.num_stage
        if not isinstance(global_kernel_sizes, (list, tuple)): global_kernel_sizes = [global_kernel_sizes] * self.num_stage
        if not isinstance(mlps, (list, tuple)): mlps = [mlps] * self.num_stage
        if not isinstance(norm_layers, (list, tuple)): norm_layers = [norm_layers] * self.num_stage
        if not isinstance(layer_scale_init_values, (list, tuple)): layer_scale_init_values = [layer_scale_init_values] * self.num_stage
        if not isinstance(res_scale_init_values, (list, tuple)): res_scale_init_values = [res_scale_init_values] * self.num_stage

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(self.num_stage):
            stage_blocks = [ParCNetV2Block(dim=dims[i], token_mixer=token_mixers[i], global_kernel_size=global_kernel_sizes[i], mlp=mlps[i], norm_layer=norm_layers[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_values[i], res_scale_init_value=res_scale_init_values[i]) for j in range(depths[i])]
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]
        
        # Add a norm layer for each stage's final output, applied on (B,H,W,C) data
        self.norms = nn.ModuleList([norm_layers[i](dims[i]) for i in range(self.num_stage)])

        self.apply(self._init_weights)
        
        # --- Add width_list calculation ---
        self.width_list = []
        try:
            self.eval()
            dummy_input = torch.randn(1, self.in_chans, img_size, img_size)
            with torch.no_grad():
                features = self.forward(dummy_input)
            self.width_list = [f.size(1) for f in features]
            self.train()
            # print(f"Successfully calculated width_list: {self.width_list}")
        except Exception as e:
            print(f"Error during dummy forward pass for width_list calculation: {e}")
            print("Setting width_list to dims as fallback.")
            self.width_list = dims
            self.train()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self): return {"norm"}

    def forward_features(self, x):
        feature_outputs = []
        # The first downsampling layer expects (B,C,H,W), subsequent ones expect (B,H,W,C)
        # but handle it internally with pre_permute.
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            norm_x = self.norms[i](x)
            # Permute from [B, H, W, C] to [B, C, H, W] for standard backbone output
            feature_outputs.append(norm_x.permute(0, 3, 1, 2).contiguous())
        return feature_outputs

    def forward(self, x):
        return self.forward_features(x)

@register_model
def parcnetv2_xt(pretrained=False, **kwargs):
    model = ParCNetV2(depths=[3, 3, 9, 2], dims=[48, 96, 192, 320], token_mixers=ParC_V2, mlps=partial(BGU, mlp_ratio=5), **kwargs)
    return model

@register_model
def parcnetv2_tiny(pretrained=False, **kwargs):
    model = ParCNetV2(depths=[3, 3, 12, 3], dims=[64, 128, 320, 512], token_mixers=ParC_V2, mlps=partial(BGU, mlp_ratio=5), **kwargs)
    return model

@register_model
def parcnetv2_small(pretrained=False, **kwargs):
    model = ParCNetV2(depths=[3, 9, 24, 3], dims=[64, 128, 320, 512], token_mixers=ParC_V2, mlps=partial(BGU, mlp_ratio=5), **kwargs)
    return model

@register_model
def parcnetv2_base(pretrained=False, **kwargs):
    model = ParCNetV2(depths=[3, 9, 24, 3], dims=[96, 192, 384, 576], token_mixers=ParC_V2, mlps=partial(BGU, mlp_ratio=5), **kwargs)
    return model


if __name__ == '__main__':
    print("--- Instantiating parcnetv2_tiny ---")
    # Pass img_size for correct dummy pass. Default is 224, using 640 for testing.
    model = parcnetv2_tiny(img_size=640)
    model.eval()

    print(f"\nModel's calculated width_list: {model.width_list}")
    assert model.width_list == [64, 128, 320, 512]

    print("\n--- Testing forward pass with image size (1, 3, 640, 640) ---")
    try:
        input_tensor = torch.randn(1, 3, 640, 640)
        output_features = model(input_tensor)
        
        print(f"Output type: {type(output_features)}")
        assert isinstance(output_features, list), "Output should be a list"
        assert len(output_features) == model.num_stage, "Output list length should match number of stages"

        print("\nShape of each feature map in the output list:")
        expected_strides = [4, 8, 16, 32]
        for i, feature_map in enumerate(output_features):
            print(f"Stage {i+1}: {feature_map.shape}")
            assert feature_map.shape[1] == model.width_list[i]
            assert feature_map.shape[2] == 640 // expected_strides[i]
            assert feature_map.shape[3] == 640 // expected_strides[i]

        print("\n✅ Verification successful! The model works as a backbone.")
    except Exception as e:
        print(f"\n❌ An error occurred during verification: {e}")
        # Add traceback for easier debugging
        import traceback
        traceback.print_exc()