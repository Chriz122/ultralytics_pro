# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import pywt.data
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:,:,0,:,:]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x
    

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)
    

class WTConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, wt_levels: int = 0):
        super().__init__()
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.dwconv = WTConv2d(
            dim,
            dim,
            kernel_size=5,
            bias=True,
            wt_levels=wt_levels,
        )
        # 使用自定義的 LayerNorm
        self.norm = LayerNorm(dim, eps=1e-6) 
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class WTConvNeXt(nn.Module):
    r""" WTConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
                           Set to 0 to disable classification head.
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        input_image_size (int): Dummy input image size for width_list calculation. Default: 224.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 input_image_size=224, wt_levels: Tuple[int, ...] = (5, 4, 3, 2) # Dummy input size for width_list
                 ):
        super().__init__()

        self.num_classes = num_classes

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[WTConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, wt_levels=wt_levels[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Optional classification head layers
        if num_classes > 0:
            self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer for classification
            self.head = nn.Linear(dims[-1], num_classes)
        else:
            self.norm = None
            self.head = None


        self.apply(self._init_weights)
        if num_classes > 0:
             self.head.weight.data.mul_(head_init_scale)
             if self.head.bias is not None:
                  self.head.bias.data.mul_(head_init_scale)

        # --- 模仿 Code 2 添加 width_list 的計算 ---
        # 創建一個 dummy input
        dummy_input = torch.randn(1, in_chans, input_image_size, input_image_size)
        # 執行一個 dummy forward pass to get shapes
        # 需要暫時切換到評估模式以確保 BatchNorm 等行為正確，並關閉梯度計算
        self.eval()
        with torch.no_grad():
             # The forward method now returns a list of feature maps
             dummy_outputs = self.forward(dummy_input)
        self.train() # 切換回訓練模式
        # 獲取每個階段輸出的 channel 維度
        # forward returns [x1, x2, x3, x4] (stage outputs)
        self.width_list = [f.size(1) for f in dummy_outputs]
        # ---------------------------------------------


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)

    # Renamed original forward_features to a helper function if still needed
    # Note: This method is NOT used by the main forward anymore unless explicitly called
    def forward_classification_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # Original classification path: global average pooling
        if self.norm is not None:
             x = self.norm(x.mean([-2, -1])) # (N, C, H, W) -> (N, C)
        return x

    def forward(self, x):
        """
        Modified forward method to return a list of intermediate stage outputs,
        suitable for use as a backbone in tasks like object detection.
        Optionally performs classification if num_classes > 0 and needed.
        """
        # MobileNetV4-like forward: return list of stage outputs
        outputs = []
        # Stem layer
        x = self.downsample_layers[0](x) 
        x = self.stages[0](x)
        outputs.append(x) # Output from stage 0 (after stem)

        # Intermediate stages
        for i in range(1, 4): # Iterate from stage 1 to 3
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outputs.append(x) # Output from stage i


        # If num_classes > 0, we might optionally also return classification logits
        # However, for a backbone, typically only the feature list is needed.
        # The calling code (like YOLO) will add its own head.
        # To strictly mimic MobileNetV4's forward output structure for backbones,
        # we only return the list of feature maps [x1, x2, x3, x4].
        # If classification is also needed, the calling code could call 
        # forward_classification_features or a separate classification head.
        # For compatibility with YOLO backbone usage based on the error, 
        # we just return the list of feature maps.

        # Return the list of feature maps from each stage
        # This list structure [stem+stage0_out, stage1_out, stage2_out, stage3_out]
        # matches the typical FPN/PAN structure requirements.
        # In ConvNeXt structure, stage 0 is after the first downsampling (stride 4),
        # stage 1 is after the second (stride 8), etc.
        # So outputs[0] is stride 4, outputs[1] is stride 8, outputs[2] is stride 16, outputs[3] is stride 32.

        return outputs


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # Handle case where normalized_shape is an integer
        if isinstance(normalized_shape, int):
             normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = normalized_shape 
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # For channels_first, normalization is done over C dimension (dim 1)
            # Weight and bias need to be broadcast across H and W
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # Reshape weight and bias to match channels_first format for broadcasting
            weight = self.weight.view(-1, 1, 1)
            bias = self.bias.view(-1, 1, 1)
            x = weight * x + bias
            return x


# Model URLs remain the same
model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

# --- ConvNeXt 不同大小模型創建方式，與 MobileNetV4 類似 ---
# 使用 timm 的 register_model 裝飾器，或移除直接呼叫

@register_model
def wtconvnext_tiny(pretrained=False, in_22k=False, **kwargs):
    """ ConvNeXt Tiny model, configured to act as a backbone. """
    # Pass num_classes=0 when used as a backbone, or let caller override
    model = WTConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        # Remove head weights if num_classes doesn't match, then load state_dict
        if 'model' in checkpoint: # handle case where checkpoint is just the model state_dict
             checkpoint = checkpoint['model']
        
        # Adjust checkpoint keys to match potential changes if any
        # For example, if classification layers are conditionally added/removed
        # Here, classification layers are kept but num_classes=0 means no head linear layer
        # If num_classes=0 was passed, model doesn't have head.weight/bias
        # If num_classes > 0 was passed, it has head.weight/bias
        # We need to check if the model instance *has* the head layer before trying to load
        if model.head is None and 'head.weight' in checkpoint:
             del checkpoint["head.weight"]
             if 'head.bias' in checkpoint:
                  del checkpoint["head.bias"]
        # If model *has* head, but num_classes is different from checkpoint, remove checkpoint head
        elif model.head is not None and model.head.out_features != checkpoint['head.weight'].size(0):
             del checkpoint["head.weight"]
             if 'head.bias' in checkpoint:
                  del checkpoint["head.bias"]

        model.load_state_dict(checkpoint, strict=False) # Use strict=False to allow partial loading
    return model

@register_model
def wtconvnext_small(pretrained=False, in_22k=False, **kwargs):
    """ ConvNeXt Small model, configured to act as a backbone. """
    model = WTConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        if 'model' in checkpoint: checkpoint = checkpoint['model']
        if model.head is None and 'head.weight' in checkpoint:
             del checkpoint["head.weight"]
             if 'head.bias' in checkpoint: del checkpoint["head.bias"]
        elif model.head is not None and model.head.out_features != checkpoint['head.weight'].size(0):
             del checkpoint["head.weight"]
             if 'head.bias' in checkpoint: del checkpoint["head.bias"]
        model.load_state_dict(checkpoint, strict=False)
    return model

@register_model
def wtconvnext_base(pretrained=False, in_22k=False, **kwargs):
    """ ConvNeXt Base model, configured to act as a backbone. """
    model = WTConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        if 'model' in checkpoint: checkpoint = checkpoint['model']
        if model.head is None and 'head.weight' in checkpoint:
             del checkpoint["head.weight"]
             if 'head.bias' in checkpoint: del checkpoint["head.bias"]
        elif model.head is not None and model.head.out_features != checkpoint['head.weight'].size(0):
             del checkpoint["head.weight"]
             if 'head.bias' in checkpoint: del checkpoint["head.bias"]
        model.load_state_dict(checkpoint, strict=False)
    return model

@register_model
def wtconvnext_large(pretrained=False, in_22k=False, **kwargs):
    """ ConvNeXt Large model, configured to act as a backbone. """
    model = WTConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        if 'model' in checkpoint: checkpoint = checkpoint['model']
        if model.head is None and 'head.weight' in checkpoint:
             del checkpoint["head.weight"]
             if 'head.bias' in checkpoint: del checkpoint["head.bias"]
        elif model.head is not None and model.head.out_features != checkpoint['head.weight'].size(0):
             del checkpoint["head.weight"]
             if 'head.bias' in checkpoint: del checkpoint["head.bias"]
        model.load_state_dict(checkpoint, strict=False)
    return model

@register_model
def wtconvnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    """ ConvNeXt Extra Large model, configured to act as a backbone. """
    model = WTConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        if 'model' in checkpoint: checkpoint = checkpoint['model']
        # 22k pre-trained models typically have 21841 classes
        if model.head is None and 'head.weight' in checkpoint:
             del checkpoint["head.weight"]
             if 'head.bias' in checkpoint: del checkpoint["head.bias"]
        elif model.head is not None and model.head.out_features != checkpoint['head.weight'].size(0):
             del checkpoint["head.weight"]
             if 'head.bias' in checkpoint: del checkpoint["head.bias"]
        model.load_state_dict(checkpoint, strict=False)
    return model

# --- 使用範例 ---
if __name__ == "__main__":
    # 創建一個 ConvNeXt Tiny 模型，作為骨幹網路使用 (類似於 MobileNetV4ConvSmall())
    # 傳入 num_classes=0 表示不要分類頭
    model = wtconvnext_tiny(pretrained=False, num_classes=0) 
    
    print("Model Architecture:")
    print(model)

    print("\nIntermediate Feature Widths (Channels):")
    print(model.width_list) # 打印 width_list

    # 測試 forward pass
    # YOLO 可能會用不同的輸入尺寸來計算 stride，比如 640x640
    # 這裡用 640x640 測試輸出形狀
    input_image_size = 640
    dummy_input = torch.randn(1, 3, input_image_size, input_image_size)
    print(f"\nInput Shape: {dummy_input.shape} (assuming {input_image_size}x{input_image_size} input)")

    # 當作為骨幹網路時，forward 方法返回特徵圖列表
    output_feature_maps = model(dummy_input)
    
    print("\nOutput Feature Map Shapes from forward:")
    # 預期的輸出階段縮小倍數是 4, 8, 16, 32
    expected_strides = [4, 8, 16, 32]
    for i, feat_map in enumerate(output_feature_maps):
        print(f"Stage {i} Output Shape:", feat_map.shape)
        # 驗證空間維度是否符合預期縮小倍數
        expected_h = input_image_size // expected_strides[i]
        expected_w = input_image_size // expected_strides[i]
        assert feat_map.shape[-2:] == (expected_h, expected_w), f"Stage {i} spatial shape mismatch! Expected ({expected_h}, {expected_w}), got {feat_map.shape[-2:]}"
        # 驗證 channel 維度是否符合 width_list
        assert feat_map.shape[1] == model.width_list[i], f"Stage {i} channel mismatch! Expected {model.width_list[i]}, got {feat_map.shape[1]}"


    # 測試其他模型大小的創建 (作為骨幹網路)
    model_base = wtconvnext_base(num_classes=0)
    print("\nConvNeXt Base Model created (as backbone).")
    print("Intermediate Feature Widths (Base):", model_base.width_list)
    
    # 如果你想測試分類功能，可以創建一個帶分類頭的模型，並呼叫 forward_classification_features
    model_tiny_classify = wtconvnext_tiny(pretrained=False, num_classes=1000)
    print("\nConvNeXt Tiny Model created (with classification head).")
    # Call classification specific forward method
    output_logits = model_tiny_classify.forward_classification_features(torch.randn(1, 3, 224, 224))
    print("Output Logits Shape (classification):", output_logits.shape)