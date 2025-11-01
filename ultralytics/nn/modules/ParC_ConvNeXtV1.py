import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

# --- LayerNorm (無變動) ---
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
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
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            weight = self.weight.view(-1, 1, 1)
            bias = self.bias.view(-1, 1, 1)
            x = weight * x + bias
            return x

# ===================【關鍵修正】===================
# ParC_operator實現延遲初始化，以適應可變輸入尺寸
class ParC_operator(nn.Module):
    def __init__(self, dim, type, use_pe=True):
        super().__init__()
        self.type = type
        self.dim = dim
        self.use_pe = use_pe
        # 不在初始化時創建層，將它們設置為 None
        self.gcc_conv = None
        self.pe = None

    def forward(self, x):
        # 延遲初始化：在第一次 forward 調用時，根據輸入 x 的大小創建層
        if self.gcc_conv is None:
            # 從輸入 x 動態獲取 H 和 W
            H, W = x.shape[-2:]
            
            # 根據類型（'H'或'W'）確定 global_kernel_size
            global_kernel_size = H if self.type == 'H' else W
            kernel_size = (global_kernel_size, 1) if self.type == 'H' else (1, global_kernel_size)

            # 創建卷積層
            self.gcc_conv = nn.Conv2d(self.dim, self.dim, kernel_size=kernel_size, groups=self.dim)
            
            # 創建位置編碼參數
            if self.use_pe:
                if self.type == 'H':
                    self.pe = nn.Parameter(torch.randn(1, self.dim, global_kernel_size, 1))
                else: # self.type == 'W'
                    self.pe = nn.Parameter(torch.randn(1, self.dim, 1, global_kernel_size))
                trunc_normal_(self.pe, std=.02)
            
            # 將新創建的層和參數移動到與輸入張量相同的設備
            self.to(x.device)

        # --- 以下是原始的 forward 邏輯 ---
        if self.use_pe:
            x = x + self.pe
        
        # 根據卷積核大小確定填充量
        padding_size_h = self.gcc_conv.kernel_size[0] - 1
        padding_size_w = self.gcc_conv.kernel_size[1] - 1
        
        if self.type == 'H':
            x_cat = torch.cat((x, x[:, :, :padding_size_h, :]), dim=2)
        else: # self.type == 'W'
            x_cat = torch.cat((x, x[:, :, :, :padding_size_w]), dim=3)
            
        x = self.gcc_conv(x_cat)
        return x

# ParC_ConvNext_Block 現在不再需要傳遞 global_kernel_size
class ParC_ConvNext_Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, use_pe=True):
        super().__init__()
        # 不再傳遞 global_kernel_size
        self.gcc_H = ParC_operator(dim//2, 'H', use_pe)
        self.gcc_W = ParC_operator(dim//2, 'W', use_pe)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x_H, x_W = torch.chunk(x, 2, dim=1)
        x_H, x_W = self.gcc_H(x_H), self.gcc_W(x_W)
        x = torch.cat((x_H, x_W), dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

# --- ConvNext_Block (無變動) ---
class ConvNext_Block(nn.Module):
    # ... (代碼與之前相同，此處省略以保持簡潔)
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

# 主模型 ParC_ConvNeXt 現在更加簡潔
class ParC_ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ParC_insert_locs=[3, 3, 6, 2],
                 # 移除 input_image_size，因為模型現在是尺寸無關的
                 ):
        super().__init__()
        self.num_classes = num_classes

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i, (depth, dim, ParC_insert_loc) in enumerate(zip(depths, dims, ParC_insert_locs)):
            blocks = []
            for j in range(depth):
                if j < ParC_insert_loc:
                    blocks.append(ConvNext_Block(dim=dim, drop_path=dp_rates[cur + j],
                                                 layer_scale_init_value=layer_scale_init_value))
                else:
                    # 不再需要傳遞 global_kernel_size
                    blocks.append(ParC_ConvNext_Block(dim=dim, drop_path=dp_rates[cur + j],
                                                     layer_scale_init_value=layer_scale_init_value,
                                                     use_pe=True))
            stage = nn.Sequential(*blocks)
            self.stages.append(stage)
            cur += depths[i]
        
        if self.num_classes > 0:
            self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
            self.head = nn.Linear(dims[-1], num_classes)
        else:
            self.norm = None
            self.head = None

        self.apply(self._init_weights)
        if self.head is not None:
            self.head.weight.data.mul_(head_init_scale)
            if self.head.bias is not None:
                self.head.bias.data.mul_(head_init_scale)

        # 為了兼容YOLO框架，在__init__中計算 width_list 是必要的
        # 我們通過一次乾跑來實現，即使ParC_operator是延遲初始化的
        # 這次乾跑會觸發ParC_operator的初始化
        self.eval()
        with torch.no_grad():
             # 使用一個標準尺寸（如224）進行乾跑來初始化所有層並獲取通道數
             dummy_input = torch.randn(1, in_chans, 224, 224)
             dummy_outputs = self.forward(dummy_input)
        self.train() 
        self.width_list = [f.size(1) for f in dummy_outputs]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_classification_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        if self.norm is not None:
             x = self.norm(x.mean([-2, -1]))
        return x

    def forward(self, x):
        outputs = []
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        outputs.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outputs.append(x)
        return outputs

# --- 工廠函數 (現在更簡潔) ---

@register_model
def parc_convnext_tiny(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError("No pretrained weights available.")
    model = ParC_ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                          ParC_insert_locs=[3, 3, 6, 2], **kwargs)
    return model

@register_model
def parc_convnext_small(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError("No pretrained weights available.")
    model = ParC_ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], 
                          ParC_insert_locs=[3, 3, 18, 2], **kwargs)
    return model

@register_model
def parc_convnext_base(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError("No pretrained weights available.")
    model = ParC_ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], 
                          ParC_insert_locs=[3, 3, 18, 2], **kwargs)
    return model

# --- 使用範例與測試 ---
if __name__ == "__main__":
    print("--- Testing ParC-ConvNeXt-Tiny as a Backbone ---")
    
    # 創建模型時不再需要指定 input_image_size
    model = parc_convnext_tiny(num_classes=0)
    
    print("\nModel Architecture created successfully.")
    print("Intermediate Feature Widths (Channels) from self.width_list:", model.width_list)

    # 測試不同輸入尺寸
    for input_image_size in [224, 512, 640]:
        print(f"\n--- Testing with input size: {input_image_size}x{input_image_size} ---")
        dummy_input = torch.randn(2, 3, input_image_size, input_image_size)
        print(f"Input Shape: {dummy_input.shape}")

        # 模型現在應該能處理任意尺寸的輸入
        output_feature_maps = model(dummy_input)
        
        print(f"Output is a: {type(output_feature_maps)}")
        print("\nOutput Feature Map Shapes from forward:")
        expected_strides = [4, 8, 16, 32]
        for i, feat_map in enumerate(output_feature_maps):
            print(f"Stage {i} Output Shape:", feat_map.shape)
            expected_h = input_image_size // expected_strides[i]
            expected_w = input_image_size // expected_strides[i]
            assert feat_map.shape[-2:] == (expected_h, expected_w)
        print(f"Shape checks for {input_image_size}x{input_image_size} passed successfully!")