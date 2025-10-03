import torch
import torch.nn as nn
import os
from torch.nn.parameter import Parameter
from collections import OrderedDict

# __all__ 可以保持不變，因為我們仍然通過這些函數創建模型
__all__ = ['sa_resnet50', 'sa_resnet101', 'sa_resnet152', 'ResNet']

# 將模型配置集中管理，類似 MobileNetV4 的 MODEL_SPECS
SA_RESNET_CONFIGS = {
    'sa_resnet50': {'block': None, 'layers': [3, 4, 6, 3], 'url': ''}, # block 會在 ResNet 中設置
    'sa_resnet101': {'block': None, 'layers': [3, 4, 23, 3], 'url': ''},
    'sa_resnet152': {'block': None, 'layers': [3, 8, 36, 3], 'url': ''},
}
# 注意：實際使用時需要填充 'url' 或提供本地權重路徑

def load_state_dict(checkpoint_path):
    """Loads state dict from a checkpoint file."""
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix if it exists
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif isinstance(checkpoint, OrderedDict):
             # Directly load if it's already a state_dict
             state_dict = checkpoint
        else:
             # Handle cases where the checkpoint is the model itself or other formats
             # This might need adjustment based on how pretrained weights are saved
             raise ValueError(f"Could not find a valid state_dict in checkpoint: {checkpoint_path}")

        print(f"Loaded state_dict from checkpoint '{checkpoint_path}'")
        return state_dict
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

# load_checkpoint 函數可以移除或保留，因為加載邏輯現在在 ResNet.__init__ 內部
# def load_checkpoint(model, checkpoint_path, strict=False):
#     state_dict = load_state_dict(checkpoint_path)
#     model.load_state_dict(state_dict, strict=strict)


class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module."""
    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        # 確保 channel 可以被 2*groups 整除，或者調整 groups
        if channel % (2 * groups) != 0:
            # 嘗試找一個可以整除的 groups 數量，或者拋出錯誤
            # 這裡簡單地調整 groups，可能需要根據實際 channel 大小調整策略
            new_groups = groups
            while channel % (2 * new_groups) != 0 and new_groups > 1:
                new_groups //= 2
            if channel % (2 * new_groups) != 0:
                 # 如果找不到合適的 groups，可能需要報錯或使用默認組（如 1）
                 # 為了示例，我們強制 groups 為 1，但這會改變原意
                #  print(f"Warning: channel {channel} not divisible by 2*groups={2*groups}. Adjusting groups might be needed.")
                 # 這裡保持原樣，讓 GroupNorm 報錯，以便用戶意識到問題
                 pass # 或者 new_groups = 1

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 計算 normalization 層的組數和通道數
        norm_groups = channel // (2 * groups) if groups > 0 and channel % (2*groups) == 0 else channel // 2 # 避免除以零或無法整除
        if norm_groups <= 0 : norm_groups = 1 # 至少為 1

        self.cweight = Parameter(torch.zeros(1, norm_groups, 1, 1))
        self.cbias = Parameter(torch.ones(1, norm_groups, 1, 1))
        self.sweight = Parameter(torch.zeros(1, norm_groups, 1, 1))
        self.sbias = Parameter(torch.ones(1, norm_groups, 1, 1))

        self.sigmoid = nn.Sigmoid()
        try:
             self.gn = nn.GroupNorm(norm_groups, norm_groups)
        except ValueError as e:
             print(f"Error initializing GroupNorm in sa_layer: {e}. Channel: {channel}, Groups: {groups}, Norm Groups: {norm_groups}")
             # 可以提供一個備用方案，例如 BatchNorm，或者讓錯誤繼續拋出
             self.gn = nn.BatchNorm2d(norm_groups) # 備選方案


    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, c, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.shape

        # 確保通道數可以被 groups 整除
        if c % self.groups != 0:
             raise ValueError(f"Input channels {c} must be divisible by groups {self.groups} in sa_layer")
        if c % 2 != 0:
             raise ValueError(f"Input channels {c} must be divisible by 2 for chunking")

        # 調整 reshape 以匹配可能的 groups 變化
        channels_per_group_total = c // self.groups
        if channels_per_group_total % 2 != 0:
             raise ValueError(f"Channels per group ({channels_per_group_total}) must be even for chunking.")

        x_reshaped = x.view(b * self.groups, channels_per_group_total, h, w)
        x_0, x_1 = x_reshaped.chunk(2, dim=1) # 在 group 內的通道維度上 chunk

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1) # gn 在 chunk 後的通道維度上操作
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.view(b, c, h, w) # Reshape back to original batch and combined channel dimensions

        # Shuffle across the two major chunks (channel and spatial)
        out = self.channel_shuffle(out, 2) # Shuffle the concatenated result
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SABottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SABottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        # 確保 sa_layer 的 channel 輸入正確
        self.sa = sa_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sa(out) # Apply SA layer

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # 修改 __init__ 以接受 model_name
    def __init__(self, model_name, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, pretrained=False):
        super(ResNet, self).__init__()

        # 從配置字典獲取模型參數
        if model_name not in SA_RESNET_CONFIGS:
            raise ValueError(f"Unknown model name: {model_name}. Available models: {list(SA_RESNET_CONFIGS.keys())}")
        config = SA_RESNET_CONFIGS[model_name]
        block = SABottleneck # 直接使用 SABottleneck
        SA_RESNET_CONFIGS[model_name]['block'] = block # 更新配置中的 block 信息 (可選)
        layers = config['layers']

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Initial layers
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # Final layers (kept for potential use, but forward returns features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                     nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                     nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, SABottleneck):
                     if m.bn3.weight is not None:
                          nn.init.constant_(m.bn3.weight, 0)

        # Load pretrained weights if requested
        if pretrained:
            checkpoint_path = config['url'] # Use URL or map to local path
            if checkpoint_path:
                try:
                    # Use the existing load_state_dict helper function
                    state_dict = load_state_dict(checkpoint_path)
                    # Load state dict, strict=False allows loading backbone weights
                    # even if FC layer size doesn't match or is missing.
                    missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
                    # print(f"Pretrained weights loaded for {model_name}.")
                    if missing_keys:
                         print(f"Warning: Missing keys: {missing_keys}")
                    if unexpected_keys:
                         print(f"Warning: Unexpected keys: {unexpected_keys}")
                except FileNotFoundError:
                    print(f"Warning: Pretrained weights file not found at {checkpoint_path}. Model initialized randomly.")
                except Exception as e:
                    print(f"Warning: Error loading pretrained weights: {e}. Model initialized randomly.")
            else:
                print(f"Warning: No pretrained weights path specified for {model_name}. Model initialized randomly.")


        # --- Add width_list calculation like in MobileNetV4 ---
        # print(f"Calculating width list for {model_name}...")
        self.eval()  # Set to evaluation mode for dummy forward pass
        try:
            with torch.no_grad(): # Disable gradient calculation
                # Use a standard input size for ResNet, e.g., 224x224 or 640x640
                # Using 224x224 as it's common for ImageNet ResNets
                dummy_input = torch.randn(1, 3, 224, 224)
                # Make sure the forward method returns the list of features
                features = self.forward(dummy_input)
                self.width_list = [f.size(1) for f in features]
        except Exception as e:
             print(f"Error during dummy forward pass for width_list calculation: {e}")
             # Assign a default or raise error if calculation fails
             self.width_list = [] # Or some default based on architecture knowledge
        finally:
             self.train() # Set back to training mode
        # print(f"Initialized {model_name}. Width list: {self.width_list}")
        # --- End of width_list calculation ---


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # Modify forward to return intermediate features
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x) # Output after stage 1 + maxpool

        x1 = self.layer1(x0) # Output after stage 2
        x2 = self.layer2(x1) # Output after stage 3
        x3 = self.layer3(x2) # Output after stage 4
        x4 = self.layer4(x3) # Output after stage 5

        # Return a list of feature maps from different stages
        # Commonly returned features for detection/segmentation are C3, C4, C5
        # Here returning outputs of layer1, layer2, layer3, layer4 for consistency
        return [x1, x2, x3, x4]

    # Optional: Add a method for classification if needed
    def forward_classify(self, x):
        features = self.forward(x)
        # Use the last feature map for classification
        out = self.avgpool(features[-1])
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# Update instantiation functions to use the modified ResNet class
def sa_resnet50(pretrained=False, **kwargs):
    """Constructs a SA-ResNet-50 model."""
    model = ResNet(model_name='sa_resnet50', pretrained=pretrained, **kwargs)
    return model

def sa_resnet101(pretrained=False, **kwargs):
    """Constructs a SA-ResNet-101 model."""
    model = ResNet(model_name='sa_resnet101', pretrained=pretrained, **kwargs)
    return model

def sa_resnet152(pretrained=False, **kwargs):
    """Constructs a SA-ResNet-152 model."""
    model = ResNet(model_name='sa_resnet152', pretrained=pretrained, **kwargs)
    return model

# Example Usage (similar to MobileNetV4 example)
if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 3, 224, 224) # Typical ResNet input size
    image = torch.rand(*image_size)

    # Model (Example: SA-ResNet-50)
    # Set pretrained=True if you have weights and configured the path/URL in SA_RESNET_CONFIGS
    model = sa_resnet50(pretrained=False)
    print(model) # Print model structure

    # Get intermediate features
    model.eval() # Set to eval mode for inference
    with torch.no_grad():
        features = model(image)

    print("\nOutput feature map shapes:")
    for i, f in enumerate(features):
        print(f"Layer {i+1} output shape: {f.shape}")

    # Access the calculated width list
    print(f"\nCalculated width list: {model.width_list}")

    # Example for classification (if needed)
    # classification_output = model.forward_classify(image)
    # print(f"\nClassification output shape: {classification_output.shape}")