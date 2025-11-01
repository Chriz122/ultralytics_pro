import torch
import torch.nn as nn
from typing import Type, Any, List, Optional, Union, Dict

__all__ = ['ResNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None
    ) -> None:
        super().__init__()
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
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ResNet 配置字典
RESNET_SPECS: Dict[str, Dict[str, Union[Type[Union[BasicBlock, Bottleneck]], List[int]]]] = {
    'ResNet18': {'block': BasicBlock, 'layers': [2, 2, 2, 2]},
    'ResNet34': {'block': BasicBlock, 'layers': [3, 4, 6, 3]},
    'ResNet50': {'block': Bottleneck, 'layers': [3, 4, 6, 3]},
    'ResNet101': {'block': Bottleneck, 'layers': [3, 4, 23, 3]},
    'ResNet152': {'block': Bottleneck, 'layers': [3, 8, 36, 3]},
}

MODEL_SPECS = RESNET_SPECS # Maintain consistency with previous examples

class ResNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
        init_weights: bool = True
    ) -> None:
        """
        ResNet 網絡基礎類 (模仿 MobileNetV4 結構)

        Args:
            model_name (str): ResNet模型的名稱 (例如 'ResNet50').
            zero_init_residual (bool): 是否將殘差分支的最後一個BN層初始化為0.
            groups (int): 分組卷積的組數 (用於 ResNeXt).
            width_per_group (int): 每個組的寬度 (用於 ResNeXt / Wide ResNet).
            replace_stride_with_dilation (Optional[List[bool]]): 是否用空洞卷積替換步幅卷積.
            norm_layer (Optional[Type[nn.Module]]): 指定歸一化層類型.
            init_weights (bool): 是否初始化權重.
        """
        super().__init__()
        assert model_name in MODEL_SPECS.keys(), f"未知模型名稱: {model_name}"
        self.model_name = model_name
        self.spec = MODEL_SPECS[self.model_name]
        block: Type[Union[BasicBlock, Bottleneck]] = self.spec['block']
        layers: List[int] = self.spec['layers']

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64 # Initial channel dimension after conv1
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # --- Initial Convolution ---
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- Main Stages ---
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # Note: Standard ResNet has avgpool and fc layer here, omitted for backbone use

        # Initialize weights (optional but recommended)
        if init_weights:
            self._initialize_weights()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


        # --- 模仿 MobileNetV4 計算 width_list ---
        # print("正在計算 width_list（可能需要一些時間）...")
        with torch.no_grad():
            # 創建一個符合模型預期的虛擬輸入 (640x640 to match other examples)
            # Note: ResNet typically uses 224x224
            dummy_input = torch.randn(1, 3, 640, 640)
            # 執行一次前向傳播以獲取中間層輸出
            intermediate_outputs = self.forward(dummy_input)
            # 從中間層輸出計算通道數
            self.width_list = [output.size(1) for output in intermediate_outputs]
        # print(f"計算得到的 width_list ({self.model_name}): {self.width_list}")
        # --- 結束模仿 MobileNetV4 計算 width_list ---

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        """構建 ResNet 的一個階段"""
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
        self.inplanes = planes * block.expansion # Update inplanes for the next block/layer
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> List[torch.Tensor]:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # Output after initial conv/pool

        # --- Execute main stages and collect outputs ---
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Return list of features after each stage
        return [x1, x2, x3, x4]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        """初始化網絡權重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # Note: Linear layer weights (if added later) should also be initialized


# --- 提供與 MobileNetV4 類似的模型創建函數 ---

def ResNet18(**kwargs: Any) -> ResNet:
    """創建 ResNet18 模型"""
    return ResNet('ResNet18', **kwargs)

def ResNet34(**kwargs: Any) -> ResNet:
    """創建 ResNet34 模型"""
    return ResNet('ResNet34', **kwargs)

def ResNet50(**kwargs: Any) -> ResNet:
    """創建 ResNet50 模型"""
    return ResNet('ResNet50', **kwargs)

def ResNet101(**kwargs: Any) -> ResNet:
    """創建 ResNet101 模型"""
    return ResNet('ResNet101', **kwargs)

def ResNet152(**kwargs: Any) -> ResNet:
    """創建 ResNet152 模型"""
    return ResNet('ResNet152', **kwargs)


if __name__ == "__main__":
    # --- 測試用例 ---
    # 生成樣本圖像 (尺寸與 MobileNetV4/VGG 範例一致)
    image_size = (1, 3, 640, 640)
    image = torch.rand(*image_size)

    # 創建模型 (例如 ResNet50)
    # init_weights=True 是預設且推薦的
    print("創建 ResNet50 模型...")
    model = ResNet50(init_weights=True)
    print("模型創建完成.")

    # 設置為評估模式
    model.eval()

    # 執行前向傳播
    print("執行前向傳播...")
    with torch.no_grad(): # 在推理時不需要計算梯度
        out = model(image)
    print("前向傳播完成.")

    # 打印輸出特徵圖的形狀
    print("\n輸出特徵圖形狀 (after layer1, layer2, layer3, layer4):")
    for i, feature_map in enumerate(out):
        print(f"  輸出 stage {i+1}: {feature_map.shape}")

    # 打印存儲的 width_list
    print(f"\n模型內部存儲的 width_list: {model.width_list}")

    # 測試創建其他 ResNet 模型
    print("\n嘗試創建 ResNet18...")
    model_resnet18 = ResNet18()
    # width_list 會在創建時打印

    print("\n嘗試創建 ResNet101...")
    model_resnet101 = ResNet101()
    # width_list 會在創建時打印