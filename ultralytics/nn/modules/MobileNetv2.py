import torch
from torch import nn
 
__all__ = ['MobileNetV2_n', 'MobileNetV2_s', 'MobileNetV2_m']
 
 
class ConvNormReLUBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        activation: bool = nn.ReLU6,
    ):
        """Constructs a block containing a combination of convolution, batchnorm and relu
        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            kernel_size (list): kernel size parameter for convolution
            stride (int, optional): stride parameter for convolution. Defaults to 1.
            padding (int, optional): padding parameter for convolution. Defaults to 0.
            groups (int, optional): number of blocked connections from input channel to output channel for convolution. Defaults to 1.
            bias (bool, optional): whether to enable bias in convolution. Defaults to False.
            activation (bool, optional): activation function to use. Defaults to nn.ReLU6.
        """
 
        super().__init__()
 
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation()
 
    def forward(self, x):
        """Perform forward pass."""
 
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
 
        return x
 
 
class InverseResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_factor: int = 6,
        stride: int = 1,
    ):
        """Constructs a inverse residual block with depthwise seperable convolution
        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            expansion_factor (int, optional): Calculating the input & output channel for depthwise convolution by multiplying the expansion factor with input channels. Defaults to 6.
            stride (int, optional): stride paramemeter for depthwise convolution. Defaults to 1.
            CSDN:Snu77
        """
 
        super().__init__()
 
        hidden_channels = in_channels * expansion_factor
        self.residual = in_channels == out_channels and stride == 1
 
        self.conv1 = (
            ConvNormReLUBlock(in_channels, hidden_channels, (1, 1))
            if in_channels != hidden_channels
            else nn.Identity() #  If it's not the first layer, then we need to add a 1x1 convolutional layer to expand the number of channels
        )
        self.depthwise_conv = ConvNormReLUBlock(
            hidden_channels,
            hidden_channels,
            (3, 3),
            stride=stride,
            padding=1,
            groups=hidden_channels,
        )
        self.conv2 = ConvNormReLUBlock(
            hidden_channels, out_channels, (1, 1), activation=nn.Identity
        )
 
    def forward(self, x):
        """Perform forward pass."""
 
        identity = x
 
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.conv2(x)
 
        if self.residual:
            x = torch.add(x, identity)
 
        return x
 
 
class MobileNetV2(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        depth_multiplier: float = 1,
    ):
        """Constructs MobileNetV2 architecture
        Args:
            n_classes (int, optional): output neuron in last layer. Defaults to 1000.
            input_channel (int, optional): input channels in first conv layer. Defaults to 3.
            dropout (float, optional): dropout in last layer. Defaults to 0.2.
        """
 
        super().__init__()
 
        # The configuration of MobileNetV2
        # input channels, expansion factor, output channels, repeat, stride,
        config = (
            (32, 1, 16, 1, 1),
            (16, 6, 24, 2, 2),
            (24, 6, 32, 3, 2),
            (32, 6, 64, 4, 2),
            (64, 6, 96, 3, 1),
            (96, 6, 160, 3, 2),
            (160, 6, 320, 1, 1),
        )
 
        layers = [
            ConvNormReLUBlock(input_channel, int(32 * depth_multiplier), (3, 3), stride=2, padding=1)
        ]
 
        # 遍历配置并添加 InverseResidualBlock 层
        for in_channels, expansion_factor, out_channels, repeat, stride in config:  # repeat不放缩了已经足够轻量化了
            for _ in range(repeat):
                layers.append(
                    InverseResidualBlock(
                        in_channels=int(in_channels * depth_multiplier),
                        out_channels=int(out_channels * depth_multiplier),
                        expansion_factor=expansion_factor,
                        stride=stride,
                    )
                )
                in_channels = out_channels  # 更新输入通道
                stride = 1  # 重复层的 stride 设为 1
 
        # 将层列表转换为 nn.Sequential
        self.model = nn.Sequential(*layers)
 
        self.width_list = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]
 
    def forward(self, x):
        """Perform forward pass."""
        unique_tensors = {}
        for model in self.model:
            x = model(x)
            width, height = x.shape[2], x.shape[3]
            unique_tensors[(width, height)] = x
        result_list = list(unique_tensors.values())[-4:]
        return result_list
 
 
def MobileNetV2_n(width_mult=0.5):
    model = MobileNetV2(depth_multiplier=0.25)
    return model
 
def MobileNetV2_s(width_mult=1.0):
    model = MobileNetV2(depth_multiplier=0.5)
    return model
 
def MobileNetV2_m(width_mult=1.5):
    model = MobileNetV2(depth_multiplier=1)
    return model
 
 
if __name__ == "__main__":
 
    # Generating Sample image
    image_size = (1, 3, 224, 224)
    image = torch.rand(*image_size)
 
    # Model
    mobilenet_v2 = MobileNetV2()
 
    out = mobilenet_v2(image)
    for i in range(len(out)):
        print(out[i].size())