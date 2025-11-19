

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

__all__ = (
    "LCA",
    "ALSS",
    "CA"
)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

def channel_shuffle(x, g):
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // g
    # Reshape
    x = x.view(batch_size, g, channels_per_group, height, width)
    # Transpose 1 and 2 axis
    x = x.transpose(1, 2).contiguous()
    # Flatten
    x = x.view(batch_size, -1, height, width)
    return x


class ALSS(nn.Module):
    def __init__(self, C_in, C_out, num_blocks=1, alpha=0.2, beta=1, stride=1, use_identity=False, shortcut_mode=False):
        super(ALSS, self).__init__()
        
        # Calculate split sizes
        self.shortcut_channels = int(C_in * alpha)
        self.main_in_channels = C_in - self.shortcut_channels
        bottleneck_channels = int(self.main_in_channels * beta)
        main_out_channels = C_out - self.shortcut_channels
        
        self.num_blocks = num_blocks
        
        # Shortcut path
        if stride == 2:
            if shortcut_mode == 0:
                self.shortcut = Conv(self.shortcut_channels, self.shortcut_channels, 3, 2)
            elif shortcut_mode == 1:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                    Conv(self.shortcut_channels, self.shortcut_channels, 3, 1)
                )
            else:
                self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.shortcut = nn.Identity() if use_identity else \
                Conv(self.shortcut_channels, self.shortcut_channels, 3, 1)

        
        # Main path
        self.initial_conv = Conv(self.main_in_channels, bottleneck_channels, 3, 1)
        
        self.middle_convs = nn.ModuleList()
        if stride == 2:
            self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, 2, g=bottleneck_channels, act=False))
            for _ in range(1, num_blocks):
                self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, 1, g=bottleneck_channels, act=False))
        else:
            for _ in range(num_blocks):
                self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, 1, g=bottleneck_channels, act=False))
        
        self.final_conv = Conv(bottleneck_channels, main_out_channels, 3, 1)

    def forward(self, x):
        # Split input into shortcut and main branches
        proportional_sizes=[self.shortcut_channels,self.main_in_channels]
        x = list(x.split(proportional_sizes, dim=1))

        
        # Process shortcut path
        shortcut_x = self.shortcut(x[0])
        
        # Process main path
        main_x = self.initial_conv(x[1])
        for conv in self.middle_convs:
            main_x = conv(main_x)
        main_x = self.final_conv(main_x)
        
        # Concatenate and shuffle channels
        out_x = torch.cat((main_x, shortcut_x), dim=1)
        out_x = channel_shuffle(out_x, 2)
        return out_x
    

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
 
    def forward(self, x):
        return self.relu(x + 3) / 6
    
    
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
 
    def forward(self, x):
        return x * self.sigmoid(x)
    
    
class LCA(nn.Module):
    def __init__(self, input_channel, reduction=32):
        super(LCA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
        self.conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.act = h_swish()
 
        self.conv_h = nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1, padding=0,groups=input_channel)
        self.conv_w = nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1, padding=0,groups=input_channel)

    
    def forward(self, x):  # torch.Size([2, 32, 64, 64])
        identity = x

        b, c, h, w = x.size()
        x_h = self.pool_h(x)  # torch.Size([2, 32, 64, 1])
        x_w = self.pool_w(x)  # torch.Size([2, 32, 1, 64]) 

        a_h = self.conv_h(x_h).sigmoid()  #  torch.Size([2, 32, 64, 1])
        a_w = self.conv_w(x_w).sigmoid()  #  torch.Size([2, 32, 1, 64])

        out = identity * a_w * a_h  # torch.Size([2, 32, 64, 64])

        return out
    

class CA(nn.Module):
    def __init__(self, input_channel, reduction=32):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
        mip = max(8, input_channel // reduction)
 
        self.conv1 = nn.Conv2d(input_channel, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
 
        self.conv_h = nn.Conv2d(mip, input_channel, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, input_channel, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        identity = x
 
        b, c, h, w = x.size()
        x_h = self.pool_h(x) #torch.Size([2, 32, 64, 1])
        x_w = self.pool_w(x).permute(0, 1, 3, 2) #torch.Size([2, 32, 64, 1])
 
        y = torch.cat([x_h, x_w], dim=2) #torch.Size([2, 32, 128, 1])
        y = self.conv1(y) #torch.Size([2, 8, 128, 1])
        y = self.bn1(y)
        y = self.act(y)
 
        x_h, x_w = torch.split(y, [h, w], dim=2) #torch.Size([2, 8, 64, 1]) torch.Size([2, 8, 64, 1])
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return out
    
    
# 测试 CoordAtt 模块
if __name__ == '__main__':
    torch.manual_seed(0)
    input_tensor = torch.rand(2, 32, 64, 64)  # 假设输入尺寸为 [batch_size, channels, height, width]
    coord_att = LCA(input_channel=32)  # 创建 CoordAtt 实例，假设输入通道为32
    output = coord_att(input_tensor)  # 获取输出

    print(f'Input shape: {input_tensor.shape}')
    print(f'Output shape: {output.shape}')

    # 如果需要，还可以使用 torchsummary 来查看模型的详细信息
    # summary(coord_att, input_size=(32, 64, 64))