import torch
from torch import nn
from torch.nn import functional as F
from .conv import Conv

class block(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()
        d = max(dim // r, L)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        # Use GroupNorm instead of BatchNorm to handle 1x1 spatial dims
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, dim, h, w = x.size()
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        a1 = self.conv1(attn1)
        a2 = self.conv2(attn2)
        attn = torch.cat([a1, a2], dim=1)

        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        # Channel attention path
        ch_attn = self.global_pool(attn)
        z = self.fc1(ch_attn)
        a_b = self.fc2(z)
        a_b = a_b.view(batch_size, 2, dim // 2, 1)
        a_b = self.softmax(a_b)

        # Split weights and apply
        w1, w2 = a_b.chunk(2, dim=1)
        w1 = w1.view(batch_size, dim // 2, 1, 1)
        w2 = w2.view(batch_size, dim // 2, 1, 1)

        out_attn = a1 * w1 + a2 * w2
        out_attn = self.conv(out_attn).sigmoid()

        return x * out_attn
    

class Attention_KSB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = block(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut
    

class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention_KSB(c)
        self.ffn = nn.Sequential(
            Conv(c, int(c * attn_ratio * 2), 1),
            Conv(int(c * attn_ratio * 2), c, 1, act=False)
        )
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
    

class C2PSA_KS(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2, "Input and output channels must match"
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, dim=1)
        b = self.m(b)
        return self.cv2(torch.cat([a, b], dim=1))

if __name__ == "__main__":
    image_size = (1, 3, 640, 640)
    image = torch.rand(*image_size)
    model = C2PSA_KS(3, 3)
    out = model(image)
    print(out.shape)


 