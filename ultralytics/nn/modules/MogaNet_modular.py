import torch
import torch.nn as nn
import torch.nn.functional as F

# 如果沒有 timm，這裡提供一個簡單的 DropPath 實現
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if self.keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

def build_act_layer(act_type):
    if act_type is None: return nn.Identity()
    if act_type == 'SiLU': return nn.SiLU()
    if act_type == 'ReLU': return nn.ReLU()
    return nn.GELU()

class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        # 假設輸入是 (B, C, H, W)，類似 BatchNorm 的行為
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )
    def forward(self, x):
        return x * self.scale

class MultiOrderDWConv(nn.Module):
    def __init__(self, embed_dims, dw_dilation=[1, 2, 3], channel_split=[1, 3, 4]):
        super(MultiOrderDWConv, self).__init__()
        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims

        self.DW_conv0 = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2, groups=self.embed_dims, stride=1, dilation=dw_dilation[0])
        self.DW_conv1 = nn.Conv2d(self.embed_dims_1, self.embed_dims_1, kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2, groups=self.embed_dims_1, stride=1, dilation=dw_dilation[1])
        self.DW_conv2 = nn.Conv2d(self.embed_dims_2, self.embed_dims_2, kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2, groups=self.embed_dims_2, stride=1, dilation=dw_dilation[2])
        self.PW_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x

class MultiOrderGatedAggregation(nn.Module):
    def __init__(self, embed_dims, attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4], attn_act_type='SiLU'):
        super(MultiOrderGatedAggregation, self).__init__()
        self.proj_1 = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(embed_dims, dw_dilation=attn_dw_dilation, channel_split=attn_channel_split)
        self.proj_2 = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)
        self.sigma = ElementScale(embed_dims, init_value=1e-5, requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        g = self.gate(x)
        v = self.value(x)
        x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        return x + shortcut

class ChannelAggregationFFN(nn.Module):
    def __init__(self, embed_dims, feedforward_channels, act_type='GELU', ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()
        self.fc1 = nn.Conv2d(embed_dims, feedforward_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(feedforward_channels, feedforward_channels, kernel_size=3,
            stride=1, padding=1, bias=True, groups=feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(feedforward_channels, embed_dims, kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)
        self.decompose = nn.Conv2d(feedforward_channels, 1, kernel_size=1)
        self.sigma = ElementScale(feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MogaBlock(nn.Module):
    def __init__(self, embed_dims, ffn_ratio=4., drop_rate=0., drop_path_rate=0., 
                 act_type='GELU', norm_type='BN', init_value=1e-5, 
                 attn_dw_dilation=[1, 2, 3]):
        super(MogaBlock, self).__init__()
        
        # Norm layer builder simplified
        def _build_norm(n_type, dims):
            if n_type == 'BN': return nn.BatchNorm2d(dims, eps=1e-5)
            if n_type == 'LN2d': return LayerNorm2d(dims, eps=1e-6)
            return nn.BatchNorm2d(dims)

        self.norm1 = _build_norm(norm_type, embed_dims)
        self.attn = MultiOrderGatedAggregation(embed_dims, attn_dw_dilation=attn_dw_dilation)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = _build_norm(norm_type, embed_dims)
        
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(embed_dims, mlp_hidden_dim, act_type, drop_rate)
        
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x

# -----------------------------------------------------------------------------------------
# 以下是專為 YOLOv8 YAML 配置設計的模塊
# -----------------------------------------------------------------------------------------

class StackConvPatchEmbed(nn.Module):
    """
    對應 Stage 1 的 Embedding
    args: [c2, kernel_size, stride]
    """
    def __init__(self, c1, c2, kernel_size=3, stride=2, act_type='GELU', norm_type='BN'):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(c1, c2 // 2, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.BatchNorm2d(c2 // 2) if norm_type=='BN' else LayerNorm2d(c2//2),
            build_act_layer(act_type),
            nn.Conv2d(c2 // 2, c2, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.BatchNorm2d(c2) if norm_type=='BN' else LayerNorm2d(c2),
        )

    def forward(self, x):
        return self.projection(x)

class ConvPatchEmbed(nn.Module):
    """
    對應 Stage 2, 3, 4 的 Embedding
    args: [c2, kernel_size, stride]
    """
    def __init__(self, c1, c2, kernel_size=3, stride=2, norm_type='BN'):
        super().__init__()
        self.projection = nn.Conv2d(c1, c2, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.norm = nn.BatchNorm2d(c2) if norm_type == 'BN' else LayerNorm2d(c2)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        return x

class MogaStage(nn.Module):
    """
    MogaNet 的一個完整 Stage，包含 N 個 Block 和一個尾部的 Norm。
    設計這個類是為了處理內部 Block 的 dilation 變化邏輯。
    YAML args: [c2, depth, ffn_ratio]
    注意：c1 由上層傳入，若 c1 != c2 則報錯 (除非添加 projection，但 MogaNet 架構通常在 Embed 層改變通道)
    """
    def __init__(self, c1, c2, depth, ffn_ratio=4.0, drop_path_rate=0.0):
        super().__init__()
        # 在 MogaNet 中，Block 不改變通道數，所以 c1 必須等於 c2
        assert c1 == c2, f"MogaStage input channels {c1} must equal output channels {c2}. Change dimensions in PatchEmbed."
        
        self.blocks = nn.ModuleList([])
        # 簡單的 Dilation 循環邏輯
        att_dilations = [1, 2, 3]
        
        for i in range(depth):
            dilation = att_dilations[i % 3] # 循環 [1, 2, 3]
            # MogaNet 每個 Stage 最後一個 block 的 dilation 有時會變，這裡簡化為循環
            
            block = MogaBlock(
                embed_dims=c2,
                ffn_ratio=ffn_ratio,
                drop_path_rate=drop_path_rate, # 這裡簡化處理，統一 drop rate
                attn_dw_dilation=[1, dilation, 1] # 簡化邏輯，原始代碼比較複雜
            )
            self.blocks.append(block)
        
        self.norm = nn.BatchNorm2d(c2) # 默認使用 BN 作為 Stage 結尾

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x