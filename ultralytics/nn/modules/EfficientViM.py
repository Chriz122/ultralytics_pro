import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, SqueezeExcite
from timm.models.registry import register_model
from fvcore.nn import flop_count
import numpy as np

__all__ = ['EfficientViM_M1', 'EfficientViM_M2', 'EfficientViM_M3', 'EfficientViM_M4']

class LayerNorm2D(nn.Module):
    """LayerNorm for channels of 2D tensor(B C H W)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias
        return x_normalized


class LayerNorm1D(nn.Module):
    """LayerNorm for channels of 1D tensor(B C L)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm1D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias
        return x_normalized
    
    
class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm2d, act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer2D, self).__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        
        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x
    
    
class ConvLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm1d, act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer1D, self).__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        
        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.fc1 = ConvLayer2D(in_dim, dim, 1)
        self.fc2 = ConvLayer2D(dim, in_dim, 1, act_layer=None, bn_weight_init=0)
        
    def forward(self, x):
        x = self.fc2(self.fc1(x))
        return x


class Stem(nn.Module):
    def __init__(self,  in_dim=3, dim=96):
        super().__init__()
        self.conv = nn.Sequential(
            ConvLayer2D(in_dim, dim // 8, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 8, dim // 4, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 4, dim // 2, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 2, dim, kernel_size=3, stride=2, padding=1, act_layer=None))

    def forward(self, x):
        x = self.conv(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self,  in_dim, out_dim, ratio=4.0):
        super().__init__()
        hidden_dim = int(out_dim * ratio)
        self.conv = nn.Sequential(
            ConvLayer2D(in_dim, hidden_dim, kernel_size=1),
            ConvLayer2D(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, groups=hidden_dim),
            SqueezeExcite(hidden_dim, .25),
            ConvLayer2D(hidden_dim, out_dim, kernel_size=1, act_layer=None)
        )
        
        self.dwconv1 = ConvLayer2D(in_dim, in_dim, 3, padding=1, groups=in_dim, act_layer=None)
        self.dwconv2 = ConvLayer2D(out_dim, out_dim, 3, padding=1, groups=out_dim, act_layer=None)

    def forward(self, x):
        x = x + self.dwconv1(x)
        x = self.conv(x)
        x = x + self.dwconv2(x)
        return x


class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim = 64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(d_model, 3*state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim*3
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3,1,1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0) 
        self.hz_proj = ConvLayer1D(d_model, 2*self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        # 這裡設定 inplace=False 以避免反向傳播時的 view 修改錯誤
        self.act = nn.SiLU(inplace=False)
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x, H, W):
        # x shape: (B, C, L)
        batch, _, L = x.shape
        
        # 原始錯誤代碼: H = int(math.sqrt(L)) -> 移除了這行假設正方形的代碼
        
        # 使用傳入的 H, W 進行 reshape
        # 先由 1D Conv 投影，再 reshape 成 2D 進行 Depthwise Conv
        BCdt = self.dw(self.BCdt_proj(x).view(batch, -1, H, W)).flatten(2)
        
        B, C, dt = torch.split(BCdt, [self.state_dim, self.state_dim,  self.state_dim], dim=1) 
        A = (dt + self.A.view(1,-1,1)).softmax(-1) 
        
        AB = (A * B) 
        h = x @ AB.transpose(-2,-1) 
        
        hz = self.hz_proj(h)
        h, z = torch.split(hz, [self.d_inner, self.d_inner], dim=1)
        
        # 使用 contiguous 確保內存連續，避免 split view inplace 錯誤
        z = z.contiguous()
        h = h.contiguous()
        
        h = self.out_proj(h * self.act(z) + h * self.D)
        y = h @ C 
        
        # Reshape 回 2D (使用傳入的 H, W)
        y = y.view(batch, -1, H, W).contiguous()
        return y, h


class EfficientViMBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        
        self.mixer = HSMSSD(d_model=dim, ssd_expand=ssd_expand,state_dim=state_dim)  
        self.norm = LayerNorm1D(dim)
        
        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer = None)
        self.dwconv2 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer = None)
        
        self.ffn = FFN(in_dim=dim, dim=int(dim * mlp_ratio))
        
        #LayerScale
        self.alpha = nn.Parameter(1e-4 * torch.ones(4,dim), requires_grad=True)
        
    def forward(self, x):
        # x input is (B, C, H, W)
        B, C, H, W = x.shape
        alpha = torch.sigmoid(self.alpha).view(4,-1,1,1)
        
        # DWconv1
        x = (1-alpha[0]) * x + alpha[0] * self.dwconv1(x)
        
        # HSM-SSD
        x_prev = x
        
        # 這裡將 H, W 傳入 mixer，並將 2D 特徵展平為 1D
        x_flat = x.flatten(2) # (B, C, L)
        y, h = self.mixer(self.norm(x_flat), H, W) # Output y is (B, C, H, W)
        
        x = (1-alpha[1]) * x_prev + alpha[1] * y
        
        # DWConv2
        x = (1-alpha[2]) * x + alpha[2] * self.dwconv2(x)
        
        # FFN
        x = (1-alpha[3]) * x + alpha[3] * self.ffn(x)
        return x, h


class EfficientViMStage(nn.Module):
    def __init__(self, in_dim, out_dim, depth,  mlp_ratio=4.,downsample=None, ssd_expand=1, state_dim=64):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([
            EfficientViMBlock(dim=in_dim, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand, state_dim=state_dim) for _ in range(depth)])
        
        self.downsample = downsample(in_dim=in_dim, out_dim =out_dim) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x, h = blk(x)
            
        x_out = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, x_out


class EfficientViM(nn.Module):
    def __init__(self, img_size=224, in_dim=3, embed_dim=[128,256,512], depths=[2, 2, 2], mlp_ratio=4., 
                 ssd_expand=1, state_dim=[49,25,9], pretrained=None, distillation=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_embed = Stem(in_dim=in_dim, dim=embed_dim[0])
        PatchMergingBlock = PatchMerging

        # build stages
        self.stages = nn.ModuleList()
        for i_layer in range(self.num_layers):
            stage = EfficientViMStage(in_dim=int(embed_dim[i_layer]),
                               out_dim=int(embed_dim[i_layer+1]) if (i_layer < self.num_layers - 1) else None,
                               depth=depths[i_layer],
                               mlp_ratio=mlp_ratio,
                               downsample=PatchMergingBlock if (i_layer < self.num_layers - 1) else None,
                               ssd_expand=ssd_expand,
                               state_dim = state_dim[i_layer])
            self.stages.append(stage)
        
        self.apply(self._init_weights)

        # Calculate width list for YOLO
        # Using 640x640 as a standard detection resolution
        test_res = 640 if img_size < 640 else img_size
        try:
            self.width_list = [i.size(1) for i in self.forward(torch.randn(1, 3, test_res, test_res))]
        except Exception as e:
            print(f"Warning: Failed to calculate width_list automatically: {e}")
            # Fallback to embed dimensions
            self.width_list = [embed_dim[0]] + [stage.blocks[-1].dim for stage in self.stages]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm2D):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm1D):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def flops(self, shape=(3, 224, 224)):
        supported_ops = {
            "aten::silu": None,  
            "aten::neg": None,  
            "aten::exp": None, 
            "aten::flip": None, 
            "aten::softmax": None,  
            "aten::sigmoid": None,  
            "aten::mul": None, 
            "aten::add": None, 
            "aten::mean": None,  
            "aten::var": None,  
            "aten::sub": None, 
            "aten::sqrt": None, 
            "aten::div": None,  
            "aten::rsub": None,  
            "aten::adaptive_avg_pool1d": None,  
        }
        import copy
        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        del model, input
        return sum(Gflops.values()) * 1e9

    def forward(self, x):
        outs = []
        x = self.patch_embed(x)
        outs.append(x) # Stem output
        
        for stage in self.stages:
            x, x_out = stage(x)
            outs.append(x_out)
            
        return outs

# Configurations
EfficientViM_m1 = {
    'embed_dim': [128, 192, 320],
    'depths': [2, 2, 2],
    'mlp_ratio': 4.,
    'ssd_expand': 1.,
    'state_dim': [49, 25, 9],
}

EfficientViM_m2 = {
    'embed_dim': [128, 256, 512],
    'depths': [2, 2, 2],
    'mlp_ratio': 4.,
    'ssd_expand': 1.,
    'state_dim': [49, 25, 9],
}

EfficientViM_m3 = {
    'embed_dim': [224, 320, 512],
    'depths': [2, 2, 2],
    'mlp_ratio': 4.,
    'ssd_expand': 1.,
    'state_dim': [49, 25, 9],
}

EfficientViM_m4 = {
    'embed_dim': [224, 320, 512],
    'depths': [3, 4, 2],
    'mlp_ratio': 4.,
    'ssd_expand': 1.,
    'state_dim': [64, 32, 16],
}

def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict

@register_model
def EfficientViM_M1(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViM_m1, **kwargs):
    model = EfficientViM(in_dim=3, distillation=distillation, **model_cfg, **kwargs)
    if pretrained:
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            if 'model' in state_dict: state_dict = state_dict['model']
            model.load_state_dict(update_weight(model.state_dict(), state_dict))
    return model

@register_model
def EfficientViM_M2(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViM_m2, **kwargs):
    model = EfficientViM(in_dim=3, distillation=distillation, **model_cfg, **kwargs)
    if pretrained:
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            if 'model' in state_dict: state_dict = state_dict['model']
            model.load_state_dict(update_weight(model.state_dict(), state_dict))
    return model

@register_model
def EfficientViM_M3(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViM_m3, **kwargs):
    model = EfficientViM(in_dim=3, distillation=distillation, **model_cfg, **kwargs)
    if pretrained:
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            if 'model' in state_dict: state_dict = state_dict['model']
            model.load_state_dict(update_weight(model.state_dict(), state_dict))
    return model

@register_model
def EfficientViM_M4(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViM_m4, **kwargs):
    model = EfficientViM(in_dim=3, distillation=distillation, **model_cfg, **kwargs)
    if pretrained:
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            if 'model' in state_dict: state_dict = state_dict['model']
            model.load_state_dict(update_weight(model.state_dict(), state_dict))
    return model