import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite

__all__ = ['SHViT_S1', 'SHViT_S2', 'SHViT_S3', 'SHViT_S4']

# 保持 Code 1 的基礎層定義，這些是 SHViT 的核心組件

class GroupNorm(torch.nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        # Note: trunc_normal_ dependency removed or needs import. 
        # Using simple normal init if timm not available, else uncomment import.
        torch.nn.init.trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    
    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x
        

class SHSA(torch.nn.Module):
    """Single-Head Self-Attention"""
    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = GroupNorm(pdim)

        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim, dim, bn_weight_init = 0))
        

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim = 1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim = 1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim = 1))

        return x


class BasicBlock(torch.nn.Module):
    def __init__(self, dim, qk_dim, pdim, type):
        super().__init__()
        if type == "s":    # for later stages
            self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups = dim, bn_weight_init = 0))
            self.mixer = Residual(SHSA(dim, qk_dim, pdim))
            self.ffn = Residual(FFN(dim, int(dim * 2)))
        elif type == "i":   # for early stages
            self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups = dim, bn_weight_init = 0))
            self.mixer = torch.nn.Identity()
            self.ffn = Residual(FFN(dim, int(dim * 2)))
    
    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))


# === 核心修改部分：重構 SHViT 類別以符合 EfficientViT (Code 2) 的風格 ===

class SHViT(torch.nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[128, 256, 384],
                 partial_dim = [32, 64, 96],
                 qk_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 types = ["s", "s", "s"],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=False,
                 frozen_stages=0, # 添加接口參數，保持一致性
                 pretrained=None): # 添加接口參數
        super().__init__()

        # Patch embedding
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1))

        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []

        # Build SHViT blocks
        for i, (ed, kd, pd, dpth, do, t) in enumerate(
                zip(embed_dim, qk_dim, partial_dim, depth, down_ops, types)):
            for d in range(dpth):
                eval('self.blocks' + str(i+1)).append(BasicBlock(ed, kd, pd, t))
            if do[0] == 'subsample':
                # Build SHViT downsample block
                blk = eval('self.blocks' + str(i+2))
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i])),
                                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2))),))
                blk.append(PatchMerging(embed_dim[i], embed_dim[i+1])) # 修正這裡的調用，Code 1 這裡傳了兩個參數
                
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1])),
                                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2))),))
        
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)
        
        # Classification head - 雖然定義了，但在 Backbone 模式下通常不使用
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        # === 關鍵修改：添加 width_list ===
        # 計算特徵圖通道數，YOLO 等檢測框架需要此屬性
        self.width_list = [i.size(1) for i in self.forward_features(torch.randn(1, 3, 640, 640))]

    def forward_features(self, x):
        # 輔助函數：執行前向傳播並返回特徵列表
        outs = []
        x = self.patch_embed(x)
        outs.append(x)
        x = self.blocks1(x)
        outs.append(x)
        x = self.blocks2(x)
        outs.append(x)
        x = self.blocks3(x)
        outs.append(x)
        return outs

    def forward(self, x):
        # === 關鍵修改：返回列表而不是 Tensor ===
        # Code 2 (EfficientViT) 的 forward 返回 outs 列表
        # 這樣可以避免 'Tensor' object has no attribute 'insert' 錯誤
        return self.forward_features(x)


# === 工具函數 (參考 Code 2) ===

def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)

def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict


# === 模型配置與構建函數 (模仿 Code 2 的 EfficientViT_MX 風格) ===

SHViT_s1_cfg = {
        'embed_dim': [128, 224, 320],
        'depth': [2, 4, 5],
        'partial_dim': [32, 48, 68],  
        'types' : ["i", "s", "s"]
    }

def SHViT_S1(pretrained='', frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=SHViT_s1_cfg, num_classes=1000):
    model = SHViT(num_classes=num_classes, distillation=distillation, frozen_stages=frozen_stages, pretrained=pretrained, **model_cfg)
    if pretrained:
        # 假設 pretrained 是一個路徑，或者你可以保留 Code 1 的 url 邏輯，這裡採用 Code 2 的通用加載方式
        checkpoint = torch.load(pretrained, map_location='cpu')
        # 處理不同儲存格式 (如果有的話)
        d = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(update_weight(model.state_dict(), d))
    if fuse:
        replace_batchnorm(model)
    return model


SHViT_s2_cfg = {
        'embed_dim': [128, 308, 448],
        'depth': [2, 4, 5],
        'partial_dim': [32, 66, 96],
        'types' : ["i", "s", "s"]
    }

def SHViT_S2(pretrained='', frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=SHViT_s2_cfg, num_classes=1000):
    model = SHViT(num_classes=num_classes, distillation=distillation, frozen_stages=frozen_stages, pretrained=pretrained, **model_cfg)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        d = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(update_weight(model.state_dict(), d))
    if fuse:
        replace_batchnorm(model)
    return model


SHViT_s3_cfg = {
        'embed_dim': [192, 352, 448],
        'depth': [3, 5, 5],
        'partial_dim': [48, 75, 96],  
        'types' : ["i", "s", "s"]
    }

def SHViT_S3(pretrained='', frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=SHViT_s3_cfg, num_classes=1000):
    model = SHViT(num_classes=num_classes, distillation=distillation, frozen_stages=frozen_stages, pretrained=pretrained, **model_cfg)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        d = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(update_weight(model.state_dict(), d))
    if fuse:
        replace_batchnorm(model)
    return model


SHViT_s4_cfg = {
        'embed_dim': [224, 336, 448],
        'depth': [4, 7, 6],
        'partial_dim': [48, 72, 96],
        'types' : ["i", "s", "s"]
    }

def SHViT_S4(pretrained='', frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=SHViT_s4_cfg, num_classes=1000):
    model = SHViT(num_classes=num_classes, distillation=distillation, frozen_stages=frozen_stages, pretrained=pretrained, **model_cfg)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        d = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(update_weight(model.state_dict(), d))
    if fuse:
        replace_batchnorm(model)
    return model