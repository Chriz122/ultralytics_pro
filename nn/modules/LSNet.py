import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import math

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite, DropPath
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import to_2tuple

# --- Start: Pytorch implementation for SKA ---
class SKA(nn.Module):
    def __init__(self):
        super(SKA, self).__init__()

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        b, c, h, w_dim = x.shape
        groups = c // w.shape[1]
        ks = int(math.sqrt(w.shape[2]))
        pad = (ks - 1) // 2
        
        x_unfold = F.unfold(x, kernel_size=ks, padding=pad)
        x_unfold = x_unfold.view(b, c, ks * ks, h * w_dim).permute(0, 1, 3, 2)

        w = w.view(b, c // groups, ks**2, h, w_dim).permute(0, 1, 3, 4, 2).reshape(b, c // groups, h * w_dim, ks**2)
        if groups > 1:
            w = w.repeat_interleave(groups, dim=1)

        out = (x_unfold * w).sum(-1)
        return out.view(b, c, h, w_dim)

# --- End: Pytorch implementation for SKA ---


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
        trunc_normal_(self.l.weight, std=std)
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
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

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

class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Conv2d_BN(dim, h, ks=1)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0))
        self.dw = Conv2d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)
        
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))
        self.init_resolution = resolution

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        
        if self.training:
            bias = self.attention_biases[:, self.attention_bias_idxs]
        else:
            if not hasattr(self, 'ab'):
                self.ab = self.attention_biases[:, self.attention_bias_idxs]
            bias = self.ab
            
        N_init = self.init_resolution * self.init_resolution
        if N != N_init:
            bias = bias.unsqueeze(0)
            bias = F.interpolate(bias, size=(N, N), mode='bicubic', align_corners=False)
            bias = bias.squeeze(0)

        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
        q = self.dw(q)
        
        q = q.view(B, self.num_heads, self.key_dim, N)
        k = k.view(B, self.num_heads, self.key_dim, N)
        v = v.view(B, self.num_heads, self.d, N)

        attn = (q.transpose(-2, -1) @ k) * self.scale + bias
        attn = attn.softmax(dim=-1)
        
        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
        x = self.proj(x)
        return x

class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
    
    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv

class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * (dim // groups), kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w

class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x

class Block(torch.nn.Module):    
    def __init__(self,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 stage=-1, depth=-1):
        super().__init__()
            
        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se = SqueezeExcite(ed, 0.25)
        else:
            self.se = torch.nn.Identity()
            if stage == 3:
                self.mixer = Residual(Attention(ed, kd, nh, ar, resolution=resolution))
            else:
                self.mixer = LSConv(ed)

        self.ffn = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))

class LSNet(torch.nn.Module):
    def __init__(self, img_size=224,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[64, 128, 192, 256],
                 key_dim=[16, 16, 16, 16],
                 depth=[1, 2, 3, 4],
                 num_heads=[4, 4, 4, 4],
                 distillation=False,
                 **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.distillation = distillation
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.img_size = img_size

        self.patch_embed = torch.nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 2, 3, 2, 1), 
            torch.nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1)
        )
        
        resolution = img_size // 4
        
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        
        self.stages = nn.ModuleList()
        current_dim = embed_dim[0]
        for i, (ed, kd, dpth, nh, ar) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio)):
            
            blocks = []
            if i > 0:
                blocks.append(Conv2d_BN(current_dim, current_dim, ks=3, stride=2, pad=1, groups=current_dim))
                blocks.append(Conv2d_BN(current_dim, ed, ks=1, stride=1, pad=0))
                resolution = (resolution + 1) // 2

            for d in range(dpth):
                blocks.append(Block(ed, kd, nh, ar, resolution, stage=i, depth=d))
            
            self.stages.append(nn.Sequential(*blocks))
            current_dim = ed

        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        self.width_list = []
        try:
            self.eval() 
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size)
            with torch.no_grad():
                 features = self.forward_features(dummy_input)
            self.width_list = [f.size(1) for f in features]
            self.train() 
        except Exception as e:
            print(f"Error during dummy forward pass for width_list calculation: {e}")
            print("Setting width_list to embed_dim as fallback.")
            self.width_list = self.embed_dim
            self.train()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward_features(self, x):
        x = self.patch_embed(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
        
    # *** FINAL FIX ***
    # This forward method now UNCONDITIONALLY returns a list of feature maps,
    # which is what detection frameworks like ultralytics expect.
    def forward(self, x):
        """
        Runs the forward pass of the model.
        Returns:
            list(torch.Tensor): A list of feature maps from each stage of the model.
        """
        return self.forward_features(x)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (4, 4),
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.1.c', 'classifier': ('head.l', 'head_dist.l'),
        **kwargs
    }

default_cfgs = dict(
    lsnet_t= _cfg(hf_hub='jameslahm/lsnet_t'),
    lsnet_t_distill= _cfg(hf_hub='jameslahm/lsnet_t_distill'),
    lsnet_s= _cfg(hf_hub='jameslahm/lsnet_s'),
    lsnet_s_distill= _cfg(hf_hub='jameslahm/lsnet_s_distill'),
    lsnet_b= _cfg(hf_hub='jameslahm/lsnet_b'),
    lsnet_b_distill= _cfg(hf_hub='jameslahm/lsnet_b_distill'),
)

def _create_lsnet(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', False):
        kwargs.pop('features_only')
    
    default_cfg = default_cfgs[variant]
    model = build_model_with_cfg(
        LSNet,
        variant,
        pretrained,
        default_cfg=default_cfg,
        **kwargs,
    )
    return model

@register_model
def LSNet_T(pretrained=False, **kwargs):
    variant = "lsnet_t_distill" if kwargs.get('distillation', False) else "lsnet_t"
    model_kwargs = dict(
        embed_dim=[64, 128, 256, 384],
        depth=[0, 2, 8, 10],
        key_dim=[16, 16, 16, 16],
        num_heads=[3, 3, 3, 4],
        **kwargs)
    return _create_lsnet(variant, pretrained, **model_kwargs)

@register_model
def LSNet_S(pretrained=False, **kwargs):
    variant = "lsnet_s_distill" if kwargs.get('distillation', False) else "lsnet_s"
    model_kwargs = dict(
        embed_dim=[96, 192, 320, 448],
        depth=[1, 2, 8, 10],
        key_dim=[16, 16, 16, 16],
        num_heads=[3, 3, 3, 4],
        **kwargs)
    return _create_lsnet(variant, pretrained, **model_kwargs)

@register_model
def LSNet_B(pretrained=False, **kwargs):
    variant = "lsnet_b_distill" if kwargs.get('distillation', False) else "lsnet_b"
    model_kwargs = dict(
        embed_dim=[128, 256, 384, 512],
        depth=[4, 6, 8, 10],
        key_dim=[16, 16, 16, 16],
        num_heads=[3, 3, 3, 4],
        **kwargs)
    return _create_lsnet(variant, pretrained, **model_kwargs)

if __name__ == '__main__':
    img_h, img_w = 224, 224 
    # num_classes can be anything, the forward method will ignore it for feature extraction
    model = LSNet_T(num_classes=80, img_size=img_h)
    model.eval()
    
    print(f"--- Creating LSNet-T model for feature extraction ---")
    print(f"Calculated width_list: {model.width_list}")

    input_tensor = torch.rand(2, 3, img_h, img_w)
    print(f"\n--- Testing LSNet-T forward pass (Input: {input_tensor.shape}) ---")

    try:
        with torch.no_grad():
            output_features = model(input_tensor)
        
        print("Forward pass successful.")
        assert isinstance(output_features, list), "Output must be a list!"
        print("Output is a list, as expected by detection frameworks.")
        
        print("Output feature shapes:")
        for i, features in enumerate(output_features):
            print(f"Stage {i+1}: {features.shape}")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()