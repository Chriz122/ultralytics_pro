# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------
#
# Modified by Gemini for integration with ultralytics framework.
# Key changes:
# 1. forward_features now returns a list of feature maps from each stage.
# 2. Added self.width_list initialization via a dummy forward pass, similar to SMT.
# 3. Refactored model creation to use factory functions (e.g., uninext_t).
# 4. The main forward method now directly returns features for detection/segmentation.
# ------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
from einops import rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'DilatedFormer_224': _cfg(),
    'DilatedFormer_384': _cfg(
        crop_pct=1.0
    ),

}


def local_group(x, H, W, ws, ds):
    '''  
    ws: window size 
    ds: dilated size
    x: BNC
    return: BG ds*ds ws*ws C
    '''
    B, _, C = x.shape
    pad_right, pad_bottom, pad_opt, pad_opt_d = 0, 0, False, False

    if H % ws != 0 or W % ws != 0:
        pad_opt =True
        x = x.view(B, H, W, C)
        pad_right = ws - W % ws
        pad_bottom = ws - H % ws
        x = F.pad(x, (0, 0, 0, pad_right, 0, pad_bottom))
        H = H + pad_bottom
        W = W + pad_right
        N = H * W
        x = x.view(B, N, C)
    Gh = H//ws
    Gw = W//ws
    x = x.view(B, Gh, ws, Gw, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B*Gh*Gw, ws*ws, C)
    
    return x, H, W, pad_right, pad_bottom, pad_opt



def img2group(x, H, W, ws, ds, num_head):
    '''
    x: B, H*W, C
    return : (B G) head  N C
    '''
    x, H, W, pad_right, pad_bottom, pad_opt = local_group(x, H, W, ws, ds)
    B, N, C =x.shape
    x = x.view(B, N, num_head, C//num_head).permute(0, 2, 1, 3).contiguous()

    return x, H, W, pad_right, pad_bottom, pad_opt


def group2image(x, H, W, pad_right, pad_bottom, pad_opt, ws):
    # Input x: (BG G) Head n C
    # Output x: B N C
    BG, Head, n, C = x.shape
    Gh, Gw = H//ws, W//ws
    Gn = Gh * Gw
    nb1 = BG//Gn
    x = x.view(nb1, Gh, Gw, Head, ws, ws, C).permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(nb1, -1, Head*C)
    
    if pad_opt:
        x = x.view(nb1, H, W, Head*C)
        x = x[:, :(H - pad_bottom), :(W - pad_right), :].contiguous()
        x = x.view(nb1, -1, Head*C)

    return x


# dwconv MLP
class Mlp(nn.Module):
    def __init__(self,
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.norm_act = nn.Sequential(
            nn.LayerNorm(hidden_features),
            nn.GELU()
        )

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        B, N, C = x.shape
        x1 = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x1 = self.dwconv(x1)
        x1 = x1.view(B, C, -1).permute(0, 2, 1).contiguous()
        x1 = self.norm_act(x1)
        x = x + x1
        x = self.fc2(x)
        x = self.drop(x)
        return x

  
class DilatedAttention(nn.Module):
    def __init__(self,
                 dim,
                 ws=7,
                 ds=7,
                 num_heads=8,
                 attn_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.ws = ws
        self.ds = ds
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.lepe = nn.Conv2d(in_channels=dim, 
                              out_channels=dim, 
                              kernel_size=3,
                              stride=1, 
                              padding=1, 
                              groups=dim, 
                              bias=True)

    def forward(self, qkv, H, W):
        q, k, v = qkv[0], qkv[1], qkv[2]
        # lepe
        B, _, vc = v.shape
        lepe = v.permute(0, 2, 1).contiguous().view(B, vc, H, W)
        lepe = self.lepe(lepe)
        lepe = lepe.view(B, vc, -1).permute(0, 2, 1).contiguous()

        N = q.size(1)
        assert N == H * W, "flatten img_tokens has wrong size"

        q, H_new, W_new, pad_right, pad_bottom, pad_opt = img2group(q, H, W, self.ws, self.ds, self.num_heads)
        k, _, _, _, _, _ = img2group(k, H, W, self.ws, self.ds, self.num_heads)
        v, _, _, _, _, _ = img2group(v, H, W, self.ws, self.ds, self.num_heads)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = group2image(x, H_new, W_new, pad_right, pad_bottom, pad_opt, self.ws)
        x = x + lepe
        return x


class DilatedBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ws=7,
                 ds=7,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 ):
        super().__init__()
        self.mlp_ratio = mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.attns = nn.ModuleList()
        self.attns.append(
            DilatedAttention(dim,
                             ws=ws,
                             ds=ds,
                             num_heads=num_heads,
                             attn_drop=attn_drop,
                             qk_scale=qk_scale)
        )
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))

        qkv = F.linear(input=img, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        attened_x = self.attns[0](qkv, H, W)

        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C_out = x.shape[:2]
        x = x.view(B, C_out, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x

class DilatedFormer_Windows(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=96, 
                 depth=[2,2,6,2], 
                 ws = [7,7,7,7], 
                 wd=[7,7,7,7],
                 num_heads=[3,6,12,24], 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 use_chk=False):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.img_size = img_size
        self.num_stages = len(depth)

        #------------- stem -----------------------
        stem_out = embed_dim//2
        self.stem1 = nn.Conv2d(in_chans, stem_out, 3, 2, 1)
        self.norm_act1 = nn.Sequential(
            nn.LayerNorm(stem_out),
            nn.GELU()
        )
        self.stem2 = nn.Conv2d(stem_out, stem_out, 3, 1, 1)
        self.norm_act2 = nn.Sequential(
            nn.LayerNorm(stem_out),
            nn.GELU()
        )
        self.stem3 = nn.Conv2d(stem_out, stem_out, 3, 1, 1)
        self.norm_act3 = nn.Sequential(
            nn.LayerNorm(stem_out),
            nn.GELU()
        )
        
        self.merge0 = Merge_Block(stem_out, embed_dim)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]
        
        # Build stages
        curr_dim = embed_dim
        for i in range(self.num_stages):
            stage = nn.ModuleList([
                DilatedBlock(
                    dim=curr_dim, 
                    num_heads=num_heads[i], 
                    ws=ws[i], 
                    ds=wd[i], 
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, 
                    qk_scale=qk_scale,
                    drop=drop_rate, 
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depth[:i]) + j])
                for j in range(depth[i])])
            
            cpe = nn.ModuleList([
                nn.Conv2d(curr_dim, curr_dim, 3, 1, 1, groups=curr_dim)
                for _ in range(depth[i])])

            setattr(self, f"stage{i + 1}", stage)
            setattr(self, f"cpe{i + 1}", cpe)

            if i < self.num_stages - 1:
                merge_block = Merge_Block(curr_dim, curr_dim*2)
                setattr(self, f"merge{i + 1}", merge_block)
                curr_dim *= 2

        self.norm = norm_layer(curr_dim)
        self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)

        # --- Add width_list calculation ---
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
            print("Setting width_list to embed_dims as fallback.")
            # Fallback based on expected dimensions
            dims = [embed_dim]
            for i in range(self.num_stages - 1):
                dims.append(dims[-1] * 2)
            self.width_list = dims
            self.train()
        # --------------------------------

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes, global_pool=''):
        if self.num_classes != num_classes:
            print ('reset head to', num_classes)
            self.num_classes = num_classes
            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head = self.head.cuda()
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x):
        B, _, H_img, W_img = x.shape
        
        # Calculate spatial dimensions for each stage
        # The first downsampling is by 2 (stem), the next is by 2 (merge0), then 3 more by 2.
        # So strides are 2, 4, 8, 16, 32.
        # We need features after stage 1, 2, 3, 4. Their strides will be 4, 8, 16, 32.
        spatial_dims = [(H_img // (2**(i+2)), W_img // (2**(i+2))) for i in range(self.num_stages)]
        
        feature_outputs = []
        
        # Stem
        x = self.stem1(x)
        c1 = x.size(1)
        x = x.view(B, c1, -1).permute(0, 2, 1).contiguous()
        x = self.norm_act1(x)
        x = x.permute(0, 2, 1).contiguous().view(B, c1, H_img // 2, W_img // 2)
        
        x = self.stem2(x)
        c2 = x.size(1)
        x = x.view(B, c2, -1).permute(0, 2, 1).contiguous()
        x = self.norm_act2(x)
        x = x.permute(0, 2, 1).contiguous().view(B, c2, H_img // 2, W_img // 2)

        x = self.stem3(x)
        c3 = x.size(1)
        x = x.view(B, c3, -1).permute(0, 2, 1).contiguous()
        x = self.norm_act3(x)

        x = self.merge0(x)
        
        # Main stages
        for i in range(self.num_stages):
            stage = getattr(self, f"stage{i+1}")
            cpe_layers = getattr(self, f"cpe{i+1}")
            H, W = spatial_dims[i]
            C = x.shape[2]
            
            for blk, cpe in zip(stage, cpe_layers):
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x, H, W)
                    pe = cpe(x.transpose(1, 2).reshape(B, C, H, W))
                    pe = pe.view(B, C, -1).transpose(1, 2)
                    x = x + pe
                else:
                    x = blk(x, H, W)
                    pe = cpe(x.transpose(1, 2).reshape(B, C, H, W))
                    pe = pe.view(B, C, -1).transpose(1, 2)
                    x = x + pe
            
            # Reshape to spatial format (B, C, H, W) and store
            feature_outputs.append(x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous())

            if i < self.num_stages - 1:
                merge = getattr(self, f"merge{i+1}")
                x = merge(x)
        
        return feature_outputs

    def forward(self, x):
        x = self.forward_features(x)
        return x


# --- Factory Functions ---

@register_model
def uninext_t(pretrained=False, **kwargs):
    model = DilatedFormer_Windows(
        embed_dim=64, 
        depth=[2, 2, 18, 2],
        ws=[7, 7, 7, 7], 
        wd=[3, 3, 3, 3], 
        num_heads=[2, 4, 8, 16], 
        mlp_ratio=4., 
        **kwargs)
    model.default_cfg = default_cfgs['DilatedFormer_224']
    return model

@register_model
def uninext_s(pretrained=False, **kwargs):
    model = DilatedFormer_Windows(
        embed_dim=96, 
        depth=[2, 2, 18, 2],
        ws=[7, 7, 7, 7], 
        wd=[3, 3, 3, 3], 
        num_heads=[3, 6, 12, 24], 
        mlp_ratio=4., 
        **kwargs)
    model.default_cfg = default_cfgs['DilatedFormer_224']
    return model

@register_model
def uninext_b(pretrained=False, **kwargs):
    model = DilatedFormer_Windows(
        embed_dim=128, 
        depth=[2, 2, 18, 2],
        ws=[7, 7, 7, 7], 
        wd=[3, 3, 3, 3], 
        num_heads=[4, 8, 16, 32], 
        mlp_ratio=4., 
        **kwargs)
    model.default_cfg = default_cfgs['DilatedFormer_224']
    return model