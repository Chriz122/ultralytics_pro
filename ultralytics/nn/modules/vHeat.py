import time
import math
from functools import partial
from typing import Optional, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# --- 以下為輔助類和函數，保持不變 ---

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()

class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.permute(0, 2, 3, 1).contiguous()

def build_norm_layer(dim, norm_layer, in_format='channels_last', out_format='channels_last', eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)

def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'build_act_layer does not support {act_layer}')

class StemLayer(nn.Module):
    def __init__(self, in_chans=3, out_chans=96, act_layer='GELU', norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans // 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer, 'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2, out_chans, kernel_size=3, stride=2, padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first', 'channels_first')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Heat2D(nn.Module):
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=True), nn.ReLU())

    def infer_init_heat2d(self, freq):
        weight_exp = self.get_decay_map((self.res, self.res), device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False)
        del self.to_k

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, H, W = x.shape
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())
        x, z = x.chunk(chunks=2, dim=-1)

        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None) is not None and getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]
        x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
        x = F.conv1d(x.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)

        if self.infer_mode:
            x = torch.einsum("bnmc,nmc->bnmc", x, self.k_exp)
        else:
            # This is where the error occurred. freq_embed must have the same H, W as weight_exp.
            weight_exp = torch.pow(weight_exp[:, :, None], self.to_k(freq_embed))
            x = torch.einsum("bnmc,nmc -> bnmc", x, weight_exp)

        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1))
        x = F.conv1d(x.contiguous().view(-1, M, C), weight_cosm.t().contiguous().view(W, M, 1)).contiguous().view(B, H, W, -1)

        x = self.out_norm(x)
        x = x * nn.functional.silu(z)
        x = self.out_linear(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class HeatBlock(nn.Module):
    def __init__(self, res: int = 14, infer_mode = False, hidden_dim: int = 0, drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 use_checkpoint: bool = False, drop: float = 0.0, act_layer: nn.Module = nn.GELU,
                 mlp_ratio: float = 4.0, post_norm = True, layer_scale = None, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.op = Heat2D(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=True)
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None
        self.infer_mode = infer_mode
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim), requires_grad=True)

    def _forward(self, x: torch.Tensor, freq_embed):
        if not self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.op(x, freq_embed)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.op(self.norm1(x), freq_embed))
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        if self.post_norm:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.norm1(self.op(x, freq_embed)))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.op(self.norm1(x), freq_embed))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor, freq_embed=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, freq_embed)
        else:
            return self._forward(input, freq_embed)

class AdditionalInputSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self:
            x = module(x, *args, **kwargs)
        return x

class vHeat(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768], drop_path_rate=0.2, patch_norm=True, post_norm=True,
                 layer_scale=None, use_checkpoint=False, mlp_ratio=4.0, img_size=224,
                 act_layer='GELU', infer_mode=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)

        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]

        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.depths = depths

        self.patch_embed = StemLayer(in_chans=in_chans, out_chans=self.embed_dim,
                                     act_layer='GELU', norm_layer='LN')

        res0 = img_size / patch_size
        self.res = [int(res0), int(res0//2), int(res0//4), int(res0//8)]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.infer_mode = infer_mode

        self.freq_embed = nn.ParameterList()
        for i in range(self.num_layers):
            self.freq_embed.append(nn.Parameter(torch.zeros(self.res[i], self.res[i], self.dims[i]), requires_grad=True))
            trunc_normal_(self.freq_embed[i], std=.02)

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i_layer in range(self.num_layers):
            stage = self.make_stage(
                res=self.res[i_layer],
                dim=self.dims[i_layer],
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=LayerNorm2d,
                post_norm=post_norm,
                layer_scale=layer_scale,
                mlp_ratio=mlp_ratio,
                infer_mode=infer_mode,
            )
            self.stages.append(stage)

            if i_layer < self.num_layers - 1:
                downsampler = self.make_downsample(
                    self.dims[i_layer],
                    self.dims[i_layer + 1],
                    norm_layer=LayerNorm2d,
                )
            else:
                downsampler = nn.Identity()
            self.downsamples.append(downsampler)

        self.classifier = nn.Sequential(
            LayerNorm2d(self.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.num_features, num_classes),
        )

        self.apply(self._init_weights)

        self.eval()
        with torch.no_grad():
            # Use a size that might be different from initialization, to test robustness
            dummy_input = torch.randn(1, in_chans, img_size, img_size)
            features = self.forward(dummy_input)
            self.width_list = [f.shape[1] for f in features]
        self.train()

    @staticmethod
    def make_downsample(dim=96, out_dim=192, norm_layer=LayerNorm2d):
        return nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_dim)
        )

    @staticmethod
    def make_stage(res=14, dim=96, depth=2, drop_path=[0.1, 0.1], use_checkpoint=False,
                   norm_layer=LayerNorm2d, post_norm=True, layer_scale=None,
                   mlp_ratio=4.0, infer_mode=False, **kwargs):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(HeatBlock(
                res=res, hidden_dim=dim, drop_path=drop_path[d], norm_layer=norm_layer,
                use_checkpoint=use_checkpoint, mlp_ratio=mlp_ratio, post_norm=post_norm,
                layer_scale=layer_scale, infer_mode=infer_mode,
            ))
        return AdditionalInputSequential(*blocks)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def infer_init(self):
        for i, stage in enumerate(self.stages):
            for block in stage:
                block.op.infer_init_heat2d(self.freq_embed[i])
        del self.freq_embed

    # --- [修正] 在 forward 方法中加入動態插值 ---
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        outputs = []

        for i in range(self.num_layers):
            if self.infer_mode:
                # 在推斷模式下，freq_embed 已被烘焙，不需要傳遞
                x = self.stages[i](x)
            else:
                B, C, H, W = x.shape
                freq_embed_param = self.freq_embed[i]

                # [核心修正] 檢查空間尺寸是否匹配，如果不匹配則進行雙線性插值
                if freq_embed_param.shape[0] != H or freq_embed_param.shape[1] != W:
                    # 將 (H_orig, W_orig, C) -> (1, C, H_orig, W_orig) 以便插值
                    freq_embed_resized = freq_embed_param.permute(2, 0, 1).unsqueeze(0)
                    # 進行插值
                    freq_embed_resized = F.interpolate(
                        freq_embed_resized, size=(H, W), mode='bilinear', align_corners=False
                    )
                    # 將 (1, C, H, W) -> (H, W, C) 以匹配 Heat2D 的期望輸入
                    freq_embed_to_pass = freq_embed_resized.squeeze(0).permute(1, 2, 0)
                else:
                    # 如果尺寸匹配，直接使用
                    freq_embed_to_pass = freq_embed_param

                x = self.stages[i](x, freq_embed_to_pass)

            outputs.append(x)
            x = self.downsamples[i](x)

        return outputs


    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward(x)
        last_feature = features[-1]
        return self.classifier(last_feature)


# --- 工廠函數保持不變 ---

def vheat_tiny(pretrained=False, **kwargs):
    model = vHeat(
        depths=[2, 2, 6, 2],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1,
        post_norm=False,
        **kwargs)
    return model

def vheat_small(pretrained=False, **kwargs):
    model = vHeat(
        depths=[2, 2, 18, 2],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.3,
        layer_scale=1.e-5,
        post_norm=True,
        **kwargs)
    return model

def vheat_base(pretrained=False, **kwargs):
    model = vHeat(
        depths=[2, 2, 18, 2],
        dims=[128, 256, 512, 1024],
        drop_path_rate=0.5,
        layer_scale=1.e-5,
        post_norm=True,
        **kwargs)
    return model

# --- 測試代碼 ---
if __name__ == "__main__":
    print("--- Testing vHeat_tiny with different input sizes ---")
    model_tiny = vheat_tiny(img_size=224).cpu() # Initialize with 224
    model_tiny.eval()
    
    print("\n[Test 1] Input size 224x224 (matches initialization)")
    input_224 = torch.randn((1, 3, 224, 224))
    try:
        features_224 = model_tiny(input_224)
        print("Success! Output shapes:")
        for i, fm in enumerate(features_224):
            print(f"  Stage {i+1}: {fm.shape}")
        print(f"Model's width_list: {model_tiny.width_list}")
    except Exception as e:
        print(f"Failed: {e}")

    print("\n[Test 2] Input size 320x320 (different from initialization)")
    input_320 = torch.randn((1, 3, 320, 320))
    try:
        features_320 = model_tiny(input_320)
        print("Success! Output shapes:")
        for i, fm in enumerate(features_320):
            print(f"  Stage {i+1}: {fm.shape}")
    except Exception as e:
        print(f"Failed: {e}")