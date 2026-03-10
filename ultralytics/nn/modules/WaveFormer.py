import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
import torch_dct  # 必須安裝: pip install torch-dct

# DropPath repr fix
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
    r""" Stem layer of InternImage """
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


class Wave2D(nn.Module):
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        self.c = nn.Parameter(torch.ones(1) * 1)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        
    def infer_init_wave2d(self, freq):
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
        x = x.permute(0, 3, 1, 2).contiguous()
        z = z.permute(0, 3, 1, 2).contiguous()

        weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (weight_cosn is not None) and (weight_cosn.device == x.device):
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
        else:
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        def dct2d(x):
            """2D DCT-II on last two dims (H, W)"""
            x = torch_dct.dct(x, norm='ortho')
            x = x.transpose(-1, -2) 
            x = torch_dct.dct(x, norm='ortho')
            x = x.transpose(-1, -2)
            return x

        def idct2d(x):
            """2D IDCT-II on last two dims (H, W)"""
            x = x.transpose(-1, -2)
            x = torch_dct.idct(x, norm='ortho')
            x = x.transpose(-1, -2)
            x = torch_dct.idct(x, norm='ortho')
            return x

        x_u0 = dct2d(x)
        x_v0 = dct2d(x)

        if freq_embed is not None:
            # --- 修正: 動態調整 freq_embed 大小以匹配輸入 x 的 (H, W) ---
            if freq_embed.shape[0] != H or freq_embed.shape[1] != W:
                # freq_embed shape: (H_param, W_param, C)
                # Permute to (1, C, H_param, W_param) for F.interpolate
                f_e = freq_embed.permute(2, 0, 1).unsqueeze(0)
                f_e = F.interpolate(f_e, size=(H, W), mode='bilinear', align_corners=True)
                # Permute back to (B, H, W, C)
                t_in = f_e.permute(0, 2, 3, 1).expand(B, -1, -1, -1)
            else:
                t_in = freq_embed.unsqueeze(0).expand(B, -1, -1, -1)
            
            t = self.to_k(t_in.contiguous())
        else:
            t = torch.zeros((B, H, W, C), device=x.device, dtype=x.dtype)
            
        cos_term = torch.cos(self.c * t).permute(0, 3, 1, 2).contiguous()
        sin_term = torch.sin(self.c * t).permute(0, 3, 1, 2).contiguous() / self.c

        wave_term = cos_term * x_u0
        velocity_term = sin_term * (x_v0 + (self.alpha / 2) * x_u0)
        final_term = wave_term + velocity_term

        x_final = idct2d(final_term)
        x = self.out_norm(x_final.permute(0, 2, 3, 1).contiguous())
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x * F.silu(z)
        x = self.out_linear(x.permute(0, 2, 3, 1).contiguous())
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class WaveBlock(nn.Module):
    def __init__(
        self, res: int = 14, infer_mode = False, hidden_dim: int = 0, drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False, drop: float = 0.0, act_layer: nn.Module = nn.GELU,
        mlp_ratio: float = 4.0, post_norm = True, layer_scale = None, **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.op = Wave2D(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)
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
            if isinstance(module, nn.Module):
                try:
                    x = module(x, *args, **kwargs)
                except TypeError:
                    x = module(x)
            else:
                x = module(x)
        return x


class WaveFormer(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], drop_path_rate=0.2, patch_norm=True, post_norm=True,
                 layer_scale=None, use_checkpoint=False, mlp_ratio=4.0, img_size=224,
                 act_layer='GELU', infer_mode=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.in_chans = in_chans
        self.img_size = img_size
        
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.depths = depths
        
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=self.embed_dim,
                                     act_layer='GELU',
                                     norm_layer='LN')
        
        res0 = img_size/patch_size
        self.res = [int(res0), int(res0//2), int(res0//4), int(res0//8)]
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.infer_mode = infer_mode
        
        self.freq_embed = nn.ParameterList()
        for i in range(self.num_layers):
            self.freq_embed.append(nn.Parameter(torch.zeros(self.res[i], self.res[i], self.dims[i]), requires_grad=True))
            trunc_normal_(self.freq_embed[i], std=.02)
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        cur = 0
        for i_layer in range(self.num_layers):
            # 1. Build Blocks for this stage
            blocks = []
            for d in range(depths[i_layer]):
                blocks.append(WaveBlock(
                    res=self.res[i_layer],
                    hidden_dim=self.dims[i_layer], 
                    drop_path=dpr[cur + d],
                    norm_layer=LayerNorm2d,
                    use_checkpoint=use_checkpoint,
                    mlp_ratio=mlp_ratio,
                    post_norm=post_norm,
                    layer_scale=layer_scale,
                    infer_mode=infer_mode,
                ))
            self.stages.append(AdditionalInputSequential(*blocks))
            cur += depths[i_layer]
            
            # 2. Build Downsample for next stage
            if i_layer < self.num_layers - 1:
                self.downsamples.append(self.make_downsample(
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=LayerNorm2d,
                ))
            else:
                self.downsamples.append(nn.Identity())
            
        self.classifier = nn.Sequential(
            LayerNorm2d(self.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.num_features, num_classes),
        )

        self.apply(self._init_weights)
        
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
            self.width_list = self.dims 
            self.train()

    @staticmethod
    def make_downsample(dim=96, out_dim=192, norm_layer=LayerNorm2d):
        return nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_dim)
        )
 
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def infer_init(self):
        for i, stage in enumerate(self.stages):
            for block in stage:
                block.op.infer_init_wave2d(self.freq_embed[i])
        del self.freq_embed
    
    def forward_features(self, x):
        outs = []
        x = self.patch_embed(x) 
        
        if self.infer_mode:
            for i in range(self.num_layers):
                x = self.stages[i](x)
                outs.append(x)
                if i < self.num_layers - 1:
                    x = self.downsamples[i](x)
        else:
            for i in range(self.num_layers):
                x = self.stages[i](x, self.freq_embed[i])
                outs.append(x)
                if i < self.num_layers - 1:
                    x = self.downsamples[i](x)
        
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x 

# --- Factory Functions ---

def waveformer_tiny(pretrained=False, img_size=224, **kwargs):
    model = WaveFormer(
        img_size=img_size,
        dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        drop_path_rate=0.1,
        post_norm=False,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

def waveformer_small(pretrained=False, img_size=224, **kwargs):
    model = WaveFormer(
        img_size=img_size,
        dims=[96, 192, 384, 768],
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3,
        post_norm=True,
        layer_scale=1.e-5,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

def waveformer_base(pretrained=False, img_size=224, **kwargs):
    model = WaveFormer(
        img_size=img_size,
        dims=[128, 256, 512, 1024], 
        depths=[2, 2, 18, 2],
        drop_path_rate=0.5,
        post_norm=True,
        layer_scale=1.e-5,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == "__main__":
    import torch
    img_h, img_w = 224, 224
    print("--- Creating WaveFormer Tiny model ---")
    try:
        model = waveformer_tiny(img_size=img_h)
        print("Model created successfully.")
        print("Calculated width_list:", model.width_list)

        input_tensor = torch.rand(2, 3, img_h, img_w)
        print(f"\n--- Testing WaveFormer Tiny forward pass (Input: {input_tensor.shape}) ---")

        model.eval()
        with torch.no_grad():
            output_features = model(input_tensor)
        
        print("Forward pass successful.")
        print(f"Output type: {type(output_features)}")
        if isinstance(output_features, (list, tuple)):
            print("Output feature shapes:")
            for i, features in enumerate(output_features):
                print(f"Stage {i+1}: {features.shape}")
        
        runtime_widths = [f.size(1) for f in output_features]
        print("\nRuntime output feature channels:", runtime_widths)
        assert model.width_list == runtime_widths, "Width list mismatch!"
        print("Width list verified successfully.")

        # Test with larger input (dynamic size test)
        print("\n--- Testing dynamic input size (512x512) ---")
        input_tensor_large = torch.rand(2, 3, 512, 512)
        with torch.no_grad():
             out_large = model(input_tensor_large)
        print("Dynamic size forward pass successful.")
        for i, f in enumerate(out_large):
            print(f"Stage {i+1} large shape: {f.shape}")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()