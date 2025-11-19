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

# 保持原有的 DropPath 顯示方式
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

__all__ = ['vHeat_MoE', 'vHeat_MoE_t', 'vHeat_MoE_s', 'vHeat_MoE_b']


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
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

class PolicyNet(nn.Module):
    def __init__(self, in_dim):
        super(PolicyNet, self).__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 3))

    def forward(self, x, temp):
        logits = self.net(x)
        hard_mask = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        return hard_mask

class Heat2D(nn.Module):
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.policy = PolicyNet(hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
    
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

    def haar_transform_1d(self, x):
        # x shape: (..., N)
        avg = (x[..., ::2] + x[..., 1::2]) / 2 
        diff = (x[..., ::2] - x[..., 1::2]) / 2 
        return torch.cat((avg, diff), dim=-1)
    
    def haar_transform(self, x, dims=(1, 2)):
        # Robust Haar Transform handling arbitrary dimensions and padding
        transformed = x.clone()
        for dim in dims:
            # 1. Move target dim to last
            perm = list(range(transformed.ndim))
            perm.pop(dim)
            perm.append(dim)
            transformed = transformed.permute(perm)
            
            # 2. Handle Padding for odd dimensions safely
            if transformed.shape[-1] % 2 != 0:
                s = transformed.shape
                # Fix for RuntimeError: Only 2D/3D/4D/5D supported for non-constant pad
                # Flatten all preceding dims: (N, L)
                # Unsqueeze to (N, 1, L) -> This is a 3D tensor which F.pad supports well
                t_flat = transformed.reshape(-1, 1, s[-1]) 
                t_padded = F.pad(t_flat, (0, 1), "replicate")
                # Reshape back to original structure + 1 padding
                transformed = t_padded.reshape(s[:-1] + (s[-1] + 1,))
            
            # 3. Transform
            transformed = self.haar_transform_1d(transformed)
            
            # 4. Move dim back
            inv_perm = [0] * transformed.ndim
            for i, p in enumerate(perm):
                inv_perm[p] = i
            transformed = transformed.permute(inv_perm)
            
        return transformed
    
    def inverse_haar_transform_1d(self, x):
        n = x.shape[-1]
        half = n // 2
        avg = x[..., :half]
        diff = x[..., half:]
        
        x_rec = torch.zeros_like(x)
        x_rec[..., ::2] = avg + diff 
        x_rec[..., 1::2] = avg - diff 
        return x_rec
    
    def inverse_haar_transform(self, x, dims=(1, 2)):
        transformed = x.clone()
        for dim in dims:
            # Move target dim to last
            perm = list(range(transformed.ndim))
            perm.pop(dim)
            perm.append(dim)
            transformed = transformed.permute(perm)
            
            # Inverse Transform
            transformed = self.inverse_haar_transform_1d(transformed)
            
            # Move dim back
            inv_perm = [0] * transformed.ndim
            for i, p in enumerate(perm):
                inv_perm[p] = i
            transformed = transformed.permute(inv_perm)
            
        return transformed

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, H, W = x.shape
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous()) 
        x, z = x.chunk(chunks=2, dim=-1) 

        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
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

        N_cos, M_cos = weight_cosn.shape[0], weight_cosm.shape[0]
        
        moe = self.policy(x.mean(dim=(1,2)), 1)

        # DCT
        x_dct = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N_cos, H, 1))
        x_dct = F.conv1d(x_dct.contiguous().view(-1, W, x.shape[-1]), weight_cosm.contiguous().view(M_cos, W, 1)).contiguous().view(B, N_cos, M_cos, -1)

        # FFT
        x_fft = torch.fft.fftn(x, dim=(1,2)) 

        # Haar
        x_haar = self.haar_transform(x, dims=(1, 2))
        
        haar_H, haar_W = x_haar.shape[1], x_haar.shape[2]
        
        if self.infer_mode:
            k_exp_used = self.k_exp
            x_dct = torch.einsum("bnmc,nmc->bnmc", x_dct, k_exp_used)
            x_fft = torch.einsum("bnmc,nmc->bnmc", x_fft, k_exp_used)
            x_haar = torch.einsum("bnmc,nmc->bnmc", x_haar, k_exp_used) 
        else:
            curr_freq_embed = freq_embed
            if freq_embed.shape[0] != H or freq_embed.shape[1] != W:
                 curr_freq_embed = F.interpolate(freq_embed.permute(2,0,1).unsqueeze(0), size=(H, W), mode='bilinear').squeeze(0).permute(1,2,0)

            weight_exp_vals = torch.pow(weight_exp[:, :, None], self.to_k(curr_freq_embed))

            x_dct = torch.einsum("bnmc,nmc -> bnmc", x_dct, weight_exp_vals)
            x_fft = torch.einsum("bnmc,nmc -> bnmc", x_fft, weight_exp_vals)
            
            # Handle Haar Padding logic for weight application
            if haar_H != H or haar_W != W:
                pad_h = haar_H - H
                pad_w = haar_W - W
                w_exp_padded = F.pad(weight_exp_vals.permute(2, 0, 1), (0, pad_w, 0, pad_h), "replicate").permute(1, 2, 0)
                x_haar = torch.einsum("bnmc,nmc -> bnmc", x_haar, w_exp_padded)
            else:
                x_haar = torch.einsum("bnmc,nmc -> bnmc", x_haar, weight_exp_vals)

        x_dct = F.conv1d(x_dct.contiguous().view(B, N_cos, -1), weight_cosn.t().contiguous().view(H, N_cos, 1))
        x_dct = F.conv1d(x_dct.contiguous().view(-1, M_cos, x.shape[-1]), weight_cosm.t().contiguous().view(W, M_cos, 1)).contiguous().view(B, H, W, -1)

        x_fft = torch.fft.ifftn(x_fft, dim=(1,2)).real

        x_haar = self.inverse_haar_transform(x_haar, dims=(1, 2))
        
        # Crop Haar result if it was padded
        if x_haar.shape[1] != H or x_haar.shape[2] != W:
            x_haar = x_haar[:, :H, :W, :]

        x_combined = torch.cat((x_dct.unsqueeze(dim=1), x_fft.unsqueeze(dim=1), x_haar.unsqueeze(dim=1)), dim=1)
        x_out = torch.einsum("brnmc,br -> bnmc", x_combined, moe)

        x_out = self.out_norm(x_out)
        x_out = x_out * nn.functional.silu(z)
        x_out = self.out_linear(x_out)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()

        return x_out


class HeatBlock(nn.Module):
    def __init__(self, res=14, infer_mode=False, hidden_dim=0, drop_path=0., 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), use_checkpoint=False, 
                 drop=0.0, act_layer=nn.GELU, mlp_ratio=4.0, post_norm=True, 
                 layer_scale=None, **kwargs):
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
            if isinstance(module, HeatBlock):
                x = module(x, *args, **kwargs)
            elif isinstance(module, nn.Module): 
                x = module(x)
        return x
    
class StemLayer(nn.Module):
    def __init__(self, in_chans=3, out_chans=96, act_layer='GELU', norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans // 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = LayerNorm2d(out_chans // 2) 
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_chans // 2, out_chans, kernel_size=3, stride=2, padding=1)
        self.norm2 = LayerNorm2d(out_chans)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class vHeat_MoE(nn.Module):
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
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=self.embed_dim,
                                     act_layer=act_layer,
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

        for i_layer in range(self.num_layers):
            blocks = self.make_blocks(
                res=self.res[i_layer],
                dim=self.dims[i_layer],
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=LayerNorm2d,
                post_norm=post_norm,
                layer_scale=layer_scale,
                mlp_ratio=mlp_ratio,
                infer_mode=infer_mode
            )
            self.stages.append(blocks)

            if i_layer < self.num_layers - 1:
                ds = self.make_downsample(
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=LayerNorm2d
                )
                self.downsamples.append(ds)
            else:
                self.downsamples.append(nn.Identity())
            
        self.classifier = nn.Sequential(
            LayerNorm2d(self.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.num_features, num_classes),
        ) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        self.width_list = []
        try:
            self.eval() 
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size)
            with torch.no_grad():
                 features = self.forward_features(dummy_input)
            self.width_list = [f.shape[1] for f in features]
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

    @staticmethod
    def make_blocks(res=14, dim=96, depth=2, drop_path=[], use_checkpoint=False, 
                    norm_layer=LayerNorm2d, post_norm=True, layer_scale=None, 
                    mlp_ratio=4.0, infer_mode=False):
        blocks = []
        for d in range(depth):
            blocks.append(HeatBlock(
                res=res,
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                post_norm=post_norm,
                layer_scale=layer_scale,
                infer_mode=infer_mode,
            ))
        return AdditionalInputSequential(*blocks)
 
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
             pass 
    
    def infer_init(self):
        for i, stage in enumerate(self.stages):
            for block in stage:
                block.op.infer_init_heat2d(self.freq_embed[i])
        del self.freq_embed
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        outs = []

        for i in range(self.num_layers):
            if self.infer_mode:
                x = self.stages[i](x) 
            else:
                x = self.stages[i](x, self.freq_embed[i])
            
            outs.append(x)
            x = self.downsamples[i](x)
        
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


# --- Factory Functions ---

def vHeat_MoE_t(pretrained=False, img_size=224, **kwargs):
    model = vHeat_MoE(
        img_size=img_size,
        drop_path_rate=0.1,
        dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

def vHeat_MoE_s(pretrained=False, img_size=224, **kwargs):
    model = vHeat_MoE(
        img_size=img_size,
        drop_path_rate=0.3,
        dims=[96, 192, 384, 768],
        depths=[2, 2, 18, 2], 
        layer_scale=1e-5,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

def vHeat_MoE_b(pretrained=False, img_size=224, **kwargs):
    model = vHeat_MoE(
        img_size=img_size,
        drop_path_rate=0.5,
        dims=[128, 256, 512, 1024],
        depths=[2, 2, 18, 2], 
        layer_scale=1e-5,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == "__main__":
    from fvcore.nn import flop_count_table, flop_count_str, FlopCountAnalysis
    
    img_size = 640
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Creating vHeat_MoE_s model (img_size={img_size})...")
    model = vHeat_MoE_t(img_size=img_size).to(device)
    
    print("Calculated width_list:", model.width_list)
    
    input_tensor = torch.randn((2, 3, img_size, img_size), device=device)
    
    # Check output format
    outs = model(input_tensor)
    print(f"Output type: {type(outs)}")
    if isinstance(outs, list):
        print(f"Number of feature maps: {len(outs)}")
        for i, o in enumerate(outs):
            print(f"Feature {i} shape: {o.shape}")
    else:
        print(f"Output shape: {outs.shape}")

    # FLOPs analysis
    try:
        analyze = FlopCountAnalysis(model, (input_tensor,))
        print(flop_count_str(analyze))
    except Exception as e:
        print(f"FLOPs analysis skipped due to: {e}")