# -*- coding: utf-8 -*-
"""
DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation
(Modified Version: RGB-only, No mmengine, MobileNetV4-like usage, List Output)

Original Author: yinbow
Original Email: bowenyin@mail.nankai.edu.cn
Original Code: https://github.com/VCIP-RGBD/DFormer

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from typing import Tuple


# --- Keep all other classes (LayerNorm2d, PatchEmbed, DWConv2d, PatchMerging, etc.) as they were in the previous correct version ---
# ... (Paste all the helper classes here) ...

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x

class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.reduction(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        return x

def angle_transform(x, sin, cos):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    sin = sin.unsqueeze(0).unsqueeze(1)
    cos = cos.unsqueeze(0).unsqueeze(1)
    rotated_pair = torch.stack([-x2, x1], dim=-1)
    rotated_flat = rotated_pair.flatten(start_dim=-2)
    return (x * cos) + (rotated_flat * sin)

class PosEncodingGen(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_dim = embed_dim // num_heads
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
        angle = 1.0 / (10000 ** torch.linspace(0, 1, head_dim // 2, dtype=torch.float))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer("angle", angle, persistent=False)
    def forward(self, HW_tuple: Tuple[int]):
        H, W = HW_tuple
        index = torch.arange(H * W, device=self.angle.device)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        cos = torch.cos(index[:, None] * self.angle[None, :])
        head_dim = self.angle.shape[0]
        sin = sin.reshape(H, W, head_dim)
        cos = cos.reshape(H, W, head_dim)
        return (sin, cos)

class Decomposed_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = self.embed_dim // num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        assert self.key_dim % 2 == 0, "Key dimension must be even for RoPE in Decomposed_GSA"
        self.scaling = self.key_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim * self.factor, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos_sincos: Tuple[torch.Tensor, torch.Tensor]):
        bsz, h, w, embed_dim_in = x.size()
        (sin, cos) = rel_pos_sincos
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)
        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)
        v = v.view(bsz, h, w, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        qr_w = qr.permute(0, 2, 1, 3, 4)
        kr_w = kr.permute(0, 2, 1, 3, 4)
        v_w = v.permute(0, 2, 1, 3, 4)
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = torch.softmax(qk_mat_w, dim=-1)
        v_w_out = torch.matmul(qk_mat_w, v_w)
        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v_h = v_w_out.permute(0, 3, 2, 1, 4)
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = torch.softmax(qk_mat_h, dim=-1)
        output = torch.matmul(qk_mat_h, v_h)
        output = output.permute(0, 2, 3, 1, 4)
        output = output.contiguous().view(bsz, h, w, self.num_heads * self.head_dim)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        if self.q_proj.bias is not None: nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        if self.k_proj.bias is not None: nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        if self.v_proj.bias is not None: nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.xavier_normal_(self.out_proj.weight)
        if self.out_proj.bias is not None: nn.init.constant_(self.out_proj.bias, 0.0)

class Full_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = self.embed_dim // num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        assert self.key_dim % 2 == 0, "Key dimension must be even for RoPE in Full_GSA"
        self.scaling = self.key_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim * self.factor, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos_sincos: Tuple[torch.Tensor, torch.Tensor]):
        bsz, h, w, _ = x.size()
        (sin, cos) = rel_pos_sincos
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)
        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)
        v = v.view(bsz, h, w, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        qr = qr.flatten(2, 3)
        kr = kr.flatten(2, 3)
        vr = v.flatten(2, 3)
        qk_mat = qr @ kr.transpose(-1, -2)
        qk_mat = torch.softmax(qk_mat, dim=-1)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2).reshape(bsz, h, w, self.num_heads * self.head_dim)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        if self.q_proj.bias is not None: nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        if self.k_proj.bias is not None: nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        if self.v_proj.bias is not None: nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.xavier_normal_(self.out_proj.weight)
        if self.out_proj.bias is not None: nn.init.constant_(self.out_proj.bias, 0.0)

class FeedForwardNetwork(nn.Module):
    def __init__(
        self, embed_dim, ffn_dim, activation_fn=F.gelu, dropout=0.0,
        activation_dropout=0.0, layernorm_eps=1e-6, subln=False, subconv=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None
    def reset_parameters(self):
        pass
    def forward(self, x: torch.Tensor):
        residual_ffn = x
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual_conv = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        if self.dwconv is not None or self.ffn_layernorm is not None:
             x = x + residual_conv
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x

class VisionBlock(nn.Module):
    def __init__(
        self, split_or_not: bool, embed_dim: int, num_heads: int, ffn_dim: int,
        drop_path=0.0, layerscale=False, layer_init_values=1e-5,
    ):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        if split_or_not:
            self.Attention = Decomposed_GSA(embed_dim, num_heads)
        else:
            self.Attention = Full_GSA(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.cnn_pos_encode = DWConv2d(embed_dim, 3, 1, 1)
        self.PosEnc = PosEncodingGen(embed_dim, num_heads)
        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
        else:
            self.gamma_1 = 1.0
            self.gamma_2 = 1.0
    def forward(self, x: torch.Tensor):
        b, h, w, d = x.size()
        x = x + self.cnn_pos_encode(x)
        shortcut = x
        pos_enc_sincos = self.PosEnc((h, w))
        x = self.layer_norm1(x)
        x = self.Attention(x, pos_enc_sincos)
        x = shortcut + self.drop_path(self.gamma_1 * x)
        shortcut = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = shortcut + self.drop_path(self.gamma_2 * x)
        return x

class BasicLayer(nn.Module):
    def __init__(
        self, embed_dim, out_dim, depth, num_heads, ffn_dim, drop_path=0.0,
        split_or_not=False, downsample: PatchMerging = None, use_checkpoint=False,
        layerscale=False, layer_init_values=1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                VisionBlock(
                    split_or_not=split_or_not, embed_dim=embed_dim, num_heads=num_heads,
                    ffn_dim=ffn_dim, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layerscale=layerscale, layer_init_values=layer_init_values,
                ) for i in range(depth)
            ]
        )
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim)
        else:
            self.downsample = None
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x_out = x
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x_out, x_down
        else:
            return x_out, x


class dformerv2(nn.Module):
    def __init__(
        self,
        out_indices=(0, 1, 2, 3),
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[4, 4, 8, 16],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.1,
        patch_norm=True,
        use_checkpoint=False,
        layerscales=[False, False, False, False],
        layer_init_values=1e-6,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=embed_dims[0])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        current_dim = embed_dims[0]
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=current_dim,
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * current_dim),
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                split_or_not=(i_layer != (self.num_layers - 1)),
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                current_dim = embed_dims[i_layer + 1]
        self.output_norms = nn.ModuleList()
        for i in self.out_indices:
             self.output_norms.append(LayerNorm2d(embed_dims[i]))
        self.apply(self._init_weights)
        self.eval()
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 256, 256)
                features = self.forward(dummy_input)
                # Ensure features is a list/tuple before accessing size
                if isinstance(features, (list, tuple)):
                    self.width_list = [f.size(1) for f in features if torch.is_tensor(f)]
                else: # Handle cases where forward might return single tensor if only one out_index
                    self.width_list = [features.size(1)] if torch.is_tensor(features) else []

        finally:
            self.train()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except AttributeError:
                 pass
        elif isinstance(m, nn.Conv2d):
             pass
        elif isinstance(m, nn.BatchNorm2d):
             nn.init.constant_(m.weight, 1.0)
             nn.init.constant_(m.bias, 0.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_decay = set()
        for name, param in self.named_parameters():
             if 'norm' in name or 'bias' in name:
                  no_decay.add(name)
        return {"params_no_decay": [p for n, p in self.named_parameters() if n in no_decay]}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        keywords = {'bias', 'norm.weight', 'norm.bias'}
        # Add gammas if they exist (check layer scale usage)
        for i in range(self.num_layers):
             if hasattr(self.layers[i], 'blocks'):
                 for block in self.layers[i].blocks:
                     if hasattr(block, 'gamma_1') and isinstance(block.gamma_1, nn.Parameter):
                         keywords.add('gamma_1')
                     if hasattr(block, 'gamma_2') and isinstance(block.gamma_2, nn.Parameter):
                         keywords.add('gamma_2')
        return keywords

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        norm_idx = 0
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x) # x becomes the downsampled output for the next stage

            if i in self.out_indices:
                # x_out is the feature map *before* downsampling in this stage
                normed_x_out = x_out.permute(0, 3, 1, 2).contiguous()
                normed_x_out = self.output_norms[norm_idx](normed_x_out)
                norm_idx += 1
                outs.append(normed_x_out)

        # Return features in (B, C, H, W) format as a LIST
        return outs # <-- Changed from tuple(outs) to outs

# --- Factory functions (DFormerv2_S, DFormerv2_B, DFormerv2_L) remain the same ---
def DFormerv2_S(**kwargs):
    """ DFormerv2-Small """
    model = dformerv2(
        embed_dims=[64, 128, 256, 512], depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16], mlp_ratios=[4, 4, 4, 4],
        layerscales=[False, False, False, False], **kwargs)
    return model

def DFormerv2_B(**kwargs):
    """ DFormerv2-Base """
    model = dformerv2(
        embed_dims=[80, 160, 320, 512], depths=[4, 8, 25, 8],
        num_heads=[5, 5, 10, 16], mlp_ratios=[4, 4, 4, 4],
        layerscales=[False, False, True, True], layer_init_values=1e-6, **kwargs)
    return model

def DFormerv2_L(**kwargs):
    """ DFormerv2-Large """
    model = dformerv2(
        embed_dims=[112, 224, 448, 640], depths=[4, 8, 25, 8],
        num_heads=[7, 7, 14, 20], mlp_ratios=[4, 4, 4, 4],
        layerscales=[False, False, True, True], layer_init_values=1e-6, **kwargs)
    return model


# --- Example Usage (remains the same) ---
if __name__ == "__main__":
    image_size = (1, 3, 256, 256)
    image = torch.rand(*image_size)

    print("Testing DFormerv2_S...")
    model_s = DFormerv2_S()
    print(f"  Model Width List: {model_s.width_list}")
    out_s = model_s(image)
    assert isinstance(out_s, list), "Output should be a list"
    for i in range(len(out_s)):
        print(f"  Output {i} shape: {out_s[i].shape}")

    print("\nTesting DFormerv2_B...")
    model_b = DFormerv2_B()
    print(f"  Model Width List: {model_b.width_list}")
    out_b = model_b(image)
    assert isinstance(out_b, list), "Output should be a list"
    for i in range(len(out_b)):
        print(f"  Output {i} shape: {out_b[i].shape}")

    print("\nTesting DFormerv2_L...")
    model_l = DFormerv2_L()
    print(f"  Model Width List: {model_l.width_list}")
    out_l = model_l(image)
    assert isinstance(out_l, list), "Output should be a list"
    for i in range(len(out_l)):
        print(f"  Output {i} shape: {out_l[i].shape}")

    print(f"\nDFormerv2_S Params: {sum(p.numel() for p in model_s.parameters() if p.requires_grad)}")
    print(f"DFormerv2_B Params: {sum(p.numel() for p in model_b.parameters() if p.requires_grad)}")
    print(f"DFormerv2_L Params: {sum(p.numel() for p in model_l.parameters() if p.requires_grad)}")