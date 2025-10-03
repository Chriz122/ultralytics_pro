import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
# from fvcore.nn import FlopCountAnalysis, flop_count_table # Not used, can be removed
import time
from einops import rearrange
# from einops.layers.torch import Rearrange # Not used, can be removed
from typing import Tuple, List

# --------------------------------------------------------------------------------
# 以下為原始代碼1中的輔助類別，保持不變
# Swish, LayerNorm2d, GateLinearAttentionNoSilu, VanillaSelfAttention,
# FeedForwardNetwork, PatchMerging, PatchEmbed, RoPE, etc.
# --------------------------------------------------------------------------------

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

class RoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # The calculation of angle has been adjusted to ensure it's compatible with head dimension
        head_dim = embed_dim // num_heads
        angle = 1.0 / (10000 ** torch.linspace(0, 1, head_dim // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    def forward(self, slen: Tuple[int]):
        h, w = slen
        index_h = torch.arange(h, device=self.angle.device).float()
        index_w = torch.arange(w, device=self.angle.device).float()

        # Correctly broadcast angles for 2D positions
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :self.angle.size(0)//2])
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :self.angle.size(0)//2])
        sin_w = torch.sin(index_w[:, None] * self.angle[None, self.angle.size(0)//2:])
        cos_w = torch.cos(index_w[:, None] * self.angle[None, self.angle.size(0)//2:])

        sin_h = sin_h.unsqueeze(1).repeat(1, w, 1)
        cos_h = cos_h.unsqueeze(1).repeat(1, w, 1)
        sin_w = sin_w.unsqueeze(0).repeat(h, 1, 1)
        cos_w = cos_w.unsqueeze(0).repeat(h, 1, 1)

        sin = torch.cat([sin_h, sin_w], dim=-1).flatten(0, 1)
        cos = torch.cat([cos_h, cos_w], dim=-1).flatten(0, 1)

        return sin, cos

class GateLinearAttentionNoSilu(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        B, C, H, W = x.shape
        qkvo = self.qkvo(x)
        qkv, o = torch.split(qkvo, [3*self.dim, self.dim], dim=1)
        lepe = self.lepe(qkv[:, 2*self.dim:, :, :])

        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads)

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        q_mean = q.mean(dim=-2, keepdim=True)
        eff = self.scale * q_mean @ k.transpose(-1, -2)
        eff = torch.softmax(eff, dim=-1).transpose(-1, -2)
        k = k * eff * (H*W)

        # Reshape sin/cos for broadcasting
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)

        q_rope = theta_shift(q, sin, cos)
        k_rope = theta_shift(k, sin, cos)

        z = 1 / (torch.einsum('b n l d, b n d -> b n l', q, k.mean(dim=-2)) + 1e-6).unsqueeze(-1)
        kv = (k_rope.transpose(-2, -1) * ((H*W) ** -0.5)) @ (v * ((H*W) ** -0.5))

        res = q_rope @ kv * z
        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        res = res + lepe
        return self.proj(res * o)

class VanillaSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        B, C, H, W = x.shape
        qkvo = self.qkvo(x)
        qkv, o = torch.split(qkvo, [3*self.dim, self.dim], dim=1)
        lepe = self.lepe(qkv[:, 2*self.dim:, :, :])

        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads)

        # Reshape sin/cos for broadcasting
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)

        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)

        attn = torch.softmax(self.scale * q @ k.transpose(-1, -2), dim=-1)
        res = attn @ v
        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        res = res + lepe
        return self.proj(res * o)

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, activation_fn=F.gelu, dropout=0.0, activation_dropout=0.0, subconv=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Conv2d(self.embed_dim, ffn_dim, 1)
        self.fc2 = nn.Conv2d(ffn_dim, self.embed_dim, 1)
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, 3, 1, 1, groups=ffn_dim) if subconv else None

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            residual = x
            x = self.dwconv(x)
            x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x

class Block(nn.Module):
    def __init__(self, flag, embed_dim, num_heads, ffn_dim, drop_path=0., layerscale=False, layer_init_value=1e-6):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.norm1 = LayerNorm2d(embed_dim, eps=1e-6)
        assert flag in ['l', 'v']
        if flag == 'l':
            self.attn = GateLinearAttentionNoSilu(embed_dim, num_heads)
        else:
            self.attn = VanillaSelfAttention(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm2d(embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, int(ffn_dim)) # Ensure ffn_dim is int
        self.pos = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_value * torch.ones(1, embed_dim, 1, 1), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_value * torch.ones(1, embed_dim, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), sin, cos))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), sin, cos))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.reduction(x)
        x = self.norm(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, flags, embed_dim, out_dim, depth, num_heads,
                 ffn_dim=96., drop_path=0.,
                 downsample: PatchMerging=None,
                 layerscale=False, layer_init_value=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        # Correctly initialize RoPE for the current dimension and head count
        self.RoPE = RoPE(embed_dim, num_heads)

        self.blocks = nn.ModuleList([
            Block(flags[i], embed_dim, num_heads, ffn_dim,
                  drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_value)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor):
        _, _, h, w = x.size()
        sin, cos = self.RoPE((h, w))
        for blk in self.blocks:
            x = blk(x, sin, cos)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x):
        x = self.proj(x)
        return x

# --------------------------------------------------------------------------------
# 以下為核心修改部分
# --------------------------------------------------------------------------------

class RAVLT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 flagss=[['l']*2, ['l']*2, ['v', 'v']*6, ['v']*2],
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1,
                 projection=1024, layerscales=[False, False, False, False],
                 layer_init_values=[1e-6, 1e-6, 1e-6, 1e-6], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.in_chans = in_chans
        self.img_size = img_size

        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                flags=flagss[i_layer],
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                layerscale=layerscales[i_layer],
                layer_init_value=layer_init_values[i_layer]
            )
            self.layers.append(layer)

        # Classification head is kept for potential standalone use, but not used in `forward`
        self.proj = nn.Conv2d(self.num_features, projection, 1)
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Conv2d(projection, num_classes, 1) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        # --- MODIFICATION: Add width_list calculation ---
        self.width_list = []
        try:
            self.eval()
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size)
            with torch.no_grad():
                features = self.forward(dummy_input)
            self.width_list = [f.size(1) for f in features]
            self.train()
            # print(f"RAVLT width_list calculated successfully: {self.width_list}")
        except Exception as e:
            print(f"Error during dummy forward pass for width_list calculation: {e}")
            print("Setting width_list to embed_dims as fallback.")
            self.width_list = self.embed_dims
            self.train()

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ======================== START OF MODIFIED CODE ========================
    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        MODIFICATION: This method now returns a list of feature maps from each stage.
        This is crucial for integration with frameworks like Ultralytics YOLO.
        The feature map is captured *before* the downsampling of each stage to get the
        correct channel dimensions corresponding to `embed_dims`.
        """
        x = self.patch_embed(x)

        feature_outputs = []
        # Iterate through each stage (BasicLayer)
        for layer in self.layers:
            # Get positional embeddings for the current feature map size
            _, _, h, w = x.size()
            sin, cos = layer.RoPE((h, w))

            # Process the input through the blocks of the current stage
            for blk in layer.blocks:
                x = blk(x, sin, cos)

            # --- KEY CHANGE ---
            # Capture the feature map *before* downsampling.
            # Its channel dimension is `layer.embed_dim`.
            feature_outputs.append(x)

            # Apply the downsampling module (if it exists) to prepare `x` for the next stage.
            if layer.downsample is not None:
                x = layer.downsample(x)

        return feature_outputs
    # ========================= END OF MODIFIED CODE =========================

    def forward(self, x) -> List[torch.Tensor]:
        """
        MODIFICATION: The forward pass now directly returns the list of feature maps.
        """
        x = self.forward_features(x)
        return x

# --------------------------------------------------------------------------------
# 以下為新的工廠函數 (Factory Functions)
# --------------------------------------------------------------------------------

@register_model
def RAVLT_T(pretrained=False, img_size=224, **kwargs):
    """ Factory function for RAVLT-Tiny """
    model = RAVLT(
        img_size=img_size,
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 6, 2],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[3.5, 3.5, 3.5, 3.5],
        drop_path_rate=0.1,
        projection=1024,
        layerscales=[True, True, True, True],
        layer_init_values=[1e-6, 1e-6, 1e-6, 1e-6], # Adjusted init values for stability
        flagss=[['l']*2, ['l']*2, ['v', 'v']*3, ['v']*2],
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def RAVLT_S(pretrained=False, img_size=224, **kwargs):
    """ Factory function for RAVLT-Small """
    model = RAVLT(
        img_size=img_size,
        embed_dims=[64, 128, 320, 512],
        depths=[3, 5, 9, 3],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[3.5, 3.5, 3.5, 3.5],
        drop_path_rate=0.15,
        projection=1024,
        layerscales=[True, True, True, True],
        layer_init_values=[1e-6, 1e-6, 1e-6, 1e-6],
        flagss=[['l']*3, ['l']*5, ['v', 'v']*4 + ['v'], ['v']*3],
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def RAVLT_B(pretrained=False, img_size=224, **kwargs):
    """ Factory function for RAVLT-Base """
    model = RAVLT(
        img_size=img_size,
        embed_dims=[96, 192, 384, 512],
        depths=[4, 6, 12, 6],
        num_heads=[1, 2, 6, 8],
        mlp_ratios=[3.5, 3.5, 3.5, 3.5],
        drop_path_rate=0.4,
        projection=1024,
        layerscales=[True, True, True, True],
        layer_init_values=[1e-6, 1e-6, 1e-6, 1e-6],
        flagss=[['l']*4, ['l']*6, ['v', 'v']*6, ['v']*6],
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def RAVLT_L(pretrained=False, img_size=224, **kwargs):
    """ Factory function for RAVLT-Large """
    model = RAVLT(
        img_size=img_size,
        embed_dims=[96, 192, 448, 640],
        depths=[4, 7, 19, 8],
        num_heads=[1, 2, 7, 10],
        mlp_ratios=[3.5, 3.5, 3.5, 3.5],
        drop_path_rate=0.55,
        projection=1024,
        layerscales=[True, True, True, True],
        layer_init_values=[1e-6, 1e-6, 1e-6, 1e-6],
        flagss=[['l']*4, ['l']*7, ['v', 'v']*9 + ['v'], ['v']*8],
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

# --------------------------------------------------------------------------------
# 新增的測試區塊
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    img_h, img_w = 224, 224

    print("--- Creating RAVLT Tiny model ---")
    # 使用新的工廠函數來創建模型
    model = RAVLT_T(img_size=img_h)
    print("Model created successfully.")
    # 檢查 width_list 是否被正確計算
    print("Calculated width_list:", model.width_list)

    # 測試前向傳播
    input_tensor = torch.rand(2, 3, img_h, img_w)
    print(f"\n--- Testing RAVLT Tiny forward pass (Input: {input_tensor.shape}) ---")

    model.eval()
    try:
        with torch.no_grad():
            output_features = model(input_tensor)
        print("Forward pass successful.")

        # 檢查輸出是否為列表
        assert isinstance(output_features, list), f"Output should be a list, but got {type(output_features)}"
        print(f"Output is a list with {len(output_features)} elements, as expected.")

        # 打印每個階段的輸出特徵圖形狀
        print("Output feature shapes:")
        for i, features in enumerate(output_features):
            print(f"Stage {i+1}: {features.shape}")

        # 驗證 width_list 是否與執行時的輸出通道匹配
        runtime_widths = [f.size(1) for f in output_features]
        print("\nRuntime output feature channels:", runtime_widths)
        assert model.width_list == runtime_widths, "Width list mismatch!"
        print("Width list verified successfully.")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()