import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

# --- Imports from timm (necessary parts) ---
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

# --- Custom Normalization Layers (from Code 1) ---
class RepBN(nn.Module):
    def __init__(self, channels):
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        if channels == 0:
            self.bn = nn.Identity()
        else:
            self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        if isinstance(self.bn, nn.Identity):
            return x
        x_transposed = x.transpose(1, 2)
        bn_out = self.bn(x_transposed)
        out = bn_out + self.alpha * x_transposed
        return out.transpose(1, 2)


class LinearNorm(nn.Module):
    def __init__(self, dim, norm1_class, norm2_class, warm=0, step=300000, r0=1.0):
        super(LinearNorm, self).__init__()
        self.register_buffer('warm', torch.tensor(warm).float())
        self.register_buffer('iter', torch.tensor(step).float())
        self.register_buffer('total_step', torch.tensor(step).float())
        self.r0 = r0

        if dim == 0:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            self.norm1 = norm1_class(dim)
            self.norm2 = norm2_class(dim)

    def forward(self, x):
        if isinstance(self.norm1, nn.Identity):
            return x

        if self.training:
            if self.warm > 0:
                self.warm.copy_(self.warm - 1)
                x = self.norm1(x)
            else:
                if self.total_step > 0:
                    lamda = self.r0 * max(0.0, self.iter.item()) / self.total_step.item()
                else:
                    lamda = 0.0 

                if self.iter > 0:
                    self.iter.copy_(self.iter - 1)

                x1 = self.norm1(x)
                x2 = self.norm2(x)
                x = lamda * x1 + (1 - lamda) * x2
        else:
            x = self.norm2(x)
        return x

ln_partial = partial(nn.LayerNorm, eps=1e-6)
linearnorm_partial = partial(LinearNorm, norm1_class=ln_partial, norm2_class=RepBN)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        if dim == 0:
            self.dwconv = nn.Identity()
        else:
            self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        if isinstance(self.dwconv, nn.Identity):
            return x
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear = linear

        if in_features == 0 or hidden_features == 0 or out_features == 0:
            self.fc1 = nn.Identity()
            self.dwconv = nn.Identity()
            self.act = nn.Identity()
            self.fc2 = nn.Identity()
            self.drop = nn.Identity()
            if self.linear:
                 self.relu = nn.Identity()
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.dwconv = DWConv(hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)
            if self.linear:
                self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if hasattr(m, 'groups') and m.groups > 0:
                fan_out //= m.groups
            if fan_out > 0:
                 m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            else:
                 m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        if isinstance(self.fc1, nn.Identity):
            return x
        x = self.fc1(x)
        if self.linear and isinstance(self.relu, nn.ReLU):
            x = self.relu(x)
        if isinstance(self.dwconv, DWConv):
            x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        self.linear = linear
        self.sr_ratio = sr_ratio

        if dim == 0 or num_heads == 0:
            self.dim = 0
            self.num_heads = 0
            self.head_dim = 0
            self.scale = 1.0
            self.q = nn.Identity()
            self.kv = nn.Identity()
            self.attn_drop = nn.Identity()
            self.proj = nn.Identity()
            self.proj_drop = nn.Identity()
            self.sr = nn.Identity()
            self.norm = nn.Identity()
            if linear:
                self.pool = nn.Identity()
                self.act = nn.Identity()
            return

        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
            else:
                self.sr = nn.Identity()
                self.norm = nn.Identity()
        else:
            pool_size = 7 if dim >= 7 else 1
            self.pool = nn.AdaptiveAvgPool2d(pool_size)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if hasattr(m, 'groups') and m.groups > 0: fan_out //= m.groups
            if fan_out > 0: m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            else: m.weight.data.normal_(0, 0.02)
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x, H, W):
        if self.dim == 0:
            return x
        B, N, C = x.shape
        if C != self.dim :
             raise ValueError(f"Input C {C} != self.dim {self.dim} in Attention")

        q_out = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1 and isinstance(self.sr, nn.Conv2d):
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv_out = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            else:
                kv_out = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            if isinstance(self.pool, nn.AdaptiveAvgPool2d): x_ = self.pool(x_)
            if isinstance(self.sr, nn.Conv2d): x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            if isinstance(self.norm, nn.LayerNorm): x_ = self.norm(x_)
            if isinstance(self.act, nn.GELU): x_ = self.act(x_)
            kv_out = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k_out, v_out = kv_out[0], kv_out[1]
        attn_scores = (q_out @ k_out.transpose(-2, -1)) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.attn_drop(attn_probs)
        
        context_layer = (attn_probs @ v_out).transpose(1, 2).reshape(B, N, C)
        context_layer = self.proj(context_layer)
        context_layer = self.proj_drop(context_layer)
        return context_layer


class SimplifiedLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, linear=False,
                 focusing_factor=3, kernel_size=5, norm_layer=nn.LayerNorm):
        super().__init__()
        self.sr_ratio = sr_ratio
        self.linear_flag_from_pvt = linear

        if dim == 0 or num_heads == 0: # num_patches can be 0 if img_size/stride leads to it
            self.dim = 0
            self.num_heads = 0
            self.head_dim = 0
            self.q = nn.Identity()
            self.kv = nn.Identity()
            self.attn_drop = nn.Identity() # Not used in this SLA variant's math directly
            self.proj = nn.Identity()
            self.proj_drop = nn.Identity()
            self.sr = nn.Identity()
            self.norm = nn.Identity()
            if self.linear_flag_from_pvt:
                self.pool = nn.Identity()
                self.act = nn.Identity()
            self.dwc = nn.Identity()
            self.positional_encoding = nn.Identity()
            return

        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) 
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if not self.linear_flag_from_pvt:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = norm_layer(dim)
            else:
                self.sr = nn.Identity()
                self.norm = nn.Identity()
        else:
            pool_size = 7 if dim >= 7 else 1
            self.pool = nn.AdaptiveAvgPool2d(pool_size)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = norm_layer(dim)
            self.act = nn.GELU()

        self.focusing_factor = focusing_factor
        if self.head_dim > 0: # DWC only if head_dim is valid
             self.dwc = nn.Conv2d(in_channels=self.head_dim, out_channels=self.head_dim, kernel_size=kernel_size,
                                 groups=self.head_dim, padding=kernel_size // 2)
        else:
            self.dwc = nn.Identity()
        
        num_reduced_patches = num_patches // (sr_ratio * sr_ratio) if sr_ratio > 0 and num_patches > 0 else num_patches
        if num_reduced_patches > 0 and dim > 0 :
             self.positional_encoding = nn.Parameter(torch.zeros(1, num_reduced_patches, dim))
             trunc_normal_(self.positional_encoding, std=.02) # Initialize here
        else:
             self.positional_encoding = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if hasattr(m, 'groups') and m.groups > 0: fan_out //= m.groups
            if fan_out > 0: m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            else: m.weight.data.normal_(0, 0.02)
            if m.bias is not None: m.bias.data.zero_()
        # Positional encoding is initialized at creation if it's a Parameter

    def forward(self, x, H, W):
        if self.dim == 0:
            return x
        
        B_orig, N_tokens, C_in_total = x.shape
        if C_in_total != self.dim:
            raise ValueError(f"Input C {C_in_total} != self.dim {self.dim} in SimplifiedLinearAttention")

        q_full = self.q(x)

        if not self.linear_flag_from_pvt:
            if self.sr_ratio > 1 and isinstance(self.sr, nn.Conv2d):
                x_kv = x.permute(0, 2, 1).reshape(B_orig, self.dim, H, W)
                x_kv = self.sr(x_kv).reshape(B_orig, self.dim, -1).permute(0, 2, 1)
                if isinstance(self.norm, nn.LayerNorm): x_kv = self.norm(x_kv) # Check instance before calling
                kv_full = self.kv(x_kv).reshape(B_orig, -1, 2, self.dim)
            else:
                kv_full = self.kv(x).reshape(B_orig, N_tokens, 2, self.dim)
        else:
            x_kv = x.permute(0, 2, 1).reshape(B_orig, self.dim, H, W)
            if isinstance(self.pool, nn.AdaptiveAvgPool2d): x_kv = self.pool(x_kv)
            if isinstance(self.sr, nn.Conv2d): x_kv = self.sr(x_kv)
            x_kv = x_kv.reshape(B_orig, self.dim, -1).permute(0, 2, 1)
            if isinstance(self.norm, nn.LayerNorm): x_kv = self.norm(x_kv)
            if isinstance(self.act, nn.GELU): x_kv = self.act(x_kv)
            kv_full = self.kv(x_kv).reshape(B_orig, -1, 2, self.dim)
        
        k_full = kv_full[..., 0, :].contiguous() # (B_orig, N_kv_eff, self.dim)
        v_full = kv_full[..., 1, :].contiguous() # (B_orig, N_kv_eff, self.dim)

        if isinstance(self.positional_encoding, nn.Parameter):
            if self.positional_encoding.shape[1] == k_full.shape[1] and \
               self.positional_encoding.shape[2] == k_full.shape[2]:
                 k_full = k_full + self.positional_encoding
        
        kernel_function = F.relu # Using F.relu to avoid creating an nn.Module here

        q_activated = kernel_function(q_full)
        k_activated = kernel_function(k_full)
        
        # q_activated: (B_orig, N_tokens, self.dim)
        # k_activated: (B_orig, N_kv_eff, self.dim)
        # v_full:      (B_orig, N_kv_eff, self.dim)

        q_multihead, k_multihead, v_multihead = \
            (rearrange(t, "b n (h c_head) -> (b h) n c_head", h=self.num_heads) \
             for t in [q_activated, k_activated, v_full])
        
        # q_multihead: (B_orig * num_heads, N_tokens, self.head_dim)
        # k_multihead: (B_orig * num_heads, N_kv_eff, self.head_dim)
        # v_multihead: (B_orig * num_heads, N_kv_eff, self.head_dim)

        # Simplified linear attention computation
        k_sum_across_seq = k_multihead.sum(dim=1) # (B_orig * num_heads, self.head_dim)
        
        # qk_sum : (B_orig*num_heads, N_tokens)
        qk_sum = torch.einsum('b n d, b d -> b n', q_multihead, k_sum_across_seq)
        D_inv = 1.0 / (qk_sum + 1e-6) # (B_orig*num_heads, N_tokens)

        # context: (B_orig*num_heads, self.head_dim, self.head_dim)
        context = torch.einsum('b m d, b m e -> b d e', k_multihead, v_multihead)
        # x_attn: (B_orig*num_heads, N_tokens, self.head_dim)
        x_attn = torch.einsum('b n d, b d e -> b n e', q_multihead, context)
        x_attn = x_attn * D_inv.unsqueeze(-1) # Element-wise mult with D_inv

        # DWC path
        if isinstance(self.dwc, nn.Conv2d) and self.head_dim > 0:
            v_for_dwc = v_multihead # (B_orig*num_heads, N_kv_eff, self.head_dim)
            if v_for_dwc.shape[1] != q_multihead.shape[1]: # If N_kv_eff != N_tokens
                 v_for_dwc = v_for_dwc.permute(0, 2, 1) # to (BH, C_head, N_kv_eff)
                 v_for_dwc = F.interpolate(v_for_dwc, size=q_multihead.shape[1], mode='linear', align_corners=False)
                 v_for_dwc = v_for_dwc.permute(0, 2, 1) # back to (BH, N_tokens, C_head)
            
            # N_for_dwc is now N_tokens (q_multihead.shape[1])
            # H, W are from the original input to the Block, corresponding to N_tokens
            if N_tokens == H * W :
                feature_map_v = rearrange(v_for_dwc, "bh (h w) c_head -> bh c_head h w", h=H, w=W)
                dwc_out = self.dwc(feature_map_v)
                feature_map_v_enhanced = rearrange(dwc_out, "bh c_head h w -> bh (h w) c_head")
                x_attn = x_attn + feature_map_v_enhanced
            # else:
                # print(f"Warning: SLA DWC N_tokens {N_tokens} != H*W {H*W}. Skipping DWC.")

        # Reshape back to (B_orig, N_tokens, self.dim)
        x_output = x_attn.reshape(B_orig, N_tokens, self.dim)

        x_output = self.proj(x_output)
        x_output = self.proj_drop(x_output)

        return x_output


class Block(nn.Module):
    def __init__(self, dim, num_patches, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False,
                 focusing_factor=3, kernel_size=5, attn_type='L'):
        super().__init__()

        if dim == 0:
            self.norm1 = nn.Identity()
            self.attn = nn.Identity()
            self.drop_path = nn.Identity()
            self.norm2 = nn.Identity()
            self.mlp = nn.Identity()
        else:
            self.norm1 = norm_layer(dim)
            assert attn_type in ['L', 'S'], f"attn_type must be 'L' or 'S', got {attn_type}"
            if attn_type == 'L':
                self.attn = SimplifiedLinearAttention(
                    dim, num_patches=num_patches,
                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear,
                    focusing_factor=focusing_factor, kernel_size=kernel_size, norm_layer=norm_layer)
            else:
                self.attn = Attention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)

            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            if dim > 0 and mlp_hidden_dim == 0: mlp_hidden_dim = dim
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if hasattr(m, 'groups') and m.groups > 0: fan_out //= m.groups
            if fan_out > 0: m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            else: m.weight.data.normal_(0, 0.02)
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x, H, W):
        if isinstance(self.norm1, nn.Identity):
            return x
        
        attn_out = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(attn_out)
        
        mlp_out = self.mlp(self.norm2(x), H, W)
        x = x + self.drop_path(mlp_out)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_layer=nn.LayerNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size 
        self.patch_size = patch_size
        self.stride = stride

        if embed_dim == 0 or in_chans == 0 or stride == 0:
            self.H, self.W = (img_size[0] // stride if stride > 0 else img_size[0],
                              img_size[1] // stride if stride > 0 else img_size[1])
            self.num_patches = self.H * self.W
            self.proj = nn.Identity()
            self.norm = nn.Identity()
        else:
            if not (isinstance(patch_size[0], int) and isinstance(patch_size[1], int) and
                    isinstance(stride, int)):
                raise TypeError(f"patch_size and stride must be int or tuple of ints. Got {patch_size}, {stride}")
            
            self.H = img_size[0] // stride
            self.W = img_size[1] // stride
            self.num_patches = self.H * self.W
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                  padding=(patch_size[0] // 2, patch_size[1] // 2))
            self.norm = norm_layer(embed_dim) if embed_dim > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if hasattr(m, 'groups') and m.groups > 0: fan_out //= m.groups
            if fan_out > 0: m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            else: m.weight.data.normal_(0, 0.02)
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x):
        if isinstance(self.proj, nn.Identity):
            B, C_in, H_in, W_in = x.shape
            H_out = H_in // self.stride if self.stride > 0 else H_in
            W_out = W_in // self.stride if self.stride > 0 else W_in
            # Output C will be C_in. If C_in is 0, then output C is 0.
            # Reshape to (B, N, C_in)
            return x.flatten(2).transpose(1,2) if C_in > 0 else torch.zeros((B, H_out * W_out, 0), device=x.device, dtype=x.dtype), H_out, W_out

        x_proj = self.proj(x)
        _, _, H_out, W_out = x_proj.shape
        x_flat = x_proj.flatten(2).transpose(1, 2)
        if isinstance(self.norm, (nn.LayerNorm, LinearNorm)):
            x_norm = self.norm(x_flat)
        else: # Is nn.Identity
            x_norm = x_flat
        return x_norm, H_out, W_out


class SlabPyramidVisionTransformerV2(nn.Module):
    arch_zoo = {
        'b0': {'embed_dims': [32, 64, 160, 256], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [8, 8, 4, 4],
               'depths': [2, 2, 2, 2], 'sr_ratios': [8, 4, 2, 1], 'la_sr_ratios': '8421',
               'attn_type': 'LLLL', 'linear': False, 'focusing_factor': 3, 'kernel_size': 5},
        'b1': {'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [8, 8, 4, 4],
               'depths': [2, 2, 2, 2], 'sr_ratios': [8, 4, 2, 1], 'la_sr_ratios': '8421',
               'attn_type': 'LLLL', 'linear': False, 'focusing_factor': 3, 'kernel_size': 5},
        'b2': {'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [8, 8, 4, 4],
               'depths': [3, 4, 6, 3], 'sr_ratios': [8, 4, 2, 1], 'la_sr_ratios': '8421',
               'attn_type': 'LLLL', 'linear': False, 'focusing_factor': 3, 'kernel_size': 5},
        'b3': {'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [8, 8, 4, 4],
               'depths': [3, 4, 18, 3], 'sr_ratios': [8, 4, 2, 1], 'la_sr_ratios': '8421',
               'attn_type': 'LLLL', 'linear': False, 'focusing_factor': 3, 'kernel_size': 5},
        'b4': {'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [8, 8, 4, 4],
               'depths': [3, 8, 27, 3], 'sr_ratios': [8, 4, 2, 1], 'la_sr_ratios': '8421',
               'attn_type': 'LLLL', 'linear': False, 'focusing_factor': 3, 'kernel_size': 5},
        'b5': {'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [4, 4, 4, 4],
               'depths': [3, 6, 40, 3], 'sr_ratios': [8, 4, 2, 1], 'la_sr_ratios': '8421',
               'attn_type': 'LLLL', 'linear': False, 'focusing_factor': 3, 'kernel_size': 5},
    }
    arch_zoo['b2_li'] = {**arch_zoo['b2'], 'linear': True}


    def __init__(self,
                 c1=3,
                 arch='b2',
                 img_size=224,
                 num_classes=0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=linearnorm_partial,
                 fork_feat=True,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.fork_feat = fork_feat

        if isinstance(arch, str):
            arch_key = arch.lower()
            if arch_key not in self.arch_zoo:
                raise KeyError(f"Arch '{arch}' is not in SlabPVTv2 archs {list(self.arch_zoo.keys())}")
            self.arch_settings = self.arch_zoo[arch_key]
        elif isinstance(arch, dict):
            default_arch_config = next(iter(self.arch_zoo.values())) # Get a default config
            self.arch_settings = {**default_arch_config, **arch}
        else:
            raise TypeError("arch must be a string or a dict")

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_heads = self.arch_settings['num_heads']
        self.mlp_ratios = self.arch_settings['mlp_ratios']
        self.depths = self.arch_settings['depths']
        self.sr_ratios = self.arch_settings['sr_ratios']
        self.la_sr_ratios = [int(r) for r in list(self.arch_settings['la_sr_ratios'])]
        self.attn_type_str = self.arch_settings['attn_type']
        self.linear_pvt_mode = self.arch_settings['linear']
        self.focusing_factor = self.arch_settings['focusing_factor']
        self.kernel_size = self.arch_settings['kernel_size']
        self.num_stages = len(self.depths)

        # Assertions for config consistency
        for config_list in [self.embed_dims, self.num_heads, self.mlp_ratios, self.sr_ratios, self.la_sr_ratios]:
            assert len(config_list) == self.num_stages, f"Config list length mismatch: {config_list} vs num_stages {self.num_stages}"
        assert len(self.attn_type_str) == self.num_stages, f"attn_type_str length mismatch"


        if self.fork_feat:
            self.width_list = [e for e in self.embed_dims if e > 0]
            if not self.width_list and any(d > 0 for d in self.embed_dims):
                self.width_list = [0] * sum(1 for e in self.embed_dims if e > 0) # Should match non-zero stages
            elif not self.width_list and not any(d > 0 for d in self.embed_dims): # All dims are zero
                 self.width_list = [0] * self.num_stages
        else: # Classification mode
            self.width_list = [self.embed_dims[-1]] if self.embed_dims and self.embed_dims[-1] > 0 else \
                              ([0] if self.embed_dims else [])


        total_depth = sum(self.depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)] if total_depth > 0 else []
        cur_idx = 0
        current_in_chans = c1
        h_w_tracker = to_2tuple(img_size)

        for i in range(self.num_stages):
            patch_embed_stride = 4 if i == 0 else 2
            patch_embed_patch_size = 7 if i == 0 else 3
            
            stage_nominal_h, stage_nominal_w = h_w_tracker

            patch_embed = OverlapPatchEmbed(
                img_size=(stage_nominal_h, stage_nominal_w),
                patch_size=patch_embed_patch_size,
                stride=patch_embed_stride,
                in_chans=current_in_chans,
                embed_dim=self.embed_dims[i],
                norm_layer=norm_layer
            )
            
            h_w_tracker = (stage_nominal_h // patch_embed_stride if patch_embed_stride > 0 else stage_nominal_h,
                           stage_nominal_w // patch_embed_stride if patch_embed_stride > 0 else stage_nominal_w)

            stage_attn_type = self.attn_type_str[i]
            current_sr_ratio = self.sr_ratios[i] if stage_attn_type == 'S' else self.la_sr_ratios[i]

            # num_patches for SLA should be based on the output of patch_embed for this stage
            # patch_embed.num_patches is based on its *own* img_size, patch_size, stride.
            num_patches_for_block = patch_embed.num_patches

            stage_blocks = nn.ModuleList([Block(
                dim=self.embed_dims[i],
                num_patches=num_patches_for_block,
                num_heads=self.num_heads[i],
                mlp_ratio=self.mlp_ratios[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur_idx + j] if dpr and (cur_idx + j < len(dpr)) else 0.0,
                norm_layer=norm_layer,
                sr_ratio=current_sr_ratio,
                linear=self.linear_pvt_mode,
                focusing_factor=self.focusing_factor,
                kernel_size=self.kernel_size,
                attn_type=stage_attn_type)
                for j in range(self.depths[i])])

            stage_out_norm = norm_layer(self.embed_dims[i]) if self.embed_dims[i] > 0 else nn.Identity()

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", stage_blocks)
            setattr(self, f"norm{i + 1}", stage_out_norm)

            current_in_chans = self.embed_dims[i]
            cur_idx += self.depths[i]

        self.head = nn.Identity()
        if self.num_classes > 0 and not self.fork_feat:
            head_in_dim = self.embed_dims[-1] if self.embed_dims and self.embed_dims[-1] > 0 else 0
            if head_in_dim > 0:
                self.head = nn.Linear(head_in_dim, self.num_classes)

        self.apply(self._init_weights_main)

    def _init_weights_main(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def freeze_patch_emb(self):
        if hasattr(self, 'patch_embed1'):
             for param in self.patch_embed1.parameters():
                 param.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        param_names = set()
        for n, p in self.named_parameters():
            param_names.add(n)
            if '.bias' in n or 'norm' in n.lower() or 'bn' in n.lower() or '.alpha' in n:
                nwd.add(n)
            elif 'positional_encoding' in n and isinstance(p, nn.Parameter): # Check if it's a Parameter
                nwd.add(n)
            elif p.ndim < 2 :
                 if 'weight' not in n or 'norm' in n.lower() or 'bn' in n.lower(): # Be more specific for 1D weights
                    nwd.add(n)
        
        no_decay_params = {name: p for name, p in self.named_parameters() if name in nwd and p.requires_grad}
        return no_decay_params


    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        current_x = x
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            blocks = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            current_x, H, W = patch_embed(current_x)

            if self.embed_dims[i] == 0:
                if self.fork_feat:
                    zero_channel_feat = torch.zeros((B, 0, H, W), device=current_x.device, dtype=current_x.dtype)
                    outs.append(zero_channel_feat)
                current_x = torch.zeros((B, 0, H, W), device=current_x.device, dtype=current_x.dtype)
                continue

            for blk in blocks:
                current_x = blk(current_x, H, W)

            if not isinstance(norm, nn.Identity):
                current_x = norm(current_x)
            
            # Ensure C dim matches expected embed_dim for this stage
            if current_x.shape[2] != self.embed_dims[i]:
                 # This can happen if embed_dim is 0, but that's handled above.
                 # If not handled, it's a mismatch.
                raise ValueError(
                    f"Stage {i+1} output C dimension {current_x.shape[2]} ({current_x.shape}) "
                    f"!= expected embed_dim {self.embed_dims[i]}"
                )

            # Reshape to (B, C, H, W) for feature output or next stage input (if not last)
            reshaped_x = current_x.reshape(B, H, W, self.embed_dims[i]).permute(0, 3, 1, 2).contiguous()

            if self.fork_feat:
                outs.append(reshaped_x)
            
            if i < self.num_stages - 1 : # If not the last stage, pass reshaped_x to next patch_embed
                current_x = reshaped_x
            else: # If last stage and not forking, current_x (B,N,C) is used for classifier.
                  # If forking, reshaped_x was already appended. current_x isn't used further.
                  if not self.fork_feat:
                      current_x = reshaped_x # For classifier, use the (B,C,H,W) format for GAP

        if self.fork_feat:
            if self.width_list is not None:
                if len(outs) != len(self.width_list):
                    # This might indicate an issue with how width_list is constructed vs. how outs are collected
                    # especially with zero-dim stages.
                    # print(f"Warning: SlabPVTv2 features len {len(outs)} != width_list len {len(self.width_list)}")
                    # print(f"  outs channels: {[o.shape[1] for o in outs]}")
                    # print(f"  width_list: {self.width_list}")
                    pass # Potentially adjust width_list or outs logic if mismatch is problematic
            return outs
        else:
            # Classification path, current_x is the output of the last stage, reshaped to (B,C,H,W)
            if current_x.shape[1] == 0: # Last stage had 0 channels
                return torch.zeros((B, self.num_classes if self.num_classes > 0 else 0), device=current_x.device, dtype=current_x.dtype)

            pooled_x = F.adaptive_avg_pool2d(current_x, 1).flatten(1)
            final_output = self.head(pooled_x)
            return final_output

    def forward(self, x):
        return self.forward_features(x)

def _create_slab_pvt_v2(arch_name, c1=3, fork_feat=True, num_classes=0, pretrained=False, **kwargs):
    model_kwargs = kwargs.copy()
    if 'norm_layer' not in model_kwargs: # Set default norm_layer if not provided
        model_kwargs['norm_layer'] = linearnorm_partial
    if 'qkv_bias' not in model_kwargs: # Default qkv_bias
        model_kwargs['qkv_bias'] = True

    model = SlabPyramidVisionTransformerV2(
        c1=c1,
        arch=arch_name,
        fork_feat=fork_feat,
        num_classes=num_classes,
        **model_kwargs
    )
    return model

def slab_pvt_v2_b0(c1=3, fork_feat=True, num_classes=0, pretrained=False, **kwargs):
    return _create_slab_pvt_v2('b0', c1, fork_feat, num_classes, pretrained, **kwargs)

def slab_pvt_v2_b1(c1=3, fork_feat=True, num_classes=0, pretrained=False, **kwargs):
    return _create_slab_pvt_v2('b1', c1, fork_feat, num_classes, pretrained, **kwargs)

def slab_pvt_v2_b2(c1=3, fork_feat=True, num_classes=0, pretrained=False, **kwargs):
    return _create_slab_pvt_v2('b2', c1, fork_feat, num_classes, pretrained, **kwargs)

def slab_pvt_v2_b2_li(c1=3, fork_feat=True, num_classes=0, pretrained=False, **kwargs):
    return _create_slab_pvt_v2('b2_li', c1, fork_feat, num_classes, pretrained, **kwargs)

def slab_pvt_v2_b3(c1=3, fork_feat=True, num_classes=0, pretrained=False, **kwargs):
    return _create_slab_pvt_v2('b3', c1, fork_feat, num_classes, pretrained, **kwargs)

def slab_pvt_v2_b4(c1=3, fork_feat=True, num_classes=0, pretrained=False, **kwargs):
    return _create_slab_pvt_v2('b4', c1, fork_feat, num_classes, pretrained, **kwargs)

def slab_pvt_v2_b5(c1=3, fork_feat=True, num_classes=0, pretrained=False, **kwargs):
    return _create_slab_pvt_v2('b5', c1, fork_feat, num_classes, pretrained, **kwargs)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_b0_backbone = slab_pvt_v2_b0(c1=3, fork_feat=True).to(device)
    model_b0_backbone.eval()
    print(f"SlabPVTv2-B0 Backbone width_list: {model_b0_backbone.width_list}")
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    print(f"\nTesting SlabPVTv2-B0 as backbone (fork_feat=True):")
    try:
        features_b0 = model_b0_backbone(dummy_input)
        if isinstance(features_b0, list):
            print(f"  Output is a list of {len(features_b0)} tensors:")
            for i, f in enumerate(features_b0):
                print(f"    Stage {i+1} feature shape: {f.shape}")
            if model_b0_backbone.width_list is not None:
                for i, f_w in enumerate(model_b0_backbone.width_list):
                    assert features_b0[i].shape[1] == f_w, \
                        f"Mismatch: Feature width {features_b0[i].shape[1]} vs width_list {f_w} at stage {i}"
                print("  Feature shapes match width_list.")
        else:
            print(f"  Output shape (expected list): {features_b0.shape}")
    except Exception as e:
        print(f"  Error during B0 backbone test: {e}")
        import traceback
        traceback.print_exc()

    model_b1_classifier = slab_pvt_v2_b1(c1=3, fork_feat=False, num_classes=100).to(device)
    model_b1_classifier.eval()
    print(f"\nTesting SlabPVTv2-B1 as classifier (fork_feat=False, num_classes=100):")
    try:
        predictions_b1 = model_b1_classifier(dummy_input)
        print(f"  Output predictions shape: {predictions_b1.shape}") 
        assert predictions_b1.shape == (dummy_input.shape[0], 100)
    except Exception as e:
        print(f"  Error during B1 classifier test: {e}")
        import traceback
        traceback.print_exc()

    model_b2li_backbone = slab_pvt_v2_b2_li(c1=3, fork_feat=True).to(device)
    model_b2li_backbone.eval()
    print(f"\nTesting SlabPVTv2-B2-li as backbone (fork_feat=True):")
    try:
        features_b2li = model_b2li_backbone(dummy_input)
        if isinstance(features_b2li, list):
            print(f"  Output is a list of {len(features_b2li)} tensors:")
            for i, f in enumerate(features_b2li):
                print(f"    Stage {i+1} feature shape: {f.shape}")
        else:
            print(f"  Output shape (expected list): {features_b2li.shape}")
    except Exception as e:
        print(f"  Error during B2-li backbone test: {e}")
        import traceback
        traceback.print_exc()

    dummy_input_diff_size = torch.randn(1, 3, 256, 320).to(device)
    print(f"\nTesting SlabPVTv2-B0 with input 256x320:")
    try:
        features_b0_ds = model_b0_backbone(dummy_input_diff_size)
        if isinstance(features_b0_ds, list):
            print(f"  Output is a list of {len(features_b0_ds)} tensors:")
            for i, f in enumerate(features_b0_ds):
                print(f"    Stage {i+1} feature shape: {f.shape}")
        else:
            print(f"  Output shape (expected list): {features_b0_ds.shape}")
    except Exception as e:
        print(f"  Error during B0 dynamic size test: {e}")
        import traceback
        traceback.print_exc()
        
    custom_arch_zero_dim_config = model_b0_backbone.arch_zoo['b0'].copy()
    custom_arch_zero_dim_config['embed_dims'] = [32, 0, 160, 256] 
    
    print(f"\nTesting SlabPVTv2 with a zero-dim stage:")
    try:
        model_zero_dim = SlabPyramidVisionTransformerV2(c1=3, arch=custom_arch_zero_dim_config, fork_feat=True).to(device)
        model_zero_dim.eval()
        print(f"  Zero-dim model width_list: {model_zero_dim.width_list}") # Expect [32, 160, 256]
        features_zero = model_zero_dim(dummy_input)
        if isinstance(features_zero, list):
            print(f"  Output is a list of {len(features_zero)} tensors:")
            for i, f in enumerate(features_zero):
                print(f"    Stage {i+1} feature shape: {f.shape}")
            if model_zero_dim.width_list is not None:
                 assert len(features_zero) == len(model_zero_dim.width_list), "Mismatch in number of output features and width_list length"
                 for i, f_w in enumerate(model_zero_dim.width_list):
                    assert features_zero[i].shape[1] == f_w, \
                        f"Mismatch: Feature width {features_zero[i].shape[1]} vs width_list {f_w} at stage {i}"
                 print("  Feature shapes match width_list for zero-dim model.")
        else:
            print(f"  Output shape (expected list): {features_zero.shape}")
    except Exception as e:
        print(f"  Error during zero-dim stage test: {e}")
        import traceback
        traceback.print_exc()