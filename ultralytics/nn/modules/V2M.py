import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional, List

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math
from math import pi
from einops import rearrange, repeat
import random

# --- 1. Mamba Import Handling with Fallbacks ---
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba = None
    print("Warning: mamba_ssm not found. Model will not be runnable without it.")

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# Fallback RMSNorm implementation if Triton version is missing
class RMSNormFallback(nn.Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

# Use fallback if import failed
if RMSNorm is None:
    RMSNorm = RMSNormFallback

# Fallback layer_norm functions for fused operations
# Since we might not have the fused kernel, we use standard pytorch implementation
if layer_norm_fn is None:
    def layer_norm_fn(x, weight, bias, residual=None, eps=1e-6, prenorm=False, residual_in_fp32=False):
        # Naive implementation to match signature
        dtype = x.dtype
        if residual is not None:
            if residual_in_fp32:
                residual = residual.to(torch.float32) + x.to(torch.float32)
                x = residual.to(dtype)
            else:
                residual = residual + x
                x = residual
        
        # LayerNorm
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + eps)
        if weight is not None:
            out = out * weight
        if bias is not None:
            out = out + bias
            
        return out, residual

if rms_norm_fn is None:
    def rms_norm_fn(x, weight, bias, residual=None, eps=1e-6, prenorm=False, residual_in_fp32=False):
        # Naive implementation
        dtype = x.dtype
        if residual is not None:
            if residual_in_fp32:
                residual = residual.to(torch.float32) + x.to(torch.float32)
                x = residual.to(dtype)
            else:
                residual = residual + x
                x = residual
        
        # RMSNorm
        rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        out = x * rrms
        if weight is not None:
            out = out * weight
        # bias is usually not used in RMSNorm but included in signature
        if bias is not None:
            out = out + bias
            
        return out, residual


# --- Helper Functions ---

def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t): 
        if t.shape[1] % 2 != 0:
            t_spatial = t[:, 1:, :]
            t_spatial = t_spatial * self.freqs_cos + rotate_half(t_spatial) * self.freqs_sin
            return torch.cat((t[:, :1, :], t_spatial), dim=1)
        else:
            return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            # ensure norm weight type
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    
    # -----------------------------------------------------------------------
    # FIX 1: Filter arguments for Mamba. 
    # Standard Mamba does not accept bimamba_type, if_devide_out, etc.
    # We only pass keys that standard Mamba likely accepts or are in ssm_cfg.
    # -----------------------------------------------------------------------
    mamba_args = {
        "layer_idx": layer_idx, 
        **ssm_cfg, 
        **factory_kwargs
    }
    # Note: If you have a custom Mamba that needs bimamba_type, add it back here manually 
    # but strictly conditionally. For standard package, we exclude it.
    
    mixer_cls = partial(Mamba, **mamba_args)
    
    # -----------------------------------------------------------------------
    # FIX 2: Ensure norm_cls is callable. RMSNorm is ensured to be not None.
    # -----------------------------------------------------------------------
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

# --- VisionMamba Class with Width List & List Output ---

class VisionMamba(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 stride=16,
                 depth=24, 
                 embed_dim=192, 
                 channels=3, 
                 num_classes=1000,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_cls_token=False,
                 if_devide_out=False,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=False,
                 out_indices=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.bimamba_type = bimamba_type
        
        # SMT Compatibility properties
        self.img_size = img_size
        self.channels = channels
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        
        if out_indices is None:
            self.out_indices = [depth - 1]
        else:
            self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            if bimamba_type == '2d':
                pass
            elif use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        # --- SMT Style: Calculate width_list ---
        self.width_list = []
        try:
            self.eval() 
            dummy_input = torch.randn(1, self.channels, self.img_size, self.img_size)
            # handle device
            p = next(self.parameters(), None)
            if p is not None:
                dummy_input = dummy_input.to(p.device).to(p.dtype)

            with torch.no_grad():
                features = self.forward_features(dummy_input)
            
            self.width_list = [f.size(1) for f in features]
            self.train() 
        except Exception as e:
            print(f"Error initializing width_list: {e}")
            self.width_list = [self.embed_dim] * len(self.out_indices)
            self.train()

    def forward_features(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        x = self.patch_embed(x)
        B, M, lat_dim = x.shape
        H_grid, W_grid = self.patch_embed.grid_size

        token_position = 0
        if self.if_cls_token:
            if self.bimamba_type == '2d':
                pass 
            elif self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]
            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                else:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        if if_random_token_rank:
            shuffle_indices = torch.randperm(M)
            x = x[:, shuffle_indices, :]
            if isinstance(token_position, list):
                new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
                token_position = new_token_position
            else:
                token_position = torch.where(shuffle_indices == token_position)[0].item()

        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        residual = None
        hidden_states = x
        
        outs = []

        if not self.if_bidirectional:
            for i, layer in enumerate(self.layers):
                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
                
                if i in self.out_indices:
                    curr_hidden = hidden_states
                    curr_residual = residual
                    
                    if not self.fused_add_norm:
                        if curr_residual is None:
                            curr_residual = curr_hidden
                        else:
                            curr_residual = curr_residual + self.drop_path(curr_hidden)
                        out = self.norm_f(curr_residual.to(dtype=self.norm_f.weight.dtype))
                    else:
                        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                        out = fused_add_norm_fn(
                            self.drop_path(curr_hidden),
                            self.norm_f.weight,
                            self.norm_f.bias,
                            eps=self.norm_f.eps,
                            residual=curr_residual,
                            prenorm=False,
                            residual_in_fp32=self.residual_in_fp32,
                        )
                    
                    if self.if_cls_token:
                        if self.use_double_cls_token:
                            seq = out[:, 1:-1, :] 
                        else:
                            if token_position == 0:
                                seq = out[:, 1:, :]
                            elif token_position == M - 1:
                                seq = out[:, :-1, :]
                            else:
                                seq = torch.cat((out[:, :token_position, :], out[:, token_position+1:, :]), dim=1)
                    else:
                        seq = out
                    
                    B_out, L_out, C_out = seq.shape
                    try:
                        seq_spatial = seq.reshape(B_out, H_grid, W_grid, C_out).permute(0, 3, 1, 2).contiguous()
                        outs.append(seq_spatial)
                    except Exception:
                        outs.append(seq)

        else:
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])
                
                current_logical_idx = i * 2 + 1
                if current_logical_idx in self.out_indices or (i*2) in self.out_indices:
                     curr_hidden = hidden_states
                     curr_residual = residual
                     
                     if not self.fused_add_norm:
                        if curr_residual is None:
                            curr_residual = curr_hidden
                        else:
                            curr_residual = curr_residual + self.drop_path(curr_hidden)
                        out = self.norm_f(curr_residual.to(dtype=self.norm_f.weight.dtype))
                     else:
                        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                        out = fused_add_norm_fn(
                            self.drop_path(curr_hidden),
                            self.norm_f.weight,
                            self.norm_f.bias,
                            eps=self.norm_f.eps,
                            residual=curr_residual,
                            prenorm=False,
                            residual_in_fp32=self.residual_in_fp32,
                        )
                     
                     if self.if_cls_token:
                        if self.use_double_cls_token:
                            seq = out[:, 1:-1, :]
                        else:
                            if token_position == 0:
                                seq = out[:, 1:, :]
                            elif token_position == M - 1:
                                seq = out[:, :-1, :]
                            else:
                                seq = torch.cat((out[:, :token_position, :], out[:, token_position+1:, :]), dim=1)
                     else:
                        seq = out
                     
                     B_out, L_out, C_out = seq.shape
                     try:
                        seq_spatial = seq.reshape(B_out, H_grid, W_grid, C_out).permute(0, 3, 1, 2).contiguous()
                        outs.append(seq_spatial)
                     except:
                        outs.append(seq)
        
        return outs

    def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        # Directly return the list of features for SMT/YOLO compatibility
        return self.forward_features(x, inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)

@register_model
def vim_tiny_patch16_224_bimamba2d_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    # Pass 'out_indices' to get intermediate layers if needed, or defaults to last
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="2d", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        pass
    return model

@register_model
def vim_small_patch16_224_bimamba2d_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="2d", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        pass
    return model

# --- Test Code ---
if __name__ == '__main__':
    print("--- Creating Modified VisionMamba (Tiny) for SMT/YOLO ---")
    img_size = 640
    
    # Example: Initialize requesting specific output layers (indices 0-based)
    # If using with YOLO, you likely want features from different scales.
    # Since ViM is constant scale, we just pick deeper layers.
    try:
        model = vim_tiny_patch16_224_bimamba2d_final_pool_mean_abs_pos_embed_with_midclstok_div2(
            img_size=img_size, 
            out_indices=[5, 11, 17, 23]
        )
        print("Model created successfully.")
        
        print(f"Calculated width_list: {model.width_list}")
        
        dummy_input = torch.randn(2, 3, img_size, img_size)
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
            
        print(f"\n--- Testing Forward Pass (Input: {dummy_input.shape}) ---")
        
        model.eval()
        with torch.no_grad():
            features = model(dummy_input)
            
        print(f"Output type: {type(features)}")
        if isinstance(features, list):
            print(f"Number of feature maps: {len(features)}")
            for i, f in enumerate(features):
                print(f"Feature {i} shape: {f.shape}")
                
            # Verify list operation compatibility (the error source)
            try:
                features.insert(0, None)
                print("Success: Output behaves like a list (insert allowed).")
            except AttributeError:
                print("FAIL: Output is NOT a list.")
        else:
            print("FAIL: Output is NOT a list.")

    except Exception as e:
        print(f"Critical Error during test: {e}")
        import traceback
        traceback.print_exc()