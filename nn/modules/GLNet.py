from functools import partial
import math
from typing import Iterable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.checkpoint import checkpoint_wrapper # May not be needed if not using fairscale
from timm.models.registry import register_model
from timm.models.layers import DropPath, LayerNorm2d, trunc_normal_


_glnet_ckpt_urls= {
    'glnet_4g': ('https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5RzlEalNfQU1xbkpRaVgxP2U9dE9raEhQ/root/content', 'model'),
    'glnet_9g': ('https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5RUdGUTZrZldWLXdWWmVpP2U9d1d3Ujh6/root/content', 'model'),
    'glnet_16g': ('https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5amFCeEMzaC1COENIV01tP2U9R2ZSMGtn/root/content', 'model_ema'), # Uses model_ema
    'glnet_stl': ('https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5QkZhQUlMRU11X2R0YmJWP2U9OUdoaGkz/root/content', 'model'),
    'glnet_stl_paramslot': ('https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5RzBlYXQ1ekNFOXVwY1FSP2U9dm1ibW8x/root/content', 'model'),
}

# Removed _load_from_url as pretrained logic will be in each model function

class ResDWConvNCHW(nn.Conv2d):
    def __init__(self, dim, ks:int=3) -> None:
        super().__init__(dim, dim, ks, 1, padding=ks//2, bias=True, groups=dim)

    def forward(self, x:torch.Tensor):
        res = super().forward(x)
        return x + res

class LayerScale(nn.Module):
    def __init__(self, chans, init_value=1e-4, in_format='nlc') -> None:
        super().__init__()
        assert in_format in {'nlc', 'nchw'}
        if in_format == 'nlc':
            self.gamma = nn.Parameter(torch.ones((chans))*init_value, requires_grad=True)
        else: # nchw
            self.gamma = nn.Parameter(torch.ones((1, chans, 1, 1))*init_value, requires_grad=True)

    def forward(self, x:torch.Tensor):
        return self.gamma * x

class MHSA_Block(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, dropout=0.,
                 mlp_ratio:float=4., drop_path:float=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 layerscale=-1) -> None:
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.mha_op = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(mlp_ratio*embed_dim)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio*embed_dim), embed_dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ls1 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nlc') if layerscale > 0 else nn.Identity()
        self.ls2 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nlc') if layerscale > 0 else nn.Identity()

    def forward(self, x:torch.Tensor):
        shortcut = x
        x_norm1 = self.norm1(x)
        # FIXME: below is just a workaround
        if not self.training and x_norm1.dtype != shortcut.dtype : # Check if types mismatch
             x_norm1 = x_norm1.to(shortcut.dtype)
        
        # If MultiheadAttention is in a mixed precision context and expects float16
        # ensure inputs are correctly cast if needed by the environment.
        # However, typically, autocast handles this. The explicit cast here is a bit unusual.
        # Let's assume for now that if types are consistent, it should work.
        # If 'x' is float16 from autocast, x_norm1 might become float32 due to LayerNorm's internal ops
        # unless LayerNorm is also AMP-aware or its params are float16.
        
        # If x_norm1 is float32 and mha_op expects float16 (e.g. due to model.half()), this could be an issue.
        # The original code's fix suggests this scenario.
        # For robustness, let's keep the explicit cast but make it conditional on type mismatch
        # if not self.training and x_norm1.dtype != self.mha_op.in_proj_weight.dtype:
        #    x_norm1 = x_norm1.to(self.mha_op.in_proj_weight.dtype)

        attn_output, attn_weights = self.mha_op(query=x_norm1, key=x_norm1, value=x_norm1, need_weights=False)
        x = shortcut + self.drop_path(self.ls1(attn_output))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x

class GLMixBlock(nn.Module):
    def __init__(self,
        embed_dim:int,
        num_heads:int,
        num_slots:int=64,
        slot_init:str='ada_avgpool',
        slot_scale_val:float=None, # Renamed from slot_scale to avoid conflict with self.slot_scale
        scale_mode='learnable',
        local_dw_ks:int=5,
        mlp_ratio:float=4.,
        drop_path:float=0.,
        norm_layer=LayerNorm2d,
        cpe_ks:int=0, 
        mlp_dw:bool=False,
        layerscale=-1,
        use_slot_attention:bool=True,
        ) -> None:

        super().__init__()
        self.embed_dim = embed_dim
        _slot_scale_init = slot_scale_val or embed_dim ** (-0.5) # Use the passed value
        self.scale_mode = scale_mode
        self.use_slot_attention = use_slot_attention
        assert scale_mode in {'learnable', 'const'}
        if scale_mode  == 'learnable':
            self.slot_scale = nn.Parameter(torch.tensor(_slot_scale_init))
        else: # const
            self.register_buffer('slot_scale', torch.tensor(_slot_scale_init))

        self.with_conv_pos_emb = (cpe_ks > 0)
        if self.with_conv_pos_emb:
            self.pos_conv = nn.Conv2d(
                embed_dim, embed_dim,
                kernel_size=cpe_ks,
                padding=cpe_ks//2, groups=embed_dim)

        assert slot_init in {'param', 'ada_avgpool'}
        self.slot_init = slot_init
        if self.slot_init == 'param':
            self.init_slots_param = nn.Parameter( # Renamed to avoid conflict if init_slots is used as var
                torch.empty(1, num_slots, embed_dim), True)
            torch.nn.init.normal_(self.init_slots_param, std=.02)
        else:
            self.pool_size = math.isqrt(num_slots)
            assert self.pool_size**2 == num_slots
        
        self.norm1 = norm_layer(embed_dim)
        self.relation_mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True) if use_slot_attention else nn.Identity()
        self.feature_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1), 
            nn.Conv2d(embed_dim, embed_dim, local_dw_ks, padding=local_dw_ks//2, groups=embed_dim), 
            nn.Conv2d(embed_dim, embed_dim, 1), 
        ) if local_dw_ks > 0 else nn.Identity()
        
        self.norm2 = norm_layer(embed_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, int(mlp_ratio*embed_dim), kernel_size=1),
            ResDWConvNCHW(int(mlp_ratio*embed_dim),ks=3) if mlp_dw else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(int(mlp_ratio*embed_dim), embed_dim, kernel_size=1)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ls1 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nchw') \
            if layerscale > 0 else nn.Identity()
        self.ls2 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nchw') \
            if layerscale > 0 else nn.Identity()
        self.vis_proxy = nn.Identity()
        
    def _forward_relation(self, x:torch.Tensor, current_init_slots:torch.Tensor): # Renamed arg
        # The original fix for slot_scale.to(x.dtype) might still be relevant in mixed precision
        # For now, let's assume slot_scale is correctly handled by its Parameter/buffer nature.
        # slot_s = self.slot_scale
        # if not self.training and x.dtype != self.slot_scale.dtype:
        #     slot_s = self.slot_scale.to(x.dtype)


        x_flatten = x.permute(0, 2, 3, 1).flatten(1, 2) # (bs, l, c)
        
        # Ensure current_init_slots and x_flatten are compatible for matmul, esp. in mixed precision
        # If current_init_slots is float32 (from param) and x_flatten is float16 (from data)
        # The matmul might upcast or require explicit casting.
        # For now, let's assume types are handled by autocast or are consistent.

        # Original: F.normalize(current_init_slots, p=2, dim=-1) @ (self.slot_scale * F.normalize(x_flatten, p=2, dim=-1).transpose(-1, -2))
        # slot_scale is a scalar or (1,) tensor.
        # x_flatten is (bs, l, c)
        # current_init_slots is (bs, num_slots, c) or (1, num_slots, c)

        norm_init_slots = F.normalize(current_init_slots, p=2, dim=-1) # (bs, num_slots, c)
        norm_x_flatten_T = F.normalize(x_flatten, p=2, dim=-1).transpose(-1, -2) # (bs, c, l)
        
        logits = norm_init_slots @ (self.slot_scale * norm_x_flatten_T) # (bs, num_slots, l)
        
        slots = torch.softmax(logits, dim=-1) @ x_flatten # (bs, num_slots, c)
        
        if self.use_slot_attention:
            # Ensure slots are of the expected dtype for relation_mha
            # if not self.training and slots.dtype != self.relation_mha.in_proj_weight.dtype:
            #    slots = slots.to(self.relation_mha.in_proj_weight.dtype)
            slots, attn_weights = self.relation_mha(query=slots, key=slots, value=slots, need_weights=False)
        else: # if not use_slot_attention
            attn_weights = None # Or some placeholder if needed for vis_proxy

        if not self.training: 
            logits, attn_weights = self.vis_proxy((logits, attn_weights))

        out = torch.softmax(logits.transpose(-1, -2), dim=-1) @ slots 
        out = out.permute(0, 2, 1).reshape_as(x) 
        out = out + self.feature_conv(x)
        return out, slots

    def forward(self, x:torch.Tensor):
        current_init_slots = None # Define to ensure it's always assigned
        if self.slot_init == 'ada_avgpool':
            current_init_slots = F.adaptive_avg_pool2d(x, output_size=self.pool_size
                ).permute(0, 2, 3, 1).flatten(1, 2) # (bs, num_slots, c)
        else: # 'param'
            current_init_slots = self.init_slots_param.expand(x.size(0), -1, -1) # (bs, num_slots, c)
        
        # Ensure dtypes match if x is float16 and current_init_slots is float32 (from param)
        if x.dtype != current_init_slots.dtype:
            current_init_slots = current_init_slots.to(x.dtype)

        if self.with_conv_pos_emb:
            x = x + self.pos_conv(x)

        shortcut = x
        # Pass norm1(x) which might be float32, and current_init_slots which might be float16
        # _forward_relation needs to handle this. Best if norm1(x) output matches x.dtype
        normed_x = self.norm1(x)
        if normed_x.dtype != current_init_slots.dtype and self.slot_init == 'param': # Be careful with ada_avgpool
             # This can happen if norm_layer (e.g. LayerNorm2d) upcasts to float32 for stability
             # while current_init_slots was cast to x.dtype (e.g. float16)
             # We need them to be compatible for F.normalize and matmul
             # Let's ensure normed_x is also cast to the prevalent dtype if necessary
            if x.dtype != normed_x.dtype: # If x was float16, norm output float32, cast back
                normed_x = normed_x.to(x.dtype)


        x_rel, updt_slots = self._forward_relation(normed_x, current_init_slots)
        x = shortcut + self.drop_path(self.ls1(x_rel))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x
    
    def extra_repr(self) -> str:
        return f"scale_mode={self.scale_mode}, "\
               f"slot_init={self.slot_init}, "\
               f"use_slot_attention={self.use_slot_attention}"

class MHSA_NCHW_Block(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, dropout=0.,
                 mlp_ratio:float=4., drop_path:float=0., norm_layer=LayerNorm2d,
                 mlp_dw:bool=False, cpe_ks:int=0, 
                 layerscale=-1,
                ) -> None:
        super().__init__()
        self.with_conv_pos_emb = (cpe_ks > 0)
        if self.with_conv_pos_emb:
            self.pos_conv = nn.Conv2d(
                embed_dim, embed_dim,
                kernel_size=cpe_ks,
                padding=cpe_ks//2, groups=embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6) # This is channels-last norm
        self.mha_op = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout)
        self.norm2 = norm_layer(embed_dim) # This is channels-first (e.g. LayerNorm2d or BatchNorm2d)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, int(mlp_ratio*embed_dim), kernel_size=1),
            ResDWConvNCHW(int(mlp_ratio*embed_dim),ks=3) if mlp_dw else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(int(mlp_ratio*embed_dim), embed_dim, kernel_size=1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ls1 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nlc') if layerscale > 0 else nn.Identity()
        self.ls2 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nchw') if layerscale > 0 else nn.Identity()

    def forward(self, x:torch.Tensor):
        if self.with_conv_pos_emb:
            x = x + self.pos_conv(x)

        nchw_shape = x.size()
        x_permuted = x.permute(0, 2, 3, 1).flatten(1, 2) # (bs, h*w, c)

        shortcut = x_permuted
        x_norm1 = self.norm1(x_permuted) # norm1 is nn.LayerNorm (channels-last)

        # FIXME for mixed precision
        if not self.training and x_norm1.dtype != shortcut.dtype:
            x_norm1 = x_norm1.to(shortcut.dtype)
        # if not self.training and x_norm1.dtype != self.mha_op.in_proj_weight.dtype:
        #    x_norm1 = x_norm1.to(self.mha_op.in_proj_weight.dtype)


        attn_output, attn_weights = self.mha_op(query=x_norm1, key=x_norm1, value=x_norm1, need_weights=False)
        x_attn = shortcut + self.drop_path(self.ls1(attn_output))
        
        x_reshaped = x_attn.permute(0, 2, 1).reshape(nchw_shape) # Back to NCHW
        # norm2 is channels-first (e.g., LayerNorm2d)
        x_mlp_out = self.mlp(self.norm2(x_reshaped)) 
        x = x_reshaped + self.drop_path(self.ls2(x_mlp_out))
        return x

class BasicLayer(nn.Module):
    def __init__(self, 
        dim, num_heads, depth:int,
        mlp_ratio=4., drop_path=0.,
        mixing_mode='glmix', 
        local_dw_ks=5, 
        slot_init:str='ada_avgpool', 
        num_slots:int=64, 
        use_slot_attention:bool=True, # Added for GLMixBlock
        norm_layer=LayerNorm2d, # Added for GLMixBlock/MHSA_NCHW_Block
        cpe_ks:int=0,
        mlp_dw:bool=False,
        layerscale=-1
        ):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mixing_mode = mixing_mode

        if self.mixing_mode == 'mha':
            # Original MHSA_Block expects nn.LayerNorm (channels-last) by default
            # If your GLNet variant using 'mha' needs channels-first norm from timm's LayerNorm2d,
            # this part would need adjustment or ensure norm_layer is appropriately passed.
            # For simplicity, assuming it uses the default nn.LayerNorm for MHSA_Block
            ln_partial = partial(nn.LayerNorm, eps=1e-6)
            self.blocks = nn.ModuleList([
                MHSA_Block(
                    embed_dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=ln_partial, layerscale=layerscale 
                ) for i in range(depth)])
        elif self.mixing_mode == 'mha_nchw':
            self.blocks = nn.ModuleList([
                MHSA_NCHW_Block(
                    embed_dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer, cpe_ks=cpe_ks, mlp_dw=mlp_dw, layerscale=layerscale
                ) for i in range(depth)])
        elif self.mixing_mode == 'glmix':
            self.blocks = nn.ModuleList([
                GLMixBlock(
                    embed_dim=dim, num_heads=num_heads, num_slots=num_slots, slot_init=slot_init,
                    slot_scale_val=dim**(-0.5), # Default from GLMixBlock
                    local_dw_ks=local_dw_ks, use_slot_attention=use_slot_attention,
                    mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    cpe_ks=cpe_ks, mlp_dw=mlp_dw, layerscale=layerscale
                )for i in range(depth)])
        elif self.mixing_mode == 'glmix.mha_nchw': # hybrid
            ln_partial_mha_nchw = norm_layer # For MHSA_NCHW_Block
            ln_partial_glmix = norm_layer    # For GLMixBlock
            self.blocks = nn.ModuleList()
            for i in range(depth):
                current_dp = drop_path[i] if isinstance(drop_path, list) else drop_path
                if i % 2 == 0: # GLMixBlock
                    block = GLMixBlock(
                        embed_dim=dim, num_heads=num_heads, num_slots=num_slots, slot_init=slot_init,
                        slot_scale_val=dim**(-0.5), local_dw_ks=local_dw_ks, use_slot_attention=use_slot_attention,
                        mlp_ratio=mlp_ratio, norm_layer=ln_partial_glmix, drop_path=current_dp,
                        cpe_ks=cpe_ks, mlp_dw=mlp_dw, layerscale=layerscale
                    )
                else: # MHSA_NCHW_Block
                    block = MHSA_NCHW_Block(
                        embed_dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                        drop_path=current_dp, norm_layer=ln_partial_mha_nchw,
                        cpe_ks=cpe_ks, mlp_dw=mlp_dw, layerscale=layerscale
                    )
                self.blocks.append(block)
        else:
            raise ValueError(f'Unknown mixing_mode: {self.mixing_mode}')

    def forward(self, x:torch.Tensor):
        if self.mixing_mode == 'mha': # MHSA_Block expects NLC input
            nchw_shape = x.size()
            x = x.permute(0, 2, 3, 1).flatten(1, 2) # NCHW to NLC
            for blk in self.blocks:
                x = blk(x) 
            x = x.transpose(1, 2).reshape(nchw_shape) # NLC to NCHW
        else: # glmix, mha_nchw, glmix.mha_nchw expect NCHW
            for blk in self.blocks:
                x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}, mixing_mode={self.mixing_mode}"


class NonOverlappedPatchEmbeddings(nn.ModuleList):
    def __init__(self, embed_dims:Iterable[int], in_chans=3,
                       midd_order='norm.proj',
                       norm_layer=nn.BatchNorm2d) -> None:
        assert midd_order in {'norm.proj', 'proj.norm'}
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims[0],kernel_size=(4, 4), stride=(4, 4)),
            norm_layer(embed_dims[0])
        )
        modules = [stem]
        for i in range(3):
            if midd_order == 'norm.proj':
                transition = nn.Sequential(
                    norm_layer(embed_dims[i]), 
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=(2, 2), stride=(2, 2)),
                )
            else:
                transition = nn.Sequential(
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=(2, 2), stride=(2, 2)),
                    norm_layer(embed_dims[i+1])
                )
            modules.append(transition)
        super().__init__(modules)


class OverlappedPacthEmbeddings(nn.ModuleList):
    def __init__(self, embed_dims:Iterable[int],
        in_chans=3,
        deep_stem=True, # This was not used in GLNet, but keeping for potential future use
        dual_patch_norm=False,
        midd_order='proj.norm',
        norm_layer=nn.BatchNorm2d
        ) -> None:
        assert midd_order in {'norm.proj', 'proj.norm'}
        # GLNet's OverlappedPacthEmbeddings's stem was equivalent to deep_stem=True
        # For compatibility, let's keep the GLNet's original stem structure
        stem_layers = []
        if dual_patch_norm: # This was part of GLNet's original definition
             stem_layers.append(LayerNorm2d(in_chans)) # LayerNorm2d is channels_first
        
        stem_layers.extend([
            nn.Conv2d(in_chans, embed_dims[0] // 2, kernel_size=3, stride=2, padding=1),
            norm_layer(embed_dims[0] // 2), # norm_layer is channels_first
            nn.GELU(),
            nn.Conv2d(embed_dims[0] // 2, embed_dims[0], kernel_size=3, stride=2, padding=1),
            norm_layer(embed_dims[0]) # norm_layer is channels_first
        ])
        stem = nn.Sequential(*stem_layers)
        
        modules = [stem]
        for i in range(3):
            if midd_order == 'norm.proj':
                transition = nn.Sequential(
                    norm_layer(embed_dims[i]), 
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=3, stride=2, padding=1),
                )
            else:
                transition = nn.Sequential(
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=3, stride=2, padding=1),
                    norm_layer(embed_dims[i+1])
                )
            modules.append(transition)
        super().__init__(modules)


class GLNet(nn.Module):
    def __init__(self,
        in_chans=3,
        num_classes=1000, # Set to 0 for backbone usage
        depth=[2, 2, 6, 2],
        embed_dim=[96, 192, 384, 768],
        head_dim=32, 
        # qk_scale=None, # Not directly used in GLNet blocks, but can be passed via kwargs
        drop_path_rate=0., 
        # drop_rate=0., # Not directly used in GLNet blocks, but can be passed via kwargs
        use_checkpoint_stages=[],
        mlp_ratios=[4, 4, 4, 4],
        norm_layer=LayerNorm2d,
        pre_head_norm_layer=None,
        mixing_modes=('glmix', 'glmix', 'glmix', 'mha'),
        local_dw_ks=5, 
        slot_init:str='param',
        num_slots:int=64, 
        use_slot_attention:bool=True, # Added for BasicLayer's GLMixBlock
        cpe_ks:int=0,
        downsample_style:str='non_ovlp',
        transition_layout:str='proj.norm',
        dual_patch_norm:bool=False,
        mlp_dw:bool=False,
        layerscale:float=-1.,
        input_image_size:int=224, # For width_list calculation
        **kwargs # Accepts unused_kwargs
        ):
        super().__init__()
        if kwargs: # Print unused kwargs like original
            print(f"Unused kwargs in GLNet initialization: {kwargs}")

        self.num_classes = num_classes
        self.embed_dim_list = embed_dim # Keep original name for consistency if used elsewhere

        ############ downsample layers (patch embeddings) ######################
        assert downsample_style in {'non_ovlp', 'ovlp'}
        if downsample_style=='ovlp':
            self.downsample_layers = OverlappedPacthEmbeddings(
                embed_dims=embed_dim, in_chans=in_chans, norm_layer=norm_layer,
                midd_order=transition_layout,
                dual_patch_norm=dual_patch_norm
                # deep_stem is implicitly True by GLNet's original design
                )
        else:
            self.downsample_layers = NonOverlappedPatchEmbeddings(
                embed_dims=embed_dim, in_chans=in_chans, norm_layer=norm_layer,
                midd_order=transition_layout
            )
        
        self.stages = nn.ModuleList() 
        nheads= [dim // head_dim for dim in embed_dim]
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        
        # Ensure local_dw_ks is a list of 4 elements
        if isinstance(local_dw_ks, int):
            local_dw_ks_list = [local_dw_ks] * 4
        else:
            assert len(local_dw_ks) == 4, "local_dw_ks must be an int or a list of 4 ints"
            local_dw_ks_list = local_dw_ks

        current_dp_idx = 0
        for i in range(4):
            stage_dp_rates = dp_rates[current_dp_idx : current_dp_idx + depth[i]]
            current_dp_idx += depth[i]
            
            stage = BasicLayer(
                dim=embed_dim[i],
                depth=depth[i],
                num_heads=nheads[i], 
                mlp_ratio=mlp_ratios[i],
                drop_path=stage_dp_rates, # Pass list of DPs for this stage
                mixing_mode=mixing_modes[i],
                local_dw_ks=local_dw_ks_list[i],
                slot_init=slot_init,
                num_slots=num_slots,
                use_slot_attention=use_slot_attention, # Pass to BasicLayer
                norm_layer=norm_layer, # Pass to BasicLayer
                cpe_ks=cpe_ks,
                mlp_dw=mlp_dw,
                layerscale=layerscale
            )
            if i in use_checkpoint_stages:
                # Ensure fairscale is available or handle its absence
                try:
                    stage = checkpoint_wrapper(stage)
                except NameError:
                    print("Warning: fairscale.nn.checkpoint.checkpoint_wrapper not available. Skipping checkpointing.")
            self.stages.append(stage)

        # Final norm and head for classification (optional)
        if self.num_classes > 0:
            _pre_head_norm_layer = pre_head_norm_layer or norm_layer 
            self.norm = _pre_head_norm_layer(embed_dim[-1]) # Applied on NCHW
            self.head = nn.Linear(embed_dim[-1], num_classes)
        else:
            self.norm = None
            self.head = None

        self.apply(self._init_weights)
        if self.head is not None and hasattr(self.head, 'weight'): # Initialize head like ConvNeXt
             trunc_normal_(self.head.weight, std=.02) # Standard init
             # head_init_scale could be a parameter, defaulting to 1.0
             # self.head.weight.data.mul_(head_init_scale)
             if self.head.bias is not None:
                 nn.init.constant_(self.head.bias, 0)
                 # self.head.bias.data.mul_(head_init_scale)


        # --- Calculate width_list ---
        self.width_list = []
        if input_image_size > 0 : # Check to allow disabling this for some reason
            self.eval() # Set to eval mode
            with torch.no_grad():
                # Create a dummy input on CPU, as model might not be on CUDA yet
                # Or ensure model parameters are on the same device as dummy_input if moved early.
                # For __init__, CPU is safest unless device is explicitly passed and model moved.
                dummy_input = torch.randn(1, in_chans, input_image_size, input_image_size)
                try:
                    dummy_outputs = self.forward(dummy_input) # This now returns a list
                    if isinstance(dummy_outputs, list):
                        self.width_list = [f.size(1) for f in dummy_outputs]
                    else: # Should not happen if forward is correct
                        print("Warning: Dummy forward pass for width_list did not return a list.")
                        # Fallback or error, for now, make it empty or based on embed_dim
                        self.width_list = embed_dim # Approximate, not from actual forward pass
                except Exception as e:
                    print(f"Error during dummy forward pass for width_list: {e}")
                    print("width_list will be based on embed_dim configuration, may not be accurate.")
                    self.width_list = embed_dim # Fallback
            self.train() # Set back to train mode
        else: # If no input_image_size, use embed_dim as a fallback
            self.width_list = embed_dim
        # -----------------------------

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm2d, nn.BatchNorm2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        # Conv2d weights are often initialized by their own specific schemes (e.g. kaiming)
        # but GLNet's original _init_weights didn't specify for Conv2d, so timm's trunc_normal
        # would apply if Linear, otherwise default. This is generally fine.

    @torch.jit.ignore
    def no_weight_decay(self): # As per original
        return {'pos_embed', 'cls_token', 'init_slots', 'init_slots_param'} # Added init_slots_param

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''): # global_pool unused by this arch
        self.num_classes = num_classes
        if num_classes > 0:
            # Re-initialize norm if it was None
            if self.norm is None:
                 _pre_head_norm_layer = getattr(self, 'pre_head_norm_layer_ref', LayerNorm2d) # Store ref or use default
                 self.norm = _pre_head_norm_layer(self.embed_dim_list[-1])

            self.head = nn.Linear(self.embed_dim_list[-1], num_classes)
            self._init_weights(self.head) # Re-initialize new head
        else:
            self.norm = None
            self.head = None

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        """
        Forward pass for classification. Assumes x is the output of the last stage.
        Input x is (N, C, H, W) from the last stage.
        """
        if self.norm is not None:
            x = self.norm(x) # NCHW norm
        
        # Global average pooling
        x = x.mean([2, 3]) # (N, C)
        
        if pre_logits:
            return x
        
        if self.head is not None:
            x = self.head(x)
        return x

    def forward(self, x:torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass as a backbone, returning a list of feature maps from each stage.
        """
        outputs: List[torch.Tensor] = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outputs.append(x)
        return outputs


# --- Model Registration Functions ---

def _load_glnet_pretrained(model: GLNet, ckpt_key_in_dict: str, url: str, progress: bool = True, check_hash: bool = True):
    """Helper to load pretrained weights, handling head layer mismatches."""
    try:
        state_dict = torch.hub.load_state_dict_from_url(
            url,
            map_location='cpu',
            progress=progress,
            check_hash=check_hash,
            file_name=f"{ckpt_key_in_dict.split('.')[0]}.pth" # Use a base name for the file
        )
        if ckpt_key_in_dict: # e.g. 'model' or 'model_ema'
             actual_state_dict = state_dict.get(ckpt_key_in_dict, None)
             if actual_state_dict is None: # Check if state_dict is already the model's state_dict
                  if 'head.weight' in state_dict or 'stages.0.blocks.0.norm1.weight' in state_dict: # Heuristic
                       actual_state_dict = state_dict
                  else:
                       raise KeyError(f"Key '{ckpt_key_in_dict}' not found in checkpoint and checkpoint doesn't seem to be a raw state_dict.")
        else: # Raw state_dict expected
             actual_state_dict = state_dict
        
        # Prune head if model's head is None (num_classes=0) or classes mismatch
        if model.head is None:
            if 'head.weight' in actual_state_dict:
                del actual_state_dict['head.weight']
            if 'head.bias' in actual_state_dict:
                del actual_state_dict['head.bias']
            # Also prune final norm if it's tied to classification head logic
            if 'norm.weight' in actual_state_dict: # check if 'norm' is final layer norm
                is_final_norm = True
                for name in actual_state_dict: # Heuristic: if no layer name contains 'norm.' then it's likely final
                    if 'norm.' in name and name != 'norm.weight' and name != 'norm.bias':
                        is_final_norm = False
                        break
                if is_final_norm:
                    del actual_state_dict['norm.weight']
                    if 'norm.bias' in actual_state_dict:
                        del actual_state_dict['norm.bias']

        elif model.head is not None and 'head.weight' in actual_state_dict:
            if model.head.weight.shape != actual_state_dict['head.weight'].shape:
                print(f"Classifier head mismatch: model has {model.head.weight.shape}, checkpoint has {actual_state_dict['head.weight'].shape}. Removing checkpoint head.")
                del actual_state_dict['head.weight']
                if 'head.bias' in actual_state_dict:
                    del actual_state_dict['head.bias']
                # If head is removed, norm might also need to be if its size depends on head's existence
                # but GLNet's self.norm is based on embed_dim[-1], so it should be fine.

        model.load_state_dict(actual_state_dict, strict=False)
        print(f"Pretrained weights loaded for {ckpt_key_in_dict.split('.')[0]} from {url.split('/')[-2]}")

    except Exception as e:
        print(f"Failed to load pretrained weights for {ckpt_key_in_dict.split('.')[0]}: {e}")


@register_model
def glnet_stl(pretrained=False, **kwargs):
    # Default num_classes to 0 for backbone usage, but allow override via kwargs
    if 'num_classes' not in kwargs:
        kwargs['num_classes'] = 0 
        
    model = GLNet(
        depth=[2, 2, 6, 2],
        embed_dim=[96, 192, 384, 768],
        mlp_ratios=[4, 4, 4, 4],
        head_dim=32,
        norm_layer=nn.BatchNorm2d,
        mixing_modes=('glmix', 'glmix', 'glmix', 'mha'),
        local_dw_ks=5,
        slot_init='ada_avgpool',
        num_slots=64,
        transition_layout='norm.proj',
        **kwargs)
    if pretrained:
        url, ckpt_key = _glnet_ckpt_urls['glnet_stl']
        _load_glnet_pretrained(model, ckpt_key, url)
    return model

@register_model
def glnet_stl_paramslot(pretrained=False, **kwargs):
    if 'num_classes' not in kwargs:
        kwargs['num_classes'] = 0
    model = GLNet(
        depth=[2, 2, 6, 2],
        embed_dim=[96, 192, 384, 768],
        mlp_ratios=[4, 4, 4, 4],
        head_dim=32,
        norm_layer=nn.BatchNorm2d,
        mixing_modes=('glmix', 'glmix', 'glmix', 'mha'),
        local_dw_ks=5,
        slot_init='param', # Key difference
        num_slots=64,
        transition_layout='norm.proj',
        **kwargs)
    if pretrained:
        url, ckpt_key = _glnet_ckpt_urls['glnet_stl_paramslot']
        _load_glnet_pretrained(model, ckpt_key, url)
    return model

@register_model
def glnet_4g(pretrained=False, **kwargs):
    if 'num_classes' not in kwargs:
        kwargs['num_classes'] = 0
    model = GLNet(
        depth=[4, 4, 18, 4],
        embed_dim=[64, 128, 256, 512],
        mlp_ratios=[3, 3, 3, 3],
        head_dim=32,
        norm_layer=nn.BatchNorm2d,
        mixing_modes=('glmix', 'glmix', 'glmix.mha_nchw', 'mha_nchw'),
        local_dw_ks=5,
        slot_init='ada_avgpool',
        num_slots=64,
        cpe_ks=3,
        downsample_style='ovlp',
        transition_layout='proj.norm',
        mlp_dw=True,
        **kwargs)
    if pretrained:
        url, ckpt_key = _glnet_ckpt_urls['glnet_4g']
        _load_glnet_pretrained(model, ckpt_key, url)
    return model

@register_model
def glnet_9g(pretrained=False, **kwargs):
    if 'num_classes' not in kwargs:
        kwargs['num_classes'] = 0
    model = GLNet(
        depth=[4, 4, 18, 4],
        embed_dim=[96, 192, 384, 768],
        mlp_ratios=[3, 3, 3, 3],
        head_dim=32,
        norm_layer=nn.BatchNorm2d,
        mixing_modes=('glmix', 'glmix', 'glmix.mha_nchw', 'mha_nchw'),
        local_dw_ks=5,
        slot_init='ada_avgpool',
        num_slots=64,
        cpe_ks=3,
        downsample_style='ovlp',
        transition_layout='proj.norm',
        mlp_dw=True,
        **kwargs)
    if pretrained:
        url, ckpt_key = _glnet_ckpt_urls['glnet_9g']
        _load_glnet_pretrained(model, ckpt_key, url)
    return model

@register_model
def glnet_16g(pretrained=False, **kwargs):
    if 'num_classes' not in kwargs:
        kwargs['num_classes'] = 0
    model = GLNet(
        depth=[4, 4, 18, 4],
        embed_dim=[128, 256, 512, 1024],
        mlp_ratios=[3, 3, 3, 3],
        head_dim=32,
        norm_layer=nn.BatchNorm2d,
        mixing_modes=('glmix', 'glmix', 'glmix.mha_nchw', 'mha_nchw'),
        local_dw_ks=5,
        slot_init='ada_avgpool',
        num_slots=64,
        cpe_ks=3,
        downsample_style='ovlp',
        transition_layout='proj.norm',
        mlp_dw=True,
        layerscale=1e-4, # Specific to 16g
        **kwargs)
    if pretrained:
        url, ckpt_key = _glnet_ckpt_urls['glnet_16g'] # Uses 'model_ema'
        _load_glnet_pretrained(model, ckpt_key, url)
    return model

# Example Usage (for testing)
if __name__ == '__main__':
    # Test backbone usage (num_classes=0 implicitly by default in new functions)
    print("Testing GLNet-STL as backbone:")
    model_stl_backbone = glnet_stl(pretrained=False, input_image_size=224)
    dummy_input = torch.randn(2, 3, 224, 224)
    features_list = model_stl_backbone(dummy_input)
    print(f"Output is a list: {isinstance(features_list, list)}")
    print(f"Number of feature maps: {len(features_list)}")
    for i, f_map in enumerate(features_list):
        print(f"Feature map {i} shape: {f_map.shape}")
    print(f"Model width_list: {model_stl_backbone.width_list}")
    print("-" * 30)

    # Test with classification head
    print("Testing GLNet-4G with classification head (10 classes):")
    model_4g_clf = glnet_4g(pretrained=False, num_classes=10, input_image_size=256) # Specify num_classes
    dummy_input_256 = torch.randn(2, 3, 256, 256)
    
    # To get classification output, you'd call forward_head after the backbone forward
    features_4g = model_4g_clf(dummy_input_256) # Returns list of features
    last_stage_features = features_4g[-1]       # Get the last feature map
    logits = model_4g_clf.forward_head(last_stage_features) # Pass to classification head
    print(f"Logits shape: {logits.shape}")
    print(f"Model width_list: {model_4g_clf.width_list}")
    print("-" * 30)

    # Test pretraining load (will print errors if URLs are inaccessible or download fails)
    # Set num_classes=0 to test backbone weight loading
    print("Testing GLNet-STL with pretrained weights (as backbone):")
    try:
        model_stl_pretrained = glnet_stl(pretrained=True, num_classes=0, input_image_size=224)
        print("Pretrained GLNet-STL (backbone) loaded successfully.")
        features_stl_pt = model_stl_pretrained(dummy_input)
        print(f"Feature map 0 shape from pretrained: {features_stl_pt[0].shape}")
    except Exception as e:
        print(f"Could not load pretrained GLNet-STL: {e}")
    print("-" * 30)

    print("Testing GLNet-16G with pretrained weights (as backbone, uses model_ema):")
    try:
        model_16g_pretrained = glnet_16g(pretrained=True, num_classes=0, input_image_size=224)
        print("Pretrained GLNet-16G (backbone) loaded successfully.")
        features_16g_pt = model_16g_pretrained(dummy_input)
        print(f"Feature map 0 shape from pretrained: {features_16g_pt[0].shape}")

    except Exception as e:
        print(f"Could not load pretrained GLNet-16G: {e}")
    print("-" * 30)

    # Test MHSA block standalone with mixed precision context (simulated)
    print("Testing MHSA_Block potential mixed precision fix:")
    mhsa_block_test = MHSA_Block(embed_dim=64, num_heads=4)
    # Simulate mixed precision: input is float16, LayerNorm might output float32
    test_input_nlc = torch.randn(2, 16, 64).half() # NLC format, float16
    
    # Manually set LayerNorm to output float32 to simulate worst-case for MHA
    # This is tricky to do without modifying LayerNorm internals.
    # The fix in MHSA_Block tries to convert norm output back to input's dtype IF not training.
    # If using torch.cuda.amp.autocast, it should handle types.
    
    # Test with training=False to trigger the fix
    mhsa_block_test.eval() 
    # If MHA layer itself is .half(), its weights are float16.
    # mhsa_block_test.mha_op.half() # Simulate MHA expecting float16
    
    # If the mha_op expects float16 due to .half() or autocast context,
    # and norm1(test_input_nlc) becomes float32, the fix should cast it back to float16.
    try:
        with torch.cuda.amp.autocast(enabled=True): # Simulate AMP context
             # Move model and data to CUDA if testing autocast properly
             if torch.cuda.is_available():
                mhsa_block_test.cuda()
                test_input_nlc = test_input_nlc.cuda()
                print("Running MHSA_Block test on CUDA with autocast.")
             else:
                print("CUDA not available, MHSA_Block autocast test will run on CPU (autocast has no effect).")

             out_mhsa = mhsa_block_test(test_input_nlc)
             print(f"MHSA_Block output dtype: {out_mhsa.dtype}, shape: {out_mhsa.shape}")
    except Exception as e:
        print(f"Error during MHSA_Block test: {e}")
    
    print("Done with tests.")