# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from einops import repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

# Assume gaterecurrent2dnoind_cuda is in a location accessible via sys.path
# For demonstration purposes, we'll check if the import works, otherwise, we'll mock it.
try:
    import gaterecurrent2dnoind_cuda
except ImportError:
    print("Warning: 'gaterecurrent2dnoind_cuda' could not be imported. Using PyTorch backend. This will be slow.")
    gaterecurrent2dnoind_cuda = None


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class GateRecurrent2dnoindFunction(Function):
		
	@staticmethod
	@torch.cuda.amp.custom_fwd
	def forward(ctx, X, B, G1, G2, G3, items_each_chunk):
		num, channels, height, width = X.size()
		output = torch.zeros(num, channels, height, width, device=X.device, dtype=X.dtype)

		ctx.hiddensize = X.size()
		ctx.items_each_chunk = items_each_chunk

		if X.is_cuda and gaterecurrent2dnoind_cuda is not None:
			gaterecurrent2dnoind_cuda.forward(items_each_chunk, X, B, G1, G2, G3, output)
		else:
			# Fallback to PyTorch implementation if CUDA extension is not available
			output = gaterecurrent2dnoind_pytorch(X, B, G1, G2, G3)

		ctx.save_for_backward(X, B, G1, G2, G3, output)

		return output

	@staticmethod
	@torch.cuda.amp.custom_bwd
	@once_differentiable
	def backward(ctx, grad_output):
		grad_output = grad_output.contiguous()
		hiddensize = ctx.hiddensize
		items_each_chunk = ctx.items_each_chunk
		X, B, G1, G2, G3, output = ctx.saved_tensors
		
		assert (hiddensize is not None and grad_output.is_cuda)
		num, channels, height, width = hiddensize

		grad_X = torch.zeros_like(X)
		grad_B = torch.zeros_like(B)
		grad_G1 = torch.zeros_like(G1)
		grad_G2 = torch.zeros_like(G2)
		grad_G3 = torch.zeros_like(G3)

		if gaterecurrent2dnoind_cuda is not None:
			gaterecurrent2dnoind_cuda.backward(items_each_chunk, output, grad_output, 
											 X, B, G1, G2, G3, 
											 grad_X, grad_B, grad_G1, grad_G2, grad_G3)
		else:
			# Note: PyTorch backward is not implemented for this custom function
			raise NotImplementedError("Backward pass for PyTorch backend of GateRecurrent2dnoindFunction not implemented.")


		return grad_X, grad_B, grad_G1, grad_G2, grad_G3, None

gaterecurrent = GateRecurrent2dnoindFunction.apply

def gaterecurrent2dnoind_pytorch(X, B, G1, G2, G3):
	"""PyTorch implementation of GateRecurrent2dnoind"""
	batch_size, channels, height, width = X.size()
	H = torch.zeros_like(X)
	
	# This is a simplified conceptual implementation. A real implementation would be more complex.
	# The logic here is illustrative.
	for w in range(width):
		for h in range(height):
			x_t = X[..., h, w]
			b_t = B[..., h, w]
			
			h_sum = 0
			if w > 0:
				if h > 0:
					h_sum += G1[..., h, w] * H[..., h-1, w-1]
				h_sum += G2[..., h, w] * H[..., h, w-1]
				if h < height-1:
					h_sum += G3[..., h, w] * H[..., h+1, w-1]
			
			H[..., h, w] = b_t * x_t + h_sum
	
	return H


class GateRecurrent2dnoind(nn.Module):
	def __init__(self, items_each_chunk_, backend='cuda'):
		super(GateRecurrent2dnoind, self).__init__()
		self.items_each_chunk = items_each_chunk_
		# If cuda extension is not available, force pytorch backend
		self.backend = backend if gaterecurrent2dnoind_cuda is not None else 'pytorch'
		if self.backend == 'cuda' and gaterecurrent2dnoind_cuda is None:
			print("Warning: CUDA backend requested but not available. Falling back to PyTorch.")
			self.backend = 'pytorch'


	def forward(self, X, B, G1, G2, G3):
		if self.backend == 'pytorch':
			return gaterecurrent2dnoind_pytorch(X, B, G1, G2, G3)
		else:  # cuda backend
			return gaterecurrent(X, B, G1, G2, G3, self.items_each_chunk)

	def __repr__(self):
		return f"{self.__class__.__name__}(backend={self.backend})"


def normalize_w(Gl, Gm, Gr, method=None): # method argument removed as it was unused
    Gl_s = torch.sigmoid(Gl)
    Gm_s = torch.sigmoid(Gm)
    Gr_s = torch.sigmoid(Gr)

    sum_s = Gl_s + Gm_s + Gr_s

    # Boundary conditions
    if sum_s.shape[2] > 1: # height > 1
        sum_s[:, :, 0, :] = Gm_s[:, :, 0, :] + Gr_s[:, :, 0, :]
        sum_s[:, :, -1, :] = Gl_s[:, :, -1, :] + Gm_s[:, :, -1, :]

    sum_s = sum_s.clamp(min=1e-7)

    return Gl_s / sum_s, Gm_s / sum_s, Gr_s / sum_s


# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        pad_h, pad_w = H % 2, W % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        pad_h, pad_w = H % 2, W % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x0 = x[..., 0::2, 0::2]
        x1 = x[..., 1::2, 0::2]
        x2 = x[..., 0::2, 1::2]
        x3 = x[..., 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], 1)
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args).contiguous()


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
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


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class GSPNv1:
    def __initgspnv1__(
        self,
        feat_size,
        items_each_chunk=8,
        d_model=96,
        ssm_ratio=2.0,
        ssm_d_state=16,
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        n_directions=4,
        channel_first=True,
        is_glayers=False,
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        self.d_state = ssm_d_state
        self.channel_first = channel_first
        self.c_group = 12
        self.n_directions = n_directions
        d_inner = int(ssm_ratio * d_model)

        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        self.forward = self.forwardv1
        self.forward_core = self.forward_corev1

        self.items_each_chunk = feat_size if is_glayers else items_each_chunk
        
        self.in_proj = Linear(d_model, d_inner, bias=bias)
        self.act: nn.Module = act_layer()
        self.d_spn = d_inner
            
        self.conv2d = nn.Conv2d(
            in_channels=self.d_spn, out_channels=self.d_spn,
            groups=self.d_spn, bias=conv_bias, kernel_size=d_conv,
            padding=(d_conv - 1) // 2, **factory_kwargs,
        )

        self.spn_core = GateRecurrent2dnoind(self.items_each_chunk)

        ks = 1
        self.x_conv_down = nn.Conv2d(self.d_spn, self.d_state, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.w_conv_up = nn.Conv2d(self.d_state, self.c_group * self.d_spn, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.l_conv_up = nn.Conv2d(self.d_state, self.n_directions * self.d_spn, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.u_conv_up = nn.Conv2d(self.d_state, self.n_directions * self.d_spn, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.d_conv = nn.Conv2d(self.d_state, self.n_directions * self.d_spn, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.m_conv = nn.Conv2d(self.n_directions, 1, kernel_size=1, bias=False)
        
        self.grn = GRN(d_inner)
        self.out_act = nn.Identity()
        self.out_norm = LayerNorm(self.d_spn)
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def spn_block(self, X, l, u, Gl, Gm, Gr, D=None, spn_module=None):
        Gl, Gm, Gr = normalize_w(Gl, Gm, Gr)
        Gl, Gm, Gr = Gl.to(X.dtype), Gm.to(X.dtype), Gr.to(X.dtype)
        
        out = spn_module(X, l, Gl, Gm, Gr)
        if D is not None:
            out = out * u + X * D
        else:
            out = out * u
        return out

    def forward_corev1(self, x: torch.Tensor=None, **kwargs):
        B, D, H, W = x.shape
        
        x_proxy = self.x_conv_down(x)
        ws = self.w_conv_up(x_proxy)
        Ls = self.l_conv_up(x_proxy).contiguous()
        Us = self.u_conv_up(x_proxy).contiguous()
        Ds = self.d_conv(x_proxy).contiguous()

        x_hwwh = torch.stack([x, x.transpose(2, 3).contiguous()], dim=1) 
        xs = torch.cat([x_hwwh, x_hwwh.flip(dims=[-1]).contiguous()], dim=1)
        xs = xs.view(B, -1, H, W).contiguous()

        Gs = torch.split(ws, D*self.n_directions, dim=1)
        G3 = [g.contiguous() for g in Gs]

        out_y = self.spn_block(xs, Ls, Us, G3[0], G3[1], G3[2], Ds, self.spn_core)

        out_y = out_y.view(B, self.n_directions, D*H, W)
        out_y = self.m_conv(out_y).view(B, D, H, W)

        y = self.out_norm(out_y)
        return y

    def forwardv1(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        x = self.act(x) # Activation is often after in_proj
        x = self.conv2d(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        y = self.grn(y)
        out = self.dropout(self.out_proj(y))
        return out

class SS2D(nn.Module, GSPNv1):
    def __init__(
        self,
        feat_size,
        items_each_chunk=8,
        d_model=96,
        ssm_ratio=2.0,
        ssm_d_state=16,
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        channel_first=False,
        is_glayers=False,
        **kwargs,
    ):
        super().__init__()
        # Pass all relevant arguments to GSPNv1 initializer
        self.__initgspnv1__(
            feat_size=feat_size, items_each_chunk=items_each_chunk, d_model=d_model, 
            ssm_ratio=ssm_ratio, ssm_d_state=ssm_d_state, act_layer=act_layer, d_conv=d_conv,
            conv_bias=conv_bias, dropout=dropout, bias=bias,
            channel_first=channel_first, is_glayers=is_glayers, **kwargs
        )


# =====================================================
class VSSBlock(nn.Module):
    def __init__(
        self,
        feat_size,
        items_each_chunk=8,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        ssm_ratio=2.0,
        ssm_d_state=16,
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v1",
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        is_glayers: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            feat_size=feat_size, 
            items_each_chunk=items_each_chunk,
            d_model=hidden_dim, 
            ssm_ratio=ssm_ratio,
            ssm_d_state=ssm_d_state,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            channel_first=channel_first,
            is_glayers=is_glayers,
            # Pass any other necessary kwargs for SS2D/GSPN
            ssm_init=ssm_init,
            forward_type=forward_type,
            **kwargs,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        _MLP = Mlp if not gmlp else gMlp
        self.norm2 = norm_layer(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        if self.post_norm:
            x = input + self.drop_path(self.norm(self.op(input)))
            x = x + self.drop_path(self.norm2(self.mlp(x)))
        else:
            x = input + self.drop_path(self.op(self.norm(input)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSM(nn.Module):
    def __init__(
        self, 
        imgsize=224,
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        items_each_chunk=8,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        forward_type="v1",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        specific_glayers=None,
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="ln2d",
        downsample_version: str = "v2",
        patchembed_version: str = "v1",
        use_checkpoint=False,  
        posembed=False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.in_chans = in_chans
        self.imgsize = imgsize

        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.dims = dims
        self.num_features = dims[-1]
        
        # Calculate feat_sizes based on imgsize and downsampling
        self.feat_sizes = []
        # The size of the feature map after the initial patch embedding
        initial_feat_size = imgsize // patch_size
        self.feat_sizes.append(initial_feat_size)
        # Sizes after each downsampling stage
        current_size = initial_feat_size
        for i in range(self.num_layers - 1):
             current_size //= 2
             self.feat_sizes.append(current_size)


        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        _NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU)

        norm_layer_mod = _NORMLAYERS.get(norm_layer.lower(), nn.LayerNorm)
        ssm_act_layer_mod = _ACTLAYERS.get(ssm_act_layer.lower(), nn.SiLU)
        mlp_act_layer_mod = _ACTLAYERS.get(mlp_act_layer.lower(), nn.GELU)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        _make_patch_embed = dict(v1=self._make_patch_embed, v2=self._make_patch_embed_v2).get(patchembed_version, self._make_patch_embed)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer_mod, channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D, v2=self._make_downsample, 
            v3=self._make_downsample_v3, none=(lambda *_, **_k: nn.Identity()),
        ).get(downsample_version, self._make_downsample)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            is_glayer_stage = specific_glayers is not None and i_layer in specific_glayers
            self.layers.append(self._make_layer(
                feat_size=self.feat_sizes[i_layer],
                items_each_chunk=items_each_chunk,
                dim=self.dims[i_layer],
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer_mod,
                downsample=_make_downsample(
                    self.dims[i_layer], self.dims[i_layer+1], norm_layer=norm_layer_mod, channel_first=self.channel_first
                ) if (i_layer < self.num_layers - 1) else nn.Identity(),
                channel_first=self.channel_first,
                ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_act_layer=ssm_act_layer_mod,
                ssm_conv=ssm_conv, ssm_conv_bias=ssm_conv_bias, ssm_drop_rate=ssm_drop_rate,
                forward_type=forward_type, mlp_ratio=mlp_ratio, mlp_act_layer=mlp_act_layer_mod,
                mlp_drop_rate=mlp_drop_rate, gmlp=gmlp, is_glayers=is_glayer_stage,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer_mod(self.num_features),
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)
        
        # === Add width_list calculation ===
        self.width_list = []
        try:
            self.eval()
            dummy_input = torch.randn(1, self.in_chans, self.imgsize, self.imgsize)
            with torch.no_grad():
                features = self.forward_features(dummy_input) # Use forward_features
            # The output of forward_features is a list of tensors from each stage
            self.width_list = [f.size(1) for f in features]
            self.train()
        except Exception as e:
            print(f"Error during dummy forward pass for width_list: {e}")
            self.width_list = self.dims # Fallback
            self.train()

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        layers = [nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)]
        if not channel_first:
            layers.append(Permute(0, 2, 3, 1))
        if patch_norm:
            layers.append(norm_layer(embed_dim))
        return nn.Sequential(*layers)


    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        stride = patch_size // 2
        layers = [
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=stride, padding=1),
        ]
        if not channel_first: layers.append(Permute(0, 2, 3, 1))
        if patch_norm: layers.append(norm_layer(embed_dim // 2))
        if not channel_first: layers.append(Permute(0, 3, 1, 2))
        layers.extend([
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=stride, padding=1),
        ])
        if not channel_first: layers.append(Permute(0, 2, 3, 1))
        if patch_norm: layers.append(norm_layer(embed_dim))
        return nn.Sequential(*layers)
    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        return nn.Sequential(
            (Permute(0, 3, 1, 2) if not channel_first else nn.Identity()),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (Permute(0, 2, 3, 1) if not channel_first else nn.Identity()),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        return nn.Sequential(
            (Permute(0, 3, 1, 2) if not channel_first else nn.Identity()),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if not channel_first else nn.Identity()),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        feat_size, items_each_chunk, dim, depth, drop_path, use_checkpoint, 
        norm_layer, downsample, channel_first, **kwargs
    ):
        # Correctly handle the 'is_glayers' argument to avoid the TypeError
        is_glayer_stage = kwargs.pop('is_glayers', False)
        
        blocks = []
        for d in range(depth):
            # The logic for determining if a specific block is a "glayer" block
            is_glayer_block = is_glayer_stage and (d < 7 or (d+1)%7==0)
            
            blocks.append(VSSBlock(
                feat_size=feat_size,
                items_each_chunk=items_each_chunk,
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                use_checkpoint=use_checkpoint,
                is_glayers=is_glayer_block,  # Pass the correctly calculated flag
                **kwargs                      # Pass the rest of the kwargs
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks),
            downsample=downsample,
        ))

    def forward_features(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1).contiguous() if not self.channel_first else self.pos_embed
            x = x + pos_embed
        
        outputs = []
        for i, layer in enumerate(self.layers):
            # Pass the input through the 'blocks' part of the layer first
            x_blocks = layer.blocks(x)
            
            # The output of the blocks is the feature map for this stage
            # This should be appended before downsampling
            # But need to handle channel_first logic correctly if it's not
            outputs.append(x_blocks)
            
            # Then apply the downsampling to prepare for the next stage
            x = layer.downsample(x_blocks)
            
        return outputs

    def forward_head(self, x: torch.Tensor):
        # This function provides the original classification capability
        # Assumes x is the output of the last feature stage
        return self.classifier(x)

    def forward(self, x: torch.Tensor):
        # Modified to return a list of feature maps for use as a backbone
        # This aligns with the behavior of SMT in Code 2
        features = self.forward_features(x)
        # To avoid the AttributeError: 'Tensor' object has no attribute 'insert'
        # we must ensure the output is a list, which forward_features does.
        return features

# --- Factory Functions ---

def gspn_tiny(pretrained=False, img_size=224, **kwargs):
    model = VSSM(
        imgsize=img_size,
        depths=[2, 2, 7, 2],
        dims=96, # Will be expanded to [96, 192, 384, 768]
        ssm_d_state=8,
        ssm_ratio=1.5,
        ssm_conv=7,
        ssm_conv_bias=False,
        mlp_ratio=4.0,
        items_each_chunk=2,
        patch_size=4,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        ssm_act_layer="silu",
        forward_type="v1",
        specific_glayers=[2, 3],
        drop_path_rate=0.2,
        **kwargs)
    return model

def gspn_small(pretrained=False, img_size=224, **kwargs):
    model = VSSM(
        imgsize=img_size,
        depths=[5, 5, 10, 2],
        dims=108, # Will be expanded
        ssm_d_state=16,
        ssm_ratio=1.5,
        ssm_conv=7,
        ssm_conv_bias=False,
        mlp_ratio=4.0,
        items_each_chunk=2,
        patch_size=4,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        ssm_act_layer="silu",
        forward_type="v1",
        specific_glayers=[3],
        drop_path_rate=0.4,
        **kwargs)
    return model

def gspn_base(pretrained=False, img_size=224, **kwargs):
    model = VSSM(
        imgsize=img_size,
        depths=[4, 4, 15, 4],
        dims=120, # Will be expanded
        ssm_d_state=8,
        ssm_ratio=1.5,
        ssm_conv=7,
        ssm_conv_bias=False,
        mlp_ratio=4.0,
        items_each_chunk=2,
        patch_size=4,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        ssm_act_layer="silu",
        forward_type="v1",
        specific_glayers=[2, 3],
        drop_path_rate=0.5,
        **kwargs)
    return model


if __name__ == '__main__':
    img_h, img_w = 640, 640
    print("--- Creating VSSM Tiny model ---")
    model = gspn_tiny(img_size=img_h)
    print("Model created successfully.")
    print("Calculated width_list:", model.width_list)

    # Test forward pass
    input_tensor = torch.rand(2, 3, img_h, img_w)
    print(f"\n--- Testing VSSM Tiny forward pass (Input: {input_tensor.shape}) ---")

    model.eval()
    try:
        with torch.no_grad():
            output_features = model(input_tensor)
        print("Forward pass successful.")
        print("Output is a list of feature maps as required.")
        print("Output feature shapes:")
        for i, features in enumerate(output_features):
            print(f"Stage {i+1}: {features.shape}")

        # Verify width_list matches runtime output
        runtime_widths = [f.size(1) for f in output_features]
        print("\nRuntime output feature channels:", runtime_widths)
        assert model.width_list == runtime_widths, "Width list mismatch!"
        print("Width list verified successfully.")

        # Test deepcopy
        print("\n--- Testing deepcopy ---")
        copied_model = copy.deepcopy(model)
        print("Deepcopy successful.")
        with torch.no_grad():
             output_copied = copied_model(input_tensor)
        print("Copied model forward pass successful.")
        assert len(output_copied) == len(output_features)
        for i in range(len(output_features)):
             assert output_copied[i].shape == output_features[i].shape
        print("Copied model output shapes verified.")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()