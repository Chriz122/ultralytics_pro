import math
from functools import partial

import torch
from einops import rearrange, reduce, repeat
from torchvision.ops import DeformConv2d
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers.activations import * # Imports GELU, Sigmoid, SiLU, etc.
from timm.models.layers import DropPath, trunc_normal_ # LayerNorm2d might come from here if not defined locally
                                                # Actually, timm.layers.LayerNorm2d is not exported by default by __init__.py

inplace = True

# Define MODEL registry if it's not available (placeholder)
# In a real scenario, this would come from a framework like mmcv or ultralytics
class ModelRegistry:
    def __init__(self):
        self._module_dict = {}

    def register_module(self, cls=None, name=None, force=False):
        if cls is None:
            return partial(self.register_module, name=name, force=force)

        if name is None:
            name = cls.__name__
        if not force and name in self._module_dict:
            raise KeyError(f'{name} is already registered in {self.__class__.__name__}')
        self._module_dict[name] = cls
        return cls

    def get(self, name):
        if name not in self._module_dict:
            raise KeyError(f'{name} is not registered in {self.__class__.__name__}')
        return self._module_dict[name]

MODEL = ModelRegistry()


# Copied from Code 2 for LayerNorm2d, as Code 1's definition was missing/commented
class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        # In EMO2, norm_layer='ln_2d' is used with embed_dims which are channels.
        # nn.LayerNorm expects normalized_shape as the last Dims.
        # For [B, C, H, W], if we want to normalize over C, H, W, normalized_shape = [C, H, W]
        # If we want to normalize over C (channel-wise like BatchNorm), normalized_shape = C
        # Timm's LayerNorm2d normalizes over the C dimension, expecting (B, C, H, W)
        # Code 2's LayerNorm2d permutes to (B, H, W, C) and applies LayerNorm over C.
        # This is equivalent to `nn.LayerNorm(normalized_shape, ...)` if normalized_shape is an int (num_channels)
        # and data_format="channels_first".
        # PyTorch LayerNorm with a single int for normalized_shape normalizes the last dimension.
        # So, (B,H,W,C) with LN(C) is correct for channel-wise LN.
        self.normalized_shape = normalized_shape
        self.norm = nn.LayerNorm(self.normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        # EMO2's ln_2d is applied on (B, C, H, W) tensors, where C is embed_dim
        # So, we want to normalize along the C dimension.
        x = x.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() # B, C, H, W
        return x

class LayerNorm3d(nn.Module): # Placeholder, not used by EMO2 configs
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

class BatchNorm2ds(nn.Module): # Placeholder, not used by EMO2 configs
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim, eps=eps)
    def forward(self, x, dim_in=None):
        # This signature is for compatibility with ConvNormAct's norm call
        return self.norm(x)


class DCN2(nn.Module):
	# ref: https://github.com/WenmuZhou/DBNet.pytorch/blob/678b2ae55e018c6c16d5ac182558517a154a91ed/models/backbone/resnet.py
	def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False, deform_groups=4):
		super().__init__()
		offset_channels = kernel_size * kernel_size * 2
		self.conv_offset = nn.Conv2d(dim_in, deform_groups * offset_channels, kernel_size=3, stride=stride, padding=1)
		self.conv = DeformConv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

	def forward(self, x):
		offset = self.conv_offset(x)
		x = self.conv(x, offset)
		return x


class Conv2ds(nn.Conv2d):
	def __init__(self, dim_in, dim_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
				 padding_mode='zeros', device=None, dtype=None):
		super().__init__(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
		self.in_channels_i = dim_in
		self.out_channels_i = dim_out
		self.groups_i = groups

	def forward(self, x, dim_in=None, dim_out=None):
		self.groups = dim_in if self.groups_i != 1 else self.groups_i
		in_channels = dim_in if dim_in else self.in_channels_i
		out_channels = dim_out if dim_out else self.out_channels_i
		weight = self.weight[:out_channels, :in_channels, :, :]
		bias = self.bias[:out_channels] if self.bias is not None else self.bias
		return self._conv_forward(x, weight, bias)


def get_conv(conv_layer='conv_2d'):
	conv_dict = {
		'conv_2d': nn.Conv2d,
		'conv_3d': nn.Conv3d,
		'dcn2_2d': DCN2,
		# 'dcn2_2d_mmcv': DeformConv2dPack,
		'conv_2ds': Conv2ds,
	}
	return conv_dict[conv_layer]


def get_norm(norm_layer='in_1d'):
	eps = 1e-6
	norm_dict = {
		'none': nn.Identity,
		'in_1d': partial(nn.InstanceNorm1d, eps=eps),
		'in_2d': partial(nn.InstanceNorm2d, eps=eps),
		'in_3d': partial(nn.InstanceNorm3d, eps=eps),
		'bn_1d': partial(nn.BatchNorm1d, eps=eps),
		'bn_2d': partial(nn.BatchNorm2d, eps=eps),
		# 'bn_2d': partial(nn.SyncBatchNorm, eps=eps), # SyncBatchNorm needs distributed setup
		'bn_3d': partial(nn.BatchNorm3d, eps=eps),
		'gn': partial(nn.GroupNorm, eps=eps),
		'ln_1d': partial(nn.LayerNorm, eps=eps),
		'ln_2d': partial(LayerNorm2d, eps=eps), # Uses the LayerNorm2d defined above
		'ln_3d': partial(LayerNorm3d, eps=eps), # Uses the LayerNorm3d defined above
		'bn_2ds': partial(BatchNorm2ds, eps=eps), # Uses the BatchNorm2ds defined above
	}
	return norm_dict[norm_layer]


def get_act(act_layer='relu'):
	act_dict = {
		'none': nn.Identity,
		'sigmoid': Sigmoid,
		'swish': Swish, # Swish == SiLU
		'mish': Mish,
		'hsigmoid': HardSigmoid,
		'hswish': HardSwish,
		'hmish': HardMish,
		'tanh': Tanh,
		'relu': nn.ReLU,
		'relu6': nn.ReLU6,
		'prelu': PReLU,
		'gelu': GELU, # timm.layers.GELU
		'silu': nn.SiLU # torch.nn.SiLU
	}
	return act_dict[act_layer]


class ConvNormAct(nn.Module):
	def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False, padding_mode='zeros', skip=False, conv_layer='conv_2d', norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
		super(ConvNormAct, self).__init__()
		self.conv_layer = conv_layer
		self.norm_layer = norm_layer
		self.has_skip = skip and dim_in == dim_out and stride == 1 # Skip connection only if dim and stride match

		# Calculate padding
		if isinstance(kernel_size, (list, tuple)): # For non-square kernels if supported
			padding = [math.ceil(((k - 1) * dilation + 1 - s) / 2) for k, s in zip(kernel_size, stride if isinstance(stride, (list,tuple)) else [stride]*len(kernel_size) )]
		else: # For square kernels
			padding = math.ceil(((kernel_size - 1) * dilation + 1 - (stride if isinstance(stride, int) else stride[0])) / 2)

		if conv_layer in ['conv_2d', 'conv_2ds', 'conv_3d']:
			self.conv = get_conv(conv_layer)(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_mode)
		elif conv_layer in ['dcn2_2d', 'dcn2_2d_mmcv']: # dcn2_2d_mmcv would need DeformConv2dPack
			self.conv = get_conv(conv_layer)(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, deform_groups=4, bias=bias) # padding for dcn might need to be 'same' or calculated carefully
		
		self.norm = get_norm(norm_layer)(dim_out)
		# For activations like ReLU, ReLU6, PReLU, GELU, SiLU, inplace can be passed if the class supports it.
		# For classes from timm.layers.activations, they often handle inplace internally or via a param.
		try:
			self.act = get_act(act_layer)(inplace=inplace)
		except TypeError: # Some activations (e.g. nn.Identity) don't take inplace
			self.act = get_act(act_layer)()
            
		self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
	
	def forward(self, x, dim_in=None, dim_out=None): # dim_in/out for Conv2ds
		shortcut = x
		if self.conv_layer in ['conv_2ds']:
			x = self.conv(x, dim_in=dim_in, dim_out=dim_out)
		else:
			x = self.conv(x)
		
		if self.norm_layer in ['bn_2ds']:
			x = self.norm(x, dim_in=dim_out) # Assuming bn_2ds takes dim_in for some reason
		else:
			x = self.norm(x)
		x = self.act(x)

		if self.has_skip:
			x = self.drop_path(x) + shortcut
		return x


class LayerScale2D(nn.Module):
	def __init__(self, dim, init_values=1e-5, inplace=True):
		super().__init__()
		self.inplace = inplace
		self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))
	
	def forward(self, x):
		return x.mul_(self.gamma) if self.inplace else x * self.gamma

# ========== basic modules and ops ==========
def get_stem(dim_in, dim_mid): # dim_mid is emb_dim_pre
	stem = nn.ModuleList([
		ConvNormAct(dim_in, dim_mid, kernel_size=3, stride=2, bias=True, norm_layer='bn_2d', act_layer='silu'),
		ConvNormAct(dim_mid, dim_mid, kernel_size=3, stride=1, groups=dim_mid, bias=False, norm_layer='bn_2d', act_layer='silu'),
		ConvNormAct(dim_mid, dim_mid, kernel_size=1, stride=1, bias=False, norm_layer='none', act_layer='none'),
	])
	return stem

# --> conv
class Conv(nn.Module):
	def __init__(self, dim_in, dim_mid, kernel_size=1, groups=1, bias=False, norm_layer='bn_2d', act_layer='relu', inplace=True):
		super().__init__()
		self.net = ConvNormAct(dim_in, dim_mid, kernel_size=kernel_size, groups=groups, bias=bias, norm_layer=norm_layer,
							   act_layer=act_layer, inplace=inplace)
	def forward(self, x):
		return self.net(x)

# --> sa - remote
class EW_MHSA_Remote(nn.Module):
	def __init__(self, dim_in, dim_mid, norm_layer='bn_2d', act_layer='relu', dim_head=64, window_size=7,
				 qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False, ls_value=1e-6): # ls_value not used here directly
		super().__init__()
		self.dim_head = dim_head
		self.window_size = window_size
		if dim_in % dim_head != 0:
			raise ValueError(f"dim_in ({dim_in}) must be divisible by dim_head ({dim_head})")
		self.num_head = dim_in // dim_head
		self.scale = self.dim_head ** -0.5
		self.attn_pre = attn_pre
		self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none', act_layer='none')
		self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias, norm_layer='none', act_layer=act_layer, inplace=inplace)
		self.attn_drop = nn.Dropout(attn_drop)

	def forward(self, x):
		B, C, H, W = x.shape
		if self.window_size <= 0:
			window_size_W, window_size_H = W, H
		else:
			window_size_W, window_size_H = self.window_size, self.window_size
		pad_l, pad_t = 0, 0
		pad_r = (window_size_W - W % window_size_W) % window_size_W
		pad_b = (window_size_H - H % window_size_H) % window_size_H
		x_padded = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,)) # Pad C last, B last to last
		
		N1, N2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
		x_windowed = rearrange(x_padded, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=N1, n2=N2).contiguous()

		b, c, h, w = x_windowed.shape
		qk = self.qk(x_windowed)
		qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head,
					   dim_head=self.dim_head).contiguous()
		q, k = qk[0], qk[1]
		attn_map = (q @ k.transpose(-2, -1)) * self.scale
		attn_map = attn_map.softmax(dim=-1)
		attn_map = self.attn_drop(attn_map)

		if self.attn_pre:
			# x_windowed here should be the input to V, which is also x_windowed
			val_pre = rearrange(x_windowed, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
			x_spa = attn_map @ val_pre
			x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
			x_spa = self.v(x_spa) # Apply V projection after attention
		else:
			v = self.v(x_windowed) # Apply V projection first
			v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
			x_spa = attn_map @ v
			x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()

		x_unwindowed = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=N1, n2=N2).contiguous()
		if pad_r > 0 or pad_b > 0:
			x_unwindowed = x_unwindowed[:, :, :H, :W].contiguous()
		return x_unwindowed


# --> sa - close
class EW_MHSA_Close(nn.Module):
	def __init__(self, dim_in, dim_mid, norm_layer='bn_2d', act_layer='relu', dim_head=64, window_size=7,
				 qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False, ls_value=1e-6):
		super().__init__()
		self.dim_head = dim_head
		self.window_size = window_size
		if dim_in % dim_head != 0:
			raise ValueError(f"dim_in ({dim_in}) must be divisible by dim_head ({dim_head})")
		self.num_head = dim_in // dim_head
		self.scale = self.dim_head ** -0.5
		self.attn_pre = attn_pre
		self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none', act_layer='none')
		self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias, norm_layer='none', act_layer=act_layer, inplace=inplace)
		self.attn_drop = nn.Dropout(attn_drop)

	def forward(self, x):
		B, C, H, W = x.shape
		if self.window_size <= 0:
			window_size_W, window_size_H = W, H
		else:
			window_size_W, window_size_H = self.window_size, self.window_size
		pad_l, pad_t = 0, 0
		pad_r = (window_size_W - W % window_size_W) % window_size_W
		pad_b = (window_size_H - H % window_size_H) % window_size_H
		x_padded = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
		
		N1, N2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
		# Corrected rearrange for close/grid attention: (n1 h1) (n2 w1) -> (b h1 w1) c n1 n2 for example
		# The original 'b c (n1 h1) (n2 w1) -> (b n1 n2) c h1 w1' means each (h1xw1) window is processed independently. This is window attention.
		# "Close" might refer to Swin-like shifted window or axial. The current is standard window attention.
		# Let's assume the einops pattern was intended for standard window attention, similar to Remote.
		x_windowed = rearrange(x_padded, 'b c (n1 h1) (n2 w1) -> (b n1 n2) c h1 w1', n1=N1, h1=window_size_H, n2=N2, w1=window_size_W).contiguous()


		b, c, h, w = x_windowed.shape # Here h=window_size_H, w=window_size_W
		qk = self.qk(x_windowed)
		qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()
		q, k = qk[0], qk[1]
		attn_map = (q @ k.transpose(-2, -1)) * self.scale
		attn_map = attn_map.softmax(dim=-1)
		attn_map = self.attn_drop(attn_map)

		if self.attn_pre:
			val_pre = rearrange(x_windowed, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
			x_spa = attn_map @ val_pre
			x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
			x_spa = self.v(x_spa)
		else:
			v = self.v(x_windowed)
			v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
			x_spa = attn_map @ v
			x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
		
		x_unwindowed = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (n1 h1) (n2 w1)', n1=N1, n2=N2, h1=h, w1=w).contiguous()

		if pad_r > 0 or pad_b > 0:
			x_unwindowed = x_unwindowed[:, :, :H, :W].contiguous()
		return x_unwindowed

class EW_MHSA_Hybrid(nn.Module):
	def __init__(self, dim_in, dim_mid, norm_layer='bn_2d', act_layer='relu', dim_head=64, window_size=7,
				 qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False, ls_value=1e-6):
		super().__init__()
		self.dim_head = dim_head
		self.window_size = window_size
		if dim_in % dim_head != 0:
			raise ValueError(f"dim_in ({dim_in}) must be divisible by dim_head ({dim_head})")
		self.num_head = dim_in // dim_head
		self.scale = self.dim_head ** -0.5
		self.attn_pre = attn_pre
		
		self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none', act_layer='none')
		self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias, norm_layer='none', act_layer=act_layer, inplace=inplace)
		self.attn_drop = nn.Dropout(attn_drop)

	def forward(self, x):
		B, C, H, W = x.shape
		if self.window_size <= 0:
			window_size_W, window_size_H = W, H
		else:
			window_size_W, window_size_H = self.window_size, self.window_size
		
		pad_l, pad_t = 0, 0
		pad_r = (window_size_W - W % window_size_W) % window_size_W
		pad_b = (window_size_H - H % window_size_H) % window_size_H
		x_padded = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
		
		Hp, Wp = H + pad_b, W + pad_r
		N1_remote, N2_remote = Hp // window_size_H, Wp // window_size_W # Num windows for remote path
		# Remote: (b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1) where h1=H/N1, w1=W/N2 (patch size for global attention)
		# This means h1 is window_size_H, w1 is window_size_W for remote based on original code.
		# x_remote = rearrange(x_padded, 'b c (h_patch n1) (w_patch n2) -> (b n1 n2) c h_patch w_patch', n1=N1_remote, n2=N2_remote).contiguous()
		x_remote_windowed = rearrange(x_padded, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=N1_remote, h1=window_size_H, n2=N2_remote, w1=window_size_W).contiguous()


		# Close: (b c (n1 h1) (n2 w1) -> (b n1 n2) c h1 w1) where h1=window_size, w1=window_size
		N1_close, N2_close = Hp // window_size_H, Wp // window_size_W # Num windows for close path
		x_close_windowed = rearrange(x_padded, 'b c (n1 h1) (n2 w1) -> (b n1 n2) c h1 w1', n1=N1_close, h1=window_size_H, n2=N2_close, w1=window_size_W).contiguous()
        # Note: The original patterns for remote and close were different.
        # 'b c (h1 n1) (w1 n2)' implies h1 = Hp / N1_remote, w1 = Wp / N2_remote. Here h1, w1 are window sizes.
        # 'b c (n1 h1) (n2 w1)' implies h1 = window_size_H, w1 = window_size_W.
        # Both seem to be standard window attention if h1,w1 are window_size.
        # For hybrid, the distinction might be how QK are derived or how V is projected.
        # Current implementation: both x_remote_windowed and x_close_windowed are identical window partitions.
        # The difference then must come from how QK are computed from the original x_padded vs windowed x.

        # QK processing: from original padded input
		qk_orig = self.qk(x_padded) # (B, 2C, Hp, Wp)

        # QK for remote path
		qk_remote_windowed = rearrange(qk_orig, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=N1_remote, h1=window_size_H, n2=N2_remote, w1=window_size_W).contiguous()
        # QK for close path
		qk_close_windowed = rearrange(qk_orig, 'b c (n1 h1) (n2 w1) -> (b n1 n2) c h1 w1', n1=N1_close, h1=window_size_H, n2=N2_close, w1=window_size_W).contiguous()
        
		b_w, c_qk, h_w, w_w = qk_remote_windowed.shape # (B*N1*N2, 2*C_head*num_head, win_H, win_W)

		_qk_remote = rearrange(qk_remote_windowed, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()
		_qk_close = rearrange(qk_close_windowed, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()

		attn_map_remote = (_qk_remote[0] @ _qk_remote[1].transpose(-2, -1)) * self.scale
		attn_map_remote = attn_map_remote.softmax(dim=-1)
		attn_map_remote = self.attn_drop(attn_map_remote)

		attn_map_close = (_qk_close[0] @ _qk_close[1].transpose(-2, -1)) * self.scale
		attn_map_close = attn_map_close.softmax(dim=-1)
		attn_map_close = self.attn_drop(attn_map_close)

		if self.attn_pre:
			# Values from x_remote_windowed and x_close_windowed
			val_remote_pre = rearrange(x_remote_windowed, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
			x_spa_remote_attended = attn_map_remote @ val_remote_pre
			x_spa_remote_attended = rearrange(x_spa_remote_attended, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h_w, w=w_w).contiguous()
			# Unwindow remote
			x_spa_remote_unwindowed = rearrange(x_spa_remote_attended, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=N1_remote, n2=N2_remote).contiguous()


			val_close_pre = rearrange(x_close_windowed, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
			x_spa_close_attended = attn_map_close @ val_close_pre
			x_spa_close_attended = rearrange(x_spa_close_attended, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h_w, w=w_w).contiguous()
			# Unwindow close
			x_spa_close_unwindowed = rearrange(x_spa_close_attended, '(b n1 n2) c h1 w1 -> b c (n1 h1) (n2 w1)', n1=N1_close, n2=N2_close).contiguous()

			x_spa_sum = x_spa_remote_unwindowed + x_spa_close_unwindowed # Summed in original spatial domain (Hp, Wp)
			x_spa = self.v(x_spa_sum) # V projection on the sum
		else:
			# V from original x_padded
			v_orig = self.v(x_padded) # (B, C_mid, Hp, Wp)

			# V for remote path
			v_remote_windowed = rearrange(v_orig, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=N1_remote, h1=window_size_H, n2=N2_remote, w1=window_size_W).contiguous()
			v_remote_windowed = rearrange(v_remote_windowed, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous() # Assuming C_mid is multiple of num_head*dim_head
			x_spa_remote_attended = attn_map_remote @ v_remote_windowed
			x_spa_remote_attended = rearrange(x_spa_remote_attended, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h_w, w=w_w).contiguous()
			x_spa_remote_unwindowed = rearrange(x_spa_remote_attended, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=N1_remote, n2=N2_remote).contiguous()

			# V for close path
			v_close_windowed = rearrange(v_orig, 'b c (n1 h1) (n2 w1) -> (b n1 n2) c h1 w1', n1=N1_close, h1=window_size_H, n2=N2_close, w1=window_size_W).contiguous()
			v_close_windowed = rearrange(v_close_windowed, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
			x_spa_close_attended = attn_map_close @ v_close_windowed
			x_spa_close_attended = rearrange(x_spa_close_attended, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h_w, w=w_w).contiguous()
			x_spa_close_unwindowed = rearrange(x_spa_close_attended, '(b n1 n2) c h1 w1 -> b c (n1 h1) (n2 w1)', n1=N1_close, n2=N2_close).contiguous()
			
			x_spa = x_spa_remote_unwindowed + x_spa_close_unwindowed

		if pad_r > 0 or pad_b > 0:
			x_spa = x_spa[:, :, :H, :W].contiguous()
		return x_spa


class iiRMB(nn.Module):
	def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
				 act_layer='relu', dw_ks=3, stride=1, dim_head=64, window_size=7, hybrid_eops=[0], conv_ks=1, conv_groups=1, qkv_bias=False,
				 attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False, ls_value=1e-6):
		super().__init__()
		self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
		self.dw_ks = dw_ks # Store dw_ks for use in forward pass logic
		dim_mid = int(dim_in * exp_ratio)

		self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
		self.hybrid_eops = hybrid_eops
		eops_list = []
		for eop_idx in self.hybrid_eops:
			if eop_idx == 0: 
				eop = Conv(dim_in, dim_mid, kernel_size=conv_ks, groups=conv_groups, bias=qkv_bias, norm_layer='none', act_layer=act_layer, inplace=inplace)
			elif eop_idx == 1: 
				eop = EW_MHSA_Remote(dim_in, dim_mid, norm_layer=norm_layer, act_layer=act_layer, dim_head=dim_head, window_size=window_size,
				 qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, drop_path=drop_path, v_group=v_group, attn_pre=attn_pre, ls_value=ls_value)
			elif eop_idx == 2: 
				eop = EW_MHSA_Close(dim_in, dim_mid, norm_layer=norm_layer, act_layer=act_layer, dim_head=dim_head, window_size=window_size,
				 qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, drop_path=drop_path, v_group=v_group, attn_pre=attn_pre, ls_value=ls_value)
			elif eop_idx == 3: 
				eop = EW_MHSA_Hybrid(dim_in, dim_mid, norm_layer=norm_layer, act_layer=act_layer, dim_head=dim_head, window_size=window_size,
				 qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, drop_path=drop_path, v_group=v_group, attn_pre=attn_pre, ls_value=ls_value)
			else:
				eop = None 
			if eop:
				eops_list.append(eop)
		self.eops = nn.ModuleList(eops_list)
		
        # Initialize conv_local, downsample_after_eops, and stride_for_dw_conv
		if self.dw_ks > 0: # Depthwise convolution part
			self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
			self.downsample_after_eops = nn.Identity() # Not used if dw_ks > 0
			# Determine stride_for_dw_conv from conv_local
			conv_stride_attr = self.conv_local.conv.stride # nn.Conv2d.stride is a tuple (sH, sW)
			self.stride_for_dw_conv = conv_stride_attr[0] 
		else: # No DW conv (dw_ks == 0)
			self.conv_local = nn.Identity()
			if stride != 1: # If no DW conv but stride is needed, use an explicit downsampling
				self.downsample_after_eops = nn.MaxPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()
				if hasattr(self.downsample_after_eops, 'stride'): # e.g. MaxPool2d
					pool_stride_attr = self.downsample_after_eops.stride
					self.stride_for_dw_conv = pool_stride_attr[0] if isinstance(pool_stride_attr, tuple) else pool_stride_attr
				else: # e.g. nn.Identity if stride was 1 and MaxPool2d was not created
					self.stride_for_dw_conv = 1
			else: # stride == 1
				self.downsample_after_eops = nn.Identity()
				self.stride_for_dw_conv = 1

		self.proj_drop = nn.Dropout(drop)
		self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)
		self.ls = LayerScale2D(dim_out, init_values=ls_value) if ls_value > 0 else nn.Identity()
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
	
	def forward(self, x):
		shortcut = x
		x_normed = self.norm(x)

		xs_eops = []
		for eop_module in self.eops:
			xs_eops.append(eop_module(x_normed))
		
		x_eops_out = sum(xs_eops) if len(self.eops) > 0 else x_normed

		# The problematic commented-out block (which the traceback might be referring to if the on-disk file is different)
		# # if self.dw_ks > 0:
		# #	x_l_path = self.conv_local(x_eops_out) # Stride applied by conv_local
		# #	if self.stride_for_dw_conv == 1: # If conv_local didn't change H,W
		# #		x_combined = x_eops_out + x_l_path
		# #	else:
		# #		x_combined = x_l_path # If conv_local strided, can't add to x_eops_out
		# # else: # No dw_ks
		# #	x_combined = self.downsample_after_eops(x_eops_out) # Stride applied if any

        # Block that calculated self.stride_for_dw_conv is now removed from here.
        # self.stride_for_dw_conv is now an instance attribute from __init__.

		x_after_eops_and_local = x_eops_out # Start with eops output
		if not isinstance(self.conv_local, nn.Identity): # If there's a conv_local (i.e. self.dw_ks > 0)
			x_l_val = self.conv_local(x_eops_out) # conv_local applies stride
			if self.stride_for_dw_conv == 1: # Can only add if conv_local didn't change H,W (stride was 1)
				x_after_eops_and_local = x_eops_out + x_l_val
			else: # If conv_local strided, it becomes the main path
				x_after_eops_and_local = x_l_val
		elif not isinstance(self.downsample_after_eops, nn.Identity): # No conv_local (self.dw_ks == 0), but explicit downsampling needed
			x_after_eops_and_local = self.downsample_after_eops(x_eops_out)
        # else: x_after_eops_and_local remains x_eops_out (dw_ks == 0 and stride was 1, so no downsampling)

		x_projected = self.proj(self.proj_drop(x_after_eops_and_local))

		if self.has_skip: 
			out = shortcut + self.drop_path(self.ls(x_projected))
		else:
			out = self.ls(x_projected)
		return out


class EMO2(nn.Module):
	def __init__(self,
				 dim_in=3, num_classes=1000, img_size=224,
				 depths=[1, 2, 4, 2],
				 embed_dims=[64, 128, 256, 512],
				 exp_ratios=[4., 4., 4., 4.],
				 norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'],
				 act_layers=['silu', 'silu', 'gelu', 'gelu'],
				 dw_kss=[3, 3, 5, 5], # dw_kss[i] used for all blocks in stage i+1, except first block uses 5 if dw_kss[i]=0
				 dim_heads=[32, 32, 32, 32], # dim_heads[i] for stage i+1
				 window_sizes=[7, 7, 7, 7],
				 hybrid_eopss=[[0], [0], [1], [1]], # hybrid_eopss[i] for non-first blocks in stage i+1
				 conv_kss=[1, 1, 1, 1],
				 conv_groupss=[1, 1, 1, 1],
				 qkv_bias=True, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False, ls_value=1e-6):
		super().__init__()
		self.num_classes = num_classes
		self.dim_in = dim_in
		self.img_size = img_size
		self.embed_dims = embed_dims # Store for reference or later use

		dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
		
		# emb_dim_pre is the output channels of the stem, and input to the first block of stage1
		# emb_dim_pre = embed_dims[0] // 2
		# First iiRMB in stage1: iiRMB(emb_dim_pre, embed_dims[0], ..., dim_head=dim_heads[0], ...)
		# This means emb_dim_pre must be divisible by dim_heads[0].
		if (embed_dims[0] // 2) % dim_heads[0] != 0:
			raise ValueError(f"Stem output channels (embed_dims[0]//2 = {embed_dims[0]//2}) "
							 f"must be divisible by dim_heads[0] ({dim_heads[0]})")
		emb_dim_pre = embed_dims[0] // 2
		
		self.stage0 = get_stem(dim_in, emb_dim_pre)
		
		current_dim = emb_dim_pre
		for i in range(len(depths)): # iterates 4 times for 4 stages
			# Check divisibility for current stage
			if embed_dims[i] % dim_heads[i] != 0:
				raise ValueError(f"embed_dims[{i}] ({embed_dims[i]}) must be divisible by "
								 f"dim_heads[{i}] ({dim_heads[i]}) for stage {i+1}")

			stage_layers = []
			stage_dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
			for j in range(depths[i]):
				is_first_block_of_stage = (j == 0)
				
				block_stride = 2 if is_first_block_of_stage else 1
				block_has_skip = False if is_first_block_of_stage else True # Skip only if stride=1 and dims match (dim_out is embed_dims[i])
				if is_first_block_of_stage and current_dim == embed_dims[i]: # if first block downsamples but in_dim == out_dim
					block_has_skip = False # No skip because stride is 2
				elif not is_first_block_of_stage and current_dim != embed_dims[i]: # Should not happen if current_dim is updated correctly
					block_has_skip = False


				block_hybrid_eops = [0] if is_first_block_of_stage else hybrid_eopss[i]
				block_exp_ratio = exp_ratios[i] * 2 if is_first_block_of_stage else exp_ratios[i] # First block has larger expansion
				block_conv_ks = 1 if is_first_block_of_stage else conv_kss[i]
				block_conv_groups = 1 if is_first_block_of_stage else conv_groupss[i]
				
				# dw_ks logic: specific for first block vs rest
				if is_first_block_of_stage:
					block_dw_ks = dw_kss[i] if dw_kss[i] > 0 else 5 # Default to 5 for first block if stage dw_ks is 0
				else:
					block_dw_ks = dw_kss[i]

				stage_layers.append(iiRMB(
					dim_in=current_dim,
					dim_out=embed_dims[i],
					norm_in=True, # Norm before block
					has_skip=block_has_skip,
					exp_ratio=block_exp_ratio,
					norm_layer=norm_layers[i], act_layer=act_layers[i], dw_ks=block_dw_ks,
					stride=block_stride, dim_head=dim_heads[i], window_size=window_sizes[i],
					hybrid_eops=block_hybrid_eops, conv_ks=block_conv_ks, conv_groups=block_conv_groups,
					qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, drop_path=stage_dpr[j],
					v_group=v_group, attn_pre=attn_pre, ls_value=ls_value
				))
				current_dim = embed_dims[i] # Update current_dim for the next block / stage
			self.__setattr__(f'stage{i + 1}', nn.ModuleList(stage_layers))
		
		# For classification head (if used)
		self.norm = get_norm(norm_layers[-1])(embed_dims[-1])
		self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
		self.pre_dim = embed_dims[-1] # Used by reset_classifier

		self.apply(self._init_weights)

		# Calculate width_list (channel dimensions of features for FPN, etc.)
		# Requires a dummy forward pass through feature extraction
		try:
			with torch.no_grad():
				dummy_input = torch.randn(1, self.dim_in, self.img_size, self.img_size)
				features = self.forward_features(dummy_input)
				self.width_list = [f.size(1) for f in features]
		except Exception as e:
			print(f"Warning: Could not compute width_list due to: {e}. Setting to empty list.")
			self.width_list = []


	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if m.bias is not None:
				nn.init.zeros_(m.bias)
		elif isinstance(m, (nn.LayerNorm, nn.GroupNorm,
							nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
							nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
			if m.bias is not None: nn.init.zeros_(m.bias)
			if m.weight is not None: nn.init.ones_(m.weight)

	@torch.jit.ignore
	def no_weight_decay(self): return {'token'}
	
	@torch.jit.ignore
	def no_weight_decay_keywords(self): return {'alpha', 'gamma', 'beta'}
	
	@torch.jit.ignore
	def no_ft_keywords(self): return {}
	
	@torch.jit.ignore
	def ft_head_keywords(self): return {'head.weight', 'head.bias'}, self.num_classes
	
	def get_classifier(self): return self.head
	
	def reset_classifier(self, num_classes):
		self.num_classes = num_classes
		self.head = nn.Linear(self.pre_dim, num_classes) if num_classes > 0 else nn.Identity()
	
	def check_bn(self):
		for name, m in self.named_modules():
			if isinstance(m, nn.modules.batchnorm._NormBase): # _BatchNorm also works
				m.running_mean = torch.nan_to_num(m.running_mean, nan=0, posinf=1, neginf=-1)
				m.running_var = torch.nan_to_num(m.running_var, nan=0, posinf=1, neginf=-1)
	
	def forward_features(self, x):
		features = []
		# stage0 (stem)
		for blk in self.stage0:
			x = blk(x)
		# After stem, x has emb_dim_pre channels. Downsampled by 2.
		# For FPN, usually start collecting features after total stride 4 or 8.
		# Stage1 output: total stride 4
		# Stage2 output: total stride 8
		# Stage3 output: total stride 16
		# Stage4 output: total stride 32

		# stage1
		for blk in self.stage1:
			x = blk(x)
		features.append(x) # Output of stage1 (embed_dims[0]), stride 4
		# stage2
		for blk in self.stage2:
			x = blk(x)
		features.append(x) # Output of stage2 (embed_dims[1]), stride 8
		# stage3
		for blk in self.stage3:
			x = blk(x)
		features.append(x) # Output of stage3 (embed_dims[2]), stride 16
		# stage4
		for blk in self.stage4:
			x = blk(x)
		features.append(x) # Output of stage4 (embed_dims[3]), stride 32
		
		return features # Returns a list of 4 feature maps
	
	def forward(self, x): # Default forward for backbone usage (e.g., YOLO)
		return self.forward_features(x)

	def forward_cls(self, x): # Forward for classification tasks
		# Get the last feature map from forward_features
		# If forward_features returns 4 maps, the last one is from stage4
		last_feature_map = self.forward_features(x)[-1]
		
		pooled_output = self.norm(last_feature_map)
		pooled_output = reduce(pooled_output, 'b c h w -> b c', 'mean').contiguous()
		# pooled_output = F.adaptive_avg_pool2d(self.norm(last_feature_map), (1,1)).flatten(1) # Alternative
		
		class_output = self.head(pooled_output)
		return {'out': class_output, 'out_kd': class_output} # Keep original output format


# Helper for scaling EMO2 model definitions
def _get_scaled_emo2_params(base_embed_dims, base_dim_heads, factor):
    scaled_embed_dims = []
    
    # Scale first embed_dim: emb_dim_pre = scaled_embed_dims[0]//2 must be div by base_dim_heads[0]
    ch_0_unscaled = base_embed_dims[0]
    ch_0_scaled_unadj = int(ch_0_unscaled * factor)
    divisor_0 = 2 * base_dim_heads[0] # emb_dim_pre must be div by dim_head[0]
    
    ch_0_scaled = ((ch_0_scaled_unadj + divisor_0 - 1) // divisor_0) * divisor_0
    if ch_0_scaled == 0 and ch_0_scaled_unadj > 0: # Ensure not zero if original was positive
        ch_0_scaled = divisor_0
    scaled_embed_dims.append(ch_0_scaled)

    # Scale subsequent embed_dims: scaled_embed_dims[i] must be div by base_dim_heads[i]
    for i in range(1, len(base_embed_dims)):
        ch_i_unscaled = base_embed_dims[i]
        ch_i_scaled_unadj = int(ch_i_unscaled * factor)
        divisor_i = base_dim_heads[i]
        
        ch_i_scaled = ((ch_i_scaled_unadj + divisor_i - 1) // divisor_i) * divisor_i
        if ch_i_scaled == 0 and ch_i_scaled_unadj > 0:
            ch_i_scaled = divisor_i
        scaled_embed_dims.append(ch_i_scaled)
            
    return scaled_embed_dims, base_dim_heads # Keep base_dim_heads, num_heads will change


# Model definition functions using the new factor argument
@MODEL.register_module
def EMO2_1M_k5_hybrid(pretrained=False, factor=1.0, **kwargs):
    base_embed_dims = [32, 48, 80, 180]
    base_dim_heads = [16, 16, 20, 20]
    scaled_embed_dims, final_dim_heads = _get_scaled_emo2_params(base_embed_dims, base_dim_heads, factor)

    return EMO2(
        dim_in=kwargs.get('dim_in', 3), img_size=kwargs.get('img_size', 224),
        depths=[2, 2, 8, 3], embed_dims=scaled_embed_dims, exp_ratios=[2., 2.5, 3.0, 3.5],
        norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
        dw_kss=[5, 5, 5, 5], dim_heads=final_dim_heads, window_sizes=kwargs.get('window_sizes', [7, 7, 7, 7]),
        hybrid_eopss=[[0], [0], [3], [3]],
        conv_kss=[1, 1, 1, 1], conv_groupss=[1, 1, 1, 1],
        qkv_bias=True, attn_drop=0., drop=0., drop_path=kwargs.get('drop_path', 0.04036),
        v_group=False, attn_pre=False, ls_value=1e-6,
        num_classes=kwargs.get('num_classes', 1000)
    )

@MODEL.register_module
def EMO2_1M_k5_hybrid_256(pretrained=False, factor=1.0, **kwargs):
    kwargs['img_size'] = 256
    kwargs['window_sizes'] = [8, 8, 8, 8]
    return EMO2_1M_k5_hybrid(pretrained=pretrained, factor=factor, **kwargs)

@MODEL.register_module
def EMO2_1M_k5_hybrid_512(pretrained=False, factor=1.0, **kwargs):
    kwargs['img_size'] = 512
    kwargs['window_sizes'] = [8, 8, 8, 8] # Or maybe larger like [16,16,16,16] for 512
    return EMO2_1M_k5_hybrid(pretrained=pretrained, factor=factor, **kwargs)


@MODEL.register_module
def EMO2_2M_k5_hybrid(pretrained=False, factor=1.0, **kwargs):
    base_embed_dims = [32, 48, 120, 200]
    base_dim_heads = [16, 16, 20, 20]
    scaled_embed_dims, final_dim_heads = _get_scaled_emo2_params(base_embed_dims, base_dim_heads, factor)
    
    return EMO2(
        dim_in=kwargs.get('dim_in', 3), img_size=kwargs.get('img_size', 224),
        depths=[3, 3, 9, 3], embed_dims=scaled_embed_dims, exp_ratios=[2., 2.5, 3.0, 3.5],
        norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
        dw_kss=[5, 5, 5, 5], dim_heads=final_dim_heads, window_sizes=kwargs.get('window_sizes', [7, 7, 7, 7]),
        hybrid_eopss=[[0], [0], [3], [3]],
        conv_kss=[1, 1, 1, 1], conv_groupss=[1, 1, 1, 1],
        qkv_bias=True, attn_drop=0., drop=0., drop_path=kwargs.get('drop_path', 0.05),
        v_group=False, attn_pre=False, ls_value=1e-6,
        num_classes=kwargs.get('num_classes', 1000)
    )

@MODEL.register_module
def EMO2_2M_k5_hybrid_256(pretrained=False, factor=1.0, **kwargs):
    kwargs['img_size'] = 256
    kwargs['window_sizes'] = [8, 8, 8, 8]
    return EMO2_2M_k5_hybrid(pretrained=pretrained, factor=factor, **kwargs)

@MODEL.register_module
def EMO2_2M_k5_hybrid_512(pretrained=False, factor=1.0, **kwargs):
    kwargs['img_size'] = 512
    kwargs['window_sizes'] = [8, 8, 8, 8] # Or larger
    return EMO2_2M_k5_hybrid(pretrained=pretrained, factor=factor, **kwargs)


@MODEL.register_module
def EMO2_5M_k5_hybrid(pretrained=False, factor=1.0, **kwargs):
    base_embed_dims = [48, 72, 160, 288]
    base_dim_heads = [16, 24, 32, 32] # Note: 48%16 ok, 72%24 ok, 160%32 ok, 288%32 ok
    scaled_embed_dims, final_dim_heads = _get_scaled_emo2_params(base_embed_dims, base_dim_heads, factor)

    return EMO2(
        dim_in=kwargs.get('dim_in', 3), img_size=kwargs.get('img_size', 224),
        depths=[3, 3, 9, 3], embed_dims=scaled_embed_dims, exp_ratios=[2., 3., 4., 4.],
        norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
        dw_kss=[5, 5, 5, 5], dim_heads=final_dim_heads, window_sizes=kwargs.get('window_sizes', [7, 7, 7, 7]),
        hybrid_eopss=[[0], [0], [3], [3]],
        conv_kss=[1, 1, 1, 1], conv_groupss=[1, 1, 1, 1],
        qkv_bias=True, attn_drop=0., drop=0., drop_path=kwargs.get('drop_path', 0.05),
        v_group=False, attn_pre=False, ls_value=1e-6,
        num_classes=kwargs.get('num_classes', 1000)
    )

@MODEL.register_module
def EMO2_5M_k5_hybrid_256(pretrained=False, factor=1.0, **kwargs):
    kwargs['img_size'] = 256
    kwargs['window_sizes'] = [8, 8, 8, 8]
    return EMO2_5M_k5_hybrid(pretrained=pretrained, factor=factor, **kwargs)

@MODEL.register_module
def EMO2_5M_k5_hybrid_512(pretrained=False, factor=1.0, **kwargs):
    kwargs['img_size'] = 512
    kwargs['window_sizes'] = [8, 8, 8, 8] # Or larger
    return EMO2_5M_k5_hybrid(pretrained=pretrained, factor=factor, **kwargs)


@MODEL.register_module
def EMO2_20M_k5_hybrid(pretrained=False, factor=1.0, **kwargs):
    base_embed_dims = [64, 128, 320, 448]
    base_dim_heads = [16, 32, 32, 32] # 64%16, 128%32, 320%32, 448%32 ok
    scaled_embed_dims, final_dim_heads = _get_scaled_emo2_params(base_embed_dims, base_dim_heads, factor)

    return EMO2(
        dim_in=kwargs.get('dim_in', 3), img_size=kwargs.get('img_size', 224),
        depths=[3, 3, 13, 3], embed_dims=scaled_embed_dims, exp_ratios=[2., 3., 4., 4.],
        norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
        dw_kss=[5, 5, 5, 5], dim_heads=final_dim_heads, window_sizes=kwargs.get('window_sizes', [7, 7, 7, 7]),
        hybrid_eopss=[[0], [0], [3], [3]],
        conv_kss=[1, 1, 1, 1], conv_groupss=[1, 1, 1, 1],
        qkv_bias=True, attn_drop=0., drop=0., drop_path=kwargs.get('drop_path', 0.1),
        v_group=False, attn_pre=False, ls_value=1e-6,
        num_classes=kwargs.get('num_classes', 1000)
    )

@MODEL.register_module
def EMO2_20M_k5_hybrid_256(pretrained=False, factor=1.0, **kwargs):
    kwargs['img_size'] = 256
    kwargs['window_sizes'] = [8, 8, 8, 8]
    return EMO2_20M_k5_hybrid(pretrained=pretrained, factor=factor, **kwargs)


@MODEL.register_module
def EMO2_50M_k5_hybrid(pretrained=False, factor=1.0, **kwargs):
    base_embed_dims = [64, 128, 384, 512]
    base_dim_heads = [16, 32, 32, 32] # 64%16, 128%32, 384%32, 512%32 ok
    scaled_embed_dims, final_dim_heads = _get_scaled_emo2_params(base_embed_dims, base_dim_heads, factor)

    return EMO2(
        dim_in=kwargs.get('dim_in', 3), img_size=kwargs.get('img_size', 224),
        depths=[5, 8, 20, 7], embed_dims=scaled_embed_dims, exp_ratios=[2., 3., 4., 4.],
        norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
        dw_kss=[5, 5, 5, 5], dim_heads=final_dim_heads, window_sizes=kwargs.get('window_sizes', [7, 7, 7, 7]),
        hybrid_eopss=[[0], [0], [3], [3]],
        conv_kss=[1, 1, 1, 1], conv_groupss=[1, 1, 1, 1],
        qkv_bias=True, attn_drop=0., drop=0., drop_path=kwargs.get('drop_path', 0.2),
        v_group=False, attn_pre=False, ls_value=1e-6,
        num_classes=kwargs.get('num_classes', 1000)
    )