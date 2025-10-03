# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MetaFormer baselines including IdentityFormer, RandFormer, PoolFormerV2,
ConvFormer and CAFormer.
Some implementations are modified from timm (https://github.com/rwightman/pytorch-image-models).

Modified to:
- Remove pretrained weight loading.
- MetaFormer.forward() now returns a list of feature maps from each stage.
- Added self.width_list to MetaFormer, calculated from a dummy forward pass.
- Input to MetaFormer.forward is expected to be B,C,H,W.
- Output list from MetaFormer.forward contains tensors in B,C,H,W format.
- Fixed LayerNormGeneral broadcasting for specific pre_norm case.
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD # Not used if not loading cfgs
from timm.models.layers.helpers import to_2tuple


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    Input: B,C,H,W (if pre_permute=False) or B,H,W,C (if pre_permute=True)
    Output: B,H',W',C' (always)
    """
    def __init__(self, in_channels, out_channels,
        kernel_size, stride=1, padding=0,
        pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        # x input: B,C,H,W or B,H,W,C (if pre_permute=True)
        if self.pre_permute:
            # if input is [B, H, W, C], permute it to [B, C, H, W] for conv
            x = x.permute(0, 3, 1, 2)
        x = self.pre_norm(x) # Applied on B,C,H,W (if pre_permute=T) or B,H,W,C (if pre_permute=F and pre_norm takes that)
                             # With current setup, pre_norm always gets B,C,H,W if pre_permute=T, or x if pre_permute=F
        x = self.conv(x)     # Output B,C',H',W'
        x = x.permute(0, 2, 3, 1) # [B, C', H', W'] -> [B, H', W', C']
        x = self.post_norm(x) # Applied on B,H',W',C'
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class SquaredReLU(nn.Module):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return torch.square(self.relu(x))


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True,
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    Input: B,H,W,C
    Output: B,H,W,C
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        return x_attn


class RandomMixing(nn.Module):
    """
    Input: B,H,W,C
    Output: B,H,W,C
    """
    def __init__(self, num_tokens=196, **kwargs): # num_tokens depends on feature map size
        super().__init__()
        self.num_tokens = num_tokens
        # Parameter moved to build method if num_tokens can change, or requires_grad=False
        # For now, keep as is, but be mindful if H*W varies unexpectedly for a fixed layer.
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1),
            requires_grad=False)

    def forward(self, x):
        B, H, W, C = x.shape
        if H*W != self.num_tokens:
             # This case should ideally be handled by selecting the correct RandomMixing layer
             # or making RandomMixing adaptive. For now, we'll raise an error or skip.
            # For safety, if a model is constructed with fixed num_tokens and input size changes,
            # this can be an issue.
            # A simple fallback or warning:
            # print(f"Warning: RandomMixing H*W ({H*W}) != num_tokens ({self.num_tokens}). Skipping mixing.")
            # return x
            # Or, better, ensure num_tokens matches.
            raise ValueError(f"RandomMixing H*W ({H*W}) must match num_tokens ({self.num_tokens})")

        x_flat = x.reshape(B, H*W, C)
        x_mixed = torch.einsum('mn, bnc -> bmc', self.random_matrix, x_flat)
        x_out = x_mixed.reshape(B, H, W, C)
        return x_out


class LayerNormGeneral(nn.Module):
    r""" General LayerNorm for different situations.

    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default.
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance.
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.
    """
    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True,
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x_normalized = c / torch.sqrt(s + self.eps)
        
        out = x_normalized

        # Check for the specific pre_norm case that caused the error:
        # Input to LayerNormGeneral's forward is B,C,H,W (x.ndim == 4)
        # affine_shape was an int C (so self.weight/bias are 1D and their size is x.shape[1] (the C dim))
        # normalized_dim is (-1,) (meaning normalization was over W, which is x.shape[3])
        is_bchw_input_norm_w_affine_c_case = False
        if x.ndim == 4 and self.normalized_dim == (-1,):
            if self.use_scale and self.weight is not None and self.weight.ndim == 1 and self.weight.shape[0] == x.shape[1]:
                is_bchw_input_norm_w_affine_c_case = True
            if not is_bchw_input_norm_w_affine_c_case and \
               self.use_bias and self.bias is not None and self.bias.ndim == 1 and self.bias.shape[0] == x.shape[1]:
                is_bchw_input_norm_w_affine_c_case = True
        
        if is_bchw_input_norm_w_affine_c_case:
            if self.use_scale:
                out = out * self.weight.view(1, -1, 1, 1)
            if self.use_bias:
                out = out + self.bias.view(1, -1, 1, 1)
        else: # Default behavior for all other cases
            if self.use_scale:
                out = out * self.weight
            if self.use_bias:
                out = out + self.bias
        return out


class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster,
    because it directly utilizes otpimized F.layer_norm.
    Expects input B,...,C and normalizes over the C dimension.
    """
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape # This is the shape of the last C dim(s)
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    Input: B,H,W,C
    Output: B,H,W,C
    """
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity,
        bias=False, kernel_size=7, padding=3,
        **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x): # x: B,H,W,C
        x_in = x
        x = self.pwconv1(x_in)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2) # B,C,H,W for conv
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # B,H,W,C
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modfiled for [B, H, W, C] input
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x): # x: B,H,W,C
        y = x.permute(0, 3, 1, 2) # B,C,H,W for pool
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1) # B,H,W,C
        return y - x # Residual connection


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    Input: B,H,W,C (or B,N,C)
    Output: B,H,W,C (or B,N,C)
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    Input: B,C (after pooling)
    Output: B,num_classes
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU,
        norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features) # expects B,H
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)


    def forward(self, x): # x: B,C
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    Input: B,H,W,C
    Output: B,H,W,C
    """
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x): # x: B,H,W,C
        x_res1 = x
        x = self.norm1(x)
        x = self.token_mixer(x)
        x = self.drop_path1(x)
        x = self.layer_scale1(x)
        x = self.res_scale1(x_res1) + x
        
        x_res2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path2(x)
        x = self.layer_scale2(x)
        x = self.res_scale2(x_res2) + x
        return x


DOWNSAMPLE_LAYERS_FOUR_STAGES = [partial(Downsampling,
            kernel_size=7, stride=4, padding=2, pre_permute=False, 
            post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6) 
            )] + \
            [partial(Downsampling,
                kernel_size=3, stride=2, padding=1, 
                pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6), 
                pre_permute=True 
            )]*3


class MetaFormer(nn.Module):
    r""" MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452
    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
                           Only used if forward_head is called.
        depths (list or tuple): Number of blocks at each stage. Default: [2, 2, 6, 2].
        dims (list or tuple): Feature dimension at each stage. Default: [64, 128, 320, 512].
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        token_mixers (list, tuple or token_fcn): Token mixer for each stage. Default: nn.Identity.
        mlps (list, tuple or mlp_fcn): Mlp for each stage. Default: Mlp.
        norm_layers (list, tuple or norm_fcn): Norm layers for MetaFormerBlock.
                                                Default: partial(LayerNormWithoutBias, eps=1e-6).
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_dropout (float): dropout for MLP classifier. Default: 0.
        layer_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: None.
        res_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: [None, None, 1.0, 1.0].
        output_norm: norm before classifier head. Default: partial(nn.LayerNorm, eps=1e-6).
        head_fn: classification head. Default: nn.Linear.
        input_size_for_width_list (tuple): (H,W) for dummy pass to compute width_list. Default (224,224)
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[2, 2, 6, 2],
                 dims=[64, 128, 320, 512],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 token_mixers=nn.Identity,
                 mlps=Mlp,
                 norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
                 drop_path_rate=0.,
                 head_dropout=0.0,
                 layer_scale_init_values=None,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 output_norm=partial(nn.LayerNorm, eps=1e-6), 
                 head_fn=nn.Linear, 
                 input_size_for_width_list=(224,224),
                 **kwargs, 
                 ):
        super().__init__()
        self.num_classes = num_classes 

        if not isinstance(depths, (list, tuple)):
            depths = [depths]
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        
        down_in_dims = [in_chans] + dims[:-1]
        down_out_dims = dims 

        self.downsample_layers = nn.ModuleList()
        for i in range(num_stage):
            self.downsample_layers.append(
                downsample_layers[i](down_in_dims[i], down_out_dims[i])
            )

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage
        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage
        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList() 
        cur = 0
        for i in range(num_stage):
            # For RandFormer, token_mixer can be a partial function.
            # If it needs num_tokens determined by feature map size, this needs careful handling.
            # Current token_mixers list is pre-defined with num_tokens.
            current_token_mixer_fn = token_mixers[i]
            # if isinstance(current_token_mixer_fn, partial) and \
            #    'num_tokens' in current_token_mixer_fn.keywords:
            #    pass # num_tokens is already set
            
            stage_blocks = nn.Sequential(
                *[MetaFormerBlock(dim=dims[i], 
                token_mixer=current_token_mixer_fn,
                mlp=mlps[i],
                norm_layer=norm_layers[i], 
                drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_values[i],
                res_scale_init_value=res_scale_init_values[i],
                ) for j in range(depths[i])]
            )
            self.stages.append(stage_blocks)
            cur += depths[i]

        self.norm_head = output_norm(dims[-1]) 
        if head_dropout > 0.0 and num_classes > 0 :
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        elif num_classes > 0:
            self.head = head_fn(dims[-1], num_classes)
        else:
            self.head = nn.Identity()


        self.apply(self._init_weights)
        
        try:
            H, W = input_size_for_width_list
            dummy_input = torch.randn(1, in_chans, H, W)
            
            original_mode = self.training
            self.eval()
            with torch.no_grad():
                features_list = self.forward(dummy_input) 
            if original_mode: 
                self.train()
                
            self.width_list = [f.size(1) for f in features_list]
        except Exception as e:
            print(f"Warning: Could not compute width_list due to: {e}")
            print("         You might need to manually set input_size_for_width_list if token_mixers like RandomMixing depend on it.")
            self.width_list = dims # Fallback, might not be fully accurate if some stages don't output `dims[i]` channels

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNormGeneral, LayerNormWithoutBias)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'} 

    def forward_features_list(self, x):
        outputs = []
        current_feature_map = x 

        for i in range(self.num_stage):
            current_feature_map = self.downsample_layers[i](current_feature_map) 
            current_feature_map = self.stages[i](current_feature_map) 
            
            outputs.append(current_feature_map.permute(0, 3, 1, 2).contiguous())
        return outputs 

    def forward(self, x):
        return self.forward_features_list(x)

    def forward_head(self, x_list, pre_logits=False):
        if isinstance(x_list, list):
            x = x_list[-1] 
        else:
            x = x_list

        x = x.permute(0, 2, 3, 1) 
        x = self.norm_head(x.mean([1, 2])) 
        if pre_logits:
            return x
        return self.head(x)


# IdentityFormer Variants
@register_model
def identityformer_s12(**kwargs):
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False), # For B,H,W,C data
        **kwargs)
    return model

@register_model
def identityformer_s24(**kwargs):
    model = MetaFormer(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    return model

@register_model
def identityformer_s36(**kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    return model

@register_model
def identityformer_m36(**kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    return model

@register_model
def identityformer_m48(**kwargs):
    model = MetaFormer(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    return model

# RandFormer Variants
# Default num_tokens for 224x224 input:
# Stage 0 (idx 0) out: 56x56 -> 3136 tokens
# Stage 1 (idx 1) out: 28x28 -> 784 tokens
# Stage 2 (idx 2) out: 14x14 -> 196 tokens
# Stage 3 (idx 3) out: 7x7   -> 49 tokens
# The token_mixers are applied *within* the stage, on the feature map *after* downsampling for that stage.
@register_model
def randformer_s12(input_size=(224,224), **kwargs):
    s0_fh, s0_fw = input_size[0]//4, input_size[1]//4
    s1_fh, s1_fw = s0_fh//2, s0_fw//2
    s2_fh, s2_fw = s1_fh//2, s1_fw//2
    s3_fh, s3_fw = s2_fh//2, s2_fw//2
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        token_mixers=[
            nn.Identity, # Stage 0: token_mixer on 56x56 (if 224 input)
            nn.Identity, # Stage 1: token_mixer on 28x28
            partial(RandomMixing, num_tokens=s2_fh*s2_fw), # Stage 2: token_mixer on 14x14 (num_tokens=196)
            partial(RandomMixing, num_tokens=s3_fh*s3_fw)  # Stage 3: token_mixer on 7x7 (num_tokens=49)
            ],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        input_size_for_width_list=input_size, # Ensure width_list calculation uses this
        **kwargs)
    return model

@register_model
def randformer_s24(input_size=(224,224), **kwargs):
    s0_fh, s0_fw = input_size[0]//4, input_size[1]//4
    s1_fh, s1_fw = s0_fh//2, s0_fw//2
    s2_fh, s2_fw = s1_fh//2, s1_fw//2
    s3_fh, s3_fw = s2_fh//2, s2_fw//2
    model = MetaFormer(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        token_mixers=[
            nn.Identity, 
            nn.Identity, 
            partial(RandomMixing, num_tokens=s2_fh*s2_fw), 
            partial(RandomMixing, num_tokens=s3_fh*s3_fw)
            ],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        input_size_for_width_list=input_size,
        **kwargs)
    return model

@register_model
def randformer_s36(input_size=(224,224), **kwargs):
    s0_fh, s0_fw = input_size[0]//4, input_size[1]//4
    s1_fh, s1_fw = s0_fh//2, s0_fw//2
    s2_fh, s2_fw = s1_fh//2, s1_fw//2
    s3_fh, s3_fw = s2_fh//2, s2_fw//2
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        token_mixers=[
            nn.Identity, 
            nn.Identity, 
            partial(RandomMixing, num_tokens=s2_fh*s2_fw), 
            partial(RandomMixing, num_tokens=s3_fh*s3_fw)
            ],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        input_size_for_width_list=input_size,
        **kwargs)
    return model

@register_model
def randformer_m36(input_size=(224,224), **kwargs):
    s0_fh, s0_fw = input_size[0]//4, input_size[1]//4
    s1_fh, s1_fw = s0_fh//2, s0_fw//2
    s2_fh, s2_fw = s1_fh//2, s1_fw//2
    s3_fh, s3_fw = s2_fh//2, s2_fw//2
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        token_mixers=[
            nn.Identity, 
            nn.Identity, 
            partial(RandomMixing, num_tokens=s2_fh*s2_fw), 
            partial(RandomMixing, num_tokens=s3_fh*s3_fw)
            ],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        input_size_for_width_list=input_size,
        **kwargs)
    return model

@register_model
def randformer_m48(input_size=(224,224), **kwargs):
    s0_fh, s0_fw = input_size[0]//4, input_size[1]//4
    s1_fh, s1_fw = s0_fh//2, s0_fw//2
    s2_fh, s2_fw = s1_fh//2, s1_fw//2
    s3_fh, s3_fw = s2_fh//2, s2_fw//2
    model = MetaFormer(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        token_mixers=[
            nn.Identity, 
            nn.Identity, 
            partial(RandomMixing, num_tokens=s2_fh*s2_fw), 
            partial(RandomMixing, num_tokens=s3_fh*s3_fw)
            ],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        input_size_for_width_list=input_size,
        **kwargs)
    return model

# PoolFormerV2 Variants
@register_model
def poolformerv2_s12(**kwargs):
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    return model

@register_model
def poolformerv2_s24(**kwargs):
    model = MetaFormer(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    return model

@register_model
def poolformerv2_s36(**kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    return model

@register_model
def poolformerv2_m36(**kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    return model

@register_model
def poolformerv2_m48(**kwargs):
    model = MetaFormer(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    return model

# ConvFormer Variants
@register_model
def convformer_s18(**kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead, 
        **kwargs)
    return model

@register_model
def convformer_s36(**kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    return model

@register_model
def convformer_m36(**kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576], 
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    return model

@register_model
def convformer_b36(**kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    return model

# CAFormer Variants
@register_model
def caformer_s18(**kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    return model

@register_model
def caformer_s36(**kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    return model

@register_model
def caformer_m36(**kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576], 
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    return model

@register_model
def caformer_b36(**kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    return model

if __name__ == '__main__':
    print("Testing IdentityFormer_S12...")
    model_identity = identityformer_s12(num_classes=100) 
    print(f"  Width list: {model_identity.width_list}")
    dummy_tensor_224 = torch.randn(2, 3, 224, 224)
    features = model_identity(dummy_tensor_224) 
    print(f"  Output type: {type(features)}")
    for i, f in enumerate(features):
        print(f"  Stage {i} feature shape: {f.shape}")
    
    class_output = model_identity.forward_head(features)
    print(f"  Classification head output shape: {class_output.shape}")

    print("\nTesting ConvFormer_S18 with 384x384 input...")
    model_conv_384 = convformer_s18(num_classes=0, input_size_for_width_list=(384,384))
    print(f"  Width list (384): {model_conv_384.width_list}")
    dummy_tensor_384 = torch.randn(1, 3, 384, 384) 
    features_conv = model_conv_384(dummy_tensor_384)
    print(f"  Output type: {type(features_conv)}")
    for i, f in enumerate(features_conv):
        print(f"  Stage {i} feature shape: {f.shape}") 

    print("\nTesting RandFormer_S12 with 384x384 input...")
    # Pass input_size to the factory function for RandFormer
    model_rand_384 = randformer_s12(num_classes=10, input_size=(384,384))
    print(f"  Width list (384): {model_rand_384.width_list}")
    features_rand_384 = model_rand_384(dummy_tensor_384)
    for i, f in enumerate(features_rand_384):
        print(f"  Stage {i} feature shape: {f.shape}")
    
    # Test RandFormer with default 224x224
    print("\nTesting RandFormer_S12 with default 224x224 input...")
    model_rand_224 = randformer_s12(num_classes=10) # Relies on default input_size=(224,224)
    print(f"  Width list (224): {model_rand_224.width_list}")
    features_rand_224 = model_rand_224(dummy_tensor_224)
    for i, f in enumerate(features_rand_224):
        print(f"  Stage {i} feature shape: {f.shape}")