# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may an copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SPANet including small, meidum, and base models.
Some implementations are modified from timm (https://github.com/rwightman/pytorch-image-models).
"""
import copy
import os # Added for os.environ.get
import logging # Added for simplified logging

from functools import partial
import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers.helpers import to_2tuple

# Simplified logger for standalone use
logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head', # Standard timm classifier name
        **kwargs
    }


default_cfgs = {
    'spanet_s': _cfg(crop_pct=0.9, first_conv='patch_embed.proj'),
    'spanet_m': _cfg(crop_pct=0.9, first_conv='patch_embed.proj'),
    'spanet_b': _cfg(crop_pct=0.95, first_conv='patch_embed.proj'),
}


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0, 
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class SPF(nn.Module):
    """ Spectral Pooling Filter 
    """
    def __init__(self, H, W, r, lamb): 
        super().__init__() 
        # Filter is float32 by default from _CircleFilter
        # Register as buffer so it's part of state_dict and moves with model.to(device)
        # persistent=False means it won't be saved in optimizer state (not a learnable param)
        # but it IS saved in the model's state_dict.
        self.register_buffer("filter_base", self._CircleFilter(H, W, r, lamb).unsqueeze(0).unsqueeze(0), persistent=True) # (1,1,H,W)

    def _CircleFilter(self, H, W, r, lamb): 
        # Determine device for meshgrid. If filter_base exists, use its device, else default.
        device = self.filter_base.device if hasattr(self, 'filter_base') and self.filter_base is not None else None
        
        x_center = int(W//2)
        y_center = int(H//2)
        
        X, Y = torch.meshgrid(
            torch.arange(0, H, 1, device=device), 
            torch.arange(0, W, 1, device=device), 
            indexing='ij'
        ) 
        circle = torch.sqrt((X.float()-x_center)**2 + (Y.float()-y_center)**2) 

        lp_F = (circle < r).to(torch.float32)
        hp_F = (circle > r).to(torch.float32)

        combined_Filter = lp_F*lamb + hp_F*(1-lamb)
        on_circle_mask = torch.isclose(circle, torch.tensor(float(r), device=device))
        combined_Filter[on_circle_mask] = 1/3.0

        return combined_Filter # Shape (H, W), dtype float32

    def _shift(self, x): 
        return torch.fft.fftshift(x, dim=(-2, -1))

    def _ishift(self, x): 
        return torch.fft.ifftshift(x, dim=(-2, -1))

    def forward(self, x):
        B, C, in_H, in_W = x.shape # x can be [B, C_chunk, H, W]
        original_dtype = x.dtype 

        if original_dtype == torch.complex32 or original_dtype == torch.complex64 or original_dtype == torch.complex128:
            # This case should not happen if input x to SPF is always real
            logger.warning(f"SPF received complex input dtype {original_dtype}. This is unexpected.")
            x_real = x.real # Try to proceed with real part
            original_dtype = x_real.dtype # Update original_dtype to the type of the real part
        else:
            x_real = x

        # Convert to float32 for FFT if not already (e.g., if it's half)
        if x_real.dtype != torch.float32:
            x_float32 = x_real.to(torch.float32)
        else:
            x_float32 = x_real
        
        x_fft = torch.fft.fft2(x_float32, dim=(-2, -1), norm='ortho') # Result is complex64
        x_fft_shifted = self._shift(x_fft) # complex64

        # Ensure filter_base is on the correct device and handle size mismatches
        current_filter = self.filter_base.to(device=x_fft_shifted.device, dtype=torch.float32) # Ensure float32
        
        _, _, f_H, f_W = current_filter.shape # current_filter is (1,1,H,W)

        if (in_H, in_W) != (f_H, f_W):
            # Runtime filter resizing/padding (can be slow)
            pad_h_total = in_H - f_H
            pad_w_total = in_W  - f_W
            
            pad_top = max(0, pad_h_total // 2 + pad_h_total % 2)
            pad_bottom = max(0, pad_h_total // 2)
            pad_left = max(0, pad_w_total // 2 + pad_w_total % 2)
            pad_right = max(0, pad_w_total // 2)
            
            padding = (pad_left, pad_right, pad_top, pad_bottom) # F.pad expects (L,R,T,B)
            
            # Use a value from the filter for padding, e.g., center or a known neutral value
            pad_value = current_filter[0, 0, f_H//2, f_W//2].item() if f_H > 0 and f_W > 0 else 0.0

            if pad_h_total < 0 or pad_w_total < 0: # Input smaller than filter (cropping needed)
                # Cropping logic:
                crop_t = max(0, (f_H - in_H) // 2)
                crop_b = f_H - max(0, (f_H - in_H) // 2 + (f_H - in_H) % 2)
                crop_l = max(0, (f_W - in_W) // 2)
                crop_r = f_W - max(0, (f_W - in_W) // 2 + (f_W - in_W) % 2)
                current_filter = current_filter[:, :, crop_t:crop_b, crop_l:crop_r]
                # logger.warning(f"SPF input ({in_H},{in_W}) smaller than filter ({f_H},{f_W}). Cropped filter to {current_filter.shape}.")

            elif pad_h_total > 0 or pad_w_total > 0: # Input larger than filter (padding needed)
                 current_filter = F.pad(current_filter, padding, mode='constant', value=pad_value)
        
        # Perform filtering: filter (real, float32) * fft_shifted_input (complex, complex64)
        # Result will be complex64 due to type promotion
        x_filtered = current_filter * x_fft_shifted # current_filter is (1,1,H,W), broadcasts over B and C_chunk
        
        x_ifft_shifted = self._ishift(x_filtered) # complex64
        x_ifft = torch.fft.ifft2(x_ifft_shifted, s=(in_H,in_W), dim=(-2,-1), norm='ortho') # complex64
        
        x_out = x_ifft.real.to(original_dtype) # Convert back to original real dtype (e.g. half)
        
        return x_out


class SPAM(nn.Module):
    def __init__(self, dim= 64, k_size=7, H=56, W=56, r=2**5):
        super().__init__()
        self.lambs = [lamb.item() for lamb in torch.arange(0.7, 1.0, 0.1)] 
        self.n_chunk = len(self.lambs)
        if dim % self.n_chunk != 0:
            # If not divisible, could adjust n_chunk or distribute dims unevenly,
            # but for simplicity, require divisibility.
            new_n_chunk = 0
            if dim % 3 == 0: new_n_chunk = 3
            elif dim % 2 == 0: new_n_chunk = 2
            elif dim % 1 == 0: new_n_chunk = 1
            
            if new_n_chunk > 0 and new_n_chunk <= len(self.lambs) :
                logger.warning(f"SPAM: dim {dim} not divisible by n_chunk {self.n_chunk}. Adjusting n_chunk to {new_n_chunk} and reusing lambs.")
                self.n_chunk = new_n_chunk
                self.lambs = self.lambs[:self.n_chunk]
            else:
                 raise ValueError(f"SPAM: Dimension {dim} must be divisible by a supported number of chunks (1, 2, or 3 based on lambs). Current n_chunk={self.n_chunk}.")

        chunk_dim = dim // self.n_chunk
        
        self.proj_in = nn.Conv2d(dim, dim, 1) 
        self.conv =  nn.Sequential(            
                nn.Conv2d(dim, dim, (1,k_size), padding=(0, k_size//2), groups=dim),
                nn.Conv2d(dim, dim, (k_size,1), padding=(k_size//2, 0), groups=dim),
            )
        self.proj_out = nn.Conv2d(dim, dim, 1) 

        self.sps = nn.ModuleList(
            [SPF(H, W, r, lamb) for lamb in self.lambs]
        )
        # Each pw_conv processes its own chunk and outputs `chunk_dim` channels.
        # These are then summed or concatenated. The original code summed, implying each pw_conv output `dim`.
        # Let's stick to the logic where each pw_conv output `dim` and then they are summed.
        self.pws = nn.ModuleList(
            [nn.Conv2d(chunk_dim, dim, 1) for _ in range(self.n_chunk)]
        )
    
    def forward(self, x):
        B, C_in, H_in, W_in = x.shape # C_in should be self.dim
        
        x_proj = self.proj_in(x) 
        x_conv = self.conv(x_proj) # Shape: (B, dim, H_in, W_in)

        chunks = torch.chunk(x_conv, self.n_chunk, dim=1) # List of n_chunk tensors, each (B, chunk_dim, H_in, W_in)
        
        feat_bank_pws_outs = []
        for i in range(self.n_chunk):
            spf_out = self.sps[i](chunks[i]) # spf_out shape (B, chunk_dim, H_in, W_in)
            pws_out = self.pws[i](spf_out)   # pws_out shape (B, dim, H_in, W_in)
            feat_bank_pws_outs.append(pws_out)
        
        # Summing the outputs from parallel paths
        ctx = torch.sum(torch.stack(feat_bank_pws_outs), dim=0) # ctx shape (B, dim, H_in, W_in)
        
        modulated_x = x_conv * ctx 
        out = self.proj_out(modulated_x) 
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale.view(1, -1, 1, 1)


class SPANetBlock(nn.Module):
    def __init__(self, dim, k_size=7, patch_dim_h=56, patch_dim_w=56, r=2**1, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., res_scale_init_value=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = SPAM(dim=dim, k_size=k_size, H=patch_dim_h,W=patch_dim_w, r=r)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + self.drop_path(self.token_mixer(self.norm1(x)))
        x = self.res_scale2(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers, 
                 k_size=7, patch_dim_h=56, patch_dim_w=56, r=2**1, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop_rate=.0, drop_path_rate=0., 
                 res_scale_init_value=None):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1) if sum(layers) > 1 else 0.0
        blocks.append(SPANetBlock(
            dim, k_size=k_size, patch_dim_h=patch_dim_h, patch_dim_w=patch_dim_w, r=r, mlp_ratio=mlp_ratio, 
            act_layer=act_layer, norm_layer=norm_layer, 
            drop=drop_rate, drop_path=block_dpr, 
            res_scale_init_value=res_scale_init_value, 
            ))
    blocks = nn.Sequential(*blocks)
    return blocks


class SPANet(nn.Module):
    def __init__(self, layers, embed_dims=None, patch_dims_hw=None, 
                 mlp_ratios=None, downsamples=None, 
                 k_size=7, 
                 radius=[2**1, 2**1, 2**0, 2**0],
                 norm_layer=GroupNorm, act_layer=nn.GELU, 
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2, 
                 down_patch_size=3, down_stride=2, down_pad=1, 
                 drop_rate=0., drop_path_rate=0.,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 fork_feat=True, 
                 img_size=224, in_chans=3, 
                 init_cfg=None, 
                 pretrained=None, 
                 **kwargs):

        super().__init__()
        
        self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.embed_dims = embed_dims
        self.img_size = img_size # Store for width_list calculation
        self.in_chans = in_chans # Store for width_list calculation


        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad, 
            in_chans=self.in_chans, embed_dim=embed_dims[0])

        network = []
        self.out_indices = [] 
        current_network_idx = 0
        for i in range(len(layers)):
            # patch_dims_hw is a list of (H,W) tuples for each stage
            patch_dim_h, patch_dim_w = patch_dims_hw[i] 
            stage = basic_blocks(embed_dims[i], i, layers, 
                                 k_size=k_size, 
                                 patch_dim_h=patch_dim_h, patch_dim_w=patch_dim_w, 
                                 r=radius[i], mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 res_scale_init_value=res_scale_init_values[i])
            network.append(stage)
            self.out_indices.append(current_network_idx)
            current_network_idx += 1
            
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride, 
                        padding=down_pad, 
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )
                current_network_idx += 1
        self.network = nn.ModuleList(network)

        # Norm layers for each feature extraction point (output of a stage)
        for i_stage_out, _ in enumerate(self.out_indices): # Iterate based on number of output stages
            current_embed_dim = embed_dims[i_stage_out]
            if i_stage_out == 0 and os.environ.get('FORK_LAST3', None):
                layer = nn.Identity()
            else:
                layer = norm_layer(current_embed_dim)
            self.add_module(f'norm_feat_stage{i_stage_out}', layer)
            
        if not self.fork_feat:
            self.norm_cls = norm_layer(embed_dims[-1]) 
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()
        else:
            # When fork_feat is True, head is typically not used by this class itself
            # but might be by a downstream model if it expects a 'head' attribute.
            self.head = nn.Identity() 

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        if not self.fork_feat and pretrained:
             self._load_pretrained_classifier(pretrained)
        # elif self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            # Ultralytics handles backbone weight loading.
            # pass

        self.width_list = []
        try:
            # Ensure model is on a device before dummy forward, and in eval mode
            device = next(self.parameters()).device 
            self.eval() 
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size, device=device)
            with torch.no_grad():
                embedded_dummy = self.patch_embed(dummy_input)
                features = self.forward_features(embedded_dummy) 
            
            self.width_list = [f.shape[1] for f in features]
            self.train() 
        except Exception as e:
            logger.warning(f"SPANet: Error during dummy forward pass for width_list: {e}. Falling back to embed_dims.")
            self.width_list = [self.embed_dims[i] for i in range(len(self.out_indices))] # Fallback
            self.train()
        # logger.info(f"SPANet initialized. Width list: {self.width_list}")


    def _load_pretrained_classifier(self, pretrained_url_or_path):
        if isinstance(pretrained_url_or_path, str) and pretrained_url_or_path.startswith('http'):
            checkpoint = torch.hub.load_state_dict_from_url(
                pretrained_url_or_path, map_location='cpu', check_hash=True)
        elif isinstance(pretrained_url_or_path, str):
            checkpoint = torch.load(pretrained_url_or_path, map_location='cpu')
        else:
            checkpoint = pretrained_url_or_path

        state_dict_key = 'state_dict' if 'state_dict' in checkpoint else 'model' if 'model' in checkpoint else None
        _state_dict = checkpoint[state_dict_key] if state_dict_key else checkpoint
        
        # Basic filtering for classifier head
        if hasattr(self, 'head') and self.head.in_features != _state_dict.get('head.weight', torch.empty(0)).shape[1]:
            logger.info("Classifier head mismatch, attempting to load without head.")
            _state_dict = {k: v for k, v in _state_dict.items() if not k.startswith('head.')}
        
        load_result = self.load_state_dict(_state_dict, strict=False)
        logger.info(f"Loaded pretrained classifier weights. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d)):
             trunc_normal_(m.weight, std=.02)
             if m.bias is not None:
                 nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''): 
        self.num_classes = num_classes
        if self.embed_dims: # Ensure embed_dims is available
            self.head = nn.Linear(
                self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
            if isinstance(self.head, nn.Linear):
                self.cls_init_weights(self.head)
        else:
            logger.warning("reset_classifier called but embed_dims not set. Head not reset.")


    def forward_features(self, x): # x is output of patch_embed
        outs = []
        current_x = x
        stage_output_counter = 0 # To index norm_feat_stage{i}
        for i, block_or_downsample in enumerate(self.network):
            current_x = block_or_downsample(current_x)
            # Check if the current network index 'i' corresponds to a designated output stage
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm_feat_stage{stage_output_counter}')
                outs.append(norm_layer(current_x))
                stage_output_counter += 1
        return outs

    def forward(self, x):
        x = self.patch_embed(x)
        features = self.forward_features(x)

        if self.fork_feat:
            return features
        else:
            # Classification path
            cls_x = features[-1] 
            if hasattr(self, 'norm_cls'): # Apply final norm before head for classification
                 cls_x = self.norm_cls(cls_x)
            # Global average pooling
            cls_out = self.head(cls_x.mean(dim=[-2, -1]))
            return cls_out


def get_patch_dims_hw_for_stages(img_size_wh, initial_stride, stage_strides):
    if isinstance(img_size_wh, int):
        img_w, img_h = img_size_wh, img_size_wh # Assuming square if int
    else: # tuple (W,H) or (H,W) - consistent with timm (H,W)
        img_h, img_w = img_size_wh[0], img_size_wh[1]


    patch_dims_list = []
    # After initial patch embed
    current_h, current_w = img_h // initial_stride, img_w // initial_stride
    patch_dims_list.append((current_h, current_w))

    # For subsequent stages after downsampling
    for s_stride in stage_strides: 
        current_h //= s_stride
        current_w //= s_stride
        patch_dims_list.append((current_h, current_w))
    return patch_dims_list


@register_model
def spanet_s(pretrained=False, img_size=224, in_chans=3, num_classes=1000, fork_feat=True, **kwargs):
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    
    # Calculate patch_dims_hw based on img_size passed to the factory function
    # Assuming in_stride=4 from default PatchEmbed, and down_stride=2 for inter-stage
    initial_embed_stride = kwargs.get('in_stride', 4)
    downsample_strides_between_stages = [kwargs.get('down_stride', 2)] * (len(layers) - 1)
    
    # img_size can be int or tuple (H,W)
    img_size_tuple = to_2tuple(img_size) # Ensures (H,W)
    patch_dims_hw = get_patch_dims_hw_for_stages(img_size_tuple, 
                                                 initial_stride=initial_embed_stride,
                                                 stage_strides=downsample_strides_between_stages)
    
    radius=[2**1, 2**1, 2**0, 2**0]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True] # Controls if PatchEmbed is added after a stage
    
    res_scale_init_values = kwargs.pop('res_scale_init_values', [None, None, 1.0, 1.0])
    if len(res_scale_init_values) != len(layers):
        res_scale_init_values = ([None]*(len(layers)-2) + [1.0]*2) if len(layers) >=2 else [None]*len(layers)


    model = SPANet(
        layers, embed_dims=embed_dims, patch_dims_hw=patch_dims_hw, radius=radius, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        num_classes=num_classes, fork_feat=fork_feat,
        img_size=img_size, in_chans=in_chans, # Pass to SPANet constructor
        res_scale_init_values=res_scale_init_values,
        **kwargs) # Pass remaining kwargs like in_stride, down_stride if needed by SPANet constructor
    
    model.default_cfg = default_cfgs['spanet_s']
    if pretrained and not fork_feat:
        if model.default_cfg.get('url'):
             model._load_pretrained_classifier(model.default_cfg['url'])
        else:
             logger.warning("spanet_s: pretrained=True for classifier but no URL in default_cfg.")
    return model


@register_model
def spanet_m(pretrained=False, img_size=224, in_chans=3, num_classes=1000, fork_feat=True, **kwargs):
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 320, 512]
    initial_embed_stride = kwargs.get('in_stride', 4)
    downsample_strides_between_stages = [kwargs.get('down_stride', 2)] * (len(layers) - 1)
    img_size_tuple = to_2tuple(img_size)
    patch_dims_hw = get_patch_dims_hw_for_stages(img_size_tuple, initial_stride=initial_embed_stride, stage_strides=downsample_strides_between_stages)
    
    radius=[2**1, 2**1, 2**0, 2**0]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    res_scale_init_values = kwargs.pop('res_scale_init_values', [None, None, 1.0, 1.0])
    if len(res_scale_init_values) != len(layers):
        res_scale_init_values = ([None]*(len(layers)-2) + [1.0]*2) if len(layers) >=2 else [None]*len(layers)

    model = SPANet(
        layers, embed_dims=embed_dims, patch_dims_hw=patch_dims_hw, radius=radius,
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        num_classes=num_classes, fork_feat=fork_feat,
        img_size=img_size, in_chans=in_chans,
        res_scale_init_values=res_scale_init_values,
        **kwargs)
    model.default_cfg = default_cfgs['spanet_m']
    if pretrained and not fork_feat:
        if model.default_cfg.get('url'):
             model._load_pretrained_classifier(model.default_cfg['url'])
        else:
             logger.warning("spanet_m: pretrained=True for classifier but no URL in default_cfg.")
    return model


@register_model
def spanet_mx(pretrained=False, img_size=224, in_chans=3, num_classes=1000, fork_feat=True, **kwargs):
    layers = [8, 8, 24, 8] 
    embed_dims = [64, 128, 320, 512] 
    initial_embed_stride = kwargs.get('in_stride', 4)
    downsample_strides_between_stages = [kwargs.get('down_stride', 2)] * (len(layers) - 1)
    img_size_tuple = to_2tuple(img_size)
    patch_dims_hw = get_patch_dims_hw_for_stages(img_size_tuple, initial_stride=initial_embed_stride, stage_strides=downsample_strides_between_stages)

    radius=[2**1, 2**1, 2**0, 2**0]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    res_scale_init_values = kwargs.pop('res_scale_init_values', [None, None, 1.0, 1.0])
    if len(res_scale_init_values) != len(layers):
        res_scale_init_values = ([None]*(len(layers)-2) + [1.0]*2) if len(layers) >=2 else [None]*len(layers)

    model = SPANet(
        layers, embed_dims=embed_dims, patch_dims_hw=patch_dims_hw, radius=radius,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        num_classes=num_classes, fork_feat=fork_feat,
        img_size=img_size, in_chans=in_chans,
        res_scale_init_values=res_scale_init_values,
        **kwargs)
    model.default_cfg = default_cfgs['spanet_m'] 
    if pretrained and not fork_feat:
        if model.default_cfg.get('url'): 
             model._load_pretrained_classifier(model.default_cfg['url'])
        else:
             logger.warning("spanet_mx: pretrained=True for classifier but no URL in default_cfg.")
    return model


@register_model
def spanet_b(pretrained=False, img_size=224, in_chans=3, num_classes=1000, fork_feat=True, **kwargs):
    layers = [6, 6, 18, 6] 
    embed_dims = [96, 192, 384, 768] 
    initial_embed_stride = kwargs.get('in_stride', 4)
    downsample_strides_between_stages = [kwargs.get('down_stride', 2)] * (len(layers) - 1)
    img_size_tuple = to_2tuple(img_size)
    patch_dims_hw = get_patch_dims_hw_for_stages(img_size_tuple, initial_stride=initial_embed_stride, stage_strides=downsample_strides_between_stages)
    
    radius=[2**1, 2**1, 2**0, 2**0]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    res_scale_init_values = kwargs.pop('res_scale_init_values', [None, None, 1.0, 1.0])
    if len(res_scale_init_values) != len(layers):
        res_scale_init_values = ([None]*(len(layers)-2) + [1.0]*2) if len(layers) >=2 else [None]*len(layers)
        
    model = SPANet(
        layers, embed_dims=embed_dims, patch_dims_hw=patch_dims_hw, radius=radius,
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        num_classes=num_classes, fork_feat=fork_feat,
        img_size=img_size, in_chans=in_chans,
        res_scale_init_values=res_scale_init_values,
        **kwargs)
    model.default_cfg = default_cfgs['spanet_b']
    if pretrained and not fork_feat:
        if model.default_cfg.get('url'):
             model._load_pretrained_classifier(model.default_cfg['url'])
        else:
             logger.warning("spanet_b: pretrained=True for classifier but no URL in default_cfg.")
    return model

@register_model
def spanet_bx(pretrained=False, img_size=224, in_chans=3, num_classes=1000, fork_feat=True, **kwargs):
    layers = [8, 8, 24, 8] 
    embed_dims = [96, 192, 384, 768] 
    initial_embed_stride = kwargs.get('in_stride', 4)
    downsample_strides_between_stages = [kwargs.get('down_stride', 2)] * (len(layers) - 1)
    img_size_tuple = to_2tuple(img_size)
    patch_dims_hw = get_patch_dims_hw_for_stages(img_size_tuple, initial_stride=initial_embed_stride, stage_strides=downsample_strides_between_stages)

    radius=[2**1, 2**1, 2**0, 2**0]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    res_scale_init_values = kwargs.pop('res_scale_init_values', [None, None, 1.0, 1.0])
    if len(res_scale_init_values) != len(layers):
        res_scale_init_values = ([None]*(len(layers)-2) + [1.0]*2) if len(layers) >=2 else [None]*len(layers)

    model = SPANet(
        layers, embed_dims=embed_dims, patch_dims_hw=patch_dims_hw, radius=radius,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        num_classes=num_classes, fork_feat=fork_feat,
        img_size=img_size, in_chans=in_chans,
        res_scale_init_values=res_scale_init_values,
        **kwargs)
    model.default_cfg = default_cfgs['spanet_b']
    if pretrained and not fork_feat:
        if model.default_cfg.get('url'):
             model._load_pretrained_classifier(model.default_cfg['url'])
        else:
             logger.warning("spanet_bx: pretrained=True for classifier but no URL in default_cfg.")
    return model


if __name__ == "__main__": 
    logging.basicConfig(level=logging.INFO)

    # Test case 1: Classification (fork_feat=False)
    print("\n--- Testing SPANet Small (Classification, CPU) ---")
    model_cls = spanet_s(num_classes=100, fork_feat=False, img_size=224)
    model_cls.eval()
    input_tensor = torch.rand(2, 3, 224, 224)
    output_cls = model_cls(input_tensor)
    print(f"Classification output shape: {output_cls.shape}")
    print(f"Width list: {model_cls.width_list}")

    # Test case 2: Feature Extraction (fork_feat=True, CPU)
    print("\n--- Testing SPANet Small (Feature Extraction, CPU) ---")
    model_feat = spanet_s(fork_feat=True, img_size=224)
    model_feat.eval()
    output_feat_list = model_feat(input_tensor)
    print(f"Feature extraction output: list of {len(output_feat_list)} tensors")
    for i, feat in enumerate(output_feat_list):
        print(f"  Feature {i} shape: {feat.shape}, dtype: {feat.dtype}")
    print(f"Width list: {model_feat.width_list}")

    # Test with different image size (CPU)
    print("\n--- Testing SPANet Medium (Feature Extraction, 256x256 input, CPU) ---")
    model_feat_m256 = spanet_m(fork_feat=True, img_size=256)
    model_feat_m256.eval()
    input_tensor_256 = torch.rand(1, 3, 256, 256)
    output_feat_list_m256 = model_feat_m256(input_tensor_256)
    print(f"Feature extraction output (256x256): list of {len(output_feat_list_m256)} tensors")
    for i, feat in enumerate(output_feat_list_m256):
        print(f"  Feature {i} shape: {feat.shape}, dtype: {feat.dtype}")
    print(f"Width list: {model_feat_m256.width_list}")

    # Test with half precision if CUDA is available
    if torch.cuda.is_available():
        print("\n--- Testing SPANet Small (Feature Extraction, CUDA, FP16) ---")
        model_feat_cuda = spanet_s(fork_feat=True, img_size=224).cuda().half()
        model_feat_cuda.eval()
        input_tensor_cuda_half = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.half)
        with torch.no_grad(): # Important for inference
            output_feat_list_cuda = model_feat_cuda(input_tensor_cuda_half)
        print(f"Feature extraction output (CUDA, FP16): list of {len(output_feat_list_cuda)} tensors")
        for i, feat in enumerate(output_feat_list_cuda):
            print(f"  Feature {i} shape: {feat.shape}, dtype: {feat.dtype}")
        print(f"Width list (from CPU model instance, as dummy fwd was on CPU): {model_feat_cuda.width_list}")
        # Note: width_list is calculated during __init__ on the device the model is initially on (usually CPU).
        # If model is moved to CUDA and then width_list is needed based on CUDA operations,
        # dummy forward for width_list should happen after .cuda() call.
        # Current implementation does it in __init__.
    else:
        print("\nCUDA not available, skipping FP16 test.")

    print("\nConceptual check: If used in Ultralytics, the model.forward(x) should return a list of feature maps.")
    print("The modified SPANet with fork_feat=True now does this.")