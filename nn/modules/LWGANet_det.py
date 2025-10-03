# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Modifications copyright (c) [Your Name/Org], 2023

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from typing import List, Type, Union, Tuple
import copy
import warnings
import os # Added for dummy file cleanup in example

try:
    import antialiased_cnns
    has_antialiased_cnns = True
except ImportError:
    warnings.warn("antialiased-cnns library not found. MRA module might not work as intended.")
    # Define a placeholder if the library is missing, or raise an error
    class BlurPool(nn.Module):
        def __init__(self, channels, stride):
            super().__init__()
            warnings.warn("Using AvgPool2d as a fallback for antialiased_cnns.BlurPool.")
            self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride) # Or MaxPool2d
        def forward(self, x):
            return self.pool(x)
    antialiased_cnns = type('obj', (object,), {'BlurPool': BlurPool})() # Mock object
    has_antialiased_cnns = False


# Helper function to update weights similar to Swin's example
def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    original_checkpoint_keys = set(weight_dict.keys())
    matched_keys_in_model = set()

    for k, v in weight_dict.items():
        model_key = k
        if k.startswith("module."):
             model_key = k[len("module."):]

        if model_key in model_dict.keys() and model_dict[model_key].shape == v.shape:
            # Skip loading classification head weights if they exist in the checkpoint
            # but not in the current model (which lacks the head)
            if not model_key.startswith('head.') and not model_key.startswith('avgpool_pre_head.'):
                temp_dict[model_key] = v
                matched_keys_in_model.add(model_key)
                idx += 1
        # Add more sophisticated matching if needed

    print(f'Attempting to load {len(weight_dict)} weights into model...')
    loaded_items = len(temp_dict)
    total_model_items = len(model_dict)
    # Adjust total count if model has fewer layers than checkpoint (e.g., no head)
    relevant_model_items = sum(1 for k in model_dict if not k.startswith('head.') and not k.startswith('avgpool_pre_head.'))
    print(f'Successfully matched {loaded_items}/{relevant_model_items} relevant items in model.')

    model_dict.update(temp_dict)

    missing_keys = [k for k in model_dict.keys() if k not in matched_keys_in_model]
    unexpected_keys = list(original_checkpoint_keys - set(temp_dict.keys()) - {k for k in original_checkpoint_keys if k.startswith("module.") and k[len("module."):] in temp_dict})
    # Filter out head keys from unexpected if they were intentionally skipped
    unexpected_keys = [k for k in unexpected_keys if not k.startswith('head.') and not k.startswith('avgpool_pre_head.')]


    return model_dict, missing_keys, unexpected_keys


# Define common type hints
NormLayerType = Type[Union[nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm]]
ActLayerType = Type[Union[nn.ReLU, nn.GELU, nn.SiLU]]

# --- DRFD, PA, LA, MRA, GA12, GA, D_GA, LWGA_Block, BasicStage, Stem ---
# (Keep these helper module classes exactly as they were in the previous version)
class DRFD(nn.Module):
    def __init__(self, dim: int, norm_layer: NormLayerType, act_layer: ActLayerType):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=2, padding=1, groups=dim*2)
        self.act_c = act_layer()
        self.norm_c = norm_layer(dim*2)
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = norm_layer(dim*2)
        self.fusion = nn.Conv2d(dim*4, self.outdim, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        max_out = self.norm_m(self.max_m(x))
        conv_out = self.norm_c(self.act_c(self.conv_c(x)))
        x = torch.cat([conv_out, max_out], dim=1)
        x = self.fusion(x)
        return x

class PA(nn.Module):
    def __init__(self, dim: int, norm_layer: NormLayerType, act_layer: ActLayerType):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim*4, 1, bias=False),
            norm_layer(dim*4),
            act_layer(),
            nn.Conv2d(dim*4, dim, 1, bias=False)
        )
        self.gate_fn = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.p_conv(x)
        x = x * self.gate_fn(att)
        return x

class LA(nn.Module):
    def __init__(self, dim: int, norm_layer: NormLayerType, act_layer: ActLayerType):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            norm_layer(dim),
            act_layer()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x

class MRA(nn.Module):
    def __init__(self, channel: int, att_kernel: int, norm_layer: NormLayerType):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = antialiased_cnns.BlurPool(channel, stride=3)
        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.norm = norm_layer(channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tem = self.max_m1(x)
        x_tem = self.max_m2(x_tem)
        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)
        x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
        x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))
        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)
        out = x[:, :self.channel, :, :] * F.interpolate(self.gate_fn(att),
                                                        size=(x.shape[-2], x.shape[-1]),
                                                        mode='nearest')
        return out

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class GA12(nn.Module):
    def __init__(self, dim: int, act_layer: ActLayerType):
        super().__init__()
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), stride=2, padding=0)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = act_layer()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        if dim // 2 == 0: # Handle case where dim is 1
             self.conv1 = nn.Identity() # Or some other placeholder
             self.conv2 = nn.Identity()
             self.conv = nn.Identity()
             warnings.warn(f"GA12 received dim={dim}, which is too small for conv1/conv2/conv. Using Identity.")
        else:
             self.conv1 = nn.Conv2d(dim, dim // 2, 1)
             self.conv2 = nn.Conv2d(dim, dim // 2, 1)
             self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)

        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_orig, W_orig = x.shape[-2:]
        pad_h = (2 - H_orig % 2) % 2
        pad_w = (2 - W_orig % 2) % 2
        if pad_h > 0 or pad_w > 0:
             x_padded = F.pad(x, (0, pad_w, 0, pad_h))
             x_, idx = self.downpool(x_padded)
        else:
             x_, idx = self.downpool(x)
             x_padded = x

        x_ = self.proj_1(x_)
        x_ = self.activation(x_)
        attn1 = self.conv0(x_)
        attn2 = self.conv_spatial(attn1)

        # Handle dim=1 case where conv1/2/ are Identity
        if isinstance(self.conv1, nn.Identity):
             attn1_split = attn1
             attn2_split = attn2
             attn_combined = torch.cat([attn1, attn2], dim=1) # For avg/max pooling
             attn_conv = attn1 # Placeholder, won't be used meaningfully
        else:
            attn1_split = self.conv1(attn1)
            attn2_split = self.conv2(attn2)
            attn_combined = torch.cat([attn1_split, attn2_split], dim=1)
            attn_conv = self.conv # Use the actual conv layer

        avg_attn = torch.mean(attn_combined, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn_combined, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()

        # Apply sigmoid weighting, handling dim=1 case
        if isinstance(self.conv1, nn.Identity):
             attn = x_ # Pass input through if conv layers are identity
        else:
             weighted_attn = attn1_split * sig[:, 0:1] + attn2_split * sig[:, 1:2] # Use slicing for channel dim
             attn = attn_conv(weighted_attn) # Apply final conv

        x_ = x_ * attn
        x_ = self.proj_2(x_)

        output_size=(x_padded.size(0), x_padded.size(1), x_padded.size(2), x_padded.size(3))
        x_unpooled = self.uppool(x_, indices=idx, output_size=output_size)

        if pad_h > 0 or pad_w > 0:
             x_unpooled = x_unpooled[:, :, :H_orig, :W_orig]

        return x_unpooled


class GA(nn.Module):
    def __init__(self, dim: int, head_dim: int = 64, num_heads: int = None, qkv_bias: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0., proj_bias: bool = False, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads if num_heads else max(1, dim // head_dim) # Ensure at least 1 head
        if dim // self.num_heads == 0 and dim > 0: # If dim < num_heads requested (and dim not 0)
             warnings.warn(f"GA dim ({dim}) < num_heads ({self.num_heads}). Setting head_dim={dim}, num_heads=1.")
             self.num_heads = 1
             self.head_dim = dim
        elif dim % self.num_heads != 0:
             old_head_dim = self.head_dim
             self.head_dim = dim // self.num_heads
             warnings.warn(f"GA dim ({dim}) not divisible by num_heads ({self.num_heads}). Adjusting head_dim from {old_head_dim} to {self.head_dim}.")
        else:
             self.head_dim = dim // self.num_heads

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Handle C=0 case gracefully (e.g., from uneven split)
        if C == 0:
             return x # Return input as is
        x_flat = x.flatten(2).transpose(1, 2)
        N = H * W
        if self.qkv.in_features != C:
             raise ValueError(f"GA input channel dim {C} != qkv layer expected dim {self.qkv.in_features}")

        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, self.attention_dim)
        x_proj = self.proj(x_attn)
        x_proj = self.proj_drop(x_proj)
        x_out = x_proj.transpose(1, 2).reshape(B, C, H, W)
        return x_out


class D_GA(nn.Module):
    def __init__(self, dim: int, norm_layer: NormLayerType, act_layer: ActLayerType):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = GA12(dim, act_layer)
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle dim=0 case
        if x.shape[1] == 0:
            warnings.warn("D_GA received input with 0 channels. Returning input.")
            return x

        H_orig, W_orig = x.shape[-2:]
        pad_h = (2 - H_orig % 2) % 2
        pad_w = (2 - W_orig % 2) % 2
        if pad_h > 0 or pad_w > 0:
             x_padded = F.pad(x, (0, pad_w, 0, pad_h))
             x_, idx = self.downpool(x_padded)
        else:
             x_, idx = self.downpool(x)
             x_padded = x

        x_attn = self.attn(x_)
        x_norm = self.norm(x_attn)

        output_size=(x_padded.size(0), x_padded.size(1), x_padded.size(2), x_padded.size(3))
        x_unpooled = self.uppool(x_norm, indices=idx, output_size=output_size)

        if pad_h > 0 or pad_w > 0:
             x_unpooled = x_unpooled[:, :, :H_orig, :W_orig]

        return x_unpooled

class LWGA_Block(nn.Module):
    def __init__(self,
                 dim: int,
                 stage: int,
                 att_kernel: int,
                 mlp_ratio: float,
                 drop_path: float,
                 act_layer: ActLayerType,
                 norm_layer: NormLayerType
                 ):
        super().__init__()
        self.stage = stage
        if dim <= 0:
             raise ValueError(f"LWGA_Block received non-positive dimension: {dim}")

        if dim < 4:
             warnings.warn(f"Dimension ({dim}) is less than 4 in LWGA_Block (Stage {stage}). Splitting may behave unexpectedly.")
             # Handle small dimensions: assign to first splits, leave last ones potentially empty
             self.dim_split = 1 if dim >=1 else 0
             self.dim_split2 = 1 if dim >=2 else 0
             self.dim_split3 = 1 if dim >=3 else 0
             self.dim_split_last = max(0, dim - self.dim_split - self.dim_split2 - self.dim_split3) # Remainder
             dims = [self.dim_split, self.dim_split2, self.dim_split3, self.dim_split_last]
        elif dim % 4 != 0:
             warnings.warn(f"Dimension ({dim}) is not divisible by 4 for splitting in LWGA_Block (Stage {stage}). Channels will be split unevenly.")
             self.dim_split = dim // 4
             self.dim_split_last = dim - 3 * self.dim_split
             dims = [self.dim_split] * 3 + [self.dim_split_last]
        else:
            self.dim_split = dim // 4
            self.dim_split_last = self.dim_split
            dims = [self.dim_split] * 4

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]
        self.mlp = nn.Sequential(*mlp_layer)

        # Instantiate modules only if dimension > 0
        self.PA = PA(dims[0], norm_layer, act_layer) if dims[0] > 0 else nn.Identity()
        self.LA = LA(dims[1], norm_layer, act_layer) if dims[1] > 0 else nn.Identity()
        self.MRA = MRA(dims[2], att_kernel, norm_layer) if dims[2] > 0 else nn.Identity()

        if dims[3] > 0:
            if stage == 2:
                self.GA_module = D_GA(dims[3], norm_layer, act_layer)
            elif stage == 3:
                self.GA_module = GA(dims[3])
                self.norm_ga = norm_layer(dims[3])
            else: # Stages 0 and 1
                self.GA_module = GA12(dims[3], act_layer)
                self.norm_ga = norm_layer(dims[3])
        else: # Handle zero dimension for GA branch
            self.GA_module = nn.Identity()
            self.norm_ga = nn.Identity()


        self.norm1 = norm_layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        if x.shape[1] < 4 and x.shape[1] > 0 : # Recalculate dims for forward pass if dim < 4
             dim = x.shape[1]
             d1 = 1 if dim >=1 else 0
             d2 = 1 if dim >=2 else 0
             d3 = 1 if dim >=3 else 0
             d4 = max(0, dim - d1 - d2 - d3)
             split_sizes = [d1, d2, d3, d4]
             # Filter out zero-sized splits for torch.split
             split_sizes_fwd = [s for s in split_sizes if s > 0]
             xs = list(torch.split(x, split_sizes_fwd, dim=1))
             # Pad xs list with None if original splits were zero
             xs_padded = [None] * 4
             current_xs_idx = 0
             for i in range(4):
                 if split_sizes[i] > 0:
                     xs_padded[i] = xs[current_xs_idx]
                     current_xs_idx += 1
             x1, x2, x3, x4 = xs_padded[0], xs_padded[1], xs_padded[2], xs_padded[3]
        elif x.shape[1] > 0: # Standard split logic
            split_sizes = [self.dim_split] * 3 + [self.dim_split_last]
            x1, x2, x3, x4 = torch.split(x, split_sizes, dim=1)
        else: # Handle input with zero channels
             x1, x2, x3, x4 = None, None, None, None


        # Apply modules, checking for Identity (if dim was 0) or None (if split resulted in 0)
        x1_res = x1 + self.PA(x1) if x1 is not None and not isinstance(self.PA, nn.Identity) else x1
        x2_att = self.LA(x2) if x2 is not None and not isinstance(self.LA, nn.Identity) else x2
        x3_att = self.MRA(x3) if x3 is not None and not isinstance(self.MRA, nn.Identity) else x3
        x4_processed = x4 # Placeholder

        if x4 is not None and not isinstance(self.GA_module, nn.Identity):
            if self.stage == 3:
                x4_att = self.GA_module(x4)
                x4_processed = self.norm_ga(x4 + x4_att) # Residual inside GA block logic for stage 3
            elif self.stage == 2:
                 x4_att = self.GA_module(x4)
                 x4_processed = x4 + x4_att # Residual outside for stage 2
            else: # Stage 0, 1
                x4_att = self.GA_module(x4)
                x4_processed = self.norm_ga(x4 + x4_att) # Residual inside for stage 0/1

        # Concatenate valid tensors
        valid_tensors = [t for t in [x1_res, x2_att, x3_att, x4_processed] if t is not None and t.shape[1]>0]
        if not valid_tensors: # If all splits were None or zero channels
            x_att = shortcut # Or handle as error
        else:
            x_att = torch.cat(valid_tensors, dim=1)


        # MLP and final residual
        x_mlp = self.mlp(x_att)
        x = shortcut + self.drop_path(self.norm1(x_mlp))

        return x


class BasicStage(nn.Module):
    def __init__(self,
                 dim: int,
                 stage: int,
                 depth: int,
                 att_kernel: int,
                 mlp_ratio: float,
                 drop_path: Union[List[float], float],
                 norm_layer: NormLayerType,
                 act_layer: ActLayerType
                 ):
        super().__init__()
        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth

        blocks_list = [
            LWGA_Block(
                dim=dim, stage=stage, att_kernel=att_kernel, mlp_ratio=mlp_ratio,
                drop_path=drop_path[i], norm_layer=norm_layer, act_layer=act_layer
            )
            for i in range(depth)
        ]
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return x

class Stem(nn.Module):
    def __init__(self, in_chans: int, stem_dim: int, norm_layer: NormLayerType = None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, stem_dim, kernel_size=4, stride=4, bias=False)
        self.norm = norm_layer(stem_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


class LWGANet(nn.Module):
    """
    LWGANet model implementation focused on feature extraction.
    Provides `width_list` for compatibility with detection frameworks.
    """
    def __init__(self,
                 in_chans: int = 3,
                 stem_dim: int = 64,
                 depths: Tuple[int, ...] = (1, 2, 4, 2),
                 att_kernel: Tuple[int, ...] = (11, 11, 11, 11),
                 norm_layer: NormLayerType = nn.BatchNorm2d,
                 act_layer: ActLayerType = nn.GELU,
                 mlp_ratio: float = 2.,
                 stem_norm: bool = True,
                 drop_path_rate: float = 0.1,
                 pretrained: str = None,  # Path to pretrained weights
                 **kwargs):
        super().__init__()

        # Model is always in feature extraction mode
        self.num_stages = len(depths)
        self.num_features_list = [int(stem_dim * 2 ** i) for i in range(self.num_stages)]
        self.num_features = self.num_features_list[-1] # Features of the last stage output

        if stem_dim == 96 and act_layer == nn.GELU:
            print("Using nn.ReLU based on stem_dim=96 as per original convention.")
            act_layer = nn.ReLU

        self.stem = Stem(
            in_chans=in_chans, stem_dim=stem_dim,
            norm_layer=norm_layer if stem_norm else None
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        current_dim = stem_dim
        dp_offset = 0
        for i_stage in range(self.num_stages):
            stage_depth = depths[i_stage]
            stage_drop_path = dpr[dp_offset : dp_offset + stage_depth]
            stage = BasicStage(
                dim=current_dim, stage=i_stage, depth=stage_depth,
                att_kernel=att_kernel[i_stage], mlp_ratio=mlp_ratio,
                drop_path=stage_drop_path, norm_layer=norm_layer, act_layer=act_layer
            )
            self.stages.append(stage)
            dp_offset += stage_depth

            if i_stage < self.num_stages - 1:
                downsample_layer = DRFD(
                    dim=current_dim, norm_layer=norm_layer, act_layer=act_layer
                )
                self.stages.append(downsample_layer)
                current_dim = downsample_layer.outdim

        # Layers for feature extraction
        self.out_indices = [i*2 for i in range(self.num_stages)] # Indices of BasicStage outputs
        for i, stage_idx in enumerate(self.out_indices):
            layer_dim = self.num_features_list[i]
            layer = norm_layer(layer_dim)
            layer_name = f'norm{stage_idx}'
            self.add_module(layer_name, layer)

        # Classification head removed
        # self.avgpool_pre_head = ...
        # self.head = ...

        # --- Calculate width_list ---
        self.width_list = None
        try:
            print("Calculating feature widths (width_list)...")
            self.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, in_chans, 640, 640)
                try:
                    param = next(self.parameters())
                    dummy_input = dummy_input.to(param.device, param.dtype)
                except StopIteration: pass
                outputs = self._forward_det_impl(dummy_input)
                self.width_list = [o.shape[1] for o in outputs]
                print(f"Calculated feature widths (width_list): {self.width_list}")
            self.train()
        except Exception as e:
            warnings.warn(f"Could not calculate width_list due to error: {e}. Setting width_list to None.")
            self.width_list = None

        # Set the main forward method directly to feature extraction
        self.forward = self.forward_det

        # Initialize weights
        self.apply(self.cls_init_weights) # Use same init func, it skips head layers if missing
        if pretrained:
            self.init_weights(pretrained)


    def cls_init_weights(self, m):
        """Initialize weights (skips non-existent head)."""
        if isinstance(m, nn.Linear):
            # Only init if it's not the (now removed) head layer
            is_head = getattr(m, '_is_head', False) # Check for a potential marker if needed
            if not is_head:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
             # Skip init for avgpool_pre_head if needed, though it's removed now
            is_pre_head = getattr(m, '_is_pre_head', False)
            if not is_pre_head:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
             if m.bias is not None:
                nn.init.constant_(m.bias, 0)
             if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained: str):
        """Initialize weights from a pretrained checkpoint."""
        print(f"Loading pretrained weights from: {pretrained}")
        if pretrained:
            try:
                checkpoint = torch.load(pretrained, map_location='cpu')
                state_dict = None
                if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint: state_dict = checkpoint['model']
                else: state_dict = checkpoint
                if state_dict is None: raise ValueError("Could not find model state_dict in checkpoint.")

                model_dict = self.state_dict()
                # Filter checkpoint dict to exclude head layers before passing to update_weight
                state_dict_filtered = {k: v for k, v in state_dict.items() if not k.startswith('head.') and not k.startswith('avgpool_pre_head.')}

                adapted_state_dict, missing_keys, unexpected_keys = update_weight(model_dict, state_dict_filtered)
                self.load_state_dict(adapted_state_dict, strict=False)

                print(f"Pretrained weights loaded.")
                if missing_keys: print("Missing keys:", missing_keys)
                if unexpected_keys: print("Unexpected keys (in checkpoint but not used by model):", unexpected_keys) # Already filtered head keys

            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
        else:
            print("No pretrained weights path provided.")


    def _forward_det_impl(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Internal implementation for feature extraction."""
        x = self.stem(x)
        outs = []
        for idx, stage_module in enumerate(self.stages):
            x = stage_module(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs

    def forward_det(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass for feature extraction (detection/segmentation)."""
        return self._forward_det_impl(x)

    # forward_cls method removed

    # Default forward is set in __init__ to self.forward_det
    # def forward(self, x):
    #     return self.forward_det(x)


# --- Model Variant Functions --- (Removed num_classes and fork_feat)

def LWGANet_L0_1242_e32_k11_GELU(pretrained: str = None, **kwargs) -> LWGANet:
    model = LWGANet(in_chans=3, stem_dim=32, depths=(1, 2, 4, 2),
                    att_kernel=(11, 11, 11, 11), norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                    drop_path_rate=0.0, pretrained=pretrained, **kwargs)
    return model

def LWGANet_L1_1242_e64_k11_GELU(pretrained: str = None, **kwargs) -> LWGANet:
    model = LWGANet(in_chans=3, stem_dim=64, depths=(1, 2, 4, 2),
                    att_kernel=(11, 11, 11, 11), norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                    drop_path_rate=0.1, pretrained=pretrained, **kwargs)
    return model

def LWGANet_L2_1442_e96_k11_ReLU(pretrained: str = None, **kwargs) -> LWGANet:
    model = LWGANet(in_chans=3, stem_dim=96, depths=(1, 4, 4, 2),
                    att_kernel=(11, 11, 11, 11), norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU,
                    drop_path_rate=0.1, pretrained=pretrained, **kwargs)
    return model


# Example Usage:
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Test Feature Extraction (Now the only mode) ---
    print("\nTesting LWGANet L1 (Feature Extraction Only Mode)...")
    # Create model (no fork_feat arg needed)
    model_det = LWGANet_L1_1242_e64_k11_GELU().to(device)
    model_det.eval()
    try:
        input_tensor_det = torch.randn(1, 3, 640, 640).to(device)
        with torch.no_grad():
            output_features = model_det(input_tensor_det) # Calls forward_det
        print("Feature extraction model created successfully.")
        print("Input shape:", input_tensor_det.shape)
        print("Number of output feature maps:", len(output_features))
        for i, feat in enumerate(output_features):
            print(f"Feature map {i} shape: {feat.shape}")
        if hasattr(model_det, 'width_list') and model_det.width_list:
             print("Calculated width_list:", model_det.width_list)
             assert len(model_det.width_list) == len(output_features)
             for i, w in enumerate(model_det.width_list):
                 assert w == output_features[i].shape[1]
             print("width_list matches feature map channels.")
        else:
             print("width_list attribute not found or is None/empty.")

    except Exception as e:
        print(f"Error during feature extraction test: {e}")
        import traceback
        traceback.print_exc()


    # --- Test Pretrained Weight Loading (Example) ---
    print("\nTesting pretrained weight loading (Example)...")
    dummy_state_dict_path = "dummy_lwganet_detonly_weights.pth"
    # Save state dict of the feature extraction model
    torch.save({'model': model_det.state_dict()}, dummy_state_dict_path)

    try:
        # Load the model specifying the dummy pretrained path
        model_pretrained = LWGANet_L1_1242_e64_k11_GELU(pretrained=dummy_state_dict_path).to(device)
        print("Model created with dummy pretrained weights.")
        model_pretrained.eval()
        with torch.no_grad():
             output_pretrained = model_pretrained(input_tensor_det) # Test forward pass
        print("Forward pass successful after loading weights.")
        print(f"Output has {len(output_pretrained)} feature maps.")

    except Exception as e:
        print(f"Error during pretrained weight loading test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(dummy_state_dict_path):
            os.remove(dummy_state_dict_path)

    # --- Test integration possibility (Conceptual) ---
    print("\nConceptual check for YOLO integration:")
    if hasattr(model_det, 'width_list') and model_det.width_list:
        print(f"Model has 'width_list': {model_det.width_list}. Should be compatible.")
        # In ultralytics, this model could be used like:
        # cfg['backbone'] = model_det
        # model = DetectionModel(cfg)
    else:
        print("Model missing 'width_list'. Would cause issues with YOLO parsing.")