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
        # Handle potential 'module.' prefix from DataParallel/DDP
        if k.startswith("module."):
             model_key = k[len("module."):]

        if model_key in model_dict.keys() and model_dict[model_key].shape == v.shape:
            temp_dict[model_key] = v
            matched_keys_in_model.add(model_key)
            idx += 1
        # Add more sophisticated matching if needed (e.g., removing "backbone.")

    print(f'Attempting to load {len(weight_dict)} weights into model...')
    loaded_items = len(temp_dict)
    total_model_items = len(model_dict)
    print(f'Successfully matched {loaded_items}/{total_model_items} items.')

    # Update the model dictionary with matched weights
    model_dict.update(temp_dict)

    # Find missing and unexpected keys
    missing_keys = [k for k in model_dict.keys() if k not in matched_keys_in_model]
    # Unexpected keys are those in the checkpoint but not matched to the model state dict
    unexpected_keys = list(original_checkpoint_keys - set(temp_dict.keys()) - {k for k in original_checkpoint_keys if k.startswith("module.") and k[len("module."):] in temp_dict})


    return model_dict, missing_keys, unexpected_keys


# Define common type hints
NormLayerType = Type[Union[nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm]]
ActLayerType = Type[Union[nn.ReLU, nn.GELU, nn.SiLU]]

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
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
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
             x_padded = x # For output size calculation if no padding

        x_ = self.proj_1(x_)
        x_ = self.activation(x_)
        attn1 = self.conv0(x_)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        x_ = x_ * attn
        x_ = self.proj_2(x_)

        # Ensure unpooling output matches the size *before* potential padding
        output_size=(x_padded.size(0), x_padded.size(1), x_padded.size(2), x_padded.size(3))
        x_unpooled = self.uppool(x_, indices=idx, output_size=output_size)

        # Crop if padding was added
        if pad_h > 0 or pad_w > 0:
             x_unpooled = x_unpooled[:, :, :H_orig, :W_orig]

        return x_unpooled


class GA(nn.Module):
    # MHSA operating on flattened sequences
    def __init__(self, dim: int, head_dim: int = 64, num_heads: int = None, qkv_bias: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0., proj_bias: bool = False, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            warnings.warn(f"GA dim ({dim}) < head_dim ({head_dim}), setting num_heads=1 and adapting head_dim.")
            self.num_heads = 1
            self.head_dim = dim
        self.attention_dim = self.num_heads * self.head_dim
        if self.attention_dim != dim:
             warnings.warn(f"GA calculated attention dim ({self.attention_dim}) != input dim ({dim}). Mismatch can occur if dim not divisible by head_dim.")
             # Proj layer below handles mapping back to `dim`
             pass

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
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
        if dim % 4 != 0:
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

        self.PA = PA(dims[0], norm_layer, act_layer)
        self.LA = LA(dims[1], norm_layer, act_layer)
        self.MRA = MRA(dims[2], att_kernel, norm_layer)

        if stage == 2:
            self.GA_module = D_GA(dims[3], norm_layer, act_layer)
        elif stage == 3:
            self.GA_module = GA(dims[3])
            self.norm_ga = norm_layer(dims[3])
        else: # Stages 0 and 1
            self.GA_module = GA12(dims[3], act_layer)
            self.norm_ga = norm_layer(dims[3])

        self.norm1 = norm_layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        split_sizes = [self.dim_split] * 3 + [self.dim_split_last]
        x1, x2, x3, x4 = torch.split(x, split_sizes, dim=1)

        x1_att = self.PA(x1)
        x2_att = self.LA(x2)
        x3_att = self.MRA(x3)

        if self.stage == 3:
            x4_att = self.GA_module(x4)
            x4 = self.norm_ga(x4 + x4_att)
        elif self.stage == 2:
             x4_att = self.GA_module(x4)
             x4 = x4 + x4_att
        else:
            x4_att = self.GA_module(x4)
            x4 = self.norm_ga(x4 + x4_att)

        x1 = x1 + x1_att # Residual only on PA branch
        x_att = torch.cat((x1, x2_att, x3_att, x4), dim=1)

        x = shortcut + self.drop_path(self.norm1(self.mlp(x_att)))
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
    LWGANet model implementation supporting both classification and feature extraction.
    Includes `width_list` calculation for compatibility with detection frameworks.
    """
    def __init__(self,
                 in_chans: int = 3,
                 num_classes: int = 1000, # Used only if fork_feat=False
                 stem_dim: int = 64,
                 depths: Tuple[int, ...] = (1, 2, 4, 2),
                 att_kernel: Tuple[int, ...] = (11, 11, 11, 11),
                 norm_layer: NormLayerType = nn.BatchNorm2d,
                 act_layer: ActLayerType = nn.GELU,
                 mlp_ratio: float = 2.,
                 stem_norm: bool = True,
                 feature_dim: int = 1280, # Dim before final classifier head
                 drop_path_rate: float = 0.1,
                 fork_feat: bool = True, # Set default to True for YOLO compatibility
                 pretrained: str = None,  # Path to pretrained weights
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.fork_feat = fork_feat # Store fork_feat setting
        self.num_stages = len(depths)
        self.num_features_list = [int(stem_dim * 2 ** i) for i in range(self.num_stages)]
        self.num_features = self.num_features_list[-1] # Features of the last stage output

        # Adjust act_layer based on stem_dim if needed
        if stem_dim == 96 and act_layer == nn.GELU:
            print("Using nn.ReLU based on stem_dim=96 as per original convention.")
            act_layer = nn.ReLU

        self.stem = Stem(
            in_chans=in_chans, stem_dim=stem_dim,
            norm_layer=norm_layer if stem_norm else None
        )

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages (ModuleList allows easier iteration for feature extraction)
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
            self.stages.append(stage) # Append BasicStage
            dp_offset += stage_depth

            # Add Patch Merging (DRFD) layer, except after the last stage
            if i_stage < self.num_stages - 1:
                downsample_layer = DRFD(
                    dim=current_dim, norm_layer=norm_layer, act_layer=act_layer
                )
                self.stages.append(downsample_layer) # Append DRFD
                current_dim = downsample_layer.outdim # Update dim for next stage

        # Layers for feature extraction (needed for width_list calculation and potentially by framework)
        # Indices corresponding to outputs of BasicStage modules in the ModuleList
        self.out_indices = [i*2 for i in range(self.num_stages)] # 0, 2, 4, 6...
        # Add a norm layer for each feature output (used in forward_det)
        for i, stage_idx in enumerate(self.out_indices):
            layer_dim = self.num_features_list[i]
            layer = norm_layer(layer_dim)
            layer_name = f'norm{stage_idx}' # Use index from ModuleList
            self.add_module(layer_name, layer)

        # Layers for classification head (created even if fork_feat=True, used by forward_cls)
        self.avgpool_pre_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
            act_layer()
        )
        self.head = nn.Linear(feature_dim, num_classes) if num_classes > 0 else nn.Identity()

        # --- Calculate width_list ---
        # Requires running forward_det logic temporarily
        self.width_list = None # Initialize
        try:
            # print("Calculating feature widths (width_list)...")
            self.eval() # Set to eval mode for dummy pass
            with torch.no_grad():
                # Use a representative input size (e.g., standard YOLO size)
                dummy_input = torch.randn(1, in_chans, 640, 640)
                # Try to match device and dtype of model parameters
                try:
                    param = next(self.parameters())
                    dummy_input = dummy_input.to(param.device, param.dtype)
                except StopIteration:
                     pass # Model has no parameters yet

                outputs = self._forward_det_impl(dummy_input) # Use internal method
                self.width_list = [o.shape[1] for o in outputs] # Channel dimension
                # print(f"Calculated feature widths (width_list): {self.width_list}")
            self.train() # Set back to train mode

        except Exception as e:
            warnings.warn(f"Could not calculate width_list due to error: {e}. Setting width_list to None.")
            self.width_list = None # Ensure it's None on failure


        # Set the main forward method based on fork_feat AFTER width_list calculation
        if self.fork_feat:
             self.forward = self.forward_det # Use feature extraction forward
        else:
             self.forward = self.forward_cls # Use classification forward

        # Initialize weights
        self.apply(self.cls_init_weights)
        # Load pretrained weights if path is provided
        if pretrained:
            self.init_weights(pretrained)


    def cls_init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
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
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

                if state_dict is None:
                     raise ValueError("Could not find model state_dict in checkpoint.")

                model_dict = self.state_dict()
                adapted_state_dict, missing_keys, unexpected_keys = update_weight(model_dict, state_dict)
                self.load_state_dict(adapted_state_dict, strict=False)

                print(f"Pretrained weights loaded.")
                if missing_keys:
                    print("Missing keys:", missing_keys)
                if unexpected_keys:
                    print("Unexpected keys:", unexpected_keys)

            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
        else:
            print("No pretrained weights path provided.")


    def _forward_det_impl(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Internal implementation for feature extraction used by forward_det and width_list calc."""
        x = self.stem(x)
        outs = []
        # Iterate through the ModuleList containing BasicStages and DRFDs
        for idx, stage_module in enumerate(self.stages):
            x = stage_module(x)
            # If the index corresponds to the output of a BasicStage
            if idx in self.out_indices:
                # Apply the corresponding normalization layer
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs

    def forward_det(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass for feature extraction (detection/segmentation)."""
        return self._forward_det_impl(x)

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification."""
        # Pass through all stages first
        x = self.stem(x)
        for stage_module in self.stages:
             x = stage_module(x)

        # Then apply classification head
        x = self.avgpool_pre_head(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    # Default forward is set in __init__ based on fork_feat
    # def forward(self, x):
    #     if self.fork_feat:
    #         return self.forward_det(x)
    #     else:
    #         return self.forward_cls(x)

# --- Model Variant Functions --- (Added fork_feat back, default True)

def LWGANet_L0_1242_e32_k11_GELU(num_classes: int = 1000, pretrained: str = None, fork_feat: bool = True, **kwargs) -> LWGANet:
    """ LWGANet-L0 variant: stem=32, depths=(1,2,4,2), act=GELU """
    model = LWGANet(in_chans=3, num_classes=num_classes, stem_dim=32, depths=(1, 2, 4, 2),
                    att_kernel=(11, 11, 11, 11), norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                    drop_path_rate=0.0, pretrained=pretrained, fork_feat=fork_feat, **kwargs)
    return model

def LWGANet_L1_1242_e64_k11_GELU(num_classes: int = 1000, pretrained: str = None, fork_feat: bool = True, **kwargs) -> LWGANet:
    """ LWGANet-L1 variant: stem=64, depths=(1,2,4,2), act=GELU """
    model = LWGANet(in_chans=3, num_classes=num_classes, stem_dim=64, depths=(1, 2, 4, 2),
                    att_kernel=(11, 11, 11, 11), norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                    drop_path_rate=0.1, pretrained=pretrained, fork_feat=fork_feat, **kwargs)
    return model

def LWGANet_L2_1442_e96_k11_ReLU(num_classes: int = 1000, pretrained: str = None, fork_feat: bool = True, **kwargs) -> LWGANet:
    """ LWGANet-L2 variant: stem=96, depths=(1,4,4,2), act=ReLU """
    model = LWGANet(in_chans=3, num_classes=num_classes, stem_dim=96, depths=(1, 4, 4, 2),
                    att_kernel=(11, 11, 11, 11), norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU,
                    drop_path_rate=0.1, pretrained=pretrained, fork_feat=fork_feat, **kwargs)
    return model


# Example Usage:
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Test Feature Extraction (Default, for YOLO compatibility) ---
    print("\nTesting LWGANet L1 (Feature Extraction Mode, fork_feat=True)...")
    model_det = LWGANet_L1_1242_e64_k11_GELU(fork_feat=True).to(device)
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


    # --- Test Classification Forward (Specify fork_feat=False) ---
    print("\nTesting LWGANet L0 (Classification Mode, fork_feat=False)...")
    model_cls = LWGANet_L0_1242_e32_k11_GELU(num_classes=100, fork_feat=False).to(device)
    model_cls.eval()
    try:
        input_tensor_cls = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output_cls = model_cls(input_tensor_cls) # Calls forward_cls
        print("Classification model created successfully.")
        print("Input shape:", input_tensor_cls.shape)
        print("Output shape (logits):", output_cls.shape)
        # width_list is still calculated during init, but forward() gives classification output
        if hasattr(model_cls, 'width_list'):
            print("width_list (calculated during init):", model_cls.width_list)
    except Exception as e:
        print(f"Error during classification test: {e}")
        import traceback
        traceback.print_exc()


    # --- Test Pretrained Weight Loading (Example - using det model) ---
    print("\nTesting pretrained weight loading (Example)...")
    dummy_state_dict_path = "dummy_lwganet_det_weights.pth"
    torch.save({'model': model_det.state_dict()}, dummy_state_dict_path) # Save state dict of det model

    try:
        model_pretrained = LWGANet_L1_1242_e64_k11_GELU(fork_feat=True, pretrained=dummy_state_dict_path).to(device)
        print("Model created with dummy pretrained weights.")
        model_pretrained.eval()
        with torch.no_grad():
             output_pretrained = model_pretrained(input_tensor_det) # Test feature extraction forward
        print("Forward pass successful after loading weights.")
        print(f"Output has {len(output_pretrained)} feature maps.")

    except Exception as e:
        print(f"Error during pretrained weight loading test: {e}")
    finally:
        if os.path.exists(dummy_state_dict_path):
            os.remove(dummy_state_dict_path)