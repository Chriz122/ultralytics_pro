import math
from typing import Optional, Union, Sequence, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

# --- Necessary imports from timm (as used in Code 2) ---
from timm.models.layers import DropPath, trunc_normal_

# --- Helper Functions (Inspired by mmcv/mmengine, implemented in PyTorch) ---

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None, min_ratio: float = 0.9) -> int:
    """
    Ensures that all layers have a channel number that is divisible by the divisor.
    This function is taken from the original tf repo.
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < min_ratio * v:
        new_v += divisor
    return new_v

def _autopad(k: Union[int, Sequence[int]], p: Optional[Union[int, Sequence[int]]] = None, d: int = 1) -> Union[int, List[int]]:
    """
    Pad to 'same' shape outputs.
    Args:
        k (int or Sequence[int]): Kernel size.
        p (int or Sequence[int], optional): Padding. If None, autopad will be applied. Defaults to None.
        d (int): Dilation. Defaults to 1.
    Returns:
        Union[int, List[int]]: Padding.
    """
    if d > 1:
        # Actual kernel-size with dilation
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # Auto-pad
    return p

class _Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims).contiguous()

def BCHW2BHWC() -> nn.Module:
    return _Permute([0, 2, 3, 1])

def BHWC2BCHW() -> nn.Module:
    return _Permute([0, 3, 1, 2])

# --- Custom ConvNormAct similar to mmcv.ConvModule ---
class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        norm_cfg: Optional[Dict[str, Any]] = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: Optional[Dict[str, Any]] = dict(type='SiLU'),
        bias: Optional[bool] = None,
        padding_mode: str = 'zeros',
        inplace: bool = True,
    ):
        super().__init__()

        if padding is None:
            padding = _autopad(kernel_size, None, dilation)
        
        if bias is None: # Default behavior for Conv2d w.r.t BN
            bias = norm_cfg is None # if no norm, bias is True, else False

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode
        )

        self.norm_name = None
        if norm_cfg is not None:
            norm_type = norm_cfg.get('type')
            if norm_type == 'BN':
                self.norm = nn.BatchNorm2d(out_channels, eps=norm_cfg.get('eps', 0.001), momentum=norm_cfg.get('momentum', 0.03))
                self.norm_name = 'bn'
            elif norm_type == 'LN':
                # LayerNorm in ConvFFN is applied after permute, so it's channel-last
                # Here, if used, it would be a spatial LayerNorm if not permuted.
                # For simplicity, assuming LayerNorm after conv means normalizing over C,H,W or specific dims.
                # PyTorch LayerNorm expects normalized_shape. For (N,C,H,W) and LN over C:
                # self.norm = nn.GroupNorm(1, out_channels) # This is equivalent to LayerNorm over C,H,W
                # Or, if it's supposed to be like nn.LayerNorm([C, H, W]):
                # self.norm = nn.LayerNorm([out_channels, H, W]) # H,W need to be known or dynamic
                # Given PKINet context, 'BN' is the primary norm for Conv.
                # LayerNorm is used specifically in ConvFFN after permute.
                # For now, let's assume LN here would be a full spatial norm or raise error.
                # self.norm = nn.LayerNorm(out_channels) # This would require BHWC format or specific handling
                raise NotImplementedError("LayerNorm directly in ConvNormAct for BCHW needs careful dim specification.")

            else:
                self.norm = nn.Identity()
        else:
            self.norm = nn.Identity()

        self.act_name = None
        if act_cfg is not None:
            act_type = act_cfg.get('type')
            if act_type == 'SiLU':
                self.act = nn.SiLU(inplace=inplace)
                self.act_name = 'silu'
            elif act_type == 'ReLU':
                self.act = nn.ReLU(inplace=inplace)
                self.act_name = 'relu'
            elif act_type == 'GELU':
                self.act = nn.GELU() # GELU does not have inplace standardly
                self.act_name = 'gelu'
            elif act_type == 'Sigmoid':
                self.act = nn.Sigmoid()
                self.act_name = 'sigmoid'
            elif act_type is None: # Explicitly no activation
                 self.act = nn.Identity()
            else:
                raise NotImplementedError(f"Activation type {act_type} not implemented in ConvNormAct")
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

# --- PKINet Modules Refactored ---

class GSiLU(nn.Module):
    """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""
    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.adpool(x))


class CAA(nn.Module):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
    ):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3) # padding = (kernel_size - 1) // 2
        self.conv1 = ConvNormAct(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvNormAct(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None) # Original has None for norm/act
        self.v_conv = ConvNormAct(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None) # Original has None for norm/act
        self.conv2 = ConvNormAct(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled_x = self.avg_pool(x)
        x1 = self.conv1(pooled_x)
        xh = self.h_conv(x1)
        xv = self.v_conv(xh)
        x2 = self.conv2(xv)
        attn_factor = self.act(x2)
        return attn_factor


class ConvFFN(nn.Module):
    """Multi-layer perceptron implemented with ConvModule"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            hidden_channels_scale: float = 4.0,
            hidden_kernel_size: int = 3,
            dropout_rate: float = 0.,
            add_identity: bool = True,
            # norm_cfg for internal ConvNormAct, PKIBlock passes None
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            # act_cfg for internal ConvNormAct, PKIBlock passes None
            act_cfg: Optional[dict] = dict(type='SiLU'),
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = _make_divisible(int(in_channels * hidden_channels_scale), 8)

        # If PKIBlock passes norm_cfg=None, these ConvNormAct will not have BN
        # If PKIBlock passes act_cfg=None, these ConvNormAct will not have SiLU (except first one, it'll use default SiLU)
        
        # LayerNorm is applied on channels (last dim after permute)
        self.ln = nn.LayerNorm(in_channels) if norm_cfg is None or norm_cfg.get('type') != 'BN_before_permute' else nn.Identity()


        self.conv1 = ConvNormAct(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0,
                                 norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dw_conv = ConvNormAct(hidden_channels, hidden_channels, kernel_size=hidden_kernel_size, stride=1,
                                   padding=hidden_kernel_size // 2, groups=hidden_channels,
                                   norm_cfg=norm_cfg, act_cfg=None) # No activation before GSiLU
        self.gsilu = GSiLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = ConvNormAct(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                 norm_cfg=norm_cfg, act_cfg=act_cfg) # Uses act_cfg from args
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.add_identity = add_identity
        self.bchw2bhwc = BCHW2BHWC()
        self.bhwc2bchw = BHWC2BCHW()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Original: BCHW2BHWC(), nn.LayerNorm(in_channels), BHWC2BCHW()
        # This implies LayerNorm on channels, in BHWC format.
        x_processed = self.bchw2bhwc(x)
        x_processed = self.ln(x_processed)
        x_processed = self.bhwc2bchw(x_processed)
        
        x_processed = self.conv1(x_processed)
        x_processed = self.dw_conv(x_processed)
        x_processed = self.gsilu(x_processed)
        x_processed = self.dropout1(x_processed)
        x_processed = self.conv2(x_processed)
        x_processed = self.dropout2(x_processed)

        return identity + x_processed if self.add_identity else x_processed


class Stem(nn.Module):
    """Stem layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion: float = 1.0,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
    ):
        super().__init__()
        self.out_channels = out_channels # Store for width_list
        hidden_channels = _make_divisible(int(out_channels * expansion), 8)

        self.down_conv = ConvNormAct(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvNormAct(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1,
                                 norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvNormAct(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                 norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownSamplingLayer(nn.Module):
    """Down sampling layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
    ):
        super().__init__()
        out_channels = out_channels or (in_channels * 2)
        self.out_channels = out_channels # Store for width_list (though PKIStage will use its own out_channels)

        self.down_conv = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_conv(x)


class InceptionBottleneck(nn.Module):
    """Bottleneck with Inception module"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 1.0, # Expansion for hidden_channels within this block
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
    ):
        super().__init__()
        actual_out_channels = out_channels or in_channels # This is the final output channels of the block
        hidden_channels = _make_divisible(int(actual_out_channels * expansion), 8) # internal processing channels

        self.pre_conv = ConvNormAct(in_channels, hidden_channels, 1, 1, 0,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Depthwise convolutions
        dw_convs = []
        for i in range(len(kernel_sizes)):
            # For depthwise, padding should be calculated to maintain spatial dimensions
            pad = _autopad(kernel_sizes[i], d=dilations[i])
            dw_convs.append(
                ConvNormAct(hidden_channels, hidden_channels, kernel_sizes[i], 1,
                            padding=pad, dilation=dilations[i],
                            groups=hidden_channels, norm_cfg=None, act_cfg=None) # Original mmcv has None
            )
        self.dw_convs = nn.ModuleList(dw_convs)
        
        self.pw_conv = ConvNormAct(hidden_channels, hidden_channels, 1, 1, 0,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.caa_factor_module = None
        if with_caa:
            # PKIBlock passes norm_cfg=None, act_cfg=None to CAA when creating it
            self.caa_factor_module = CAA(hidden_channels, caa_kernel_size, caa_kernel_size, norm_cfg=None, act_cfg=None)

        self.add_identity = add_identity and in_channels == actual_out_channels # Original logic for skip connection

        self.post_conv = ConvNormAct(hidden_channels, actual_out_channels, 1, 1, 0,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x # Save original input for potential skip connection
        x_pre = self.pre_conv(x)

        # Apply DW convs and sum their outputs
        # x_dw_sum = self.dw_convs[0](x_pre) # Initialize with the first DW conv output
        # for i in range(1, len(self.dw_convs)):
        #     x_dw_sum = x_dw_sum + self.dw_convs[i](x_pre) # Add subsequent DW conv outputs
        # The original code has:
        # x = self.dw_conv(x)
        # x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)
        # This is a sequential sum, not parallel from x_pre. Let's replicate that.
        
        x_dw_processed = x_pre.clone() # Use clone if inplace operations happen in dw_convs
                                      # Assuming dw_convs don't operate inplace on their input directly.
        
        # Replicating: y = x_pre; x_dw = dw_conv0(x_pre); x_dw = x_dw + dw_conv1(x_dw) ...
        # This interpretation is likely wrong. More plausible: sum of parallel branches.
        # x_sum_branches = self.dw_convs[0](x_pre)
        # for i in range(1, len(self.dw_convs)):
        #     x_sum_branches = x_sum_branches + self.dw_convs[i](x_pre)
        
        # Looking at "Poly Kernel Inception Network.png" from a source if it exists, or common inception patterns.
        # The code implies:
        # x_branch = x_pre
        # x_out = self.dw_conv(x_branch)
        # x_out = x_out + self.dw_conv1(x_out) # This is unusual. Let's assume parallel branches from x_pre
        # x_out = x_out + self.dw_conv2(x_out) 
        # x_out = x_out + self.dw_conv3(x_out) 
        # x_out = x_out + self.dw_conv4(x_out) 
        # This implies output of previous DW conv is input to next sum.

        # Let's re-read: "x = self.dw_conv(x); x = x + self.dw_conv1(x) + ..."
        # This means x is modified in place.
        # Let's assume x_pre is the input to all dw_conv branches, and their results are summed.
        # y = x_pre (used for caa_factor and potentially identity add)
        
        # dw_outputs = [dw_conv(x_pre) for dw_conv in self.dw_convs]
        # x_processed = sum(dw_outputs) # This is standard Inception
        
        # Let's follow the original code's sequential update literally:
        # x = self.dw_conv(x) # x is now x_pre
        # x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)
        # This would mean self.dw_conv1 takes the output of self.dw_conv(x_pre) as input.
        # This is an odd pattern. More likely, it means all dw_convs operate on x_pre (or a copy)
        # and the results are combined.
        # x_after_dw0 = self.dw_convs[0](x_pre)
        # x_combined = x_after_dw0 + self.dw_convs[1](x_after_dw0) + ... -> still seems off.

        # Most logical interpretation of the original:
        # x_temp = self.dw_convs[0](x_pre)
        # x_temp = x_temp + self.dw_convs[1](x_temp.clone()) # If dw_convs are stateful or modify input for next in chain
        # x_temp = x_temp + self.dw_convs[2](x_temp.clone())
        # ...
        # This is very unconventional.
        # A more standard Inception-like structure is parallel branches from x_pre, then sum:
        # dw_outs = [dw(x_pre) for dw in self.dw_convs]
        # x_processed_dw = sum(dw_outs)
        # Let's go with this common pattern for Inception sum unless original paper shows otherwise.
        # The original code literally is:
        #   x = self.dw_conv(x)
        #   x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)
        # This implies `self.dw_conv1` takes `self.dw_conv(x_pre)` as input.
        
        # Given `y = x` and then `x = self.dw_conv(x)` etc.
        # `y` is `x_pre`. `x` becomes `self.dw_convs[0](x_pre)`.
        # Then `x_new = x + self.dw_convs[1](x) + self.dw_convs[2](x) ...`
        # Let's map variables: current_x = x_pre
        y_for_caa_and_skip = x_pre # Original uses `y=x` *before* dw_convs
        
        current_x = self.dw_convs[0](x_pre)
        if len(self.dw_convs) > 1: # If there are more dw_convs
            for i in range(1, len(self.dw_convs)):
                current_x = current_x + self.dw_convs[i](current_x) # This is what the code implies literally
        
        x_processed = self.pw_conv(current_x)

        caa_mult_factor = 1.0
        if self.caa_factor_module is not None:
            caa_mult_factor = self.caa_factor_module(y_for_caa_and_skip) # CAA operates on x_pre (y)

        # Original logic:
        # if self.add_identity:
        #     y = x * y  # y was x_pre, x is x_processed. So y_for_caa_and_skip * caa_mult_factor if caa
        #                  # y becomes x_processed * caa_mult_factor (if no caa) or * (caa_output) (if caa)
        #                  # This is complex. Let's re-read the original carefully:
        #     # if self.caa_factor is not None:
        #     #     y = self.caa_factor(y) # y is x_pre. So y becomes caa_out(x_pre)
        #     # if self.add_identity:
        #     #     y = x * y           # x is pw_conv output. y is caa_out(x_pre) or just x_pre
        #     #     x = x + y
        #     # else:
        #     #     x = x * y
        # This implies y is modified by CAA, then used as a multiplier.

        if self.caa_factor_module is not None:
            y_multiplier = caa_mult_factor # This is already self.caa_factor(y_for_caa_and_skip)
        else:
            y_multiplier = y_for_caa_and_skip # If no CAA, use x_pre as multiplier (this seems odd, usually identity is added)

        # Re-interpreting based on typical attention/gating:
        # x_gated = x_processed * caa_mult_factor (if with_caa)
        # else: x_gated = x_processed (if no_caa, but original has x = x * y which would be x_processed * x_pre)

        # Let's follow the original code's logic as closely as possible:
        # y_val = y_for_caa_and_skip # This is x_pre
        # x_val = x_processed # This is output of pw_conv

        # if self.caa_factor_module is not None:
        #     # y_val was x_pre, now it becomes caa_factor applied to x_pre
        #     y_val = self.caa_factor_module(y_val)

        # if self.add_identity:
        #     # y_val is either caa_factor(x_pre) or just x_pre
        #     # x_val is pw_conv output
        #     mult_term = x_val * y_val
        #     final_x = x_val + mult_term # This seems like the skip: x + x * caa(identity) or x + x * identity
        # else: # Not adding original identity, just gating
        #     final_x = x_val * y_val

        # Simpler approach from many models: Attention * Features + Features
        # Or: Features * Attention + Skip_from_Input
        # The original seems to be:  X_out = PW_out * (CAA(X_in) or X_in)  (if not add_identity)
        #                        or: X_out = PW_out + PW_out * (CAA(X_in) or X_in) (if add_identity)

        # Let's use the direct variable mapping:
        # `x` in original is `x_processed` (output of `self.pw_conv`)
        # `y` in original is `y_for_caa_and_skip` (which is `x_pre`)
        
        # Original:
        # y_orig = y_for_caa_and_skip # x_pre
        # x_orig = x_processed      # after pw_conv
        # if self.caa_factor_module is not None:
        #     y_orig = self.caa_factor_module(y_orig) # y_orig is now caa(x_pre)
        # if self.add_identity: # add_identity is tied to in_channels == out_channels
        #     # This means skip connection is possible because dimensions match
        #     # Here y_orig is either caa(x_pre) or just x_pre
        #     # x_orig is pw_conv output
        #     gated_x_orig = x_orig * y_orig
        #     x_final_before_post_conv = x_orig + gated_x_orig # This means x_processed + x_processed * (caa(x_pre) or x_pre)
        # else:
        #     x_final_before_post_conv = x_orig * y_orig

        # This implies that if add_identity is true, the effective skip is x_orig * y_orig, added to x_orig.
        # If in_channels != out_channels, add_identity is False.
        # Then it's just x_orig * y_orig. This is a modulation.

        # Let's simplify to a more common pattern if the original is too convoluted or potentially suboptimal.
        # Common: output = attention_module(features) * features + skip_connection(original_input)
        # Here, `self.add_identity` refers to `in_channels == out_channels` for the *final* skip.
        # The internal logic `x = x + y` where `y = x * y` is complex.

        # Assume:
        # 1. Calculate CAA factor from `x_pre` (i.e., `y_for_caa_and_skip`).
        # 2. Modulate `x_processed` (output of `pw_conv`) with this CAA factor.
        # 3. If `self.add_identity` (meaning `in_channels == out_channels` for the whole InceptionBottleneck),
        #    add the original `identity` (input `x` to the block) to the modulated `x_processed`.
        #    Else, just use the modulated `x_processed`. This is a common residual attention.

        current_features = x_processed # Output of self.pw_conv
        
        if self.caa_factor_module:
            # caa_mult_factor = self.caa_factor_module(y_for_caa_and_skip) # Already calculated
            gated_features = current_features * caa_mult_factor
        else:
            gated_features = current_features # No CAA, no gating from it.
        
        # The original code: `if self.add_identity: y = x * y; x = x + y else: x = x * y`
        # This `add_identity` is `self.add_identity` for `InceptionBottleneck`.
        # `x` is `x_processed`, `y` is `y_for_caa_and_skip` (or `caa_factor(y_for_caa_and_skip)`)
        # Let's re-implement literally based on their variable names
        # `x_after_pw_conv = x_processed`
        # `y_modifier = y_for_caa_and_skip`
        # if self.caa_factor_module:
        #      y_modifier = self.caa_factor_module(y_for_caa_and_skip)

        # if self.add_identity: # This is `self.add_identity` which is `in_C == out_C`
        #      # This is the part: y = x * y ; x = x + y
        #      # means: y_modified_further = x_after_pw_conv * y_modifier
        #      #        x_after_pw_conv = x_after_pw_conv + y_modified_further
        #      # So effectively: x_after_pw_conv * (1 + y_modifier)
        #      x_interim = x_after_pw_conv * (1 + y_modifier)
        # else:
        #      # This is the part: x = x * y
        #      x_interim = x_after_pw_conv * y_modifier
        # This looks like a plausible interpretation of the original code's intent.
        
        # Let's use y_modifier = caa_mult_factor if with_caa, else 1.0 for neutral element for multiplication
        # This is different from original where y_modifier would be y_for_caa_and_skip if no caa
        # Let's stick to original: y_modifier is caa_output or y_for_caa_and_skip
        
        y_mult = self.caa_factor_module(y_for_caa_and_skip) if self.caa_factor_module else y_for_caa_and_skip

        if self.add_identity: # This is the InceptionBottleneck's add_identity (in_ch == out_ch)
            # Corresponds to: y_temp = x_processed * y_mult; x_processed = x_processed + y_temp
            x_result = x_processed * (1 + y_mult)
        else:
            # Corresponds to: x_processed = x_processed * y_mult
            x_result = x_processed * y_mult
            
        output = self.post_conv(x_result)

        # If original add_identity for the entire block (x_input + block(x_input))
        # The parameter `add_identity` to __init__ is `True` if `in_channels == out_channels` for the block.
        # The internal `x = x + y` is always there if `add_identity` is true.
        # It seems the output of post_conv might then be added to the original input `x` if `self.add_identity`
        # is true for the *whole block* not just the internal operation.
        # However, `PKIBlock` manages its own residual connections *outside* this InceptionBottleneck.
        # The `add_identity=True` passed to `InceptionBottleneck` from `PKIBlock` means the internal logic.
        # Let's assume `output` is the final output of this module as calculated.

        return output


class PKIBlock(nn.Module):
    """Poly Kernel Inception Block"""
    def __init__(
            self,
            in_channels: int, # This will be hidden_channels from PKIStage
            out_channels: Optional[int] = None, # This will be hidden_channels from PKIStage
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            expansion: float = 1.0, # Expansion for InceptionBottleneck's hidden, and for this block's hidden
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale: Optional[float] = 1e-6, # Common value for LayerScale
            add_identity: bool = True, # For the residual connections in this block
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
    ):
        super().__init__()
        # In PKIStage, PKIBlock is called with (hidden_channels, hidden_channels, ...)
        # So, `in_channels` here is PKIStage's hidden_channels.
        # `out_channels` here, if None, becomes `in_channels`.
        actual_out_channels = out_channels or in_channels 
        
        # This hidden_channels is for the output of InceptionBottleneck if expansion was > 1.
        # But InceptionBottleneck's expansion is set to 1.0 from PKIStage's call.
        # PKIBlock's `expansion` argument (default 1.0) defines hidden dimension for InceptionBottleneck.
        # So, InceptionBottleneck input: `in_channels` (from PKIBlock arg)
        #      InceptionBottleneck output: `_make_divisible(int(actual_out_channels * expansion), 8)`
        # If expansion=1.0, then InceptionBottleneck output is `actual_out_channels`.
        # If `in_channels` and `actual_out_channels` are the same (e.g. `hidden_ch` from stage), it's fine.
        
        block_out_channels = _make_divisible(int(actual_out_channels * expansion), 8)

        self.norm1 = nn.BatchNorm2d(in_channels, eps=norm_cfg.get('eps', 0.001), momentum=norm_cfg.get('momentum', 0.03)) if norm_cfg and norm_cfg.get('type') == 'BN' else nn.Identity()
        
        self.block = InceptionBottleneck(
            in_channels, block_out_channels, kernel_sizes, dilations,
            expansion=1.0, # Internal expansion within InceptionBottleneck, set to 1.0 as per original
            add_identity=True, # This is for the internal connection style of InceptionBottleneck
            with_caa=with_caa, caa_kernel_size=caa_kernel_size,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        
        self.norm2 = nn.BatchNorm2d(block_out_channels, eps=norm_cfg.get('eps', 0.001), momentum=norm_cfg.get('momentum', 0.03)) if norm_cfg and norm_cfg.get('type') == 'BN' else nn.Identity()
        
        # ConvFFN input is block_out_channels, output is actual_out_channels
        self.ffn = ConvFFN(
            block_out_channels, actual_out_channels, ffn_scale, ffn_kernel_size, dropout_rate, 
            add_identity=False, # FFN's internal residual. PKIBlock handles its own.
            norm_cfg=None, act_cfg=None # As per original, PKIBlock passes None
        )
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layer_scale_1 = layer_scale is not None
        self.layer_scale_2 = layer_scale is not None
        
        if self.layer_scale_1:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(block_out_channels), requires_grad=True)
        if self.layer_scale_2:
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(actual_out_channels), requires_grad=True)
            
        self.add_identity = add_identity # For the two main residual branches of PKIBlock

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # First branch (InceptionBottleneck)
        x_block = self.norm1(x)
        x_block = self.block(x_block)
        if self.layer_scale_1:
            x_block = self.gamma1.view(1, -1, 1, 1) * x_block
        x_block = self.drop_path(x_block)
        
        # Second branch (ConvFFN)
        # The original code implies FFN processes the output of the first branch (x_block + x)
        # x = x + self.drop_path(self.gamma1 * self.block(self.norm1(x)))
        # x = x + self.drop_path(self.gamma2 * self.ffn(self.norm2(x)))
        # This means FFN takes the *updated x* as input for its norm2.
        # This is a sequential structure like ConvNeXt.

        if self.add_identity:
            x = identity + x_block
        else: # This case was not explicitly in original PKIBlock forward logic, but for completeness
            x = x_block
        
        identity_after_block = x # Save for FFN residual

        x_ffn = self.norm2(x) # Norm is applied to the output of the first residual sum
        x_ffn = self.ffn(x_ffn)
        if self.layer_scale_2:
            x_ffn = self.gamma2.view(1, -1, 1, 1) * x_ffn
        x_ffn = self.drop_path(x_ffn)

        if self.add_identity: # If FFN also has a residual connection
            x = identity_after_block + x_ffn
        else:
            x = x_ffn # This means output of FFN is the final output (if add_identity is false for the FFN part)
                      # But original code always adds: x = x + drop_path(...)
        return x


class PKIStage(nn.Module):
    """Poly Kernel Inception Stage"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 0.5, # Expansion for hidden_channels in this stage
            ffn_scale: float = 4.0, # For PKIBlock's FFN
            ffn_kernel_size: int = 3, # For PKIBlock's FFN
            dropout_rate: float = 0.,
            drop_path_rate: Union[float, list] = 0.,
            layer_scale: Optional[float] = 1e-6,
            shortcut_with_ffn: bool = True,
            shortcut_ffn_scale: float = 4.0,
            shortcut_ffn_kernel_size: int = 5,
            add_identity: bool = True, # For PKIBlock's residual
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
    ):
        super().__init__()
        self.out_channels = out_channels # Store for width_list

        # Hidden channels for this stage, affects channels for PKIBlocks
        hidden_channels_stage = _make_divisible(int(out_channels * expansion), 8)

        self.downsample = DownSamplingLayer(in_channels, out_channels, norm_cfg, act_cfg)

        # Split into two branches x and y
        self.conv1_pre_split = ConvNormAct(out_channels, 2 * hidden_channels_stage, kernel_size=1,
                                           norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.ffn_shortcut = None
        if shortcut_with_ffn:
            self.ffn_shortcut = ConvFFN(hidden_channels_stage, hidden_channels_stage, shortcut_ffn_scale,
                                        shortcut_ffn_kernel_size, 0., add_identity=True,
                                        norm_cfg=None, act_cfg=None) # Original passed None

        self.blocks = nn.ModuleList()
        if not isinstance(drop_path_rate, list):
            dpr_list = [drop_path_rate] * num_blocks
        else:
            dpr_list = drop_path_rate
            assert len(dpr_list) == num_blocks

        for i in range(num_blocks):
            # PKIBlocks operate on hidden_channels_stage
            block = PKIBlock(
                hidden_channels_stage, hidden_channels_stage, kernel_sizes, dilations, with_caa,
                caa_kernel_size + 2 * i, # Increasing caa_kernel_size per block
                1.0, # PKIBlock's own expansion, original is 1.0
                ffn_scale, ffn_kernel_size, dropout_rate,
                dpr_list[i],
                layer_scale, add_identity, norm_cfg, act_cfg
            )
            self.blocks.append(block)

        self.conv2_post_merge = ConvNormAct(2 * hidden_channels_stage, out_channels, kernel_size=1,
                                            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3_final = ConvNormAct(out_channels, out_channels, kernel_size=1,
                                       norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_down = self.downsample(x)
        
        x_split = self.conv1_pre_split(x_down)
        branch_x, branch_y = torch.chunk(x_split, 2, dim=1)

        if self.ffn_shortcut is not None:
            branch_x = self.ffn_shortcut(branch_x)

        # Process branch_y through blocks
        # Original: t = torch.zeros_like(branch_y); for block in self.blocks: t = t + block(branch_y)
        # This means branch_y is input to *every* block, and results are summed.
        # This is different from sequential application.
        
        processed_branch_y_parts = []
        for block in self.blocks:
            processed_branch_y_parts.append(block(branch_y))
        
        if processed_branch_y_parts: # Ensure there's at least one block
             branch_y_aggregated = sum(processed_branch_y_parts)
        else: # Handle num_blocks = 0 case, though typically num_blocks > 0
             branch_y_aggregated = torch.zeros_like(branch_y)


        # Original: z = [branch_x]; t = ...; z.append(t); z = torch.cat(z, dim=1)
        merged_features = torch.cat([branch_x, branch_y_aggregated], dim=1)
        
        output = self.conv2_post_merge(merged_features)
        output = self.conv3_final(output)
        
        return output


class PKINet(nn.Module):
    """Poly Kernel Inception Network"""
    # arch_zoo defines parameters for each stage AFTER the stem
    # Each sub-list corresponds to a PKIStage:
    # [in_channels_to_stage(0), out_channels_from_stage(1), num_blocks(2), kernel_sizes(3), dilations(4), 
    #  expansion_in_stage(5), ffn_scale_in_block(6), ffn_kernel_size_in_block(7), 
    #  dropout_rate_in_block(8), layer_scale_in_block(9), shortcut_with_ffn_in_stage(10),
    #  shortcut_ffn_scale_in_stage(11), shortcut_ffn_kernel_size_in_stage(12), 
    #  add_identity_in_block(13), with_caa_in_block(14), caa_kernel_size_in_block_start(15)]
    arch_zoo = {
        'T': [[16, 32, 4, (3,5,7,9,11), (1,1,1,1,1), 0.5, 4.0, 3, 0.1, 1e-6, True, 8.0, 5, True, True, 11],
              [32, 64, 14, (3,5,7,9,11), (1,1,1,1,1), 0.5, 4.0, 3, 0.1, 1e-6, True, 8.0, 7, True, True, 11],
              [64, 128, 22, (3,5,7,9,11), (1,1,1,1,1), 0.5, 4.0, 3, 0.1, 1e-6, True, 4.0, 9, True, True, 11],
              [128, 256, 4, (3,5,7,9,11), (1,1,1,1,1), 0.5, 4.0, 3, 0.1, 1e-6, True, 4.0, 11, True, True, 11]],
        'S': [[32, 64, 4, (3,5,7,9,11), (1,1,1,1,1), 0.5, 4.0, 3, 0.1, 1e-6, True, 8.0, 5, True, True, 11],
              [64, 128, 12, (3,5,7,9,11), (1,1,1,1,1), 0.5, 4.0, 3, 0.1, 1e-6, True, 8.0, 7, True, True, 11],
              [128, 256, 20, (3,5,7,9,11), (1,1,1,1,1), 0.5, 4.0, 3, 0.1, 1e-6, True, 4.0, 9, True, True, 11],
              [256, 512, 4, (3,5,7,9,11), (1,1,1,1,1), 0.5, 4.0, 3, 0.1, 1e-6, True, 4.0, 11, True, True, 11]],
        'B': [[40, 80, 6, (3,5,7,9,11), (1,1,1,1,1), 0.5, 4.0, 3, 0.1, 1e-6, True, 8.0, 5, True, True, 11],
              [80, 160, 16, (3,5,7,9,11), (1,1,1,1,1), 0.5, 4.0, 3, 0.1, 1e-6, True, 8.0, 7, True, True, 11],
              [160, 320, 24, (3,5,7,9,11), (1,1,1,1,1), 0.5, 4.0, 3, 0.1, 1e-6, True, 4.0, 9, True, True, 11],
              [320, 640, 6, (3,5,7,9,11), (1,1,1,1,1), 0.5, 4.0, 3, 0.1, 1e-6, True, 4.0, 11, True, True, 11]],
    }

    def __init__(
            self,
            c1: int = 3, # Input channels
            arch: Union[str, Dict] = 'S',
            out_indices: Sequence[int] = (1, 2, 3, 4), # Indices of stages to output: 0 for stem, 1-N for PKIStages
            drop_path_rate: float = 0.1,
            frozen_stages: int = -1, # Stages to freeze (0=stem, 1=stem+stage1, etc.)
            norm_eval: bool = False, # Set BN to eval mode during training
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            # Kaiming init for Conv2d, specific for others
            default_init_method: str = 'kaiming_conv_trunc_normal_linear',
    ):
        super().__init__()
        
        if isinstance(arch, str):
            if arch not in self.arch_zoo:
                raise KeyError(f"Arch '{arch}' is not in default PKINet archs {set(self.arch_zoo.keys())}")
            arch_setting_stages = self.arch_zoo[arch]
        elif isinstance(arch, dict): # Assuming dict provides the list of stage_params
            arch_setting_stages = arch.get('stages_params', []) # Or however custom dict is structured
            if not arch_setting_stages:
                 raise ValueError("Custom arch dict must contain 'stages_params' list.")
        else:
            raise TypeError("arch must be a string or a dict with 'stages_params'.")

        self.num_feature_stages = len(arch_setting_stages) + 1 # +1 for stem
        
        if not out_indices: # If empty, output from last stage
            out_indices = (self.num_feature_stages - 1,) 
        assert all(i < self.num_feature_stages for i in out_indices), \
            f"out_indices {out_indices} out of range for {self.num_feature_stages} stages."
        if frozen_stages not in range(-1, self.num_feature_stages):
            raise ValueError(f'frozen_stages must be in range(-1, {self.num_feature_stages}). But received {frozen_stages}')

        self.out_indices = sorted(list(set(out_indices))) # Ensure unique and sorted
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.default_init_method = default_init_method
        self.stages_modulelist = nn.ModuleList() # Renamed to avoid conflict if self.stages is used elsewhere

        # Stem
        # arch_setting_stages[0][0] is the output channels of the stem
        stem_out_channels = arch_setting_stages[0][0]
        self.stem = Stem(c1, stem_out_channels, expansion=1.0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stages_modulelist.append(self.stem)

        # PKIStages
        depths_per_pki_stage = [s_params[2] for s_params in arch_setting_stages]
        total_pki_blocks = sum(depths_per_pki_stage)
        # Distribute Dpr across PKIBlocks only
        dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, total_pki_blocks)] if total_pki_blocks > 0 else []
        
        current_block_idx_for_dpr = 0
        current_in_channels = stem_out_channels

        for i, stage_params in enumerate(arch_setting_stages):
            # stage_params[0] is in_channels for this PKIStage, should match current_in_channels
            # stage_params[1] is out_channels for this PKIStage
            # Ensure consistency:
            if stage_params[0] != current_in_channels:
                 print(f"Warning: Stage {i} arch in_channels {stage_params[0]} != previous out_channels {current_in_channels}")
                 # Could override current_in_channels = stage_params[0] if arch is king.
            
            num_blocks_in_stage = stage_params[2]
            stage_dpr = dpr_list[current_block_idx_for_dpr : current_block_idx_for_dpr + num_blocks_in_stage]
            current_block_idx_for_dpr += num_blocks_in_stage

            pki_stage = PKIStage(
                in_channels=current_in_channels, # Use current_in_channels
                out_channels=stage_params[1],
                num_blocks=stage_params[2],
                kernel_sizes=stage_params[3],
                dilations=stage_params[4],
                expansion=stage_params[5],
                ffn_scale=stage_params[6],
                ffn_kernel_size=stage_params[7],
                dropout_rate=stage_params[8],
                drop_path_rate=stage_dpr, # Pass the slice for this stage
                layer_scale=stage_params[9],
                shortcut_with_ffn=stage_params[10],
                shortcut_ffn_scale=stage_params[11],
                shortcut_ffn_kernel_size=stage_params[12],
                add_identity=stage_params[13],
                with_caa=stage_params[14],
                caa_kernel_size=stage_params[15],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
            self.stages_modulelist.append(pki_stage)
            current_in_channels = stage_params[1] # Update for next stage

        # Populate width_list for Ultralytics compatibility
        self.width_list = []
        for i in self.out_indices:
            if hasattr(self.stages_modulelist[i], 'out_channels'):
                self.width_list.append(self.stages_modulelist[i].out_channels)
            else: # Fallback, should not happen if Stem and PKIStage have out_channels
                if i == 0: # Stem
                    self.width_list.append(arch_setting_stages[0][0]) 
                else: # PKIStage i-1
                    self.width_list.append(arch_setting_stages[i-1][1])
        
        self._initialize_weights()
        self._freeze_stages()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.default_init_method == 'kaiming_conv_trunc_normal_linear':
                    # Parameters from original init_cfg for PKINet
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
                else: # Fallback or other schemes
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if self.default_init_method == 'kaiming_conv_trunc_normal_linear':
                     trunc_normal_(m.weight, std=.02) # As per MMLab default for Linear
                else:
                     nn.init.normal_(m.weight, 0, 0.01) # A common default
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Specific parameter initialization (e.g., gamma in PKIBlock)
            # These are initialized directly in their respective modules' __init__

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.frozen_stages >= 0: # Freeze stem
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            
            # Freeze PKIStages up to frozen_stages -1 (since stage 0 is stem)
            for i in range(1, self.frozen_stages + 1): # Corrected loop range
                if i < len(self.stages_modulelist): # Ensure index is valid
                    m = self.stages_modulelist[i]
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm): # Includes nn.BatchNorm2d
                    m.eval()
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for i, stage_module in enumerate(self.stages_modulelist):
            x = stage_module(x)
            if i in self.out_indices:
                outs.append(x)
        return outs # Return list for Ultralytics compatibility (avoids .insert error on tuple)

# --- Factory Functions ---
def pkinet_t(c1: int = 3, out_indices: Sequence[int] = (1, 2, 3, 4), **kwargs) -> PKINet:
    return PKINet(c1=c1, arch='T', out_indices=out_indices, **kwargs)

def pkinet_s(c1: int = 3, out_indices: Sequence[int] = (1, 2, 3, 4), **kwargs) -> PKINet:
    return PKINet(c1=c1, arch='S', out_indices=out_indices, **kwargs)

def pkinet_b(c1: int = 3, out_indices: Sequence[int] = (1, 2, 3, 4), **kwargs) -> PKINet:
    return PKINet(c1=c1, arch='B', out_indices=out_indices, **kwargs)


if __name__ == '__main__':
    # Example Usage:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing PKINet-T...")
    model_t = pkinet_t(out_indices=(0, 1, 2, 3, 4)).to(device) # Output all 5 feature levels
    print(f"PKINet-T width_list: {model_t.width_list}")
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    features_t = model_t(dummy_input)
    print(f"PKINet-T output features ({len(features_t)} levels):")
    for i, f in enumerate(features_t):
        print(f"  Level {i}: {f.shape}")

    print("\nTesting PKINet-S with specific out_indices...")
    # Typical for detection: P2, P3, P4 (if P0 is stem, P1 is stage0_out, P2 is stage1_out etc.)
    # Original out_indices was (2,3,4) referring to stages_modulelist[2], [3], [4]
    # which are PKIStage1, PKIStage2, PKIStage3 outputs.
    # If Stem is stage 0, PKIStage0 is stage 1, PKIStage1 is stage 2, ...
    # So (2,3,4) means outputs of PKIStage1, PKIStage2, PKIStage3.
    model_s = pkinet_s(out_indices=(2, 3, 4)).to(device) 
    print(f"PKINet-S width_list: {model_s.width_list}")
    features_s = model_s(dummy_input)
    print(f"PKINet-S output features ({len(features_s)} levels):")
    for i, f in enumerate(features_s):
        print(f"  Level {i}: {f.shape}")

    print(f"\nPKINet-S total parameters: {sum(p.numel() for p in model_s.parameters() if p.requires_grad) / 1e6:.2f}M")

    print("\nFreezing stages example (Stem + first PKIStage):")
    model_s_frozen = pkinet_s(out_indices=(2,3,4), frozen_stages=1).to(device) # Freeze stem (idx 0) and PKIStage0 (idx 1)
    model_s_frozen.train() # Call train to ensure freezing logic and norm_eval is applied
    
    print("Stem parameters frozen:")
    for name, param in model_s_frozen.stem.named_parameters():
        print(f"  {name}: requires_grad={param.requires_grad}")
    print("First PKIStage (stages_modulelist[1]) parameters frozen:")
    for name, param in model_s_frozen.stages_modulelist[1].named_parameters():
        print(f"  {name}: requires_grad={param.requires_grad}")
    print("Second PKIStage (stages_modulelist[2]) parameters (should not be frozen):")
    # Check a few params from the next stage
    all_true=True
    for name, param in model_s_frozen.stages_modulelist[2].named_parameters():
        if not param.requires_grad:
            all_true=False
            break
    print(f"  All params require_grad: {all_true}")

    # Test with a smaller input for faster run
    dummy_input_small = torch.randn(2, 3, 64, 64).to(device)
    features_frozen = model_s_frozen(dummy_input_small)
    print(f"PKINet-S (frozen) output features ({len(features_frozen)} levels):")
    for i, f in enumerate(features_frozen):
        print(f"  Level {i}: {f.shape}")