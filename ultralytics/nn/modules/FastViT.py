# #
# # For licensing see accompanying LICENSE file.
# # Copyright (C) 2023 Apple Inc. All Rights Reserved.
# #
# Modifications copyright (C) 2024 <Your Name/Org>
# - Removed mmcv/mmdet/mmseg dependencies
# - Refactored model instantiation similar to MobileNetV4 example
# - Added width_list calculation
# #
import os
import copy
from functools import partial
from typing import List, Tuple, Optional, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming timm is available for layers like DropPath and trunc_normal_
try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    print("timm library not found. DropPath and trunc_normal_ might not work.")
    # Provide dummy implementations if timm is not available
    class DropPath(nn.Module):
        def __init__(self, drop_prob=None):
            super().__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            return x
    def trunc_normal_(tensor, mean=0., std=1.):
        # Simple normal initialization as a fallback
        nn.init.normal_(tensor, mean=mean, std=std)


# --- Helper Modules (MobileOneBlock, ReparamLargeKernelConv, SEBlock, etc.) ---
# These modules are kept largely the same as in the original FastViT code,
# as they define the core building blocks.

class ReparamLargeKernelConv(nn.Module):
    """Building Block of RepLKNet (modified for FastViT use)"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: Optional[int], # Made optional
        inference_mode: bool = False,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        super(ReparamLargeKernelConv, self).__init__()

        self.stride = stride
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation # Note: Activation might not be used directly in forward in original FastViT PatchEmbed
        self.inference_mode = inference_mode

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.padding = kernel_size // 2

        if inference_mode:
            self.lkb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = self._conv_bn(
                kernel_size=kernel_size, padding=self.padding
            )
            if small_kernel is not None and small_kernel > 0: # Check small_kernel > 0
                assert (
                    small_kernel <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = self._conv_bn(
                    kernel_size=small_kernel, padding=small_kernel // 2
                )
            else:
                self.small_conv = None # Ensure it's None if not used

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        # Activation is typically applied *after* this block in FastViT's structure
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if self.small_conv is not None:
                out += self.small_conv(x)
        # Original FastViT doesn't apply activation within ReparamLargeKernelConv forward
        # out = self.activation(out) # Remove this line if activation is external
        return out

    def get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias."""
        eq_k, eq_b = self._fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv") and self.small_conv is not None:
             small_k, small_b = self._fuse_bn(self.small_conv.conv, self.small_conv.bn)
             eq_b += small_b
             # Pad the small kernel correctly
             pad_val = (self.kernel_size - self.small_kernel) // 2
             eq_k += nn.functional.pad(small_k, [pad_val] * 4) # Pad H_in, H_out, W_in, W_out

        return eq_k, eq_b

    def reparameterize(self) -> None:
        """Reparameterize multi-branched architecture to single conv."""
        if self.inference_mode: return
        eq_k, eq_b = self.get_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.lkb_origin.conv.dilation, # Use dilation from original conv
            groups=self.groups,
            bias=True,
        )

        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")
        self.inference_mode = True # Set inference mode flag

    @staticmethod
    def _fuse_bn(
        conv: nn.Conv2d, bn: nn.BatchNorm2d
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse batchnorm layer with conv layer."""
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int = 0) -> nn.Sequential:
        """Helper method to construct conv-batchnorm layers."""
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class SEBlock(nn.Module):
    """Squeeze and Excite module."""
    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """MobileOne building block."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        activation: nn.Module = nn.GELU(), # Use GELU as default like original
    ) -> None:
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

        if use_act:
            # Use the provided activation module instance directly
            self.activation = activation
        else:
            self.activation = nn.Identity()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.rbr_skip = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            if num_conv_branches > 0:
                rbr_conv = list()
                for _ in range(self.num_conv_branches):
                    rbr_conv.append(
                        self._conv_bn(kernel_size=kernel_size, padding=padding)
                    )
                self.rbr_conv = nn.ModuleList(rbr_conv)
            else:
                self.rbr_conv = None # Ensure it's None if not used

            self.rbr_scale = None
            # In original FastViT, scale branch is used even for kernel_size=1 in some cases (like stem)
            # Let's stick to the original condition `(kernel_size > 1)` unless specified otherwise
            # Correction: Looking at `convolutional_stem`, it seems kernel_size=1 *does* use it. Let's allow it.
            # if (kernel_size > 1) and use_scale_branch:
            if use_scale_branch: # Use scale branch whenever requested
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        for para in self.parameters():
            para.detach_()
        if hasattr(self, "rbr_conv"): self.__delattr__("rbr_conv")
        if hasattr(self, "rbr_scale"): self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"): self.__delattr__("rbr_skip")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            pad = self.kernel_size // 2
            if pad > 0: # Only pad if kernel size > 1
                 kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(
        self, branch: Union[nn.Sequential, nn.BatchNorm2d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                 input_dim = self.in_channels // self.groups
                 # Ensure id_tensor has correct shape based on kernel_size
                 ks = self.kernel_size
                 kernel_value = torch.zeros(
                     (self.in_channels, input_dim, ks, ks), # Use self.kernel_size
                     dtype=branch.weight.dtype,
                     device=branch.weight.device,
                 )
                 for i in range(self.in_channels):
                     # Place the '1' at the center of the kernel
                     kernel_value[i, i % input_dim, ks // 2, ks // 2] = 1
                 self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


# --- Attention and FFN Modules ---
class MHSA(nn.Module):
    """Multi-headed Self Attention module."""
    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        B, C, H, W = shape
        N = H * W
        is_4d = len(shape) == 4
        if is_4d:
            x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if is_4d:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x

class ConvFFN(nn.Module):
    """Convolutional FFN Module."""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU, # Keep GELU default
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        # Use MobileOneBlock for the depthwise conv part? Original used standard Conv+BN.
        # Sticking to original Conv+BN for ConvFFN's depthwise part for now.
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels, # DW conv keeps channels same
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=in_channels), # Use in_channels here
        )
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights) # Keep local init

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
             nn.init.constant_(m.weight, 1)
             nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
             trunc_normal_(m.weight, std=0.02)
             if m.bias is not None:
                 nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x) # Apply DW conv block
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# --- Positional Encoding ---
class RepCPE(nn.Module):
    """Reparameterizable conditional positional encoding."""
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
        inference_mode=False,
    ) -> None:
        super(RepCPE, self).__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, Tuple) and len(spatial_shape) == 2

        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim # CPE uses groups == embed_dim
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.groups, # Groups = embed_dim
                bias=True,
            )
        else:
            self.pe = nn.Conv2d(
                in_channels,
                embed_dim,
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                bias=True,
                groups=self.groups, # Groups = embed_dim
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            return self.reparam_conv(x)
        else:
            return self.pe(x) + x # Additive positional encoding

    def reparameterize(self) -> None:
        if self.inference_mode: return

        # Build equivalent Id tensor
        input_dim = self.in_channels // self.groups # Should be 1 if in_channels==embed_dim
        kernel_value = torch.zeros(
            (
                self.in_channels,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ),
            dtype=self.pe.weight.dtype,
            device=self.pe.weight.device,
        )
        for i in range(self.in_channels):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        # Reparameterize Id tensor and conv
        w_final = id_tensor + self.pe.weight
        b_final = self.pe.bias

        # Introduce reparam conv
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        for para in self.parameters():
            para.detach_()
        self.__delattr__("pe")
        self.inference_mode = True # Set inference mode flag

# --- Core Blocks (RepMixer, AttentionBlock) ---

class RepMixer(nn.Module):
    """Reparameterizable token mixer."""
    def __init__(
        self,
        dim,
        kernel_size=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            # Norm branch: BatchNorm only (MobileOneBlock with 0 conv branches, no activation, no scale branch)
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
                use_scale_branch=False, # No scale branch for norm
                num_conv_branches=0, # No conv branch for norm
                inference_mode=inference_mode, # Pass inference mode
            )
            # Mixer branch: Full MobileOneBlock (includes scale branch if k>1)
            self.mixer = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False, # No activation within mixer itself
                use_scale_branch=True, # Default from original code
                num_conv_branches=1,   # Default from original code
                inference_mode=inference_mode, # Pass inference mode
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(
                    layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
                )
            else:
                 self.layer_scale = None # Explicitly set to None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            # Shortcut in inference mode
            return self.reparam_conv(x)
        else:
            # Training mode logic
            mx = self.mixer(x)
            nx = self.norm(x) # This is just the BatchNorm pass
            if self.use_layer_scale:
                 return x + self.layer_scale * (mx - nx)
            else:
                 return x + (mx - nx)


    def reparameterize(self) -> None:
        if self.inference_mode: return

        # Reparameterize mixer and norm branches first
        self.mixer.reparameterize()
        self.norm.reparameterize()

        # Get the weights and biases from the reparameterized conv layers
        # Note: mixer.reparam_conv and norm.reparam_conv should now exist.
        w_mixer = self.mixer.reparam_conv.weight
        b_mixer = self.mixer.reparam_conv.bias
        w_norm = self.norm.reparam_conv.weight # This is the BN reparameterized
        b_norm = self.norm.reparam_conv.bias

        # Fuse them based on the formula: x + ls * (mixer(x) - norm(x))
        # The reparameterized equivalent is a single convolution:
        # Conv(x) = (Id + ls * (W_mixer - W_norm)) * x + ls * (b_mixer - b_norm)

        # Build the identity kernel tensor (like in MobileOneBlock._fuse_bn_tensor)
        input_dim = self.dim // self.dim # groups = dim, so input_dim = 1
        ks = self.kernel_size
        id_tensor = torch.zeros(
            (self.dim, input_dim, ks, ks),
            dtype=w_mixer.dtype, device=w_mixer.device
        )
        for i in range(self.dim):
            id_tensor[i, i % input_dim, ks // 2, ks // 2] = 1

        if self.use_layer_scale:
            # Squeeze layer scale for broadcasting with bias
            ls_val = torch.squeeze(self.layer_scale)
            # Unsqueeze layer scale for broadcasting with weight
            ls_weight_val = self.layer_scale.unsqueeze(-1)

            w_final = id_tensor + ls_weight_val * (w_mixer - w_norm)
            b_final = ls_val * (b_mixer - b_norm) # Bias is only affected by the scaled difference
        else:
            # If no layer scale, the formula simplifies: x + mixer(x) - norm(x)
            # Conv(x) = (Id + W_mixer - W_norm) * x + (b_mixer - b_norm)
            w_final = id_tensor + w_mixer - w_norm
            b_final = b_mixer - b_norm

        # Create the final reparameterized convolution
        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        # Clean up
        for para in self.parameters():
            para.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")
        self.inference_mode = True # Set inference mode flag


class RepMixerBlock(nn.Module):
    """RepMixer block structure (TokenMixer -> ConvFFN)."""
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU, # Keep GELU default
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ):
        super().__init__()
        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )

        assert mlp_ratio > 0
        mlp_hidden_dim = int(dim * mlp_ratio)
        # Pass the activation layer *type* to ConvFFN
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer, # Pass type
            drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer scale for the FFN branch
        self.use_ffn_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_ffn = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )
        else:
            self.layer_scale_ffn = None # Explicitly None

    def forward(self, x):
        # Apply token mixer first
        x = self.token_mixer(x)
        # Apply FFN with skip connection and optional layer scale
        if self.use_ffn_layer_scale:
            x = x + self.drop_path(self.layer_scale_ffn * self.convffn(x))
        else:
            x = x + self.drop_path(self.convffn(x))
        return x

    def reparameterize(self):
        # Reparameterize the token mixer if it has the method
        if hasattr(self.token_mixer, 'reparameterize'):
            self.token_mixer.reparameterize()
        # ConvFFN does not have reparameterization in this setup


class AttentionBlock(nn.Module):
    """Attention block structure (Norm -> TokenMixer -> Skip -> FFN -> Skip)."""
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU, # Keep GELU default
        norm_layer: nn.Module = nn.BatchNorm2d,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        # Removed inference_mode, MHSA/ConvFFN don't use it directly here
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        # Assuming MHSA is used as the token mixer here
        self.token_mixer = MHSA(dim=dim, attn_drop=drop, proj_drop=drop) # Pass drop rates

        assert mlp_ratio > 0
        mlp_hidden_dim = int(dim * mlp_ratio)
        # Pass activation layer *type* to ConvFFN
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer, # Pass type
            drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            # Two layer scales: one after attention, one after FFN
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )
        else:
            self.layer_scale_1 = None
            self.layer_scale_2 = None


    def forward(self, x):
        # Pre-normalization structure
        if self.use_layer_scale:
            # Apply norm, token mixer, layer scale, drop path, skip connection
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
            # Apply FFN, layer scale, drop path, skip connection
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x

    # AttentionBlock does not have reparameterization itself

# --- Stem and Patch Embedding ---

def convolutional_stem(
    in_channels: int, out_channels: int, inference_mode: bool = False
) -> nn.Sequential:
    """Build convolutional stem with MobileOne blocks."""
    # Activation is typically part of MobileOneBlock
    act_layer = nn.GELU() # Define activation once
    return nn.Sequential(
        MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1, # First conv is standard
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
            activation=copy.deepcopy(act_layer) # Pass instance
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels, # Depthwise conv
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
            activation=copy.deepcopy(act_layer) # Pass instance
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1, # Pointwise conv
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
            activation=copy.deepcopy(act_layer) # Pass instance
        ),
    )


class PatchEmbed(nn.Module):
    """Convolutional patch embedding layer using RepLK + MobileOne."""
    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        inference_mode: bool = False,
    ) -> None:
        super().__init__()
        self.inference_mode = inference_mode
        # Use ReparamLargeKernelConv for the main downsampling conv
        # Activation is usually applied *after* the block
        self.rkl_conv = ReparamLargeKernelConv(
            in_channels=in_channels,
            out_channels=embed_dim, # Output embed_dim directly
            kernel_size=patch_size,
            stride=stride,
            groups=in_channels, # Grouped conv for efficiency? Original used groups=in_channels
            small_kernel=3,      # Use a small kernel branch
            inference_mode=inference_mode,
            # activation=nn.Identity() # Activation applied later
        )
        # Followed by a MobileOne block (1x1 conv)
        self.m1_block = MobileOneBlock(
            in_channels=embed_dim, # Input is embed_dim
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
            activation=nn.GELU() # Apply activation here
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rkl_conv(x)
        x = self.m1_block(x)
        return x

    def reparameterize(self):
        # Reparameterize internal blocks if they have the method
        if hasattr(self.rkl_conv, 'reparameterize'):
            self.rkl_conv.reparameterize()
        if hasattr(self.m1_block, 'reparameterize'):
            self.m1_block.reparameterize()
        self.inference_mode = True # Set flag for the PatchEmbed block


# --- Model Definition ---

FASTVIT_SPECS = {
    "fastvit_t8": {
        "layers": [2, 2, 4, 2],
        "embed_dims": [48, 96, 192, 384],
        "mlp_ratios": [3.0, 3.0, 3.0, 3.0], # Use float
        "downsamples": [True, True, True, True], # Corresponds to entering the next stage
        "token_mixers": ("repmixer", "repmixer", "repmixer", "repmixer"),
        "repmixer_kernel_size": 3,
        "pos_embs": [None, None, None, None], # No CPE by default in T8
        "layer_scale_init_value": 1e-5,
        "cls_ratio": 2.0,
    },
    "fastvit_t12": {
        "layers": [2, 2, 6, 2],
        "embed_dims": [64, 128, 256, 512],
        "mlp_ratios": [3.0, 3.0, 3.0, 3.0],
        "downsamples": [True, True, True, True],
        "token_mixers": ("repmixer", "repmixer", "repmixer", "repmixer"),
        "repmixer_kernel_size": 3,
        "pos_embs": [None, None, None, None],
        "layer_scale_init_value": 1e-5,
        "cls_ratio": 2.0,
    },
    "fastvit_s12": {
        "layers": [2, 2, 6, 2],
        "embed_dims": [64, 128, 256, 512],
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
        "downsamples": [True, True, True, True],
        "token_mixers": ("repmixer", "repmixer", "repmixer", "repmixer"),
        "repmixer_kernel_size": 3,
        "pos_embs": [None, None, None, None],
        "layer_scale_init_value": 1e-5,
        "cls_ratio": 2.0,
    },
    "fastvit_sa12": {
        "layers": [2, 2, 6, 2],
        "embed_dims": [64, 128, 256, 512],
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
        "downsamples": [True, True, True, True],
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
        "repmixer_kernel_size": 3, # Used in first 3 stages
        "pos_embs": [None, None, None, partial(RepCPE, spatial_shape=(7, 7))], # CPE in last stage
        "layer_scale_init_value": 1e-5,
        "cls_ratio": 2.0,
    },
    "fastvit_sa24": {
        "layers": [4, 4, 12, 4],
        "embed_dims": [64, 128, 256, 512],
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
        "downsamples": [True, True, True, True],
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
        "repmixer_kernel_size": 3,
        "pos_embs": [None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
        "layer_scale_init_value": 1e-5,
        "cls_ratio": 2.0,
    },
    "fastvit_sa36": {
        "layers": [6, 6, 18, 6],
        "embed_dims": [64, 128, 256, 512],
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
        "downsamples": [True, True, True, True],
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
        "repmixer_kernel_size": 3,
        "pos_embs": [None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
        "layer_scale_init_value": 1e-6, # Different init value
        "cls_ratio": 2.0,
    },
     "fastvit_ma36": {
        "layers": [6, 6, 18, 6],
        "embed_dims": [76, 152, 304, 608], # Different embedding dims
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
        "downsamples": [True, True, True, True],
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
        "repmixer_kernel_size": 3,
        "pos_embs": [None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
        "layer_scale_init_value": 1e-6,
        "cls_ratio": 2.0,
    },
}


def basic_blocks(
    dim: int,
    stage_index: int, # Renamed from block_index for clarity
    num_stages: int,  # Total number of stages (e.g., 4)
    layers_per_stage: List[int], # e.g., [2, 2, 4, 2]
    token_mixer_type: str,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    act_layer: nn.Module = nn.GELU,
    norm_layer: nn.Module = nn.BatchNorm2d,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
    inference_mode=False,
) -> nn.Sequential:
    """Build FastViT blocks within a stage."""
    blocks = []
    total_blocks = sum(layers_per_stage)
    global_block_offset = sum(layers_per_stage[:stage_index])

    for block_idx in range(layers_per_stage[stage_index]):
        global_block_idx = global_block_offset + block_idx
        # Calculate drop path rate for the current block
        block_dpr = drop_path_rate * global_block_idx / (total_blocks - 1) if total_blocks > 1 else 0.0

        if token_mixer_type == "repmixer":
            blocks.append(
                RepMixerBlock(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer, # Pass type
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode,
                )
            )
        elif token_mixer_type == "attention":
            blocks.append(
                AttentionBlock(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer, # Pass type
                    norm_layer=norm_layer, # Pass type
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    # inference_mode not needed by AttentionBlock directly
                )
            )
        else:
            raise ValueError(f"Token mixer type: {token_mixer_type} not supported")
    blocks = nn.Sequential(*blocks)
    return blocks


class FastViT(nn.Module):
    """
    Implementation of FastViT architecture.

    Modified so the main `forward` method returns a list of features suitable
    for use as a backbone in detection/segmentation frameworks like Ultralytics YOLO.
    """
    def __init__(
        self,
        model_name: str,
        in_chans: int = 3,
        num_classes: int = 1000, # Still useful if using the head separately
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU,
        down_patch_size: int = 7,
        down_stride: int = 2,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = True,
        inference_mode: bool = False,
        resolution: int = 256,
        **kwargs,
    ) -> None:
        super().__init__()

        if model_name not in FASTVIT_SPECS:
            raise ValueError(f"Unknown model_name: {model_name}. Available: {list(FASTVIT_SPECS.keys())}")

        specs = FASTVIT_SPECS[model_name]
        self.model_name = model_name
        self.num_classes = num_classes
        self.inference_mode = inference_mode

        # Extract specs
        layers = specs['layers']
        embed_dims = specs['embed_dims']
        mlp_ratios = specs['mlp_ratios']
        downsamples = specs['downsamples']
        token_mixers = specs['token_mixers']
        repmixer_kernel_size = specs['repmixer_kernel_size']
        pos_embs = specs['pos_embs']
        layer_scale_init_value = specs['layer_scale_init_value']
        cls_ratio = specs['cls_ratio']

        if pos_embs is None:
            pos_embs = [None] * len(layers)
        elif len(pos_embs) != len(layers):
             raise ValueError(f"Length of pos_embs ({len(pos_embs)}) must match number of layers ({len(layers)})")

        # --- Build Network ---
        self.patch_embed = convolutional_stem(in_chans, embed_dims[0], inference_mode)

        self.stages = nn.ModuleList()
        num_stages = len(layers)
        curr_dim = embed_dims[0]

        for i in range(num_stages):
            stage_modules = []
            if pos_embs[i] is not None:
                if curr_dim != embed_dims[i]:
                     raise ValueError(f"CPE at stage {i}: current dim ({curr_dim}) != expected embed_dim ({embed_dims[i]})")
                stage_modules.append(
                    pos_embs[i](
                        in_channels=curr_dim,
                        embed_dim=curr_dim,
                        inference_mode=inference_mode
                    )
                )

            stage_blocks = basic_blocks(
                dim=curr_dim,
                stage_index=i,
                num_stages=num_stages,
                layers_per_stage=layers,
                token_mixer_type=token_mixers[i],
                kernel_size=repmixer_kernel_size if token_mixers[i] == "repmixer" else 3,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
            )
            stage_modules.append(stage_blocks)
            self.stages.append(nn.Sequential(*stage_modules))

            if i < num_stages - 1:
                if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                    next_dim = embed_dims[i+1]
                    patch_embed_layer = PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        in_channels=curr_dim,
                        embed_dim=next_dim,
                        inference_mode=inference_mode,
                    )
                    self.stages.append(patch_embed_layer)
                    curr_dim = next_dim

        # --- CLASSIFICATION HEAD (Separated from main forward) ---
        # This part is NOT used when FastViT acts as a backbone in Ultralytics
        self.final_norm = norm_layer(curr_dim) # Renamed from self.norm to avoid potential conflicts
        final_expanded_dim = int(curr_dim * cls_ratio)
        self.conv_exp = MobileOneBlock(
            in_channels=curr_dim,
            out_channels=final_expanded_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=curr_dim,
            inference_mode=inference_mode,
            use_se=True,
            num_conv_branches=1,
            activation=act_layer()
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.head = nn.Linear(final_expanded_dim, num_classes) if num_classes > 0 else nn.Identity()
        # --- END CLASSIFICATION HEAD ---

        self.apply(self._init_weights)

        # Calculate width_list (intermediate feature dimensions)
        # Needs to run in eval mode and handle potential errors during init
        self._is_initializing = True # Flag to prevent issues during width calculation
        try:
            self.width_list = self._calculate_width_list(in_chans, resolution)
        except Exception as e:
             print(f"Warning: Failed to calculate width_list during init: {e}. Using embed_dims as fallback.")
             self.width_list = specs['embed_dims']
        finally:
             self._is_initializing = False


    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None: nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        # Conv2d handled by submodules or ConvFFN internal init

    def _calculate_width_list(self, in_chans: int, resolution: int) -> List[int]:
        """Performs a dummy forward pass to get intermediate feature dimensions."""
        # Temporarily switch to eval mode if not already initializing
        original_mode_is_train = self.training
        if not self._is_initializing: self.eval()

        dummy_input = torch.randn(1, in_chans, resolution, resolution)
        width_list = []
        try:
            with torch.no_grad():
                # Use the actual forward logic to get features
                features = self.forward(dummy_input) # Forward now returns the list
                if isinstance(features, list):
                    width_list = [f.size(1) for f in features]
                else: # Should not happen with the corrected forward, but handle just in case
                     print("Warning: _calculate_width_list expected a list from forward, got Tensor. Using embed_dims.")
                     specs = FASTVIT_SPECS[self.model_name]
                     width_list = specs['embed_dims']

        except Exception as e:
             # Fallback if any error occurs during dummy forward
             print(f"Warning: Exception during width_list calculation: {e}. Using embed_dims.")
             specs = FASTVIT_SPECS[self.model_name]
             width_list = specs['embed_dims']
        finally:
            # Restore original training mode if needed
            if not self._is_initializing and original_mode_is_train: self.train()
        return width_list


    def reparameterize(self):
        """Reparameterize all reparameterizable modules in the network."""
        print(f"Starting reparameterization for model: {self.model_name}")
        for module in self.modules():
            if module is self: continue # Skip self
            if hasattr(module, 'reparameterize'):
                module.reparameterize()
        self.inference_mode = True # Set model's main inference flag
        print(f"Reparameterization complete for model: {self.model_name}")

    # --- FORWARD METHODS ---

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through stem and stages, returning intermediate features.
        This is the core logic used by the main `forward` method now.
        """
        x = self.patch_embed(x)
        features = []
        stage_output = x
        for block in self.stages:
            stage_output = block(stage_output)
            # Store output *after* each stage block sequence (nn.Sequential)
            # Do NOT store output after PatchEmbed layers between stages
            if isinstance(block, nn.Sequential):
                features.append(stage_output)
        return features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Main forward pass. Returns a list of feature maps from different stages.
        This is expected by frameworks like Ultralytics YOLO when using FastViT as a backbone.
        """
        return self.forward_features(x)

    def forward_head(self, features: List[torch.Tensor], pre_logits: bool = False) -> torch.Tensor:
        """
        Applies the classification head to the features obtained from `forward` or `forward_features`.

        Args:
            features: A list of feature tensors (output of `forward` or `forward_features`).
            pre_logits: If True, return the features before the final linear layer.

        Returns:
            Tensor of classification logits or pre-logits.
        """
        # Use the output of the last stage
        x = features[-1]

        # Apply final norm, conv expansion, GAP
        x = self.final_norm(x)
        x = self.conv_exp(x)
        x = self.gap(x)
        x = x.flatten(1)

        if pre_logits:
            return x
        # Apply final linear layer
        x = self.head(x)
        return x

# ... (Keep the model instantiation functions like fastvit_t8, fastvit_sa12, etc.) ...
def _create_fastvit(model_name: str, pretrained: bool = False, **kwargs):
    if pretrained:
        print(f"Warning: pretrained=True is not yet implemented for {model_name}. Loading with random weights.")
    kwargs.setdefault('resolution', 256) # Default resolution if not provided
    model = FastViT(model_name=model_name, **kwargs)
    return model

def fastvit_t8(pretrained=False, **kwargs): return _create_fastvit("fastvit_t8", pretrained=pretrained, **kwargs)
def fastvit_t12(pretrained=False, **kwargs): return _create_fastvit("fastvit_t12", pretrained=pretrained, **kwargs)
def fastvit_s12(pretrained=False, **kwargs): return _create_fastvit("fastvit_s12", pretrained=pretrained, **kwargs)
def fastvit_sa12(pretrained=False, **kwargs): return _create_fastvit("fastvit_sa12", pretrained=pretrained, **kwargs)
def fastvit_sa24(pretrained=False, **kwargs): return _create_fastvit("fastvit_sa24", pretrained=pretrained, **kwargs)
def fastvit_sa36(pretrained=False, **kwargs): return _create_fastvit("fastvit_sa36", pretrained=pretrained, **kwargs)
def fastvit_ma36(pretrained=False, **kwargs): return _create_fastvit("fastvit_ma36", pretrained=pretrained, **kwargs)


# --- Example Usage ---
if __name__ == "__main__":
    # --- Training Mode Example ---
    print("--- Training Mode ---")
    model_train = fastvit_sa12(num_classes=100, inference_mode=False)
    # print(model_train)
    print("Model created in training mode.")
    print(f"Output channels per stage (width_list): {model_train.width_list}")

    dummy_input = torch.randn(2, 3, 256, 256)
    output_train = model_train(dummy_input)
    print(f"Output shape (train): {output_train.shape}")

    # --- Inference Mode Example (after reparameterization) ---
    print("\n--- Inference Mode (Post Reparameterization) ---")
    # Create a new instance or deepcopy for reparameterization
    model_infer = copy.deepcopy(model_train)
    model_infer.reparameterize()
    model_infer.eval() # Set to eval mode for inference
    print("Model reparameterized for inference.")

    # Verify output shape and consistency (optional)
    with torch.no_grad():
        output_infer = model_infer(dummy_input)
    print(f"Output shape (infer): {output_infer.shape}")

    # Optional: Check if outputs are close (should be numerically very close)
    # diff = torch.mean(torch.abs(output_train - output_infer))
    # print(f"Mean absolute difference between train/infer outputs: {diff.item()}") # This requires running train forward again

    # --- Inference Mode Example (direct instantiation) ---
    print("\n--- Inference Mode (Direct Instantiation) ---")
    model_infer_direct = fastvit_sa12(num_classes=100, inference_mode=True)
    model_infer_direct.eval()
    print("Model created directly in inference mode.")
    print(f"Output channels per stage (width_list): {model_infer_direct.width_list}")

    with torch.no_grad():
        output_infer_direct = model_infer_direct(dummy_input)
    print(f"Output shape (infer direct): {output_infer_direct.shape}")

    # Optional: Check difference between reparameterized and direct inference models
    # diff_direct = torch.mean(torch.abs(output_infer - output_infer_direct))
    # print(f"Mean absolute difference between reparam/direct infer outputs: {diff_direct.item()}")


    # --- Test another variant ---
    print("\n--- Testing fastvit_ma36 ---")
    model_ma36 = fastvit_ma36(num_classes=50, inference_mode=False)
    print(f"MA36 width_list: {model_ma36.width_list}")
    output_ma36 = model_ma36(dummy_input)
    print(f"MA36 output shape: {output_ma36.shape}")

    model_ma36.reparameterize()
    model_ma36.eval()
    with torch.no_grad():
        output_ma36_infer = model_ma36(dummy_input)
    print(f"MA36 output shape (infer): {output_ma36_infer.shape}")