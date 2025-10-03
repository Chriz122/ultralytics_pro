import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple
# from timm.models.registry import register_model # Removed as we use local factory functions now

# Helper functions (unchanged)
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

def gcd(a,b):
    if a<b:
        a,b=b,a
    while(a%b != 0):
        c = a%b
        a=b
        b=c
    return b

def MyNorm(dim):
    return nn.GroupNorm(1, dim)

# Modules (mostly unchanged, removed 'args')
class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

class GhostModule(nn.Module):
    # Removed 'args' from init
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode=None):
        super(GhostModule, self).__init__()
        self.mode = mode if mode else 'ori' # Default to 'ori' if None
        self.gate_loc = 'before'

        self.inter_mode = 'nearest'
        self.scale = 1.0

        self.infer_mode = False
        self.num_conv_branches = 3
        self.dconv_scale = True
        # Defaulting gate_fn to Sigmoid as args is removed.
        # Adjust if other gate functions were intended via args.
        self.gate_fn = nn.Sigmoid()
        # print(f"GhostModule init: inp={inp}, oup={oup}, ks={kernel_size}, ratio={ratio}, dw_size={dw_size}, stride={stride}, relu={relu}, mode={self.mode}")

        if self.mode in ['ori']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels*(ratio-1)
            if new_channels < 0 : # Avoid negative channels if oup/ratio < 1
                new_channels = 0
                init_channels = oup

            # Ensure new_channels + init_channels >= oup
            if init_channels + new_channels < oup:
                init_channels = oup - new_channels # Adjust init_channels if needed

            if init_channels <= 0 or new_channels < 0 :
                 raise ValueError(f"GhostModule: Invalid channel calculation for mode='{self.mode}'. inp={inp}, oup={oup}, ratio={ratio} -> init={init_channels}, new={new_channels}")


            self.primary_conv_module = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Identity(),
            )
            if new_channels > 0:
                self.cheap_operation_module = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                    nn.BatchNorm2d(new_channels),
                    nn.ReLU(inplace=True) if relu else nn.Identity(),
                )
            else:
                 self.cheap_operation_module = None # No cheap operation if new_channels is 0

            # --- Training time branches ---
            self.primary_rpr_skip = nn.BatchNorm2d(inp) if inp == init_channels and stride == 1 else None
            primary_rpr_conv = list()
            for _ in range(self.num_conv_branches):
                primary_rpr_conv.append(self._conv_bn(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False))
            self.primary_rpr_conv = nn.ModuleList(primary_rpr_conv)
            self.primary_rpr_scale = None
            if kernel_size > 1:
                self.primary_rpr_scale = self._conv_bn(inp, init_channels, 1, stride, 0, bias=False) # Match stride
            self.primary_activation = nn.ReLU(inplace=True) if relu else nn.Identity()


            if self.cheap_operation_module is not None:
                # Check if init_channels == new_channels (can happen if ratio=2, oup is odd)
                # And if the cheap operation doesn't change dimensionality (stride=1 implicitly here)
                self.cheap_rpr_skip = nn.BatchNorm2d(init_channels) if init_channels == new_channels else None
                cheap_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    # Ensure groups <= in_channels and groups <= out_channels for Conv2d
                    valid_groups = min(init_channels, new_channels) if init_channels > 0 else 1
                    if init_channels % valid_groups != 0 or new_channels % valid_groups != 0:
                        valid_groups = gcd(init_channels, new_channels) if init_channels > 0 and new_channels > 0 else 1 # Fallback if groups not divisible


                    cheap_rpr_conv.append(self._conv_bn(init_channels, new_channels, dw_size, 1, dw_size//2, groups=valid_groups, bias=False))
                self.cheap_rpr_conv = nn.ModuleList(cheap_rpr_conv)
                self.cheap_rpr_scale = None
                if dw_size > 1:
                     valid_groups = min(init_channels, new_channels) if init_channels > 0 else 1
                     if init_channels % valid_groups != 0 or new_channels % valid_groups != 0:
                         valid_groups = gcd(init_channels, new_channels) if init_channels > 0 and new_channels > 0 else 1
                     self.cheap_rpr_scale = self._conv_bn(init_channels, new_channels, 1, 1, 0, groups=valid_groups, bias=False)
                self.cheap_activation = nn.ReLU(inplace=True) if relu else nn.Identity()
                self.in_channels_cheap = init_channels # Store for reparam id_tensor
                self.groups_cheap = valid_groups
                self.kernel_size_cheap = dw_size
            else:
                 self.cheap_rpr_skip = None
                 self.cheap_rpr_conv = None
                 self.cheap_rpr_scale = None
                 self.cheap_activation = nn.Identity()


            self.in_channels_primary = inp # Store for reparam id_tensor
            self.groups_primary = 1 # Pointwise conv
            self.kernel_size_primary = kernel_size # Store for reparam padding

        elif self.mode in ['ori_shortcut_mul_conv15']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels*(ratio-1)
            if new_channels < 0 :
                new_channels = 0
                init_channels = oup
            if init_channels + new_channels < oup:
                init_channels = oup - new_channels

            if init_channels <= 0 or new_channels < 0 :
                 raise ValueError(f"GhostModule: Invalid channel calculation for mode='{self.mode}'. inp={inp}, oup={oup}, ratio={ratio} -> init={init_channels}, new={new_channels}")

            # Shortcut path
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1,5), stride=1, padding=(0,2), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5,1), stride=1, padding=(2,0), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
            )

            # Main path (identical to 'ori' mode for building blocks)
            self.primary_conv_module = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Identity(),
            )
            if new_channels > 0:
                self.cheap_operation_module = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                    nn.BatchNorm2d(new_channels),
                    nn.ReLU(inplace=True) if relu else nn.Identity(),
                )
            else:
                self.cheap_operation_module = None

            # --- Training time branches ---
            self.primary_rpr_skip = nn.BatchNorm2d(inp) if inp == init_channels and stride == 1 else None
            primary_rpr_conv = list()
            for _ in range(self.num_conv_branches):
                primary_rpr_conv.append(self._conv_bn(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False))
            self.primary_rpr_conv = nn.ModuleList(primary_rpr_conv)
            self.primary_rpr_scale = None
            if kernel_size > 1:
                self.primary_rpr_scale = self._conv_bn(inp, init_channels, 1, stride, 0, bias=False) # Match stride
            self.primary_activation = nn.ReLU(inplace=True) if relu else nn.Identity()


            if self.cheap_operation_module is not None:
                self.cheap_rpr_skip = nn.BatchNorm2d(init_channels) if init_channels == new_channels else None
                cheap_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    valid_groups = min(init_channels, new_channels) if init_channels > 0 else 1
                    if init_channels % valid_groups != 0 or new_channels % valid_groups != 0:
                       valid_groups = gcd(init_channels, new_channels) if init_channels > 0 and new_channels > 0 else 1

                    cheap_rpr_conv.append(self._conv_bn(init_channels, new_channels, dw_size, 1, dw_size//2, groups=valid_groups, bias=False))
                self.cheap_rpr_conv = nn.ModuleList(cheap_rpr_conv)
                self.cheap_rpr_scale = None
                if dw_size > 1:
                    valid_groups = min(init_channels, new_channels) if init_channels > 0 else 1
                    if init_channels % valid_groups != 0 or new_channels % valid_groups != 0:
                         valid_groups = gcd(init_channels, new_channels) if init_channels > 0 and new_channels > 0 else 1
                    self.cheap_rpr_scale = self._conv_bn(init_channels, new_channels, 1, 1, 0, groups=valid_groups, bias=False)
                self.cheap_activation = nn.ReLU(inplace=True) if relu else nn.Identity()
                self.in_channels_cheap = init_channels
                self.groups_cheap = valid_groups
                self.kernel_size_cheap = dw_size
            else:
                 self.cheap_rpr_skip = None
                 self.cheap_rpr_conv = None
                 self.cheap_rpr_scale = None
                 self.cheap_activation = nn.Identity()


            self.in_channels_primary = inp
            self.groups_primary = 1
            self.kernel_size_primary = kernel_size
        else:
            raise ValueError(f"Unsupported GhostModule mode: {self.mode}")

    def forward(self, x):
        if self.infer_mode:
            # Inference mode uses fused modules
            x1 = self.primary_conv_module(x)
            if self.cheap_operation_module is not None:
                 x2 = self.cheap_operation_module(x1)
                 out = torch.cat([x1, x2], dim=1)
            else: # Handle case where cheap op might not exist
                 out = x1

            if self.mode == 'ori_shortcut_mul_conv15':
                 # Apply shortcut and gating
                 res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2)) # Original uses stride=2 pool
                 if out.shape[-2:] != res.shape[-2:]:
                     gate = F.interpolate(self.gate_fn(res/self.scale), size=out.shape[-2:], mode=self.inter_mode)
                 else:
                     gate = self.gate_fn(res/self.scale)

                 # Ensure output channels match self.oup before applying gate
                 return out[:, :self.oup, :, :] * gate
            else: # 'ori' mode
                 # Ensure output channels match self.oup
                 return out[:, :self.oup, :, :]

        else:
            # Training mode uses reparameterization branches
            identity_out = 0
            if self.primary_rpr_skip is not None:
                identity_out = self.primary_rpr_skip(x)
            scale_out = 0
            if self.primary_rpr_scale is not None and self.dconv_scale:
                scale_out = self.primary_rpr_scale(x)
            x1_train = scale_out + identity_out
            for ix in range(self.num_conv_branches):
                x1_train += self.primary_rpr_conv[ix](x)
            x1_train = self.primary_activation(x1_train)

            if self.cheap_operation_module is not None and self.cheap_rpr_conv is not None:
                cheap_identity_out = 0
                if self.cheap_rpr_skip is not None:
                    cheap_identity_out = self.cheap_rpr_skip(x1_train)
                cheap_scale_out = 0
                if self.cheap_rpr_scale is not None and self.dconv_scale:
                    cheap_scale_out = self.cheap_rpr_scale(x1_train)
                x2_train = cheap_scale_out + cheap_identity_out
                for ix in range(self.num_conv_branches):
                    x2_train += self.cheap_rpr_conv[ix](x1_train)
                x2_train = self.cheap_activation(x2_train)
                out_train = torch.cat([x1_train, x2_train], dim=1)
            else:
                out_train = x1_train


            if self.mode == 'ori_shortcut_mul_conv15':
                 res = self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2)) # Original uses stride=2 pool
                 if out_train.shape[-2:] != res.shape[-2:]:
                      gate = F.interpolate(self.gate_fn(res/self.scale),size=out_train.shape[-2:],mode=self.inter_mode)
                 else:
                     gate = self.gate_fn(res/self.scale)

                 if self.gate_loc=='before':
                     return out_train[:,:self.oup,:,:] * gate
                 else: # gate_loc == 'after' ? (Original code logic was slightly ambiguous here)
                     # This applies gate *after* interpolation if sizes differ
                     interpolated_res = F.interpolate(res, size=out_train.shape[-2:], mode=self.inter_mode)
                     return out_train[:,:self.oup,:,:] * self.gate_fn(interpolated_res / self.scale)
            else: # 'ori' mode
                 return out_train[:,:self.oup,:,:] # Truncate output channels to self.oup

    def reparameterize(self):
        if self.infer_mode:
            return

        # --- Reparameterize Primary Conv ---
        primary_kernel, primary_bias = self._get_kernel_bias_primary()
        # Create the final Conv2d layer for primary path
        fused_primary_conv = nn.Conv2d(
            in_channels=self.primary_rpr_conv[0].conv.in_channels,
            out_channels=self.primary_rpr_conv[0].conv.out_channels,
            kernel_size=self.primary_rpr_conv[0].conv.kernel_size,
            stride=self.primary_rpr_conv[0].conv.stride,
            padding=self.primary_rpr_conv[0].conv.padding,
            dilation=self.primary_rpr_conv[0].conv.dilation,
            groups=self.primary_rpr_conv[0].conv.groups,
            bias=True)
        fused_primary_conv.weight.data = primary_kernel
        fused_primary_conv.bias.data = primary_bias
        # Replace the sequential module with the fused conv + activation
        self.primary_conv_module = nn.Sequential(
            fused_primary_conv,
            self.primary_activation if not isinstance(self.primary_activation, nn.Identity) else nn.Identity()
        )


        # --- Reparameterize Cheap Operation (if exists) ---
        if hasattr(self, 'cheap_rpr_conv') and self.cheap_rpr_conv is not None and len(self.cheap_rpr_conv)>0 :
            cheap_kernel, cheap_bias = self._get_kernel_bias_cheap()
            # Create the final Conv2d layer for cheap path
            fused_cheap_conv = nn.Conv2d(
                in_channels=self.cheap_rpr_conv[0].conv.in_channels,
                out_channels=self.cheap_rpr_conv[0].conv.out_channels,
                kernel_size=self.cheap_rpr_conv[0].conv.kernel_size,
                stride=self.cheap_rpr_conv[0].conv.stride,
                padding=self.cheap_rpr_conv[0].conv.padding,
                dilation=self.cheap_rpr_conv[0].conv.dilation,
                groups=self.cheap_rpr_conv[0].conv.groups,
                bias=True)
            fused_cheap_conv.weight.data = cheap_kernel
            fused_cheap_conv.bias.data = cheap_bias
            # Replace the sequential module with the fused conv + activation
            self.cheap_operation_module = nn.Sequential(
                fused_cheap_conv,
                self.cheap_activation if not isinstance(self.cheap_activation, nn.Identity) else nn.Identity()
            )
        elif hasattr(self, 'cheap_operation_module'): # Ensure module exists even if no reparam happens
             pass # Keep the original sequential if no reparam branches existed
        else:
             self.cheap_operation_module = None # Ensure it's None if no cheap op existed


        # --- Cleanup Training Branches ---
        for para in self.parameters():
            para.detach_() # Detach all parameters first

        # Delete specific reparam attributes safely
        if hasattr(self, 'primary_rpr_conv'): delattr(self, 'primary_rpr_conv')
        if hasattr(self, 'primary_rpr_scale'): delattr(self, 'primary_rpr_scale')
        if hasattr(self, 'primary_rpr_skip'): delattr(self, 'primary_rpr_skip')
        if hasattr(self, 'primary_activation'): delattr(self, 'primary_activation') # Activation is now part of fused module

        if hasattr(self, 'cheap_rpr_conv'): delattr(self, 'cheap_rpr_conv')
        if hasattr(self, 'cheap_rpr_scale'): delattr(self, 'cheap_rpr_scale')
        if hasattr(self, 'cheap_rpr_skip'): delattr(self, 'cheap_rpr_skip')
        if hasattr(self, 'cheap_activation'): delattr(self, 'cheap_activation') # Activation is now part of fused module

        # Cleanup potentially created id_tensors
        if hasattr(self, 'id_tensor_primary'): delattr(self, 'id_tensor_primary')
        if hasattr(self, 'id_tensor_cheap'): delattr(self, 'id_tensor_cheap')


        self.infer_mode = True

    def _get_kernel_bias_primary(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel_scale, bias_scale = 0, 0
        if self.primary_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.primary_rpr_scale, 'primary')
            pad = self.kernel_size_primary // 2
            if pad > 0:
                 kernel_scale = F.pad(kernel_scale, [pad] * 4)

        kernel_identity, bias_identity = 0, 0
        if self.primary_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.primary_rpr_skip, 'primary')

        kernel_conv, bias_conv = 0, 0
        for ix in range(self.num_conv_branches):
             if hasattr(self.primary_rpr_conv[ix], 'conv'): # Ensure conv exists
                _kernel, _bias = self._fuse_bn_tensor(self.primary_rpr_conv[ix], 'primary')
                kernel_conv += _kernel
                bias_conv += _bias

        return kernel_conv + kernel_scale + kernel_identity, bias_conv + bias_scale + bias_identity

    def _get_kernel_bias_cheap(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel_scale, bias_scale = 0, 0
        if self.cheap_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.cheap_rpr_scale, 'cheap')
            pad = self.kernel_size_cheap // 2
            if pad > 0:
                kernel_scale = F.pad(kernel_scale, [pad] * 4)

        kernel_identity, bias_identity = 0, 0
        if self.cheap_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.cheap_rpr_skip, 'cheap')

        kernel_conv, bias_conv = 0, 0
        # Check if cheap_rpr_conv exists and is iterable
        if hasattr(self, 'cheap_rpr_conv') and self.cheap_rpr_conv is not None:
             for ix in range(self.num_conv_branches):
                # Check if the specific conv module exists
                if len(self.cheap_rpr_conv) > ix and hasattr(self.cheap_rpr_conv[ix], 'conv'):
                     _kernel, _bias = self._fuse_bn_tensor(self.cheap_rpr_conv[ix], 'cheap')
                     kernel_conv += _kernel
                     bias_conv += _bias

        return kernel_conv + kernel_scale + kernel_identity, bias_conv + bias_scale + bias_identity

    def _fuse_bn_tensor(self, branch, branch_type) -> Tuple[torch.Tensor, torch.Tensor]:
         # Select attributes based on branch type
        if branch_type == 'primary':
            in_channels = self.in_channels_primary
            groups = self.groups_primary
            kernel_size = self.kernel_size_primary
            id_tensor_name = 'id_tensor_primary'
        elif branch_type == 'cheap':
            in_channels = self.in_channels_cheap
            groups = self.groups_cheap
            kernel_size = self.kernel_size_cheap
            id_tensor_name = 'id_tensor_cheap'
        else:
            raise ValueError("Invalid branch_type for _fuse_bn_tensor")

        if isinstance(branch, nn.Sequential):
            # Make sure 'conv' and 'bn' attributes exist
            if not hasattr(branch, 'conv') or not hasattr(branch, 'bn'):
                 raise AttributeError(f"Branch Sequential module missing 'conv' or 'bn': {branch}")
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
             # Create identity kernel if it doesn't exist for this branch type
             if not hasattr(self, id_tensor_name):
                 # input_dim calculation needs careful check for groups > 1
                 # For BN identity, the input is effectively ungrouped before BN
                 # But the kernel shape needs to match the *output* of the implicit identity conv
                 # which has out_channels = in_channels
                 out_channels = in_channels
                 # input_dim = in_channels // groups # This is for the *conv* input, not BN identity

                 # Kernel shape: (out_channels, in_channels // groups, k, k)
                 # For identity, k=1 initially, then padded. Groups=in_channels for DW-like identity
                 identity_groups = in_channels # Treat identity like a DW conv for kernel creation
                 identity_kernel_size = kernel_size # Use the target kernel size
                 # print(f"Creating id_tensor_{branch_type}: out={out_channels}, in={in_channels}, groups={identity_groups}, ks={identity_kernel_size}")

                 # Check if groups is valid before division
                 if identity_groups <= 0:
                      raise ValueError(f"Groups must be positive, got {identity_groups}")
                 if in_channels % identity_groups != 0:
                      raise ValueError(f"in_channels {in_channels} not divisible by groups {identity_groups}")

                 input_dim_per_group = in_channels // identity_groups

                 # Kernel shape: (out_channels, input_dim_per_group, k, k)
                 kernel_value = torch.zeros((out_channels, input_dim_per_group, identity_kernel_size, identity_kernel_size),
                                            dtype=branch.weight.dtype,
                                            device=branch.weight.device)

                 # Fill the center element for identity mapping
                 center = identity_kernel_size // 2
                 for i in range(out_channels):
                     # Map output channel 'i' to its corresponding input channel within its group
                     # Input channel index = i % input_dim_per_group (for the specific group)
                     # Group index = i // input_dim_per_group
                     # Overall input channel = group_index * input_dim_per_group + (i % input_dim_per_group) --> simplifies to 'i' when groups=out_channels

                     # Correct indexing: kernel_value[output_channel, input_channel_in_group, row, col]
                     kernel_value[i, i % input_dim_per_group, center, center] = 1

                 setattr(self, id_tensor_name, kernel_value)

             kernel = getattr(self, id_tensor_name)
             running_mean = branch.running_mean
             running_var = branch.running_var
             gamma = branch.weight
             beta = branch.bias
             eps = branch.eps
        else:
             raise TypeError(f"Unsupported branch type for fusing: {type(branch)}")


        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1) # Reshape gamma/std for broadcasting

        # Ensure kernel and t are compatible for multiplication
        # Kernel shape: (out_channels, in_channels // groups, k, k)
        # t shape: (out_channels, 1, 1, 1) - broadcasts correctly
        fused_kernel = kernel * t

        # Bias shape: (out_channels)
        fused_bias = beta - running_mean * gamma / std

        return fused_kernel, fused_bias


    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        """ Helper method to construct conv-batchnorm layers. """
        mod_list = nn.Sequential()
         # Ensure groups is valid
        if groups <= 0:
            raise ValueError("groups must be positive")
        if in_channels % groups != 0:
             # Fallback: try greatest common divisor? Or just use 1?
             print(f"Warning: in_channels {in_channels} not divisible by groups {groups}. Setting groups=1.")
             groups = 1
             # groups = gcd(in_channels, out_channels) # Alternative, might break DW assumption
        if out_channels % groups != 0:
             print(f"Warning: out_channels {out_channels} not divisible by groups {groups}. Setting groups=1.")
             groups = 1
             # groups = gcd(in_channels, out_channels)

        mod_list.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              groups=groups,
                                              bias=bias))
        mod_list.add_module('bn', nn.BatchNorm2d(out_channels))
        return mod_list

class GhostBottleneck(nn.Module):
    # Removed 'args' from init
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., layer_id=None):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        self.layer_id = layer_id # Store layer_id

        self.num_conv_branches = 3
        self.infer_mode = False
        self.dconv_scale = True

        # Point-wise expansion
        # Decide mode based on layer_id
        ghost1_mode = 'ori_shortcut_mul_conv15' if layer_id is not None and layer_id > 1 else 'ori'
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True, mode=ghost1_mode)

        # Depth-wise convolution (if stride > 1)
        self.conv_dw = None
        self.bn_dw = None
        self.dw_rpr_skip = None
        self.dw_rpr_conv = None
        self.dw_rpr_scale = None
        if self.stride > 1:
            # --- Inference Path ---
            # Note: These will be created during reparameterize() if not in infer_mode initially
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                 padding=(dw_kernel_size-1)//2,
                                 groups=mid_chs, bias=False) # bias=False initially
            self.bn_dw = nn.BatchNorm2d(mid_chs) # BN comes after conv in inference path


            # --- Training Path ---
            # Note: stride = 1 case handled by lack of dw_rpr_skip definition
            # self.dw_rpr_skip = nn.BatchNorm2d(mid_chs) # only created if stride=1? Original logic was ambiguous here
            # Let's be explicit: skip branch only useful if stride=1, but DW conv itself only happens if stride>1
            # Therefore, dw_rpr_skip seems unnecessary for stride>1 DW. Let's keep it None.
            self.dw_rpr_skip = None # Explicitly None for stride > 1 DW conv

            dw_rpr_conv = list()
            for _ in range(self.num_conv_branches):
                dw_rpr_conv.append(self._conv_bn(mid_chs, mid_chs, dw_kernel_size, stride, (dw_kernel_size-1)//2, groups=mid_chs, bias=False))
            self.dw_rpr_conv = nn.ModuleList(dw_rpr_conv)

            self.dw_rpr_scale = None
            # Scale branch should also match stride
            if dw_kernel_size > 1:
                # Original code had stride=2 here fixed. Should match self.stride.
                self.dw_rpr_scale = self._conv_bn(mid_chs, mid_chs, 1, stride, 0, groups=mid_chs, bias=False)

            # Store info needed for reparameterization of DW conv
            self.kernel_size_dw = dw_kernel_size
            self.in_channels_dw = mid_chs
            self.groups_dw = mid_chs # groups = in_channels for depthwise


        # Squeeze-and-excitation
        self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio) if has_se else None

        # Point-wise linear projection
        # Original code always used 'ori' mode here, regardless of layer_id
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False, mode='ori')

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Identity() # Use Identity for clarity
        else:
            # Original shortcut used DW+BN -> PW+BN
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride, # Stride applied here
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False), # PW has stride 1
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            if self.infer_mode:
                 if self.conv_dw is None: raise RuntimeError("conv_dw is None during inference mode with stride > 1")
                 x = self.conv_dw(x)
                 x = self.bn_dw(x) # BN is separate in infer mode
            else:
                # Training mode DW conv (Reparameterized branches)
                # dw_identity_out = 0 # dw_rpr_skip is None for stride > 1
                # if self.dw_rpr_skip is not None:
                #     dw_identity_out = self.dw_rpr_skip(x)
                dw_scale_out = 0
                if self.dw_rpr_scale is not None and self.dconv_scale:
                    dw_scale_out = self.dw_rpr_scale(x)

                x_dw = dw_scale_out # Start with scale branch output (+ 0 from identity)
                if self.dw_rpr_conv is not None: # Check if list exists
                    for ix in range(self.num_conv_branches):
                        x_dw += self.dw_rpr_conv[ix](x)
                else:
                     # Should not happen if stride > 1, but safeguard
                     print("Warning: dw_rpr_conv is None during training with stride > 1")
                     x_dw = x # Pass through if no branches exist? Or raise error?

                x = x_dw # Update x with DW result


        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        # Shortcut connection
        shortcut_out = self.shortcut(residual)
        x += shortcut_out
        return x

    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        """ Helper method to construct conv-batchnorm layers specific to bottleneck. """
        mod_list = nn.Sequential()
        # Ensure groups is valid
        if groups <= 0:
            raise ValueError("groups must be positive")
        if in_channels % groups != 0:
             print(f"Warning (Bottleneck): in_channels {in_channels} not divisible by groups {groups}. Setting groups=1.")
             groups = 1
        if out_channels % groups != 0:
             print(f"Warning (Bottleneck): out_channels {out_channels} not divisible by groups {groups}. Setting groups=1.")
             groups = 1

        mod_list.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              groups=groups,
                                              bias=bias))
        mod_list.add_module('bn', nn.BatchNorm2d(out_channels))
        return mod_list

    def reparameterize(self):
        # Reparameterize internal GhostModules first
        self.ghost1.reparameterize()
        self.ghost2.reparameterize()

        # Reparameterize the optional Depthwise convolution layer
        if self.infer_mode or self.stride == 1: # Only reparam DW if stride > 1 and not already done
            self.infer_mode = True # Ensure infer_mode is set even if no DW reparam needed
            return

        # Proceed with DW reparameterization
        dw_kernel, dw_bias = self._get_kernel_bias_dw()

        # Check if self.dw_rpr_conv is valid before accessing its attributes
        if self.dw_rpr_conv is None or len(self.dw_rpr_conv) == 0 or not hasattr(self.dw_rpr_conv[0], 'conv'):
             raise RuntimeError("Cannot reparameterize DW conv: training branches are missing or invalid.")


        # Recreate self.conv_dw with fused parameters
        self.conv_dw = nn.Conv2d(in_channels=self.dw_rpr_conv[0].conv.in_channels,
                                      out_channels=self.dw_rpr_conv[0].conv.out_channels,
                                      kernel_size=self.dw_rpr_conv[0].conv.kernel_size,
                                      stride=self.dw_rpr_conv[0].conv.stride,
                                      padding=self.dw_rpr_conv[0].conv.padding,
                                      dilation=self.dw_rpr_conv[0].conv.dilation,
                                      groups=self.dw_rpr_conv[0].conv.groups,
                                      bias=True) # Bias is now True
        self.conv_dw.weight.data = dw_kernel
        self.conv_dw.bias.data = dw_bias
        self.bn_dw = nn.Identity() # BN is fused into conv_dw

        # --- Cleanup DW Training Branches ---
        # Detach parameters (redundant if GhostModule.reparam also detaches, but safe)
        # for para in self.parameters():
        #     para.detach_()
        if hasattr(self, 'dw_rpr_conv'): delattr(self, 'dw_rpr_conv')
        if hasattr(self, 'dw_rpr_scale'): delattr(self, 'dw_rpr_scale')
        if hasattr(self, 'dw_rpr_skip'): delattr(self, 'dw_rpr_skip') # Should be None anyway
        # Cleanup potentially created id_tensor for DW
        if hasattr(self, 'id_tensor_dw'): delattr(self, 'id_tensor_dw')

        self.infer_mode = True # Mark bottleneck as reparameterized

    def _get_kernel_bias_dw(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias for DW conv. """
        kernel_scale, bias_scale = 0, 0
        if self.dw_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor_dw(self.dw_rpr_scale)
            pad = self.kernel_size_dw // 2
            if pad > 0:
                kernel_scale = F.pad(kernel_scale, [pad] * 4)

        kernel_identity, bias_identity = 0, 0
        # dw_rpr_skip should be None for stride>1 DW, so this part should yield 0
        if self.dw_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor_dw(self.dw_rpr_skip)

        kernel_conv, bias_conv = 0, 0
        if self.dw_rpr_conv is not None: # Check exists
             for ix in range(self.num_conv_branches):
                 if len(self.dw_rpr_conv) > ix and hasattr(self.dw_rpr_conv[ix], 'conv'): # Check valid index and conv exists
                     _kernel, _bias = self._fuse_bn_tensor_dw(self.dw_rpr_conv[ix])
                     kernel_conv += _kernel
                     bias_conv += _bias

        return kernel_conv + kernel_scale + kernel_identity, bias_conv + bias_scale + bias_identity

    def _fuse_bn_tensor_dw(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer specifically for the DW conv branches. """
        in_channels = self.in_channels_dw
        groups = self.groups_dw # Should be equal to in_channels for DW
        kernel_size = self.kernel_size_dw
        id_tensor_name = 'id_tensor_dw'

        if isinstance(branch, nn.Sequential):
            if not hasattr(branch, 'conv') or not hasattr(branch, 'bn'):
                 raise AttributeError(f"DW Branch Sequential module missing 'conv' or 'bn': {branch}")
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
             if not hasattr(self, id_tensor_name):
                  out_channels = in_channels # For identity
                  identity_groups = groups # Use DW groups
                  identity_kernel_size = kernel_size
                  if identity_groups <= 0: raise ValueError("Groups must be positive")
                  if in_channels % identity_groups != 0: raise ValueError("in_channels not divisible by groups")
                  input_dim_per_group = in_channels // identity_groups

                  kernel_value = torch.zeros((out_channels, input_dim_per_group, identity_kernel_size, identity_kernel_size),
                                             dtype=branch.weight.dtype, device=branch.weight.device)
                  center = identity_kernel_size // 2
                  for i in range(out_channels):
                       kernel_value[i, i % input_dim_per_group, center, center] = 1
                  setattr(self, id_tensor_name, kernel_value)

             kernel = getattr(self, id_tensor_name)
             running_mean = branch.running_mean
             running_var = branch.running_var
             gamma = branch.weight
             beta = branch.bias
             eps = branch.eps
        else:
             raise TypeError(f"Unsupported DW branch type for fusing: {type(branch)}")

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        fused_kernel = kernel * t
        fused_bias = beta - running_mean * gamma / std
        return fused_kernel, fused_bias


# Define Architectures using SPECS dictionary
GHOSTNET_SPECS = {
    "GhostNet1.0": [
        # Format: List[List[List[k, exp_size, c, SE, s]]]
        # Each inner list represents a stage config
        # Each innermost list represents a block config: [kernel, hidden_channels, out_channels, se_ratio, stride]
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2], [3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2], [5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2], [3, 200,  80, 0, 1], [3, 184,  80, 0, 1], [3, 184,  80, 0, 1], [3, 480, 112, 0.25, 1], [3, 672, 112, 0.25, 1]],
        # stage5
        [[5, 672, 160, 0.25, 2], [5, 960, 160, 0, 1], [5, 960, 160, 0.25, 1], [5, 960, 160, 0, 1], [5, 960, 160, 0.25, 1]]
    ]
    # Add other variants like "GhostNet0.5" here if needed
}


class GhostNet(nn.Module):
    # Removed cfgs, num_classes, dropout, block, args from init
    # Added model_name
    def __init__(self, model_name="GhostNet1.0", width=1.0):
        super(GhostNet, self).__init__()

        if model_name not in GHOSTNET_SPECS:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(GHOSTNET_SPECS.keys())}")

        cfgs = GHOSTNET_SPECS[model_name]
        self.width = width
        block = GhostBottleneck # Keep default block

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks (stages)
        self.stages = nn.ModuleList() # Use ModuleList to store stages
        layer_id = 0
        exp_size = 0 # Keep track of last exp_size for final conv
        for stage_cfg in cfgs:
            stage_layers = []
            for k, hidden_channel_base, c, se_ratio, s in stage_cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(hidden_channel_base * width, 4) # hidden_channel is exp_size in original cfgs
                exp_size = hidden_channel_base # Store the base expansion size for the final conv

                # Pass necessary params to GhostBottleneck
                stage_layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                          se_ratio=se_ratio, layer_id=layer_id))
                input_channel = output_channel
                layer_id += 1
            self.stages.append(nn.Sequential(*stage_layers))

        # Add the final ConvBnAct layer (similar to original structure's last stage)
        output_channel_final_conv = _make_divisible(exp_size * width, 4) # Use last exp_size
        self.conv_last = ConvBnAct(input_channel, output_channel_final_conv, 1) # Kernel size 1
        # input_channel = output_channel_final_conv # Update input_channel if needed later

        # --- Calculate width_list ---
        # Perform a dummy forward pass to get intermediate feature map sizes
        self.eval() # Set to eval mode for dummy pass (esp. for BN)
        with torch.no_grad():
             # Use a standard input size, e.g., 224x224 or adapt as needed
             # Note: MobileNetV4 example used 640x640
             dummy_input = torch.randn(1, 3, 224, 224)
             features = self._forward_features(dummy_input)
             # Store channel dimension of stage 2, 3, 4, 5 outputs
             # Indexing: features[0]=stage2_out, features[1]=stage3_out, etc.
             if len(features) != 4:
                  print(f"Warning: Expected 4 feature maps from _forward_features, got {len(features)}")
                  self.width_list = []
             else:
                  self.width_list = [f.size(1) for f in features]
        self.train() # Set back to train mode


    def _forward_features(self, x):
        # Helper function for forward pass, used by both forward() and width_list calculation
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Pass through stages
        s1_out = self.stages[0](x)
        s2_out = self.stages[1](s1_out)
        s3_out = self.stages[2](s2_out)
        s4_out = self.stages[3](s3_out)
        s5_out = self.stages[4](s4_out)

        # Apply final conv layer
        s5_out_conv = self.conv_last(s5_out) # Output of final conv might be useful too

        # Return outputs of stages 2, 3, 4, and the final conv output
        # Adjust which features to return based on requirements
        # Returning stages 2, 3, 4, 5 output *before* final conv:
        # return [s2_out, s3_out, s4_out, s5_out]

        # Returning stages 2, 3, 4 output, and final conv output (like some detection backbones)
        return [s2_out, s3_out, s4_out, s5_out_conv]


    def forward(self, x):
        return self._forward_features(x)

    def reparameterize(self):
        print("Reparameterizing GhostNet...")
        for name, module in self.named_modules():
            # Check if the module itself is GhostModule/GhostBottleneck OR
            # if it's a Sequential containing them (like self.stages)
             if isinstance(module, (GhostModule, GhostBottleneck)):
                  # Check if the module has the reparameterize method before calling
                  if hasattr(module, 'reparameterize') and callable(module.reparameterize):
                       # print(f" Calling reparameterize on: {name} ({type(module).__name__})")
                       module.reparameterize()
             # No need to explicitly recurse into Sequential, named_modules handles it.
        print("Reparameterization complete.")


# Factory functions similar to MobileNetV4
def GhostNet_1_0(width=1.0, **kwargs):
    """ Constructs a GhostNet 1.0 model with specified width """
    # Pass any extra kwargs if needed by GhostNet init in the future
    return GhostNet(model_name="GhostNet1.0", width=width)

# Example: Define a 0.5 width variant
# def GhostNet_0_5(**kwargs):
#    """ Constructs a GhostNet 0.5 model """
#    # You might define a "GhostNet0.5" entry in GHOSTNET_SPECS
#    # or just use the width multiplier with the 1.0 spec
#    return GhostNet(model_name="GhostNet1.0", width=0.5)


if __name__=='__main__':
    # --- Test Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model instance using factory function
    model = GhostNet_1_0(width=1.0).to(device)
    model.eval() # Start in eval mode

    print(f"Model Width List: {model.width_list}") # Print calculated widths

    # Dummy Inputs
    input1 = torch.randn(4, 3, 224, 224).to(device) # Standard size
    input2 = torch.randn(4, 3, 256, 320).to(device) # Different aspect ratio

    # --- Test before reparameterization ---
    print("\n--- Testing before reparameterization ---")
    with torch.inference_mode(): # Use inference_mode for eval
        y1_list = model(input1)
        y2_list = model(input2)
    print("Output shapes (Input 1):", [y.shape for y in y1_list])
    print("Output shapes (Input 2):", [y.shape for y in y2_list])

    # --- Test after reparameterization ---
    print("\n--- Testing after reparameterization ---")
    model.reparameterize()
    # print(model) # Print model structure after reparameterization (optional)
    model.eval() # Ensure model is in eval mode after reparam

    with torch.inference_mode():
        y1_reparam_list = model(input1)
        y2_reparam_list = model(input2)
    print("Reparam Output shapes (Input 1):", [y.shape for y in y1_reparam_list])
    print("Reparam Output shapes (Input 2):", [y.shape for y in y2_reparam_list])

    # --- Verification ---
    print("\n--- Verification ---")
    all_close1 = True
    total_norm1 = 0
    if len(y1_list) == len(y1_reparam_list):
        for i in range(len(y1_list)):
            is_close = torch.allclose(y1_list[i], y1_reparam_list[i], atol=1e-5) # Adjust tolerance if needed
            norm_diff = torch.norm(y1_list[i] - y1_reparam_list[i])
            print(f"Input 1 - Feature {i}: Allclose={is_close}, Norm Diff={norm_diff.item():.6f}")
            if not is_close: all_close1 = False
            total_norm1 += norm_diff.item()
        print(f"Input 1 - Overall Allclose: {all_close1}, Total Norm Diff: {total_norm1:.6f}")
    else:
        print("Input 1 - Error: Output list lengths differ after reparameterization.")

    all_close2 = True
    total_norm2 = 0
    if len(y2_list) == len(y2_reparam_list):
         for i in range(len(y2_list)):
            is_close = torch.allclose(y2_list[i], y2_reparam_list[i], atol=1e-5)
            norm_diff = torch.norm(y2_list[i] - y2_reparam_list[i])
            print(f"Input 2 - Feature {i}: Allclose={is_close}, Norm Diff={norm_diff.item():.6f}")
            if not is_close: all_close2 = False
            total_norm2 += norm_diff.item()
         print(f"Input 2 - Overall Allclose: {all_close2}, Total Norm Diff: {total_norm2:.6f}")
    else:
         print("Input 2 - Error: Output list lengths differ after reparameterization.")