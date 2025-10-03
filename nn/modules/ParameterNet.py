import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple
from functools import partial

# Helper functions (from Code 2 and adapted)
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
    if a == 0: return b
    if b == 0: return a
    a = abs(a)
    b = abs(b)
    if a<b:
        a,b=b,a
    while(a%b != 0):
        c = a%b
        a=b
        b=c
    return b

# Modules (Adapted from Code 1 and Code 2)
class SqueezeExcite(nn.Module):
    # Using Code 2's SqueezeExcite implementation style
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        # Original SE in timm uses relu here, Code 1 _SE_LAYER uses hard_sigmoid.
        # Let's stick to Code 2's SE structure which aligns with MobileNet/EfficientNet SE.
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

# Define the SE layer based on the above SqueezeExcite
_SE_LAYER = partial(SqueezeExcite, gate_fn=hard_sigmoid, divisor=4) # As used in Code 1 GhostBottleneck

class ConvBnAct(nn.Module):
    # Simplified ConvBnAct similar to Code 2, removing skip/drop_path from Code 1's version
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
    # Adapted from Code 2's GhostModule with Reparam logic
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, act_layer=nn.ReLU, mode='ori'): # Added mode, act_layer
        super(GhostModule, self).__init__()
        self.mode = mode # Should be 'ori' for standard GhostNet
        self.oup = oup
        self.stride = stride
        self.kernel_size_primary = kernel_size
        self.dw_size = dw_size # store dw_size

        self.infer_mode = False
        self.num_conv_branches = 3 # Number of parallel conv branches during training
        self.dconv_scale = True # Use scale branch

        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # Handle potential edge case where oup < ratio
        if new_channels < 0 :
            new_channels = 0
            init_channels = oup

        # Ensure total channels cover oup (can happen if ratio=2 and oup is odd)
        if init_channels + new_channels < oup:
           # print(f"Adjusting channels: init={init_channels}, new={new_channels}, oup={oup}")
           init_channels = oup - new_channels

        if init_channels <= 0:
             raise ValueError(f"GhostModule: Invalid channel calculation. inp={inp}, oup={oup}, ratio={ratio} -> init={init_channels}, new={new_channels}")

        self.init_channels = init_channels # Store for reparam
        self.new_channels = new_channels   # Store for reparam

        # --- Inference Path Modules ---
        # These will be populated/overwritten during reparameterize()
        self.primary_conv_module = nn.Sequential(
            nn.Conv2d(inp, self.init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(self.init_channels),
            act_layer(inplace=True) if act_layer is not None else nn.Identity(),
        )

        self.cheap_operation_module = None
        if self.new_channels > 0:
            # Ensure groups are valid for Conv2d
            cheap_groups = self.init_channels
            if cheap_groups <= 0: cheap_groups = 1 # Should not happen if init_channels > 0
            elif self.init_channels % cheap_groups != 0 or self.new_channels % cheap_groups != 0:
                cheap_groups = gcd(self.init_channels, self.new_channels)
                if cheap_groups == 0: cheap_groups = 1 # Fallback

            self.cheap_operation_module = nn.Sequential(
                nn.Conv2d(self.init_channels, self.new_channels, dw_size, 1, dw_size//2, groups=cheap_groups, bias=False),
                nn.BatchNorm2d(self.new_channels),
                act_layer(inplace=True) if act_layer is not None else nn.Identity(),
            )
            self.groups_cheap = cheap_groups # Store for reparam
            self.kernel_size_cheap = dw_size
        else:
             self.groups_cheap = 0
             self.kernel_size_cheap = 0


        # --- Training Path Branches ---
        self.primary_activation = act_layer(inplace=True) if act_layer is not None else nn.Identity()
        self.cheap_activation = act_layer(inplace=True) if act_layer is not None else nn.Identity()

        # Primary conv branches
        self.primary_rpr_skip = nn.BatchNorm2d(inp) if inp == self.init_channels and stride == 1 else None
        primary_rpr_conv = list()
        for _ in range(self.num_conv_branches):
            primary_rpr_conv.append(self._conv_bn(inp, self.init_channels, kernel_size, stride, kernel_size//2, groups=1, bias=False)) # groups=1 for primary conv
        self.primary_rpr_conv = nn.ModuleList(primary_rpr_conv)
        self.primary_rpr_scale = None
        if kernel_size > 1:
            self.primary_rpr_scale = self._conv_bn(inp, self.init_channels, 1, stride, 0, groups=1, bias=False) # Match stride

        # Cheap operation branches (only if new_channels > 0)
        self.cheap_rpr_skip = None
        self.cheap_rpr_conv = None
        self.cheap_rpr_scale = None
        if self.cheap_operation_module is not None and self.new_channels > 0:
             # Skip connection possible if input/output channels match and stride=1 (cheap op is always stride 1)
             if self.init_channels == self.new_channels:
                 self.cheap_rpr_skip = nn.BatchNorm2d(self.init_channels)

             cheap_rpr_conv = list()
             for _ in range(self.num_conv_branches):
                 cheap_rpr_conv.append(self._conv_bn(self.init_channels, self.new_channels, dw_size, 1, dw_size//2, groups=self.groups_cheap, bias=False))
             self.cheap_rpr_conv = nn.ModuleList(cheap_rpr_conv)

             if dw_size > 1:
                  self.cheap_rpr_scale = self._conv_bn(self.init_channels, self.new_channels, 1, 1, 0, groups=self.groups_cheap, bias=False)

        # Store info needed for reparameterization id_tensor creation
        self.in_channels_primary = inp
        self.groups_primary = 1 # groups=1 for primary conv

        # Check if cheap operation exists before accessing its attributes
        if self.cheap_operation_module is not None:
             self.in_channels_cheap = self.init_channels
             # groups_cheap and kernel_size_cheap are already stored above
        else:
            self.in_channels_cheap = 0
            self.groups_cheap = 0
            self.kernel_size_cheap = 0


    def forward(self, x):
        if self.infer_mode:
            # Inference mode uses fused modules
            x1 = self.primary_conv_module(x)
            if self.cheap_operation_module is not None:
                 x2 = self.cheap_operation_module(x1)
                 # Check if concat is needed or possible
                 if x1 is not None and x2 is not None:
                     out = torch.cat([x1, x2], dim=1)
                 elif x1 is not None:
                     out = x1 # Should only happen if new_channels was 0
                 else:
                     raise ValueError("Primary conv output (x1) is None in inference mode") # Should not happen
            else:
                 out = x1 # No cheap operation

            # Truncate to the desired output dimension
            return out[:, :self.oup, :, :]

        else:
            # Training mode uses reparameterization branches
            # Primary Conv Path
            identity_out_p = 0
            if self.primary_rpr_skip is not None:
                identity_out_p = self.primary_rpr_skip(x)
            scale_out_p = 0
            if self.primary_rpr_scale is not None and self.dconv_scale:
                scale_out_p = self.primary_rpr_scale(x)
            x1_train = scale_out_p + identity_out_p
            if self.primary_rpr_conv is not None: # Check if list exists
                 for ix in range(self.num_conv_branches):
                     x1_train += self.primary_rpr_conv[ix](x)
            x1_train = self.primary_activation(x1_train)

            # Cheap Operation Path (if exists)
            if self.cheap_operation_module is not None and self.cheap_rpr_conv is not None and self.new_channels > 0:
                identity_out_c = 0
                if self.cheap_rpr_skip is not None:
                    identity_out_c = self.cheap_rpr_skip(x1_train) # Input is x1_train
                scale_out_c = 0
                if self.cheap_rpr_scale is not None and self.dconv_scale:
                    scale_out_c = self.cheap_rpr_scale(x1_train)
                x2_train = scale_out_c + identity_out_c
                for ix in range(self.num_conv_branches):
                    x2_train += self.cheap_rpr_conv[ix](x1_train)
                x2_train = self.cheap_activation(x2_train)

                # Concatenate
                if x1_train is not None and x2_train is not None:
                    out_train = torch.cat([x1_train, x2_train], dim=1)
                elif x1_train is not None:
                    out_train = x1_train
                else:
                    raise ValueError("Primary conv output (x1_train) is None in training mode")
            else:
                # No cheap operation branches
                out_train = x1_train

            # Truncate to the desired output dimension
            return out_train[:, :self.oup, :, :]


    def reparameterize(self):
        if self.infer_mode:
            return

        # --- Reparameterize Primary Conv ---
        if hasattr(self, 'primary_rpr_conv') and self.primary_rpr_conv is not None and len(self.primary_rpr_conv) > 0:
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
                bias=True) # Bias is now True
            fused_primary_conv.weight.data = primary_kernel
            fused_primary_conv.bias.data = primary_bias
            # Replace the sequential module with the fused conv + activation
            # Need to handle case where primary_activation might be Identity
            if isinstance(self.primary_activation, nn.Identity):
                 self.primary_conv_module = fused_primary_conv
            else:
                 self.primary_conv_module = nn.Sequential(fused_primary_conv, self.primary_activation)

        # --- Reparameterize Cheap Operation (if exists) ---
        if hasattr(self, 'cheap_rpr_conv') and self.cheap_rpr_conv is not None and len(self.cheap_rpr_conv) > 0:
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
                bias=True) # Bias is now True
            fused_cheap_conv.weight.data = cheap_kernel
            fused_cheap_conv.bias.data = cheap_bias
            # Replace the sequential module with the fused conv + activation
            if isinstance(self.cheap_activation, nn.Identity):
                 self.cheap_operation_module = fused_cheap_conv
            else:
                 self.cheap_operation_module = nn.Sequential(fused_cheap_conv, self.cheap_activation)
        elif not hasattr(self, 'cheap_rpr_conv') or self.cheap_rpr_conv is None:
             # If no cheap reparam branches existed, ensure the original module is kept or set to None correctly
             if self.new_channels <= 0:
                 self.cheap_operation_module = None
             # else: keep the originally defined self.cheap_operation_module (pre-fusion logic)
             # This case might occur if dw_size=1 and init_channels != new_channels,
             # where no reparam branches might be created but the module exists.

        # --- Cleanup Training Branches ---
        # Detach all parameters first (safety measure)
        # for para in self.parameters():
        #    para.detach_()

        # Delete specific reparam attributes safely using delattr and hasattr
        if hasattr(self, 'primary_rpr_conv'): delattr(self, 'primary_rpr_conv')
        if hasattr(self, 'primary_rpr_scale'): delattr(self, 'primary_rpr_scale')
        if hasattr(self, 'primary_rpr_skip'): delattr(self, 'primary_rpr_skip')
        # Don't delete primary_activation, it's part of the fused module now

        if hasattr(self, 'cheap_rpr_conv'): delattr(self, 'cheap_rpr_conv')
        if hasattr(self, 'cheap_rpr_scale'): delattr(self, 'cheap_rpr_scale')
        if hasattr(self, 'cheap_rpr_skip'): delattr(self, 'cheap_rpr_skip')
        # Don't delete cheap_activation

        # Cleanup potentially created id_tensors
        if hasattr(self, 'id_tensor_primary'): delattr(self, 'id_tensor_primary')
        if hasattr(self, 'id_tensor_cheap'): delattr(self, 'id_tensor_cheap')

        self.infer_mode = True

    def _get_kernel_bias_primary(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel_scale, bias_scale = 0, 0
        if self.primary_rpr_scale is not None:
            # Ensure the scale branch exists and has conv/bn before fusing
            if hasattr(self.primary_rpr_scale, 'conv') and hasattr(self.primary_rpr_scale, 'bn'):
                 kernel_scale, bias_scale = self._fuse_bn_tensor(self.primary_rpr_scale, 'primary')
                 # Pad the 1x1 kernel from scale branch to match the main kernel size
                 pad = self.kernel_size_primary // 2
                 if pad > 0:
                      # kernel_scale shape: (out_ch, in_ch/groups, 1, 1) -> (out_ch, in_ch/groups, k, k)
                      kernel_scale = F.pad(kernel_scale, [pad] * 4) # Pad H and W dimensions
            else:
                 print("Warning: primary_rpr_scale missing conv or bn, skipping fusion.")


        kernel_identity, bias_identity = 0, 0
        if self.primary_rpr_skip is not None:
             # Ensure the skip branch exists (BatchNorm2d) before fusing
             if isinstance(self.primary_rpr_skip, nn.BatchNorm2d):
                 kernel_identity, bias_identity = self._fuse_bn_tensor(self.primary_rpr_skip, 'primary')
             else:
                 print("Warning: primary_rpr_skip is not BatchNorm2d, skipping fusion.")


        kernel_conv, bias_conv = 0, 0
        if hasattr(self, 'primary_rpr_conv') and self.primary_rpr_conv is not None:
             for ix in range(self.num_conv_branches):
                 # Ensure the conv branch exists and has conv/bn before fusing
                 if len(self.primary_rpr_conv) > ix and hasattr(self.primary_rpr_conv[ix], 'conv') and hasattr(self.primary_rpr_conv[ix], 'bn'):
                     _kernel, _bias = self._fuse_bn_tensor(self.primary_rpr_conv[ix], 'primary')
                     kernel_conv += _kernel
                     bias_conv += _bias
                 else:
                      print(f"Warning: primary_rpr_conv[{ix}] missing or invalid, skipping fusion.")

        return kernel_conv + kernel_scale + kernel_identity, bias_conv + bias_scale + bias_identity

    def _get_kernel_bias_cheap(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Only proceed if cheap operation actually exists and has branches
        if not hasattr(self, 'cheap_rpr_conv') or self.cheap_rpr_conv is None:
            return 0, 0 # Return zero tensors if no cheap branches to fuse

        kernel_scale, bias_scale = 0, 0
        if self.cheap_rpr_scale is not None:
            if hasattr(self.cheap_rpr_scale, 'conv') and hasattr(self.cheap_rpr_scale, 'bn'):
                 kernel_scale, bias_scale = self._fuse_bn_tensor(self.cheap_rpr_scale, 'cheap')
                 pad = self.kernel_size_cheap // 2
                 if pad > 0:
                     kernel_scale = F.pad(kernel_scale, [pad] * 4)
            else:
                 print("Warning: cheap_rpr_scale missing conv or bn, skipping fusion.")


        kernel_identity, bias_identity = 0, 0
        if self.cheap_rpr_skip is not None:
             if isinstance(self.cheap_rpr_skip, nn.BatchNorm2d):
                 kernel_identity, bias_identity = self._fuse_bn_tensor(self.cheap_rpr_skip, 'cheap')
             else:
                  print("Warning: cheap_rpr_skip is not BatchNorm2d, skipping fusion.")


        kernel_conv, bias_conv = 0, 0
        # Check if cheap_rpr_conv exists and is iterable
        if hasattr(self, 'cheap_rpr_conv') and self.cheap_rpr_conv is not None:
             for ix in range(self.num_conv_branches):
                # Check if the specific conv module exists
                if len(self.cheap_rpr_conv) > ix and hasattr(self.cheap_rpr_conv[ix], 'conv') and hasattr(self.cheap_rpr_conv[ix], 'bn'):
                     _kernel, _bias = self._fuse_bn_tensor(self.cheap_rpr_conv[ix], 'cheap')
                     kernel_conv += _kernel
                     bias_conv += _bias
                else:
                     print(f"Warning: cheap_rpr_conv[{ix}] missing or invalid, skipping fusion.")


        return kernel_conv + kernel_scale + kernel_identity, bias_conv + bias_scale + bias_identity

    def _fuse_bn_tensor(self, branch, branch_type) -> Tuple[torch.Tensor, torch.Tensor]:
         # Select attributes based on branch type
        if branch_type == 'primary':
            in_channels = self.in_channels_primary
            groups = self.groups_primary
            # Kernel size for identity/scale needs to match the main conv branch's target size
            kernel_size = self.kernel_size_primary
            id_tensor_name = 'id_tensor_primary'
        elif branch_type == 'cheap':
            # Ensure cheap attributes are valid before using
            if self.in_channels_cheap <= 0 or self.groups_cheap <= 0 or self.kernel_size_cheap <= 0:
                 # This can happen if cheap_operation_module was None
                 # print(f"Warning: Invalid cheap params for fusion ({self.in_channels_cheap}, {self.groups_cheap}, {self.kernel_size_cheap}), returning zero tensors.")
                 # Need to return tensors of correct dtype/device but zero value and appropriate shape?
                 # This case should ideally be handled by _get_kernel_bias_cheap returning 0,0 earlier.
                 # If called directly, need a fallback. Let's assume the caller checked.
                 pass # Continue, assuming attributes were set if this function is called.

            in_channels = self.in_channels_cheap
            groups = self.groups_cheap
            kernel_size = self.kernel_size_cheap
            id_tensor_name = 'id_tensor_cheap'
        else:
            raise ValueError(f"Invalid branch_type '{branch_type}' for _fuse_bn_tensor")

        # --- Get Kernel and BN parameters ---
        if isinstance(branch, nn.Sequential):
            # Make sure 'conv' and 'bn' attributes exist
            if not hasattr(branch, 'conv') or not hasattr(branch, 'bn'):
                 raise AttributeError(f"Branch Sequential module missing 'conv' or 'bn': {branch}")
            kernel = branch.conv.weight
            bn_module = branch.bn
            # kernel_size = branch.conv.kernel_size[0] # Get k from the conv itself for non-identity branches
        elif isinstance(branch, nn.BatchNorm2d):
             # Create identity kernel if it doesn't exist for this branch type
             if not hasattr(self, id_tensor_name):
                 # Determine output channels for identity - should match bn_module.num_features
                 out_channels = branch.num_features
                 # Identity kernel: needs shape (out_channels, in_channels // groups, k, k)
                 # For BN identity, input channels == output channels.
                 # Groups: How is the identity mapped? Usually group=1 for identity on dense,
                 # but for DW-like structures (cheap op), maybe groups = in_channels?
                 # Code 2 logic suggests groups=in_channels for DW identity kernel. Let's use that.
                 identity_groups = out_channels # Assume DW-like identity mapping
                 identity_kernel_size = kernel_size # Use the target kernel size passed in

                 if identity_groups <= 0:
                      raise ValueError(f"Identity groups must be positive, got {identity_groups}")
                 # For identity, in_channels = out_channels
                 if out_channels % identity_groups != 0:
                      # This happens if groups=out_channels, unless out_channels=1? Fallback?
                      print(f"Warning: Cannot create identity kernel, out_channels {out_channels} not divisible by groups {identity_groups}. Using groups=1.")
                      identity_groups = 1
                      # raise ValueError(f"in_channels {out_channels} not divisible by groups {identity_groups} for identity kernel")

                 input_dim_per_group = out_channels // identity_groups

                 # Kernel shape: (out_channels, input_dim_per_group, k, k)
                 kernel_value = torch.zeros((out_channels, input_dim_per_group, identity_kernel_size, identity_kernel_size),
                                            dtype=branch.weight.dtype,
                                            device=branch.weight.device)

                 # Fill the center element for identity mapping
                 center = identity_kernel_size // 2
                 for i in range(out_channels):
                      # Input channel index for group 'g' = i % input_dim_per_group
                      # Group index = i // input_dim_per_group
                      # Overall input channel index = group_index * input_dim_per_group + (i % input_dim_per_group) => simplifies to 'i'
                      # For DW (groups=out_channels), input_dim_per_group=1. Index is always 0.
                     kernel_value[i, 0, center, center] = 1 # DW identity: shape(out, 1, k, k), fill [i, 0, center, center]

                 setattr(self, id_tensor_name, kernel_value)
                 # print(f"Created {id_tensor_name} with shape: {kernel_value.shape}")


             kernel = getattr(self, id_tensor_name)
             # Pad the identity kernel if its inherent size (usually 1x1 effective) is smaller than target kernel size
             # Example: target k=3, identity is effectively 1x1 centered. Pad to 3x3.
             # This padding is handled *inside* _get_kernel_bias_* now. The id_tensor has the target size.
             bn_module = branch
             # kernel_size = kernel.shape[-1] # Use size from created id_tensor
        else:
             raise TypeError(f"Unsupported branch type for fusing: {type(branch)}")

        # --- Fuse BN parameters ---
        if isinstance(bn_module, nn.BatchNorm2d):
            running_mean = bn_module.running_mean
            running_var = bn_module.running_var
            gamma = bn_module.weight
            beta = bn_module.bias
            eps = bn_module.eps
        else:
            # Should not happen if checks above pass
             raise TypeError(f"bn_module is not BatchNorm2d: {type(bn_module)}")


        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1) # Reshape gamma/std for broadcasting

        # Kernel shape: (out_channels, in_channels // groups, k, k)
        # t shape: (out_channels, 1, 1, 1)
        fused_kernel = kernel * t

        # Bias shape: (out_channels)
        fused_bias = beta - running_mean * gamma / std

        return fused_kernel, fused_bias


    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        """ Helper method to construct conv-batchnorm layers. """
        mod_list = nn.Sequential()
         # Ensure groups is valid
        if groups <= 0:
            # print(f"Warning: groups must be positive, got {groups}. Setting groups=1.")
            groups = 1 # Fallback to groups=1
        if in_channels % groups != 0:
             # Try greatest common divisor? Or just use 1?
             new_groups = gcd(in_channels, groups)
             # print(f"Warning: in_channels {in_channels} not divisible by groups {groups}. Trying gcd: {new_groups}.")
             if new_groups > 0 and in_channels % new_groups == 0:
                 groups = new_groups
             else:
                 # print("Warning: Falling back to groups=1.")
                 groups = 1
        if out_channels % groups != 0:
             new_groups = gcd(out_channels, groups)
             # print(f"Warning: out_channels {out_channels} not divisible by groups {groups}. Trying gcd: {new_groups}.")
             if new_groups > 0 and out_channels % new_groups == 0:
                 groups = new_groups
             else:
                # print("Warning: Falling back to groups=1.")
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

class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE and Reparameterization"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.): # Removed drop_path, num_experts, layer_id
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        self.in_chs = in_chs # Store for shortcut check
        self.out_chs = out_chs # Store for shortcut check

        self.infer_mode = False
        self.num_conv_branches = 3 # Match GhostModule
        self.dconv_scale = True    # Match GhostModule

        # Point-wise expansion (GhostModule 1)
        # Use 'ori' mode as standard GhostNet doesn't have the complex shortcut from Code 2's example
        self.ghost1 = GhostModule(in_chs, mid_chs, act_layer=act_layer, mode='ori', ratio=2, stride=1) # Primary conv is 1x1, stride=1 here

        # Depth-wise convolution (if stride > 1)
        self.conv_dw = None
        self.bn_dw = None # BN separate in inference mode initially
        self.dw_rpr_skip = None
        self.dw_rpr_conv = None
        self.dw_rpr_scale = None
        self.activation_dw = nn.Identity() # No activation after DW usually needed before SE/Ghost2

        if self.stride > 1:
            # --- Inference Path DW ---
            # Note: These will be populated/overwritten by reparameterize()
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                 padding=(dw_kernel_size-1)//2,
                                 groups=mid_chs, bias=False) # bias=False initially
            self.bn_dw = nn.BatchNorm2d(mid_chs)

            # --- Training Path DW ---
            self.dw_rpr_skip = None # No skip branch for strided DW conv

            dw_rpr_conv = list()
            # Ensure mid_chs is valid for groups
            dw_groups = mid_chs
            if dw_groups <= 0: dw_groups = 1

            for _ in range(self.num_conv_branches):
                dw_rpr_conv.append(self._conv_bn(mid_chs, mid_chs, dw_kernel_size, stride, (dw_kernel_size-1)//2, groups=dw_groups, bias=False))
            self.dw_rpr_conv = nn.ModuleList(dw_rpr_conv)

            self.dw_rpr_scale = None
            if dw_kernel_size > 1:
                 self.dw_rpr_scale = self._conv_bn(mid_chs, mid_chs, 1, stride, 0, groups=dw_groups, bias=False) # Scale branch is 1x1

            # Store info needed for reparameterization of DW conv
            self.kernel_size_dw = dw_kernel_size
            self.in_channels_dw = mid_chs
            self.groups_dw = dw_groups # groups = in_channels for depthwise

        # Squeeze-and-excitation
        # Use the _SE_LAYER partial defined earlier, matching Code 1's usage
        self.se = _SE_LAYER(mid_chs, se_ratio=se_ratio, act_layer=act_layer) if has_se else None

        # Point-wise linear projection (GhostModule 2)
        # Output activation is usually off here (relu=False / act_layer=None)
        self.ghost2 = GhostModule(mid_chs, out_chs, act_layer=None, mode='ori', ratio=2, stride=1) # Stride is 1 here

        # Shortcut connection
        self.has_shortcut = (in_chs == out_chs and self.stride == 1)
        if self.has_shortcut:
            self.shortcut = nn.Identity()
        else:
            # Replicate Code 1's shortcut structure: DW -> BN -> PW -> BN
            # Ensure groups are valid for DW part of shortcut
            sc_dw_groups = in_chs
            if sc_dw_groups <=0 : sc_dw_groups = 1

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride, # Stride applied in DW conv
                       padding=(dw_kernel_size-1)//2, groups=sc_dw_groups, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False), # PW conv is 1x1, stride 1
                nn.BatchNorm2d(out_chs),
            )

        # Removed DropPath from Code 1

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            if self.infer_mode:
                 if self.conv_dw is None: raise RuntimeError("conv_dw is None during inference mode with stride > 1")
                 x = self.conv_dw(x)
                 # self.bn_dw is Identity after reparam, fused into conv_dw
                 # If not reparameterized yet, bn_dw should be applied (but logic assumes reparam before inference)
                 # Let's assume reparam happened if infer_mode is True.
                 # x = self.bn_dw(x) # BN is fused into conv_dw in infer_mode
            else:
                # Training mode DW conv (Reparameterized branches)
                identity_out_dw = 0 # dw_rpr_skip is None for stride > 1
                # if self.dw_rpr_skip is not None: identity_out_dw = self.dw_rpr_skip(x)
                scale_out_dw = 0
                if self.dw_rpr_scale is not None and self.dconv_scale:
                    scale_out_dw = self.dw_rpr_scale(x)

                x_dw = scale_out_dw + identity_out_dw # Start with scale/identity
                if self.dw_rpr_conv is not None: # Check if list exists
                    for ix in range(self.num_conv_branches):
                         if len(self.dw_rpr_conv) > ix: # Check index validity
                            x_dw += self.dw_rpr_conv[ix](x)
                else:
                     # Should not happen if stride > 1, but safeguard
                     print("Warning: dw_rpr_conv is None during training with stride > 1")
                     # Fallback? Or let it error? Let's assume branches exist if stride > 1.

                x = self.activation_dw(x_dw) # Apply DW activation (Identity by default)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        # Shortcut connection
        if self.has_shortcut:
            x += residual
        else:
            x += self.shortcut(residual)

        return x

    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        """ Helper method identical to GhostModule._conv_bn """
        mod_list = nn.Sequential()
        if groups <= 0: groups = 1
        if in_channels == 0: # Avoid division by zero if called incorrectly
             print(f"Warning (_conv_bn): in_channels is 0. Cannot create Conv2d.")
             return mod_list # Return empty sequential

        if in_channels % groups != 0:
             new_groups = gcd(in_channels, groups)
             if new_groups > 0 and in_channels % new_groups == 0: groups = new_groups
             else: groups = 1
        if out_channels % groups != 0:
             new_groups = gcd(out_channels, groups)
             if new_groups > 0 and out_channels % new_groups == 0: groups = new_groups
             else: groups = 1

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
        if hasattr(self.ghost1, 'reparameterize'):
             self.ghost1.reparameterize()
        if hasattr(self.ghost2, 'reparameterize'):
             self.ghost2.reparameterize()

        # Reparameterize the optional Depthwise convolution layer
        if self.infer_mode or self.stride == 1: # Only reparam DW if stride > 1 and not already done
            self.infer_mode = True # Ensure infer_mode is set even if no DW reparam needed
            return

        # Proceed with DW reparameterization (only if stride > 1)
        if hasattr(self, 'dw_rpr_conv') and self.dw_rpr_conv is not None and len(self.dw_rpr_conv) > 0:
            dw_kernel, dw_bias = self._get_kernel_bias_dw()

            # Check if self.dw_rpr_conv is valid before accessing its attributes
            if self.dw_rpr_conv is None or len(self.dw_rpr_conv) == 0 or not hasattr(self.dw_rpr_conv[0], 'conv'):
                 # This should ideally not be reached if the outer check passed, but safety first
                 raise RuntimeError("Cannot reparameterize DW conv: training branches are missing or invalid.")

            # Recreate self.conv_dw with fused parameters
            fused_dw_conv = nn.Conv2d(in_channels=self.dw_rpr_conv[0].conv.in_channels,
                                          out_channels=self.dw_rpr_conv[0].conv.out_channels,
                                          kernel_size=self.dw_rpr_conv[0].conv.kernel_size,
                                          stride=self.dw_rpr_conv[0].conv.stride,
                                          padding=self.dw_rpr_conv[0].conv.padding,
                                          dilation=self.dw_rpr_conv[0].conv.dilation,
                                          groups=self.dw_rpr_conv[0].conv.groups,
                                          bias=True) # Bias is now True
            fused_dw_conv.weight.data = dw_kernel
            fused_dw_conv.bias.data = dw_bias

            # Replace conv_dw and set bn_dw to Identity
            self.conv_dw = fused_dw_conv
            self.bn_dw = nn.Identity() # BN is fused into conv_dw

            # --- Cleanup DW Training Branches ---
            # Detach parameters (maybe redundant, but safe)
            # for para in self.parameters(): para.detach_()
            if hasattr(self, 'dw_rpr_conv'): delattr(self, 'dw_rpr_conv')
            if hasattr(self, 'dw_rpr_scale'): delattr(self, 'dw_rpr_scale')
            if hasattr(self, 'dw_rpr_skip'): delattr(self, 'dw_rpr_skip') # Should be None anyway
            if hasattr(self, 'activation_dw'): delattr(self, 'activation_dw') # No activation needed after fused DW usually

            # Cleanup potentially created id_tensor for DW
            if hasattr(self, 'id_tensor_dw'): delattr(self, 'id_tensor_dw')
        else:
            # Case where stride > 1 but somehow no dw_rpr_conv exists (shouldn't happen with init logic)
            print(f"Warning: Stride is {self.stride} but no DW reparam branches found. Keeping original conv_dw/bn_dw.")
            # We might want to keep the original conv_dw and bn_dw if no fusion happens?
            # Or ensure init always creates branches if stride > 1. Let's assume init is correct.


        self.infer_mode = True # Mark bottleneck as reparameterized

    def _get_kernel_bias_dw(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias for DW conv. """
        kernel_scale, bias_scale = 0, 0
        if self.dw_rpr_scale is not None:
            if hasattr(self.dw_rpr_scale, 'conv') and hasattr(self.dw_rpr_scale, 'bn'):
                 kernel_scale, bias_scale = self._fuse_bn_tensor_dw(self.dw_rpr_scale)
                 # Pad the 1x1 kernel from scale branch
                 pad = self.kernel_size_dw // 2
                 if pad > 0:
                     kernel_scale = F.pad(kernel_scale, [pad] * 4)
            else:
                 print("Warning: dw_rpr_scale missing conv or bn, skipping fusion.")


        kernel_identity, bias_identity = 0, 0
        # dw_rpr_skip should be None for stride>1 DW, so this part should yield 0
        if self.dw_rpr_skip is not None:
             if isinstance(self.dw_rpr_skip, nn.BatchNorm2d):
                 kernel_identity, bias_identity = self._fuse_bn_tensor_dw(self.dw_rpr_skip)
             else:
                  print("Warning: dw_rpr_skip is not BatchNorm2d, skipping fusion.")


        kernel_conv, bias_conv = 0, 0
        if hasattr(self, 'dw_rpr_conv') and self.dw_rpr_conv is not None:
             for ix in range(self.num_conv_branches):
                 if len(self.dw_rpr_conv) > ix and hasattr(self.dw_rpr_conv[ix], 'conv') and hasattr(self.dw_rpr_conv[ix], 'bn'):
                     _kernel, _bias = self._fuse_bn_tensor_dw(self.dw_rpr_conv[ix])
                     kernel_conv += _kernel
                     bias_conv += _bias
                 else:
                     print(f"Warning: dw_rpr_conv[{ix}] missing or invalid, skipping fusion.")


        return kernel_conv + kernel_scale + kernel_identity, bias_conv + bias_scale + bias_identity

    def _fuse_bn_tensor_dw(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer specifically for the DW conv branches. """
        # Attributes needed for identity kernel creation if branch is BatchNorm2d
        in_channels = self.in_channels_dw # Input channels to the DW stage
        groups = self.groups_dw         # Groups for the DW stage (should == in_channels)
        kernel_size = self.kernel_size_dw # Kernel size for the DW stage
        id_tensor_name = 'id_tensor_dw'

        # --- Get Kernel and BN parameters ---
        if isinstance(branch, nn.Sequential):
            if not hasattr(branch, 'conv') or not hasattr(branch, 'bn'):
                 raise AttributeError(f"DW Branch Sequential module missing 'conv' or 'bn': {branch}")
            kernel = branch.conv.weight
            bn_module = branch.bn
        elif isinstance(branch, nn.BatchNorm2d):
             # Create identity kernel if it doesn't exist
             if not hasattr(self, id_tensor_name):
                  out_channels = bn_module.num_features # Should == in_channels for DW identity
                  identity_groups = groups             # Use DW groups
                  identity_kernel_size = kernel_size

                  if identity_groups <= 0: raise ValueError("DW Identity groups must be positive")
                  if out_channels % identity_groups != 0:
                      print(f"Warning: DW identity kernel - out_channels {out_channels} not divisible by groups {identity_groups}. Using groups=1.")
                      identity_groups = 1
                  input_dim_per_group = out_channels // identity_groups

                  kernel_value = torch.zeros((out_channels, input_dim_per_group, identity_kernel_size, identity_kernel_size),
                                             dtype=branch.weight.dtype, device=branch.weight.device)
                  center = identity_kernel_size // 2
                  for i in range(out_channels):
                       # For DW identity (groups=out_channels), input_dim_per_group=1. Index is always 0.
                       kernel_value[i, 0, center, center] = 1
                  setattr(self, id_tensor_name, kernel_value)
                  # print(f"Created {id_tensor_name} with shape: {kernel_value.shape}")

             kernel = getattr(self, id_tensor_name)
             bn_module = branch
        else:
             raise TypeError(f"Unsupported DW branch type for fusing: {type(branch)}")

        # --- Fuse BN parameters ---
        if isinstance(bn_module, nn.BatchNorm2d):
            running_mean = bn_module.running_mean
            running_var = bn_module.running_var
            gamma = bn_module.weight
            beta = bn_module.bias
            eps = bn_module.eps
        else:
             raise TypeError(f"DW bn_module is not BatchNorm2d: {type(bn_module)}")

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        fused_kernel = kernel * t
        fused_bias = beta - running_mean * gamma / std
        return fused_kernel, fused_bias


# Define Architecture Spec based on Code 1's parameternet_600m cfgs
PARAMETERTNET_SPECS = {
    "parameternet_600m": [
        # Format: List[List[List[k, exp_size_base, c_base, SE, s]]]
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[3, 120,  40, 0.25, 1],
         [3, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 200,  80, 0, 1],
         [3, 200,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[3, 960, 160, 0.25, 1],
         [3, 960, 160, 0.25, 1],
         [3, 960, 160, 0.25, 1],
         [3, 960, 160, 0.25, 1],
         [3, 960, 160, 0.25, 1]],
    ]
    # Add other variants if needed
}


class GhostNet_Reparam(nn.Module):
    # Adapted from Code 1's GhostNet and Code 2's GhostNet structure
    def __init__(self, model_name="parameternet_600m", width=1.0, in_chans=3, act_layer=nn.ReLU): # Using nn.ReLU to match Code 2 default GhostNet
        super(GhostNet_Reparam, self).__init__()

        if model_name not in PARAMETERTNET_SPECS:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(PARAMETERTNET_SPECS.keys())}")

        cfgs = PARAMETERTNET_SPECS[model_name]
        self.width = width
        self.act_layer = act_layer
        block = GhostBottleneck # Use the reparameterizable bottleneck

        # building first layer (stem) - Replace DynamicConv with Conv2d
        stem_chs = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(in_chans, stem_chs, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_chs)
        self.act1 = act_layer(inplace=True)
        input_channel = stem_chs

        # building inverted residual blocks (stages)
        self.stages = nn.ModuleList() # Use ModuleList to store stages
        self.stage_output_indices = [] # Store indices of stages ending with stride=2
        last_exp_size = 0 # Keep track of last exp_size for final conv

        stage_idx = 0
        current_stride = 2 # Initial stride after stem
        for stage_cfg_list in cfgs:
            stage_layers = []
            stage_has_stride = False
            for block_cfg in stage_cfg_list:
                 # Handle different lengths of block_cfg (some stages have multiple blocks defined in one sublist in Code 1)
                 if isinstance(block_cfg[0], int): # Single block definition: [k, exp, c, se, s]
                      k, exp_size_base, c_base, se_ratio, s = block_cfg
                      out_chs = _make_divisible(c_base * width, 4)
                      mid_chs = _make_divisible(exp_size_base * width, 4)
                      stage_layers.append(block(input_channel, mid_chs, out_chs, k, s,
                                                act_layer=act_layer, se_ratio=se_ratio))
                      input_channel = out_chs
                      last_exp_size = exp_size_base # Store the base expansion size
                      if s > 1:
                          stage_has_stride = True
                          current_stride *= s
                 else: # Multiple blocks definition: [[k, exp, c, se, s], [k, exp, c, se, s], ...]
                     # This structure appeared in Code 1's cfg, need to iterate
                     for sub_block_cfg in block_cfg:
                         k, exp_size_base, c_base, se_ratio, s = sub_block_cfg
                         out_chs = _make_divisible(c_base * width, 4)
                         mid_chs = _make_divisible(exp_size_base * width, 4)
                         stage_layers.append(block(input_channel, mid_chs, out_chs, k, s,
                                                   act_layer=act_layer, se_ratio=se_ratio))
                         input_channel = out_chs
                         last_exp_size = exp_size_base
                         if s > 1:
                             stage_has_stride = True
                             current_stride *= s

            self.stages.append(nn.Sequential(*stage_layers))
            if stage_has_stride:
                 self.stage_output_indices.append(stage_idx)
            stage_idx += 1


        # Add the final ConvBnAct layer (similar to Code 2's structure's conv_last)
        # Code 1 added ConvBnAct(prev_chs, out_chs, 1) where out_chs=make_divisible(exp_size*width,4)
        # Let's replicate that using the last seen exp_size_base
        output_channel_final_conv = _make_divisible(last_exp_size * width, 4)
        self.conv_last = ConvBnAct(input_channel, output_channel_final_conv, 1, act_layer=act_layer) # Kernel size 1
        self.final_stage_idx = stage_idx # Store index after last real stage

        # --- Calculate width_list ---
        # Perform a dummy forward pass to get intermediate feature map sizes
        # Similar to Code 2, but using _forward_features
        self.eval() # Set to eval mode for dummy pass (esp. for BN)
        with torch.no_grad():
             # Use a standard input size, e.g., 224x224
             # Adjust size if specific application requires different default (e.g., 640x640 for detection)
             # Let's use 224x224 as it's common for classification pretraining
             dummy_input = torch.randn(1, in_chans, 224, 224)
             features = self._forward_features(dummy_input, return_all=True) # Get all intermediate features
             # Select features based on stage_output_indices and the final conv output
             # Typically want features with strides 4, 8, 16, 32
             # Stride 4: Output of first stage with stride=2 (index in stage_output_indices[0])
             # Stride 8: Output of second stage with stride=2 (index in stage_output_indices[1])
             # Stride 16: Output of third stage with stride=2 (index in stage_output_indices[2])
             # Stride 32: Output of final conv layer (index self.final_stage_idx)
             selected_features = []
             if len(self.stage_output_indices) >= 1:
                 selected_features.append(features[self.stage_output_indices[0]]) # Stride 4 (after stage index 0 or 1 typically)
             if len(self.stage_output_indices) >= 2:
                  selected_features.append(features[self.stage_output_indices[1]]) # Stride 8
             if len(self.stage_output_indices) >= 3:
                   selected_features.append(features[self.stage_output_indices[2]]) # Stride 16
             # Always include the final conv output (last element in features list)
             selected_features.append(features[-1]) # Stride 32

             # Check if we got the expected number of features (usually 4 for FPN-like usage)
             if len(selected_features) < 4:
                 print(f"Warning: Expected at least 4 feature maps for width_list, got {len(selected_features)}. Using available features.")
                 # Fallback: use all features returned by _forward_features if selection fails
                 if not selected_features:
                      selected_features = features

             self.width_list = [f.size(1) for f in selected_features if f is not None]


        self.train() # Set back to train mode


    def _forward_features(self, x, return_all=False):
        # Helper function for forward pass, returns list of features
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Pass through stages and collect outputs
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)

        # Apply final conv layer
        x = self.conv_last(features[-1]) # Apply to output of last stage
        features.append(x) # Add final conv output to the list

        if return_all:
             return features # Return all stage outputs + final conv output
        else:
             # Return features selected during width_list calculation by default
             selected_features = []
             if len(self.stage_output_indices) >= 1 and len(features) > self.stage_output_indices[0]:
                 selected_features.append(features[self.stage_output_indices[0]])
             if len(self.stage_output_indices) >= 2 and len(features) > self.stage_output_indices[1]:
                  selected_features.append(features[self.stage_output_indices[1]])
             if len(self.stage_output_indices) >= 3 and len(features) > self.stage_output_indices[2]:
                   selected_features.append(features[self.stage_output_indices[2]])
             # Always include the final conv output (last element)
             if features: # Ensure features list is not empty
                 selected_features.append(features[-1])

             if not selected_features and features: # Fallback if indices are wrong
                 print("Warning: Failed to select features based on indices, returning final feature map only.")
                 return [features[-1]]
             elif not features:
                 print("Warning: No features generated by _forward_features.")
                 return []

             return selected_features


    def forward(self, x):
        # Main forward pass returns the selected feature maps
        return self._forward_features(x)

    def reparameterize(self):
        print(f"Reparameterizing {self.__class__.__name__}...")
        for name, module in self.named_modules():
             # Check if the module has the reparameterize method before calling
             # This will cover GhostModule and GhostBottleneck instances
             if hasattr(module, 'reparameterize') and callable(module.reparameterize):
                  # print(f" Calling reparameterize on: {name} ({type(module).__name__})")
                  # Check if it's the module itself, not a submodule already called
                  # (named_modules handles recursion, so only call on the target module type)
                  if isinstance(module, (GhostModule, GhostBottleneck)):
                      module.reparameterize()
        print("Reparameterization complete.")


# Factory function for the reparameterizable parameternet_600m
def parameternet_600m_reparam(width=1.9, pretrained=False, **kwargs): # Width 1.9 from original parameternet_600m factory
    """ Constructs a parameternet_600m model with Reparameterization support """
    if pretrained:
        print("Warning: Pretrained weights are not available for the reparameterized version.")
        # Here you could add logic to load weights into the non-reparameterized
        # version first, then reparameterize, but that's complex.
        # For now, just returning untrained model.

    # Use nn.Hardswish as activation like original parameternet_600m factory if needed
    # act_layer = nn.Hardswish
    # Or keep nn.ReLU to align with Code 2's base GhostNet
    act_layer = nn.ReLU

    # Pass width and activation layer to the constructor
    # Filter out kwargs not accepted by GhostNet_Reparam if necessary
    accepted_kwargs = {'in_chans'}
    final_kwargs = {k: v for k, v in kwargs.items() if k in accepted_kwargs}

    model = GhostNet_Reparam(model_name="parameternet_600m", width=width, act_layer=act_layer, **final_kwargs)
    return model


if __name__=='__main__':
    # --- Test Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model instance using factory function
    # Using default width=1.9 for parameternet_600m
    model = parameternet_600m_reparam(width=1.9).to(device)

    print(f"Model Width List: {model.width_list}") # Print calculated widths

    # Set to eval mode *after* width_list calculation (which uses eval internally)
    model.eval()

    # Dummy Inputs
    input1 = torch.randn(4, 3, 224, 224).to(device) # Standard size
    input2 = torch.randn(2, 3, 256, 320).to(device) # Different size and batch

    # --- Test before reparameterization ---
    print("\n--- Testing before reparameterization (eval mode) ---")
    model.eval() # Ensure eval mode
    with torch.no_grad(): # Use no_grad for inference testing
        y1_list_train = model(input1)
        y2_list_train = model(input2)
    print("Output shapes (Input 1, Train Mode):", [y.shape for y in y1_list_train])
    print("Output shapes (Input 2, Train Mode):", [y.shape for y in y2_list_train])

    # --- Test after reparameterization ---
    print("\n--- Testing after reparameterization ---")
    model.reparameterize()
    # print(model) # Print model structure after reparameterization (optional, can be very long)
    model.eval() # Ensure model is in eval mode after reparam

    with torch.no_grad():
        y1_list_reparam = model(input1)
        y2_list_reparam = model(input2)
    print("Reparam Output shapes (Input 1):", [y.shape for y in y1_list_reparam])
    print("Reparam Output shapes (Input 2):", [y.shape for y in y2_list_reparam])

    # --- Verification ---
    print("\n--- Verification (Comparing Train mode output vs Reparam mode output) ---")
    all_close1 = True
    total_norm1 = 0
    if len(y1_list_train) == len(y1_list_reparam):
        for i in range(len(y1_list_train)):
            if y1_list_train[i] is None or y1_list_reparam[i] is None:
                 print(f"Input 1 - Feature {i}: Skipped comparison due to None output.")
                 all_close1 = False
                 continue
            # Increase tolerance slightly for complex models and mixed precision potential
            is_close = torch.allclose(y1_list_train[i], y1_list_reparam[i], atol=1e-4, rtol=1e-3)
            norm_diff = torch.norm(y1_list_train[i].float() - y1_list_reparam[i].float()) # Use float for norm calculation stability
            print(f"Input 1 - Feature {i}: Allclose={is_close}, Norm Diff={norm_diff.item():.6f}")
            if not is_close: all_close1 = False
            total_norm1 += norm_diff.item()
        print(f"Input 1 - Overall Allclose: {all_close1}, Total Norm Diff: {total_norm1:.6f}")
    else:
        print(f"Input 1 - Error: Output list lengths differ after reparameterization ({len(y1_list_train)} vs {len(y1_list_reparam)}).")
        all_close1 = False

    all_close2 = True
    total_norm2 = 0
    if len(y2_list_train) == len(y2_list_reparam):
         for i in range(len(y2_list_train)):
            if y2_list_train[i] is None or y2_list_reparam[i] is None:
                 print(f"Input 2 - Feature {i}: Skipped comparison due to None output.")
                 all_close2 = False
                 continue
            is_close = torch.allclose(y2_list_train[i], y2_list_reparam[i], atol=1e-4, rtol=1e-3)
            norm_diff = torch.norm(y2_list_train[i].float() - y2_list_reparam[i].float())
            print(f"Input 2 - Feature {i}: Allclose={is_close}, Norm Diff={norm_diff.item():.6f}")
            if not is_close: all_close2 = False
            total_norm2 += norm_diff.item()
         print(f"Input 2 - Overall Allclose: {all_close2}, Total Norm Diff: {total_norm2:.6f}")
    else:
         print(f"Input 2 - Error: Output list lengths differ after reparameterization ({len(y2_list_train)} vs {len(y2_list_reparam)}).")
         all_close2 = False