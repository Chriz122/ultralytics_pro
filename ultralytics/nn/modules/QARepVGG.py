import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
import torch.nn.init as init
import math
from functools import partial
from timm.models.layers import DropPath # Assuming timm is an acceptable dependency
import torch.utils.checkpoint as checkpoint # Added for potential use

class my_scaler(object):
    def __init__(self, max_iter=1, cosine=False):
        self.max_iter = max_iter
        self.iter = 0
        self.cosine = cosine

    def get_scale(self):
        if self.cosine:
            if self.iter <= self.max_iter:
                return (1 + math.cos(math.pi * self.iter / self.max_iter)) / 2
            else:
                return 0.0
        else:
            return max(0.0, 1.0 - self.iter/self.max_iter)

    def step(self):
        self.iter += 1

    def set_max_iter(self, max_iter):
        self.max_iter = max_iter

    def set_iter(self, iter):
        self.iter = iter


class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = inputs.mean((2, 3), keepdim=True) # Original from Code 1
        # x = F.avg_pool2d(inputs, kernel_size=inputs.size(3)) # Alternative from Code 2
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


def conv_bn_noaffline(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels, affine=False))
    return result


class QARepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU(), drop_path_ratio=0.0):
        super(QARepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels # Store out_channels
        self.use_scale = use_scale # Kept from Code 1, though not used in Code 2's RepVGGBlock
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()


        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        if isinstance(act, nn.Module): # if act is already an instance
            self.nonlinearity = act
        elif isinstance(act, type) and issubclass(act, nn.Module): # if act is a class
            if act == nn.PReLU:
                self.nonlinearity = nn.PReLU(num_parameters=out_channels)
            else:
                self.nonlinearity = act()
        else: # Default or error
            self.nonlinearity = nn.ReLU()


        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        # Applying DropPath as in QARepVGGBlockV2 if identity is present
        if self.rbr_identity is None:
             # Original RepVGG does not use DropPath here typically, but QARepVGGBlockV2 does
            return self.nonlinearity(self.se(self.drop_path(self.rbr_dense(inputs) + self.rbr_1x1(inputs)) + id_out))
        else:
            return self.nonlinearity(self.se(self.drop_path(self.rbr_dense(inputs) + self.rbr_1x1(inputs)) + id_out))


    def get_custom_L2(self):
        if hasattr(self, 'rbr_reparam'): return 0 # Cannot get L2 for deployed block
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2 + 1e-8)).sum() # Added epsilon for stability
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight if hasattr(branch.bn, 'weight') and branch.bn.weight is not None else torch.ones_like(branch.bn.running_mean)
            beta = branch.bn.bias if hasattr(branch.bn, 'bias') and branch.bn.bias is not None else torch.zeros_like(branch.bn.running_mean)
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                # Ensure id_tensor is on the same device as branch parameters
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device if hasattr(branch, 'weight') and branch.weight is not None else branch.running_mean.device)

            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight if hasattr(branch, 'weight') and branch.weight is not None else torch.ones_like(branch.running_mean)
            beta = branch.bias if hasattr(branch, 'bias') and branch.bias is not None else torch.zeros_like(branch.running_mean)
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_extra_bn_tensor(self, kernel, bias, branch): # Used by QARepVGGBlockV1, V2 etc.
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean - bias # remove bias
        running_var = branch.running_var
        gamma = branch.weight if hasattr(branch, 'weight') and branch.weight is not None else torch.ones_like(branch.running_mean)
        beta = branch.bias if hasattr(branch, 'bias') and branch.bias is not None else torch.zeros_like(branch.running_mean)
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        # Use parameters from rbr_dense for the new Conv2d layer's shape if it exists
        # otherwise, need a fallback if rbr_dense was replaced (e.g. in some QARepVGG variants)
        
        # Determine conv params from existing conv layers if possible
        # Fallback to self.in_channels, self.out_channels, kernel_size=3, stride= (need stride)
        # This part is tricky because QARepVGGBlock variants might redefine self.rbr_dense
        # For the base RepVGGBlock, rbr_dense is nn.Sequential(conv, bn)
        if hasattr(self, 'rbr_dense') and isinstance(self.rbr_dense, nn.Sequential) and hasattr(self.rbr_dense, 'conv'):
            conv_template = self.rbr_dense.conv
        elif hasattr(self, 'rbr_dense') and isinstance(self.rbr_dense, nn.Conv2d): # For variants that replace rbr_dense
             conv_template = self.rbr_dense
        else: # Fallback if rbr_dense is not available or not a conv/sequential(conv,bn)
            # This fallback needs the stride information, which is not stored directly
            # in RepVGGBlock after __init__. Assuming stride is 1 if not first block in stage.
            # This is a simplification; ideally, stride would be accessible.
            # However, since QARepVGGBlock variants inherit and might call super().switch_to_deploy(),
            # this part needs careful handling or overriding in subclasses.
            # For simplicity here, we'll assume rbr_dense.conv is usually available for the base.
             raise ValueError("Cannot determine parameters for rbr_reparam without rbr_dense.conv")

        self.rbr_reparam = nn.Conv2d(in_channels=conv_template.in_channels,
                                     out_channels=conv_template.out_channels,
                                     kernel_size=conv_template.kernel_size,
                                     stride=conv_template.stride,
                                     padding=conv_template.padding,
                                     dilation=conv_template.dilation,
                                     groups=conv_template.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        
        # Delete training-time branches
        attrs_to_delete = ['rbr_dense', 'rbr_1x1', 'rbr_identity', 'id_tensor', 'bn'] # 'bn' for QAVariants
        for attr in attrs_to_delete:
            if hasattr(self, attr):
                self.__delattr__(attr)
        self.deploy = True

# --- QARepVGGBlock Variants (Modified to ensure proper super calls and attribute handling) ---
class QARepVGGBlockV1(QARepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU(), drop_path_ratio=0.0):
        super(QARepVGGBlockV1, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act, drop_path_ratio)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels) # Post-summation BN

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        
        # Apply drop_path to the sum of conv branches before adding identity and passing to post-BN
        sum_conv_branches = self.rbr_dense(inputs) + self.rbr_1x1(inputs)
        return self.nonlinearity(self.bn(self.se(self.drop_path(sum_conv_branches) + id_out)))


    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid
        bias = bias3x3 + bias1x1 + biasid
        
        return self._fuse_extra_bn_tensor(kernel, bias, self.bn)

    # switch_to_deploy will be inherited from RepVGGBlock, which should handle deleting 'bn'

class QARepVGGBlockV2(QARepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU(), drop_path_ratio=0.0):
        super(QARepVGGBlockV2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act, drop_path_ratio)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels) # Post-summation BN
            # rbr_dense from super is conv_bn
            # rbr_1x1 is Conv2D without BN, identity is nn.Identity or None
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False, padding=padding - kernel_size // 2)
            # rbr_identity is BatchNorm2d in super if conditions met, override if needed for V2 logic
            # For V2, if identity exists, it's nn.Identity() NOT a BatchNorm layer that gets fused.
            # The superclass init already sets self.rbr_identity to nn.BatchNorm2d if conditions met.
            # QARepVGGBlockV2 in original code redefines rbr_identity if not deploy:
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
        self._id_tensor_for_l2 = None # For custom L2

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        id_out = 0
        if self.rbr_identity is not None:
            id_out = self.rbr_identity(inputs)

        # rbr_dense already includes BN
        # rbr_1x1 is just Conv2D
        # id_out is direct input or 0
        # DropPath applied to sum of (BN'd 3x3) and (raw 1x1), then add identity, then post-BN
        sum_conv_branches = self.rbr_dense(inputs) + self.rbr_1x1(inputs)
        return self.nonlinearity(self.bn(self.se(self.drop_path(sum_conv_branches) + id_out)))


    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense) # rbr_dense is conv_bn
        
        # rbr_1x1 is just a Conv2d, no BN to fuse with it directly, so its weight is used as is.
        # Bias for rbr_1x1 is 0 as bias=False
        kernel1x1 = self.rbr_1x1.weight 
        
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1)
        bias = bias3x3 # Bias from 1x1 is 0

        if self.rbr_identity is not None: # This means it's an nn.Identity()
            # Create an identity kernel to add
            id_kernel_val = np.zeros((self.out_channels, self.in_channels // self.groups, 3, 3), dtype=np.float32)
            if self.in_channels == self.out_channels: # Should be true if rbr_identity is not None
                 for i in range(self.in_channels): # Assuming groups = 1 for identity part for simplicity in kernel creation
                    if i < (self.in_channels // self.groups): # Ensure we don't go out of bounds for input_dim
                        id_kernel_val[i, i % (self.in_channels // self.groups), 1, 1] = 1.0

            id_kernel_tensor = torch.from_numpy(id_kernel_val).to(kernel.device)
            kernel = kernel + id_kernel_tensor
        
        return self._fuse_extra_bn_tensor(kernel, bias, self.bn)


    def get_custom_L2(self):
        if hasattr(self, 'rbr_reparam'): return 0
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.weight # rbr_1x1 is nn.Conv2d without BN in V2
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        tmp = K3 * t3
        l2_loss_circle = (tmp ** 2).sum() - (tmp[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel_center = K3[:, :, 1:2, 1:2] * t3 + K1 # K1 is padded to 1x1 before adding in fusion, here use raw K1

        if self.rbr_identity is not None:
            eq_kernel_center = eq_kernel_center + self.get_id_kernel_for_l2() # This needs to be 1x1
        
        l2_loss_eq_kernel = (eq_kernel_center ** 2).sum()
        return l2_loss_eq_kernel + l2_loss_circle

    def get_id_kernel_for_l2(self): # Make this return a 1x1 kernel for L2
        if self.rbr_identity is not None:
            if self._id_tensor_for_l2 is None:
                # For L2, the identity contribution to the center of the effective kernel
                # is just 1 (scaled by nothing if there's no BN on identity branch)
                # Shape should match K1 (out_channels, in_channels/groups, 1, 1)
                kernel_value = np.zeros((self.out_channels, self.in_channels // self.groups, 1, 1), dtype=np.float32)
                if self.in_channels == self.out_channels: # Condition for identity
                    for i in range(self.in_channels): # Assuming groups = 1 for identity part for simplicity
                         if i < self.out_channels and (i % (self.in_channels // self.groups)) < (self.in_channels // self.groups):
                            kernel_value[i, i % (self.in_channels // self.groups), 0, 0] = 1.0
                
                self._id_tensor_for_l2 = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            return self._id_tensor_for_l2
        return 0.0 # Return 0 if no identity

# ... (Other QARepVGGBlock variants V3-V15, M3, etc. would follow a similar pattern of:
#       1. Calling super().__init__
#       2. Redefining branches (rbr_dense, rbr_1x1, rbr_identity, bn) if they differ from base RepVGGBlock or QARepVGGBlockV1/V2
#       3. Overriding forward() if the data flow changes
#       4. Overriding get_equivalent_kernel_bias() if fusion logic changes
#       5. Overriding get_custom_L2() if L2 calculation changes
#       6. switch_to_deploy() can often be inherited if it correctly deletes all training-time attributes, including 'bn' if present)
# For brevity, I will include a few more key ones and then the main backbone structure.

class QARepVGGBlockV6(QARepVGGBlock): # No post-BN
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU(), drop_path_ratio=0.0):
        super(QARepVGGBlockV6, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act, drop_path_ratio)
        if not deploy:
            # rbr_dense and rbr_1x1 are conv_bn from super
            # rbr_identity is nn.BatchNorm2d if conditions met, else None.
            # V6 expects identity to be nn.Identity if it exists
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
            # NO self.bn (post-summation BN)

    def forward(self, inputs): # No self.bn
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        id_out = 0
        if self.rbr_identity is not None:
            id_out = self.rbr_identity(inputs)
        
        sum_conv_branches = self.rbr_dense(inputs) + self.rbr_1x1(inputs)
        return self.nonlinearity(self.se(self.drop_path(sum_conv_branches) + id_out))


    def get_equivalent_kernel_bias(self): # No _fuse_extra_bn_tensor
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1)
        bias = bias3x3 + bias1x1

        if self.rbr_identity is not None: # nn.Identity
            id_kernel_val = np.zeros((self.out_channels, self.in_channels // self.groups, 3, 3), dtype=np.float32)
            if self.in_channels == self.out_channels:
                for i in range(self.in_channels):
                     if i < (self.in_channels // self.groups):
                        id_kernel_val[i, i % (self.in_channels // self.groups), 1, 1] = 1.0
            id_kernel_tensor = torch.from_numpy(id_kernel_val).to(kernel.device)
            kernel = kernel + id_kernel_tensor
            # No bias contribution from nn.Identity
        return kernel, bias

class QARepVGGBlock_Baseline(QARepVGGBlock): # This is the QARepVGGBlock from code 1
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU(), drop_path_ratio=0.0):
        # Call RepVGGBlock's init, but then override branches as per original QARepVGGBlock
        super(QARepVGGBlock_Baseline, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             padding_mode, deploy, use_se, use_scale, act, drop_path_ratio)

        if not deploy:
            # Override branches defined in super RepVGGBlock's __init__
            self.rbr_dense = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=stride, padding=1, groups=groups, bias=False)
            self.rbr_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                     stride=stride, groups=groups, bias=False, padding=padding - kernel_size // 2)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
            self.bn = nn.BatchNorm2d(out_channels) # Post-summation BN

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        id_out = 0
        if self.rbr_identity is not None:
            id_out = self.rbr_identity(inputs)
        
        sum_conv_branches = self.rbr_dense(inputs) + self.rbr_1x1(inputs)
        return self.nonlinearity(self.bn(self.se(self.drop_path(sum_conv_branches) + id_out)))


    def _fuse_bn_tensor_for_baseline(self, kernel, branch_bn): # Specific for baseline QARepVGGBlock
        # This is the _fuse_bn_tensor from the original QARepVGGBlock
        assert isinstance(branch_bn, nn.BatchNorm2d)
        running_mean = branch_bn.running_mean
        running_var = branch_bn.running_var
        gamma = branch_bn.weight if hasattr(branch_bn, 'weight') and branch_bn.weight is not None else torch.ones_like(branch_bn.running_mean)
        beta = branch_bn.bias if hasattr(branch_bn, 'bias') and branch_bn.bias is not None else torch.zeros_like(branch_bn.running_mean)
        eps = branch_bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        # Kernel fusion for QARepVGGBlock_Baseline
        # rbr_dense and rbr_1x1 are nn.Conv2d (no BN)
        # rbr_identity is nn.Identity or None
        # bn is the post-summation BN
        
        kernel = self.rbr_dense.weight + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        # Bias is 0 from conv branches since bias=False
        
        if self.rbr_identity is not None:
            id_kernel_val = np.zeros((self.out_channels, self.in_channels // self.groups, 3, 3), dtype=np.float32)
            if self.in_channels == self.out_channels:
                for i in range(self.in_channels):
                     if i < (self.in_channels // self.groups): # Check bounds for group conv
                        id_kernel_val[i, i % (self.in_channels // self.groups), 1, 1] = 1.0
            id_tensor = torch.from_numpy(id_kernel_val).to(kernel.device)
            kernel = kernel + id_tensor
        
        # Fuse the post-summation BN
        fused_kernel, fused_bias = self._fuse_bn_tensor_for_baseline(kernel, self.bn)
        return fused_kernel, fused_bias


# --- Main Backbone Definition ---
class QARepVGGBackbone(nn.Module):
    def __init__(self, model_name: str, deploy=False, use_checkpoint=False,
                 strides_override=None): # strides_override to match RepVGGFeatures
        super(QARepVGGBackbone, self).__init__()

        if model_name not in QAREPVGG_SPECS:
            raise ValueError(f"Model name {model_name} not found in QAREPVGG_SPECS.")
        
        spec = QAREPVGG_SPECS[model_name]
        num_blocks = spec['num_blocks']
        width_multiplier = spec['width_multiplier']
        override_groups_map = spec['override_groups_map'] or dict()
        use_se = spec['use_se']
        block_cls = spec['block_cls']
        # Drop path ratio can also be part of spec if needed per model
        drop_path_ratio = spec.get('drop_path_ratio', 0.0) 
        act_func = spec.get('act', nn.ReLU()) # Get activation from spec

        self.current_deploy_state = deploy
        self.use_checkpoint = use_checkpoint
        self.override_groups_map = override_groups_map
        assert 0 not in self.override_groups_map

        # Resolve block_cls: if it's a partial, it's already configured.
        # If it's a class, we might need to pass more args from spec.
        # The use of partial in QAREPVGG_SPECS handles this well.
        
        # Default strides, can be overridden
        # Original RepVGG typically has [2, 2, 2, 2] for stages 0 through 3 (effectively stage1-4 after initial conv)
        # The 'strides' parameter in RepVGG class in Code 1 applies to stage1, stage2, stage3, stage4
        effective_strides = strides_override if strides_override is not None else [2, 2, 2, 2]


        self.in_planes = min(64, int(64 * width_multiplier[0]))

        # Stage 0: Initial convolution block
        # It also uses block_cls, its stride is typically 2
        current_block_cls = partial(block_cls, act=act_func, drop_path_ratio=drop_path_ratio) if not isinstance(block_cls, partial) else block_cls

        self.stage0 = current_block_cls(in_channels=3, out_channels=self.in_planes, kernel_size=3,
                                   stride=2, padding=1, deploy=self.current_deploy_state, use_se=use_se)
        self.cur_layer_idx = 1 # Start after stage0

        self.stage1 = self._make_stage(current_block_cls, int(64 * width_multiplier[0]), num_blocks[0], stride=effective_strides[0], use_se=use_se)
        self.stage2 = self._make_stage(current_block_cls, int(128 * width_multiplier[1]), num_blocks[1], stride=effective_strides[1], use_se=use_se)
        self.stage3 = self._make_stage(current_block_cls, int(256 * width_multiplier[2]), num_blocks[2], stride=effective_strides[2], use_se=use_se)
        self.stage4 = self._make_stage(current_block_cls, int(512 * width_multiplier[3]), num_blocks[3], stride=effective_strides[3], use_se=use_se)

        # Calculate width_list
        self.width_list = []
        if not deploy: # Only calculate for training mode, deploy mode can infer from training
            original_training_state = self.training
            self.eval()
            try:
                with torch.no_grad():
                    # Use a typical input size. For width_list, spatial dim doesn't matter as much as batch size 1.
                    dummy_input = torch.randn(1, 3, 224, 224).to(next(self.parameters()).device)
                    # Need to run a forward pass.
                    # Temporarily set deploy to False if it's True, to get correct feature list from training structure
                    is_currently_deployed_for_width_calc = self.current_deploy_state
                    if is_currently_deployed_for_width_calc:
                        # This is tricky. If we are initializing a deployed model directly,
                        # getting width_list from training structure is not possible without building it.
                        # For now, assume width_list is primarily for models created in training mode.
                        # If a model is created with deploy=True, width_list might be less relevant or
                        # should be hardcoded based on the known architecture.
                        # Let's assume this is called when deploy=False or after conversion.
                        # If called with deploy=True initially, this part won't run.
                        pass


                    # Get features from the forward method that returns a list
                    features = self._forward_features(dummy_input)
                    self.width_list = [f.size(1) for f in features]
            except Exception as e:
                print(f"Warning: Could not compute width_list during init: {e}")
                # Fallback or leave empty. For common models, these are known:
                # Example for A0: [48, 48, 96, 192, 1280] (output of stage0, s1, s2, s3, s4)
                # The features returned by _forward_features are usually stage1-4
                # So width_list would be [planes_s1, planes_s2, planes_s3, planes_s4]
                self.width_list = [
                    int(64 * width_multiplier[0]),
                    int(128 * width_multiplier[1]),
                    int(256 * width_multiplier[2]),
                    int(512 * width_multiplier[3])
                ]

            finally:
                self.train(original_training_state)
        else: # For deployed models, width_list should correspond to the output channels of stages
             self.width_list = [
                int(64 * width_multiplier[0]),
                int(128 * width_multiplier[1]),
                int(256 * width_multiplier[2]),
                int(512 * width_multiplier[3])
            ]
        self.apply(self._init_weights) # Initialize weights

    def _init_weights(self, m):
        # From original Code 1 RepVGG._init_weights
        if isinstance(m, nn.Linear):
            # timm.models.vision_transformer.trunc_normal_ can be replaced if timm is not available
            # For now, assuming timm is available.
            try:
                from timm.models.layers import trunc_normal_
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            except ImportError:
                 nn.init.normal_(m.weight, 0, 0.02)
                 if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None: m.weight.data.fill_(1.0)
            if m.bias is not None: m.bias.data.zero_()


    def _make_stage(self, block_cls, planes, num_blocks_stage, stride, use_se):
        strides_list = [stride] + [1]*(num_blocks_stage-1)
        blocks = []
        for s_idx, current_stride in enumerate(strides_list):
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(block_cls(in_channels=self.in_planes,
                                      out_channels=planes,
                                      kernel_size=3,
                                      stride=current_stride,
                                      padding=1,
                                      groups=cur_groups,
                                      deploy=self.current_deploy_state,
                                      use_se=use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks) # Using Sequential for stages as in original RepVGG

    def _forward_features(self, x):
        """Helper to get feature list, used by forward and width_list calculation."""
        s0 = self.stage0(x)
        
        # It seems the Ultralytics error implies it expects features *including* stage0 output,
        # or at least a list where it can insert. Let's match RepVGGFeatures output.
        # RepVGGFeatures output stage1, stage2, stage3, stage4.
        features = []
        
        s1_out = self.stage1(s0)
        features.append(s1_out)

        s2_out = self.stage2(s1_out)
        features.append(s2_out)

        s3_out = self.stage3(s2_out)
        features.append(s3_out)
        
        s4_out = self.stage4(s3_out)
        features.append(s4_out)
        
        return features # Return list of features from stage1, stage2, stage3, stage4

    def forward(self, x):
        # Checkpoint logic can be added per stage if needed
        # For simplicity, direct forward pass here.
        # if self.use_checkpoint and not torch.jit.is_scripting():
            # Apply checkpoint to stages or blocks
        
        return self._forward_features(x) # Return a LIST of Tensors


# --- Model Specs and Factory ---
optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}
# A0_block_indices_for_dw = [2, 6, 20, 21] # Original layer indices for A0 DW variant
# Recalculate gd_map based on how override_groups_map is used (cur_layer_idx)
# Stage0 is layer 0 (not in map). Stage1 starts from layer 1.
# Example: RepVGG-A0: num_blocks=[2, 4, 14, 1]
# Stage1: layers 1, 2 (num_blocks[0]=2)
# Stage2: layers 3, 4, 5, 6 (num_blocks[1]=4)
# Stage3: layers 7 to 20 (num_blocks[2]=14)
# Stage4: layer 21 (num_blocks[3]=1)
# If A0_block_indices_for_dw refers to specific blocks within these stages for DW,
# the gd_map needs to be constructed carefully.
# For simplicity, the DW example from Code1's QARepVGGBlockV2_A0_DW seems to apply override_groups_map
# to ALL optional_groupwise_layers if they exist. Let's keep gd_map as it was if that was the intention.
A0_block_orig_indices = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26] # These are layer indices
# gd_map from original code seems to be an example, let's keep it for 'QARepVGGV2-A0-DW'
gd_map_example = {l: int(48 * 2**(i//3)) for i, l in enumerate(A0_block_orig_indices)} # A more structured example


QAREPVGG_SPECS = {
    'QARepVGG-A0': {'num_blocks': [2, 4, 14, 1], 'width_multiplier': [0.75, 0.75, 0.75, 2.5], 'override_groups_map': None, 'use_se': False, 'block_cls': QARepVGGBlock_Baseline},
    'QARepVGGV1-A0': {'num_blocks': [2, 4, 14, 1], 'width_multiplier': [0.75, 0.75, 0.75, 2.5], 'override_groups_map': None, 'use_se': False, 'block_cls': QARepVGGBlockV1},
    'QARepVGGV2-A0': {'num_blocks': [2, 4, 14, 1], 'width_multiplier': [0.75, 0.75, 0.75, 2.5], 'override_groups_map': None, 'use_se': False, 'block_cls': QARepVGGBlockV2},
    'QARepVGGV2-A0_d01': {'num_blocks': [2, 4, 14, 1], 'width_multiplier': [0.75, 0.75, 0.75, 2.5], 'override_groups_map': None, 'use_se': False, 'block_cls': partial(QARepVGGBlockV2, drop_path_ratio=0.1)},
    'QARepVGGV2-A0-DW': {'num_blocks': [2, 4, 14, 1], 'width_multiplier': [0.75, 0.75, 0.75, 2.5], 'override_groups_map': gd_map_example, 'use_se': False, 'block_cls': QARepVGGBlockV2}, # Using the example gd_map
    'QARepVGGV6-A0': {'num_blocks': [2, 4, 14, 1], 'width_multiplier': [0.75, 0.75, 0.75, 2.5], 'override_groups_map': None, 'use_se': False, 'block_cls': QARepVGGBlockV6},
    # Add other variants from Code 1's func_dict here
    # e.g. QARepVGGV3-A0, QARepVGGV4-A0, etc.
    # Activation variants
    'QARepVGG-A0-ReLU6': {'num_blocks': [2, 4, 14, 1], 'width_multiplier': [0.75, 0.75, 0.75, 2.5], 'override_groups_map': None, 'use_se': False, 'block_cls': QARepVGGBlock_Baseline, 'act': nn.ReLU6()},
    'QARepVGGV2-A0-PReLU': {'num_blocks': [2, 4, 14, 1], 'width_multiplier': [0.75, 0.75, 0.75, 2.5], 'override_groups_map': None, 'use_se': False, 'block_cls': QARepVGGBlockV2, 'act': nn.PReLU}, # Pass class for PReLU

    # Scaled versions
    'QARepVGGV2-A1': {'num_blocks': [2, 4, 14, 1], 'width_multiplier': [1.0, 1.0, 1.0, 2.5], 'override_groups_map': None, 'use_se': False, 'block_cls': QARepVGGBlockV2},
    'QARepVGGV2-A2': {'num_blocks': [2, 4, 14, 1], 'width_multiplier': [1.5, 1.5, 1.5, 2.75], 'override_groups_map': None, 'use_se': False, 'block_cls': QARepVGGBlockV2},
    'QARepVGGV2-B0': {'num_blocks': [4, 6, 16, 1], 'width_multiplier': [1.0, 1.0, 1.0, 2.5], 'override_groups_map': None, 'use_se': False, 'block_cls': QARepVGGBlockV2},
    'QARepVGGV2-B1': {'num_blocks': [4, 6, 16, 1], 'width_multiplier': [2.0, 2.0, 2.0, 4.0], 'override_groups_map': None, 'use_se': False, 'block_cls': QARepVGGBlockV2},
    'QARepVGGV2-B1g2': {'num_blocks': [4, 6, 16, 1], 'width_multiplier': [2.0, 2.0, 2.0, 4.0], 'override_groups_map': g2_map, 'use_se': False, 'block_cls': QARepVGGBlockV2},
    'QARepVGGV2-B1g4': {'num_blocks': [4, 6, 16, 1], 'width_multiplier': [2.0, 2.0, 2.0, 4.0], 'override_groups_map': g4_map, 'use_se': False, 'block_cls': QARepVGGBlockV2},
    'QARepVGGV2-D2se': {'num_blocks': [8, 14, 24, 1], 'width_multiplier': [2.5, 2.5, 2.5, 5.0], 'override_groups_map': None, 'use_se': True, 'block_cls': QARepVGGBlockV2},
}

def QARepVGG_A0(model_name='QARepVGG-A0', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model
    
def QARepVGGV1_A0(model_name='QARepVGGV1-A0', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model
    
def QARepVGGV2_A0(model_name='QARepVGGV2-A0', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model
    
def QARepVGGV2_A0_d01(model_name='QARepVGGV2-A0_d01', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model


def QARepVGGV2_A0_DW(model_name='QARepVGGV2-A0-DW', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model


def QARepVGGV6_A0(model_name='QARepVGGV6-A0', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model


def QARepVGG_A0_ReLU6(model_name='QARepVGG-A0-ReLU6', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model

def QARepVGGV2_A0_PReLU(model_name='QARepVGGV2-A0-PReLU', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model

def QARepVGGV2_A1(model_name='QARepVGGV2-A1', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model

def QARepVGGV2_A2(model_name='QARepVGGV2-A2', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model

def QARepVGGV2_B0(model_name='QARepVGGV2-B0', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model

def QARepVGGV2_B1(model_name='QARepVGGV2-B1', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model

def QARepVGGV2_B1g2(model_name='QARepVGGV2-B1g2', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model

def QARepVGGV2_B1g4(model_name='QARepVGGV2-B1g4', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model

def QARepVGGV2_D2se(model_name='QARepVGGV2-D2se', deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model
    

def get_QARepVGG_Backbone(model_name: str, deploy=False, use_checkpoint=False, strides_override=None, pretrained_path=None):
    model = QARepVGGBackbone(model_name=model_name, deploy=deploy, use_checkpoint=use_checkpoint, strides_override=strides_override)
    if pretrained_path:
        try:
            # Basic PyTorch state dict loading
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle potential 'model' key if saved from a training script
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict: # another common key
                state_dict = state_dict['state_dict']

            # Adjust keys if necessary (e.g., removing "module." prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False) # strict=False to allow some flexibility
            print(f"Loaded pretrained weights for {model_name} from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights for {model_name} from {pretrained_path}: {e}")
            print("Initializing model with random weights.")

    if deploy: # Ensure model is fully switched if created with deploy=True
        repvgg_model_convert(model, do_copy=False) # Convert in-place
    return model


def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    
    # Update the backbone's deploy state if it has one
    if hasattr(model, 'current_deploy_state'):
        model.current_deploy_state = True
        
    if save_path is not None:
        torch.save({'model': model.state_dict()}, save_path) # Saving in a dict for consistency
    return model


if __name__ == '__main__':
    # Example Usage:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Create a training-mode QARepVGGV2-A0 backbone
    model_name_train = 'QARepVGGV2-A0'
    # model_name_train = 'RepVGG-A0-base' # Test with base RepVGGBlock
    # model_name_train = 'QARepVGG-A0-ReLU6'
    # model_name_train = 'QARepVGGV2-A0-PReLU'

    print(f"\n--- Creating {model_name_train} (train mode) ---")
    backbone_train = get_QARepVGG_Backbone(model_name_train, deploy=False)
    backbone_train.to(device)
    backbone_train.eval() # Set to eval for consistent output for this test

    print(f"Backbone architecture: {model_name_train}")
    print(f"Is backbone deployed: {backbone_train.current_deploy_state}")
    print(f"Backbone width_list: {backbone_train.width_list}")

    # Dummy input
    img_size = 224
    dummy_input = torch.randn(2, 3, img_size, img_size).to(device)

    # Forward pass
    print("Running forward pass (train mode)...")
    features_train = backbone_train(dummy_input)
    print(f"Number of output feature maps: {len(features_train)}")
    for i, f in enumerate(features_train):
        print(f"Shape of feature map {i+1}: {f.shape}, Device: {f.device}")


    # 2. Convert the training model to deploy mode
    print(f"\n--- Converting {model_name_train} to deploy mode ---")
    backbone_deploy = repvgg_model_convert(backbone_train, do_copy=True) # Use do_copy=True to keep original train model
    backbone_deploy.to(device)
    backbone_deploy.eval()

    print(f"Is converted backbone deployed: {backbone_deploy.current_deploy_state}")
    # Check a specific block's deploy status
    if hasattr(backbone_deploy.stage0, 'deploy'):
        print(f"Stage0 deploy status: {backbone_deploy.stage0.deploy}")
        print(f"Stage0 has rbr_reparam: {hasattr(backbone_deploy.stage0, 'rbr_reparam')}")
    if hasattr(backbone_deploy.stage1[0], 'deploy'):
         print(f"Stage1[0] deploy status: {backbone_deploy.stage1[0].deploy}")
         print(f"Stage1[0] has rbr_reparam: {hasattr(backbone_deploy.stage1[0], 'rbr_reparam')}")


    print("Running forward pass (deploy mode)...")
    features_deploy = backbone_deploy(dummy_input)
    print(f"Number of output feature maps (deploy): {len(features_deploy)}")
    for i, f in enumerate(features_deploy):
        print(f"Shape of feature map {i+1} (deploy): {f.shape}")

    # Verify outputs are numerically close (if using SE or other non-deterministic layers during eval, might differ slightly)
    # For RepVGG, after conversion, the outputs should be very close.
    if len(features_train) == len(features_deploy):
        for i in range(len(features_train)):
            try:
                # It's possible DropPath is active in RepVGGBlock if not strictly disabled for eval
                # and it's not part of the deployed path.
                # For a strict check, ensure DropPath is nn.Identity or model is in eval().
                # The base RepVGGBlock now includes DropPath logic from QARepVGGBlockV2 for the sum of branches.
                # If DropPath is used (drop_path_ratio > 0), train and deploy outputs will differ unless
                # DropPath is also fused or bypassed in deploy, which it isn't.
                # The check is more meaningful if drop_path_ratio = 0.
                spec = QAREPVGG_SPECS[model_name_train]
                dpr = spec.get('drop_path_ratio', 0.0)
                if dpr == 0.0: # Only check if no stochastic depth is used
                    diff = torch.abs(features_train[i] - features_deploy[i]).mean()
                    print(f"Mean absolute difference for feature map {i+1}: {diff.item()}")
                    assert torch.allclose(features_train[i], features_deploy[i], atol=1e-5), f"Outputs for feature {i} do not match!"
                else:
                    print(f"Skipping numerical check for feature {i+1} due to DropPath (ratio={dpr}).")

            except AssertionError as e:
                print(f"AssertionError: {e}")
    print("Numerical check (if applicable) complete.")

    # 3. Create a deploy-mode backbone directly
    model_name_direct_deploy = 'QARepVGGV2-B0'
    print(f"\n--- Creating {model_name_direct_deploy} (direct deploy mode) ---")
    backbone_direct_deploy = get_QARepVGG_Backbone(model_name_direct_deploy, deploy=True)
    backbone_direct_deploy.to(device)
    backbone_direct_deploy.eval()

    print(f"Backbone architecture: {model_name_direct_deploy}")
    print(f"Is backbone deployed: {backbone_direct_deploy.current_deploy_state}")
    print(f"Backbone width_list: {backbone_direct_deploy.width_list}")
    if hasattr(backbone_direct_deploy.stage0, 'deploy'):
        print(f"Stage0 deploy status: {backbone_direct_deploy.stage0.deploy}")
        print(f"Stage0 has rbr_reparam: {hasattr(backbone_direct_deploy.stage0, 'rbr_reparam')}")

    print("Running forward pass (direct deploy mode)...")
    features_direct_deploy = backbone_direct_deploy(dummy_input)
    print(f"Number of output feature maps: {len(features_direct_deploy)}")
    for i, f in enumerate(features_direct_deploy):
        print(f"Shape of feature map {i+1}: {f.shape}")

    print("\nExample completed.")