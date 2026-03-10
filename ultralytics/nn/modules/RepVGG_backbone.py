# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch.nn as nn
import numpy as np
import torch
import copy
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F # Added for SEBlock

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3)) # Use F.avg_pool2d
        x = self.down(x)
        x = F.relu(x) # Use F.relu
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels # Store out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

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
            # Commenting out print for cleaner output during model creation
            # print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_custom_L2(self):
        if hasattr(self, 'rbr_reparam'): # Cannot get L2 for deployed block
            return 0
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()
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
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                # Ensure in_channels and out_channels match for identity if it's a BN layer
                # For RepVGG, identity branch exists if in_channels == out_channels
                # The BN in identity is on num_features=in_channels
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels): # This should be self.out_channels if BN is on output side
                                                 # But identity BN is on input features, so self.in_channels
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        # Use parameters from rbr_dense for the new Conv2d layer's shape
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, 
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, 
                                     stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, 
                                     dilation=self.rbr_dense.conv.dilation, 
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'): # also remove id_tensor if it exists
            self.__delattr__('id_tensor')
        self.deploy = True


# Define RepVGG model specifications
REPVGG_SPECS = {
    'RepVGG-A0': {'num_blocks': [2, 4, 14, 1], 'width_multiplier': [0.75, 0.75, 0.75, 2.5], 'override_groups_map': None, 'use_se': False},
    'RepVGG-A1': {'num_blocks': [2, 4, 14, 1], 'width_multiplier': [1.0, 1.0, 1.0, 2.5], 'override_groups_map': None, 'use_se': False},
    'RepVGG-A2': {'num_blocks': [2, 4, 14, 1], 'width_multiplier': [1.5, 1.5, 1.5, 2.75], 'override_groups_map': None, 'use_se': False},
    'RepVGG-B0': {'num_blocks': [4, 6, 16, 1], 'width_multiplier': [1.0, 1.0, 1.0, 2.5], 'override_groups_map': None, 'use_se': False},
    'RepVGG-B1': {'num_blocks': [4, 6, 16, 1], 'width_multiplier': [2.0, 2.0, 2.0, 4.0], 'override_groups_map': None, 'use_se': False},
    'RepVGG-B1g2': {'num_blocks': [4, 6, 16, 1], 'width_multiplier': [2.0, 2.0, 2.0, 4.0], 'override_groups_map': {l: 2 for l in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]}, 'use_se': False},
    'RepVGG-B1g4': {'num_blocks': [4, 6, 16, 1], 'width_multiplier': [2.0, 2.0, 2.0, 4.0], 'override_groups_map': {l: 4 for l in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]}, 'use_se': False},
    'RepVGG-B2': {'num_blocks': [4, 6, 16, 1], 'width_multiplier': [2.5, 2.5, 2.5, 5.0], 'override_groups_map': None, 'use_se': False},
    'RepVGG-B2g2': {'num_blocks': [4, 6, 16, 1], 'width_multiplier': [2.5, 2.5, 2.5, 5.0], 'override_groups_map': {l: 2 for l in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]}, 'use_se': False},
    'RepVGG-B2g4': {'num_blocks': [4, 6, 16, 1], 'width_multiplier': [2.5, 2.5, 2.5, 5.0], 'override_groups_map': {l: 4 for l in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]}, 'use_se': False},
    'RepVGG-B3': {'num_blocks': [4, 6, 16, 1], 'width_multiplier': [3.0, 3.0, 3.0, 5.0], 'override_groups_map': None, 'use_se': False},
    'RepVGG-B3g2': {'num_blocks': [4, 6, 16, 1], 'width_multiplier': [3.0, 3.0, 3.0, 5.0], 'override_groups_map': {l: 2 for l in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]}, 'use_se': False},
    'RepVGG-B3g4': {'num_blocks': [4, 6, 16, 1], 'width_multiplier': [3.0, 3.0, 3.0, 5.0], 'override_groups_map': {l: 4 for l in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]}, 'use_se': False},
    'RepVGG-D2se': {'num_blocks': [8, 14, 24, 1], 'width_multiplier': [2.5, 2.5, 2.5, 5.0], 'override_groups_map': None, 'use_se': True},
}

class RepVGGBackbone(nn.Module):
    def __init__(self, model_name: str, use_checkpoint=False):
        super(RepVGGBackbone, self).__init__()
        
        if model_name not in REPVGG_SPECS:
            raise ValueError(f"Model name {model_name} not found in REPVGG_SPECS.")
            
        spec = REPVGG_SPECS[model_name]
        num_blocks = spec['num_blocks']
        width_multiplier = spec['width_multiplier']
        override_groups_map = spec['override_groups_map'] or dict()
        use_se = spec['use_se']

        # RepVGGBlocks are always initialized with deploy=False for backbone structure
        # Conversion to deploy mode is handled by repvgg_model_convert if needed
        self.current_deploy_state = False 
        self.use_checkpoint = use_checkpoint
        self.use_se = use_se
        self.override_groups_map = override_groups_map
        assert 0 not in self.override_groups_map # From original code

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, 
                                  stride=2, padding=1, deploy=self.current_deploy_state, use_se=self.use_se)
        self.cur_layer_idx = 1 # Start after stage0
        
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)

        # Calculate width_list for compatibility, using outputs of stage1, stage2, stage3, stage4
        # Ensure model is on CPU for this dummy forward pass to avoid device issues if not yet moved
        # We need to make sure that the model is in eval mode for batchnorm stats if they are not frozen
        # and then restore its training mode. For width calculation, this should be fine.
        original_training_state = self.training
        self.eval() # Consistent behavior for BN/Dropout during dummy forward
        try:
            with torch.no_grad():
                # Use a typical input size, e.g., 224x224 or 640x640.
                # For width_list, the exact spatial dim doesn't matter as much as batch size 1.
                # Create dummy input on CPU to avoid potential CUDA errors if model not yet moved.
                dummy_input = torch.randn(1, 3, 224, 224) 
                features = self.forward(dummy_input) # This will return a list of 4 tensors
            self.width_list = [f.size(1) for f in features]
        finally:
            self.train(original_training_state) # Restore original training state


    def _make_stage(self, planes, num_blocks_stage, stride):
        strides = [stride] + [1]*(num_blocks_stage-1)
        blocks = []
        for s_idx, current_stride in enumerate(strides):
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, 
                                      out_channels=planes, 
                                      kernel_size=3,
                                      stride=current_stride, 
                                      padding=1, 
                                      groups=cur_groups, 
                                      deploy=self.current_deploy_state, # Always False during construction
                                      use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks) # Use ModuleList for proper registration

    def forward(self, x):
        x = self.stage0(x)
        
        outputs = []
        # Pass through stage1
        s1_out = x
        for block in self.stage1:
            if self.use_checkpoint and not torch.jit.is_scripting(): # checkpoint is not scriptable
                s1_out = checkpoint.checkpoint(block, s1_out)
            else:
                s1_out = block(s1_out)
        outputs.append(s1_out)

        # Pass through stage2
        s2_out = s1_out
        for block in self.stage2:
            if self.use_checkpoint and not torch.jit.is_scripting():
                s2_out = checkpoint.checkpoint(block, s2_out)
            else:
                s2_out = block(s2_out)
        outputs.append(s2_out)

        # Pass through stage3
        s3_out = s2_out
        for block in self.stage3:
            if self.use_checkpoint and not torch.jit.is_scripting():
                s3_out = checkpoint.checkpoint(block, s3_out)
            else:
                s3_out = block(s3_out)
        outputs.append(s3_out)
        
        # Pass through stage4
        s4_out = s3_out
        for block in self.stage4:
            if self.use_checkpoint and not torch.jit.is_scripting():
                s4_out = checkpoint.checkpoint(block, s4_out)
            else:
                s4_out = block(s4_out)
        outputs.append(s4_out)
        
        return outputs # Return list of features from stage1, stage2, stage3, stage4

# Factory functions
def RepVGG_A0(deploy=False, use_checkpoint=False):
    model = RepVGGBackbone(model_name='RepVGG-A0', use_checkpoint=use_checkpoint)
    if deploy:
        model = repvgg_model_convert(model, do_copy=False) # Modify in-place for consistency
    return model

def RepVGG_A1(deploy=False, use_checkpoint=False):
    model = RepVGGBackbone(model_name='RepVGG-A1', use_checkpoint=use_checkpoint)
    if deploy:
        model = repvgg_model_convert(model, do_copy=False)
    return model

def RepVGG_A2(deploy=False, use_checkpoint=False):
    model = RepVGGBackbone(model_name='RepVGG-A2', use_checkpoint=use_checkpoint)
    if deploy:
        model = repvgg_model_convert(model, do_copy=False)
    return model

def RepVGG_B0(deploy=False, use_checkpoint=False):
    model = RepVGGBackbone(model_name='RepVGG-B0', use_checkpoint=use_checkpoint)
    if deploy:
        model = repvgg_model_convert(model, do_copy=False)
    return model

def RepVGG_B1(deploy=False, use_checkpoint=False):
    model = RepVGGBackbone(model_name='RepVGG-B1', use_checkpoint=use_checkpoint)
    if deploy:
        model = repvgg_model_convert(model, do_copy=False)
    return model

def RepVGG_B1g2(deploy=False, use_checkpoint=False):
    model = RepVGGBackbone(model_name='RepVGG-B1g2', use_checkpoint=use_checkpoint)
    if deploy:
        model = repvgg_model_convert(model, do_copy=False)
    return model

def RepVGG_B1g4(deploy=False, use_checkpoint=False):
    model = RepVGGBackbone(model_name='RepVGG-B1g4', use_checkpoint=use_checkpoint)
    if deploy:
        model = repvgg_model_convert(model, do_copy=False)
    return model

def RepVGG_B2(deploy=False, use_checkpoint=False):
    model = RepVGGBackbone(model_name='RepVGG-B2', use_checkpoint=use_checkpoint)
    if deploy:
        model = repvgg_model_convert(model, do_copy=False)
    return model

def RepVGG_B2g2(deploy=False, use_checkpoint=False):
    model = RepVGGBackbone(model_name='RepVGG-B2g2', use_checkpoint=use_checkpoint)
    if deploy:
        model = repvgg_model_convert(model, do_copy=False)
    return model

def RepVGG_B2g4(deploy=False, use_checkpoint=False):
    model = RepVGGBackbone(model_name='RepVGG-B2g4', use_checkpoint=use_checkpoint)
    if deploy:
        model = repvgg_model_convert(model, do_copy=False)
    return model

def RepVGG_B3(deploy=False, use_checkpoint=False):
    model = RepVGGBackbone(model_name='RepVGG-B3', use_checkpoint=use_checkpoint)
    if deploy:
        model = repvgg_model_convert(model, do_copy=False)
    return model

def RepVGG_B3g2(deploy=False, use_checkpoint=False):
    model = RepVGGBackbone(model_name='RepVGG-B3g2', use_checkpoint=use_checkpoint)
    if deploy:
        model = repvgg_model_convert(model, do_copy=False)
    return model

def RepVGG_B3g4(deploy=False, use_checkpoint=False):
    model = RepVGGBackbone(model_name='RepVGG-B3g4', use_checkpoint=use_checkpoint)
    if deploy:
        model = repvgg_model_convert(model, do_copy=False)
    return model

def RepVGG_D2se(deploy=False, use_checkpoint=False):
    model = RepVGGBackbone(model_name='RepVGG-D2se', use_checkpoint=use_checkpoint)
    if deploy:
        model = repvgg_model_convert(model, do_copy=False)
    return model

# Helper to get model by name string, now using the new factory functions
def get_RepVGG_func_by_name(name_str):
    # This is a simple map to the new factory functions
    # In a real scenario, you might want a more robust lookup or directly use the factory functions.
    factory_map = {
        'RepVGG-A0': RepVGG_A0, 'RepVGG-A1': RepVGG_A1, 'RepVGG-A2': RepVGG_A2,
        'RepVGG-B0': RepVGG_B0, 'RepVGG-B1': RepVGG_B1, 'RepVGG-B1g2': RepVGG_B1g2, 'RepVGG-B1g4': RepVGG_B1g4,
        'RepVGG-B2': RepVGG_B2, 'RepVGG-B2g2': RepVGG_B2g2, 'RepVGG-B2g4': RepVGG_B2g4,
        'RepVGG-B3': RepVGG_B3, 'RepVGG-B3g2': RepVGG_B3g2, 'RepVGG-B3g4': RepVGG_B3g4,
        'RepVGG-D2se': RepVGG_D2se,
    }
    if name_str in factory_map:
        return factory_map[name_str]
    else:
        raise ValueError(f"Model factory for {name_str} not found.")

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
            # After switching, update the backbone's internal deploy state if it has one
            if hasattr(model, 'current_deploy_state'):
                 model.current_deploy_state = True
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


if __name__ == '__main__':
    # Example usage:
    # 1. Create a training-mode RepVGG-A0 backbone
    model_train = RepVGG_A0(deploy=False)
    print(f"Created RepVGG-A0 (train mode). Backbone width_list: {model_train.width_list}")
    
    # Dummy input
    img_size = 224 # for RepVGG typical training
    # img_size = 640 # for detection typical input
    dummy_input = torch.randn(2, 3, img_size, img_size) # Batch size 2

    # Forward pass
    features = model_train(dummy_input)
    print(f"Number of output feature maps: {len(features)}")
    for i, f in enumerate(features):
        print(f"Shape of feature map {i+1}: {f.shape}")

    # 2. Create a deploy-mode RepVGG-A0 backbone
    model_deploy = RepVGG_A0(deploy=True)
    # Note: model_deploy.width_list will still reflect the training architecture's widths.
    # The actual channels of the deployed model's layers would be the same,
    # but the internal structure is simplified to single Conv2D per block.
    print(f"\nCreated RepVGG-A0 (deploy mode). Backbone width_list: {model_deploy.width_list}")

    # Forward pass with deployed model
    # If the model is on CUDA, move input to CUDA
    # model_deploy.to('cuda')
    # dummy_input = dummy_input.to('cuda')
    features_deploy = model_deploy(dummy_input)
    print(f"Number of output feature maps (deploy): {len(features_deploy)}")
    for i, f in enumerate(features_deploy):
        print(f"Shape of feature map {i+1} (deploy): {f.shape}")

    # Check if a specific block is deployed
    print(f"\nIs stage0 of deploy model a RepVGGBlock? {isinstance(model_deploy.stage0, RepVGGBlock)}")
    if isinstance(model_deploy.stage0, RepVGGBlock):
        print(f"Is stage0 of deploy model in deploy mode? {model_deploy.stage0.deploy}")
        print(f"Does stage0 of deploy model have rbr_reparam? {hasattr(model_deploy.stage0, 'rbr_reparam')}")

    # Test with a different model, e.g., RepVGG-D2se
    model_d2se_train = RepVGG_D2se(deploy=False, use_checkpoint=True)
    print(f"\nCreated RepVGG-D2se (train mode with checkpointing). Backbone width_list: {model_d2se_train.width_list}")
    features_d2se = model_d2se_train(dummy_input)
    print(f"Number of output feature maps (D2se): {len(features_d2se)}")
    for i, f in enumerate(features_d2se):
        print(f"Shape of feature map {i+1} (D2se): {f.shape}")

    # Test get_RepVGG_func_by_name
    factory_fn = get_RepVGG_func_by_name('RepVGG-B0')
    model_b0_from_factory = factory_fn(deploy=True)
    print(f"\nCreated RepVGG-B0 via factory (deploy mode). Backbone width_list: {model_b0_from_factory.width_list}")