import torch
import torch.nn as nn
from torch import autograd
from typing import List

# --- GetMask (Unchanged) ---
class GetMask(autograd.Function):
    @staticmethod
    def forward(ctx, scores):  # binarization
        # This is where the error occurs if scores is None
        expert_pred = torch.argmax(scores, dim=1)  # [bs]
        expert_pred_one_hot = torch.zeros_like(scores).scatter_(1, expert_pred.unsqueeze(-1), 1)
        return expert_pred, expert_pred_one_hot

    @staticmethod
    def backward(ctx, g1, g2):
        return g2

# --- MoEBase (Modified set_score for clarity and debug) ---
class MoEBase(nn.Module):
    def __init__(self):
        super(MoEBase, self).__init__()
        self.scores = None
        self.router = None # Router for ResNet is defined in ResNet itself
        # print(f"DEBUG MoEBase __init__: {type(self).__name__} ID {id(self)}, self.scores is None.")

    def set_score(self, scores_arg):
        # print(f"DEBUG MoEBase set_score: Called by {type(self).__name__} ID {id(self)}. Input scores_arg is None: {scores_arg is None}")
        # if scores_arg is not None:
        #     print(f"  Input scores_arg shape: {scores_arg.shape}")

        self.scores = scores_arg # Set for the current MoEBase instance (e.g., ResNet)
        
        # Propagate to children MoEBase instances (e.g., MoEConv)
        for module_name, module in self.named_modules():
            if module is self: # Skip self
                continue
            if isinstance(module, MoEBase):
                # print(f"  Propagating scores from {type(self).__name__} to {module_name} ({type(module).__name__} ID: {id(module)})")
                module.scores = self.scores # Assign the tensor (or None) directly
                # if module.scores is None:
                #     print(f"    DEBUG: {module_name} (ID {id(module)}) self.scores is now None after propagation.")
                # elif hasattr(module.scores, 'shape'):
                #      print(f"    DEBUG: {module_name} (ID {id(module)}) self.scores shape after propagation: {module.scores.shape}")


# --- MoEConv (Modified forward for robust error checking) ---
class MoEConv(nn.Conv2d, MoEBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=False,
                 n_expert=5):
        MoEBase.__init__(self)
        super(MoEConv, self).__init__(in_channels, out_channels * n_expert, kernel_size, stride, padding, dilation,
                                      groups, bias)
        
        self.true_in_channels = in_channels
        self.true_out_channels = out_channels 
        self.n_expert = n_expert
        assert self.n_expert >= 1

        self.layer_selection = torch.zeros([n_expert, out_channels * n_expert])
        for expert_id in range(n_expert):
            start = expert_id * self.true_out_channels
            end = (expert_id + 1) * self.true_out_channels
            self.layer_selection[expert_id, start:end] = 1
        # Initial self.scores is None due to MoEBase.__init__

    def forward(self, x):
        if self.n_expert > 1:
            # print(f"DEBUG MoEConv forward: ID {id(self)}, self.scores is None: {self.scores is None}")
            # if self.scores is not None:
            #     print(f"  MoEConv ID {id(self)} scores shape: {self.scores.shape}, device: {self.scores.device}")
            # else:
            #     # This print will show up if scores are None right before GetMask
            #     print(f"  ERROR MoEConv ID {id(self)} (in:{self.true_in_channels}, out:{self.true_out_channels}, exp:{self.n_expert}) has self.scores=None before GetMask!")


            if self.scores is None:
                # This is a critical failure if MoE is intended to be active.
                # It means scores were not propagated from ResNet, or ResNet's router failed.
                raise ValueError(
                    f"MoEConv (in_channels={self.true_in_channels}, out_channels={self.true_out_channels}, "
                    f"n_expert={self.n_expert}, ID:{id(self)}) encountered self.scores = None. "
                    "If MoE is active (use_moe=True in ResNet and n_expert > 1), "
                    "ResNet.router must produce scores and ResNet.set_score() must propagate them."
                )

            # GetMask.apply expects scores for the current batch: [bs, n_expert]
            _expert_pred, expert_pred_one_hot = GetMask.apply(self.scores) # Get both outputs

            mask = torch.matmul(expert_pred_one_hot, self.layer_selection.to(x.device))
            out_all_experts = super(MoEConv, self).forward(x)
            out_masked = out_all_experts * mask.unsqueeze(-1).unsqueeze(-1)
            
            bs, _, H, W = out_masked.shape
            out_reshaped_experts = out_masked.view(bs, self.n_expert, self.true_out_channels, H, W)
            out_selected = out_reshaped_experts.sum(dim=1)
        else: 
            out_selected = super(MoEConv, self).forward(x)
        
        return out_selected

# --- Convolution helpers (Unchanged) ---
def conv3x3(in_planes, out_planes, conv_layer, stride=1, groups=1, dilation=1, **kwargs):
    return conv_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=dilation, groups=groups, bias=False, dilation=dilation, **kwargs)

def conv1x1(in_planes, out_planes, conv_layer, stride=1, **kwargs):
    return conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False, **kwargs)

# --- BasicBlock (Unchanged from previous correct version) ---
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']
    def __init__(self, inplanes, planes, conv_layer, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, **kwargs): 
        super(BasicBlock, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64: raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1: raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        n_expert_kwargs = {'n_expert': kwargs.get('n_expert', 1)} if conv_layer == MoEConv else {}
        self.conv1 = conv3x3(inplanes, planes, conv_layer, stride, **n_expert_kwargs)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, conv_layer, **n_expert_kwargs)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; out = self.relu(out)
        return out

# --- Bottleneck (Unchanged from previous correct version) ---
class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']
    def __init__(self, inplanes, planes, conv_layer, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        n_expert_kwargs = {'n_expert': kwargs.get('n_expert', 1)} if conv_layer == MoEConv else {}
        self.conv1 = conv1x1(inplanes, width, conv_layer, **n_expert_kwargs)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, conv_layer, stride, groups, dilation, **n_expert_kwargs)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, conv_layer, **n_expert_kwargs)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; out = self.relu(out)
        return out

# --- ResNet Class (Modified forward for robust error checking) ---
class ResNet(MoEBase):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, n_expert=1, ratio=1.0, use_moe=False, router_class=None):
        super(ResNet, self).__init__() 
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.use_moe = use_moe
        self.n_expert = n_expert

        if self.use_moe and self.n_expert > 1:
            self.conv_layer = MoEConv
            # print(f"DEBUG ResNet __init__: MoE enabled with n_expert={self.n_expert}. Using MoEConv.")
        else:
            if self.use_moe and self.n_expert <= 1:
                # print(f"DEBUG ResNet __init__: use_moe is True but n_expert ({self.n_expert}) <= 1. Fallback to standard nn.Conv2d.")
                self.use_moe = False # Effectively disable MoE
            self.conv_layer = nn.Conv2d
            self.n_expert = 1 # Ensure n_expert is 1 if not effectively using MoE

        self.ratio = ratio
        self.normalize = None 

        self.inplanes = int(ratio * 64)
        self.dilation = 1
        if replace_stride_with_dilation is None: replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3: raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.router = None # Initialize router
        if self.use_moe: # This check uses the potentially updated self.use_moe
            # Define router here if MoE is effectively active
            # (self.use_moe is True implies self.n_expert > 1 at this point)
            if router_class is None:
                # print(f"DEBUG ResNet __init__: Defining SimpleRouter for {self.inplanes} in_feat, {self.n_expert} experts.")
                class SimpleRouter(nn.Module):
                    def __init__(self, in_feat, n_exp):
                        super().__init__()
                        self.pool = nn.AdaptiveAvgPool2d((1,1))
                        self.fc = nn.Linear(in_feat, n_exp)
                    def forward(self, x_route):
                        x_route = self.pool(x_route)
                        x_route = torch.flatten(x_route, 1)
                        return self.fc(x_route)
                self.router = SimpleRouter(self.inplanes, self.n_expert)
            else:
                # print(f"DEBUG ResNet __init__: Using provided router_class.")
                self.router = router_class(in_features=self.inplanes, num_experts=self.n_expert)
        # else:
            # print(f"DEBUG ResNet __init__: MoE not active or n_expert <=1, router will not be used.")


        layer_kwargs = {'n_expert': self.n_expert} if self.use_moe else {}

        self.layer1 = self._make_layer(block, int(64*ratio), layers[0], **layer_kwargs)
        self.layer2 = self._make_layer(block, int(128*ratio), layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], **layer_kwargs)
        self.layer3 = self._make_layer(block, int(256*ratio), layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], **layer_kwargs)
        self.layer4 = self._make_layer(block, int(512*ratio), layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], **layer_kwargs)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * self.ratio * block.expansion), num_classes)

        self._initialize_weights(zero_init_residual)

        self.width_list = []
        if hasattr(self, 'forward'): # Ensure forward method is defined before calling
            try:
                original_training_mode = self.training
                self.eval() 
                with torch.no_grad():
                    dummy_input_size = 224 # Or a size more relevant to your use case
                    # print(f"DEBUG ResNet __init__ width_list: Creating dummy_input (1, 3, {dummy_input_size}, {dummy_input_size})")
                    dummy_input = torch.randn(1, 3, dummy_input_size, dummy_input_size)
                    if next(self.parameters()).is_cuda: # Move dummy input to GPU if model is on GPU
                        dummy_input = dummy_input.to(next(self.parameters()).device)
                    
                    # print(f"DEBUG ResNet __init__ width_list: Calling self.forward for width_list calculation.")
                    features = self.forward(dummy_input) 
                    self.width_list = [f.size(1) for f in features]
                    # print(f"DEBUG ResNet __init__ width_list: Computed width_list: {self.width_list}")
                self.train(original_training_mode)
            except Exception as e:
                print(f"Warning: Could not compute width_list during ResNet init: {e}")
                import traceback
                traceback.print_exc()
                self.width_list = []
        # else:
            # print("DEBUG ResNet __init__ width_list: self.forward not fully available yet (should not happen here).")


    def _initialize_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None: nn.init.constant_(m.weight, 1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck): nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock): nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate: self.dilation *= stride; stride = 1
        
        n_expert_kwargs = {'n_expert': kwargs.get('n_expert', 1)} if self.conv_layer == MoEConv else {}

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, self.conv_layer, stride, **n_expert_kwargs),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, self.conv_layer, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.conv_layer, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, **kwargs)) 
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.normalize is not None: x_normalized = self.normalize(x)
        else: x_normalized = x

        x_stem = self.conv1(x_normalized)
        x_stem = self.bn1(x_stem)
        x_stem = self.relu(x_stem)
        x_pooled = self.maxpool(x_stem)

        current_scores_tensor = None # Define outside the if block
        if self.use_moe and self.router is not None:
            # print(f"DEBUG ResNet.forward: Router is active. Calling router with x_pooled device: {x_pooled.device}")
            current_scores_tensor = self.router(x_pooled)
            if current_scores_tensor is None: # CRITICAL CHECK
                raise ValueError(
                    "ResNet.router returned None. This is unexpected and will cause MoE to fail. "
                    f"Router type: {type(self.router)}, x_pooled shape: {x_pooled.shape}"
                )
            # print(f"DEBUG ResNet.forward: Router produced scores. Shape: {current_scores_tensor.shape}, Device: {current_scores_tensor.device}. Calling self.set_score.")
            self.set_score(current_scores_tensor)
        # elif self.use_moe and self.router is None:
            # print("DEBUG ResNet.forward: self.use_moe is True but self.router is None. Scores will not be set. MoEConv layers will likely fail if n_expert > 1.")


        features = []
        # print(f"DEBUG ResNet.forward: Processing layer1 with x_pooled device: {x_pooled.device}")
        x1 = self.layer1(x_pooled); features.append(x1)
        # print(f"DEBUG ResNet.forward: Processing layer2 with x1 device: {x1.device}")
        x2 = self.layer2(x1);       features.append(x2)
        # print(f"DEBUG ResNet.forward: Processing layer3 with x2 device: {x2.device}")
        x3 = self.layer3(x2);       features.append(x3)
        # print(f"DEBUG ResNet.forward: Processing layer4 with x3 device: {x3.device}")
        x4 = self.layer4(x3);       features.append(x4)
        
        if self.use_moe: # Reset scores only if MoE was potentially active
            # print("DEBUG ResNet.forward: Resetting scores to None after forward pass.")
            self.scores = None 
            for module in self.modules():
                if isinstance(module, MoEBase) and module is not self:
                    module.scores = None
        return features

    def forward_classification(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward(x) 
        out = features[-1] 
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# --- ResNet Model Instantiation Functions (Unchanged from previous correct version) ---
def resnet18_moe(**kwargs) -> ResNet:
    defaults = {'block': BasicBlock, 'layers': [2, 2, 2, 2], 'use_moe': False, 'n_expert': 1, 'router_class': None}
    if kwargs.get('use_moe', False) and 'n_expert' not in kwargs: kwargs['n_expert'] = 5 
    final_kwargs = {**defaults, **kwargs}
    return ResNet(**final_kwargs)

def resnet34_moe(**kwargs) -> ResNet:
    defaults = {'block': BasicBlock, 'layers': [3, 4, 6, 3], 'use_moe': False, 'n_expert': 1, 'router_class': None}
    if kwargs.get('use_moe', False) and 'n_expert' not in kwargs: kwargs['n_expert'] = 5
    final_kwargs = {**defaults, **kwargs}
    return ResNet(**final_kwargs)

def resnet50_moe(**kwargs) -> ResNet:
    defaults = {'block': Bottleneck, 'layers': [3, 4, 6, 3], 'use_moe': False, 'n_expert': 1, 'router_class': None}
    if kwargs.get('use_moe', False) and 'n_expert' not in kwargs: kwargs['n_expert'] = 5
    final_kwargs = {**defaults, **kwargs}
    return ResNet(**final_kwargs)

def resnet101_moe(**kwargs) -> ResNet:
    defaults = {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'use_moe': False, 'n_expert': 1, 'router_class': None}
    if kwargs.get('use_moe', False) and 'n_expert' not in kwargs: kwargs['n_expert'] = 5
    final_kwargs = {**defaults, **kwargs}
    return ResNet(**final_kwargs)

def resnet152_moe(**kwargs) -> ResNet:
    defaults = {'block': Bottleneck, 'layers': [3, 8, 36, 3], 'use_moe': False, 'n_expert': 1, 'router_class': None}
    if kwargs.get('use_moe', False) and 'n_expert' not in kwargs: kwargs['n_expert'] = 5
    final_kwargs = {**defaults, **kwargs}
    return ResNet(**final_kwargs)


if __name__ == '__main__':
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nTesting Standard ResNet-18:")
    model_resnet18_std = resnet18_moe(num_classes=100, use_moe=False).to(device)
    print(f"ResNet-18 Standard Width List: {model_resnet18_std.width_list}")
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    features_std = model_resnet18_std(dummy_input)
    print(f"ResNet-18 Standard Output Features ({len(features_std)}): {[f.shape for f in features_std]}")
    
    print("\nTesting ResNet-50 with MoE:")
    model_resnet50_moe = resnet50_moe(num_classes=100, use_moe=True, n_expert=4, ratio=0.5).to(device)
    print(f"ResNet-50 MoE Width List (may be empty if init failed, check warnings): {model_resnet50_moe.width_list}")
    
    is_moe_conv_present = any(isinstance(m, MoEConv) for m in model_resnet50_moe.modules())
    print(f"Are MoEConv layers present in ResNet-50 MoE: {is_moe_conv_present}")
    if model_resnet50_moe.router:
        print(f"Router type: {type(model_resnet50_moe.router)}")
    else:
        print(f"Router is None for ResNet-50 MoE (use_moe={model_resnet50_moe.use_moe}, n_expert={model_resnet50_moe.n_expert})")

    # Perform a forward pass if width_list calculation might have failed, to see error directly
    try:
        print("Attempting forward pass for ResNet-50 MoE...")
        dummy_input_moe = torch.randn(2, 3, 224, 224).to(device)
        features_moe = model_resnet50_moe(dummy_input_moe)
        print(f"ResNet-50 MoE Output Features ({len(features_moe)}): {[f.shape for f in features_moe]}")
    except Exception as e:
        print(f"ERROR during ResNet-50 MoE forward pass: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting ResNet-18 with MoE (n_expert=1, should behave like standard):")
    model_resnet18_moe_1exp = resnet18_moe(num_classes=100, use_moe=True, n_expert=1).to(device)
    is_moe_conv_present_1exp = any(isinstance(m, MoEConv) for m in model_resnet18_moe_1exp.modules())
    print(f"Are MoEConv layers present in ResNet-18 MoE (1 expert): {is_moe_conv_present_1exp}") # Should be False
    print(f"Actual conv_layer type in ResNet-18 MoE (1 expert) layer1[0].conv1: {type(model_resnet18_moe_1exp.layer1[0].conv1)}") # Should be nn.Conv2d
    features_moe_1exp = model_resnet18_moe_1exp(dummy_input)
    print(f"ResNet-18 MoE (1 expert) Output Features ({len(features_moe_1exp)}): {[f.shape for f in features_moe_1exp]}")
    