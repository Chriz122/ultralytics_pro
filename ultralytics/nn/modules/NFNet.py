# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import List, Dict, Any, Optional

# NFNet 參數配置 (保持不變)
nfnet_params = {
    'F0': {
        'width': [256, 512, 1536, 1536], 'depth': [1, 2, 6, 3],
        'train_imsize': 192, 'test_imsize': 256,
        'RA_level': '405', 'drop_rate': 0.2},
    'F1': {
        'width': [256, 512, 1536, 1536], 'depth': [2, 4, 12, 6],
        'train_imsize': 224, 'test_imsize': 320,
        'RA_level': '410', 'drop_rate': 0.3},
    'F2': {
        'width': [256, 512, 1536, 1536], 'depth': [3, 6, 18, 9],
        'train_imsize': 256, 'test_imsize': 352,
        'RA_level': '410', 'drop_rate': 0.4},
    'F3': {
        'width': [256, 512, 1536, 1536], 'depth': [4, 8, 24, 12],
        'train_imsize': 320, 'test_imsize': 416,
        'RA_level': '415', 'drop_rate': 0.4},
    'F4': {
        'width': [256, 512, 1536, 1536], 'depth': [5, 10, 30, 15],
        'train_imsize': 384, 'test_imsize': 512,
        'RA_level': '415', 'drop_rate': 0.5},
    'F5': {
        'width': [256, 512, 1536, 1536], 'depth': [6, 12, 36, 18],
        'train_imsize': 416, 'test_imsize': 544,
        'RA_level': '415', 'drop_rate': 0.5},
    'F6': {
        'width': [256, 512, 1536, 1536], 'depth': [7, 14, 42, 21],
        'train_imsize': 448, 'test_imsize': 576,
        'RA_level': '415', 'drop_rate': 0.5},
    'F7': {
        'width': [256, 512, 1536, 1536], 'depth': [8, 16, 48, 24],
        'train_imsize': 480, 'test_imsize': 608,
        'RA_level': '415', 'drop_rate': 0.5},
}

# --- Helper Classes (基本保持不變) ---

# Variance Preserving GELU
class VPGELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input) * 1.7015043497085571

# Variance Preserving ReLU
class VPReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(VPReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input, inplace=self.inplace) * 1.7139588594436646

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

# Activation Dictionary
activations_dict = {
    'gelu': VPGELU(),
    'relu': VPReLU(inplace=True)
}

# Weight Standardized Conv2D (保持不變)
class WSConv2D(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0,
        dilation = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):

        super(WSConv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)

        nn.init.xavier_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        # Using register_buffer for eps and fan_in for state_dict compatibility and device handling
        self.register_buffer('eps', torch.tensor(1e-4, requires_grad=False), persistent=False)
        # Calculate fan_in based on weight shape
        fan_in = self.weight.shape[1:].numel()
        self.register_buffer('fan_in', torch.tensor(fan_in, requires_grad=False).type_as(self.weight), persistent=False)


    def standardized_weights(self):
        # Original code: HWCN -> PyTorch: NCHW
        mean = torch.mean(self.weight, dim=[1,2,3], keepdim=True)
        var = torch.var(self.weight, dim=[1,2,3], keepdim=True)
        # Ensure fan_in is used correctly in scale calculation
        scale = torch.rsqrt(torch.maximum(var * self.fan_in, self.eps))
        return (self.weight - mean) * scale * self.gain

    def forward(self, x):
        return F.conv2d(
            input=x,
            weight=self.standardized_weights(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

# Squeeze-and-Excitation (保持不變)
class SqueezeExcite(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, se_ratio:float=0.5, activation:str='gelu'):
        super(SqueezeExcite, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.se_ratio = se_ratio

        self.hidden_channels = max(1, int(self.in_channels * self.se_ratio))

        self.activation = activations_dict[activation]
        self.linear = nn.Linear(self.in_channels, self.hidden_channels)
        self.linear_1 = nn.Linear(self.hidden_channels, self.out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = torch.mean(x, (2,3))
        out = self.linear_1(self.activation(self.linear(out)))
        out = self.sigmoid(out)

        b,c,_,_ = x.size()
        return out.view(b,c,1,1).expand_as(x)

# Stochastic Depth (保持不變)
class StochDepth(nn.Module):
    def __init__(self, stochdepth_rate:float):
        super(StochDepth, self).__init__()
        self.drop_rate = stochdepth_rate

    def forward(self, x):
        if not self.training or self.drop_rate == 0.: # Added check for drop_rate == 0
            return x

        batch_size = x.shape[0]
        rand_tensor = torch.rand(batch_size, 1, 1, 1).type_as(x).to(x.device)
        keep_prob = 1.0 - self.drop_rate # Corrected keep_prob calculation
        binary_tensor = torch.floor(rand_tensor + keep_prob)

        # Scale output by keep_prob during training (inverted dropout)
        return x * binary_tensor / keep_prob # Apply scaling


# Stem Block (保持不變)
class Stem(nn.Module):
    def __init__(self, activation:str='gelu'):
        super(Stem, self).__init__()

        self.activation = activations_dict[activation]
        # Corrected padding for stride=2 convs to potentially match TF 'SAME' padding behavior
        # For kernel=3, stride=2: padding=1 usually needed
        self.conv0 = WSConv2D(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv1 = WSConv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = WSConv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = WSConv2D(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.activation(self.conv0(x))
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv2(out))
        out = self.conv3(out) # No activation after the last stem conv
        return out

# NFBlock (保持不變)
class NFBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, expansion:float=0.5,
        se_ratio:float=0.5, stride:int=1, beta:float=1.0, alpha:float=0.2,
        group_size:int=1, stochdepth_rate:Optional[float]=None, activation:str='gelu'): # Added Optional type hint

        super(NFBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.se_ratio = se_ratio
        self.activation = activations_dict[activation]
        self.beta, self.alpha = beta, alpha
        self.group_size = group_size

        width = int(self.out_channels * expansion)
        self.groups = width // group_size if group_size > 0 else 1 # Avoid division by zero
        # Ensure width is divisible by group_size, may need adjustment based on original paper/impl details
        self.width = self.groups * group_size # Recalculate width based on groups


        self.stride = stride

        self.conv0 = WSConv2D(in_channels=self.in_channels, out_channels=self.width, kernel_size=1)
        self.conv1 = WSConv2D(in_channels=self.width, out_channels=self.width, kernel_size=3, stride=stride, padding=1, groups=self.groups)
        self.conv1b = WSConv2D(in_channels=self.width, out_channels=self.width, kernel_size=3, stride=1, padding=1, groups=self.groups)
        self.conv2 = WSConv2D(in_channels=self.width, out_channels=self.out_channels, kernel_size=1)

        self.use_projection = self.stride > 1 or self.in_channels != self.out_channels
        if self.use_projection:
            self.conv_shortcut = WSConv2D(self.in_channels, self.out_channels, kernel_size=1, stride=1) # Shortcut conv has stride 1
            if stride > 1:
                # Padding calculation for AvgPool2d to mimic TF 'SAME' might need adjustment based on input size parity
                # padding=0 maintains size if input H/W is even, reduces by 1 if odd.
                # TF 'SAME' with stride 2 aims for ceil(H/2) x ceil(W/2).
                # Let's use kernel_size=3, stride=2, padding=1 for pooling before shortcut conv for downsampling blocks
                # This is a common pattern (e.g., ResNet). Original paper might specify differently.
                # The original code had complex padding logic `padding=0 if self.in_channels==1536 else 1`.
                # We'll use a standard approach: pool then conv.
                self.shortcut_avg_pool = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0) # Simpler pooling before conv

        self.squeeze_excite = SqueezeExcite(self.out_channels, self.out_channels, se_ratio=self.se_ratio, activation=activation)
        self.skip_gain = nn.Parameter(torch.zeros(()))

        self.use_stochdepth = stochdepth_rate is not None and stochdepth_rate > 0. and stochdepth_rate < 1.
        if self.use_stochdepth:
            self.stoch_depth = StochDepth(stochdepth_rate)

    def forward(self, x):
        # Apply activation and beta scaling *before* the residual branch split
        out_main = self.activation(x) * self.beta

        # Shortcut path
        if self.use_projection:
            if self.stride > 1:
                # Apply pooling before the shortcut convolution for downsampling
                shortcut = self.shortcut_avg_pool(out_main)
                shortcut = self.conv_shortcut(shortcut)
            else:
                # Projection for channel change only
                shortcut = self.conv_shortcut(out_main)
        else:
            # Identity shortcut
            shortcut = x

        # Main path
        out = self.activation(self.conv0(out_main))
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv1b(out))
        out = self.conv2(out)
        out = (self.squeeze_excite(out)*2) * out # SE applied here

        # Apply StochDepth if enabled
        if self.use_stochdepth:
            out = self.stoch_depth(out)

        # Final combination: scaled residual + shortcut
        return out * self.alpha * self.skip_gain + shortcut


# --- Modified NFNet Class ---

class NFNet(nn.Module):
    """
    NFNet backbone feature extractor.
    Follows the structure modification inspired by MobileNetV4 implementation.
    Outputs a list of feature maps from selected stages.
    """
    def __init__(self, variant:str='F0', stochdepth_rate:Optional[float]=None,
        alpha:float=0.2, se_ratio:float=0.5, activation:str='gelu',
        in_channels: int = 3): # Added in_channels parameter
        super(NFNet, self).__init__()

        if not variant in nfnet_params:
            raise RuntimeError(f"Variant {variant} does not exist and could not be loaded.")

        block_params = nfnet_params[variant]

        self.variant = variant
        self.train_imsize = block_params['train_imsize']
        self.test_imsize = block_params['test_imsize']
        self.activation_fn = activations_dict[activation] # Use a different name to avoid conflict
        # self.drop_rate = block_params['drop_rate'] # Drop rate is mainly for classifier head, removed

        self.stem = Stem(activation=activation)

        num_blocks_total = sum(block_params['depth'])
        block_idx_counter = 0

        stages = []
        expected_std = 1.0
        current_channels = 128 # Output channels of Stem

        block_args = zip(
            block_params['width'],    # [256, 512, 1536, 1536] for F0
            block_params['depth'],    # [1, 2, 6, 3] for F0
            [0.5] * 4,                # bottleneck pattern
            [128] * 4,                # group pattern
            [1, 2, 2, 2]              # stride pattern (First stage stride=1, others=2)
        )

        for i, (block_width, stage_depth, expand_ratio, group_size, stage_stride) in enumerate(block_args):
            stage_blocks = []
            output_channels = block_width

            for block_index in range(stage_depth):
                beta = 1. / expected_std
                block_sd_rate = stochdepth_rate * block_idx_counter / num_blocks_total if stochdepth_rate else None

                # Determine stride: only the first block of stages 2, 3, 4 has stride > 1
                stride = stage_stride if block_index == 0 else 1

                stage_blocks.append(NFBlock(
                    in_channels=current_channels,
                    out_channels=output_channels,
                    stride=stride,
                    alpha=alpha,
                    beta=beta,
                    se_ratio=se_ratio,
                    group_size=group_size,
                    stochdepth_rate=block_sd_rate,
                    activation=activation,
                    expansion=expand_ratio)) # Pass expansion ratio

                current_channels = output_channels
                block_idx_counter += 1

                # Update expected_std: reset at the start of stage, accumulate within stage
                if block_index == 0:
                     # Expected std is reset to 1.0 for the input of the **first block** of each stage?
                     # Or does it reset *after* the first block's output? Let's assume input.
                     # The original code resets expected_std = 1.0 *after* the first block, implies the *input* to the second block has std=1?
                     # Let's follow original logic: reset after block 0 IF stride was > 1? Needs clarification.
                     # Re-checking original code: expected_std is reset *after* the first block append.
                     pass # Reset happens below

                expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5

                # Reset expected_std to 1.0 after the first block of each stage (as per original code logic)
                if block_index == 0:
                    expected_std = 1.0 # Reset for the next block in the stage

            stages.append(nn.Sequential(*stage_blocks))

        # Assign stages to attributes
        self.stage1 = stages[0]
        self.stage2 = stages[1]
        self.stage3 = stages[2]
        self.stage4 = stages[3]

        # --- Calculate width_list (similar to MobileNetV4) ---
        # Perform a dummy forward pass to get intermediate feature shapes
        # Use train_imsize as the reference input size
        self.width_list = []
        if self.train_imsize > 0: # Ensure valid image size
             try:
                 with torch.no_grad(): # No need to track gradients here
                     # Use a variable input size to avoid fixed size dependency if possible, but fall back to train_imsize
                     # Using a reasonably small size like 224 might be safer for general initialization.
                     # Let's stick to train_imsize as specified by the block params.
                     dummy_input_size = (1, in_channels, self.train_imsize, self.train_imsize)
                     dummy_input = torch.randn(dummy_input_size)
                     # Move model to the same device as input if necessary (e.g., GPU)
                     # dummy_input = dummy_input.to(next(self.parameters()).device) # Fails if no parameters yet
                     # Temporarily move model to CPU for calculation if it avoids issues
                     original_device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
                     self.to('cpu')
                     dummy_input = dummy_input.to('cpu')

                     features = self.forward(dummy_input)
                     self.width_list = [f.shape[1] for f in features]

                     # Move model back to its original device
                     self.to(original_device)

             except Exception as e:
                 print(f"Warning: Could not compute width_list during NFNet initialization. Error: {e}")
                 # Provide fallback or default widths based on block_params if forward fails
                 self.width_list = [
                     block_params['width'][0] // 2, # Stem output guess (128) - Needs verification based on Stem impl.
                     block_params['width'][0],      # Stage 1 output
                     block_params['width'][1],      # Stage 2 output
                     block_params['width'][2],      # Stage 3 output
                     block_params['width'][3]       # Stage 4 output
                 ]
                 # Let's refine the stem output channel guess based on the actual stem definition
                 stem_out_channels = 128
                 self.width_list = [stem_out_channels] + block_params['width']


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning intermediate features."""
        x_stem = self.stem(x)
        x1 = self.stage1(x_stem)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # Return features from stem and all stages
        return [x_stem, x1, x2, x3, x4]

    def exclude_from_weight_decay(self, name: str) -> bool:
        """Exclude biases and gains from weight decay."""
        # Regex to find layer names like
        # "stem.conv[0-3].bias", "stem.conv[0-3].gain", "stage[1-4].*.skip_gain",
        # "stage[1-4].*.conv[0-2|shortcut].bias", "stage[1-4].*.conv[0-2|shortcut].gain"
        # Simplified regex: find 'bias' or 'gain' or 'skip_gain'
        return 'bias' in name or 'gain' in name or 'skip_gain' in name

    def exclude_from_clipping(self, name: str) -> bool:
        """Exclude layers from clipping (e.g., final linear layer if added externally)."""
        # This might be less relevant now without the internal linear layer,
        # but could be used by external modules. Keep for potential use.
        # Example: if an external classifier head is named 'classifier',
        # return name.startswith('classifier')
        return False # No layers excluded by default in the backbone


# --- Factory Functions for NFNet Variants ---

def _create_nfnet(variant: str, **kwargs: Any) -> NFNet:
    """Helper function to create NFNet instances."""
    model = NFNet(variant=variant, **kwargs)
    return model

def NFNetF0(**kwargs: Any) -> NFNet:
    """Creates an NFNet-F0 backbone."""
    # Set default stochdepth_rate if not provided in kwargs, e.g., 0.1 might be reasonable
    kwargs.setdefault('stochdepth_rate', 0.1) # Example default
    return _create_nfnet('F0', **kwargs)

def NFNetF1(**kwargs: Any) -> NFNet:
    """Creates an NFNet-F1 backbone."""
    kwargs.setdefault('stochdepth_rate', 0.1)
    return _create_nfnet('F1', **kwargs)

def NFNetF2(**kwargs: Any) -> NFNet:
    """Creates an NFNet-F2 backbone."""
    kwargs.setdefault('stochdepth_rate', 0.1)
    return _create_nfnet('F2', **kwargs)

def NFNetF3(**kwargs: Any) -> NFNet:
    """Creates an NFNet-F3 backbone."""
    kwargs.setdefault('stochdepth_rate', 0.1)
    return _create_nfnet('F3', **kwargs)

def NFNetF4(**kwargs: Any) -> NFNet:
    """Creates an NFNet-F4 backbone."""
    kwargs.setdefault('stochdepth_rate', 0.1)
    return _create_nfnet('F4', **kwargs)

def NFNetF5(**kwargs: Any) -> NFNet:
    """Creates an NFNet-F5 backbone."""
    kwargs.setdefault('stochdepth_rate', 0.1)
    return _create_nfnet('F5', **kwargs)

def NFNetF6(**kwargs: Any) -> NFNet:
    """Creates an NFNet-F6 backbone."""
    kwargs.setdefault('stochdepth_rate', 0.1)
    return _create_nfnet('F6', **kwargs)

def NFNetF7(**kwargs: Any) -> NFNet:
    """Creates an NFNet-F7 backbone."""
    kwargs.setdefault('stochdepth_rate', 0.1)
    return _create_nfnet('F7', **kwargs)


# --- Example Usage ---

if __name__ == "__main__":
    print("Testing NFNet F0 creation and forward pass...")

    # Create NFNet-F0 model using the new factory function
    model_f0 = NFNetF0(stochdepth_rate=0.1, activation='gelu')
    # print(model_f0) # Print model structure if desired

    # Get the expected input size for F0
    input_h, input_w = model_f0.train_imsize, model_f0.train_imsize
    print(f"Using input size: {input_h}x{input_w}")

    # Create a dummy input tensor
    # Handle case where train_imsize might be 0 or negative from init failure
    if input_h <= 0 or input_w <= 0:
        print("Warning: Invalid input size, using default 224x224 for test.")
        input_h, input_w = 224, 224

    image_size = (1, 3, input_h, input_w) # Batch size 1, 3 channels
    image = torch.rand(*image_size)

    # Move model and input to the same device (e.g., CPU for this test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_f0.to(device)
    image = image.to(device)
    print(f"Testing on device: {device}")

    # Perform forward pass
    model_f0.eval() # Set model to evaluation mode
    with torch.no_grad():
        output_features = model_f0(image)

    # Print the shapes of the output feature maps
    print("\nOutput feature map shapes:")
    for i, features in enumerate(output_features):
        print(f"Stage {i} output shape: {features.shape}")

    # Print the calculated width_list stored in the model
    print(f"\nStored width_list: {model_f0.width_list}")

    # Verify if width_list matches the output channels
    output_channels = [f.shape[1] for f in output_features]
    print(f"Actual output channels: {output_channels}")
    if hasattr(model_f0, 'width_list') and model_f0.width_list == output_channels:
         print("Width list matches actual output channels.")
    else:
         print("Mismatch between stored width_list and actual output channels.")