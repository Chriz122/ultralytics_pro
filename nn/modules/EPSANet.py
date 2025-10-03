import torch
import torch.nn as nn
import math
from typing import List, Dict, Optional

# --- Helper Functions and Modules (Unchanged) ---
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SEWeightModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        return weight

class PSAModule(nn.Module):
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        # Concatenate weighted features directly is simpler
        out_list = []
        for i in range(4):
            out_list.append(feats_weight[:, i, :, :, :])
        out = torch.cat(out_list, dim=1)

        return out

class EPSABlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, conv_kernels=[3, 5, 7, 9],
                 conv_groups=[1, 4, 8, 16]):
        super(EPSABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = PSAModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# --- EPSANet Configurations ---
EPSANET_CONFIGS: Dict[str, Dict] = {
    "50": {"block": EPSABlock, "layers": [3, 4, 6, 3]},
    "101": {"block": EPSABlock, "layers": [3, 4, 23, 3]},
    # Can add more variants here if needed
}

# --- Modified EPSANet Class ---
class EPSANet(nn.Module):
    def __init__(self, variant: str = "50", norm_layer: Optional[nn.Module] = None):
        """
        EPSANet backbone modified to follow MobileNetV4 structure.

        Args:
            variant (str): Specifies the EPSANet variant ('50', '101').
                           Defaults to "50".
            norm_layer (Optional[nn.Module]): Normalization layer to use.
                                               Defaults to nn.BatchNorm2d.
        """
        super(EPSANet, self).__init__()
        if variant not in EPSANET_CONFIGS:
            raise ValueError(f"Unknown EPSANet variant: {variant}. Available: {list(EPSANET_CONFIGS.keys())}")

        config = EPSANET_CONFIGS[variant]
        block = config["block"]
        layers = config["layers"]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1 # Not used in standard ResNet but keep for potential future compatibility

        # Initial stem
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Main layers (stages)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Initialize weights
        self._initialize_weights()

        # Calculate output channel dimensions (similar to MobileNetV4)
        # Use a smaller dummy input size common for classification backbones
        # Ensure the forward pass is defined correctly before this runs
        self.eval() # Set to eval mode for dummy forward pass
        try:
            # Important: Ensure the forward method returns a list of features
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad(): # No need to track gradients
                 features = self.forward(dummy_input)
            self.width_list = [f.size(1) for f in features]
        except Exception as e:
            print(f"Warning: Could not compute width_list during init: {e}")
            self.width_list = [] # Set to empty list or handle appropriately
        self.train() # Set back to train mode

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate: # Not used in standard config but keep structure
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer)) # Pass norm_layer
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer)) # Pass norm_layer

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                 m.weight.data.normal_(0, 0.01)
                 nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass for feature extraction.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, H, W).

        Returns:
            List[torch.Tensor]: List of feature maps from layer1 to layer4.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x) # Output after stem relu
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Return intermediate features similar to Code 2's MobileNetV4
        return [x1, x2, x3, x4]

# --- Helper Functions for Instantiation (Mimicking Code 2) ---
def epsanet50() -> EPSANet:
    """Constructs an EPSANet-50 backbone."""
    model = EPSANet(variant="50")
    return model

def epsanet101() -> EPSANet:
    """Constructs an EPSANet-101 backbone."""
    model = EPSANet(variant="101")
    return model

# --- Example Usage ---
if __name__ == "__main__":
    # Instantiate EPSANet-50
    model50 = epsanet50()
    print("EPSANet-50 Instantiated.")
    print(f"Output channels (width_list): {model50.width_list}")

    # Instantiate EPSANet-101
    model101 = epsanet101()
    print("\nEPSANet-101 Instantiated.")
    print(f"Output channels (width_list): {model101.width_list}")


    # Example forward pass with EPSANet-50
    print("\nTesting forward pass with EPSANet-50...")
    # Use a spatial size compatible with the strides (e.g., 224x224)
    dummy_image = torch.rand(2, 3, 224, 224)
    model50.eval() # Set to evaluation mode for inference
    with torch.no_grad():
        features = model50(dummy_image)

    print("Output features shapes:")
    for i, f in enumerate(features):
        print(f"Layer {i+1}: {f.shape}")

    # Verify width_list matches the output channels
    calculated_widths = [f.shape[1] for f in features]
    print(f"\nCalculated widths from forward pass: {calculated_widths}")
    print(f"Stored width_list: {model50.width_list}")
    assert model50.width_list == calculated_widths, "Mismatch between stored width_list and forward pass output channels!"
    print("Width list verified.")