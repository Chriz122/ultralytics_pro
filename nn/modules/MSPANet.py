# -*- coding: utf-8 -*-
import torch.nn as nn
import math
import torch
from typing import List # Import List for type hinting

__all__ = ['MSPANet', 'mspanet50', 'mspanet101'] # Corrected __all__


# --- Helper Convolution Functions (Unchanged) ---
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def convdilated(in_planes, out_planes, kSize=3, stride=1, dilation=1):
    """3x3 convolution with dilation"""
    padding = int((kSize - 1) / 2) * dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=kSize, stride=stride, padding=padding,
                     dilation=dilation, bias=False)


# --- SPRModule (Corrected) ---
class SPRModule(nn.Module):
    def __init__(self, channels, reduction=16): # 'channels' is defined here in init
        super(SPRModule, self).__init__()
        self.channels = channels # Store channels as an instance attribute

        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)

        # Use self.channels here for consistency
        self.fc1 = nn.Conv2d(self.channels * 5, self.channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(self.channels // reduction, self.channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x has shape (batch, channels, H, W)
        # Get the actual channel size from the stored 'self.channels'
        batch_size = x.size(0)
        input_channels = self.channels # Use the stored channel value

        out1 = self.avg_pool1(x) # Shape: (batch, input_channels, 1, 1)
        out2 = self.avg_pool2(x) # Shape: (batch, input_channels, 2, 2)

        out1_flat = out1.view(batch_size, input_channels, -1) # Shape: (batch, input_channels, 1)
        out2_flat = out2.view(batch_size, input_channels, -1) # Shape: (batch, input_channels, 4)

        out = torch.cat((out1_flat, out2_flat), 2) # Shape: (batch, input_channels, 5)

        # *** Correction: Use 'input_channels' (derived from self.channels) ***
        # Reshape for Conv2d: (batch, input_channels*5, 1, 1)
        out = out.view(batch_size, input_channels * 5, 1, 1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out) # Shape: (batch, input_channels, 1, 1)

        return weight


# --- MSAModule (Unchanged from previous version) ---
class MSAModule(nn.Module):
    def __init__(self, inplanes, scale=3, stride=1, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality (width per scale).
            scale: number of scale.
            stride: conv stride.
            stype: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(MSAModule, self).__init__()

        self.width = inplanes # This is the width *per scale*
        self.nums = scale
        self.stride = stride
        assert stype in ['stage', 'normal'], 'One of these is suppported (stage or normal)'
        self.stype = stype

        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])

        for i in range(self.nums):
            if self.stype == 'stage' and self.stride != 1:
                self.convs.append(convdilated(self.width, self.width, kSize=3, stride=stride, dilation=int(i + 1)))
            else:
                self.convs.append(conv3x3(self.width, self.width, stride))

            self.bns.append(nn.BatchNorm2d(self.width))

        # Attention module operates on the width *per scale*
        self.attention = SPRModule(self.width) # Pass width per scale

        self.softmax = nn.Softmax(dim=1) # Softmax over scales

    def forward(self, x):
        # Input x has shape (batch_size, width * scale, H, W)
        batch_size = x.shape[0]

        spx = torch.split(x, self.width, 1) # List of tensors, each (batch, width, H, W)

        processed_scales = []
        for i in range(self.nums):
            if i == 0 or (self.stype == 'stage' and self.stride != 1):
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](sp)
            processed_scales.append(sp)

        out = torch.cat(processed_scales, 1) # Shape: (batch, width * scale, H', W')

        # --- Attention Mechanism ---
        feats = out.view(batch_size, self.nums, self.width, out.shape[2], out.shape[3])

        attn_weight = []
        sp_inp = torch.split(out, self.width, 1) # Split the concatenated output
        for inp in sp_inp:
             # self.attention expects input with 'self.width' channels
            attn_weight.append(self.attention(inp)) # Each weight is (batch, self.width, 1, 1)

        # Concatenate attention weights
        # Resulting shape: (batch, self.width * self.nums, 1, 1)
        attn_weight = torch.cat(attn_weight, dim=1)

        # Reshape weights to match feature dimensions for broadcasting
        # Target: (batch, nums, width, 1, 1)
        attn_vectors = attn_weight.view(batch_size, self.nums, self.width, 1, 1)

        attn_vectors = self.softmax(attn_vectors)

        feats_weight = feats * attn_vectors # Element-wise multiplication

        output_scales = []
        for i in range(self.nums):
            x_attn_weight = feats_weight[:, i, :, :, :]
            output_scales.append(x_attn_weight)

        out = torch.cat(output_scales, 1) # Concatenate along channel dim -> (batch, width * scale, H', W')

        return out


# --- MSPABlock (Unchanged from previous version) ---
class MSPABlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=30, scale=3,
                 norm_layer=None, stype='normal'):
        super(MSPABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(math.floor(planes * (baseWidth / 64.0)))
        total_width = width * scale

        self.conv1 = conv1x1(inplanes, total_width)
        self.bn1 = norm_layer(total_width)
        self.relu = nn.ReLU(inplace=True) # Added missing relu after bn1

        # MSAModule takes width per scale as 'inplanes' argument
        self.conv2 = MSAModule(width, scale=scale, stride=stride, stype=stype)
        self.bn2 = norm_layer(total_width)
        # self.relu = nn.ReLU(inplace=True) # Relu is applied after residual add

        self.conv3 = conv1x1(total_width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out) # Apply relu here

        out = self.conv2(out)
        out = self.bn2(out)
        # No relu here, apply after residual

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out) # Apply final relu

        return out


# --- Main MSPANet Class (Unchanged from previous version regarding width_list) ---
class MSPANet(nn.Module):
    def __init__(self, block, layers, baseWidth=30, scale=3, norm_layer=None):
        super(MSPANet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self._initialize_weights()

        # --- Calculate width_list (will now work correctly) ---
        self.width_list = [] # Initialize as empty list
        try:
            # Store original training mode
            original_mode = self.training
            self.eval() # Set to eval mode for consistent behavior (esp. BN)
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224) # Use a standard size
                features = self._forward_extract(dummy_input)
                self.width_list = [f.size(1) for f in features]
            # Restore original training mode
            self.train(original_mode)
        except Exception as e:
            print(f"Warning: Could not compute width_list during init: {e}")
            # Optionally, print traceback:
            # import traceback
            # traceback.print_exc()
            self.width_list = [] # Keep it empty on failure
        # --- End of width_list calculation ---


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            baseWidth=self.baseWidth, scale=self.scale, norm_layer=norm_layer, stype='stage'))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale,
                                norm_layer=norm_layer, stype='normal'))

        return nn.Sequential(*layers)

    def _forward_extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        """ Internal forward pass for feature extraction """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """ Main forward pass, returning intermediate features """
        return self._forward_extract(x)


# --- Helper functions (Unchanged) ---
def mspanet50(**kwargs) -> MSPANet:
    defaults = {'baseWidth': 30, 'scale': 3}
    defaults.update(kwargs)
    model = MSPANet(MSPABlock, [3, 4, 6, 3], **defaults)
    return model


def mspanet101(**kwargs) -> MSPANet:
    defaults = {'baseWidth': 30, 'scale': 3}
    defaults.update(kwargs)
    model = MSPANet(MSPABlock, [3, 4, 23, 3], **defaults)
    return model


# --- Example Usage (Should now run without NameError) ---
if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 3, 640, 640)
    image = torch.rand(*image_size)

    # --- Test mspanet50 ---
    print("--- Testing MSPANet-50 ---")
    model50 = mspanet50()
    print(f"MSPANet-50 width_list: {model50.width_list}") # Should now be populated
    try:
        output_features_50 = model50(image)
        print("MSPANet-50 Output Feature Shapes:")
        for i, feat in enumerate(output_features_50):
            print(f"Layer {i+1} output shape: {feat.shape}")
    except Exception as e:
        print(f"Error during MSPANet-50 forward pass: {e}")
        import traceback
        traceback.print_exc()


    print("\n" + "="*30 + "\n")

    # --- Test mspanet101 ---
    print("--- Testing MSPANet-101 ---")
    model101 = mspanet101()
    print(f"MSPANet-101 width_list: {model101.width_list}") # Should now be populated
    try:
        output_features_101 = model101(image)
        print("MSPANet-101 Output Feature Shapes:")
        for i, feat in enumerate(output_features_101):
            print(f"Layer {i+1} output shape: {feat.shape}")
    except Exception as e:
        print(f"Error during MSPANet-101 forward pass: {e}")
        import traceback
        traceback.print_exc()