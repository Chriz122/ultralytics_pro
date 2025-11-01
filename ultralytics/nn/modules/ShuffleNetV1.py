import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable # Kept as in original code 1, although not explicitly used after modification
from collections import OrderedDict
from torch.nn import init


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    """3x3 convolution with padding
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def conv1x1(in_channels, out_channels, groups=1):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
        channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3,
                 grouped_conv=True, combine='add'):

        super(ShuffleUnit, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4

        # define the type of ShuffleUnit
        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
            self._combine_func = self._add
            # Ensure input and output channels are the same for residual connection
            assert self.in_channels == self.out_channels, \
                "Input and output channels must be same for 'add' combine type"
        elif self.combine == 'concat':
            # ShuffleUnit Figure 2c
            self.depthwise_stride = 2
            self._combine_func = self._concat

            # ensure output of concat has the same channels as
            # original output channels. For 'concat', the output channel
            # is the sum of residual (input channel) and block output channel.
            # So the block itself only outputs self.out_channels - self.in_channels
            # before concat.
            self.block_out_channels = self.out_channels - self.in_channels
            self.bottleneck_channels = self.block_out_channels // 4 # Bottleneck relative to block output
        else:
            raise ValueError("Cannot combine tensors with \"{}\"" \
                             "Only \"add\" and \"concat\" are" \
                             "supported".format(self.combine))

        # Use a 1x1 grouped or non-grouped convolution to reduce input channels
        # to bottleneck channels, as in a ResNet bottleneck module.
        # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
        self.first_1x1_groups = self.groups if grouped_conv else 1

        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            self.in_channels,
            self.bottleneck_channels,
            self.first_1x1_groups,
            batch_norm=True,
            relu=True
            )

        # 3x3 depthwise convolution followed by batch normalization
        self.depthwise_conv3x3 = conv3x3(
            self.bottleneck_channels, self.bottleneck_channels,
            stride=self.depthwise_stride, groups=self.bottleneck_channels)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)

        # Use 1x1 grouped convolution to expand from
        # bottleneck_channels to out_channels
        expand_out_channels = self.out_channels if self.combine == 'add' else self.block_out_channels
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(
            self.bottleneck_channels,
            expand_out_channels,
            self.groups,
            batch_norm=True,
            relu=False # No ReLU after last conv in residual block before add/concat
            )


    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out


    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)


    def _make_grouped_conv1x1(self, in_channels, out_channels, groups,
        batch_norm=True, relu=False):

        modules = OrderedDict()

        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU(inplace=True) # Added inplace=True

        return nn.Sequential(modules) if len(modules) > 1 else conv


    def forward(self, x):
        # save for combining later with output
        residual = x

        if self.combine == 'concat':
            # Downsample residual branch for concat unit
            residual = F.avg_pool2d(residual, kernel_size=3,
                stride=2, padding=1)
            # The number of channels in the residual branch is the input channels
            # Need to handle potential channel mismatch in concat if input channels != block_out_channels + input_channels
            # Wait, in concat unit, the output channels is `self.stage_out_channels[stage]`.
            # The residual branch contributes `self.stage_out_channels[stage-1]` channels after pooling.
            # The block branch must output `self.stage_out_channels[stage] - self.stage_out_channels[stage-1]` channels.
            # Let's re-calculate block_out_channels for concat type
            self.block_out_channels = self.out_channels - self.in_channels
            self.bottleneck_channels = self.block_out_channels // 4
            # Re-create the expansion layer in forward for concat only? No, better adjust in init.
            # Let's fix the init based on the paper's figure 2c
            # In concat unit:
            # Input: C_in
            # 1x1 GConv -> C_bottleneck
            # Shuffle
            # 3x3 DWConv (stride 2) -> C_bottleneck
            # BN
            # 1x1 GConv -> C_out - C_in
            # Add with AVG Pool(C_in)
            # ReLU
            # Total output channels = C_out
            # This means the expand_out_channels in init should be self.out_channels - self.in_channels for concat.
            # And bottleneck_channels should be (self.out_channels - self.in_channels) // 4.
            # Yes, the init logic was already doing this. Let's trust the original calculation.

        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out) # Output channels here is self.out_channels (add) or self.block_out_channels (concat)

        # The combination adds or concatenates 'residual' and 'out'.
        # After combination, ReLU is applied.
        combined_out = self._combine_func(residual, out)
        return F.relu(combined_out, inplace=True) # Added inplace=True


class ShuffleNet(nn.Module):
    """ShuffleNet implementation as a feature extractor backbone.
       Modified based on Code 2 to return feature maps and support factory functions.
    """

    def __init__(self, groups=3, in_channels=3, input_size=(224, 224)):
        """ShuffleNet constructor for feature extraction.

        Arguments:
            groups (int, optional): number of groups to be used in grouped
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            input_size (tuple, optional): Dummy input size (H, W) used to calculate
                width_list. Default is (224, 224).
        """
        super(ShuffleNet, self).__init__()

        self.groups = groups
        self.stage_repeats = [3, 7, 3] # Repeats for Stage 2, 3, 4 respectively
        self.in_channels =  in_channels
        # Note: num_classes and final fc layer are removed for feature extraction

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        # These are output channels *after* each stage
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))

        # Stage 1: Conv1 and MaxPool
        # Output channels of Stage 1 is stage_out_channels[1] (which is always 24)
        self.conv1 = conv3x3(self.in_channels,
                             self.stage_out_channels[1], # stage 1 output = 24
                             stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2
        # Input channels: self.stage_out_channels[1] (24)
        # Output channels: self.stage_out_channels[2]
        self.stage2 = self._make_stage(2, self.stage_out_channels[1], self.stage_out_channels[2])
        # Stage 3
        # Input channels: self.stage_out_channels[2]
        # Output channels: self.stage_out_channels[3]
        self.stage3 = self._make_stage(3, self.stage_out_channels[2], self.stage_out_channels[3])
        # Stage 4
        # Input channels: self.stage_out_channels[3]
        # Output channels: self.stage_out_channels[4]
        self.stage4 = self._make_stage(4, self.stage_out_channels[3], self.stage_out_channels[4])

        # Removed global pooling and final classification layer

        # Calculate width_list based on dummy input (similar to Code 2)
        self.width_list = self._calculate_width_list(input_size)

        self.init_params()


    def init_params(self):
        # Initialize parameters, excluding the removed Linear layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use modern init functions
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            # Removed initialization for nn.Linear


    def _make_stage(self, stage_idx, in_channels, out_channels):
        modules = OrderedDict()
        stage_name = "ShuffleUnit_Stage{}".format(stage_idx)

        # First ShuffleUnit in the stage is always 'concat' type and uses stride 2
        # It reduces resolution by half and increases channels from in_channels to out_channels
        # For Stage 2, the first 1x1 conv is non-grouped. For Stage 3 and 4, it's grouped.
        grouped_conv = stage_idx > 2 # Grouped conv for first 1x1 only in Stage 3 and 4

        first_module = ShuffleUnit(
            in_channels=in_channels,
            out_channels=out_channels, # Total output channels after concat
            groups=self.groups,
            grouped_conv=grouped_conv, # Non-grouped for Stage 2, Grouped for Stage 3/4
            combine='concat'
            )
        modules[stage_name+"_0"] = first_module

        # Add more ShuffleUnits depending on pre-defined number of repeats
        # These units are 'add' type, maintain resolution and channel count
        # They always use grouped 1x1 convolutions
        num_repeats = self.stage_repeats[stage_idx-2] # stage_idx 2, 3, 4 maps to repeats index 0, 1, 2
        for i in range(num_repeats):
            name = stage_name + "_{}".format(i+1)
            module = ShuffleUnit(
                in_channels=out_channels, # Input and output channels are the same for 'add'
                out_channels=out_channels,
                groups=self.groups,
                grouped_conv=True, # Always grouped conv for 1x1s in 'add' units
                combine='add'
                )
            modules[name] = module

        return nn.Sequential(modules)

    def _calculate_width_list(self, input_size):
        """Helper to calculate output channel sizes using a dummy input."""
        with torch.no_grad(): # Ensure no gradients are computed
            dummy_input = torch.randn(1, self.in_channels, input_size[0], input_size[1])
            x = self.conv1(dummy_input)
            x = self.maxpool(x)
            out_maxpool = x # Output after initial layers

            x = self.stage2(x)
            out_stage2 = x # Output after Stage 2

            x = self.stage3(x)
            out_stage3 = x # Output after Stage 3

            x = self.stage4(x)
            out_stage4 = x # Output after Stage 4

            # Return the channel counts of the feature maps at each desired stage
            return [out_maxpool.size(1), out_stage2.size(1), out_stage3.size(1), out_stage4.size(1)]


    def forward(self, x):
        # This forward pass is modified to return a list of feature maps
        # at different stages, suitable for feature extraction tasks.

        x = self.conv1(x)
        x = self.maxpool(x)
        out_maxpool = x # Output after initial layers

        x = self.stage2(x)
        out_stage2 = x # Output after Stage 2

        x = self.stage3(x)
        out_stage3 = x # Output after Stage 3

        x = self.stage4(x)
        out_stage4 = x # Output after Stage 4

        # Return a list of feature maps from different stages
        # The stages correspond to different spatial resolutions
        return [out_maxpool, out_stage2, out_stage3, out_stage4]


# Factory functions for different ShuffleNet group sizes
def ShuffleNetG1(in_channels=3, input_size=(224, 224)):
    """ShuffleNet with groups=1"""
    return ShuffleNet(groups=1, in_channels=in_channels, input_size=input_size)

def ShuffleNetG2(in_channels=3, input_size=(224, 224)):
    """ShuffleNet with groups=2"""
    return ShuffleNet(groups=2, in_channels=in_channels, input_size=input_size)

def ShuffleNetG3(in_channels=3, input_size=(224, 224)):
    """ShuffleNet with groups=3"""
    return ShuffleNet(groups=3, in_channels=in_channels, input_size=input_size)

def ShuffleNetG4(in_channels=3, input_size=(224, 224)):
    """ShuffleNet with groups=4"""
    return ShuffleNet(groups=4, in_channels=in_channels, input_size=input_size)

def ShuffleNetG8(in_channels=3, input_size=(224, 224)):
    """ShuffleNet with groups=8"""
    return ShuffleNet(groups=8, in_channels=in_channels, input_size=input_size)


if __name__ == "__main__":
    """Testing
    """
    # Example usage similar to Code 2's test block
    # Use a common image size like 224x224 or 640x640
    dummy_input_size = (1, 3, 224, 224)
    dummy_input = torch.randn(*dummy_input_size)

    # --- Test ShuffleNetG3 (default groups=3) ---
    print("-" * 20)
    print("Testing ShuffleNetG3")
    model_g3 = ShuffleNetG3(input_size=(dummy_input_size[2], dummy_input_size[3]))
    print(f"Created ShuffleNet with {model_g3.groups} groups")
    print(f"Calculated width_list: {model_g3.width_list}")

    print(f"Input shape: {dummy_input.shape}")
    outputs_g3 = model_g3(dummy_input)

    print("Output feature map shapes:")
    for i, output in enumerate(outputs_g3):
        print(f"Output {i}: {output.shape}")

    # Verify width_list matches output channels
    print("Verifying width_list against output shapes:")
    for i, width in enumerate(model_g3.width_list):
         print(f"width_list[{i}] ({width}) == output[{i}].size(1) ({outputs_g3[i].size(1)})")
    print("-" * 20)


    # --- Test ShuffleNetG1 ---
    print("Testing ShuffleNetG1")
    model_g1 = ShuffleNetG1(input_size=(dummy_input_size[2], dummy_input_size[3]))
    print(f"Created ShuffleNet with {model_g1.groups} groups")
    print(f"Calculated width_list: {model_g1.width_list}")

    print(f"Input shape: {dummy_input.shape}")
    outputs_g1 = model_g1(dummy_input)

    print("Output feature map shapes:")
    for i, output in enumerate(outputs_g1):
        print(f"Output {i}: {output.shape}")
    print("-" * 20)