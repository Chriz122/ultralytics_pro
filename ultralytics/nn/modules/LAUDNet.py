import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.nn.functional as F
from typing import List, Tuple, Optional # Added List, Tuple, Optional

# from torchvision.models import resnet50, ResNet50_Weights


__all__ = ['uni_resnet50', 'uni_resnet101', 'ResNet'] # Added ResNet to __all__ for direct import if needed


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def apply_channel_mask(x, mask):
    b, c, h, w = x.shape
    if mask is None: # Allow None mask for non-dynamic paths
        return x
    _, g = mask.shape
    if (g > 1) and (g != c):
        mask = mask.repeat(1,c//g).view(b, c//g, g).transpose(-1,-2).reshape(b,c,1,1)
    else:
        mask = mask.view(b,g,1,1)
    return x * mask

def apply_spatial_mask(x, mask):
    if mask is None: # Allow None mask
        return x
    b, c, h, w = x.shape
    _, g, hw_mask, _ = mask.shape
    if (g > 1) and (g != c):
        mask = mask.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,hw_mask,hw_mask)
    return x * mask

class Masker_spatial(nn.Module):
    def __init__(self, in_channels, mask_channel_group, mask_size):
        super(Masker_spatial, self).__init__()
        self.mask_channel_group = mask_channel_group
        self.mask_size = mask_size
        self.conv = conv1x1(in_channels, mask_channel_group*2,bias=True)
        self.conv_flops_pp = self.conv.weight.shape[0] * self.conv.weight.shape[1] + (self.conv.weight.shape[1] if self.conv.bias is not None else 0) # Corrected bias flops
        if self.conv.bias is not None:
            self.conv.bias.data[:mask_channel_group] = 5.0
            self.conv.bias.data[mask_channel_group:] = 0.0 # Corrected slicing for bias init

    def forward(self, x, temperature):
        if self.mask_size == 0: # Handle case where masking is effectively disabled by mask_size
            b, c, h, w = x.shape
            # Return a dummy mask that doesn't alter the input, and zero sparsity/flops
            dummy_mask = torch.ones(b, self.mask_channel_group, 1, 1, device=x.device) 
            return dummy_mask, torch.tensor(1.0, device=x.device), 0.0

        mask =  F.adaptive_avg_pool2d(x, self.mask_size) if self.mask_size < x.shape[2] else x
        flops = mask.shape[0] * mask.shape[1] * mask.shape[2] * mask.shape[3] # Input elements for AAP
        
        mask = self.conv(mask)
        flops += self.conv_flops_pp * mask.shape[2] * mask.shape[3]
        
        b,c_out,h_mask,w_mask = mask.shape # c_out is mask_channel_group*2
        mask = mask.view(b,2,c_out//2,h_mask,w_mask)
        if self.training:
            mask = F.gumbel_softmax(mask, dim=1, tau=temperature, hard=True)
            mask = mask[:,0]
        else:
            mask = (mask[:,0]>=mask[:,1]).float()
        sparsity = mask.mean()
        return mask, sparsity, flops

class ExpandMask(nn.Module):
    def __init__(self, stride, padding=1, mask_channel_group=1): 
        super(ExpandMask, self).__init__()
        self.stride=stride
        self.padding = padding
        self.mask_channel_group = mask_channel_group
        
    def forward(self, x):
        if x is None : return None # If input mask is None, output None
        if self.stride > 1:
            # Ensure pad_kernel is created on the correct device and for the correct number of groups
            pad_kernel = torch.zeros((x.size(1),1,self.stride, self.stride), device=x.device)
            for i in range(x.size(1)): # Initialize for each group
                 pad_kernel[i,0,0,0] = 1
        
        # Dilate kernel should also be group-aware if mask_channel_group is involved
        # Or simpler: if x has G channels, dilate_kernel is Gx1xKxK for grouped conv
        dilate_kernel = torch.ones((x.size(1),1,1+2*self.padding,1+2*self.padding), device=x.device)
        
        x = x.float()
        
        if self.stride > 1:
            x = F.conv_transpose2d(x, pad_kernel, stride=self.stride, groups=x.size(1))
        
        # If padding is 0, kernel is 1x1, essentially no change unless it's about >0.5
        if self.padding > 0 or (self.padding == 0 and (dilate_kernel.shape[2]>1 or dilate_kernel.shape[3]>1)):
             x = F.conv2d(x, dilate_kernel, padding=self.padding, stride=1, groups=x.size(1))
        return x > 0.5


class Masker_channel_MLP(nn.Module):
    def __init__(self, in_channels, channel_dyn_group, layers=2, reduction=16):
        super(Masker_channel_MLP, self).__init__()
        assert(layers in [1,2])
        
        self.channel_dyn_group = channel_dyn_group
        if channel_dyn_group == 0: # Handle case for no channel masking
            self.conv = None
            self.conv_flops = 0
            return

        width = max(channel_dyn_group//reduction, 16)
        self.conv = nn.Sequential(
            nn.Linear(in_channels, width),
            nn.ReLU(),
            nn.Linear(width, channel_dyn_group*2,bias=True)
        ) if layers == 2 else nn.Linear(in_channels, channel_dyn_group*2,bias=True)
        
        self.conv_flops = in_channels * width + width * channel_dyn_group*2 if layers == 2 else in_channels * channel_dyn_group*2
        if layers == 2:
            self.conv[-1].bias.data[:channel_dyn_group] = 2.0
            self.conv[-1].bias.data[channel_dyn_group:] = -2.0 # Corrected slicing
        else:
            self.conv.bias.data[:channel_dyn_group] = 2.0
            self.conv.bias.data[channel_dyn_group:] = -2.0 # Corrected slicing

    def forward(self, x, temperature):
        if self.conv is None or self.channel_dyn_group == 0: # No channel masking
            return None, torch.tensor(1.0, device=x.device), 0.0

        b, c, h, w = x.shape
        flops = c # For the GAP operation (c * h * w -> c)
        mask_input =  F.adaptive_avg_pool2d(x, (1,1)).view(b,c)
        
        mask = self.conv(mask_input)
        flops += self.conv_flops
        
        b_m, c_m = mask.shape # c_m is channel_dyn_group*2
        mask = mask.view(b_m,2,c_m//2)
        if self.training:
            mask = F.gumbel_softmax(mask, dim=1, tau=temperature, hard=True)
            mask = mask[:,0]
        else:
            mask = (mask[:,0]>=mask[:,1]).float()
        
        sparsity = torch.mean(mask)
        return mask, sparsity, flops

class Masker_channel_conv_linear(nn.Module):
    def __init__(self, in_channels, channel_dyn_group, reduction=16):
        super(Masker_channel_conv_linear, self).__init__()
        self.channel_dyn_group = channel_dyn_group
        
        if channel_dyn_group == 0: # Handle case for no channel masking
            self.conv = None
            self.linear = None
            self.masker_flops = 0
            return

        self.conv = nn.Sequential(
            conv1x1(in_channels, in_channels//reduction),
            nn.BatchNorm2d(in_channels//reduction),
            nn.ReLU(),
        )
        self.linear = nn.Linear(in_channels//reduction, channel_dyn_group*2,bias=True)
        
        self.linear.bias.data[:channel_dyn_group] = 2.0
        self.linear.bias.data[channel_dyn_group:] = -2.0 # Corrected slicing
        
        self.masker_flops = (in_channels * (in_channels // reduction) + # conv1x1
                             (in_channels // reduction) * channel_dyn_group*2) # linear

    def forward(self, x, temperature):
        if self.conv is None or self.linear is None or self.channel_dyn_group == 0: # No channel masking
            return None, torch.tensor(1.0, device=x.device), 0.0

        mask_intermediate = self.conv(x) # spatial dimensions remain
        b, c_int, h_int, w_int = mask_intermediate.shape
        # Flops for conv part: (in_channels * (in_channels // reduction)) * h_int * w_int (approx)
        # This is tricky to calculate precisely here without knowing input h,w to this module
        # Let's stick to the precomputed self.masker_flops for the masker's own ops.
        flops = 0 # Flops calculation will be handled by Bottleneck

        mask_input_linear =  F.adaptive_avg_pool2d(mask_intermediate, (1,1)).view(b,c_int)
        flops += c_int # For GAP
        
        mask = self.linear(mask_input_linear)
        # self.masker_flops already accounts for linear layer, so no need to add here
        # We add only GAP flops here
        
        b_m,c_m = mask.shape # c_m is channel_dyn_group*2
        mask = mask.view(b_m,2,c_m//2)
        if self.training:
            mask = F.gumbel_softmax(mask, dim=1, tau=temperature, hard=True)
            mask = mask[:,0]
        else:
            mask = (mask[:,0]>=mask[:,1]).float()
        
        sparsity = torch.mean(mask)
        
        return mask, sparsity, flops


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_width=1,
                 dilation=1, norm_layer=None,
                 spatial_mask_channel_group=1,
                 channel_dyn_granularity=1,
                 output_size=56, 
                 mask_spatial_granularity=1,
                 dyn_mode='both',
                 channel_masker_type='conv_linear',
                 channel_masker_layers=2,
                 reduction=16):
        super(Bottleneck, self).__init__()
        
        assert dyn_mode in ['channel', 'spatial', 'both', 'layer', 'none']
        assert channel_masker_type in ['conv_linear', 'MLP']
        
        self.dyn_mode = dyn_mode
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        bottleneck_planes = planes

        if channel_dyn_granularity == 0 or dyn_mode not in ['channel', 'both']:
            channel_dyn_group = 0
        else:
            assert bottleneck_planes % channel_dyn_granularity == 0, f"bottleneck_planes {bottleneck_planes} not divisible by channel_dyn_granularity {channel_dyn_granularity}"
            channel_dyn_group = bottleneck_planes // channel_dyn_granularity
        
        self.conv1 = conv1x1(inplanes, bottleneck_planes)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = conv3x3(bottleneck_planes, bottleneck_planes, stride, groups=group_width, dilation=dilation)
        self.bn2 = norm_layer(bottleneck_planes)
        self.conv3 = conv1x1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.conv1_flops_per_pixel = inplanes*bottleneck_planes
        self.conv2_flops_per_pixel = bottleneck_planes*bottleneck_planes*9 // self.conv2.groups
        self.conv3_flops_per_pixel = bottleneck_planes*planes*self.expansion

        if self.downsample is not None:
            if isinstance(self.downsample, nn.Sequential) and len(self.downsample) > 0 and isinstance(self.downsample[0], nn.Conv2d):
                ds_conv = self.downsample[0]
                self.downsample_flops = ds_conv.in_channels * ds_conv.out_channels * ds_conv.kernel_size[0] * ds_conv.kernel_size[1] # Stride already accounts for output size
            else: 
                self.downsample_flops = 0 
        else:
            self.downsample_flops = 0

        self.output_size = output_size # Expected H/W of the feature map for this stage (used for mask scaling)
        self.mask_spatial_granularity = mask_spatial_granularity if dyn_mode not in ['none', 'channel'] else 0
        
        if self.output_size == 0 or self.mask_spatial_granularity == 0 or dyn_mode not in ['spatial', 'layer', 'both']:
            self.mask_size = 0 
        else:
            self.mask_size = self.output_size // self.mask_spatial_granularity
        
        self.masker_spatial = None
        self.masker_channel = None
        
        if dyn_mode in ['spatial', 'layer', 'both'] and self.mask_size > 0:
            self.masker_spatial = Masker_spatial(inplanes, spatial_mask_channel_group, self.mask_size)
            self.mask_expander2 = ExpandMask(stride=1, padding=1, mask_channel_group=spatial_mask_channel_group) 
            self.mask_expander1 = ExpandMask(stride=stride, padding=1, mask_channel_group=spatial_mask_channel_group)
             
        if dyn_mode in ['channel', 'both'] and channel_dyn_group > 0:
            if channel_masker_type == 'conv_linear':
                self.masker_channel = Masker_channel_conv_linear(inplanes, channel_dyn_group, reduction=reduction)
            else:
                self.masker_channel = Masker_channel_MLP(inplanes, channel_dyn_group, layers=channel_masker_layers, reduction=reduction)

    def forward(self, forward_input: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor], 
                temperature: float = 1.0) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        
        x, s_sparsity_c3_list, s_sparsity_c2_list, s_sparsity_c1_list, c_sparsity_list, flops_p_list, current_flops = forward_input
        identity = x
        
        channel_mask: Optional[torch.Tensor] = None
        spatial_mask_conv3_raw: Optional[torch.Tensor] = None # Raw output from Masker_spatial
        
        channel_sparsity = torch.tensor(1.0, device=x.device)
        spatial_sparsity_conv1 = torch.tensor(1.0, device=x.device) # For stats
        spatial_sparsity_conv2 = torch.tensor(1.0, device=x.device) # For stats
        spatial_sparsity_conv3 = torch.tensor(1.0, device=x.device) # Sparsity of spatial_mask_conv3_raw
        channel_mask_flops = 0.0
        spatial_mask_flops = 0.0

        if self.dyn_mode == 'channel':
            if self.masker_channel:
                channel_mask, channel_sparsity, cm_flops = self.masker_channel(x, temperature)
                channel_mask_flops += cm_flops
        elif self.dyn_mode in ['spatial', 'layer']:
            if self.masker_spatial:
                spatial_mask_conv3_raw, spatial_sparsity_conv3, sm_flops = self.masker_spatial(x, temperature)
                spatial_mask_flops += sm_flops
        elif self.dyn_mode == 'both':
            if self.masker_channel:
                channel_mask, channel_sparsity, cm_flops = self.masker_channel(x, temperature)
                channel_mask_flops += cm_flops
            if self.masker_spatial:
                spatial_mask_conv3_raw, spatial_sparsity_conv3, sm_flops = self.masker_spatial(x, temperature)
                spatial_mask_flops += sm_flops
        
        # This variable will hold the spatial mask interpolated to self.output_size (stage-level resolution)
        # It's used by expanders for stats and as a base for the application mask.
        base_spatial_mask_for_stats_and_app: Optional[torch.Tensor] = None

        if spatial_mask_conv3_raw is not None: # True if masker_spatial ran and produced a mask
            base_spatial_mask_for_stats_and_app = spatial_mask_conv3_raw
            if self.output_size > 0 and \
               (spatial_mask_conv3_raw.shape[2] != self.output_size or spatial_mask_conv3_raw.shape[3] != self.output_size):
                base_spatial_mask_for_stats_and_app = F.interpolate(spatial_mask_conv3_raw, size=self.output_size, mode='nearest')

            if self.mask_expander2 is not None and base_spatial_mask_for_stats_and_app is not None:
                spatial_mask_conv2_for_stats = self.mask_expander2(base_spatial_mask_for_stats_and_app)
                if spatial_mask_conv2_for_stats is not None:
                    spatial_sparsity_conv2 = spatial_mask_conv2_for_stats.float().mean()
            
            if self.mask_expander1 is not None and 'spatial_mask_conv2_for_stats' in locals() and spatial_mask_conv2_for_stats is not None:
                spatial_mask_conv1_for_stats = self.mask_expander1(spatial_mask_conv2_for_stats)
                if spatial_mask_conv1_for_stats is not None:
                    spatial_sparsity_conv1 = spatial_mask_conv1_for_stats.float().mean()
        
        current_block_sparse_flops = channel_mask_flops + spatial_mask_flops
        current_block_dense_flops = channel_mask_flops + spatial_mask_flops
        
        # --- Conv1 ---
        out = self.conv1(x)
        out_c1_shape = out.shape
        out = apply_channel_mask(out, channel_mask) # No spatial mask applied on conv1 output
        out = self.bn1(out)
        out = self.relu(out)
        
        dense_flops_c1 = self.conv1_flops_per_pixel * out_c1_shape[2] * out_c1_shape[3]
        current_block_dense_flops += dense_flops_c1
        current_block_sparse_flops += dense_flops_c1 * channel_sparsity * spatial_sparsity_conv1 # spatial_sparsity_conv1 for stats

        # --- Conv2 ---
        out = self.conv2(out)
        out_c2_shape = out.shape
        out = apply_channel_mask(out, channel_mask) # No spatial mask applied on conv2 output
        out = self.bn2(out)
        out = self.relu(out)
        
        dense_flops_c2 = self.conv2_flops_per_pixel * out_c2_shape[2] * out_c2_shape[3]
        current_block_dense_flops += dense_flops_c2
        current_block_sparse_flops += dense_flops_c2 * (channel_sparsity**2 if channel_mask is not None else channel_sparsity) * spatial_sparsity_conv2 # spatial_sparsity_conv2 for stats

        # --- Conv3 ---
        out = self.conv3(out)
        out_c3_shape = out.shape # Shape of conv3's output
        out = self.bn3(out)
        
        # Apply spatial mask to conv3's output
        if self.dyn_mode in ['layer', 'spatial', 'both'] and base_spatial_mask_for_stats_and_app is not None:
            # Ensure the mask matches the actual current spatial dimensions of 'out'
            target_h, target_w = out_c3_shape[2], out_c3_shape[3]
            current_mask_h, current_mask_w = base_spatial_mask_for_stats_and_app.shape[2], base_spatial_mask_for_stats_and_app.shape[3]

            final_mask_for_conv3_application = base_spatial_mask_for_stats_and_app
            if target_h != current_mask_h or target_w != current_mask_w:
                final_mask_for_conv3_application = F.interpolate(base_spatial_mask_for_stats_and_app, 
                                                                 size=(target_h, target_w), 
                                                                 mode='nearest')
            out = apply_spatial_mask(out, final_mask_for_conv3_application)
        
        dense_flops_c3 = self.conv3_flops_per_pixel * out_c3_shape[2] * out_c3_shape[3]
        current_block_dense_flops += dense_flops_c3
        # spatial_sparsity_conv3 is from the raw mask, reflecting sparsity if mask applied at conv3 output
        current_block_sparse_flops += dense_flops_c3 * channel_sparsity * spatial_sparsity_conv3 
        
        if self.downsample is not None:
            identity = self.downsample(x)
            ds_flops_spatial_factor = identity.shape[2] * identity.shape[3] if identity.ndim == 4 and identity.shape[2]*identity.shape[3] > 0 else 1
            ds_flops = self.downsample_flops * ds_flops_spatial_factor

            current_block_dense_flops += ds_flops
            current_block_sparse_flops += ds_flops
        
        out += identity
        out = self.relu(out)

        current_flops += current_block_sparse_flops
        flops_perc_this_block = current_block_sparse_flops / current_block_dense_flops if current_block_dense_flops > 0 else torch.tensor(1.0, device=x.device)

        s_sparsity_c3_list = spatial_sparsity_conv3.unsqueeze(0) if s_sparsity_c3_list is None else \
            torch.cat((s_sparsity_c3_list, spatial_sparsity_conv3.unsqueeze(0)), dim=0)
        s_sparsity_c2_list = spatial_sparsity_conv2.unsqueeze(0) if s_sparsity_c2_list is None else \
            torch.cat((s_sparsity_c2_list, spatial_sparsity_conv2.unsqueeze(0)), dim=0)
        s_sparsity_c1_list = spatial_sparsity_conv1.unsqueeze(0) if s_sparsity_c1_list is None else \
            torch.cat((s_sparsity_c1_list, spatial_sparsity_conv1.unsqueeze(0)), dim=0)
        c_sparsity_list = channel_sparsity.unsqueeze(0) if c_sparsity_list is None else \
            torch.cat((c_sparsity_list, channel_sparsity.unsqueeze(0)), dim=0)
        flops_p_list = flops_perc_this_block.unsqueeze(0) if flops_p_list is None else \
            torch.cat((flops_p_list, flops_perc_this_block.unsqueeze(0)), dim=0)
        
        return out, s_sparsity_c3_list, s_sparsity_c2_list, s_sparsity_c1_list, c_sparsity_list, flops_p_list, current_flops


class ResNet(nn.Module):
    def __init__(self, block: type[Bottleneck], layers: List[int], num_classes: int = 1000, zero_init_residual: bool = False,
                 groups: int = 1, width_per_group: int = 64, replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[type[nn.Module]] = None, width_mult: float = 1.,
                 input_size: int = 224, # Added for width_list calculation consistency
                 spatial_mask_channel_group: List[int]=[1,1,1,1],
                 mask_spatial_granularity: List[int]=[1,1,1,1],
                 channel_dyn_granularity: List[int]=[1,1,1,1],
                 dyn_mode: List[str]=['both','both','both','both'], # 'none' can be used here
                 channel_masker_type: List[str]=['MLP','MLP','MLP','MLP'], # Renamed from channel_masker
                 channel_masker_layers: List[int]=[1,1,1,1],
                 reduction_ratio: List[int]=[16,16,16,16],
                 lr_mult: float =1.0,
                 **kwargs): # Absorb potential extra kwargs
        super(ResNet, self).__init__()
        
        self.input_size = input_size # Store for dummy input
        self.dyn_mode_config = dyn_mode # Store for reference
        assert lr_mult is not None
        self.lr_mult = lr_mult

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64*width_mult)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple.")
        
        self.groups = groups # For ResNeXt type blocks, if Bottleneck uses it
        self.base_width = width_per_group # For ResNeXt type blocks

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Calculate output sizes for each stage dynamically based on input_size
        # Stage 0 (after conv1, maxpool): input_size / 4
        # Stage 1 (layer1): input_size / 4 (stride 1)
        # Stage 2 (layer2): input_size / 8 (stride 2)
        # Stage 3 (layer3): input_size / 16 (stride 2)
        # Stage 4 (layer4): input_size / 32 (stride 2)
        
        s0_size = input_size // 4
        s1_size = s0_size 
        s2_size = s1_size // (2 if not replace_stride_with_dilation[0] else 1)
        s3_size = s2_size // (2 if not replace_stride_with_dilation[1] else 1)
        s4_size = s3_size // (2 if not replace_stride_with_dilation[2] else 1)


        self.layer1 = self._make_layer(block, int(64*width_mult), layers[0], stride=1,
                                       dilate=False, # First layer of ResNet typically doesn't dilate like this
                                       output_size=s1_size,
                                       spatial_mask_channel_group=spatial_mask_channel_group[0],
                                       mask_spatial_granularity=mask_spatial_granularity[0],
                                       channel_dyn_granularity=channel_dyn_granularity[0],
                                       dyn_mode=dyn_mode[0],
                                       channel_masker_type=channel_masker_type[0],
                                       channel_masker_layers=channel_masker_layers[0],
                                       reduction_ratio=reduction_ratio[0])
        
        self.layer2 = self._make_layer(block, int(128*width_mult), layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       output_size=s2_size,
                                       spatial_mask_channel_group=spatial_mask_channel_group[1],
                                       mask_spatial_granularity=mask_spatial_granularity[1],
                                       channel_dyn_granularity=channel_dyn_granularity[1],
                                       dyn_mode=dyn_mode[1],
                                       channel_masker_type=channel_masker_type[1],
                                       channel_masker_layers=channel_masker_layers[1],
                                       reduction_ratio=reduction_ratio[1])
        
        self.layer3 = self._make_layer(block, int(256*width_mult), layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       output_size=s3_size,
                                       spatial_mask_channel_group=spatial_mask_channel_group[2],
                                       mask_spatial_granularity=mask_spatial_granularity[2],
                                       channel_dyn_granularity=channel_dyn_granularity[2],
                                       dyn_mode=dyn_mode[2],
                                       channel_masker_type=channel_masker_type[2],
                                       channel_masker_layers=channel_masker_layers[2],
                                       reduction_ratio=reduction_ratio[2])
        
        self.layer4 = self._make_layer(block, int(512*width_mult), layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       output_size=s4_size,
                                       spatial_mask_channel_group=spatial_mask_channel_group[3],
                                       mask_spatial_granularity=mask_spatial_granularity[3],
                                       channel_dyn_granularity=channel_dyn_granularity[3],
                                       dyn_mode=dyn_mode[3],
                                       channel_masker_type=channel_masker_type[3],
                                       channel_masker_layers=channel_masker_layers[3],
                                       reduction_ratio=reduction_ratio[3])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512*width_mult * block.expansion), num_classes)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and 'masker' not in name and m.weight is not None : # Check for m.weight not None
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None: nn.init.constant_(m.weight, 1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    if m.bn3.weight is not None : nn.init.constant_(m.bn3.weight, 0)
        
        # --- Calculate width_list (similar to MSPANet) ---
        self.width_list: List[int] = []
        try:
            original_mode = self.training
            self.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, self.input_size, self.input_size)
                features = self._forward_extract(dummy_input, temperature=1.0) # Provide default temp
                self.width_list = [f.size(1) for f in features]
            self.train(original_mode)
        except Exception as e:
            print(f"Warning: Could not compute width_list during ResNet init: {e}")
            # import traceback # Optional: for more detailed error
            # traceback.print_exc()
            self.width_list = [] # Keep it empty on failure


    def _make_layer(self, block: type[Bottleneck], planes: int, blocks: int, stride: int = 1, dilate: bool = False,
                    output_size: int = 56, # Expected H/W of feature maps from this layer
                    spatial_mask_channel_group: int =1,
                    mask_spatial_granularity: int =1,
                    channel_dyn_granularity: int =1,
                    dyn_mode: str ='both',
                    channel_masker_type: str ='MLP',
                    channel_masker_layers: int =1,
                    reduction_ratio: int =16) -> nn.ModuleList: # Changed to ModuleList
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate: # This means replace_stride_with_dilation for this stage was True
            self.dilation *= stride # Increase dilation factor
            stride = 1 # Actual stride of conv becomes 1
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), # Stride applied here for downsampling
                norm_layer(planes * block.expansion),
            )

        layers = []
        # First block of the layer handles downsampling (if any) and stride
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample, 
                            group_width=self.groups, # groups from ResNeXt
                            dilation=previous_dilation, # Dilation for the first block
                            norm_layer=norm_layer, 
                            output_size=output_size,
                            spatial_mask_channel_group=spatial_mask_channel_group,
                            mask_spatial_granularity=mask_spatial_granularity,
                            channel_dyn_granularity=channel_dyn_granularity,
                            dyn_mode=dyn_mode,
                            channel_masker_type=channel_masker_type,
                            channel_masker_layers=channel_masker_layers,
                            reduction=reduction_ratio))
        
        self.inplanes = planes * block.expansion # Update inplanes for subsequent blocks/layers
        
        # Subsequent blocks in the layer
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                group_width=self.groups,
                                dilation=self.dilation, # Use current self.dilation (potentially increased)
                                norm_layer=norm_layer, 
                                output_size=output_size, # All blocks in a stage have same output spatial size
                                spatial_mask_channel_group=spatial_mask_channel_group,
                                mask_spatial_granularity=mask_spatial_granularity,
                                channel_dyn_granularity=channel_dyn_granularity,
                                dyn_mode=dyn_mode,
                                channel_masker_type=channel_masker_type,
                                channel_masker_layers=channel_masker_layers,
                                reduction=reduction_ratio))

        return nn.ModuleList(layers) # Return as ModuleList

    def _forward_stages_dynamic(self, x: torch.Tensor, temperature: float = 1.0) -> \
        Tuple[torch.Tensor, List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], Optional[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """
        Internal forward pass that processes all stages and returns detailed dynamic info + feature list.
        Returns:
            - final_tensor_output (after layer4)
            - List of stage-wise spatial_sparsity_conv3 tensors
            - List of stage-wise spatial_sparsity_conv2 tensors
            - List of stage-wise spatial_sparsity_conv1 tensors
            - List of stage-wise channel_sparsity tensors
            - Concatenated flops_perc_list from all blocks
            - Total accumulated sparse flops
            - List of feature maps [x_s1, x_s2, x_s3, x_s4]
        """
        # --- Initial operations + FLOPs for them ---
        c_in = x.shape[1]
        x = self.conv1(x)
        flops = c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.kernel_size[0] * self.conv1.kernel_size[1] / self.conv1.groups
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        flops += x.shape[1]*x.shape[2]*x.shape[3]*9 # Maxpool approx 9 ops per element

        # --- Initialize accumulators for sparsity and flops percentage ---
        # These will store lists of tensors, one tensor per stage, each tensor containing per-block sparsities
        s_sparsity_c3_stages: List[Optional[torch.Tensor]] = []
        s_sparsity_c2_stages: List[Optional[torch.Tensor]] = []
        s_sparsity_c1_stages: List[Optional[torch.Tensor]] = []
        c_sparsity_stages: List[Optional[torch.Tensor]] = []
        
        # This will be a single tensor concatenating all per-block flop percentages
        all_blocks_flops_perc_list: Optional[torch.Tensor] = None
        
        intermediate_features: List[torch.Tensor] = []

        # --- Stage 1 ---
        # For each stage, the per-block sparsity lists start as None
        block_input_tuple = (x, None, None, None, None, None, flops)
        for i in range(len(self.layer1)):
            block_input_tuple = self.layer1[i](block_input_tuple, temperature)
        x_s1, ssc3_s1, ssc2_s1, ssc1_s1, cs_s1, fp_s1, flops = block_input_tuple
        s_sparsity_c3_stages.append(ssc3_s1)
        s_sparsity_c2_stages.append(ssc2_s1)
        s_sparsity_c1_stages.append(ssc1_s1)
        c_sparsity_stages.append(cs_s1)
        all_blocks_flops_perc_list = fp_s1
        intermediate_features.append(x_s1.clone()) # Save feature map

        # --- Stage 2 ---
        block_input_tuple = (x_s1, None, None, None, None, all_blocks_flops_perc_list, flops) # Pass accumulated flops_perc and flops
        for i in range(len(self.layer2)):
            block_input_tuple = self.layer2[i](block_input_tuple, temperature)
        x_s2, ssc3_s2, ssc2_s2, ssc1_s2, cs_s2, fp_s2, flops = block_input_tuple
        s_sparsity_c3_stages.append(ssc3_s2)
        s_sparsity_c2_stages.append(ssc2_s2)
        s_sparsity_c1_stages.append(ssc1_s2)
        c_sparsity_stages.append(cs_s2)
        all_blocks_flops_perc_list = fp_s2 # This is now the concatenated list up to this stage
        intermediate_features.append(x_s2.clone())

        # --- Stage 3 ---
        block_input_tuple = (x_s2, None, None, None, None, all_blocks_flops_perc_list, flops)
        for i in range(len(self.layer3)):
            block_input_tuple = self.layer3[i](block_input_tuple, temperature)
        x_s3, ssc3_s3, ssc2_s3, ssc1_s3, cs_s3, fp_s3, flops = block_input_tuple
        s_sparsity_c3_stages.append(ssc3_s3)
        s_sparsity_c2_stages.append(ssc2_s3)
        s_sparsity_c1_stages.append(ssc1_s3)
        c_sparsity_stages.append(cs_s3)
        all_blocks_flops_perc_list = fp_s3
        intermediate_features.append(x_s3.clone())

        # --- Stage 4 ---
        block_input_tuple = (x_s3, None, None, None, None, all_blocks_flops_perc_list, flops)
        for i in range(len(self.layer4)):
            block_input_tuple = self.layer4[i](block_input_tuple, temperature)
        x_s4, ssc3_s4, ssc2_s4, ssc1_s4, cs_s4, fp_s4, flops = block_input_tuple
        s_sparsity_c3_stages.append(ssc3_s4)
        s_sparsity_c2_stages.append(ssc2_s4)
        s_sparsity_c1_stages.append(ssc1_s4)
        c_sparsity_stages.append(cs_s4)
        all_blocks_flops_perc_list = fp_s4
        intermediate_features.append(x_s4.clone())
        
        return x_s4, s_sparsity_c3_stages, s_sparsity_c2_stages, s_sparsity_c1_stages, c_sparsity_stages, all_blocks_flops_perc_list, flops, intermediate_features

    def _forward_extract(self, x: torch.Tensor, temperature: float = 1.0) -> List[torch.Tensor]:
        """ Feature extraction forward, returns list of features from each stage. """
        _, _, _, _, _, _, _, features = self._forward_stages_dynamic(x, temperature)
        return features

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> List[torch.Tensor]:
        """ 
        Main forward pass for feature extraction (e.g., for YOLO).
        Returns a list of feature maps from different stages.
        """
        return self._forward_extract(x, temperature)

    def forward_classification(self, x: torch.Tensor, temperature: float = 1.0) -> \
        Tuple[torch.Tensor, List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass for classification, returning final logits and all dynamic statistics.
        This matches the return signature of the original Code 1's ResNet.forward.
        """
        x_s4, s_sparsity_c3_stages, s_sparsity_c2_stages, s_sparsity_c1_stages, \
            c_sparsity_stages, all_blocks_flops_perc_list, current_flops, _ = \
            self._forward_stages_dynamic(x, temperature)

        out = self.avgpool(x_s4)
        current_flops += out.shape[1]*out.shape[2]*out.shape[3] # Approx for avgpool
        
        out = torch.flatten(out, 1)
        
        fc_in_channels = out.shape[1]
        out = self.fc(out)
        current_flops += fc_in_channels * out.shape[1] # FLOPs for FC layer
        
        return out, s_sparsity_c3_stages, s_sparsity_c2_stages, s_sparsity_c1_stages, c_sparsity_stages, all_blocks_flops_perc_list, current_flops
    
    def get_optim_policies(self) -> List[dict]: # Type hint for return
        # This method seems fine, just added type hint
        backbone_params = []
        masker_params = []

        for name, param in self.named_parameters(): # Iterate over parameters directly
            if not param.requires_grad:
                continue
            
            module_name = name.split('.')[0] # e.g. layer1, masker_spatial
            is_masker_param = False
            # Check if any part of the name indicates a masker module
            # This is a bit fragile; depends on module naming conventions in Bottleneck
            # A more robust way would be to iterate self.named_modules() and check isinstance,
            # then get params for those modules. But current logic is based on 'masker' in name.
            
            # Let's refine based on module type as in original code, but use named_parameters better
            # The original code iterated named_modules(), this is more precise for parameters.
            # To match original, we need to check the module a param belongs to.
            # For simplicity, let's assume 'masker' in the parameter name string is sufficient.
            # (e.g., self.masker_spatial.conv.weight)

            # Reverting to module iteration for policy separation:
            pass # Will re-implement this part more carefully

        # Re-implementation of get_optim_policies closer to original intent
        backbone_params_set = set() # Use set to avoid duplicate tensors
        masker_params_set = set()

        for name, m in self.named_modules():
            is_masker_module = 'masker' in name or isinstance(m, (Masker_spatial, Masker_channel_MLP, Masker_channel_conv_linear, ExpandMask))
            
            # Check for specific nn.Module types that have parameters
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                for param_name, param in m.named_parameters(recurse=False): # Get params of this module only
                    if not param.requires_grad:
                        continue
                    if is_masker_module:
                        masker_params_set.add(param)
                    else:
                        backbone_params_set.add(param)
        
        # Parameters directly in ResNet (e.g. self.fc.weight) not in a 'masker' named module
        # will be backbone by default unless explicitly assigned.
        # The self.fc is not a masker. self.conv1, self.bn1 are not masker.

        # Ensure no overlap (though set inherently handles it)
        # common_params = backbone_params_set.intersection(masker_params_set)
        # if common_params:
        #     print(f"Warning: Overlapping params in optimizer policies: {common_params}")

        return [
            {'params': list(backbone_params_set), 'lr_mult': self.lr_mult, 'decay_mult': 1.0, 'name': "backbone_params"},
            {'params': list(masker_params_set), 'lr_mult': 1.0, 'decay_mult': 1.0, 'name': "masker_params"}, # Default lr_mult=1.0 for maskers
        ]

def _resnet(arch: str, block: type[Bottleneck], layers: List[int], pretrained: bool, progress: bool, **kwargs) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        
        # Filter out unexpected keys (like width_list if it were saved, or mismatch in num_classes for fc)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        
        # If fc layer size mismatch, remove it from pretrained_dict
        if 'fc.weight' in pretrained_dict and model_dict['fc.weight'].shape != pretrained_dict['fc.weight'].shape:
            print(f"Removing fc.weight ({pretrained_dict['fc.weight'].shape}) due to shape mismatch with model ({model_dict['fc.weight'].shape})")
            del pretrained_dict['fc.weight']
        if 'fc.bias' in pretrained_dict and model_dict['fc.bias'].shape != pretrained_dict['fc.bias'].shape:
            print(f"Removing fc.bias ({pretrained_dict['fc.bias'].shape}) due to shape mismatch with model ({model_dict['fc.bias'].shape})")
            del pretrained_dict['fc.bias']

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False) # strict=False to allow for width_list etc.
    return model


def uni_resnet50(pretrained: bool = False, progress: bool = True, **kwargs) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        **kwargs: Extra arguments passed to the ResNet constructor.
    """
    # print('Model: UniResnet 50')
    # Default ResNet-50 parameters that can be overridden by kwargs
    defaults = {
        "groups": 1,
        "width_per_group": 64,
    }
    defaults.update(kwargs) # kwargs can override defaults
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **defaults)


def uni_resnet101(pretrained: bool = False, progress: bool = True, **kwargs) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        **kwargs: Extra arguments passed to the ResNet constructor.
    """
    # print('Model: UniResnet 101')
    defaults = {
        "groups": 1,
        "width_per_group": 64,
    }
    defaults.update(kwargs)
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **defaults)

# Example Usage (for testing purposes):
if __name__ == '__main__':
    # Test with default settings (should enable dynamic masking by default)
    print("--- Testing uni_resnet50 (default dynamic) ---")
    model50_dynamic = uni_resnet50(pretrained=False, input_size=224)
    print(f"ResNet50 width_list: {model50_dynamic.width_list}")
    dummy_input = torch.randn(2, 3, 640, 640)
    
    # Test feature extraction forward
    features = model50_dynamic(dummy_input, temperature=0.5)
    print("Feature extraction output (shapes):")
    for i, f in enumerate(features):
        print(f"  Stage {i+1}: {f.shape}")

    # Test classification forward
    logits, ssc3, ssc2, ssc1, cs, fp, fl = model50_dynamic.forward_classification(dummy_input, temperature=0.5)
    print(f"\nClassification output logits shape: {logits.shape}")
    print(f"Total sparse FLOPs: {fl.item()}")
    if fp is not None:
        print(f"Avg FLOPs percentage (per block): {fp.mean().item() if fp.numel() > 0 else 'N/A'}")

    # Test with dyn_mode='none' for all stages
    print("\n--- Testing uni_resnet50 (dyn_mode='none') ---")
    model50_static = uni_resnet50(pretrained=False, input_size=224, dyn_mode=['none','none','none','none'])
    print(f"ResNet50 (static) width_list: {model50_static.width_list}")
    features_static = model50_static(dummy_input) # Default temperature
    print("Static Feature extraction output (shapes):")
    for i, f in enumerate(features_static):
        print(f"  Stage {i+1}: {f.shape}")
    
    logits_static, _, _, _, _, _, fl_static = model50_static.forward_classification(dummy_input)
    print(f"\nStatic Classification output logits shape: {logits_static.shape}")
    print(f"Total static FLOPs: {fl_static.item()}")


    # Test get_optim_policies
    policies = model50_dynamic.get_optim_policies()
    print("\nOptimizer policies:")
    for pol in policies:
        print(f"  {pol['name']}: {len(pol['params'])} parameters")
        # Check if any parameter is shared (should not happen with sets)
        # for other_pol in policies:
        #     if pol['name'] != other_pol['name']:
        #         common = set(pol['params']).intersection(set(other_pol['params']))
        #         if common:
        #             print(f"    WARNING: Common params between {pol['name']} and {other_pol['name']}: {len(common)}")

    print("\n--- Testing uni_resnet101 ---")
    model101 = uni_resnet101(pretrained=False, input_size=256)
    print(f"ResNet101 width_list: {model101.width_list}")
    dummy_input_101 = torch.randn(1, 3, 256, 256)
    features_101 = model101(dummy_input_101)
    print("ResNet101 Feature shapes:")
    for f in features_101:
        print(f"  {f.shape}")