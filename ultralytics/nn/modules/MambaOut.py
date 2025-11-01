# """
# MambaOut models for image classification / feature extraction.
# Some implementations are modified from:
# timm (https://github.com/rwightman/pytorch-image-models),
# MetaFormer (https://github.com/sail-sg/metaformer),
# InceptionNeXt (https://github.com/sail-sg/inceptionnext)
# """
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _cfg(url='', **kwargs):
    # Note: URL is now informational only if default_cfg is used, as loading is removed.
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

# Default configs remain for metadata, but URL loading is removed from model functions
default_cfgs = {
    'mambaout_femto': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_femto.pth'),
    'mambaout_kobe': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_kobe.pth'),
    'mambaout_tiny': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_tiny.pth'),
    'mambaout_small': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_small.pth'),
    'mambaout_base': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_base.pth'),
}

# --- Model Configuration Dictionaries ---
MAMBAOUT_CONFIGS = {
    "mambaout_femto": {
        "depths": [3, 3, 9, 3],
        "dims": [48, 96, 192, 288],
        "default_cfg": default_cfgs['mambaout_femto'],
    },
    "mambaout_kobe": { # Kobe Memorial Version with 24 Gated CNN blocks
        "depths": [3, 3, 15, 3],
        "dims": [48, 96, 192, 288],
        "default_cfg": default_cfgs['mambaout_kobe'],
    },
    "mambaout_tiny": {
        "depths": [3, 3, 9, 3],
        "dims": [96, 192, 384, 576],
        "default_cfg": default_cfgs['mambaout_tiny'],
    },
    "mambaout_small": {
        "depths": [3, 4, 27, 3],
        "dims": [96, 192, 384, 576],
        "default_cfg": default_cfgs['mambaout_small'],
    },
    "mambaout_base": {
        "depths": [3, 4, 27, 3],
        "dims": [128, 256, 512, 768],
        "default_cfg": default_cfgs['mambaout_base'],
    }
}


class StemLayer(nn.Module):
    r""" Code modified from InternImage:
        https://github.com/OpenGVLab/InternImage
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=96,
                 act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = norm_layer(out_channels // 2)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(out_channels // 2,
                               out_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = norm_layer(out_channels)

    def forward(self, x):
        # Input x: [B, C, H, W]
        x = self.conv1(x) # [B, C/2, H/2, W/2]
        x = x.permute(0, 2, 3, 1).contiguous() # [B, H/2, W/2, C/2] (Use contiguous after permute)
        x = self.norm1(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2).contiguous() # [B, C/2, H/2, W/2] -> Back to N C H W for Conv2D

        x = self.conv2(x) # [B, C, H/4, W/4]
        x = x.permute(0, 2, 3, 1).contiguous() # [B, H/4, W/4, C] -> N H W C format for subsequent layers
        x = self.norm2(x)
        # Output: [B, H/4, W/4, C] (NHWC)
        return x


class DownsampleLayer(nn.Module):
    r""" Code modified from InternImage:
        https://github.com/OpenGVLab/InternImage
    """
    def __init__(self, in_channels=96, out_channels=198, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        # Input is expected in N H W C format from previous layer
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        # Input x: [B, H, W, C]
        x = x.permute(0, 3, 1, 2).contiguous() # [B, C, H, W] for Conv2D
        x = self.conv(x)          # [B, C_out, H/2, W/2]
        x = x.permute(0, 2, 3, 1).contiguous() # [B, H/2, W/2, C_out] for LayerNorm and subsequent layers
        x = self.norm(x)
        # Output: [B, H/2, W/2, C_out] (NHWC)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, act_layer=nn.GELU, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # Input x: [B, C] (after pooling)
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class GatedCNNBlock(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6),
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        self.conv_ratio = conv_ratio
        # Determine conv_channels based on hidden dimension
        conv_channels = int(conv_ratio * hidden)
        conv_channels = max(1, conv_channels) # Ensure at least 1 channel

        if self.conv_ratio == 1.0:
            self.split_indices = (hidden, hidden) # g, c split
            conv_channels = hidden # Conv acts on the 'c' part
        else:
            # Ensure i_channels is non-negative
            i_channels = hidden - conv_channels
            if i_channels < 0:
                 # This case might happen if conv_ratio > 1 or expansion_ratio is small
                 # Defaulting to splitting equally if calculation is invalid? Or raise error?
                 # Let's assume conv_ratio <= 1 and expansion is reasonable.
                 # If i_channels is 0, it means conv takes the whole second split.
                 i_channels = max(0, i_channels) # Clamp at 0
                 conv_channels = hidden - i_channels
            self.split_indices = (hidden, i_channels, conv_channels) # g, i, c split


        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim) # Project back from the combined hidden dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # Input x: [B, H, W, C] (dim=C) NHWC format
        shortcut = x
        x = self.norm(x)

        # Split logic based on conv_ratio
        if self.conv_ratio == 1.0:
            g, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
            i = None # No identity split when conv_ratio is 1.0
        else:
            split_result = torch.split(self.fc1(x), self.split_indices, dim=-1)
            g = split_result[0]
            i = split_result[1] if self.split_indices[1] > 0 else None # Handle zero-size split for i
            c = split_result[2]


        c = c.permute(0, 3, 1, 2).contiguous() # [B, H, W, C_conv] -> [B, C_conv, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1).contiguous() # [B, C_conv, H, W] -> [B, H, W, C_conv]

        if i is not None:
            hidden_out = self.act(g) * torch.cat((i, c), dim=-1)
        else: # Handles both conv_ratio = 1.0 and cases where i_channels is 0
            hidden_out = self.act(g) * c # Multiply gate with conv output directly

        x = self.fc2(hidden_out)
        x = self.drop_path(x)
        # Output: [B, H, W, C] (NHWC)
        return x + shortcut

r"""
downsampling (stem) for the first stage is two layer of conv with k3, s2 and p1
downsamplings for the last 3 stages is a layer of conv with k3, s2 and p1
DOWNSAMPLE_LAYERS_FOUR_STAGES format: [Downsampling, Downsampling, Downsampling, Downsampling]
use `partial` to specify some arguments
"""
DOWNSAMPLE_LAYERS_FOUR_STAGES = [StemLayer] + [DownsampleLayer]*3


class MambaOut(nn.Module):
    r""" MambaOut Model Base - Modified for Backbone Usage in Detection/Segmentation
    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head (IGNORED if used as backbone). Default: 1000.
        depths (list or tuple): Number of blocks at each stage.
        dims (list or tuple): Feature dimension at each stage.
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        norm_layer: Normalization layer. Default: partial(nn.LayerNorm, eps=1e-6).
        act_layer: Activation layer. Default: nn.GELU.
        conv_ratio (float): Conv ratio in GatedCNNBlock. Default: 1.0.
        kernel_size (int): Kernel size in GatedCNNBlock. Default: 7.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        output_norm: Norm before classifier head (IGNORED if used as backbone). Default: partial(nn.LayerNorm, eps=1e-6).
        head_fn: Classification head module/fn (IGNORED if used as backbone). Default: MlpHead.
        head_dropout (float): Dropout for MLP classifier (IGNORED if used as backbone). Default: 0.0.
        input_size (tuple): Input image size, used for width_list calculation. Default: (3, 224, 224).
        out_indices (tuple): Indices of stages to output features from. Default: (0, 1, 2, 3).
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 576],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 conv_ratio=1.0,
                 kernel_size=7,
                 drop_path_rate=0.,
                 output_norm=partial(nn.LayerNorm, eps=1e-6), # Keep for potential classification use
                 head_fn=MlpHead, # Keep for potential classification use
                 head_dropout=0.0, # Keep for potential classification use
                 input_size=(3, 224, 224), # Add input_size for dummy pass
                 out_indices=(0, 1, 2, 3), # Indices of stages to output features
                 **kwargs,
                 ):
        super().__init__()
        # num_classes, output_norm, head_fn, head_dropout are stored but unused by forward if used as backbone
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.num_stage = len(depths)
        self.out_indices = out_indices

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * self.num_stage

        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList()
        # Stem layer needs only in_channels and first dim
        self.downsample_layers.append(
            downsample_layers[0](down_dims[0], down_dims[1], act_layer=act_layer, norm_layer=norm_layer)
        )
        # Subsequent downsample layers need previous dim and current dim
        for i in range(1, self.num_stage):
             self.downsample_layers.append(
                downsample_layers[i](down_dims[i], down_dims[i+1], norm_layer=norm_layer)
             )


        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(self.num_stage):
            stage = nn.Sequential(
                *[GatedCNNBlock(dim=dims[i],
                expansion_ratio=kwargs.get('expansion_ratio', 8/3), # Allow expansion ratio override
                norm_layer=norm_layer,
                act_layer=act_layer,
                kernel_size=kernel_size,
                conv_ratio=conv_ratio,
                drop_path=dp_rates[cur + j],
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Output Norm and Head are kept for potential standalone classification use,
        # but will NOT be used in the default forward pass for backbone integration.
        self.norm = output_norm(dims[-1])
        if head_fn is not None:
            if head_dropout > 0.0:
                self.head = head_fn(dims[-1], num_classes, act_layer=act_layer, norm_layer=output_norm, head_dropout=head_dropout)
            else:
                if head_fn == MlpHead:
                     self.head = head_fn(dims[-1], num_classes, act_layer=act_layer, norm_layer=output_norm)
                else:
                     self.head = head_fn(dims[-1], num_classes)
        else:
            self.head = nn.Identity() # No head if head_fn is None


        self.apply(self._init_weights)

        # --- Calculate self.width_list (based on selected out_indices) ---
        self.eval() # Set to eval mode for dummy pass
        with torch.no_grad(): # Disable gradient calculation
            dummy_input = torch.randn(1, *input_size)
            # Pass through forward_features_for_init to get intermediate outputs in NHWC format
            # Note: This internal helper avoids the final permutation needed for YOLO output
            intermediate_features_nhwc, _ = self._forward_features_nhwc(dummy_input)

            if intermediate_features_nhwc and all(isinstance(f, torch.Tensor) for f in intermediate_features_nhwc):
                 # Select widths based on out_indices
                 self.width_list = [intermediate_features_nhwc[i].size(-1) for i in self.out_indices]
                 # Store the strides as well, YOLO might need them
                 self.stride = torch.tensor([input_size[-1] / intermediate_features_nhwc[i].size(2) for i in self.out_indices])

            else:
                 print("Warning: Could not determine width_list/stride. _forward_features_nhwc did not return expected intermediate tensors.")
                 self.width_list = []
                 self.stride = torch.tensor([])


        self.train() # Set back to train mode

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_decay = {'norm'}
        for name, m in self.named_modules():
             if isinstance(m, (nn.LayerNorm)): # Only LayerNorm used here
                 no_decay.add(f'{name}.weight')
                 if hasattr(m, 'bias') and m.bias is not None:
                    no_decay.add(f'{name}.bias')
        # Exclude biases from linear/conv layers if desired
        # for name, param in self.named_parameters():
        #     if param.ndim == 1: # Check if it's a bias or other 1D param
        #         no_decay.add(name)
        return no_decay

    def _forward_features_nhwc(self, x):
        """Internal helper to get features in NHWC format for width/stride calculation."""
        intermediate_outputs_nhwc = []
        # Input x: [B, C, H, W]
        x = self.downsample_layers[0](x) # Stem output: NHWC
        x = self.stages[0](x) # Stage 0 output: NHWC
        intermediate_outputs_nhwc.append(x)

        for i in range(1, self.num_stage):
            x = self.downsample_layers[i](x) # Downsample output: NHWC
            x = self.stages[i](x)            # Stage i output: NHWC
            intermediate_outputs_nhwc.append(x)

        # Keep pooled output calculation (might be useful elsewhere, but not used by forward)
        pooled_output = self.norm(x.mean([1, 2])) # (B, H, W, C) -> (B, C)
        return intermediate_outputs_nhwc, pooled_output

    def forward_features(self, x):
        """ Main feature extraction, outputs features in NCHW format for compatibility. """
        intermediate_outputs_nchw = []
        # Input x: [B, C, H, W]
        x = self.downsample_layers[0](x) # Stem output: NHWC
        x = self.stages[0](x) # Stage 0 output: NHWC
        if 0 in self.out_indices: # Check if stage 0 output is needed
             intermediate_outputs_nchw.append(x.permute(0, 3, 1, 2).contiguous()) # Convert to NCHW

        for i in range(1, self.num_stage):
            x = self.downsample_layers[i](x) # Downsample output: NHWC
            x = self.stages[i](x)            # Stage i output: NHWC
            if i in self.out_indices: # Check if stage i output is needed
                intermediate_outputs_nchw.append(x.permute(0, 3, 1, 2).contiguous()) # Convert to NCHW

        # Note: We don't return the pooled_output here as it's not standard for backbone output
        return intermediate_outputs_nchw # Return list of selected features in NCHW

    def forward(self, x):
        """
        Forward pass for using MambaOut as a backbone.
        Returns a list of feature maps (in NCHW format) from specified stages.
        """
        x = self.forward_features(x)
        return x

    def forward_cls(self, x):
        """
        Forward pass for classification task (if needed separately).
        Uses the final stage output, pooling, norm, and head.
        """
        # Get NHWC features (doesn't matter which forward_features is used if only last stage is needed)
        intermediate_features_nhwc, pooled_output = self._forward_features_nhwc(x)
        # pooled_output is already calculated correctly after norm in _forward_features_nhwc
        cls_output = self.head(pooled_output)
        return cls_output


# --- Model Registration Functions (Modified) ---

@register_model
def mambaout_femto(**kwargs):
    """ Instantiates MambaOut-Femto model """
    config = MAMBAOUT_CONFIGS['mambaout_femto']
    # Ensure out_indices is passed if not specified by user
    kwargs.setdefault('out_indices', (0, 1, 2, 3)) # Default to output all stages for backbone use
    model = MambaOut(
        depths=config['depths'],
        dims=config['dims'],
        **kwargs) # Pass any overrides
    model.default_cfg = config['default_cfg'] # Assign default_cfg
    return model

@register_model
def mambaout_kobe(**kwargs):
    """ Instantiates MambaOut-Kobe model """
    config = MAMBAOUT_CONFIGS['mambaout_kobe']
    kwargs.setdefault('out_indices', (0, 1, 2, 3))
    model = MambaOut(
        depths=config['depths'],
        dims=config['dims'],
        **kwargs)
    model.default_cfg = config['default_cfg']
    return model

@register_model
def mambaout_tiny(**kwargs):
    """ Instantiates MambaOut-Tiny model """
    config = MAMBAOUT_CONFIGS['mambaout_tiny']
    kwargs.setdefault('out_indices', (0, 1, 2, 3))
    model = MambaOut(
        depths=config['depths'],
        dims=config['dims'],
        **kwargs)
    model.default_cfg = config['default_cfg']
    return model

@register_model
def mambaout_small(**kwargs):
    """ Instantiates MambaOut-Small model """
    config = MAMBAOUT_CONFIGS['mambaout_small']
    kwargs.setdefault('out_indices', (0, 1, 2, 3))
    model = MambaOut(
        depths=config['depths'],
        dims=config['dims'],
        **kwargs)
    model.default_cfg = config['default_cfg']
    return model

@register_model
def mambaout_base(**kwargs):
    """ Instantiates MambaOut-Base model """
    config = MAMBAOUT_CONFIGS['mambaout_base']
    kwargs.setdefault('out_indices', (0, 1, 2, 3)) # Usually detection uses stages 1, 2, 3 (e.g., indices (1,2,3)) adjust if needed
    model = MambaOut(
        depths=config['depths'],
        dims=config['dims'],
        **kwargs)
    model.default_cfg = config['default_cfg']
    return model

# Example Usage (Testing Backbone Output)
if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 3, 640, 640) # Typical detection input size
    image = torch.rand(*image_size)

    # Model Instantiation (example: tiny, selecting features for detection P3, P4, P5)
    # Assuming stages correspond roughly to strides 8, 16, 32, 64 after stem (4x downsample)
    # For P3, P4, P5 (strides 8, 16, 32), we might need outputs from stages 0, 1, 2
    out_indices_det = (1, 2, 3) # Example: Use features after stages 1, 2, 3

    print(f"--- MambaOut Tiny as Backbone (out_indices={out_indices_det}) ---")
    model_tiny_backbone = mambaout_tiny(out_indices=out_indices_det, input_size=image_size[1:])
    model_tiny_backbone.eval() # Set to eval mode

    # Standard forward pass (for backbone usage)
    features = model_tiny_backbone(image)

    print(f"Number of output feature maps: {len(features)}")
    for i, f in enumerate(features):
        print(f"Feature map {i} shape (NCHW): {f.shape}")

    print("\nStored Widths (channels) corresponding to out_indices:", model_tiny_backbone.width_list)
    print("Stored Strides corresponding to out_indices:", model_tiny_backbone.stride)

    # --- Example: Using for Classification ---
    print(f"\n--- MambaOut Tiny for Classification ---")
    # Re-instantiate or use the same instance but call forward_cls
    model_tiny_cls = mambaout_tiny(num_classes=50, input_size=(3, 224, 224)) # Example with 50 classes and typical input size
    cls_image = torch.rand(1, 3, 224, 224)
    cls_output = model_tiny_cls.forward_cls(cls_image)
    print(f"Classification output shape: {cls_output.shape}")