import torch
import copy
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torch.nn import ModuleList

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
# from timm.models.registry import register_model # No longer needed for this structure

import random
import warnings
warnings.filterwarnings('ignore')


# IMAGENET defaults (can be used for classifier head if added later)
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None, # Default input size, but can be overridden
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head', # Refers to the final classifier layer name
        **kwargs
    }


default_cfgs = {
    'mobilevigv2_ti': _cfg(crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'mobilevigv2_s': _cfg(crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'mobilevigv2_m': _cfg(crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'mobilevigv2_b': _cfg(crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
}

# --- Model Component Classes (Stem, DepthWiseSeparable, InvertedResidual, MRConv4d, RepCPE, Grapher, MGC, Downsample) ---
# (Keep these class definitions exactly as they were in the original Code 1)
class Stem(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)


class DepthWiseSeparable(nn.Module):
    def __init__(self, in_dim, kernel, e=4):
        super().__init__()

        self.pw1 = nn.Conv2d(in_dim, in_dim * e, 1) # kernel size = 1
        self.norm1 = nn.BatchNorm2d(in_dim * e)
        self.act1 = nn.GELU()

        self.dw = nn.Conv2d(in_dim * e, in_dim * e, kernel_size=kernel, stride=1, padding=1, groups=in_dim * e) # kernel size = 3
        self.norm2 = nn.BatchNorm2d(in_dim * e)
        self.act2 = nn.GELU()

        self.pw2 = nn.Conv2d(in_dim * e, in_dim, 1)
        self.norm3 = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        x = self.pw1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.dw(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.pw2(x)
        x = self.norm3(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, dim, kernel, expansion_ratio=4., drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.dws = DepthWiseSeparable(in_dim=dim, kernel=kernel, e=expansion_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.dws(x))
        else:
            x = x + self.drop_path(self.dws(x))
        return x


class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type

    K is the number of superpatches, therefore hops equals res // K.
    """
    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        # The input channels for the nn will be doubled (original + max-relative)
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1), # Adjusted input channels here
            nn.BatchNorm2d(out_channels),
            nn.GELU()
            )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape

        # Ensure K is not larger than H or W to avoid errors
        safe_K_H = min(self.K, H // 2) if H > 1 else 0
        safe_K_W = min(self.K, W // 2) if W > 1 else 0

        if safe_K_H == 0 and safe_K_W == 0:
            # If K is too large or feature map is 1x1, just return original features
            # through a modified nn layer that expects C channels
             temp_nn = nn.Sequential(
                nn.Conv2d(C, C, 1),
                nn.BatchNorm2d(C),
                nn.GELU()
             ).to(x.device)
             # Pass original 'x' through a 1x1 conv equivalent,
             # because concatenation won't happen
             # Note: This assumes the output channel should be the same as input if K is invalid.
             # Or adjust temp_nn's output channel if needed.
             return temp_nn(x)


        '''
        This is the 5 connection graph construction
        '''
        x_j = torch.zeros_like(x) # Initialize x_j properly

        if safe_K_H > 0:
            x_c_neg = torch.cat([x[:, :, -safe_K_H:, :], x[:, :, :-safe_K_H, :]], dim=2)
            x_j = torch.max(x_j, x_c_neg - x)
            x_c_pos = torch.cat([x[:, :, safe_K_H:, :], x[:, :, :safe_K_H, :]], dim=2)
            x_j = torch.max(x_j, x_c_pos - x)

        if safe_K_W > 0:
            x_r_neg = torch.cat([x[:, :, :, -safe_K_W:], x[:, :, :, :-safe_K_W]], dim=3)
            x_j = torch.max(x_j, x_r_neg - x)
            x_r_pos = torch.cat([x[:, :, :, safe_K_W:], x[:, :, :, :safe_K_W]], dim=3)
            x_j = torch.max(x_j, x_r_pos - x)

        x_cat = torch.cat([x, x_j], dim=1)
        return self.nn(x_cat)


class RepCPE(nn.Module):
    """
    This implementation of reparameterized conditional positional encoding was originally implemented
    in the following repository: https://github.com/apple/ml-fastvit

    Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        spatial_shape = (7, 7),
        inference_mode=False, # Keep track if reparameterized
    ) -> None:
        """Build reparameterizable conditional positional encoding

        Args:
            in_channels: Number of input channels.
            embed_dim: Number of embedding dimensions. Default: 768
            spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """
        super(RepCPE, self).__init__()
        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim
        self._inference_mode = inference_mode # Store inference mode state

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias=True,
            )
        else:
            # Ensure groups does not exceed in_channels
            effective_groups = min(self.embed_dim, self.in_channels)
            self.pe = nn.Conv2d(
                in_channels,
                embed_dim, # Output should be embed_dim
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                bias=True,
                groups=effective_groups, # Use effective_groups
            )

    def forward(self, x: torch.Tensor):
        if hasattr(self, "reparam_conv"):
            return self.reparam_conv(x)
        else:
             # Check if input channels match embed_dim for the addition
            if self.in_channels == self.embed_dim:
                return self.pe(x) + x
            else:
                # If channels don't match (e.g., first layer), just apply PE
                # Or consider a projection layer if addition is strictly needed
                return self.pe(x)


    def reparameterize(self):
        if self._inference_mode or not hasattr(self, 'pe'): # Check if already reparameterized or pe doesn't exist
             return

        # Build equivalent Id tensor only if in_channels == embed_dim
        if self.in_channels == self.embed_dim:
            input_dim_per_group = self.in_channels // self.groups # Must use self.groups used in pe init
            kernel_value = torch.zeros(
                (
                    self.in_channels,
                    input_dim_per_group, # Channels per group
                    self.spatial_shape[0],
                    self.spatial_shape[1],
                ),
                dtype=self.pe.weight.dtype,
                device=self.pe.weight.device,
            )
            for i in range(self.in_channels):
                 # Correctly calculate the index within the group
                group_index = i // input_dim_per_group
                index_in_group = i % input_dim_per_group
                kernel_value[
                    i,
                    index_in_group, # Use index within the group
                    self.spatial_shape[0] // 2,
                    self.spatial_shape[1] // 2,
                ] = 1
            id_tensor = kernel_value
            # Reparameterize Id tensor and conv
            w_final = id_tensor + self.pe.weight
            b_final = self.pe.bias
        else:
            # If channels don't match, the reparameterized conv is just the pe conv
            w_final = self.pe.weight
            b_final = self.pe.bias


        # Introduce reparam conv
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.pe.groups, # Use groups from original pe
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        # Cleanup
        for para in self.parameters():
            para.detach_()
        if hasattr(self, 'pe'):
             self.__delattr__("pe")
        self._inference_mode = True # Mark as reparameterized


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, K, cpe_spatial_shape=(7,7)):
        super(Grapher, self).__init__()
        # Pass spatial_shape to RepCPE
        self.cpe = RepCPE(in_channels=in_channels, embed_dim=in_channels, spatial_shape=cpe_spatial_shape)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        # MRConv4d input is in_channels, output is in_channels (it handles doubling internally)
        self.graph_conv = MRConv4d(in_channels, in_channels, K=K)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        x = self.cpe(x) # Apply CPE first
        shortcut = x # Store shortcut after potential CPE modification
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        return x + shortcut # Add shortcut

    def reparameterize(self):
        if hasattr(self.cpe, 'reparameterize'):
            self.cpe.reparameterize()


class MGC(nn.Module):
    def __init__(self, in_dim, drop_path=0., K=2, use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.mixer = Grapher(in_dim, K) # Pass K to Grapher
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim * 4),
            nn.GELU(),
            nn.Conv2d(in_dim * 4, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((in_dim, 1, 1)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((in_dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x
        if self.use_layer_scale:
            x = shortcut + self.drop_path(self.layer_scale_1 * self.mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.ffn(x)) # FFN path also adds to the result
        else:
            x = shortcut + self.drop_path(self.mixer(x))
            x = x + self.drop_path(self.ffn(x)) # FFN path also adds to the result
        return x

    def reparameterize(self):
         # Reparameterize components if they have the method
        if hasattr(self.mixer, 'reparameterize'):
            self.mixer.reparameterize()
        # FFN does not have reparameterization


class Downsample(nn.Module):
    """
    Convolution-based downsample
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# --- MobileViG Specifications ---
MOBILEVIG_SPECS = {
    'mobilevigv2_ti': {
        'blocks': [[2,0], [2,2], [6,2], [2,2]],
        'channels': [32, 64, 128, 256],
        'emb_dims': 512,
        'K': [0, 8, 4, 2], # K=0 for stage 0 means MGC won't use graph conv effectively
    },
    'mobilevigv2_s': {
        'blocks': [[3,0], [3,3], [9,3], [3,3]],
        'channels': [32, 64, 128, 256],
        'emb_dims': 512,
        'K': [0, 8, 4, 2],
    },
    'mobilevigv2_m': {
        'blocks': [[3,0], [3,3], [9,3], [3,3]],
        'channels': [32, 64, 192, 384],
        'emb_dims': 512,
        'K': [0, 8, 4, 2],
    },
    'mobilevigv2_b': {
        'blocks': [[3,0], [3,3], [9,3], [3,3]],
        'channels': [64, 128, 256, 512],
        'emb_dims': 768,
        'K': [0, 8, 4, 2],
    }
}


# --- Main MobileViG Class ---
class MobileViG(torch.nn.Module):
    def __init__(self, model_name: str,
                 input_size=(3, 640, 640), # Default input size including channels
                 dropout=0.,
                 drop_path=0.1,
                 num_classes=1000, # Kept for potential classifier head usage
                 distillation=True # Kept for potential classifier head usage
                 ):
        super(MobileViG, self).__init__()

        if model_name not in MOBILEVIG_SPECS:
            raise ValueError(f"Unknown model_name: {model_name}. Available: {list(MOBILEVIG_SPECS.keys())}")

        spec = MOBILEVIG_SPECS[model_name]
        blocks = spec['blocks']
        channels = spec['channels']
        emb_dims = spec['emb_dims']
        K = spec['K']

        self.model_name = model_name
        self.input_size = input_size
        self.distillation = distillation # Store for potential later use
        self.num_classes = num_classes  # Store for potential later use

        # Calculate total number of blocks for DropPath
        n_blocks = sum([sum(x) for x in blocks])
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]  # stochastic depth decay rule
        dpr_idx = 0

        # --- Build Network Stages ---
        # Stem
        self.stem = Stem(input_dim=input_size[0], output_dim=channels[0])

        # Backbone Stages
        self.stages = ModuleList()
        current_channels = channels[0]
        for i in range(len(blocks)):
            stage_blocks = []
            target_channels = channels[i]

            # Add Downsample block if not the first stage
            if i > 0:
                stage_blocks.append(Downsample(current_channels, target_channels))
                current_channels = target_channels

            # Local Blocks (InvertedResidual)
            local_stages_count = blocks[i][0]
            for _ in range(local_stages_count):
                stage_blocks.append(InvertedResidual(dim=current_channels, kernel=3, expansion_ratio=4, drop_path=dpr[dpr_idx]))
                dpr_idx += 1 # Increment dpr_idx for each block that uses it

            # Global Blocks (MGC)
            global_stages_count = blocks[i][1]
            for _ in range(global_stages_count):
                # Only add MGC if K[i] > 0, otherwise it's just like another local block (or skip)
                if K[i] > 0:
                     stage_blocks.append(MGC(current_channels, drop_path=dpr[dpr_idx], K=K[i]))
                else:
                    # If K=0, potentially add an equivalent local block or skip
                    # Adding InvertedResidual here to maintain block count for dpr
                     stage_blocks.append(InvertedResidual(dim=current_channels, kernel=3, expansion_ratio=4, drop_path=dpr[dpr_idx]))
                dpr_idx += 1 # Increment dpr_idx for each block that uses it

            self.stages.append(nn.Sequential(*stage_blocks))

        # --- Classification Head (Defined but not used in forward) ---
        # These are kept if you want to add classification capability later
        # outside the main forward pass which returns features.
        self.prediction = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(channels[-1], emb_dims, kernel_size=1, bias=True),
                                        nn.BatchNorm2d(emb_dims),
                                        nn.GELU(),
                                        nn.Dropout(dropout))

        self.head = nn.Conv2d(emb_dims, num_classes, kernel_size=1, bias=True) if num_classes > 0 else nn.Identity()

        if self.distillation:
            self.dist_head = nn.Conv2d(emb_dims, num_classes, 1, bias=True) if num_classes > 0 else nn.Identity()
        else:
            self.dist_head = None

        # --- Initialize Weights ---
        self.model_init()

        # --- Calculate Width List (similar to MobileNetV4) ---
        self.eval() # Set to eval mode for dummy forward pass
        try:
            # Use specified input size for width calculation
            dummy_input = torch.randn(1, *self.input_size)
            features = self.forward(dummy_input)
            # Width list includes stem output and each stage's output channels
            self.width_list = [f.size(1) for f in features]
        except Exception as e:
             print(f"Error during dummy forward pass for width calculation: {e}")
             # Provide default or empty list in case of error
             self.width_list = [channels[0]] + channels # Fallback based on config
        self.train() # Set back to train mode

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                try:
                     torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                     m.weight.requires_grad = True
                     if m.bias is not None:
                         torch.nn.init.zeros_(m.bias)
                         m.bias.requires_grad = True
                except Exception as e:
                     print(f"Could not initialize layer {m}: {e}") # Add error handling
            elif isinstance(m, (torch.nn.BatchNorm2d, nn.GroupNorm)):
                 if m.weight is not None:
                     torch.nn.init.ones_(m.weight)
                     m.weight.requires_grad = True
                 if m.bias is not None:
                     torch.nn.init.zeros_(m.bias)
                     m.bias.requires_grad = True


    def forward(self, inputs: Tensor) -> list[Tensor]:
        """
        Forward pass returning a list of feature maps from stem and each stage.
        """
        feature_maps = []
        x = self.stem(inputs)
        feature_maps.append(x) # Add stem output

        # Pass through each stage
        for stage in self.stages:
            x = stage(x)
            feature_maps.append(x)

        # Returns list of features [stem_out, stage1_out, stage2_out, ...]
        return feature_maps

    def get_classifier_output(self, inputs: Tensor):
        """
        Helper method to get classification output using the defined head.
        You would typically call forward() first to get features if needed elsewhere.
        This provides the original classification behavior.
        """
        features = self.forward(inputs)
        # Use the output of the last stage for prediction
        last_features = features[-1]
        x = self.prediction(last_features)

        if self.distillation and self.dist_head is not None:
            cls_out = self.head(x).squeeze(-1).squeeze(-1)
            dist_out = self.dist_head(x).squeeze(-1).squeeze(-1)
            if not self.training:
                # Average heads during inference
                return (cls_out + dist_out) / 2
            else:
                # Return both during training
                return cls_out, dist_out
        else:
            # No distillation or head not defined properly
            return self.head(x).squeeze(-1).squeeze(-1)

    def reparameterize(self):
        """Reparameterize RepCPE modules within the model."""
        print("Reparameterizing RepCPE modules...")
        for module in self.modules():
             # Specifically target RepCPE or modules containing it (like Grapher)
            if isinstance(module, RepCPE):
                 module.reparameterize()
            elif hasattr(module, 'reparameterize') and not isinstance(module, MobileViG):
                 # Call reparameterize on submodules like Grapher if they have it
                 module.reparameterize()
        print("Reparameterization complete.")
        return self

# --- New Factory Functions (similar to MobileNetV4 style) ---

def mobilevigv2_ti(**kwargs):
    """ MobileViGv2-Tiny """
    model = MobileViG(model_name='mobilevigv2_ti', **kwargs)
    model.default_cfg = default_cfgs['mobilevigv2_ti'] # Attach default cfg
    return model

def mobilevigv2_s(**kwargs):
    """ MobileViGv2-Small """
    model = MobileViG(model_name='mobilevigv2_s', **kwargs)
    model.default_cfg = default_cfgs['mobilevigv2_s'] # Attach default cfg
    return model

def mobilevigv2_m(**kwargs):
    """ MobileViGv2-Medium """
    model = MobileViG(model_name='mobilevigv2_m', **kwargs)
    model.default_cfg = default_cfgs['mobilevigv2_m'] # Attach default cfg
    return model

def mobilevigv2_b(**kwargs):
    """ MobileViGv2-Base """
    model = MobileViG(model_name='mobilevigv2_b', **kwargs)
    model.default_cfg = default_cfgs['mobilevigv2_b'] # Attach default cfg
    return model


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where RepCPE modules are re-parameterized
        for inference.

    :param model: MobileViG model in train mode.
    :return: MobileViG model in inference mode (re-parameterized).
    """
    # Avoid editing original graph
    inference_model = copy.deepcopy(model)
    if hasattr(inference_model, 'reparameterize'):
         inference_model.reparameterize() # Call the model's reparameterize method
    else:
         print("Warning: Model does not have a reparameterize method.")
    inference_model.eval() # Set to eval mode
    return inference_model


# --- Example Usage ---
if __name__ == "__main__":
    # Define input size
    image_size = (1, 3, 640, 640) # Batch size 1, 3 channels, 640x640
    image = torch.randn(*image_size)

    # --- Test Tiny Model ---
    print("--- Testing mobilevigv2_ti ---")
    try:
        model_ti = mobilevigv2_ti(input_size=image_size[1:], num_classes=1000, distillation=True) # Pass input size without batch
        model_ti.eval() # Set to evaluation mode

        # 1. Get Feature Maps
        print(f"Input shape: {image.shape}")
        features_ti = model_ti(image)
        print("Output feature map shapes:")
        for i, f in enumerate(features_ti):
            print(f"  Layer {i}: {f.shape}")
        print(f"Width List (channels): {model_ti.width_list}")

        # 2. Get Classifier Output (Example)
        # Note: This uses the internal head. In practice, you might define your own head.
        classifier_output_ti = model_ti.get_classifier_output(image)
        print(f"Classifier output shape: {classifier_output_ti.shape}") # Should be [1, num_classes]

        # 3. Test Reparameterization
        print("\nReparameterizing model_ti...")
        model_ti_reparam = reparameterize_model(model_ti)
        print("Testing reparameterized model...")
        with torch.no_grad(): # Inference should not require gradients
             features_ti_reparam = model_ti_reparam(image)
             classifier_output_ti_reparam = model_ti_reparam.get_classifier_output(image)

        print("Output feature map shapes (reparameterized):")
        for i, f in enumerate(features_ti_reparam):
            print(f"  Layer {i}: {f.shape}")
        print(f"Classifier output shape (reparameterized): {classifier_output_ti_reparam.shape}")

        # Optional: Check if outputs are close (should be numerically very similar)
        # print(f"Feature difference (last layer): {torch.abs(features_ti[-1] - features_ti_reparam[-1]).mean()}")
        # print(f"Classifier output difference: {torch.abs(classifier_output_ti - classifier_output_ti_reparam).mean()}")


    except Exception as e:
        print(f"Error testing mobilevigv2_ti: {e}")
        import traceback
        traceback.print_exc()


    print("\n--- Testing mobilevigv2_b (Base model) ---")
    try:
        model_b = mobilevigv2_b(input_size=image_size[1:], num_classes=1000, distillation=False) # Example: No distillation
        model_b.eval()

        print(f"Input shape: {image.shape}")
        features_b = model_b(image)
        print("Output feature map shapes:")
        for i, f in enumerate(features_b):
            print(f"  Layer {i}: {f.shape}")
        print(f"Width List (channels): {model_b.width_list}")

        classifier_output_b = model_b.get_classifier_output(image)
        print(f"Classifier output shape: {classifier_output_b.shape}")

    except Exception as e:
        print(f"Error testing mobilevigv2_b: {e}")
        import traceback
        traceback.print_exc()