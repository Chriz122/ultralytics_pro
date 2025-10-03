# -*- coding: utf-8 -*-
"""
RDNet (Refactored)
Based on original RDNet code:
Copyright (c) 2024-present NAVER Cloud Corp.
Apache-2.0

Refactored to follow MobileNetV4 usage patterns.
"""

from functools import partial
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD # Keep for potential metadata
from timm.models.layers.squeeze_excite import EffectiveSEModule
# from timm.models import register_model, build_model_with_cfg, named_apply, generate_default_cfgs # Removed timm registration/build helpers
from timm.models.layers import DropPath, LayerNorm2d # Keep necessary layers and helpers
from timm.models.helpers import named_apply

# __all__ = ["RDNet"] # Let's export the variant functions instead


class RDNetClassifierHead(nn.Module):
    """Classifier head for RDNet, kept separate from feature extraction."""
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        drop_rate: float = 0.,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_features = in_features

        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def reset(self, num_classes):
        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, pre_logits: bool = False):
        # Assumes x is the output of the last feature stage
        x = x.mean([-2, -1]) # Global Average Pooling
        x = self.norm(x)
        x = self.drop(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x


class PatchifyStem(nn.Module):
    def __init__(self, num_input_channels, num_init_features, patch_size=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(num_input_channels, num_init_features, kernel_size=patch_size, stride=patch_size),
            LayerNorm2d(num_init_features),
        )

    def forward(self, x):
        return self.stem(x)


class Block(nn.Module):
    """D == Dw conv, N == Norm, F == Feed Forward, A == Activation"""
    def __init__(self, in_chs, inter_chs, out_chs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, groups=in_chs, kernel_size=7, stride=1, padding=3),
            LayerNorm2d(in_chs, eps=1e-6),
            nn.Conv2d(in_chs, inter_chs, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(inter_chs, out_chs, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.layers(x)


class BlockESE(nn.Module):
    """D == Dw conv, N == Norm, F == Feed Forward, A == Activation"""
    def __init__(self, in_chs, inter_chs, out_chs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, groups=in_chs, kernel_size=7, stride=1, padding=3),
            LayerNorm2d(in_chs, eps=1e-6),
            nn.Conv2d(in_chs, inter_chs, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(inter_chs, out_chs, kernel_size=1, stride=1, padding=0),
            EffectiveSEModule(out_chs),
        )

    def forward(self, x):
        return self.layers(x)


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_input_features,
        growth_rate,
        bottleneck_width_ratio,
        drop_path_rate,
        drop_rate=0.0, # drop_rate seems unused within DenseBlock itself, maybe intended for head?
        # rand_gather_step_prob=0.0, # This param was unused in original code
        block_idx=0,
        block_type="Block",
        ls_init_value=1e-6,
        **kwargs, # Consume potential extra kwargs
    ):
        super().__init__()
        # self.drop_rate = drop_rate # Store if needed later
        self.drop_path_rate = drop_path_rate
        # self.rand_gather_step_prob = rand_gather_step_prob
        self.block_idx = block_idx
        self.growth_rate = growth_rate

        self.gamma = nn.Parameter(ls_init_value * torch.ones(growth_rate)) if ls_init_value > 0 else None
        growth_rate = int(growth_rate)
        inter_chs = int(num_input_features * bottleneck_width_ratio / 8) * 8

        if self.drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)
        else:
            self.drop_path = nn.Identity() # Use Identity if drop_path_rate is 0

        # Instantiate the correct block type
        if block_type == "Block":
            self.layers = Block(
                in_chs=num_input_features,
                inter_chs=inter_chs,
                out_chs=growth_rate,
            )
        elif block_type == "BlockESE":
             self.layers = BlockESE(
                in_chs=num_input_features,
                inter_chs=inter_chs,
                out_chs=growth_rate,
            )
        else:
            raise ValueError(f"Unknown block_type: {block_type}")


    def forward(self, x):
        # DenseNet-style forward: input is a list of features, concat them
        if isinstance(x, List):
            x = torch.cat(x, 1)

        new_features = self.layers(x)

        if self.gamma is not None:
            new_features = new_features.mul(self.gamma.reshape(1, -1, 1, 1))

        # Apply DropPath
        # Note: Original code applied drop_path only if > 0 *and* self.training.
        # DropPath itself handles the training check internally.
        new_features = self.drop_path(new_features)

        return new_features


class DenseStage(nn.Sequential):
    """A stage containing multiple DenseBlocks."""
    def __init__(self, num_block, num_input_features, drop_path_rates, growth_rate, **kwargs):
        super().__init__()
        self.num_out_features_before_cat = num_input_features # Track features *before* final concat
        current_features = num_input_features
        for i in range(num_block):
            layer = DenseBlock(
                num_input_features=current_features,
                growth_rate=growth_rate,
                drop_path_rate=drop_path_rates[i],
                block_idx=i,
                **kwargs,
            )
            current_features += growth_rate
            self.add_module(f"dense_block{i}", layer)
        self.num_out_features = current_features # Features *after* final concat in forward

    def forward(self, init_feature):
        features = [init_feature]
        for module in self:
            new_feature = module(features) # Pass the list of features
            features.append(new_feature)
        # Concatenate all features generated within the stage + initial feature
        return torch.cat(features, 1)


class RDNet(nn.Module):
    """
    Refactored RDNet Feature Extractor Backbone.

    Outputs a list of feature maps from specified stages.
    Includes `width_list` attribute similar to the provided MobileNetV4 example.
    """
    def __init__(
        self,
        num_init_features: int = 64,
        growth_rates: tuple = (64, 104, 128, 128, 128, 128, 224),
        num_blocks_list: tuple = (3, 3, 3, 3, 3, 3, 3),
        bottleneck_width_ratio: float = 4,
        zero_head: bool = False, # Affects classifier head initialization if used separately
        in_chans: int = 3,
        num_classes: int = 1000, # For classifier head if used separately
        drop_rate: float = 0.0, # For classifier head if used separately
        drop_path_rate: float = 0.0,
        # checkpoint_path=None, # Checkpoint loading should be handled externally now
        transition_compression_ratio: float = 0.5,
        ls_init_value: float = 1e-6,
        is_downsample_block: tuple = (None, True, True, False, False, False, True),
        block_type: str or List[str] = "Block",
        head_init_scale: float = 1.,
        output_stages: Optional[List[int]] = None, # Specify which stages to output (0-indexed)
        input_size: tuple = (224, 224), # For width_list calculation
        **kwargs, # Consume potential extra kwargs
    ):
        super().__init__()
        assert len(growth_rates) == len(num_blocks_list) == len(is_downsample_block)

        self.num_classes = num_classes # Store for potential classifier use
        self.drop_rate = drop_rate     # Store for potential classifier use
        self.num_stages = len(growth_rates)

        if isinstance(block_type, str):
            block_type = [block_type] * self.num_stages

        # Stem
        self.stem = PatchifyStem(in_chans, num_init_features, patch_size=4)
        num_features = num_init_features
        self.feature_info = [] # Store info about output features if needed elsewhere
        curr_stride = 4

        # Calculate DropPath rates per block
        total_blocks = sum(num_blocks_list)
        dp_rates = [
            x.tolist() for x in torch.linspace(0, drop_path_rate, total_blocks).split(num_blocks_list)
        ] if total_blocks > 0 else [[0.0]*n for n in num_blocks_list] # Handle zero blocks case


        self.stages = nn.ModuleList() # Use ModuleList to store stages
        for i in range(self.num_stages):
            stage_layers = nn.ModuleDict() # Use ModuleDict for clarity within a stage

            # --- Transition Layer (if not the first stage) ---
            if i != 0:
                compressed_num_features = int(num_features * transition_compression_ratio / 8) * 8
                k_size = stride = 1
                if is_downsample_block[i]:
                    curr_stride *= 2
                    k_size = stride = 2
                stage_layers['norm'] = LayerNorm2d(num_features)
                stage_layers['conv'] = nn.Conv2d(
                    num_features, compressed_num_features, kernel_size=k_size, stride=stride, padding=0
                )
                num_features = compressed_num_features # Input features for the DenseStage

            # --- Dense Stage ---
            stage = DenseStage(
                num_block=num_blocks_list[i],
                num_input_features=num_features,
                growth_rate=growth_rates[i],
                bottleneck_width_ratio=bottleneck_width_ratio,
                # drop_rate=drop_rate, # Pass relevant params
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                block_type=block_type[i],
            )
            stage_layers['dense'] = stage
            # Update num_features to reflect the output of the DenseStage (concatenated features)
            num_features = stage.num_out_features

            # Store feature info (optional, based on original code logic)
            # This condition determines if the *output* of this stage is considered a feature extraction point
            is_feature_stage = (i + 1 == self.num_stages) or \
                               (i + 1 != self.num_stages and is_downsample_block[i + 1])
            if is_feature_stage:
                 self.feature_info.append(
                    dict(
                        num_chs=num_features,
                        reduction=curr_stride,
                        module=f'stages.{i}', # Adjusted module path
                        stage_index=i
                    )
                )

            self.stages.append(stage_layers) # Add the ModuleDict for this stage

        # Define which stages to output features from
        if output_stages is None:
            # Default: Output features from stages marked in feature_info
            self.output_stage_indices = sorted([info['stage_index'] for info in self.feature_info])
            # Alternative Default: Output from *all* stages
            # self.output_stage_indices = list(range(self.num_stages))
        else:
            self.output_stage_indices = sorted(list(set(output_stages)))
            # Validate indices
            if not all(0 <= idx < self.num_stages for idx in self.output_stage_indices):
                 raise ValueError(f"Invalid output_stages. Must be between 0 and {self.num_stages - 1}")

        # --- Classifier Head (kept separate) ---
        self.head_in_features = num_features # Features going into the head if used
        self.head = RDNetClassifierHead(self.head_in_features, num_classes, drop_rate=drop_rate)

        # --- Initialize Weights ---
        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)
        if zero_head:
            if hasattr(self.head, 'fc') and isinstance(self.head.fc, nn.Linear):
                nn.init.zeros_(self.head.fc.weight)
                if self.head.fc.bias is not None:
                    nn.init.zeros_(self.head.fc.bias)

        # --- Calculate Output Width List (like MobileNetV4) ---
        self.eval() # Set to eval mode for dummy pass (disables dropout, etc.)
        with torch.no_grad():
            try:
                # Use the provided input_size or default to 224x224
                h, w = input_size if isinstance(input_size, (list, tuple)) and len(input_size) == 2 else (224, 224)
                dummy_input = torch.randn(1, in_chans, h, w)
                features = self._forward_features_list(dummy_input) # Use the internal feature extraction
                self.width_list = [f.size(1) for f in features]
            except Exception as e:
                print(f"Warning: Could not compute width_list during init: {e}")
                self.width_list = [] # Set empty list on failure
        self.train() # Set back to train mode

    def _forward_features_list(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Internal method to forward through stages and collect specified outputs."""
        x = self.stem(x)
        output_features = []
        current_feature_idx = 0 # Index into self.output_stage_indices

        for i, stage_module_dict in enumerate(self.stages):
            # Apply transition layers if they exist
            if 'norm' in stage_module_dict:
                x = stage_module_dict['norm'](x)
            if 'conv' in stage_module_dict:
                x = stage_module_dict['conv'](x)

            # Apply dense stage
            x = stage_module_dict['dense'](x)

            # Check if the output of this stage should be collected
            if current_feature_idx < len(self.output_stage_indices) and i == self.output_stage_indices[current_feature_idx]:
                 output_features.append(x)
                 current_feature_idx += 1

        # Ensure all requested features were collected (debugging check)
        # if current_feature_idx != len(self.output_stage_indices):
        #     print(f"Warning: Expected {len(self.output_stage_indices)} features, collected {current_feature_idx}")

        return output_features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass for feature extraction. Returns a list of feature maps
        from the stages specified in `output_stages` during initialization.
        """
        return self._forward_features_list(x)

    # --- Methods related to the separate classifier head ---
    @torch.jit.ignore
    def get_classifier(self):
        """Return the classifier Linear layer."""
        return self.head.fc

    def reset_classifier(self, num_classes: int = 0):
        """Reset the classifier head."""
        self.num_classes = num_classes
        self.head.reset(num_classes)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        """
        Forward pass through the classifier head.
        Assumes x is the output of the final feature stage.
        You might need to select the last element from the list returned by forward().
        e.g., model.forward_head(model(input)[-1])
        """
        return self.head(x, pre_logits=pre_logits)


# --- Weight Initialization ---
def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
             nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
         # Keep original logic, but apply scaling specifically to the head's fc layer
        nn.init.normal_(module.weight, std=0.001) # Typical linear init
        nn.init.constant_(module.bias, 0)
        if name and 'head.fc' in name: # Check if it's the head's linear layer
            module.weight.data.mul_(head_init_scale)
            if module.bias is not None:
                module.bias.data.mul_(head_init_scale)


# --- Model Variant Definitions (New Style) ---

def rdnet_tiny(**kwargs: Any) -> RDNet:
    """ RDNet-Tiny """
    n_layer = 7
    model_args: Dict[str, Any] = {
        "num_init_features": 64,
        "growth_rates": tuple([64] + [104] + [128] * 4 + [224]),
        "num_blocks_list": tuple([3] * n_layer),
        "is_downsample_block": (None, True, True, False, False, False, True),
        "transition_compression_ratio": 0.5,
        "block_type": ["Block"] * 2 + ["BlockESE"] * 5, # Corrected length based on n_layer=7
        # Add other default params if needed, e.g., drop_path_rate
        "drop_path_rate": kwargs.get("drop_path_rate", 0.02), # Example default dpr
    }
    model_args.update(kwargs) # Allow overriding defaults
    model = RDNet(**model_args)
    return model

def rdnet_small(**kwargs: Any) -> RDNet:
    """ RDNet-Small """
    n_layer = 11
    model_args: Dict[str, Any] = {
        "num_init_features": 72,
        "growth_rates": tuple([64] + [128] + [128] * (n_layer - 4) + [240] * 2), # n_layer-4 = 7
        "num_blocks_list": tuple([3] * n_layer),
        "is_downsample_block": (None, True, True, False, False, False, False, False, False, True, False),
        "transition_compression_ratio": 0.5,
        "block_type": ["Block"] * 2 + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2, # 2 + 7 + 2 = 11
        "drop_path_rate": kwargs.get("drop_path_rate", 0.05), # Example default dpr
    }
    model_args.update(kwargs)
    model = RDNet(**model_args)
    return model

def rdnet_base(**kwargs: Any) -> RDNet:
    """ RDNet-Base """
    n_layer = 11
    model_args: Dict[str, Any] = {
        "num_init_features": 120,
        "growth_rates": tuple([96] + [128] + [168] * (n_layer - 4) + [336] * 2), # n_layer-4 = 7
        "num_blocks_list": tuple([3] * n_layer),
        "is_downsample_block": (None, True, True, False, False, False, False, False, False, True, False),
        "transition_compression_ratio": 0.5,
        "block_type": ["Block"] * 2 + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2, # 2 + 7 + 2 = 11
        "drop_path_rate": kwargs.get("drop_path_rate", 0.1), # Example default dpr
    }
    model_args.update(kwargs)
    model = RDNet(**model_args)
    return model

def rdnet_large(**kwargs: Any) -> RDNet:
    """ RDNet-Large """
    n_layer = 12
    model_args: Dict[str, Any] = {
        "num_init_features": 144,
        "growth_rates": tuple([128] + [192] + [256] * (n_layer - 4) + [360] * 2), # n_layer-4 = 8
        "num_blocks_list": tuple([3] * n_layer),
        "is_downsample_block": (None, True, True, False, False, False, False, False, False, False, True, False),
        "transition_compression_ratio": 0.5,
        "block_type": ["Block"] * 2 + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2, # 2 + 8 + 2 = 12
        "drop_path_rate": kwargs.get("drop_path_rate", 0.2), # Example default dpr
    }
    model_args.update(kwargs)
    model = RDNet(**model_args)
    return model

__all__ = ['rdnet_tiny', 'rdnet_small', 'rdnet_base', 'rdnet_large', 'RDNet'] # Export functions and class


# --- Example Usage (similar to MobileNetV4 example) ---
if __name__ == "__main__":
    # Generating Sample image
    image_size_h, image_size_w = 224, 224 # Default RDNet size
    # image_size_h, image_size_w = 640, 640 # Or use a larger size like MobileNetV4 example
    image = torch.rand(2, 3, image_size_h, image_size_w) # Batch size 2

    # --- Instantiate different models ---
    print("--- RDNet Tiny ---")
    # Specify which stages to output, e.g., last 4 defined in feature_info
    # Or let it use the default based on feature_info:
    model_tiny = rdnet_tiny(input_size=(image_size_h, image_size_w))
    # Example: Output features from stages 2, 4, and 6 (0-indexed)
    # model_tiny_custom_out = rdnet_tiny(output_stages=[2, 4, 6], input_size=(image_size_h, image_size_w))

    print(f"Width List (Tiny): {model_tiny.width_list}")
    out_tiny = model_tiny(image)
    print("Output Shapes (Tiny):")
    for i, feature_map in enumerate(out_tiny):
        print(f"  Stage {model_tiny.output_stage_indices[i]}: {feature_map.shape}") # Show stage index and shape

    # Example of using the classifier head on the *last* feature map
    final_features_tiny = out_tiny[-1]
    logits_tiny = model_tiny.forward_head(final_features_tiny)
    print(f"Logits Shape (Tiny): {logits_tiny.shape}") # Should be [batch_size, num_classes]

    print("\n--- RDNet Small ---")
    model_small = rdnet_small(drop_path_rate=0.1, input_size=(image_size_h, image_size_w)) # Override drop_path_rate
    print(f"Width List (Small): {model_small.width_list}")
    out_small = model_small(image)
    print("Output Shapes (Small):")
    for i, feature_map in enumerate(out_small):
        print(f"  Stage {model_small.output_stage_indices[i]}: {feature_map.shape}")

    print("\n--- RDNet Base ---")
    # Output all stages
    model_base = rdnet_base(output_stages=list(range(11)), input_size=(image_size_h, image_size_w))
    print(f"Width List (Base - all stages): {model_base.width_list}")
    out_base = model_base(image)
    print("Output Shapes (Base - all stages):")
    for i, feature_map in enumerate(out_base):
        print(f"  Stage {model_base.output_stage_indices[i]}: {feature_map.shape}")

    print("\n--- RDNet Large ---")
    model_large = rdnet_large(input_size=(image_size_h, image_size_w))
    print(f"Width List (Large): {model_large.width_list}")
    out_large = model_large(image)
    print("Output Shapes (Large):")
    for i, feature_map in enumerate(out_large):
        print(f"  Stage {model_large.output_stage_indices[i]}: {feature_map.shape}")