# -*- coding: utf-8 -*-
"""
SwiftFormer modified for easier integration, inspired by MogaNet structure.
"""
import math # Added for weight init like MogaNet
import copy # Keep for potential future use, though less critical now
import torch
import torch.nn as nn
import torch.nn.functional as F # Useful for functional calls if needed

# Keep necessary timm imports used by SwiftFormer's components
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
# Removed timm.models.registry and layers.helpers.to_2tuple if not strictly needed
# from timm.models.layers.helpers import to_2tuple # Replaced with local version if needed
from timm.models.layers import to_2tuple # Keep this one as Embedding uses it
import einops

# --- SwiftFormer Architecture Definitions ---
# Keep these global or move them into the class like MogaNet's arch_zoo if preferred
SwiftFormer_width = {
    'XS': [48, 56, 112, 220],
    'S':  [48, 64, 168, 224],
    'L1': [48, 96, 192, 384], # Renamed l1 -> L1 for consistency
    'L3': [64, 128, 320, 512], # Renamed l3 -> L3 for consistency
}

SwiftFormer_depth = {
    'XS': [3, 3, 6, 4],
    'S':  [3, 3, 9, 6],
    'L1': [4, 3, 10, 5], # Renamed l1 -> L1
    'L3': [4, 4, 12, 6], # Renamed l3 -> L3
}

# --- SwiftFormer Building Blocks (Mostly unchanged) ---

def stem(in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Output: sequence of layers with final shape of [B, C, H/4, W/4]
    """
    # Ensure non-zero channels for BN
    mid_chs = max(1, out_chs // 2)
    out_chs = max(1, out_chs)
    if mid_chs == 0 or out_chs == 0:
        return nn.Identity() # Handle zero channel case
    return nn.Sequential(
        nn.Conv2d(in_chs, mid_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(mid_chs),
        nn.ReLU(),
        nn.Conv2d(mid_chs, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), )

class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        # Ensure non-zero channels for Conv/Norm
        embed_dim = max(1, embed_dim)
        if in_chans == 0 or embed_dim == 0:
             self.proj = nn.Identity()
             self.norm = nn.Identity()
        else:
             self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                   stride=stride, padding=padding)
             self.norm = norm_layer(embed_dim) if norm_layer and embed_dim > 0 else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class ConvEncoder(nn.Module):
    """
    Implementation of ConvEncoder with 3*3 and 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """
    def __init__(self, dim, hidden_dim=64, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        # Ensure non-zero channels
        dim = max(1, dim)
        hidden_dim = max(1, hidden_dim)
        if dim == 0:
            self.dwconv = nn.Identity()
            self.norm = nn.Identity()
            self.pwconv1 = nn.Identity()
            self.act = nn.Identity()
            self.pwconv2 = nn.Identity()
            self.drop_path = nn.Identity()
            self.layer_scale = nn.Identity()
            self.use_layer_scale = False
            return

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            # Ensure scale shape matches dim, handle dim=0 implicitly via check above
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        # Initialize weights here or rely on global apply
        # self.apply(self._init_weights) # Removed: Apply globally later

    # Removed _init_weights from here, will use global one

    def forward(self, x):
        if isinstance(self.dwconv, nn.Identity): # Check for zero dim case
            return x
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            # Ensure layer_scale is applied correctly
            if isinstance(self.layer_scale, nn.Parameter):
                 x = input + self.drop_path(self.layer_scale * x)
            else: # Should not happen if use_layer_scale is True and dim > 0
                 x = input + self.drop_path(x)
        else:
            x = input + self.drop_path(x)
        return x

class Mlp(nn.Module):
    """
    Implementation of MLP layer with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Ensure non-zero channels
        in_features = max(1, in_features)
        hidden_features = max(1, hidden_features)
        out_features = max(1, out_features)

        if in_features == 0 or hidden_features == 0 or out_features == 0:
            self.norm1 = nn.Identity()
            self.fc1 = nn.Identity()
            self.act = nn.Identity()
            self.fc2 = nn.Identity()
            self.drop = nn.Identity()
            return

        self.norm1 = nn.BatchNorm2d(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        # Initialize weights here or rely on global apply
        # self.apply(self._init_weights) # Removed: Apply globally later

    # Removed _init_weights from here

    def forward(self, x):
        if isinstance(self.norm1, nn.Identity): # Check for zero dim case
            return x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """
    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()
        # Ensure non-zero dims
        in_dims = max(1, in_dims)
        token_dim = max(1, token_dim)
        num_heads = max(1, num_heads)
        head_dim = max(1, token_dim * num_heads)

        if in_dims == 0 or token_dim == 0 or head_dim == 0:
             self.to_query = nn.Identity()
             self.to_key = nn.Identity()
             self.w_g = None # No parameter needed
             self.scale_factor = 1.0
             self.Proj = nn.Identity()
             self.final = nn.Identity()
             return

        self.to_query = nn.Linear(in_dims, head_dim)
        self.to_key = nn.Linear(in_dims, head_dim)

        self.w_g = nn.Parameter(torch.randn(head_dim, 1))
        self.scale_factor = head_dim ** -0.5 # Use head_dim for scaling
        self.Proj = nn.Linear(head_dim, head_dim)
        self.final = nn.Linear(head_dim, token_dim) # Output should be token_dim

    def forward(self, x):
        if isinstance(self.to_query, nn.Identity): # Check for zero dim case
            # Need to return something of the expected output dim (token_dim)
            # If input x has correct shape [B,N,in_dims], we can't easily map
            # For simplicity, return input. This block shouldn't be used if dims are 0.
            print("Warning: EfficientAdditiveAttention called with zero dimensions.")
            return x # Or raise error

        query = self.to_query(x)
        key = self.to_key(x)

        query_norm = torch.nn.functional.normalize(query, dim=-1) # BxNxD
        key_norm = torch.nn.functional.normalize(key, dim=-1) # BxNxD

        query_weight = query_norm @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1

        # Note: Normalizing A across N might make sense depending on interpretation
        A = torch.nn.functional.normalize(A, dim=1) # BxNx1

        # Weighted sum of query features based on A
        # G = torch.sum(A * query_norm, dim=1) # BxD (Using normalized query)
        G = torch.sum(A * query, dim=1) # BxD (Using original query might be intended)

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key_norm.shape[1]
        ) # BxNxD

        # Project weighted key features, add query (residual)
        # out = self.Proj(G * key_norm) + query_norm # Using normalized key/query
        out = self.Proj(G * key_norm) + query # Using normalized key, original query

        out = self.final(out) # BxNxD (shape: B, N, token_dim)

        return out

class SwiftFormerLocalRepresentation(nn.Module):
    """
    Local Representation module for SwiftFormer that is implemented by 3*3 depth-wise and point-wise convolutions.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """
    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        # Ensure non-zero channels
        dim = max(1, dim)
        if dim == 0:
            self.dwconv = nn.Identity()
            self.norm = nn.Identity()
            self.pwconv1 = nn.Identity()
            self.act = nn.Identity()
            self.pwconv2 = nn.Identity()
            self.drop_path = nn.Identity()
            self.layer_scale = nn.Identity()
            self.use_layer_scale = False
            return

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim, kernel_size=1) # Original has dim -> dim
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim, dim, kernel_size=1) # Original has dim -> dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        # self.apply(self._init_weights) # Removed: Apply globally later

    # Removed _init_weights from here

    def forward(self, x):
        if isinstance(self.dwconv, nn.Identity): # Check for zero dim case
            return x
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            if isinstance(self.layer_scale, nn.Parameter):
                 x = input + self.drop_path(self.layer_scale * x)
            else:
                 x = input + self.drop_path(x)
        else:
            x = input + self.drop_path(x)
        return x

class SwiftFormerEncoder(nn.Module):
    """
    SwiftFormer Encoder Block. Consists of Local Representation, Attention, and MLP.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        # Ensure non-zero channels
        dim = max(1, dim)
        mlp_hidden_dim = max(1, int(dim * mlp_ratio))
        if dim == 0:
            self.local_representation = nn.Identity()
            self.attn = nn.Identity()
            self.linear = nn.Identity()
            self.drop_path = nn.Identity()
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
            self.use_layer_scale = False
            return

        self.local_representation = SwiftFormerLocalRepresentation(
            dim=dim, kernel_size=3, drop_path=0., use_layer_scale=use_layer_scale # Pass scale usage down
        )
        # In original SwiftFormer, attn output dim = input dim.
        # Setting token_dim=dim ensures final Linear maps back to dim.
        self.attn = EfficientAdditiveAttnetion(in_dims=dim, token_dim=dim, num_heads=1)
        self.linear = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            # Ensure scale shape matches dim
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        if isinstance(self.local_representation, nn.Identity): # Check zero dim
            return x

        # Apply Local Representation Block first
        x = self.local_representation(x) # Includes residual connection inside

        # Attention Block
        B, C, H, W = x.shape
        attn_input = x # Input to attention branch

        # Reshape for attention: B, C, H, W -> B, H*W, C
        attn_input_seq = attn_input.permute(0, 2, 3, 1).reshape(B, H * W, C)
        attn_output_seq = self.attn(attn_input_seq) # Output: B, H*W, C

        # Reshape back: B, H*W, C -> B, C, H, W
        attn_output = attn_output_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Apply scale and residual connection for Attention
        if self.use_layer_scale:
             if isinstance(self.layer_scale_1, nn.Parameter):
                 x = x + self.drop_path(self.layer_scale_1 * attn_output)
             else:
                 x = x + self.drop_path(attn_output)
        else:
            x = x + self.drop_path(attn_output)

        # MLP Block
        mlp_output = self.linear(x) # MLP includes Norm, Conv, Act, Drop, Conv, Drop

        # Apply scale and residual connection for MLP
        if self.use_layer_scale:
             if isinstance(self.layer_scale_2, nn.Parameter):
                 x = x + self.drop_path(self.layer_scale_2 * mlp_output)
             else:
                 x = x + self.drop_path(mlp_output)
        else:
            x = x + self.drop_path(mlp_output)

        return x

def Stage(dim, index, layers, mlp_ratio=4.,
          act_layer=nn.GELU,
          drop_rate=.0, drop_path_rate=0.,
          use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1,
          # Add drop_path base index for correct calculation across stages
          dp_base_idx=0, total_depth=1):
    """
    Implementation of each SwiftFormer stage.
    Uses ConvEncoder for early blocks and SwiftFormerEncoder for `vit_num` last blocks.
    """
    blocks = []
    # Ensure dim is non-zero before creating blocks
    dim = max(1, dim)
    if dim == 0:
        return nn.Sequential(*blocks) # Return empty sequential

    for block_idx in range(layers[index]):
        # Correct drop path rate calculation for the specific block
        block_dpr = drop_path_rate * (dp_base_idx + block_idx) / (total_depth - 1) if total_depth > 1 else 0.0

        if layers[index] - block_idx <= vit_num:
            # Use SwiftFormerEncoder for the last 'vit_num' blocks
            blocks.append(SwiftFormerEncoder(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, # Pass main drop rate to MLP dropout
                drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value))
        else:
            # Use ConvEncoder for the initial blocks
            blocks.append(ConvEncoder(
                dim=dim, hidden_dim=max(1, int(mlp_ratio * dim)), # Ensure hidden_dim > 0
                kernel_size=3, drop_path=block_dpr, # Pass dpr here too
                use_layer_scale=use_layer_scale))

    blocks = nn.Sequential(*blocks)
    return blocks

# --- Main SwiftFormer Model Class ---
class SwiftFormer(nn.Module):
    """
    SwiftFormer implementation, refactored for integration.
    Includes `width_list` attribute and factory function pattern.
    """
    # Removed arch_zoo from inside, using global dicts SwiftFormer_width/depth

    def __init__(self,
                 arch='XS',           # Architecture type (e.g., 'XS', 'S')
                 c1=3,                # Input channels (e.g., 3 for RGB)
                 num_classes=1000,    # Num classes for classification head (if not fork_feat)
                 mlp_ratios=4,        # MLP expansion ratio (can be list or single value)
                 downsamples=[True, True, True, True], # Whether to downsample between stages
                 act_layer=nn.GELU,   # Activation layer type
                 down_patch_size=3,   # Patch size for Embedding/Downsampling
                 down_stride=2,       # Stride for Embedding/Downsampling
                 down_pad=1,          # Padding for Embedding/Downsampling
                 drop_rate=0.,        # Dropout rate for MLP
                 drop_path_rate=0.1,  # Stochastic depth rate
                 use_layer_scale=True,# Use layer scale
                 layer_scale_init_value=1e-5, # Initial value for layer scale
                 fork_feat=True,     # Return features from different stages (for detection/segmentation)
                 vit_num=1,           # Number of SwiftFormerEncoder blocks at the end of each stage
                 distillation=False,  # Use distillation head (for classification)
                 **kwargs):           # Allow absorbing extra args
        super().__init__()

        # --- Determine architecture settings ---
        arch = arch.upper() # Ensure consistency (XS, S, L1, L3)
        if arch not in SwiftFormer_depth or arch not in SwiftFormer_width:
            raise ValueError(f"Unknown SwiftFormer architecture: {arch}. Available: {list(SwiftFormer_depth.keys())}")
        layers = SwiftFormer_depth[arch]
        embed_dims = SwiftFormer_width[arch]
        self.embed_dims = embed_dims # Store for reference
        self.num_stages = len(layers)
        self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.distillation = distillation

        # Expand mlp_ratios if it's a single value
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * self.num_stages

        # --- Set width_list based on embed_dims (Crucial for YOLO integration) ---
        if self.fork_feat:
            # Output channels of each stage for feature extraction
            self.width_list = list(self.embed_dims)
        else:
            # If not forking, width_list might be unused or contain only the final dim
            # Setting it to final dim for potential consistency if needed elsewhere
            self.width_list = [self.embed_dims[-1]] if self.embed_dims else []
        # --- End width_list ---

        # --- Stem Layer ---
        # Use c1 for input channels
        current_channels = embed_dims[0]
        self.patch_embed = stem(c1, current_channels)

        # --- Stochastic depth ---
        total_depth = sum(layers)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)] if total_depth > 0 else []
        cur_block_idx = 0

        # --- Build Network Stages ---
        self.network = nn.ModuleList()
        self.norm_layers = nn.ModuleList() # Store norms separately for fork_feat

        for i in range(self.num_stages):
            stage_layers = layers[i]
            stage_embed_dim = embed_dims[i]
            stage_mlp_ratio = mlp_ratios[i]

            stage = Stage(
                dim=stage_embed_dim, index=i, layers=layers, mlp_ratio=stage_mlp_ratio,
                act_layer=act_layer, drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
                vit_num=vit_num,
                dp_base_idx=cur_block_idx, # Pass base index for dpr calculation
                total_depth=total_depth
            )
            self.network.append(stage)
            cur_block_idx += stage_layers

            # Add norm layer for this stage's output if fork_feat
            # Norm is applied *after* the stage blocks
            if self.fork_feat:
                 # Using BatchNorm2d as in original fork_feat logic
                 norm_layer = nn.BatchNorm2d(stage_embed_dim) if stage_embed_dim > 0 else nn.Identity()
                 self.norm_layers.append(norm_layer)
                 # Keep layer name consistent if needed elsewhere, but ModuleList is cleaner
                 # layer_name = f'norm{i}'
                 # self.add_module(layer_name, norm_layer)


            # Add downsampling layer if not the last stage and required
            if i < self.num_stages - 1:
                next_embed_dim = embed_dims[i+1]
                # Original condition: downsamples[i] or embed_dims[i] != embed_dims[i+1]
                # Simplified: Always add Embedding if not last stage, handles dim change and downsampling
                # (Assuming downsamples[i] is always True based on original SwiftFormer_* calls)
                downsample_layer = Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=stage_embed_dim, embed_dim=next_embed_dim
                    )
                self.network.append(downsample_layer)
                current_channels = next_embed_dim # Update current_channels for next stage input (if needed)


        # --- Final Norm & Classifier Head (Only if not fork_feat) ---
        if not self.fork_feat:
            final_embed_dim = self.embed_dims[-1]
            if final_embed_dim > 0:
                 self.norm = nn.BatchNorm2d(final_embed_dim)
                 self.head = nn.Linear(final_embed_dim, num_classes) if num_classes > 0 else nn.Identity()
                 if self.distillation:
                     self.dist_head = nn.Linear(final_embed_dim, num_classes) if num_classes > 0 else nn.Identity()
                 else:
                     self.dist_head = nn.Identity()
            else:
                 self.norm = nn.Identity()
                 self.head = nn.Identity()
                 self.dist_head = nn.Identity()
        else:
            # If fork_feat, these are not needed for the backbone output
            self.norm = nn.Identity()
            self.head = nn.Identity()
            self.dist_head = nn.Identity()

        # --- Initialize Weights ---
        # Removed complex pretrained loading logic, rely on external handling if needed.
        # Apply generic weight initialization.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ Generic weight init """
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)): # Added GroupNorm
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """ Forward pass returning features based on fork_feat. """
        outs = []
        network_block_idx = 0 # Index for iterating through self.network

        for i in range(self.num_stages):
            # Get stage blocks
            stage = self.network[network_block_idx]
            x = stage(x)
            network_block_idx += 1

            # Apply norm if fork_feat
            if self.fork_feat:
                norm_layer = self.norm_layers[i]
                outs.append(norm_layer(x))

            # Apply downsampling if not the last stage
            if i < self.num_stages - 1:
                downsampler = self.network[network_block_idx]
                x = downsampler(x)
                network_block_idx += 1

        if self.fork_feat:
            # Verify output length matches width_list length if width_list exists
            if self.width_list is not None and len(outs) != len(self.width_list):
                 print(f"Warning: SwiftFormer forward_features output count ({len(outs)}) "
                       f"mismatches stored width_list length ({len(self.width_list)}).")
                 print(f" Output dims: {[o.shape[1] for o in outs]}")
                 print(f" Width list: {self.width_list}")
            return outs # Return list of features [P2, P3, P4, P5] etc. (depends on strides)
        else:
            # If not forking, return only the last stage's output (before final norm/head)
            return x

    def forward(self, x):
        """ Default forward pass. """
        x = self.patch_embed(x)
        x = self.forward_features(x)

        if self.fork_feat:
            # Output features of stages for dense prediction
            return x
        else:
            # Classification forward pass
            x = self.norm(x) # Apply final norm

            # Global average pooling
            pooled_output = x.flatten(2).mean(-1) # B, C

            if self.distillation:
                cls_out = self.head(pooled_output), self.dist_head(pooled_output)
                # For inference, average predictions
                if not self.training:
                    cls_out = (cls_out[0] + cls_out[1]) / 2
            else:
                cls_out = self.head(pooled_output)

            return cls_out


# --- Default CFG Function (Similar to timm's _cfg) ---
def _cfg(url='', **kwargs):
    # Removed registration, just return cfg dict
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head', # Main head name
        **kwargs
    }

# --- Factory Functions (Modified) ---
# Removed @register_model, directly instantiate SwiftFormer

def SwiftFormer_XS(pretrained=False, **kwargs):
    """ SwiftFormer-XS """
    # pretrained argument is kept for compatibility but not used for loading here
    if pretrained:
        print("Warning: pretrained=True ignored in SwiftFormer_XS factory. Load weights externally.")
    model = SwiftFormer(arch='XS', **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9) # Set default config if needed
    return model

def SwiftFormer_S(pretrained=False, **kwargs):
    """ SwiftFormer-S """
    if pretrained:
        print("Warning: pretrained=True ignored in SwiftFormer_S factory. Load weights externally.")
    model = SwiftFormer(arch='S', **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model

def SwiftFormer_L1(pretrained=False, **kwargs):
    """ SwiftFormer-L1 """
    if pretrained:
        print("Warning: pretrained=True ignored in SwiftFormer_L1 factory. Load weights externally.")
    model = SwiftFormer(arch='L1', **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model

def SwiftFormer_L3(pretrained=False, **kwargs):
    """ SwiftFormer-L3 """
    if pretrained:
        print("Warning: pretrained=True ignored in SwiftFormer_L3 factory. Load weights externally.")
    model = SwiftFormer(arch='L3', **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("--- Testing Refactored SwiftFormer ---")
    image_size_test = (1, 3, 640, 640) # Test with YOLO-like input size
    image = torch.rand(*image_size_test)

    print("\n--- Testing Feature Extraction Mode (fork_feat=True) ---")
    try:
        # Test feature extraction mode (default fork_feat=True)
        model_feat = SwiftFormer_S(c1=3, fork_feat=True) # Use factory
        # Or directly: model_feat = SwiftFormer(arch='S', c1=3, fork_feat=True)
        model_feat.eval()

        print(f"SwiftFormer-S (features) Initialized.")
        # Check if width_list exists and matches embed_dims
        print(f"  Architecture embed_dims: {model_feat.embed_dims}")
        print(f"  Stored width_list: {model_feat.width_list}")
        assert hasattr(model_feat, 'width_list')
        assert model_feat.width_list == model_feat.embed_dims, \
            f"Mismatch: width_list {model_feat.width_list} vs embed_dims {model_feat.embed_dims}"

        with torch.no_grad():
            out_feat = model_feat(image) # Calls forward -> forward_features

        print(f"  Input shape: {image.shape}")
        print(f"  Output: List of {len(out_feat)} tensors")
        output_channels = []
        for i, feat in enumerate(out_feat):
            print(f"    Stage {i+1} Feature Shape: {feat.shape}")
            output_channels.append(feat.shape[1])

        # Verify output channels match width_list
        print(f"  Output channels: {output_channels}")
        assert output_channels == model_feat.width_list, \
             f"Output channel mismatch: {output_channels} vs {model_feat.width_list}"
        print("Feature extraction test passed.")

    except Exception as e:
        print(f"Error during feature extraction test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing Classification Mode (fork_feat=False) ---")
    try:
         # Test classification mode
         model_cls = SwiftFormer_XS(c1=3, num_classes=100, fork_feat=False) # Use factory
         model_cls.eval()
         print(f"SwiftFormer-XS (classification) Initialized.")
         print(f"  Stored width_list: {model_cls.width_list}") # Should be last embed_dim
         assert model_cls.width_list == [SwiftFormer_width['XS'][-1]], \
             f"Classification width_list mismatch: {model_cls.width_list} vs {[SwiftFormer_width['XS'][-1]]}"

         with torch.no_grad():
            out_cls = model_cls(image)
         print(f"  Input shape: {image.shape}")
         print(f"  Classification Output Shape: {out_cls.shape}")
         assert out_cls.shape == (image_size_test[0], 100), \
             f"Classification output shape mismatch: {out_cls.shape} vs {(image_size_test[0], 100)}"
         print("Classification test passed.")

    except Exception as e:
        print(f"Error during classification test: {e}")
        import traceback
        traceback.print_exc()