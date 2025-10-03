import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite, DropPath, to_2tuple
# from timm.models.registry import register_model # Removed, using factory functions instead

# --- Helper Modules (Mostly unchanged from Code 1) ---

class GroupNorm(torch.nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        # Corrected Conv2d initialization arguments for fused module
        m = torch.nn.Conv2d(
            in_channels=c.in_channels,
            out_channels=c.out_channels,
            kernel_size=c.kernel_size,
            stride=c.stride,
            padding=c.padding,
            dilation=c.dilation,
            groups=c.groups,
            bias=True,  # Bias is now True
            device=c.weight.device
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    # Keep this if a final classification head is needed separately
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - bn.running_mean * \
            bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            # Calculate bias for the fused linear layer when original linear layer has no bias
            b = b @ l.weight.T
        else:
            # Calculate bias for the fused linear layer when original linear layer has bias
            b = (l.weight @ b[:, None]).view(-1) + l.bias
        # Corrected Linear initialization arguments for fused module
        m = torch.nn.Linear(
            in_features=l.in_features,
            out_features=l.out_features,
            bias=True,  # Bias is now True
            device=l.weight.device
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    # SHViT's specific PatchMerging/Downsampling block
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4) # Increased intermediate dimension based on original code's logic
        # Use ReLU consistently as activation based on original
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.act1 = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim) # Stride 2 for downsampling
        self.act2 = torch.nn.ReLU()
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.se(x)
        x = self.conv3(x)
        return x

# Original Residual used stochastic depth, let's align with Swin's DropPath
# class Residual(torch.nn.Module):
#     def __init__(self, m, drop=0.):
#         super().__init__()
#         self.m = m
#         self.drop = drop # This was stochastic depth rate

#     def forward(self, x):
#         if self.training and self.drop > 0:
#             # Original implementation was different from standard stochastic depth
#             return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
#                                               device=x.device).ge_(self.drop).div(1 - self.drop).detach()
#         else:
#              return x + self.m(x)

# Replaced with standard DropPath for consistency and fuse compatibility
class Residual(nn.Module):
    def __init__(self, m: nn.Module, drop_path=0.):
        super().__init__()
        self.m = m
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.m(x))

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m_fused = self.m.fuse()
            # Check if it's a depthwise convolution (groups == in_channels == out_channels)
            if m_fused.groups == m_fused.in_channels and m_fused.groups == m_fused.out_channels and m_fused.kernel_size == (3,3) and m_fused.padding == (1,1):
                # Create identity kernel
                identity = torch.zeros_like(m_fused.weight)
                identity_indices = torch.arange(m_fused.out_channels)
                # Set center element to 1 for each output channel's corresponding input channel
                identity[identity_indices, 0, 1, 1] = 1.0 # Assuming groups=in_channels, so input channel index is 0

                # Add identity to the weight
                m_fused.weight.data += identity
                # The bias already exists from fusing Conv+BN
                # No need to add bias here unless the original Residual logic required it differently
                return m_fused
            else:
                 # Cannot fuse identity for non-depthwise or non-3x3 convs this way
                 print(f"Warning: Residual fusion skipped for non-depthwise 3x3 Conv2d_BN: {self.m}")
                 return self # Return unfused module
        elif hasattr(self.m, 'fuse'): # Attempt to fuse inner module if possible (e.g., SHSA might have fusable parts)
             try:
                 self.m = self.m.fuse() # Fuse the inner module
             except Exception as e:
                 print(f"Warning: Could not fuse inner module of Residual: {e}")
             return self # Return the Residual block (possibly with fused inner module)
        else:
            # If inner module is not Conv2d_BN or fusable, return original
            return self


class FFN(torch.nn.Module):
    def __init__(self, ed, h, drop_path=0.): # Added drop_path
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU() # Changed to ReLU based on original code
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)
        # Drop path is typically applied *after* the residual connection in blocks like Swin/ViT
        # If FFN is wrapped in Residual, drop path is handled there.
        # If FFN is used standalone, it might need its own drop path, but usually not.

    def forward(self, x):
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x


class SHSA(torch.nn.Module):
    """Single-Head Self-Attention"""
    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim # Partial dimension for attention

        # Use GroupNorm as per original definition
        self.pre_norm = GroupNorm(pdim)

        # Combine QKV projection into one Conv + BN for potential fusion
        # Output channels: qk_dim (Q) + qk_dim (K) + pdim (V)
        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = torch.nn.Sequential(
            torch.nn.ReLU(), # Activation after projection, common practice
            Conv2d_BN(dim, dim, bn_weight_init=0) # Final projection
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Split input into the part going through attention (pdim) and the part kept aside
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)

        # Apply pre-normalization only to the attention part
        x1_norm = self.pre_norm(x1)
        qkv = self.qkv(x1_norm)

        # Split Q, K, V
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)

        # Reshape for attention calculation (B, num_heads * C_head, H, W) -> (B, num_heads, C_head, N) -> (B, num_heads, N, C_head) for Q
        # Here, num_heads = 1 effectively
        q = q.flatten(2) # (B, qk_dim, H*W)
        k = k.flatten(2) # (B, qk_dim, H*W)
        v = v.flatten(2) # (B, pdim, H*W)

        # Attention calculation: Q^T * K
        attn = (q.transpose(-2, -1) @ k) * self.scale # (B, N, qk_dim) @ (B, qk_dim, N) -> (B, N, N)
        attn = attn.softmax(dim=-1)

        # Apply attention to V: V * Attn^T
        x1_attn = (v @ attn.transpose(-2, -1)) # (B, pdim, N) @ (B, N, N) -> (B, pdim, N)
        x1_attn = x1_attn.reshape(B, self.pdim, H, W) # Reshape back to spatial dimensions

        # Concatenate attended part and the rest
        x_concat = torch.cat([x1_attn, x2], dim=1)

        # Final projection
        x_out = self.proj(x_concat)

        return x_out

    @torch.no_grad()
    def fuse(self):
        # Potentially fuse Conv2d_BN layers inside qkv and proj
        if hasattr(self.qkv, 'fuse'):
             self.qkv = self.qkv.fuse()
        if hasattr(self.proj[1], 'fuse'): # proj[1] is Conv2d_BN
             self.proj[1] = self.proj[1].fuse()
        # Note: GroupNorm is generally not fused. Pre-norm stays.
        return self


class BasicBlock(torch.nn.Module):
    # Aligned more with Swin Block structure (Norm -> Attention/Conv -> DropPath -> Norm -> FFN -> DropPath)
    # But SHViT uses Conv -> Mixer -> FFN, each wrapped in Residual. Let's keep that structure.
    def __init__(self, dim, qk_dim, pdim, block_type="s", mlp_ratio=2., drop_path=0.):
        super().__init__()
        self.type = block_type

        # 1. Depthwise Convolution Block
        self.conv = Residual(
            Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0),
            drop_path=drop_path
        )

        # 2. Mixer Block (SHSA or Identity)
        if block_type == "s":  # Use SHSA for later stages
            self.mixer = Residual(
                SHSA(dim, qk_dim, pdim),
                drop_path=drop_path
            )
        elif block_type == "i": # Use Identity for early stages (no attention)
            # If mixer is Identity, no need for Residual or DropPath around it
            self.mixer = nn.Identity()
        else:
            raise ValueError(f"Unknown BasicBlock type: {block_type}")

        # 3. FFN Block
        ffn_hidden_dim = int(dim * mlp_ratio)
        self.ffn = Residual(
            FFN(dim, ffn_hidden_dim),
            drop_path=drop_path
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.mixer(x)
        x = self.ffn(x)
        return x

    @torch.no_grad()
    def fuse(self):
         # Fuse each residual block
         if hasattr(self.conv, 'fuse'):
             self.conv = self.conv.fuse()
         if hasattr(self.mixer, 'fuse'): # This will attempt to fuse SHSA if it exists and is Residual wrapped
             self.mixer = self.mixer.fuse()
         if hasattr(self.ffn, 'fuse'):
             self.ffn = self.ffn.fuse()
         return self

# --- Main SHViT Model ---

class SHViT(torch.nn.Module):
    def __init__(self,
                 in_chans=3,
                 # num_classes=1000, # Removed: Backbone doesn't need num_classes
                 embed_dim=[128, 256, 384],
                 partial_dim=[32, 64, 96],
                 qk_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 types=["s", "s", "s"],
                 mlp_ratio=2., # Standard MLP ratio for FFN
                 drop_path_rate=0.1, # Overall drop path rate, will be distributed
                 # down_ops=[['subsample', 2], ['subsample', 2], ['']], # Replaced by embed_dim transitions
                 norm_layer=GroupNorm, # Use GroupNorm consistent with SHSA, Swin uses LayerNorm
                 out_indices=(0, 1, 2), # Indices of stages whose outputs to return
                 use_checkpoint=False # Added checkpointing option like Swin
                 ):
        super().__init__()

        self.num_layers = len(depth)
        self.embed_dim = embed_dim
        self.partial_dim = partial_dim
        self.qk_dim = qk_dim
        self.types = types
        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint
        self.num_features = embed_dim # List of feature dimensions at each stage output

        # 1. Patch Embedding (Original SHViT style) - stride 16 total
        self.patch_embed = torch.nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1), torch.nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1), torch.nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), torch.nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1)
        )
        patch_embed_out_dim = embed_dim[0]

        # Stochastic depth decay rule (like Swin)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        # dpr_idx = 0 # Original used fixed drop rate per block in Residual

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dp_offset = 0 # Offset for stochastic depth indices

        # 2. Build Stages and Downsampling Layers
        current_dim = patch_embed_out_dim
        for i in range(self.num_layers):
            target_dim = embed_dim[i]
            stage_depth = depth[i]
            stage_pdim = partial_dim[i]
            stage_qkdim = qk_dim[i]
            stage_type = types[i]

            # Downsampling layer before the stage (if dimensions change)
            if current_dim != target_dim:
                 # Use the specific SHViT PatchMerging block for downsampling
                 downsample_layer = PatchMerging(current_dim, target_dim)
                 # Note: Original code added extra blocks around merging.
                 # Keeping it simpler here, assuming PatchMerging handles the transition.
                 # If those extra blocks are crucial, they need to be added here or inside PatchMerging.
                 self.downsamples.append(downsample_layer)
                 current_dim = target_dim # Update current_dim after downsampling
            else:
                 # If no dimension change, add Identity (needed to match stages list length if first stage uses embed_dim[0])
                 # We apply downsampling *after* the stage, so need one less downsample layer.
                 # Let's rethink: Apply downsampling *after* stage i, before stage i+1.
                 pass # Downsampling logic will be handled after the stage loop


            # Stage blocks
            stage_dpr = dpr[dp_offset : dp_offset + stage_depth] # Stochastic depth rates for this stage
            blocks = []
            for j in range(stage_depth):
                block = BasicBlock(
                    dim=current_dim,
                    qk_dim=stage_qkdim,
                    pdim=stage_pdim,
                    block_type=stage_type,
                    mlp_ratio=mlp_ratio,
                    drop_path=stage_dpr[j] # Use per-block stochastic depth
                )
                if use_checkpoint:
                    # block = checkpoint_wrapper(block) # Requires importing checkpoint_wrapper or using torch.utils.checkpoint
                    pass # Checkpointing not fully implemented here to avoid extra imports
                blocks.append(block)
            self.stages.append(nn.Sequential(*blocks))
            dp_offset += stage_depth

            # Add downsampling layer *after* the stage (except for the last one)
            if i < self.num_layers - 1:
                downsample_layer = PatchMerging(current_dim, embed_dim[i+1])
                self.downsamples.append(downsample_layer)
                current_dim = embed_dim[i+1] # Update dim for the next iteration's logic (if needed)
            else:
                 # No downsampling after the last stage
                 self.downsamples.append(nn.Identity())


        # 3. Add normalization layer for each output feature map (like Swin)
        for i in out_indices:
            if i >= len(self.num_features):
                raise ValueError(f"out_indice {i} is out of bounds for {len(self.num_features)} stages.")
            layer = norm_layer(self.num_features[i])
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        # Removed classification head - it belongs to downstream task model

        # 4. Calculate output widths (like Swin) - requires a forward pass
        # Use a dummy input size representative of expected inputs, e.g., 224x224
        # Crucially, call self.forward, which should now return a list of features
        self.eval() # Ensure model is in eval mode for width calculation (e.g., disable dropout)
        with torch.no_grad():
             # Make sure forward returns a list/tuple
             dummy_input = torch.randn(1, in_chans, 224, 224)
             features = self.forward(dummy_input)
             if isinstance(features, (list, tuple)):
                 self.width_list = [f.size(1) for f in features] # Get channel dimension (C in NCHW)
             else:
                  print("Warning: Forward pass for width calculation did not return a list/tuple. Width list may be incorrect.")
                  # Handle case where forward might still return single tensor if out_indices is problematic
                  self.width_list = [features.size(1)] if isinstance(features, torch.Tensor) else []

        self.train() # Return model to train mode

    def forward(self, x):
        """Forward function to extract features from specified stages."""
        x = self.patch_embed(x)
        outs = []
        current_stage_output = x
        for i in range(self.num_layers):
            stage = self.stages[i]
            downsample = self.downsamples[i]

            # Apply stage blocks
            if self.use_checkpoint:
                 current_stage_output = torch.utils.checkpoint.checkpoint_sequential(stage, len(stage), current_stage_output)
            else:
                 current_stage_output = stage(current_stage_output)

            # Check if output of this stage is requested
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                normed_output = norm_layer(current_stage_output)
                outs.append(normed_output)

            # Apply downsampling to prepare for the next stage (or if it's the last stage's identity op)
            # Important: Downsample *after* saving the output for the current stage
            current_stage_output = downsample(current_stage_output)


        # Return the list of feature maps
        # This directly addresses the "AttributeError: 'Tensor' object has no attribute 'insert'"
        return outs

    @torch.no_grad()
    def fuse(self):
         # Fuse patch_embed
         for i in range(len(self.patch_embed)):
              if hasattr(self.patch_embed[i], 'fuse'):
                  self.patch_embed[i] = self.patch_embed[i].fuse()
         # Fuse stages
         for i in range(len(self.stages)):
             for j in range(len(self.stages[i])): # Iterate through blocks in Sequential stage
                  if hasattr(self.stages[i][j], 'fuse'):
                      self.stages[i][j] = self.stages[i][j].fuse()
         # Fuse downsamples
         for i in range(len(self.downsamples)):
              if hasattr(self.downsamples[i], 'fuse'):
                  self.downsamples[i] = self.downsamples[i].fuse()
              elif isinstance(self.downsamples[i], nn.Sequential): # If downsample is Sequential, fuse its parts
                  for j in range(len(self.downsamples[i])):
                       if hasattr(self.downsamples[i][j], 'fuse'):
                           self.downsamples[i][j] = self.downsamples[i][j].fuse()

         # Note: Norm layers (GroupNorm) are typically not fused.
         return self

# --- Configuration Dictionaries (like Code 1) ---

SHViT_s1_cfg = {
        'embed_dim': [128, 224, 320],
        'depth': [2, 4, 5],
        'partial_dim': [32, 48, 68],
        'qk_dim': [16, 16, 16], # Original SHViT paper might have details on qk_dim per size
        'types' : ["i", "s", "s"], # Example: first stage is conv only, rest use SHSA
        'drop_path_rate': 0.1 # Example drop path rate
}

SHViT_s2_cfg = {
        'embed_dim': [128, 308, 448],
        'depth': [2, 4, 5],
        'partial_dim': [32, 66, 96],
        'qk_dim': [16, 16, 16],
        'types' : ["i", "s", "s"],
        'drop_path_rate': 0.15
}

SHViT_s3_cfg = {
        'embed_dim': [192, 352, 448],
        'depth': [3, 5, 5],
        'partial_dim': [48, 75, 96],
        'qk_dim': [16, 16, 16],
        'types' : ["i", "s", "s"],
        'drop_path_rate': 0.2
}

SHViT_s4_cfg = {
        'embed_dim': [224, 336, 448],
        'depth': [4, 7, 6],
        'partial_dim': [48, 72, 96],
        'qk_dim': [16, 16, 16],
        'types' : ["i", "s", "s"],
        'drop_path_rate': 0.25
}


# --- Factory Functions (like SwinTransformer) ---

__all__ = ['SHViT_S1', 'SHViT_S2', 'SHViT_S3', 'SHViT_S4'] # Export factory functions

def _load_pretrained(model, weights_path=None, fuse=False):
    """Helper function to load weights and potentially fuse."""
    if weights_path:
        print(f"Loading weights from: {weights_path}")
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            # Adjust keys based on how weights were saved (e.g., 'model', 'state_dict')
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

            # --- Key Matching Logic (similar to Swin's update_weight) ---
            current_dict = model.state_dict()
            loaded_keys = state_dict.keys()
            matched_keys, unmatched_keys, shape_mismatch_keys = [], [], []
            new_state_dict = {}

            for k in current_dict.keys():
                if k in loaded_keys:
                    if current_dict[k].shape == state_dict[k].shape:
                        new_state_dict[k] = state_dict[k]
                        matched_keys.append(k)
                    else:
                        # Handle potential shape mismatches if necessary (e.g., adapting FC layer)
                        # Example: adapting a classification head (if it existed)
                        # if 'head' in k and current_dict[k].shape != state_dict[k].shape:
                        #     print(f"Adapting shape for {k}")
                        #     # Add specific adaptation logic here if needed
                        # else:
                        shape_mismatch_keys.append(k)
                        # Keep the model's initialized weight for shape mismatches
                        new_state_dict[k] = current_dict[k]
                else:
                    unmatched_keys.append(k)
                    # Keep the model's initialized weight for unmatched keys
                    new_state_dict[k] = current_dict[k]

            print(f"Matched keys: {len(matched_keys)}/{len(current_dict)}")
            if unmatched_keys:
                 print(f"Warning: Unmatched keys in model state_dict: {unmatched_keys}")
            if shape_mismatch_keys:
                 print(f"Warning: Shape mismatch for keys: {shape_mismatch_keys}")
            # --- End Key Matching ---

            model.load_state_dict(new_state_dict, strict=False) # strict=False because we handled mismatches

        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Proceeding with randomly initialized weights.")

    if fuse:
        print("Fusing layers...")
        model.fuse()
    return model

def SHViT_S1(pretrained=False, weights_path=None, fuse=False, **kwargs):
    """Instantiates SHViT S1 model."""
    model = SHViT(**SHViT_s1_cfg, **kwargs) # Pass specific config and any overrides
    model = _load_pretrained(model, weights_path if pretrained else None, fuse)
    return model

def SHViT_S2(pretrained=False, weights_path=None, fuse=False, **kwargs):
    """Instantiates SHViT S2 model."""
    model = SHViT(**SHViT_s2_cfg, **kwargs)
    model = _load_pretrained(model, weights_path if pretrained else None, fuse)
    return model

def SHViT_S3(pretrained=False, weights_path=None, fuse=False, **kwargs):
    """Instantiates SHViT S3 model."""
    model = SHViT(**SHViT_s3_cfg, **kwargs)
    model = _load_pretrained(model, weights_path if pretrained else None, fuse)
    return model

def SHViT_S4(pretrained=False, weights_path=None, fuse=False, **kwargs):
    """Instantiates SHViT S4 model."""
    model = SHViT(**SHViT_s4_cfg, **kwargs)
    model = _load_pretrained(model, weights_path if pretrained else None, fuse)
    return model

# --- BN Fusion Function (from Code 1, slightly adapted fuse methods above) ---
def replace_batchnorm(net):
    # This function is implicitly replaced by calling model.fuse()
    # Keeping it here for reference, but the logic is now within the modules' fuse methods
    # and the Residual fuse method.
    print("Note: replace_batchnorm is deprecated. Call model.fuse() instead.")
    net.fuse() # Call the top-level fuse method
    # for child_name, child in net.named_children():
    #     if hasattr(child, 'fuse'):
    #         fused = child.fuse()
    #         setattr(net, child_name, fused)
    #         # replace_batchnorm(fused) # Recursion handled within fuse methods now
    #     elif isinstance(child, torch.nn.BatchNorm2d):
    #         # Fusing should handle BN replacement within Conv2d_BN etc.
    #         # If standalone BNs exist, they might need explicit handling or indicate an issue.
    #         print(f"Warning: Found standalone BatchNorm2d '{child_name}', which might not be handled by fusion.")
    #         # setattr(net, child_name, torch.nn.Identity()) # This might be too aggressive
    #     else:
    #          replace_batchnorm(child) # Recurse for non-fusable modules


# --- Example Usage / Test ---
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Instantiate a model (e.g., S1)
    # Set out_indices=[0, 1, 2] to get features after each stage
    model = SHViT_S1(out_indices=(0, 1, 2)).to(device)
    model.eval()

    # Create a dummy input tensor
    # Input size example: (Batch, Channels, Height, Width)
    # YOLO often uses larger inputs like 640x640
    inputs = torch.randn((1, 3, 224, 224)).to(device)
    # inputs = torch.randn((1, 3, 640, 640)).to(device) # Example for YOLO

    # Perform a forward pass
    with torch.no_grad():
        features = model(inputs)

    # Check the output
    print(f"\nModel: {model.__class__.__name__} (S1 variant)")
    print(f"Number of output feature maps: {len(features)}")
    for i, f in enumerate(features):
        print(f"Feature map {i} shape: {f.shape}") # Should be [B, C, H, W]

    # Check the calculated width list
    print(f"Calculated width_list: {model.width_list}")
    # Verify width_list matches output channel dimensions
    assert model.width_list == [f.size(1) for f in features]

    # Example of fusion
    print("\nTesting Fusion:")
    model_fused = SHViT_S1(out_indices=(0, 1, 2), fuse=True).to(device)
    model_fused.eval()
    with torch.no_grad():
        features_fused = model_fused(inputs)
    print("Fused model output shapes:")
    for i, f in enumerate(features_fused):
        print(f"Feature map {i} shape: {f.shape}")

    # Optional: Compare outputs (should be numerically close if fusion is correct)
    # print(torch.allclose(features[0], features_fused[0], atol=1e-5))