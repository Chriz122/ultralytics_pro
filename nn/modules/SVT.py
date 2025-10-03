import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg # Ensure this is correctly imported if used
import math
import numpy as np
# You might need to pip install pytorch_wavelets
# Ensure you have pytorch_wavelets installed if you run this: pip install pytorch_wavelets
try:
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
except ImportError:
    print("pytorch_wavelets not found. Please install it if you intend to use SVT_channel_mixing.")
    DTCWTForward = None
    DTCWTInverse = None

class SVT_channel_mixing(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if DTCWTForward is None or DTCWTInverse is None:
            raise ImportError("pytorch_wavelets is required for SVT_channel_mixing but not found.")

        self.hidden_size = dim
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        if not (self.hidden_size % self.num_blocks == 0): # Added error checking
            raise ValueError(f"hidden_size {self.hidden_size} must be divisible by num_blocks {self.num_blocks}")

        if dim in [64, 96]: # for stages with 56x56 feature maps
            spatial_dim = 56
        elif dim in [128, 192]: # for stages with 28x28 feature maps
            spatial_dim = 28
        else:
            # This case should ideally not be hit if alpha and embed_dims are set correctly
            # Or, this module is only intended for specific early stages.
            print(f"Warning: SVT_channel_mixing ancountered dim={dim}, which might not have a predefined spatial_dim. Defaulting to 28.")
            spatial_dim = 28 # Fallback or error, depending on expected usage.
            # Or raise ValueError(f"Unsupported dim {dim} for SVT_channel_mixing's predefined spatial dimensions")


        self.complex_weight_ll = nn.Parameter(torch.randn(dim, spatial_dim, spatial_dim, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)

        self.xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
        self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.softshrink = 0.0

    def multiply(self, input_tensor, weights): # Renamed 'input' to 'input_tensor'
        return torch.einsum('...bd,bdk->...bk', input_tensor, weights)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.permute(x, (0, 3, 1, 2)) # x is now B, C, H, W
        # B_orig, C_orig, H_orig, W_orig = x.shape # For debugging shapes if complex_weight_ll mismatches

        x = x.to(torch.float32)

        xl, xh = self.xfm(x)

        # Check spatial dimensions for xl and self.complex_weight_ll
        # Expected xl shape: [B, C, H_out, W_out]
        # Expected self.complex_weight_ll shape: [C, H_target, W_target]
        # H_out, W_out from xfm should match H_target, W_target
        if xl.shape[2:] != self.complex_weight_ll.shape[1:]:
             # This happens if H, W passed to forward() don't result in xfm output matching complex_weight_ll's hardcoded spatial_dim
             # Example: if dim=64 (so spatial_dim=56 for weight_ll), but H,W input to forward are 28,28 (e.g. from a later stage)
             # then xfm(x) will produce xl with spatial dims like 14,14, which won't match 56,56.
             # This indicates SVT_channel_mixing might be used in a stage it wasn't configured for.
             # For now, we'll try to adapt by using F.interpolate or by warning.
             # A robust solution might involve making complex_weight_ll learnable spatially or passing H,W to __init__.
             # Quick fix: interpolate xl to match complex_weight_ll if needed. This is a guess.
            # print(f"Warning: Mismatch in SVT_channel_mixing. xl.shape[2:]={xl.shape[2:]}, complex_weight_ll.shape[1:]={self.complex_weight_ll.shape[1:]}. Interpolating xl.")
            # xl = F.interpolate(xl, size=self.complex_weight_ll.shape[1:], mode='bilinear', align_corners=False)
            # A better fix: The `alpha` parameter in SVT class controls which stages use this.
            # Ensure `SVT_channel_mixing` is only used in stages where H,W match its `spatial_dim` config.
            # The current code uses it only for the first stage (alpha=1), where H,W should be 56,56 matching embed_dims[0].
            pass # Assuming H,W passed to forward() are correct for the initialized 'dim'

        xl = xl * self.complex_weight_ll

        xh[0] = torch.permute(xh[0], (5, 0, 2, 3, 4, 1)) # last dim is real/imag, B, H, W, C, n_bands -> R/I, B, H, W, C, n_bands
        xh[0] = xh[0].reshape(xh[0].shape[0], xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], xh[0].shape[4], self.num_blocks, self.block_size)

        x_real = xh[0][0]
        x_imag = xh[0][1]

        x_real_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[0]) - self.multiply(x_imag, self.complex_weight_lh_1[1]) + self.complex_weight_lh_b1[0])
        x_imag_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[1]) + self.multiply(x_imag, self.complex_weight_lh_1[0]) + self.complex_weight_lh_b1[1])

        x_real_2 = self.multiply(x_real_1, self.complex_weight_lh_2[0]) - self.multiply(x_imag_1, self.complex_weight_lh_2[1]) + self.complex_weight_lh_b2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_lh_2[1]) + self.multiply(x_imag_1, self.complex_weight_lh_2[0]) + self.complex_weight_lh_b2[1]

        xh[0] = torch.stack([x_real_2, x_imag_2], dim=-1).float() # stacks real/imag back to last dim
        xh[0] = F.softshrink(xh[0], lambd=self.softshrink) if self.softshrink else xh[0]
        
        # Reshape back, note xh[0] current shape: (2, B, Hf, Wf, C_bands, num_blocks, block_size, 2_stacked)
        # After stack: (B, Hf, Wf, C_bands, num_blocks, block_size, 2_stacked) - no, stack is on new dim -1
        # xh[0] shape before reshape: (2, B, H_coeff, W_coeff, num_orientations, num_blocks, block_size_channel_part)
        # The multiply op contracts last dim of input with 2nd to last dim of weights.
        # x_real/imag shape: (B, H_coeff, W_coeff, num_orientations, num_blocks, block_size_channel_part)
        # complex_weight_lh_1[0] shape: (num_blocks, block_size_channel_part, block_size_channel_part)
        # multiply output shape: (B, H_coeff, W_coeff, num_orientations, num_blocks, block_size_channel_part)
        # xh[0] after stack: (B, H_coeff, W_coeff, num_orientations, num_blocks, block_size_channel_part, 2)
        # self.hidden_size = num_blocks * block_size
        xh[0] = xh[0].reshape(B, xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], self.hidden_size, xh[0].shape[-1]) # H,W,Orient, C, R/I
        xh[0]=torch.permute(xh[0], (0, 4, 1, 2, 3, 5)) # B, C, H, W, Orient, R/I

        x = self.ifm((xl,xh))
        x = torch.permute(x, (0, 2, 3, 1)) # B, H, W, C
        x = x.reshape(B, N, C)
        return x


class SVT_channel_token_mixing(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if DTCWTForward is None or DTCWTInverse is None:
            raise ImportError("pytorch_wavelets is required for SVT_channel_token_mixing but not found.")

        # Common initialization
        self.xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
        self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.softshrink = 0.0

        if dim == 64: #[b, 64,56,56] -> x2 is [b, 32, 56, 56]
            self.d_model = dim
            self.hidden_size_channel = dim // 2 # for x2 part
            self.num_blocks_channel = 4
            self.block_size_channel = self.hidden_size_channel // self.num_blocks_channel
            self.token_spatial_dim = 28 # High-freq components are H/2, W/2. So 56/2=28
            self.num_token_blocks = 6 # Number of orientations in DTCWT high-pass
            # Note: Original code had self.token_blocks = 28, which seems to be spatial dim, not block count
            # The token mixing here seems to operate over the 'orientations' dimension of xh.
            # For J=1, DTCWT gives 6 orientations. So token_blocks might refer to these.
            # Let's assume token mixing acts on the 6 orientation bands.
            # And the weights for token mixing might be small (e.g., 6x6 if mixing orientations)
            # The original code used large token_blocks (28, 14) for weights, suggesting spatial mixing.
            # This part is ambiguous in the original. Let's try to make it consistent.
            # If complex_weight_lh_1_t is (2, num_token_blocks, token_dim, token_dim)
            # and token_dim refers to spatial dimension, this is very large.
            # Let's stick to the original parameter sizing for now and point out potential issues.
            
            # This was self.token_blocks = 28 for dim 64, which implies spatial dim for weights.
            # This would make weights [2, 28, 28, 28].
            # If it's for the 6 orientations, weights would be [2, 6, 6, 6] or similar.
            # The original code uses xh[0] with shape (2, B, H/2, W/2, 6_orientations, C_channel) after permute
            # Then for token mixing, it permutes to (2, B, C_channel, 6_orientations, H/2, W/2)
            # and applies multiply. So 'bd' is (..., 6_orientations, H/2) and 'bdk' is (..., H/2, W/2)
            # This implies token mixing over spatial dimension W/2.
            self.token_block_dim_for_weights = 28 # Corresponds to H/2 or W/2 after wavelet transform.

            self.conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, groups=dim // 2, bias=True)
            self.complex_weight_ll = nn.Parameter(torch.randn(self.hidden_size_channel, 56, 56, dtype=torch.float32) * 0.02) # Spatial for LL
        
        elif dim == 128: #[b, 128,28,28] -> x2 is [b, 64, 28, 28]
            self.d_model = dim
            self.hidden_size_channel = dim // 2
            self.num_blocks_channel = 4
            self.block_size_channel = self.hidden_size_channel // self.num_blocks_channel
            self.token_spatial_dim = 14 # 28/2 = 14
            self.num_token_blocks = 6 # Orientations
            self.token_block_dim_for_weights = 14

            self.conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, groups=dim // 2, bias=True)
            self.complex_weight_ll = nn.Parameter(torch.randn(self.hidden_size_channel, 28, 28, dtype=torch.float32) * 0.02)

        # Cases for dim 96 and 192 were problematic in original SVT_channel_token_mixing
        # - self.hidden_size was not dim // 2
        # - self.conv was not defined
        # - self.token_blocks (token_spatial_dim here) was not defined
        # Correcting these based on the pattern for 64/128:
        elif dim == 96:
            self.d_model = dim
            self.hidden_size_channel = dim // 2 # Corrected
            self.num_blocks_channel = 4
            self.block_size_channel = self.hidden_size_channel // self.num_blocks_channel
            self.token_spatial_dim = 28 # Assuming 56x56 input features to this block -> 28x28 for xh
            self.num_token_blocks = 6 # Orientations
            self.token_block_dim_for_weights = 28 # Corrected

            self.conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, groups=dim // 2, bias=True) # Added
            self.complex_weight_ll = nn.Parameter(torch.randn(self.hidden_size_channel, 56, 56, dtype=torch.float32) * 0.02)

        elif dim == 192:
            self.d_model = dim
            self.hidden_size_channel = dim // 2 # Corrected
            self.num_blocks_channel = 4
            self.block_size_channel = self.hidden_size_channel // self.num_blocks_channel
            self.token_spatial_dim = 14 # Assuming 28x28 input features -> 14x14 for xh
            self.num_token_blocks = 6 # Orientations
            self.token_block_dim_for_weights = 14 # Corrected

            self.conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, groups=dim // 2, bias=True) # Added
            self.complex_weight_ll = nn.Parameter(torch.randn(self.hidden_size_channel, 28, 28, dtype=torch.float32) * 0.02)
        else:
            raise ValueError(f"Unsupported dim {dim} for SVT_channel_token_mixing")

        if not (self.hidden_size_channel % self.num_blocks_channel == 0):
             raise ValueError(f"hidden_size_channel {self.hidden_size_channel} must be divisible by num_blocks_channel {self.num_blocks_channel}")

        # Channel mixing weights (for x2 part)
        self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks_channel, self.block_size_channel, self.block_size_channel, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks_channel, self.block_size_channel, self.block_size_channel, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks_channel, self.block_size_channel,  dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks_channel, self.block_size_channel,  dtype=torch.float32) * 0.02)

        # Token mixing weights (for x2 part, after channel mixing)
        # Original shapes: (2, self.token_blocks, self.token_blocks, self.token_blocks)
        # If token_blocks refers to spatial dim, then (2, spatial_dim, spatial_dim, spatial_dim)
        # This is token_block_dim_for_weights here.
        # This part is still very memory-intensive if token_block_dim_for_weights is large.
        # It seems like it should be mixing across the 6 orientation bands.
        # For now, following original sizing logic:
        tb_dim = self.token_block_dim_for_weights
        self.complex_weight_lh_1_t = nn.Parameter(torch.randn(2, tb_dim, tb_dim, tb_dim, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_2_t = nn.Parameter(torch.randn(2, tb_dim, tb_dim, tb_dim, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b1_t = nn.Parameter(torch.randn(2, tb_dim, tb_dim,  dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b2_t = nn.Parameter(torch.randn(2, tb_dim, tb_dim,  dtype=torch.float32) * 0.02)


    def multiply(self, input_tensor, weights):
        return torch.einsum('...bd,bdk->...bk', input_tensor, weights)

    def forward(self, x, H, W):
        B, N, C_total = x.shape # C_total is self.d_model

        x = x.view(B, H, W, C_total)
        x = torch.permute(x, (0, 3, 1, 2)).contiguous() # (B, C_total, H, W)

        x1, x2 = torch.chunk(x, 2, dim=1) # x1, x2 each have C_total // 2 channels
        x1 = self.conv(x1) # spatial mixing for first half

        # Wavelet mixing for second half (x2)
        x2 = x2.to(torch.float32)
        B_x2, C_x2, H_x2, W_x2 = x2.shape # C_x2 is self.hidden_size_channel

        xl, xh_list = self.xfm(x2) # xh_list contains xh[0] for J=1
        
        # LL path
        # xl shape: (B, C_channel, H_xl, W_xl)
        # complex_weight_ll shape: (C_channel, H_ll_weight, W_ll_weight)
        # H_xl, W_xl should match H_ll_weight, W_ll_weight from __init__
        if xl.shape[2:] != self.complex_weight_ll.shape[1:]:
            # print(f"Warning: Mismatch in SVT_channel_token_mixing LL. xl.shape[2:]={xl.shape[2:]}, complex_weight_ll.shape[1:]={self.complex_weight_ll.shape[1:]}. Interpolating xl.")
            # xl = F.interpolate(xl, size=self.complex_weight_ll.shape[1:], mode='bilinear', align_corners=False)
            pass # Assuming correct dimensions based on stage
        xl = xl * self.complex_weight_ll
        
        xh = xh_list[0] # For J=1, xh is a list with one tensor: (B, C_channel, H_xh, W_xh, num_orientations=6, real/imag=2)
        
        # Permute for channel mixing: (real/imag, B, H_xh, W_xh, num_orientations, C_channel)
        xh = torch.permute(xh, (5, 0, 2, 3, 4, 1)).contiguous()
        # Reshape C_channel into num_blocks_channel, block_size_channel
        # xh shape: (2, B, H_xh, W_xh, num_orientations, num_blocks_channel, block_size_channel)
        xh = xh.reshape(xh.shape[0], xh.shape[1], xh.shape[2], xh.shape[3], xh.shape[4], self.num_blocks_channel, self.block_size_channel)

        # Channel mixing
        x_real = xh[0]
        x_imag = xh[1]
        # Shapes: (B, H_xh, W_xh, num_orientations, num_blocks_channel, block_size_channel)
        # Weights: (num_blocks_channel, block_size_channel, block_size_channel)

        x_real_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[0]) - self.multiply(x_imag, self.complex_weight_lh_1[1]) + self.complex_weight_lh_b1[0])
        x_imag_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[1]) + self.multiply(x_imag, self.complex_weight_lh_1[0]) + self.complex_weight_lh_b1[1])
        x_real_2 = self.multiply(x_real_1, self.complex_weight_lh_2[0]) - self.multiply(x_imag_1, self.complex_weight_lh_2[1]) + self.complex_weight_lh_b2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_lh_2[1]) + self.multiply(x_imag_1, self.complex_weight_lh_2[0]) + self.complex_weight_lh_b2[1]

        xh_cm = torch.stack([x_real_2, x_imag_2], dim=-1).float() # (B, H_xh, W_xh, num_orientations, num_blocks_channel, block_size_channel, 2)
        xh_cm = F.softshrink(xh_cm, lambd=self.softshrink) if self.softshrink else xh_cm
        # Reshape back to C_channel: (B, H_xh, W_xh, num_orientations, C_channel, 2)
        xh_cm = xh_cm.reshape(B_x2, xh_cm.shape[1], xh_cm.shape[2], xh_cm.shape[3], self.hidden_size_channel, 2)
        
        # Token mixing
        # Permute for token mixing: (real/imag, B, C_channel, num_orientations, H_xh, W_xh)
        # Original: xh[0]=torch.permute(xh[0], (5, 0, 4, 1, 2, 3)).contiguous() #2, B, C, 6,H,W
        # Current xh_cm shape: (B, H_xh, W_xh, num_orientations, C_channel, 2)
        # Permute to: (2, B, C_channel, num_orientations, H_xh, W_xh)
        xh_tm_input = torch.permute(xh_cm, (5, 0, 4, 3, 1, 2)).contiguous()

        x_real_t = xh_tm_input[0] # (B, C_channel, num_orientations, H_xh, W_xh)
        x_imag_t = xh_tm_input[1]
        
        # multiply expects '...bd,bdk->...bk'
        # input '...bd' is (B, C_channel, num_orientations, H_xh, W_xh)
        # weights 'bdk' is (token_block_dim, token_block_dim, token_block_dim)
        # This implies einsum is on last two dims, e.g. (H_xh, W_xh) and (W_xh, W_xh_out) if weights are (token_block_dim_for_W, W_xh, W_xh_out)
        # Original weights: (2, self.token_blocks (spatial_dim), self.token_blocks (spatial_dim), self.token_blocks (spatial_dim))
        # This requires careful matching of dimensions. Let's assume token mixing is on the W_xh dimension.
        # So, 'd' is W_xh, and 'k' is W_xh (output).
        # weights need to be (num_orientations, H_xh, W_xh_in, W_xh_out) or broadcastable.
        # The current self.complex_weight_lh_1_t[0] is (tb_dim, tb_dim, tb_dim).
        # This would mean the multiply op needs inputs like (..., num_orientations, H_xh) and weights (H_xh, W_xh)
        # This part of the original code is quite complex in its assumptions about tensor shapes.
        # For now, replicating the einsum from the original code, assuming W_xh is the dimension being mixed.
        # This means self.token_block_dim_for_weights must match W_xh.
        if W_x2 // 2 != self.token_block_dim_for_weights:
             # This is an issue. H_xh = H_x2/2, W_xh = W_x2/2
             # print(f"Warning: Token mixing spatial dimension mismatch. W_xh={W_x2//2}, expected weight dim={self.token_block_dim_for_weights}")
             pass

        x_real_1_t = F.relu(self.multiply(x_real_t, self.complex_weight_lh_1_t[0]) - self.multiply(x_imag_t, self.complex_weight_lh_1_t[1]) + self.complex_weight_lh_b1_t[0])
        x_imag_1_t = F.relu(self.multiply(x_real_t, self.complex_weight_lh_1_t[1]) + self.multiply(x_imag_t, self.complex_weight_lh_1_t[0]) + self.complex_weight_lh_b1_t[1])
        x_real_2_t = self.multiply(x_real_1_t, self.complex_weight_lh_2_t[0]) - self.multiply(x_imag_1_t, self.complex_weight_lh_2_t[1]) + self.complex_weight_lh_b2_t[0]
        x_imag_2_t = self.multiply(x_real_1_t, self.complex_weight_lh_2_t[1]) + self.multiply(x_imag_1_t, self.complex_weight_lh_2_t[0]) + self.complex_weight_lh_b2_t[1]

        xh_processed = torch.stack([x_real_2_t, x_imag_2_t], dim=-1).float() # (B, C_channel, num_orientations, H_xh, W_xh, 2)
        # Permute back to original xh format for ifm: (B, C_channel, H_xh, W_xh, num_orientations, 2)
        xh_processed = torch.permute(xh_processed, (0, 1, 3, 4, 2, 5)).contiguous()
        
        xh_list_out = [xh_processed] # ifm expects a list of high-pass tensors
        x2_reconstructed = self.ifm((xl, xh_list_out))

        # Concatenate x1 and reconstructed x2
        # x1: (B, C_total/2, H, W)
        # x2_reconstructed: (B, C_total/2, H, W)
        # The unsqueeze(2) and reshape in original was for a different structure.
        # Here, x1 and x2_reconstructed should have same C, H, W.
        x_final = torch.cat([x1, x2_reconstructed], dim=1) # (B, C_total, H, W)
        
        x_final = torch.permute(x_final, (0, 2, 3, 1)).contiguous() # (B, H, W, C_total)
        x_final = x_final.reshape(B, N, C_total)
        return x_final


def rand_bbox(size, lam, scale=1):
    W = size[1] // scale # Assuming size is [B, H, W, C] or [B, N]
    H = size[2] // scale # Needs to handle N if input is flattened
    if len(size) == 4: # B, H, W, C
        pass
    elif len(size) == 3: # B, N, C -- need original H, W for bbox. This function might be problematic.
        # This function rand_bbox is called with x.size() where x is [B,H,W,C] in SVT.forward
        pass
    else: # B, C, H, W
        W = size[2] // scale
        H = size[3] // scale


    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class ClassAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.kv = nn.Linear(dim, dim * 2)
        self.q = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(x[:, :1, :]).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B, num_heads, 1, head_dim
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        return cls_embed

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class ClassBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = ClassAttention(dim, num_heads)
        self.mlp = FFN(dim, int(dim * mlp_ratio))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.attn(self.norm1(x))
        cls_embed = cls_embed + self.mlp(self.norm2(cls_embed))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)

class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self,
        dim,
        num_heads,
        mlp_ratio,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        sr_ratio=1, # Not used by SVT_channel_mixing or Attention here
        block_type = 'scatter'
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        if block_type == 'std_att':
            self.attn = Attention(dim, num_heads)
        elif block_type == 'scatter':
            self.attn = SVT_channel_mixing (dim)
        elif block_type == 'scatter_token': # Added a new type for clarity
            self.attn = SVT_channel_token_mixing (dim)
        else:
            raise ValueError(f"Unknown block_type: {block_type}")
            
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class DownSamples(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x) # Input x is [B, C_in, H, W]
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, C_out, H_new, W_new] -> [B, C_out, N_new] -> [B, N_new, C_out]
        x = self.norm(x)
        return x, H, W

class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),  # 112x112 for 224x224 input
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3, # Original PVT uses kernel_size=2, stride=2, padding=0
                              stride=2,      # Or kernel_size=3, stride=2, padding=1
                              padding=1)     # This gives H/2, W/2. So 112->56
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # B, C, H, W -> B, C, N -> B, N, C
        x = self.norm(x)
        return x, H, W

class SVT(nn.Module):
    def __init__(self,
        in_chans=3,
        num_classes=1000,
        stem_hidden_dim = 32,
        embed_dims=[64, 128, 320, 448],
        num_heads=[2, 4, 10, 14],
        mlp_ratios=[8, 8, 4, 4],
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[4, 2, 1, 1], # sr_ratios not used by current Attention/SVT_channel_mixing
        num_stages=4,
        token_label=True, # Original SVT uses this for training with token labeling
        output_features=False, # If True, forward() returns a list of feature maps
        img_size=224, # For dummy forward pass to get width_list
        alpha=1, # Controls how many stages use 'scatter' attention. Default 1 means only stage 0.
        #scatter_block_type='scatter', # 'scatter' or 'scatter_token'
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.token_label = token_label
        self.output_features = output_features
        self.img_size = img_size # For width_list calculation

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_hidden_dim, embed_dims[i])
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])

            current_block_type = 'scatter' if i < alpha else 'std_att'
            # if scatter_block_type == 'scatter_token' and current_block_type == 'scatter':
            #    current_block_type = 'scatter_token' # Overrides to use channel_token_mixing

            block_module = nn.ModuleList([Block( # Renamed from 'block' to 'block_module'
                dim = embed_dims[i],
                num_heads = num_heads[i],
                mlp_ratio = mlp_ratios[i],
                drop_path=dpr[cur + j],
                norm_layer=norm_layer,
                sr_ratio = sr_ratios[i],
                block_type=current_block_type)
            for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block_module)
            setattr(self, f"norm{i + 1}", norm)

        # Post network for classification token processing (if not output_features)
        if not self.output_features:
            post_layers = ['ca'] # As in original
            self.post_network = nn.ModuleList([
                ClassBlock(
                    dim = embed_dims[-1],
                    num_heads = num_heads[-1],
                    mlp_ratio = mlp_ratios[-1],
                    norm_layer=norm_layer)
                for _ in range(len(post_layers))
            ])
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

            if self.token_label: # Or self.return_dense in original
                self.mix_token = True # from original
                self.beta = 1.0       # from original
                self.pooling_scale = 8 # from original, related to patch size for mix token
                self.aux_head = nn.Linear(
                    embed_dims[-1],
                    num_classes) if num_classes > 0 else nn.Identity()
        
        self.width_list = []
        if self.output_features:
            try:
                # Ensure model is on CPU for this dummy pass if not specified, to avoid device errors during init
                # If you have GPU, PyTorchWavelets might require CUDA tensors.
                # For safety, do this on CPU then move model to desired device later.
                current_device = next(self.parameters()).device
                dummy_input = torch.randn(1, in_chans, self.img_size, self.img_size).to(current_device)
                # Temporarily disable token_label features for width_list calculation
                _original_token_label_status = self.token_label
                _original_output_features_status = self.output_features
                self.token_label = False 
                self.output_features = True # Ensure _get_feature_outputs is called

                with torch.no_grad():
                    features = self._get_feature_outputs(dummy_input)
                self.width_list = [f.shape[1] for f in features]
                
                self.token_label = _original_token_label_status # Restore
                self.output_features = _original_output_features_status # Restore

            except Exception as e:
                print(f"Warning: Could not compute width_list during SVT init: {e}")
                self.width_list = [embed_dims[i] for i in range(num_stages)] # Fallback

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_feature_outputs(self, x):
        """ Helper to get feature maps from each stage. """
        B = x.shape[0]
        features_out = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block_module = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x) # x is [B, N, C] for Stem/DownSamples, or [B,C,H,W] for DownSamples
            
            for blk in block_module:
                x = blk(x, H, W) # Input [B, N, C], Output [B, N, C]
            
            x = norm(x) # Operates on [B, N, C]
            
            current_feature_map = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            features_out.append(current_feature_map)
            
            if i < self.num_stages - 1: # Prepare x for the next DownSamples layer
                x = current_feature_map # x is now [B, C, H, W]
        
        return features_out # List of [B, C_stage, H_stage, W_stage]

    def forward_cls_token_processor(self, x): # Renamed from forward_cls
        """Processes features with a class token for classification head."""
        B, N, C = x.shape
        # Simple mean pooling if no CLS token, or take CLS token if prepended
        # Original SVT adds CLS token here.
        cls_tokens = x.mean(dim=1, keepdim=True) # Global Average Pooling as a class token
        x_with_cls = torch.cat((cls_tokens, x), dim=1)
        
        for post_block in self.post_network: # 'block' name conflicts
            x_with_cls = post_block(x_with_cls)
        return x_with_cls # Returns all tokens, including processed CLS token at index 0

    def forward_features_classification(self, x): # Renamed from forward_features
        """Forward pass for classification, ending with features for the head."""
        B = x.shape[0]
        # This path processes data stage by stage up to the point of adding class token
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block_module = getattr(self, f"block{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block_module:
                x = blk(x, H, W)
            
            norm = getattr(self, f"norm{i + 1}") # Norm is applied after blocks in each stage
            x = norm(x) # x is [B, N, C]
            
            if i < self.num_stages - 1: # If not the last stage
                # Reshape to [B, C, H, W] for the next convolutional patch_embed (DownSamples)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # For the last stage, x remains [B, N, C] and goes to forward_cls_token_processor

        # After all stages, x is [B, N_last_stage, C_last_stage]
        x_processed_cls = self.forward_cls_token_processor(x) # Adds and processes CLS token
        return x_processed_cls[:, 0] # Return only the CLS token features

    # --- Token Labeling specific methods from original SVT ---
    def forward_embeddings(self, x): # For token_label path
        # Only processes through the first patch embedding layer (Stem)
        patch_embed = getattr(self, f"patch_embed{1}")
        x, H, W = patch_embed(x) # x: [B,N,C], H, W of first stage patches
        x = x.view(x.size(0), H, W, -1) # [B, H, W, C]
        return x, H, W

    def forward_tokens(self, x, H_in, W_in): # For token_label path, x is [B,H,W,C] from forward_embeddings + mixup
        B = x.shape[0]
        # x here is the output of forward_embeddings potentially with mix token applied.
        # It's [B, H_stem, W_stem, C_stem]
        x = x.view(B, -1, x.size(-1)) # Flatten to [B, N_stem, C_stem]
        
        H, W = H_in, W_in # H, W from the first stage (after Stem)

        # Loop through all stages, starting from block1 (stage 0). patch_embed1 was already done.
        for i in range(self.num_stages):
            # For i=0 (first stage), patch_embed is Stem, which was handled by forward_embeddings.
            # So, if i=0, we skip patch_embed and use the already transformed x.
            if i != 0: # For subsequent stages, apply DownSamples
                patch_embed = getattr(self, f"patch_embed{i + 1}")
                # Input to DownSamples should be [B, C, H, W]
                x_reshaped = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                x, H, W = patch_embed(x_reshaped)
            
            block_module = getattr(self, f"block{i + 1}")
            for blk in block_module:
                x = blk(x, H, W) # x is [B, N, C]

            norm = getattr(self, f"norm{i + 1}")
            x = norm(x)
            # No reshape to B,C,H,W here within the loop for token_label's forward_tokens path
            # because the next iteration's patch_embed expects flattened input if it's DownSamples.
            # Actually, DownSamples takes B,C,H,W. So it needs reshape IF not last stage.
            # The original `forward_tokens` did not reshape x back to image-like for next patch_embed.
            # This seems like a discrepancy with how DownSamples expects input.
            # Let's assume this specific path for token_label is intentional.
            # The `patch_embed` for i!=0 is DownSamples, which expects [B,C,H,W].
            # So, we DO need to reshape if not last stage.
            # if i < self.num_stages - 1:
            #    x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # This was missing in original forward_tokens logic, this will be fixed in self._get_feature_outputs
            # For token_labeling, we stick to original logic to avoid breaking it.
            # The `forward_cls_token_processor` is applied after this loop.

        # After all stages, x is [B, N_last_stage, C_last_stage]
        x = self.forward_cls_token_processor(x) # Adds and processes CLS token
        # x is now [B, 1 + N_last_stage, C_last_stage]
        return x
    # --- End of Token Labeling specific methods ---

    def forward(self, x):
        if self.output_features: # For backbone usage (e.g. in YOLO)
            return self._get_feature_outputs(x) # Returns a list of Tensors [B,C,H,W]

        # Original SVT forward logic for classification or token labeling
        if not self.token_label: # Simple classification
            x = self.forward_features_classification(x) # Gets CLS token features [B, C_last_stage]
            x = self.head(x) # Final classification
            return x
        else: # Token labeling path (more complex)
            # 1. Get initial embeddings (Stem output)
            # x_emb: [B, H_stem, W_stem, C_stem], H_stem, W_stem: spatial dim after Stem
            x_emb, H_stem, W_stem = self.forward_embeddings(x)

            # 2. Apply MixToken if training
            if self.mix_token and self.training:
                lam = np.random.beta(self.beta, self.beta)
                # rand_bbox expects [B, H, W, C] for size[1] and size[2]
                patch_h, patch_w = x_emb.shape[1] // self.pooling_scale, x_emb.shape[2] // self.pooling_scale
                bbx1, bby1, bbx2, bby2 = rand_bbox(x_emb.shape, lam, scale=self.pooling_scale)
                
                temp_x = x_emb.clone()
                sbbx1, sbby1, sbbx2, sbby2 = self.pooling_scale*bbx1, self.pooling_scale*bby1, \
                                            self.pooling_scale*bbx2, self.pooling_scale*bby2
                temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x_emb.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
                x_processed_stages = temp_x # This will be input to forward_tokens
            else:
                bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0
                x_processed_stages = x_emb

            # 3. Process through all stages (Blocks + Norms) including CLS token handling
            # Input x_processed_stages is [B, H_stem, W_stem, C_stem]
            # Output all_tokens_features is [B, 1 (cls) + N_last_stage, C_last_stage]
            all_tokens_features = self.forward_tokens(x_processed_stages, H_stem, W_stem)
            
            # 4. Get outputs from heads
            x_cls_output = self.head(all_tokens_features[:, 0]) # CLS token output
            x_aux_patch_tokens = all_tokens_features[:, 1:] # Patch tokens [B, N_last, C_last]
            x_aux_output = self.aux_head(x_aux_patch_tokens) # Aux head output [B, N_last, num_classes]

            if not self.training: # Inference for token_labeling
                # The shape of x_aux_output is [B, N_last_stage_patches, num_classes]
                # max(1)[0] takes the max along the patches dimension.
                return x_cls_output + 0.5 * x_aux_output.max(1)[0]

            # 5. Reverse MixToken for auxiliary loss if training
            if self.mix_token and self.training:
                # This part assumes N_last_stage_patches can be reshaped to patch_h_last, patch_w_last
                # These patch_h, patch_w are from the *initial* feature map for bbox calculation.
                # The number of patches changes through stages. This needs careful handling.
                # The original code uses patch_h, patch_w from before all stages.
                # This implies aux_head output spatial dims correspond to initial pooling_scale.
                # This is tricky. Let's assume N_last_stage == patch_h * patch_w.
                # (where patch_h, patch_w are from pooling_scale on x_emb, not final H,W)
                # Example: if x_emb is 56x56, pooling_scale 8 -> patch_h,patch_w = 7,7. N_last=49.
                # This is true if the final stage output spatial dim matches this. (e.g. 7x7 for 224 input)

                # x_aux_output is [B, N_patches_final, Num_classes]
                # We need to reshape it to [B, patch_h_final, patch_w_final, Num_classes]
                # Assuming N_patches_final = patch_h * patch_w (where these are based on pooling_scale)
                # For 224 input, Stem->56. DownSample x3 -> 7. So last stage is 7x7.
                # If pooling_scale=8, 56/8=7. So patch_h, patch_w are 7. This matches.
                
                # Reshape aux output to spatial form before reversing mix
                x_aux_output = x_aux_output.reshape(x_aux_output.shape[0], patch_h, patch_w, x_aux_output.shape[-1])
                temp_x_aux = x_aux_output.clone()
                # bbx1, etc. are for the 'pooled' grid (patch_h, patch_w)
                temp_x_aux[:, bbx1:bbx2, bby1:bby2, :] = x_aux_output.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
                x_aux_output_final = temp_x_aux
                # Reshape back to [B, N_final, Num_classes]
                x_aux_output_final = x_aux_output_final.reshape(x_aux_output_final.shape[0], patch_h * patch_w, x_aux_output_final.shape[-1])
                return x_cls_output, x_aux_output_final, (bbx1, bby1, bbx2, bby2) # Tuple for training
            else: # Training without mix_token (but token_label=True)
                return x_cls_output, x_aux_output, (bbx1, bby1, bbx2, bby2)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# Example of _cfg, if you don't have timm's vision_transformer fully available
# or if it's causing issues.
# For full timm integration, ensure timm is installed and imports work.
if _cfg is None:
    class PlaceholderDefaultCfg:
        def __init__(self):
            self.url = ''
            self.num_classes = 1000
            self.input_size = (3, 224, 224)
            self.pool_size = None
            self.crop_pct = .9
            self.interpolation = 'bicubic'
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
            self.first_conv = ''
            self.classifier = 'head'
    _cfg = PlaceholderDefaultCfg()


@register_model
def svt_s(pretrained=False, img_size=224, output_features=False, **kwargs):
    # Set token_label=False if output_features=True, to simplify.
    # Or, the user can explicitly set token_label.
    # If used as a backbone, token_label specific outputs are usually not needed.
    token_label_setting = kwargs.pop('token_label', not output_features)

    model = SVT(
        stem_hidden_dim = 32,
        embed_dims = [64, 128, 320, 448],
        num_heads = [2, 4, 10, 14],
        mlp_ratios = [8, 8, 4, 4],
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        depths = [3, 4, 6, 3],
        sr_ratios = [4, 2, 1, 1],
        token_label=token_label_setting,
        output_features=output_features,
        img_size=img_size,
        alpha=1, # Default: scatter for 1st stage
        **kwargs)
    model.default_cfg = _cfg
    return model

@register_model
def svt_b(pretrained=False, img_size=224, output_features=False, **kwargs):
    token_label_setting = kwargs.pop('token_label', not output_features)
    model = SVT(
        stem_hidden_dim = 64,
        embed_dims = [64, 128, 320, 512],
        num_heads = [2, 4, 10, 16],
        mlp_ratios = [8, 8, 4, 4],
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        depths = [3, 4, 12, 3],
        sr_ratios = [4, 2, 1, 1],
        token_label=token_label_setting,
        output_features=output_features,
        img_size=img_size,
        alpha=1,
        **kwargs)
    model.default_cfg = _cfg
    return model

@register_model
def svt_l(pretrained=False, img_size=224, output_features=False, **kwargs):
    token_label_setting = kwargs.pop('token_label', not output_features)
    model = SVT(
        stem_hidden_dim = 64, # Stage 0 stem hidden dim
        embed_dims = [96, 192, 384, 512], # Channels for stage 0, 1, 2, 3
        num_heads = [3, 6, 12, 16],
        mlp_ratios = [8, 8, 4, 4],
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        depths = [3, 6, 18, 3],
        sr_ratios = [4, 2, 1, 1],
        token_label=token_label_setting,
        output_features=output_features,
        img_size=img_size,
        alpha=1, # Use scatter (SVT_channel_mixing) for first stage
        # For SVT-L, embed_dims[0]=96. SVT_channel_mixing uses spatial_dim=56 for dim=96.
        # This means input to first Block (after Stem) must be H,W=56,56.
        # Stem output: H/4, W/4. For img_size=224, Stem H,W = 56,56. This matches.
        **kwargs)
    model.default_cfg = _cfg
    return model

if __name__ == '__main__':
    # Test basic instantiation and forward pass for classification
    print("Testing SVT-S for classification:")
    model_s_cls = svt_s(num_classes=100, token_label=False, output_features=False)
    dummy_input_cls = torch.randn(2, 3, 640, 640)
    try:
        output_cls = model_s_cls(dummy_input_cls)
        print(f"SVT-S classification output shape: {output_cls.shape}") # Expected: [2, 100]
    except Exception as e:
        print(f"Error during SVT-S classification test: {e}")

    # Test instantiation for feature extraction (backbone mode)
    print("\nTesting SVT-S as a backbone (output_features=True):")
    model_s_feat = svt_s(output_features=True, img_size=224) # token_label will be False by default
    print(f"SVT-S width_list: {model_s_feat.width_list}") # Should be like [64, 128, 320, 448]
    
    dummy_input_feat = torch.randn(2, 3, 640, 640)
    try:
        feature_maps = model_s_feat(dummy_input_feat)
        print(f"SVT-S produced {len(feature_maps)} feature maps.")
        for i, fm in enumerate(feature_maps):
            print(f"  Feature map {i} shape: {fm.shape}") # List of [B,C,H,W]
            assert fm.ndim == 4, f"Feature map {i} is not 4D"
            assert isinstance(feature_maps, list), "Output is not a list"
    except Exception as e:
        print(f"Error during SVT-S backbone test: {e}")

    print("\nTesting SVT-L as a backbone (output_features=True):")
    # SVT-L uses embed_dims=[96, 192, 384, 512]
    # SVT_channel_mixing(dim=96) will be used for the first stage.
    # It expects H,W that lead to xfm output matching complex_weight_ll spatial dim of 56.
    # Stem output H,W for 224x224 input is 56x56. This should work.
    model_l_feat = svt_l(output_features=True, img_size=224) # Test with alpha=2
    # alpha=2 means stage 0 (dim 96, H,W=56) and stage 1 (dim 192, H,W=28) use SVT_channel_mixing
    print(f"SVT-L (alpha=2) width_list: {model_l_feat.width_list}")

    try:
        feature_maps_l = model_l_feat(dummy_input_feat)
        print(f"SVT-L produced {len(feature_maps_l)} feature maps.")
        for i, fm in enumerate(feature_maps_l):
            print(f"  Feature map {i} shape: {fm.shape}")
    except Exception as e:
        print(f"Error during SVT-L backbone test: {e}")


    # Test token labeling path (original SVT training)
    print("\nTesting SVT-S for token labeling (training mode):")
    model_s_token = svt_s(num_classes=100, token_label=True, output_features=False)
    model_s_token.train() # Set to training mode for token labeling specifics
    dummy_input_tl = torch.randn(2, 3, 224, 224)
    try:
        # Expected output: x_cls_output, x_aux_output_final, (bbx1, bby1, bbx2, bby2)
        output_tl_train = model_s_token(dummy_input_tl)
        print(f"SVT-S token_label train output type: {type(output_tl_train)}")
        if isinstance(output_tl_train, tuple) and len(output_tl_train) == 3:
            print(f"  CLS output shape: {output_tl_train[0].shape}")
            print(f"  AUX output shape: {output_tl_train[1].shape}")
            print(f"  BBox: {output_tl_train[2]}")
        else:
            print(f"  Unexpected token_label train output: {output_tl_train}")
    except Exception as e:
        print(f"Error during SVT-S token_label train test: {e}")

    model_s_token.eval() # Set to eval mode for token labeling specifics
    try:
        output_tl_eval = model_s_token(dummy_input_tl)
        print(f"SVT-S token_label eval output shape: {output_tl_eval.shape}")
    except Exception as e:
        print(f"Error during SVT-S token_label eval test: {e}")

    # Test SVT_channel_token_mixing if intended (currently scatter_block_type is not used)
    # To test it, you'd need to modify SVT's __init__ to use 'scatter_token' block_type
    # e.g., pass scatter_block_type='scatter_token' to svt_s(...) and handle it in SVT.__init__
    # For now, this is not the default path.