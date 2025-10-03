# -*- coding: utf-8 -*-
from typing import Tuple, Optional, List, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import timm
import math # 為了 try-except 中的日誌添加

__all__: List[str] = [
    "SwinTransformerStage",
    "SwinTransformerBlock",
    "DeformableSwinTransformerBlock",
    "SwinTransformerV2", # 添加主類到 __all__
    "swin_transformer_v2_t",
    "swin_transformer_v2_s",
    "swin_transformer_v2_b",
    "swin_transformer_v2_l",
    "swin_transformer_v2_h",
    "swin_transformer_v2_g"
]


class FeedForward(nn.Sequential):
    """
    Feed forward module used in the transformer encoder.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 dropout: float = 0.) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input features
        :param hidden_features: (int) Number of hidden features
        :param out_features: (int) Number of output features
        :param dropout: (float) Dropout factor
        """
        # Call super constructor and init modules
        super().__init__(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_features, out_features=out_features),
            nn.Dropout(p=dropout)
        )


def bchw_to_bhwc(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, height, width, channels]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, channels, height, width]
    :return: (torch.Tensor) Output tensor of the shape [batch size, height, width, channels]
    """
    return input.permute(0, 2, 3, 1)


def bhwc_to_bchw(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, channels, height, width]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, channels, height, width]
    """
    return input.permute(0, 3, 1, 2)


def unfold(input: torch.Tensor,
           window_size: int) -> torch.Tensor:
    """
    Unfolds (non-overlapping) a given feature map by the given window size (stride = window size)
    :param input: (torch.Tensor) Input feature map of the shape [batch size, channels, height, width]
    :param window_size: (int) Window size to be applied
    :return: (torch.Tensor) Unfolded tensor of the shape [batch size * windows, channels, window size, window size]
    """
    # Get original shape
    _, channels, height, width = input.shape  # type: int, int, int, int

    # Handle cases where image dimension is smaller than window_size
    # This might happen if the input resolution is dynamically changed or very small.
    # We'll pad if necessary to make unfolding work, although this might not be
    # the ideal solution for all use cases. Alternatively, one could skip unfolding.
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        input = F.pad(input, (0, pad_w, 0, pad_h))
        _, _, height, width = input.shape # Update shape after padding

    # Unfold input
    output: torch.Tensor = input.unfold(dimension=3, size=window_size, step=window_size) \
        .unfold(dimension=2, size=window_size, step=window_size)
    # Reshape to [batch size * windows, channels, window size, window size]
    # The number of windows needs to account for the potentially padded dimensions
    num_windows_h = height // window_size
    num_windows_w = width // window_size
    output: torch.Tensor = output.permute(0, 2, 3, 1, 5, 4).reshape(-1, channels, window_size, window_size)

    # Ensure the output shape matches the expectation based on calculated windows
    # expected_num_windows = input.shape[0] * num_windows_h * num_windows_w # Using original batch size
    # assert output.shape[0] == expected_num_windows, f"Shape mismatch: expected {expected_num_windows} windows, got {output.shape[0]}"

    return output


def fold(input: torch.Tensor,
         window_size: int,
         height: int,
         width: int) -> torch.Tensor:
    """
    Fold a tensor of windows again to a 4D feature map
    :param input: (torch.Tensor) Input tensor of windows [batch size * windows, channels, window size, window size]
    :param window_size: (int) Window size to be reversed
    :param height: (int) Height of the *original* feature map (before potential padding in unfold)
    :param width: (int) Width of the *original* feature map (before potential padding in unfold)
    :return: (torch.Tensor) Folded output tensor of the shape [batch size, channels, height, width]
    """
    # Calculate padded height/width based on original dimensions and window size
    padded_height = math.ceil(height / window_size) * window_size
    padded_width = math.ceil(width / window_size) * window_size
    num_windows_h = padded_height // window_size
    num_windows_w = padded_width // window_size

    # Get channels of windows
    channels: int = input.shape[1]
    # Get original batch size
    # Make sure the division is safe, handle potential zero dimensions gracefully
    if num_windows_h == 0 or num_windows_w == 0:
        # This case implies height or width was 0, which is unusual.
        # Return an appropriately shaped tensor, perhaps empty or matching input batch size.
        # For simplicity, let's assume valid dimensions for now, but a robust implementation would check.
        # If input.shape[0] is 0, this calculation will also fail.
        if input.shape[0] == 0:
             batch_size = 0 # Or handle as error
        else:
             batch_size = int(input.shape[0] // (num_windows_h * num_windows_w))
    else:
        batch_size: int = int(input.shape[0] // (num_windows_h * num_windows_w))

    # Check if batch_size calculation is valid
    if batch_size * num_windows_h * num_windows_w != input.shape[0] and input.shape[0] != 0:
        # This could happen if the input tensor shape doesn't match the expected number of windows
        raise ValueError(f"Cannot infer batch size. Input windows {input.shape[0]}, expected windows per batch item {num_windows_h * num_windows_w}")

    # Reshape input to [batch_size, num_windows_h, num_windows_w, channels, window_size, window_size]
    output: torch.Tensor = input.view(batch_size, num_windows_h, num_windows_w, channels,
                                      window_size, window_size)
    # Permute and reshape to [batch_size, channels, padded_height, padded_width]
    output: torch.Tensor = output.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, padded_height, padded_width)

    # Crop back to the original height and width if padding was added
    if padded_height != height or padded_width != width:
        output = output[:, :, :height, :width]

    return output

class WindowMultiHeadAttention(nn.Module):
    """
    This class implements window-based Multi-Head-Attention.
    NOTE: Swin v2 uses scaled cosine attention instead of dot-product attention
          and utilizes log-spaced continuous relative position bias.
    """

    def __init__(self,
                 in_features: int,
                 window_size: int,
                 number_of_heads: int,
                 dropout_attention: float = 0.,
                 dropout_projection: float = 0.,
                 meta_network_hidden_features: int = 256, # As suggested in v2 paper section 3.3
                 sequential_self_attention: bool = False) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input features
        :param window_size: (int) Window size
        :param number_of_heads: (int) Number of attention heads
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_projection: (float) Dropout rate after projection
        :param meta_network_hidden_features: (int) Number of hidden features in the two layer MLP meta network for CPB
        :param sequential_self_attention: (bool) If true sequential self-attention is performed (memory saving)
        """
        # Call super constructor
        super(WindowMultiHeadAttention, self).__init__()
        # Check parameter
        assert (in_features % number_of_heads) == 0, \
            "The number of input features (in_features) are not divisible by the number of heads (number_of_heads)."
        # Save parameters
        self.in_features: int = in_features
        self.window_size: int = window_size
        self.number_of_heads: int = number_of_heads
        self.sequential_self_attention: bool = sequential_self_attention
        head_dim = in_features // number_of_heads # Define head_dim

        # Init query, key and value mapping as a single layer
        self.mapping_qkv: nn.Module = nn.Linear(in_features=in_features, out_features=in_features * 3, bias=True)
        # Init attention dropout
        self.attention_dropout: nn.Module = nn.Dropout(dropout_attention)
        # Init projection mapping
        self.projection: nn.Module = nn.Linear(in_features=in_features, out_features=in_features, bias=True)
        # Init projection dropout
        self.projection_dropout: nn.Module = nn.Dropout(dropout_projection)

        # --- Continuous Relative Position Bias (CPB) Meta Network ---
        self.meta_network: nn.Module = nn.Sequential(
            nn.Linear(in_features=2, out_features=meta_network_hidden_features, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=meta_network_hidden_features, out_features=number_of_heads, bias=True))

        # --- learnable scalar tau for scaled cosine attention ---
        # Initialized with log(10) as in the v2 paper appendix A.2
        # self.register_parameter("tau", torch.nn.Parameter(torch.ones(1, number_of_heads, 1, 1) * math.log(10.0)))
        # Simpler initialization for demonstration:
        self.register_parameter("tau", torch.nn.Parameter(torch.ones(number_of_heads)))
        # Initialize pair-wise relative positions (log-spaced coordinates)
        self.__make_pair_wise_relative_positions()

    def __make_pair_wise_relative_positions(self) -> None:
        """
        Method initializes the pair-wise relative positions to compute the positional biases.
        Uses log-spaced coordinates as per Swin v2.
        """
        indexes: torch.Tensor = torch.arange(self.window_size) # Device placement happens implicitly later or could be explicit
        coordinates: torch.Tensor = torch.stack(torch.meshgrid([indexes, indexes], indexing='ij'), dim=0) # Use 'ij' indexing
        coordinates: torch.Tensor = torch.flatten(coordinates, start_dim=1) # Shape: [2, Wh*Ww]
        # Relative coordinates: [2, Wh*Ww, Wh*Ww]
        relative_coordinates: torch.Tensor = coordinates[:, :, None] - coordinates[:, None, :]
        # Permute and reshape: [Wh*Ww * Wh*Ww, 2]
        relative_coordinates: torch.Tensor = relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()

        # Log-spaced coordinates (using sign-preserving log)
        relative_coordinates_log: torch.Tensor = torch.sign(relative_coordinates) \
                                                 * torch.log1p(relative_coordinates.abs()) # Use log1p for stability

        self.register_buffer("relative_coordinates_log", relative_coordinates_log, persistent=False) # Register as buffer

    def update_resolution(self,
                          new_window_size: int,
                          **kwargs: Any) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param kwargs: (Any) Unused
        """
        # Set new window size
        self.window_size: int = new_window_size
        # Make new pair-wise relative positions
        self.__make_pair_wise_relative_positions()

    def __get_relative_positional_encodings(self) -> torch.Tensor:
        """
        Method computes the relative positional encodings using the meta network.
        :return: (torch.Tensor) Relative positional encodings [num_heads, window_size**2, window_size**2]
        """
        # Ensure coordinates are on the same device as the meta network parameters
        relative_coords = self.relative_coordinates_log.to(self.meta_network[0].weight.device)
        relative_position_bias: torch.Tensor = self.meta_network(relative_coords) # Shape: [N*N, num_heads]
        relative_position_bias: torch.Tensor = relative_position_bias.permute(1, 0) # Shape: [num_heads, N*N]
        N = self.window_size * self.window_size
        relative_position_bias: torch.Tensor = relative_position_bias.reshape(self.number_of_heads, N, N)
        return relative_position_bias # Shape: [num_heads, N, N]


    def __self_attention(self,
                         query: torch.Tensor, # Shape: [B*num_windows, num_heads, N, head_dim]
                         key: torch.Tensor,   # Shape: [B*num_windows, num_heads, N, head_dim]
                         value: torch.Tensor, # Shape: [B*num_windows, num_heads, N, head_dim]
                         batch_size_windows: int,
                         tokens: int, # N = window_size * window_size
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function performs standard (non-sequential) scaled cosine self-attention (Swin v2).
        :param query: (torch.Tensor) Query tensor of the shape [batch size * windows, heads, tokens, head_dim]
        :param key: (torch.Tensor) Key tensor of the shape [batch size * windows, heads, tokens, head_dim]
        :param value: (torch.Tensor) Value tensor of the shape [batch size * windows, heads, tokens, head_dim]
        :param batch_size_windows: (int) Size of the first dimension of the input tensor (batch size * windows)
        :param tokens: (int) Number of tokens in the input (N = window_size * window_size)
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case [num_mask_windows, N, N]
        :return: (torch.Tensor) Output feature map of the shape [batch size * windows, tokens, channels]
        """
        head_dim = query.shape[-1]

        # Scaled Cosine Attention (Swin v2)
        # Normalize query and key
        query_norm = F.normalize(query, p=2, dim=-1)
        key_norm = F.normalize(key, p=2, dim=-1)

        # Cosine similarity: einsum is efficient here
        # attention_map: [B*nW, nH, N, N]
        attention_map: torch.Tensor = torch.einsum("bhnd, bhmd -> bhnm", query_norm, key_norm)

        # Scale by learnable tau (clamp to avoid instability)
        # Tau shape is [num_heads], needs broadcasting: [1, nH, 1, 1]
        # Clamp tau to prevent extreme values, as suggested in paper
        # A reasonable range might be [0.01, 100] or similar depending on log/exp usage.
        # Using exp() as tau is often defined in log space. Clamping the original parameter.
        # tau = torch.clamp(self.tau, min=math.log(0.01), max=math.log(100.0)).exp() # Example if tau stored as log
        tau = torch.clamp(self.tau, min=0.01).view(1, self.number_of_heads, 1, 1) # Clamp positive tau directly
        attention_map = attention_map / tau

        # Apply relative positional encodings: [nH, N, N] -> broadcasted to [B*nW, nH, N, N]
        # Needs to be on the same device.
        rel_pos_bias = self.__get_relative_positional_encodings().to(attention_map.device)
        attention_map = attention_map + rel_pos_bias.unsqueeze(0) # Add batch dim for broadcasting

        # Apply mask if utilized
        if mask is not None:
            nW_mask = mask.shape[0] # Number of unique mask patterns

            # Infer the *expected* number of windows per image based on the resolution used for mask generation
            # This assumes self.window_attention holds a reference to the parent block or its properties accessible
            # Let's try to access parent block's resolution directly if possible, otherwise we might need to pass it
            # Assuming self.input_resolution IS the resolution of the feature map ENTERING the SwinBlock
            # Note: This resolution info might need to be more reliably obtained or passed down.
            # Let's find the resolution associated with the mask generation context.
            # This usually resides in the SwinTransformerBlock instance.
            # We need a way to access SwinTransformerBlock's input_resolution and window_size here.
            # This indicates a potential design issue - attention block might need resolution context.
            # --- TEMPORARY WORKAROUND: Recompute expected nW based on known N and nW_mask ---
            # This is brittle, assumes a somewhat standard layout.
            # A better fix involves passing the block's current input_resolution here.
            # Heuristic: Estimate grid size from nW_mask. sqrt(nW_mask) roughly gives grid dims.
            # Example: if nW_mask=9, grid is 3x3. if nW_mask=64, grid=8x8.
            # This assumes the mask generation covers the full grid.

            # --- More Robust Approach: Ensure B_win = B * nW and nW is divisible by nW_mask ---
            # Calculate nW (number of windows per image) based on the *actual* input tensor shape
            # We need batch size B. We can get it if we know nW.
            # B_win = input.shape[0] # Passed as batch_size_windows
            # We don't know B or nW reliably here without more context.

            # --- Let's stick to the tiling logic assuming B_win = B * nW ---
            # The mask needs to be applied cyclically. B_win is the total number of windows.
            # Each window `i` (0 to B_win-1) should use mask `i % nW_mask`.

            # Tile the mask to match the total number of windows B_win
            # Number of times to repeat the mask patterns = B_win / nW_mask
            # This division must be exact IF the input resolution perfectly tiled the mask patterns.
            if batch_size_windows % nW_mask != 0:
                 # This condition suggests the input size doesn't align with the mask pattern count.
                 # Could be due to padding or dynamic shapes mismatching the static mask.
                 # Let's proceed with flooring division for repeats and handle the remainder separately,
                 # although ideally this shouldn't happen.
                 # Or, more likely, the mask needs to be generated based on the *actual* input shape.
                 # For now, let's assume it *should* be divisible and raise error if not.
                 # Or maybe the mask should just be indexed? mask[window_index % nW_mask]
                #  print(f"Warning: batch_size_windows ({batch_size_windows}) not divisible by nW_mask ({nW_mask}). "
                #        f"Mask application might be incorrect for non-aligned inputs.")
                 # Fallback: Use integer division, might lead to shape errors later if size mismatch
                 n_repeats = batch_size_windows // nW_mask
            else:
                 n_repeats = batch_size_windows // nW_mask

            # Tile the mask: [nW_mask, N, N] -> [B_win, N, N]
            # mask_tiled = mask.repeat(n_repeats, 1, 1)[:batch_size_windows] # Ensure exact size
            # Alternative Tiling (handles non-divisible case better):
            indices = torch.arange(batch_size_windows, device=mask.device) % nW_mask
            mask_applied = mask[indices] # Shape: [B_win, N, N]

            # Add the mask, broadcasting over the head dimension (nH)
            # mask_applied: [B_win, N, N] -> [B_win, 1, N, N]
            # attention_map: [B_win, nH, N, N]
            attention_map = attention_map + mask_applied.unsqueeze(1)
            # No reshaping needed before/after if we use this broadcasting approach.

        # Apply softmax
        attention_map: torch.Tensor = attention_map.softmax(dim=-1)

        # Perform attention dropout
        attention_map: torch.Tensor = self.attention_dropout(attention_map)

        # Apply attention map to value: [B*nW, nH, N, N] @ [B*nW, nH, N, Vd] -> [B*nW, nH, N, Vd]
        # output: torch.Tensor = torch.einsum("bhnm, bhmd -> bhnd", attention_map, value) # If Vd == Kd
        output: torch.Tensor = torch.matmul(attention_map, value)

        # Reshape output: [B*nW, nH, N, Vd] -> [B*nW, N, nH, Vd] -> [B*nW, N, C]
        output: torch.Tensor = output.transpose(1, 2).reshape(batch_size_windows, tokens, -1)
        return output

    def __sequential_self_attention(self,
                                    query: torch.Tensor, # [B*nW, nH, N, head_dim]
                                    key: torch.Tensor,   # [B*nW, nH, N, head_dim]
                                    value: torch.Tensor, # [B*nW, nH, N, head_dim]
                                    batch_size_windows: int,
                                    tokens: int,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function performs sequential scaled cosine self-attention.
        Less memory intensive but potentially slower.
        :param query: (torch.Tensor) Query tensor of the shape [batch size * windows, heads, tokens, head_dim]
        :param key: (torch.Tensor) Key tensor of the shape [batch size * windows, heads, tokens, head_dim]
        :param value: (torch.Tensor) Value tensor of the shape [batch size * windows, heads, tokens, head_dim]
        :param batch_size_windows: (int) Size of the first dimension of the input tensor (batch size * windows)
        :param tokens: (int) Number of tokens in the input (N)
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case [num_mask_windows, N, N]
        :return: (torch.Tensor) Output feature map of the shape [batch size * windows, tokens, channels]
        """
        # Init output tensor
        output: torch.Tensor = torch.zeros_like(query) # Use zeros_like for correct shape and device
        head_dim = query.shape[-1]

        # Precompute relative positional encodings on the correct device
        rel_pos_bias = self.__get_relative_positional_encodings().to(query.device) # [nH, N, N]

        # Normalize key globally first
        key_norm = F.normalize(key, p=2, dim=-1) # [B*nW, nH, N, head_dim]

        # Get tau ready for use
        tau = torch.clamp(self.tau, min=0.01).view(1, self.number_of_heads, 1) # [1, nH, 1]

        # Iterate over query tokens
        for token_index_query in range(tokens):
            # Get current query slice: [B*nW, nH, head_dim]
            query_slice = query[:, :, token_index_query, :]
            # Normalize query slice
            query_slice_norm = F.normalize(query_slice, p=2, dim=-1) # [B*nW, nH, head_dim]

            # Compute cosine similarity with all keys for this query token
            # [B*nW, nH, 1, head_dim] * [B*nW, nH, head_dim, N] -> [B*nW, nH, 1, N]
            attention_map_slice: torch.Tensor = torch.einsum("bhd, bhmd -> bhn", query_slice_norm, key_norm)

            # Scale by tau
            attention_map_slice = attention_map_slice / tau # Broadcasting works

            # Apply positional encodings for this query token slice: [nH, N] -> [1, nH, N]
            rel_pos_bias_slice = rel_pos_bias[:, token_index_query, :] # Shape [nH, N]
            attention_map_slice = attention_map_slice + rel_pos_bias_slice.unsqueeze(0) # Add batch dim

            # Apply mask if utilized for this query token
            if mask is not None:
                nW_mask = mask.shape[0]
                # Get the correct mask slice for each window cyclically
                indices = torch.arange(batch_size_windows, device=mask.device) % nW_mask
                # mask shape [nW_mask, N, N], select slice for current query token [nW_mask, N]
                mask_slice_all_patterns = mask[:, token_index_query, :] # [nW_mask, N]
                # Apply the cyclic index: [B_win, N]
                mask_slice_applied = mask_slice_all_patterns[indices]

                # Add the mask slice, broadcasting over the head dimension (nH)
                # mask_slice_applied: [B_win, N] -> [B_win, 1, N]
                # attention_map_slice: [B_win, nH, N]
                attention_map_slice = attention_map_slice + mask_slice_applied.unsqueeze(1)
                # No reshaping needed here either.

            # Apply softmax over key dimension (N)
            attention_map_slice: torch.Tensor = attention_map_slice.softmax(dim=-1) # Shape: [B*nW, nH, N]

            # Perform attention dropout
            attention_map_slice: torch.Tensor = self.attention_dropout(attention_map_slice)

            # Apply attention map slice to values: [B*nW, nH, N] @ [B*nW, nH, N, Vd] -> [B*nW, nH, Vd]
            # output_slice = torch.einsum("bhn, bhnv -> bhv", attention_map_slice, value)
            # Need to reshape value for matmul: [B*nW, nH, N, Vd]
            output_slice = torch.matmul(attention_map_slice.unsqueeze(2), value).squeeze(2) # [B*nW, nH, 1, N] @ [B*nW, nH, N, Vd] -> [B*nW, nH, 1, Vd] -> [B*nW, nH, Vd]

            # Store result in the corresponding output position
            output[:, :, token_index_query, :] = output_slice

        # Reshape output: [B*nW, nH, N, Vd] -> [B*nW, N, nH, Vd] -> [B*nW, N, C]
        output: torch.Tensor = output.transpose(1, 2).reshape(batch_size_windows, tokens, -1)
        return output


    def forward(self,
                input: torch.Tensor, # Shape: [B*num_windows, C, Wh, Ww]
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size * windows, channels, window_size, window_size]
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case [num_mask_windows, N, N]
        :return: (torch.Tensor) Output tensor of the shape [batch size * windows, channels, window_size, window_size]
        """
        # Save original shape
        batch_size_windows, channels, height, width = input.shape  # Wh, Ww = window_size
        tokens: int = height * width # N = Wh * Ww
        head_dim = channels // self.number_of_heads

        # Reshape input to [batch size * windows, tokens (N), channels (C)]
        input_reshaped: torch.Tensor = input.flatten(2).transpose(1, 2) # [B*nW, N, C]

        # Perform query, key, and value mapping
        query_key_value: torch.Tensor = self.mapping_qkv(input_reshaped) # [B*nW, N, 3*C]
        # Reshape and permute for multi-head attention
        # [B*nW, N, 3, num_heads, head_dim] -> [3, B*nW, num_heads, N, head_dim]
        query_key_value: torch.Tensor = query_key_value.view(batch_size_windows, tokens, 3, self.number_of_heads,
                                                             head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value[0], query_key_value[1], query_key_value[2]

        # Perform attention
        if self.sequential_self_attention:
            output_attention: torch.Tensor = self.__sequential_self_attention(query=query, key=key, value=value,
                                                                    batch_size_windows=batch_size_windows,
                                                                    tokens=tokens,
                                                                    mask=mask)
        else:
            output_attention: torch.Tensor = self.__self_attention(query=query, key=key, value=value,
                                                         batch_size_windows=batch_size_windows, tokens=tokens,
                                                         mask=mask)
        # output_attention shape: [B*nW, N, C]

        # Perform linear projection and dropout
        output: torch.Tensor = self.projection_dropout(self.projection(output_attention)) # [B*nW, N, C]

        # Reshape output to original shape [batch size * windows, channels, height, width]
        output: torch.Tensor = output.transpose(1, 2).view(batch_size_windows, channels, height, width)
        return output


class SwinTransformerBlock(nn.Module):
    """
    This class implements the Swin transformer block (v1 and v2 compatible).
    Differences for v2 mainly lie within WindowMultiHeadAttention and LayerNorm placement.
    """

    def __init__(self,
                 in_channels: int,
                 input_resolution: Tuple[int, int],
                 number_of_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: float = 0.0,
                 sequential_self_attention: bool = False,
                 post_norm: bool = False) -> None: # Added post_norm flag (Swin v1 uses pre-norm)
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param input_resolution: (Tuple[int, int]) Input resolution (H, W) for mask generation
        :param number_of_heads: (int) Number of attention heads to be utilized
        :param window_size: (int) Window size to be utilized
        :param shift_size: (int) Shifting size to be used (0 for W-MSA, >0 for SW-MSA)
        :param ff_feature_ratio: (int) Ratio of the hidden dimension in the FFN to the input channels
        :param dropout: (float) Dropout in FFN and projection layers
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_path: (float) Stochastic depth rate
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        :param post_norm: (bool) If true, use post-normalization (like ViT, Swin v2), else pre-normalization (Swin v1)
        """
        # Call super constructor
        super(SwinTransformerBlock, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        self.input_resolution: Tuple[int, int] = input_resolution
        self.number_of_heads = number_of_heads # Needed for Deformable block subclass
        self.post_norm = post_norm

        # Catch case if resolution is smaller than the window size
        if min(self.input_resolution) <= window_size:
            # If resolution is smaller than window size, adapt window size and disable shifting
            self.window_size: int = min(self.input_resolution)
            self.shift_size: int = 0
            self.make_windows: bool = (self.window_size > 1) # Only make windows if size > 1
            # print(f"Warning: Input resolution {self.input_resolution} is smaller than window size {window_size}. Adjusting window size to {self.window_size} and disabling shift.")
        else:
            self.window_size: int = window_size
            self.shift_size: int = shift_size
            self.make_windows: bool = True

        # Check shift size
        if self.shift_size >= self.window_size:
            # print(f"Warning: Shift size {self.shift_size} >= window size {self.window_size}. Setting shift size to 0.")
            self.shift_size = 0


        # Init normalization layers
        # Swin v2 uses one LayerNorm before the MLP block. Pre-norm has one before attn too.
        self.normalization_1: nn.Module = nn.LayerNorm(normalized_shape=in_channels)
        if not self.post_norm: # Only need second norm for pre-norm case
            self.normalization_2: nn.Module = nn.LayerNorm(normalized_shape=in_channels)

        # Init window attention module (using the updated Swin v2 attention)
        self.window_attention: WindowMultiHeadAttention = WindowMultiHeadAttention(
            in_features=in_channels,
            window_size=self.window_size,
            number_of_heads=number_of_heads,
            dropout_attention=dropout_attention,
            dropout_projection=dropout, # Use general dropout for projection
            sequential_self_attention=sequential_self_attention)

        # Init dropout layer (stochastic depth)
        self.dropout: nn.Module = timm.models.layers.DropPath(
            drop_prob=dropout_path) if dropout_path > 0. else nn.Identity()

        # Init feed-forward network
        self.feed_forward_network: nn.Module = FeedForward(in_features=in_channels,
                                                           hidden_features=int(in_channels * ff_feature_ratio),
                                                           dropout=dropout, # Use general dropout here
                                                           out_features=in_channels)
        # Make attention mask (needed only if shifting)
        self.__make_attention_mask()

    def __make_attention_mask(self) -> None:
        """
        Method generates the attention mask used in shift case (SW-MSA).
        The mask ensures that attention is only computed within the same sub-window.
        """
        attention_mask: Optional[torch.Tensor] = None # Default to None
        if self.shift_size > 0 and self.make_windows:
            height, width = self.input_resolution
            # Calculate the image configuration after shifting
            img_mask = torch.zeros((1, height, width, 1)) # Use 4D tensor for compatibility with window_partition
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # Use unfold to partition into windows (simulate timm's window_partition)
            # Need BCHW format for unfold
            img_mask_bchw = img_mask.permute(0, 3, 1, 2) # [1, 1, H, W]

            # We need unfold to work correctly even if H, W are not divisible by window_size
            # Since we are only creating a mask, we can assume the *actual* input
            # will be handled (e.g., padded) if necessary by the main forward pass.
            # Here, we work with the specified input_resolution.
            # If input_resolution is smaller than window_size, make_windows should be false.
            pad_h = (self.window_size - height % self.window_size) % self.window_size
            pad_w = (self.window_size - width % self.window_size) % self.window_size
            if pad_h > 0 or pad_w > 0:
                 mask_unfold_input = F.pad(img_mask_bchw, (0, pad_w, 0, pad_h))
            else:
                 mask_unfold_input = img_mask_bchw

            mask_windows = unfold(mask_unfold_input, self.window_size) # [num_windows, 1, Wh, Ww]
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # [num_windows, N]

            # Create the attention mask: [num_windows, N, N]
            # Pairwise comparison: if values are different, they belong to different original sub-windows
            attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # Fill with -100 (or large negative number) where values differ, 0 where they are the same
            attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0))
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float(0.0))

        # Register the buffer (even if it's None)
        self.register_buffer("attention_mask", attention_mask, persistent=False)

    def update_resolution(self,
                          new_window_size: int,
                          new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size and input resolution, recalculating the mask.
        :param new_window_size: (int) New window size
        :param new_input_resolution: (Tuple[int, int]) New input resolution
        """
        # Update input resolution
        self.input_resolution: Tuple[int, int] = new_input_resolution
        # Check if resolution is smaller than the new window size
        if min(self.input_resolution) <= new_window_size:
            self.window_size: int = min(self.input_resolution)
            self.shift_size: int = 0 # Disable shifting if window size is adapted
            self.make_windows: bool = (self.window_size > 1)
            # print(f"Warning: New input resolution {self.input_resolution} is smaller than new window size {new_window_size}. Adjusting window size to {self.window_size} and disabling shift.")
        else:
            self.window_size: int = new_window_size
            # Keep original shift logic relative to potentially new window size, but ensure shift < window
            original_shift_fraction = self.shift_size / self.window_attention.window_size if self.window_attention.window_size > 0 else 0
            self.shift_size = int(original_shift_fraction * self.window_size) if (original_shift_fraction > 0) else 0
            if self.shift_size >= self.window_size:
                self.shift_size = 0 # Disable if calculation leads to invalid shift
            self.make_windows: bool = True

        # Update attention mask
        self.__make_attention_mask()
        # Update attention module's window size and relative positions
        self.window_attention.update_resolution(new_window_size=self.window_size)


    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Swin Transformer Block.
        Supports both pre-norm (Swin v1) and post-norm (Swin v2) configurations.
        :param input: (torch.Tensor) Input tensor of the shape [B, C, H, W]
        :return: (torch.Tensor) Output tensor of the shape [B, C, H, W]
        """
        # Save shape and shortcut
        batch_size, channels, height, width = input.shape
        shortcut = input

        # --- Pre-Normalization Logic (Swin v1 style) ---
        if not self.post_norm:
            # Apply first LayerNorm (Input: BCHW -> BHWC -> Norm -> BHCW -> BCHW)
            norm1_input = bhwc_to_bchw(self.normalization_1(bchw_to_bhwc(input)))
        else:
            # For post-norm, normalization happens *after* the main ops
            norm1_input = input # Pass input directly to attention/shifting

        # --- Cyclic Shift ---
        if self.shift_size > 0 and self.make_windows:
            shifted_input: torch.Tensor = torch.roll(input=norm1_input, shifts=(-self.shift_size, -self.shift_size),
                                                    dims=(2, 3)) # Shift H and W dims
        else:
            shifted_input: torch.Tensor = norm1_input

        # --- Window Partitioning ---
        if self.make_windows:
            # Unfold/Partition input into windows: [B, C, H, W] -> [B*nW, C, Wh, Ww]
            input_windows: torch.Tensor = unfold(input=shifted_input, window_size=self.window_size)
        else:
            # If not making windows (e.g., resolution <= window size), treat the whole input as one "window"
            input_windows: torch.Tensor = shifted_input

        # --- Window Multi-Head Self-Attention (W-MSA / SW-MSA) ---
        # Input shape: [B*nW, C, Wh, Ww], Mask shape: [nW_mask, N, N] or None
        attn_windows: torch.Tensor = self.window_attention(input_windows, mask=self.attention_mask)
        # Output shape: [B*nW, C, Wh, Ww]

        # --- Reverse Window Partitioning ---
        if self.make_windows:
            # Fold/Merge windows back: [B*nW, C, Wh, Ww] -> [B, C, H, W] (handles padding removal)
            merged_windows: torch.Tensor = fold(input=attn_windows, window_size=self.window_size, height=height, width=width)
        else:
            merged_windows: torch.Tensor = attn_windows # No merging needed

        # --- Reverse Cyclic Shift ---
        if self.shift_size > 0 and self.make_windows:
            # Roll back the shift
            attention_output: torch.Tensor = torch.roll(input=merged_windows, shifts=(self.shift_size, self.shift_size),
                                                      dims=(2, 3))
        else:
            attention_output: torch.Tensor = merged_windows

        # --- Post-Normalization & Residual Logic ---
        if self.post_norm: # Swin v2 style
            # First residual connection
            x = shortcut + self.dropout(attention_output)
            # Apply LayerNorm, FFN, DropPath, and second residual connection
            # Norm input: BCHW -> BHWC -> Norm -> BHWC
            norm_out = self.normalization_1(bchw_to_bhwc(x))
            # FFN input: BHWC -> FFN -> BHWC
            ffn_out = self.feed_forward_network(norm_out)
            # Reshape back: BHWC -> BCHW
            ffn_out_bchw = bhwc_to_bchw(ffn_out)
            # Second residual connection
            output = x + self.dropout(ffn_out_bchw)

        else: # Swin v1 style (pre-norm)
            # First residual connection (after attention and DropPath)
            x = shortcut + self.dropout(attention_output)
            # Apply second LayerNorm, FFN, DropPath, and second residual connection
            # Norm input: BCHW -> BHWC -> Norm -> BHWC
            norm2_input = bchw_to_bhwc(x)
            norm2_out = self.normalization_2(norm2_input)
            # FFN input: BHWC -> FFN -> BHWC
            ffn_out = self.feed_forward_network(norm2_out)
            # Reshape back: BHWC -> BCHW
            ffn_out_bchw = bhwc_to_bchw(ffn_out)
             # Second residual connection
            output = x + self.dropout(ffn_out_bchw)

        return output


class DeformableSwinTransformerBlock(SwinTransformerBlock):
    """
    This class implements a deformable version of the Swin Transformer block.
    Inspired by: https://arxiv.org/pdf/2201.00520.pdf (Not a direct implementation)
    This version applies offsets *before* the standard Swin block logic.
    """

    def __init__(self,
                 in_channels: int,
                 input_resolution: Tuple[int, int],
                 number_of_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: float = 0.0,
                 sequential_self_attention: bool = False,
                 post_norm: bool = False, # Inherit post_norm flag
                 offset_downscale_factor: int = 2, # Downscale factor for offset prediction conv
                 offset_kernel_size: int = 5, # Kernel size for offset prediction conv
                 offset_groups: int = 1 # Groups for offset prediction conv (1=dense, in_channels=depthwise)
                 ) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param input_resolution: (Tuple[int, int]) Input resolution
        :param number_of_heads: (int) Number of attention heads to be utilized
        :param window_size: (int) Window size to be utilized
        :param shift_size: (int) Shifting size to be used
        :param ff_feature_ratio: (int) Ratio of the hidden dimension in the FFN to the input channels
        :param dropout: (float) Dropout in input mapping
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_path: (float) Dropout in main path
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        :param post_norm: (bool) If true, use post-normalization style
        :param offset_downscale_factor: (int) Downscale factor of offset network stride
        :param offset_kernel_size: (int) Kernel size for offset prediction conv
        :param offset_groups: (int) Groups for offset prediction conv
        """
        # Call super constructor
        super(DeformableSwinTransformerBlock, self).__init__(
            in_channels=in_channels,
            input_resolution=input_resolution,
            number_of_heads=number_of_heads,
            window_size=window_size,
            shift_size=shift_size,
            ff_feature_ratio=ff_feature_ratio,
            dropout=dropout,
            dropout_attention=dropout_attention,
            dropout_path=dropout_path,
            sequential_self_attention=sequential_self_attention,
            post_norm=post_norm # Pass post_norm setting to parent
        )
        # Save parameters specific to deformable part
        self.offset_downscale_factor: int = offset_downscale_factor
        # Note: The number of offset *pairs* should match the number of heads
        # The offset network predicts 2 values (dx, dy) for each head.
        num_offset_outputs = 2 * self.number_of_heads

        # Make default sampling grid [-1, 1]
        self.__make_default_offsets()

        # Init offset network
        # Predicts offsets based on the input features
        # Conv -> GELU -> Conv (predicts 2*num_heads outputs)
        # Padding calculation: 'same' equivalent for stride > 1 is complex.
        # Calculate padding to keep spatial dims roughly H/factor, W/factor
        # Example padding: (offset_kernel_size - offset_downscale_factor) // 2 might work for stride=factor
        # Let's use a simpler padding and let dimensions adjust. Pad=kernel//2 usually works well.
        self.offset_network: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels, # Intermediate channels often same as input
                      kernel_size=offset_kernel_size,
                      stride=offset_downscale_factor,
                      padding=offset_kernel_size // 2, # Standard padding
                      groups=offset_groups, # Allow depthwise or grouped conv
                      bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_offset_outputs, # Predict 2 values per head
                      kernel_size=1, # 1x1 conv to project to offset dimension
                      stride=1,
                      padding=0,
                      bias=True)
        )
        # Initialize the bias of the final offset conv to zero for stable training start
        nn.init.constant_(self.offset_network[-1].bias, 0)

    def __make_default_offsets(self) -> None:
        """
        Method generates the default sampling grid (normalized to [-1, 1]).
        """
        height, width = self.input_resolution
        # Handle zero dimensions if they occur
        if height == 0 or width == 0:
             # Set grid to something valid but small, or raise error
            #  print(f"Warning: Input resolution {self.input_resolution} has zero dimension. Creating minimal grid.")
             grid = torch.zeros(1, 1, 1, 2) # Minimal grid
        else:
            # Init x and y coordinates
            # Use torch.linspace for consistency, requires device later
            x: torch.Tensor = torch.linspace(-1, 1, width)
            y: torch.Tensor = torch.linspace(-1, 1, height)

            # Make grid [H, W, 2] using meshgrid (ensure 'xy' indexing for grid_sample)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij') # H, W
            grid: torch.Tensor = torch.stack((grid_x, grid_y), dim=-1) # H, W, 2

            # Reshape grid to [1, H, W, 2] (batch dim 1)
            grid: torch.Tensor = grid.unsqueeze(dim=0)

        # Register in module, not persistent if it depends only on resolution
        self.register_buffer("default_grid", grid, persistent=False)


    def update_resolution(self, new_window_size: int, new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size, input resolution, and recalculates the default grid.
        :param new_window_size: (int) New window size
        :param new_input_resolution: (Tuple[int, int]) New input resolution
        """
        # Update resolution and window size in the parent class (handles mask etc.)
        super(DeformableSwinTransformerBlock, self).update_resolution(new_window_size=new_window_size,
                                                                      new_input_resolution=new_input_resolution)
        # Update default sampling grid based on the new resolution
        self.__make_default_offsets()

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Deformable Swin Transformer Block.
        1. Predict offsets.
        2. Resample input features based on offsets.
        3. Pass resampled features through the standard SwinTransformerBlock logic.
        :param input: (torch.Tensor) Input tensor [B, C, H, W]
        :return: (torch.Tensor) Output tensor [B, C, H, W]
        """
        # Get input shape
        batch_size, channels, height, width = input.shape

        # 1. Predict offsets
        # Offsets shape: [B, 2 * num_heads, H_offset, W_offset]
        offsets: torch.Tensor = self.offset_network(input)
        # Upscale offsets to the original input resolution [B, 2 * num_heads, H, W]
        # Using bilinear interpolation
        offsets: torch.Tensor = F.interpolate(input=offsets,
                                              size=(height, width),
                                              mode="bilinear",
                                              align_corners=False) # Align corners often False for feature maps

        # 2. Prepare for grid_sample
        # Reshape offsets: [B, 2 * num_heads, H, W] -> [B, num_heads, 2, H, W] -> [B * num_heads, H, W, 2]
        # Offsets represent deviations from the regular grid [-1, 1]
        # Scaling offsets (e.g., by tanh) can limit their range, common practice.
        offsets = offsets.tanh() # Scale offsets to [-1, 1] range

        # Reshape for grid_sample: each head gets its own offset map
        # [B, 2*nH, H, W] -> [B, nH, 2, H, W] -> [B*nH, H, W, 2]
        offsets_reshaped = offsets.view(batch_size, self.number_of_heads, 2, height, width)
        offsets_permuted = offsets_reshaped.permute(0, 1, 3, 4, 2) # [B, nH, H, W, 2]
        offsets_final = offsets_permuted.reshape(batch_size * self.number_of_heads, height, width, 2)

        # Prepare the default grid: [1, H, W, 2] -> [B * num_heads, H, W, 2]
        # Ensure grid is on the same device and dtype as offsets
        default_grid_aligned = self.default_grid.to(offsets_final.device, dtype=offsets_final.dtype)
        # Repeat grid for each head and batch item
        grid_repeated = default_grid_aligned.repeat(batch_size * self.number_of_heads, 1, 1, 1)

        # Construct the final sampling grid: default grid + predicted offsets
        # Clip grid values to [-1, 1] as required by grid_sample
        sampling_grid: torch.Tensor = (grid_repeated + offsets_final).clamp_(min=-1, max=1)

        # Prepare input for grid_sample: Reshape input per head
        # [B, C, H, W] -> [B, nH, C//nH, H, W] -> [B*nH, C//nH, H, W]
        head_dim = channels // self.number_of_heads
        input_reshaped_per_head = input.view(batch_size, self.number_of_heads, head_dim, height, width)
        input_flattened_per_head = input_reshaped_per_head.reshape(batch_size * self.number_of_heads, head_dim, height, width)

        # 3. Apply sampling grid using F.grid_sample
        input_resampled_per_head: torch.Tensor = F.grid_sample(
                                                        input=input_flattened_per_head,
                                                        grid=sampling_grid,
                                                        mode="bilinear",
                                                        align_corners=False, # Match interpolate setting
                                                        padding_mode="reflection") # Reflection padding often works well
        # Output shape: [B*nH, C//nH, H, W]

        # 4. Reshape resampled tensor back to [B, C, H, W]
        input_resampled: torch.Tensor = input_resampled_per_head.view(batch_size, self.number_of_heads, head_dim, height, width)
        input_resampled: torch.Tensor = input_resampled.reshape(batch_size, channels, height, width)

        # 5. Pass the *resampled* input through the standard SwinTransformerBlock forward method
        # We call the *parent* class's forward method using super()
        output: torch.Tensor = super(DeformableSwinTransformerBlock, self).forward(input=input_resampled)

        return output


class PatchMerging(nn.Module):
    """
    Patch Merging Layer. Downsamples the feature map by 2x and increases channels by 2x.
    Input: B, C, H, W
    Output: B, 2*C, H/2, W/2
    """

    def __init__(self,
                 in_channels: int,
                 norm_layer=nn.LayerNorm) -> None: # Allow specifying norm layer
        """
        Constructor method
        :param in_channels: (int) Number of input channels.
        :param norm_layer: Normalization layer to use. Default is nn.LayerNorm.
        """
        super().__init__()
        self.in_channels = in_channels
        # Linear layer to reduce features after concatenation. Input features = 4 * in_channels.
        self.reduction: nn.Module = nn.Linear(in_features=4 * in_channels, out_features=2 * in_channels, bias=False)
        # Normalization layer applied *before* the reduction.
        self.norm: nn.Module = norm_layer(4 * in_channels)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [B, C, H, W]
        :return: (torch.Tensor) Output tensor of the shape [B, 2*C, H/2, W/2]
        """
        batch_size, channels, height, width = input.shape

        # --- Input Checks and Padding ---
        # Ensure height and width are even for 2x2 merging. Pad if necessary.
        pad_input = False
        if height % 2 == 1 or width % 2 == 1:
            pad_input = True
            # Pad right and bottom: (padding_left, padding_right, padding_top, padding_bottom)
            padding = (0, width % 2, 0, height % 2)
            input = F.pad(input, padding)
            # Update shape after padding
            batch_size, channels, height, width = input.shape

        # Permute to [B, H, W, C] for easier selection
        input_bhwc = bchw_to_bhwc(input)

        # --- Select Patches ---
        # Select elements with 2x2 stride
        # Top-left:    input_bhwc[:, 0::2, 0::2, :]
        # Top-right:   input_bhwc[:, 0::2, 1::2, :]
        # Bottom-left: input_bhwc[:, 1::2, 0::2, :]
        # Bottom-right:input_bhwc[:, 1::2, 1::2, :]

        # Concatenate along the channel dimension (last dimension)
        # Output shape: [B, H/2, W/2, 4*C]
        concatenated_features = torch.cat([input_bhwc[:, 0::2, 0::2, :],
                                           input_bhwc[:, 0::2, 1::2, :],
                                           input_bhwc[:, 1::2, 0::2, :],
                                           input_bhwc[:, 1::2, 1::2, :]], dim=-1)

        # --- Normalization and Reduction ---
        # Apply LayerNorm
        norm_output = self.norm(concatenated_features) # Input/Output: [B, H/2, W/2, 4*C]
        # Apply linear reduction
        reduced_output = self.reduction(norm_output)  # Output: [B, H/2, W/2, 2*C]

        # Permute back to [B, 2*C, H/2, W/2]
        output = bhwc_to_bchw(reduced_output)

        return output


class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding Layer.
    Uses a Conv2d layer for patch extraction and embedding.
    Input: B, C_in, H, W
    Output: B, C_embed, H/patch_size, W/patch_size
    """

    def __init__(self,
                 in_channels: int = 3,
                 embedding_channels: int = 96,
                 patch_size: int = 4,
                 norm_layer=None) -> None: # Allow optional normalization layer
        """
        Constructor method
        :param in_channels: (int) Number of input image channels (e.g., 3 for RGB).
        :param embedding_channels: (int) Number of output channels (embedding dimension).
        :param patch_size: (int) Size of the square patch (e.g., 4 means 4x4 patches).
        :param norm_layer: Normalization layer to apply after convolution. Default is None.
        """
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.embedding_channels = embedding_channels

        # Use Conv2d to implement patch embedding
        self.projection: nn.Module = nn.Conv2d(in_channels=in_channels,
                                               out_channels=embedding_channels,
                                               kernel_size=self.patch_size,
                                               stride=self.patch_size)

        # Optional normalization layer
        if norm_layer is not None:
            # LayerNorm expects shape [*, normalized_shape]
            # For BCHW output of Conv2d, need to apply norm on C dimension,
            # often done by permuting to BHWC, applying norm, permuting back.
            # Or use LayerNorm directly if suitable (might need GroupNorm for channel-wise norm).
            # Timm uses LayerNorm(embedding_channels) applied after permuting to N, L, C.
            # Here, let's apply it after conv, assuming BHWC intermediate if LayerNorm is used.
            self.norm = norm_layer(embedding_channels)
        else:
            self.norm = nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass transforms an image into patch embeddings.
        :param input: (torch.Tensor) Input images of the shape [B, C_in, H, W]
        :return: (torch.Tensor) Patch embedding of the shape [B, C_embed, H/patch_size, W/patch_size]
        """
        # Get input shape
        batch_size, channels, height, width = input.shape

        # --- Input Padding (Optional but recommended) ---
        # Ensure H and W are divisible by patch_size. Pad if necessary.
        pad_h = (self.patch_size[0] - height % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - width % self.patch_size[1]) % self.patch_size[1]
        if pad_h > 0 or pad_w > 0:
            # Pad right and bottom
            input = F.pad(input, (0, pad_w, 0, pad_h))

        # --- Convolutional Projection ---
        # [B, C_in, H, W] -> [B, C_embed, H/patch, W/patch]
        embedding: torch.Tensor = self.projection(input)

        # --- Normalization ---
        # If LayerNorm is used, typically requires BHWC format.
        if isinstance(self.norm, nn.LayerNorm):
            embedding_bhwc = bchw_to_bhwc(embedding)
            norm_embedding_bhwc = self.norm(embedding_bhwc)
            embedding = bhwc_to_bchw(norm_embedding_bhwc)
        else:
            # Apply other norms (like Identity or BatchNorm) directly
            embedding = self.norm(embedding)

        return embedding


class SwinTransformerStage(nn.Module):
    """
    This class implements a stage of the Swin transformer, consisting of multiple blocks.
    Optionally includes downsampling via PatchMerging at the beginning.
    """

    def __init__(self,
                 in_channels: int,          # Channels from previous stage or embedding
                 depth: int,                # Number of transformer blocks in this stage
                 downscale: bool,           # Whether to apply PatchMerging at the start
                 input_resolution: Tuple[int, int], # Input resolution to this stage (H, W)
                 number_of_heads: int,      # Number of attention heads in this stage's blocks
                 window_size: int = 7,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: Union[List[float], float] = 0.0, # Can be list or float
                 use_checkpoint: bool = False,    # Use gradient checkpointing for memory saving
                 sequential_self_attention: bool = False,
                 use_deformable_block: bool = False,
                 post_norm: bool = False) -> None: # Added post_norm flag
        """
        Constructor method
        :param in_channels: (int) Number of input channels to the stage (before downsampling)
        :param depth: (int) Number of transformer blocks in this stage.
        :param downscale: (bool) If true, apply PatchMerging at the beginning of the stage.
        :param input_resolution: (Tuple[int, int]) Spatial resolution (H, W) of the input to this stage.
        :param number_of_heads: (int) Number of attention heads for blocks in this stage.
        :param window_size: (int) Window size for W-MSA/SW-MSA.
        :param ff_feature_ratio: (int) Feed-forward network hidden feature ratio.
        :param dropout: (float) General dropout rate for FFN/projection.
        :param dropout_attention: (float) Dropout rate for attention maps.
        :param dropout_path: (Union[List[float], float]) Stochastic depth rate(s) for blocks in this stage.
        :param use_checkpoint: (bool) Whether to use gradient checkpointing for transformer blocks.
        :param sequential_self_attention: (bool) Use memory-efficient sequential attention.
        :param use_deformable_block: (bool) Use DeformableSwinTransformerBlock instead of SwinTransformerBlock.
        :param post_norm: (bool) Use post-normalization style blocks.
        """
        super(SwinTransformerStage, self).__init__()
        self.use_checkpoint: bool = use_checkpoint
        self.downscale: bool = downscale

        # --- Downsampling (Patch Merging) ---
        if downscale:
            self.downsample: nn.Module = PatchMerging(in_channels=in_channels, norm_layer=nn.LayerNorm)
            # Update resolution and channels for the rest of the stage
            current_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
            current_channels = in_channels * 2
        else:
            self.downsample: nn.Module = nn.Identity()
            current_resolution = input_resolution
            current_channels = in_channels

        # Store the resolution *after* potential downsampling, used by blocks
        self.current_resolution = current_resolution

        # --- Transformer Blocks ---
        # Select block type
        block_class = DeformableSwinTransformerBlock if use_deformable_block else SwinTransformerBlock

        # Stochastic depth decay rule (if dropout_path is a list)
        if isinstance(dropout_path, list):
            assert len(dropout_path) == depth, "Length of dropout_path list must match stage depth."
            dpr = dropout_path
        else: # Assume it's a float, create linearly increasing dropout rates
              # This is common practice, but the original code allowed a single float too.
              # Let's keep the possibility of a single float applying to all blocks.
              # If a single float is given, just use it directly.
            dpr = [dropout_path] * depth # Apply same drop path rate to all blocks if single float


        self.blocks: nn.ModuleList = nn.ModuleList([
            block_class(in_channels=current_channels,         # Use potentially increased channels
                        input_resolution=current_resolution, # Use potentially decreased resolution
                        number_of_heads=number_of_heads,
                        window_size=window_size,
                        # Alternate shift_size: 0 for even blocks, window_size // 2 for odd blocks
                        shift_size=0 if ((index % 2) == 0) else window_size // 2,
                        ff_feature_ratio=ff_feature_ratio,
                        dropout=dropout,
                        dropout_attention=dropout_attention,
                        dropout_path=dpr[index], # Use the specific drop path rate for this block
                        sequential_self_attention=sequential_self_attention,
                        post_norm=post_norm) # Pass post_norm setting to blocks
            for index in range(depth)])

    def update_resolution(self, new_window_size: int, new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the resolution and window size for the stage and its blocks.
        :param new_window_size: (int) New window size.
        :param new_input_resolution: (Tuple[int, int]) New input resolution *to the stage*.
        """
        # Determine the resolution *after* potential downsampling for this stage
        if self.downscale:
            current_resolution = (new_input_resolution[0] // 2, new_input_resolution[1] // 2)
        else:
            current_resolution = new_input_resolution

        # Update the stored current resolution
        self.current_resolution = current_resolution

        # Update resolution and window size for each block in the stage
        for block in self.blocks:
            # The block's update_resolution method expects the resolution it actually receives
            block.update_resolution(new_window_size=new_window_size, new_input_resolution=self.current_resolution)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Swin Transformer Stage.
        :param input: (torch.Tensor) Input tensor of the shape [B, C_in, H_in, W_in]
        :return: (torch.Tensor) Output tensor of the shape [B, C_out, H_out, W_out]
                 (C_out=2*C_in, H_out=H_in/2 etc. if downscale=True)
        """
        # 1. Apply downsampling (Patch Merging) if configured
        x: torch.Tensor = self.downsample(input)

        # 2. Pass through transformer blocks
        for block in self.blocks:
            # Use gradient checkpointing if enabled
            if self.use_checkpoint and self.training: # Checkpointing only during training
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        return x


class SwinTransformerV2(nn.Module):
    """
    Swin Transformer V2 backbone.
    Based on the papers:
    Swin Transformer V1: https://arxiv.org/abs/2103.14030
    Swin Transformer V2: https://arxiv.org/abs/2111.09883
    Deformable Swin: https://arxiv.org/abs/2201.00520 (Optional block type)
    """

    def __init__(self,
                 input_resolution: Tuple[int, int] = (224, 224), # Default ImageNet size
                 patch_size: int = 4,
                 in_channels: int = 3,
                 embedding_channels: int = 96, # Base channel dimension C (e.g., 96 for T/S, 128 for B)
                 depths: Tuple[int, ...] = (2, 2, 6, 2), # Number of blocks in each stage
                 number_of_heads: Tuple[int, ...] = (3, 6, 12, 24), # Number of heads in each stage
                 window_size: int = 7,
                 ff_feature_ratio: int = 4, # MLP expansion ratio
                 dropout: float = 0.0,      # General dropout rate
                 dropout_attention: float = 0.0, # Attention dropout rate
                 dropout_path: float = 0.1,   # Stochastic depth rate (max value, linearly increased)
                 norm_layer = nn.LayerNorm,   # Normalization layer
                 post_norm: bool = True,      # Use post-normalization (Swin v2 default)
                 patch_norm: bool = True,     # Add norm layer after patch embedding (Swin v1/v2 default)
                 use_checkpoint: bool = False, # Use gradient checkpointing
                 sequential_self_attention: bool = False, # Use memory-efficient sequential attention
                 use_deformable_block_stages: Tuple[bool, ...] = (False, False, False, False) # Control deformable blocks per stage
                 ) -> None:
        """
        Constructor method
        :param input_resolution: (Tuple[int, int]) Input image resolution (H, W).
        :param patch_size: (int) Size of patches (patch_size x patch_size).
        :param in_channels: (int) Number of input image channels.
        :param embedding_channels: (int) Dimension of patch embeddings (C).
        :param depths: (Tuple[int, ...]) Number of blocks in each of the 4 stages.
        :param number_of_heads: (Tuple[int, ...]) Number of attention heads in each of the 4 stages.
        :param window_size: (int) Window size for attention.
        :param ff_feature_ratio: (int) MLP hidden dimension ratio.
        :param dropout: (float) Dropout rate for MLP and projection layers.
        :param dropout_attention: (float) Dropout rate for attention weights.
        :param dropout_path: (float) Maximum stochastic depth rate for the last block. Linearly increases from 0.
        :param norm_layer: Normalization layer constructor (default: nn.LayerNorm).
        :param post_norm: (bool) Whether to use post-normalization (True for Swin v2) or pre-normalization (False for Swin v1).
        :param patch_norm: (bool) Whether to apply normalization after patch embedding.
        :param use_checkpoint: (bool) Whether to use gradient checkpointing in stages.
        :param sequential_self_attention: (bool) Whether to use sequential self-attention in blocks.
        :param use_deformable_block_stages: (Tuple[bool, ...]) Tuple indicating which stages should use DeformableSwinTransformerBlock. Length must match `depths`.
        """
        super().__init__()

        # --- Store configuration ---
        self.num_layers = len(depths)
        self.embedding_channels = embedding_channels
        self.patch_norm = patch_norm
        self.num_features = int(embedding_channels * 2**(self.num_layers - 1)) # Features after last stage
        self.ff_feature_ratio = ff_feature_ratio
        self.post_norm = post_norm
        self.patch_size = patch_size # Store patch size for resolution updates

        assert len(use_deformable_block_stages) == self.num_layers, \
            "Length of use_deformable_block_stages must match the number of stages (length of depths)."

        # --- Patch Embedding ---
        self.patch_embedding: nn.Module = PatchEmbedding(
            in_channels=in_channels,
            embedding_channels=embedding_channels,
            patch_size=patch_size,
            norm_layer=norm_layer if self.patch_norm else None)

        # Calculate spatial resolution after patch embedding
        # Need integer division
        patch_resolution: Tuple[int, int] = (input_resolution[0] // patch_size,
                                             input_resolution[1] // patch_size)
        self.patch_resolution = patch_resolution

        # --- Stochastic Depth ---
        # Calculate dropout path rates for each block, increasing linearly across stages/blocks
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, dropout_path, total_blocks)]  # stochastic depth decay rule

        # --- Build Swin Transformer Stages ---
        self.stages: nn.ModuleList = nn.ModuleList()
        current_input_resolution = patch_resolution
        current_in_channels = embedding_channels

        for i_layer in range(self.num_layers):
            stage = SwinTransformerStage(
                in_channels=current_in_channels,
                depth=depths[i_layer],
                # Downscale starting from the *second* stage (index 1)
                downscale=(i_layer > 0),
                input_resolution=current_input_resolution,
                number_of_heads=number_of_heads[i_layer],
                window_size=window_size,
                ff_feature_ratio=ff_feature_ratio,
                dropout=dropout,
                dropout_attention=dropout_attention,
                # Slice the dropout path rates for the current stage
                dropout_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                sequential_self_attention=sequential_self_attention,
                # Use deformable block based on the flag for this stage
                use_deformable_block=use_deformable_block_stages[i_layer],
                post_norm=post_norm # Pass post_norm setting to stage
            )
            self.stages.append(stage)

            # Update channels and resolution for the next stage
            if i_layer > 0: # Downscaling happens from stage 1 onwards
                current_input_resolution = (current_input_resolution[0] // 2, current_input_resolution[1] // 2)
                current_in_channels = current_in_channels * 2
            # Ensure the stage's internal resolution is consistent (already handled in SwinTransformerStage init)


        # --- Final Normalization Layer (Optional, often used before head) ---
        # Timm models often have a final norm layer applied to the output of the last stage
        self.final_norm = norm_layer(self.num_features)

        # --- Calculate Output Channels (width_list) ---
        # Similar to MobileNetV4, perform a dummy forward pass to get output channels
        try:
            # Create dummy input matching the *initial* expected dimensions (B, C_in, H, W)
            # Use the provided input_resolution
            dummy_input = torch.randn(1, in_channels, input_resolution[0], input_resolution[1])
            # Perform a forward pass (no_grad to save memory/computation)
            # Ensure model is in eval mode if dropout/etc affects shapes (though unlikely)
            self.eval() # Set to eval mode temporarily
            with torch.no_grad():
                # The forward pass returns features from each stage
                dummy_features = self.forward(dummy_input)
            self.train() # Set back to train mode
            # Extract channel dimensions (dim 1 for BCHW format)
            # Features are from the output of each *stage*
            self.width_list = [f.size(1) for f in dummy_features]
        except Exception as e:
            # print(f"Warning: Could not automatically determine width_list due to error during dummy forward pass: {e}")
            # print("Setting width_list to empty list. Check input_resolution and model parameters.")
            # Fallback if the dummy forward pass fails
            self.width_list = []

        # --- Initialize weights ---
        self.apply(self._init_weights)


    def _init_weights(self, m):
        """ Initializes weights for layers (common practice). """
        if isinstance(m, nn.Linear):
            # Timm's trunc_normal_ performs well for transformers
            timm.models.layers.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
             # He initialization for Conv layers
             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             if m.bias is not None:
                 nn.init.constant_(m.bias, 0)


    @torch.jit.ignore # Tell torchscript this isn't part of the main forward path
    def no_weight_decay(self):
        """ Which parameters should not have weight decay applied? (e.g., biases, norms, pos embeds)"""
        no_decay = set()
        for name, param in self.named_parameters():
            if param.ndim < 2 or 'bias' in name or 'norm' in name or 'tau' in name or 'relative_position' in name:
                 # Includes LayerNorm/BatchNorm weights/biases, Linear biases, pos embeds, tau scalar
                 no_decay.add(name)
        return no_decay


    def update_resolution(self, new_window_size: int, new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size and input resolution for the entire network.
        :param new_window_size: (int) New window size for all blocks.
        :param new_input_resolution: (Tuple[int, int]) New input image resolution (H, W).
        """
        # Calculate the new patch resolution based on the new input image resolution
        new_patch_resolution: Tuple[int, int] = (new_input_resolution[0] // self.patch_size,
                                                 new_input_resolution[1] // self.patch_size)
        self.patch_resolution = new_patch_resolution # Update stored patch resolution

        # Update resolution and window size for each stage
        current_input_res = new_patch_resolution
        for i, stage in enumerate(self.stages): # type: int, SwinTransformerStage
            stage.update_resolution(new_window_size=new_window_size,
                                    new_input_resolution=current_input_res)
            # Update the resolution for the next stage's input if it downscales
            if stage.downscale: # Check if the *current* stage performed downscaling
                 current_input_res = (current_input_res[0] // 2, current_input_res[1] // 2)

        # Note: The dummy width_list calculation in __init__ is based on the initial
        # resolution. If the resolution changes dynamically *after* init, width_list
        # will not be updated unless manually recalculated.

    def forward_features(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through patch embedding and stages, returning features from each stage.
        :param input: (torch.Tensor) Input tensor [B, C_in, H, W]
        :return: (List[torch.Tensor]) List of feature maps from each stage [B, C_stage, H_stage, W_stage]
        """
        # 1. Patch Embedding
        # [B, C_in, H, W] -> [B, C_embed, H/p, W/p]
        x: torch.Tensor = self.patch_embedding(input)

        # Store features from each stage output
        stage_features: List[torch.Tensor] = []

        # 2. Forward pass through each stage
        for stage in self.stages:
            x = stage(x)
            stage_features.append(x)

        return stage_features

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Main forward pass. Returns features from each stage.
        :param input: (torch.Tensor) Input tensor [B, C_in, H, W]
        :return: (List[torch.Tensor]) List of feature maps from each stage [B, C_stage, H_stage, W_stage]
        """
        features = self.forward_features(input)
        return features

# --- Model Instantiation Functions (MobileNetV4 Style) ---

def swin_transformer_v2_t(input_resolution: Tuple[int, int] = (224, 224),
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          post_norm: bool = True, # Swin v2 default
                          **kwargs) -> SwinTransformerV2:
    """
    Builds a Tiny Swin Transformer V2 (SwinV2-T)
    Configuration: C=96, Heads=(3, 6, 12, 24), Depths=(2, 2, 6, 2)
    :param input_resolution: (Tuple[int, int]) Input image resolution (H, W).
    :param window_size: (int) Window size for attention.
    :param in_channels: (int) Number of input channels.
    :param use_checkpoint: (bool) Enable gradient checkpointing.
    :param sequential_self_attention: (bool) Enable sequential self-attention.
    :param post_norm: (bool) Use post-normalization (default True for V2).
    :param kwargs: Additional arguments passed to SwinTransformerV2 constructor.
    :return: (SwinTransformerV2) Configured SwinV2-T model.
    """
    model = SwinTransformerV2(input_resolution=input_resolution,
                              window_size=window_size,
                              in_channels=in_channels,
                              use_checkpoint=use_checkpoint,
                              sequential_self_attention=sequential_self_attention,
                              embedding_channels=96,
                              depths=(2, 2, 6, 2),
                              number_of_heads=(3, 6, 12, 24),
                              post_norm=post_norm,
                              **kwargs)
    return model


def swin_transformer_v2_s(input_resolution: Tuple[int, int] = (224, 224),
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          post_norm: bool = True,
                          **kwargs) -> SwinTransformerV2:
    """
    Builds a Small Swin Transformer V2 (SwinV2-S)
    Configuration: C=96, Heads=(3, 6, 12, 24), Depths=(2, 2, 18, 2)
    """
    model = SwinTransformerV2(input_resolution=input_resolution,
                              window_size=window_size,
                              in_channels=in_channels,
                              use_checkpoint=use_checkpoint,
                              sequential_self_attention=sequential_self_attention,
                              embedding_channels=96,
                              depths=(2, 2, 18, 2),
                              number_of_heads=(3, 6, 12, 24),
                              post_norm=post_norm,
                              **kwargs)
    return model


def swin_transformer_v2_b(input_resolution: Tuple[int, int] = (224, 224),
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          post_norm: bool = True,
                          **kwargs) -> SwinTransformerV2:
    """
    Builds a Base Swin Transformer V2 (SwinV2-B)
    Configuration: C=128, Heads=(4, 8, 16, 32), Depths=(2, 2, 18, 2)
    """
    model = SwinTransformerV2(input_resolution=input_resolution,
                              window_size=window_size,
                              in_channels=in_channels,
                              use_checkpoint=use_checkpoint,
                              sequential_self_attention=sequential_self_attention,
                              embedding_channels=128,
                              depths=(2, 2, 18, 2),
                              number_of_heads=(4, 8, 16, 32),
                              post_norm=post_norm,
                              **kwargs)
    return model


def swin_transformer_v2_l(input_resolution: Tuple[int, int] = (224, 224),
                          window_size: int = 7,
                          in_channels: int = 3,
                          use_checkpoint: bool = False,
                          sequential_self_attention: bool = False,
                          post_norm: bool = True,
                          **kwargs) -> SwinTransformerV2:
    """
    Builds a Large Swin Transformer V2 (SwinV2-L)
    Configuration: C=192, Heads=(6, 12, 24, 48), Depths=(2, 2, 18, 2)
    """
    model = SwinTransformerV2(input_resolution=input_resolution,
                              window_size=window_size,
                              in_channels=in_channels,
                              use_checkpoint=use_checkpoint,
                              sequential_self_attention=sequential_self_attention,
                              embedding_channels=192,
                              depths=(2, 2, 18, 2),
                              number_of_heads=(6, 12, 24, 48),
                              post_norm=post_norm,
                              **kwargs)
    return model


def swin_transformer_v2_h(input_resolution: Tuple[int, int] = (224, 224), # Needs larger window/res usually
                          window_size: int = 16, # Example: Adjust window for larger models
                          in_channels: int = 3,
                          use_checkpoint: bool = True, # Often needed for H/G models
                          sequential_self_attention: bool = False,
                          post_norm: bool = True,
                          **kwargs) -> SwinTransformerV2:
    """
    Builds a Huge Swin Transformer V2 (SwinV2-H) - Note: Fictional based on pattern
    Configuration: C=352, Heads=(11, 22, 44, 88), Depths=(2, 2, 18, 2) - Example config
    Requires significant memory. Checkpoint=True recommended. May need larger window_size.
    """
    # print("Warning: SwinV2-H configuration is illustrative and may require specific adjustments (e.g., window size, resolution).")
    model = SwinTransformerV2(input_resolution=input_resolution,
                              window_size=window_size,
                              in_channels=in_channels,
                              use_checkpoint=use_checkpoint,
                              sequential_self_attention=sequential_self_attention,
                              embedding_channels=352, # Example C
                              depths=(2, 2, 18, 2), # Example depths
                              number_of_heads=(11, 22, 44, 88), # Example heads
                              post_norm=post_norm,
                              **kwargs)
    return model


def swin_transformer_v2_g(input_resolution: Tuple[int, int] = (224, 224), # Needs larger window/res usually
                          window_size: int = 16, # Example: Adjust window for larger models
                          in_channels: int = 3,
                          use_checkpoint: bool = True, # Often needed for H/G models
                          sequential_self_attention: bool = False,
                          post_norm: bool = True,
                          **kwargs) -> SwinTransformerV2:
    """
    Builds a Giant Swin Transformer V2 (SwinV2-G)
    Configuration: C=512, Heads=(16, 32, 64, 128), Depths=(2, 2, 42, 2) - From paper appendix
    Requires significant memory. Checkpoint=True recommended. May need larger window_size.
    """
    # print("Warning: SwinV2-G requires substantial compute resources.")
    model = SwinTransformerV2(input_resolution=input_resolution,
                              window_size=window_size,
                              in_channels=in_channels,
                              use_checkpoint=use_checkpoint,
                              sequential_self_attention=sequential_self_attention,
                              embedding_channels=512, # From paper
                              depths=(2, 2, 42, 2), # From paper
                              number_of_heads=(16, 32, 64, 128), # From paper
                              post_norm=post_norm,
                              dropout_path=0.2, # G model used higher dropout path in paper
                              **kwargs)
    return model


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Select a model variant
    # model = swin_transformer_v2_t(input_resolution=(224, 224))
    # model = swin_transformer_v2_s(input_resolution=(256, 256), window_size=8)
    model = swin_transformer_v2_b(
        input_resolution=(640, 640),
        window_size=12,
        use_checkpoint=False, # Example: Disable checkpointing
        # Example: Enable deformable blocks in stages 2 and 3
        use_deformable_block_stages=(False, True, True, False),
        post_norm=True # Explicitly use V2 style post-norm
    )
    # model = swin_transformer_v2_g(input_resolution=(256, 256), window_size=16, use_checkpoint=True)


    # 2. Create a dummy input tensor
    # Match the input resolution used for model creation
    input_res = (640, 640) # Should match model's input_resolution
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, input_res[0], input_res[1])

    # 3. Perform a forward pass
    model.eval() # Set to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for inference
        features = model(dummy_image)

    # 4. Print output shapes and width_list
    print(f"Model: SwinTransformerV2 (Base with Deformable Example)")
    print(f"Input shape: {dummy_image.shape}")
    print("Output features per stage:")
    for i, f in enumerate(features):
        print(f"  Stage {i}: {f.shape}")

    print(f"\nCalculated width_list (channels per stage output): {model.width_list}")

    # 5. Example: Update resolution dynamically (if needed)
    # new_res = (512, 512)
    # new_window = 16
    # print(f"\nUpdating resolution to {new_res} and window size to {new_window}...")
    # model.update_resolution(new_window_size=new_window, new_input_resolution=new_res)
    # dummy_image_new_res = torch.randn(batch_size, 3, new_res[0], new_res[1])
    # with torch.no_grad():
    #      features_new = model(dummy_image_new_res)
    # print("Output features after resolution update:")
    # for i, f in enumerate(features_new):
    #     print(f"  Stage {i}: {f.shape}")
    # print(f"(Note: width_list {model.width_list} is from initializaiton, not updated dynamically)")
