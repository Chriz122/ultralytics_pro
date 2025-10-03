# -*- coding: utf-8 -*-
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# Adapted from Evo-ViT (https://github.com/YifanXu74/Evo-ViT) and UniFormer (https://github.com/Sense-X/UniFormer)

import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import _cfg # Assumes timm is installed
from timm.models.layers import trunc_normal_, DropPath # Assumes timm is installed
from torch.nn.modules.batchnorm import _BatchNorm
import warnings # Use warnings for checkpoint loading feedback

# --- Global Configuration (Consider passing these as args in a real application) ---
layer_scale = False
init_value = 1e-6
# --- Global Variables used during forward pass (Reset per forward call) ---
global_attn = None
token_indices = None

# -------------------- Helper Functions (from Evo-ViT) --------------------

def easy_gather(x, indices):
    # x => B x N x C
    # indices => B x N
    B, N, C = x.shape
    N_new = indices.shape[1]
    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
    indices = indices + offset
    # only select the informative tokens
    out = x.reshape(B * N, C)[indices.view(-1)].reshape(B, N_new, C)
    return out

def merge_tokens(x_drop, score):
    # x_drop => B x N_drop x C
    # score => B x N_drop
    # Normalize score to create weights
    score_sum = torch.sum(score, dim=1, keepdim=True)
    # Handle potential division by zero if all scores are zero for a batch element
    weight = score / torch.clamp(score_sum, min=1e-6) # Add epsilon for stability
    x_drop = weight.unsqueeze(-1) * x_drop
    return torch.sum(x_drop, dim=1, keepdim=True)

# -------------------- Core Building Blocks --------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., trade_off=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.trade_off = trade_off # updating weight for global score

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # update global score (using global variable, needs care)
        global global_attn
        tradeoff = self.trade_off
        if isinstance(global_attn, int): # Initialize global_attn in the first relevant block
            # Exclude cls token [0] and potentially representative token [-1] if present later
            if N > 1: # Ensure there are tokens other than cls/rep
                 global_attn = torch.mean(attn[:, :, 0, 1:], dim=1) # B, N-1
            else:
                 # Handle case with only cls token (shouldn't happen with patch tokens)
                 global_attn = torch.zeros((B, 0), device=x.device, dtype=x.dtype)

        elif global_attn is not None and N > 1:
            current_num_patch_tokens = N - 1 # Assuming only cls token added initially
            target_num_patch_tokens_in_attn = attn.shape[-1] - 1 # Tokens attn is calculated over (excl. cls)
            num_tokens_to_update = min(global_attn.shape[1], target_num_patch_tokens_in_attn) # Avoid index errors

            # Calculate cls_attn over relevant tokens (exclude cls token [0], exclude potential rep token [-1])
            if N > 2 and global_attn.shape[1] > num_tokens_to_update : # Has rep token merged in
                cls_attn = torch.mean(attn[:, :, 0, 1:-1], dim=1) # B, N-2
                num_tokens_to_update = N - 2 # Only update non-rep tokens
            else: # No rep token merged yet or only cls + patches
                cls_attn = torch.mean(attn[:, :, 0, 1:], dim=1) # B, N-1

            # Ensure cls_attn has the correct shape for update
            cls_attn = cls_attn[:, :num_tokens_to_update]

            if self.training:
                 # Update existing scores, potentially keeping others if pruning happened
                 temp_attn = (1 - tradeoff) * global_attn[:, :num_tokens_to_update] + tradeoff * cls_attn
                 if global_attn.shape[1] > num_tokens_to_update: # Keep non-updated scores (e.g., from pruned stage)
                     global_attn = torch.cat((temp_attn, global_attn[:, num_tokens_to_update:]), dim=1)
                 else:
                     global_attn = temp_attn
            else:
                 # Inference: In-place update for speed
                 global_attn[:, :num_tokens_to_update] = (1 - tradeoff) * global_attn[:, :num_tokens_to_update] + tradeoff * cls_attn
        # else: global_attn is None or N <= 1, no update needed

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d): # Use BatchNorm for CBlock
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim) # Simplified attention-like op
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            # print(f"CBlock Use layer_scale: {layer_scale}, init_values: {init_value}") # Optional print
            self.gamma_1 = nn.Parameter(init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = x + self.pos_embed(x)
        if self.ls:
            x = x + self.drop_path(self.gamma_1 * self.conv2(self.attn(self.conv1(self.norm1(x)))))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class EvoSABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, prune_ratio=1.0, # Default to no pruning
                 trade_off=0.5, downsample=False): # Default trade_off
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, trade_off=trade_off)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.prune_ratio = prune_ratio
        self.downsample = downsample
        if downsample:
            # Use adaptive avg pool for flexibility if H, W are not always known/even
            # self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((None, None)) # Placeholder, spatial dims set in forward

        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            # print(f"EvoSABlock Use layer_scale: {layer_scale}, init_values: {init_value}") # Optional print
            self.gamma_1 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
            if self.prune_ratio != 1.0: # Only need gamma_3 if pruning
                self.gamma_3 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)

    def forward(self, cls_token, x):
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape
        x_patch = x.flatten(2).transpose(1, 2) # B, N, C (where N = H*W)

        if self.prune_ratio == 1.0: # No pruning/merging
            x_combined = torch.cat([cls_token, x_patch], dim=1)
            if self.ls:
                x_combined = x_combined + self.drop_path(self.gamma_1 * self.attn(self.norm1(x_combined)))
                x_combined = x_combined + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_combined)))
            else:
                x_combined = x_combined + self.drop_path(self.attn(self.norm1(x_combined)))
                x_combined = x_combined + self.drop_path(self.mlp(self.norm2(x_combined)))

            cls_token_out, x_patch_out = x_combined[:, :1], x_combined[:, 1:]
            x_out = x_patch_out.transpose(1, 2).reshape(B, C, H, W)

            # Handle downsampling for the *next* stage if needed
            if self.downsample:
                global global_attn
                if global_attn is not None and isinstance(global_attn, torch.Tensor):
                     # Downsample global attention map (assuming it corresponds to x_patch)
                     H_attn, W_attn = H, W # Dimensions matching global_attn before pooling
                     global_attn_map = global_attn.reshape(B, 1, H_attn, W_attn)
                     # Set target size for AdaptiveAvgPool2d based on input H, W halved
                     target_H, target_W = H // 2, W // 2
                     self.avgpool.output_size = (target_H, target_W)
                     global_attn_pooled = self.avgpool(global_attn_map).view(B, -1)

                     # Normalize pooled attention
                     old_global_scale = torch.sum(global_attn, dim=1, keepdim=True)
                     new_global_scale = torch.sum(global_attn_pooled, dim=1, keepdim=True)
                     # Avoid division by zero
                     scale = old_global_scale / torch.clamp(new_global_scale, min=1e-6)
                     global_attn = global_attn_pooled * scale

            return cls_token_out, x_out

        else: # Pruning/merging path
            global token_indices
            if global_attn is None or token_indices is None:
                 # Should not happen if global_attn is initialized correctly before first EvoSABlock
                 # Fallback to no pruning for safety, though this indicates an issue
                 warnings.warn("global_attn or token_indices not initialized before pruning block. Skipping pruning.")
                 # Fallback to the non-pruning path logic:
                 x_combined = torch.cat([cls_token, x_patch], dim=1)
                 if self.ls:
                    x_combined = x_combined + self.drop_path(self.gamma_1 * self.attn(self.norm1(x_combined)))
                    x_combined = x_combined + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_combined)))
                 else:
                    x_combined = x_combined + self.drop_path(self.attn(self.norm1(x_combined)))
                    x_combined = x_combined + self.drop_path(self.mlp(self.norm2(x_combined)))
                 cls_token_out, x_patch_out = x_combined[:, :1], x_combined[:, 1:]
                 x_out = x_patch_out.transpose(1, 2).reshape(B, C, H, W)
                 # Still handle potential downsampling even in fallback
                 if self.downsample and isinstance(global_attn, torch.Tensor):
                     # Downsample global attention map (assuming it corresponds to x_patch)
                     H_attn, W_attn = H, W # Dimensions matching global_attn before pooling
                     global_attn_map = global_attn.reshape(B, 1, H_attn, W_attn)
                     target_H, target_W = H // 2, W // 2
                     self.avgpool.output_size = (target_H, target_W)
                     global_attn_pooled = self.avgpool(global_attn_map).view(B, -1)
                     old_global_scale = torch.sum(global_attn, dim=1, keepdim=True)
                     new_global_scale = torch.sum(global_attn_pooled, dim=1, keepdim=True)
                     scale = old_global_scale / torch.clamp(new_global_scale, min=1e-6)
                     global_attn = global_attn_pooled * scale
                 return cls_token_out, x_out


            N_patch = x_patch.shape[1]
            N_keep = int(N_patch * self.prune_ratio)
            N_drop = N_patch - N_keep

            if N_keep == 0 or N_patch == 0: # Cannot prune further
                 # Essentially acts like prune_ratio=1.0 block from here
                 x_combined = torch.cat([cls_token, x_patch], dim=1)
                 # Apply Attention and MLP as usual
                 if self.ls:
                     x_combined = x_combined + self.drop_path(self.gamma_1 * self.attn(self.norm1(x_combined)))
                     x_combined = x_combined + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_combined)))
                 else:
                     x_combined = x_combined + self.drop_path(self.attn(self.norm1(x_combined)))
                     x_combined = x_combined + self.drop_path(self.mlp(self.norm2(x_combined)))
                 cls_token_out, x_patch_out = x_combined[:, :1], x_combined[:, 1:]
                 x_out = x_patch_out.transpose(1, 2).reshape(B, C, H, W)
                 # Handle downsampling for the *next* stage if needed
                 if self.downsample and isinstance(global_attn, torch.Tensor):
                     H_attn, W_attn = H, W
                     global_attn_map = global_attn.reshape(B, 1, H_attn, W_attn)
                     target_H, target_W = H // 2, W // 2
                     self.avgpool.output_size = (target_H, target_W)
                     global_attn_pooled = self.avgpool(global_attn_map).view(B, -1)
                     old_global_scale = torch.sum(global_attn, dim=1, keepdim=True)
                     new_global_scale = torch.sum(global_attn_pooled, dim=1, keepdim=True)
                     scale = old_global_scale / torch.clamp(new_global_scale, min=1e-6)
                     global_attn = global_attn_pooled * scale
                 return cls_token_out, x_out


            # Sort global attention scores to find tokens to keep/drop
            # Ensure global_attn corresponds to the *patch tokens* only
            if global_attn.shape[1] != N_patch:
                warnings.warn(f"Mismatch between global_attn size ({global_attn.shape[1]}) and N_patch ({N_patch}). Skipping pruning.")
                # Fallback to no pruning logic (copy from above)
                x_combined = torch.cat([cls_token, x_patch], dim=1)
                if self.ls:
                    x_combined = x_combined + self.drop_path(self.gamma_1 * self.attn(self.norm1(x_combined)))
                    x_combined = x_combined + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_combined)))
                else:
                    x_combined = x_combined + self.drop_path(self.attn(self.norm1(x_combined)))
                    x_combined = x_combined + self.drop_path(self.mlp(self.norm2(x_combined)))
                cls_token_out, x_patch_out = x_combined[:, :1], x_combined[:, 1:]
                x_out = x_patch_out.transpose(1, 2).reshape(B, C, H, W)
                if self.downsample and isinstance(global_attn, torch.Tensor):
                     H_attn, W_attn = H, W
                     global_attn_map = global_attn.reshape(B, 1, H_attn, W_attn)
                     target_H, target_W = H // 2, W // 2
                     self.avgpool.output_size = (target_H, target_W)
                     global_attn_pooled = self.avgpool(global_attn_map).view(B, -1)
                     old_global_scale = torch.sum(global_attn, dim=1, keepdim=True)
                     new_global_scale = torch.sum(global_attn_pooled, dim=1, keepdim=True)
                     scale = old_global_scale / torch.clamp(new_global_scale, min=1e-6)
                     global_attn = global_attn_pooled * scale
                return cls_token_out, x_out

            # Proceed with pruning
            indices_sort = torch.argsort(global_attn, dim=1, descending=True) # B, N_patch

            # Gather tokens based on sorted indices
            # Combine patch tokens, their global attn scores, and original indices for sorting/unsorting
            x_patch_ga_ti = torch.cat((x_patch, global_attn.unsqueeze(-1), token_indices.unsqueeze(-1)), dim=-1)
            x_patch_ga_ti_sorted = easy_gather(x_patch_ga_ti, indices_sort)

            x_patch_sorted = x_patch_ga_ti_sorted[..., :-2]     # B, N_patch, C
            global_attn_sorted = x_patch_ga_ti_sorted[..., -2]  # B, N_patch
            token_indices_sorted = x_patch_ga_ti_sorted[..., -1] # B, N_patch

            # Keep top N_keep tokens
            x_info = x_patch_sorted[:, :N_keep]         # B, N_keep, C
            global_attn_info = global_attn_sorted[:, :N_keep] # B, N_keep
            token_indices_info = token_indices_sorted[:, :N_keep] # B, N_keep

            # Tokens to be dropped/merged
            x_drop = x_patch_sorted[:, N_keep:]         # B, N_drop, C
            global_attn_drop = global_attn_sorted[:, N_keep:] # B, N_drop
            token_indices_drop = token_indices_sorted[:, N_keep:] # B, N_drop

            # Merge dropped tokens into a single representative token
            rep_token = merge_tokens(x_drop, global_attn_drop) # B, 1, C

            # Concatenate class token, kept tokens, and representative token for Attention/MLP
            x_attn_mlp_input = torch.cat((cls_token, x_info, rep_token), dim=1) # B, 1+N_keep+1, C

            # === Apply Attention and MLP on the combined set ===
            if self.ls:
                tmp_x = self.attn(self.norm1(x_attn_mlp_input))
                # The "fast update" in original code seems to apply the update from the rep_token
                # back to the dropped tokens. Let's capture the update related to the rep_token.
                fast_update_delta = tmp_x[:, -1:] # B, 1, C (update corresponding to rep_token)
                x_attn_mlp_input = x_attn_mlp_input + self.drop_path(self.gamma_1 * tmp_x)

                tmp_x = self.mlp(self.norm2(x_attn_mlp_input))
                fast_update_delta = fast_update_delta + tmp_x[:, -1:] # Accumulate update
                x_processed = x_attn_mlp_input + self.drop_path(self.gamma_2 * tmp_x)

                # Apply the "fast update" back to the original dropped tokens
                # Expand delta and add to x_drop, scaled by gamma_3
                x_drop_updated = x_drop + self.gamma_3 * fast_update_delta.expand(-1, N_drop, -1)
            else:
                tmp_x = self.attn(self.norm1(x_attn_mlp_input))
                fast_update_delta = tmp_x[:, -1:]
                x_attn_mlp_input = x_attn_mlp_input + self.drop_path(tmp_x)

                tmp_x = self.mlp(self.norm2(x_attn_mlp_input))
                fast_update_delta = fast_update_delta + tmp_x[:, -1:]
                x_processed = x_attn_mlp_input + self.drop_path(tmp_x)

                # Apply the "fast update" (unscaled)
                x_drop_updated = x_drop + fast_update_delta.expand(-1, N_drop, -1)
            # ===================================================

            # Separate class token and processed patch tokens (kept + rep)
            cls_token_out = x_processed[:, :1]         # B, 1, C
            x_info_processed = x_processed[:, 1:-1]    # B, N_keep, C
            # rep_token_processed = x_processed[:, -1:] # B, 1, C (don't need this directly)

            # Reconstruct the full set of patch tokens in the *sorted* order
            # (Updated kept tokens + updated dropped tokens)
            x_patch_sorted_updated = torch.cat((x_info_processed, x_drop_updated), dim=1) # B, N_patch, C

            # Reconstruct the global attention scores and token indices in the *sorted* order
            # The attn scores for dropped tokens don't change, keep attn for kept tokens
            global_attn_sorted_updated = torch.cat((global_attn_info, global_attn_drop), dim=1) # B, N_patch
            token_indices_sorted_updated = torch.cat((token_indices_info, token_indices_drop), dim=1) # B, N_patch

            # === Recover original token order ===
            indices_unsort = torch.argsort(token_indices_sorted_updated, dim=1) # B, N_patch

            # Combine updated patches, their attn scores, and original indices for unsorting
            x_patch_ga_ti_updated_sorted = torch.cat(
                (x_patch_sorted_updated, global_attn_sorted_updated.unsqueeze(-1), token_indices_sorted_updated.unsqueeze(-1)),
                dim=-1
            )
            x_patch_ga_ti_unsorted = easy_gather(x_patch_ga_ti_updated_sorted, indices_unsort)

            x_patch_out = x_patch_ga_ti_unsorted[..., :-2]      # B, N_patch, C
            global_attn_out = x_patch_ga_ti_unsorted[..., -2]   # B, N_patch
            token_indices_out = x_patch_ga_ti_unsorted[..., -1] # B, N_patch (should be 0, 1, ..., N_patch-1)

            # Reshape patch tokens back to image format
            x_out = x_patch_out.transpose(1, 2).reshape(B, C, H, W)

            # Update global state for the next layer
            global_attn = global_attn_out
            token_indices = token_indices_out.long() # Ensure it's long dtype

            # === Handle Downsampling (if required) ===
            if self.downsample:
                # Downsample the *updated* global attention map before returning
                H_attn, W_attn = H, W # Dimensions matching global_attn_out
                global_attn_map = global_attn_out.reshape(B, 1, H_attn, W_attn)

                # Set target size for AdaptiveAvgPool2d based on input H, W halved
                target_H, target_W = H // 2, W // 2
                self.avgpool.output_size = (target_H, target_W)
                global_attn_pooled = self.avgpool(global_attn_map).view(B, -1) # B, N_patch/4

                # Normalize pooled attention
                old_global_scale = torch.sum(global_attn_out, dim=1, keepdim=True)
                new_global_scale = torch.sum(global_attn_pooled, dim=1, keepdim=True)
                scale = old_global_scale / torch.clamp(new_global_scale, min=1e-6) # Avoid division by zero
                global_attn = global_attn_pooled * scale # Update global_attn to the pooled version

                # Update token_indices for the downsampled grid (this part was missing in original?)
                # If downsampling happens, the indices need to be regenerated for the smaller grid
                # Assuming standard pooling, indices are reset for the new grid size
                # Note: This might deviate from original Evo-ViT if they had a specific index mapping strategy
                new_N_patch = target_H * target_W
                token_indices = torch.arange(new_N_patch, dtype=torch.long, device=x_out.device).unsqueeze(0).expand(B, -1)

            return cls_token_out, x_out


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding with Norm """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # B, H*W, C
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W) # B, C, H, W
        return x

class ConvStemEmbed(nn.Module):
    """ Convolutional Stem """
    def __init__(self, in_chans=3, embed_dim=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # Using two conv layers with stride 2 each, like MobileNetV4 conv0 and layer1 approx.
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(embed_dim // 2),
            nn.GELU(), # Changed to GELU like ViT
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(embed_dim),
            # Removed final GELU here, typically applied after block in ViT style
        )
        # This stem results in 4x downsampling

    def forward(self, x):
        x = self.proj(x)
        return x

class DownsampleEmbed(nn.Module):
    """ Downsampling Embedding Layer (Conv2d) """
    def __init__(self, in_embed_dim, out_embed_dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = norm_layer(out_embed_dim)
        # Removed activation, usually applied after block

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


# -------------------- Main Model Definition --------------------

class UniFormer_Light(nn.Module):
    """ UniFormer Light Backbone (Standalone) """
    def __init__(self, depth=[3, 4, 8, 3], in_chans=3, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=[4., 4., 4., 4.], qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), conv_stem=False,
                 prune_ratio=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], # Example prune ratios
                 trade_off=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],    # Example trade-offs
                 norm_eval=False, pretrained_path=None):
        super().__init__()
        self.norm_eval = norm_eval
        self.num_stages = len(depth)
        assert len(embed_dim) == self.num_stages
        assert len(mlp_ratio) == self.num_stages
        assert len(prune_ratio) == self.num_stages
        assert len(trade_off) == self.num_stages

        # Choose Norm layers based on block type (rough heuristic)
        # Stages 1 & 2 (CBlocks) often use BatchNorm
        # Stages 3 & 4 (EvoSABlocks) often use LayerNorm
        conv_norm_layer = partial(nn.BatchNorm2d, eps=1e-5, momentum=0.1) # Common BN defaults
        sa_norm_layer = norm_layer # Use provided norm_layer for SA blocks

        # Patch/Stem Embedding
        if conv_stem:
            self.patch_embed1 = ConvStemEmbed(in_chans=in_chans, embed_dim=embed_dim[0], norm_layer=conv_norm_layer)
        else:
            self.patch_embed1 = PatchEmbed(patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0], norm_layer=sa_norm_layer)

        # Subsequent Downsampling Embeddings
        self.patch_embed2 = DownsampleEmbed(embed_dim[0], embed_dim[1], norm_layer=conv_norm_layer)
        self.patch_embed3 = DownsampleEmbed(embed_dim[1], embed_dim[2], norm_layer=conv_norm_layer)
        self.patch_embed4 = DownsampleEmbed(embed_dim[2], embed_dim[3], norm_layer=conv_norm_layer)

        # Class token (only used in stages 3 & 4)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim[2]))
        self.cls_upsample = nn.Linear(embed_dim[2], embed_dim[3]) # To match dim in stage 4
        trunc_normal_(self.cls_token, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))] # Stochastic depth decay rule

        num_heads = [dim // head_dim for dim in embed_dim]

        # --- Stage 1 (Convolutional Blocks) ---
        cur = 0
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=conv_norm_layer)
            for i in range(depth[0])])
        self.norm1 = conv_norm_layer(embed_dim[0]) # Norm after stage 1 blocks
        cur += depth[0]

        # --- Stage 2 (Convolutional Blocks) ---
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=conv_norm_layer)
            for i in range(depth[1])])
        self.norm2 = conv_norm_layer(embed_dim[1]) # Norm after stage 2 blocks
        cur += depth[1]

        # --- Stage 3 (Evo Self-Attention Blocks) ---
        assert len(prune_ratio[2]) == depth[2], f"Prune ratio list length mismatch for stage 3 ({len(prune_ratio[2])} vs {depth[2]})"
        assert len(trade_off[2]) == depth[2], f"Trade off list length mismatch for stage 3 ({len(trade_off[2])} vs {depth[2]})"
        self.blocks3 = nn.ModuleList([
            EvoSABlock(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=sa_norm_layer,
                prune_ratio=prune_ratio[2][i], trade_off=trade_off[2][i],
                downsample=(i == depth[2] - 1) # Downsample attention map *after* the last block of stage 3
                )
            for i in range(depth[2])])
        self.norm3 = sa_norm_layer(embed_dim[2]) # Norm after stage 3 blocks
        cur += depth[2]

        # --- Stage 4 (Evo Self-Attention Blocks) ---
        assert len(prune_ratio[3]) == depth[3], f"Prune ratio list length mismatch for stage 4 ({len(prune_ratio[3])} vs {depth[3]})"
        assert len(trade_off[3]) == depth[3], f"Trade off list length mismatch for stage 4 ({len(trade_off[3])} vs {depth[3]})"
        self.blocks4 = nn.ModuleList([
            EvoSABlock(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=sa_norm_layer,
                prune_ratio=prune_ratio[3][i], trade_off=trade_off[3][i],
                downsample=False # No attention downsampling after last stage
                )
            for i in range(depth[3])])
        self.norm4 = sa_norm_layer(embed_dim[3]) # Norm after stage 4 blocks

        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained_path)

        # Calculate output feature dimensions like MobileNetV4
        self.eval() # Ensure model is in eval mode for consistent output during init
        with torch.no_grad():
            # Use a typical input size, e.g., 224x224 for ImageNet, or the requested 640x640
            # Use the requested 640x640 size
            dummy_input = torch.randn(1, 3, 640, 640)
            features = self.forward(dummy_input)
            self.width_list = [f.shape[1] for f in features] # Get channel dim (C in B, C, H, W)
        self.train() # Return model to train mode

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
             # He initialization for Conv layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # or 'gelu' if used
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            try:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

                # Adjust state dict keys if necessary (e.g., remove 'backbone.' prefix)
                if list(state_dict.keys())[0].startswith('backbone.'):
                     state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
                # Adjust for potential 'module.' prefix from DataParallel/DDP
                if list(state_dict.keys())[0].startswith('module.'):
                     state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                # --- Specific key adjustments if loading ImageNet weights ---
                # Example: Rename patch_embed keys if structure differs
                # Example: Ignore classifier head weights ('head.weight', 'head.bias')
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}

                # Load the weights
                missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

                if missing_keys:
                    warnings.warn(f"Missing keys when loading pretrained weights: {missing_keys}")
                if unexpected_keys:
                    warnings.warn(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
                print(f"Successfully loaded pretrained weights from {pretrained}")

            except Exception as e:
                warnings.warn(f"Could not load pretrained weights from {pretrained}. Error: {e}")
        elif pretrained is None:
            # print("No pretrained weights provided, initializing from scratch.")
            # self.apply(self._init_weights) # Already applied in __init__
            pass # Weights are initialized in __init__
        else:
            raise TypeError('pretrained must be a str or None')


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        global global_attn, token_indices # Use the global variables

        outs = [] # Initialize outs as a list
        # Stage 1
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x_norm = self.norm1(x)
        outs.append(x_norm) # Append to list

        # Stage 2
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x_norm = self.norm2(x)
        outs.append(x_norm) # Append to list

        # Stage 3
        x = self.patch_embed3(x)
        B, C, H, W = x.shape
        cls_token = self.cls_token.expand(B, -1, -1)
        global_attn = 0
        token_indices = torch.arange(H * W, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, -1)

        for i, blk in enumerate(self.blocks3):
            cls_token, x = blk(cls_token, x)

        x_patch = x.flatten(2).transpose(1, 2)
        x_patch_norm = self.norm3(x_patch)
        _, _, H_out3, W_out3 = x.shape
        x_norm = x_patch_norm.transpose(1, 2).view(B, C, H_out3, W_out3)
        outs.append(x_norm) # Append to list

        # Stage 4
        x = self.patch_embed4(x)
        cls_token = self.cls_upsample(cls_token)
        B, C, H, W = x.shape
        token_indices = torch.arange(H * W, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, -1)

        for i, blk in enumerate(self.blocks4):
             cls_token, x = blk(cls_token, x)

        x_patch = x.flatten(2).transpose(1, 2)
        x_patch_norm = self.norm4(x_patch)
        _, _, H_out4, W_out4 = x.shape
        x_norm = x_patch_norm.transpose(1, 2).view(B, C, H_out4, W_out4)
        outs.append(x_norm) # Append to list

        # ---- Change is here ----
        # Return the list directly instead of converting to tuple
        return outs
        # ---- End of Change ----

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Override train modes to potentially keep norm layers in eval mode."""
        super(UniFormer_Light, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # Trick: eval() has effect on BatchNorm and Dropout
                if isinstance(m, (_BatchNorm)): # Keep BatchNorm in eval
                    m.eval()
                # Optionally keep Dropout in eval as well if needed
                # if isinstance(m, nn.Dropout):
                #     m.eval()

# -------------------- Model Instantiation Functions (like MobileNetV4) --------------------

def uniformer_light_xxs(pretrained=False, pretrained_path=None, **kwargs):
    """ UniFormer-Light XXS variant """
    # Define prune ratios and trade-offs specific to XXS
    # These values are examples, adjust based on actual XXS config if available
    prune_ratio_xxs = [[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5]] # 8 blocks stage3, 2 blocks stage4
    trade_off_xxs   = [[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5]] # 8 blocks stage3, 2 blocks stage4
    depth_xxs       = [2, 5, 8, 2] # Example depths for XXS
    embed_dim_xxs   = [56, 112, 224, 448] # Example dims for XXS
    head_dim_xxs    = 28 # Example head dim

    model = UniFormer_Light(
        depth=depth_xxs, conv_stem=True, # Typically smaller models use conv stem
        prune_ratio=prune_ratio_xxs,
        trade_off=trade_off_xxs,
        embed_dim=embed_dim_xxs, head_dim=head_dim_xxs, mlp_ratio=[3, 3, 3, 3], qkv_bias=True,
        pretrained_path=pretrained_path if pretrained else None,
        **kwargs)
    model.default_cfg = _cfg() # Keep timm config if useful
    return model


def uniformer_light_xs(pretrained=False, pretrained_path=None, **kwargs):
    """ UniFormer-Light XS variant """
    # Define prune ratios and trade-offs specific to XS
    prune_ratio_xs = [[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]] # 9 blocks stage3, 3 blocks stage4
    trade_off_xs   = [[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]] # 9 blocks stage3, 3 blocks stage4
    depth_xs       = [3, 5, 9, 3]
    embed_dim_xs   = [64, 128, 256, 512]
    head_dim_xs    = 32

    model = UniFormer_Light(
        depth=depth_xs, conv_stem=True, # Typically smaller models use conv stem
        prune_ratio=prune_ratio_xs,
        trade_off=trade_off_xs,
        embed_dim=embed_dim_xs, head_dim=head_dim_xs, mlp_ratio=[3, 3, 3, 3], qkv_bias=True,
        pretrained_path=pretrained_path if pretrained else None,
        **kwargs)
    model.default_cfg = _cfg() # Keep timm config if useful
    return model

# Add more variants like uniformer_light_s, uniformer_light_b if needed, following the pattern

# -------------------- Example Usage --------------------
if __name__ == "__main__":
    # Set global config (optional, can be passed to __init__)
    layer_scale = False
    init_value = 1e-6

    print("--- Testing UniFormer-Light XXS ---")
    # Instantiate using the helper function
    model_xxs = uniformer_light_xxs(
        drop_rate=0.0,       # Example override
        drop_path_rate=0.1, # Example stochastic depth rate
        norm_eval=False      # Example norm eval setting
    )
    model_xxs.eval() # Set to eval mode for inference testing

    # Create a dummy input tensor matching the size used for width_list calculation
    input_size = (1, 3, 640, 640)
    dummy_input = torch.randn(*input_size)

    # Perform inference
    with torch.no_grad():
        output_features = model_xxs(dummy_input)

    # Print output shapes and calculated width_list
    print(f"Input shape: {dummy_input.shape}")
    print("Output feature shapes:")
    for i, feat in enumerate(output_features):
        print(f"  Stage {i+1}: {feat.shape}")

    print(f"Calculated width_list: {model_xxs.width_list}")
    print("-" * 30)


    print("--- Testing UniFormer-Light XS ---")
    model_xs = uniformer_light_xs()
    model_xs.eval()

    with torch.no_grad():
        output_features_xs = model_xs(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print("Output feature shapes:")
    for i, feat in enumerate(output_features_xs):
        print(f"  Stage {i+1}: {feat.shape}")

    print(f"Calculated width_list: {model_xs.width_list}")
    print("-" * 30)

    # Example: Test with a different input size
    print("--- Testing UniFormer-Light XS with 224x224 input ---")
    dummy_input_224 = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output_features_224 = model_xs(dummy_input_224)
    print(f"Input shape: {dummy_input_224.shape}")
    print("Output feature shapes:")
    for i, feat in enumerate(output_features_224):
        print(f"  Stage {i+1}: {feat.shape}")
    print("-" * 30)

    # Example: Loading non-existent pretrained weights (will trigger warning)
    # print("--- Testing Pretrained Loading (Warning Expected) ---")
    # model_xxs_pt = uniformer_light_xxs(pretrained=True, pretrained_path="non_existent_file.pth")
    # print("-" * 30)