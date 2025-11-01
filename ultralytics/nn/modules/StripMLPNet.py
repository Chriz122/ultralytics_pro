# --------------------------------------------------------
# Copyright (c) 2023 CVIP of SUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Guiping Cao
# Modified by AI based on SwinTransformer structure
# --------------------------------------------------------

from ast import Pass
# from matplotlib.pyplot import axis # Not used
# from numpy import append # Not used
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import to_2tuple, trunc_normal_, DropPath

class BN_Activ_Conv(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride=(1, 1), dilation=(1, 1), groups=1):
        super(BN_Activ_Conv, self).__init__()
        # For BN -> Act -> Conv pattern, BN operates on its input channels
        self.BN = nn.BatchNorm2d(in_channels)
        self.Activation = activation
        padding = [int((dilation[j] * (kernel_size[j] - 1) - stride[j] + 1) / 2) for j in range(2)]
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=False)

    def forward(self, img):
        img = self.BN(img)
        img = self.Activation(img)
        img = self.Conv(img)
        return img

class DepthWise_Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_merge = BN_Activ_Conv(channels, nn.GELU(), channels, (3, 3), groups=channels)

    def forward(self, img):
        img = self.conv_merge(img)
        return img

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        trunc_normal_(self.embeddings_table, std=.02)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q, device=self.embeddings_table.device)
        range_vec_k = torch.arange(length_k, device=self.embeddings_table.device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat.to(self.embeddings_table.device)]
        return embeddings

class StripMLP_Block(nn.Module):
    def __init__(self, channels, H, W):
        super().__init__()
        self.channels = channels # Total channels for this block
        self.activation = nn.GELU()
        
        # This BN is applied after fuse_h, which outputs channels//2
        self.BN_x_h_path = nn.BatchNorm2d(channels // 2) # CORRECTED: For the x_h path, after fuse_h

        if channels % 80 == 0:
            patch_token_divider = 2
        else:
            patch_token_divider = 4

        self.C_internal_base = int(channels * 0.5 / patch_token_divider)
        self.ratio = 1
        self.chan_proj = self.ratio * self.C_internal_base

        self.proj_h_in_channels = H * self.C_internal_base
        self.proj_h_out_channels = self.chan_proj * H
        self.proj_h = nn.Conv2d(self.proj_h_in_channels, self.proj_h_out_channels, (1, 3), stride=1, padding=(0, 1), groups=self.C_internal_base, bias=True)

        self.proj_w_in_channels = W * self.C_internal_base
        self.proj_w_out_channels = self.chan_proj * W
        self.proj_w = nn.Conv2d(self.proj_w_in_channels, self.proj_w_out_channels, (1, 3), stride=1, padding=(0, 1), groups=self.C_internal_base, bias=True)

        # fuse_h input is cat([x_h_intermediate (C//2), x_strip (C//2)]) -> C channels total. Output C//2.
        self.fuse_h = nn.Conv2d(channels, channels // 2, (1, 1), (1, 1), bias=False)
        # fuse_w input is cat([x_strip (C//2), x_w_intermediate (C//2)]) -> C channels total. Output C//2.
        self.fuse_w = nn.Conv2d(channels, channels // 2, (1, 1), (1, 1), bias=False)

        self.mlp_pre = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, bias=True),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        dim_half = channels // 2
        self.fc_h = nn.Conv2d(dim_half, dim_half, (3, 7), stride=1, padding=(1, 7 // 2), groups=dim_half, bias=False)
        self.fc_w = nn.Conv2d(dim_half, dim_half, (7, 3), stride=1, padding=(7 // 2, 1), groups=dim_half, bias=False)
        self.reweight = Mlp(dim_half, dim_half // 4, dim_half * 3)

        self.fuse_post = nn.Conv2d(channels, channels, (1, 1), (1, 1), bias=False)

        self.relate_pos_h_lookup = RelativePosition(dim_half, H)
        self.relate_pos_w_lookup = RelativePosition(dim_half, W) # Original was (dim_half, W), assuming W-dim interaction
        self.H_init, self.W_init = H, W

    def forward(self, x_input_block):
        N, C_orig, H_feat, W_feat = x_input_block.shape
        
        assert H_feat == self.H_init and W_feat == self.W_init, \
            f"Input H,W ({H_feat},{W_feat}) must match H,W at init ({self.H_init},{self.W_init}). " \
            f"Block configured for H={self.H_init}, W={self.W_init}."

        x = self.mlp_pre(x_input_block)

        x_1 = x[:, :C_orig // 2, :, :] # This is x_strip for strip_mlp_path, has C_orig // 2 channels
        x_2 = x[:, C_orig // 2:, :, :] # This is for the other path, has C_orig // 2 channels
        
        x_1_processed = self.strip_mlp_path(x_1, H_feat, W_feat)

        x_w_path = self.fc_h(x_2)
        x_h_path = self.fc_w(x_2)
        
        att = F.adaptive_avg_pool2d(x_h_path + x_w_path + x_2, output_size=1)
        att = self.reweight(att).reshape(N, C_orig // 2, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x_2_processed = x_h_path * att[0] + x_w_path * att[1] + x_2 * att[2]

        # x_1_processed has C_orig//2, x_2_processed has C_orig//2. Concat gives C_orig.
        # self.fuse_post is Conv2d(channels, channels, ...), so C_orig -> C_orig.
        x_fused = self.fuse_post(torch.cat([x_1_processed, x_2_processed], dim=1))
        return x_fused

    def strip_mlp_path(self, x_strip, H_feat, W_feat):
        N_strip, C_strip, _, _ = x_strip.shape # C_strip is C_orig // 2

        # relate_pos_h_lookup initialized with (C_strip, H_feat)
        # relate_pos_w_lookup initialized with (C_strip, W_feat)
        # Original code: self.relate_pos_h(H, W) and self.relate_pos_w(H, W)
        # This implies the pos embedding might be (H,W,C_strip)-shaped or similar.
        # Let's assume relate_pos_h_lookup is for H-dim interactions, relate_pos_w_lookup for W-dim.
        # And their outputs are (H_feat, H_feat, C_strip) and (W_feat, W_feat, C_strip) respectively.
        # Permuted: (1, C_strip, H_feat, H_feat) and (1, C_strip, W_feat, W_feat)

        # The original implementation of relative position addition was:
        # pos_h = self.relate_pos_h(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        # pos_w = self.relate_pos_w(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        # x_h = x + pos_h  (where x was x_1, so C_strip channels)
        # This implies pos_h and pos_w were (1, C_strip, H_feat, W_feat)
        _pos_h_bias = self.relate_pos_h_lookup(H_feat, W_feat).unsqueeze(0).permute(0, 3, 1, 2)
        _pos_w_bias = self.relate_pos_w_lookup(H_feat, W_feat).unsqueeze(0).permute(0, 3, 1, 2)
        # Note: relate_pos_h_lookup was init with (C_strip, H_feat), so forward(H_feat,W_feat) gives (H_feat,W_feat,C_strip)
        # relate_pos_w_lookup was init with (C_strip, W_feat), so forward(H_feat,W_feat) gives (H_feat,W_feat,C_strip)
        # This seems okay.

        C1_groups = C_strip // self.C_internal_base
        
        x_h_content = x_strip + _pos_h_bias

        reshaped_for_proj_h = x_h_content.view(N_strip, C1_groups, self.C_internal_base, H_feat, W_feat).permute(0,3,2,1,4).contiguous()
        reshaped_for_proj_h = reshaped_for_proj_h.view(N_strip, H_feat*self.C_internal_base, C1_groups, W_feat)
        
        x_h_projected = self.proj_h(reshaped_for_proj_h)
        
        x_h_projected = x_h_projected.view(N_strip, self.chan_proj, H_feat, C1_groups, W_feat)
        x_h_projected = x_h_projected.permute(0,3,1,2,4).contiguous()
        # Output channels: C1_groups * self.chan_proj. Since chan_proj = C_internal_base, this is C_strip.
        x_h_intermediate = x_h_projected.view(N_strip, C_strip, H_feat, W_feat)
        
        # Input to fuse_h: cat([x_h_intermediate (C_strip), x_strip (C_strip)]) -> 2*C_strip = C_orig channels.
        # fuse_h: Conv2d(C_orig, C_strip, ...)
        fused_h_path = self.fuse_h(torch.cat([x_h_intermediate, x_strip], dim=1)) # Output C_strip channels
        
        # BN_x_h_path is nn.BatchNorm2d(C_strip), fused_h_path has C_strip channels. This is correct.
        activated_h_path = self.activation(self.BN_x_h_path(fused_h_path)) + _pos_w_bias

        reshaped_for_proj_w = activated_h_path.view(N_strip, C1_groups, self.C_internal_base, H_feat, W_feat).permute(0,4,2,1,3).contiguous()
        reshaped_for_proj_w = reshaped_for_proj_w.view(N_strip, W_feat*self.C_internal_base, C1_groups, H_feat)

        x_w_projected = self.proj_w(reshaped_for_proj_w)
        
        x_w_projected = x_w_projected.view(N_strip, self.chan_proj, W_feat, C1_groups, H_feat)
        x_w_projected = x_w_projected.permute(0,3,1,4,2).contiguous()
        x_w_intermediate = x_w_projected.view(N_strip, C_strip, H_feat, W_feat)

        # Input to fuse_w: cat([x_strip (C_strip), x_w_intermediate (C_strip)]) -> 2*C_strip = C_orig channels
        # fuse_w: Conv2d(C_orig, C_strip, ...)
        output_strip_mlp = self.fuse_w(torch.cat([x_strip, x_w_intermediate], dim=1)) # Output C_strip channels
        
        return output_strip_mlp

class TokenMixing(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()
        self.smlp_block = StripMLP_Block(C, H, W)
        self.dwsc = DepthWise_Conv(C)
    
    def forward(self, x):
        x_res = x
        x = self.dwsc(x)
        x = self.smlp_block(x)
        # Original code implies residual connection for TokenMixing as part of BasicBlock
        # x = x_res + self.drop_path(self.token_mixing(x))
        # So TokenMixing itself shouldn't add x_res yet.
        return x

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x): # x expected as (N, H, W, C)
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ChannelMixing(nn.Module):
    def __init__(self, in_channel, mlp_ratio, use_dropout=False, drop_rate=0.):
        super().__init__()
        # self.use_dropout = use_dropout # Not directly used, drop_rate is used
        self.conv_77 = nn.Conv2d(in_channel, in_channel, 7, 1, 3, groups=in_channel, bias=False)
        self.layer_norm = nn.LayerNorm(in_channel)
        hidden_dim = int(mlp_ratio * in_channel)
        self.fc1 = nn.Linear(in_channel, hidden_dim)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_dim, in_channel)
        self.grn = GRN(hidden_dim)
    
    def forward(self, x): # x is (N, C, H, W)
        x_res = x
        x = self.conv_77(x)
        
        x = x.permute(0, 2, 3, 1) # (N, H, W, C)
        x = self.layer_norm(x)
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x) # Original Mlp has drop after fc2 too

        x = x.permute(0, 3, 1, 2) # (N, C, H, W)
        # Original code implies residual connection for ChannelMixing as part of BasicBlock
        # x = x_res + self.drop_path(self.channel_mixing(x))
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channel, H, W, mlp_ratio, drop_path_rate=0., token_drop_rate=0.): # use_dropout removed as it's implicit with token_drop_rate > 0
        super().__init__()
        self.token_mixing = TokenMixing(in_channel, H, W)
        self.channel_mixing = ChannelMixing(in_channel, mlp_ratio, drop_rate=token_drop_rate) # Pass token_drop_rate to ChannelMixing's Mlp
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.token_mixing(x))
        x = x + self.drop_path(self.channel_mixing(x))
        return x

class StripMLPNet(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=80, layers=[2, 8, 14, 2], token_mlp_ratio=3,
                 drop_rate=0.0, drop_path_rate=0.1,
                 norm_layer=nn.BatchNorm2d, patch_norm=True, 
                 out_indices=(0, 1, 2, 3), use_checkpoint=False, **kwargs):
        super(StripMLPNet, self).__init__()

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)

        self.num_layers = len(layers)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        # drop_rate is for MLP/Attn-like dropout inside blocks (passed as token_drop_rate to BasicBlock)
        # drop_path_rate is for stochastic depth

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        if patch_norm:
            self.patch_norm_layer = norm_layer(embed_dim) # Renamed to avoid clash
        else:
            self.patch_norm_layer = nn.Identity()

        patches_resolution_h = img_size[0] // patch_size
        patches_resolution_w = img_size[1] // patch_size
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        current_dim = embed_dim
        current_H, current_W = patches_resolution_h, patches_resolution_w
        
        stage_blocks1 = []
        for i in range(layers[0]):
            stage_blocks1.append(BasicBlock(
                in_channel=current_dim, H=current_H, W=current_W, mlp_ratio=token_mlp_ratio,
                drop_path_rate=dpr[sum(layers[:0]) + i], token_drop_rate=drop_rate))
        self.stage1 = nn.Sequential(*stage_blocks1)
        if 0 in self.out_indices:
            self.norm0 = norm_layer(current_dim)

        self.merging1 = nn.Conv2d(current_dim, current_dim * 2, kernel_size=2, stride=2, bias=False)
        current_dim *= 2; current_H //= 2; current_W //= 2
        self.conv_s1_28 = nn.Conv2d(current_dim, current_dim * 2, kernel_size=2, stride=2, groups=current_dim, bias=False)
        stage_blocks2 = []
        for i in range(layers[1]):
            stage_blocks2.append(BasicBlock(
                in_channel=current_dim, H=current_H, W=current_W, mlp_ratio=token_mlp_ratio,
                drop_path_rate=dpr[sum(layers[:1]) + i], token_drop_rate=drop_rate))
        self.stage2 = nn.Sequential(*stage_blocks2)
        if 1 in self.out_indices:
            self.norm1 = norm_layer(current_dim)
        
        self.merging2 = nn.Conv2d(current_dim, current_dim * 2, kernel_size=2, stride=2, bias=False)
        current_dim *= 2; current_H //= 2; current_W //= 2
        self.conv_s1_14 = nn.Conv2d(current_dim, current_dim * 2, kernel_size=2, stride=2, groups=current_dim, bias=False)
        self.conv_s2_14 = nn.Conv2d(current_dim, current_dim * 2, kernel_size=2, stride=2, groups=current_dim, bias=False)
        stage_blocks3 = []
        for i in range(layers[2]):
            stage_blocks3.append(BasicBlock(
                in_channel=current_dim, H=current_H, W=current_W, mlp_ratio=token_mlp_ratio,
                drop_path_rate=dpr[sum(layers[:2]) + i], token_drop_rate=drop_rate))
        self.stage3 = nn.Sequential(*stage_blocks3)
        if 2 in self.out_indices:
            self.norm2 = norm_layer(current_dim)

        self.merging3 = nn.Conv2d(current_dim, current_dim * 2, kernel_size=2, stride=2, bias=False)
        current_dim *= 2; current_H //= 2; current_W //= 2
        stage_blocks4 = []
        for i in range(layers[3]):
            stage_blocks4.append(BasicBlock(
                in_channel=current_dim, H=current_H, W=current_W, mlp_ratio=token_mlp_ratio,
                drop_path_rate=dpr[sum(layers[:3]) + i], token_drop_rate=drop_rate))
        self.stage4 = nn.Sequential(*stage_blocks4)
        if 3 in self.out_indices:
            self.norm3 = norm_layer(current_dim)
        
        self.apply(self._init_weights)

        self.eval()
        # Use a consistent device for dummy input, especially if model is moved to GPU later
        # For init, cpu is fine.
        dummy_input = torch.randn(1, in_chans, img_size[0], img_size[1])
        try:
            # Need to ensure dummy input is on same device as model if model moved before this
            features = self.forward(dummy_input.to(next(self.parameters()).device if sum(p.numel() for p in self.parameters()) > 0 else "cpu") )
            self.width_list = [f.shape[1] for f in features]
        except Exception as e:
            print(f"Warning: Could not compute width_list during init: {e}")
            self.width_list = [] 
        self.train()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): # Added GroupNorm just in case
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.patch_norm_layer(x)
        
        outs = []

        x_s1_out = self.stage1(x)
        if 0 in self.out_indices:
            outs.append(self.norm0(x_s1_out))
        
        x_s2_in = self.merging1(x_s1_out)
        x_s1_14_skip = self.conv_s1_28(x_s2_in)
        x_s1_7_skip = self.conv_s1_14(x_s1_14_skip)

        x_s2_out = self.stage2(x_s2_in)
        if 1 in self.out_indices:
            outs.append(self.norm1(x_s2_out))

        x_s3_in = self.merging2(x_s2_out)
        x_s2_7_skip = self.conv_s2_14(x_s3_in)
        
        x_s3_out = self.stage3(x_s3_in + x_s1_14_skip) 
        if 2 in self.out_indices:
            outs.append(self.norm2(x_s3_out))

        x_s4_in = self.merging3(x_s3_out)
        x_s4_out = self.stage4(x_s4_in + x_s1_7_skip + x_s2_7_skip)
        if 3 in self.out_indices:
            outs.append(self.norm3(x_s4_out))
            
        return outs

def StripMLPNet_LightTiny(img_size=224, in_chans=3, drop_path_rate=0.1, out_indices=(0,1,2,3), weights=None, **kwargs):
    model = StripMLPNet(img_size=img_size, in_chans=in_chans, embed_dim=80, layers=[2, 2, 6, 2], 
                          drop_path_rate=drop_path_rate, out_indices=out_indices, **kwargs)
    if weights: pass
    return model

def StripMLPNet_Tiny(img_size=224, in_chans=3, drop_path_rate=0.1, out_indices=(0,1,2,3), weights=None, **kwargs):
    model = StripMLPNet(img_size=img_size, in_chans=in_chans, embed_dim=80, layers=[2, 2, 12, 2],
                          drop_path_rate=drop_path_rate, out_indices=out_indices, **kwargs)
    if weights: pass
    return model

def StripMLPNet_Small(img_size=224, in_chans=3, drop_path_rate=0.2, out_indices=(0,1,2,3), weights=None, **kwargs):
    model = StripMLPNet(img_size=img_size, in_chans=in_chans, embed_dim=96, layers=[2, 2, 18, 2],
                          drop_path_rate=drop_path_rate, out_indices=out_indices, **kwargs)
    if weights: pass
    return model

def StripMLPNet_Base(img_size=224, in_chans=3, drop_path_rate=0.3, out_indices=(0,1,2,3), weights=None, **kwargs):
    model = StripMLPNet(img_size=img_size, in_chans=in_chans, embed_dim=112, layers=[2, 2, 18, 2],
                          drop_path_rate=drop_path_rate, out_indices=out_indices, **kwargs)
    if weights: pass
    return model


if __name__ == "__main__":
    import os
    data = torch.rand((1, 3, 640, 640)) # .cuda() if testing on GPU

    smlp = StripMLPNet_Base(img_size=640, out_indices=(0,1,2,3)) # .cuda()
    # smlp = StripMLPNet_Tiny(img_size=224, out_indices=(0,1,2,3))
    
    # Ensure model and data are on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smlp.to(device)
    data = data.to(device)

    print(f"Model: {type(smlp).__name__} on {device}")
    
    outs = smlp(data)
    print("Output features:")
    for i, out_feat in enumerate(outs):
        print(f"  Stage {i}: {out_feat.shape}, Channels: {smlp.width_list[i] if smlp.width_list else 'N/A'}")
    
    print(f"Model channel_list: {smlp.width_list}")

    try:
        from ptflops import get_model_complexity_info
        ops, params = get_model_complexity_info(smlp, (3, 224, 224), as_strings=True,
                                                print_per_layer_stat=False, verbose=False)
        print(f"The model parameters: {params}")
        print(f"The model FLOPS: {ops}")
    except ImportError:
        print("ptflops not installed. Skipping complexity calculation.")
    except Exception as e:
        print(f"Error during ptflops calculation: {e}")
    
    print("Get output succeeded!...")