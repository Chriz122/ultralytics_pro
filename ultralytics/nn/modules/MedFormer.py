# -*- coding: utf-8 -*-
from collections import OrderedDict
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor
from typing import Optional, Tuple
from einops import rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) #NHWC

        return x

class Attention(nn.Module):
    """
    vanilla attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        args:
            x: NHWC tensor
        return:
            NHWC tensor
        """
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')
        
        #######################################
        B, N, C = x.shape        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #######################################

        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x

class AttentionLePE(nn.Module):
    """
    vanilla attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)

    def forward(self, x):
        """
        args:
            x: NHWC tensor
        return:
            NHWC tensor
        """
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')
        
        #######################################
        B, N, C = x.shape        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        lepe = self.lepe(rearrange(x, 'n (h w) c -> n c h w', h=H, w=W))
        lepe = rearrange(lepe, 'n c h w -> n (h w) c')

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + lepe

        x = self.proj(x)
        x = self.proj_drop(x)
        #######################################

        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x



class nchwAttentionLePE(nn.Module):
    """
    Attention with LePE, takes nchw input
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)

    def forward(self, x:torch.Tensor):
        """
        args:
            x: NCHW tensor
        return:
            NCHW tensor
        """
        B, C, H, W = x.size()
        q, k, v = self.qkv.forward(x).chunk(3, dim=1) # B, C, H, W

        attn = q.view(B, self.num_heads, self.head_dim, H*W).transpose(-1, -2) @ \
               k.view(B, self.num_heads, self.head_dim, H*W)
        attn = torch.softmax(attn*self.scale, dim=-1)
        attn = self.attn_drop(attn)

        # (B, nhead, HW, HW) @ (B, nhead, HW, head_dim) -> (B, nhead, HW, head_dim)
        output:torch.Tensor = attn @ v.view(B, self.num_heads, self.head_dim, H*W).transpose(-1, -2)
        output = output.permute(0, 1, 3, 2).reshape(B, C, H, W)
        output = output + self.lepe(v)

        output = self.proj_drop(self.proj(output))

        return output


def _grid2seq(x: Tensor, region_size: Tuple[int], num_heads: int):
    """
    Args:
        x: BCHW tensor
        region size: int
        num_heads: number of attention heads
    Return:
        out: rearranged x, has a shape of (bs, nhead, nregion, reg_size, head_dim)
        region_h, region_w: number of regions per col/row
    """
    B, C, H, W = x.size()
    region_h, region_w = H // region_size[0], W // region_size[1]
    x = x.view(B, num_heads, C // num_heads, region_h, region_size[0], region_w, region_size[1])
    x = torch.einsum('bmdhpwq->bmhwpqd', x).flatten(2, 3).flatten(-3, -2)
    return x, region_h, region_w


def _seq2grid(x: Tensor, region_h: int, region_w: int, region_size: Tuple[int]):
    """
    Args: 
        x: (bs, nhead, nregion, reg_size^2, head_dim)
    Return:
        x: (bs, C, H, W)
    """
    bs, nhead, nregion, reg_size_square, head_dim = x.size()
    x = x.view(bs, nhead, region_h, region_w, region_size[0], region_size[1], head_dim)
    x = torch.einsum('bmhwpqd->bmdhpwq', x).reshape(bs, nhead * head_dim,
                                                    region_h * region_size[0], region_w * region_size[1])
    return x


def dual_sparse_attention(
                            query: Tensor, key: Tensor, value: Tensor, scale: float,
                            region_graph: LongTensor, region_size: Tuple[int],
                            kv_region_size: Optional[Tuple[int]] = None,
                            auto_pad=True) -> Tensor:
    """
    Args:
        query, key, value: (B, C, H, W) tensor
        scale: the scale/temperature for dot product attention
        region_graph: (B, nhead, h_q*w_q, topk) tensor, topk <= h_k*w_k
        region_size: region/window size for queries, (rh, rw)
        key_region_size: optional, if None, key_region_size=region_size
        auto_pad: required to be true if the input sizes are not divisible by the region_size
    Return:
        output: (B, C, H, W) tensor
        attn: (bs, nhead, q_nregion, reg_size, topk*kv_region_size) attention matrix
    """
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()

    # =========================================================================
    # MODIFICATION 1 START: Save original query size for correct un-padding later
    # =========================================================================
    _, _, Hq_orig, Wq_orig = query.size()
    # =========================================================================
    # MODIFICATION 1 END
    # =========================================================================

    # Auto pad to deal with any input size 
    q_pad_b, q_pad_r, kv_pad_b, kv_pad_r = 0, 0, 0, 0
    if auto_pad:
        _, _, Hq, Wq = query.size()
        q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
        q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
        if (q_pad_b > 0 or q_pad_r > 0):
            query = F.pad(query, (0, q_pad_r, 0, q_pad_b))  # zero padding

        _, _, Hk, Wk = key.size()
        kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
        kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
        if (kv_pad_r > 0 or kv_pad_b > 0):
            key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b))  # zero padding
            value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b))  # zero padding

    # to sequence format, i.e. (bs, nhead, nregion, reg_size, head_dim)
    query, q_region_h, q_region_w = _grid2seq(query, region_size=region_size, num_heads=nhead)
    key, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)

    # gather key and values.
    # TODO: is seperate gathering slower than fused one (our old version) ?
    # torch.gather does not support broadcasting, hence we do it manually
    bs, nhead, kv_nregion, kv_region_size_sq, head_dim = key.size()
    broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1). \
        expand(-1, -1, -1, -1, kv_region_size_sq, head_dim)
    key_g = torch.gather(key.view(bs, nhead, 1, kv_nregion, kv_region_size_sq, head_dim). \
                         expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
                         index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size_sq, head_dim)


    value_g = torch.gather(value.view(bs, nhead, 1, kv_nregion, kv_region_size_sq, head_dim). \
                           expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
                           index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size_sq, head_dim)

    # token-to-token attention
    # (bs, nhead, q_nregion, reg_size, head_dim) @ (bs, nhead, q_nregion, head_dim, topk*kv_region_size)
    # -> (bs, nhead, q_nregion, reg_size, topk*kv_region_size)
    # TODO: mask padding region
    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
    # (bs, nhead, q_nregion, reg_size, k2)
    score, index = attn.topk(topk * (kv_region_size[0] * kv_region_size[1]) // 8, dim=-1)
    v_g_un = value_g.flatten(-3, -2).unsqueeze(-3).expand(-1, -1, -1, attn.size(3), -1, -1)# (bs, nhead, q_nregion, reg_size, topk*kv_region_size, head_dim)
    idx = index.unsqueeze(-1).expand(-1, -1, -1, -1, -1, head_dim)# (bs, nhead, q_nregion, reg_size, k2, head_dim)
    v_g_select = torch.gather(v_g_un, dim=4, index=idx)
    a_g = score.unsqueeze(-2)
    a_g = torch.softmax(a_g, dim=-1)
    output = (a_g @ v_g_select).squeeze(-2)
    # to BCHW format
    output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_size=region_size)

    # remove paddings if needed
    if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
        # =========================================================================
        # MODIFICATION 2 START: Use the saved original size to correctly crop the output
        # =========================================================================
        output = output[:, :, :Hq_orig, :Wq_orig]
        # =========================================================================
        # MODIFICATION 2 END
        # =========================================================================

    return output, attn


class DSSA(nn.Module):
    """Bi-Level Routing Attention that takes nchw input

    Compared to legacy version, this implementation:
    * removes unused args and components
    * uses nchw input format to avoid frequent permutation

    When the size of inputs is not divisible by the region size, there is also a numerical difference
    than legacy implementation, due to:
    * different way to pad the input feature map (padding after linear projection)
    * different pooling behavior (count_include_pad=False)

    Current implementation is more reasonable, hence we do not keep backward numerical compatiability
    """

    def __init__(self, dim, num_heads=8, n_win=7, qk_scale=None, topk=4, side_dwconv=3, auto_pad=True,
                 attn_backend='torch'):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.a_r = None
        self.idx_r = None
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5  # NOTE: to be consistent with old models.
        self.auto_pad = auto_pad

        ################side_dwconv (i.e. LCE in Shunted Transformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        ################ regional routing setting #################
        self.topk = topk
        self.n_win = n_win  # number of windows per row/col

        ##########################################

        self.qkv_linear = nn.Conv2d(self.dim, 3 * self.dim, kernel_size=1)
        self.output_linear = nn.Conv2d(self.dim, self.dim, kernel_size=1)

        if attn_backend == 'torch':
            self.attn_fn = dual_sparse_attention
        else:
            raise ValueError('CUDA implementation is not available yet. Please stay tuned.')

    def forward(self, x: Tensor, ret_attn_mask=False):
        """
        Args:
            x: NCHW tensor, better to be channel_last (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        Return:
            NCHW tensor
        """

        x = x.permute(0, 3, 1, 2)
        N, C, H, W = x.size()
        region_size = (H // self.n_win, W // self.n_win)

        # STEP 1: linear projection 
        qkv = self.qkv_linear.forward(x)  # ncHW
        q, k, v = qkv.chunk(3, dim=1)  # ncHW

        # STEP 2: regional sparse selection
        # NOTE: ceil_mode=True, count_include_pad=False = auto padding
        # NOTE: gradients backward through token-to-token attention. See Appendix A for the intuition.
        q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)  # nchw
        q_r: Tensor = q_r.permute(0, 2, 3, 1).flatten(1, 2)  # n(hw)c
        k_r: Tensor = k_r.flatten(2, 3)  # nc(hw)
        a_r = q_r @ k_r  # n(hw)(hw), adj matrix of regional graph
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1)  # n(hw)k long tensor


        idx_r: LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1)

        # STEP 3: token-level sparse selection
        output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        region_graph=idx_r, region_size=region_size,
                                        auto_pad=self.auto_pad
                                        )
        output = output + self.lepe(v)  # ncHW
        output = self.output_linear(output)  # ncHW
        output = output.permute(0, 2, 3, 1)  # ncHW->nHWc
        if ret_attn_mask:
            return output, attn_mat

        return output


def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'PE name {name} is not surpported!')


class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio,
                 stage,
                 drop_path=0.,
                 layer_scale_init_value=-1,
                 num_heads=8,
                 n_win=7,
                 qk_scale=None,
                 topk=4,
                 mlp_dwconv=False,
                 side_dwconv=5,
                 before_attn_dwconv=3,
                 pre_norm=True,
                 # =========================================================================
                 # MODIFICATION 3 START: Change default value to True to enable auto-padding
                 # =========================================================================
                 auto_pad=True,
                 # =========================================================================
                 # MODIFICATION 3 END
                 # =========================================================================
                 ):
        super().__init__()

        # 1.DWConv 3*3
        if before_attn_dwconv > 0:  # before_attn_dwconv=3
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)  # DWConv
        else:
            self.pos_embed = lambda x: 0
        # 2.LN
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        # 3.Dual Sparse Selection Attention
        if topk > 0:
            self.attn = DSSA(dim=dim, num_heads=num_heads, n_win=n_win, qk_scale=qk_scale, topk=topk,
                             side_dwconv=side_dwconv,
                             auto_pad=auto_pad, attn_backend='torch')
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        # 4.LN
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        # 5.MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            # mlp_dwconv=False
            DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * dim), dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm
        self.stage = stage

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)
        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:  # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)
        return x


class MedFormer(nn.Module):
    def __init__(self,
                 depth=[3, 4, 8, 3],
                 in_chans=3,
                 num_classes=1000,
                 img_size=224,
                 embed_dim=[64, 128, 320, 512],
                 head_dim=64,
                 qk_scale=None,
                 representation_size=None,
                 drop_path_rate=0.,
                 drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 n_win=7,
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[64, 128, 320, 512],
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=True,
                 mlp_ratios=[4, 4, 4, 4],
                 mlp_dwconv=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.in_chans = in_chans
        self.img_size = img_size
        ################################################
        # downsample layers (patch embeddings)
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        # pe_stages=0
        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i],
                        stage=[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_scale=qk_scale,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad,
                        ) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim[-1], representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1],
                              num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        
        # --- Add width_list calculation ---
        self.width_list = []
        try:
            self.eval() 
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size)
            with torch.no_grad():
                 features = self.forward_features(dummy_input)

            self.width_list = [f.size(1) for f in features]
            self.train()
        except Exception as e:
            print(f"Error during dummy forward pass for width_list calculation: {e}")
            print("Setting width_list to embed_dims as fallback.")
            self.width_list = self.embed_dim
            self.train()


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        feature_outputs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i < 3: # For the first 3 stages, append output
                feature_outputs.append(x)

        x = self.norm(x)
        # x = self.pre_logits(x) # pre_logits usually for classification, skip for feature extraction
        feature_outputs.append(x) # Append the final stage output
        return feature_outputs

    def forward(self, x):
        x = self.forward_features(x)
        return x


@register_model
def medformer_tiny(pretrained=False, **kwargs):
    model = MedFormer(
        depth=[2, 2, 8, 2],
        embed_dim=[32, 64, 128, 256], mlp_ratios=[3, 3, 3, 3],
        n_win=7,
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[32, 64, 128, 256],
        head_dim=32,
        pre_norm=True,
        pe=None,
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def medformer_small(pretrained=False, **kwargs):
    model = MedFormer(
        depth=[2, 2, 8, 2],
        embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
        n_win=7,
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        pre_norm=True,
        pe=None,
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def medformer_base(pretrained=False, **kwargs):
    model = MedFormer(
        depth=[2, 2, 8, 2],
        embed_dim=[96, 192, 384, 768], mlp_ratios=[3, 3, 3, 3],
        use_checkpoint_stages=[],
        n_win=7,
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[96, 192, 384, 768],
        head_dim=32,
        pre_norm=True,
        pe=None,
        **kwargs)
    model.default_cfg = _cfg()
    return model
    
if __name__ == '__main__':
    # 測試代碼
    img_h, img_w = 640, 640
    print("--- Creating MedFormer Tiny model ---")
    model = medformer_tiny(img_size=img_h)
    print("Model created successfully.")
    print("Calculated width_list:", model.width_list)

    # 測試前向傳播
    input_tensor = torch.rand(2, 3, img_h, img_w)
    print(f"\n--- Testing MedFormer Tiny forward pass (Input: {input_tensor.shape}) ---")

    model.eval()
    try:
        with torch.no_grad():
            output_features = model(input_tensor)
        print("Forward pass successful.")
        print("Output feature shapes:")
        for i, features in enumerate(output_features):
            print(f"Stage {i+1}: {features.shape}")

        # 驗證 width_list 是否與執行時的輸出相符
        runtime_widths = [f.size(1) for f in output_features]
        print("\nRuntime output feature channels:", runtime_widths)
        assert model.width_list == runtime_widths, "Width list mismatch!"
        print("Width list verified successfully.")

        # --- 測試 deepcopy ---
        print("\n--- Testing deepcopy ---")
        import copy
        copied_model = copy.deepcopy(model)
        print("Deepcopy successful.")

        # 可選：測試複製後模型的前向傳播
        with torch.no_grad():
             output_copied = copied_model(input_tensor)
        print("Copied model forward pass successful.")
        assert len(output_copied) == len(output_features)
        for i in range(len(output_features)):
             assert output_copied[i].shape == output_features[i].shape
        print("Copied model output shapes verified.")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()