from typing import Type, Callable, Tuple, Optional, Set, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

def _gelu_ignore_parameters(
        *args,
        **kwargs
) -> nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation


class MBConv(nn.Module):
    """ MBConv block as described in: https://arxiv.org/pdf/2204.01697.pdf.

        Without downsampling:
        x ← x + Proj(SE(DWConv(Conv(Norm(x)))))

        With downsampling:
        x ← Proj(Pool2D(x)) + Proj(SE(DWConv ↓(Conv(Norm(x))))).

        Conv is a 1 X 1 convolution followed by a Batch Normalization layer and a GELU activation.
        SE is the Squeeze-Excitation layer.
        Proj is the shrink 1 X 1 convolution.

        Note: This implementation differs slightly from the original MobileNet implementation!

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downscale (bool, optional): If true downscale by a factor of two is performed. Default: False
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        drop_path (float, optional): Dropout rate to be applied during training. Default 0.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_path: float = 0.,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MBConv, self).__init__()
        # Save parameter
        self.drop_path_rate: float = drop_path
        # Check parameters for downscaling
        if not downscale:
            assert in_channels == out_channels, "If downscaling is utilized input and output channels must be equal."
        # Ignore inplace parameter if GELU is used
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        # Make main path
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1)),
            DepthwiseSeparableConv(in_chs=in_channels, out_chs=out_channels, stride=2 if downscale else 1,
                                   act_layer=act_layer, norm_layer=norm_layer, drop_path_rate=drop_path),
            SqueezeExcite(in_chs=out_channels, rd_ratio=0.25),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))
        )
        # Make skip path
        self.skip_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        ) if downscale else nn.Identity()

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)] (downscaling is optional).
        """
        output = self.main_path(input)
        if self.drop_path_rate > 0.:
            output = drop_path(output, self.drop_path_rate, self.training)
        output = output + self.skip_path(input)
        return output


def window_partition(
        input: torch.Tensor,
        window_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Window partition function.

    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        window_size (Tuple[int, int], optional): Window size to be applied. Default (7, 7)

    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape [B * windows, window_size[0], window_size[1], C].
    """
    # Get size of input
    B, C, H, W = input.shape
    # Unfold input
    windows = input.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    # Permute and reshape to [B * windows, window_size[0], window_size[1], channels]
    windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(
        windows: torch.Tensor,
        original_size: Tuple[int, int],
        window_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Reverses the window partition.

    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0], window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)

    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return output


def grid_partition(
        input: torch.Tensor,
        grid_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Grid partition function.

    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        grid_size (Tuple[int, int], optional): Grid size to be applied. Default (7, 7)

    Returns:
        grid (torch.Tensor): Unfolded input tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
    """
    # Get size of input
    B, C, H, W = input.shape
    # Unfold input
    grid = input.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
    # Permute and reshape [B * (H // grid_size[0]) * (W // grid_size[1]), grid_size[0], window_size[1], C]
    grid = grid.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return grid


def grid_reverse(
        grid: torch.Tensor,
        original_size: Tuple[int, int],
        grid_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Reverses the grid partition.

    Args:
        Grid (torch.Tensor): Grid tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        grid_size (Tuple[int, int], optional): Grid size which have been applied. Default (7, 7)

    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    # Get height, width, and channels
    (H, W), C = original_size, grid.shape[-1]
    # Compute original batch size
    B = int(grid.shape[0] / (H * W / grid_size[0] / grid_size[1]))
    # Fold grid tensor
    output = grid.view(B, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
    return output


def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.

    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.

    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class RelativeSelfAttention(nn.Module):
    """ Relative Self-Attention similar to Swin V1. Implementation inspired by Timms Swin V1 implementation.

    Args:
        in_channels (int): Number of input channels.
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
            self,
            in_channels: int,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(RelativeSelfAttention, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.grid_window_size: Tuple[int, int] = grid_window_size
        self.scale: float = (in_channels // num_heads) ** -0.5
        self.attn_area: int = grid_window_size[0] * grid_window_size[1]
        # Init layers
        self.qkv_mapping = nn.Linear(in_features=in_channels, out_features=3 * in_channels, bias=True)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * grid_window_size[0] - 1) * (2 * grid_window_size[1] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(grid_window_size[0],
                                                                                    grid_window_size[1]), persistent=False)
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.

        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B_, N, C].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B_, N, C].
        """
        # Get shape of input
        B_, N, C = input.shape
        # Perform query key value mapping
        qkv = self.qkv_mapping(input).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Scale query
        q = q * self.scale
        # Compute attention maps
        attn = self.softmax(q @ k.transpose(-2, -1) + self._get_relative_positional_bias())
        attn = self.attn_drop(attn)
        # Map value with attention maps
        output = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # Perform final projection and dropout
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


class MaxViTTransformerBlock(nn.Module):
    """ MaxViT Transformer block.

        With block partition:
        x ← x + Unblock(RelAttention(Block(LN(x))))
        x ← x + MLP(LN(x))

        With grid partition:
        x ← x + Ungrid(RelAttention(Grid(LN(x))))
        x ← x + MLP(LN(x))

        Layer Normalization (LN) is applied after the grid/window partition to prevent multiple reshaping operations.
        Grid/window reverse (Unblock/Ungrid) is performed on the final output for the same reason.

    Args:
        in_channels (int): Number of input channels.
        partition_function (Callable): Partition function to be utilized (grid or window partition).
        reverse_function (Callable): Reverse function to be utilized  (grid or window reverse).
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
    """

    def __init__(
            self,
            in_channels: int,
            partition_function: Callable,
            reverse_function: Callable,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        """ Constructor method """
        super(MaxViTTransformerBlock, self).__init__()
        # Save parameters
        self.partition_function: Callable = partition_function
        self.reverse_function: Callable = reverse_function
        self.grid_window_size: Tuple[int, int] = grid_window_size
        # Init layers
        self.norm_1 = norm_layer(in_channels)
        self.attention = RelativeSelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(
            in_features=in_channels,
            hidden_features=int(mlp_ratio * in_channels),
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H, W].
        """
        # Save original shape
        B, C, H, W = input.shape
        shortcut = input

        # --- Padding ---
        # Calculate padding for height and width
        win_h, win_w = self.grid_window_size
        pad_h = (win_h - H % win_h) % win_h
        pad_w = (win_w - W % win_w) % win_w
        
        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            input_padded = F.pad(input, (0, pad_w, 0, pad_h))
        else:
            input_padded = input

        _, _, H_padded, W_padded = input_padded.shape

        # Perform partition
        input_partitioned = self.partition_function(input_padded, self.grid_window_size)
        input_partitioned = input_partitioned.view(-1, self.grid_window_size[0] * self.grid_window_size[1], C)

        # Perform normalization, attention, and drop path
        attn_out = self.attention(self.norm_1(input_partitioned))
        attn_out_reversed = self.reverse_function(attn_out, (H_padded, W_padded), self.grid_window_size)
        
        # Remove padding if it was added
        if pad_h > 0 or pad_w > 0:
            attn_out_reversed = attn_out_reversed[:, :, :H, :W]

        # First residual connection
        x = shortcut + self.drop_path(attn_out_reversed)
        
        shortcut2 = x
        
        # Apply padding again for the MLP path
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_padded = x

        # Re-partition for MLP
        x_partitioned = self.partition_function(x_padded, self.grid_window_size)
        x_partitioned = x_partitioned.view(-1, self.grid_window_size[0] * self.grid_window_size[1], C)
        
        # Perform normalization, MLP, and drop path
        mlp_out = self.mlp(self.norm_2(x_partitioned))
        mlp_out_reversed = self.reverse_function(mlp_out, (H_padded, W_padded), self.grid_window_size)
        
        # Remove padding if it was added
        if pad_h > 0 or pad_w > 0:
            mlp_out_reversed = mlp_out_reversed[:, :, :H, :W]

        # Second residual connection
        output = shortcut2 + self.drop_path(mlp_out_reversed)
        
        return output


class MaxViTBlock(nn.Module):
    """ MaxViT block composed of MBConv block, Block Attention, and Grid Attention.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downscale (bool, optional): If true spatial downscaling is performed. Default: False
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            norm_layer_transformer: Type[nn.Module] = nn.LayerNorm
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MaxViTBlock, self).__init__()
        # Init MBConv block
        self.mb_conv = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            downscale=downscale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path
        )
        # Init Block and Grid Transformer
        self.block_transformer = MaxViTTransformerBlock(
            in_channels=out_channels,
            partition_function=window_partition,
            reverse_function=window_reverse,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer
        )
        self.grid_transformer = MaxViTTransformerBlock(
            in_channels=out_channels,
            partition_function=grid_partition,
            reverse_function=grid_reverse,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W]

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2] (downscaling is optional)
        """
        x = self.mb_conv(input)
        x = self.block_transformer(x)
        output = self.grid_transformer(x)
        return output


class MaxViTStage(nn.Module):
    """ Stage of the MaxViT.

    Args:
        depth (int): Depth of the stage.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
    """

    def __init__(
            self,
            depth: int,
            in_channels: int,
            out_channels: int,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: Union[List[float], float] = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            norm_layer_transformer: Type[nn.Module] = nn.LayerNorm
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MaxViTStage, self).__init__()
        # Init blocks
        self.blocks = nn.ModuleList()
        self.blocks.append(MaxViTBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                downscale=True,
                num_heads=num_heads,
                grid_window_size=grid_window_size,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path if isinstance(drop_path, float) else drop_path[0],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_transformer=norm_layer_transformer
            ))
        for index in range(1, depth):
            self.blocks.append(MaxViTBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                downscale=False,
                num_heads=num_heads,
                grid_window_size=grid_window_size,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path if isinstance(drop_path, float) else drop_path[index],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_transformer=norm_layer_transformer
            ))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2].
        """
        output = input
        for block in self.blocks:
            output = block(output)
        return output

class MaxViT(nn.Module):
    """ Implementation of the MaxViT proposed in:
        https://arxiv.org/pdf/2204.01697.pdf

    Args:
        in_channels (int, optional): Number of input channels to the convolutional stem. Default 3
        img_size (int): Input image size.
        depths (Tuple[int, ...], optional): Depth of each network stage. Default (2, 2, 5, 2)
        channels (Tuple[int, ...], optional): Number of channels in each network stage. Default (64, 128, 256, 512)
        num_classes (int, optional): Number of classes to be predicted. Default 1000
        embed_dim (int, optional): Embedding dimension of the convolutional stem. Default 64
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path_rate (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
        global_pool (str, optional): Global polling type to be utilized. Default "avg"
    """

    def __init__(
            self,
            in_chans: int = 3,
            img_size: int = 224,
            depths: Tuple[int, ...] = (2, 2, 5, 2),
            channels: Tuple[int, ...] = (64, 128, 256, 512),
            num_classes: int = 1000,
            embed_dim: int = 64,
            num_heads: int = 32,
            grid_window_size: int = 7,
            attn_drop: float = 0.,
            drop_rate=0.,
            drop_path_rate=0.,
            mlp_ratio=4.,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            norm_layer_transformer=nn.LayerNorm,
            global_pool: str = "avg",
            **kwargs,
    ) -> None:
        """ Constructor method """
        super().__init__()
        # Check parameters
        assert len(depths) == len(channels), "For each stage a channel dimension must be given."
        assert global_pool in ["avg", "max"], f"Only avg and max is supported but {global_pool} is given"
        
        self.num_classes = num_classes
        self.depths = depths
        self.channels = channels
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.in_chans = in_chans
        
        # Init convolutional stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=(3, 3), stride=(2, 2),
                      padding=(1, 1)),
            act_layer(),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            act_layer(),
        )
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Init blocks
        stages = []
        in_channels_stage = embed_dim
        for i in range(self.num_stages):
            stage = MaxViTStage(
                depth=depths[i],
                in_channels=in_channels_stage,
                out_channels=channels[i],
                num_heads=num_heads,
                grid_window_size=(grid_window_size, grid_window_size),
                attn_drop=attn_drop,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_transformer=norm_layer_transformer
            )
            stages.append(stage)
            in_channels_stage = channels[i]
            
        self.stages = nn.ModuleList(stages)
        self.global_pool: str = global_pool
        # Classification head, mainly for pre-training or as a default
        self.head = nn.Linear(channels[-1], num_classes) if num_classes > 0 else nn.Identity()

        # --- Add width_list calculation ---
        self.width_list = []
        try:
            self.eval()
            # Use a dummy input size that is divisible by the downscaling factors and window size
            dummy_input_size = self._get_safe_dummy_input_size()
            dummy_input = torch.randn(1, self.in_chans, dummy_input_size, dummy_input_size)
            with torch.no_grad():
                features = self.forward_features(dummy_input)
            self.width_list = [f.size(1) for f in features]
            self.train()
        except Exception as e:
            print(f"Error during dummy forward pass for width_list calculation: {e}")
            print("Setting width_list to channels as fallback.")
            self.width_list = list(self.channels)
            self.train()
            
    def _get_safe_dummy_input_size(self):
        # A simple way to get a safe input size is to use a size that is a multiple of 
        # the total downscaling factor and the window size.
        # Total downscaling = 2 (stem) * 2^num_stages
        # Let's assume a reasonable size like 224 which is often used and divisible by 7.
        # A more robust way might involve calculating LCM, but for now, 224 is a safe bet.
        if self.img_size % (2**(self.num_stages + 1)) == 0 and all((self.img_size // (2**(i+2))) % 7 == 0 for i in range(self.num_stages)):
             return self.img_size
        return 224
        
    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """ Gets the names of parameters to not apply weight decay to.

        Returns:
            nwd (Set[str]): Set of parameter names to not apply weight decay to.
        """
        nwd = set()
        for n, p in self.named_parameters():
            if "relative_position_bias_table" in n:
                nwd.add(n)
            if 'bias' in n:
                nwd.add(n)
            if 'norm' in n:
                nwd.add(n)
        return nwd

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Method results the classification head

        Args:
            num_classes (int): Number of classes to be predicted
            global_pool (str, optional): If not global pooling is updated
        """
        self.num_classes: int = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.channels[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, input: torch.Tensor) -> List[torch.Tensor]:
        """ Forward pass of feature extraction.

        Args:
            input (torch.Tensor): Input images of the shape [B, C, H, W].

        Returns:
            output (List[torch.Tensor]): List of image features from the backbone.
        """
        feature_outputs = []
        x = self.stem(input)
        for stage in self.stages:
            x = stage(x)
            feature_outputs.append(x)
        return feature_outputs

    def forward_head(self, input: torch.Tensor, pre_logits: bool = False):
        """ Forward pass of classification head.

        Args:
            input (torch.Tensor): Input features
            pre_logits (bool, optional): If true pre-logits are returned

        Returns:
            output (torch.Tensor): Classification output of the shape [B, num_classes].
        """
        if self.global_pool == "avg":
            x = input.mean(dim=(2, 3))
        elif self.global_pool == "max":
            x = torch.amax(input, dim=(2, 3))
        else:
            x = input # No pooling
            
        return x if pre_logits else self.head(x)

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        """ Forward pass for feature extraction.

        Args:
            input (torch.Tensor): Input images of the shape [B, C, H, W].

        Returns:
            output (List[torch.Tensor]): List of feature maps from each stage.
        """
        output = self.forward_features(input)
        return output

# --- Factory Functions ---

@register_model
def maxvit_tiny(pretrained=False, img_size=224, **kwargs):
    """ MaxViT-Tiny for 224x224 input """
    model = MaxViT(
        img_size=img_size,
        depths=(2, 2, 5, 2),
        channels=(64, 128, 256, 512),
        embed_dim=64,
        num_heads=32,
        grid_window_size=7,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def maxvit_small(pretrained=False, img_size=224, **kwargs):
    """ MaxViT-Small for 224x224 input """
    model = MaxViT(
        img_size=img_size,
        depths=(2, 2, 5, 2),
        channels=(96, 192, 384, 768),
        embed_dim=64,
        num_heads=32,
        grid_window_size=7,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def maxvit_base(pretrained=False, img_size=224, **kwargs):
    """ MaxViT-Base for 224x224 input """
    model = MaxViT(
        img_size=img_size,
        depths=(2, 6, 14, 2),
        channels=(96, 192, 384, 768),
        embed_dim=64,
        num_heads=32,
        grid_window_size=7,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def maxvit_large(pretrained=False, img_size=224, **kwargs):
    """ MaxViT-Large for 224x224 input """
    model = MaxViT(
        img_size=img_size,
        depths=(2, 6, 14, 2),
        channels=(128, 256, 512, 1024),
        embed_dim=128,
        num_heads=32,
        grid_window_size=7,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    img_h, img_w = 640, 640
    print("--- Creating MaxViT Tiny model ---")
    # You can choose other models like maxvit_small(), maxvit_base(), etc.
    model = maxvit_tiny(img_size=img_h)
    print("Model created successfully.")
    print("Calculated width_list:", model.width_list)

    # Test forward pass
    input_tensor = torch.rand(2, 3, img_h, img_w)
    print(f"\n--- Testing MaxViT Tiny forward pass (Input: {input_tensor.shape}) ---")

    model.eval()
    try:
        with torch.no_grad():
            output_features = model(input_tensor)
        print("Forward pass successful.")
        
        # Check if the output is a list
        assert isinstance(output_features, list), f"Output should be a list, but got {type(output_features)}"
        print(f"Output is a list with {len(output_features)} elements.")

        print("\nOutput feature shapes:")
        for i, features in enumerate(output_features):
            print(f"Stage {i+1}: {features.shape}")

        # Verify width_list matches runtime output
        runtime_widths = [f.size(1) for f in output_features]
        print("\nRuntime output feature channels:", runtime_widths)
        assert model.width_list == runtime_widths, "Width list mismatch!"
        print("Width list verified successfully.")
        
        # --- Test deepcopy ---
        print("\n--- Testing deepcopy ---")
        import copy
        copied_model = copy.deepcopy(model)
        print("Deepcopy successful.")

        # Optional: Test copied model forward pass
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