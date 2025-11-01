import torch.nn as nn
import torch
from inspect import signature
from typing import Any, Optional
from torch.cuda.amp import autocast
import torch.nn.functional as F

def val2list(x: list | tuple | Any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)

class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res
    
 
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out
 
 
REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
}
 
# register activation function here
REGISTERED_ACT_DICT: dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
}
 
 
class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
 
        mid_channels = mid_channels or round(in_channels * expand_ratio)
 
        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x
 
 
class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()
 
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
 
        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x
 
 
class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()
 
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)
 
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x
 
 
class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        norm="bn2d",
        act_func="hswish",
    ):
        super(EfficientViTBlock, self).__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
            ),
            IdentityLayer(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x
 
class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
 
        mid_channels = mid_channels or round(in_channels * expand_ratio)
 
        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x
 
 
 
class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""
 
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu6",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)
 
        total_dim = heads * dim
 
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
 
        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)
 
        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )
 
    @autocast(enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())
 
        if qkv.dtype == torch.float16:
            qkv = qkv.float()
 
        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (
            qkv[..., 0 : self.dim],
            qkv[..., self.dim : 2 * self.dim],
            qkv[..., 2 * self.dim :],
        )
 
        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)
 
        # linear matmul
        trans_k = k.transpose(-1, -2)
 
        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = torch.clone(out)
        out = out[..., :-1] / (out[..., -1:] + self.eps)
 
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        return out
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        device, types = qkv.device, qkv.dtype
        for op in self.aggreg:
            if device.type == 'cuda' and types == torch.float32:
                qkv = qkv.to(torch.float16)
            x1 = op(qkv)
            multi_scale_qkv.append(x1)
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)
        out = self.relu_linear_att(multi_scale_qkv)
        out = self.proj(out)
        return out
 
    @staticmethod
    def configure_litemla(model: nn.Module, **kwargs) -> None:
        eps = kwargs.get("eps", None)
        for m in model.modules():
            if isinstance(m, LiteMLA):
                if eps is not None:
                    m.eps = eps
 
 
def build_kwargs_from_config(config: dict, target_func: callable) -> dict[str, any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs
 
 
 
def build_norm(name="bn2d", num_features=None, **kwargs) -> nn.Module or None:
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None
 
def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2
 
 
def build_act(name: str, **kwargs) -> nn.Module or None:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None
 
 
class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()
 
        padding = get_same_padding(kernel_size)
        padding *= dilation
 
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        device, type = x.device, x.dtype
        choose = False
        if device.type == 'cuda' and type == torch.float32:
            x = x.to(torch.float16)
            choose = True
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        if choose:
            x = x.to(torch.float16)
        return x
 
 
class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
 
 
class OpSequential(nn.Module):
    def __init__(self, op_list: list[nn.Module or None]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x
 
 
 
class EfficientViTBackbone(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="ln2d",
        act_func="hswish",
    ) -> None:
        super().__init__()
 
        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)
 
        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
 
        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w
 
            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)
 
    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block
 
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = []
        for stage_id, stage in enumerate(self.stages):
            x = stage(x)
            if x.device.type == 'cuda':
                x = x.to(torch.float16)
            outputs.append(x)
        return outputs
 
 
def efficientvit_backbone_b0(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[3, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone
 
def efficientvit_backbone_b1(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[3, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone
 
 
def efficientvit_backbone_b2(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[3, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone
 
 
def efficientvit_backbone_b3(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[3, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


