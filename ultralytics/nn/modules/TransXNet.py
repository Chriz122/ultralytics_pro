import os
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from timm.models.layers import DropPath, to_2tuple

# Helper function for GroupNorm, making it consistent
def build_group_norm(num_channels, num_groups=1, eps=1e-6):
    """Creates a GroupNorm layer. num_groups=1 is equivalent to LayerNorm for CNNs."""
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps)

class PatchEmbed(nn.Module):
    """Patch Embedding module implemented by a layer of convolution."""
    def __init__(self,
                 patch_size=16,
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768,
                 use_norm=True):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = build_group_norm(embed_dim) if use_norm else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class Attention(nn.Module):  ### OSRA
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1,):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio + 3, stride=sr_ratio, padding=(sr_ratio + 3) // 2, groups=dim, bias=False),
                build_group_norm(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=False),
                build_group_norm(dim),
            )
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C//self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:],
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)

class DynamicConv2d(nn.Module): ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=1,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        
        # ***** FIX: Replaced BatchNorm2d with GroupNorm *****
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim // reduction_ratio, kernel_size=1),
            build_group_norm(dim // reduction_ratio), # <-- FIX
            nn.GELU(),
            nn.Conv2d(dim//reduction_ratio, dim*num_groups, kernel_size=1),
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            # This is the line that causes the error with BatchNorm
            scale_bias = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale_bias = torch.softmax(scale_bias.reshape(B, self.num_groups, C), dim=1)
            bias = scale_bias * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K//2,
                     groups=B*C,
                     bias=bias)

        return x.reshape(B, C, H, W)

class HybridTokenMixer(nn.Module): ### D-Mixer
    def __init__(self,
                 dim,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 reduction_ratio=8):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv2d(
            dim=dim//2, kernel_size=kernel_size, num_groups=num_groups)
        self.global_unit = Attention(
            dim=dim//2, num_heads=num_heads, sr_ratio=sr_ratio)

        inner_dim = max(16, dim//reduction_ratio)
        # ***** FIX: Replaced BatchNorm2d with GroupNorm *****
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            build_group_norm(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            build_group_norm(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            build_group_norm(dim),
        )

    def forward(self, x, relative_pos_enc=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2, relative_pos_enc)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x ## STE
        return x

class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i]//2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x_splits = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x_splits):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x

class Mlp(nn.Module):  ### MS-FFN
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # ***** FIX: Replaced BatchNorm2d with GroupNorm *****
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.GELU(),
            build_group_norm(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = nn.GELU()
        self.norm = build_group_norm(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=False),
            build_group_norm(out_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value,
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x

class Block(nn.Module):
    def __init__(self,
                 dim=64,
                 kernel_size=3,
                 sr_ratio=1,
                 num_groups=2,
                 num_heads=1,
                 mlp_ratio=4,
                 drop=0,
                 drop_path=0,
                 layer_scale_init_value=1e-5,
                 grad_checkpoint=False):

        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm1 = build_group_norm(dim)
        self.token_mixer = HybridTokenMixer(dim,
                                            kernel_size=kernel_size,
                                            num_groups=num_groups,
                                            num_heads=num_heads,
                                            sr_ratio=sr_ratio)
        self.norm2 = build_group_norm(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       out_features=dim,
                       drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(dim, layer_scale_init_value)
            self.layer_scale_2 = LayerScale(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def _forward_impl(self, x, relative_pos_enc=None):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.layer_scale_1(
                self.token_mixer(self.norm1(x), relative_pos_enc)))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))
        return x

    def forward(self, x, relative_pos_enc=None):
        if self.grad_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self._forward_impl, x, relative_pos_enc)
        else:
            x = self._forward_impl(x, relative_pos_enc)
        return x

def basic_blocks(dim,
                 index,
                 layers,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 mlp_ratio=4,
                 drop_rate=0,
                 drop_path_rate=0,
                 layer_scale_init_value=1e-5,
                 grad_checkpoint=False):

    blocks = nn.ModuleList()
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(
            Block(
                dim,
                kernel_size=kernel_size,
                num_groups=num_groups,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=block_dpr,
                layer_scale_init_value=layer_scale_init_value,
                grad_checkpoint=grad_checkpoint,
            ))
    return blocks

class TransXNet(nn.Module):
    arch_settings = {
        **dict.fromkeys(['t', 'tiny', 'T'],
                        {'layers': [3, 3, 9, 3],
                         'embed_dims': [48, 96, 224, 448],
                         'kernel_size': [7, 7, 7, 7],
                         'num_groups': [2, 2, 2, 2],
                         'sr_ratio': [8, 4, 2, 1],
                         'num_heads': [1, 2, 4, 8],
                         'mlp_ratios': [4, 4, 4, 4],
                         'layer_scale_init_value': 1e-5,}),

        **dict.fromkeys(['s', 'small', 'S'],
                        {'layers': [4, 4, 12, 4],
                         'embed_dims': [64, 128, 320, 512],
                         'kernel_size': [7, 7, 7, 7],
                         'num_groups': [2, 2, 3, 4],
                         'sr_ratio': [8, 4, 2, 1],
                         'num_heads': [1, 2, 5, 8],
                         'mlp_ratios': [6, 6, 4, 4],
                         'layer_scale_init_value': 1e-5,}),

        **dict.fromkeys(['b', 'base', 'B'],
                        {'layers': [4, 4, 21, 4],
                         'embed_dims': [76, 152, 336, 672],
                         'kernel_size': [7, 7, 7, 7],
                         'num_groups': [2, 2, 4, 4],
                         'sr_ratio': [8, 4, 2, 1],
                         'num_heads': [2, 4, 8, 16],
                         'mlp_ratios': [8, 8, 4, 4],
                         'layer_scale_init_value': 1e-5,}),
    }

    def __init__(self,
                 image_size=224,
                 arch='tiny',
                 in_chans=3,
                 num_classes=1000,
                 in_patch_size=7,
                 in_stride=4,
                 in_pad=3,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0,
                 drop_path_rate=0,
                 grad_checkpoint=False,
                 checkpoint_stage=[0] * 4,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.grad_checkpoint = grad_checkpoint

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from {set(self.arch_settings.keys())}'
            self.arch = arch.lower()
            arch_config = self.arch_settings[self.arch]
        else:
            arch_config = arch

        layers = arch_config['layers']
        embed_dims = arch_config['embed_dims']
        kernel_size = arch_config['kernel_size']
        num_groups = arch_config['num_groups']
        sr_ratio = arch_config['sr_ratio']
        num_heads = arch_config['num_heads']
        mlp_ratios = arch_config.get('mlp_ratios', [4, 4, 4, 4])
        layer_scale_init_value = arch_config.get('layer_scale_init_value', 1e-5)

        if not grad_checkpoint:
            checkpoint_stage = [0] * 4

        self.patch_embed = PatchEmbed(patch_size=in_patch_size,
                                      stride=in_stride,
                                      padding=in_pad,
                                      in_chans=in_chans,
                                      embed_dim=embed_dims[0])

        self.relative_pos_enc = []
        image_size_tuple = to_2tuple(image_size)
        current_size = [math.ceil(image_size_tuple[0] / in_stride),
                        math.ceil(image_size_tuple[1] / in_stride)]
        for i in range(4):
            num_patches = current_size[0] * current_size[1]
            sr_patches = math.ceil(current_size[0] / sr_ratio[i]) * math.ceil(current_size[1] / sr_ratio[i])
            self.relative_pos_enc.append(
                nn.Parameter(torch.zeros(1, num_heads[i], num_patches, sr_patches), requires_grad=True))
            current_size = [math.ceil(current_size[0] / 2), math.ceil(current_size[1] / 2)]
        self.relative_pos_enc = nn.ParameterList(self.relative_pos_enc)

        self.network = nn.ModuleList()
        self.out_indices = [0, 2, 4, 6]

        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i], i, layers, kernel_size=kernel_size[i], num_groups=num_groups[i],
                num_heads=num_heads[i], sr_ratio=sr_ratio[i], mlp_ratio=mlp_ratios[i],
                drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                grad_checkpoint=checkpoint_stage[i])
            self.network.append(stage)
            if i < len(layers) - 1:
                self.network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride, padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))

        for i_emb in embed_dims:
            layer = build_group_norm(i_emb)
            layer_name = f'norm{i_emb}' # Use a unique name
            self.add_module(layer_name, layer)
        
        # Add a norm layer for each output feature map, using unique names
        self.norm_out_layers = nn.ModuleList([build_group_norm(dim) for dim in embed_dims])

        self.classifier_head = nn.Sequential(
            build_group_norm(embed_dims[-1]),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1),
        ) if num_classes > 0 else nn.Identity()

        self.apply(self._init_model_weights)

        self.width_list = []
        try:
            self.eval()
            dummy_input = torch.randn(1, in_chans, image_size, image_size)
            with torch.no_grad():
                features = self.forward(dummy_input)
            self.width_list = [f.size(1) for f in features]
            self.train()
        except Exception as e:
            print(f"Error during dummy forward pass for width_list: {e}")
            print("Setting width_list to embed_dims as fallback.")
            self.width_list = embed_dims
            self.train()

    def _init_model_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_tokens(self, x):
        outs = []
        pos_idx = 0
        for idx, module in enumerate(self.network):
            if isinstance(module, nn.ModuleList): # This is a stage with blocks
                for blk in module:
                    x = blk(x, self.relative_pos_enc[pos_idx])
                
                # Apply the corresponding output norm and save the output
                norm_layer = self.norm_out_layers[pos_idx]
                outs.append(norm_layer(x))
                pos_idx += 1
            else: # This is a downsampling PatchEmbed layer
                x = module(x)
        return outs

    def forward(self, x):
        x = self.patch_embed(x)
        return self.forward_tokens(x)

def _filter_checkpoint(checkpoint, model):
    state_dict = checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']

    filtered_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if 'classifier' in k:
            continue
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
            filtered_dict[k] = v
        else:
            print(f"Warning: Skipping key {k} from checkpoint due to shape mismatch or not found.")
            
    return filtered_dict

# --- Factory Functions ---
def transxnet_tiny(pretrained=False, **kwargs):
    model = TransXNet(arch='t', **kwargs)
    if pretrained:
        url = 'https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-t.pth.tar'
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        state_dict = _filter_checkpoint(checkpoint, model)
        model.load_state_dict(state_dict, strict=False)
    return model

def transxnet_small(pretrained=False, **kwargs):
    model = TransXNet(arch='s', **kwargs)
    if pretrained:
        url = 'https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-s.pth.tar'
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        state_dict = _filter_checkpoint(checkpoint, model)
        model.load_state_dict(state_dict, strict=False)
    return model

def transxnet_base(pretrained=False, **kwargs):
    model = TransXNet(arch='b', **kwargs)
    if pretrained:
        url = 'https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-b.pth.tar'
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        state_dict = _filter_checkpoint(checkpoint, model)
        model.load_state_dict(state_dict, strict=False)
    return model

# Example of how to use it
if __name__ == '__main__':
    model = transxnet_tiny(pretrained=True, image_size=640)
    model.eval()

    print("\nCalculated width_list:", model.width_list)

    dummy_input = torch.randn(1, 3, 640, 640)
    features = model(dummy_input)

    print(f"\nOutput type: {type(features)}")
    print(f"Number of feature maps: {len(features)}")
    for i, f in enumerate(features):
        print(f"  Feature map {i+1} shape: {f.shape}")
        
    output_channels = [f.shape[1] for f in features]
    print("\nOutput feature channels:", output_channels)
    assert model.width_list == output_channels
    print("Assertion passed: width_list matches output channels.")