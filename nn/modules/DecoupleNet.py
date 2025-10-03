import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from typing import List, Tuple, Dict, Any
from torch import Tensor
import copy
import antialiased_cnns # Make sure this is installed: pip install antialiased-cnns
import torch.nn.functional as F

# --- Configuration for DecoupleNet Variants ---
DECOUPLE_NET_SPECS: Dict[str, Dict[str, Any]] = {
    'decouplenet_d0': {
        'embed_dim': 32,
        'depths': (1, 6, 6, 2),
        'att_kernel': (9, 9, 9, 9),
        'drop_path_rate': 0.1,
        'feature_dim': 1280,
    },
    'decouplenet_d1': {
        'embed_dim': 48,
        'depths': (1, 6, 6, 2),
        'att_kernel': (9, 9, 9, 9),
        'drop_path_rate': 0.15,
        'feature_dim': 1280,
    },
    'decouplenet_d2': {
        'embed_dim': 64,
        'depths': (1, 6, 6, 2),
        'att_kernel': (9, 9, 9, 9),
        'drop_path_rate': 0.2,
        'feature_dim': 1280,
    },
}

def _cfg(url='', **kwargs): # Minimal cfg for potential timm compatibility
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head', **kwargs
    }

default_cfgs = {
    'decouplenet_d0': _cfg(),
    'decouplenet_d1': _cfg(),
    'decouplenet_d2': _cfg(),
}

class FID(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.Gconv = nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.pii = PII(dim*2, 8)
        self.conv_D = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=2, padding=1, groups=dim*2)
        self.act = nn.GELU()
        self.batch_norm_c = nn.BatchNorm2d(dim*2)
        self.max_m1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.max_m2 = antialiased_cnns.BlurPool(dim*2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(dim*2)
        self.fusion = nn.Conv2d(dim*4, self.outdim, kernel_size=1, stride=1)

    def forward(self, x):
        x_branch = self.Gconv(x)
        x_branch = self.pii(x_branch)
        max_branch = self.max_m1(x_branch)
        max_branch = self.max_m2(max_branch)
        max_branch = self.batch_norm_m(max_branch)
        conv_branch = self.conv_D(x_branch)
        conv_branch = self.act(conv_branch)
        conv_branch = self.batch_norm_c(conv_branch)
        x_out = torch.cat([conv_branch, max_branch], dim=1)
        x_out = self.fusion(x_out)
        return x_out

class PII(nn.Module):
    def __init__(self, dim, n_div): # dim is the input channel to PII (e.g., dim*2 from FID's Gconv)
        super().__init__()
        # Defines how input 'x' to PII's forward is split
        self.dim_conv_effective = dim // n_div # Size of each of the two parts that get convolved
        self.dim_untouched_effective = (dim // 2) - self.dim_conv_effective # Size of each of the two untouched parts

        # The conv layer takes the concatenation of the two 'dim_conv_effective' parts
        self.conv = nn.Conv2d(self.dim_conv_effective * 2, self.dim_conv_effective * 2, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x has 'dim' channels
        # Split x into four parts: x1_c, x1_u, x2_c, x2_u
        # x1_c and x2_c are the 'conv' parts (each of size self.dim_conv_effective)
        # x1_u and x2_u are the 'untouched' parts (each of size self.dim_untouched_effective)
        
        current_dim = x.size(1)
        expected_sum_split = 2 * self.dim_conv_effective + 2 * self.dim_untouched_effective
        
        if current_dim != expected_sum_split:
            # This case should ideally not happen if 'dim' for PII is chosen carefully.
            # For robustness, adjust the second untouched part if there's a slight mismatch.
            # This can happen if 'dim' is not perfectly divisible by 'n_div' or by 2 in a way that aligns.
            # Let's assume dim_conv_effective and the first dim_untouched_effective are primary.
            dim_untouched_last = current_dim - (2 * self.dim_conv_effective + self.dim_untouched_effective)
            if dim_untouched_last < 0: # Should not happen with typical positive dimensions
                 raise ValueError(f"PII split calculation resulted in negative dimension: {dim_untouched_last}")
            split_sizes = [self.dim_conv_effective, self.dim_untouched_effective, 
                           self.dim_conv_effective, dim_untouched_last]
        else:
            split_sizes = [self.dim_conv_effective, self.dim_untouched_effective,
                           self.dim_conv_effective, self.dim_untouched_effective]

        x1_c, x1_u, x2_c, x2_u = torch.split(x, split_sizes, dim=1)
        
        x_to_conv = torch.cat((x1_c, x2_c), 1) # Concatenate the two parts intended for convolution
        x_convolved = self.conv(x_to_conv)     # Convolve them
        
        # Reconstruct: convolved_parts, first_untouched_part, second_untouched_part
        x_out = torch.cat((x_convolved, x1_u, x2_u), 1)
        return x_out

class MRLA(nn.Module):
    def __init__(self, channel, att_kernel):
        super(MRLA, self).__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        channels12 = int(channel / 2) # Ensure integer
        self.primary_conv = nn.Sequential(
            nn.Conv2d(channel, channels12, 1, 1, bias=False),
            nn.BatchNorm2d(channels12), nn.GELU(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(channels12, channels12, 3, 1, 1, groups=channels12, bias=False),
            nn.BatchNorm2d(channels12), nn.GELU(),
        )
        self.init = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, bias=False), nn.BatchNorm2d(channel),
        )
        self.H_att = nn.Conv2d(channel, channel, (att_kernel, 1), 1, (att_padding, 0), groups=channel, bias=False)
        self.V_att = nn.Conv2d(channel, channel, (1, att_kernel), 1, (0, att_padding), groups=channel, bias=False)
        self.batchnorm = nn.BatchNorm2d(channel)

    def forward(self, x):
        x_tem = self.init(F.avg_pool2d(x, kernel_size=2, stride=2))
        x_h, x_w = self.H_att(x_tem), self.V_att(x_tem)
        mrla = self.batchnorm(x_h + x_w)
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1) # out has 'channel' channels
        out = out * F.interpolate(self.gate_fn(mrla), size=(out.shape[-2], out.shape[-1]), mode='nearest')
        return out

class GA(nn.Module):
    def __init__(self, dim, head_dim=4, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0: self.num_heads = 1
        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1)
        N = H * W
        qkv = self.qkv(x_perm).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x_out = self.proj(x_attn)
        x_out = self.proj_drop(x_out)
        return x_out.permute(0, 3, 1, 2)

class MBFD(nn.Module):
    def __init__(self, dim, stage, att_kernel):
        super().__init__()
        self.dim, self.stage = dim, stage
        self.dim_learn = dim // 4
        self.dim_untouched = dim - 2 * self.dim_learn
        self.Conv = nn.Conv2d(self.dim_learn, self.dim_learn, 3, 1, 1, bias=False)
        self.MRLA = MRLA(self.dim_learn, att_kernel)
        if stage > 2:
            self.GA = GA(self.dim_untouched)
            self.norm = nn.BatchNorm2d(self.dim_untouched)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2, x3 = torch.split(x, [self.dim_learn, self.dim_learn, self.dim_untouched], dim=1)
        x1_out, x2_out = self.Conv(x1), self.MRLA(x2)
        x3_out = self.norm(x3 + self.GA(x3)) if self.stage > 2 else x3
        return torch.cat((x1_out, x2_out, x3_out), 1)

class MLPBlock(nn.Module):
    def __init__(self, dim, stage, att_kernel, mlp_ratio, drop_path, layer_scale_init_value, act_layer, norm_layer):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False), norm_layer(mlp_hidden_dim), act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        )
        self.MBFD = MBFD(dim, stage, att_kernel)
        self.use_layer_scale = layer_scale_init_value > 0
        if self.use_layer_scale:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x_mbfd = self.MBFD(x)
        x_mlp = self.mlp(x_mbfd)
        if self.use_layer_scale:
            x_mlp = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x_mlp
        return shortcut + self.drop_path(x_mlp)

class BasicStage(nn.Module):
    def __init__(self, dim, stage, depth, att_kernel, mlp_ratio, drop_path, layer_scale_init_value, norm_layer, act_layer):
        super().__init__()
        self.blocks = nn.Sequential(*(
            MLPBlock(dim, stage, att_kernel, mlp_ratio, drop_path[i], layer_scale_init_value, act_layer, norm_layer)
            for i in range(depth)))
    def forward(self, x: Tensor) -> Tensor: return self.blocks(x)

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x: Tensor) -> Tensor: return self.norm(self.proj(x))

class DecoupleNet(nn.Module):
    def __init__(self,
                 model_name: str,
                 in_chans: int = 3,
                 num_classes: int = 1000, # For get_classification_output's head
                 input_size: Tuple[int, int, int] = (3, 224, 224),
                 mlp_ratio: float = 2.,
                 patch_size: int = 4,
                 patch_stride: int = 4,
                 patch_norm: bool = True,
                 layer_scale_init_value: float = 0,
                 # fork_feat is effectively True for forward(), but kept for clarity in __init__ logic
                 fork_feat_setup: bool = True, # Used to control norm layer setup and width_list source
                 **kwargs):
        super().__init__()

        if model_name not in DECOUPLE_NET_SPECS:
            raise ValueError(f"Unknown model_name: {model_name}. Available: {list(DECOUPLE_NET_SPECS.keys())}")
        
        spec = DECOUPLE_NET_SPECS[model_name]
        embed_dim = spec['embed_dim']
        depths = spec['depths']
        att_kernel = spec['att_kernel']
        drop_path_rate = spec['drop_path_rate']
        classifier_feature_dim = spec['feature_dim']

        self.model_name = model_name
        self.in_chans = in_chans
        self.num_classes = num_classes # For the classification head
        self.input_size_for_dummy_pass = input_size

        norm_layer = nn.BatchNorm2d
        act_layer = nn.GELU

        self.num_stages = len(depths)
        self.num_features_before_cls_head = int(embed_dim * 2 ** (self.num_stages - 1))

        self.patch_embed = PatchEmbed(patch_size, patch_stride, in_chans, embed_dim, norm_layer if patch_norm else None)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        current_dim = embed_dim
        for i_stage in range(self.num_stages):
            stage_module = BasicStage(
                current_dim, i_stage, depths[i_stage], att_kernel[i_stage], mlp_ratio,
                dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                layer_scale_init_value, norm_layer, act_layer
            )
            self.stages.append(stage_module)
            if i_stage < self.num_stages - 1:
                self.stages.append(FID(dim=current_dim))
                current_dim *= 2
        
        # Classifier head parts (used by get_classification_output)
        self.avgpool_pre_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.num_features_before_cls_head, classifier_feature_dim, 1, bias=False),
            act_layer()
        )
        self.head = nn.Linear(classifier_feature_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Norm layers for feature list (if fork_feat_setup is True)
        self.out_indices_for_features = []
        if fork_feat_setup:
            _out_idx_counter = 0
            for i, stage_module_in_list in enumerate(self.stages):
                if isinstance(stage_module_in_list, BasicStage):
                    self.out_indices_for_features.append(i)
                    # Dim for norm layer is output dim of that BasicStage
                    norm_out_dim = int(embed_dim * 2**_out_idx_counter)
                    self.add_module(f'norm_feat_stage{i}', norm_layer(norm_out_dim))
                    _out_idx_counter += 1
        
        self.apply(self._init_weights)

        # Calculate width_list (channels of features returned by forward())
        self.width_list: List[int] = []
        self.eval()
        try:
            dummy_input = torch.randn(1, *self.input_size_for_dummy_pass)
            features = self.forward_features_list(dummy_input) # This is what forward() will return
            self.width_list = [f.size(1) for f in features]
        except Exception as e:
            print(f"Warning: Error during dummy forward pass for width_list: {e}. Falling back.")
            _potential_widths = [embed_dim] # After patch_embed
            _c_dim_tracker = embed_dim
            for i in range(self.num_stages):
                _potential_widths.append(_c_dim_tracker)
                if i < self.num_stages -1: _c_dim_tracker *=2
            # Only take as many as forward_features_list would produce
            num_expected_features = 1 + len(self.out_indices_for_features) if fork_feat_setup else 1
            self.width_list = _potential_widths[:num_expected_features]
        self.train()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward_features_list(self, x: Tensor) -> List[Tensor]:
        x_embed = self.patch_embed(x)
        outs = [x_embed]
        current_x = x_embed
        for idx, stage_module in enumerate(self.stages):
            current_x = stage_module(current_x)
            if idx in self.out_indices_for_features: # Output of a BasicStage
                norm_layer = getattr(self, f'norm_feat_stage{idx}')
                outs.append(norm_layer(current_x))
        return outs

    # Main forward method for compatibility with frameworks like Ultralytics
    def forward(self, x: Tensor) -> List[Tensor]:
        return self.forward_features_list(x)

    # Helper method for standalone classification
    def get_classification_output(self, x: Tensor) -> Tensor:
        # Pass input through patch_embed and all stages to get final feature map
        current_x = self.patch_embed(x)
        for stage_module in self.stages:
            current_x = stage_module(current_x)
        
        # Now current_x is the output of the final stage (e.g., BS3)
        # This is self.num_features_before_cls_head dimensional
        out = self.avgpool_pre_head(current_x)
        out = torch.flatten(out, 1)
        out = self.head(out)
        return out

# --- Factory Functions ---
def decouplenet_d0(num_classes: int = 1000, input_size=(3,224,224), **kwargs):
    model = DecoupleNet(model_name='decouplenet_d0', num_classes=num_classes, input_size=input_size, **kwargs)
    model.default_cfg = default_cfgs['decouplenet_d0']
    return model

def decouplenet_d1(num_classes: int = 1000, input_size=(3,224,224), **kwargs):
    model = DecoupleNet(model_name='decouplenet_d1', num_classes=num_classes, input_size=input_size, **kwargs)
    model.default_cfg = default_cfgs['decouplenet_d1']
    return model

def decouplenet_d2(num_classes: int = 1000, input_size=(3,224,224), **kwargs):
    model = DecoupleNet(model_name='decouplenet_d2', num_classes=num_classes, input_size=input_size, **kwargs)
    model.default_cfg = default_cfgs['decouplenet_d2']
    return model

# --- Example Usage (for testing the module itself) ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_input_size_config = (3, 224, 224)
    dummy_image = torch.randn(2, *test_input_size_config).to(device)

    print(f"\n--- Testing DecoupleNet D0 ---")
    # When used with ultralytics, num_classes for the DecoupleNet itself might be for its aux head.
    # Ultralytics will add its own detection/segmentation head.
    model_d0 = decouplenet_d0(num_classes=10, input_size=test_input_size_config).to(device)
    model_d0.eval()
    
    # Test forward() (should return list of features)
    with torch.no_grad():
        feature_maps = model_d0(dummy_image)
    print(f"DecoupleNet D0 forward() produced {len(feature_maps)} feature maps:")
    for i, fm in enumerate(feature_maps):
        print(f"  Feature map {i} shape: {fm.shape}, Channels: {fm.size(1)}")
    print(f"DecoupleNet D0 width_list: {model_d0.width_list}")
    assert len(model_d0.width_list) == len(feature_maps), "Mismatch between width_list and number of feature maps"
    for i in range(len(feature_maps)):
        assert model_d0.width_list[i] == feature_maps[i].size(1), f"Mismatch in channel count for feature map {i}"
    print("Width_list matches feature map channels: True")

    # Test get_classification_output()
    with torch.no_grad():
        cls_output = model_d0.get_classification_output(dummy_image)
    print(f"DecoupleNet D0 get_classification_output() shape: {cls_output.shape}") # Should be [B, num_classes_for_head]
    assert cls_output.shape == (dummy_image.size(0), 10)

    print("\n--- Testing DecoupleNet D1 (different input size for dummy pass) ---")
    test_input_size_large_config = (3, 384, 384)
    dummy_image_large = torch.randn(1, *test_input_size_large_config).to(device)
    model_d1_large = decouplenet_d1(num_classes=5, input_size=test_input_size_large_config).to(device)
    model_d1_large.eval()
    with torch.no_grad():
        feature_maps_large = model_d1_large(dummy_image_large)
    print(f"DecoupleNet D1 (large input) produced {len(feature_maps_large)} feature maps:")
    for i, fm in enumerate(feature_maps_large):
        print(f"  Feature map {i} shape: {fm.shape}")
    print(f"DecoupleNet D1 (large input) width_list: {model_d1_large.width_list}")

    print("\nAll tests passed.")