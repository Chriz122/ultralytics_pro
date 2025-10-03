import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from typing import List, Tuple, Dict, Any, Union # Added for type hinting

# --- Configuration for FANet Variants ---
FANET_SPECS: Dict[str, Dict[str, Any]] = {
    'fanet_tiny': {
        'depths': [2, 2, 8, 2],
        'dims': [96, 192, 384, 768], # [96, 192, 384, 768]
        'drop_path_rate': 0.1,
        'expan_ratio': 4,
        'kernel_sizes': [5, 5, 3, 3],
    },
    'fanet_small': { # Example for another variant
        'depths': [3, 3, 12, 3],
        'dims': [96, 192, 384, 768],
        'drop_path_rate': 0.2,
        'expan_ratio': 4,
        'kernel_sizes': [5, 5, 3, 3],
    },
    # Add more variants as needed
}

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class FeatureRefinementModule(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, down_kernel=5, down_stride=4):
        super().__init__()

        self.lconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.hconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.norm1 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.down = nn.Conv2d(in_dim, in_dim, kernel_size=down_kernel, stride=down_stride, padding=down_kernel//2, groups=in_dim)
        self.proj = nn.Conv2d(in_dim*2, out_dim, kernel_size=1, stride=1, padding=0)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B,C,H,W = x.shape

        dx = self.down(x)
        udx = F.interpolate(dx, size=(H,W), mode='bilinear', align_corners=False)
        lx = self.norm1(self.lconv(self.act(x * udx)))
        hx = self.norm2(self.hconv(self.act(x - udx)))

        out = self.act(self.proj(torch.cat([lx, hx], dim=1))) 
        return out


class AFE(nn.Module): # Attentive Feature Enhancement
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim//2, 1, padding=0)
        self.proj2 = nn.Conv2d(dim, dim, 1, padding=0) # Takes concat of ctx and enh_x, so input is (dim//2 + dim//2) = dim

        self.ctx_conv = nn.Conv2d(dim//2, dim//2, kernel_size=7, padding=3, groups=dim//2 if dim//2 % 4 ==0 and dim//2 > 0 else 1) # Ensure groups <= in_channels

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(dim//2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(dim//2, eps=1e-6, data_format="channels_first")

        self.enhance = FeatureRefinementModule(in_dim=dim//2, out_dim=dim//2, down_kernel=3, down_stride=2)        
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        
        x_res = x
        x = self.act(self.norm1(self.dwconv(x))) # Original: x = x + self.norm1(self.act(self.dwconv(x)))
        x = x + x_res # Apply residual connection after norm and act

        x = self.norm2(self.act(self.proj1(x))) # Now x has dim//2 channels
        
        ctx = self.norm3(self.act(self.ctx_conv(x)))

        enh_x = self.enhance(x) # enhance takes dim//2 and outputs dim//2
        
        # Concatenate ctx (dim//2) and enh_x (dim//2)
        x_cat = torch.cat([ctx, enh_x], dim=1) # Shape: B, dim, H, W
        x = self.act(self.proj2(x_cat)) # proj2 takes dim and outputs dim

        return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.1, expan_ratio=4,
                 kernel_size=3, use_dilated_mlp=False): # Added use_dilated_mlp
        super().__init__()
        
        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        if use_dilated_mlp: # Not used by default in FANET_SPECS for tiny
            self.mlp = AtrousMLP(dim=dim, mlp_ratio=expan_ratio)
        else:
            self.mlp = MLP(dim=dim, mlp_ratio=expan_ratio)
        self.attn = AFE(dim, kernel_size=kernel_size)

        self.drop_path_1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        # B, C, H, W = x.shape # Not needed here

        inp_copy = x
        x = self.layer_norm1(inp_copy)
        x = self.drop_path_1(self.attn(x))
        x_attn_out = x + inp_copy # First residual

        x = self.layer_norm2(x_attn_out)
        x = self.drop_path_2(self.mlp(x))
        out = x_attn_out + x # Second residual

        return out


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4): # Removed use_dcn as it's not in original FANet
        super().__init__()
        
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        # B, C, H, W = x.shape # Not needed here

        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x)) # Element-wise sum
        x = self.fc2(x)

        return x


class AtrousMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        
        hidden_dim = dim * mlp_ratio
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        # For AtrousMLP, the concatenated features (x1, x2) should sum up to hidden_dim
        # So, pos1 and pos2 should output hidden_dim / 2 each
        # And their groups should also be hidden_dim / 2
        # Original code has dim*2, which might be a typo if hidden_dim = dim * mlp_ratio (typically mlp_ratio=4)
        # Let's assume the original intent was for pos1 and pos2 to operate on half of the hidden_dim channels each effectively
        # Or, if hidden_dim is the target for concatenation, then pos1 and pos2 output hidden_dim/2
        
        # Correcting AtrousMLP based on typical patterns:
        # fc1 expands to hidden_dim. Then this hidden_dim is split or processed.
        # The original implementation splits hidden_dim into two paths (dim*2 each)
        # which means mlp_ratio must be 4 for this to make sense (hidden_dim = dim*4, each path gets dim*2).

        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be divisible by 2 for AtrousMLP's split")
        
        self.pos1_out_channels = hidden_dim // 2
        self.pos2_out_channels = hidden_dim // 2
        
        self.pos1 = nn.Conv2d(hidden_dim, self.pos1_out_channels, 3, padding=1, groups=self.pos1_out_channels)
        self.pos2 = nn.Conv2d(hidden_dim, self.pos2_out_channels, 3, padding=2, dilation=2, groups=self.pos2_out_channels)
        
        self.fc2 = nn.Conv2d(self.pos1_out_channels + self.pos2_out_channels, dim, 1) # input is concatenation
        self.act = nn.GELU()

    def forward(self, x):
        # B, C, H, W = x.shape # Not needed here
        
        x = self.act(self.fc1(x)) # x is now [B, hidden_dim, H, W]
        
        # The original AtrousMLP implementation implies pos1 and pos2 take the *same* input 'x' (expanded by fc1)
        # and their outputs are concatenated.
        x1 = self.act(self.pos1(x)) # Output: B, hidden_dim/2, H, W
        x2 = self.act(self.pos2(x)) # Output: B, hidden_dim/2, H, W
        
        x_a = torch.cat([x1,x2], dim=1) # Concatenated: B, hidden_dim, H, W
        x = self.fc2(x_a)

        return x


class FANet(nn.Module):
    def __init__(self, 
                 model_name: str,
                 in_chans: int = 3, 
                 input_size: Union[int, Tuple[int, int]] = 224, # H or (H,W)
                 **kwargs): # Allow other kwargs like num_classes if a head is re-added
        super().__init__()

        if model_name not in FANET_SPECS:
            raise ValueError(f"Unknown model_name: {model_name}. Available: {list(FANET_SPECS.keys())}")
        
        spec = FANET_SPECS[model_name]
        depths = spec['depths']
        dims = spec['dims']
        drop_path_rate = spec['drop_path_rate']
        expan_ratio = spec['expan_ratio']
        kernel_sizes = spec['kernel_sizes']
        # use_dilated_mlp is not in tiny spec, default to False for Block
        # if you add it to specs, you can retrieve it:
        # use_dilated_mlp_stages = spec.get('use_dilated_mlp_stages', [False]*4) 

        self.model_name = model_name
        self.in_chans = in_chans
        if isinstance(input_size, int):
            self.input_h_w = (input_size, input_size)
        elif isinstance(input_size, tuple) and len(input_size) == 2:
            self.input_h_w = input_size
        else:
            raise ValueError(f"input_size must be int or tuple of 2 ints, got {input_size}")

        self.downsample_layers = nn.ModuleList()
        stem = nn.Conv2d(in_chans, dims[0], kernel_size=5, stride=4, padding=2) # padding = (kernel_size-1)//2 if stride=1
                                                                               # For stride 4, output H/4. E.g. 224 -> 56
                                                                               # (224 - 5 + 2*2)/4 + 1 = (223)/4 + 1 = 55 + 1 = 56. Padding is correct.
        self.downsample_layers.append(stem)

        for i in range(3): # 3 more downsampling layers
            downsample_layer = nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1)
                                                                                # (H - 3 + 2*1)/2 + 1 = (H-1)/2 + 1. Output H/2. Correct.
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4): # 4 stages
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                                          expan_ratio=expan_ratio, kernel_size=kernel_sizes[i],
                                          use_dilated_mlp=False) # Defaulting to False, can be made configurable per stage
                                    )
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]

        # Classification head (commented out as per original, for backbone usage)
        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        # self.head = nn.Linear(dims[-1], num_classes) # num_classes would need to be an arg

        self.apply(self._init_weights) # Initialize weights

        # Calculate width_list
        self.width_list: List[int] = []
        self.eval() # Set to eval mode for dummy pass; affects dropout, batchnorm
        try:
            dummy_input = torch.randn(1, self.in_chans, *self.input_h_w)
            features = self.forward_features(dummy_input)
            self.width_list = [f.size(1) for f in features]
        except Exception as e:
            print(f"Warning: Error during dummy forward pass for FANet width_list: {e}.")
            # Fallback: use dims directly, assuming forward_features returns one feature per stage
            self.width_list = list(dims) 
            print(f"Falling back to width_list: {self.width_list}")
        self.train() # Set back to train mode

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)): # Handles custom LayerNorm and nn.LayerNorm
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d): # Added for completeness if BatchNorm were used elsewhere
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


    def init_weights(self, pretrained: str = None):
        if pretrained is not None:
            try:
                checkpoint = torch.load(pretrained, map_location="cpu")
                state_dict_key = "state_dict" if "state_dict" in checkpoint else "model" # Common keys
                if state_dict_key in checkpoint:
                    state_dict = checkpoint[state_dict_key]
                else: # Try if checkpoint is the state_dict itself
                    state_dict = checkpoint 
                
                # Filter out unnecessary keys (e.g. classifier head if not present)
                # and adapt keys if necessary (e.g. remove 'module.' prefix from DataParallel)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                
                msg = self.load_state_dict(new_state_dict, strict=False)
                print(f"Pretrained weights loaded from {pretrained}. Load message: {msg}")
            except Exception as e:
                print(f"Error loading pretrained weights from {pretrained}: {e}")
        else:
            # Weights are already initialized by self.apply(self._init_weights) in __init__
            print("No pretrained weights provided, using random initialization.")


    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        x = self.downsample_layers[0](x) # Stem
        x = self.stages[0](x)
        feats.append(x) # Feature from stage 0

        for i in range(1, 4): # For stages 1, 2, 3
            x = self.downsample_layers[i](x) # Downsample
            x = self.stages[i](x)           # Pass through blocks
            feats.append(x)
        return feats # Returns a list of 4 feature maps

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # The forward_features method already returns a list of tensors.
        # The original code unpacks and repacks, which is fine but redundant.
        # Directly returning the list from forward_features is cleaner.
        return self.forward_features(x)


# --- Factory Functions ---
def fanet_tiny(input_size: Tuple[int, int, int] = (3, 224, 224), pretrained: str = None, **kwargs) -> FANet:
    """
    Args:
        input_size (Tuple[int, int, int]): Input image size (channels, height, width).
        pretrained (str, optional): Path to pre-trained weights.
    """
    in_chans = input_size[0]
    img_h_w = (input_size[1], input_size[2])
    model = FANet(model_name='fanet_tiny', in_chans=in_chans, input_size=img_h_w, **kwargs)
    if pretrained:
        model.init_weights(pretrained)
    return model

def fanet_small(input_size: Tuple[int, int, int] = (3, 224, 224), pretrained: str = None, **kwargs) -> FANet:
    in_chans = input_size[0]
    img_h_w = (input_size[1], input_size[2])
    model = FANet(model_name='fanet_small', in_chans=in_chans, input_size=img_h_w, **kwargs)
    if pretrained:
        model.init_weights(pretrained)
    return model

# --- Example Usage (for testing the module itself) ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_input_size_config = (3, 224, 224) # C, H, W
    dummy_image = torch.randn(2, *test_input_size_config).to(device) # Batch size 2

    print(f"\n--- Testing FANet Tiny ---")
    # Pass input_size tuple directly to the factory function
    model_tiny = fanet_tiny(input_size=test_input_size_config).to(device)
    model_tiny.eval()

    print(f"FANet Tiny width_list (from __init__): {model_tiny.width_list}")

    with torch.no_grad():
        feature_maps = model_tiny(dummy_image) # Calls FANet.forward()

    print(f"FANet Tiny forward() produced {len(feature_maps)} feature maps (output type: {type(feature_maps)}):")
    for i, fm in enumerate(feature_maps):
        print(f"  Feature map {i} shape: {fm.shape}, Channels: {fm.size(1)}")

    assert isinstance(feature_maps, list), "Output is not a list!"
    print("Output is a list: True")

    assert len(model_tiny.width_list) == len(feature_maps), \
        f"Mismatch: width_list len {len(model_tiny.width_list)} vs num feature maps {len(feature_maps)}"
    
    all_channels_match = True
    for i in range(len(feature_maps)):
        if model_tiny.width_list[i] != feature_maps[i].size(1):
            print(f"Mismatch in channel count for feature map {i}: width_list says {model_tiny.width_list[i]}, actual is {feature_maps[i].size(1)}")
            all_channels_match = False
    
    if all_channels_match:
        print("Width_list channels match actual feature map channels: True")
    else:
        print("ERROR: Width_list channels DO NOT match actual feature map channels.")


    print(f"\n--- Testing FANet Small (example with different size) ---")
    test_input_size_large_config = (3, 384, 384)
    dummy_image_large = torch.randn(1, *test_input_size_large_config).to(device)
    model_small = fanet_small(input_size=test_input_size_large_config).to(device)
    model_small.eval()
    print(f"FANet Small width_list (from __init__): {model_small.width_list}")
    with torch.no_grad():
        feature_maps_large = model_small(dummy_image_large)
    print(f"FANet Small (large input) produced {len(feature_maps_large)} feature maps:")
    for i, fm in enumerate(feature_maps_large):
        print(f"  Feature map {i} shape: {fm.shape}")
    assert isinstance(feature_maps_large, list), "Output for large FANet is not a list!"


    # Test AFE groups parameter more robustly
    print("\n--- Testing AFE with various dimensions ---")
    try:
        afe_test1 = AFE(dim=10) # dim//2 = 5, 5%4 != 0
        print("AFE(dim=10) created with groups=1 for ctx_conv.")
        test_tensor_afe1 = torch.randn(1, 10, 32, 32)
        out_afe1 = afe_test1(test_tensor_afe1)
        print(f"AFE(dim=10) output shape: {out_afe1.shape}")

        afe_test2 = AFE(dim=16) # dim//2 = 8, 8%4 == 0
        print("AFE(dim=16) created with groups=dim//2 for ctx_conv.")
        test_tensor_afe2 = torch.randn(1, 16, 32, 32)
        out_afe2 = afe_test2(test_tensor_afe2)
        print(f"AFE(dim=16) output shape: {out_afe2.shape}")

    except Exception as e:
        print(f"Error during AFE test: {e}")


    print("\nAll FANet tests completed.")