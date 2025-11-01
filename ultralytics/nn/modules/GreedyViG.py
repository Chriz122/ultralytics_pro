import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential as Seq

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD # No longer needed for backbone definition
from timm.models.layers import DropPath
# from timm.models.registry import register_model # No longer registering directly

import random
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Configuration Specs for GreedyViG Variants ---
GREEDYVIG_SPECS = {
    'GreedyViG_S': {
        'blocks': [[2, 2], [2, 2], [6, 2], [2, 2]],
        'channels': [48, 96, 192, 384],
        'K': [8, 4, 2, 1],
        # 'emb_dims': 768, # Removed as head is removed
    },
    'GreedyViG_M': {
        'blocks': [[3, 3], [3, 3], [9, 3], [3, 3]],
        'channels': [56, 112, 224, 448],
        'K': [8, 4, 2, 1],
        # 'emb_dims': 768, # Removed as head is removed
    },
    'GreedyViG_B': {
        'blocks': [[4, 4], [4, 4], [12, 4], [3, 3]],
        'channels': [64, 128, 256, 512],
        'K': [8, 4, 2, 1],
        # 'emb_dims': 768, # Removed as head is removed
    }
}

# --- Helper Modules (Unchanged from original Code 1) ---
class Stem(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)

class DepthWiseSeparable(nn.Module):
    def __init__(self, in_dim, kernel, expansion=4):
        super().__init__()
        hidden_dim = int(in_dim * expansion) # Calculate hidden dim based on expansion
        self.pw1 = nn.Conv2d(in_dim, hidden_dim, 1) # kernel size = 1
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.GELU()

        self.dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel, stride=1, padding=kernel // 2, groups=hidden_dim) # Correct padding
        self.norm2 = nn.BatchNorm2d(hidden_dim)
        self.act2 = nn.GELU()

        self.pw2 = nn.Conv2d(hidden_dim, in_dim, 1)
        self.norm3 = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        x = self.pw1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.dw(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.pw2(x)
        x = self.norm3(x)
        return x

class InvertedResidual(nn.Module):
    # Corrected expansion_ratio usage
    def __init__(self, dim, kernel=3, expansion_ratio=4., drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.dws = DepthWiseSeparable(in_dim=dim, kernel=kernel, expansion=expansion_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        residual = x
        x = self.dws(x)
        if self.use_layer_scale:
            x = self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * x)
        else:
            x = self.drop_path(x)
        return residual + x


class DynamicMRConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        # The input to nn is doubled because we concatenate x and x_j
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1), # Corrected input channels
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.K = K
        # Removed mean/std init here, calculated per forward pass

    def forward(self, x):
        B, C, H, W = x.shape
        x_j = torch.zeros_like(x) # Initialize x_j with zeros

        # get an estimate of the mean distance by computing the distance of points b/w quadrants. This is for efficiency to minimize computations.
        x_rolled_est = torch.roll(x, shifts=(-H//2, -W//2), dims=(2, 3))

        # Norm, Manhattan Distance (p=1)
        # Avoid in-place ops on x by cloning if necessary, though norm shouldn't be in-place
        norm = torch.norm((x - x_rolled_est), p=1, dim=1, keepdim=True)

        mean_dist = torch.mean(norm, dim=[1, 2, 3], keepdim=True) # Mean over C, H, W
        std_dist = torch.std(norm, dim=[1, 2, 3], keepdim=True)   # Std over C, H, W

        # Ensure K is at least 1 and not larger than H or W
        step_h = max(1, self.K)
        step_w = max(1, self.K)

        for i in range(0, H, step_h):
            if i == 0: continue # Skip rolling by 0
            x_rolled = torch.roll(x, shifts=(-i,), dims=2)
            dist = torch.norm((x - x_rolled), p=1, dim=1, keepdim=True)
            # Apply threshold using estimated mean/std
            mask = torch.where(dist < mean_dist - std_dist, 1.0, 0.0).float() # Ensure float mask
            x_rolled_and_masked = (x_rolled - x) * mask # Diff * mask
            x_j = torch.max(x_j, x_rolled_and_masked)

        for j in range(0, W, step_w):
            if j == 0: continue # Skip rolling by 0
            x_rolled = torch.roll(x, shifts=(-j,), dims=3)
            dist = torch.norm((x - x_rolled), p=1, dim=1, keepdim=True)
            mask = torch.where(dist < mean_dist - std_dist, 1.0, 0.0).float() # Ensure float mask
            x_rolled_and_masked = (x_rolled - x) * mask # Diff * mask
            x_j = torch.max(x_j, x_rolled_and_masked)

        x_cat = torch.cat([x, x_j], dim=1)
        return self.nn(x_cat)

class ConditionalPositionEncoding(nn.Module):
    def __init__(self, in_channels, kernel_size=7): # Default kernel 7
        super().__init__()
        self.pe = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=in_channels
        )

    def forward(self, x):
        x = self.pe(x) + x
        return x

class Grapher(nn.Module):
    def __init__(self, in_channels, K):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K

        self.cpe = ConditionalPositionEncoding(in_channels, kernel_size=7)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.GELU() # Added activation after first FC based on common patterns
        )
        # DynamicMRConv4d takes in_channels * 2 as input and outputs in_channels
        self.graph_conv = DynamicMRConv4d(in_channels, in_channels, K=self.K)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.GELU() # Added activation
        )

    def forward(self, x):
        shortcut = x
        x = self.cpe(x)
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        # Add residual connection common in graph/mixer blocks
        return x + shortcut


class DynamicGraphConvBlock(nn.Module):
    def __init__(self, in_dim, drop_path=0., K=2, use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.mixer = Grapher(in_dim, K)
        # Standard FFN with expansion=4
        ffn_hidden_dim = int(in_dim * 4)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, ffn_hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ffn_hidden_dim),
            nn.GELU(),
            nn.Conv2d(ffn_hidden_dim, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(in_dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(in_dim), requires_grad=True)

        # Add norm layers before mixer and ffn, common practice in transformers/mixers
        self.norm1 = nn.BatchNorm2d(in_dim)
        self.norm2 = nn.BatchNorm2d(in_dim)


    def forward(self, x):
        residual1 = x
        x = self.norm1(x) # Pre-normalization
        if self.use_layer_scale:
            x = residual1 + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(x))
        else:
            x = residual1 + self.drop_path(self.mixer(x))

        residual2 = x
        x = self.norm2(x) # Pre-normalization
        if self.use_layer_scale:
            x = residual2 + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.ffn(x))
        else:
            x = residual2 + self.drop_path(self.ffn(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            # Removed activation - typically just downsample+norm
        )
    def forward(self, x):
        x = self.conv(x)
        return x

# --- Main GreedyViG Model (Refactored) ---
class GreedyViG(torch.nn.Module):
    def __init__(self, model_name='GreedyViG_S', dropout=0., drop_path=0.):
        super(GreedyViG, self).__init__()

        assert model_name in GREEDYVIG_SPECS, f"Model name {model_name} not found in GREEDYVIG_SPECS"
        specs = GREEDYVIG_SPECS[model_name]
        blocks = specs['blocks']
        channels = specs['channels']
        K = specs['K']
        # kernels = 3 # Seems fixed in original, ensure used correctly
        # stride = 1 # Seems fixed in original
        # act_func = 'gelu' # Fixed in original modules

        # Calculate total blocks for drop path rate distribution
        n_blocks = sum([sum(x) for x in blocks])
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]  # stochastic depth decay rule
        dpr_idx = 0

        # --- Stem ---
        self.stem = Stem(input_dim=3, output_dim=channels[0])

        # --- Stages ---
        self.stages = nn.ModuleList() # Use ModuleList for stages
        current_channels = channels[0]
        for i in range(len(blocks)):
            stage_modules = []
            # Add Downsample layer if not the first stage
            if i > 0:
                stage_modules.append(Downsample(current_channels, channels[i]))
                current_channels = channels[i]

            # Add local and global blocks for the current stage
            local_stages_count = blocks[i][0]
            global_stages_count = blocks[i][1]

            for _ in range(local_stages_count):
                stage_modules.append(InvertedResidual(dim=current_channels, kernel=3, expansion_ratio=4, drop_path=dpr[dpr_idx]))
                dpr_idx += 1
            for _ in range(global_stages_count):
                stage_modules.append(DynamicGraphConvBlock(current_channels, drop_path=dpr[dpr_idx], K=K[i]))
                dpr_idx += 1

            self.stages.append(nn.Sequential(*stage_modules))

        # --- Initialize Weights ---
        self.model_init()

        # --- Calculate Output Width List ---
        self.eval() # Set model to evaluation mode for deterministic dummy forward pass
        try:
            # Temporarily move model to the target device for the dummy forward pass
            # This assumes 'device' is defined globally ('cuda' or 'cpu')
            original_device = next(self.parameters()).device # Store original device
            self.to(device)
            with torch.no_grad():
                # Use the specified input size (640x640) for width calculation
                dummy_input = torch.randn(1, 3, 640, 640, device=device)
                features = self.forward(dummy_input)
            self.width_list = [f.size(1) for f in features]
            self.to(original_device) # Move model back to its original device

        except Exception as e:
            print(f"Error during dummy forward pass for width calculation: {e}")
            self.width_list = [] # Set empty list on error
        self.train() # Set model back to training mode


    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # Common initialization
                # m.weight.requires_grad = True # This is True by default
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                    # m.bias.requires_grad = True # True by default
            elif isinstance(m, (torch.nn.BatchNorm2d, nn.GroupNorm)):
                 torch.nn.init.ones_(m.weight)
                 torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter): # Handle layer scale parameters
                 if m.requires_grad:
                     # Assuming layer scale params are small, initialized elsewhere correctly
                     pass


    def forward(self, inputs):
        x = self.stem(inputs)
        output_features = []
        # Sequentially pass through stages and collect outputs
        current_feature = x
        for stage in self.stages:
            current_feature = stage(current_feature)
            output_features.append(current_feature)

        # Return list of features from each stage's output
        # Corresponds to features after stage1, stage2, stage3, stage4 blocks
        return output_features


# --- Functions to instantiate specific models ---
def GreedyViG_S(dropout=0., drop_path=0.1, **kwargs):
    """ Instantiates the GreedyViG Small model """
    model = GreedyViG(model_name='GreedyViG_S', dropout=dropout, drop_path=drop_path)
    return model

def GreedyViG_M(dropout=0., drop_path=0.1, **kwargs):
    """ Instantiates the GreedyViG Medium model """
    model = GreedyViG(model_name='GreedyViG_M', dropout=dropout, drop_path=drop_path)
    return model

def GreedyViG_B(dropout=0., drop_path=0.1, **kwargs):
    """ Instantiates the GreedyViG Base model """
    model = GreedyViG(model_name='GreedyViG_B', dropout=dropout, drop_path=drop_path)
    return model


# --- Example Usage ---
if __name__ == "__main__":
    print(f"Using device: {device}")

    # Generating Sample image (as requested)
    image_size = (1, 3, 640, 640)
    image = torch.rand(*image_size).to(device)
    print(f"Input image shape: {image.shape}")

    # --- Test GreedyViG Variants ---
    models_to_test = {
        "GreedyViG_S": GreedyViG_S,
        "GreedyViG_M": GreedyViG_M,
        "GreedyViG_B": GreedyViG_B,
    }

    for name, model_builder in models_to_test.items():
        print(f"\n--- Testing {name} ---")
        model = model_builder(drop_path=0.1) # Example drop_path
        model.to(device)
        model.eval() # Set to eval for inference test

        try:
            with torch.no_grad():
                 output_features = model(image)

            print(f"{name} Width List: {model.width_list}")
            print(f"{name} Output Feature Shapes:")
            for i, feat in enumerate(output_features):
                print(f"  Stage {i+1}: {feat.shape}")

            # Basic check
            assert len(output_features) == len(GREEDYVIG_SPECS[name]['blocks']), "Mismatch in number of output features and stages"
            assert len(model.width_list) == len(output_features), "Mismatch between width_list and output features"
            print(f"{name} executed successfully.")

        except Exception as e:
            print(f"Error testing {name}: {e}")
            import traceback
            traceback.print_exc()

    # --- Compare with MobileNetV4 style (if code exists) ---
    # try:
    #     # Assuming Code 2 (MobileNetV4) is available in the same scope or imported
    #     print("\n--- Testing MobileNetV4 Example (requires Code 2) ---")
    #     mnv4_model = MobileNetV4HybridMedium() # Example from Code 2
    #     mnv4_model.to(device)
    #     mnv4_model.eval()
    #     with torch.no_grad():
    #         mnv4_out = mnv4_model(image)
    #     print(f"MobileNetV4 Width List: {mnv4_model.width_list}")
    #     print("MobileNetV4 Output Feature Shapes:")
    #     for i, feat in enumerate(mnv4_out):
    #         print(f"  Layer {i+1}: {feat.shape}")
    # except NameError:
    #     print("\nMobileNetV4 code (Code 2) not found, skipping comparison.")
    # except Exception as e:
    #      print(f"Error testing MobileNetV4: {e}")