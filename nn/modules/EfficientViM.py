import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.layers import trunc_normal_, SqueezeExcite # Keep SqueezeExcite if used
from timm.models.layers import SqueezeExcite, trunc_normal_ # Make sure trunc_normal_ is imported if used in _init_weights
# from timm.models import register_model # Removed, using direct function calls now
# from fvcore.nn import flop_count # Removed, can be added back if needed separately

# --- LayerNorm Classes (Unchanged) ---
class LayerNorm2D(nn.Module):
    """LayerNorm for channels of 2D tensor(B C H W)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # Original implementation was correct for LayerNorm across C dimension
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if self.affine:
            x = self.weight * x + self.bias
        return x


class LayerNorm1D(nn.Module):
    """LayerNorm for channels of 1D tensor(B C L)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm1D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # Original implementation was correct for LayerNorm across C dimension
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)

        if self.affine:
            x = self.weight * x + self.bias # Corrected broadcasting needed self.weight * x_normalized + self.bias
        return x

# --- ConvLayer Classes (Unchanged) ---
class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm2d, act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer2D, self).__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=False if norm else True # Bias usually False if Norm layer follows
        )
        self.norm = norm(out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm and isinstance(self.norm, (nn.BatchNorm2d, nn.GroupNorm)):
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm1d, act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer1D, self).__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False if norm else True # Bias usually False if Norm layer follows
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm and isinstance(self.norm, (nn.BatchNorm1d, nn.GroupNorm)):
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


# --- FFN Class (Unchanged) ---
class FFN(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        # Use LayerNorm2D consistent with input format
        self.fc1 = ConvLayer2D(in_dim, dim, 1, norm=LayerNorm2D, act_layer=nn.SiLU) # Using SiLU as common in modern architectures
        self.fc2 = ConvLayer2D(dim, in_dim, 1, act_layer=None, norm=LayerNorm2D, bn_weight_init=0) # Changed bn_weight_init to 0 as per original

    def forward(self, x):
        x = self.fc2(self.fc1(x))
        return x


# --- Stem Class (Unchanged) ---
class Stem(nn.Module):
    def __init__(self,  in_dim=3, dim=96):
        super().__init__()
        self.conv = nn.Sequential(
            ConvLayer2D(in_dim, dim // 8, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 8, dim // 4, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 4, dim // 2, kernel_size=3, stride=2, padding=1),
            # Last layer often doesn't have activation, depends on design
            ConvLayer2D(dim // 2, dim, kernel_size=3, stride=2, padding=1, act_layer=None)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# --- PatchMerging Class (Minor tweak for clarity) ---
class PatchMerging(nn.Module):
    def __init__(self,  in_dim, out_dim, ratio=4.0):
        super().__init__()
        hidden_dim = int(in_dim * ratio) # Original code might have intended in_dim * ratio for hidden
                                         # Or out_dim * ratio. The provided code uses out_dim for hidden_dim calculation.
                                         # Let's stick to what was provided: int(out_dim * ratio)
                                         # but if this is meant for typical ViT style PM, it might be int(in_dim * 2) for hidden if out_dim is in_dim * 2.
                                         # Given it's Conv-based, out_dim * ratio is plausible for expansion.
        # For clarity, let's use the provided logic: hidden_dim = int(out_dim * ratio)
        # However, if PatchMerging is also responsible for channel increase (e.g. C -> 2C),
        # then `out_dim` is the target channel dimension after downsampling.
        # The original structure of PM in some ViTs is: LN -> Linear(C -> 2C) -> Reshape (or Conv equivalent)
        # Here, it seems more complex. Let's assume `out_dim` is the final output channel dimension.
        # A common pattern for PatchMerging's internal expansion is based on its input dimension.
        # If `out_dim` is the target channel count, then `hidden_dim` expansion could be `int(in_dim * ratio)` or `int(out_dim * (ratio/2))` if `out_dim = 2*in_dim`.
        # The provided code has `hidden_dim = int(out_dim * ratio)`. This means the hidden_dim is very large if ratio=4.
        # E.g. in_dim=128, out_dim=192. hidden_dim = 192*4 = 768.
        # Let's assume this large expansion is intended.

        self.conv = nn.Sequential(
            ConvLayer2D(in_dim, hidden_dim, kernel_size=1, act_layer=nn.SiLU), # Added SiLU
            ConvLayer2D(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, groups=hidden_dim, act_layer=nn.SiLU), # stride=2, Added SiLU
            SqueezeExcite(hidden_dim, .25),
            ConvLayer2D(hidden_dim, out_dim, kernel_size=1, act_layer=None)
        )
        self.dwconv1 = ConvLayer2D(in_dim, in_dim, 3, padding=1, groups=in_dim, act_layer=None, norm=LayerNorm2D, bn_weight_init=1.0)
        self.dwconv2 = ConvLayer2D(out_dim, out_dim, 3, padding=1, groups=out_dim, act_layer=None, norm=LayerNorm2D, bn_weight_init=1.0)

    def forward(self, x):
        x_res = x
        x = x + self.dwconv1(x) # Pre-addition
        x = self.conv(x)       # Main path with downsampling
        x = x + self.dwconv2(x) # Post-addition
        return x

# --- HSMSSD Class (MODIFIED) ---
class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim = 64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(d_model, 3*state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim*3
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3,1,1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0)
        self.hz_proj = ConvLayer1D(d_model, 2*self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU() # Default inplace=False
        self.D = nn.Parameter(torch.ones(d_model))
        self.D._no_weight_decay = True

    def forward(self, x_in_2d):
        batch, d_model, H, W = x_in_2d.shape
        L = H * W
        x = x_in_2d.flatten(2)

        BCdt_flat = self.BCdt_proj(x)
        BCdt_2d = BCdt_flat.view(batch, -1, H, W)
        BCdt_dw = self.dw(BCdt_2d)
        BCdt = BCdt_dw.flatten(2)

        B, C, dt = torch.split(BCdt, [self.state_dim, self.state_dim,  self.state_dim], dim=1)

        A_base = self.A.view(1, -1, 1)
        A = (dt + A_base).softmax(dim=-1) # Softmax over L

        AB = (A * B)
        h = x @ AB.transpose(-2, -1)

        h_projected, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        
        # MODIFICATION: Use z.clone() to prevent inplace error on view
        h_gated = h_projected * self.act(z.clone()) 
        
        h_out_proj = self.out_proj(h_gated)
        y = h_out_proj @ C
        
        y = y.view(batch, d_model, H, W)
        y = y + x_in_2d * self.D.view(1, -1, 1, 1)

        return y.contiguous(), h


# --- EfficientViMBlock (Unchanged from snippet, check norm consistency) ---
class EfficientViMBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm = LayerNorm2D(dim)
        self.mixer = HSMSSD(d_model=dim, ssd_expand=ssd_expand,state_dim=state_dim)

        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, norm=LayerNorm2D, bn_weight_init=1.0, act_layer=None)
        self.dwconv2 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, norm=LayerNorm2D, bn_weight_init=0.0, act_layer=None)

        self.ffn = FFN(in_dim=dim, dim=int(dim * mlp_ratio))
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim), requires_grad=True)

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha).view(4, 1, -1, 1, 1) # (4, B, C, H, W) -> (4, 1, C, 1, 1) for broadcasting with dim

        # Original alpha shape: (4, dim) -> view(4, -1, 1, 1) implies (4, C, 1, 1) which is fine for (B,C,H,W)
        alpha_reshaped = torch.sigmoid(self.alpha).view(4, self.dim, 1, 1)


        x = (1 - alpha_reshaped[0]) * x + alpha_reshaped[0] * self.dwconv1(x)
        
        x_prev_mixer = x
        x_norm = self.norm(x)
        y_mixer, h = self.mixer(x_norm)
        x = (1 - alpha_reshaped[1]) * x_prev_mixer + alpha_reshaped[1] * y_mixer

        x = (1 - alpha_reshaped[2]) * x + alpha_reshaped[2] * self.dwconv2(x)
        
        x_prev_ffn = x # Save before FFN for residual
        x_ffn_out = self.ffn(x) # FFN output
        x = (1 - alpha_reshaped[3]) * x_prev_ffn + alpha_reshaped[3] * x_ffn_out


        return x, h


# --- EfficientViMStage (Unchanged from snippet) ---
class EfficientViMStage(nn.Module):
    def __init__(self, in_dim, out_dim, depth, mlp_ratio=4., downsample=None, ssd_expand=1, state_dim=64):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([
            EfficientViMBlock(dim=in_dim, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand, state_dim=state_dim) for _ in range(depth)])

        self.downsample = downsample(in_dim=in_dim, out_dim=out_dim) if downsample is not None else None

    def forward(self, x):
        h_stage = None 
        for blk in self.blocks:
            x, h = blk(x) 
            h_stage = h 

        x_out = x 

        if self.downsample is not None:
            x = self.downsample(x)
        return x, x_out, h_stage


# --- Refactored EfficientViM Backbone (Unchanged from snippet) ---
class EfficientViM(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[128, 192, 320], 
                 depths=[2, 2, 2],
                 mlp_ratio=4.,
                 ssd_expand=1.,
                 state_dims=[49, 25, 9], 
                 out_indices=(0, 1, 2), 
                 **kwargs): 
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.out_indices = out_indices

        self.patch_embed = Stem(in_dim=in_chans, dim=embed_dims[0])
        PatchMergingBlock = PatchMerging 

        self.stages = nn.ModuleList()
        in_dim = embed_dims[0]
        for i_layer in range(self.num_layers):
            current_embed_dim = embed_dims[i_layer] # This should be the in_dim for the stage blocks
            
            # Determine out_dim for PatchMerging: if it's not the last stage, it's the next embed_dim
            # If it's the last stage, PatchMerging is not used, so out_dim for stage is current_embed_dim
            pm_out_dim = embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else current_embed_dim

            stage = EfficientViMStage(in_dim=current_embed_dim, # blocks operate on current_embed_dim
                               out_dim=pm_out_dim, # PatchMerging (if exists) maps current_embed_dim to pm_out_dim
                               depth=depths[i_layer],
                               mlp_ratio=mlp_ratio,
                               downsample=PatchMergingBlock if (i_layer < self.num_layers - 1) else None,
                               ssd_expand=ssd_expand,
                               state_dim = state_dims[i_layer])
            self.stages.append(stage)
            # The input dimension for the *next* stage is the output dimension of the current stage's downsampler
            in_dim = pm_out_dim # This was correctly updated in the original logic too.
                                # The local `current_embed_dim` ensures blocks within the stage use the correct input dimension.

        self.num_features = [embed_dims[i] for i in out_indices] 
        for i in out_indices:
             # The features output by stage `i` will have `embed_dims[i]` channels
             # because `x_out` from `EfficientViMStage` is before downsampling.
             layer = LayerNorm2D(embed_dims[i]) # So, use embed_dims[i] directly
             layer_name = f'norm{i}'
             self.add_module(layer_name, layer)

        self.apply(self._init_weights)

        self.eval()
        with torch.no_grad():
             try:
                 dummy_input = torch.randn(1, in_chans, 224, 224) # Using a more common default size
                 first_param = next(self.parameters(), None)
                 if first_param is not None:
                      dummy_input = dummy_input.to(first_param.device)
                 outputs = self.forward(dummy_input)
                 self.width_list = [o.size(1) for o in outputs]
             except Exception as e:
                  print(f"Warning: Could not compute width_list during init: {e}")
                  self.width_list = list(self.num_features) # Fallback, ensure it's a list
        self.train() 

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm2D, LayerNorm1D, nn.LayerNorm)): 
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
             if m.bias is not None: nn.init.constant_(m.bias, 0)
             if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
             # Default PyTorch init for Conv2d is kaiming_uniform_
             # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # Optionally Kaiming normal
             if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
        elif isinstance(m, HSMSSD):
             pass # A and D are initialized in HSMSSD's __init__

    def forward(self, x):
        x = self.patch_embed(x) 

        outs = []
        current_x = x # Use a different variable name for the loop
        for i, stage in enumerate(self.stages):
            # Stage output: next_stage_input, output_before_downsample, last_hidden_state
            current_x, x_out_stage, h = stage(current_x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_norm = norm_layer(x_out_stage) # Normalize the output *before* downsampling
                outs.append(x_norm)
        return outs


# --- Factory Functions (Unchanged from snippet) ---
def _load_weights(model, weights_path):
    if weights_path:
        print(f"Attempting to load weights from: {weights_path}")
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

            if state_dict:
                model_dict = model.state_dict()
                # Filter out unnecessary keys and adapt to potentially different key names
                new_state_dict = {}
                loaded_keys = []
                skipped_keys_mismatch = []
                skipped_keys_missing = []

                for k_ckpt, v_ckpt in state_dict.items():
                    if k_ckpt in model_dict:
                        if model_dict[k_ckpt].shape == v_ckpt.shape:
                            new_state_dict[k_ckpt] = v_ckpt
                            loaded_keys.append(k_ckpt)
                        else:
                            skipped_keys_mismatch.append(
                                f"Skipping {k_ckpt}: shape_ckpt={v_ckpt.shape}, shape_model={model_dict[k_ckpt].shape}")
                    else:
                        skipped_keys_missing.append(f"Skipping {k_ckpt}: key not in model")
                
                if skipped_keys_mismatch:
                    print("Warning: Shape mismatches for some keys:")
                    for msg in skipped_keys_mismatch: print(f"  {msg}")
                if skipped_keys_missing:
                    print("Warning: Some keys from checkpoint not found in model:")
                    for msg in skipped_keys_missing: print(f"  {msg}")

                model_dict.update(new_state_dict)
                model.load_state_dict(model_dict, strict=False) # Use strict=False if you expect some missing keys
                print(f"Successfully loaded {len(loaded_keys)}/{len(model_dict)} items from checkpoint ({len(state_dict)} in file).")
            else:
                print(f"Warning: Could not find state dict in {weights_path}")

        except FileNotFoundError:
            print(f"Error: Weight file not found at {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print("No weights path provided, using initialized weights.")
    return model

def EfficientViM_M1(pretrained=False, weights='', **kwargs):
    model = EfficientViM(
        embed_dims=[128, 192, 320], 
        depths=[2, 2, 2],
        mlp_ratio=4.,
        ssd_expand=1.,
        state_dims=[49, 25, 9], 
        out_indices=(0, 1, 2), 
        **kwargs)
    if pretrained or weights: # Allow either pretrained=True (to fetch default) or specific weights path
        model = _load_weights(model, weights if weights else "path/to/default/m1_weights.pth") # Placeholder for default
    return model

def EfficientViM_M2(pretrained=False, weights='', **kwargs):
    model = EfficientViM(
        embed_dims=[128, 256, 512],
        depths=[2, 2, 2],
        mlp_ratio=4.,
        ssd_expand=1.,
        state_dims=[49, 25, 9], 
        out_indices=(0, 1, 2),
        **kwargs)
    if pretrained or weights:
        model = _load_weights(model, weights if weights else "path/to/default/m2_weights.pth")
    return model

def EfficientViM_M3(pretrained=False, weights='', **kwargs):
    model = EfficientViM(
        embed_dims=[224, 320, 512],
        depths=[2, 2, 2],
        mlp_ratio=4.,
        ssd_expand=1.,
        state_dims=[49, 25, 9], 
        out_indices=(0, 1, 2),
        **kwargs)
    if pretrained or weights:
        model = _load_weights(model, weights if weights else "path/to/default/m3_weights.pth")
    return model

def EfficientViM_M4(pretrained=False, weights='', **kwargs):
    model = EfficientViM(
        embed_dims=[224, 320, 512],
        depths=[3, 4, 2],
        mlp_ratio=4.,
        ssd_expand=1.,
        state_dims=[64, 32, 16], 
        out_indices=(0, 1, 2),
        **kwargs)
    if pretrained or weights:
        model = _load_weights(model, weights if weights else "path/to/default/m4_weights.pth")
    return model


def EfficientViM_M5(pretrained=False, weights='', **kwargs):
    model = EfficientViM(
        embed_dims=[224, 320, 512, 800], # Has 4 stages
        depths=[3, 4, 2, 2],
        mlp_ratio=4.,
        ssd_expand=1.,
        state_dims=[64, 32, 16, 8], 
        out_indices=(0, 1, 2, 3), # Output from all 4 stages
        **kwargs)
    if pretrained or weights:
        model = _load_weights(model, weights if weights else "path/to/default/m5_weights.pth")
    return model


# --- Example Usage ---
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test M2 variant
    model = EfficientViM_M2().to(device)
    # print(model)

    # Example input (adjust size as needed)
    input_tensor = torch.randn(2, 3, 224, 224).to(device) # Smaller size for faster test
    # input_tensor = torch.randn(1, 3, 640, 640).to(device) # Larger size like in YOLO

    model.eval() # Set to evaluation mode
    with torch.no_grad():
        outputs = model(input_tensor)

    print(f"\nModel: EfficientViM_M2")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Number of output feature maps: {len(outputs)}")
    print("Output feature map shapes and channels (width_list):")
    for i, out in enumerate(outputs):
        print(f"  Stage {model.out_indices[i]}: Shape = {out.shape}, Channels = {out.size(1)}")

    print(f"Calculated width_list during init: {model.width_list}")

    # Verify width_list matches output channels
    assert model.width_list == [o.size(1) for o in outputs], "Mismatch between width_list and actual output channels!"
    print("\nwidth_list matches runtime output channels.")

    # Test factory function with dummy weights path
    model_m1_weights = EfficientViM_M1(weights='non_existent_path.pth').to(device)