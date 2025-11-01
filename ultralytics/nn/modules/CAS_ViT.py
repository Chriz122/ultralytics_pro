# """
# Code for CAS-ViT, modified to align with SwinTransformer usage and incorporate width_list.
# """

import torch
import torch.nn as nn
# from torch.cuda.amp import autocast # autocast is not used in the provided snippet

import numpy as np
# from einops import rearrange, repeat # Not used in the final model structure directly
# import itertools # Not used
import os
import copy

from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from timm.models.registry import register_model

# ======================================================================================================================
def stem(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), )

class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class Mlp(nn.Module):
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

class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class LocalIntegration(nn.Module):
    """
    """
    def __init__(self, dim, ratio=1, act_layer=nn.ReLU, norm_layer_mlp=nn.GELU): # Renamed norm_layer to norm_layer_mlp to avoid clash
        super().__init__()
        mid_dim = round(ratio * dim)
        # Original RCViT used GELU for norm here, but AdditiveBlock passes BatchNorm.
        # Let's make it flexible or stick to one. Given AdditiveBlock uses BatchNorm for its norms,
        # using BatchNorm here might be more consistent if norm_layer_mlp becomes BatchNorm2d.
        # However, the original code used norm_layer=nn.GELU in AdditiveBlock for LocalIntegration's norm.
        # This is confusing. Let's assume norm_layer_mlp is the activation *inside* the conv block.
        # And the actual norm is nn.BatchNorm2d.
        # The original code in AdditiveBlock passes norm_layer=nn.BatchNorm2d, which would be used by self.norm1, self.norm2.
        # But LocalIntegration's norm_layer argument was nn.GELU. This is likely an error or misunderstanding in my original interpretation.
        # Let's stick to what was passed: act_layer for activation, norm_layer_mlp for the norm *within* LocalIntegration
        # If AdditiveBlock sets norm_layer=nn.BatchNorm2d, then norm_layer_mlp here will be BatchNorm2d.
        
        # Re-evaluating: The AdditiveBlock passes `norm_layer=nn.BatchNorm2d` to the Stage,
        # which then passes it to AdditiveBlock. Inside AdditiveBlock,
        # `self.local_perception = LocalIntegration(..., norm_layer=norm_layer)`
        # So, norm_layer_mlp in LocalIntegration IS `nn.BatchNorm2d`.
        # The `act_layer` is `nn.ReLU`.
        self.network = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0),
            norm_layer_mlp(mid_dim), # This will be nn.BatchNorm2d(mid_dim)
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1, groups=mid_dim),
            act_layer(), # This will be nn.ReLU()
            nn.Conv2d(mid_dim, dim, 1, 1, 0),
        )


    def forward(self, x):
        return self.network(x)


class AdditiveTokenMixer(nn.Module):
    """
    """
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out



class AdditiveBlock(nn.Module):
    """
    """
    def __init__(self, dim, mlp_ratio=4., attn_bias=False, drop=0., drop_path=0.,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d): # Changed default norm_layer to BatchNorm2d
        super().__init__()
        # norm_layer for LocalIntegration was originally nn.GELU in its definition,
        # but here it's passed what AdditiveBlock receives (e.g. BatchNorm2d).
        # Let's assume LocalIntegration's internal norm should be `norm_layer` (e.g. BatchNorm2d)
        # and its activation should be `act_layer` (e.g. ReLU).
        self.local_perception = LocalIntegration(dim, ratio=1, act_layer=act_layer, norm_layer_mlp=norm_layer)
        self.norm1 = norm_layer(dim)
        self.attn = AdditiveTokenMixer(dim, attn_bias=attn_bias, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop) # Original Mlp uses GELU by default

    def forward(self, x):
        x = x + self.local_perception(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def Stage(dim, index, layers, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, attn_bias=False, drop=0., drop_path_rate=0.):
    """
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)

        blocks.append(
            AdditiveBlock(
                dim, mlp_ratio=mlp_ratio, attn_bias=attn_bias, drop=drop, drop_path=block_dpr,
                act_layer=act_layer, norm_layer=norm_layer) # Pass norm_layer here
        )
    blocks = nn.Sequential(*blocks)
    return blocks

class RCViT(nn.Module):
    def __init__(self, layers, embed_dims, mlp_ratios=4, downsamples=[True, True, True, True], norm_layer=nn.BatchNorm2d, attn_bias=False,
                 act_layer=nn.GELU, num_classes=1000, drop_rate=0., drop_path_rate=0., fork_feat=False,
                 distillation=True, pretrained=None, dummy_input_size=(224,224), **kwargs): # Added dummy_input_size
        super().__init__()

        self.fork_feat = fork_feat
        self.num_classes = num_classes # Keep for classification mode
        self.distillation = distillation # Keep for classification mode

        self.patch_embed = stem(3, embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = Stage(embed_dims[i], i, layers, mlp_ratio=mlp_ratios if isinstance(mlp_ratios, (int, float)) else mlp_ratios[i],
                          act_layer=act_layer, norm_layer=norm_layer,
                          attn_bias=attn_bias, drop=drop_rate, drop_path_rate=drop_path_rate)

            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    Embedding(
                        patch_size=3, stride=2, padding=1, in_chans=embed_dims[i],
                        embed_dim=embed_dims[i+1], norm_layer=norm_layer) # Use passed norm_layer
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # These indices should point to the output of a Stage block in self.network
            # Stage 0: self.network[0]
            # Stage 1: self.network[2]
            # Stage 2: self.network[4]
            # Stage 3: self.network[6]
            self.out_indices = [0, 2, 4, 6] # Corresponds to the output of each of the 4 stages
            for i_emb, i_layer_in_network in enumerate(self.out_indices):
                # We need to ensure embed_dims[i_emb] matches the output dim of network[i_layer_in_network]
                # embed_dims are [dim_stage0, dim_stage1, dim_stage2, dim_stage3]
                # network[0] (Stage 0) outputs embed_dims[0]
                # network[2] (Stage 1) outputs embed_dims[1]
                # network[4] (Stage 2) outputs embed_dims[2]
                # network[6] (Stage 3) outputs embed_dims[3]
                current_embed_dim = embed_dims[i_emb]
                if i_emb == 0 and os.environ.get('FORK_LAST3', None): # This seems like a specific experimental setup
                    layer = nn.Identity()
                else:
                    layer = norm_layer(current_embed_dim)
                layer_name = f'norm{i_layer_in_network}' # Use network index for clarity
                self.add_module(layer_name, layer)

            # Calculate width_list for feature extraction mode
            try:
                dummy_h, dummy_w = to_2tuple(dummy_input_size)
                dummy_input = torch.randn(1, 3, dummy_h, dummy_w)
                # Store current training state and set to eval for dummy pass
                original_training_state = self.training
                self.eval()
                with torch.no_grad():
                    features = self.forward(dummy_input) # self.forward will use self.fork_feat
                self.width_list = [f.size(1) for f in features]
                self.train(original_training_state) # Restore original training state
            except Exception as e:
                print(f"RCViT Warning: Could not compute width_list during init: {e}")
                self.width_list = [] # Fallback
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()
            if self.distillation:
                self.dist_head = nn.Linear(
                    embed_dims[-1], num_classes) if num_classes > 0 \
                    else nn.Identity()
            self.apply(self.cls_init_weights) # Initialize classifier weights
            self.width_list = [] # Not typically needed for classification mode directly

        # Simplified weight initialization / loading
        if pretrained:
            self.load_pretrained(pretrained)

    def load_pretrained(self, pretrained_path):
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict_key = 'model' if 'model' in checkpoint else 'state_dict' if 'state_dict' in checkpoint else ''
            
            if state_dict_key:
                state_dict = checkpoint[state_dict_key]
            else: # Assume the checkpoint is the state_dict itself
                state_dict = checkpoint

            # Filter out unnecessary keys (e.g., classifier head if fork_feat=True)
            if self.fork_feat:
                # Remove classifier specific weights if we are in fork_feat mode
                # and the checkpoint contains them.
                for k in list(state_dict.keys()):
                    if k.startswith('head.') or k.startswith('norm.'): # final norm before head
                        if not hasattr(self, k.split('.')[0]): # if self doesn't have 'head' or 'norm' (final one)
                           print(f"  Ignoring {k} from pretrained checkpoint for fork_feat=True mode.")
                           del state_dict[k]
            
            # Adjust for distillation head if necessary
            if not self.distillation and self.fork_feat==False:
                 for k in list(state_dict.keys()):
                    if k.startswith('dist_head.'):
                        print(f"  Ignoring {k} from pretrained checkpoint as distillation is False.")
                        del state_dict[k]

            msg = self.load_state_dict(state_dict, strict=False)
            print(f"  Pretrained weights loaded with message: {msg}")
        else:
            print(f"RCViT Warning: Pretrained path {pretrained_path} does not exist.")


    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None): # Original, for mmdet. Kept for reference but simplified.
        # This method was complex and tied to mmdetection's logger and _load_checkpoint.
        # We'll use a simpler load_pretrained for now.
        pass


    def forward_tokens(self, x):
        outs = []
        for idx, block_module in enumerate(self.network): # Changed 'block' to 'block_module' to avoid clash
            x = block_module(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer_module = getattr(self, f'norm{idx}')
                x_out = norm_layer_module(x)
                outs.append(x_out)
        
        if self.fork_feat:
            # outs should be a list of 4 feature maps
            return outs # This is a list
        return x # This is the final feature map for classification path

    def forward(self, x):
        x = self.patch_embed(x)
        x_features = self.forward_tokens(x) # This returns a list if fork_feat, else a tensor

        if self.fork_feat:
            # output features of four stages for dense prediction
            return x_features # x_features is already a list of tensors

        # Classification path (fork_feat is False)
        # Here x_features is the single tensor output from forward_tokens
        x_final = self.norm(x_features)
        
        if hasattr(self, 'dist_head') and self.distillation: # Check if dist_head exists
            # When distillation=True, head and dist_head are present
            cls_out_main = self.head(x_final.flatten(2).mean(-1))
            cls_out_dist = self.dist_head(x_final.flatten(2).mean(-1))
            if not self.training: # Average outputs during inference
                cls_out = (cls_out_main + cls_out_dist) / 2
            else: # Return both during training
                cls_out = cls_out_main, cls_out_dist # This is a TUPLE
        else:
            cls_out = self.head(x_final.flatten(2).mean(-1)) # This is a TENSOR

        # To consistently return a list from forward to potentially aid ultralytics,
        # even in classification mode, we could wrap cls_out.
        # However, standard classification models return a tensor/tuple.
        # The key is that when used as a backbone (fork_feat=True), it returns a list.
        return cls_out


# ======================================================================================================================
# Helper function for loading weights, similar to SwinTransformer's
def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'Loading weights... {idx}/{len(model_dict)} items loaded successfully.')
    return model_dict

# New factory functions, similar to SwinTransformer
@register_model
def RCViT_XS(weights='', pretrained_strict=False, **kwargs): # Added pretrained_strict
    model = RCViT(
        layers=[2, 2, 4, 2], embed_dims=[48, 56, 112, 220], mlp_ratios=4, 
        downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU,
        fork_feat=True, # Default to True for backbone usage
        **kwargs)
    if weights:
        # Using the simpler load_pretrained method inside RCViT
        model.load_pretrained(weights)
        # Or, if you prefer the Swin-style external loading:
        # state_dict = torch.load(weights)['model'] # Adjust key if necessary
        # model.load_state_dict(update_weight(model.state_dict(), state_dict), strict=pretrained_strict)
    return model

@register_model
def RCViT_S(weights='', pretrained_strict=False, **kwargs):
    model = RCViT(
        layers=[3, 3, 6, 3], embed_dims=[48, 64, 128, 256], mlp_ratios=4, 
        downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU,
        fork_feat=True,
        **kwargs)
    if weights:
        model.load_pretrained(weights)
    return model

@register_model
def RCViT_M(weights='', pretrained_strict=False, **kwargs):
    model = RCViT(
        layers=[3, 3, 6, 3], embed_dims=[64, 96, 192, 384], mlp_ratios=4, 
        downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU,
        fork_feat=True,
        **kwargs)
    if weights:
        model.load_pretrained(weights)
    return model

@register_model
def RCViT_T(weights='', pretrained_strict=False, **kwargs): # Assuming 'T' means Tiny or a different variant
    model = RCViT(
        layers=[3, 3, 6, 3], embed_dims=[96, 128, 256, 512], mlp_ratios=4, 
        downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU,
        fork_feat=True,
        **kwargs)
    if weights:
        model.load_pretrained(weights)
    return model

# ======================================================================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("Testing RCViT_XS as a backbone (fork_feat=True):")
    # Use dummy_input_size consistent with width_list calculation for this test
    test_input_size = (224, 224) # Or (640,640) if preferred for testing large inputs
    net_backbone = RCViT_XS(dummy_input_size=test_input_size) # Pass dummy_input_size
    
    # Ensure model is in eval mode if not training, for consistency with width_list calculation
    net_backbone.eval() 

    print(f"  Model out_indices: {net_backbone.out_indices if hasattr(net_backbone, 'out_indices') else 'N/A'}")
    print(f"  Model width_list: {net_backbone.width_list}")
    
    x = torch.rand((1, 3, test_input_size[0], test_input_size[1]))
    
    # Test with autocast if using mixed precision and CUDA is available
    # if torch.cuda.is_available():
    #     net_backbone = net_backbone.cuda()
    #     x = x.cuda()
    #     with torch.cuda.amp.autocast():
    #         out_features = net_backbone(x)
    # else:
    #     out_features = net_backbone(x)
    out_features = net_backbone(x)

    if isinstance(out_features, list):
        print(f"  Output is a list of {len(out_features)} tensors (features):")
        for i, f in enumerate(out_features):
            print(f"    Feature {i} shape: {f.shape}") # Should be [B, C, H, W]
            if net_backbone.width_list: # Check if width_list was successfully computed
                 assert f.size(1) == net_backbone.width_list[i], \
                    f"Mismatch: Feature {i} channels {f.size(1)} vs width_list {net_backbone.width_list[i]}"
    else:
        print(f"  Output shape (classification): {out_features.shape if isinstance(out_features, torch.Tensor) else [o.shape for o in out_features]}")

    print('  Net Params: {:d}'.format(int(count_parameters(net_backbone))))
    print("-" * 30)

    print("\nTesting RCViT_XS for classification (fork_feat=False):")
    # For classification, num_classes matters.
    net_classifier = RCViT(
        layers=[2, 2, 4, 2], embed_dims=[48, 56, 112, 220], mlp_ratios=4,
        downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU,
        num_classes=100, fork_feat=False, distillation=True, # Test with distillation
        dummy_input_size=test_input_size
    )
    net_classifier.eval() # Set to eval for testing
    
    # x is already defined
    out_classification = net_classifier(x)

    if isinstance(out_classification, tuple): # Distillation returns a tuple during training
        print(f"  Output is a tuple of {len(out_classification)} tensors (classification with distillation, eval mode averages):")
        # In eval mode with distillation, it should be a single tensor if correctly averaged.
        # My forward logic for classification:
        # if not self.training: cls_out = (cls_out_main + cls_out_dist) / 2 -> tensor
        # else: cls_out = cls_out_main, cls_out_dist -> tuple
        # Since net_classifier.eval() is set, it should be a tensor.
        # Let's test training mode to see the tuple.
        net_classifier.train()
        out_classification_train = net_classifier(x)
        if isinstance(out_classification_train, tuple):
             print(f"  Output (train mode, distillation) is a tuple of shapes: {[o.shape for o in out_classification_train]}")
        else:
             print(f"  Output (train mode, distillation) shape: {out_classification_train.shape}")
        
        net_classifier.eval() # back to eval
        out_classification_eval = net_classifier(x)
        print(f"  Output (eval mode, distillation averaged) shape: {out_classification_eval.shape}")


    elif isinstance(out_classification, torch.Tensor):
        print(f"  Output shape (classification): {out_classification.shape}")
    
    print('  Net Params (classifier): {:d}'.format(int(count_parameters(net_classifier))))
    print(f"  Classifier width_list: {net_classifier.width_list}") # Should be empty for fork_feat=False