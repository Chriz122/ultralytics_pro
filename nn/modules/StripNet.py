import torch
import torch.nn as nn
import math
from functools import partial
import warnings

# 從 timm 導入必要的工具
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# --- 輔助函數區 (替代 mmcv 功能) ---

def get_norm_layer(norm_cfg, num_features):
    """根據配置字典創建標準化層。"""
    if norm_cfg is None:
        # 預設使用 BatchNorm2d，與原始代碼行為一致
        return nn.BatchNorm2d(num_features)
    
    norm_type = norm_cfg.get('type')
    if norm_type == 'SyncBN':
        # 注意：SyncBatchNorm 需要在 DDP 環境下才能正常工作
        return nn.SyncBatchNorm(num_features, **norm_cfg.get('kwargs', {}))
    elif norm_type == 'BN':
        return nn.BatchNorm2d(num_features, **norm_cfg.get('kwargs', {}))
    elif norm_type == 'LN':
        return nn.LayerNorm(num_features, **norm_cfg.get('kwargs', {}))
    else:
        raise NotImplementedError(f"Normalization layer type '{norm_type}' is not implemented.")

# --- 模型組件區 (原始代碼，稍作修改以移除依賴) ---

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class StripBlock(nn.Module):
    def __init__(self, dim, k1, k2):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim, dim, kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)     
        self.conv_spatial2 = nn.Conv2d(dim, dim, kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):   
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)
        return x * attn

class Attention(nn.Module):
    def __init__(self, d_model, k1, k2):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripBlock(d_model, k1, k2)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., k1=1, k2=19, drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        self.norm1 = get_norm_layer(norm_cfg, dim)
        self.attn = Attention(dim, k1, k2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = get_norm_layer(norm_cfg, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = get_norm_layer(norm_cfg, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W

# --- 主要模型 StripNet (已重構) ---

class StripNet(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4], k1s=[1, 1, 1, 1], k2s=[19, 19, 19, 19], 
                 drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4, 
                 pretrained=None,
                 norm_cfg=None):
        super().__init__()
        
        self.depths = depths
        self.num_stages = num_stages
        self.embed_dims = embed_dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], k1=k1s[i], k2=k2s[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_cfg=norm_cfg)
                for j in range(depths[i])])
            
            # 使用 nn.LayerNorm 作为 Transformer-style block 结尾的 norm
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # 初始化權重
        self.apply(self._init_weights)
        if pretrained:
            self.load_pretrained(pretrained)

        # 參考 EMO 模型，在初始化最後計算 width_list
        # 這有助於與需要知道各階段輸出通道數的框架(如YOLOv8)集成
        # 注意: 這裡的 forward() 會被調用，所以要確保模型結構在此時已完整
        self.width_list = [i.shape[1] for i in self.forward(torch.randn(1, in_chans, 640, 640))]
        # print(f"StripNet initialized. Output channels per stage: {self.width_list}")


    def _init_weights(self, m):
        """權重初始化函數，替代 mmcv.cnn.utils.weight_init"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def load_pretrained(self, pretrained_path):
        """加載預訓練權重"""
        if pretrained_path:
            print(f"Loading pretrained weights from {pretrained_path}")
            try:
                # 加載 checkpoint
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                
                # 處理不同 checkpoint 格式 (例如 'state_dict', 'model', or root)
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

                # 移除 'head.' or 'backbone.' 等前綴（如果有的話）
                # 這一步驟對於從不同訓練框架導出的權重很重要
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('backbone.'):
                        new_state_dict[k[len('backbone.'):]] = v
                    elif k.startswith('head.'):
                        continue # 通常不加載分類頭
                    else:
                        new_state_dict[k] = v
                
                # 加載權重
                msg = self.load_state_dict(new_state_dict, strict=False)
                print(f"Pretrained weights loaded with message: {msg}")

            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            
            # 原始的 norm 應用方式是針對 Transformer 的 (B, N, C)
            # 這裡我們保持 CNN 的 (B, C, H, W) 格式
            x_reshaped = x.flatten(2).transpose(1, 2)
            x_normed = norm(x_reshaped)
            x = x_normed.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            
            outs.append(x)
        
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # 確保輸出是 list，以避免 'tuple' object has no attribute 'insert' 錯誤
        # 即使 forward_features 已經返回 list，這裡再次轉換可以確保萬無一失
        return list(x)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

# --- 工廠函數 (類似 EMO 的使用方式) ---

def StripNet_tiny(pretrained=None, **kwargs):
    """
    StripNet-Tiny model.
    """
    model = StripNet(
        embed_dims=[32, 64, 160, 256],
        k1s=[1, 1, 1, 1],
        k2s=[19, 19, 19, 19],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.1,
        drop_path_rate=0.15,
        depths=[3, 3, 5, 2],
        norm_cfg=dict(type='BN'),  # 使用普通 BN，或根據需要改為 'SyncBN'
        pretrained=pretrained,
        **kwargs
    )
    return model

def StripNet_small(pretrained=None, **kwargs):
    """
    StripNet-Small model.
    """
    model = StripNet(
        embed_dims=[64, 128, 320, 512],
        k1s=[1, 1, 1, 1],
        k2s=[19, 19, 19, 19],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.1,
        drop_path_rate=0.15,
        depths=[2, 2, 4, 2],
        norm_cfg=dict(type='BN'), # 使用普通 BN，或根據需要改為 'SyncBN'
        pretrained=pretrained,
        **kwargs
    )
    return model

# --- 使用範例 ---
if __name__ == '__main__':
    # 創建一個 StripNet-Tiny 模型
    print("Creating StripNet-Tiny model...")
    # 假設預訓練權重 'pretrained/stripnet_s.pth.tar' 存在
    # model_tiny = StripNet_tiny(pretrained='path/to/your/weights.pth')
    model_tiny = StripNet_tiny()

    # 創建一個 StripNet-Small 模型
    print("\nCreating StripNet-Small model...")
    model_small = StripNet_small()

    # 打印模型結構
    # print(model_small)
    
    # 測試前向傳播
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 3, 640, 640)
    output_features = model_small(dummy_input)

    # 檢查輸出
    print(f"Output is a: {type(output_features)}")
    print(f"Number of output feature maps: {len(output_features)}")
    for i, feature_map in enumerate(output_features):
        print(f"  - Feature map {i+1} shape: {feature_map.shape}")
        
    # 檢查 width_list 是否正確
    print(f"\nModel's pre-calculated width_list: {model_small.width_list}")
    
    # 驗證 width_list 是否與實際輸出通道數一致
    actual_widths = [f.shape[1] for f in output_features]
    assert model_small.width_list == actual_widths
    print("width_list matches actual output channels. Verification successful.")