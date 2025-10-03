import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import scipy.io as sio
import math
from functools import partial
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import time

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
                
    def forward(self, x):
        return x + self.conv(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True, downsample=False, kernel_size=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
               
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()         
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.conv = ResDWC(hidden_features, 3)
        
    def forward(self, x):       
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)        
        x = self.conv(x)        
        x = self.fc2(x)               
        x = self.drop(x)
        return x
 
class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = qk_scale or head_dim ** -0.5
                
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3, dim=2)
        attn = (k.transpose(-1, -2) @ q) * self.scale
        attn = attn.softmax(dim=-2)
        attn = self.attn_drop(attn)
        x = (v @ attn).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.weights, stride=1, padding=self.kernel_size//2)        
        return x.reshape(b, c*9, h*w)

class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
    def forward(self, x):
        # b, _, h, w = x.shape # Unused variables
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x

class StokenAttention(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, refine=True, refine_attention=True, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_iter = n_iter
        self.stoken_size = stoken_size
        self.refine = refine
        self.refine_attention = refine_attention  
        self.scale = dim ** - 0.5
        self.unfold = Unfold(3)
        self.fold = Fold(3)
        
        if refine:
            if refine_attention:
                self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
            else:
                self.stoken_refine = nn.Sequential(
                    nn.Conv2d(dim, dim, 1, 1, 0),
                    nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
                    nn.Conv2d(dim, dim, 1, 1, 0)
                )
        
    def stoken_forward(self, x):
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size
        
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            
        _, _, H, W = x.shape
        hh, ww = H//h, W//w
        
        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww))
        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h*w, C)
        
        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale
                affinity_matrix = affinity_matrix.softmax(-1)
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
                    stoken_features = stoken_features/(affinity_matrix_sum + 1e-12)
        
        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix
        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
        stoken_features = stoken_features/(affinity_matrix_sum.detach() + 1e-12)
        
        if self.refine:
            stoken_features = self.stoken_refine(stoken_features)
            
        stoken_features = self.unfold(stoken_features)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9)
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2)
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
                
        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]
        
        return pixel_features
    
    def direct_forward(self, x):
        stoken_features = x
        if self.refine:
            stoken_features = self.stoken_refine(stoken_features)
        return stoken_features
        
    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)

class StokenAttentionLayer(nn.Module):
    def __init__(self, dim, n_iter, stoken_size, 
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5):
        super().__init__()
        self.layerscale = layerscale
        self.pos_embed = ResDWC(dim, 3)
        self.norm1 = LayerNorm2d(dim)
        self.attn = StokenAttention(dim, stoken_size=stoken_size, 
                                    n_iter=n_iter,                                     
                                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                    attn_drop=attn_drop, proj_drop=drop)   
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer, drop=drop)
                
        if layerscale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
        
    def forward(self, x):
        x = self.pos_embed(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp2(self.norm2(x))) 
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp2(self.norm2(x)))        
        return x

class BasicLayer(nn.Module):        
    def __init__(self, num_layers, dim, n_iter, stoken_size, 
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5,
                 downsample=False,
                 use_checkpoint=False, checkpoint_num=None):
        super().__init__()        
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
                
        self.blocks = nn.ModuleList([StokenAttentionLayer(
                                           dim=dim[0],  n_iter=n_iter, stoken_size=stoken_size,                                           
                                           num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                           drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                           act_layer=act_layer, 
                                           layerscale=layerscale, init_values=init_values) for i in range(num_layers)])
                                                                                           
        if downsample:            
            self.downsample = PatchMerging(dim[0], dim[1])
        else:
            self.downsample = None
         
    def forward(self, x):
        # 遍歷此階段的所有 block
        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint and idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        # 將通過 block 後的特徵圖儲存為此階段的輸出
        stage_output = x
        
        # 如果有降採樣層，則處理特徵圖以作為下一階段的輸入
        if self.downsample is not None:
            x = self.downsample(x)
            
        # 返回給下一階段的輸入 x 和此階段的特徵輸出 stage_output
        return x, stage_output
       
class PatchEmbed(nn.Module):        
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),            
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class STViT(nn.Module):   
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 embed_dim=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 n_iter=[1, 1, 1, 1], stoken_size=[8, 4, 2, 1],                
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 projection=None, freeze_bn=False,
                 use_checkpoint=False, checkpoint_num=[0,0,0,0], 
                 layerscale=[False, False, False, False], init_values=1e-6, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim        
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        self.in_chans = in_chans
        self.img_size = img_size
        
        self.freeze_bn = freeze_bn

        self.patch_embed = PatchEmbed(in_chans, embed_dim[0])
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
                
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(num_layers=depths[i_layer],
                               dim=[embed_dim[i_layer], embed_dim[i_layer+1] if i_layer<self.num_layers-1 else None],                              
                               n_iter=n_iter[i_layer],
                               stoken_size=to_2tuple(stoken_size[i_layer]),                                                       
                               num_heads=num_heads[i_layer], 
                               mlp_ratio=self.mlp_ratio, 
                               qkv_bias=qkv_bias, qk_scale=qk_scale, 
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               downsample=i_layer < self.num_layers - 1,
                               use_checkpoint=use_checkpoint,
                               checkpoint_num=checkpoint_num[i_layer],                               
                               layerscale=layerscale[i_layer],
                               init_values=init_values)
            self.layers.append(layer)
    
        # 為了潛在用途保留分類頭和投影層，但在特徵提取的主前向傳播中不使用
        self.proj = nn.Conv2d(self.num_features, projection, 1) if projection else None
        if self.proj:
            self.norm = nn.BatchNorm2d(projection)
        else:
            self.norm = nn.BatchNorm2d(self.num_features) # 為最後的特徵圖進行 Norm
        self.swish = MemoryEfficientSwish()        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(projection or self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        
        # --- 新增 width_list 計算 ---
        self.width_list = []
        try:
            self.eval() 
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size)
            with torch.no_grad():
                 features = self.forward(dummy_input)
            self.width_list = [f.size(1) for f in features]
            self.train()
        except Exception as e:
            print(f"在為 width_list 計算進行虛擬前向傳播時出錯: {e}")
            print("將 width_list 設置為 embed_dim 作為備用方案。")
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
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def forward_features(self, x):
        feature_outputs = []
        x = self.patch_embed(x)        
        x = self.pos_drop(x)
        
        for layer in self.layers:
            # layer 返回 (下一階段的輸入, 此階段的輸出)
            x, stage_output = layer(x)
            feature_outputs.append(stage_output)
        
        return feature_outputs

    def forward(self, x):
        # 現在返回一個特徵圖列表，符合 Ultralytics 等框架的預期
        x = self.forward_features(x)
        return x

@register_model
def stvit_small(pretrained=False, **kwargs):
    model = STViT(embed_dim=[64, 128, 320, 512],
                    depths=[3, 5, 9, 3],
                    num_heads=[1, 2, 5, 8],
                    n_iter=[1, 1, 1, 1], 
                    stoken_size=[8, 4, 1, 1],
                    projection=1024,                    
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=kwargs.get('drop_rate', 0.0),
                    drop_path_rate=kwargs.get('drop_path_rate', 0.1),
                    use_checkpoint=False,
                    checkpoint_num = [0, 0, 0, 0],
                    layerscale=[False, False, False, False],
                    init_values=1e-5,
                    **kwargs)
    model.default_cfg = _cfg()
    return model    

@register_model
def stvit_base(pretrained=False, **kwargs):
    model = STViT(embed_dim=[96, 192, 384, 512],
                    depths=[4, 6, 14, 6],
                    num_heads=[2, 3, 6, 8],
                    n_iter=[1, 1, 1, 1], 
                    stoken_size=[8, 4, 1, 1],
                    projection=1024,                   
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=kwargs.get('drop_rate', 0.0),
                    drop_path_rate=kwargs.get('drop_path_rate', 0.1), 
                    use_checkpoint=False,
                    checkpoint_num = [0, 0, 0, 0],
                    layerscale=[False, False, True, True],
                    init_values=1e-6,
                    **kwargs)
    model.default_cfg = _cfg()
    return model   

@register_model
def stvit_large(pretrained=False, **kwargs):
    model = STViT(embed_dim=[96, 192, 448, 640],
                    depths=[4, 7, 19, 8],
                    num_heads=[2, 3, 7, 10],
                    n_iter=[1, 1, 1, 1], 
                    stoken_size=[8, 4, 1, 1],
                    projection=1024,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=kwargs.get('drop_rate', 0.0),
                    drop_path_rate=kwargs.get('drop_path_rate', 0.1), 
                    use_checkpoint=False,
                    checkpoint_num=[4,7,15,0],
                    layerscale=[False, False, True, True],
                    init_values=1e-6,
                    **kwargs)
    model.default_cfg = _cfg()
    return model

if __name__ == '__main__':
    img_h, img_w = 224, 224
    print("--- Creating STViT Small model ---")
    model = stvit_base(img_size=img_h)
    print("Model created successfully.")
    # width_list 應該在初始化時自動計算
    print("Calculated width_list:", model.width_list)

    # 測試前向傳播
    input_tensor = torch.rand(2, 3, img_h, img_w)
    print(f"\n--- Testing STViT Small forward pass (Input: {input_tensor.shape}) ---")

    model.eval()
    try:
        with torch.no_grad():
            # 輸出現在是一個張量列表
            output_features = model(input_tensor)
        print("Forward pass successful.")
        print("Output is a list of feature maps.")
        print("Output feature shapes:")
        for i, features in enumerate(output_features):
            # Shape 應該是 [B, C_i, H_i, W_i]
            print(f"Stage {i+1}: {features.shape}")

        # 驗證 width_list 是否與執行的輸出通道匹配
        runtime_widths = [f.size(1) for f in output_features]
        print("\nRuntime output feature channels:", runtime_widths)
        assert model.width_list == runtime_widths, "Width list mismatch!"
        print("Width list verified successfully.")
        
        # 驗證通道數是否正確
        expected_widths = [96, 192, 384, 512]
        assert runtime_widths == expected_widths, f"Channel dimensions are incorrect! Expected {expected_widths}, but got {runtime_widths}"
        print(f"Channel dimensions verified successfully: {runtime_widths}")


    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()