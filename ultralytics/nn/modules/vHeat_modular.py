import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

# --- 基礎組件 (保持不變或微調) ---

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
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

class Heat2D(nn.Module):
    def __init__(self, dim, res=14, hidden_dim=None, infer_mode=False):
        super().__init__()
        self.res = res
        hidden_dim = hidden_dim or dim
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=True), nn.ReLU())
        
        # 緩存屬性
        self.register_buffer("weight_cosn", None, persistent=False)
        self.register_buffer("weight_cosm", None, persistent=False)
        self.register_buffer("weight_exp", None, persistent=False)
        self.last_res = (0, 0)

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, H, W = x.shape
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())
        x, z = x.chunk(chunks=2, dim=-1)

        # 緩存機制：避免每次都重新計算 DCT 權重
        if (H, W) != self.last_res:
            self.weight_cosn = self.get_cos_map(H, device=x.device).detach()
            self.weight_cosm = self.get_cos_map(W, device=x.device).detach()
            self.weight_exp = self.get_decay_map((H, W), device=x.device).detach()
            self.last_res = (H, W)

        N, M = self.weight_cosn.shape[0], self.weight_cosm.shape[0]
        
        # 2D DCT
        x = F.conv1d(x.contiguous().view(B, H, -1), self.weight_cosn.contiguous().view(N, H, 1))
        x = F.conv1d(x.contiguous().view(-1, W, C), self.weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)

        # 頻域調製
        weight_exp = torch.pow(self.weight_exp[:, :, None], self.to_k(freq_embed))
        x = torch.einsum("bnmc,nmc -> bnmc", x, weight_exp)

        # 2D IDCT
        x = F.conv1d(x.contiguous().view(B, N, -1), self.weight_cosn.t().contiguous().view(H, N, 1))
        x = F.conv1d(x.contiguous().view(-1, M, C), self.weight_cosm.t().contiguous().view(W, M, 1)).contiguous().view(B, H, W, -1)

        x = self.out_norm(x)
        x = x * F.silu(z)
        x = self.out_linear(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class HeatBlock(nn.Module):
    def __init__(self, dim, res=14, drop_path=0., layer_scale=None, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.op = Heat2D(dim=dim, res=res)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = LayerNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x, freq_embed):
        # Post-norm 架構
        if self.layer_scale:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.norm1(self.op(x, freq_embed)))
            x = x + self.drop_path(self.gamma2[:, None, None] * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(self.norm1(self.op(x, freq_embed)))
            x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

# --- YOLO 兼容模塊 ---

class vHeatStem(nn.Module):
    """vHeat 的 Stem 層：將圖像快速下採樣 4 倍"""
    def __init__(self, c1, c2, kernel_size=3, stride=2):
        super().__init__()
        # vHeat Stem: 3 -> c2//2 (stride 2) -> c2 (stride 2) => total stride 4
        # 這裡我們假設 YOLO 傳入的 c1=3, c2=embed_dim
        mid_c = c2 // 2
        self.conv1 = nn.Conv2d(c1, mid_c, kernel_size=3, stride=2, padding=1)
        self.norm1 = LayerNorm2d(mid_c)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid_c, c2, kernel_size=3, stride=2, padding=1)
        self.norm2 = LayerNorm2d(c2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x

class vHeatDownsample(nn.Module):
    """層間下採樣"""
    def __init__(self, c1, c2):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(c2)
        )
    
    def forward(self, x):
        return self.down(x)

class vHeatStage(nn.Module):
    """
    對應 YOLO 的主要 Block (如 C2f, C3)。
    包含多個 HeatBlock 以及該 Stage 專屬的 freq_embed。
    """
    def __init__(self, c1, c2, n=1, res=64, drop_path=0.0):
        # c1: 輸入通道, c2: 輸出通道 (通常 c1==c2)
        # n: 深度 (blocks 數量)
        # res: 初始化的解析度 (用於 freq_embed)
        super().__init__()
        assert c1 == c2, "vHeatStage blocks keep channels same. Use vHeatDownsample for channel change."
        self.dim = c1
        self.res = res
        
        self.blocks = nn.ModuleList([
            HeatBlock(dim=c1, res=res, drop_path=drop_path) 
            for _ in range(n)
        ])
        
        # 每個 Stage 獨立學習一個 freq_embed
        self.freq_embed = nn.Parameter(torch.zeros(res, res, c1))
        trunc_normal_(self.freq_embed, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 處理 freq_embed 的空間尺寸匹配 (動態插值)
        # vHeat 原論文的核心：Visual Embeddings 隨解析度變化
        if self.freq_embed.shape[0] != H or self.freq_embed.shape[1] != W:
            # (H_orig, W_orig, C) -> (1, C, H_orig, W_orig)
            freq_embed_resized = self.freq_embed.permute(2, 0, 1).unsqueeze(0)
            freq_embed_resized = F.interpolate(
                freq_embed_resized, size=(H, W), mode='bilinear', align_corners=False
            )
            # -> (H, W, C)
            freq_embed_to_pass = freq_embed_resized.squeeze(0).permute(1, 2, 0)
        else:
            freq_embed_to_pass = self.freq_embed

        for block in self.blocks:
            x = block(x, freq_embed_to_pass)
            
        return x