import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 核心 JBU/UPA 算法部分 (保留並適配)
# ==========================================

@torch.no_grad()
def _build_offsets(R_max: int, device: torch.device):
    """Return flattened neighbor offsets within square radius R_max."""
    offs = torch.arange(-R_max, R_max + 1, device=device)
    dY, dX = torch.meshgrid(offs, offs, indexing='ij')
    return dY.reshape(-1), dX.reshape(-1)

def _tanh_bound_pi(raw: torch.Tensor):
    """Map R -> (-pi, pi) smoothly."""
    return math.pi * torch.tanh(raw)

def gather_lr_scalar_general(map_lr: torch.Tensor, Ui: torch.Tensor, Vi: torch.Tensor):
    """
    Gather values from specific coordinates.
    """
    Hl, Wl = map_lr.shape[-2:]
    flat = Hl * Wl
    idx = (Ui * Wl + Vi).reshape(-1)
    t = map_lr.view(flat)
    vals = t.index_select(0, idx)
    return vals.view(Ui.shape[0], Ui.shape[1], Ui.shape[2])

def gs_jbu_aniso_noparent(
    feat_lr: torch.Tensor,     # [1,C,Hl,Wl]
    guide_hr: torch.Tensor,    # [1,3,Hh,Wh]
    scale: int,
    sigma_x_map: torch.Tensor, # [1,1,Hl,Wl]
    sigma_y_map: torch.Tensor, # [1,1,Hl,Wl]
    theta_map: torch.Tensor,   # [1,1,Hl,Wl]
    sigma_r_map: torch.Tensor, # [1,1,Hl,Wl]
    R_max: int = 4,
    alpha_dyn: float = 2.0,
    C_chunk: int = 512,
    Nn_chunk: int = 81,
    center_mode: str = "nearest",
    use_autocast: bool = True,
):
    """
    執行各向異性聯合雙邊上採樣 (Anisotropic JBU)。
    注意：此函數原本設計為處理 Batch=1。
    """
    _, C, Hl, Wl = feat_lr.shape
    _, _, Hh, Wh = guide_hr.shape

    dev = feat_lr.device
    dtype_feat = feat_lr.dtype
    dtype_acc = torch.float32

    # 建立 HR 坐標網格
    y = torch.arange(Hh, device=dev, dtype=torch.float32)
    x = torch.arange(Wh, device=dev, dtype=torch.float32)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    u = (Y + 0.5) / scale - 0.5
    v = (X + 0.5) / scale - 0.5

    if center_mode == "nearest":
        uc = torch.round(u).clamp(0, Hl - 1).to(torch.long)
        vc = torch.round(v).clamp(0, Wl - 1).to(torch.long)
    else: # floor
        uc = torch.floor(u).clamp(0, Hl - 1).to(torch.long)
        vc = torch.floor(v).clamp(0, Wl - 1).to(torch.long)

    # 動態半徑計算
    sigma_eff = torch.maximum(sigma_x_map, sigma_y_map)
    sigma_eff_hr = F.interpolate(sigma_eff, (Hh, Wh), mode='bilinear', align_corners=False)
    R_map = torch.ceil(alpha_dyn * sigma_eff_hr).clamp_(min=1, max=R_max).to(torch.int64)

    dY_all, dX_all = _build_offsets(R_max, dev)
    K = dY_all.numel()

    num_s = torch.zeros(C, Hh, Wh, device=dev, dtype=dtype_acc)
    den_s = torch.zeros(   Hh, Wh, device=dev, dtype=dtype_acc)
    m     = torch.full((Hh, Wh), float("-inf"), device=dev, dtype=dtype_acc)

    guide32 = guide_hr.to(torch.float32, copy=False)
    sx_map32 = sigma_x_map.to(torch.float32, copy=False)
    sy_map32 = sigma_y_map.to(torch.float32, copy=False)
    th_map32 = theta_map.to(torch.float32, copy=False)
    sr_map32 = sigma_r_map.to(torch.float32, copy=False)

    flat = Hl * Wl
    feat_flat = feat_lr[0].permute(1, 2, 0).reshape(flat, C).contiguous()

    # 將 Guide 下採樣到 LR 空間以進行查詢
    guide_lr = F.interpolate(guide32, size=(Hl, Wl), mode='bilinear', align_corners=False)

    autocast_ctx = torch.cuda.amp.autocast(enabled=use_autocast, dtype=torch.float16)

    with autocast_ctx:
        for n0 in range(0, K, Nn_chunk):
            n1 = min(n0 + Nn_chunk, K)
            dY = dY_all[n0:n1].view(-1, 1, 1)
            dX = dX_all[n0:n1].view(-1, 1, 1)
            Bn = dY.shape[0]

            Ui = torch.clamp(uc.unsqueeze(0) + dY, 0, Hl - 1)
            Vi = torch.clamp(vc.unsqueeze(0) + dX, 0, Wl - 1)

            rad2 = (dY ** 2 + dX ** 2)
            mask = (rad2 <= (R_map ** 2)).squeeze(0).squeeze(0)

            cy = (Ui.to(torch.float32) + 0.5) * scale - 0.5
            cx = (Vi.to(torch.float32) + 0.5) * scale - 0.5
            dx = X.unsqueeze(0) - cx
            dy = Y.unsqueeze(0) - cy

            sx = gather_lr_scalar_general(sx_map32, Ui, Vi).clamp_min(1e-6)
            sy = gather_lr_scalar_general(sy_map32, Ui, Vi).clamp_min(1e-6)
            th = gather_lr_scalar_general(th_map32, Ui, Vi)
            sr = gather_lr_scalar_general(sr_map32, Ui, Vi).clamp_min(1e-6)

            cos_t, sin_t = torch.cos(th), torch.sin(th)
            x_p = dx * cos_t + dy * sin_t
            y_p = -dx * sin_t + dy * cos_t
            
            # 空間權重
            log_ws = -(x_p ** 2) / (2 * sx ** 2 + 1e-8) - (y_p ** 2) / (2 * sy ** 2 + 1e-8)

            # 範圍(Range)權重 / 引導圖差異
            # 注意：這裡硬編碼了 guide 是 3 通道 (RGB/Lab)
            g0 = gather_lr_scalar_general(guide_lr[0, 0, ...], Ui, Vi)
            g1 = gather_lr_scalar_general(guide_lr[0, 1, ...], Ui, Vi)
            g2 = gather_lr_scalar_general(guide_lr[0, 2, ...], Ui, Vi)
            diff2 = (guide32[0, 0] - g0) ** 2 + (guide32[0, 1] - g1) ** 2 + (guide32[0, 2] - g2) ** 2
            log_wr = -diff2 / (2.0 * sr * sr + 1e-8)

            log_w = log_ws + log_wr
            log_w = torch.where(mask, log_w, torch.full_like(log_w, float("-inf")))

            m_chunk = torch.max(log_w, dim=0).values
            valid = torch.isfinite(m_chunk)
            if not valid.any():
                continue

            m_new = m.clone()
            m_new[valid] = torch.maximum(m[valid], m_chunk[valid])

            delta = (m - m_new).clamp_max(0)
            scale_old = torch.ones_like(den_s)
            scale_old[valid] = torch.exp(delta[valid])
            den_s.mul_(scale_old)
            num_s.mul_(scale_old.unsqueeze(0))

            log_w_shift = log_w - m_new.unsqueeze(0)
            log_w_shift[:, ~valid] = float("-inf")
            s = torch.exp(log_w_shift)

            den_s.add_(s.sum(0))

            idx_flat = (Ui * Wl + Vi).reshape(-1)
            # 特徵聚合
            for c0 in range(0, C, C_chunk):
                c1 = min(c0 + C_chunk, C)
                feat_sel = feat_flat.index_select(0, idx_flat)[:, c0:c1]
                feat_sel = feat_sel.view(Bn, Hh, Wh, c1 - c0)
                num_s[c0:c1].add_((feat_sel * s[..., None]).sum(dim=0).permute(2, 0, 1))

            m = m_new

    out_raw = (num_s / den_s.clamp_min(1e-8)).unsqueeze(0).to(dtype_feat)
    
    # 若權重過小，回退到雙線性插值
    fallback = F.interpolate(feat_lr, size=(Hh, Wh), mode='bilinear', align_corners=False)
    tiny = (den_s < 1e-6).unsqueeze(0).unsqueeze(0)
    out = torch.where(tiny, fallback, out_raw)
    return out


# ==========================================
# 修改後的 UPA 類別 (介面適配 CARAFE)
# ==========================================

class UPA(nn.Module):
    """
    Unified Pixelwise Anisotropic Upsampling (UPA) adapted to match CARAFE usage.
    
    Usage:
        model = UPA(c1, c2, kernel_size=5, up_factor=2)
        out = model(x)
        
    Mechanism:
        1. Predicts JBU parameters (sigma_x, sigma_y, theta, sigma_r) from input features.
        2. Generates a pseudo-HR guide by compressing and upsampling the input.
        3. Applies Anisotropic JBU.
    """
    def __init__(self, c1, c2, kernel_size=5, up_factor=2):
        super(UPA, self).__init__()
        self.up_factor = int(up_factor)
        # CARAFE 的 kernel_size 是一個卷積核大小，在 JBU 中我們將其映射為半徑 R
        self.R_max = kernel_size // 2 
        self.alpha_dyn = 2.0
        
        # 1. 參數預測器 (Parameter Predictor)
        # 類似 CARAFE 的 Kernel Prediction Module
        # 輸入: c1 -> 輸出: 4 通道 (sx, sy, theta, sr)
        mid_channels = max(c1 // 4, 32)
        self.param_encoder = nn.Sequential(
            nn.Conv2d(c1, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 4, kernel_size=3, padding=1)
        )
        
        # 2. 引導圖生成器 (Guide Generator)
        # JBU 需要一個 3 通道的 Guide。我們將高維特徵壓縮為 3 通道作為結構引導。
        self.guide_compress = nn.Conv2d(c1, 3, 1)
        
        # 3. 輸出投影 (Output Projection)
        # 如果輸入輸出通道不同，最後進行調整
        if c1 != c2:
            self.out_conv = nn.Conv2d(c1, c2, 1)
        else:
            self.out_conv = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W] Input features
        Returns:
            out: [N, C_out, H*up, W*up] Upsampled features
        """
        N, C, H, W = x.shape
        scale = self.up_factor
        Hh, Wh = H * scale, W * scale
        
        # 1. 預測 JBU 參數 [N, 4, H, W]
        # 我們讓網絡學習 raw 數值，然後映射到物理意義的範圍
        params = self.param_encoder(x)
        
        sx_raw = params[:, 0:1, :, :]
        sy_raw = params[:, 1:2, :, :]
        th_raw = params[:, 2:3, :, :]
        sr_raw = params[:, 3:4, :, :]
        
        # 參數映射 (Mapping functions)
        sigma_x = torch.exp(sx_raw)      # > 0
        sigma_y = torch.exp(sy_raw)      # > 0
        theta   = _tanh_bound_pi(th_raw) # (-pi, pi)
        sigma_r = torch.exp(sr_raw)      # > 0

        # 2. 準備 Guide
        # 自引導策略：壓縮通道 -> 雙線性上採樣到目標解析度
        guide_lr = self.guide_compress(x) # [N, 3, H, W]
        guide_hr = F.interpolate(guide_lr, scale_factor=scale, mode='bilinear', align_corners=False)

        # 3. 執行 JBU 上採樣
        # 由於 gs_jbu_aniso_noparent 是基於 scatter/gather 的複雜運算，目前僅支持 batch=1
        # 我們在 batch 維度上循環 (雖然效率較低，但保證了功能正確性與介面一致)
        outputs = []
        for i in range(N):
            # 取出單個樣本
            feat_i = x[i:i+1]       # [1, C, H, W]
            guide_i = guide_hr[i:i+1] # [1, 3, Hh, Wh]
            sx_i = sigma_x[i:i+1]
            sy_i = sigma_y[i:i+1]
            th_i = theta[i:i+1]
            sr_i = sigma_r[i:i+1]
            
            # 呼叫 JBU 核心 (這部分運算較重)
            out_i = gs_jbu_aniso_noparent(
                feat_lr=feat_i,
                guide_hr=guide_i,
                scale=scale,
                sigma_x_map=sx_i,
                sigma_y_map=sy_i,
                theta_map=th_i,
                sigma_r_map=sr_i,
                R_max=self.R_max,
                alpha_dyn=self.alpha_dyn,
                center_mode="nearest",
                use_autocast=True
            )
            outputs.append(out_i)
        
        # 拼接回 Batch
        out_tensor = torch.cat(outputs, dim=0) # [N, C, Hh, Wh]
        
        # 4. 最終通道調整
        out_tensor = self.out_conv(out_tensor)
        
        return out_tensor

# 測試用例
if __name__ == '__main__':
    # 模擬 CARAFE 的用法
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c1, c2 = 64, 64
    upsample_layer = UPA(c1, c2, kernel_size=5, up_factor=2).to(device)
    
    input_tensor = torch.randn(2, 64, 32, 32).to(device) # Batch=2
    
    output = upsample_layer(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")