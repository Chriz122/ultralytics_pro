import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .conv import Conv
import math


class AgentAttention(nn.Module):
    """ AgentAttention module that dynamically determines H, W in forward pass. """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert int(agent_num**0.5 + 0.5)**2 == agent_num, f"agent_num ({agent_num}) must be a perfect square"

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        pool_size = int(agent_num ** 0.5)
        agent_patch_H, agent_patch_W = pool_size, pool_size

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)

        # --- Initialize biases ---
        # 2D biases (interpolated based on target H, W)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, agent_patch_H, agent_patch_W))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, agent_patch_H, agent_patch_W))
        # 1D biases (interpolated based on target H or W)
        # Storing them with intended spatial dimension at the end for easier interpolation
        placeholder_size = 1 # Will be resized
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, placeholder_size)) # Interpolate last dim (H)
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, placeholder_size)) # Interpolate last dim (W)
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, placeholder_size)) # Interpolate last dim (H)
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, placeholder_size)) # Interpolate last dim (W)

        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, c = x.shape
        h_w = int(math.sqrt(n))
        assert h_w * h_w == n, f"Input sequence length {n} is not a perfect square."
        H, W = h_w, h_w

        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)

        if self.sr_ratio > 1:
            assert H % self.sr_ratio == 0 and W % self.sr_ratio == 0, \
                f"Runtime H={H}, W={W} not divisible by sr_ratio={self.sr_ratio}"
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
            kv_n = n // (self.sr_ratio ** 2)
            kv_H, kv_W = H // self.sr_ratio, W // self.sr_ratio
        else:
            kv = self.kv(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
            kv_n = n
            kv_H, kv_W = H, W
        k, v = kv[0], kv[1]

        q_for_pool = q.permute(0, 2, 1).reshape(b, c, H, W)
        agent_tokens = self.pool(q_for_pool).reshape(b, c, -1).permute(0, 2, 1)

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, kv_n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, kv_n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # --- Agent -> K/V Attention Biases ---
        kv_size = (kv_H, kv_W)
        # Interpolate 2D bias (an_bias)
        position_bias1 = F.interpolate(self.an_bias, size=kv_size, mode='bilinear', align_corners=False) # (num_heads, agent_num, kv_H, kv_W)
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, kv_n).repeat(b, 1, 1, 1) # (b, num_heads, agent_num, kv_n)

        # Interpolate 1D biases (ah_bias, aw_bias)
        # Reshape for interpolate: treat (1 * num_heads * agent_num) as channels, (H_placeholder,) or (W_placeholder,) as spatial dim
        orig_ah_shape = self.ah_bias.shape # (1, num_heads, agent_num, H_placeholder)
        ah_bias_reshaped = self.ah_bias.reshape(orig_ah_shape[0] * orig_ah_shape[1] * orig_ah_shape[2], 1, orig_ah_shape[3]) # (N', C=1, H)
        ah_bias_resized = F.interpolate(ah_bias_reshaped, size=kv_H, mode='linear', align_corners=False) # Interpolate H dim -> (N', C=1, kv_H)
        ah_bias_final = ah_bias_resized.reshape(orig_ah_shape[0], orig_ah_shape[1], orig_ah_shape[2], kv_H).unsqueeze(-1) # (1, num_heads, agent_num, kv_H, 1)

        orig_aw_shape = self.aw_bias.shape # (1, num_heads, agent_num, W_placeholder)
        aw_bias_reshaped = self.aw_bias.reshape(orig_aw_shape[0] * orig_aw_shape[1] * orig_aw_shape[2], 1, orig_aw_shape[3]) # (N', C=1, W)
        aw_bias_resized = F.interpolate(aw_bias_reshaped, size=kv_W, mode='linear', align_corners=False) # Interpolate W dim -> (N', C=1, kv_W)
        aw_bias_final = aw_bias_resized.reshape(orig_aw_shape[0], orig_aw_shape[1], orig_aw_shape[2], kv_W).unsqueeze(-2) # (1, num_heads, agent_num, 1, kv_W)

        # Combine 1D biases
        position_bias2 = (ah_bias_final + aw_bias_final) # Broadcasts to (1, num_heads, agent_num, kv_H, kv_W)
        position_bias2 = position_bias2.reshape(1, num_heads, self.agent_num, kv_n).repeat(b, 1, 1, 1) # (b, num_heads, agent_num, kv_n)

        # Final Agent->KV position bias
        position_bias = position_bias1 + position_bias2

        # Agent->KV Attention calculation
        attn_agent_kv = (agent_tokens * self.scale) @ k.transpose(-2, -1)
        agent_attn = self.softmax(attn_agent_kv + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        # --- Q -> Agent Attention Biases ---
        q_size = (H, W)
        # Interpolate 2D bias (na_bias)
        agent_bias1 = F.interpolate(self.na_bias, size=q_size, mode='bilinear', align_corners=False) # (num_heads, agent_num, H, W)
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, n).permute(0, 1, 3, 2).repeat(b, 1, 1, 1) # (b, num_heads, n, agent_num)

        # Interpolate 1D biases (ha_bias, wa_bias) - Apply same reshape logic
        orig_ha_shape = self.ha_bias.shape # (1, num_heads, agent_num, H_placeholder)
        ha_bias_reshaped = self.ha_bias.reshape(orig_ha_shape[0] * orig_ha_shape[1] * orig_ha_shape[2], 1, orig_ha_shape[3]) # (N', C=1, H)
        ha_bias_resized = F.interpolate(ha_bias_reshaped, size=H, mode='linear', align_corners=False) # Interpolate H dim -> (N', C=1, H)
        ha_bias_final = ha_bias_resized.reshape(orig_ha_shape[0], orig_ha_shape[1], orig_ha_shape[2], H).unsqueeze(-1) # (1, num_heads, agent_num, H, 1)
        # Permute to match (b, num_heads, n, agent_num) structure: need (1, num_heads, H, 1, agent_num)
        ha_bias_final = ha_bias_final.permute(0, 1, 3, 4, 2) # (1, num_heads, H, 1, agent_num)


        orig_wa_shape = self.wa_bias.shape # (1, num_heads, agent_num, W_placeholder)
        wa_bias_reshaped = self.wa_bias.reshape(orig_wa_shape[0] * orig_wa_shape[1] * orig_wa_shape[2], 1, orig_wa_shape[3]) # (N', C=1, W)
        wa_bias_resized = F.interpolate(wa_bias_reshaped, size=W, mode='linear', align_corners=False) # Interpolate W dim -> (N', C=1, W)
        wa_bias_final = wa_bias_resized.reshape(orig_wa_shape[0], orig_wa_shape[1], orig_wa_shape[2], W).unsqueeze(-2) # (1, num_heads, agent_num, 1, W)
        # Permute to match (b, num_heads, n, agent_num) structure: need (1, num_heads, 1, W, agent_num)
        wa_bias_final = wa_bias_final.permute(0, 1, 3, 4, 2) # (1, num_heads, 1, W, agent_num)


        # Combine 1D biases
        agent_bias2 = (ha_bias_final + wa_bias_final) # Broadcasts to (1, num_heads, H, W, agent_num)
        agent_bias2 = agent_bias2.reshape(1, num_heads, n, self.agent_num).repeat(b, 1, 1, 1) # (b, num_heads, n, agent_num)

        # Final Q->Agent position bias
        agent_bias = agent_bias1 + agent_bias2

        # Q->Agent Attention calculation
        attn_q_agent = (q * self.scale) @ agent_tokens.transpose(-2, -1)
        q_attn = self.softmax(attn_q_agent + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        # --- Combine Heads and DWC Path ---
        x = x.transpose(1, 2).reshape(b, n, c)
        v_for_dwc = v.transpose(1, 2).reshape(b, kv_n, c)
        v_for_dwc = v_for_dwc.permute(0, 2, 1).reshape(b, c, kv_H, kv_W)
        if self.sr_ratio > 1:
            v_for_dwc = F.interpolate(v_for_dwc, size=(H, W), mode='bilinear', align_corners=False)
        dwc_out = self.dwc(v_for_dwc)
        dwc_out = dwc_out.permute(0, 2, 3, 1).reshape(b, n, c)
        x = x + dwc_out

        # --- Final Projection ---
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --- PSABlock and C2PSA remain the same as the previous working version ---
# Make sure PSABlock uses Linear FFN and C2PSA takes args correctly from YAML

#===========================================
# Modified PSABlock (No input_resolution in init)
#===========================================
class PSABlock(nn.Module):
    """ PSABlock using AgentAttention, determining H, W dynamically. """
    def __init__(self, c, num_heads=8, qkv_bias=False, sr_ratio=1, agent_num=49,
                 attn_drop=0., proj_drop=0., ffn_exp_ratio=2.0, shortcut=True): # Removed input_resolution, added ffn_exp_ratio
        super().__init__()
        self.attn = AgentAttention(
            dim=c,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            sr_ratio=sr_ratio,
            agent_num=agent_num
        )
        ffn_hidden_dim = int(c * ffn_exp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(c, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(ffn_hidden_dim, c),
            nn.Dropout(proj_drop)
        )
        self.add = shortcut # Controls internal residuals
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        x_attn_input = x.flatten(2).transpose(1, 2).contiguous() # (B, N, C)

        # Attention Block
        normed_x = self.norm1(x_attn_input)
        attn_out = self.attn(normed_x)
        # Residual connection for attention
        x_attn_output = x_attn_input + attn_out # Assumes self.add controls this path implicitly

        # FFN Block
        normed_ffn_input = self.norm2(x_attn_output)
        ffn_out = self.ffn(normed_ffn_input)
        # Residual connection for FFN
        x_ffn_output = x_attn_output + ffn_out # Assumes self.add controls this path implicitly

        # Reshape final output back to (B, C, H, W)
        x_final = x_ffn_output.transpose(1, 2).reshape(B, C, H, W)

        # If shortcut=True in C2PSA's PSABlock call, internal residuals are added.
        # C2PSA handles the parallel branch summation itself.
        return x_final


#===========================================
# Modified C2PSA (No input_resolution in init)
#===========================================
class C2PSA_Agent(nn.Module):
    """ C2PSA using PSABlock with dynamic H/W determination. """
    def __init__(self, c1, c2, n=1, e=0.5, num_heads=8, sr_ratio=1, agent_num=49,
                 qkv_bias=False, attn_drop=0., proj_drop=0., shortcut=True): # Removed input_resolution
        super().__init__()
        assert c1 == c2, "C2PSA requires c1 == c2 typically. Check YAML definition."
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        ffn_exp_ratio_psa = 2.0 # Keep consistent with PSABlock internal

        self.m = nn.Sequential(*(
            PSABlock(
                c=self.c,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                sr_ratio=sr_ratio,
                agent_num=agent_num,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                ffn_exp_ratio=ffn_exp_ratio_psa,
                shortcut=shortcut # Controls internal PSABlock residuals
            ) for _ in range(n)
        ))

    def forward(self, x):
        split_features = self.cv1(x)
        a, b = split_features.split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), dim=1))

# --- Example Usage (Standalone) ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 4
    # C must be divisible by num_heads
    C = 256 # Input/Output channels for C2PSA
    H, W = 20, 20 # Example input resolution (MUST BE SQUARE & divisible by sr_ratio if > 1)
    num_heads_test = 4
    sr_ratio_test = 2
    agent_num_test = 49 # 7x7

    # Create a C2PSA module instance
    c2psa_module = C2PSA(
        c1=C,
        c2=C,
        n=2,
        num_heads=num_heads_test,
        sr_ratio=sr_ratio_test,
        agent_num=agent_num_test,
        # e=0.5, shortcut=True # Defaults
    ).to(device)

    # Create a dummy input tensor
    input_tensor = torch.randn(B, C, H, W).to(device)

    # Perform a forward pass
    print(f"Input shape: {input_tensor.shape}")
    try:
        output_tensor = c2psa_module(input_tensor)
        print(f"Output shape: {output_tensor.shape}")
        assert input_tensor.shape == output_tensor.shape
        print("\nC2PSA with dynamic H/W determination and bias interpolation created and tested successfully.")
    except Exception as e:
        print(f"\nError during forward pass: {e}")
        import traceback
        traceback.print_exc()
