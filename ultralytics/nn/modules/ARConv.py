# --- ARConv Module Code (ARConv.py) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback # For detailed error printing in test block

class ARConv(nn.Module):
    def __init__(
        self,
        inc,
        outc,
        kernel_size=3, # Note: kernel_size is not directly used for the main deformable convs
        padding=1,     # Padding value for the initial ZeroPad2d
        stride=1,      # Stride for the auxiliary convolutions (m_conv, b_conv, p_conv, l_conv, w_conv)
        l_max=9,
        w_max=9,
        flag=False, # Note: flag argument is defined but not used
        modulation=True # Note: modulation argument is defined but not used to conditionally disable
    ):
        super(ARConv, self).__init__()
        self.inc = inc
        self.outc = outc
        self.padding = padding # Padding for self.zero_pad
        self.stride = stride   # Stride for auxiliary convs, affects H_out/W_out calculation relative to H_in/W_in

        self.lmax = l_max
        self.wmax = w_max

        # Supported kernel shapes (Height, Width) and corresponding strides for the final conv
        self.i_list = [33, 35, 53, 37, 73, 55, 57, 75, 77] # Format: KernelHeight*10 + KernelWidth
        self.convs = nn.ModuleList([
            nn.Conv2d(inc, outc,
                      kernel_size=(i // 10, i % 10),
                      stride=(i // 10, i % 10), # Stride matches kernel size
                      padding=0) # No padding needed here, handled by sampling/reshape
            for i in self.i_list
        ])

        # Modulation & bias branches (using the provided stride)
        self.m_conv = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1, stride=self.stride),
            nn.LeakyReLU(inplace=False), nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, 3, padding=1, stride=self.stride),
            nn.LeakyReLU(inplace=False), nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, 3, padding=1, stride=self.stride),
            nn.Tanh()
        )
        self.b_conv = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1, stride=self.stride),
            nn.LeakyReLU(inplace=False), nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, 3, padding=1, stride=self.stride),
            nn.LeakyReLU(inplace=False), nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, 3, padding=1, stride=self.stride)
        )

        # Offset pre-processing branch (using the provided stride)
        self.p_conv = nn.Sequential(
            nn.Conv2d(inc, inc, 3, padding=1, stride=self.stride),
            nn.BatchNorm2d(inc), nn.LeakyReLU(inplace=False),
            nn.Conv2d(inc, inc, 3, padding=1, stride=self.stride),
            nn.BatchNorm2d(inc), nn.LeakyReLU(inplace=False)
        )
        # Kernel height/width prediction branches (using the provided stride)
        self.l_conv = nn.Sequential(
            nn.Conv2d(inc, 1, 3, padding=1, stride=self.stride),
            nn.BatchNorm2d(1), nn.LeakyReLU(inplace=False),
            nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.w_conv = nn.Sequential(
            nn.Conv2d(inc, 1, 3, padding=1, stride=self.stride),
            nn.BatchNorm2d(1), nn.LeakyReLU(inplace=False),
            nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid()
        )

        # Zero padding for the input tensor 'x'
        self.zero_pad = nn.ZeroPad2d(self.padding)
        # Dropout for the sampled features before final convolution
        self.dropout2 = nn.Dropout2d(0.3)

        # Backward hooks for learning rate scaling (applied to first Conv of relevant sequences)
        self.hooks = []
        for seq in (self.m_conv, self.b_conv, self.p_conv, self.l_conv, self.w_conv):
             # Ensure the sequence is not empty and has a Conv2d layer at index 0
             if len(seq) > 0 and isinstance(seq[0], nn.Conv2d):
                 # print(f"Registering hook for {seq[0]}") # Debug print
                 self.hooks.append(seq[0].register_full_backward_hook(self._set_lr))

        # Buffer for storing fixed N_X, N_Y after initial epochs
        self.register_buffer('reserved_NXY', torch.tensor([3, 3], dtype=torch.int32), persistent=True)

        # State variables
        self.epoch = 0
        self.hw_range = [1, self.lmax] # Default range for predicted kernel size

    @staticmethod
    def _set_lr(module, grad_in, grad_out):
        """
        Backward hook to scale gradients for the module's parameters (weight & bias).
        Clones parameter gradients before modification. Leaves grad_input untouched.
        """
        # grad_in for Conv2d is typically (grad_input, grad_weight, grad_bias)

        # Create a mutable list from the input tuple
        new_grad_in = list(grad_in)

        # Check if grad_weight exists and is not None (index 1)
        if len(new_grad_in) > 1 and new_grad_in[1] is not None:
            # Clone and scale the weight gradient
            new_grad_in[1] = new_grad_in[1].clone() * 0.1
            # print(f"Hook: Scaled grad_weight for {module}") # Optional debug print

        # Check if grad_bias exists and is not None (index 2)
        if len(new_grad_in) > 2 and new_grad_in[2] is not None:
            # Clone and scale the bias gradient
            new_grad_in[2] = new_grad_in[2].clone() * 0.1
            # print(f"Hook: Scaled grad_bias for {module}") # Optional debug print

        # Return the modified gradients as a tuple
        return tuple(new_grad_in)


    def __del__(self):
        """Remove hooks when object is deleted."""
        # print("Removing ARConv hooks...") # Debug print
        for handle in self.hooks:
            handle.remove()
        self.hooks = [] # Clear the list

    def set_epoch(self, e):
        self.epoch = e

    def set_hw_range(self, hw):
        assert isinstance(hw, list) and len(hw)==2, "hw_range must be a list of length 2"
        self.hw_range = hw

    def forward(self, x):
        # Get current state
        epoch = self.epoch
        hw0, hw1 = self.hw_range # Min/Max sampling kernel dimension range

        # Modulation & bias branches
        m = self.m_conv(x)
        b = self.b_conv(x)
        B, C_out, H_out, W_out = m.shape

        # Offset pre-processing features
        offset_features = self.p_conv(x * 100) # Scaling factor might need adjustment

        # Predict kernel dimensions (l, w) per pixel
        l = self.l_conv(offset_features) * (hw1 - 1) + 1 # Scale sigmoid output [0,1] to [1, hw1]
        w = self.w_conv(offset_features) * (hw1 - 1) + 1

        # Determine N_X, N_Y (sampling grid dimensions)
        if self.training and epoch <= 100: # Dynamic during early training
            scale = hw1 // 9 if not (hw0 == 1 and hw1 == 3) else 1
            scale = max(1, scale) # Avoid scale=0
            N_X_float = l.mean().item() / scale
            N_Y_float = w.mean().item() / scale

            def phi(z_float): # Make odd and clamp to [3, 7]
                z = int(round(z_float))
                z = z - 1 if z % 2 == 0 else z
                return max(3, min(7, z))

            N_X, N_Y = phi(N_X_float), phi(N_Y_float)

            if epoch == 100: # Store the final dynamic values
                 # Ensure tensor is created on the same device as input x
                 if self.reserved_NXY.device != x.device:
                     self.reserved_NXY = torch.tensor([N_X, N_Y], dtype=torch.int32, device=x.device)
                 else:
                     # Update in place if already on correct device
                     self.reserved_NXY.data = torch.tensor([N_X, N_Y], dtype=torch.int32, device=x.device)


        else: # Fixed after epoch 100 or during evaluation
            # Ensure NXY is on the correct device if model moved after initialization
            if self.reserved_NXY.device != x.device:
                self.reserved_NXY = self.reserved_NXY.to(x.device)
            N_X, N_Y = self.reserved_NXY.tolist()

        # Ensure N_X, N_Y correspond to a supported kernel shape in i_list
        target_i = N_X * 10 + N_Y
        if target_i not in self.i_list:
            # Fallback to the closest supported shape
            distances = [abs(N_X - (i // 10)) + abs(N_Y - (i % 10)) for i in self.i_list]
            closest_idx = distances.index(min(distances))
            target_i = self.i_list[closest_idx]
            N_X, N_Y = target_i // 10, target_i % 10
            # Update reserved_NXY if this fallback happens at epoch 100
            if self.training and epoch == 100:
                if self.reserved_NXY.device != x.device:
                    self.reserved_NXY = torch.tensor([N_X, N_Y], dtype=torch.int32, device=x.device)
                else:
                    self.reserved_NXY.data = torch.tensor([N_X, N_Y], dtype=torch.int32, device=x.device)


        N = N_X * N_Y # Total number of sampling points

        # Pad the original input tensor 'x' if self.padding > 0
        x_pad = self.zero_pad(x) if self.padding > 0 else x
        B_pad, C_in_pad, H_pad, W_pad = x_pad.shape

        # Calculate absolute sampling coordinates 'p' in the padded input space
        p = self._get_p(l, w, x.dtype, N_X, N_Y, H_out, W_out, B, self.stride, self.padding)
        p = p.permute(0, 2, 3, 1) # Shape: (B, H_out, W_out, 2*N)

        # --- Manual Bilinear Interpolation Sampling (using floor as per original logic) ---
        p_y = torch.clamp(p[..., :N], 0, H_pad - 1) # Y coordinates (rows)
        p_x = torch.clamp(p[..., N:], 0, W_pad - 1) # X coordinates (columns)
        q_lt_y = p_y.floor()
        q_lt_x = p_x.floor()
        q_lt = torch.cat([q_lt_y, q_lt_x], dim=-1) # Shape: (B, H, W, 2*N) with (y1..yN, x1..xN)

        # Sample features from x_pad using the calculated integer coordinates q_lt
        x_sampled = self._get_x_q(x_pad, q_lt.long(), N) # Shape: (B, C_in, H, W, N)

        # Reshape sampled features into grid format for the final convolution
        x_off = self._reshape_x_offset(x_sampled, N_X, N_Y) # Shape: (B, C_in, H*NX, W*NY)

        # Apply dropout to the reshaped features
        x_off = self.dropout2(x_off)

        # Convolve the reshaped features using the selected conv layer
        conv_idx = self.i_list.index(target_i)
        out = self.convs[conv_idx](x_off) # Shape: (B, C_out, H_out, W_out)

        # Apply modulation and bias
        out = out * m + b

        return out

    def _get_p_n(self, N, dtype, n_x, n_y):
        """Calculates the relative offsets (integer grid) for the N_X x N_Y sampling pattern."""
        py, px = torch.meshgrid(
            torch.arange(-(n_x - 1) // 2, (n_x - 1) // 2 + 1, dtype=dtype), # Rows (relative y)
            torch.arange(-(n_y - 1) // 2, (n_y - 1) // 2 + 1, dtype=dtype), # Cols (relative x)
            indexing='ij'
        )
        p_n = torch.cat([py.flatten(), px.flatten()], dim=0) # Shape: (2*N) -> (y1..yN, x1..xN)
        return p_n.view(1, 2 * N, 1, 1)

    def _get_p(self, l, w, dtype, nx, ny, h_out, w_out, batch_size, stride, pad):
        """
        Calculates the absolute sampling coordinates p in the padded input space.
        Combines base grid (p0) with scaled relative offsets (scaled_offsets).
        """
        N = nx * ny
        device = l.device # Ensure all tensors are on the same device

        # 1. Base grid coordinates (p0): centers of output pixels in padded input space
        center_offset = (stride - 1) / 2.0
        p0_y_centers = torch.arange(pad + center_offset, pad + center_offset + h_out * stride, stride, dtype=dtype, device=device)
        p0_x_centers = torch.arange(pad + center_offset, pad + center_offset + w_out * stride, stride, dtype=dtype, device=device)
        p0_y = p0_y_centers[:h_out].view(h_out, 1).expand(h_out, w_out)
        p0_x = p0_x_centers[:w_out].view(1, w_out).expand(h_out, w_out)
        p0 = torch.stack([p0_y, p0_x], dim=0) # Stack: (Y coord, X coord) -> shape (2, H_out, W_out)
        p0 = p0.unsqueeze(0).repeat(batch_size, 1, 1, 1) # Shape: (B, 2, H_out, W_out)

        # 2. Relative grid offsets (pn): integer grid scaled by predicted l, w
        pn = self._get_p_n(N, dtype, nx, ny).to(device) # Shape: (1, 2*N, 1, 1) -> (y_rel, x_rel)
        pn_y = pn[:, :N, :, :]
        pn_x = pn[:, N:, :, :]
        step_y = (l / max(1, nx)) # Use nx (kernel height) for l (predicted height)
        step_x = (w / max(1, ny)) # Use ny (kernel width) for w (predicted width)
        scaled_offset_y = step_y * pn_y # Shape (B, N, H, W)
        scaled_offset_x = step_x * pn_x # Shape (B, N, H, W)
        scaled_offsets = torch.cat([scaled_offset_y, scaled_offset_x], dim=1) # Shape (B, 2*N, H, W)

        # 3. Combine base grid (p0) and scaled offsets using broadcasting
        p0_reshaped = p0.unsqueeze(1) # Shape: (B, 1, 2, H, W)
        scaled_offsets_reshaped = scaled_offsets.view(batch_size, N, 2, h_out, w_out) # Shape: (B, N, 2, H, W)
        p_combined = p0_reshaped + scaled_offsets_reshaped # Shape: (B, N, 2, H, W)
        p = p_combined.permute(0, 2, 1, 3, 4).contiguous().view(batch_size, 2 * N, h_out, w_out) # Shape: (B, 2*N, H, W)

        return p

    def _get_x_q(self, x_pad, q, N):
        """Sample features from x_pad at specified integer coordinates q using gather."""
        b, h, w, _ = q.shape # H=H_out, W=W_out
        c = x_pad.size(1) # C_in
        H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

        rows = q[..., :N] # Shape: (B, H, W, N)
        cols = q[..., N:] # Shape: (B, H, W, N)
        idx = rows * W_pad + cols # Shape: (B, H, W, N)

        x_flat = x_pad.view(b, c, -1) # Shape: (B, C_in, H_pad * W_pad)
        idx = idx.view(b, 1, -1).expand(-1, c, -1) # Shape: (B, C_in, H*W*N)

        sampled_flat = x_flat.gather(2, idx.long()) # Shape: (B, C_in, H*W*N)
        sampled = sampled_flat.view(b, c, h, w, N) # Shape: (B, C, H, W, N)
        return sampled

    @staticmethod
    def _reshape_x_offset(x_off, nx, ny):
        """Reshape sampled features from (B, C, H, W, N) to (B, C, H*nx, W*ny) for conv."""
        b, c, h, w, N = x_off.shape
        assert N == nx * ny, f"N={N} does not match nx*ny={nx*ny}"
        x_off = x_off.permute(0, 1, 4, 2, 3)       # (B, C, N, H, W)
        x_off = x_off.reshape(b, c, nx, ny, h, w)  # (B, C, nx, ny, H, W)
        x_off = x_off.permute(0, 1, 2, 4, 3, 5)    # (B, C, nx, H, ny, W) - Group nx with H, ny with W
        x_off = x_off.contiguous().view(b, c, nx * h, ny * w) # (B, C, nx*H, ny*W)
        return x_off