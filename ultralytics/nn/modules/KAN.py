from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
import numpy as np
import math
from torch.autograd import Function
 
__all__ = ['RBFKANConv2d', 'ReLUKANConv2d', 'KANConv2d', 'FasterKANConv2d', 'WavKANConv2d', 'ChebyKANConv2d', 'JacobiKANConv2d', 'FastKANConv2d', 'GRAMKANConv2d']
 
#  ################################################各种工具###########################################################
class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
 
    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)
 
 
class RSWAFFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, inv_denominator, train_grid, train_inv_denominator):
        # Compute the forward pass
        # print('\n')
        # print(f"Forward pass - grid: {(grid[0].item(),grid[-1].item())}, inv_denominator: {inv_denominator.item()}")
 
        # print(f"grid.shape: {grid.shape }")
        # print(f"grid: {(grid[0],grid[-1]) }")
        # print(f"inv_denominator.shape: {inv_denominator.shape }")
        # print(f"inv_denominator: {inv_denominator }")
        diff = (input[..., None] - grid)
        diff_mul = diff.mul(inv_denominator)
        tanh_diff = torch.tanh(diff)
        tanh_diff_deriviative = -tanh_diff.mul(tanh_diff) + 1  # sech^2(x) = 1 - tanh^2(x)
 
        # Save tensors for backward pass
        ctx.save_for_backward(input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator)
        ctx.train_grid = train_grid
        ctx.train_inv_denominator = train_inv_denominator
 
        return tanh_diff_deriviative
 
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator = ctx.saved_tensors
        grad_grid = None
        grad_inv_denominator = None
 
        # print(f"tanh_diff_deriviative shape: {tanh_diff_deriviative.shape }")
        # print(f"tanh_diff shape: {tanh_diff.shape }")
        # print(f"grad_output shape: {grad_output.shape }")
 
        # Compute the backward pass for the input
        grad_input = -2 * tanh_diff * tanh_diff_deriviative * grad_output
        # print(f"Backward pass 1 - grad_input: {(grad_input.min().item(), grad_input.max().item())}")
        # print(f"grad_input shape: {grad_input.shape }")
        # print(f"grad_input.sum(dim=-1): {grad_input.sum(dim=-1).shape}")
        grad_input = grad_input.sum(dim=-1).mul(inv_denominator)
        # print(f"Backward pass 2 - grad_input: {(grad_input.min().item(), grad_input.max().item())}")
        # print(f"grad_input: {grad_input}")
        # print(f"grad_input shape: {grad_input.shape }")
 
        # Compute the backward pass for grid
        if ctx.train_grid:
            # print('\n')
            # print(f"grad_grid shape: {grad_grid.shape }")
            grad_grid = -inv_denominator * grad_output.sum(dim=0).sum(
                dim=0)  # -(inv_denominator * grad_output * tanh_diff_deriviative).sum(dim=0) #-inv_denominator * grad_output.sum(dim=0).sum(dim=0)
            # print(f"Backward pass - grad_grid: {(grad_grid[0].item(),grad_grid[-1].item())}")
            # print(f"grad_grid.shape: {grad_grid.shape }")
            # print(f"grad_grid: {(grad_grid[0],grad_grid[-1]) }")
            # print(f"inv_denominator shape: {inv_denominator.shape }")
            # print(f"grad_grid shape: {grad_grid.shape }")
 
        # Compute the backward pass for inv_denominator
        if ctx.train_inv_denominator:
            grad_inv_denominator = (
                        grad_output * diff).sum()  # (grad_output * diff * tanh_diff_deriviative).sum() #(grad_output* diff).sum()
            # print(f"Backward pass - grad_inv_denominator: {grad_inv_denominator.item()}")
            # print(f"diff shape: {diff.shape }")
 
            # print(f"grad_inv_denominator shape: {grad_inv_denominator.shape }")
            # print(f"grad_inv_denominator : {grad_inv_denominator }")
 
        return grad_input, grad_grid, grad_inv_denominator, None, None  # same number as tensors or parameters
 
 
class ReflectionalSwitchFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -1.2,
            grid_max: float = 0.2,
            num_grids: int = 8,
            exponent: int = 2,
            inv_denominator: float = 0.5,
            train_grid: bool = False,
            train_inv_denominator: bool = False,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.train_grid = torch.tensor(train_grid, dtype=torch.bool)
        self.train_inv_denominator = torch.tensor(train_inv_denominator, dtype=torch.bool)
        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        # print(f"grid initial shape: {self.grid.shape }")
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator, dtype=torch.float32),
                                                  requires_grad=train_inv_denominator)  # Cache the inverse of the denominator
 
    def forward(self, x):
        return RSWAFFunction.apply(x, self.grid, self.inv_denominator, self.train_grid, self.train_inv_denominator)
 
 
# ####################################各种激活函数##############################################################
class KANLayer(nn.Module):
    def __init__(self, input_features, output_features, grid_size=5, spline_order=3, base_activation=nn.GELU,
                 grid_range=[-1, 1]):
        super(KANLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
 
        # The number of points in the grid for the spline interpolation.
        self.grid_size = grid_size
        # The order of the spline used in the interpolation.
        self.spline_order = spline_order
        # Activation function used for the initial transformation of the input.
        self.base_activation = base_activation()
        # The range of values over which the grid for spline interpolation is defined.
        self.grid_range = grid_range
 
        # Initialize the base weights with random values for the linear transformation.
        self.base_weight = nn.Parameter(torch.randn(output_features, input_features))
        # Initialize the spline weights with random values for the spline transformation.
        self.spline_weight = nn.Parameter(torch.randn(output_features, input_features, grid_size + spline_order))
        # Add a layer normalization for stabilizing the output of this layer.
        self.layer_norm = nn.LayerNorm(output_features)
        # Add a PReLU activation for this layer to provide a learnable non-linearity.
        self.prelu = nn.PReLU()
 
        # Compute the grid values based on the specified range and grid size.
        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        ).expand(input_features, -1).contiguous()
 
        # Initialize the weights using Kaiming uniform distribution for better initial values.
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.spline_weight, nonlinearity='linear')
 
    def forward(self, x):
        # Process each layer using the defined base weights, spline weights, norms, and activations.
        grid = self.grid.to(x.device)
        # Move the input tensor to the device where the weights are located.
 
        # Perform the base linear transformation followed by the activation function.
        base_output = F.linear(self.base_activation(x), self.base_weight)
        x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations.
        # Compute the basis for the spline using intervals and input values.
        bases = ((x_uns >= grid[:, :-1]) & (x_uns < grid[:, 1:])).to(x.dtype).to(x.device)
 
        # Compute the spline basis over multiple orders.
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[:, :-(k + 1)]
            right_intervals = grid[:, k:-1]
            delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
                                right_intervals - left_intervals)
            bases = ((x_uns - left_intervals) / delta * bases[:, :, :-1]) + \
                    ((grid[:, k + 1:] - x_uns) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        bases = bases.contiguous()
 
        # Compute the spline transformation and combine it with the base transformation.
        spline_output = F.linear(bases.view(x.size(0), -1), self.spline_weight.view(self.spline_weight.size(0), -1))
        # Apply layer normalization and PReLU activation to the combined output.
        x = self.prelu(self.layer_norm(base_output + spline_output))
 
        return x
 
 
class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)
 
    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
 
 
class FastKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            use_base_update: bool = True,
            base_activation=nn.SiLU,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation()
            self.base_linear = nn.Linear(input_dim, output_dim)
 
    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret
 
 
# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
 
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))
 
    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        # x = torch.tanh(x)
        x = torch.clamp(x, -1.0, 1.0)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y
 
 
class GRAMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, degree=3, act=nn.SiLU):
        super(GRAMLayer, self).__init__()
 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degrees = degree
 
        self.act = act()
 
        self.norm = nn.LayerNorm(out_channels).to(dtype=torch.float32)
 
        self.beta_weights = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))
 
        self.grams_basis_weights = nn.Parameter(
            torch.zeros(in_channels, out_channels, degree + 1, dtype=torch.float32)
        )
 
        self.base_weights = nn.Parameter(
            torch.zeros(out_channels, in_channels, dtype=torch.float32)
        )
 
        self.init_weights()
 
    def init_weights(self):
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / (self.in_channels * (self.degrees + 1.0)),
        )
 
        nn.init.xavier_uniform_(self.grams_basis_weights)
 
        nn.init.xavier_uniform_(self.base_weights)
 
    def beta(self, n, m):
        return (
                ((m + n) * (m - n) * n ** 2) / (m ** 2 / (4.0 * n ** 2 - 1.0))
        ) * self.beta_weights[n]
 
    @lru_cache(maxsize=128)
    def gram_poly(self, x, degree):
        p0 = x.new_ones(x.size())
 
        if degree == 0:
            return p0.unsqueeze(-1)
 
        p1 = x
        grams_basis = [p0, p1]
 
        for i in range(2, degree + 1):
            p2 = x * p1 - self.beta(i - 1, i) * p0
            grams_basis.append(p2)
            p0, p1 = p1, p2
 
        return torch.stack(grams_basis, dim=-1)
 
    def forward(self, x):
 
        basis = F.linear(self.act(x), self.base_weights)
 
        x = torch.tanh(x).contiguous()
 
        grams_basis = self.act(self.gram_poly(x, self.degrees))
 
        y = einsum(
            grams_basis,
            self.grams_basis_weights,
            "b l d, l o d -> b o",
        )
 
        y = self.act(self.norm(y + basis))
 
        y = y.view(-1, self.out_channels)
 
        return y
 
 
class WavKANLayer(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(WavKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
 
        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
 
        # Linear weights for combining outputs
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight1 = nn.Parameter(torch.Tensor(out_features,
                                                 in_features))  # not used; you may like to use it for wieghting base activation and adding it like Spl-KAN paper
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))
 
        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
 
        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU()
 
        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)
 
    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x
 
        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded
 
        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2) - 1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
 
        elif self.wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet
            wavelet = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
        elif self.wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi
 
            def meyer_aux(v):
                return torch.where(v <= 1 / 2, torch.ones_like(v),
                                   torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))
 
            def nu(t):
                return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)
 
            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
        elif self.wavelet_type == 'shannon':
            # Windowing the sinc function to limit its support
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)
 
            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype,
                                          device=x_scaled.device)
            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window
 
            # You can try many more wavelet types ...
        else:
            raise ValueError("Unsupported wavelet type")
        wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
        wavelet_output = wavelet_weighted.sum(dim=2)
        return wavelet_output
 
    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        # You may like test the cases like Spl-KAN
        # wav_output = F.linear(wavelet_output, self.weight)
        base_output = F.linear(self.base_activation(x), self.weight1)
 
        # base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output + base_output
 
        # Apply batch normalization
        return self.bn(combined_output)
 
 
class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0, act=nn.SiLU):
        super(JacobiKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree
 
        self.act = act()
        self.norm = nn.LayerNorm(output_dim,).to(dtype=torch.float32)
 
        self.base_weights = nn.Parameter(
            torch.zeros(output_dim, input_dim, dtype=torch.float32)
        )
 
        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
 
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        nn.init.xavier_uniform_(self.base_weights)
 
    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
 
        basis = F.linear(self.act(x), self.base_weights)
 
        # Since Jacobian polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:  ## degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k = (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 1) / (2 * i * (i + self.a + self.b))
            theta_k1 = (2 * i + self.a + self.b - 1) * (self.a * self.a - self.b * self.b) / (
                    2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            theta_k2 = (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b) / (
                    i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone() - theta_k2 * jacobi[:, :,
                                                                                                  i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
 
        y = self.act(self.norm(y + basis))
        return y
 
 
class ReLUKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int, train_ab: bool = True):
        super().__init__()
        self.g, self.k, self.r = g, k, 4 * g * g / ((k + 1) * (k + 1))
        self.input_size, self.output_size = input_size, output_size
        phase_low = np.arange(-k, g) / g
        phase_height = phase_low + (k + 1) / g
        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])),
                                      requires_grad=train_ab)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])),
                                         requires_grad=train_ab)
        self.equal_size_conv = nn.Conv2d(1, output_size, (g + k, input_size))
 
    def forward(self, x):
        # Expand dimensions of x to match the shape of self.phase_low
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))
 
        # Perform the subtraction with broadcasting
        x1 = torch.relu(x_expanded - self.phase_low)
        x2 = torch.relu(self.phase_height - x_expanded)
 
        # Continue with the rest of the operations
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
        # x = x.reshape((len(x), self.output_size, 1))
        x = x.reshape((len(x), self.output_size))
        return x
 
 
class SplineLinear_fstr(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)
 
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)  # Using Xavier Uniform initialization
 
 
class FasterKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -1.2,
            grid_max: float = 0.2,
            num_grids: int = 8,
            exponent: int = 2,
            inv_denominator: float = 0.5,
            train_grid: bool = False,
            train_inv_denominator: bool = False,
            # use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.667,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = ReflectionalSwitchFunction(grid_min, grid_max, num_grids, exponent, inv_denominator, train_grid,
                                              train_inv_denominator)
        self.spline_linear = SplineLinear_fstr(input_dim * num_grids, output_dim, spline_weight_init_scale)
        # self.use_base_update = use_base_update
        # if use_base_update:
        #    self.base_activation = base_activation
        #    self.base_linear = nn.Linear(input_dim, output_dim)
 
    def forward(self, x):
        # print("Shape before LayerNorm:", x.shape)  # Debugging line to check the input shape
        x = self.layernorm(x)
        # print("Shape After LayerNorm:", x.shape)
        spline_basis = self.rbf(x).view(x.shape[0], -1)
        # print("spline_basis:", spline_basis.shape)
 
        # print("-------------------------")
        # ret = 0
        ret = self.spline_linear(spline_basis)
        # print("spline_basis.shape[:-2]:", spline_basis.shape[:-2])
        # print("*spline_basis.shape[:-2]:", *spline_basis.shape[:-2])
        # print("spline_basis.view(*spline_basis.shape[:-2], -1):", spline_basis.view(*spline_basis.shape[:-2], -1).shape)
        # print("ret:", ret.shape)
        # print("-------------------------")
        # if self.use_base_update:
        # base = self.base_linear(self.base_activation(x))
        # print("self.base_activation(x):", self.base_activation(x).shape)
        # print("base:", base.shape)
        # print("@@@@@@@@@")
        # ret += base
        return ret
 
 
class RBFLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_min=-2., grid_max=2., num_grids=8, spline_weight_init_scale=0.1):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids), requires_grad=False)
        self.spline_weight = nn.Parameter(torch.randn(in_features * num_grids, out_features) * spline_weight_init_scale)
 
    def forward(self, x):
        x = x.unsqueeze(-1)
        basis = torch.exp(-((x - self.grid) / ((self.grid_max - self.grid_min) / (self.num_grids - 1))) ** 2)
        return basis.reshape(basis.size(0), -1).matmul(self.spline_weight)
 
 
class RBFKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_min=-2., grid_max=2., num_grids=8, use_base_update=True,
                 base_activation=nn.SiLU(), spline_weight_init_scale=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.spline_weight_init_scale = spline_weight_init_scale
        self.rbf_linear = RBFLinear(input_dim, output_dim, grid_min, grid_max, num_grids, spline_weight_init_scale)
        self.base_linear = nn.Linear(input_dim, output_dim) if use_base_update else None
 
    def forward(self, x):
        ret = self.rbf_linear(x)
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret
 
 
class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
 
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
 
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
 
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
 
        self.reset_parameters()
 
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
 
    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
 
        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )
 
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()
 
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).
        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
 
        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        # CSDN作者Snu77注明 此处torch.linalg.lstsq需要torch1.9及以上才可以.
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)
 
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()
 
    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )
 
    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
 
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output
 
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
 
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)
 
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]
 
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )
 
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
 
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))
 
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.
        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.
        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )
 
 
 
# """""""""""""""""""""""""""""""""""""""""""""正式代码"""""""""""""""""""""""""""""""""""""""""""
 
 
class KANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(KANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = KANLinear(in_channels * kernel_size * kernel_size, out_channels)
 
    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels
 
        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)
 
        out_unfold = self.kanlayer(x_unfold)
 
        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)
 
        return out
 
 
class ChebyKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, degree=4):
        super(ChebyKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = ChebyKANLayer(in_channels * kernel_size * kernel_size, out_channels, degree=degree)
 
    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels
 
        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)
 
        out_unfold = self.kanlayer(x_unfold)
 
        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)
 
        return out
 
 
class FastKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(FastKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = FastKANLayer(in_channels * kernel_size * kernel_size, out_channels)
 
    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels
 
        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)
 
        out_unfold = self.kanlayer(x_unfold)
 
        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)
 
        return out
 
 
 
class GRAMKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(GRAMKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = GRAMLayer(in_channels * kernel_size * kernel_size, out_channels)
 
    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels
 
        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)
 
        out_unfold = self.kanlayer(x_unfold)
 
        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)
 
        return out
 
 
 
 
class WavKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, wavelet_type='mexican_hat'):
        super(WavKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = WavKANLayer(in_channels * kernel_size * kernel_size, out_channels, wavelet_type=wavelet_type)
 
    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels
 
        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)
 
        out_unfold = self.kanlayer(x_unfold)
 
        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)
 
        return out
 
 
 
class JacobiKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1,degree=4):
        super(JacobiKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = JacobiKANLayer(in_channels * kernel_size * kernel_size, out_channels, degree=degree)
 
    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels
 
        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)
 
        out_unfold = self.kanlayer(x_unfold)
 
        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)
 
        return out
 
 
 
 
class ReLUKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ReLUKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = ReLUKANLayer(in_channels * kernel_size * kernel_size, 5, 3, out_channels)
 
    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels
 
        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)
 
        out_unfold = self.kanlayer(x_unfold)
 
        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)
 
        return out
 
 
 
 
class FasterKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(FasterKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = FasterKANLayer(in_channels * kernel_size * kernel_size, out_channels)
 
    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels
 
        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)
 
        out_unfold = self.kanlayer(x_unfold)
 
        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)
 
        return out
 
 
 
class RBFKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(RBFKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = RBFKANLayer(in_channels * kernel_size * kernel_size, out_channels)
 
    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels
 
        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)
 
        out_unfold = self.kanlayer(x_unfold)
 
        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)
 
        return out
 
 
 
if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)
    # KANConv2d需要torch1.9以上才可以.
    Convs = ['RBFKANConv2d', 'ReLUKANConv2d','FasterKANConv2d', 'ChebyKANConv2d', 'JacobiKANConv2d', 'FastKANConv2d', 'GRAMKANConv2d']
    qu = ['WavKANConv2d'] # 需要大量显存
    e = ['KANConv2d'] # 需要torch 1.9以上
    with torch.no_grad():
        for i in range(len(Convs)):
            model = eval(Convs[i])
            # Model
            mobilenet_v1 = model(64, 64, kernel_size=3, stride=1, padding=1)
 
            out = mobilenet_v1(image)
            print(out.size())