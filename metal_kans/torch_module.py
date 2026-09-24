"""
PyTorch Integration for metal-KANs.
Drop-in torch.nn.Module layers powered by direct Apple Metal Shading Language (MSL) compute shaders.
Provides seamless autograd interoperability between PyTorch and Metal GPU kernels.
"""

from typing import Union, Sequence, Optional, Any
import numpy as np
import torch
import torch.nn as nn

from .cheby_kan import ChebyKAN
from .fast_kan import FastKAN
from .bspline_kan import BSplineKAN
from .relu_kan import ReLUKAN
from .metal_kan import MetalKAN


class _MetalKANAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, layer: Any) -> torch.Tensor:
        ctx.layer = layer
        device = x.device
        dtype = x.dtype

        # Fast zero-copy CPU view if CPU, or transfer to CPU numpy for Metal execution
        x_np = x.detach().contiguous().cpu().numpy().astype(np.float32)
        y_np = layer.forward(x_np)

        ctx.save_for_backward(x)
        out = torch.from_numpy(y_np).to(device=device, dtype=dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        layer = ctx.layer
        device = grad_output.device
        dtype = grad_output.dtype

        grad_np = grad_output.detach().contiguous().cpu().numpy().astype(np.float32)
        dx_np = layer.backward(grad_np)

        dx = torch.from_numpy(dx_np).to(device=device, dtype=dtype)
        return dx, None


class TorchMetalKANLinear(nn.Module):
    """
    Drop-in PyTorch Linear Layer powered by Direct Metal Shading Language compute shaders.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        basis: Basis type ("cheby", "fastkan", "bspline", "relu")
        degree: Degree or grid size
        bias: Whether to include bias term
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis: str = "cheby",
        degree: int = 4,
        bias: bool = True,
        use_base: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis = basis.lower()
        self.degree = degree
        self.use_bias = bias

        if self.basis == "cheby":
            self.metal_layer = ChebyKAN(in_features, out_features, degree=degree, bias=bias, use_base=use_base)
        elif self.basis in ("fastkan", "rbf"):
            self.metal_layer = FastKAN(in_features, out_features, num_centers=degree, bias=bias, use_base=use_base)
        elif self.basis in ("bspline", "kan"):
            self.metal_layer = BSplineKAN(in_features, out_features, grid_size=degree, bias=bias)
        elif self.basis in ("relu", "relukan"):
            self.metal_layer = ReLUKAN(in_features, out_features, num_grids=degree, bias=bias, use_base=use_base)
        else:
            raise ValueError(f"Unsupported basis type: {basis}")

        # Register trainable PyTorch parameters mapped to Metal weights
        w_main = getattr(self.metal_layer, "w_cheby", getattr(self.metal_layer, "w_rbf", getattr(self.metal_layer, "w_spline", getattr(self.metal_layer, "w_relu", None))))
        if w_main is not None:
            self.weight = nn.Parameter(torch.from_numpy(w_main.copy()))
        else:
            self.weight = nn.Parameter(torch.randn(out_features, in_features))

        if bias and hasattr(self.metal_layer, "bias") and self.metal_layer.bias is not None:
            self.bias_param = nn.Parameter(torch.from_numpy(self.metal_layer.bias.copy()))
        else:
            self.bias_param = None

    def _sync_to_metal(self):
        """Synchronizes PyTorch parameter weights into Metal GPU buffers."""
        w_np = self.weight.detach().cpu().numpy().astype(np.float32)
        if hasattr(self.metal_layer, "w_cheby"):
            self.metal_layer.w_cheby[:] = w_np
        elif hasattr(self.metal_layer, "w_rbf"):
            self.metal_layer.w_rbf[:] = w_np
        elif hasattr(self.metal_layer, "w_spline"):
            self.metal_layer.w_spline[:] = w_np
        elif hasattr(self.metal_layer, "w_relu"):
            self.metal_layer.w_relu[:] = w_np

        if self.bias_param is not None and hasattr(self.metal_layer, "bias"):
            self.metal_layer.bias[:] = self.bias_param.detach().cpu().numpy().astype(np.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._sync_to_metal()
        return _MetalKANAutogradFunction.apply(x, self.metal_layer)


class TorchMetalKAN(nn.Module):
    """
    Multi-layer PyTorch Network powered by Direct Metal KAN kernels.
    """
    def __init__(
        self,
        layers_hidden: Sequence[int],
        basis: str = "cheby",
        degree: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TorchMetalKANLinear(in_f, out_f, basis=basis, degree=degree, bias=bias)
            for in_f, out_f in zip(layers_hidden, layers_hidden[1:])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
