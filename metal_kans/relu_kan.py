"""
Direct Metal Fused ReLUKAN Layer (Piecewise Linear Tent Basis).
Zero transcendental operations with hardware tent basis evaluation and GEMM fusion.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional
from .device import get_metal_bridge


class ReLUKAN:
    """
    Direct Metal Fused ReLUKAN Layer.

    Parameters:
        in_features: Number of input features.
        out_features: Number of output features.
        num_grids: Number of tent basis grid points (default 8).
        grid_range: Tuple of (min, max) range for grid placement (default (-1.0, 1.0)).
        bias: Whether to add trainable additive bias (default True).
        use_base: Whether to include residual SiLU base connection (default True).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_grids: int = 8,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        bias: bool = True,
        use_base: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids
        self.has_base = 1 if use_base else 0
        self.has_bias = 1 if bias else 0

        self.grid = np.linspace(grid_range[0], grid_range[1], num_grids, dtype=np.float32)
        h = (grid_range[1] - grid_range[0]) / (num_grids - 1)
        self.inv_h = float(1.0 / h)

        bound = 1.0 / math.sqrt(in_features)
        self.w_relu = np.random.uniform(-bound, bound, (out_features, in_features * num_grids)).astype(np.float32)
        self.w_base = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32) if use_base else np.zeros((1,), dtype=np.float32)
        self.bias = np.zeros((out_features,), dtype=np.float32) if bias else np.zeros((1,), dtype=np.float32)

        self.grad_w_relu = np.zeros_like(self.w_relu)
        self.grad_w_base = np.zeros_like(self.w_base) if use_base else None
        self.grad_bias = np.zeros_like(self.bias) if bias else None
        self._saved_x: Optional[np.ndarray] = None

        self._bridge = get_metal_bridge()

    @property
    def dtype(self) -> np.dtype:
        return self.w_relu.dtype

    def half(self) -> ReLUKAN:
        """Converts layer parameters to FP16 half precision."""
        self.w_relu = self.w_relu.astype(np.float16)
        self.grid = self.grid.astype(np.float16)
        if self.has_base:
            self.w_base = self.w_base.astype(np.float16)
        if self.has_bias:
            self.bias = self.bias.astype(np.float16)
        return self

    def float(self) -> ReLUKAN:
        """Converts layer parameters to FP32 single precision."""
        self.w_relu = self.w_relu.astype(np.float32)
        self.grid = self.grid.astype(np.float32)
        if self.has_base:
            self.w_base = self.w_base.astype(np.float32)
        if self.has_bias:
            self.bias = self.bias.astype(np.float32)
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes fused Metal forward pass."""
        target_dtype = self.w_relu.dtype
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=target_dtype)
        elif x.dtype != target_dtype:
            x = x.astype(target_dtype)

        orig_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, self.in_features)
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)

        self._saved_x = x
        self._saved_shape = orig_shape

        B, D_in = x.shape
        if D_in != self.in_features:
            raise ValueError(f"Expected in_features={self.in_features}, got {D_in}")

        y = np.empty((B, self.out_features), dtype=target_dtype)

        if target_dtype == np.float16:
            self._bridge.metal_kan_relu_forward_fp16(
                x.ctypes.data,
                self.w_relu.ctypes.data,
                self.w_base.ctypes.data,
                self.grid.ctypes.data,
                self.bias.ctypes.data,
                y.ctypes.data,
                B, D_in, self.out_features, self.num_grids, self.inv_h,
                self.has_base, self.has_bias
            )
        else:
            self._bridge.metal_kan_relu_forward(
                x.ctypes.data,
                self.w_relu.ctypes.data,
                self.w_base.ctypes.data,
                self.grid.ctypes.data,
                self.bias.ctypes.data,
                y.ctypes.data,
                B, D_in, self.out_features, self.num_grids, self.inv_h,
                self.has_base, self.has_bias
            )

        if len(orig_shape) > 2:
            return y.reshape(*orig_shape[:-1], self.out_features)
        return y

    __call__ = forward

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """Executes GPU backward pass: computes weight gradients and returns dX."""
        if self._saved_x is None:
            raise RuntimeError("Cannot run backward before forward() has been called.")

        x = self._saved_x
        B, D_in = x.shape

        if not isinstance(dY, np.ndarray):
            dY = np.asarray(dY, dtype=np.float32)
        elif dY.dtype != np.float32:
            dY = dY.astype(np.float32)

        if dY.ndim > 2:
            dY = dY.reshape(B, self.out_features)
        if not dY.flags['C_CONTIGUOUS']:
            dY = np.ascontiguousarray(dY)

        if self.grad_w_relu is None:
            self.grad_w_relu = np.zeros_like(self.w_relu, dtype=np.float32)
        else:
            self.grad_w_relu.fill(0)

        if self.has_base:
            if self.grad_w_base is None:
                self.grad_w_base = np.zeros_like(self.w_base, dtype=np.float32)
            else:
                self.grad_w_base.fill(0)
        else:
            self.grad_w_base = np.zeros((1,), dtype=np.float32)

        if self.has_bias:
            if self.grad_bias is None:
                self.grad_bias = np.zeros_like(self.bias, dtype=np.float32)
            else:
                self.grad_bias.fill(0)
        else:
            self.grad_bias = np.zeros((1,), dtype=np.float32)

        dX = np.empty((B, D_in), dtype=np.float32)

        grad_base_ptr = self.grad_w_base.ctypes.data if self.has_base else None
        grad_bias_ptr = self.grad_bias.ctypes.data if self.has_bias else None
        w_base_ptr = self.w_base.ctypes.data if self.has_base else None

        self._bridge.metal_kan_relu_backward(
            dY.ctypes.data,
            x.ctypes.data,
            self.w_relu.ctypes.data,
            w_base_ptr,
            self.grid.ctypes.data,
            self.grad_w_relu.ctypes.data,
            grad_base_ptr,
            grad_bias_ptr,
            dX.ctypes.data,
            B, D_in, self.out_features, self.num_grids, self.inv_h,
            self.has_base, self.has_bias
        )

        if hasattr(self, '_saved_shape') and len(self._saved_shape) > 2:
            return dX.reshape(self._saved_shape)
        return dX

    def zero_grad(self) -> None:
        """Zeros stored parameter gradients."""
        if self.grad_w_relu is not None:
            self.grad_w_relu.fill(0)
        if self.grad_w_base is not None:
            self.grad_w_base.fill(0)
        if self.grad_bias is not None:
            self.grad_bias.fill(0)

    def benchmark(self, x: np.ndarray, warmup: int = 10, iters: int = 50) -> float:
        """Benchmarks kernel execution time in milliseconds directly on GPU."""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        elif x.dtype != np.float32:
            x = x.astype(np.float32)

        x_flat = x.reshape(-1, self.in_features)
        if not x_flat.flags['C_CONTIGUOUS']:
            x_flat = np.ascontiguousarray(x_flat)

        B, D_in = x_flat.shape
        y = np.empty((B, self.out_features), dtype=np.float32)

        return self._bridge.benchmark_metal_relu(
            x_flat.ctypes.data,
            self.w_relu.ctypes.data,
            self.w_base.ctypes.data,
            self.grid.ctypes.data,
            self.bias.ctypes.data,
            y.ctypes.data,
            B, D_in, self.out_features, self.num_grids, self.inv_h,
            self.has_base, self.has_bias,
            warmup, iters
        )
