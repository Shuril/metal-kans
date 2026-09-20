"""
Direct Metal Fused B-Spline KAN Layer (Cox-de Boor Algorithm).
Classic Kolmogorov-Arnold Network with adaptive B-spline basis evaluation in GPU registers.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Sequence
from .device import get_metal_bridge


class BSplineKAN:
    """
    Direct Metal Fused B-Spline KAN Layer.

    Parameters:
        in_features: Number of input features.
        out_features: Number of output features.
        grid_size: Number of grid intervals G (default: 5).
        spline_order: Order of the B-spline (e.g. 3 for cubic splines).
        grid_range: Tuple of (min, max) for knot grid (default: (-1.0, 1.0)).
        bias: Whether to add trainable additive bias (default: True).
        use_base: Whether to include residual SiLU base connection (default: True).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Sequence[float] = (-1.0, 1.0),
        bias: bool = True,
        use_base: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_bases = grid_size + 3
        self.has_base = 1 if use_base else 0
        self.has_bias = 1 if bias else 0

        # Closed-form cubic B-splines parameterization: [grid_min, inv_h]
        self.grid_min = float(grid_range[0])
        self.grid_max = float(grid_range[1])
        self.inv_h = float(grid_size) / (self.grid_max - self.grid_min)
        self.grid_params = np.array([self.grid_min, self.inv_h], dtype=np.float32)
        self.grid = self.grid_params

        bound = 1.0 / math.sqrt(in_features)
        self.w_spline = np.random.uniform(-bound, bound, (out_features, in_features * self.num_bases)).astype(np.float32)
        self.w_base = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32) if use_base else np.zeros((1,), dtype=np.float32)
        self.bias = np.zeros((out_features,), dtype=np.float32) if bias else np.zeros((1,), dtype=np.float32)

        self._bridge = get_metal_bridge()

    @property
    def dtype(self) -> np.dtype:
        return self.w_spline.dtype

    def half(self) -> BSplineKAN:
        """Converts layer parameters to FP16 half precision."""
        self.w_spline = self.w_spline.astype(np.float16)
        if self.has_base:
            self.w_base = self.w_base.astype(np.float16)
        if self.has_bias:
            self.bias = self.bias.astype(np.float16)
        return self

    def float(self) -> BSplineKAN:
        """Converts layer parameters to FP32 single precision."""
        self.w_spline = self.w_spline.astype(np.float32)
        if self.has_base:
            self.w_base = self.w_base.astype(np.float32)
        if self.has_bias:
            self.bias = self.bias.astype(np.float32)
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes fused Metal forward pass."""
        target_dtype = self.w_spline.dtype
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=target_dtype)
        elif x.dtype != target_dtype:
            x = x.astype(target_dtype)

        orig_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, self.in_features)
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)

        B, D_in = x.shape
        if D_in != self.in_features:
            raise ValueError(f"Expected in_features={self.in_features}, got {D_in}")

        y = np.empty((B, self.out_features), dtype=target_dtype)

        if target_dtype == np.float16:
            self._bridge.metal_kan_bspline_forward_fp16(
                x.ctypes.data,
                self.w_spline.ctypes.data,
                self.w_base.ctypes.data,
                self.grid.ctypes.data,
                self.bias.ctypes.data,
                y.ctypes.data,
                B, D_in, self.out_features, self.grid_size, self.spline_order,
                self.has_base, self.has_bias
            )
        else:
            self._bridge.metal_kan_bspline_forward(
                x.ctypes.data,
                self.w_spline.ctypes.data,
                self.w_base.ctypes.data,
                self.grid.ctypes.data,
                self.bias.ctypes.data,
                y.ctypes.data,
                B, D_in, self.out_features, self.grid_size, self.spline_order,
                self.has_base, self.has_bias
            )

        if len(orig_shape) > 2:
            return y.reshape(*orig_shape[:-1], self.out_features)
        return y

    __call__ = forward

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

        return self._bridge.benchmark_metal_bspline(
            x_flat.ctypes.data,
            self.w_spline.ctypes.data,
            self.w_base.ctypes.data,
            self.grid.ctypes.data,
            self.bias.ctypes.data,
            y.ctypes.data,
            B, D_in, self.out_features, self.grid_size, self.spline_order,
            self.has_base, self.has_bias,
            warmup, iters
        )


# Alias KAN to BSplineKAN
KAN = BSplineKAN
