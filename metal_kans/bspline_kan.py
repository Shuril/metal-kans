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
        self.num_bases = grid_size + spline_order
        self.num_knots = grid_size + 2 * spline_order + 1
        self.has_base = 1 if use_base else 0
        self.has_bias = 1 if bias else 0

        # Uniform knot vector extended by spline_order on both sides
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid_start = grid_range[0] - spline_order * h
        grid_end = grid_range[1] + spline_order * h
        knots_1d = np.linspace(grid_start, grid_end, self.num_knots, dtype=np.float32)
        self.grid = np.tile(knots_1d, (in_features, 1)).astype(np.float32)

        bound = 1.0 / math.sqrt(in_features)
        self.w_spline = np.random.uniform(-bound, bound, (out_features, in_features * self.num_bases)).astype(np.float32)
        self.w_base = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32) if use_base else np.zeros((1,), dtype=np.float32)
        self.bias = np.zeros((out_features,), dtype=np.float32) if bias else np.zeros((1,), dtype=np.float32)

        self._bridge = get_metal_bridge()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes fused Metal forward pass."""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        elif x.dtype != np.float32:
            x = x.astype(np.float32)

        orig_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, self.in_features)
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)

        B, D_in = x.shape
        if D_in != self.in_features:
            raise ValueError(f"Expected in_features={self.in_features}, got {D_in}")

        y = np.empty((B, self.out_features), dtype=np.float32)

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


# Alias KAN to BSplineKAN
KAN = BSplineKAN
