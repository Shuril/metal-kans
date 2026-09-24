"""
Direct Metal Fused RationalKAN Layer (Padé-Chebyshev Rational Functions).
Evaluates P(x) / (1 + |Q(x)|) directly in GPU registers for pole and singularity modeling.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional
from .device import get_metal_bridge


class RationalKAN:
    """
    Direct Metal Fused RationalKAN Layer.

    Approximates activations as trainable rational functions:
        phi(x) = P(x) / (1 + |Q(x)|)
    where P and Q are Chebyshev polynomial expansions.

    Parameters:
        in_features: Number of input features.
        out_features: Number of output features.
        p_degree: Numerator Chebyshev degree (default: 4).
        q_degree: Denominator Chebyshev degree (default: 2).
        bias: Whether to add trainable additive bias (default: True).
        use_base: Whether to include residual SiLU base connection (default: True).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        p_degree: int = 4,
        q_degree: int = 2,
        bias: bool = True,
        use_base: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.p_degree = p_degree
        self.q_degree = q_degree
        self.has_base = 1 if use_base else 0
        self.has_bias = 1 if bias else 0

        bound = 1.0 / math.sqrt(in_features)
        self.w_p = np.random.uniform(-bound, bound, (out_features, in_features * p_degree)).astype(np.float32)
        self.w_q = np.random.uniform(-bound * 0.1, bound * 0.1, (out_features, in_features * q_degree)).astype(np.float32)
        self.w_base = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32) if use_base else np.zeros((1,), dtype=np.float32)
        self.bias = np.zeros((out_features,), dtype=np.float32) if bias else np.zeros((1,), dtype=np.float32)

        self._bridge = get_metal_bridge()

    @property
    def dtype(self) -> np.dtype:
        return self.w_p.dtype

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

        self._bridge.metal_kan_rational_forward(
            x.ctypes.data,
            self.w_p.ctypes.data,
            self.w_q.ctypes.data,
            self.w_base.ctypes.data,
            self.bias.ctypes.data,
            y.ctypes.data,
            B, D_in, self.out_features, self.p_degree, self.q_degree,
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

        return self._bridge.benchmark_metal_rational(
            x_flat.ctypes.data,
            self.w_p.ctypes.data,
            self.w_q.ctypes.data,
            self.w_base.ctypes.data,
            self.bias.ctypes.data,
            y.ctypes.data,
            B, D_in, self.out_features, self.p_degree, self.q_degree,
            self.has_base, self.has_bias,
            warmup, iters
        )
