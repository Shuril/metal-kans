"""
Direct Metal Fused JacobiKAN Layer (Orthogonal Jacobi Polynomials).
Recurrence evaluation for parameters (alpha, beta) executed directly in GPU registers.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional
from .device import get_metal_bridge


class JacobiKAN:
    """
    Direct Metal Fused JacobiKAN Layer.

    Parameters:
        in_features: Number of input features.
        out_features: Number of output features.
        degree: Degree of Jacobi polynomials (default: 4).
        alpha: First Jacobi parameter (alpha > -1). Default: 0.0 (Legendre).
        beta: Second Jacobi parameter (beta > -1). Default: 0.0 (Legendre).
        bias: Whether to add trainable additive bias (default: True).
        use_base: Whether to include residual SiLU base connection (default: True).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 4,
        alpha: float = 0.0,
        beta: float = 0.0,
        bias: bool = True,
        use_base: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.has_base = 1 if use_base else 0
        self.has_bias = 1 if bias else 0

        bound = 1.0 / math.sqrt(in_features)
        self.w_jacobi = np.random.uniform(-bound, bound, (out_features, in_features * degree)).astype(np.float32)
        self.w_base = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32) if use_base else np.zeros((1,), dtype=np.float32)
        self.bias = np.zeros((out_features,), dtype=np.float32) if bias else np.zeros((1,), dtype=np.float32)

        self._bridge = get_metal_bridge()

    @property
    def dtype(self) -> np.dtype:
        return self.w_jacobi.dtype

    def half(self) -> JacobiKAN:
        """Converts layer parameters to FP16 half precision."""
        self.w_jacobi = self.w_jacobi.astype(np.float16)
        if self.has_base:
            self.w_base = self.w_base.astype(np.float16)
        if self.has_bias:
            self.bias = self.bias.astype(np.float16)
        return self

    def float(self) -> JacobiKAN:
        """Converts layer parameters to FP32 single precision."""
        self.w_jacobi = self.w_jacobi.astype(np.float32)
        if self.has_base:
            self.w_base = self.w_base.astype(np.float32)
        if self.has_bias:
            self.bias = self.bias.astype(np.float32)
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes fused Metal forward pass."""
        target_dtype = self.w_jacobi.dtype
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
            self._bridge.metal_kan_jacobi_forward_fp16(
                x.ctypes.data,
                self.w_jacobi.ctypes.data,
                self.w_base.ctypes.data,
                self.bias.ctypes.data,
                y.ctypes.data,
                B, D_in, self.out_features, self.degree, self.alpha, self.beta,
                self.has_base, self.has_bias
            )
        else:
            self._bridge.metal_kan_jacobi_forward(
                x.ctypes.data,
                self.w_jacobi.ctypes.data,
                self.w_base.ctypes.data,
                self.bias.ctypes.data,
                y.ctypes.data,
                B, D_in, self.out_features, self.degree, self.alpha, self.beta,
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

        return self._bridge.benchmark_metal_jacobi(
            x_flat.ctypes.data,
            self.w_jacobi.ctypes.data,
            self.w_base.ctypes.data,
            self.bias.ctypes.data,
            y.ctypes.data,
            B, D_in, self.out_features, self.degree, self.alpha, self.beta,
            self.has_base, self.has_bias,
            warmup, iters
        )
