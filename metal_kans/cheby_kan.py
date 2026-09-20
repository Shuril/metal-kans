"""
Direct Metal Fused ChebyKAN Layer (Chebyshev Polynomials).
Evaluates polynomial basis recurrence and linear projection in GPU registers and threadgroup SRAM.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional
from .device import get_metal_bridge


class ChebyKAN:
    """
    Direct Metal Fused ChebyKAN Layer.

    Parameters:
        in_features: Number of input features.
        out_features: Number of output features.
        degree: Degree of Chebyshev polynomials (default 4).
        bias: Whether to add trainable additive bias (default True).
        use_base: Whether to include residual SiLU base connection (default True).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 4,
        bias: bool = True,
        use_base: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.has_base = 1 if use_base else 0
        self.has_bias = 1 if bias else 0

        # Initialize weights with Kaiming uniform
        bound = 1.0 / math.sqrt(in_features)
        self.w_cheby = np.random.uniform(-bound, bound, (out_features, in_features * degree)).astype(np.float32)
        self.w_base = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32) if use_base else np.zeros((1,), dtype=np.float32)
        self.bias = np.zeros((out_features,), dtype=np.float32) if bias else np.zeros((1,), dtype=np.float32)

        self.grad_w_cheby = np.zeros_like(self.w_cheby)
        self.grad_w_base = np.zeros_like(self.w_base) if use_base else None
        self.grad_bias = np.zeros_like(self.bias) if bias else None
        self._saved_x: Optional[np.ndarray] = None

        self._bridge = get_metal_bridge()

    @property
    def dtype(self) -> np.dtype:
        return self.w_cheby.dtype

    def half(self) -> ChebyKAN:
        """Converts layer parameters to FP16 half precision."""
        self.w_cheby = self.w_cheby.astype(np.float16)
        if self.has_base:
            self.w_base = self.w_base.astype(np.float16)
        if self.has_bias:
            self.bias = self.bias.astype(np.float16)
        return self

    def float(self) -> ChebyKAN:
        """Converts layer parameters to FP32 single precision."""
        self.w_cheby = self.w_cheby.astype(np.float32)
        if self.has_base:
            self.w_base = self.w_base.astype(np.float32)
        if self.has_bias:
            self.bias = self.bias.astype(np.float32)
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Executes fused Metal forward pass.
        Expects 2D NumPy array of shape [B, in_features].
        """
        target_dtype = self.w_cheby.dtype
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
            self._bridge.metal_kan_cheby_forward_fp16(
                x.ctypes.data,
                self.w_cheby.ctypes.data,
                self.w_base.ctypes.data,
                self.bias.ctypes.data,
                y.ctypes.data,
                B, D_in, self.out_features, self.degree,
                self.has_base, self.has_bias
            )
        else:
            self._bridge.metal_kan_cheby_forward(
                x.ctypes.data,
                self.w_cheby.ctypes.data,
                self.w_base.ctypes.data,
                self.bias.ctypes.data,
                y.ctypes.data,
                B, D_in, self.out_features, self.degree,
                self.has_base, self.has_bias
            )

        if len(orig_shape) > 2:
            return y.reshape(*orig_shape[:-1], self.out_features)
        return y

    __call__ = forward

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Executes GPU backward pass: computes weight gradients and returns dX.
        Expects dY with shape matching the forward output.
        """
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

        if self.grad_w_cheby is None:
            self.grad_w_cheby = np.zeros_like(self.w_cheby, dtype=np.float32)
        else:
            self.grad_w_cheby.fill(0)

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

        self._bridge.metal_kan_cheby_backward(
            dY.ctypes.data,
            x.ctypes.data,
            self.w_cheby.ctypes.data,
            w_base_ptr,
            self.grad_w_cheby.ctypes.data,
            grad_base_ptr,
            grad_bias_ptr,
            dX.ctypes.data,
            B, D_in, self.out_features, self.degree,
            self.has_base, self.has_bias
        )

        if hasattr(self, '_saved_shape') and len(self._saved_shape) > 2:
            return dX.reshape(self._saved_shape)
        return dX

    def zero_grad(self) -> None:
        """Zeros stored parameter gradients."""
        if self.grad_w_cheby is not None:
            self.grad_w_cheby.fill(0)
        if self.grad_w_base is not None:
            self.grad_w_base.fill(0)
        if self.grad_bias is not None:
            self.grad_bias.fill(0)

    def benchmark(self, x: np.ndarray, warmup: int = 10, iters: int = 50) -> float:
        """Benchmarks kernel execution time in milliseconds."""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        elif x.dtype != np.float32:
            x = x.astype(np.float32)

        x_flat = x.reshape(-1, self.in_features)
        if not x_flat.flags['C_CONTIGUOUS']:
            x_flat = np.ascontiguousarray(x_flat)

        B, D_in = x_flat.shape
        y = np.empty((B, self.out_features), dtype=np.float32)

        return self._bridge.benchmark_metal_cheby(
            x_flat.ctypes.data,
            self.w_cheby.ctypes.data,
            self.w_base.ctypes.data,
            self.bias.ctypes.data,
            y.ctypes.data,
            B, D_in, self.out_features, self.degree,
            self.has_base, self.has_bias,
            warmup, iters
        )
