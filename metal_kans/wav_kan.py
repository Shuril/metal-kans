"""
Direct Metal Fused WavKAN Layer (Continuous Wavelets: Mexican Hat, Morlet, DOG).
Multiresolution localization in time and frequency executed entirely in GPU registers.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional
from .device import get_metal_bridge


class WavKAN:
    """
    Direct Metal Fused WavKAN Layer.

    Parameters:
        in_features: Number of input features.
        out_features: Number of output features.
        num_wavelets: Number of wavelet scales/translations (default: 8).
        wavelet_type: "mexican_hat" (default), "morlet", or "dog".
        bias: Whether to add trainable additive bias (default: True).
        use_base: Whether to include residual SiLU base connection (default: True).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_wavelets: int = 8,
        wavelet_type: str = "mexican_hat",
        bias: bool = True,
        use_base: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.num_wavelets = num_wavelets
        self.wavelet_type_str = wavelet_type.lower()
        if self.wavelet_type_str == "morlet":
            self.wavelet_type = 1
        elif self.wavelet_type_str in ("dog", "derivative_of_gaussian"):
            self.wavelet_type = 2
        else:
            self.wavelet_type = 0  # mexican_hat
        self.has_base = 1 if use_base else 0
        self.has_bias = 1 if bias else 0

        # Translation & scale parameters
        self.translation = np.random.uniform(-1.0, 1.0, (in_features, num_wavelets)).astype(np.float32)
        self._scale = np.random.uniform(0.5, 2.0, (in_features, num_wavelets)).astype(np.float32)
        self._inv_scale = (1.0 / (np.abs(self._scale) + 1e-4)).astype(np.float32)

        bound = 1.0 / math.sqrt(in_features)
        self.w_wav = np.random.uniform(-bound, bound, (out_features, in_features * num_wavelets)).astype(np.float32)
        self.w_base = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32) if use_base else np.zeros((1,), dtype=np.float32)
        self.bias = np.zeros((out_features,), dtype=np.float32) if bias else np.zeros((1,), dtype=np.float32)

        self._bridge = get_metal_bridge()

    @property
    def dtype(self) -> np.dtype:
        return self.w_wav.dtype

    def half(self) -> WavKAN:
        """Converts layer parameters to FP16 half precision."""
        self.w_wav = self.w_wav.astype(np.float16)
        self.translation = self.translation.astype(np.float16)
        self._inv_scale = self._inv_scale.astype(np.float16)
        if self.has_base:
            self.w_base = self.w_base.astype(np.float16)
        if self.has_bias:
            self.bias = self.bias.astype(np.float16)
        return self

    def float(self) -> WavKAN:
        """Converts layer parameters to FP32 single precision."""
        self.w_wav = self.w_wav.astype(np.float32)
        self.translation = self.translation.astype(np.float32)
        self._inv_scale = self._inv_scale.astype(np.float32)
        if self.has_base:
            self.w_base = self.w_base.astype(np.float32)
        if self.has_bias:
            self.bias = self.bias.astype(np.float32)
        return self

    @property
    def scale(self) -> np.ndarray:
        return self._scale

    @scale.setter
    def scale(self, val: np.ndarray):
        target_dtype = self.w_wav.dtype
        self._scale = np.asarray(val, dtype=target_dtype)
        self._inv_scale = (1.0 / (np.abs(self._scale) + 1e-4)).astype(target_dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes fused Metal forward pass."""
        target_dtype = self.w_wav.dtype
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
            self._bridge.metal_kan_wavkan_forward_fp16(
                x.ctypes.data,
                self.w_wav.ctypes.data,
                self.w_base.ctypes.data,
                self.translation.ctypes.data,
                self._inv_scale.ctypes.data,
                self.bias.ctypes.data,
                y.ctypes.data,
                B, D_in, self.out_features, self.num_wavelets, self.wavelet_type,
                self.has_base, self.has_bias
            )
        else:
            self._bridge.metal_kan_wavkan_forward(
                x.ctypes.data,
                self.w_wav.ctypes.data,
                self.w_base.ctypes.data,
                self.translation.ctypes.data,
                self._inv_scale.ctypes.data,
                self.bias.ctypes.data,
                y.ctypes.data,
                B, D_in, self.out_features, self.num_wavelets, self.wavelet_type,
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

        return self._bridge.benchmark_metal_wavkan(
            x_flat.ctypes.data,
            self.w_wav.ctypes.data,
            self.w_base.ctypes.data,
            self.translation.ctypes.data,
            self._inv_scale.ctypes.data,
            self.bias.ctypes.data,
            y.ctypes.data,
            B, D_in, self.out_features, self.num_wavelets, self.wavelet_type,
            self.has_base, self.has_bias,
            warmup, iters
        )
