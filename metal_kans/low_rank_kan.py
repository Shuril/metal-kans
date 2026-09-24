"""
Direct Metal Fused LowRankKAN Layer (LoRA / Bottleneck Factorization).
Reduces parameter complexity from O(d_out * d_in * K) to O((d_out + d_in * K) * rank).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Sequence
from .fast_kan import FastKAN
from .device import get_metal_bridge


class LowRankKAN:
    """
    Direct Metal Fused LowRankKAN Layer.

    Uses a rank-factorized bottleneck projection:
    W_spline = W_up @ W_down, where W_down has rank r << D_out.

    Parameters:
        in_features: Number of input features.
        out_features: Number of output features.
        rank: Bottleneck rank r (default: 8).
        num_grids: Number of basis grid points (default: 8).
        bias: Whether to add trainable additive bias (default: True).
        use_base: Whether to include residual SiLU base connection (default: True).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        num_grids: int = 8,
        bias: bool = True,
        use_base: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, out_features, in_features * num_grids)
        self.num_grids = num_grids
        self.has_bias = 1 if bias else 0
        self.has_base = 1 if use_base else 0
        self.use_base = use_base

        # Down-projection fused kernel to rank r
        self.down_layer = FastKAN(
            in_features=in_features,
            out_features=self.rank,
            num_centers=num_grids,
            bias=False,
            use_base=False,
        )
        self.w_down = self.down_layer.w_rbf
        self.grid = self.down_layer.grid
        self.inv_denominator = self.down_layer.inv_denominator

        # Up-projection matrix (rank -> out_features)
        bound = 1.0 / math.sqrt(self.rank)
        self.w_up = np.random.uniform(-bound, bound, (out_features, self.rank)).astype(np.float32)

        # Base connection (SiLU) and bias
        bound_base = 1.0 / math.sqrt(in_features)
        self.w_base = np.random.uniform(-bound_base, bound_base, (out_features, in_features)).astype(np.float32) if use_base else np.zeros((1,), dtype=np.float32)
        self.bias = np.zeros((out_features,), dtype=np.float32) if bias else np.zeros((1,), dtype=np.float32)

        self._bridge = get_metal_bridge()

    @property
    def dtype(self) -> np.dtype:
        return self.w_up.dtype

    def half(self) -> LowRankKAN:
        """Converts layer parameters to FP16 half precision."""
        self.down_layer.half()
        self.w_down = self.down_layer.w_rbf
        self.grid = self.down_layer.grid
        self.w_up = self.w_up.astype(np.float16)
        if self.has_base:
            self.w_base = self.w_base.astype(np.float16)
        if self.has_bias:
            self.bias = self.bias.astype(np.float16)
        return self

    def float(self) -> LowRankKAN:
        """Converts layer parameters to FP32 single precision."""
        self.down_layer.float()
        self.w_down = self.down_layer.w_rbf
        self.grid = self.down_layer.grid
        self.w_up = self.w_up.astype(np.float32)
        if self.has_base:
            self.w_base = self.w_base.astype(np.float32)
        if self.has_bias:
            self.bias = self.bias.astype(np.float32)
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes chained down-projection and up-projection on Metal GPU."""
        target_dtype = self.w_up.dtype
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
            self._bridge.metal_kan_lowrank_forward_fp16(
                x.ctypes.data,
                self.w_down.ctypes.data,
                self.w_up.ctypes.data,
                self.w_base.ctypes.data,
                self.grid.ctypes.data,
                self.bias.ctypes.data,
                y.ctypes.data,
                B, D_in, self.out_features, self.rank, self.num_grids, self.inv_denominator,
                self.has_base, self.has_bias
            )
        else:
            self._bridge.metal_kan_lowrank_forward(
                x.ctypes.data,
                self.w_down.ctypes.data,
                self.w_up.ctypes.data,
                self.w_base.ctypes.data,
                self.grid.ctypes.data,
                self.bias.ctypes.data,
                y.ctypes.data,
                B, D_in, self.out_features, self.rank, self.num_grids, self.inv_denominator,
                self.has_base, self.has_bias
            )

        if len(orig_shape) > 2:
            return y.reshape(*orig_shape[:-1], self.out_features)
        return y

    __call__ = forward

    def benchmark(self, x: np.ndarray, warmup: int = 10, iters: int = 50) -> float:
        """Benchmarks forward execution in milliseconds directly on GPU."""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        elif x.dtype != np.float32:
            x = x.astype(np.float32)

        x_flat = x.reshape(-1, self.in_features)
        if not x_flat.flags['C_CONTIGUOUS']:
            x_flat = np.ascontiguousarray(x_flat)

        B, D_in = x_flat.shape
        y = np.empty((B, self.out_features), dtype=np.float32)

        return self._bridge.benchmark_metal_lowrank(
            x_flat.ctypes.data,
            self.w_down.ctypes.data,
            self.w_up.ctypes.data,
            self.w_base.ctypes.data,
            self.grid.ctypes.data,
            self.bias.ctypes.data,
            y.ctypes.data,
            B, D_in, self.out_features, self.rank, self.num_grids, self.inv_denominator,
            self.has_base, self.has_bias,
            warmup, iters
        )
