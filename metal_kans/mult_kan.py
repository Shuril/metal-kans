"""
Direct Metal Fused MultKAN Layer (KAN 2.0 with Multiplication Nodes).
Based on Ziming Liu et al. (MIT / Caltech 2024).
Introduces explicit multiplicative nodes u * v alongside standard additive nodes.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional
from .fast_kan import FastKAN


class MultKAN:
    """
    Direct Metal Fused MultKAN Layer.

    Produces `num_add` additive channels and `num_mult` multiplicative channels (u * v),
    enabling exact representations of physical conservation laws and polynomial product terms.

    Parameters:
        in_features: Number of input features.
        out_features: Total output dimension (num_add + num_mult).
        num_mult: Number of multiplicative channels (default: max(1, out_features // 2)).
        num_grids: Number of basis grid points (default: 8).
        bias: Whether to add trainable additive bias (default: True).
        use_base: Whether to include residual SiLU base connection (default: True).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_mult: Optional[int] = None,
        num_grids: int = 8,
        bias: bool = True,
        use_base: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features

        if num_mult is None:
            self.num_mult = max(1, out_features // 2) if out_features > 1 else 0
        else:
            self.num_mult = min(num_mult, out_features)

        self.num_add = out_features - self.num_mult
        self.internal_out = self.num_add + 2 * self.num_mult

        # Fused Metal sub-layer
        self.sub_layer = FastKAN(
            in_features=in_features,
            out_features=self.internal_out,
            num_centers=num_grids,
            bias=bias,
            use_base=use_base,
        )
        self.num_grids = num_grids
        self.w_rbf = self.sub_layer.w_rbf
        self.w_base = self.sub_layer.w_base
        self.grid = self.sub_layer.grid
        self.bias = self.sub_layer.bias
        self.has_base = self.sub_layer.has_base
        self.has_bias = self.sub_layer.has_bias
        self.inv_denominator = self.sub_layer.inv_denominator
        self._bridge = self.sub_layer._bridge

    @property
    def dtype(self) -> np.dtype:
        return self.sub_layer.dtype

    def half(self) -> MultKAN:
        """Converts layer parameters to FP16 half precision."""
        self.sub_layer.half()
        self.w_rbf = self.sub_layer.w_rbf
        self.w_base = self.sub_layer.w_base
        self.grid = self.sub_layer.grid
        self.bias = self.sub_layer.bias
        return self

    def float(self) -> MultKAN:
        """Converts layer parameters to FP32 single precision."""
        self.sub_layer.float()
        self.w_rbf = self.sub_layer.w_rbf
        self.w_base = self.sub_layer.w_base
        self.grid = self.sub_layer.grid
        self.bias = self.sub_layer.bias
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes fused Metal sub-layer and multiplicative channel combination in a single GPU command buffer."""
        target_dtype = self.sub_layer.dtype
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

        out = np.empty((B, self.out_features), dtype=target_dtype)

        if target_dtype == np.float16:
            self._bridge.metal_kan_mult_forward_fp16(
                x.ctypes.data,
                self.w_rbf.ctypes.data,
                self.w_base.ctypes.data,
                self.grid.ctypes.data,
                self.bias.ctypes.data,
                out.ctypes.data,
                B, D_in, self.out_features, self.num_add, self.num_mult, self.num_grids, self.inv_denominator,
                self.has_base, self.has_bias
            )
        else:
            self._bridge.metal_kan_mult_forward(
                x.ctypes.data,
                self.w_rbf.ctypes.data,
                self.w_base.ctypes.data,
                self.grid.ctypes.data,
                self.bias.ctypes.data,
                out.ctypes.data,
                B, D_in, self.out_features, self.num_add, self.num_mult, self.num_grids, self.inv_denominator,
                self.has_base, self.has_bias
            )

        if len(orig_shape) > 2:
            return out.reshape(*orig_shape[:-1], self.out_features)
        return out

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
        out = np.empty((B, self.out_features), dtype=np.float32)

        return self._bridge.benchmark_metal_mult(
            x_flat.ctypes.data,
            self.w_rbf.ctypes.data,
            self.w_base.ctypes.data,
            self.grid.ctypes.data,
            self.bias.ctypes.data,
            out.ctypes.data,
            B, D_in, self.out_features, self.num_add, self.num_mult, self.num_grids, self.inv_denominator,
            self.has_base, self.has_bias,
            warmup, iters
        )

