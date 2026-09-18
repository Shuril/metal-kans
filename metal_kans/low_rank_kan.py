"""
Direct Metal Fused LowRankKAN Layer (LoRA / Bottleneck Factorization).
Reduces parameter complexity from O(d_out * d_in * K) to O((d_out + d_in * K) * rank).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Sequence
from .fast_kan import FastKAN


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
        self.has_bias = bias

        # Down-projection fused kernel to rank r
        self.down_layer = FastKAN(
            in_features=in_features,
            out_features=self.rank,
            num_centers=num_grids,
            bias=False,
            use_base=False,
        )

        # Up-projection matrix (rank -> out_features)
        bound = 1.0 / math.sqrt(self.rank)
        self.w_up = np.random.uniform(-bound, bound, (out_features, self.rank)).astype(np.float32)

        # Base connection (SiLU) and bias
        self.use_base = use_base
        bound_base = 1.0 / math.sqrt(in_features)
        self.w_base = np.random.uniform(-bound_base, bound_base, (out_features, in_features)).astype(np.float32) if use_base else None
        self.bias = np.zeros((out_features,), dtype=np.float32) if bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes down-projection on Metal GPU followed by up-projection."""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        elif x.dtype != np.float32:
            x = x.astype(np.float32)

        orig_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, self.in_features)

        # 1. Down-projection to bottleneck rank on GPU
        z = self.down_layer(x)  # [B, rank]

        # 2. Up-projection
        y = z @ self.w_up.T  # [B, out_features]

        # 3. Base residual
        if self.use_base and self.w_base is not None:
            silu_x = x / (1.0 + np.exp(-x))
            y = y + (silu_x @ self.w_base.T)

        if self.bias is not None:
            y = y + self.bias

        if len(orig_shape) > 2:
            return y.reshape(*orig_shape[:-1], self.out_features)
        return y

    __call__ = forward
