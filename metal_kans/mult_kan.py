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

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes fused Metal sub-layer and multiplicative channel combination."""
        internal = self.sub_layer(x)

        if self.num_mult == 0:
            return internal

        orig_shape = internal.shape
        flat = internal.reshape(-1, self.internal_out)

        y_add = flat[:, :self.num_add]
        u = flat[:, self.num_add : self.num_add + self.num_mult]
        v = flat[:, self.num_add + self.num_mult :]
        y_mult = u * v

        if self.num_add > 0:
            out = np.concatenate([y_add, y_mult], axis=-1)
        else:
            out = y_mult

        if len(orig_shape) > 2:
            return out.reshape(*orig_shape[:-1], self.out_features)
        return out

    __call__ = forward
