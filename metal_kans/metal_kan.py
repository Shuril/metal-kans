"""
Sequential Multi-Layer Pure Metal KAN Network.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence, List, Union
from .cheby_kan import ChebyKAN
from .fast_kan import FastKAN
from .relu_kan import ReLUKAN


class MetalKAN:
    """
    Multi-Layer Pure Metal KAN Network.
    Executes sequentially on Apple Silicon GPU without requiring MLX or PyTorch.

    Parameters:
        layers_hidden: List or tuple of layer widths, e.g. [4, 16, 8, 2].
        basis_type: Type of basis ('cheby', 'fastkan', 'relu'). Default: 'cheby'.
        degree: Basis order or grid points (default 4).
        bias: Whether to include bias term (default True).
        use_base: Whether to include residual base connections (default True).
    """
    def __init__(
        self,
        layers_hidden: Sequence[int],
        basis_type: str = "cheby",
        degree: int = 4,
        bias: bool = True,
        use_base: bool = True,
    ):
        self.layers_hidden = list(layers_hidden)
        self.basis_type = basis_type.lower()
        self.layers: List[Union[ChebyKAN, FastKAN, ReLUKAN]] = []

        for i in range(len(layers_hidden) - 1):
            in_f = layers_hidden[i]
            out_f = layers_hidden[i + 1]
            if self.basis_type in ("cheby", "chebyshev"):
                layer = ChebyKAN(in_f, out_f, degree=degree, bias=bias, use_base=use_base)
            elif self.basis_type in ("fastkan", "rbf"):
                layer = FastKAN(in_f, out_f, num_centers=degree, bias=bias, use_base=use_base)
            elif self.basis_type in ("relukan", "relu", "tent"):
                layer = ReLUKAN(in_f, out_f, num_grids=degree, bias=bias, use_base=use_base)
            else:
                raise ValueError(f"Unknown basis_type: {basis_type}. Expected 'cheby', 'fastkan', or 'relu'.")
            self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes forward pass sequentially through all Metal layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    __call__ = forward

    def __repr__(self) -> str:
        layers_str = " -> ".join(map(str, self.layers_hidden))
        return f"MetalKAN(layers=[{layers_str}], basis={self.basis_type!r})"
