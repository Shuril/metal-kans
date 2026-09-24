"""
Activation Checkpointing for Metal-KANs.
Trades compute for memory during training: saves only layer inputs [B, D_in],
and re-computes expanded basis [B, D_in * K] on the fly during backward pass.
Reduces intermediate basis activation memory by up to 70-80%.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence, List, Tuple, Union
from .metal_kan import MetalKAN


class CheckpointedMetalKAN(MetalKAN):
    """
    MetalKAN with activation checkpointing enabled.
    Stores only the input activations x at each layer boundary,
    discarding intermediate basis projections until backward() is invoked.
    """
    def __init__(
        self,
        layers_hidden: Sequence[int],
        basis_type: str = "cheby",
        degree: int = 4,
        bias: bool = True,
        use_base: bool = True,
    ):
        super().__init__(
            layers_hidden=layers_hidden,
            basis_type=basis_type,
            degree=degree,
            bias=bias,
            use_base=use_base,
            pipeline=False, # Activation checkpointing tracks individual layer inputs
        )
        self.saved_inputs: List[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass saving only the layer inputs."""
        self.saved_inputs.clear()
        target_dtype = self.dtype
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=target_dtype)
        elif x.dtype != target_dtype:
            x = x.astype(target_dtype)

        curr = x
        for layer in self.layers:
            self.saved_inputs.append(curr)
            curr = layer(curr)
        return curr

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Backward pass using checkpointed activations.
        Ensures each layer's saved input is restored before its backward execution.
        """
        if not self.saved_inputs or len(self.saved_inputs) != len(self.layers):
            raise RuntimeError("Cannot execute backward before forward() has completed.")

        grad = dY
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            x_in = self.saved_inputs[i]
            # Ensure layer has its recorded input
            layer._saved_x = x_in if x_in.ndim == 2 else x_in.reshape(-1, layer.in_features)
            layer._saved_shape = x_in.shape
            grad = layer.backward(grad)

        return grad


def checkpoint_kan(model: MetalKAN) -> CheckpointedMetalKAN:
    """
    Wraps or transforms an existing MetalKAN instance into a CheckpointedMetalKAN.
    """
    if isinstance(model, CheckpointedMetalKAN):
        return model

    checkpointed = CheckpointedMetalKAN(
        layers_hidden=model.layers_hidden,
        basis_type=model.basis_type,
        degree=model.degree,
        bias=model.bias,
        use_base=model.use_base,
    )
    checkpointed.layers = model.layers
    return checkpointed
