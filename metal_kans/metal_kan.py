"""
Sequential Multi-Layer Pure Metal KAN Network.
Supports all 10 KAN architectures with GPU-pipelined execution.
"""

from __future__ import annotations
import numpy as np
import ctypes
from typing import Sequence, List, Union, Optional
from .cheby_kan import ChebyKAN
from .fast_kan import FastKAN
from .relu_kan import ReLUKAN
from .wav_kan import WavKAN
from .fourier_kan import FourierKAN
from .jacobi_kan import JacobiKAN
from .rational_kan import RationalKAN
from .bspline_kan import BSplineKAN, KAN
from .mult_kan import MultKAN
from .low_rank_kan import LowRankKAN
from .device import get_metal_bridge

# Backward compatibility aliases
MetalChebyKAN = ChebyKAN
MetalFastKAN = FastKAN
MetalReLUKAN = ReLUKAN


class MetalKAN:
    """
    Multi-Layer Pure Metal KAN Network.
    Executes sequentially on Apple Silicon GPU without requiring PyTorch or MLX.

    Parameters:
        layers_hidden: List or tuple of layer widths, e.g. [4, 16, 8, 2].
        basis_type: Type of basis ('cheby', 'fastkan', 'relu', 'wav', 'fourier', 'jacobi', 'rational', 'bspline', 'mult', 'lowrank'). Default: 'cheby'.
        degree: Basis order or grid points (default 4).
        bias: Whether to include bias term (default True).
        use_base: Whether to include residual base connections (default True).
        pipeline: Whether to use single-dispatch chained GPU pipeline (for ChebyKAN).
    """
    def __init__(
        self,
        layers_hidden: Sequence[int],
        basis_type: str = "cheby",
        degree: int = 4,
        bias: bool = True,
        use_base: bool = True,
        pipeline: bool = False,
    ):
        self.layers_hidden = list(layers_hidden)
        self.basis_type = basis_type.lower()
        self.degree = degree
        self.bias = bias
        self.use_base = use_base
        self.pipeline = pipeline and (self.basis_type in ("cheby", "chebyshev"))
        self.layers = []

        for i in range(len(layers_hidden) - 1):
            in_f = layers_hidden[i]
            out_f = layers_hidden[i + 1]

            if self.basis_type in ("cheby", "chebyshev"):
                layer = ChebyKAN(in_f, out_f, degree=degree, bias=bias, use_base=use_base)
            elif self.basis_type in ("fastkan", "rbf"):
                layer = FastKAN(in_f, out_f, num_centers=degree, bias=bias, use_base=use_base)
            elif self.basis_type in ("relukan", "relu", "tent"):
                layer = ReLUKAN(in_f, out_f, num_grids=degree, bias=bias, use_base=use_base)
            elif self.basis_type in ("wav", "wavkan", "wavelet"):
                layer = WavKAN(in_f, out_f, num_wavelets=degree, bias=bias, use_base=use_base)
            elif self.basis_type in ("fourier", "fourierkan"):
                layer = FourierKAN(in_f, out_f, num_frequencies=degree, bias=bias, use_base=use_base)
            elif self.basis_type in ("jacobi", "jacobikan"):
                layer = JacobiKAN(in_f, out_f, degree=degree, bias=bias, use_base=use_base)
            elif self.basis_type in ("rational", "rationalkan"):
                layer = RationalKAN(in_f, out_f, p_degree=degree, q_degree=max(2, degree // 2), bias=bias, use_base=use_base)
            elif self.basis_type in ("bspline", "spline", "kan"):
                layer = BSplineKAN(in_f, out_f, grid_size=degree, bias=bias, use_base=use_base)
            elif self.basis_type in ("mult", "multkan"):
                layer = MultKAN(in_f, out_f, num_grids=degree, bias=bias, use_base=use_base)
            elif self.basis_type in ("lowrank", "lowrankkan"):
                layer = LowRankKAN(in_f, out_f, rank=degree, num_grids=degree, bias=bias, use_base=use_base)
            else:
                raise ValueError(f"Unknown basis_type: {basis_type}")

            self.layers.append(layer)

    @property
    def dtype(self) -> np.dtype:
        return getattr(self.layers[0], "dtype", np.dtype(np.float32)) if self.layers else np.dtype(np.float32)

    def half(self) -> MetalKAN:
        """Converts all network layers to FP16 half precision."""
        for layer in self.layers:
            if hasattr(layer, "half"):
                layer.half()
        return self

    def float(self) -> MetalKAN:
        """Converts all network layers to FP32 single precision."""
        for layer in self.layers:
            if hasattr(layer, "float"):
                layer.float()
        return self

    def async_stream(self, batches: Sequence[np.ndarray]) -> List[np.ndarray]:
        """
        Executes a sequence of batches asynchronously using ring-buffered GPU pipelining.
        Synchronizes once at the end of the batch sequence, eliminating per-iteration CPU idle latency.
        """
        from .device import set_async, sync
        results = []
        set_async(True)
        try:
            for b in batches:
                results.append(self.forward(b))
        finally:
            sync()
            set_async(False)
        return results

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes forward pass sequentially through all Metal layers."""
        target_dtype = self.dtype
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=target_dtype)
        elif x.dtype != target_dtype:
            x = x.astype(target_dtype)

        # Chained single-dispatch pipeline optimization for ChebyKAN
        if self.pipeline and len(self.layers) > 1 and target_dtype == np.float32:
            return self._forward_pipelined_cheby(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def _forward_pipelined_cheby(self, x: np.ndarray) -> np.ndarray:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.layers_hidden[0])
        if not x_flat.flags['C_CONTIGUOUS']:
            x_flat = np.ascontiguousarray(x_flat)

        B = x_flat.shape[0]
        out_dim = self.layers_hidden[-1]
        y = np.empty((B, out_dim), dtype=np.float32)

        num_layers = len(self.layers)
        c_layer_dims = (ctypes.c_int * (num_layers + 1))(*self.layers_hidden)
        c_degrees = (ctypes.c_int * num_layers)(*[l.degree for l in self.layers])

        c_w_chebys = (ctypes.c_void_p * num_layers)(*[ctypes.c_void_p(l.w_cheby.ctypes.data) for l in self.layers])
        c_w_bases = (ctypes.c_void_p * num_layers)(*[ctypes.c_void_p(l.w_base.ctypes.data) if l.has_base else None for l in self.layers])
        c_biases = (ctypes.c_void_p * num_layers)(*[ctypes.c_void_p(l.bias.ctypes.data) if l.has_bias else None for l in self.layers])

        bridge = get_metal_bridge()
        bridge.metal_kan_chain_pipeline_cheby(
            x_flat.ctypes.data,
            y.ctypes.data,
            B,
            num_layers,
            c_layer_dims,
            c_degrees,
            c_w_chebys,
            c_w_bases,
            c_biases
        )

        if len(orig_shape) > 2:
            return y.reshape(*orig_shape[:-1], out_dim)
        return y

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Executes backward pass in reverse layer order.
        Returns gradient with respect to network input dX.
        """
        grad = dY
        for layer in reversed(self.layers):
            if hasattr(layer, "backward"):
                grad = layer.backward(grad)
            else:
                raise NotImplementedError(f"Layer {type(layer).__name__} does not implement backward().")
        return grad

    def zero_grad(self) -> None:
        """Zeros gradients across all layers."""
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()

    def parameters(self) -> List[tuple[np.ndarray, np.ndarray]]:
        """
        Returns list of (param_array, grad_array) tuples for all trainable parameters.
        Only includes parameters that have active gradients.
        """
        params = []
        for layer in self.layers:
            # Main basis weights
            for attr in ("w_cheby", "w_rbf", "w_relu", "w_spline", "w_wav", "w_fourier", "w_jacobi"):
                if hasattr(layer, attr):
                    w = getattr(layer, attr)
                    grad_attr = f"grad_{attr}"
                    grad = getattr(layer, grad_attr, None)
                    if grad is not None:
                        params.append((w, grad))
                    break

            # Residual base weights
            if getattr(layer, "has_base", 0):
                w_b = getattr(layer, "w_base", None)
                grad_b = getattr(layer, "grad_w_base", None)
                if w_b is not None and grad_b is not None:
                    params.append((w_b, grad_b))

            # Bias
            if getattr(layer, "has_bias", 0):
                bias = getattr(layer, "bias", None)
                grad_bias = getattr(layer, "grad_bias", None)
                if bias is not None and grad_bias is not None:
                    params.append((bias, grad_bias))

        return params

    def train_step(
        self,
        x: np.ndarray,
        target: np.ndarray,
        optimizer: Optional[Any] = None,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        opt_state: Optional[dict] = None
    ) -> float:
        """
        Executes fused Metal GPU training step.
        Dispatches directly to GPU monolithic kernel when applicable.
        """
        if len(self.layers) == 1 and hasattr(self.layers[0], "train_step"):
            return self.layers[0].train_step(
                x, target,
                lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay,
                opt_state=opt_state
            )

        pred = self.forward(x)
        B = x.shape[0]

        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        if (
            bridge is not None
            and pred.dtype == np.float32
            and pred.flags['C_CONTIGUOUS']
            and target.flags['C_CONTIGUOUS']
            and hasattr(bridge, "metal_kan_calc_mse_loss_backward")
        ):
            diff = np.empty_like(pred)
            code = bridge.metal_kan_calc_mse_loss_backward(
                pred.ctypes.data,
                target.ctypes.data,
                diff.ctypes.data,
                pred.size,
                2.0 / float(B)
            )
            if code != 0:
                diff = 2.0 * (pred - target) / float(B)
        else:
            diff = 2.0 * (pred - target) / float(B)

        self.backward(diff)
        if optimizer is not None:
            optimizer.step()
            self.zero_grad()
        return 0.0

    def __repr__(self) -> str:
        layers_str = " -> ".join(map(str, self.layers_hidden))
        return f"MetalKAN(layers=[{layers_str}], basis={self.basis_type!r})"
