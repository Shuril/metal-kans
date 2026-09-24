"""
GatedKAN: Drop-in Replacement for SwiGLU / MLP Transformer Blocks.
Designed for System Level Cache (SLC) residency on Apple Silicon.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Union, Tuple, Sequence

from .cheby_kan import ChebyKAN
from .quantization import to_int8, to_int4, to_ternary, to_int2


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _d_silu(x: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
    return sig * (1.0 + x * (1.0 - sig))


class GatedKAN:
    """
    Gated Kolmogorov-Arnold Network (GatedKAN) block.
    Computes:
        gate = Gate_KAN(x)
        up = Up_KAN(x)
        hidden = SiLU(gate) * up
        out = Down_KAN(hidden)

    Matches and exceeds SwiGLU expressiveness with 2-4x smaller hidden dimension (d_ffn),
    maximizing Apple Silicon System Level Cache (SLC) residency.
    """
    def __init__(
        self,
        d_model: int,
        d_ffn: Optional[int] = None,
        degree: int = 4,
        has_base: bool = True,
        has_bias: bool = False,
        dtype: np.dtype = np.float32,
    ):
        self.d_model = int(d_model)
        # Default d_ffn is 2 * d_model (compared to 4 * d_model in standard Transformers)
        self.d_ffn = int(d_ffn if d_ffn is not None else 2 * d_model)
        self.degree = int(degree)
        self.has_base = bool(has_base)
        self.has_bias = bool(has_bias)
        self.dtype = np.dtype(dtype)

        # Three KAN components
        self.gate_kan = ChebyKAN(
            self.d_model,
            self.d_ffn,
            degree=self.degree,
            bias=self.has_bias,
            use_base=self.has_base,
        )
        self.up_kan = ChebyKAN(
            self.d_model,
            self.d_ffn,
            degree=self.degree,
            bias=self.has_bias,
            use_base=self.has_base,
        )
        self.down_kan = ChebyKAN(
            self.d_ffn,
            self.d_model,
            degree=self.degree,
            bias=self.has_bias,
            use_base=self.has_base,
        )
        if self.dtype == np.float16:
            self.gate_kan.half()
            self.up_kan.half()
            self.down_kan.half()

        # Caching for backward pass
        self._saved_x: Optional[np.ndarray] = None
        self._saved_gate: Optional[np.ndarray] = None
        self._saved_up: Optional[np.ndarray] = None
        self._saved_hidden: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of GatedKAN block.
        Accepts x of shape (B, d_model) or (B, Seq, d_model).
        """
        orig_shape = x.shape
        if len(orig_shape) > 2:
            x_2d = x.reshape(-1, self.d_model)
        else:
            x_2d = x

        self._saved_x = x_2d

        gate = self.gate_kan(x_2d)
        up = self.up_kan(x_2d)

        self._saved_gate = gate
        self._saved_up = up

        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        if bridge is not None and self.dtype == np.float32 and gate.dtype == np.float32 and up.dtype == np.float32:
            gate_c = np.ascontiguousarray(gate, dtype=np.float32)
            up_c = np.ascontiguousarray(up, dtype=np.float32)
            hidden = np.empty_like(gate_c)
            code = bridge.metal_kan_swiglu_forward(
                gate_c.ctypes.data,
                up_c.ctypes.data,
                hidden.ctypes.data,
                gate_c.size
            )
            if code == 0:
                self._saved_hidden = hidden
                out = self.down_kan(hidden)
                if len(orig_shape) > 2:
                    return out.reshape(*orig_shape[:-1], self.d_model)
                return out

        # Fallback: SiLU(gate) * up
        silu_gate = _silu(gate)
        hidden = silu_gate * up
        self._saved_hidden = hidden

        out = self.down_kan(hidden)

        if len(orig_shape) > 2:
            return out.reshape(*orig_shape[:-1], self.d_model)
        return out

    __call__ = forward

    def backward(self, dY: np.ndarray, fused: bool = True) -> np.ndarray:
        """
        Full backward pass through Down_KAN -> Gating -> Gate_KAN & Up_KAN.
        Returns dX of same shape as input x.
        """
        if self._saved_x is None:
            raise RuntimeError("Cannot run backward before forward() has been called.")

        orig_shape = dY.shape
        if len(orig_shape) > 2:
            dY_2d = dY.reshape(-1, self.d_model)
        else:
            dY_2d = dY

        # 1. Backprop through Down_KAN
        d_hidden = self.down_kan.backward(dY_2d, fused=fused)

        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        if (
            bridge is not None
            and self.dtype == np.float32
            and d_hidden.dtype == np.float32
            and self._saved_gate.dtype == np.float32
            and self._saved_up.dtype == np.float32
        ):
            gate_c = np.ascontiguousarray(self._saved_gate, dtype=np.float32)
            up_c = np.ascontiguousarray(self._saved_up, dtype=np.float32)
            d_hidden_c = np.ascontiguousarray(d_hidden, dtype=np.float32)
            d_gate = np.empty_like(gate_c)
            d_up = np.empty_like(up_c)

            code = bridge.metal_kan_swiglu_backward(
                d_hidden_c.ctypes.data,
                gate_c.ctypes.data,
                up_c.ctypes.data,
                d_gate.ctypes.data,
                d_up.ctypes.data,
                gate_c.size
            )
            if code == 0:
                dX_gate = self.gate_kan.backward(d_gate, fused=fused)
                dX_up = self.up_kan.backward(d_up, fused=fused)
                dX_gate_c = np.ascontiguousarray(dX_gate, dtype=np.float32)
                dX_up_c = np.ascontiguousarray(dX_up, dtype=np.float32)
                dX = np.empty_like(dX_gate_c)
                bridge.metal_kan_elementwise_add(
                    dX_gate_c.ctypes.data,
                    dX_up_c.ctypes.data,
                    dX.ctypes.data,
                    dX.size
                )
                if len(orig_shape) > 2:
                    return dX.reshape(orig_shape)
                return dX

        # Fallback: Backprop through SwiGLU gating: hidden = silu(gate) * up
        gate = self._saved_gate
        up = self._saved_up
        silu_gate = _silu(gate)
        d_silu_gate = _d_silu(gate)

        d_up = d_hidden * silu_gate
        d_gate = d_hidden * up * d_silu_gate

        # 3. Backprop through Gate_KAN and Up_KAN
        dX_gate = self.gate_kan.backward(d_gate, fused=fused)
        dX_up = self.up_kan.backward(d_up, fused=fused)

        dX = dX_gate + dX_up

        if len(orig_shape) > 2:
            return dX.reshape(orig_shape)
        return dX

    def zero_grad(self) -> None:
        """Zeros stored parameter gradients across all KAN layers."""
        self.gate_kan.zero_grad()
        self.up_kan.zero_grad()
        self.down_kan.zero_grad()

    def get_trainable_params(self) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        """Returns sequence of (param, grad) pairs for optimizers."""
        params = []
        for kan in (self.gate_kan, self.up_kan, self.down_kan):
            params.append((kan.w_cheby, kan.grad_w_cheby))
            if kan.has_base and kan.w_base is not None:
                params.append((kan.w_base, kan.grad_w_base))
            if kan.has_bias and kan.bias is not None:
                params.append((kan.bias, kan.grad_bias))
        return params

    def to_int8(self, group_size: int = 64) -> GatedKAN:
        """Quantizes all sub-layers to INT8 in-place."""
        to_int8(self.gate_kan, group_size=group_size)
        to_int8(self.up_kan, group_size=group_size)
        to_int8(self.down_kan, group_size=group_size)
        return self

    def to_int4(self, group_size: int = 64) -> GatedKAN:
        """Quantizes all sub-layers to INT4 in-place."""
        to_int4(self.gate_kan, group_size=group_size)
        to_int4(self.up_kan, group_size=group_size)
        to_int4(self.down_kan, group_size=group_size)
        return self

    def to_ternary(self, group_size: int = 32) -> GatedKAN:
        """Quantizes all sub-layers to 1.58-bit Ternary in-place."""
        to_ternary(self.gate_kan, group_size=group_size)
        to_ternary(self.up_kan, group_size=group_size)
        to_ternary(self.down_kan, group_size=group_size)
        return self

    def to_int2(self, group_size: int = 32) -> GatedKAN:
        """Quantizes all sub-layers to 2-bit codebook quantized in-place."""
        to_int2(self.gate_kan, group_size=group_size)
        to_int2(self.up_kan, group_size=group_size)
        to_int2(self.down_kan, group_size=group_size)
        return self

    def parameter_count(self) -> int:
        """Returns total number of scalar parameters in the GatedKAN block."""
        count = 0
        for kan in (self.gate_kan, self.up_kan, self.down_kan):
            count += kan.w_cheby.size
            if kan.has_base and kan.w_base is not None:
                count += kan.w_base.size
            if kan.has_bias and kan.bias is not None:
                count += kan.bias.size
        return count

    def __repr__(self) -> str:
        return (
            f"GatedKAN(d_model={self.d_model}, d_ffn={self.d_ffn}, degree={self.degree}, "
            f"has_base={self.has_base}, params={self.parameter_count():,})"
        )
