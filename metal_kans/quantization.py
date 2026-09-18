"""
INT8 and INT4 Quantization Engine for metal-KANs.
Compresses model weights up to 8x with minimal accuracy loss using block-affine scale/offset packing.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Any, Dict
from .utils import get_model_size


class QuantizedWeight:
    """
    Affine block-quantized weight matrix.
    Supports INT8 (1 byte per weight) and INT4 (2 weights packed into 1 uint8 byte).
    """
    def __init__(self, weight: np.ndarray, bits: int = 8, group_size: int = 64):
        self.bits = bits
        self.group_size = group_size
        self.shape = weight.shape
        out_features, in_features = weight.shape

        # Pad in_features to multiple of group_size
        pad_in = (group_size - (in_features % group_size)) % group_size
        if pad_in > 0:
            padded = np.pad(weight, ((0, 0), (0, pad_in)), mode="constant", constant_values=0.0)
        else:
            padded = weight.copy()

        self.padded_in_features = padded.shape[1]
        grouped = padded.reshape(-1, group_size)

        # Affine min-max scaling
        min_vals = grouped.min(axis=1, keepdims=True)
        max_vals = grouped.max(axis=1, keepdims=True)
        range_vals = np.maximum(max_vals - min_vals, 1e-7)

        max_int = (1 << bits) - 1
        scales = range_vals / max_int
        biases = min_vals

        q = np.round((grouped - biases) / scales).astype(np.int32)
        q = np.clip(q, 0, max_int)

        self.scales = scales.astype(np.float32)
        self.biases = biases.astype(np.float32)

        if bits == 8:
            self.qweight = q.astype(np.uint8)
        elif bits == 4:
            # Pack 2 4-bit values per byte
            q_even = q[:, 0::2]
            q_odd = q[:, 1::2]
            self.qweight = (q_even | (q_odd << 4)).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported bits: {bits}. Choose 8 or 4.")

    def dequantize(self) -> np.ndarray:
        """Dequantizes packed representation back into FP32 array."""
        if self.bits == 8:
            q = self.qweight.astype(np.float32)
        elif self.bits == 4:
            q_even = (self.qweight & 0x0F).astype(np.float32)
            q_odd = ((self.qweight >> 4) & 0x0F).astype(np.float32)
            q = np.empty((self.qweight.shape[0], self.group_size), dtype=np.float32)
            q[:, 0::2] = q_even
            q[:, 1::2] = q_odd

        restored_grouped = q * self.scales + self.biases
        restored = restored_grouped.reshape(-1, self.padded_in_features)
        return restored[:self.shape[0], :self.shape[1]].astype(np.float32)


def quantize(model: Any, bits: int = 8, group_size: int = 64) -> Any:
    """
    Quantizes all weight matrices of a metal-KAN layer or network in-place.
    """
    if hasattr(model, "layers"):
        for layer in model.layers:
            quantize(layer, bits=bits, group_size=group_size)
        return model

    weight_attrs = ["w_cheby", "w_rbf", "w_relu", "w_wav", "w_fourier", "w_jacobi",
                    "w_p", "w_q", "w_spline", "w_up", "w_base"]

    for attr in weight_attrs:
        if hasattr(model, attr):
            w = getattr(model, attr)
            if w is not None and isinstance(w, np.ndarray) and w.ndim == 2:
                qw = QuantizedWeight(w, bits=bits, group_size=group_size)
                # Store dequantized array in weight attribute for seamless execution
                setattr(model, f"_{attr}_quantized", qw)
                setattr(model, attr, qw.dequantize())

    if hasattr(model, "sub_layer"):
        quantize(model.sub_layer, bits=bits, group_size=group_size)
    if hasattr(model, "down_layer"):
        quantize(model.down_layer, bits=bits, group_size=group_size)

    return model


def to_int8(model: Any, group_size: int = 64) -> Any:
    """Quantizes all weights to INT8."""
    return quantize(model, bits=8, group_size=group_size)


def to_int4(model: Any, group_size: int = 64) -> Any:
    """Quantizes all weights to INT4."""
    return quantize(model, bits=4, group_size=group_size)
