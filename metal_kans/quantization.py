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
        if not padded.flags['C_CONTIGUOUS']:
            padded = np.ascontiguousarray(padded, dtype=np.float32)
        elif padded.dtype != np.float32:
            padded = padded.astype(np.float32)

        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        num_groups = padded.size // group_size
        if bridge is not None:
            self.scales = np.empty((num_groups, 1), dtype=np.float32)
            self.biases = np.empty((num_groups, 1), dtype=np.float32)
            if bits == 8:
                self.qweight = np.empty((num_groups, group_size), dtype=np.uint8)
                code = bridge.metal_kan_quantize_int8(
                    padded.ctypes.data,
                    self.qweight.ctypes.data,
                    self.scales.ctypes.data,
                    self.biases.ctypes.data,
                    num_groups,
                    group_size
                )
                if code == 0:
                    return
            elif bits == 4:
                self.qweight = np.empty((num_groups, group_size // 2), dtype=np.uint8)
                code = bridge.metal_kan_quantize_int4(
                    padded.ctypes.data,
                    self.qweight.ctypes.data,
                    self.scales.ctypes.data,
                    self.biases.ctypes.data,
                    num_groups,
                    group_size
                )
                if code == 0:
                    return

        # Fallback NumPy implementation
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
        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        if bridge is not None and self.qweight.flags['C_CONTIGUOUS'] and self.scales.flags['C_CONTIGUOUS'] and self.biases.flags['C_CONTIGUOUS']:
            num_groups = self.scales.shape[0]
            padded_out = np.empty(num_groups * self.group_size, dtype=np.float32)
            if self.bits == 8:
                code = bridge.metal_kan_dequantize_int8(
                    self.qweight.ctypes.data,
                    self.scales.ctypes.data,
                    self.biases.ctypes.data,
                    padded_out.ctypes.data,
                    num_groups,
                    self.group_size
                )
                if code == 0:
                    return padded_out.reshape(-1, self.padded_in_features)[:self.shape[0], :self.shape[1]]
            elif self.bits == 4:
                code = bridge.metal_kan_dequantize_int4(
                    self.qweight.ctypes.data,
                    self.scales.ctypes.data,
                    self.biases.ctypes.data,
                    padded_out.ctypes.data,
                    num_groups,
                    self.group_size
                )
                if code == 0:
                    return padded_out.reshape(-1, self.padded_in_features)[:self.shape[0], :self.shape[1]]

        # Fallback NumPy implementation
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


class TernaryQuantizedWeight:
    """
    1.58-bit Ternary Quantized weight matrix (weights in {-1, 0, +1} * per-group scale).
    Packs 16 weights into a single 32-bit uint32 (2 bits per weight).
    """
    def __init__(self, weight: np.ndarray, group_size: int = 32):
        self.group_size = group_size
        self.shape = weight.shape
        D_out, K_dim = weight.shape

        if not weight.flags['C_CONTIGUOUS']:
            w = np.ascontiguousarray(weight, dtype=np.float32)
        else:
            w = weight.astype(np.float32)

        words_per_row = (K_dim + 15) // 16
        groups_per_row = (K_dim + group_size - 1) // group_size

        self.w_packed = np.zeros((D_out, words_per_row), dtype=np.uint32)
        self.scales = np.zeros((D_out, groups_per_row), dtype=np.float32)

        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        if bridge is not None and hasattr(bridge, "metal_kan_quantize_ternary"):
            code = bridge.metal_kan_quantize_ternary(
                w.ctypes.data,
                self.w_packed.ctypes.data,
                self.scales.ctypes.data,
                D_out,
                K_dim,
                group_size
            )
            if code == 0:
                return

        # Fallback NumPy
        for j in range(D_out):
            w_row = w[j]
            for g in range(groups_per_row):
                start = g * group_size
                end = min(start + group_size, K_dim)
                slice_w = w_row[start:end]
                sc = float(np.mean(np.abs(slice_w))) if len(slice_w) > 0 and np.mean(np.abs(slice_w)) > 0 else 1e-4
                self.scales[j, g] = sc
                norm_w = slice_w / sc
                ternary = np.where(norm_w < -0.5, 0, np.where(norm_w > 0.5, 2, 1)).astype(np.uint32)
                for idx, code_val in enumerate(ternary):
                    k = start + idx
                    word_idx = k // 16
                    shift = (k % 16) * 2
                    self.w_packed[j, word_idx] |= (code_val << shift)

    def dequantize(self) -> np.ndarray:
        D_out, K_dim = self.shape
        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        if (
            bridge is not None
            and self.w_packed.flags['C_CONTIGUOUS']
            and self.scales.flags['C_CONTIGUOUS']
            and hasattr(bridge, "metal_kan_dequantize_ternary")
        ):
            out = np.empty((D_out, K_dim), dtype=np.float32)
            code = bridge.metal_kan_dequantize_ternary(
                self.w_packed.ctypes.data,
                self.scales.ctypes.data,
                out.ctypes.data,
                D_out,
                K_dim,
                self.group_size
            )
            if code == 0:
                return out

        out = np.zeros((D_out, K_dim), dtype=np.float32)
        for j in range(D_out):
            w_row = self.w_packed[j]
            for k in range(K_dim):
                word_idx = k // 16
                shift = (k % 16) * 2
                code = (w_row[word_idx] >> shift) & 0x3
                w_val = float(code) - 1.0 if code < 3 else 0.0
                g = k // self.group_size
                sc = self.scales[j, g]
                out[j, k] = w_val * sc
        return out


class Int2QuantizedWeight:
    """
    2-bit Lloyd-Max / uniform codebook quantized weight matrix.
    Packs 16 2-bit weights into a single uint32.
    """
    def __init__(self, weight: np.ndarray, group_size: int = 32, codebook: Any = None):
        self.group_size = group_size
        self.shape = weight.shape
        D_out, K_dim = weight.shape

        if codebook is None:
            codebook = np.array([-1.0, -0.33333, 0.33333, 1.0], dtype=np.float32)
        else:
            codebook = np.ascontiguousarray(codebook, dtype=np.float32)
        self.codebook = codebook

        if not weight.flags['C_CONTIGUOUS']:
            w = np.ascontiguousarray(weight, dtype=np.float32)
        else:
            w = weight.astype(np.float32)

        words_per_row = (K_dim + 15) // 16
        groups_per_row = (K_dim + group_size - 1) // group_size

        self.w_packed = np.zeros((D_out, words_per_row), dtype=np.uint32)
        self.scales = np.zeros((D_out, groups_per_row), dtype=np.float32)

        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        if bridge is not None and hasattr(bridge, "metal_kan_quantize_int2"):
            code = bridge.metal_kan_quantize_int2(
                w.ctypes.data,
                self.w_packed.ctypes.data,
                self.scales.ctypes.data,
                codebook.ctypes.data,
                D_out,
                K_dim,
                group_size
            )
            if code == 0:
                return

        # Fallback NumPy
        for j in range(D_out):
            w_row = w[j]
            for g in range(groups_per_row):
                start = g * group_size
                end = min(start + group_size, K_dim)
                slice_w = w_row[start:end]
                sc = float(np.max(np.abs(slice_w))) if len(slice_w) > 0 and np.max(np.abs(slice_w)) > 0 else 1e-4
                self.scales[j, g] = sc
                norm_w = slice_w / sc
                dists = np.abs(norm_w[:, None] - codebook[None, :])
                best_c = np.argmin(dists, axis=1).astype(np.uint32)
                for idx, c_idx in enumerate(best_c):
                    k = start + idx
                    word_idx = k // 16
                    shift = (k % 16) * 2
                    self.w_packed[j, word_idx] |= (c_idx << shift)

    def dequantize(self) -> np.ndarray:
        D_out, K_dim = self.shape
        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        if (
            bridge is not None
            and self.w_packed.flags['C_CONTIGUOUS']
            and self.scales.flags['C_CONTIGUOUS']
            and self.codebook.flags['C_CONTIGUOUS']
            and hasattr(bridge, "metal_kan_dequantize_int2")
        ):
            out = np.empty((D_out, K_dim), dtype=np.float32)
            code = bridge.metal_kan_dequantize_int2(
                self.w_packed.ctypes.data,
                self.scales.ctypes.data,
                self.codebook.ctypes.data,
                out.ctypes.data,
                D_out,
                K_dim,
                self.group_size
            )
            if code == 0:
                return out

        out = np.zeros((D_out, K_dim), dtype=np.float32)
        for j in range(D_out):
            w_row = self.w_packed[j]
            for k in range(K_dim):
                word_idx = k // 16
                shift = (k % 16) * 2
                c_idx = (w_row[word_idx] >> shift) & 0x3
                g = k // self.group_size
                sc = self.scales[j, g]
                out[j, k] = self.codebook[c_idx] * sc
        return out


def to_ternary(model: Any, group_size: int = 32) -> Any:
    """
    Quantizes all weight matrices of a metal-KAN layer or network to 1.58-bit Ternary in-place.
    """
    if hasattr(model, "layers"):
        for layer in model.layers:
            to_ternary(layer, group_size=group_size)
        return model

    weight_attrs = ["w_cheby", "w_rbf", "w_relu", "w_wav", "w_fourier", "w_jacobi",
                    "w_p", "w_q", "w_spline", "w_up", "w_base"]

    for attr in weight_attrs:
        if hasattr(model, attr):
            w = getattr(model, attr)
            if w is not None and isinstance(w, np.ndarray) and w.ndim == 2:
                qw = TernaryQuantizedWeight(w, group_size=group_size)
                setattr(model, f"_{attr}_ternary", qw)
                setattr(model, attr, qw.dequantize())

    if hasattr(model, "sub_layer"):
        to_ternary(model.sub_layer, group_size=group_size)
    if hasattr(model, "down_layer"):
        to_ternary(model.down_layer, group_size=group_size)

    return model


def to_int2(model: Any, group_size: int = 32, codebook: Any = None) -> Any:
    """
    Quantizes all weight matrices of a metal-KAN layer or network to 2-bit codebook quantized in-place.
    """
    if hasattr(model, "layers"):
        for layer in model.layers:
            to_int2(layer, group_size=group_size, codebook=codebook)
        return model

    weight_attrs = ["w_cheby", "w_rbf", "w_relu", "w_wav", "w_fourier", "w_jacobi",
                    "w_p", "w_q", "w_spline", "w_up", "w_base"]

    for attr in weight_attrs:
        if hasattr(model, attr):
            w = getattr(model, attr)
            if w is not None and isinstance(w, np.ndarray) and w.ndim == 2:
                qw = Int2QuantizedWeight(w, group_size=group_size, codebook=codebook)
                setattr(model, f"_{attr}_int2", qw)
                setattr(model, attr, qw.dequantize())

    if hasattr(model, "sub_layer"):
        to_int2(model.sub_layer, group_size=group_size, codebook=codebook)
    if hasattr(model, "down_layer"):
        to_int2(model.down_layer, group_size=group_size, codebook=codebook)

    return model

