"""
Utility functions for parameter counting and model size calculation in metal-KANs.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict


def count_parameters(model: Any) -> int:
    """Counts total trainable floating-point parameters in a metal-KAN layer or model."""
    total = 0
    if hasattr(model, "layers"):
        for layer in model.layers:
            total += count_parameters(layer)
        return total

    for attr in ("w_cheby", "w_rbf", "w_relu", "w_wav", "w_fourier", "w_jacobi",
                 "w_p", "w_q", "w_spline", "w_up", "w_base", "bias", "translation", "scale"):
        if hasattr(model, attr):
            val = getattr(model, attr)
            if val is not None and isinstance(val, np.ndarray):
                total += val.size

    if hasattr(model, "sub_layer"):
        total += count_parameters(model.sub_layer)
    if hasattr(model, "down_layer"):
        total += count_parameters(model.down_layer)

    return total


def get_model_size(model: Any) -> Dict[str, Any]:
    """Computes total memory size of model parameters in bytes and megabytes."""
    total_bytes = 0
    total_params = 0

    if hasattr(model, "layers"):
        for layer in model.layers:
            sub = get_model_size(layer)
            total_bytes += sub["bytes"]
            total_params += sub["parameters"]
    else:
        for attr in ("w_cheby", "w_rbf", "w_relu", "w_wav", "w_fourier", "w_jacobi",
                     "w_p", "w_q", "w_spline", "w_up", "w_base", "bias", "translation", "scale",
                     "weight_int8", "weight_int4"):
            if hasattr(model, attr):
                val = getattr(model, attr)
                if val is not None and isinstance(val, np.ndarray):
                    total_bytes += val.nbytes
                    total_params += val.size
        if hasattr(model, "sub_layer"):
            sub = get_model_size(model.sub_layer)
            total_bytes += sub["bytes"]
            total_params += sub["parameters"]
        if hasattr(model, "down_layer"):
            sub = get_model_size(model.down_layer)
            total_bytes += sub["bytes"]
            total_params += sub["parameters"]

    mb = total_bytes / (1024 * 1024)
    return {
        "bytes": total_bytes,
        "mb": mb,
        "parameters": total_params,
        "summary": f"{mb:.3f} MB ({total_bytes:,} bytes, {total_params:,} elements)"
    }
