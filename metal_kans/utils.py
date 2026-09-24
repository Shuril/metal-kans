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


def build_train_step(model: Any, optimizer: Any, loss_fn: Any = None):
    """
    Compiles forward pass, MSE loss gradient, backward pass, and optimizer update
    into a high-performance training step. Dispatches to fused Metal GPU kernels
    for maximum throughput on Apple Silicon.
    """
    lr = getattr(optimizer, "lr", 1e-3)
    b1 = getattr(optimizer, "beta1", 0.9)
    b2 = getattr(optimizer, "beta2", 0.999)
    eps = getattr(optimizer, "eps", 1e-8)
    wd = getattr(optimizer, "weight_decay", 0.01)

    if hasattr(model, "train_step"):
        def fused_step(x: np.ndarray, y: np.ndarray) -> float:
            return model.train_step(x, y, optimizer=optimizer, lr=lr, beta1=b1, beta2=b2, eps=eps, weight_decay=wd)
        return fused_step

    def standard_step(x: np.ndarray, y: np.ndarray) -> float:
        pred = model(x)
        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        if (
            bridge is not None
            and isinstance(pred, np.ndarray)
            and isinstance(y, np.ndarray)
            and pred.dtype == np.float32
            and pred.flags['C_CONTIGUOUS']
            and y.flags['C_CONTIGUOUS']
            and hasattr(bridge, "metal_kan_calc_mse_loss_backward")
        ):
            diff = np.empty_like(pred)
            code = bridge.metal_kan_calc_mse_loss_backward(
                pred.ctypes.data,
                y.ctypes.data,
                diff.ctypes.data,
                pred.size,
                2.0 / float(len(x))
            )
            if code != 0:
                diff = 2.0 * (pred - y) / float(len(x))
        else:
            diff = 2.0 * (pred - y) / float(len(x))

        model.backward(diff)
        optimizer.step()
        model.zero_grad()
        return 0.0

    return standard_step

