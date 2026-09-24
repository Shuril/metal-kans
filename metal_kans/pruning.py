"""
Structural Pruning and Node Compaction for metal-KANs.
Computes node importance and physically eliminates inactive hidden neurons.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Any


def _get_layer_weights(layer: Any) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """Extracts base weight, basis weight, basis count, and attribute name."""
    base_w = getattr(layer, "w_base", None)
    basis_keys = ["w_cheby", "w_rbf", "w_relu", "w_wav", "w_fourier", "w_jacobi", "w_spline", "w_p"]
    basis_w = None
    basis_attr = None

    for k in basis_keys:
        if hasattr(layer, k):
            w = getattr(layer, k)
            if w is not None and isinstance(w, np.ndarray):
                basis_w = w
                basis_attr = k
                break

    if basis_w is None:
        raise ValueError(f"Layer {type(layer).__name__} does not contain recognized basis weights.")

    in_dim = layer.in_features
    num_basis = basis_w.shape[1] // in_dim
    return base_w, basis_w, num_basis, basis_attr


def compute_node_importance(model: Any) -> List[np.ndarray]:
    """
    Computes importance scores for hidden neurons across all hidden layers.
    I_j = magnitude_out(layer_l, j) * magnitude_in(layer_{l+1}, j)
    """
    if not hasattr(model, "layers") or len(model.layers) < 2:
        raise ValueError("Model must have at least 2 layers to compute hidden node importance.")

    importances = []
    for l in range(len(model.layers) - 1):
        l1 = model.layers[l]
        l2 = model.layers[l + 1]

        base_w1, basis_w1, k1, _ = _get_layer_weights(l1)
        base_w2, basis_w2, k2, _ = _get_layer_weights(l2)

        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
            if basis_w1.dtype == np.float32 and basis_w2.dtype == np.float32:
                hidden_dim = l1.out_features
                scores = np.empty(hidden_dim, dtype=np.float32)
                b1_ptr = base_w1.ctypes.data if base_w1 is not None and base_w1.size > 1 else None
                b2_ptr = base_w2.ctypes.data if base_w2 is not None and base_w2.size > 1 else None
                has_b1 = 1 if b1_ptr else 0
                has_b2 = 1 if b2_ptr else 0
                code = bridge.metal_kan_node_importance_pair(
                    basis_w1.ctypes.data, b1_ptr,
                    l1.out_features, l1.in_features, k1, has_b1,
                    basis_w2.ctypes.data, b2_ptr,
                    l2.out_features, l2.in_features, k2, has_b2,
                    scores.ctypes.data
                )
                if code == 0:
                    importances.append(scores)
                    continue
        except Exception:
            pass

        # Outgoing magnitude from l1: shape [hidden_dim]
        # basis_w1 has shape [out_features, in_features * k1]
        out_mag = np.sum(np.abs(basis_w1), axis=1)
        if base_w1 is not None and base_w1.size > 1:
            out_mag = out_mag + np.sum(np.abs(base_w1), axis=1)

        # Incoming magnitude to l2: shape [hidden_dim]
        # basis_w2 has shape [out_features2, hidden_dim * k2]
        hidden_dim = l1.out_features
        reshaped_w2 = basis_w2.reshape(l2.out_features, hidden_dim, k2)
        in_mag = np.sum(np.abs(reshaped_w2), axis=(0, 2))
        if base_w2 is not None and base_w2.size > 1:
            in_mag = in_mag + np.sum(np.abs(base_w2), axis=0)

        score = out_mag * in_mag
        importances.append(score)

    return importances


def prune(model: Any, threshold: float = 1e-3) -> Any:
    """
    Zeros out weights belonging to edges whose L1 norm is below threshold.
    """
    if hasattr(model, "layers"):
        for layer in model.layers:
            prune(layer, threshold=threshold)
        return model

    base_w, basis_w, k, attr = _get_layer_weights(model)
    in_dim = model.in_features
    out_dim = model.out_features

    try:
        from .device import get_metal_bridge
        bridge = get_metal_bridge()
        if basis_w.dtype == np.float32 and basis_w.flags['C_CONTIGUOUS']:
            b_ptr = base_w.ctypes.data if base_w is not None and base_w.size > 1 else None
            has_b = 1 if b_ptr else 0
            code = bridge.metal_kan_prune_layer(
                basis_w.ctypes.data,
                b_ptr,
                out_dim,
                in_dim,
                k,
                has_b,
                float(threshold)
            )
            if code == 0:
                return model
    except Exception:
        pass

    reshaped = basis_w.reshape(out_dim, in_dim, k)
    edge_norms = np.sum(np.abs(reshaped), axis=2)  # [out_dim, in_dim]

    mask = (edge_norms >= threshold)
    reshaped_mask = np.repeat(mask[:, :, np.newaxis], k, axis=2)

    new_basis = reshaped * reshaped_mask
    setattr(model, attr, new_basis.reshape(out_dim, in_dim * k).astype(np.float32))

    if base_w is not None and base_w.size > 1:
        new_base = base_w * mask
        setattr(model, "w_base", new_base.astype(np.float32))

    return model



def compact_kan(model: Any, threshold: float = 1e-4) -> Any:
    """
    Physically removes inactive hidden nodes with importance below threshold.
    Returns a new, physically smaller MetalKAN model.
    """
    from .metal_kan import MetalKAN

    if not hasattr(model, "layers") or len(model.layers) < 2:
        return model

    importances = compute_node_importance(model)
    new_hidden_dims = [model.layers[0].in_features]

    active_indices_per_layer = []
    for l, score in enumerate(importances):
        active = np.where(score > threshold)[0]
        if len(active) == 0:
            # Keep at least the top-1 most important neuron
            active = np.array([int(np.argmax(score))])
        active_indices_per_layer.append(active)
        new_hidden_dims.append(len(active))

    new_hidden_dims.append(model.layers[-1].out_features)

    # Instantiate compacted model
    compact_model = MetalKAN(
        layers_hidden=new_hidden_dims,
        basis_type=model.basis_type,
        degree=model.degree,
        bias=model.bias,
        use_base=model.use_base,
    )

    # Slice and copy weights into compacted model
    prev_active = None
    for l, (old_layer, new_layer) in enumerate(zip(model.layers, compact_model.layers)):
        base_w, basis_w, k, attr = _get_layer_weights(old_layer)
        curr_active = active_indices_per_layer[l] if l < len(active_indices_per_layer) else None

        reshaped_basis = basis_w.reshape(old_layer.out_features, old_layer.in_features, k)

        if prev_active is not None:
            reshaped_basis = reshaped_basis[:, prev_active, :]
            if base_w is not None and base_w.size > 1:
                base_w = base_w[:, prev_active]

        if curr_active is not None:
            reshaped_basis = reshaped_basis[curr_active, :, :]
            if base_w is not None and base_w.size > 1:
                base_w = base_w[curr_active, :]
            if old_layer.bias is not None and old_layer.has_bias:
                new_layer.bias = old_layer.bias[curr_active].copy()
        elif old_layer.bias is not None and old_layer.has_bias:
            new_layer.bias = old_layer.bias.copy()

        setattr(new_layer, attr, reshaped_basis.reshape(new_layer.out_features, -1).astype(np.float32))
        if base_w is not None and base_w.size > 1:
            setattr(new_layer, "w_base", base_w.astype(np.float32))

        prev_active = curr_active

    return compact_model
