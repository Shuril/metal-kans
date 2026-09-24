"""
Symbolic Regression and C Code Export for metal-KANs.
Converts numerical KAN edge activations into exact symbolic formulas and standalone C headers.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Callable, Any, Optional

CANDIDATE_FUNCTIONS = [
    ("linear", lambda x: x, "x"),
    ("x^2", lambda x: x**2, "x*x"),
    ("x^3", lambda x: x**3, "x*x*x"),
    ("sin(x)", lambda x: np.sin(x), "sinf(x)"),
    ("cos(x)", lambda x: np.cos(x), "cosf(x)"),
    ("exp(x)", lambda x: np.exp(np.clip(x, -5.0, 5.0)), "expf(x)"),
    ("tanh(x)", lambda x: np.tanh(x), "tanhf(x)"),
    ("gaussian", lambda x: np.exp(-x**2), "expf(-x*x)"),
]


class SymbolicEdge:
    """Represents a symbolic univariate activation function phi(x) = c * f(x)."""
    def __init__(self, name: str, fn: Callable[[np.ndarray], np.ndarray], c_expr: str, coef: float):
        self.name = name
        self.fn = fn
        self.c_expr = c_expr
        self.coef = float(coef)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.coef * self.fn(x)

    def to_c_code(self, var_name: str) -> str:
        if abs(self.coef) < 1e-6:
            return "0.0f"
        import re
        expr = re.sub(r'\bx\b', var_name, self.c_expr)
        return f"({self.coef:.6f}f * {expr})"


class SymbolicKAN:
    """
    Symbolic representation of a KAN model.
    Can be evaluated directly in Python or exported to standalone C code.
    """
    def __init__(self, layers_edges: List[List[List[SymbolicEdge]]], biases: List[Optional[np.ndarray]]):
        self.layers_edges = layers_edges
        self.biases = biases
        self.num_layers = len(layers_edges)

    def forward(self, x: np.ndarray) -> np.ndarray:
        curr = x
        for edges_layer, bias in zip(self.layers_edges, self.biases):
            out_dim = len(edges_layer)
            in_dim = len(edges_layer[0])
            B = curr.shape[0]
            next_out = np.zeros((B, out_dim), dtype=np.float32)

            for j in range(out_dim):
                for i in range(in_dim):
                    next_out[:, j] += edges_layer[j][i](curr[:, i])
                if bias is not None:
                    next_out[:, j] += bias[j]
            curr = next_out
        return curr

    __call__ = forward

    def to_c_code(self, func_name: str = "kan_predict") -> str:
        """Generates standalone C code for inference."""
        lines = []
        in_dim = len(self.layers_edges[0][0])
        out_dim = len(self.layers_edges[-1])

        lines.append(f"void {func_name}(const float* x, float* y) {{")

        # Declare temporary layer buffers
        prev_var = "x"
        for l, (layer, bias) in enumerate(zip(self.layers_edges, self.biases)):
            l_out = len(layer)
            l_in = len(layer[0])
            curr_var = f"layer_{l}"

            if l == self.num_layers - 1:
                curr_var = "y"
            else:
                lines.append(f"    float {curr_var}[{l_out}];")

            for j in range(l_out):
                terms = []
                for i in range(l_in):
                    edge = layer[j][i]
                    if abs(edge.coef) >= 1e-6:
                        terms.append(edge.to_c_code(f"{prev_var}[{i}]"))
                if bias is not None and abs(bias[j]) >= 1e-6:
                    terms.append(f"{bias[j]:.6f}f")

                expr_str = " + ".join(terms) if terms else "0.0f"
                lines.append(f"    {curr_var}[{j}] = {expr_str};")

            prev_var = curr_var

        lines.append("}")
        return "\n".join(lines)

    def to_c_header(self, guard: str = "KAN_MODEL_H", func_name: str = "kan_predict") -> str:
        """Generates complete standalone C header file."""
        c_code = self.to_c_code(func_name=func_name)
        header = f"""#ifndef {guard}
#define {guard}

#include <math.h>

#ifdef __cplusplus
extern "C" {{
#endif

{c_code}

#ifdef __cplusplus
}}
#endif

#endif // {guard}
"""
        return header

    def export_c(self, filepath: str, guard: str = "KAN_MODEL_H", func_name: str = "kan_predict"):
        """Exports standalone C header file to disk."""
        content = self.to_c_header(guard=guard, func_name=func_name)
        with open(filepath, "w") as f:
            f.write(content)


def to_symbolic(model: Any, sample_points: int = 200, grid_range: tuple = (-1.0, 1.0)) -> SymbolicKAN:
    """
    Fits symbolic functional candidates to each univariate edge of a trained KAN model.
    """
    if not hasattr(model, "layers"):
        layers = [model]
    else:
        layers = model.layers

    layers_edges = []
    biases = []

    xs = np.linspace(grid_range[0], grid_range[1], sample_points, dtype=np.float32)

    # Precompute candidate functions and norms
    F_mat = np.stack([fn(xs) for _, fn, _ in CANDIDATE_FUNCTIONS], axis=0)  # [num_cands, S]

    F_norms = np.sum(F_mat ** 2, axis=1, keepdims=True)                     # [num_cands, 1]
    F_norms = np.maximum(F_norms, 1e-8)

    for layer in layers:
        in_dim = layer.in_features
        out_dim = layer.out_features

        # Pre-allocate grid of SymbolicEdge: [out_dim, in_dim]
        layer_edges = [[None for _ in range(in_dim)] for _ in range(out_dim)]

        for i in range(in_dim):
            # Construct test input with only feature i active
            X_test = np.zeros((sample_points, in_dim), dtype=np.float32)
            X_test[:, i] = xs

            # Single batched forward call for all output channels
            Y_resp = layer(X_test)  # [S, out_dim]
            if Y_resp.ndim > 2:
                Y_resp = Y_resp.reshape(sample_points, out_dim)

            # Vectorized linear least squares for all candidate functions: C = (F @ Y) / norms
            # F_mat is [K, S], Y_resp is [S, out_dim] -> proj is [K, out_dim]
            proj = F_mat @ Y_resp
            coefs = proj / F_norms  # [K, out_dim]

            for j in range(out_dim):
                y_j = Y_resp[:, j]
                best_err = float("inf")
                best_edge = None

                for k, (name, fn, c_expr) in enumerate(CANDIDATE_FUNCTIONS):
                    c = float(coefs[k, j])
                    fx = F_mat[k]
                    diff = np.clip(y_j - c * fx, -1e4, 1e4)
                    err = float(np.mean(diff ** 2))
                    if err < best_err:
                        best_err = err
                        best_edge = SymbolicEdge(name, fn, c_expr, c)

                layer_edges[j][i] = best_edge

        layers_edges.append(layer_edges)

        bias = layer.bias.copy() if hasattr(layer, "bias") and layer.bias is not None and layer.has_bias else None
        biases.append(bias)

    return SymbolicKAN(layers_edges, biases)
