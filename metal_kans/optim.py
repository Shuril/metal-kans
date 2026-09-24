"""
Pure Metal/NumPy Optimizers for Metal-KANs training.
Includes SGD, Adam, AdamW, Muon, RMSprop, and Lion without external PyTorch or MLX dependencies.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Sequence, Tuple, List, Optional, Union, Dict, Any


def _zeropower_via_newtonschulz5(G: np.ndarray, steps: int = 5, eps: float = 1e-7, use_gpu: bool = True) -> np.ndarray:
    """
    Newton-Schulz iteration (order 5) to compute the approximate matrix sign / orthogonalization.
    Used by the Muon optimizer (MomentUm Orthogonalized by Newton-schulz).
    Formula coefficients: a=3.4445, b=-4.7750, c=2.0315
    """
    assert G.ndim == 2, f"Expected 2D matrix for Newton-Schulz, got {G.shape}"

    if G.dtype == np.float32:
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
            G_contig = np.ascontiguousarray(G, dtype=np.float32)
            out = np.empty_like(G_contig)
            if use_gpu and hasattr(bridge, "metal_kan_newton_schulz5_gpu"):
                code = bridge.metal_kan_newton_schulz5_gpu(
                    G_contig.ctypes.data,
                    out.ctypes.data,
                    G_contig.shape[0],
                    G_contig.shape[1],
                    steps,
                    float(eps)
                )
                if code == 0:
                    return out
            code = bridge.metal_kan_newton_schulz5(
                G_contig.ctypes.data,
                out.ctypes.data,
                G_contig.shape[0],
                G_contig.shape[1],
                steps,
                float(eps)
            )
            if code == 0:
                return out
        except Exception:
            pass

    a, b, c = 3.4445, -4.7750, 2.0315

    transpose = G.shape[0] > G.shape[1]
    X = G.T if transpose else G.copy()

    # Spectral norm normalization
    norm = np.linalg.norm(X) + eps
    X = X / norm

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transpose:
        X = X.T
    return X



class Optimizer:
    """Base class for Metal-KAN optimizers."""
    def __init__(self, params: Sequence[Tuple[np.ndarray, np.ndarray]], lr: float = 1e-3):
        self.params = list(params)
        self.lr = float(lr)

    def zero_grad(self) -> None:
        for param, grad in self.params:
            if grad is not None:
                grad.fill(0)

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum, Nesterov acceleration, and weight decay.
    """
    def __init__(
        self,
        params: Sequence[Tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(params, lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self.velocities: List[Optional[np.ndarray]] = [
            np.zeros_like(p) if momentum > 0.0 else None for p, _ in self.params
        ]

    def step(self) -> None:
        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        for i, (param, grad) in enumerate(self.params):
            if grad is None:
                continue

            v = self.velocities[i]
            if v is None:
                v = np.zeros_like(param)
                self.velocities[i] = v

            if bridge is not None and param.dtype == np.float32 and param.flags['C_CONTIGUOUS'] and grad.flags['C_CONTIGUOUS']:
                bridge.metal_kan_sgd_step(
                    param.ctypes.data,
                    grad.ctypes.data,
                    v.ctypes.data,
                    self.lr,
                    self.momentum,
                    self.weight_decay,
                    1 if self.nesterov else 0,
                    param.size
                )
                continue

            # Fallback
            g = grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * param

            if self.momentum != 0.0:
                v[:] = self.momentum * v + g
                if self.nesterov:
                    update = g + self.momentum * v
                else:
                    update = v
            else:
                update = g

            param -= self.lr * update


class Adam(Optimizer):
    """
    Standard Adam optimizer with L2 weight decay (coupled).
    """
    def __init__(
        self,
        params: Sequence[Tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0

        self.m = [np.zeros_like(p) for p, _ in self.params]
        self.v = [np.zeros_like(p) for p, _ in self.params]

    def step(self) -> None:
        self.t += 1
        bias_correction1 = 1.0 - self.beta1 ** self.t
        bias_correction2 = 1.0 - self.beta2 ** self.t
        lr_t = self.lr * math.sqrt(bias_correction2) / bias_correction1

        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        for i, (param, grad) in enumerate(self.params):
            if grad is None:
                continue

            if (
                bridge is not None
                and param.dtype == np.float32
                and param.flags['C_CONTIGUOUS']
                and grad.flags['C_CONTIGUOUS']
                and hasattr(bridge, "metal_kan_adam_step")
            ):
                m = self.m[i]
                v = self.v[i]
                code = bridge.metal_kan_adam_step(
                    param.ctypes.data,
                    grad.ctypes.data,
                    m.ctypes.data,
                    v.ctypes.data,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.weight_decay,
                    lr_t,
                    param.size
                )
                if code == 0:
                    continue

            if self.weight_decay == 0.0 and bridge is not None and param.dtype == np.float32 and param.flags['C_CONTIGUOUS'] and grad.flags['C_CONTIGUOUS']:
                m = self.m[i]
                v = self.v[i]
                bridge.metal_kan_adamw_step(
                    param.ctypes.data,
                    grad.ctypes.data,
                    m.ctypes.data,
                    v.ctypes.data,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    0.0,
                    lr_t,
                    param.size
                )
                continue

            g = grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * param

            m = self.m[i]
            v = self.v[i]

            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g * g)

            denom = np.sqrt(v) + self.eps
            param -= lr_t * (m / denom)


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay (Loshchilov & Hutter, 2019).
    """
    def __init__(
        self,
        params: Sequence[Tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0

        self.m = [np.zeros_like(p) for p, _ in self.params]
        self.v = [np.zeros_like(p) for p, _ in self.params]

    def step(self) -> None:
        self.t += 1
        bias_correction1 = 1.0 - self.beta1 ** self.t
        bias_correction2 = 1.0 - self.beta2 ** self.t
        lr_t = self.lr * math.sqrt(bias_correction2) / bias_correction1

        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        for i, (param, grad) in enumerate(self.params):
            if grad is None:
                continue

            if bridge is not None and param.dtype == np.float32 and param.flags['C_CONTIGUOUS'] and grad.flags['C_CONTIGUOUS']:
                m = self.m[i]
                v = self.v[i]
                bridge.metal_kan_adamw_step(
                    param.ctypes.data,
                    grad.ctypes.data,
                    m.ctypes.data,
                    v.ctypes.data,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.weight_decay,
                    lr_t,
                    param.size
                )
                continue

            # Fallback
            if self.weight_decay != 0.0:
                param -= self.lr * self.weight_decay * param

            g = grad
            m = self.m[i]
            v = self.v[i]

            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g * g)

            denom = np.sqrt(v) + self.eps
            param -= lr_t * (m / denom)


class Muon(Optimizer):
    """
    Muon (MomentUm Orthogonalized by Newton-schulz) optimizer.
    Uses 5th-order Newton-Schulz polynomial iteration to orthogonalize updates on 2D weight matrices,
    providing exceptional convergence speed on neural networks and KAN layers.
    Falls back to AdamW/SGD for 1D vectors (biases).
    """
    def __init__(
        self,
        params: Sequence[Tuple[np.ndarray, np.ndarray]],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        super().__init__(params, lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self.ns_steps = int(ns_steps)
        self.v = [np.zeros_like(p) for p, _ in self.params]

    def step(self) -> None:
        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        for i, (param, grad) in enumerate(self.params):
            if grad is None:
                continue

            v = self.v[i]
            if (
                bridge is not None
                and param.ndim >= 2
                and param.dtype == np.float32
                and param.flags['C_CONTIGUOUS']
                and grad.flags['C_CONTIGUOUS']
                and v.flags['C_CONTIGUOUS']
                and hasattr(bridge, "metal_kan_muon_step")
            ):
                rows = param.shape[0]
                cols = param.size // rows
                ret = bridge.metal_kan_muon_step(
                    param.ctypes.data,
                    grad.ctypes.data,
                    v.ctypes.data,
                    self.lr,
                    self.momentum,
                    self.weight_decay,
                    1 if self.nesterov else 0,
                    self.ns_steps,
                    rows,
                    cols
                )
                if ret == 0:
                    continue

            # Fallback
            g = grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * param

            v[:] = self.momentum * v + (1.0 - self.momentum) * g

            if self.nesterov:
                update = (1.0 - self.momentum) * g + self.momentum * v
            else:
                update = v.copy()

            step_lr = self.lr

            # Orthogonalize 2D weight matrices via Newton-Schulz
            if update.ndim >= 2:
                orig_shape = update.shape
                mat = update.reshape(orig_shape[0], -1) if update.ndim > 2 else update
                mat_ortho = _zeropower_via_newtonschulz5(mat, steps=self.ns_steps)
                update = mat_ortho.reshape(orig_shape)
                # Scale step by aspect ratio
                aspect = max(1.0, float(mat.shape[0]) / float(mat.shape[1])) ** 0.5
                step_lr = step_lr * aspect

            param -= step_lr * update


class RMSprop(Optimizer):
    """
    RMSprop optimizer (Hinton, Coursera lecture 6).
    """
    def __init__(
        self,
        params: Sequence[Tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        super().__init__(params, lr)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)

        self.v = [np.zeros_like(p) for p, _ in self.params]
        self.buf = [np.zeros_like(p) if momentum > 0.0 else None for p, _ in self.params]

    def step(self) -> None:
        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        for i, (param, grad) in enumerate(self.params):
            if grad is None:
                continue

            if bridge is not None and param.dtype == np.float32 and param.flags['C_CONTIGUOUS'] and grad.flags['C_CONTIGUOUS']:
                v = self.v[i]
                buf = self.buf[i]
                buf_ptr = buf.ctypes.data if buf is not None else 0
                has_mom = 1 if self.momentum > 0.0 else 0
                bridge.metal_kan_rmsprop_step(
                    param.ctypes.data,
                    grad.ctypes.data,
                    v.ctypes.data,
                    buf_ptr,
                    self.lr,
                    self.alpha,
                    self.eps,
                    self.weight_decay,
                    self.momentum,
                    has_mom,
                    param.size
                )
                continue

            # Fallback
            g = grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * param

            v = self.v[i]
            v[:] = self.alpha * v + (1.0 - self.alpha) * (g * g)

            avg = g / (np.sqrt(v) + self.eps)

            if self.momentum > 0.0:
                buf = self.buf[i]
                buf[:] = self.momentum * buf + avg
                param -= self.lr * buf
            else:
                param -= self.lr * avg


class Lion(Optimizer):
    """
    Lion (EvoLved Sign Momentum) optimizer (Chen et al., Google Brain, 2023).
    Tracks momentum and uses only the sign of updates, yielding high memory efficiency.
    """
    def __init__(
        self,
        params: Sequence[Tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.weight_decay = float(weight_decay)
        self.m = [np.zeros_like(p) for p, _ in self.params]

    def step(self) -> None:
        bridge = None
        try:
            from .device import get_metal_bridge
            bridge = get_metal_bridge()
        except Exception:
            pass

        for i, (param, grad) in enumerate(self.params):
            if grad is None:
                continue

            if bridge is not None and param.dtype == np.float32 and param.flags['C_CONTIGUOUS'] and grad.flags['C_CONTIGUOUS']:
                m = self.m[i]
                bridge.metal_kan_lion_step(
                    param.ctypes.data,
                    grad.ctypes.data,
                    m.ctypes.data,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.weight_decay,
                    param.size
                )
                continue

            # Fallback
            if self.weight_decay != 0.0:
                param -= self.lr * self.weight_decay * param

            g = grad
            m = self.m[i]

            # Update direction: sign(beta1 * m + (1 - beta1) * g)
            update = np.sign(self.beta1 * m + (1.0 - self.beta1) * g)
            param -= self.lr * update

            # Update momentum buffer: m = beta2 * m + (1 - beta2) * g
            m[:] = self.beta2 * m + (1.0 - self.beta2) * g

