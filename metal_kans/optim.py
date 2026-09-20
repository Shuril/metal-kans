"""
Pure Metal/NumPy Optimizers for Metal-KANs training.
No dependency on PyTorch or MLX.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Sequence, Tuple, List, Optional, Union


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
    Stochastic Gradient Descent with optional momentum and weight decay.
    """
    def __init__(
        self,
        params: Sequence[Tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.velocities: List[Optional[np.ndarray]] = [
            np.zeros_like(p) if momentum > 0.0 else None for p, _ in self.params
        ]

    def step(self) -> None:
        for i, (param, grad) in enumerate(self.params):
            if grad is None:
                continue

            g = grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * param

            if self.momentum != 0.0:
                v = self.velocities[i]
                v[:] = self.momentum * v + g
                update = v
            else:
                update = g

            param -= self.lr * update


class Adam(Optimizer):
    """
    Adam optimizer with decoupled or standard weight decay.
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

        for i, (param, grad) in enumerate(self.params):
            if grad is None:
                continue

            g = grad
            if self.weight_decay != 0.0:
                param -= self.lr * self.weight_decay * param

            m = self.m[i]
            v = self.v[i]

            # m = beta1 * m + (1 - beta1) * g
            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            # v = beta2 * v + (1 - beta2) * g^2
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g * g)

            # update param = param - lr_t * m / (sqrt(v) + eps)
            denom = np.sqrt(v) + self.eps
            param -= lr_t * (m / denom)
