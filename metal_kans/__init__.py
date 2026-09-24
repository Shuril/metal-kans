"""
metal-KANs: Pure Metal Shading Language (MSL) Kolmogorov-Arnold Networks for Apple Silicon GPU.
High-throughput, zero-allocation GPU compute shaders with clean Python/NumPy interface.
"""

from .device import is_metal_available, get_metal_bridge, set_async, sync
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
from .metal_kan import MetalKAN
from .quantization import (
    QuantizedWeight,
    TernaryQuantizedWeight,
    Int2QuantizedWeight,
    quantize,
    to_int8,
    to_int4,
    to_ternary,
    to_int2,
)
from .pruning import compute_node_importance, prune, compact_kan
from .symbolic import to_symbolic, SymbolicKAN, SymbolicEdge
from .utils import count_parameters, get_model_size, build_train_step
from .optim import Optimizer, Adam, AdamW, Muon, RMSprop, Lion, SGD
from .checkpoint import CheckpointedMetalKAN, checkpoint_kan
from .gated_kan import GatedKAN
try:
    from .torch_module import TorchMetalKANLinear, TorchMetalKAN
except ImportError:
    TorchMetalKANLinear = None
    TorchMetalKAN = None

__version__ = "0.6.0"


__all__ = [
    # Asynchronous Execution
    "set_async",
    "sync",
    # 10 KAN Architectures
    "ChebyKAN",
    "FastKAN",
    "ReLUKAN",
    "WavKAN",
    "FourierKAN",
    "JacobiKAN",
    "RationalKAN",
    "BSplineKAN",
    "KAN",
    "MultKAN",
    "LowRankKAN",
    # Multi-Layer Network & Transformer Blocks
    "MetalKAN",
    "GatedKAN",
    "CheckpointedMetalKAN",
    "checkpoint_kan",
    # Optimizers
    "Optimizer",
    "Adam",
    "AdamW",
    "Muon",
    "RMSprop",
    "Lion",
    "SGD",
    # Quantization (INT8, INT4, INT2, Ternary 1.58-bit)
    "QuantizedWeight",
    "TernaryQuantizedWeight",
    "Int2QuantizedWeight",
    "quantize",
    "to_int8",
    "to_int4",
    "to_ternary",
    "to_int2",
    # Pruning & Compaction
    "compute_node_importance",
    "prune",
    "compact_kan",
    # Symbolic Discovery & C Export
    "to_symbolic",
    "SymbolicKAN",
    "SymbolicEdge",
    # Utilities & Runtime
    "build_train_step",
    "count_parameters",
    "get_model_size",
    "is_metal_available",
    "get_metal_bridge",
]
