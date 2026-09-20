"""
metal-KANs: Pure Metal Shading Language (MSL) Kolmogorov-Arnold Networks for Apple Silicon GPU.
High-throughput, zero-allocation GPU compute shaders with clean Python/NumPy interface.
"""

from .device import is_metal_available, get_metal_bridge
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
from .quantization import QuantizedWeight, quantize, to_int8, to_int4
from .pruning import compute_node_importance, prune, compact_kan
from .symbolic import to_symbolic, SymbolicKAN, SymbolicEdge
from .utils import count_parameters, get_model_size

__version__ = "0.3.1"

__all__ = [
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
    # Multi-Layer Network
    "MetalKAN",
    # INT8 / INT4 Quantization
    "QuantizedWeight",
    "quantize",
    "to_int8",
    "to_int4",
    # Pruning & Compaction
    "compute_node_importance",
    "prune",
    "compact_kan",
    # Symbolic Discovery & C Export
    "to_symbolic",
    "SymbolicKAN",
    "SymbolicEdge",
    # Utilities & Runtime
    "count_parameters",
    "get_model_size",
    "is_metal_available",
    "get_metal_bridge",
]
