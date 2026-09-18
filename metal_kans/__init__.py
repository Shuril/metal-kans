"""
metal-KANs: Pure Metal Shading Language (MSL) Kolmogorov-Arnold Networks for Apple Silicon GPU.
Zero-allocation GPU compute shaders with direct register blocking and threadgroup SRAM tiling.
"""

from .device import is_metal_available, get_metal_bridge
from .cheby_kan import ChebyKAN
from .fast_kan import FastKAN
from .relu_kan import ReLUKAN
from .metal_kan import MetalKAN

__version__ = "0.1.0"

__all__ = [
    "ChebyKAN",
    "FastKAN",
    "ReLUKAN",
    "MetalKAN",
    "is_metal_available",
    "get_metal_bridge",
]
