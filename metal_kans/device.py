"""
Device management, dynamic compilation, and ctypes bridge for pure Metal compute.
"""

from __future__ import annotations
import os
import sys
import ctypes
import subprocess
from typing import Optional

_KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_DYLIB_PATH = os.path.join(_KERNELS_DIR, "libmetal_kan.dylib")
_SHADER_PATH = os.path.join(_KERNELS_DIR, "fused_kan.metal")
_MM_PATH = os.path.join(_KERNELS_DIR, "metal_kan_bridge.mm")

_lib: Optional[ctypes.CDLL] = None


def is_metal_available() -> bool:
    """Checks if macOS Metal GPU is available."""
    if not sys.platform.startswith("darwin"):
        return False
    try:
        bridge = get_metal_bridge()
        return bridge is not None
    except Exception:
        return False


def get_metal_bridge() -> ctypes.CDLL:
    """
    Returns the loaded Metal C++ bridge CDLL.
    If the shared library does not exist, it compiles on the fly using clang++.
    """
    global _lib
    if _lib is not None:
        return _lib

    if not sys.platform.startswith("darwin"):
        raise RuntimeError("metal-KANs requires Apple Silicon macOS with Metal support.")

    if not os.path.exists(_DYLIB_PATH):
        if not os.path.exists(_MM_PATH):
            raise FileNotFoundError(f"Cannot find bridge source: {_MM_PATH}")
        # Compile dynamic library using clang++ with Metal and Foundation frameworks
        cmd = [
            "clang++", "-O3", "-dynamiclib",
            "-framework", "Metal", "-framework", "Foundation", "-framework", "MetalPerformanceShaders",
            _MM_PATH, "-o", _DYLIB_PATH
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Compilation of Metal bridge failed:\n{res.stderr}")

    lib = ctypes.CDLL(_DYLIB_PATH)

    # Function signatures
    lib.metal_kan_init.argtypes = [ctypes.c_char_p]
    lib.metal_kan_init.restype = ctypes.c_int

    lib.metal_kan_cheby_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_cheby_forward.restype = ctypes.c_int

    lib.metal_kan_fastkan_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_fastkan_forward.restype = ctypes.c_int

    lib.metal_kan_relu_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_relu_forward.restype = ctypes.c_int

    lib.metal_kan_wavkan_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_wavkan_forward.restype = ctypes.c_int

    lib.metal_kan_fourier_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_fourier_forward.restype = ctypes.c_int

    lib.metal_kan_jacobi_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_jacobi_forward.restype = ctypes.c_int

    lib.metal_kan_rational_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_rational_forward.restype = ctypes.c_int

    lib.metal_kan_bspline_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_bspline_forward.restype = ctypes.c_int

    lib.metal_kan_chain_pipeline_cheby.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)
    ]
    lib.metal_kan_chain_pipeline_cheby.restype = ctypes.c_int

    lib.benchmark_metal_cheby.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int
    ]
    lib.benchmark_metal_cheby.restype = ctypes.c_double

    lib.benchmark_metal_fastkan.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int
    ]
    lib.benchmark_metal_fastkan.restype = ctypes.c_double

    lib.benchmark_metal_wavkan.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int
    ]
    lib.benchmark_metal_wavkan.restype = ctypes.c_double

    lib.benchmark_metal_bspline.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int
    ]
    lib.benchmark_metal_bspline.restype = ctypes.c_double

    lib.metal_kan_combine_mult_nodes.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_combine_mult_nodes.restype = ctypes.c_int

    lib.benchmark_metal_relu.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int
    ]
    lib.benchmark_metal_relu.restype = ctypes.c_double

    lib.benchmark_metal_fourier.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int
    ]
    lib.benchmark_metal_fourier.restype = ctypes.c_double

    lib.benchmark_metal_jacobi.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int
    ]
    lib.benchmark_metal_jacobi.restype = ctypes.c_double

    lib.benchmark_metal_rational.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int
    ]
    lib.benchmark_metal_rational.restype = ctypes.c_double

    lib.metal_kan_lowrank_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_lowrank_forward.restype = ctypes.c_int

    lib.benchmark_metal_lowrank.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int
    ]
    lib.benchmark_metal_lowrank.restype = ctypes.c_double

    lib.metal_kan_mult_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_mult_forward.restype = ctypes.c_int

    lib.benchmark_metal_mult.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int
    ]
    lib.benchmark_metal_mult.restype = ctypes.c_double

    init_code = lib.metal_kan_init(_SHADER_PATH.encode("utf-8"))
    if init_code != 0:
        raise RuntimeError(f"Metal KAN shader initialization failed with code {init_code}")

    _lib = lib
    return _lib
