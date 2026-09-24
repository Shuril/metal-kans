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

    rebuild = False
    if not os.path.exists(_DYLIB_PATH):
        rebuild = True
    elif os.path.exists(_MM_PATH) and os.path.getmtime(_MM_PATH) > os.path.getmtime(_DYLIB_PATH):
        rebuild = True

    if rebuild:
        if not os.path.exists(_MM_PATH):
            raise FileNotFoundError(f"Cannot find bridge source: {_MM_PATH}")
        # Compile dynamic library using clang++ with Metal and Foundation frameworks
        cmd = [
            "clang++", "-O3", "-dynamiclib",
            "-framework", "Metal", "-framework", "Foundation", "-framework", "MetalPerformanceShaders", "-framework", "Accelerate",
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

    # Asynchronous pipelining
    lib.metal_kan_set_async.argtypes = [ctypes.c_int]
    lib.metal_kan_set_async.restype = None

    lib.metal_kan_sync.argtypes = []
    lib.metal_kan_sync.restype = None

    # SIMD GEMM
    lib.metal_kan_gemm_simd_fp32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float
    ]
    lib.metal_kan_gemm_simd_fp32.restype = ctypes.c_int

    # FP16 Forwards
    lib.metal_kan_cheby_forward_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_cheby_forward_fp16.restype = ctypes.c_int

    lib.metal_kan_fastkan_forward_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_fastkan_forward_fp16.restype = ctypes.c_int

    lib.metal_kan_relu_forward_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_relu_forward_fp16.restype = ctypes.c_int

    lib.metal_kan_wavkan_forward_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_wavkan_forward_fp16.restype = ctypes.c_int

    lib.metal_kan_fourier_forward_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_fourier_forward_fp16.restype = ctypes.c_int

    lib.metal_kan_jacobi_forward_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_jacobi_forward_fp16.restype = ctypes.c_int

    lib.metal_kan_bspline_forward_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_bspline_forward_fp16.restype = ctypes.c_int

    lib.metal_kan_lowrank_forward_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_lowrank_forward_fp16.restype = ctypes.c_int

    lib.metal_kan_mult_forward_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_mult_forward_fp16.restype = ctypes.c_int

    # Backward gradient and basis derivative signatures
    lib.metal_kan_cheby_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_cheby_backward.restype = ctypes.c_int

    lib.metal_kan_fastkan_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_fastkan_backward.restype = ctypes.c_int

    lib.metal_kan_relu_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_relu_backward.restype = ctypes.c_int

    lib.metal_kan_bspline_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_bspline_backward.restype = ctypes.c_int

    # Native GPU Optimizers & Monolithic Fused Training Steps
    lib.metal_kan_adamw_step.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int
    ]
    lib.metal_kan_adamw_step.restype = ctypes.c_int

    lib.metal_kan_sgd_step.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_sgd_step.restype = ctypes.c_int

    lib.metal_kan_cheby_train_step.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float
    ]
    lib.metal_kan_cheby_train_step.restype = ctypes.c_int

    lib.metal_kan_fastkan_train_step.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float
    ]
    lib.metal_kan_fastkan_train_step.restype = ctypes.c_int

    lib.metal_kan_bspline_train_step.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float
    ]
    lib.metal_kan_bspline_train_step.restype = ctypes.c_int

    lib.metal_kan_lion_step.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int
    ]
    lib.metal_kan_lion_step.restype = ctypes.c_int

    lib.metal_kan_rmsprop_step.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_rmsprop_step.restype = ctypes.c_int

    lib.metal_kan_newton_schulz5.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float
    ]
    lib.metal_kan_newton_schulz5.restype = ctypes.c_int

    lib.metal_kan_newton_schulz5_gpu.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float
    ]
    lib.metal_kan_newton_schulz5_gpu.restype = ctypes.c_int

    lib.metal_kan_cheby_backward_fused.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_cheby_backward_fused.restype = ctypes.c_int

    lib.metal_kan_quantize_ternary.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_quantize_ternary.restype = ctypes.c_int

    lib.metal_kan_forward_ternary_cheby.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_forward_ternary_cheby.restype = ctypes.c_int

    lib.metal_kan_quantize_int2.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_quantize_int2.restype = ctypes.c_int

    lib.metal_kan_forward_int2_cheby.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_forward_int2_cheby.restype = ctypes.c_int

    lib.metal_kan_quantize_int8.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_quantize_int8.restype = ctypes.c_int

    lib.metal_kan_dequantize_int8.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_dequantize_int8.restype = ctypes.c_int

    lib.metal_kan_quantize_int4.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_quantize_int4.restype = ctypes.c_int

    lib.metal_kan_dequantize_int4.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_dequantize_int4.restype = ctypes.c_int

    lib.metal_kan_prune_layer.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float
    ]
    lib.metal_kan_prune_layer.restype = ctypes.c_int

    lib.metal_kan_node_importance_pair.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p
    ]
    lib.metal_kan_node_importance_pair.restype = ctypes.c_int

    # Native Fused SwiGLU Gating & Add
    lib.metal_kan_swiglu_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
    ]
    lib.metal_kan_swiglu_forward.restype = ctypes.c_int

    lib.metal_kan_swiglu_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
    ]
    lib.metal_kan_swiglu_backward.restype = ctypes.c_int

    lib.metal_kan_elementwise_add.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
    ]
    lib.metal_kan_elementwise_add.restype = ctypes.c_int

    # Fused Coupled Adam & Muon Step
    lib.metal_kan_adam_step.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int
    ]
    lib.metal_kan_adam_step.restype = ctypes.c_int

    lib.metal_kan_muon_step.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_muon_step.restype = ctypes.c_int

    # Fast GPU Sub-4-bit Dequantization
    lib.metal_kan_dequantize_ternary.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_dequantize_ternary.restype = ctypes.c_int

    lib.metal_kan_dequantize_int2.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.metal_kan_dequantize_int2.restype = ctypes.c_int

    # MSE Loss Gradient
    lib.metal_kan_calc_mse_loss_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    ]
    lib.metal_kan_calc_mse_loss_backward.restype = ctypes.c_int

    init_code = lib.metal_kan_init(_SHADER_PATH.encode("utf-8"))

    if init_code != 0:
        raise RuntimeError(f"Metal KAN shader initialization failed with code {init_code}")

    _lib = lib
    return _lib


def set_async(enabled: bool = True) -> None:
    """Enable or disable asynchronous command dispatch without per-step synchronization."""
    bridge = get_metal_bridge()
    bridge.metal_kan_set_async(1 if enabled else 0)


def sync() -> None:
    """Wait for all pending GPU commands to finish execution."""
    bridge = get_metal_bridge()
    bridge.metal_kan_sync()
