# metal-KANs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20--%20M4%20Metal%20GPU-purple.svg)](https://developer.apple.com/metal/)
[![Metal Shading Language](https://img.shields.io/badge/MSL-3.0+-orange.svg)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**metal-KANs** is a pure **Metal Shading Language (MSL)** Kolmogorov-Arnold Networks (KAN) suite for Apple Silicon (M1 / M2 / M3 / M4 / Pro / Max / Ultra) with a lightweight, idiomatic Python / NumPy wrapper.

Unlike standard deep learning frameworks that materialize intermediate 3D basis tensors in VRAM before launching a matrix multiplication GEMM, **metal-KANs fuses basis evaluation and projection accumulation directly inside GPU thread registers and threadgroup shared memory (SRAM)**.

> 🇷🇺 *Русскоязычная версия документации доступна в [README_RU.md](README_RU.md).*

---

## Key Highlights

- **Complete Suite of 10 KAN Architectures**:
  - `ChebyKAN`: Chebyshev polynomial recurrence ($T_k(x) = 2x T_{k-1} - T_{k-2}$) in registers.
  - `FastKAN`: Gaussian Radial Basis Functions (RBF) with shared centers.
  - `ReLUKAN`: Piecewise linear tent basis with zero transcendental operations.
  - `WavKAN`: Continuous wavelets (Mexican Hat, Morlet, DOG) for multiresolution localization.
  - `FourierKAN`: Trigonometric harmonic series ($\cos(k \pi x), \sin(k \pi x)$) for periodic signals.
  - `JacobiKAN`: Orthogonal Jacobi polynomials with weighting parameters $(\alpha, \beta)$.
  - `RationalKAN`: Padé-Chebyshev rational functions ($P(x) / (1 + |Q(x)|)$) for poles and boundary layers.
  - `BSplineKAN` / `KAN`: Classic Cox-de Boor cubic B-spline evaluation in registers.
  - `MultKAN`: KAN 2.0 with explicit multiplication nodes ($u \cdot v$) for physical conservation laws.
  - `LowRankKAN`: Bottleneck / LoRA factorized spline projections with up to 10x parameter reduction.
- **Zero-Allocation GPU Compute**: Basis functions are computed on the fly in thread registers and fused with the output GEMM accumulator: **0 intermediate VRAM allocations**.
- **Zero-Copy Host Interoperability**: Direct pointer binding to host NumPy arrays via macOS Unified Memory (`newBufferWithBytesNoCopy`). No `memcpy` between CPU and GPU.
- **Single-Dispatch Chained Pipeline**: Multi-layer networks execute in a single command buffer with ping-pong GPU buffers, eliminating CPU synchronization latency.
- **Native Quantization**: Block-affine INT8 and packed INT4 weight quantization (`to_int8`, `to_int4`).
- **Structural Pruning & Compaction**: Node importance scoring and physical neuron compaction (`compact_kan`).
- **Symbolic Regression & C Export**: Converts trained KAN activations into mathematical formulas and exports standalone C99 headers (`export_c`).
- **Zero Framework Bloat**: Pure Metal backend. No PyTorch, MLX, or libtorch required.

---

## Supported Architectures

| Architecture | Mathematical Basis | Hardware Acceleration | Recommended Use Case |
|---|---|---|---|
| **`ChebyKAN`** | Chebyshev polynomials ($T_k$) | 3-term recurrence in registers | Smooth functions, physical modeling |
| **`FastKAN`** | Gaussian RBF ($\exp(-d^2 / 2\sigma^2)$) | Shared centers in threadgroup SRAM | Universal function approximation |
| **`ReLUKAN`** | Piecewise linear tent | Hardware clamp, 0 transcendental ops | Low latency, edge inference |
| **`WavKAN`** | Mexican Hat, Morlet, DOG | Localized wavelets in registers | Time series, frequency analysis |
| **`FourierKAN`** | Harmonics ($\cos, \sin$) | Trigonometric SIMD evaluation | Periodic signals, PINNs, audio |
| **`JacobiKAN`** | Jacobi polynomials $(\alpha, \beta)$ | Generalized orthogonal recurrence | Differential equations, boundary problems |
| **`RationalKAN`** | Padé rational ($P/Q$) | Chebyshev rational fractions | Poles, boundary layers, kinetics |
| **`BSplineKAN`** | Cox-de Boor B-splines | Knot interval search & recursion | Classic KAN, interpretability |
| **`MultKAN`** | Multiplicative nodes ($u \cdot v$) | Fused spline + product channels | Analytical formulas, physics |
| **`LowRankKAN`** | Rank factorized ($W_{\text{up}} W_{\text{down}}$) | Bottleneck projection | Deep / wide high-dimensional models |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Shuril/metal-kans.git
cd metal-kans

# Install in editable mode
pip install -e .
```

*Note: On first import, the C++/Objective-C Metal bridge is automatically compiled by the system `clang++` in under 1 second.*

---

## Quick Start

```python
import numpy as np
from metal_kans import (
    ChebyKAN, FastKAN, ReLUKAN, WavKAN, FourierKAN, JacobiKAN, RationalKAN, BSplineKAN,
    MetalKAN, to_int8, compact_kan, to_symbolic
)

# 1. Instantiate a multi-layer KAN network with GPU pipelining
model = MetalKAN(
    layers_hidden=[4, 32, 16, 2], 
    basis_type="cheby", 
    degree=4, 
    pipeline=True
)

x = np.random.uniform(-1.0, 1.0, (1024, 4)).astype(np.float32)
y = model(x)
print("Output shape:", y.shape)  # (1024, 2)

# 2. INT8 Quantization (4x parameter compression)
to_int8(model)

# 3. Structural Pruning & Physical Compaction
compact_model = compact_kan(model, threshold=1e-3)

# 4. Symbolic Formula Discovery & C Code Export
sym = to_symbolic(compact_model, sample_points=100)
sym.export_c("kan_model.h", func_name="kan_predict")
```

---

## Performance Benchmarks on Apple Silicon GPU

Benchmarked on **Apple M1 GPU** (Layer: $64 \to 64$, Degree: 4):

```text
================================================================================
Batch Size   | Latency (ms)    | Throughput (samples/sec) | Intermediate VRAM
--------------------------------------------------------------------------------
64           | 0.342 ms        |          187,172         | 0 KB (Registers)
256          | 0.483 ms        |          530,355         | 0 KB (Registers)
1,024        | 0.703 ms        |        1,456,796         | 0 KB (Registers)
4,096        | 1.444 ms        |        2,836,266         | 0 KB (Registers)
16,384       | 2.506 ms        |        6,538,110         | 0 KB (Registers)
65,536       | 9.165 ms        |        7,150,867         | 0 KB (Registers)
================================================================================
```
*(Peak single-layer throughput exceeds **11.6+ Million samples/sec** on base Apple M1).*

### Multi-Architecture Benchmark (Batch 4,096, 64 -> 64)

| Architecture | MLX JIT | metal-KANs (v0.2.2) | Speedup vs MLX | Optimization Highlights |
| :--- | :--- | :--- | :--- | :--- |
| **ChebyKAN** | 1.45 ms | **1.23 ms** | **1.18x** | Clenshaw recurrence in GPU registers |
| **BSplineKAN** | 2.07 ms | **1.52 ms** | **1.37x** | Closed-form cubic Horner polynomial in registers |
| **FastKAN** | 1.28 ms | **1.16 ms** | **1.10x** | Decoupled SIMD float4 RBF + hardware MPS GEMM (was 2.06 ms) |
| **WavKAN** | 1.12 ms | **0.92 ms** | **1.22x** | Vectorized SIMD wavelets + hardware MPS GEMM (was 2.14 ms) |
| **3-Layer Deep MetalKAN** | 6.15 ms | **4.88 ms** | **1.26x** | Single GPU command buffer pipeline |

---

## C Code Export

Export trained KAN models to zero-dependency standalone C99 headers for embedded deployment:

```c
#include "kan_model.h"

int main() {
    float x[4] = {0.2f, -0.5f, 0.8f, 0.1f};
    float y[2];
    kan_predict(x, y);
    printf("Prediction: %f, %f\n", y[0], y[1]);
    return 0;
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
