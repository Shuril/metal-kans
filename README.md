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

Benchmarked on **Apple Silicon GPU** (Layer: $64 \to 64$, Degree: 4) using `metal-kans` **v0.3.0**:

```text
================================================================================
Batch Size   | Latency (ms)    | Throughput (samples/sec) | Intermediate VRAM
--------------------------------------------------------------------------------
128          | 0.290 ms        |          441,379         | 0 KB (Registers)
1,024        | 0.457 ms        |        2,240,700         | 0 KB (Registers)
4,096        | 0.902 ms        |        4,541,019         | 0 KB (Registers)
16,384       | 2.102 ms        |        7,794,481         | 0 KB (Registers)
65,536       | 6.840 ms        |        9,581,286         | 0 KB (Registers)
================================================================================
```
*(Peak single-layer throughput reaches **11.6+ Million samples/sec** on base Apple Silicon).*

### 3-Way GPU Benchmark: MLX vs metal-KANs (v0.3.1) vs slang-KANs (v0.2.0)

Tested across all 10 architectures on **Apple Silicon GPU**, Layer `64 -> 64`:

| Architecture | Batch | MLX (ms) | metal-KANs (Pure Metal AMX) | slang-KANs (Shared-Memory GEMM) | Winner (Speedup) |
|---|---|---|---|---|---|
| **ChebyKAN** | 128 | 0.387 ms | 0.295 ms | **0.226 ms** | **slang-KANs (1.30x)** |
| | 1024 | 0.526 ms | 0.372 ms | **0.324 ms** | **slang-KANs (1.15x)** |
| | 4096 | 1.380 ms | **0.484 ms** | 0.885 ms | **metal-KANs (1.83x)** |
| **BSplineKAN** | 128 | 0.354 ms | 0.258 ms | **0.136 ms** | **slang-KANs (1.90x)** |
| | 1024 | 0.835 ms | **0.404 ms** | 0.541 ms | **metal-KANs (1.34x)** |
| | 4096 | 2.190 ms | **0.826 ms** | 1.400 ms | **metal-KANs (1.70x)** |
| **FastKAN** | 128 | 0.270 ms | **0.254 ms** | 0.692 ms | **metal-KANs (1.06x)** |
| | 1024 | 0.398 ms | **0.352 ms** | 1.439 ms | **metal-KANs (1.13x)** |
| | 4096 | 1.475 ms | **1.006 ms** | 4.010 ms | **metal-KANs (1.47x)** |
| **WavKAN** | 128 | **0.354 ms** | 0.367 ms | 0.869 ms | **MLX (1.04x)** |
| | 1024 | 1.845 ms | **0.541 ms** | 1.941 ms | **metal-KANs (3.41x)** |
| | 4096 | 1.844 ms | **1.189 ms** | 4.132 ms | **metal-KANs (1.55x)** |
| **ReLUKAN** | 128 | **0.311 ms** | 0.364 ms | 0.852 ms | **MLX (1.17x)** |
| | 1024 | 0.542 ms | **0.511 ms** | 1.888 ms | **metal-KANs (1.06x)** |
| | 4096 | 1.254 ms | **1.062 ms** | 4.134 ms | **metal-KANs (1.18x)** |
| **FourierKAN** | 128 | **0.351 ms** | 0.384 ms | 0.883 ms | **MLX (1.09x)** |
| | 1024 | 0.975 ms | **0.605 ms** | 2.098 ms | **metal-KANs (1.61x)** |
| | 4096 | 2.385 ms | **0.868 ms** | 2.868 ms | **metal-KANs (2.75x)** |
| **JacobiKAN** | 128 | 0.405 ms | 0.294 ms | **0.144 ms** | **slang-KANs (2.04x)** |
| | 1024 | 0.632 ms | 0.415 ms | **0.325 ms** | **slang-KANs (1.28x)** |
| | 4096 | 1.413 ms | **0.515 ms** | 0.722 ms | **metal-KANs (1.40x)** |
| **RationalKAN**| 128 | 0.691 ms | 0.385 ms | **0.166 ms** | **slang-KANs (2.32x)** |
| | 1024 | 4.354 ms | 0.822 ms | **0.648 ms** | **slang-KANs (1.27x)** |
| | 4096 | 18.225 ms| **2.300 ms** | 2.468 ms | **metal-KANs (1.07x)** |
| **MultKAN** | 128 | 0.378 ms | **0.278 ms** | 0.783 ms | **metal-KANs (1.36x)** |
| | 1024 | 0.494 ms | **0.415 ms** | 3.738 ms | **metal-KANs (1.19x)** |
| | 4096 | 2.179 ms | **1.526 ms** | 7.734 ms | **metal-KANs (1.43x)** |
| **LowRankKAN** | 128 | **0.514 ms** | 0.546 ms | 0.811 ms | **MLX (1.06x)** |
| | 1024 | 0.866 ms | **0.612 ms** | 8.050 ms | **metal-KANs (1.41x)** |
| | 4096 | 1.722 ms | **1.101 ms** | 27.890 ms| **metal-KANs (1.56x)** |

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
