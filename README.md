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

### 3-Way GPU Benchmark: MLX vs metal-KANs (v0.3.0) vs slang-KANs (v0.2.0)

Tested across all 10 architectures on **Apple Silicon GPU**, Layer `64 -> 64`:

| Architecture | Batch | MLX (ms) | metal-KANs (Pure Metal AMX) | slang-KANs (Shared-Memory GEMM) | Winner (Speedup) |
|---|---|---|---|---|---|
| **ChebyKAN** | 128 | 0.321 ms | 0.350 ms | **0.166 ms** | **slang-KANs (1.94x)** |
| | 1024 | 1.269 ms | 0.509 ms | **0.369 ms** | **slang-KANs (1.38x)** |
| | 4096 | 1.383 ms | **0.653 ms** | 0.892 ms | **metal-KANs (1.37x)** |
| **BSplineKAN** | 128 | 0.359 ms | 0.290 ms | **0.132 ms** | **slang-KANs (2.20x)** |
| | 1024 | 0.777 ms | **0.457 ms** | 0.545 ms | **metal-KANs (1.19x)** |
| | 4096 | 2.092 ms | **0.902 ms** | 1.402 ms | **metal-KANs (1.56x)** |
| **FastKAN** | 128 | **0.267 ms** | 0.297 ms | 0.696 ms | **MLX (1.11x)** |
| | 1024 | 0.446 ms | **0.399 ms** | 1.463 ms | **metal-KANs (1.12x)** |
| | 4096 | 1.445 ms | **1.177 ms** | 4.739 ms | **metal-KANs (1.23x)** |
| **WavKAN** | 128 | **0.335 ms** | 0.411 ms | 0.833 ms | **MLX (1.23x)** |
| | 1024 | 1.528 ms | **0.567 ms** | 1.835 ms | **metal-KANs (2.69x)** |
| | 4096 | 1.222 ms | **0.984 ms** | 3.762 ms | **metal-KANs (1.24x)** |
| **ReLUKAN** | 128 | **0.309 ms** | 0.378 ms | 0.861 ms | **MLX (1.22x)** |
| | 1024 | **0.513 ms** | 0.557 ms | 1.840 ms | **MLX (1.08x)** |
| | 4096 | **1.093 ms** | 1.135 ms | 3.250 ms | **MLX (1.04x)** |
| **FourierKAN** | 128 | **0.321 ms** | 0.381 ms | 0.896 ms | **MLX (1.19x)** |
| | 1024 | 0.872 ms | **0.641 ms** | 1.969 ms | **metal-KANs (1.36x)** |
| | 4096 | 2.219 ms | **0.928 ms** | 2.565 ms | **metal-KANs (2.39x)** |
| **JacobiKAN** | 128 | 0.292 ms | 0.298 ms | **0.144 ms** | **slang-KANs (2.03x)** |
| | 1024 | 0.578 ms | 0.465 ms | **0.359 ms** | **slang-KANs (1.30x)** |
| | 4096 | 1.521 ms | **0.574 ms** | 0.727 ms | **metal-KANs (1.27x)** |
| **RationalKAN**| 128 | 0.679 ms | 0.403 ms | **0.132 ms** | **slang-KANs (3.05x)** |
| | 1024 | 4.070 ms | 0.850 ms | **0.652 ms** | **slang-KANs (1.30x)** |
| | 4096 | 18.252 ms| 2.453 ms | **2.393 ms** | **slang-KANs (1.02x)** |
| **MultKAN** | 128 | **0.320 ms** | 0.509 ms | 0.803 ms | **MLX (1.59x)** |
| | 1024 | **0.428 ms** | 0.707 ms | 1.561 ms | **MLX (1.65x)** |
| | 4096 | **1.529 ms** | 1.853 ms | 6.638 ms | **MLX (1.21x)** |
| **LowRankKAN** | 128 | 0.344 ms | 0.454 ms | **0.301 ms** | **slang-KANs (1.14x)** |
| | 1024 | **0.608 ms** | 0.829 ms | 3.306 ms | **MLX (1.36x)** |
| | 4096 | **1.442 ms** | 2.257 ms | 11.785 ms| **MLX (1.57x)** |

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
