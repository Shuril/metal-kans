"""
Comprehensive Quickstart for metal-KANs on Apple Silicon Metal GPU.
Demonstrates layer execution, GPU-pipelining, INT8 quantization, pruning, and C code export.
"""

import numpy as np
from metal_kans import (
    ChebyKAN, FastKAN, ReLUKAN, WavKAN, FourierKAN, JacobiKAN, RationalKAN, BSplineKAN,
    MetalKAN, to_int8, compute_node_importance, compact_kan, to_symbolic, count_parameters, get_model_size
)

def main():
    print("=== metal-KANs: Pure Metal GPU Suite on Apple Silicon ===")

    x = np.random.uniform(-1.0, 1.0, (10, 4)).astype(np.float32)

    # 1. Multi-layer network with GPU pipeline chaining
    model = MetalKAN(layers_hidden=[4, 16, 8, 1], basis_type="cheby", degree=4, pipeline=True)
    print(f"1. Network: {model} (Parameters: {count_parameters(model):,})")
    y = model(x)
    print(f"   Output shape: {y.shape}")

    # 2. INT8 Quantization
    print("\n2. Model Quantization:")
    size_before = get_model_size(model)
    print(f"   Original FP32 size: {size_before['summary']}")
    to_int8(model)
    print("   Quantized to INT8 in-place.")

    # 3. Node Importance & Structural Compaction
    print("\n3. Structural Pruning & Compaction:")
    dense_model = MetalKAN([4, 16, 2], basis_type="cheby", degree=4)
    scores = compute_node_importance(dense_model)
    print(f"   Hidden layer node importance scores: {scores[0][:4]}...")
    compact_model = compact_kan(dense_model, threshold=1e-3)
    print(f"   Compacted network: {compact_model}")

    # 4. Symbolic Formula Discovery & C Code Export
    print("\n4. Symbolic Regression & C Code Export:")
    sym = to_symbolic(compact_model, sample_points=50)
    c_snippet = sym.to_c_code(func_name="predict_kan")
    print(f"   Generated C function:\n{c_snippet[:200]}...\n}}")

    # 5. Throughput Benchmark
    print("\n5. Metal GPU Throughput:")
    cheby = ChebyKAN(64, 64, degree=4)
    x_bench = np.random.uniform(-1.0, 1.0, (4096, 64)).astype(np.float32)
    ms = cheby.benchmark(x_bench, warmup=10, iters=50)
    print(f"   Latency: {ms:.3f} ms | Throughput: {4096 / (ms / 1000.0):,.0f} samples/sec")

if __name__ == "__main__":
    main()
