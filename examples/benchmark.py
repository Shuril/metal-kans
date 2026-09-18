"""
Benchmark suite for metal-KANs comparing various batch sizes on Apple Silicon Metal GPU.
"""

import numpy as np
from metal_kans import ChebyKAN, FastKAN, ReLUKAN

def run_benchmark():
    batch_sizes = [64, 256, 1024, 4096, 16384, 65536]
    in_dim, out_dim, deg = 64, 64, 4

    print("=" * 80)
    print(f"metal-KANs Apple Silicon GPU Benchmark (Layer: {in_dim} -> {out_dim}, Degree: {deg})")
    print("=" * 80)
    print(f"{'Batch Size':<12} | {'ChebyKAN (ms)':<15} | {'Throughput (samples/s)':<22}")
    print("-" * 80)

    cheby = ChebyKAN(in_dim, out_dim, degree=deg)

    for B in batch_sizes:
        x = np.random.uniform(-1.0, 1.0, (B, in_dim)).astype(np.float32)
        ms = cheby.benchmark(x, warmup=5, iters=20)
        throughput = B / (ms / 1000.0)
        print(f"{B:<12} | {ms:<15.3f} | {throughput:>20,.0f}")

    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
