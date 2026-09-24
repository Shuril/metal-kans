"""
Benchmark suite for metal-KANs comparing various architectures and batch sizes on Apple Silicon GPU.
"""

import numpy as np
from metal_kans import ChebyKAN, FastKAN, BSplineKAN, ReLUKAN


def run_benchmark():
    batch_sizes = [64, 128, 256, 1024, 4096, 16384, 65536]
    in_dim, out_dim = 64, 64

    models = {
        "ChebyKAN (deg=4)": ChebyKAN(in_dim, out_dim, degree=4),
        "FastKAN (centers=8)": FastKAN(in_dim, out_dim, num_centers=8),
        "BSplineKAN (grid=5)": BSplineKAN(in_dim, out_dim, grid_size=5),
        "ReLUKAN (grids=8)": ReLUKAN(in_dim, out_dim, num_grids=8),
    }

    print("=" * 86)
    print(f"metal-KANs Apple Silicon GPU Benchmark (Layer: {in_dim} -> {out_dim})")
    print("=" * 86)

    for name, model in models.items():
        print(f"\n--- {name} ---")
        print(f"{'Batch Size':<12} | {'Latency (ms)':<15} | {'Throughput (samples/s)':<24}")
        print("-" * 58)
        for B in batch_sizes:
            x = np.random.uniform(-1.0, 1.0, (B, in_dim)).astype(np.float32)
            ms = model.benchmark(x, warmup=5, iters=20)
            throughput = B / (ms / 1000.0)
            print(f"{B:<12} | {ms:<15.3f} | {throughput:>22,.0f}")

    print("\n" + "=" * 86)


if __name__ == "__main__":
    run_benchmark()
