"""
Quickstart example for metal-KANs.
Demonstrates ChebyKAN, FastKAN, ReLUKAN, and multi-layer MetalKAN.
"""

import numpy as np
from metal_kans import ChebyKAN, FastKAN, ReLUKAN, MetalKAN

def main():
    print("=== metal-KANs Quickstart on Apple Silicon Metal ===")

    # 1. Single ChebyKAN layer
    cheby = ChebyKAN(in_features=4, out_features=2, degree=4)
    x = np.random.uniform(-1.0, 1.0, (10, 4)).astype(np.float32)
    y_cheby = cheby(x)
    print(f"ChebyKAN forward output shape: {y_cheby.shape}")

    # 2. Single FastKAN layer (Gaussian RBF)
    fast = FastKAN(in_features=4, out_features=2, num_centers=8)
    y_fast = fast(x)
    print(f"FastKAN forward output shape:  {y_fast.shape}")

    # 3. Single ReLUKAN layer (Piecewise Linear Tent)
    relu = ReLUKAN(in_features=4, out_features=2, num_grids=8)
    y_relu = relu(x)
    print(f"ReLUKAN forward output shape:  {y_relu.shape}")

    # 4. Multi-layer MetalKAN network
    model = MetalKAN(layers_hidden=[4, 16, 8, 1], basis_type="cheby", degree=4)
    print(f"Network: {model}")
    y_net = model(x)
    print(f"MetalKAN multi-layer output:   {y_net.shape}")

    # 5. Measure latency on large batch
    x_large = np.random.uniform(-1.0, 1.0, (4096, 4)).astype(np.float32)
    ms = cheby.benchmark(x_large, warmup=10, iters=50)
    throughput = 4096 / (ms / 1000.0)
    print(f"ChebyKAN (B=4096) Latency:     {ms:.3f} ms ({throughput:,.0f} samples/sec)")

if __name__ == "__main__":
    main()
