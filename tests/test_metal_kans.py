import unittest
import numpy as np
import sys
import os

# Ensure metal_kans package is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metal_kans import (
    is_metal_available,
    ChebyKAN,
    FastKAN,
    ReLUKAN,
    MetalKAN,
)


class TestMetalKANs(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_metal_device(self):
        self.assertTrue(is_metal_available(), "Metal device should be detected on Apple Silicon")

    def test_cheby_kan_shapes(self):
        layer = ChebyKAN(in_features=8, out_features=16, degree=4)
        # Small batch
        x1 = np.random.uniform(-1.0, 1.0, (1, 8)).astype(np.float32)
        y1 = layer(x1)
        self.assertEqual(y1.shape, (1, 16))
        self.assertFalse(np.isnan(y1).any())

        # Large batch
        x2 = np.random.uniform(-1.0, 1.0, (1024, 8)).astype(np.float32)
        y2 = layer(x2)
        self.assertEqual(y2.shape, (1024, 16))
        self.assertFalse(np.isnan(y2).any())

    def test_fast_kan_shapes(self):
        layer = FastKAN(in_features=12, out_features=6, num_centers=8)
        x = np.random.uniform(-1.0, 1.0, (256, 12)).astype(np.float32)
        y = layer(x)
        self.assertEqual(y.shape, (256, 6))
        self.assertFalse(np.isnan(y).any())

    def test_relu_kan_shapes(self):
        layer = ReLUKAN(in_features=10, out_features=5, num_grids=6)
        x = np.random.uniform(-1.0, 1.0, (128, 10)).astype(np.float32)
        y = layer(x)
        self.assertEqual(y.shape, (128, 5))
        self.assertFalse(np.isnan(y).any())

    def test_multidim_input(self):
        layer = ChebyKAN(in_features=4, out_features=8, degree=3)
        x = np.random.uniform(-1.0, 1.0, (4, 16, 4)).astype(np.float32)
        y = layer(x)
        self.assertEqual(y.shape, (4, 16, 8))

    def test_metal_kan_multilayer(self):
        net = MetalKAN([4, 16, 8, 2], basis_type="cheby", degree=4)
        x = np.random.uniform(-1.0, 1.0, (64, 4)).astype(np.float32)
        y = net(x)
        self.assertEqual(y.shape, (64, 2))
        self.assertFalse(np.isnan(y).any())

    def test_benchmark(self):
        layer = ChebyKAN(in_features=16, out_features=16, degree=4)
        x = np.random.uniform(-1.0, 1.0, (512, 16)).astype(np.float32)
        ms = layer.benchmark(x, warmup=2, iters=5)
        self.assertGreater(ms, 0.0)


if __name__ == "__main__":
    unittest.main()
