"""
Unit tests for PyTorch interop with metal-KANs.
"""

import unittest
import numpy as np
import torch

from metal_kans import TorchMetalKANLinear, TorchMetalKAN, is_metal_available


class TestTorchMetalKAN(unittest.TestCase):
    def setUp(self):
        if not is_metal_available():
            self.skipTest("Metal GPU not available")

    def test_torch_cheby_forward_and_backward(self):
        B, D_in, D_out = 16, 4, 8
        layer = TorchMetalKANLinear(D_in, D_out, basis="cheby", degree=4)

        x = torch.randn(B, D_in, requires_grad=True)
        y = layer(x)

        self.assertEqual(y.shape, (B, D_out))
        self.assertFalse(torch.isnan(y).any())

        # Test autograd backward pass
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (B, D_in))
        self.assertFalse(torch.isnan(x.grad).any())

    def test_torch_fastkan_forward_and_backward(self):
        B, D_in, D_out = 8, 6, 4
        layer = TorchMetalKANLinear(D_in, D_out, basis="fastkan", degree=6)

        x = torch.randn(B, D_in, requires_grad=True)
        y = layer(x)

        self.assertEqual(y.shape, (B, D_out))
        loss = (y ** 2).mean()
        loss.backward()

        self.assertIsNotNone(x.grad)

    def test_torch_multilayer_network(self):
        net = TorchMetalKAN([4, 8, 2], basis="cheby", degree=3)
        x = torch.randn(10, 4)
        y = net(x)
        self.assertEqual(y.shape, (10, 2))


if __name__ == "__main__":
    unittest.main()
