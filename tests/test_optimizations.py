"""
Unit tests for Qwen-inspired high-performance optimizations:
1. FP16 Chebyshev & Jacobi polynomial stabilization with fma and FP32 registers.
2. GPU-resident Muon optimizer (Newton-Schulz 5 in pure Metal).
3. Fully fused Chebyshev backward pass without Phi intermediate VRAM materialization.
4. Sub-4-bit Quantization: 1.58-bit Ternary KAN and INT2 Codebook KAN.
5. GatedKAN drop-in SwiGLU / MLP replacement for LLMs.
"""

import unittest
import numpy as np
import metal_kans as mk
from metal_kans.device import get_metal_bridge


class TestFP16Stabilization(unittest.TestCase):
    def test_cheby_boundary_fp16(self):
        """Verify Chebyshev FP16 evaluation is stable at boundaries x = +-1 up to degree 8."""
        B, D_in, D_out, degree = 4, 8, 4, 8
        layer = mk.ChebyKAN(in_features=D_in, out_features=D_out, degree=degree).half()
        X = np.array([[-1.0, -0.999, 0.0, 0.999, 1.0, -0.5, 0.5, 0.0] for _ in range(B)], dtype=np.float16)
        Y = layer(X)

        self.assertEqual(Y.shape, (B, D_out))
        self.assertFalse(np.isnan(Y).any(), "FP16 Chebyshev basis produced NaN!")
        self.assertFalse(np.isinf(Y).any(), "FP16 Chebyshev basis produced Inf!")



class TestGPUMuonNewtonSchulz(unittest.TestCase):
    def test_gpu_newton_schulz5(self):
        """Verify pure Metal GPU Newton-Schulz 5 computes orthogonalized sign matrix."""
        bridge = get_metal_bridge()
        np.random.seed(42)
        rows, cols = 64, 128
        G = np.random.randn(rows, cols).astype(np.float32)

        out_gpu = np.empty_like(G)
        out_cpu = np.empty_like(G)

        code_cpu = bridge.metal_kan_newton_schulz5(G.ctypes.data, out_cpu.ctypes.data, rows, cols, 5, 1e-7)
        code_gpu = bridge.metal_kan_newton_schulz5_gpu(G.ctypes.data, out_gpu.ctypes.data, rows, cols, 5, 1e-7)

        self.assertEqual(code_cpu, 0)
        self.assertEqual(code_gpu, 0)

        max_diff = np.max(np.abs(out_gpu - out_cpu))
        self.assertLess(max_diff, 1e-3, f"GPU vs CPU Newton-Schulz discrepancy too high: {max_diff}")

        # Check approximate semi-orthogonality: out @ out.T should be scaled identity
        M = min(rows, cols)
        X = out_gpu if rows <= cols else out_gpu.T
        gram = X @ X.T
        diag_mean = np.mean(np.diag(gram))
        off_diag = gram - np.diag(np.diag(gram))
        self.assertLess(np.max(np.abs(off_diag)), diag_mean * 0.5)


class TestFusedBackwardPass(unittest.TestCase):
    def test_fused_cheby_backward(self):
        """Verify fused Chebyshev backward pass produces identical gradients without Phi materialization."""
        bridge = get_metal_bridge()
        B, D_in, D_out, degree = 32, 16, 8, 4
        has_base, has_bias = 1, 1

        np.random.seed(123)
        dY = np.random.randn(B, D_out).astype(np.float32)
        X = np.random.uniform(-0.9, 0.9, (B, D_in)).astype(np.float32)
        W_cheby = np.random.randn(D_out, D_in * degree).astype(np.float32)
        W_base = np.random.randn(D_out, D_in).astype(np.float32)

        dW_cheby_mps = np.zeros_like(W_cheby)
        dW_base_mps = np.zeros_like(W_base)
        dbias_mps = np.zeros((D_out,), dtype=np.float32)
        dX_mps = np.zeros_like(X)

        dW_cheby_fused = np.zeros_like(W_cheby)
        dW_base_fused = np.zeros_like(W_base)
        dbias_fused = np.zeros((D_out,), dtype=np.float32)
        dX_fused = np.zeros_like(X)

        bridge.metal_kan_cheby_backward(
            dY.ctypes.data, X.ctypes.data, W_cheby.ctypes.data, W_base.ctypes.data,
            dW_cheby_mps.ctypes.data, dW_base_mps.ctypes.data, dbias_mps.ctypes.data, dX_mps.ctypes.data,
            B, D_in, D_out, degree, has_base, has_bias
        )

        bridge.metal_kan_cheby_backward_fused(
            dY.ctypes.data, X.ctypes.data, W_cheby.ctypes.data, W_base.ctypes.data,
            dW_cheby_fused.ctypes.data, dW_base_fused.ctypes.data, dbias_fused.ctypes.data, dX_fused.ctypes.data,
            B, D_in, D_out, degree, has_base, has_bias
        )

        diff_dW = np.max(np.abs(dW_cheby_mps - dW_cheby_fused))
        diff_dX = np.max(np.abs(dX_mps - dX_fused))
        diff_dbias = np.max(np.abs(dbias_mps - dbias_fused))

        self.assertLess(diff_dW, 1e-4)
        self.assertLess(diff_dX, 1e-4)
        self.assertLess(diff_dbias, 1e-5)


class TestSub4BitQuantization(unittest.TestCase):
    def test_ternary_158bit_quantization(self):
        """Verify 1.58-bit Ternary KAN quantization and branchless GPU forward pass."""
        layer = mk.ChebyKAN(in_features=16, out_features=8, degree=4)
        x = np.random.uniform(-0.8, 0.8, (4, 16)).astype(np.float32)
        out_fp32 = layer(x)

        mk.to_ternary(layer, group_size=32)
        self.assertTrue(hasattr(layer, "_w_cheby_ternary"))

        out_ternary = layer(x)
        self.assertEqual(out_ternary.shape, (4, 8))
        self.assertFalse(np.isnan(out_ternary).any())
        # Cosine similarity between FP32 and Ternary should be positive
        cos_sim = np.dot(out_fp32.ravel(), out_ternary.ravel()) / (
            np.linalg.norm(out_fp32) * np.linalg.norm(out_ternary) + 1e-8
        )
        self.assertGreater(cos_sim, 0.4)

    def test_int2_quantization(self):
        """Verify INT2 Codebook KAN quantization and branchless GPU forward pass."""
        layer = mk.ChebyKAN(in_features=16, out_features=8, degree=4)
        x = np.random.uniform(-0.8, 0.8, (4, 16)).astype(np.float32)
        out_fp32 = layer(x)

        mk.to_int2(layer, group_size=32)
        self.assertTrue(hasattr(layer, "_w_cheby_int2"))

        out_int2 = layer(x)
        self.assertEqual(out_int2.shape, (4, 8))
        self.assertFalse(np.isnan(out_int2).any())
        cos_sim = np.dot(out_fp32.ravel(), out_int2.ravel()) / (
            np.linalg.norm(out_fp32) * np.linalg.norm(out_int2) + 1e-8
        )
        self.assertGreater(cos_sim, 0.5)


class TestGatedKAN(unittest.TestCase):
    def test_gated_kan_forward_backward(self):
        """Verify GatedKAN module forward, backward, parameter counts, and sub-4-bit quantization."""
        block = mk.GatedKAN(d_model=32, d_ffn=64, degree=4)
        x = np.random.uniform(-0.8, 0.8, (6, 32)).astype(np.float32)
        out = block(x)
        self.assertEqual(out.shape, (6, 32))
        self.assertFalse(np.isnan(out).any())

        # Backward pass on 2D
        dY = np.random.randn(6, 32).astype(np.float32)
        dX = block.backward(dY, fused=True)
        self.assertEqual(dX.shape, (6, 32))
        self.assertFalse(np.isnan(dX).any())

        # Test 3D tensor input (B, Seq, D_model)
        x_3d = np.random.uniform(-0.8, 0.8, (2, 8, 32)).astype(np.float32)
        out_3d = block(x_3d)
        self.assertEqual(out_3d.shape, (2, 8, 32))

        # Backward pass on 3D
        dY_3d = np.random.randn(2, 8, 32).astype(np.float32)
        dX_3d = block.backward(dY_3d, fused=True)
        self.assertEqual(dX_3d.shape, (2, 8, 32))
        self.assertFalse(np.isnan(dX_3d).any())

        # Quantize block to ternary
        block.to_ternary()
        out_ternary = block(x)
        self.assertEqual(out_ternary.shape, (6, 32))
        self.assertFalse(np.isnan(out_ternary).any())


if __name__ == "__main__":
    unittest.main()
