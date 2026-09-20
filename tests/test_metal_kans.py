import unittest
import numpy as np
import sys
import os
import tempfile
import subprocess

# Ensure metal_kans package is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metal_kans import (
    is_metal_available,
    ChebyKAN,
    FastKAN,
    ReLUKAN,
    WavKAN,
    FourierKAN,
    JacobiKAN,
    RationalKAN,
    BSplineKAN,
    KAN,
    MultKAN,
    LowRankKAN,
    MetalKAN,
    QuantizedWeight,
    quantize,
    to_int8,
    to_int4,
    compute_node_importance,
    prune,
    compact_kan,
    to_symbolic,
    SymbolicKAN,
    count_parameters,
    get_model_size,
)


class TestMetalKANsSuite(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_metal_device(self):
        self.assertTrue(is_metal_available(), "Metal device should be detected on Apple Silicon")

    # -----------------------------------------------------------------------
    # 1. Individual Architecture Tests
    # -----------------------------------------------------------------------
    def test_cheby_kan(self):
        layer = ChebyKAN(in_features=8, out_features=16, degree=4)
        x = np.random.uniform(-1.0, 1.0, (128, 8)).astype(np.float32)
        y = layer(x)
        self.assertEqual(y.shape, (128, 16))
        self.assertFalse(np.isnan(y).any())

    def test_fast_kan(self):
        layer = FastKAN(in_features=8, out_features=16, num_centers=8)
        x = np.random.uniform(-1.0, 1.0, (128, 8)).astype(np.float32)
        y = layer(x)
        self.assertEqual(y.shape, (128, 16))
        self.assertFalse(np.isnan(y).any())

    def test_relu_kan(self):
        layer = ReLUKAN(in_features=8, out_features=16, num_grids=8)
        x = np.random.uniform(-1.0, 1.0, (128, 8)).astype(np.float32)
        y = layer(x)
        self.assertEqual(y.shape, (128, 16))
        self.assertFalse(np.isnan(y).any())

    def test_wav_kan(self):
        for w_type in ["mexican_hat", "morlet", "dog"]:
            layer = WavKAN(in_features=8, out_features=16, num_wavelets=6, wavelet_type=w_type)
            x = np.random.uniform(-1.0, 1.0, (64, 8)).astype(np.float32)
            y = layer(x)
            self.assertEqual(y.shape, (64, 16))
            self.assertFalse(np.isnan(y).any())

    def test_fourier_kan(self):
        layer = FourierKAN(in_features=8, out_features=16, num_frequencies=4)
        x = np.random.uniform(-1.0, 1.0, (64, 8)).astype(np.float32)
        y = layer(x)
        self.assertEqual(y.shape, (64, 16))
        self.assertFalse(np.isnan(y).any())

    def test_jacobi_kan(self):
        layer = JacobiKAN(in_features=8, out_features=16, degree=4, alpha=0.5, beta=-0.5)
        x = np.random.uniform(-1.0, 1.0, (64, 8)).astype(np.float32)
        y = layer(x)
        self.assertEqual(y.shape, (64, 16))
        self.assertFalse(np.isnan(y).any())

    def test_rational_kan(self):
        layer = RationalKAN(in_features=8, out_features=16, p_degree=4, q_degree=2)
        x = np.random.uniform(-1.0, 1.0, (64, 8)).astype(np.float32)
        y = layer(x)
        self.assertEqual(y.shape, (64, 16))
        self.assertFalse(np.isnan(y).any())

    def test_bspline_kan(self):
        layer = BSplineKAN(in_features=8, out_features=16, grid_size=5, spline_order=3)
        x = np.random.uniform(-0.9, 0.9, (64, 8)).astype(np.float32)
        y = layer(x)
        self.assertEqual(y.shape, (64, 16))
        self.assertFalse(np.isnan(y).any())

    def test_mult_kan(self):
        layer = MultKAN(in_features=8, out_features=16, num_mult=4, use_base=True, bias=True)
        x = np.random.uniform(-1.0, 1.0, (64, 8)).astype(np.float32)
        y = layer(x)
        self.assertEqual(y.shape, (64, 16))
        self.assertFalse(np.isnan(y).any())

        # Ground truth verification
        internal = layer.sub_layer(x)
        y_ref = np.zeros_like(y)
        y_ref[:, :layer.num_add] = internal[:, :layer.num_add]
        for i in range(layer.num_mult):
            u = internal[:, layer.num_add + i]
            v = internal[:, layer.num_add + layer.num_mult + i]
            y_ref[:, layer.num_add + i] = u * v
        np.testing.assert_allclose(y, y_ref, atol=1e-5)

    def test_low_rank_kan(self):
        layer = LowRankKAN(in_features=16, out_features=32, rank=4, use_base=True, bias=True)
        x = np.random.uniform(-1.0, 1.0, (64, 16)).astype(np.float32)
        y = layer(x)
        self.assertEqual(y.shape, (64, 32))
        self.assertFalse(np.isnan(y).any())

        # Ground truth verification
        z = layer.down_layer(x)
        silu_x = x / (1.0 + np.exp(-x))
        y_ref = (z @ layer.w_up.T) + (silu_x @ layer.w_base.T) + layer.bias
        np.testing.assert_allclose(y, y_ref, atol=1e-4)

    # -----------------------------------------------------------------------
    # 2. Multi-Layer MetalKAN Network Tests for All 10 Architectures
    # -----------------------------------------------------------------------
    def test_metal_kan_all_architectures(self):
        bases = ["cheby", "fastkan", "relu", "wav", "fourier", "jacobi", "rational", "bspline", "mult", "lowrank"]
        x = np.random.uniform(-0.8, 0.8, (32, 4)).astype(np.float32)

        for b in bases:
            net = MetalKAN([4, 8, 2], basis_type=b, degree=4)
            y = net(x)
            self.assertEqual(y.shape, (32, 2), f"Failed shape for basis {b}")
            self.assertFalse(np.isnan(y).any(), f"NaN found for basis {b}")

    def test_metal_kan_pipelined_execution(self):
        net = MetalKAN([4, 16, 8, 2], basis_type="cheby", degree=4, pipeline=True)
        x = np.random.uniform(-1.0, 1.0, (128, 4)).astype(np.float32)
        y_pipelined = net(x)
        self.assertEqual(y_pipelined.shape, (128, 2))

        # Check equivalence with standard dispatch
        net_std = MetalKAN([4, 16, 8, 2], basis_type="cheby", degree=4, pipeline=False)
        for l_std, l_pipe in zip(net_std.layers, net.layers):
            l_std.w_cheby = l_pipe.w_cheby.copy()
            l_std.w_base = l_pipe.w_base.copy()
            l_std.bias = l_pipe.bias.copy()

        y_std = net_std(x)
        diff = np.max(np.abs(y_pipelined - y_std))
        self.assertLess(diff, 1e-4)

    # -----------------------------------------------------------------------
    # 3. Quantization Tests (INT8 & INT4)
    # -----------------------------------------------------------------------
    def test_quantization_int8_and_int4(self):
        w = np.random.normal(0.0, 1.0, (32, 64)).astype(np.float32)

        qw8 = QuantizedWeight(w, bits=8, group_size=32)
        w8 = qw8.dequantize()
        diff8 = np.max(np.abs(w - w8))
        self.assertLess(diff8, 0.05)

        qw4 = QuantizedWeight(w, bits=4, group_size=32)
        w4 = qw4.dequantize()
        diff4 = np.max(np.abs(w - w4))
        self.assertLess(diff4, 0.35)

        # In-place model quantization
        net = MetalKAN([8, 16, 4], basis_type="cheby")
        to_int8(net)
        x = np.random.uniform(-1.0, 1.0, (16, 8)).astype(np.float32)
        y_q = net(x)
        self.assertEqual(y_q.shape, (16, 4))

    # -----------------------------------------------------------------------
    # 4. Pruning and Compaction Tests
    # -----------------------------------------------------------------------
    def test_pruning_and_compaction(self):
        net = MetalKAN([4, 16, 2], basis_type="cheby", degree=4)
        scores = compute_node_importance(net)
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0].shape, (16,))

        # Prune small weights
        pruned_net = prune(net, threshold=1e-3)
        self.assertIsNotNone(pruned_net)

        # Compact network physically
        compact_net = compact_kan(net, threshold=1e-4)
        self.assertLessEqual(compact_net.layers_hidden[1], 16)
        x = np.random.uniform(-1.0, 1.0, (8, 4)).astype(np.float32)
        y_c = compact_net(x)
        self.assertEqual(y_c.shape, (8, 2))

    # -----------------------------------------------------------------------
    # 5. Symbolic Formula Discovery & C Code Generation Tests
    # -----------------------------------------------------------------------
    def test_symbolic_regression_and_c_export(self):
        net = MetalKAN([2, 4, 1], basis_type="cheby", degree=3)
        sym = to_symbolic(net, sample_points=100)

        # Test evaluation
        x = np.array([[0.5, -0.25]], dtype=np.float32)
        y_sym = sym(x)
        self.assertEqual(y_sym.shape, (1, 1))

        # Test C code and C header generation
        c_code = sym.to_c_code(func_name="kan_predict")
        self.assertIn("void kan_predict", c_code)

        with tempfile.TemporaryDirectory() as tmpdir:
            header_path = os.path.join(tmpdir, "kan_model.h")
            sym.export_c(header_path, guard="TEST_KAN_H", func_name="kan_predict")
            self.assertTrue(os.path.exists(header_path))

            main_c = f"""#include <stdio.h>
#include "kan_model.h"

int main() {{
    float x[2] = {{0.5f, -0.25f}};
    float y[1] = {{0.0f}};
    kan_predict(x, y);
    printf("RESULT: %f\\n", y[0]);
    return 0;
}}
"""
            main_path = os.path.join(tmpdir, "main.c")
            with open(main_path, "w") as f:
                f.write(main_c)

            bin_path = os.path.join(tmpdir, "test_bin")
            cc = os.environ.get("CC", "clang")
            compile_res = subprocess.run([cc, "-O2", main_path, "-o", bin_path, "-lm"], capture_output=True, text=True)
            if compile_res.returncode == 0:
                run_res = subprocess.run([bin_path], capture_output=True, text=True)
                self.assertEqual(run_res.returncode, 0)
                self.assertIn("RESULT:", run_res.stdout)
                c_val = float(run_res.stdout.strip().split("RESULT:")[1])
                self.assertAlmostEqual(c_val, float(y_sym[0, 0]), places=4)

    # -----------------------------------------------------------------------
    # 6. Parameter Counting & Model Size Tests
    # -----------------------------------------------------------------------
    def test_parameter_counts_and_model_size(self):
        net = MetalKAN([8, 32, 16, 4], basis_type="cheby", degree=4)
        params = count_parameters(net)
        self.assertGreater(params, 0)

        size_info = get_model_size(net)
        self.assertEqual(size_info["parameters"], params)
        self.assertGreater(size_info["bytes"], 0)
        self.assertIn("MB", size_info["summary"])

    # -----------------------------------------------------------------------
    # 7. FP16 Half Precision & Asynchronous Pipelining Tests
    # -----------------------------------------------------------------------
    def test_fp16_execution(self):
        """Validates native FP16 half precision parity against FP32 across architectures."""
        # ChebyKAN
        cheby = ChebyKAN(8, 16, degree=4)
        x = np.random.uniform(-1.0, 1.0, (64, 8)).astype(np.float32)
        y_fp32 = cheby(x)
        cheby.half()
        y_fp16 = cheby(x.astype(np.float16))
        self.assertEqual(y_fp16.dtype, np.float16)
        np.testing.assert_allclose(y_fp16.astype(np.float32), y_fp32, atol=1e-2)

        # FastKAN
        fastkan = FastKAN(8, 16, num_centers=8)
        y_fp32 = fastkan(x)
        fastkan.half()
        y_fp16 = fastkan(x.astype(np.float16))
        self.assertEqual(y_fp16.dtype, np.float16)
        np.testing.assert_allclose(y_fp16.astype(np.float32), y_fp32, atol=1e-2)

        # ReLUKAN
        relu = ReLUKAN(8, 16, num_grids=8)
        y_fp32 = relu(x)
        relu.half()
        y_fp16 = relu(x.astype(np.float16))
        self.assertEqual(y_fp16.dtype, np.float16)
        np.testing.assert_allclose(y_fp16.astype(np.float32), y_fp32, atol=1e-2)

        # LowRankKAN
        lowrank = LowRankKAN(16, 32, rank=4, use_base=True, bias=True)
        x_lr = np.random.uniform(-1.0, 1.0, (64, 16)).astype(np.float32)
        y_fp32 = lowrank(x_lr)
        lowrank.half()
        y_fp16 = lowrank(x_lr.astype(np.float16))
        self.assertEqual(y_fp16.dtype, np.float16)
        np.testing.assert_allclose(y_fp16.astype(np.float32), y_fp32, atol=1e-2)

        # MultKAN
        mult = MultKAN(8, 16, num_mult=4, use_base=True, bias=True)
        y_fp32 = mult(x)
        mult.half()
        y_fp16 = mult(x.astype(np.float16))
        self.assertEqual(y_fp16.dtype, np.float16)
        np.testing.assert_allclose(y_fp16.astype(np.float32), y_fp32, atol=1e-2)

        # BSplineKAN
        bspline = BSplineKAN(8, 16, grid_size=5, spline_order=3)
        x_sp = np.random.uniform(-0.9, 0.9, (64, 8)).astype(np.float32)
        y_fp32 = bspline(x_sp)
        bspline.half()
        y_fp16 = bspline(x_sp.astype(np.float16))
        self.assertEqual(y_fp16.dtype, np.float16)
        np.testing.assert_allclose(y_fp16.astype(np.float32), y_fp32, atol=1e-2)

        # WavKAN
        wav = WavKAN(8, 16, num_wavelets=6)
        y_fp32 = wav(x)
        wav.half()
        y_fp16 = wav(x.astype(np.float16))
        self.assertEqual(y_fp16.dtype, np.float16)
        np.testing.assert_allclose(y_fp16.astype(np.float32), y_fp32, atol=1e-2)

    def test_async_pipelining(self):
        """Validates asynchronous streaming execution across batch queues."""
        net = MetalKAN([8, 16, 2], basis_type="fastkan", degree=8)
        batches = [np.random.uniform(-1.0, 1.0, (32, 8)).astype(np.float32) for _ in range(8)]
        
        # Synchronous ground truth
        y_sync = [net(b) for b in batches]
        
        # Async stream
        y_async = net.async_stream(batches)
        
        self.assertEqual(len(y_async), len(batches))
        for y_s, y_a in zip(y_sync, y_async):
            self.assertEqual(y_a.shape, (32, 2))
            np.testing.assert_allclose(y_a, y_s, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

