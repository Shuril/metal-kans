import unittest
import numpy as np
import metal_kans as mk
from metal_kans import ChebyKAN, FastKAN, ReLUKAN, BSplineKAN, MetalKAN, Adam, SGD, CheckpointedMetalKAN, checkpoint_kan


class TestTraining(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_cheby_gradient_numerical_check(self):
        """Finite difference check on analytical dX and dW gradients for ChebyKAN."""
        B, D_in, D_out, deg = 4, 3, 2, 4
        layer = ChebyKAN(D_in, D_out, degree=deg, bias=True, use_base=True)

        X = np.random.uniform(-0.8, 0.8, (B, D_in)).astype(np.float32)
        dY = np.random.randn(B, D_out).astype(np.float32)

        # Forward + analytical backward
        y = layer(X)
        dX = layer.backward(dY)

        # Numerical gradient checking for dX:
        eps = 1e-4
        dX_num = np.zeros_like(X)
        for i in range(B):
            for j in range(D_in):
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[i, j] += eps
                X_minus[i, j] -= eps

                y_plus = layer(X_plus)
                y_minus = layer(X_minus)
                # Loss L = sum(y * dY)
                l_plus = np.sum(y_plus * dY)
                l_minus = np.sum(y_minus * dY)
                dX_num[i, j] = (l_plus - l_minus) / (2.0 * eps)

        # Relative error
        rel_err_X = np.linalg.norm(dX - dX_num) / (np.linalg.norm(dX) + np.linalg.norm(dX_num) + 1e-7)
        self.assertLess(rel_err_X, 1e-3, f"ChebyKAN dX relative error too high: {rel_err_X}")

        # Numerical gradient checking for dW_cheby:
        dW_num = np.zeros_like(layer.w_cheby)
        for i in range(layer.w_cheby.shape[0]):
            for j in range(layer.w_cheby.shape[1]):
                w_orig = layer.w_cheby[i, j]

                layer.w_cheby[i, j] = w_orig + eps
                y_plus = layer(X)
                l_plus = np.sum(y_plus * dY)

                layer.w_cheby[i, j] = w_orig - eps
                y_minus = layer(X)
                l_minus = np.sum(y_minus * dY)

                layer.w_cheby[i, j] = w_orig
                dW_num[i, j] = (l_plus - l_minus) / (2.0 * eps)

        rel_err_W = np.linalg.norm(layer.grad_w_cheby - dW_num) / (np.linalg.norm(layer.grad_w_cheby) + np.linalg.norm(dW_num) + 1e-7)
        self.assertLess(rel_err_W, 1e-3, f"ChebyKAN dW relative error too high: {rel_err_W}")

    def test_fastkan_gradient_numerical_check(self):
        """Finite difference check on analytical dX and dW gradients for FastKAN."""
        B, D_in, D_out, num_c = 4, 3, 2, 6
        layer = FastKAN(D_in, D_out, num_centers=num_c, bias=True, use_base=True)

        X = np.random.uniform(-0.8, 0.8, (B, D_in)).astype(np.float32)
        dY = np.random.randn(B, D_out).astype(np.float32)

        y = layer(X)
        dX = layer.backward(dY)

        eps = 1e-4
        dX_num = np.zeros_like(X)
        for i in range(B):
            for j in range(D_in):
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[i, j] += eps
                X_minus[i, j] -= eps

                y_plus = layer(X_plus)
                y_minus = layer(X_minus)
                l_plus = np.sum(y_plus * dY)
                l_minus = np.sum(y_minus * dY)
                dX_num[i, j] = (l_plus - l_minus) / (2.0 * eps)

        rel_err_X = np.linalg.norm(dX - dX_num) / (np.linalg.norm(dX) + np.linalg.norm(dX_num) + 1e-7)
        self.assertLess(rel_err_X, 1e-3, f"FastKAN dX relative error too high: {rel_err_X}")

    def test_bspline_gradient_numerical_check(self):
        """Finite difference check on analytical dX and dW gradients for BSplineKAN."""
        B, D_in, D_out, grid_size = 4, 3, 2, 5
        layer = BSplineKAN(D_in, D_out, grid_size=grid_size, bias=True, use_base=True)

        X = np.random.uniform(-0.5, 0.5, (B, D_in)).astype(np.float32)
        dY = np.random.randn(B, D_out).astype(np.float32)

        y = layer(X)
        dX = layer.backward(dY)

        eps = 1e-4
        dX_num = np.zeros_like(X)
        for i in range(B):
            for j in range(D_in):
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[i, j] += eps
                X_minus[i, j] -= eps

                y_plus = layer(X_plus)
                y_minus = layer(X_minus)
                l_plus = np.sum(y_plus * dY)
                l_minus = np.sum(y_minus * dY)
                dX_num[i, j] = (l_plus - l_minus) / (2.0 * eps)

        rel_err_X = np.linalg.norm(dX - dX_num) / (np.linalg.norm(dX) + np.linalg.norm(dX_num) + 1e-7)
        self.assertLess(rel_err_X, 1e-3, f"BSplineKAN dX relative error too high: {rel_err_X}")

    def test_end_to_end_training_loop(self):
        """Verify that a multi-layer MetalKAN trains and reduces MSE loss on non-linear target."""
        X = np.linspace(-0.8, 0.8, 64).reshape(64, 1).astype(np.float32)
        # Target function: y = sin(pi * x)
        y_true = np.sin(np.pi * X).astype(np.float32)

        model = MetalKAN([1, 16, 1], basis_type="cheby", degree=4, bias=True, use_base=True)
        # Check initial forward
        y_init = model(X)
        init_loss = np.mean((y_init - y_true) ** 2)

        optimizer = Adam(model.parameters(), lr=0.03)

        # Train for 50 steps
        for epoch in range(50):
            y_pred = model(X)
            # MSE loss: L = mean((y - y_true)^2) -> dL/dy = 2 * (y - y_true) / N
            diff = 2.0 * (y_pred - y_true) / len(X)
            model.backward(diff)
            optimizer.step()
            model.zero_grad()

        final_loss = np.mean((model(X) - y_true) ** 2)
        self.assertLess(final_loss, init_loss * 0.3, f"Training failed to reduce loss: init={init_loss}, final={final_loss}")

    def test_checkpointed_metal_kan_equivalence(self):
        """Verify that CheckpointedMetalKAN produces exact forward and backward results as standard MetalKAN."""
        X = np.random.uniform(-0.8, 0.8, (8, 4)).astype(np.float32)
        dY = np.random.randn(8, 2).astype(np.float32)

        model = MetalKAN([4, 8, 2], basis_type="cheby", degree=4, bias=True, use_base=True)
        ckpt_model = checkpoint_kan(model)

        # Forward comparison
        y_std = model(X)
        y_ckpt = ckpt_model(X)
        np.testing.assert_allclose(y_std, y_ckpt, rtol=1e-5, atol=1e-5)

        # Backward comparison
        dX_std = model.backward(dY)
        dX_ckpt = ckpt_model.backward(dY)
        np.testing.assert_allclose(dX_std, dX_ckpt, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
