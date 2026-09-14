"""Tests for soft-collision RBF: ~zero outside the influence band."""

import numpy as np

from mm_control.MPCCostFunctions import RBF


class TestRBFSoftCost:
    def test_near_zero_outside_band(self):
        mu, zeta = 0.05, 0.04
        # Smooth-max leaves a small tail; require it negligible far out.
        for h in [zeta + 0.05, 0.1, 0.5, 1.5, 3.0]:
            B = float(RBF.B_fcn(h, mu, zeta))
            g = float(RBF.B_grad_fcn(h, mu, zeta))
            H = float(RBF.B_hess_fcn(h, mu, zeta))
            assert np.isfinite(B) and np.isfinite(g) and np.isfinite(H)
            assert B < 1e-4, f"B({h})={B}"
            assert abs(g) < 1e-3, f"grad({h})={g}"

    def test_positive_inside_band(self):
        mu, zeta = 0.05, 0.04
        for h in [-0.05, 0.0, 0.5 * zeta]:
            B = float(RBF.B_fcn(h, mu, zeta))
            assert B > 0.0, f"B({h})={B}"
            assert np.isfinite(float(RBF.B_grad_fcn(h, mu, zeta)))
            assert np.isfinite(float(RBF.B_hess_fcn(h, mu, zeta)))

    def test_small_at_boundary(self):
        mu, zeta = 0.05, 0.04
        B = float(RBF.B_fcn(zeta, mu, zeta))
        g = float(RBF.B_grad_fcn(zeta, mu, zeta))
        # smooth-max at 0 → gap = eps/2
        assert B < 1e-3
        assert abs(g) < 0.5

    def test_mu_scales_interior(self):
        zeta = 0.04
        h = 0.0
        b_small = float(RBF.B_fcn(h, 1e-4, zeta))
        b_large = float(RBF.B_fcn(h, 0.05, zeta))
        np.testing.assert_allclose(b_large / b_small, 0.05 / 1e-4, rtol=1e-6)
        assert float(RBF.B_fcn(1.5, 1e-4, zeta)) < 1e-8
        assert float(RBF.B_fcn(1.5, 0.05, zeta)) < 1e-8

    def test_interior_near_quadratic_hinge(self):
        mu, zeta = 0.05, 0.04
        # Deep inside the band, smooth-max ≈ hinge.
        h = -0.05
        expected = 0.5 * mu * ((zeta - h) / zeta) ** 2
        np.testing.assert_allclose(float(RBF.B_fcn(h, mu, zeta)), expected, rtol=1e-3)
        expected_g = -mu * (zeta - h) / zeta**2
        np.testing.assert_allclose(
            float(RBF.B_grad_fcn(h, mu, zeta)), expected_g, rtol=1e-3
        )
        np.testing.assert_allclose(
            float(RBF.B_hess_fcn(h, mu, zeta)), mu / zeta**2, rtol=5e-3
        )

    def test_no_nan_deep_penetration(self):
        mu, zeta = 0.05, 0.04
        for h in [-10.0, -1.0, -0.5]:
            B = float(RBF.B_fcn(h, mu, zeta))
            g = float(RBF.B_grad_fcn(h, mu, zeta))
            H = float(RBF.B_hess_fcn(h, mu, zeta))
            assert np.isfinite(B) and np.isfinite(g) and np.isfinite(H)
            assert B > 0.0
