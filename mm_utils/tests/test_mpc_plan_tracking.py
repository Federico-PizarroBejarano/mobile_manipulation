"""Interpolation commands the first-interval velocity, not the current-state knot.

v_bar[0] is the OCP initial condition (measured velocity). The command that
implements u_bar[0] is v_bar[1]. With ctrl_rate faster than 1/dt, t_elapsed
never reaches dt before the next solve, so sampling v(t) would never leave
the neighbourhood of v_bar[0].
"""

import unittest

import numpy as np

from mm_utils.math import compute_velocity_command_interpolation
from mm_utils.mpc_plan_tracking import build_plan_interpolators, low_level_velocity_step


def _interp_step(v_bar, t_elapsed, mpc_dt=0.1):
    n_v, nu = v_bar.shape
    u_bar = np.zeros((n_v - 1, nu))
    interps = build_plan_interpolators(mpc_dt, u_bar, v_bar, None, "interpolation")
    return low_level_velocity_step(
        np.zeros(nu),
        t_elapsed,
        0.01,
        interps,
        np.zeros(nu),
        kp=None,
        lb_u=None,
        ub_u=None,
    )


class TestInterpolationKnot(unittest.TestCase):
    def setUp(self):
        self.mpc_dt = 0.1
        self.v_bar = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        )

    def test_plan_arrival_uses_first_interval_knot(self):
        cmd = _interp_step(self.v_bar, 0.0, self.mpc_dt)
        np.testing.assert_allclose(cmd, [2.0, 0.0])

    def test_mid_interval_blends_first_and_second_interval(self):
        cmd = _interp_step(self.v_bar, 0.5 * self.mpc_dt, self.mpc_dt)
        np.testing.assert_allclose(cmd, [2.5, 0.0])

    def test_math_helper_matches_first_interval_knot(self):
        cmd = compute_velocity_command_interpolation(self.v_bar, 0.0, self.mpc_dt)
        np.testing.assert_allclose(cmd, [2.0, 0.0])

    def test_p_term_uses_same_knot_as_velocity(self):
        """q_ref must be q(t+dt), matching v(t+dt). q(t) is the measured knot."""
        mpc_dt = 0.1
        v_bar = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        q_bar = np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]])
        u_bar = np.zeros((2, 2))
        interps = build_plan_interpolators(mpc_dt, u_bar, v_bar, q_bar, "interpolation")
        cmd, diag = low_level_velocity_step(
            np.zeros(2),
            0.0,
            0.01,
            interps,
            q_meas=np.zeros(2),
            kp=np.ones(2),
            lb_u=None,
            ub_u=None,
            return_diagnostics=True,
        )
        np.testing.assert_allclose(diag["q_ref"], [4.0, 0.0])
        np.testing.assert_allclose(diag["v_ff"], [1.0, 0.0])
        np.testing.assert_allclose(cmd, [5.0, 0.0])


if __name__ == "__main__":
    unittest.main()
