"""Interpolation commands the first-interval velocity, not the current-state knot.

v_bar[0] is the OCP initial condition (measured velocity). The command that
implements u_bar[0] is v_bar[1]. With ctrl_rate faster than 1/dt, t_elapsed
never reaches dt before the next solve, so sampling v(t) would never leave
the neighbourhood of v_bar[0].
"""

import unittest

import numpy as np

from mm_utils.math import compute_velocity_command_interpolation
from mm_utils.mpc_plan_tracking import (
    apply_replan_continuity,
    build_plan_interpolators,
    low_level_velocity_step,
)


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


class TestReplanContinuity(unittest.TestCase):
    def test_handoff_matches_previous_feedforward(self):
        mpc_dt = 0.1
        v_bar_old = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        v_bar_new = np.array([[0.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        u_bar = np.zeros((2, 2))
        old_interps = build_plan_interpolators(
            mpc_dt, u_bar, v_bar_old, None, "interpolation"
        )
        new_interps = build_plan_interpolators(
            mpc_dt, u_bar, v_bar_new, None, "interpolation"
        )
        t_handoff = 0.05
        before = _interp_step(v_bar_old, t_handoff, mpc_dt)
        apply_replan_continuity(new_interps, old_interps, t_handoff, blend_s=mpc_dt)
        after, diag = low_level_velocity_step(
            np.zeros(2),
            0.0,
            0.01,
            new_interps,
            np.zeros(2),
            kp=None,
            lb_u=None,
            ub_u=None,
            return_diagnostics=True,
        )
        np.testing.assert_allclose(diag["v_ff"], before, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(after, before, rtol=0.0, atol=1e-12)

    def test_crossfade_reaches_new_plan_after_blend(self):
        mpc_dt = 0.1
        v_bar_old = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        v_bar_new = np.array([[0.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        u_bar = np.zeros((2, 2))
        old_interps = build_plan_interpolators(
            mpc_dt, u_bar, v_bar_old, None, "interpolation"
        )
        new_interps = build_plan_interpolators(
            mpc_dt, u_bar, v_bar_new, None, "interpolation"
        )
        apply_replan_continuity(new_interps, old_interps, 0.05, blend_s=mpc_dt)
        _, diag_start = low_level_velocity_step(
            np.zeros(2),
            0.0,
            0.01,
            new_interps,
            np.zeros(2),
            kp=None,
            lb_u=None,
            ub_u=None,
            return_diagnostics=True,
        )
        _, diag_end = low_level_velocity_step(
            np.zeros(2),
            mpc_dt,
            0.01,
            new_interps,
            np.zeros(2),
            kp=None,
            lb_u=None,
            ub_u=None,
            return_diagnostics=True,
        )
        self.assertLess(diag_start["v_ff"][0], 5.0)
        np.testing.assert_allclose(diag_end["v_ff"], [6.0, 0.0])


class TestReplanHandoffTiming(unittest.TestCase):
    """Replan continuity at realistic sim vs hardware handoff times."""

    def setUp(self):
        self.mpc_dt = 0.1
        self.ctrl_period = 0.05  # ctrl_rate 20 Hz with dt 0.1
        self.replan_ff_blend_s = 0.3  # MPC.yaml default

    def _handoff_continuity(self, t_handoff, blend_s):
        v_bar_old = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        v_bar_new = np.array([[0.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        u_bar = np.zeros((2, 2))
        old_interps = build_plan_interpolators(
            self.mpc_dt, u_bar, v_bar_old, None, "interpolation"
        )
        new_interps = build_plan_interpolators(
            self.mpc_dt, u_bar, v_bar_new, None, "interpolation"
        )
        before = _interp_step(v_bar_old, t_handoff, self.mpc_dt)
        apply_replan_continuity(new_interps, old_interps, t_handoff, blend_s)
        after, diag = low_level_velocity_step(
            np.zeros(2),
            0.0,
            0.01,
            new_interps,
            np.zeros(2),
            kp=None,
            lb_u=None,
            ub_u=None,
            return_diagnostics=True,
        )
        return before, after, diag

    def test_sim_handoff_at_ctrl_period_is_continuous(self):
        """experiment.py: t_handoff = t - last_controller_time ≈ ctrl_period."""
        before, after, diag = self._handoff_continuity(
            self.ctrl_period, self.replan_ff_blend_s
        )
        np.testing.assert_allclose(diag["v_ff"], before, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(after, before, rtol=0.0, atol=1e-12)

    def test_hardware_stamp_handoff_matches_sim_timing(self):
        """low_level_cmd_node: t_handoff = new_stamp - old_stamp (same 0.05 s gap)."""
        before_sim, after_sim, _ = self._handoff_continuity(
            self.ctrl_period, self.replan_ff_blend_s
        )
        before_hw, after_hw, diag_hw = self._handoff_continuity(
            0.05, self.replan_ff_blend_s
        )
        np.testing.assert_allclose(before_hw, before_sim)
        np.testing.assert_allclose(after_hw, after_sim)
        np.testing.assert_allclose(diag_hw["v_ff"], before_hw, rtol=0.0, atol=1e-12)

    def test_replan_ff_blend_s_longer_than_mpc_dt_keeps_old_plan_mid_horizon(self):
        """replan_ff_blend_s=0.3 crossfades slowly; at t=0.15 still mostly old plan."""
        t_mid = 0.15
        v_bar_old = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        v_bar_new = np.array([[0.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        u_bar = np.zeros((2, 2))
        old_interps = build_plan_interpolators(
            self.mpc_dt, u_bar, v_bar_old, None, "interpolation"
        )
        new_interps = build_plan_interpolators(
            self.mpc_dt, u_bar, v_bar_new, None, "interpolation"
        )
        apply_replan_continuity(new_interps, old_interps, 0.05, self.replan_ff_blend_s)
        _, diag_long = low_level_velocity_step(
            np.zeros(2),
            t_mid,
            0.01,
            new_interps,
            np.zeros(2),
            kp=None,
            lb_u=None,
            ub_u=None,
            return_diagnostics=True,
        )
        new_interps_short = build_plan_interpolators(
            self.mpc_dt, u_bar, v_bar_new, None, "interpolation"
        )
        apply_replan_continuity(new_interps_short, old_interps, 0.05, self.mpc_dt)
        _, diag_short = low_level_velocity_step(
            np.zeros(2),
            t_mid,
            0.01,
            new_interps_short,
            np.zeros(2),
            kp=None,
            lb_u=None,
            ub_u=None,
            return_diagnostics=True,
        )
        self.assertLess(diag_long["v_ff"][0], diag_short["v_ff"][0])

    def test_nested_replan_during_crossfade_stays_continuous(self):
        """If a new plan arrives mid-blend, sample the blended old reference, not raw v_bar."""
        mpc_dt = 0.1
        u_bar = np.zeros((2, 2))
        interps_a = build_plan_interpolators(
            mpc_dt,
            u_bar,
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            None,
            "interpolation",
        )
        interps_b = build_plan_interpolators(
            mpc_dt,
            u_bar,
            np.array([[0.0, 0.0], [5.0, 0.0], [6.0, 0.0]]),
            None,
            "interpolation",
        )
        apply_replan_continuity(interps_b, interps_a, 0.05, blend_s=0.5)
        v_mid, _ = low_level_velocity_step(
            np.zeros(2),
            0.05,
            0.01,
            interps_b,
            np.zeros(2),
            kp=None,
            lb_u=None,
            ub_u=None,
            return_diagnostics=True,
        )
        interps_c = build_plan_interpolators(
            mpc_dt,
            u_bar,
            np.array([[0.0, 0.0], [8.0, 0.0], [9.0, 0.0]]),
            None,
            "interpolation",
        )
        apply_replan_continuity(interps_c, interps_b, 0.05, blend_s=0.5)
        v_after, diag = low_level_velocity_step(
            np.zeros(2),
            0.0,
            0.01,
            interps_c,
            np.zeros(2),
            kp=None,
            lb_u=None,
            ub_u=None,
            return_diagnostics=True,
        )
        np.testing.assert_allclose(diag["v_ff"], v_mid, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(v_after, v_mid, rtol=0.0, atol=1e-12)

    def test_replan_chain_depth_stays_bounded(self):
        """Frequent replans must not retain an unbounded prev_interps chain."""
        mpc_dt = 0.05
        blend_s = 0.4
        handoff = 0.05
        u_bar = np.zeros((2, 2))
        v_bar = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        interps = build_plan_interpolators(mpc_dt, u_bar, v_bar, None, "interpolation")
        for i in range(100):
            nxt = build_plan_interpolators(
                mpc_dt,
                u_bar,
                np.array([[0.0, 0.0], [float(i + 2), 0.0], [float(i + 3), 0.0]]),
                None,
                "interpolation",
            )
            apply_replan_continuity(nxt, interps, handoff, blend_s=blend_s)
            interps = nxt

        depth = 0
        node = interps
        while node is not None and node.prev_interps is not None:
            depth += 1
            node = node.prev_interps
            self.assertLessEqual(depth, 16, "replan chain grew without bound")

        # Steady-state depth is about blend_s / handoff (~8), not 100.
        self.assertLessEqual(depth, int(blend_s / handoff) + 2)


if __name__ == "__main__":
    unittest.main()
