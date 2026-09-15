"""Tests for MoMa-style EE motion inference."""

import numpy as np

from mm_utils.ee_motion_infer import (
    DEFAULT_HORIZON_M,
    DEFAULT_RESOLUTION_M,
    integrate_ee_motion,
    signal_from_ee_twist,
)


def test_idle_zeros():
    ee_pos = np.array([0.1, 0.2, 0.8])
    ee_orn = np.array([0.0, 0.0, 0.0, 1.0])
    out = integrate_ee_motion(ee_pos, ee_orn, np.zeros(6), dt=0.05)
    assert out["active"] is False
    assert np.allclose(out["v_ee"], 0.0)
    assert np.allclose(out["hat_pos"], ee_pos)
    assert np.allclose(out["goal_pos"], ee_pos)


def test_forward_translation_horizon():
    ee_pos = np.zeros(3)
    ee_orn = np.array([0.0, 0.0, 0.0, 1.0])
    # +x linear command
    cmd = np.array([0.12, 0.0, 0.0, 0.0, 0.0, 0.0])
    out = integrate_ee_motion(ee_pos, ee_orn, cmd, dt=0.05)
    assert out["active"] is True
    assert out["n_steps"] >= int(np.ceil(DEFAULT_HORIZON_M / DEFAULT_RESOLUTION_M))
    # Goal ~1.5 m along +x
    assert abs(out["goal_pos"][0] - DEFAULT_HORIZON_M) < 1e-6
    assert np.allclose(out["goal_pos"][1:], 0.0, atol=1e-9)
    # Next pose one resolution step ahead
    assert abs(out["hat_pos"][0] - DEFAULT_RESOLUTION_M) < 1e-6
    # v_ee points +x
    assert out["v_ee"][0] > 0.0


def test_signal_normalizes_translation():
    v, q, active = signal_from_ee_twist(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert active
    assert abs(np.linalg.norm(v) - DEFAULT_RESOLUTION_M) < 1e-9
    assert abs(q[3] - 1.0) < 1e-9


if __name__ == "__main__":
    test_idle_zeros()
    test_forward_translation_horizon()
    test_signal_normalizes_translation()
    print("ee_motion_infer tests passed")
