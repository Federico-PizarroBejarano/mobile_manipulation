"""Tests for pose-path EEPlanner."""

import numpy as np

from mm_rl.env.ee_planner import MIN_PATH_STEPS, EEPlanner
from mm_utils.math import quat_slerp


def test_slerp_endpoints():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(quat_slerp(q0, q1, 0.0), q0, atol=1e-6)
    np.testing.assert_allclose(np.abs(quat_slerp(q0, q1, 1.0)), np.abs(q1), atol=1e-6)


def test_pure_orientation_min_steps():
    goal_orn = np.array([0.70710678, 0.0, 0.0, 0.70710678])
    start_orn = np.array([0.0, 0.0, 0.0, 1.0])
    p = EEPlanner(
        np.zeros(3),
        goal_orn,
        np.zeros(3),
        start_orn,
        max_linear_speed=0.1,
        dt=0.03,
    )
    assert p.n_path_steps == MIN_PATH_STEPS
    for _ in range(MIN_PATH_STEPS):
        p.step()
    _, do = p.get_desired_pose()
    assert 1.0 - abs(np.dot(do, goal_orn)) ** 2 < 1e-3


def test_translation_respects_max_linear_speed():
    start_orn = np.array([0.0, 0.0, 0.0, 1.0])
    p = EEPlanner(
        np.array([2.0, 0.0, 0.0]),
        start_orn,
        np.zeros(3),
        start_orn,
        max_linear_speed=0.1,
        dt=0.03,
    )
    dist = 2.0
    mean_speed = dist / (p.n_path_steps * p.dt)
    assert mean_speed <= 0.1 + 1e-6


def test_plateau_at_goal():
    start_orn = np.array([0.0, 0.0, 0.0, 1.0])
    p = EEPlanner(
        np.array([0.03, 0.0, 0.0]),
        start_orn,
        np.zeros(3),
        start_orn,
        max_linear_speed=0.1,
        dt=0.03,
    )
    for _ in range(p.n_path_steps + 5):
        vl, va = p.step()
        if p.s >= 1.0:
            assert np.linalg.norm(vl) < 1e-9
            assert np.linalg.norm(va) < 1e-9
    dp, _ = p.get_desired_pose()
    np.testing.assert_allclose(dp, p._goal_pos, atol=1e-5)


if __name__ == "__main__":
    test_slerp_endpoints()
    test_pure_orientation_min_steps()
    test_translation_respects_max_linear_speed()
    test_plateau_at_goal()
    print("all ok")
