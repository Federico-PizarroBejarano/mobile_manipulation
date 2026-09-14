"""Tests for MPSF goal-following velocity from calculate_desired_velocity."""

import numpy as np

from mm_run.scripts.mpsf_experiment import calculate_desired_velocity


class _MockRobot:
    lb_u = np.array([-1.0] * 9)
    ub_u = np.array([1.0] * 9)


class _MockController:
    robot = _MockRobot()


def _states(base_pose=None, ee_pose=None):
    return {
        "base": {"pose": np.zeros(3) if base_pose is None else np.asarray(base_pose)},
        "EE": {"pose": np.zeros(6) if ee_pose is None else np.asarray(ee_pose)},
    }


class TestCalculateDesiredVelocity:
    def test_far_from_goal_caps_at_configured_max(self):
        mpsf_params = {
            "goal_velocity": {
                "max_ee_vel": [0.15, 0.15, 0.15, 0.3, 0.3, 0.3],
                "ee_threshold": [0.2, 0.2, 0.2, 0.3, 0.3, 0.3],
            }
        }
        ee_goal = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, ee_vel = calculate_desired_velocity(
            None, ee_goal, _states(), _MockController(), mpsf_params
        )
        np.testing.assert_allclose(ee_vel[0], 0.15)
        np.testing.assert_allclose(ee_vel[1:], 0.0)

    def test_within_threshold_uses_proportional_error(self):
        ee_goal = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, ee_vel = calculate_desired_velocity(
            None, ee_goal, _states(), _MockController(), None
        )
        np.testing.assert_allclose(ee_vel[0], 0.1)

    def test_base_goal_none_returns_none_base_vel(self):
        ee_goal = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        base_vel, ee_vel = calculate_desired_velocity(
            None, ee_goal, _states(), _MockController(), None
        )
        assert base_vel is None
        assert ee_vel is not None

    def test_clips_to_robot_velocity_limits(self):
        mpsf_params = {
            "goal_velocity": {
                "max_base_vel": [0.9, 0.9, 0.9],
                "base_threshold": [0.1, 0.1, 0.1],
            }
        }
        base_goal = np.array([2.0, 0.0, 0.0])
        base_vel, _ = calculate_desired_velocity(
            base_goal, None, _states(), _MockController(), mpsf_params
        )
        np.testing.assert_allclose(base_vel[0], 0.9)
