"""Tests for base velocity guard helper."""

import numpy as np

from mm_utils.base_velocity_guard import (
    base_velocity_command_from_references,
    body_twist_to_world,
    ee_velocity_command_from_references,
    sanitize_base_velocity,
    zero_idle_arm_velocity,
    zero_idle_base_velocity,
)


class TestBodyTwistToWorld:
    def test_zero_yaw_passthrough_xy(self):
        v = body_twist_to_world(np.array([0.5, -0.2, 0.1]), 0.0)
        np.testing.assert_allclose(v, [0.5, -0.2, 0.1], atol=1e-9)

    def test_yaw_rotates_xy(self):
        v = body_twist_to_world(np.array([1.0, 0.0, 0.0]), np.pi / 2)
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-9)


class TestSanitizeBaseVelocity:
    def test_no_replace_when_agreement(self):
        q = np.zeros(9)
        v = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v_out, replaced = sanitize_base_velocity(
            q, v, np.array([0.5, 0.0, 0.0]), max_disagreement=0.25
        )
        assert not replaced
        np.testing.assert_allclose(v_out[:3], [0.5, 0.0, 0.0])

    def test_replace_when_spike(self):
        q = np.zeros(9)
        v = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v_out, replaced = sanitize_base_velocity(
            q, v, np.array([-0.7, 0.0, 0.0]), max_disagreement=0.25
        )
        assert replaced
        np.testing.assert_allclose(v_out[:3], [-0.7, 0.0, 0.0])

    def test_arm_velocities_unchanged(self):
        q = np.zeros(9)
        v = np.array([-2.0, 0.0, 0.0, 1.1, 2.2, 3.3, 0.0, 0.0, 0.0])
        v_out, replaced = sanitize_base_velocity(
            q, v, np.array([-0.7, 0.0, 0.0]), max_disagreement=0.25
        )
        assert replaced
        np.testing.assert_allclose(v_out[3:], [1.1, 2.2, 3.3, 0.0, 0.0, 0.0])


class TestZeroIdleBaseVelocity:
    def test_zeros_when_cmd_and_meas_idle(self):
        v = np.array([0.004, -0.002, 0.001, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        v_out, zeroed = zero_idle_base_velocity(
            v, np.zeros(3), cmd_eps=1e-3, meas_eps=0.02
        )
        assert zeroed
        np.testing.assert_allclose(v_out[:3], 0.0)
        np.testing.assert_allclose(v_out[3:], v[3:])

    def test_skips_when_cmd_nonzero(self):
        v = np.array([0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v_out, zeroed = zero_idle_base_velocity(
            v, np.array([0.1, 0.0, 0.0]), cmd_eps=1e-3, meas_eps=0.02
        )
        assert not zeroed
        np.testing.assert_allclose(v_out, v)

    def test_skips_when_meas_above_eps(self):
        v = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v_out, zeroed = zero_idle_base_velocity(
            v, np.zeros(3), cmd_eps=1e-3, meas_eps=0.02
        )
        assert not zeroed
        np.testing.assert_allclose(v_out, v)

    def test_skips_when_cmd_none(self):
        v = np.array([0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v_out, zeroed = zero_idle_base_velocity(v, None, cmd_eps=1e-3, meas_eps=0.02)
        assert not zeroed
        np.testing.assert_allclose(v_out, v)

    def test_accepts_horizon_cmd(self):
        v = np.array([0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v_cmd = np.zeros((11, 3))
        v_out, zeroed = zero_idle_base_velocity(v, v_cmd, cmd_eps=1e-3, meas_eps=0.02)
        assert zeroed
        np.testing.assert_allclose(v_out[:3], 0.0)


class TestBaseVelocityCommandFromReferences:
    def test_explicit_base_command(self):
        cmd = np.array([0.1, 0.0, 0.0])
        out = base_velocity_command_from_references(
            {"desired_velocity": {"base_velocity": cmd}}
        )
        np.testing.assert_allclose(out, cmd)

    def test_ee_teleop_missing_base_is_idle_zero(self):
        out = base_velocity_command_from_references(
            {"desired_velocity": {"ee_velocity": np.ones(6)}}
        )
        np.testing.assert_allclose(out, np.zeros(3))

    def test_planner_base_velocity_fallback(self):
        cmd = np.array([0.2, 0.0, 0.0])
        out = base_velocity_command_from_references({"base_velocity": cmd})
        np.testing.assert_allclose(out, cmd)

    def test_none_when_no_velocity_refs(self):
        assert base_velocity_command_from_references({}) is None
        assert base_velocity_command_from_references(None) is None

    def test_ee_idle_gate_zeros_measured_base(self):
        v = np.array([0.004, -0.002, 0.001, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        v_cmd = base_velocity_command_from_references(
            {"desired_velocity": {"ee_velocity": np.zeros(6)}}
        )
        v_out, zeroed = zero_idle_base_velocity(v, v_cmd, cmd_eps=1e-3, meas_eps=0.02)
        assert zeroed
        np.testing.assert_allclose(v_out[:3], 0.0)


class TestEeVelocityCommandFromReferences:
    def test_explicit_ee_command(self):
        cmd = np.ones(6) * 0.1
        out = ee_velocity_command_from_references(
            {"desired_velocity": {"ee_velocity": cmd}}
        )
        np.testing.assert_allclose(out, cmd)

    def test_base_teleop_missing_ee_is_idle_zero(self):
        out = ee_velocity_command_from_references(
            {"desired_velocity": {"base_velocity": np.ones(3)}}
        )
        np.testing.assert_allclose(out, np.zeros(6))

    def test_planner_ee_velocity_fallback(self):
        cmd = np.ones(6) * 0.2
        out = ee_velocity_command_from_references({"ee_velocity": cmd})
        np.testing.assert_allclose(out, cmd)

    def test_none_when_no_velocity_refs(self):
        assert ee_velocity_command_from_references({}) is None
        assert ee_velocity_command_from_references(None) is None


class TestZeroIdleArmVelocity:
    def test_zeros_when_ee_cmd_and_arm_meas_idle(self):
        v = np.array([0.5, 0.0, 0.0, 0.01, -0.005, 0.002, 0.0, 0.0, 0.0])
        v_out, zeroed = zero_idle_arm_velocity(
            v, np.zeros(6), cmd_eps=1e-3, arm_meas_eps=0.05
        )
        assert zeroed
        np.testing.assert_allclose(v_out[:3], v[:3])
        np.testing.assert_allclose(v_out[3:9], 0.0)

    def test_skips_when_ee_cmd_nonzero(self):
        v = np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        v_out, zeroed = zero_idle_arm_velocity(
            v, np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]), cmd_eps=1e-3, arm_meas_eps=0.05
        )
        assert not zeroed
        np.testing.assert_allclose(v_out, v)

    def test_skips_when_arm_meas_above_eps(self):
        v = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        v_out, zeroed = zero_idle_arm_velocity(
            v, np.zeros(6), cmd_eps=1e-3, arm_meas_eps=0.05
        )
        assert not zeroed
        np.testing.assert_allclose(v_out, v)

    def test_base_mode_missing_ee_key_zeros_tiny_arm(self):
        v = np.array([0.004, 0.0, 0.0, 0.01, -0.005, 0.002, 0.0, 0.0, 0.0])
        v_ee_cmd = ee_velocity_command_from_references(
            {"desired_velocity": {"base_velocity": np.zeros(3)}}
        )
        v_out, zeroed = zero_idle_arm_velocity(
            v, v_ee_cmd, cmd_eps=1e-3, arm_meas_eps=0.05
        )
        assert zeroed
        np.testing.assert_allclose(v_out[3:9], 0.0)

    def test_accepts_horizon_ee_cmd(self):
        v = np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        v_cmd = np.zeros((11, 6))
        v_out, zeroed = zero_idle_arm_velocity(
            v, v_cmd, cmd_eps=1e-3, arm_meas_eps=0.05
        )
        assert zeroed
        np.testing.assert_allclose(v_out[3:9], 0.0)
