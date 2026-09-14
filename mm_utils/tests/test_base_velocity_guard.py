"""Tests for base velocity guard helper."""

import numpy as np

from mm_utils.base_velocity_guard import body_twist_to_world, sanitize_base_velocity


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
