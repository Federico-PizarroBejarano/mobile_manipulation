"""Tests for the gripper cable keep-out."""

import numpy as np

from mm_utils.gripper_cable_guard import (
    CONTACT_FROM_TRAY_RAD,
    DEFAULT_RELEASE_MARGIN_RAD,
    MARGIN_RAD,
    STOP_EXCESS,
    GripperCableGuard,
    cable_orbit_excess,
)

_TRAY_W2 = 1.57079864
_TRAY_W3 = 1.28475273


def _arm_q(wrist_2, wrist_3=_TRAY_W3):
    q = np.zeros(6)
    q[4] = wrist_2
    q[5] = wrist_3
    return q


def _guard():
    return GripperCableGuard(
        enabled=True, release_margin_rad=DEFAULT_RELEASE_MARGIN_RAD
    )


def _at_stop(extra_deg=2.0):
    return _TRAY_W2 + CONTACT_FROM_TRAY_RAD - MARGIN_RAD + np.deg2rad(extra_deg)


class TestCableOrbitExcess:
    def test_tray_is_zero(self):
        assert abs(cable_orbit_excess(_TRAY_W2, _TRAY_W3)) < 1e-9

    def test_positive_wrist_2_moves_toward_the_link(self):
        excess = cable_orbit_excess(_TRAY_W2 + CONTACT_FROM_TRAY_RAD, _TRAY_W3)
        assert abs(excess - CONTACT_FROM_TRAY_RAD) < np.deg2rad(1.0)

    def test_negative_wrist_2_opens_the_gap(self):
        excess = cable_orbit_excess(_TRAY_W2 - CONTACT_FROM_TRAY_RAD, _TRAY_W3)
        assert excess < 0.0


class TestGripperCableGuard:
    def test_disabled_is_a_noop_at_the_stop(self):
        guard = GripperCableGuard(enabled=False)
        cmd = np.array([0.1, -0.2, 0.3, 0.0, 0.5, 0.1])
        out, clipped = guard.apply(_arm_q(_at_stop()), cmd)
        assert not clipped
        assert not guard.latched
        np.testing.assert_allclose(out, cmd)

    def test_clear_of_the_stop_passes_the_command(self):
        cmd = np.zeros(6)
        cmd[4] = 0.4
        out, clipped = _guard().apply(_arm_q(_TRAY_W2), cmd)
        assert not clipped
        np.testing.assert_allclose(out, cmd)

    def test_stop_zeros_the_arm_when_closing(self):
        cmd = np.array([0.1, -0.2, 0.3, 0.05, 0.5, 0.1])
        out, clipped = _guard().apply(_arm_q(_at_stop()), cmd)
        assert clipped
        np.testing.assert_allclose(out, 0.0)

    def test_stop_keeps_opening_wrist_and_zeros_proximal(self):
        cmd = np.array([0.1, -0.2, 0.3, 0.05, -0.4, 0.02])
        out, clipped = _guard().apply(_arm_q(_at_stop()), cmd)
        assert clipped
        np.testing.assert_allclose(out[:4], 0.0)
        np.testing.assert_allclose(out[4:], cmd[4:])

    def test_hysteresis_holds_until_the_release_margin(self):
        guard = _guard()
        closing = np.array([0.2, 0.0, 0.0, 0.0, 0.3, 0.0])
        guard.apply(_arm_q(_at_stop()), closing)
        assert guard.latched

        # Still inside the band: a few degrees back is not enough.
        inside = _TRAY_W2 + STOP_EXCESS - 0.5 * DEFAULT_RELEASE_MARGIN_RAD
        out, clipped = guard.apply(
            _arm_q(inside), np.array([0.2, 0.1, 0.1, 0.1, 0.2, 0.0])
        )
        assert guard.latched
        assert clipped
        np.testing.assert_allclose(out, 0.0)

        clear = _TRAY_W2 + STOP_EXCESS - DEFAULT_RELEASE_MARGIN_RAD - 0.05
        cmd = np.array([0.2, -0.1, 0.05, 0.0, 0.2, 0.0])
        out, clipped = guard.apply(_arm_q(clear), cmd)
        assert not guard.latched
        assert not clipped
        np.testing.assert_allclose(out, cmd)

    def test_base_command_is_kept(self):
        q = np.zeros(9)
        q[7] = _at_stop()
        q[8] = _TRAY_W3
        cmd = np.array([0.2, -0.1, 0.05, 0.3, 0.2, -0.4, 0.1, 0.3, 0.0])
        out, clipped = _guard().apply(q, cmd)
        assert clipped
        np.testing.assert_allclose(out[:3], cmd[:3])
        np.testing.assert_allclose(out[3:], 0.0)

    def test_safe_direction_from_tray_is_not_limited(self):
        q = np.zeros(9)
        q[7] = _TRAY_W2 - 1.2
        q[8] = _TRAY_W3
        cmd = np.zeros(9)
        cmd[7] = -0.5
        out, clipped = _guard().apply(q, cmd)
        assert not clipped
        np.testing.assert_allclose(out, cmd)
