"""Stop gripper cables from being driven into the proximal wrist link.

Two cables leave the top of the gripper (camera side) and stand proud of the
housing. Yawing the gripper about ``ur10_arm_wrist_2_joint`` swings that bulge
into the link between wrist 1 and wrist 2. From the tray pose that contact is
about +90 deg of wrist 2: positive wrist 2 moves the camera-side cables toward
the forearm / wrist-1 body. Wrist 3 only shifts the same orbit by a few tens
of degrees, so the limit is on the cable's orbit angle, not a fixed wrist-2 box.

At the stop, the arm velocity is zeroed until the cables yaw back out of a
hysteresis band. An opening yaw keeps wrist 2 and wrist 3 so the arm can
retreat, and still zeros the proximal arm joints.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

# Tray pose (home.yaml). Contact was observed about one quarter-turn from here.
_TRAY_WRIST_2 = 1.57079864
_TRAY_WRIST_3 = 1.28475273
# How far past the tray clocking the cables meet the link.
CONTACT_FROM_TRAY_RAD = 0.5 * np.pi
# Hold short of that pose. 25 deg leaves clearance for the bulge.
MARGIN_RAD = np.deg2rad(25.0)
# Stay latched until the cables are this far back inside the stop.
DEFAULT_RELEASE_MARGIN_RAD = 0.14
CABLE_GUARD_ENABLED_PARAM = "/mm_run/gripper_cable_guard/enabled"
CABLE_GUARD_RELEASE_PARAM = "/mm_run/gripper_cable_guard/release_margin_rad"
_OPENING_RATE = -1e-3

_W2_XYZ = np.array([-1.744730553720786e-05, -0.1156796703530661, 9.578054833275211e-06])
_W2_RPY = np.array([1.570713528713072, 0.0, -5.018411682050918e-05])
_W3_XYZ = np.array([7.947493484628657e-05, 0.09205088147541438, 2.139222642043671e-05])
_W3_RPY = np.array([1.571028722462297, np.pi, -3.141565446353047])
# tool0 -> gripper, including the 15 deg mounting yaw.
_GRIP_YAW = 0.2617993877991494
# Cable bulge on the camera side of the gripper, just above the flange.
_CAM_XY = np.array([0.0218279, -0.16487507])
_CABLE_POINT_GRIPPER = np.array(
    [
        *(_CAM_XY / np.linalg.norm(_CAM_XY) * 0.075),
        0.015,
        1.0,
    ]
)
# Wrist-2 axis point in the wrist-1 link. Orbit angle is measured in the XZ plane.
_AXIS_ORIGIN = np.array([0.0, -0.1156796703530661, 0.0])


def _wrap_pi(theta: float) -> float:
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


def _joint_transform(xyz, rpy, q: float) -> np.ndarray:
    rot = R.from_euler("xyz", rpy).as_matrix() @ R.from_euler("z", q).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = xyz
    return transform


def _gripper_in_wrist1(wrist_2: float, wrist_3: float) -> np.ndarray:
    """Homogeneous transform of the gripper frame in the wrist-1 link."""
    flange = _joint_transform(
        np.zeros(3), np.array([0.0, -0.5 * np.pi, -0.5 * np.pi]), 0.0
    )
    tool0 = _joint_transform(
        np.zeros(3), np.array([0.5 * np.pi, 0.0, 0.5 * np.pi]), 0.0
    )
    gripper = _joint_transform(np.zeros(3), np.array([0.0, 0.0, _GRIP_YAW]), 0.0)
    transform = _joint_transform(_W2_XYZ, _W2_RPY, wrist_2)
    transform = transform @ _joint_transform(_W3_XYZ, _W3_RPY, wrist_3)
    return transform @ flange @ tool0 @ gripper


def cable_orbit_angle(wrist_2: float, wrist_3: float) -> float:
    """Angle of the cable bulge around the wrist-2 axis, in (-pi, pi]."""
    point = (_gripper_in_wrist1(wrist_2, wrist_3) @ _CABLE_POINT_GRIPPER)[:3]
    radial = point - _AXIS_ORIGIN
    return float(np.arctan2(radial[2], radial[0]))


_TRAY_ORBIT = cable_orbit_angle(_TRAY_WRIST_2, _TRAY_WRIST_3)
STOP_EXCESS = CONTACT_FROM_TRAY_RAD - MARGIN_RAD
_STOP_EXCESS = STOP_EXCESS


def cable_orbit_excess(wrist_2: float, wrist_3: float) -> float:
    """Signed yaw of the cables away from the tray clocking, in (-pi, pi].

    Positive excess swings the cables toward the proximal wrist link.
    """
    return _wrap_pi(cable_orbit_angle(wrist_2, wrist_3) - _TRAY_ORBIT)


def _arm_index_groups(size: int):
    """Return proximal arm indices and (wrist 2, wrist 3).

    Layout is either the 6 arm joints or base (3) plus arm (6).
    """
    if size == 9:
        return (3, 4, 5, 6), (7, 8)
    if size == 6:
        return (0, 1, 2, 3), (4, 5)
    return None


def _excess_rate(wrist_2: float, wrist_3: float, v2: float, v3: float) -> float:
    eps = 1e-5
    base = cable_orbit_angle(wrist_2, wrist_3)
    d2 = _wrap_pi(cable_orbit_angle(wrist_2 + eps, wrist_3) - base) / eps
    d3 = _wrap_pi(cable_orbit_angle(wrist_2, wrist_3 + eps) - base) / eps
    return float(d2 * v2 + d3 * v3)


class GripperCableGuard:
    """Latch the arm at the cable stop until the wrist yaws clear.

    Disabled by default. ``q`` and ``cmd_vel`` are the 6 arm joints or the
    9-DoF mobile manipulator command (base, then arm).
    """

    def __init__(self, enabled=False, release_margin_rad=DEFAULT_RELEASE_MARGIN_RAD):
        self.enabled = bool(enabled)
        self.release_margin_rad = float(release_margin_rad)
        self.latched = False

    def apply(self, q, cmd_vel):
        """Return ``(cmd_out, clipped)``.

        While latched, proximal arm rates are zero. Wrist 2 and wrist 3 are
        kept only when they decrease the cable excess.
        """
        cmd = np.asarray(cmd_vel, dtype=float).reshape(-1).copy()
        if not self.enabled:
            self.latched = False
            return cmd, False

        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != cmd.size:
            return cmd, False
        groups = _arm_index_groups(q.size)
        if groups is None:
            return cmd, False
        proximal, (i2, i3) = groups

        excess = cable_orbit_excess(float(q[i2]), float(q[i3]))
        if excess >= _STOP_EXCESS:
            self.latched = True
        elif excess <= _STOP_EXCESS - self.release_margin_rad:
            self.latched = False
        if not self.latched:
            return cmd, False

        opening = (
            _excess_rate(float(q[i2]), float(q[i3]), float(cmd[i2]), float(cmd[i3]))
            < _OPENING_RATE
        )
        original = cmd.copy()
        cmd[list(proximal)] = 0.0
        if not opening:
            cmd[i2] = 0.0
            cmd[i3] = 0.0
        return cmd, not np.allclose(cmd, original)
