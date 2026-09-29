"""Pure helpers for joystick teleop (MPSF and direct; no ROS dependencies)."""

import numpy as np

from mm_utils.base_velocity_guard import body_twist_to_world
from mm_utils.teleop_mapping import (
    EE_PITCH_BUTTONS,
    EE_ROLL_BUTTONS,
    EE_YAW_AXIS,
    ENABLE_TRIGGER_AXIS,
    ENABLE_TRIGGER_THRESHOLD,
)

# low_level_cmd_node zeros output when this param is False (teleop gate).
STICKS_ACTIVE_PARAM = "/teleop_sticks_active"

# Set True by direct_teleop_ros so low_level disables P tracking (shared YAML kp).
FORCE_ZERO_LL_KP_PARAM = "/mm_run/force_zero_ll_kp"

VALID_TELEOP_MODES = ("none", "mpsf", "direct", "rl")
TELEOP_MODE_PARAM = "/mm_run/teleop_mode"

TELEOP_DEFAULTS = {
    "enabled": False,
    "max_base_vel": [0.3, 0.3, 0.3],
    "max_ee_vel": [0.12, 0.12, 0.12, 0.25, 0.25, 0.25],
}

GOAL_VELOCITY_DEFAULTS = {
    "max_base_vel": [0.3, 0.3, 0.3],
    "max_ee_vel": [0.15, 0.15, 0.15, 0.3, 0.3, 0.3],
    "base_threshold": [0.3, 0.3, 0.3],
    "ee_threshold": [0.2, 0.2, 0.2, 0.3, 0.3, 0.3],
}


def _planar_yaw_rotation(yaw):
    """3x3 rotation that maps chassis-frame vectors into the world frame."""
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def chassis_base_twist_to_world(v_chassis, yaw):
    """Map chassis-frame base twist ``[vx, vy, vyaw]`` to world frame."""
    return body_twist_to_world(v_chassis, float(yaw))


def teleop_ee_twist_for_control(twist_teleop, yaw, R_ee_wb):
    """Map teleop EE twist to spatial-Jacobian / MPC ``EEVel`` convention.

    ``twist_teleop`` is ``[vx, vy, vz, wx, wy, wz]`` with linear and angular
    velocity in the chassis frame (stick yaw about vertical, d-pad pitch/roll
    about chassis axes). Returns world-frame linear velocity and EE-body
    angular velocity (matching the tool spatial Jacobian and MPC ``EEVel``).
    """
    tw = np.asarray(twist_teleop, dtype=float).reshape(6)
    rot = _planar_yaw_rotation(float(yaw))
    R = np.asarray(R_ee_wb, dtype=float).reshape(3, 3)
    lin_w = rot @ tw[:3]
    ang_w = rot @ tw[3:]
    return np.concatenate([lin_w, R.T @ ang_w])


def teleop_ee_twist_to_world(twist_teleop, yaw, R_ee_wb=None):
    """Map teleop EE twist to a full world-frame twist.

    Linear and angular parts are chassis→world via planar yaw. ``R_ee_wb`` is
    accepted for call-site compatibility and ignored.
    """
    tw = np.asarray(twist_teleop, dtype=float).reshape(6)
    rot = _planar_yaw_rotation(float(yaw))
    return np.concatenate([rot @ tw[:3], rot @ tw[3:]])


def ee_yaw_from_buttons(buttons, left_idx, right_idx):
    """EE yaw rate factor from two digital buttons: right (+1) minus left (-1)."""
    left = float(buttons[left_idx]) if left_idx < len(buttons) else 0.0
    right = float(buttons[right_idx]) if right_idx < len(buttons) else 0.0
    return right - left


def gate_teleop_velocity(desired_vel, enabled):
    """Return desired velocity when enabled, else zeros of the same shape."""
    desired_vel = np.asarray(desired_vel, dtype=float)
    if enabled:
        return desired_vel
    return np.zeros_like(desired_vel)


def teleop_enable_held(axes):
    """True when the remapped enable trigger is pulled past threshold."""
    if ENABLE_TRIGGER_AXIS >= len(axes):
        return False
    return float(axes[ENABLE_TRIGGER_AXIS]) > ENABLE_TRIGGER_THRESHOLD


def axes_to_base_velocity(joy_axes, max_base_vel):
    """Map joy axes to chassis-frame base twist ``[vx, vy, vyaw]``.

    Callers must convert with :func:`chassis_base_twist_to_world` before feeding
    world-frame MPC / cmd_vel consumers.
    """
    joy_axes = np.asarray(joy_axes, dtype=float).reshape(-1)
    max_base_vel = np.asarray(max_base_vel, dtype=float).reshape(3)
    return np.array(
        [
            joy_axes[1] * max_base_vel[0],
            joy_axes[0] * max_base_vel[1],
            joy_axes[2] * max_base_vel[2],
        ],
        dtype=float,
    )


def axes_to_ee_velocity(joy_axes, buttons, max_ee_vel):
    """Map joy axes + d-pad buttons to teleop EE twist.

    Returns ``[vx, vy, vz, wx, wy, wz]`` in the chassis frame (right stick X →
    yaw about vertical, d-pad up/down → pitch, d-pad left/right → roll). Callers
    must convert with :func:`teleop_ee_twist_for_control` (MPC / spatial IK) or
    :func:`teleop_ee_twist_to_world` (world-frame consumers).
    """
    joy_axes = np.asarray(joy_axes, dtype=float).reshape(-1)
    max_ee_vel = np.asarray(max_ee_vel, dtype=float).reshape(6)
    joy_wx = ee_yaw_from_buttons(buttons, *EE_ROLL_BUTTONS)
    joy_wy = ee_yaw_from_buttons(buttons, *EE_PITCH_BUTTONS)
    return np.array(
        [
            joy_axes[1] * max_ee_vel[0],
            joy_axes[0] * max_ee_vel[1],
            joy_axes[3] * max_ee_vel[2],
            -joy_wx * max_ee_vel[3],
            joy_wy * max_ee_vel[4],
            joy_axes[EE_YAW_AXIS] * max_ee_vel[5],
        ],
        dtype=float,
    )


def joint_velocity_command(mode, base_vel, ee_vel, nu):
    """Map teleop twists to length-``nu`` joint cmd_vel (world-frame base).

    ``base_vel`` must already be in the world frame. Base mode copies it into
    ``[:3]``. EE mode returns zeros here; callers that support differential IK
    should solve arm rates separately.
    """
    cmd = np.zeros(int(nu), dtype=float)
    if mode == "base":
        base_vel = np.asarray(base_vel, dtype=float).reshape(-1)
        n = min(3, int(nu), base_vel.size)
        cmd[:n] = base_vel[:n]
    return cmd


def parse_teleop_config(controller_config=None):
    """Parse ``controller.teleop`` stick/deadman settings."""
    controller_config = controller_config or {}
    section = controller_config.get("teleop") or {}
    return {
        "enabled": bool(section.get("enabled", TELEOP_DEFAULTS["enabled"])),
        "max_base_vel": np.asarray(
            section.get("max_base_vel", TELEOP_DEFAULTS["max_base_vel"]), dtype=float
        ),
        "max_ee_vel": np.asarray(
            section.get("max_ee_vel", TELEOP_DEFAULTS["max_ee_vel"]), dtype=float
        ),
    }


def parse_goal_velocity_params(mpsf_params=None):
    """Parse MPSF goal-following velocity limits from ``mpsf_params.goal_velocity``."""
    mpsf_params = mpsf_params or {}
    section = mpsf_params.get("goal_velocity", mpsf_params)
    return {
        key: np.asarray(section.get(key, default), dtype=float)
        for key, default in GOAL_VELOCITY_DEFAULTS.items()
    }


def store_joy_axes(msg_axes, out_axes):
    """Fill length-6 ``out_axes`` from a raw Joy axes sequence (in-place)."""
    n_axes = len(msg_axes)
    if n_axes > 0:
        out_axes[0] = msg_axes[0]
    if n_axes > 1:
        out_axes[1] = msg_axes[1]
    if n_axes > 3:
        out_axes[2] = msg_axes[3]
    if n_axes > 4:
        out_axes[3] = msg_axes[4]
    if n_axes > 2:
        out_axes[4] = max(0.0, (1.0 - msg_axes[2]) / 2.0)
    if n_axes > 5:
        out_axes[5] = max(0.0, (1.0 - msg_axes[5]) / 2.0)
    return out_axes
