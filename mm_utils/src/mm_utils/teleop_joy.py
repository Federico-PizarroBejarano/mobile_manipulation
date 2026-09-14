"""Pure helpers for joystick teleop (MPSF and direct; no ROS dependencies)."""

import numpy as np

# D-pad up. Same index as the hardware relay deadman in
# mobile_manipulation_central/joy_stick_relay.py (enable_button).
HARDWARE_DEADMAN_BUTTON = 13

# low_level_cmd_node zeros output when this param is False (teleop gate).
STICKS_ACTIVE_PARAM = "/teleop_sticks_active"

# Set True by direct_teleop_ros so low_level disables P tracking (shared YAML kp).
FORCE_ZERO_LL_KP_PARAM = "/mm_run/force_zero_ll_kp"

VALID_TELEOP_MODES = ("none", "mpsf", "direct")
TELEOP_MODE_PARAM = "/mm_run/teleop_mode"

TELEOP_DEFAULTS = {
    "enabled": False,
    "enable_button": HARDWARE_DEADMAN_BUTTON,
    "ee_yaw_buttons": [14, 15],
    "max_base_vel": [0.3, 0.3, 0.3],
    "max_ee_vel": [0.12, 0.12, 0.12, 0.25, 0.25, 0.25],
}

GOAL_VELOCITY_DEFAULTS = {
    "max_base_vel": [0.3, 0.3, 0.3],
    "max_ee_vel": [0.15, 0.15, 0.15, 0.3, 0.3, 0.3],
    "base_threshold": [0.3, 0.3, 0.3],
    "ee_threshold": [0.2, 0.2, 0.2, 0.3, 0.3, 0.3],
}


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


def teleop_enable_held(buttons, enable_button_index):
    """True when teleop enable button is pressed; always True if index is None."""
    if enable_button_index is None:
        return True
    idx = int(enable_button_index)
    if idx < 0 or idx >= len(buttons):
        return False
    return bool(buttons[idx])


def axes_to_base_velocity(joy_axes, max_base_vel):
    """Map stored joy axes [lx, ly, rx, ry, lt, rt] to base [vx, vy, vyaw]."""
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


def axes_to_ee_velocity(joy_axes, buttons, max_ee_vel, ee_yaw_buttons):
    """Map joy axes + yaw buttons to EE twist [vx, vy, vz, wx, wy, wz]."""
    joy_axes = np.asarray(joy_axes, dtype=float).reshape(-1)
    max_ee_vel = np.asarray(max_ee_vel, dtype=float).reshape(6)
    left_idx, right_idx = ee_yaw_buttons
    joy_wy = float(joy_axes[5]) - float(joy_axes[4])
    joy_wz = ee_yaw_from_buttons(buttons, left_idx, right_idx)
    return np.array(
        [
            joy_axes[1] * max_ee_vel[0],
            joy_axes[0] * max_ee_vel[1],
            joy_axes[3] * max_ee_vel[2],
            joy_axes[2] * max_ee_vel[3],
            joy_wy * max_ee_vel[4],
            joy_wz * max_ee_vel[5],
        ],
        dtype=float,
    )


def joint_velocity_command(mode, base_vel, ee_vel, nu):
    """Map teleop twists to length-``nu`` joint cmd_vel (world frame).

    Base mode copies base twist into ``[:3]``. EE mode returns zeros here;
    callers that support differential IK should solve arm rates separately.
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
    enable_btn = section.get("enable_button", TELEOP_DEFAULTS["enable_button"])
    return {
        "enabled": bool(section.get("enabled", TELEOP_DEFAULTS["enabled"])),
        "enable_button": int(enable_btn) if enable_btn is not None else None,
        "ee_yaw_buttons": list(
            section.get("ee_yaw_buttons", TELEOP_DEFAULTS["ee_yaw_buttons"])
        ),
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
