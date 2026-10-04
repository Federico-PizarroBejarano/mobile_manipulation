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

# Per-axis slew of the stick reference. Base is [vx, vy, vyaw] in m/s^2, m/s^2,
# rad/s^2. EE is [vx, vy, vz, wx, wy, wz] in m/s^2 and rad/s^2.
REFERENCE_RAMP_DEFAULTS = {
    "base": [1.0, 1.0, 1.0],
    "ee": [1.0, 1.0, 1.0, 1.5, 1.5, 1.5],
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


def teleop_ee_twist_for_control(twist_teleop, yaw):
    """Map teleop EE twist to spatial-Jacobian / MPC ``EEVel`` convention.

    ``twist_teleop`` is ``[vx, vy, vz, wx, wy, wz]`` with linear velocity in the
    chassis frame and angular velocity in the tool frame (right stick X → yaw
    about the gripper, d-pad pitch/roll about gripper axes). Returns world-frame
    linear velocity and the same tool-frame angular velocity (matching the tool
    spatial Jacobian and MPC ``EEVel``).
    """
    tw = np.asarray(twist_teleop, dtype=float).reshape(6)
    rot = _planar_yaw_rotation(float(yaw))
    return np.concatenate([rot @ tw[:3], tw[3:]])


def teleop_ee_twist_to_world(twist_teleop, yaw, R_ee_wb):
    """Map teleop EE twist to a full world-frame twist.

    Linear velocity is chassis→world via planar yaw. Angular velocity is
    tool→world via ``R_ee_wb``.
    """
    tw = np.asarray(twist_teleop, dtype=float).reshape(6)
    rot = _planar_yaw_rotation(float(yaw))
    R = np.asarray(R_ee_wb, dtype=float).reshape(3, 3)
    return np.concatenate([rot @ tw[:3], R @ tw[3:]])


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

    Returns ``[vx, vy, vz, wx, wy, wz]``. Linear velocity is chassis-frame;
    angular velocity is tool-frame (right stick X → yaw about the gripper,
    d-pad up/down → pitch, d-pad left/right → roll). Callers must convert with
    :func:`teleop_ee_twist_for_control` (MPC / spatial IK) or
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
            joy_wx * max_ee_vel[3],
            -joy_wy * max_ee_vel[4],
            joy_axes[EE_YAW_AXIS] * max_ee_vel[5],
        ],
        dtype=float,
    )


def slew_toward(current, target, max_rate, dt):
    """Move ``current`` toward ``target`` by at most ``max_rate * dt`` per axis.

    ``max_rate`` is the absolute rate limit, same shape as ``current`` and
    ``target``. A very large rate leaves the step unchanged.
    """
    current = np.asarray(current, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    max_rate = np.abs(np.asarray(max_rate, dtype=float).reshape(-1))
    if current.shape != target.shape or max_rate.shape != current.shape:
        raise ValueError(
            "slew_toward shapes must match, got "
            f"current {current.shape}, target {target.shape}, rate {max_rate.shape}"
        )
    dt = float(dt)
    if dt < 0.0:
        raise ValueError(f"dt must be non-negative, got {dt}")
    step = max_rate * dt
    delta = np.clip(target - current, -step, step)
    return current + delta


class TeleopTwistRamp:
    """Slew stick-frame base and EE twists between control cycles.

    Apply this before the chassis-to-world / tool-frame conversion so a yaw
    change rotates an already-ramped forward command.
    """

    def __init__(self, max_base_rate, max_ee_rate):
        self.max_base_rate = np.asarray(max_base_rate, dtype=float).reshape(3)
        self.max_ee_rate = np.asarray(max_ee_rate, dtype=float).reshape(6)
        self._base = np.zeros(3, dtype=float)
        self._ee = np.zeros(6, dtype=float)

    def reset(self):
        """Snap both channels to zero (deadman release or base/EE toggle)."""
        self._base[:] = 0.0
        self._ee[:] = 0.0

    def base(self, target, dt):
        """Ramp a chassis-frame base twist ``[vx, vy, vyaw]``. Returns a copy."""
        self._base = slew_toward(self._base, target, self.max_base_rate, dt)
        return self._base.copy()

    def ee(self, target, dt):
        """Ramp a teleop EE twist ``[vx, vy, vz, wx, wy, wz]``. Returns a copy."""
        self._ee = slew_toward(self._ee, target, self.max_ee_rate, dt)
        return self._ee.copy()


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


# ControlEffort parameter order used by MPC._set_control_effort_params.
CONTROL_EFFORT_NAMES = ("Qqa", "Qqb", "Qva", "Qvb", "Qua", "Qub")
_EE_EFFORT_SCALE = frozenset({"Qvb", "Qub"})
_BASE_EFFORT_SCALE = frozenset({"Qva", "Qua"})


def _nonnegative_strictness(section, key):
    value = float(section.get(key, 1.0))
    if value < 0.0:
        raise ValueError(f"controller.teleop.{key} must be >= 0, got {value}")
    return value


def scale_teleop_control_effort(values, mode, ee_strictness, base_strictness):
    """Scale base or arm effort weights for the active teleop mode.

    ``values`` is the nominal ControlEffort vector in ``CONTROL_EFFORT_NAMES``
    order. ``mode is None`` returns copies of those values. EE mode multiplies
    ``Qvb`` and ``Qub`` by ``ee_strictness``. Base mode multiplies ``Qva`` and
    ``Qua`` by ``base_strictness``. The input arrays are not modified.
    """
    values = [np.asarray(v, dtype=float) for v in values]
    if len(values) != len(CONTROL_EFFORT_NAMES):
        raise ValueError(
            f"expected {len(CONTROL_EFFORT_NAMES)} effort weights, got {len(values)}"
        )
    ee_strictness = float(ee_strictness)
    base_strictness = float(base_strictness)
    if ee_strictness < 0.0 or base_strictness < 0.0:
        raise ValueError(
            "teleop strictness must be >= 0, got "
            f"ee={ee_strictness}, base={base_strictness}"
        )
    if mode is None:
        return [v.copy() for v in values]
    if mode == "ee":
        scale_names = _EE_EFFORT_SCALE
        scale = ee_strictness
    elif mode == "base":
        scale_names = _BASE_EFFORT_SCALE
        scale = base_strictness
    else:
        raise ValueError(
            f"teleop effort mode must be 'base', 'ee', or None, got {mode}"
        )

    scaled = []
    for name, value in zip(CONTROL_EFFORT_NAMES, values):
        scaled.append(value * scale if name in scale_names else value.copy())
    return scaled


def _reference_ramp_rates(section):
    """Per-axis stick slew limits from ``controller.teleop.reference_ramp``."""
    ramp = section.get("reference_ramp") or {}
    return {
        "base": np.asarray(
            ramp.get("base", REFERENCE_RAMP_DEFAULTS["base"]), dtype=float
        ),
        "ee": np.asarray(ramp.get("ee", REFERENCE_RAMP_DEFAULTS["ee"]), dtype=float),
    }


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
        "reference_ramp": _reference_ramp_rates(section),
        "ee_teleop_strictness": _nonnegative_strictness(
            section, "ee_teleop_strictness"
        ),
        "base_teleop_strictness": _nonnegative_strictness(
            section, "base_teleop_strictness"
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
