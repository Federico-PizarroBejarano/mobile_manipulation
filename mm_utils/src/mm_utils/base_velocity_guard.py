"""Sanitize base velocity from joint_states using odom twist."""

from __future__ import annotations

import numpy as np

from mm_utils.math import wrap_pi_scalar


def body_twist_to_world(v_body: np.ndarray, yaw: float) -> np.ndarray:
    """Map Ridgeback body-frame twist to world-frame base velocity."""
    v_body = np.asarray(v_body, dtype=float).reshape(3)
    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot @ v_body


def sanitize_base_velocity(
    q: np.ndarray,
    v: np.ndarray,
    v_odom_body: np.ndarray,
    max_disagreement: float,
) -> tuple[np.ndarray, bool]:
    """Replace base velocity with odom if joint_states disagree too much.

    Args:
        q: Generalized positions, shape (nq,).
        v: Generalized velocities, shape (nv,); base block is updated in-place copy.
        v_odom_body: Odom twist [vx, vy, vyaw] in body frame.
        max_disagreement: L2 threshold on world-frame base velocity [m/s, m/s, rad/s].

    Returns:
        (v_out, replaced): Updated velocity and whether base block was replaced.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    v_out = np.asarray(v, dtype=float).reshape(-1).copy()
    v_odom_body = np.asarray(v_odom_body, dtype=float).reshape(3)

    yaw = wrap_pi_scalar(float(q[2]))
    v_base = v_out[:3]
    v_odom_world = body_twist_to_world(v_odom_body, yaw)

    disagreement = float(np.linalg.norm(v_base - v_odom_world))
    if disagreement <= max_disagreement:
        return v_out, False

    v_out[:3] = v_odom_world
    return v_out, True


def _as_base_twist(v_cmd: np.ndarray) -> np.ndarray:
    """Normalize a base twist to shape (3,); horizon rows use the first sample."""
    v_cmd = np.asarray(v_cmd, dtype=float)
    if v_cmd.ndim == 2:
        v_cmd = v_cmd[0]
    v_cmd = v_cmd.reshape(-1)
    if v_cmd.size < 3:
        raise ValueError(f"base twist must have at least 3 elements, got {v_cmd.size}")
    return v_cmd[:3]


def base_velocity_command_from_references(references: dict | None) -> np.ndarray | None:
    """Extract commanded base twist for the idle gate.

    - ``desired_velocity["base_velocity"]`` when present (base teleop / explicit cmd)
    - zeros when ``desired_velocity`` exists but has no base key (EE teleop: not
      commanding the base, still idle for coasting prevention)
    - else ``references["base_velocity"]`` (planner), or None to skip the gate
    """
    references = references or {}
    desired = references.get("desired_velocity")
    if isinstance(desired, dict):
        if "base_velocity" in desired:
            return desired["base_velocity"]
        # EE teleop (or any desired_velocity without base): not commanding base.
        return np.zeros(3, dtype=float)
    return references.get("base_velocity")


def ee_velocity_command_from_references(references: dict | None) -> np.ndarray | None:
    """Extract commanded EE twist for the arm idle gate.

    - ``desired_velocity["ee_velocity"]`` when present (EE teleop / explicit cmd)
    - zeros when ``desired_velocity`` exists but has no EE key (base teleop: not
      commanding the EE, still idle for arm coasting prevention)
    - else ``references["ee_velocity"]`` (planner), or None to skip the gate
    """
    references = references or {}
    desired = references.get("desired_velocity")
    if isinstance(desired, dict):
        if "ee_velocity" in desired:
            return desired["ee_velocity"]
        # Base teleop (or any desired_velocity without EE): not commanding EE.
        return np.zeros(6, dtype=float)
    return references.get("ee_velocity")


def _as_ee_twist(v_cmd: np.ndarray) -> np.ndarray:
    """Normalize an EE twist to shape (6,); horizon rows use the first sample."""
    v_cmd = np.asarray(v_cmd, dtype=float)
    if v_cmd.ndim == 2:
        v_cmd = v_cmd[0]
    v_cmd = v_cmd.reshape(-1)
    if v_cmd.size < 6:
        raise ValueError(f"EE twist must have at least 6 elements, got {v_cmd.size}")
    return v_cmd[:6]


def zero_idle_base_velocity(
    v: np.ndarray,
    v_cmd: np.ndarray | None,
    cmd_eps: float,
    meas_eps: float,
) -> tuple[np.ndarray, bool]:
    """Zero measured base velocity when command and measurement are both near zero.

    Stops estimator noise / bias from becoming an MPC coasting reference when the
    teleop or planner base-velocity command is idle. Does not freeze the base:
    constraints and costs can still accelerate from a true rest state, and any
    nontrivial measured velocity is left unchanged.

    Args:
        v: Generalized velocities, shape (nv,).
        v_cmd: Commanded base twist [vx, vy, vyaw], or horizon (N+1, 3). None skips.
        cmd_eps: L2 threshold on commanded base twist.
        meas_eps: L2 threshold on measured base twist.

    Returns:
        (v_out, zeroed): Updated velocity and whether the base block was zeroed.
    """
    v_out = np.asarray(v, dtype=float).reshape(-1).copy()
    if v_cmd is None:
        return v_out, False

    v_cmd_base = _as_base_twist(v_cmd)
    if float(np.linalg.norm(v_cmd_base)) > float(cmd_eps):
        return v_out, False
    if float(np.linalg.norm(v_out[:3])) > float(meas_eps):
        return v_out, False

    v_out[:3] = 0.0
    return v_out, True


def zero_idle_arm_velocity(
    v: np.ndarray,
    v_ee_cmd: np.ndarray | None,
    cmd_eps: float,
    arm_meas_eps: float,
) -> tuple[np.ndarray, bool]:
    """Zero measured arm joint rates when EE command and arm rates are both near zero.

    Same coasting prevention as ``zero_idle_base_velocity``, for the arm block
    ``v[3:9]`` when EE is not being commanded (e.g. base teleop).

    Args:
        v: Generalized velocities, shape (nv,); requires at least 9 elements.
        v_ee_cmd: Commanded EE twist (6,), or horizon (N+1, 6). None skips.
        cmd_eps: L2 threshold on commanded EE twist.
        arm_meas_eps: L2 threshold on measured arm joint rates [rad/s].

    Returns:
        (v_out, zeroed): Updated velocity and whether the arm block was zeroed.
    """
    v_out = np.asarray(v, dtype=float).reshape(-1).copy()
    if v_ee_cmd is None:
        return v_out, False
    if v_out.size < 9:
        raise ValueError(f"velocity must have at least 9 elements, got {v_out.size}")

    v_cmd_ee = _as_ee_twist(v_ee_cmd)
    if float(np.linalg.norm(v_cmd_ee)) > float(cmd_eps):
        return v_out, False
    if float(np.linalg.norm(v_out[3:9])) > float(arm_meas_eps):
        return v_out, False

    v_out[3:9] = 0.0
    return v_out, True
