"""Differential (Jacobian) IK helpers shared by RL and direct teleop."""

from __future__ import annotations

import numpy as np

from mm_utils.math import clamp_ee_velocity


def build_ik_params_from_config(config, nu, nq):
    """Build the parameter dict used by :func:`solve_diff_ik` from merged YAML config.

    Args:
        config (dict): Top-level config containing optional ``ik`` and ``robot``
            sections and ``controller.robot.limits.state`` for velocity bounds.
        nu (int): Total number of velocity inputs (dimension of ``u`` / Jacobian columns).
        nq (int): Number of generalized positions; velocity limits are taken from
            index ``nq:`` onward in the state's lower/upper arrays.

    Returns:
        dict: Keys ``nu``, ``use_weighted_regularization``, ``regularization_strength``,
            ``joint_vel_lower``, ``joint_vel_upper`` (arm + base velocity bounds),
            ``ee_max_linear_vel``, ``ee_max_angular_vel`` (for clamping desired EE twist).
    """
    ik_cfg = config.get("ik", {})
    robot_cfg = config.get("robot", {})
    lim = (
        config.get("controller", {}).get("robot", {}).get("limits", {}).get("state", {})
    )
    vel_lo = np.array(lim.get("lower", [0] * nu)[nq:], dtype=float)
    vel_hi = np.array(lim.get("upper", [0] * nu)[nq:], dtype=float)
    return {
        "nu": int(nu),
        "use_weighted_regularization": ik_cfg.get("use_weighted_regularization", True),
        "regularization_strength": float(ik_cfg.get("regularization_strength", 0.1)),
        "joint_vel_lower": vel_lo,
        "joint_vel_upper": vel_hi,
        "ee_max_linear_vel": float(robot_cfg.get("ee_linear_vel_limit", 0.5)),
        "ee_max_angular_vel": float(robot_cfg.get("ee_angular_vel_limit", 0.75)),
    }


def solve_diff_ik(J, desired_ee_vel, base_vel, ik_params):
    """Compute joint velocities: base command plus arm rates from Jacobian IK.

    Clamps ``desired_ee_vel``, subtracts the end-effector velocity induced by
    ``base_vel`` via the base columns of ``J``, then solves for arm joint rates
    with weighted regularization or damped least squares. Result is clipped to
    ``joint_vel_lower`` / ``joint_vel_upper``.

    Args:
        J (ndarray): Spatial Jacobian, shape ``(6, nu)``.
        desired_ee_vel (ndarray): 6D EE twist matching Jacobian rows
            (for ``MobileManipulator3D`` spatial J: world linear + body angular).
        base_vel (ndarray): Base velocities ``[vx, vy, vyaw]`` (3,).
        ik_params (dict): From :func:`build_ik_params_from_config`.

    Returns:
        ndarray: Joint velocity command ``u`` of shape ``(nu,)``.
    """
    nu = int(ik_params["nu"])
    jlo = np.asarray(ik_params["joint_vel_lower"], dtype=float).reshape(-1)
    jhi = np.asarray(ik_params["joint_vel_upper"], dtype=float).reshape(-1)
    J = np.asarray(J, dtype=float).reshape(6, nu)
    base_vel = np.asarray(base_vel, dtype=float).reshape(3)

    v = clamp_ee_velocity(
        desired_ee_vel,
        float(ik_params.get("ee_max_linear_vel", 0.5)),
        float(ik_params.get("ee_max_angular_vel", 0.75)),
    )
    base_idx = [0, 1, 2]
    arm_idx = list(range(3, nu))
    v_ee_from_base = J[:, base_idx] @ base_vel
    v_arm = v - v_ee_from_base
    J_arm = J[:, arm_idx]
    if ik_params["use_weighted_regularization"]:
        max_vels = np.maximum(np.abs(jhi[arm_idx]), 1e-6)
        W = np.diag(1.0 / max_vels)
        A = J_arm.T @ J_arm + ik_params["regularization_strength"] * (W.T @ W)
        arm_vel = np.linalg.pinv(A) @ (J_arm.T @ v_arm)
    else:
        damp = 0.01
        Jpinv = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T + damp * np.eye(6))
        arm_vel = Jpinv @ v_arm
    u = np.zeros(nu, dtype=float)
    u[base_idx] = base_vel
    u[arm_idx] = arm_vel
    for i in range(nu):
        u[i] = np.clip(u[i], jlo[i], jhi[i])
    return u


def spatial_jacobian(robot_mdl, q):
    """Evaluate ``MobileManipulator3D`` tool spatial Jacobian at ``q``."""
    tool = robot_mdl.tool_link_name
    key = tool + "_spatial"
    J = robot_mdl.jacSymMdls[key](np.asarray(q, dtype=float).reshape(-1))
    return np.asarray(J, dtype=float).reshape(6, int(robot_mdl.DoF))
