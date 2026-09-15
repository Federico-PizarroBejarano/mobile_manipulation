"""MoMa-Teleop-style end-effector motion inference from a 6D EE command.

Integrates a linear dynamical system forward from the current EE pose to build
a short motion plan. From that plan the RL agent receives:
  - ``v_ee``: twist toward the next pose
  - ``hat_ee``: next desired pose
  - ``g``: last pose on the horizon (subgoal)
"""

from __future__ import annotations

import numpy as np

from mm_utils.math import (
    omega_from_quat_step,
    quat_multiply,
    quat_normalize,
    quat_rotate,
)

# MoMa-Teleop defaults (arxiv 2409.15095 §III-C)
DEFAULT_HORIZON_M = 1.5
DEFAULT_RESOLUTION_M = 0.1
DEFAULT_MIN_STEPS = 5
DEFAULT_MAX_ANGLE_STEP = 0.1875  # rad per integration step
IDLE_LIN_EPS = 1e-4
IDLE_ANG_EPS = 1e-4


def _axis_angle_quat(axis, angle):
    """Unit quaternion (xyzs) for rotation by ``angle`` about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-12 or abs(angle) < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis / n
    half = 0.5 * float(angle)
    s = np.sin(half)
    return np.array(
        [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)], dtype=np.float64
    )


def signal_from_ee_twist(
    v_ee_cmd, resolution_m=DEFAULT_RESOLUTION_M, max_angle=DEFAULT_MAX_ANGLE_STEP
):
    """Map a 6D EE twist command to per-step translation and orientation deltas.

    Active translational sticks are normalized to length ``resolution_m`` (MoMa
    joystick treatment). Angular magnitude is clipped to ``max_angle`` rad/step.

    Args:
        v_ee_cmd (ndarray): World-frame EE twist ``[vx,vy,vz, wx,wy,wz]``.
        resolution_m (float): Translation step length when translating.
        max_angle (float): Max rotation magnitude per integration step (rad).

    Returns:
        tuple: ``(v_signal, q_signal, active)`` where ``v_signal`` is (3,),
        ``q_signal`` is xyzs quaternion, ``active`` is bool.
    """
    tw = np.asarray(v_ee_cmd, dtype=np.float64).reshape(6)
    lin = tw[:3]
    ang = tw[3:]
    lin_n = float(np.linalg.norm(lin))
    ang_n = float(np.linalg.norm(ang))
    active = lin_n > IDLE_LIN_EPS or ang_n > IDLE_ANG_EPS
    if not active:
        return (
            np.zeros(3, dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            False,
        )

    if lin_n > IDLE_LIN_EPS:
        v_signal = (lin / lin_n) * float(resolution_m)
    else:
        v_signal = np.zeros(3, dtype=np.float64)

    if ang_n > IDLE_ANG_EPS:
        angle = min(ang_n, float(max_angle))
        # Stick ang is a direction in world; rotate about that axis in world,
        # expressed as a body-fixed delta applied as q_signal * q_ee in integrate.
        # Convert world-axis rotation into a quaternion; for small steps applying
        # q_new = q_delta_world * q works when q_delta is world-fixed rotation.
        q_signal = _axis_angle_quat(ang / ang_n, angle)
    else:
        q_signal = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    return v_signal, q_signal, True


def integrate_ee_motion(
    ee_pos,
    ee_orn,
    v_ee_cmd,
    dt,
    horizon_m=DEFAULT_HORIZON_M,
    resolution_m=DEFAULT_RESOLUTION_M,
    min_steps=DEFAULT_MIN_STEPS,
    max_angle_step=DEFAULT_MAX_ANGLE_STEP,
):
    """Integrate EE command into ``v_ee``, next pose, and horizon subgoal.

    Args:
        ee_pos (ndarray): Current EE position (world), shape (3,).
        ee_orn (ndarray): Current EE orientation xyzs quaternion, shape (4,).
        v_ee_cmd (ndarray): Commanded EE twist (world), shape (6,).
        dt (float): Control period used to form ``v_ee`` from the first step.
        horizon_m (float): Planning horizon distance ``d_g`` (m).
        resolution_m (float): Path spacing ``res`` (m).
        min_steps (int): Minimum number of integration steps when active.
        max_angle_step (float): Max orientation change per step (rad).

    Returns:
        dict: Keys ``active``, ``v_ee`` (6,), ``hat_pos``, ``hat_orn``, ``goal_pos``,
        ``goal_orn``, ``n_steps``. Poses are world-frame.
    """
    ee_pos = np.asarray(ee_pos, dtype=np.float64).reshape(3)
    ee_orn = quat_normalize(ee_orn)
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    v_signal, q_signal, active = signal_from_ee_twist(
        v_ee_cmd, resolution_m=resolution_m, max_angle=max_angle_step
    )
    if not active:
        z6 = np.zeros(6, dtype=np.float64)
        return {
            "active": False,
            "v_ee": z6,
            "hat_pos": ee_pos.copy(),
            "hat_orn": ee_orn.copy(),
            "goal_pos": ee_pos.copy(),
            "goal_orn": ee_orn.copy(),
            "n_steps": 0,
        }

    # Paper: T = max(||v_signal|| * d_g / res, 5); with ||v||=res this is max(d_g/res, 5)
    step_len = float(np.linalg.norm(v_signal))
    if step_len < 1e-12:
        # Pure rotation: still take min_steps of orientation-only updates
        n_steps = int(min_steps)
    else:
        n_steps = int(max(step_len * float(horizon_m) / float(resolution_m), min_steps))
        # Prefer full horizon when translating at nominal resolution
        n_horizon = int(
            max(min_steps, int(np.ceil(float(horizon_m) / float(resolution_m))))
        )
        n_steps = max(n_steps, n_horizon)

    pos = ee_pos.copy()
    orn = ee_orn.copy()
    poses = []
    for _ in range(n_steps):
        # Eq. 2: pos += R(q_signal) @ v_signal is wrong for identity q_signal;
        # paper writes q_signal · v_signal as rotating the step into world.
        pos = pos + quat_rotate(q_signal, v_signal)
        orn = quat_normalize(quat_multiply(q_signal, orn))
        poses.append((pos.copy(), orn.copy()))

    hat_pos, hat_orn = poses[0]
    goal_pos, goal_orn = poses[-1]

    v_lin = (hat_pos - ee_pos) / dt
    v_ang = omega_from_quat_step(ee_orn, hat_orn, dt)
    v_ee = np.concatenate([v_lin, v_ang]).astype(np.float64)

    return {
        "active": True,
        "v_ee": v_ee,
        "hat_pos": hat_pos,
        "hat_orn": hat_orn,
        "goal_pos": goal_pos,
        "goal_orn": goal_orn,
        "n_steps": n_steps,
    }
