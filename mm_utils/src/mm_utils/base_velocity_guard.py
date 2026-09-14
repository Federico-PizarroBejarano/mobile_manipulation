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
