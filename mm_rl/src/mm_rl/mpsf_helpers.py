"""Shared helpers for MPSF rollouts, tests, and evaluation.

Keeps MPC setup, goal sampling, and Jacobian IK in one importable module so
evaluation scripts and tests do not import each other.
"""

import numpy as np

from mm_utils import math as mm_math
from mm_utils import parsing
from mm_utils.diff_ik import build_ik_params_from_config  # noqa: F401
from mm_utils.diff_ik import solve_diff_ik, spatial_jacobian
from mm_utils.parsing import recursive_dict_update


def ensure_controller_config(config):
    """Ensure a full MPC controller block and scene entry exist on ``config``.

    If ``config`` has no ``controller`` or ``controller.type`` is not ``'MPC'``,
    loads ``mm_run/config/controller/MPC.yaml`` and merges it into
    ``config[\"controller\"]``. If ``controller.scene`` is missing, loads
    ``mm_run/config/scene/empty.yaml`` and attaches a default scene dict.

    Args:
        config (dict): Top-level experiment config (mutated in place). Expected
            to follow the same structure as merged YAML (e.g. training config).

    Returns:
        None
    """
    if "controller" not in config or config.get("controller", {}).get("type") != "MPC":
        path = parsing.parse_ros_path(
            {"package": "mm_run", "path": "config/controller/MPC.yaml"}
        )
        ctrl = parsing.load_config(path)
        config.setdefault("controller", {})
        config["controller"] = recursive_dict_update(
            ctrl.get("controller", {}), config["controller"]
        )
    if "scene" not in config.get("controller", {}):
        path = parsing.parse_ros_path(
            {"package": "mm_run", "path": "config/scene/empty.yaml"}
        )
        scene = parsing.load_config(path)
        config["controller"]["scene"] = scene.get("controller", {}).get(
            "scene",
            {
                "enabled": False,
                "collision_link_names": {"static_obstacles": ["ground"]},
            },
        )


def setup_mpsf_config(ctrl_config):
    """Mutate ``ctrl_config`` for MPSF-style evaluation (EE tracking, free base).

    Sets base and arm command masks to zero, enables full MPSF EE velocity tracking
    with increased weights, tightens ``EEVel`` cost matrices, and ensures a minimal
    ``scene`` entry exists.

    Args:
        ctrl_config (dict): The ``config["controller"]`` subtree (mutated in place).

    Returns:
        None
    """
    ctrl_config["base_mask"] = [0, 0, 0]
    ctrl_config["ee_mask"] = [0, 0, 0, 0, 0, 0]
    ctrl_config.setdefault("mpsf_params", {})
    ctrl_config["mpsf_params"]["mpsf_base_mask"] = [0, 0, 0]
    ctrl_config["mpsf_params"]["mpsf_ee_mask"] = [1, 1, 1, 1, 1, 1]
    ctrl_config["mpsf_params"]["mpsf_weight_multiplier"] = 100.0
    ctrl_config.setdefault("cost_params", {})
    ctrl_config["cost_params"].setdefault("EEVel", {})
    ctrl_config["cost_params"]["EEVel"]["Qk"] = [10, 10, 10, 10, 10, 10]
    ctrl_config["cost_params"]["EEVel"]["P"] = [10, 10, 10, 10, 10, 10]
    ctrl_config.setdefault(
        "scene",
        {"enabled": False, "collision_link_names": {"static_obstacles": ["ground"]}},
    )


def generate_goal(pos_range, orn_range, np_random):
    """Sample a random goal position and orientation quaternion.

    Position is uniform in the axis-aligned box given by ``pos_range``.
    Orientation is uniform on the sphere of rotations with rotation angle in
    ``[0, orn_range]`` radians about a uniformly random axis (xyzs quaternion).

    Args:
        pos_range (list): ``[min, max]`` with each a length-3 array-like
            ``[x, y, z]`` in meters.
        orn_range (float): Maximum rotation angle magnitude in radians (``0`` gives identity).
        np_random (numpy.random.RandomState): RNG instance for reproducibility.

    Returns:
        tuple: ``(goal_pos, goal_orn)`` where ``goal_pos`` is shape ``(3,)`` and
            ``goal_orn`` is shape ``(4,)`` xyzs unit quaternion.
    """
    pos_min = np.array(pos_range[0])
    pos_max = np.array(pos_range[1])
    goal_pos = np_random.uniform(pos_min, pos_max)

    max_angle = float(orn_range)
    axis = np_random.uniform(-1, 1, size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    angle = np_random.uniform(0, max_angle)
    goal_orn = np.array(
        [
            axis[0] * np.sin(angle / 2),
            axis[1] * np.sin(angle / 2),
            axis[2] * np.sin(angle / 2),
            np.cos(angle / 2),
        ]
    )
    return goal_pos, goal_orn


def solve_ik(robot_mdl, q, desired_ee_vel_world, base_vel, ik_params):
    """Compute joint velocities via Casadi spatial Jacobian IK.

    Converts world-frame angular velocity to body frame to match
    ``MobileManipulator3D`` spatial Jacobian convention, then calls
    :func:`solve_diff_ik`.

    Args:
        robot_mdl: ``MobileManipulator3D`` instance.
        q (ndarray): Joint configuration, shape ``(nq,)``.
        desired_ee_vel_world (ndarray): 6D EE twist (world linear + world angular).
        base_vel (ndarray): Base velocities ``[vx, vy, vyaw]`` (3,).
        ik_params (dict): From :func:`build_ik_params_from_config`.

    Returns:
        ndarray: Joint velocity command ``u`` of shape ``(nu,)``.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    tw = np.asarray(desired_ee_vel_world, dtype=float).reshape(6)
    _, ee_orn = robot_mdl.getEE(q)
    _, twist_ik = mm_math.ee_twist_world_and_mpc_reference(
        tw[:3],
        tw[3:],
        ee_orn,
        clamp_limits=(
            float(ik_params.get("ee_max_linear_vel", 0.5)),
            float(ik_params.get("ee_max_angular_vel", 0.75)),
        ),
    )
    J = spatial_jacobian(robot_mdl, q)
    return solve_diff_ik(J, twist_ik, base_vel, ik_params)
