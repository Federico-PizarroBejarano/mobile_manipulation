"""Shared helpers for MPSF rollouts, tests, and evaluation.

Keeps MPC setup, goal sampling, and Jacobian IK in one importable module so
evaluation scripts and tests do not import each other.
"""

import numpy as np

from mm_utils import math as mm_math
from mm_utils import parsing
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


def build_ik_params_from_config(config, nu, nq):
    """Build the parameter dict used by :func:`solve_ik` from merged YAML config.

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
    vel_lo = np.array(lim.get("lower", [0] * nu)[nq:])
    vel_hi = np.array(lim.get("upper", [0] * nu)[nq:])
    return {
        "nu": nu,
        "use_weighted_regularization": ik_cfg.get("use_weighted_regularization", True),
        "regularization_strength": ik_cfg.get("regularization_strength", 0.1),
        "joint_vel_lower": vel_lo,
        "joint_vel_upper": vel_hi,
        "ee_max_linear_vel": float(robot_cfg.get("ee_linear_vel_limit", 0.5)),
        "ee_max_angular_vel": float(robot_cfg.get("ee_angular_vel_limit", 0.75)),
    }


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


def solve_ik(robot, desired_ee_vel, base_vel, ik_params):
    """Compute joint velocities: base command plus arm velocities from Jacobian IK.

    Clamps ``desired_ee_vel`` with :func:`mm_utils.math.clamp_ee_velocity`, subtracts
    the end-effector velocity induced by ``base_vel`` via the base columns of the
    Jacobian, then solves for arm joint rates with weighted regularization or
    damped least squares. Result is clipped to ``joint_vel_lower`` /
    ``joint_vel_upper``.

    Args:
        robot: Simulator robot with ``joint_states()``, ``jacobian(q)``, and
            omnidirectional base in joints ``0:3``.
        desired_ee_vel (ndarray): 6D spatial EE velocity in the world frame
            ``[v_x, v_y, v_z, \\omega_x, \\omega_y, \\omega_z]``.
        base_vel (ndarray): Base velocities ``[v_x, v_y, \\omega_{yaw}]`` (3,),
            matching the first three controlled joints.
        ik_params (dict): From :func:`build_ik_params_from_config` (must include
            ``nu``, ``joint_vel_*``, ``use_weighted_regularization``,
            ``regularization_strength``, and optional ``ee_max_*`` clamp limits).

    Returns:
        ndarray: Joint velocity command ``u`` of shape ``(nu,)``.
    """
    nu = ik_params["nu"]
    jlo = ik_params["joint_vel_lower"]
    jhi = ik_params["joint_vel_upper"]
    v = mm_math.clamp_ee_velocity(
        desired_ee_vel,
        float(ik_params.get("ee_max_linear_vel", 0.5)),
        float(ik_params.get("ee_max_angular_vel", 0.75)),
    )
    q, _ = robot.joint_states()
    J = robot.jacobian(q)
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
    u = np.zeros(nu)
    u[base_idx] = base_vel
    u[arm_idx] = arm_vel
    for i in range(nu):
        u[i] = np.clip(u[i], jlo[i], jhi[i])
    return u
