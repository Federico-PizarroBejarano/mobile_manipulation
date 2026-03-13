"""Reward calculation functions for RL training."""

import numpy as np

from mm_utils import math as mm_math


def compute_ik_reward(
    achieved_ee_pos,
    achieved_ee_orn,
    desired_ee_pos,
    desired_ee_orn,
    rot_weight=0.5,
    ik_penalty_multiplier=1.0,
):
    """Compute reward/penalty based on IK solution quality.

    This measures how well the achieved end-effector pose matches the desired
    pose from the planner.

    Args:
        achieved_ee_pos: Achieved end-effector position (3,)
        achieved_ee_orn: Achieved end-effector orientation quaternion (4,)
        desired_ee_pos: Desired end-effector position from planner (3,)
        desired_ee_orn: Desired end-effector orientation from planner (4,)
        rot_weight: Weight for orientation error relative to position error
        ik_penalty_multiplier: Multiplier for IK reward (λ_ik in paper)

    Returns:
        float: IK reward value (negative, closer to desired = less negative)
    """
    # Position distance
    pos_dist = np.linalg.norm(achieved_ee_pos - desired_ee_pos)

    # Orientation distance (quaternion distance)
    rot_dist = mm_math.quat_orientation_error(achieved_ee_orn, desired_ee_orn)

    ik_reward = -ik_penalty_multiplier * (pos_dist**2 + rot_weight * rot_dist)
    return ik_reward


def compute_acceleration_penalty(
    current_action, prev_action, acceleration_penalty_multiplier=0.01
):
    """Compute penalty for action changes (acceleration).

    This encourages smooth control by penalizing large changes in actions
    between consecutive steps.

    Args:
        current_action: Current action vector
        prev_action: Previous action vector
        acceleration_penalty_multiplier: Multiplier for acceleration penalty

    Returns:
        float: Acceleration penalty value (negative)
    """
    action_diff = current_action - prev_action
    return -acceleration_penalty_multiplier * np.sum(np.square(action_diff))


def compute_velocity_reward(ee_vel_scale, v_ee_max=1.0, vel_reward_multiplier=0.1):
    """Velocity reward to incentivize fast motions when possible (N²M² Eq. 6).

    r_vel = -(v_ee_max - n_ee)^2, with n_ee = ee_vel_scale * v_ee_max, so
    r_vel = -(v_ee_max * (1 - ee_vel_scale))^2. With v_ee_max=1: r_vel = -(1 - ee_vel_scale)^2.

    Args:
        ee_vel_scale: Normalized EE velocity scale in [ee_vel_scale_min, ee_vel_scale_max] (n_vel in paper).
        v_ee_max: Max EE velocity (paper sets equal to max base velocity); use 1 for normalized scale.
        vel_reward_multiplier: λ_vel in paper (Table II: 0.1).

    Returns:
        float: λ_vel * r_vel (negative when ee_vel_scale < 1).
    """
    r_vel = -((v_ee_max - ee_vel_scale) ** 2)
    return vel_reward_multiplier * r_vel


def compute_total_reward(
    ee_pos,
    ee_orn,
    action,
    prev_action,
    desired_ee_pos,
    desired_ee_orn,
    rot_weight=0.5,
    ik_penalty_multiplier=1.0,
    acceleration_penalty_multiplier=0.01,
    base_action_penalty_multiplier=0.0,
    base_action_dim=None,
    ee_vel_scale=1.0,
    v_ee_max=1.0,
    vel_reward_multiplier=0.0,
):
    """Compute total reward for a step (N²M² Eq. 8).

    r = n_vel * (λ_ik * r_ik + r_coll) + λ_vel * r_vel + λ_acc * r_acc + base_penalty.
    We omit r_coll. n_vel scales the task reward so it is "per distance"; r_vel encourages using speed.

    Args:
        ee_pos: End-effector position (3,)
        ee_orn: End-effector orientation quaternion (4,)
        action: Action vector (full, including ee_vel_scale if present)
        prev_action: Previous action vector (for acceleration penalty)
        desired_ee_pos: Desired end-effector position from planner (3,) for IK reward
        desired_ee_orn: Desired end-effector orientation from planner (4,) for IK reward
        rot_weight: Weight for orientation error relative to position error in IK reward
        ik_penalty_multiplier: Multiplier for IK reward (λ_ik in paper)
        acceleration_penalty_multiplier: Multiplier for acceleration penalty (λ_acc in paper)
        base_action_penalty_multiplier: Multiplier for base action penalty (λ_base in paper)
        base_action_dim: If set, only first this many action dims are penalized (e.g. 3 for base only).
        ee_vel_scale: n_vel in paper (normalized EE velocity scale, [ee_vel_scale_min, ee_vel_scale_max]).
            Used to scale task reward and for r_vel.
        v_ee_max: Max EE velocity (paper sets equal to max base velocity); use 1 for normalized scale.
        vel_reward_multiplier: λ_vel in paper (Table II: 0.1). If 0, velocity reward is omitted.

    Returns:
        float: Total reward
    """
    # IK reward (will be multiplied by n_vel per paper Eq. 8)
    ik_reward_val = compute_ik_reward(
        ee_pos,
        ee_orn,
        desired_ee_pos,
        desired_ee_orn,
        rot_weight,
        ik_penalty_multiplier,
    )

    # N²M²: task term scaled by n_vel (reward per distance, Eq. 7–8)
    task_term = ee_vel_scale * ik_reward_val

    # N²M²: velocity reward r_vel = -(v_ee_max - n_ee)^2 to incentivize fast motions (Eq. 6)
    vel_reward_val = compute_velocity_reward(
        ee_vel_scale,
        v_ee_max=v_ee_max,
        vel_reward_multiplier=vel_reward_multiplier,
    )

    # Acceleration penalty (all actions)
    acceleration_penalty_val = compute_acceleration_penalty(
        action, prev_action, acceleration_penalty_multiplier
    )

    action_for_base_penalty = (
        action[:base_action_dim] if base_action_dim is not None else action
    )
    base_action_penalty_val = -base_action_penalty_multiplier * np.sum(
        np.square(action_for_base_penalty)
    )

    return (
        task_term + vel_reward_val + acceleration_penalty_val + base_action_penalty_val
    )
