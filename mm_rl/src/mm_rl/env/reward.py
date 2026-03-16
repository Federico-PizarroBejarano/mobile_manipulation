"""Reward calculation functions for RL training."""

import numpy as np

from mm_utils import math as mm_math


def compute_ik_reward(
    achieved_ee_pos,
    achieved_ee_orn,
    desired_ee_pos,
    desired_ee_orn,
    ik_penalty_multiplier=1.0,
    pos_scale=0.1,
    rot_scale=0.05,
):
    """Compute reward/penalty based on IK solution quality.

    Uses modulation_rl-style normalization: scale position and rotation errors
    so both terms are comparable (-0.5 when at pos_scale/rot_scale away).

    Formula: -0.5 * ((pos_dist/pos_scale)^2 + (rot_dist/rot_scale)^2) * ik_penalty_multiplier

    Args:
        achieved_ee_pos: Achieved end-effector position (3,)
        achieved_ee_orn: Achieved end-effector orientation quaternion (4,)
        desired_ee_pos: Desired end-effector position from planner (3,)
        desired_ee_orn: Desired end-effector orientation from planner (4,)
        ik_penalty_multiplier: Multiplier for IK reward (λ_ik in paper)
        pos_scale: Scale for position error
        rot_scale: Scale for rotation error

    Returns:
        float: IK reward value (negative, closer to desired = less negative)
    """
    pos_dist = np.linalg.norm(achieved_ee_pos - desired_ee_pos)
    rot_dist = mm_math.quat_orientation_error(achieved_ee_orn, desired_ee_orn)

    scaled = -0.5 * ((pos_dist / pos_scale) ** 2 + (rot_dist / rot_scale) ** 2)
    return ik_penalty_multiplier * scaled


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


def compute_total_reward(
    ee_pos,
    ee_orn,
    action,
    prev_action,
    desired_ee_pos,
    desired_ee_orn,
    ik_penalty_multiplier=1.0,
    pos_scale=0.1,
    rot_scale=0.05,
    acceleration_penalty_multiplier=0.01,
    base_action_penalty_multiplier=0.0,
):
    """Compute total reward for a step.

    Reward formula: r = λ_ik * r_ik + r_acc + r_base
    with r_ik using normalized position/rotation (modulation_rl-style).

    Args:
        ee_pos: End-effector position (3,)
        ee_orn: End-effector orientation quaternion (4,)
        action: Action vector (base actions, already unscaled)
        prev_action: Previous action vector (for acceleration penalty)
        desired_ee_pos: Desired end-effector position from planner (3,) for IK reward
        desired_ee_orn: Desired end-effector orientation from planner (4,) for IK reward
        ik_penalty_multiplier: Multiplier for IK reward (λ_ik in paper)
        pos_scale: Scale for position error
        rot_scale: Scale for rotation error
        acceleration_penalty_multiplier: Multiplier for acceleration penalty (λ_acc in paper)
        base_action_penalty_multiplier: Multiplier for base action penalty (λ_base in paper)

    Returns:
        float: Total reward
    """
    # IK reward
    ik_reward_val = compute_ik_reward(
        ee_pos,
        ee_orn,
        desired_ee_pos,
        desired_ee_orn,
        ik_penalty_multiplier=ik_penalty_multiplier,
        pos_scale=pos_scale,
        rot_scale=rot_scale,
    )

    # Acceleration penalty
    acceleration_penalty_val = compute_acceleration_penalty(
        action, prev_action, acceleration_penalty_multiplier
    )

    base_action_penalty_val = -base_action_penalty_multiplier * np.sum(
        np.square(action)
    )

    return ik_reward_val + acceleration_penalty_val + base_action_penalty_val
