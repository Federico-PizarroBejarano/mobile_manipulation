"""Reward calculation functions for RL training."""

import numpy as np


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
    q_dot = np.abs(np.dot(achieved_ee_orn, desired_ee_orn))
    rot_dist = 1.0 - q_dot

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
):
    """Compute total reward for a step.

    Reward formula: r = λ_ik * r_ik + λ_acc * r_acc
    where r_ik is the IK penalty and r_acc is the acceleration penalty.

    Args:
        ee_pos: End-effector position (3,)
        ee_orn: End-effector orientation quaternion (4,)
        action: Action vector (base actions, already unscaled)
        prev_action: Previous action vector (for acceleration penalty)
        desired_ee_pos: Desired end-effector position from planner (3,) for IK reward
        desired_ee_orn: Desired end-effector orientation from planner (4,) for IK reward
        rot_weight: Weight for orientation error relative to position error in IK reward
        ik_penalty_multiplier: Multiplier for IK reward (λ_ik in paper)
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
        rot_weight,
        ik_penalty_multiplier,
    )

    # Acceleration penalty
    acceleration_penalty_val = compute_acceleration_penalty(
        action, prev_action, acceleration_penalty_multiplier
    )

    base_action_penalty_val = base_action_penalty_multiplier * np.sum(np.square(action))

    return ik_reward_val + acceleration_penalty_val + base_action_penalty_val
