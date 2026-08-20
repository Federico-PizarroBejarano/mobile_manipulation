"""Simple goal-reaching environment without obstacles."""

import numpy as np

from mm_rl.env.base_env import BaseRLEnv
from mm_rl.env.reward import compute_total_reward
from mm_utils import math as mm_math


class SimpleGoalEnv(BaseRLEnv):
    """Simple goal-reaching task without obstacles."""

    def __init__(self, config):
        """Initialize simple goal environment.

        Args:
            config (dict): Must include ``goal``, ``reward``, and base keys expected by
                :class:`BaseRLEnv`.
        """
        super().__init__(config)

        # Goal generation parameters
        self.goal_config = config.get("goal")
        self.goal_pos_range = self.goal_config.get("pos_range")
        self.goal_orn_range = self.goal_config.get("orn_range")

        # Success thresholds
        self.success_pos_threshold = self.goal_config.get("success_pos_threshold")
        self.success_orn_threshold = self.goal_config.get("success_orn_threshold")

        # Reward parameters (IK uses modulation_rl-style pos_scale/rot_scale normalization)
        self.reward_config = config.get("reward")
        self.ik_penalty_multiplier = self.reward_config.get("ik_penalty_multiplier")
        self.pos_scale = self.reward_config.get("pos_scale", 0.1)
        self.rot_scale = self.reward_config.get("rot_scale", 0.05)
        self.acceleration_penalty_multiplier = self.reward_config.get(
            "acceleration_penalty_multiplier"
        )
        self.base_action_penalty_multiplier = self.reward_config.get(
            "base_action_penalty_multiplier", 0.0
        )

        # Initialize goal
        self.goal_pos = None
        self.goal_orn = None

    def reset(self, seed=None, options=None):
        """Reset environment and generate new goal.

        Args:
            seed (int, optional): Forwarded to the parent reset for RNG.
            options (dict, optional): May set ``goal_pos`` and ``goal_orn`` together to
                skip random goal sampling; ``disable_early_termination`` is forwarded to
                :meth:`BaseRLEnv.reset`.

        Returns:
            tuple: ``(observation, info)`` with ``info`` also containing ``goal_pos`` and
            ``goal_orn`` copies.
        """
        options = options or {}
        if "goal_pos" in options and "goal_orn" in options:
            self.goal_pos = np.asarray(options["goal_pos"], dtype=np.float64).reshape(3)
            self.goal_orn = np.asarray(options["goal_orn"], dtype=np.float64).reshape(4)
        else:
            self._generate_goal()

        # Now reset (this will initialize the EE planner with the goal)
        obs, info = super().reset(seed=seed, options=options)

        # Update observation with new goal
        obs = self._get_observation()

        info["goal_pos"] = self.goal_pos.copy()
        info["goal_orn"] = self.goal_orn.copy()

        return obs, info

    def _generate_goal(self):
        """Generate a random goal within the specified range."""
        # Random position
        pos_min = np.array(self.goal_pos_range[0])
        pos_max = np.array(self.goal_pos_range[1])
        self.goal_pos = self.np_random.uniform(pos_min, pos_max)

        # Generate random quaternion using axis-angle representation
        # Random axis (uniform on unit sphere)
        axis = self.np_random.uniform(-1, 1, size=3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        # Random angle within range
        angle = self.np_random.uniform(0, self.goal_orn_range)
        # Convert to quaternion (xyzs order)
        self.goal_orn = np.array(
            [
                axis[0] * np.sin(angle / 2),
                axis[1] * np.sin(angle / 2),
                axis[2] * np.sin(angle / 2),
                np.cos(angle / 2),
            ]
        )

    def _compute_reward(self, action, prev_action):
        """Compute reward based on goal distance, IK quality, and action penalties.

        Args:
            action: Action vector (scaled, for observation)
            prev_action: Previous action vector (for acceleration penalty)

        Returns:
            float: Reward value
        """
        # Get current end-effector pose (achieved after IK)
        ee_pos_w, ee_orn_w = self.sim.robot.link_pose()
        base_pos_w, base_orn_w = self.sim.robot.link_pose(link_idx=-1)

        # Transform to base frame for reward computation
        ee_pos_b, ee_orn_b = self._world_to_base_frame(
            ee_pos_w, ee_orn_w, base_pos_w, base_orn_w
        )

        # Get desired EE pose directly from planner (for IK reward)
        # This uses the planner's tracked desired pose, not something derived from actual robot pose
        desired_ee_pos_w, desired_ee_orn_w = self.ee_planner.get_desired_pose()

        # Transform desired EE pose to base frame for IK reward
        desired_ee_pos_b, desired_ee_orn_b = self._world_to_base_frame(
            desired_ee_pos_w, desired_ee_orn_w, base_pos_w, base_orn_w
        )

        reward = compute_total_reward(
            ee_pos_b,
            ee_orn_b,
            action,
            prev_action,
            desired_ee_pos_b,
            desired_ee_orn_b,
            ik_penalty_multiplier=self.ik_penalty_multiplier,
            pos_scale=self.pos_scale,
            rot_scale=self.rot_scale,
            acceleration_penalty_multiplier=self.acceleration_penalty_multiplier,
            base_action_penalty_multiplier=self.base_action_penalty_multiplier,
        )

        return reward

    def _check_termination(self):
        """Check if episode should terminate.

        Checks for:
        1. Early termination: deviation from desired pose (checked by parent)
        2. Goal reached: current pose within success thresholds

        Returns:
            bool: True if episode should terminate
        """
        # First check for early termination (deviation-based)
        if super()._check_termination():
            return True

        # Then check if goal is reached
        ee_pos_w, ee_orn_w = self.sim.robot.link_pose()

        # Check position distance
        pos_error = np.linalg.norm(ee_pos_w - self.goal_pos)
        if pos_error > self.success_pos_threshold:
            return False

        # Check orientation distance
        orn_error = mm_math.quat_orientation_error(ee_orn_w, self.goal_orn)
        if orn_error > self.success_orn_threshold:
            return False

        # Goal reached!
        return True
