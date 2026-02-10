"""Simple goal-reaching environment without obstacles."""

import numpy as np

from mm_rl.env.base_env import BaseRLEnv
from mm_rl.env.reward import compute_total_reward
from mm_utils import math


class SimpleGoalEnv(BaseRLEnv):
    """Simple goal-reaching task without obstacles."""

    def __init__(self, config):
        """Initialize simple goal environment.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Goal generation parameters
        self.goal_config = config.get("goal")
        self.goal_pos_range = self.goal_config.get("pos_range")
        self.goal_orn_range = self.goal_config.get("orn_range")

        # Success thresholds
        self.success_pos_threshold = self.goal_config.get("success_pos_threshold")
        self.success_orn_threshold = self.goal_config.get("success_orn_threshold")

        # Reward parameters
        self.reward_config = config.get("reward")
        self.rot_weight = self.reward_config.get("rot_weight")
        self.ik_penalty_multiplier = self.reward_config.get("ik_penalty_multiplier")
        self.acceleration_penalty_multiplier = self.reward_config.get(
            "acceleration_penalty_multiplier"
        )

        # Initialize goal
        self.goal_pos = None
        self.goal_orn = None

    def reset(self, seed=None, options=None):
        """Reset environment and generate new goal.

        Args:
            seed: Random seed
            options: Optional dict with reset options

        Returns:
            observation, info
        """
        # Generate random goal first (before calling super().reset())
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
        max_angle = float(self.goal_orn_range)

        # Random axis (uniform on unit sphere)
        axis = self.np_random.uniform(-1, 1, size=3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        # Random angle within range
        angle = self.np_random.uniform(0, max_angle)
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

        # Compute desired EE pose from velocity command (for IK reward)
        # Integrate velocity to get desired next pose
        dt = float(self.sim.timestep)
        if np.linalg.norm(self.desired_ee_vel) > 0:
            # Linear: integrate velocity
            desired_ee_pos_w = ee_pos_w + self.desired_ee_vel[:3] * dt

            # Angular: convert angular velocity to quaternion increment
            ang_vel = self.desired_ee_vel[3:]
            ang_vel_norm = np.linalg.norm(ang_vel)
            if ang_vel_norm > 1e-6:
                # Create rotation quaternion from angular velocity
                angle = ang_vel_norm * dt
                axis = ang_vel / ang_vel_norm
                q_inc = np.array(
                    [
                        axis[0] * np.sin(angle / 2),
                        axis[1] * np.sin(angle / 2),
                        axis[2] * np.sin(angle / 2),
                        np.cos(angle / 2),
                    ]
                )
                desired_ee_orn_w = math.quat_multiply(q_inc, ee_orn_w)
            else:
                desired_ee_orn_w = ee_orn_w.copy()
        else:
            # No velocity command, desired pose is current pose
            desired_ee_pos_w = ee_pos_w.copy()
            desired_ee_orn_w = ee_orn_w.copy()

        # Transform desired EE pose to base frame for IK reward
        desired_ee_pos_b, desired_ee_orn_b = self._world_to_base_frame(
            desired_ee_pos_w, desired_ee_orn_w, base_pos_w, base_orn_w
        )

        # Compute reward: r = λ_ik * r_ik + λ_acc * r_acc
        reward = compute_total_reward(
            ee_pos_b,
            ee_orn_b,
            action,  # action: action vector
            prev_action,  # prev_action: previous action vector
            desired_ee_pos_b,  # desired_ee_pos: desired pose from planner
            desired_ee_orn_b,  # desired_ee_orn: desired orientation from planner
            rot_weight=self.rot_weight,
            ik_penalty_multiplier=self.ik_penalty_multiplier,
            acceleration_penalty_multiplier=self.acceleration_penalty_multiplier,
        )

        return reward

    def _check_termination(self):
        """Check if goal is reached.

        Returns:
            bool: True if goal is reached
        """
        # Get current end-effector pose
        ee_pos_w, ee_orn_w = self.sim.robot.link_pose()

        # Check position distance
        pos_error = np.linalg.norm(ee_pos_w - self.goal_pos)
        if pos_error > self.success_pos_threshold:
            return False

        # Check orientation distance
        q_dot = np.abs(np.dot(ee_orn_w, self.goal_orn))
        orn_error = 1.0 - q_dot
        if orn_error > self.success_orn_threshold:
            return False

        # Goal reached!
        return True
