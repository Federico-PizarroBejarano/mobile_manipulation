"""End-effector planner for generating desired velocities toward goal."""

import numpy as np

from mm_utils import math


class EEPlanner:
    """Simple linear end-effector planner that generates velocities toward goal.

    The planner commands velocities at a fixed magnitude (set randomly on reset)
    independent of the RL agent's actions, acting as a surrogate teleoperator.
    """

    def __init__(
        self, goal_pos, goal_orn, vel_range=(0.3, 0.5), dt=0.03, np_random=None
    ):
        """Initialize end-effector planner.

        Args:
            goal_pos: Goal position in world frame (3,)
            goal_orn: Goal orientation quaternion in world frame (4,)
            vel_range: Tuple (min, max) for random velocity selection (m/s)
            dt: Time step (s)
            np_random: Optional numpy random number generator
        """
        self.goal_pos = np.array(goal_pos)
        self.goal_orn = np.array(goal_orn)
        self.vel_range = vel_range
        self.dt = dt
        self.np_random = np_random

        # Set on reset
        self.current_pos = None
        self.current_orn = None
        self.planner_vel = None

    def reset(self, current_pos, current_orn):
        """Reset planner with current end-effector pose.

        Args:
            current_pos: Current end-effector position (3,)
            current_orn: Current end-effector orientation quaternion (4,)
        """
        self.update_current_pose(current_pos, current_orn)
        # Set random fixed velocity for this episode
        self.planner_vel = self.np_random.uniform(self.vel_range[0], self.vel_range[1])

    def update_current_pose(self, current_pos, current_orn):
        """Update planner's current pose with actual robot pose.

        Args:
            current_pos: Actual current end-effector position (3,)
            current_orn: Actual current end-effector orientation quaternion (4,)
        """
        self.current_pos = np.array(current_pos)
        self.current_orn = np.array(current_orn)

    def step(self):
        """Step planner forward using fixed velocity set on reset.

        Returns desired end-effector velocity command (teleoperator command).

        Returns:
            tuple: (desired_lin_vel, desired_ang_vel) in world frame (3,), (3,)
        """
        # Compute desired linear velocity toward goal
        pos_error = self.goal_pos - self.current_pos
        pos_error_norm = np.linalg.norm(pos_error)

        if pos_error_norm > 1e-6:
            # Move toward goal at fixed velocity
            pos_dir = pos_error / pos_error_norm
            desired_lin_vel = pos_dir * min(pos_error_norm / 2.0, self.planner_vel)
        else:
            # Reached goal position
            desired_lin_vel = np.zeros(3)

        # Compute desired angular velocity toward goal orientation
        q_dot = np.abs(np.dot(self.current_orn, self.goal_orn))
        orn_error = 1.0 - q_dot

        if orn_error > 1e-6:
            # Compute quaternion difference
            q_inv = math.quat_inverse(self.current_orn)
            q_diff = math.quat_multiply(self.goal_orn, q_inv)

            # Convert to axis-angle for angular velocity
            q_diff_norm = np.linalg.norm(q_diff[:3])
            if q_diff_norm > 1e-6:
                axis = q_diff[:3] / q_diff_norm
                angle = 2 * np.arccos(np.clip(q_diff[3], -1, 1))

                # Angular velocity scale: rad/s per m/s
                ang_vel_scale = 0.5
                max_ang_vel = self.planner_vel * ang_vel_scale

                # Desired angular velocity magnitude
                # If we're close, reduce velocity proportionally
                desired_ang_vel_mag = min(angle / self.dt, max_ang_vel)
                desired_ang_vel = axis * desired_ang_vel_mag
            else:
                desired_ang_vel = np.zeros(3)
        else:
            # Reached goal orientation
            desired_ang_vel = np.zeros(3)

        return desired_lin_vel, desired_ang_vel
