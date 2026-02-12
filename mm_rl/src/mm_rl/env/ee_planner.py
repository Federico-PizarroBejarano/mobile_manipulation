"""End-effector planner for generating desired velocities toward goal."""

import numpy as np

from mm_utils import math


class EEPlanner:
    """Simple linear end-effector planner that generates velocities toward goal.

    The planner commands velocities at a fixed magnitude (set randomly on reset)
    independent of the RL agent's actions, acting as a surrogate teleoperator.

    During training, uses last desired pose instead of current actual pose to
    prevent RL agent from influencing the trajectory shape.
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

        # Track desired pose (not actual robot pose)
        self.desired_pos = None
        self.desired_orn = None
        self.planner_vel = None

    def reset(self, current_pos, current_orn):
        """Reset planner with current end-effector pose.

        Initializes desired pose to current actual pose at reset.

        Args:
            current_pos: Current end-effector position (3,)
            current_orn: Current end-effector orientation quaternion (4,)
        """
        # Initialize desired pose to current pose at reset
        self.desired_pos = np.array(current_pos)
        self.desired_orn = np.array(current_orn)
        # Set random fixed velocity for this episode
        self.planner_vel = self.np_random.uniform(self.vel_range[0], self.vel_range[1])

    def step(self):
        """Step planner forward using fixed velocity set on reset.

        Returns desired end-effector velocity command (teleoperator command).

        Returns:
            tuple: (desired_lin_vel, desired_ang_vel) in world frame (3,), (3,)
        """
        # Compute desired linear velocity toward goal from last desired pose
        pos_error = self.goal_pos - self.desired_pos
        pos_error_norm = np.linalg.norm(pos_error)

        if pos_error_norm > 1e-6:
            # Move toward goal at fixed velocity
            pos_dir = pos_error / pos_error_norm
            desired_lin_vel = pos_dir * min(pos_error_norm / 2.0, self.planner_vel)
        else:
            # Reached goal position
            desired_lin_vel = np.zeros(3)

        # Compute desired angular velocity toward goal orientation
        q_dot = np.abs(np.dot(self.desired_orn, self.goal_orn))
        orn_error = 1.0 - q_dot

        if orn_error > 1e-6:
            # Compute quaternion difference
            q_inv = math.quat_inverse(self.desired_orn)
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
                desired_ang_vel_mag = min(angle / self.dt, max_ang_vel)
                desired_ang_vel = axis * desired_ang_vel_mag
            else:
                desired_ang_vel = np.zeros(3)
        else:
            # Reached goal orientation
            desired_ang_vel = np.zeros(3)

        # Update desired pose based on computed velocities
        self.desired_pos = self.desired_pos + desired_lin_vel * self.dt

        # Update desired orientation using quaternion integration
        if np.linalg.norm(desired_ang_vel) > 1e-6:
            # Quaternion derivative: q_dot = 0.5 * q * [0, wx, wy, wz]
            # For small rotations: q_new ≈ q * [1, 0.5*dt*wx, 0.5*dt*wy, 0.5*dt*wz]
            ang_vel_quat = np.zeros(4)
            ang_vel_quat[:3] = desired_ang_vel * self.dt / 2.0
            ang_vel_quat[3] = 1.0
            # Normalize the delta quaternion
            ang_vel_quat = ang_vel_quat / np.linalg.norm(ang_vel_quat)
            # quat_multiply normalizes by default
            self.desired_orn = math.quat_multiply(self.desired_orn, ang_vel_quat)

        return desired_lin_vel, desired_ang_vel

    def get_desired_pose(self):
        """Get current desired pose from planner.

        Returns:
            tuple: (desired_pos, desired_orn) in world frame
        """
        return self.desired_pos.copy(), self.desired_orn.copy()

    def get_intermediate_goal(self, distance_ahead=1.5):
        """Get intermediate goal roughly distance_ahead meters ahead along trajectory.

        N²M² paper: "we do not let the agent observe the final end-effector goal which
        can often be far away, but we repeatedly apply the end-effector motion generator
        fee to generate an intermediate goal roughly 1.5 m ahead of the agent."

        Args:
            distance_ahead: Distance ahead along trajectory in meters (default 1.5m)

        Returns:
            tuple: (intermediate_pos, intermediate_orn) in world frame
        """
        # Compute direction from current desired pose to final goal
        pos_error = self.goal_pos - self.desired_pos
        pos_error_norm = np.linalg.norm(pos_error)

        if pos_error_norm < 1e-6:
            # Already at goal, return goal itself
            return self.goal_pos.copy(), self.goal_orn.copy()

        # Direction toward final goal
        pos_dir = pos_error / pos_error_norm

        # Move distance_ahead along the trajectory (or less if closer to goal)
        distance = min(distance_ahead, pos_error_norm)
        intermediate_pos = self.desired_pos + pos_dir * distance

        # Interpolate orientation based on progress toward goal
        # Use SLERP (spherical linear interpolation) for quaternions
        progress = distance / max(pos_error_norm, 1e-6)
        progress = np.clip(progress, 0.0, 1.0)

        # Quaternion interpolation using SLERP
        goal_orn_copy = self.goal_orn.copy()
        q_dot = np.dot(self.desired_orn, goal_orn_copy)
        # Handle quaternion double cover (q and -q represent same rotation)
        if q_dot < 0:
            goal_orn_copy = -goal_orn_copy
            q_dot = -q_dot

        # SLERP formula
        theta = np.arccos(np.clip(np.abs(q_dot), -1.0, 1.0))
        if theta < 1e-6:
            # Quaternions are very close, just use goal orientation
            intermediate_orn = goal_orn_copy
        else:
            sin_theta = np.sin(theta)
            w1 = np.sin((1 - progress) * theta) / sin_theta
            w2 = np.sin(progress * theta) / sin_theta
            intermediate_orn = w1 * self.desired_orn + w2 * goal_orn_copy
            # Normalize
            intermediate_orn = intermediate_orn / np.linalg.norm(intermediate_orn)

        return intermediate_pos, intermediate_orn
