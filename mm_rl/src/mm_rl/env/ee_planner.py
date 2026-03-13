"""End-effector planner for generating desired velocities toward goal."""

import numpy as np

from mm_utils import math as mm_math


class EEPlanner:
    """Simple linear end-effector planner that generates velocities toward goal.

    The planner commands velocities at a fixed magnitude (set randomly on reset)
    independent of the RL agent's actions, acting as a surrogate teleoperator.

    During training, uses last desired pose instead of current actual pose to
    prevent RL agent from influencing the trajectory shape.
    """

    def __init__(
        self,
        goal_pos,
        goal_orn,
        vel_range=(0.3, 0.5),
        dt=0.03,
        np_random=None,
        slowdown_distance=0.1,
    ):
        """Initialize end-effector planner.

        Args:
            goal_pos: Goal position in world frame (3,)
            goal_orn: Goal orientation quaternion in world frame (4,)
            vel_range: Tuple (min, max) for random velocity selection (m/s)
            dt: Time step (s)
            np_random: Optional numpy random number generator
            slowdown_distance: Distance in meters from goal where planner starts slowing down
        """
        self.goal_pos = np.array(goal_pos)
        self.goal_orn = np.array(goal_orn)
        self.vel_range = vel_range
        self.dt = dt
        self.np_random = np_random
        self.slowdown_distance = slowdown_distance

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

    def step(self, scale=1.0):
        """Step planner forward using fixed velocity set on reset.

        The scale parameter is intended for the *commanded* EE velocity only.
        The internal desired pose is always advanced using the unscaled
        velocity defined by the motion generator. This means the reference
        trajectory (desired pose over time) is independent of the agent's
        current speed choice and corresponds to the "default" motion.

        Returns:
            tuple: (desired_lin_vel, desired_ang_vel) in world frame (3,), (3,)
                   These are unscaled; the caller applies `scale` when sending
                   commands to the robot/IK.
        """
        # Compute desired linear velocity toward goal from last desired pose
        pos_error = self.goal_pos - self.desired_pos
        pos_error_norm = np.linalg.norm(pos_error)

        if pos_error_norm > 1e-6:
            pos_dir = pos_error / pos_error_norm
            # Maintain full velocity until within slowdown_distance, then slow down
            if pos_error_norm > self.slowdown_distance:
                # Use full planner velocity when far from goal
                desired_lin_vel = pos_dir * self.planner_vel
            else:
                # Slow down proportionally when close to goal
                desired_lin_vel = (
                    pos_dir
                    * (pos_error_norm / self.slowdown_distance)
                    * self.planner_vel
                )
        else:
            # Reached goal position
            desired_lin_vel = np.zeros(3)

        # Compute desired angular velocity toward goal orientation
        orn_error = mm_math.quat_orientation_error(self.desired_orn, self.goal_orn)

        if orn_error > 1e-6:
            # Compute quaternion difference
            q_inv = mm_math.quat_inverse(self.desired_orn)
            q_diff = mm_math.quat_multiply(self.goal_orn, q_inv)

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

        # Update desired pose based on unscaled velocities (reference trajectory)
        self.desired_pos, self.desired_orn = self.integrate_pose(
            self.desired_pos,
            self.desired_orn,
            desired_lin_vel,
            desired_ang_vel,
            self.dt,
            scale=1.0,
        )

        return desired_lin_vel, desired_ang_vel

    @staticmethod
    def integrate_pose(pos, orn, lin_vel, ang_vel, dt, scale=1.0):
        """Integrate pose by one step given velocities.

        Args:
            pos: Position (3,) in world frame.
            orn: Orientation quaternion (4,) in world frame.
            lin_vel: Linear velocity (3,) in world frame.
            ang_vel: Angular velocity (3,) in world frame (rad/s).
            dt: Time step (s).
            scale: Scale applied to both velocities (e.g. ee_vel_scale). Use 1.0 for unscaled.

        Returns:
            tuple: (new_pos, new_orn) after integration.
        """
        pos = np.asarray(pos)
        orn = np.asarray(orn)
        scaled_lin = np.asarray(lin_vel) * scale
        scaled_ang = np.asarray(ang_vel) * scale
        new_pos = pos + scaled_lin * dt
        if np.linalg.norm(scaled_ang) > 1e-6:
            ang_vel_quat = np.zeros(4)
            ang_vel_quat[:3] = scaled_ang * dt / 2.0
            ang_vel_quat[3] = 1.0
            ang_vel_quat = ang_vel_quat / np.linalg.norm(ang_vel_quat)
            new_orn = mm_math.quat_multiply(orn, ang_vel_quat)
        else:
            new_orn = orn.copy()
        return new_pos, new_orn

    def get_desired_pose(self):
        """Get current desired pose from planner.

        Returns:
            tuple: (desired_pos, desired_orn) in world frame
        """
        return self.desired_pos.copy(), self.desired_orn.copy()
