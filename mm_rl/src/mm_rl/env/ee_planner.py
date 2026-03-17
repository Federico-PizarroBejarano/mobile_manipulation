"""Pose-path EE planner: straight-line position + SLERP orientation."""

import numpy as np

from mm_utils.math import omega_from_quat_step, quat_normalize, quat_slerp

# Fixed path discretization (not configurable)
MIN_PATH_STEPS = 300


class EEPlanner:
    """Linear position + SLERP orientation along path parameter s ∈ [0,1]."""

    def __init__(self, goal_pos, goal_orn, start_pos, start_orn, max_linear_speed, dt):
        """Build pose path from start EE pose to goal.

        Args:
            goal_pos: Goal position in world frame (3,)
            goal_orn: Goal orientation quaternion in world frame (4,)
            start_pos: Start EE position in world frame (3,)
            start_orn: Start EE orientation quaternion (4,)
            max_linear_speed: Maximum linear speed (m/s)
            dt: Control time step (s)
        """
        self.goal_pos = np.asarray(goal_pos, dtype=np.float64)
        self.goal_orn = quat_normalize(goal_orn)
        self.max_linear_speed = float(max_linear_speed)
        self.dt = float(dt)

        self._start_pos = np.asarray(start_pos, dtype=np.float64).reshape(3)
        self._start_orn = quat_normalize(start_orn)
        self._goal_pos = np.asarray(self.goal_pos, dtype=np.float64).reshape(3)
        self._goal_orn = quat_normalize(self.goal_orn)

        dist = float(np.linalg.norm(self._goal_pos - self._start_pos))
        self.n_path_steps = self._compute_n_eff(dist)
        self.delta_s = 1.0 / self.n_path_steps

        self.s = 0.0
        self.desired_pos = self._start_pos.copy()
        self.desired_orn = self._start_orn.copy()

    def _path_pose(self, s):
        """Compute position and orientation at path parameter s ∈ [0,1].

        Args:
            s: Path parameter, in [0, 1]

        Returns:
            tuple: (position, orientation) in world frame
        """
        s = float(np.clip(s, 0.0, 1.0))
        pos = (1.0 - s) * self._start_pos + s * self._goal_pos
        orn = quat_slerp(self._start_orn, self._goal_orn, s)
        return pos, quat_normalize(orn)

    def _compute_n_eff(self, dist):
        """Compute number of path steps needed to reach goal.

        Args:
            dist: Distance to goal (m)

        Returns:
            int: Number of path steps
        """
        v_max = self.max_linear_speed
        n_speed = max(1, int(np.ceil(dist / (v_max * self.dt + 1e-12))))
        n_eff = max(MIN_PATH_STEPS, n_speed)
        while dist / (n_eff * self.dt) > v_max + 1e-9:
            n_eff += 1
        return n_eff

    def step(self):
        """Compute desired end-effector velocity.

        Returns:
            tuple: (desired_lin_vel, desired_ang_vel) in world frame (3,), (3,)
        """
        if self.s >= 1.0 - 1e-12:
            self.s = 1.0
            self.desired_pos = self._goal_pos.copy()
            self.desired_orn = self._goal_orn.copy()
            return np.zeros(3), np.zeros(3)

        s_prev = self.s
        self.s = min(self.s + self.delta_s, 1.0)

        pos_prev, orn_prev = self._path_pose(s_prev)
        pos_new, orn_new = self._path_pose(self.s)

        v_lin = (pos_new - pos_prev) / self.dt
        v_ang = omega_from_quat_step(orn_prev, orn_new, self.dt)

        self.desired_pos = pos_new
        self.desired_orn = orn_new

        return v_lin, v_ang

    def get_desired_pose(self):
        """Get current desired pose from planner.

        Returns:
            tuple: (desired_pos, desired_orn) in world frame
        """
        return self.desired_pos.copy(), self.desired_orn.copy()
