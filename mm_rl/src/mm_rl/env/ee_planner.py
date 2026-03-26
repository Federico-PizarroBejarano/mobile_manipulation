"""Pose-path EE planner: straight-line position + SLERP orientation."""

import numpy as np

from mm_utils.math import omega_from_quat_step, quat_normalize, quat_slerp

# Fixed path discretization (not configurable)
MIN_PATH_STEPS = 300


class EEPlanner:
    """Linear position + SLERP orientation along path parameter s ∈ [0,1]."""

    def __init__(
        self,
        goal_pos,
        goal_orn,
        start_pos,
        start_orn,
        max_linear_speed,
        dt,
        planner_mode="open_loop",
    ):
        """Build pose path from start EE pose to goal.

        Args:
            goal_pos: Goal position in world frame (3,)
            goal_orn: Goal orientation quaternion in world frame (4,)
            start_pos: Start EE position in world frame (3,)
            start_orn: Start EE orientation quaternion (4,)
            max_linear_speed: Maximum linear speed (m/s)
            dt: Control time step (s)
            planner_mode: "open_loop" (path-parameter rollout) or
                "closed_loop" (reactive to current EE pose).
        """
        self.goal_pos = np.asarray(goal_pos, dtype=np.float64)
        self.goal_orn = quat_normalize(goal_orn)
        self.max_linear_speed = float(max_linear_speed)
        self.dt = float(dt)
        if planner_mode not in ("open_loop", "closed_loop"):
            raise ValueError(
                f"Unsupported planner_mode '{planner_mode}'. "
                "Expected 'open_loop' or 'closed_loop'."
            )
        self.planner_mode = planner_mode

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

    def step(self, current_pos=None, current_orn=None, linear_speed_override=None):
        """Compute desired end-effector velocity.

        For open_loop mode, velocities are generated from the internal path
        parameter regardless of measured EE state. For closed_loop mode, the
        planner reacts to the provided current EE pose.

        Args:
            current_pos (ndarray, optional): Measured EE position in world frame; required
                when ``planner_mode == "closed_loop"``.
            current_orn (ndarray, optional): Measured EE quaternion (xyzs); required
                when ``planner_mode == "closed_loop"``.
            linear_speed_override (float, optional): Runtime linear speed (m/s) used
                to scale planner progression (open-loop) or step bound (closed-loop).

        Returns:
            tuple: ``(desired_lin_vel, desired_ang_vel)`` in world frame, each shape ``(3,)``.
        """
        speed_scale = 1.0
        if linear_speed_override is not None:
            speed_scale = float(linear_speed_override) / (self.max_linear_speed + 1e-12)
            speed_scale = float(np.clip(speed_scale, 1e-4, 1.0))

        if self.planner_mode == "closed_loop":
            return self._step_closed_loop(current_pos, current_orn, speed_scale)

        if self.s >= 1.0 - 1e-12:
            self.s = 1.0
            self.desired_pos = self._goal_pos.copy()
            self.desired_orn = self._goal_orn.copy()
            return np.zeros(3), np.zeros(3)

        s_prev = self.s
        self.s = min(self.s + speed_scale * self.delta_s, 1.0)

        pos_prev, orn_prev = self._path_pose(s_prev)
        pos_new, orn_new = self._path_pose(self.s)

        v_lin = (pos_new - pos_prev) / self.dt
        v_ang = omega_from_quat_step(orn_prev, orn_new, self.dt)

        self.desired_pos = pos_new
        self.desired_orn = orn_new

        return v_lin, v_ang

    def _step_closed_loop(self, current_pos, current_orn, speed_scale):
        """Advance desired pose toward the goal using measured EE pose (closed-loop).

        Args:
            current_pos (ndarray): Current EE position in world frame, shape ``(3,)``.
            current_orn (ndarray): Current EE orientation quaternion (xyzs), shape ``(4,)``.

        Returns:
            tuple: ``(v_lin, v_ang)`` world-frame linear and angular velocity, shape ``(3,)`` each.
        """
        if current_pos is None or current_orn is None:
            raise ValueError(
                "closed_loop planner_mode requires current_pos and current_orn in step()."
            )

        pos_curr = np.asarray(current_pos, dtype=np.float64).reshape(3)
        orn_curr = quat_normalize(current_orn)
        pos_err = self._goal_pos - pos_curr
        dist = float(np.linalg.norm(pos_err))
        max_step = self.max_linear_speed * speed_scale * self.dt

        if dist <= 1e-12:
            pos_new = self._goal_pos.copy()
            alpha = self.delta_s
        else:
            step_len = min(dist, max_step)
            pos_new = pos_curr + (pos_err / dist) * step_len
            alpha = max(self.delta_s, step_len / dist)

        alpha = float(np.clip(alpha, 0.0, 1.0))
        orn_new = quat_slerp(orn_curr, self._goal_orn, alpha)
        orn_new = quat_normalize(orn_new)

        v_lin = (pos_new - pos_curr) / self.dt
        v_ang = omega_from_quat_step(orn_curr, orn_new, self.dt)

        self.desired_pos = pos_new
        self.desired_orn = orn_new
        return v_lin, v_ang

    def get_desired_pose(self):
        """Get current desired pose from planner.

        Returns:
            tuple: (desired_pos, desired_orn) in world frame
        """
        return self.desired_pos.copy(), self.desired_orn.copy()
