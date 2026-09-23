"""Base Gym environment wrapper for BulletSimulation."""

from datetime import datetime

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mm_control.robot import MobileManipulator3D
from mm_rl.env.ee_planner import EEPlanner
from mm_simulator.simulation import BulletSimulation
from mm_utils import math as mm_math
from mm_utils.diff_ik import (
    build_ik_params_from_config,
    solve_diff_ik,
    spatial_jacobian,
)


class BaseRLEnv(gym.Env):
    """Base Gym environment wrapping BulletSimulation for RL training."""

    def __init__(self, config):
        """Initialize base RL environment.

        Args:
            config (dict): Simulation, controller, planner, IK, reward, and termination
                settings (see YAML merged by :mod:`mm_utils.parsing`).
        """
        super().__init__()

        # Store config
        self.config = config
        self.sim_config = config.get("simulation")
        # Robot config might be nested under controller.robot
        self.robot_config = config.get("controller").get("robot")
        self.planner_mode = config.get("planner_mode", "open_loop")

        # Initialize simulation
        timestamp = datetime.now()
        self.sim = BulletSimulation(self.sim_config, timestamp, cli_args=None)

        # Get robot dimensions
        self.nq = self.sim.robot.nq  # Number of joint positions
        self.nv = self.sim.robot.nv  # Number of joint velocities
        self.nu = self.sim.robot.nu  # Number of inputs

        # Casadi kinematics (same model as hardware teleop / MPC)
        ctrl = dict(config.get("controller", {}))
        if "dt" not in ctrl:
            ctrl["dt"] = float(self.sim_config.get("timestep", 0.03))
        self.robot_mdl = MobileManipulator3D(ctrl)
        self.ik_params = build_ik_params_from_config(config, self.nu, self.nq)
        # Mutable mirrors for tests / sweeps (synced into ik_params in _solve_ik)
        self.ik_regularization_strength = float(
            self.ik_params["regularization_strength"]
        )
        self.use_weighted_regularization = bool(
            self.ik_params["use_weighted_regularization"]
        )

        # Action space: N²M² approach - 3D action space
        # [base_x, base_y, base_yaw]
        # All actions in range [-1, 1], will be scaled appropriately
        self.action_space = spaces.Box(
            low=-np.ones(3),
            high=np.ones(3),
            dtype=np.float32,
        )

        # Action scaling parameters (base velocities: x, y, yaw)
        input_limits = self.robot_config.get("limits").get("input")
        self.base_input_low = np.array(input_limits["lower"][:3])
        self.base_input_high = np.array(input_limits["upper"][:3])
        # Optional override to match modulation_rl speeds (~0.2 m/s linear, ~0.75 rad/s yaw)
        robot_overrides = config.get("robot", {})
        if "base_linear_vel_limit" in robot_overrides:
            lim = float(robot_overrides["base_linear_vel_limit"])
            self.base_input_low[0] = -lim
            self.base_input_high[0] = lim
            self.base_input_low[1] = -lim
            self.base_input_high[1] = lim
        if "base_angular_vel_limit" in robot_overrides:
            lim = float(robot_overrides["base_angular_vel_limit"])
            self.base_input_low[2] = -lim
            self.base_input_high[2] = lim
        self.ee_max_linear_vel = float(robot_overrides.get("ee_linear_vel_limit", 0.5))
        self.ee_max_angular_vel = float(
            robot_overrides.get("ee_angular_vel_limit", 0.75)
        )
        # Keep ik_params EE clamps in sync with robot overrides
        self.ik_params["ee_max_linear_vel"] = self.ee_max_linear_vel
        self.ik_params["ee_max_angular_vel"] = self.ee_max_angular_vel

        # Clamp joint velocities to limits from config
        velocity_limits = self.robot_config.get("limits").get("state")
        self.joint_vel_lower = np.array(velocity_limits["lower"][self.nq :])
        self.joint_vel_upper = np.array(velocity_limits["upper"][self.nq :])

        planner_config = config.get("planner", {})
        self.planner_max_linear_speed = float(planner_config["max_linear_speed"])

        # End-effector planner (will be initialized in reset)
        self.ee_planner = None

        # Store previous action for observation (a_{t-1})
        self.prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)

        # Observation space will be defined by subclasses
        obs_dim = self._get_observation_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Episode tracking
        self.max_episode_steps = self.sim_config.get("max_episode_steps")
        self.current_step = 0

        # Goal (will be set by task)
        self.goal_pos = None
        self.goal_orn = None

        # Early termination: modulation_rl-style slack + cumulative count (no reset when back within slack)
        termination_config = config.get("termination", {})
        self.slack_pos_threshold = termination_config.get("slack_pos_threshold", 0.1)
        self.slack_rot_threshold = termination_config.get("slack_rot_threshold", 0.05)
        self.ik_fail_thresh = termination_config.get("ik_fail_thresh", 20)
        self.nr_kin_failures = 0

    def _get_observation_dim(self):
        """Get observation dimension. Override in subclasses if needed.

        Observation structure per paper: [v_{ee}, ee, \hat{ee}, g, s_{robot}, a_{t-1}]
        - v_{ee}: EE velocities (6D: 3D linear + 3D angular)
        - ee: current EE pose (12D: 3D position + 9D rotation matrix)
        - \hat{ee}: desired EE pose (12D: 3D position + 9D rotation matrix)
        - g: goal pose (12D: 3D position + 9D rotation matrix)
        - s_{robot}: joint positions (nq)
        - a_{t-1}: previous action (action_dim)

        Returns:
            int: Flat observation size ``6 + 12*3 + nq + action_dim``.
        """
        return 6 + 12 + 12 + 12 + self.nq + self.action_space.shape[0]

    def _ee_pose_w(self, q=None):
        """Casadi tool pose in world frame."""
        if q is None:
            q, _ = self.sim.robot.joint_states()
        return self.robot_mdl.getEE(np.asarray(q, dtype=float).reshape(-1))

    def _base_pose_w(self, q=None):
        """Base pose in world frame from planar ``q[:3]``."""
        if q is None:
            q, _ = self.sim.robot.joint_states()
        q = np.asarray(q, dtype=float).reshape(-1)
        yaw = float(q[2])
        base_pos = np.array([q[0], q[1], 0.0], dtype=np.float64)
        base_orn = np.array(
            [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=np.float64
        )
        return base_pos, base_orn

    def _get_observation(self):
        """Get current observation matching paper structure.

        Observation: [v_{ee}, ee, \hat{ee}, g, s_{robot}, a_{t-1}]
        - v_{ee}: EE velocities from planner (6D: linear + angular in base frame)
        - ee: current EE pose (12D: position + rotation matrix in base frame)
        - \hat{ee}: desired EE pose from planner (12D: position + rotation matrix in base frame)
        - g: goal pose (12D: position + rotation matrix in base frame)
        - s_{robot}: joint positions (nq)
        - a_{t-1}: previous action (action_dim)

        Returns:
            np.ndarray: Observation vector
        """
        # Get joint states (only positions, not velocities)
        q, _ = self.sim.robot.joint_states()

        # Casadi FK (matches hardware)
        ee_pos_w, ee_orn_w = self._ee_pose_w(q)
        base_pos_w, base_orn_w = self._base_pose_w(q)

        # Transform current end-effector pose to base frame
        ee_pos_b, ee_orn_b = self._world_to_base_frame(
            ee_pos_w, ee_orn_w, base_pos_w, base_orn_w
        )

        # Get desired EE pose directly from planner (already computed in step())
        # This uses the planner's tracked desired pose, not the actual robot pose
        desired_ee_pos_w, desired_ee_orn_w = self.ee_planner.get_desired_pose()

        # Transform desired end-effector pose to base frame
        desired_ee_pos_b, desired_ee_orn_b = self._world_to_base_frame(
            desired_ee_pos_w, desired_ee_orn_w, base_pos_w, base_orn_w
        )

        # Final goal in base frame
        goal_pos_b, goal_orn_b = self._world_to_base_frame(
            self.goal_pos, self.goal_orn, base_pos_w, base_orn_w
        )

        # Get EE velocities (v_{ee}) from planner command (teleoperator), not actual robot velocity
        # Transform commanded velocity to base frame
        ee_lin_vel_b = mm_math.quat_rotate(
            mm_math.quat_inverse(base_orn_w), self.desired_ee_vel[:3]
        )
        ee_ang_vel_b = mm_math.quat_rotate(
            mm_math.quat_inverse(base_orn_w), self.desired_ee_vel[3:]
        )
        ee_velocities = np.concatenate(
            [ee_lin_vel_b, ee_ang_vel_b]
        )  # 6D: [linear(3), angular(3)]

        # Concatenate observation per paper structure
        obs = np.concatenate(
            [
                ee_velocities,  # v_{ee}: EE velocities (6D)
                ee_pos_b,  # ee: current EE position (3D)
                mm_math.quat_to_rot(
                    ee_orn_b
                ).flatten(),  # ee: current EE rotation matrix (9D)
                desired_ee_pos_b,  # \hat{ee}: desired EE position (3D)
                mm_math.quat_to_rot(
                    desired_ee_orn_b
                ).flatten(),  # \hat{ee}: desired EE rotation matrix (9D)
                goal_pos_b,  # g: goal position (3D)
                mm_math.quat_to_rot(
                    goal_orn_b
                ).flatten(),  # g: goal rotation matrix (9D)
                q,  # s_{robot}: joint positions (nq)
                self.prev_action,  # a_{t-1}: previous action (action_dim)
            ]
        )

        return obs.astype(np.float32)

    def _world_to_base_frame(self, pos_w, orn_w, base_pos_w, base_orn_w):
        """Transform pose from world frame to base frame.

        Args:
            pos_w: Position in world frame (3,)
            orn_w: Orientation in world frame (4,) quaternion
            base_pos_w: Base position in world frame (3,)
            base_orn_w: Base orientation in world frame (4,) quaternion

        Returns:
            tuple: (pos_b, orn_b) position and orientation in base frame
        """
        # Position: transform to base frame
        pos_rel = pos_w - base_pos_w
        q_base_inv = mm_math.quat_inverse(base_orn_w)
        pos_b = mm_math.quat_rotate(q_base_inv, pos_rel)

        # Orientation: relative rotation
        orn_b = mm_math.quat_multiply(q_base_inv, orn_w)

        return pos_b, orn_b

    def reset(self, seed=None, options=None):
        """Reset environment.

        Args:
            seed (int, optional): Passed to :meth:`gymnasium.Env.reset` for RNG.
            options (dict, optional): May include ``disable_early_termination`` (bool):
                when True, :meth:`_check_termination` never fires from slack failures.

        Returns:
            tuple: ``(observation, info)`` with ``observation`` shape matching
            :attr:`observation_space` and ``info`` containing at least ``step`` and ``time``.
        """
        super().reset(seed=seed)

        # Reset to home (no start randomization)
        self.sim.robot.reset_joint_configuration(self.sim.robot.home)

        # Reset step counter
        self.current_step = 0

        # Reset slack-failure count (cumulative per episode)
        self.nr_kin_failures = 0

        opts = options or {}
        self.disable_early_termination = bool(
            opts.get("disable_early_termination", False)
        )

        # Initialize/reset end-effector planner if goal is set
        ee_pos, ee_orn = self._ee_pose_w()
        self.ee_planner = EEPlanner(
            self.goal_pos,
            self.goal_orn,
            ee_pos,
            ee_orn,
            self.planner_max_linear_speed,
            self.sim.timestep,
            planner_mode=self.planner_mode,
        )

        # Initialize desired EE velocity (zero for first observation)
        self.desired_ee_vel = np.zeros(6, dtype=np.float32)  # [lin_vel(3), ang_vel(3)]
        self.prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)

        # Get initial observation
        obs = self._get_observation()

        info = {
            "step": 0,
            "time": 0,
        }

        return obs, info

    def step(self, action):
        """Execute one step in the environment.

        Args:
            action: Action vector [base_x, base_y, base_yaw] in [-1, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip action to action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Convert policy actions to environment actions (unscaled base velocities)
        base_vel = self._convert_policy_to_env_actions(action)

        # Get desired end-effector velocity from planner (teleoperator command)
        if self.planner_mode == "closed_loop":
            q_now, _ = self.sim.robot.joint_states()
            ee_pos_now, ee_orn_now = self._ee_pose_w(q_now)
            desired_ee_lin_vel, desired_ee_ang_vel = self.ee_planner.step(
                ee_pos_now, ee_orn_now
            )
        else:
            desired_ee_lin_vel, desired_ee_ang_vel = self.ee_planner.step()
        desired_ee_vel = np.concatenate([desired_ee_lin_vel, desired_ee_ang_vel])  # 6D

        # Store desired velocity for observation (world frame)
        self.desired_ee_vel = desired_ee_vel.copy()

        # Solve IK to get joint velocities using desired velocity directly
        joint_velocities = self._solve_ik(desired_ee_vel, base_vel)

        # Command robot with computed joint velocities
        self.sim.robot.command_velocity(joint_velocities)

        # Step simulation
        t = self.current_step * self.sim.timestep
        next_t, _ = self.sim.step(t)

        # Increment step counter
        self.current_step += 1

        # Get observation
        obs = self._get_observation()

        # Slack-based failure count (modulation_rl-style: cumulative, no reset when back within slack)
        q, _ = self.sim.robot.joint_states()
        ee_pos_w, ee_orn_w = self._ee_pose_w(q)
        desired_ee_pos_w, desired_ee_orn_w = self.ee_planner.get_desired_pose()
        pos_deviation = np.linalg.norm(ee_pos_w - desired_ee_pos_w)
        orn_deviation = mm_math.quat_orientation_error(ee_orn_w, desired_ee_orn_w)
        if (
            pos_deviation > self.slack_pos_threshold
            or orn_deviation > self.slack_rot_threshold
        ):
            self.nr_kin_failures += 1

        # Compute reward (to be implemented by subclasses)
        reward = self._compute_reward(action, self.prev_action)
        # Store action for next observation (a_{t-1})
        self.prev_action = action.copy()

        # Check termination
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps

        info = {
            "step": self.current_step,
            "time": next_t,
            "nr_kin_failures": self.nr_kin_failures,
            "terminated_by_slack": (
                terminated and self.nr_kin_failures >= self.ik_fail_thresh
            ),
        }

        return obs, reward, terminated, truncated, info

    def _convert_policy_to_env_actions(self, action):
        """Convert policy actions from [-1, 1] to actual velocities.

        Args:
            action: Policy action [base_x, base_y, base_yaw] in [-1, 1]

        Returns:
            ndarray: Base velocities [x, y, yaw] (3,)
        """
        # Unscale base velocities
        base_vel = np.array(
            [
                self._unscale_action(
                    action[0], self.base_input_low[0], self.base_input_high[0]
                ),
                self._unscale_action(
                    action[1], self.base_input_low[1], self.base_input_high[1]
                ),
                self._unscale_action(
                    action[2], self.base_input_low[2], self.base_input_high[2]
                ),
            ]
        )

        return base_vel

    def _unscale_action(self, action, low, high):
        """Unscale action from [-1, 1] to [low, high].

        Args:
            action: Scaled action in [-1, 1]
            low: Lower bound
            high: Upper bound

        Returns:
            float: Unscaled action
        """
        return low + (action + 1.0) * 0.5 * (high - low)

    def _solve_ik(
        self,
        desired_ee_vel,
        base_vel,
    ):
        """Solve inverse kinematics to get joint velocities.

        Uses Casadi spatial Jacobian + shared :func:`solve_diff_ik`. Planner twists
        are world-frame; angular part is converted to body frame to match the
        spatial Jacobian convention.

        Args:
            desired_ee_vel: Desired end-effector velocity [lin_vel(3), ang_vel(3)]
                in world frame (6,)
            base_vel: Base velocities [x, y, yaw] (3,)

        Returns:
            ndarray: Joint velocities (nu,)
        """
        q, _ = self.sim.robot.joint_states()
        q = np.asarray(q, dtype=float).reshape(-1)
        self.ik_params["regularization_strength"] = float(
            self.ik_regularization_strength
        )
        self.ik_params["use_weighted_regularization"] = bool(
            self.use_weighted_regularization
        )
        _, ee_orn_w = self._ee_pose_w(q)
        _, twist_for_ik = mm_math.ee_twist_world_and_mpc_reference(
            desired_ee_vel[:3],
            desired_ee_vel[3:],
            ee_orn_w,
            clamp_limits=(self.ee_max_linear_vel, self.ee_max_angular_vel),
        )
        J = spatial_jacobian(self.robot_mdl, q)
        return solve_diff_ik(J, twist_for_ik, base_vel, self.ik_params)

    def _compute_reward(self, action, prev_action):
        """Compute reward. Override in subclasses.

        Args:
            action: Action vector
            prev_action: Previous action vector (for acceleration penalty)

        Returns:
            float: Reward value
        """
        # Default: zero reward
        return 0.0

    def _check_termination(self):
        """Check if episode should terminate. Override in subclasses.

        Uses modulation_rl-style: terminate when cumulative steps exceeding
        slack (pos or rot) >= ik_fail_thresh. Counter does not reset when back within slack.

        Returns:
            bool: ``True`` if slack-based failure count reached the threshold; ``False``
            if early termination is disabled or the count is still below the threshold.
        """
        if self.disable_early_termination:
            return False
        if self.nr_kin_failures >= self.ik_fail_thresh:
            return True
        return False

    def close(self):
        """Disconnect the PyBullet client used by the wrapped simulation."""
        import pybullet as pyb

        pyb.disconnect()
