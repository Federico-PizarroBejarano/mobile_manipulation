"""Base Gym environment wrapper for BulletSimulation."""

from datetime import datetime

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mm_rl.env.ee_planner import EEPlanner
from mm_simulator.simulation import BulletSimulation
from mm_utils import math as mm_math


class BaseRLEnv(gym.Env):
    """Base Gym environment wrapping BulletSimulation for RL training."""

    def __init__(self, config):
        """Initialize base RL environment.

        Args:
            config: Configuration dictionary with simulation and robot parameters
        """
        super().__init__()

        # Store config
        self.config = config
        self.sim_config = config.get("simulation")
        # Robot config might be nested under controller.robot
        self.robot_config = config.get("controller").get("robot")

        # Initialize simulation
        timestamp = datetime.now()
        self.sim = BulletSimulation(self.sim_config, timestamp, cli_args=None)

        # Get robot dimensions
        self.nq = self.sim.robot.nq  # Number of joint positions
        self.nv = self.sim.robot.nv  # Number of joint velocities
        self.nu = self.sim.robot.nu  # Number of inputs

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

        # Clamp joint velocities to limits from config
        velocity_limits = self.robot_config.get("limits").get("state")
        self.joint_vel_lower = np.array(velocity_limits["lower"][self.nq :])
        self.joint_vel_upper = np.array(velocity_limits["upper"][self.nq :])

        # IK solver parameters (N²M² paper: minimum displacement regularization)
        ik_config = config.get("ik", {})
        self.ik_regularization_strength = ik_config.get("regularization_strength", 0.1)
        self.use_weighted_regularization = ik_config.get(
            "use_weighted_regularization", True
        )

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
        """
        return 6 + 12 + 12 + 12 + self.nq + self.action_space.shape[0]

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

        # Get end-effector pose in world frame
        ee_pos_w, ee_orn_w = self.sim.robot.link_pose()  # Tool link

        # Get base pose in world frame
        base_pos_w, base_orn_w = self.sim.robot.link_pose(link_idx=-1)  # Base link

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

        # Use final goal (original behavior)
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
            seed: Random seed
            options: Optional dict with reset options

        Returns:
            observation, info
        """
        super().reset(seed=seed)

        # Reset simulation
        self.sim.robot.reset_joint_configuration(self.sim.robot.home)

        # Reset step counter
        self.current_step = 0

        # Reset slack-failure count (cumulative per episode)
        self.nr_kin_failures = 0

        # Initialize/reset end-effector planner if goal is set
        ee_pos, ee_orn = self.sim.robot.link_pose()
        self.ee_planner = EEPlanner(
            self.goal_pos,
            self.goal_orn,
            ee_pos,
            ee_orn,
            self.planner_max_linear_speed,
            self.sim.timestep,
        )

        # Initialize desired EE velocity (zero for first observation)
        self.desired_ee_vel = np.zeros(6, dtype=np.float32)  # [lin_vel(3), ang_vel(3)]

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
        desired_ee_lin_vel, desired_ee_ang_vel = self.ee_planner.step()
        desired_ee_vel = np.concatenate([desired_ee_lin_vel, desired_ee_ang_vel])  # 6D

        # Store desired velocity for observation
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
        ee_pos_w, ee_orn_w = self.sim.robot.link_pose()
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

        Uses Jacobian-based IK with minimum displacement regularization weighted by
        maximum joint velocities (N²M² paper approach). This prefers solutions that
        keep joints close to their current positions, with faster joints penalized less.

        Args:
            desired_ee_vel: Desired end-effector velocity [lin_vel(3), ang_vel(3)] in world frame (6,)
            base_vel: Base velocities [x, y, yaw] (3,)

        Returns:
            ndarray: Joint velocities (nu,)
        """
        # Get current joint state
        q, _ = self.sim.robot.joint_states()

        # Clamp desired velocities to reasonable limits (make a copy to avoid modifying input)
        desired_ee_vel_clamped = desired_ee_vel.copy()
        max_lin_vel = 1.0
        if np.linalg.norm(desired_ee_vel_clamped[:3]) > max_lin_vel:
            desired_ee_vel_clamped[:3] = (
                desired_ee_vel_clamped[:3]
                / np.linalg.norm(desired_ee_vel_clamped[:3])
                * max_lin_vel
            )

        max_ang_vel = 2.0
        if np.linalg.norm(desired_ee_vel_clamped[3:]) > max_ang_vel:
            desired_ee_vel_clamped[3:] = (
                desired_ee_vel_clamped[3:]
                / np.linalg.norm(desired_ee_vel_clamped[3:])
                * max_ang_vel
            )

        # Get Jacobian
        J = self.sim.robot.jacobian(q)

        # For Thing robot with omnidirectional base:
        # Joint 0: x_to_world_joint (base x translation)
        # Joint 1: y_to_x_joint (base y translation)
        # Joint 2: base_to_y_joint (base yaw rotation)
        # Joints 3-8: arm joints

        # Base joint indices: [x, y, yaw] -> joints [0, 1, 2]
        base_joint_indices = [0, 1, 2]

        # Compute contribution of base velocities to end-effector velocity
        J_base = J[:, base_joint_indices]
        v_ee_from_base = J_base @ base_vel

        # Remaining desired velocity for arm to achieve
        v_ee_arm_desired = desired_ee_vel_clamped - v_ee_from_base

        # Solve for arm joint velocities
        arm_joint_indices = list(range(3, self.nu))  # Arm joints (3-8)
        J_arm = J[:, arm_joint_indices]

        if self.use_weighted_regularization:
            # N²M² paper approach: minimum displacement regularization weighted by 1/max_velocity
            # Minimize: ||J_arm * q_dot - v_ee_arm_desired||² + λ * ||W * q_dot||²
            # where W_ii = 1 / max_vel_i

            # Get max velocities for arm joints
            max_vels_arm = self.joint_vel_upper[arm_joint_indices]
            # Avoid division by zero and ensure positive values
            max_vels_arm = np.maximum(np.abs(max_vels_arm), 1e-6)

            # Build weight matrix: W_ii = 1 / max_vel_i
            # This means faster joints (higher max_vel) get less penalty
            W = np.diag(1.0 / max_vels_arm)

            # Weighted regularization term: W^T * W (since W is diagonal, this is W^2)
            W_squared = W.T @ W

            # Solve: (J_arm^T * J_arm + λ * W^T * W) * q_dot = J_arm^T * v_ee_arm_desired
            A = J_arm.T @ J_arm + self.ik_regularization_strength * W_squared
            b = J_arm.T @ v_ee_arm_desired

            # Use pseudo-inverse for numerical stability
            arm_vel = np.linalg.pinv(A) @ b
        else:
            # Fallback to simple damped least squares (original approach)
            damping = 0.01
            J_arm_pinv = J_arm.T @ np.linalg.inv(
                J_arm @ J_arm.T + damping * np.eye(J_arm.shape[0])
            )
            arm_vel = J_arm_pinv @ v_ee_arm_desired

        # Combine all joint velocities
        joint_velocities = np.zeros(self.nu)
        joint_velocities[base_joint_indices] = base_vel
        joint_velocities[arm_joint_indices] = arm_vel

        # Clip to limits
        for i in range(self.nu):
            joint_velocities[i] = np.clip(
                joint_velocities[i], self.joint_vel_lower[i], self.joint_vel_upper[i]
            )

        return joint_velocities

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
        """
        if self.nr_kin_failures >= self.ik_fail_thresh:
            return True
        return False

    def close(self):
        """Clean up environment."""
        import pybullet as pyb

        pyb.disconnect()
