#!/usr/bin/env python3

import threading

import numpy as np
import rospy
from mobile_manipulation_central.ros_interface import JoystickButtonInterface
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import Joy

# Import the base controller node
from mm_run.nodes.mpc_ros import ControllerROSNode

# Import MPSF-specific functions
from mm_run.scripts.mpsf_experiment import calculate_desired_velocity

# Import metrics collection
from mm_utils.metrics import MPSFMetricsCollector  # noqa: E402


class MPSFControllerROSNode(ControllerROSNode):
    """ROS node for MPSF (Model Predictive Shared Framework) experiments.

    Inherits from ControllerROSNode and adds MPSF-specific functionality:
    - Calculates desired velocities from MPSF goals
    - Adds desired velocities to controller references
    """

    def __init__(self):
        self.base_goal = None
        self.ee_goal = None
        self._mpsf_goals_initialized = False

        # Teleoperation support
        self.teleop_enabled = False
        # Store latest joystick axes [left_x, left_y, right_x, right_y, left_trigger, right_trigger]
        self.joy_axes = np.zeros(6)
        self.joy_buttons = np.zeros(10)  # Store latest joystick buttons
        self.joy_lock = threading.Lock()
        self.teleop_max_base_vel = np.array([1.0, 1.0, 1.0])
        self.teleop_max_ee_vel = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.teleop_control_mode = "base"  # "base" or "ee"
        self._last_toggle_button_state = False  # Track button 0 (A) state for toggle

        # Metrics collection
        self.metrics_collector = MPSFMetricsCollector()
        self._current_references = None
        self._current_desired_base_vel = None
        self._current_desired_ee_vel = None
        self._sim_timestep = None

        super().__init__()

        # Subscribe to joystick messages for teleoperation
        self.joy_sub = rospy.Subscriber("/teleop/joy", Joy, self._joy_callback)

        # Check if teleop is enabled in config
        mpsf_params = self.ctrl_config.get("mpsf_params")
        self.teleop_enabled = mpsf_params.get("teleop_enabled", False)
        self.use_joy = self.teleop_enabled

        # Get simulation timestep for metrics (use controller dt as fallback)
        sim_config = self.ctrl_config.get("simulation", {})
        self._sim_timestep = sim_config.get("timestep", getattr(self, "mpc_dt", 0.01))

        if self.teleop_enabled:
            rospy.loginfo("Teleoperation mode enabled - MPSF goals will be ignored")
            # Get teleop parameters from config if available
            if "teleop_max_base_vel" in mpsf_params:
                self.teleop_max_base_vel = np.array(mpsf_params["teleop_max_base_vel"])
            if "teleop_max_ee_vel" in mpsf_params:
                self.teleop_max_ee_vel = np.array(mpsf_params["teleop_max_ee_vel"])
            # Create task_switch_button_interface if it doesn't exist (needed for parent code)
            if not hasattr(self, "task_switch_button_interface"):
                # PS4: circle button or Xbox: B button
                self.task_switch_button_interface = JoystickButtonInterface(1)

            # Update masks based on initial control mode
            self._update_mpsf_masks()

    def _extract_states_for_mpsf(self, robot_states, use_vicon_tool_data):
        """Extract robot states in the format needed for MPSF calculations.

        Args:
            robot_states (tuple): (q, v) tuple from robot interface.
            use_vicon_tool_data (bool): Whether Vicon tool data is available.

        Returns:
            dict: Dictionary with "base" and "EE" keys, each containing "pose" and "velocity".
        """
        if use_vicon_tool_data:
            ee_pos = self.vicon_tool_interface.position
            ee_quat = self.vicon_tool_interface.orientation
            # For Vicon data, velocity is not directly available, set to zeros
            ee_vel = np.zeros(6)
        else:
            ee_pos, ee_quat = self.robot.getEE(robot_states[0])
            # Compute EE velocity using spatial Jacobian if available
            tool_name = self.robot.tool_link_name
            spatial_jac_key = tool_name + "_spatial"
            if spatial_jac_key in self.robot.jacSymMdls:
                J_spatial = self.robot.jacSymMdls[spatial_jac_key](robot_states[0])
                ee_vel = (J_spatial @ robot_states[1]).toarray().flatten()
            else:
                # Fallback: use position Jacobian and pad with zeros for angular velocity
                J_pos = self.robot.jacSymMdls[tool_name](robot_states[0])
                ee_lin_vel = (J_pos @ robot_states[1]).toarray().flatten()
                ee_vel = np.hstack([ee_lin_vel, np.zeros(3)])

        ee_euler = Rot.from_quat(ee_quat).as_euler("xyz")
        ee_pose = np.hstack([ee_pos, ee_euler])

        base_pose = robot_states[0][:3]  # [x, y, yaw] already in world frame
        base_vel = robot_states[1][:3]  # [vx, vy, vyaw]

        return {
            "base": {"pose": base_pose, "velocity": base_vel},
            "EE": {"pose": ee_pose, "velocity": ee_vel},
        }

    def _joy_callback(self, msg):
        """Callback for joystick messages. Stores axes and button values thread-safely."""
        self.joy_lock.acquire()

        # Joystick axes
        self.joy_axes[0] = msg.axes[0]  # Left stick X
        self.joy_axes[1] = msg.axes[1]  # Left stick Y
        self.joy_axes[2] = msg.axes[3]  # Right stick X for roll (wx)
        self.joy_axes[3] = msg.axes[4]  # Right stick Y for vertical (EE mode)

        # Joystick triggers
        # Triggers go from 1.0 (not pressed) to -1.0 (fully pressed)
        lt_val = msg.axes[2]
        self.joy_axes[4] = max(0.0, (1.0 - lt_val) / 2.0)
        rt_val = msg.axes[5]
        self.joy_axes[5] = max(0.0, (1.0 - rt_val) / 2.0)

        # Store button states
        self.joy_buttons = np.array(msg.buttons)
        if self.joy_buttons[0] == 1 and not self._last_toggle_button_state:
            if self.teleop_control_mode == "base":
                self.teleop_control_mode = "ee"
                rospy.loginfo("Switched to EE control mode")
            else:
                self.teleop_control_mode = "base"
                rospy.loginfo("Switched to base control mode")
            # Update MPSF masks when control mode changes
            self._update_mpsf_masks()
        self._last_toggle_button_state = self.joy_buttons[0] == 1

        self.joy_lock.release()

    def _joystick_to_base_velocity(self):
        """Convert joystick axes to desired base velocity.

        Returns:
            np.ndarray: Desired base velocity [vx, vy, vyaw] or None if no input.
        """
        self.joy_lock.acquire()
        joy_x = self.joy_axes[1]  # Left stick X
        joy_y = self.joy_axes[0]  # Left stick Y
        joy_yaw = self.joy_axes[2]  # Right stick X for yaw
        self.joy_lock.release()

        # Map joystick values [-1, 1] to velocity commands
        desired_vel = np.array(
            [
                joy_x * self.teleop_max_base_vel[0],  # vx
                joy_y * self.teleop_max_base_vel[1],  # vy
                joy_yaw * self.teleop_max_base_vel[2],  # vyaw
            ]
        )

        return desired_vel

    def _joystick_to_ee_velocity(self):
        """Convert joystick axes to desired end-effector velocity.

        Returns:
            np.ndarray: Desired EE velocity [vx, vy, vz, wx, wy, wz] or None if no input.
        """
        self.joy_lock.acquire()

        joy_x = self.joy_axes[1]  # Left stick X -> EE vx
        joy_y = self.joy_axes[0]  # Left stick Y -> EE vy
        joy_z = self.joy_axes[3]  # Right stick Y -> EE vz
        joy_wx = self.joy_axes[2]  # Right stick X -> EE wx (roll)

        # Pitch (wy) from triggers: RT (positive) - LT (negative)
        left_trigger = self.joy_axes[4]
        right_trigger = self.joy_axes[5]
        joy_wy = right_trigger - left_trigger

        # Yaw (wz) from bumpers: Right Bumper (positive) - Left Bumper (negative)
        left_bumper = float(self.joy_buttons[4])
        right_bumper = float(self.joy_buttons[5])
        joy_wz = right_bumper - left_bumper

        self.joy_lock.release()

        # Map joystick values to velocity commands
        desired_vel = np.array(
            [
                joy_x * self.teleop_max_ee_vel[0],  # vx
                joy_y * self.teleop_max_ee_vel[1],  # vy
                joy_z * self.teleop_max_ee_vel[2],  # vz
                joy_wx * self.teleop_max_ee_vel[3],  # wx (roll) from right stick X
                joy_wy * self.teleop_max_ee_vel[4],  # wy (pitch) from triggers
                joy_wz * self.teleop_max_ee_vel[5],  # wz (yaw) from bumpers
            ]
        )

        return desired_vel

    def _update_mpsf_masks(self):
        """Update MPSF masks based on current teleop control mode."""
        if not self.teleop_enabled:
            return

        if self.teleop_control_mode == "base":
            # Teleop controls base -> MPSF masks base, can control EE
            self.controller.mpsf_base_mask = np.array([1, 1, 1], dtype=bool)
            self.controller.mpsf_ee_mask = np.array([0, 0, 0, 0, 0, 0], dtype=bool)
            rospy.loginfo("MPSF masks updated: ee masked, base enabled")
        else:  # "ee" mode
            # Teleop controls EE -> MPSF masks EE, can control base
            self.controller.mpsf_base_mask = np.array([0, 0, 0], dtype=bool)
            self.controller.mpsf_ee_mask = np.array([1, 1, 1, 1, 1, 1], dtype=bool)
            rospy.loginfo("MPSF masks updated: base masked, ee enabled")

    def update_references(self, references, robot_states):
        """Update references for MPSF calculations or teleoperation.

        Args:
            references (dict): References dictionary.
            robot_states (tuple): (q, v) tuple from robot interface.
        """
        if self.teleop_enabled:
            # Check control mode and use appropriate joystick mapping
            if self.teleop_control_mode == "ee":
                desired_ee_vel = self._joystick_to_ee_velocity()
                desired_velocity = {"ee_velocity": desired_ee_vel}
                self._current_desired_base_vel = None
                self._current_desired_ee_vel = desired_ee_vel
            else:  # base mode
                desired_base_vel = self._joystick_to_base_velocity()
                desired_velocity = {"base_velocity": desired_base_vel}
                self._current_desired_base_vel = desired_base_vel
                self._current_desired_ee_vel = None
            references["desired_velocity"] = desired_velocity
            print(f"desired_velocity ({self.teleop_control_mode}): {desired_velocity}")
        else:
            if not self._mpsf_goals_initialized:
                mpsf_goals = self.ctrl_config.get("mpsf_params", {})
                if mpsf_goals.get("mpsf_base_goal") is not None:
                    self.base_goal = np.array(mpsf_goals.get("mpsf_base_goal"))
                if mpsf_goals.get("mpsf_ee_goal") is not None:
                    self.ee_goal = np.array(mpsf_goals.get("mpsf_ee_goal"))
                self._mpsf_goals_initialized = True

                if self.base_goal is None and self.ee_goal is None:
                    raise ValueError("MPSF goals not found in config")
                else:
                    rospy.loginfo(
                        f"MPSF goals - Base: {self.base_goal}, EE: {self.ee_goal}"
                    )

            states = self._extract_states_for_mpsf(
                robot_states, self.use_vicon_tool_data
            )
            desired_base_vel, desired_ee_vel = calculate_desired_velocity(
                self.base_goal, self.ee_goal, states, self.controller
            )

            # Store desired velocities for metrics collection
            self._current_desired_base_vel = desired_base_vel
            self._current_desired_ee_vel = desired_ee_vel

            # Add desired velocities to references if not None
            desired_velocity = {}
            if desired_base_vel is not None:
                desired_velocity["base_velocity"] = desired_base_vel
            if desired_ee_vel is not None:
                desired_velocity["ee_velocity"] = desired_ee_vel
            references["desired_velocity"] = desired_velocity

        # Store references for metrics collection
        self._current_references = references

    def _after_control_step(self, t, robot_states, states, references, u_current):
        """Override to collect metrics after each control step."""
        if self._current_references is not None:
            self.metrics_collector.update(
                self._current_references,
                states,
                u_current,
                self._current_desired_base_vel,
                self._current_desired_ee_vel,
                self.controller,
                robot_states,
                self._sim_timestep,
            )


if __name__ == "__main__":
    rospy.init_node("controller_ros_mpsf")

    node = MPSFControllerROSNode()
    node.run()
    node.metrics_collector.print_summary()
