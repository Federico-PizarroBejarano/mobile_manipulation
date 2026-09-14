#!/usr/bin/env python3

import threading

import numpy as np
import rospy
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import Joy

from mm_run.nodes.mpc_ros import ControllerROSNode
from mm_run.scripts.mpsf_experiment import calculate_desired_velocity
from mm_utils.metrics import MPSFMetricsCollector  # noqa: E402
from mm_utils.teleop_joy import (
    STICKS_ACTIVE_PARAM,
    axes_to_base_velocity,
    axes_to_ee_velocity,
    gate_teleop_velocity,
    parse_teleop_config,
    store_joy_axes,
    teleop_enable_held,
)
from mm_utils.teleop_session_logging import session_root


class MPSFControllerROSNode(ControllerROSNode):
    """ROS node for MPSF experiments with optional joystick teleop.

    Uses the same split ROS stack as MPC: this node publishes ``MpcPlan``;
    ``low_level_cmd_node`` publishes ``cmd_vel`` (see ``controller.launch``).
    """

    def __init__(self):
        self.base_goal = None
        self.ee_goal = None
        self._mpsf_goals_initialized = False

        self.teleop_enabled = False
        self.joy_axes = np.zeros(6)
        self.joy_buttons = np.zeros(16)
        self.joy_lock = threading.Lock()
        self.teleop_max_base_vel = np.array([1.0, 1.0, 1.0])
        self.teleop_max_ee_vel = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.teleop_control_mode = "base"
        self._last_toggle_button_state = False
        self.teleop_enable_button = None
        self.teleop_ee_yaw_buttons = [14, 15]
        self._teleop_enable_active = False
        self._sticks_active_param = True

        self.metrics_collector = MPSFMetricsCollector()
        self._metrics_saved = False
        self._current_references = None
        self._current_desired_base_vel = None
        self._current_desired_ee_vel = None
        self._sim_timestep = None

        super().__init__()

        self.joy_sub = rospy.Subscriber(
            "/bluetooth_teleop/joy", Joy, self._joy_callback
        )

        teleop_cfg = parse_teleop_config(self.ctrl_config)
        self.teleop_enabled = teleop_cfg["enabled"]
        self._sim_timestep = self.simulation_timestep

        if self.teleop_enabled:
            rospy.loginfo("Teleoperation mode enabled - MPSF goals will be ignored")
            self.teleop_max_base_vel = teleop_cfg["max_base_vel"]
            self.teleop_max_ee_vel = teleop_cfg["max_ee_vel"]
            self.teleop_enable_button = teleop_cfg["enable_button"]
            self.teleop_ee_yaw_buttons = teleop_cfg["ee_yaw_buttons"]
            if self.teleop_enable_button is not None:
                rospy.loginfo(
                    "MPSF teleop enable button: %d (hold to apply stick inputs)",
                    self.teleop_enable_button,
                )
            rospy.loginfo(
                "MPSF EE yaw buttons: left=%d right=%d",
                self.teleop_ee_yaw_buttons[0],
                self.teleop_ee_yaw_buttons[1],
            )
            self._sticks_active_param = self.teleop_enable_button is None
            rospy.set_param(STICKS_ACTIVE_PARAM, self._sticks_active_param)
            self._update_mpsf_masks()

    def _extract_states_for_mpsf(self, robot_states, use_vicon_tool_data):
        q, v = robot_states
        base_pose = q[:3].copy()
        base_vel = v[:3].copy()

        if use_vicon_tool_data:
            ee_pose = np.zeros(6)
            ee_pose[:3] = self.vicon_tool_interface.position
            ee_pose[3:] = Rot.from_quat(self.vicon_tool_interface.orientation).as_euler(
                "xyz"
            )
            ee_vel = np.zeros(6)
        else:
            ee_pose = self.controller.robot.ee_pose(q)
            ee_vel = self.controller.robot.ee_velocity(q, v)

        return {
            "base": {"pose": base_pose, "velocity": base_vel},
            "EE": {"pose": ee_pose, "velocity": ee_vel},
        }

    def _joy_callback(self, msg):
        """Store axes/buttons; update sticks-active param at joy rate."""
        self.joy_lock.acquire()
        try:
            store_joy_axes(msg.axes, self.joy_axes)
            self.joy_buttons = np.array(msg.buttons, dtype=float)
            if len(self.joy_buttons) > 0:
                pressed = self.joy_buttons[0] == 1
                if pressed and not self._last_toggle_button_state:
                    if self.teleop_control_mode == "base":
                        self.teleop_control_mode = "ee"
                        rospy.loginfo("Switched to EE control mode")
                    else:
                        self.teleop_control_mode = "base"
                        rospy.loginfo("Switched to base control mode")
                    self._update_mpsf_masks()
                    if self.teleop_enable_button is not None:
                        rospy.loginfo(
                            "Mode toggled — keep d-pad up (btn %d) held "
                            "or stick inputs stay gated",
                            self.teleop_enable_button,
                        )
                self._last_toggle_button_state = pressed

            if self.teleop_enabled:
                enabled = teleop_enable_held(
                    self.joy_buttons, self.teleop_enable_button
                )
                if enabled != self._sticks_active_param:
                    self._sticks_active_param = enabled
                    rospy.set_param(STICKS_ACTIVE_PARAM, bool(enabled))
        finally:
            self.joy_lock.release()

    def _joystick_to_base_velocity(self):
        with self.joy_lock:
            axes = self.joy_axes.copy()
        return axes_to_base_velocity(axes, self.teleop_max_base_vel)

    def _joystick_to_ee_velocity(self):
        with self.joy_lock:
            axes = self.joy_axes.copy()
            buttons = self.joy_buttons.copy()
        return axes_to_ee_velocity(
            axes, buttons, self.teleop_max_ee_vel, self.teleop_ee_yaw_buttons
        )

    def _update_mpsf_masks(self):
        if not self.teleop_enabled:
            return

        if self.teleop_control_mode == "base":
            self.controller.mpsf_base_mask = np.array([1, 1, 1], dtype=bool)
            self.controller.mpsf_ee_mask = np.array([0, 0, 0, 0, 0, 0], dtype=bool)
            rospy.loginfo("MPSF masks updated: ee masked, base enabled")
        else:
            self.controller.mpsf_base_mask = np.array([0, 0, 0], dtype=bool)
            self.controller.mpsf_ee_mask = np.array([1, 1, 1, 1, 1, 1], dtype=bool)
            rospy.loginfo("MPSF masks updated: base masked, ee enabled")

    def update_references(self, references, robot_states):
        if self.teleop_enabled:
            self.joy_lock.acquire()
            enabled = teleop_enable_held(self.joy_buttons, self.teleop_enable_button)
            stick_mag = float(np.linalg.norm(self.joy_axes[:4]))
            pressed = [i for i, b in enumerate(self.joy_buttons) if b == 1]
            self.joy_lock.release()
            prev_active = self._teleop_enable_active
            if enabled != prev_active:
                self._teleop_enable_active = enabled
                if enabled:
                    self._update_mpsf_masks()
                else:
                    self.controller.reset()
                    rospy.loginfo("MPSF teleop released: cleared MPC warm start")
                if self.teleop_enable_button is not None:
                    state = "active" if enabled else "idle (enable button released)"
                    rospy.loginfo(
                        "MPSF teleop sticks %s (enable_button=%s, pressed=%s)",
                        state,
                        self.teleop_enable_button,
                        pressed,
                    )
            elif (
                self.teleop_enable_button is not None
                and not enabled
                and stick_mag > 0.2
            ):
                rospy.logwarn_throttle(
                    2.0,
                    "Sticks deflected but d-pad up (btn %s) not held "
                    "(pressed buttons=%s). Hold d-pad up to drive.",
                    self.teleop_enable_button,
                    pressed,
                )

            if not enabled:
                self.controller.mpsf_base_mask = np.ones(3, dtype=float)
                self.controller.mpsf_ee_mask = np.ones(6, dtype=float)
                desired_velocity = {
                    "base_velocity": np.zeros(3, dtype=float),
                    "ee_velocity": np.zeros(6, dtype=float),
                }
                self._current_desired_base_vel = desired_velocity["base_velocity"]
                self._current_desired_ee_vel = desired_velocity["ee_velocity"]
            else:
                if self.teleop_control_mode == "ee":
                    desired_ee_vel = gate_teleop_velocity(
                        self._joystick_to_ee_velocity(), True
                    )
                    desired_velocity = {"ee_velocity": desired_ee_vel}
                    self._current_desired_base_vel = None
                    self._current_desired_ee_vel = desired_ee_vel
                else:
                    desired_base_vel = gate_teleop_velocity(
                        self._joystick_to_base_velocity(), True
                    )
                    desired_velocity = {"base_velocity": desired_base_vel}
                    self._current_desired_base_vel = desired_base_vel
                    self._current_desired_ee_vel = None
            references["desired_velocity"] = desired_velocity
            self._sticks_active_param = bool(enabled)
            rospy.set_param(STICKS_ACTIVE_PARAM, self._sticks_active_param)
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
                rospy.loginfo(
                    f"MPSF goals - Base: {self.base_goal}, EE: {self.ee_goal}"
                )

            states = self._extract_states_for_mpsf(
                robot_states, self.use_vicon_tool_data
            )
            desired_base_vel, desired_ee_vel = calculate_desired_velocity(
                self.base_goal,
                self.ee_goal,
                states,
                self.controller,
                mpsf_params=self.ctrl_config.get("mpsf_params"),
            )

            self._current_desired_base_vel = desired_base_vel
            self._current_desired_ee_vel = desired_ee_vel

            desired_velocity = {}
            if desired_base_vel is not None:
                desired_velocity["base_velocity"] = desired_base_vel
            if desired_ee_vel is not None:
                desired_velocity["ee_velocity"] = desired_ee_vel
            references["desired_velocity"] = desired_velocity

        self._current_references = references

    def _teleop_log_fields(self):
        with self.joy_lock:
            joy_axes = self.joy_axes.copy()
            joy_buttons = self.joy_buttons.copy()

        desired_base_vel = (
            np.zeros(3)
            if self._current_desired_base_vel is None
            else np.asarray(self._current_desired_base_vel, dtype=float).copy()
        )
        desired_ee_vel = (
            np.zeros(6)
            if self._current_desired_ee_vel is None
            else np.asarray(self._current_desired_ee_vel, dtype=float).copy()
        )
        return {
            "joy_axes": joy_axes,
            "joy_buttons": joy_buttons,
            "teleop_mode": self.teleop_control_mode,
            "teleop_enabled": bool(self.teleop_enabled and self._teleop_enable_active),
            "desired_base_vel": desired_base_vel,
            "desired_ee_vel": desired_ee_vel,
        }

    def _after_control_step(self, t, robot_states, states, references, u_current):
        if self._current_references is not None:
            metrics_enabled = bool(self.teleop_enabled and self._teleop_enable_active)
            metric_desired_base = (
                self._current_desired_base_vel if metrics_enabled else None
            )
            metric_desired_ee = (
                self._current_desired_ee_vel if metrics_enabled else None
            )
            self.metrics_collector.update(
                self._current_references,
                states,
                u_current,
                metric_desired_base,
                metric_desired_ee,
                self.controller,
                robot_states,
                self._sim_timestep,
                teleop_mode=self.teleop_control_mode,
                teleop_enabled=metrics_enabled,
                measured_base_vel=states["base"]["velocity"],
                measured_ee_vel=states["EE"]["velocity"],
                dt=1.0 / self._mpc_loop_hz,
            )

    def shutdownhook(self):
        super().shutdownhook()
        self._save_metrics()

    def _save_metrics(self):
        if self._metrics_saved:
            return
        metrics_dir = (
            session_root(self.logger.base_directory, self.session_timestamp) / "metrics"
        )
        try:
            self.metrics_collector.save(metrics_dir)
            self.metrics_collector.print_summary()
            self._metrics_saved = True
        except Exception as exc:
            rospy.logerr("Failed to save MPSF metrics: %s", exc)


if __name__ == "__main__":
    rospy.init_node("controller_ros_mpsf")

    node = MPSFControllerROSNode()
    node.run()
    node._save_metrics()
