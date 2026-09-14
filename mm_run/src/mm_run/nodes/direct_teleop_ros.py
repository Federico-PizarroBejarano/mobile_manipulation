"""Direct teleop: joy → synthetic MpcPlan → low_level_cmd_node (no MPC/MPSF)."""

from __future__ import annotations

import argparse
import sys
import threading

import numpy as np
import rospy
from mobile_manipulation_central.ros_interface import (
    JoystickButtonInterface,
    MobileManipulatorROSInterface,
)
from robotiq_3f_gripper_articulated_msgs.msg import Robotiq3FGripperRobotOutput
from sensor_msgs.msg import Joy

from mm_control.robot import MobileManipulator3D
from mm_run.msg import MpcPlan
from mm_utils import parsing
from mm_utils.diff_ik import (
    build_ik_params_from_config,
    solve_diff_ik,
    spatial_jacobian,
)
from mm_utils.robotiq_gripper import (
    GRIPPER_TOGGLE_BUTTON,
    gripper_position,
    toggle_gripper_open,
)
from mm_utils.teleop_joy import (
    FORCE_ZERO_LL_KP_PARAM,
    STICKS_ACTIVE_PARAM,
    axes_to_base_velocity,
    axes_to_ee_velocity,
    joint_velocity_command,
    parse_teleop_config,
    store_joy_axes,
    teleop_enable_held,
)


class DirectTeleopROSNode:
    """Direct teleop that publishes short constant-velocity MpcPlans (no MPC)."""

    def __init__(self):
        argv = rospy.myargv(argv=sys.argv)
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", required=True)
        parser.add_argument("--ctrl_config", type=str, default="default")
        parser.add_argument("--planner_config", type=str, default="default")
        parser.add_argument("--logging_sub_folder", type=str, default="default")
        args = parser.parse_args(argv[1:])

        config = parsing.load_config(args.config)
        if args.ctrl_config and args.ctrl_config != "default":
            config = parsing.recursive_dict_update(
                config, parsing.load_config(args.ctrl_config)
            )
        if args.planner_config and args.planner_config != "default":
            config = parsing.recursive_dict_update(
                config, parsing.load_config(args.planner_config)
            )
        self.config = config

        self.ctrl_config = config["controller"]
        teleop_cfg = parse_teleop_config(self.ctrl_config)
        if not teleop_cfg["enabled"]:
            raise ValueError(
                "direct_teleop_ros requires controller.teleop.enabled: true"
            )

        self.teleop_max_base_vel = teleop_cfg["max_base_vel"]
        self.teleop_max_ee_vel = teleop_cfg["max_ee_vel"]
        self.teleop_enable_button = teleop_cfg["enable_button"]
        self.teleop_ee_yaw_buttons = teleop_cfg["ee_yaw_buttons"]

        self.nu = int(self.ctrl_config["robot"]["dims"]["u"])
        self.dof = int(self.ctrl_config["robot"]["dims"]["q"])
        self.mpc_dt = float(self.ctrl_config["dt"])
        self.rate_hz = float(self.ctrl_config["ctrl_rate"])
        if self.rate_hz <= 0.0:
            raise ValueError(f"ctrl_rate must be positive, got {self.rate_hz}")

        self.robot_mdl = MobileManipulator3D(self.ctrl_config)
        self.ik_params = build_ik_params_from_config(config, self.nu, self.dof)

        self.joy_axes = np.zeros(6)
        self.joy_buttons = np.zeros(16)
        self.joy_lock = threading.Lock()
        self.teleop_control_mode = "base"
        self._last_toggle_button_state = False
        self._sticks_active_param = self.teleop_enable_button is None
        self.ctrl_c = False

        self.robot_interface = MobileManipulatorROSInterface()
        self.start_end_button_interface = JoystickButtonInterface(2)
        self.gripper_button_interface = JoystickButtonInterface(GRIPPER_TOGGLE_BUTTON)
        self._gripper_open = True
        self._gripper_no_sub_warned = False
        self._gripper_pub = rospy.Publisher(
            "/Robotiq3FGripperRobotOutput",
            Robotiq3FGripperRobotOutput,
            queue_size=1,
        )
        self.mpc_plan_pub = rospy.Publisher("mpc_plan", MpcPlan, queue_size=1)

        rospy.Subscriber("/bluetooth_teleop/joy", Joy, self._joy_callback)
        rospy.set_param(STICKS_ACTIVE_PARAM, self._sticks_active_param)
        # Shared YAML may set low_level kp for MPC/MPSF; disable P for direct teleop
        # (q_bar is a frozen snapshot, so kp fights stick velocity).
        rospy.set_param(FORCE_ZERO_LL_KP_PARAM, True)
        rospy.set_param("/controller_finished", False)
        rospy.set_param("/controller_started", False)
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            "Direct teleop: enable_button=%s max_base_vel=%s (EE uses diff IK)",
            self.teleop_enable_button,
            self.teleop_max_base_vel,
        )

    def _on_shutdown(self):
        self.ctrl_c = True
        self._publish_velocity_plan(np.zeros(self.nu), rospy.Time.now().to_sec())
        self.robot_interface.brake()
        rospy.set_param(FORCE_ZERO_LL_KP_PARAM, False)

    def _joy_callback(self, msg):
        with self.joy_lock:
            store_joy_axes(msg.axes, self.joy_axes)
            self.joy_buttons = np.array(msg.buttons, dtype=float)
            if len(self.joy_buttons) > 0:
                pressed = self.joy_buttons[0] == 1
                if pressed and not self._last_toggle_button_state:
                    if self.teleop_control_mode == "base":
                        self.teleop_control_mode = "ee"
                        rospy.loginfo("Switched to EE control mode (diff IK, arm only)")
                    else:
                        self.teleop_control_mode = "base"
                        rospy.loginfo("Switched to base control mode")
                self._last_toggle_button_state = pressed

            enabled = teleop_enable_held(self.joy_buttons, self.teleop_enable_button)
            if enabled != self._sticks_active_param:
                self._sticks_active_param = enabled
                rospy.set_param(STICKS_ACTIVE_PARAM, bool(enabled))

    def _wait_for_joint_states(self, rate):
        rospy.loginfo("Waiting for joint states ...")
        while not self.robot_interface.ready() and not rospy.is_shutdown():
            self.robot_interface.brake()
            rate.sleep()
        if rospy.is_shutdown():
            raise rospy.ROSInterruptException("shutdown while waiting for joints")
        rospy.loginfo("Direct teleop received joint states.")

    def _wait_for_start(self, rate):
        self.start_end_button_interface.reset_button()
        enter_pressed = threading.Event()

        if sys.stdin.isatty():
            print("----- Press Square (controller) or Enter to start -----")

            def _wait_enter():
                try:
                    input()
                    enter_pressed.set()
                except EOFError:
                    pass

            threading.Thread(target=_wait_enter, daemon=True).start()
        elif self.start_end_button_interface.ready():
            print("----- Press Square (controller) to start -----")
        else:
            print("----- Non-interactive start: skipping prompt -----")
            return

        while not self.ctrl_c and not rospy.is_shutdown():
            if self.start_end_button_interface.button == 1:
                self.start_end_button_interface.reset_button()
                return
            if enter_pressed.is_set():
                return
            rate.sleep()
        raise rospy.ROSInterruptException("shutdown while waiting to start")

    def _make_gripper_output_msg(self, is_open):
        msg = Robotiq3FGripperRobotOutput()
        msg.rACT = 1
        msg.rMOD = 0
        msg.rGTO = 1
        msg.rATR = 0
        msg.rICF = 0
        msg.rPRA = gripper_position(is_open)
        msg.rSPA = 255
        msg.rFRA = 150
        return msg

    def _poll_gripper_toggle(self):
        if self.gripper_button_interface.button != 1:
            return
        self.gripper_button_interface.reset_button()
        if self._gripper_pub.get_num_connections() == 0:
            if not self._gripper_no_sub_warned:
                rospy.logwarn(
                    "Gripper toggle ignored: no subscribers on "
                    "/Robotiq3FGripperRobotOutput"
                )
                self._gripper_no_sub_warned = True
            return
        self._gripper_open = toggle_gripper_open(self._gripper_open)
        self._gripper_pub.publish(self._make_gripper_output_msg(self._gripper_open))
        rospy.loginfo(
            "Gripper %s (Triangle/Y)",
            "open" if self._gripper_open else "closed",
        )

    def _compute_v_cmd(self):
        with self.joy_lock:
            axes = self.joy_axes.copy()
            buttons = self.joy_buttons.copy()
            mode = self.teleop_control_mode
            enabled = teleop_enable_held(buttons, self.teleop_enable_button)

        if not enabled:
            return np.zeros(self.nu, dtype=float), enabled

        if mode == "base":
            base_vel = axes_to_base_velocity(axes, self.teleop_max_base_vel)
            ee_vel = np.zeros(6, dtype=float)
            return joint_velocity_command(mode, base_vel, ee_vel, self.nu), enabled

        ee_vel = axes_to_ee_velocity(
            axes, buttons, self.teleop_max_ee_vel, self.teleop_ee_yaw_buttons
        )
        q = np.asarray(self.robot_interface.q, dtype=float).reshape(-1)[: self.dof]
        J = spatial_jacobian(self.robot_mdl, q)
        return (
            solve_diff_ik(J, ee_vel, base_vel=np.zeros(3), ik_params=self.ik_params),
            enabled,
        )

    def _publish_velocity_plan(self, v_cmd, t):
        v_cmd = np.asarray(v_cmd, dtype=float).reshape(self.nu)
        N = 1
        u_bar = np.zeros((N, self.nu))
        v_bar = np.tile(v_cmd, (N + 1, 1))
        q = np.asarray(self.robot_interface.q, dtype=float).reshape(-1)[: self.dof]
        q_bar = np.tile(q, (N + 1, 1))

        msg = MpcPlan()
        msg.header.stamp = rospy.Time.from_sec(t)
        msg.mpc_dt = self.mpc_dt
        msg.N = N
        msg.nu = self.nu
        msg.dof = self.dof
        msg.u_flat = u_bar.ravel(order="C").tolist()
        msg.v_flat = v_bar.ravel(order="C").tolist()
        msg.q_flat = q_bar.ravel(order="C").tolist()
        self.mpc_plan_pub.publish(msg)

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        self._wait_for_joint_states(rate)
        self._wait_for_start(rate)

        rospy.set_param("/controller_started", True)
        rospy.loginfo("Direct teleop started")

        while not self.ctrl_c and not rospy.is_shutdown():
            if self.start_end_button_interface.button == 1:
                self.start_end_button_interface.reset_button()
                break

            self._poll_gripper_toggle()
            v_cmd, enabled = self._compute_v_cmd()
            self._sticks_active_param = bool(enabled)
            rospy.set_param(STICKS_ACTIVE_PARAM, self._sticks_active_param)
            self._publish_velocity_plan(v_cmd, rospy.Time.now().to_sec())
            rate.sleep()

        self._publish_velocity_plan(np.zeros(self.nu), rospy.Time.now().to_sec())
        self.robot_interface.brake()
        rospy.set_param(FORCE_ZERO_LL_KP_PARAM, False)
        rospy.set_param("/controller_finished", True)
        rospy.loginfo("Direct teleop stopped")


if __name__ == "__main__":
    rospy.init_node("controller_direct_teleop")
    DirectTeleopROSNode().run()
