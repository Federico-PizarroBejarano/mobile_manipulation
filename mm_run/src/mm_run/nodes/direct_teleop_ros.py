"""Direct teleop: joy → synthetic MpcPlan → low_level_cmd_node (no MPC/MPSF)."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import rospy
from mobile_manipulation_central.ros_interface import (
    JoystickButtonInterface,
    MobileManipulatorROSInterface,
)
from robotiq_3f_gripper_articulated_msgs.msg import Robotiq3FGripperRobotOutput
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import Joy

from mm_control.robot import MobileManipulator3D
from mm_run.msg import MpcPlan
from mm_utils import parsing
from mm_utils.diff_ik import (
    build_ik_params_from_config,
    solve_diff_ik,
    spatial_jacobian,
)
from mm_utils.logging import DataLogger
from mm_utils.metrics import MPSFMetricsCollector
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
    chassis_base_twist_to_world,
    chassis_ee_twist_to_world,
    joint_velocity_command,
    parse_teleop_config,
    store_joy_axes,
    teleop_enable_held,
)
from mm_utils.teleop_session_logging import (
    TrialBagRecorder,
    append_teleop_sample,
    clear_experiment_timestamp,
    commanded_ee_twist,
    resolve_experiment_timestamp,
    session_root,
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
        if args.logging_sub_folder and args.logging_sub_folder != "default":
            config["logging"]["log_dir"] = os.path.join(
                config["logging"]["log_dir"], args.logging_sub_folder
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

        self.logger = DataLogger(copy.deepcopy(config), name="control")
        dims = self.ctrl_config["robot"]["dims"]
        for key in ("q", "v", "x", "u"):
            if key in dims:
                self.logger.add(f"n{key}", dims[key])

        self.session_timestamp = resolve_experiment_timestamp(create=True)
        record_bag = rospy.get_param(
            "~record_bag", rospy.get_param("/mm_run/record_bag", False)
        )
        bag_all = rospy.get_param("~bag_all", rospy.get_param("/mm_run/bag_all", False))
        self._record_bag = bool(record_bag)
        self._bag_all = bool(bag_all)
        self.bag_recorder = TrialBagRecorder(
            session_root(self.logger.base_directory, self.session_timestamp),
            record_bag=self._record_bag,
            bag_all=self._bag_all,
            tool_vicon_name=self.ctrl_config["robot"]["tool_vicon_name"],
            plan_topic=rospy.resolve_name("mpc_plan"),
        )
        self.metrics_collector = MPSFMetricsCollector()
        self.metrics_controller = SimpleNamespace(
            mpsf_base_mask=np.ones(3),
            mpsf_ee_mask=np.ones(6),
            base_mask=np.ones(3),
            ee_mask=np.ones(6),
            constraints=[],
            log={},
        )
        self._cycle_t0 = None
        self._saved = False

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

    def _sync_session_timestamp(self):
        """Re-read shared stamp (sim may set it after our __init__)."""
        ts = resolve_experiment_timestamp(create=True)
        if ts == self.session_timestamp:
            return
        rospy.loginfo(
            "Adopting experiment timestamp %s (was %s)", ts, self.session_timestamp
        )
        self.session_timestamp = ts
        self.bag_recorder.session_root = session_root(
            self.logger.base_directory, self.session_timestamp
        )

    def _on_shutdown(self):
        self.ctrl_c = True
        self._publish_velocity_plan(np.zeros(self.nu), rospy.Time.now().to_sec())
        self.robot_interface.brake()
        rospy.set_param(FORCE_ZERO_LL_KP_PARAM, False)
        self._save_results()
        clear_experiment_timestamp()

    def _save_results(self):
        if self._saved:
            return
        self.bag_recorder.stop()
        metrics_dir = (
            session_root(self.logger.base_directory, self.session_timestamp) / "metrics"
        )
        metrics_saved = False
        logger_saved = False
        try:
            self.metrics_collector.save(metrics_dir)
            self.metrics_collector.print_summary()
            metrics_saved = True
        except Exception as exc:
            rospy.logerr("Failed to save direct teleop metrics: %s", exc)
        try:
            self.logger.save(session_timestamp=self.session_timestamp)
            logger_saved = True
        except Exception as exc:
            rospy.logerr("Failed to save direct teleop control log: %s", exc)
        self._saved = metrics_saved and logger_saved

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

        desired_base_vel = np.zeros(3, dtype=float)
        desired_ee_vel = np.zeros(6, dtype=float)
        if not enabled:
            return (
                np.zeros(self.nu, dtype=float),
                enabled,
                mode,
                desired_base_vel,
                desired_ee_vel,
                axes,
                buttons,
            )

        q = np.asarray(self.robot_interface.q, dtype=float).reshape(-1)[: self.dof]
        yaw = float(q[2])
        if mode == "base":
            desired_base_vel = chassis_base_twist_to_world(
                axes_to_base_velocity(axes, self.teleop_max_base_vel), yaw
            )
            v_cmd = joint_velocity_command(
                mode, desired_base_vel, desired_ee_vel, self.nu
            )
            return (
                v_cmd,
                enabled,
                mode,
                desired_base_vel,
                desired_ee_vel,
                axes,
                buttons,
            )

        desired_ee_vel = chassis_ee_twist_to_world(
            axes_to_ee_velocity(
                axes, buttons, self.teleop_max_ee_vel, self.teleop_ee_yaw_buttons
            ),
            yaw,
        )
        J = spatial_jacobian(self.robot_mdl, q)
        v_cmd = solve_diff_ik(
            J,
            desired_ee_vel,
            base_vel=desired_base_vel,
            ik_params=self.ik_params,
        )
        return (
            v_cmd,
            enabled,
            mode,
            desired_base_vel,
            desired_ee_vel,
            axes,
            buttons,
        )

    def _record_cycle(
        self,
        timestamp,
        cycle_period,
        v_cmd,
        enabled,
        mode,
        desired_base_vel,
        desired_ee_vel,
        joy_axes,
        joy_buttons,
    ):
        if not enabled:
            desired_base_vel = np.zeros(3, dtype=float)
            desired_ee_vel = np.zeros(6, dtype=float)

        q = np.asarray(self.robot_interface.q, dtype=float).reshape(-1)[: self.dof]
        v = np.asarray(self.robot_interface.v, dtype=float).reshape(-1)[: self.nu]
        J = spatial_jacobian(self.robot_mdl, q)
        ee_pos, ee_quat = self.robot_mdl.getEE(q)
        ee_pose = np.hstack([ee_pos, Rot.from_quat(ee_quat).as_euler("xyz")])
        ee_vel = commanded_ee_twist(J, v)
        ee_cmd_twist = commanded_ee_twist(J, v_cmd)
        base_pose = q[:3].copy()
        base_vel = v[:3].copy()

        append_teleop_sample(
            self.logger,
            ts=timestamp,
            q=q,
            v=v,
            base_pose=base_pose,
            base_vel=base_vel,
            ee_pose=ee_pose,
            ee_vel=ee_vel,
            joy_axes=joy_axes,
            joy_buttons=joy_buttons,
            teleop_mode=mode,
            teleop_enabled=enabled,
            desired_base_vel=desired_base_vel,
            desired_ee_vel=desired_ee_vel,
            u_cmd=v_cmd,
            cycle_period=cycle_period,
        )

        states = {
            "base": {"pose": base_pose, "velocity": base_vel},
            "EE": {"pose": ee_pose, "velocity": ee_vel},
        }
        metric_desired_base = desired_base_vel if enabled and mode == "base" else None
        metric_desired_ee = desired_ee_vel if enabled and mode == "ee" else None
        self.metrics_collector.update(
            {},
            states,
            v_cmd,
            metric_desired_base,
            metric_desired_ee,
            self.metrics_controller,
            (q, v),
            self.mpc_dt,
            teleop_mode=mode,
            teleop_enabled=enabled,
            commanded_ee_twist=ee_cmd_twist if mode == "ee" else None,
            measured_base_vel=base_vel,
            measured_ee_vel=ee_vel,
            dt=cycle_period,
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
        self._sync_session_timestamp()

        self.bag_recorder.start()
        if self.bag_recorder.last_error is not None:
            rospy.logwarn(
                "rosbag record failed to start: %s", self.bag_recorder.last_error
            )

        rospy.set_param("/controller_started", True)
        rospy.loginfo("Direct teleop started (session %s)", self.session_timestamp)
        self._cycle_t0 = time.perf_counter()

        while not self.ctrl_c and not rospy.is_shutdown():
            if self.start_end_button_interface.button == 1:
                self.start_end_button_interface.reset_button()
                break

            cycle_t0 = time.perf_counter()
            cycle_period = cycle_t0 - self._cycle_t0
            self._cycle_t0 = cycle_t0
            self._poll_gripper_toggle()
            (
                v_cmd,
                enabled,
                mode,
                desired_base_vel,
                desired_ee_vel,
                joy_axes,
                joy_buttons,
            ) = self._compute_v_cmd()
            self._sticks_active_param = bool(enabled)
            rospy.set_param(STICKS_ACTIVE_PARAM, self._sticks_active_param)
            timestamp = rospy.Time.now().to_sec()
            self._publish_velocity_plan(v_cmd, timestamp)
            self._record_cycle(
                timestamp,
                cycle_period,
                v_cmd,
                enabled,
                mode,
                desired_base_vel,
                desired_ee_vel,
                joy_axes,
                joy_buttons,
            )
            rate.sleep()

        self._publish_velocity_plan(np.zeros(self.nu), rospy.Time.now().to_sec())
        self.robot_interface.brake()
        rospy.set_param(FORCE_ZERO_LL_KP_PARAM, False)
        rospy.set_param("/controller_finished", True)
        self._save_results()
        rospy.loginfo("Direct teleop stopped")


if __name__ == "__main__":
    rospy.init_node("controller_direct_teleop")
    DirectTeleopROSNode().run()
