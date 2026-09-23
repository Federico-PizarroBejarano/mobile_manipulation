#!/usr/bin/env python3
"""RL teleop: joy EE (MoMa integrate) + SAC base → MpcPlan → low_level_cmd_node."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rospy
import torch
from mobile_manipulation_central.ros_interface import (
    JoystickButtonInterface,
    MobileManipulatorROSInterface,
)
from robotiq_3f_gripper_articulated_msgs.msg import (
    Robotiq3FGripperRobotInput,
    Robotiq3FGripperRobotOutput,
)
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import Joy

from mm_control.robot import MobileManipulator3D
from mm_rl.sac.sac import SAC
from mm_run.msg import MpcPlan
from mm_utils import math as mm_math
from mm_utils import parsing
from mm_utils.diff_ik import (
    build_ik_params_from_config,
    solve_diff_ik,
    spatial_jacobian,
)
from mm_utils.ee_motion_infer import integrate_ee_motion
from mm_utils.logging import DataLogger
from mm_utils.metrics import MPSFMetricsCollector
from mm_utils.robotiq_gripper import (
    GRIPPER_TOGGLE_BUTTON,
    gripper_position,
    seed_gripper_mode_position,
    toggle_gripper_open,
)
from mm_utils.teleop_joy import (
    FORCE_ZERO_LL_KP_PARAM,
    STICKS_ACTIVE_PARAM,
    axes_to_ee_velocity,
    chassis_ee_twist_to_world,
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


def _base_pose_from_q(q):
    q = np.asarray(q, dtype=float).reshape(-1)
    yaw = float(q[2])
    base_pos = np.array([q[0], q[1], 0.0], dtype=np.float64)
    base_orn = np.array(
        [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=np.float64
    )
    return base_pos, base_orn


def _world_to_base_frame(pos_w, orn_w, base_pos_w, base_orn_w):
    pos_rel = np.asarray(pos_w, dtype=float).reshape(3) - np.asarray(
        base_pos_w, dtype=float
    ).reshape(3)
    q_inv = mm_math.quat_inverse(base_orn_w)
    pos_b = mm_math.quat_rotate(q_inv, pos_rel)
    orn_b = mm_math.quat_multiply(q_inv, orn_w)
    return pos_b, orn_b


def build_rl_observation(
    q,
    ee_pos_w,
    ee_orn_w,
    v_ee_world,
    hat_pos_w,
    hat_orn_w,
    goal_pos_w,
    goal_orn_w,
    prev_action,
):
    """Build observation vector matching ``BaseRLEnv._get_observation``."""
    base_pos_w, base_orn_w = _base_pose_from_q(q)
    ee_pos_b, ee_orn_b = _world_to_base_frame(
        ee_pos_w, ee_orn_w, base_pos_w, base_orn_w
    )
    hat_pos_b, hat_orn_b = _world_to_base_frame(
        hat_pos_w, hat_orn_w, base_pos_w, base_orn_w
    )
    goal_pos_b, goal_orn_b = _world_to_base_frame(
        goal_pos_w, goal_orn_w, base_pos_w, base_orn_w
    )
    q_inv = mm_math.quat_inverse(base_orn_w)
    v_ee = np.asarray(v_ee_world, dtype=np.float64).reshape(6)
    ee_lin_b = mm_math.quat_rotate(q_inv, v_ee[:3])
    ee_ang_b = mm_math.quat_rotate(q_inv, v_ee[3:])
    prev = np.asarray(prev_action, dtype=np.float32).reshape(-1)
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    obs = np.concatenate(
        [
            np.concatenate([ee_lin_b, ee_ang_b]),
            ee_pos_b,
            mm_math.quat_to_rot(ee_orn_b).flatten(),
            hat_pos_b,
            mm_math.quat_to_rot(hat_orn_b).flatten(),
            goal_pos_b,
            mm_math.quat_to_rot(goal_orn_b).flatten(),
            q,
            prev,
        ]
    )
    return obs.astype(np.float32)


def _unscale_action(action, low, high):
    return low + (action + 1.0) * 0.5 * (high - low)


class RLTeleopROSNode:
    """Joystick EE + SAC base policy, published as short constant-velocity MpcPlans."""

    def __init__(self):
        argv = rospy.myargv(argv=sys.argv)
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", required=True)
        parser.add_argument("--ctrl_config", type=str, default="default")
        parser.add_argument("--planner_config", type=str, default="default")
        parser.add_argument("--logging_sub_folder", type=str, default="default")
        parser.add_argument(
            "--checkpoint",
            type=str,
            default="",
            help="Override rl.checkpoint path",
        )
        parser.add_argument("--device", type=str, default="cpu")
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
            raise ValueError("rl_teleop_ros requires controller.teleop.enabled: true")

        self.teleop_max_ee_vel = teleop_cfg["max_ee_vel"]
        self.teleop_enable_button = teleop_cfg["enable_button"]
        self.teleop_ee_yaw_buttons = teleop_cfg["ee_yaw_buttons"]

        self.nu = int(self.ctrl_config["robot"]["dims"]["u"])
        self.dof = int(self.ctrl_config["robot"]["dims"]["q"])
        self.mpc_dt = float(self.ctrl_config["dt"])
        self.rate_hz = float(self.ctrl_config["ctrl_rate"])
        if self.rate_hz <= 0.0:
            raise ValueError(f"ctrl_rate must be positive, got {self.rate_hz}")
        self.dt = 1.0 / self.rate_hz

        robot_overrides = config.get("robot", {})
        base_lin = float(robot_overrides.get("base_linear_vel_limit", 0.3))
        base_ang = float(robot_overrides.get("base_angular_vel_limit", 0.3))
        self.base_input_low = np.array([-base_lin, -base_lin, -base_ang], dtype=float)
        self.base_input_high = np.array([base_lin, base_lin, base_ang], dtype=float)

        # MoMa EE-integrator look-ahead (deploy only; training uses final goal in obs)
        self.obs_horizon_m = float(config.get("goal", {}).get("obs_horizon_m", 1.5))

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
        self.bag_recorder = TrialBagRecorder(
            session_root(self.logger.base_directory, self.session_timestamp),
            record_bag=bool(record_bag),
            bag_all=bool(bag_all),
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
        self.ik_params["ee_max_linear_vel"] = float(
            robot_overrides.get(
                "ee_linear_vel_limit", self.ik_params["ee_max_linear_vel"]
            )
        )
        self.ik_params["ee_max_angular_vel"] = float(
            robot_overrides.get(
                "ee_angular_vel_limit", self.ik_params["ee_max_angular_vel"]
            )
        )

        # SAC policy
        rl_cfg = config.get("rl", {})
        sac_cfg = rl_cfg.get("sac", {})
        state_dim = 6 + 12 + 12 + 12 + self.dof + 3
        action_dim = 3
        self.agent = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            action_range=(-1.0, 1.0),
            lr=sac_cfg.get("lr", 1e-4),
            gamma=sac_cfg.get("gamma", 0.99),
            tau=sac_cfg.get("tau", 0.001),
            alpha=sac_cfg.get("alpha", 0.2),
            auto_alpha=sac_cfg.get("auto_alpha", True),
            min_alpha=sac_cfg.get("min_alpha", 0.0),
            hidden_layers=sac_cfg.get("hidden_layers", [512, 512, 256]),
            buffer_size=sac_cfg.get("buffer_size", 100000),
            device=args.device,
            infinite_horizon=sac_cfg.get("infinite_horizon", True),
        )
        ckpt = args.checkpoint or rl_cfg.get("checkpoint", "")
        if not ckpt:
            raise ValueError(
                "rl_teleop_ros requires --checkpoint or rl.checkpoint in config"
            )
        ckpt_path = Path(ckpt).expanduser()
        if not ckpt_path.is_file():
            # Try relative to mm_rl package share / source
            try:
                pkg = Path(parsing.parse_ros_path({"package": "mm_rl", "path": "."}))
                alt = (pkg / ckpt).resolve()
                if alt.is_file():
                    ckpt_path = alt
            except Exception:
                pass
        if not ckpt_path.is_file():
            # Workspace-relative fallback used in docs
            alt2 = Path.cwd() / ckpt
            if alt2.is_file():
                ckpt_path = alt2
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"RL checkpoint not found: {ckpt}")
        rospy.loginfo("Loading RL checkpoint: %s", ckpt_path)
        self.agent.load(str(ckpt_path))

        self.prev_action = np.zeros(3, dtype=np.float32)
        self.joy_axes = np.zeros(6)
        self.joy_buttons = np.zeros(16)
        self.joy_lock = threading.Lock()
        self._sticks_active_param = self.teleop_enable_button is None
        self.ctrl_c = False

        self.robot_interface = MobileManipulatorROSInterface()
        self.start_end_button_interface = JoystickButtonInterface(2)
        self.gripper_button_interface = JoystickButtonInterface(GRIPPER_TOGGLE_BUTTON)
        self._gripper_open = True
        self._gripper_no_sub_warned = False
        self._gripper_status = None
        self._gripper_cmd = None
        self._gripper_pub = rospy.Publisher(
            "/Robotiq3FGripperRobotOutput",
            Robotiq3FGripperRobotOutput,
            queue_size=1,
        )

        self.mpc_plan_pub = rospy.Publisher("mpc_plan", MpcPlan, queue_size=1)
        rospy.Subscriber("/bluetooth_teleop/joy", Joy, self._joy_cb, queue_size=1)
        rospy.Subscriber(
            "/Robotiq3FGripperRobotInput",
            Robotiq3FGripperRobotInput,
            self._gripper_status_cb,
            queue_size=1,
        )
        rospy.on_shutdown(self._on_shutdown)

        rospy.set_param(FORCE_ZERO_LL_KP_PARAM, True)
        rospy.loginfo(
            "RL teleop: enable_button=%s max_ee_vel=%s checkpoint=%s",
            self.teleop_enable_button,
            self.teleop_max_ee_vel,
            ckpt_path,
        )

    def _joy_cb(self, msg):
        with self.joy_lock:
            store_joy_axes(msg.axes, self.joy_axes)
            self.joy_buttons = np.array(msg.buttons, dtype=float)

    def _on_shutdown(self):
        self.ctrl_c = True
        try:
            self._publish_velocity_plan(np.zeros(self.nu), rospy.Time.now().to_sec())
            self.robot_interface.brake()
        except Exception:
            pass
        self._save_results()

    def _save_results(self):
        if self._saved:
            return
        self._saved = True
        try:
            self.bag_recorder.stop()
            self.logger.save(session_timestamp=self.session_timestamp)
            clear_experiment_timestamp()
        except Exception as exc:
            rospy.logwarn("RL teleop save failed: %s", exc)

    def _sync_session_timestamp(self):
        self.session_timestamp = resolve_experiment_timestamp(create=False) or (
            self.session_timestamp
        )

    def _wait_for_joint_states(self, rate):
        rospy.loginfo("Waiting for joint states...")
        while not self.ctrl_c and not rospy.is_shutdown():
            if self.robot_interface.ready():
                return
            rate.sleep()
        raise rospy.ROSInterruptException("shutdown while waiting for joint states")

    def _wait_for_start(self, rate):
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

    def _gripper_status_cb(self, msg):
        self._gripper_status = msg

    def _ensure_gripper_cmd(self):
        """Keep one Robotiq output message; only rPRA changes on later toggles."""
        if self._gripper_cmd is not None:
            return self._gripper_cmd
        cmd = Robotiq3FGripperRobotOutput()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rATR = 0
        cmd.rICF = 0
        cmd.rSPA = 255
        cmd.rFRA = 150
        if self._gripper_status is not None:
            r_mod, r_pra = seed_gripper_mode_position(
                self._gripper_status.gMOD,
                self._gripper_status.gPRA,
                self._gripper_open,
            )
        else:
            r_mod, r_pra = seed_gripper_mode_position(None, None, self._gripper_open)
        cmd.rMOD = r_mod
        cmd.rPRA = r_pra
        self._gripper_cmd = cmd
        return cmd

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
        cmd = self._ensure_gripper_cmd()
        self._gripper_open = toggle_gripper_open(self._gripper_open)
        cmd.rPRA = gripper_position(self._gripper_open)
        cmd.rACT = 1
        cmd.rGTO = 1
        self._gripper_pub.publish(cmd)
        rospy.loginfo(
            "Gripper %s (Triangle/Y)",
            "open" if self._gripper_open else "closed",
        )

    def _policy_base_vel(self, action):
        action = np.clip(np.asarray(action, dtype=float).reshape(3), -1.0, 1.0)
        return np.array(
            [
                _unscale_action(
                    action[0], self.base_input_low[0], self.base_input_high[0]
                ),
                _unscale_action(
                    action[1], self.base_input_low[1], self.base_input_high[1]
                ),
                _unscale_action(
                    action[2], self.base_input_low[2], self.base_input_high[2]
                ),
            ],
            dtype=float,
        )

    def _compute_v_cmd(self):
        with self.joy_lock:
            axes = self.joy_axes.copy()
            buttons = self.joy_buttons.copy()
            enabled = teleop_enable_held(buttons, self.teleop_enable_button)

        desired_ee_cmd = np.zeros(6, dtype=float)
        desired_base_vel = np.zeros(3, dtype=float)
        if not enabled:
            self.prev_action = np.zeros(3, dtype=np.float32)
            return (
                np.zeros(self.nu, dtype=float),
                enabled,
                desired_base_vel,
                desired_ee_cmd,
                axes,
                buttons,
            )

        q = np.asarray(self.robot_interface.q, dtype=float).reshape(-1)[: self.dof]
        yaw = float(q[2])
        desired_ee_cmd = chassis_ee_twist_to_world(
            axes_to_ee_velocity(
                axes, buttons, self.teleop_max_ee_vel, self.teleop_ee_yaw_buttons
            ),
            yaw,
        )
        ee_pos, ee_orn = self.robot_mdl.getEE(q)
        motion = integrate_ee_motion(
            ee_pos,
            ee_orn,
            desired_ee_cmd,
            dt=self.dt,
            horizon_m=self.obs_horizon_m,
        )

        if not motion["active"]:
            # Idle stick: hold; no base policy command
            self.prev_action = np.zeros(3, dtype=np.float32)
            return (
                np.zeros(self.nu, dtype=float),
                enabled,
                desired_base_vel,
                desired_ee_cmd,
                axes,
                buttons,
            )

        obs = build_rl_observation(
            q,
            ee_pos,
            ee_orn,
            motion["v_ee"],
            motion["hat_pos"],
            motion["hat_orn"],
            motion["goal_pos"],
            motion["goal_orn"],
            self.prev_action,
        )
        with torch.no_grad():
            action = self.agent.select_action(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(3)
        desired_base_vel = self._policy_base_vel(action)
        self.prev_action = action.copy()

        # IK: world EE twist from integrator → body ang for Casadi J
        _, twist_ik = mm_math.ee_twist_world_and_mpc_reference(
            motion["v_ee"][:3],
            motion["v_ee"][3:],
            ee_orn,
            clamp_limits=(
                float(self.ik_params["ee_max_linear_vel"]),
                float(self.ik_params["ee_max_angular_vel"]),
            ),
        )
        J = spatial_jacobian(self.robot_mdl, q)
        v_cmd = solve_diff_ik(
            J, twist_ik, base_vel=desired_base_vel, ik_params=self.ik_params
        )
        return v_cmd, enabled, desired_base_vel, desired_ee_cmd, axes, buttons

    def _record_cycle(
        self,
        timestamp,
        cycle_period,
        v_cmd,
        enabled,
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
            teleop_mode="rl",
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
        self.metrics_collector.update(
            {},
            states,
            v_cmd,
            desired_base_vel if enabled else None,
            desired_ee_vel if enabled else None,
            self.metrics_controller,
            (q, v),
            self.mpc_dt,
            teleop_mode="rl",
            teleop_enabled=enabled,
            commanded_ee_twist=ee_cmd_twist,
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
        rospy.loginfo("RL teleop started (session %s)", self.session_timestamp)
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
        rospy.loginfo("RL teleop stopped")


if __name__ == "__main__":
    rospy.init_node("controller_rl_teleop")
    RLTeleopROSNode().run()
