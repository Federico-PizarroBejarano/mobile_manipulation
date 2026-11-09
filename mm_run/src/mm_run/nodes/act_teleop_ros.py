"""ACT teleop: q → ACT chunk → synthetic MpcPlan → low_level_cmd_node (no MPC/MPSF)."""

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
from robotiq_3f_gripper_articulated_msgs.msg import (
    Robotiq3FGripperRobotInput,
    Robotiq3FGripperRobotOutput,
)
from scipy.spatial.transform import Rotation as Rot

from mm_control.robot import MobileManipulator3D
from mm_run.msg import MpcPlan
from mm_utils import parsing
from mm_utils.act_policy_runtime import ActChunkRunner
from mm_utils.diff_ik import spatial_jacobian
from mm_utils.logging import DataLogger
from mm_utils.metrics import MPSFMetricsCollector, populate_collector_from_log
from mm_utils.path_coverage import (
    apply_path_coverage_to_metrics,
    print_trial_metrics_summary,
)
from mm_utils.robotiq_gripper import (
    GRIPPER_TOGGLE_BUTTON,
    gripper_position,
    seed_gripper_mode_position,
    toggle_gripper_open,
)
from mm_utils.teleop_joy import FORCE_ZERO_LL_KP_PARAM, STICKS_ACTIVE_PARAM
from mm_utils.teleop_mapping import (
    END_TRIAL_BUTTON,
    START_TRIAL_BUTTON,
    trial_button_action,
)
from mm_utils.teleop_session_logging import (
    TrialBagRecorder,
    append_teleop_sample,
    apply_logging_profile,
    apply_teleop_backend,
    clear_experiment_timestamp,
    commanded_ee_twist,
    commanded_ee_twists,
    logging_profile,
    maybe_prompt_trial_metadata,
    resolve_experiment_timestamp,
    session_root,
)


class ActTeleopROSNode:
    """ACT teleop that publishes short constant-velocity MpcPlans (no MPC)."""

    def __init__(self):
        argv = rospy.myargv(argv=sys.argv)
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", required=True)
        parser.add_argument("--ctrl_config", type=str, default="default")
        parser.add_argument("--planner_config", type=str, default="default")
        parser.add_argument("--logging_sub_folder", type=str, default="default")
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--n-action-steps", type=int, default=25)
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
        self.nu = int(self.ctrl_config["robot"]["dims"]["u"])
        self.dof = int(self.ctrl_config["robot"]["dims"]["q"])
        self.mpc_dt = float(self.ctrl_config["dt"])
        self.rate_hz = float(self.ctrl_config["ctrl_rate"])
        if self.rate_hz <= 0.0:
            raise ValueError(f"ctrl_rate must be positive, got {self.rate_hz}")

        self.logger = DataLogger(copy.deepcopy(config), name="control")
        apply_logging_profile(self.logger, logging_profile(config))
        apply_teleop_backend(self.logger, "act")
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
        self._sticks_active_param = True
        self.ctrl_c = False

        # Lazy import: lerobot/torch are only needed on the IL env for this node.
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
            from lerobot.policies.factory import make_pre_post_processors
        except ImportError as exc:
            raise ImportError(
                "act_teleop_ros requires lerobot>=0.4 (ACTPolicy and "
                "make_pre_post_processors) and torch on this node's Python "
                "(run from the IL environment)"
            ) from exc
        import torch

        self._torch = torch
        self.policy = ACTPolicy.from_pretrained(args.checkpoint)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config, pretrained_path=args.checkpoint
        )
        self.runner = ActChunkRunner(
            self._predict_chunk, n_action_steps=int(args.n_action_steps)
        )

        self.robot_interface = MobileManipulatorROSInterface()
        self.start_button_interface = JoystickButtonInterface(START_TRIAL_BUTTON)
        self.end_button_interface = JoystickButtonInterface(END_TRIAL_BUTTON)
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

        rospy.Subscriber(
            "/Robotiq3FGripperRobotInput",
            Robotiq3FGripperRobotInput,
            self._gripper_status_cb,
            queue_size=1,
        )
        rospy.set_param(STICKS_ACTIVE_PARAM, self._sticks_active_param)
        # Shared YAML may set low_level kp for MPC/MPSF; disable P for ACT teleop
        # (q_bar is a frozen snapshot, so kp fights stick velocity).
        rospy.set_param(FORCE_ZERO_LL_KP_PARAM, True)
        rospy.set_param("/controller_finished", False)
        rospy.set_param("/controller_started", False)
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            "ACT teleop: checkpoint=%s n_action_steps=%d",
            args.checkpoint,
            self.runner.n_action_steps,
        )

    def _sync_session_timestamp(self):
        """Re-read shared stamp (sim may set it after our __init__).

        Always rebinds the bag path so a user folder applied during the start
        prompt is included even when the timestamp itself did not change.
        """
        ts = resolve_experiment_timestamp(create=True)
        if ts != self.session_timestamp:
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
        root = session_root(self.logger.base_directory, self.session_timestamp)
        metrics_saved = False
        logger_saved = False
        try:
            self._populate_metrics()
        except Exception as exc:
            rospy.logerr("Failed to reduce ACT teleop metrics: %s", exc)
        try:
            self.logger.save(session_timestamp=self.session_timestamp)
            logger_saved = True
        except Exception as exc:
            rospy.logerr("Failed to save ACT teleop control log: %s", exc)
        try:
            self.metrics_collector.save(root / "metrics")
            metrics_saved = True
        except Exception as exc:
            rospy.logerr("Failed to save ACT teleop metrics: %s", exc)
        pcfg = self.logger.config.get("logging", {}).get("path_coverage")
        if metrics_saved and isinstance(pcfg, dict):
            try:
                cov = apply_path_coverage_to_metrics(root, pcfg)
                if cov is None:
                    rospy.logwarn(
                        "path_coverage skipped: missing control/data.npz or ee_pose"
                    )
            except ValueError as exc:
                rospy.logerr("path_coverage: %s", exc)
            except Exception as exc:
                rospy.logerr("Failed to apply path_coverage: %s", exc)
        if metrics_saved and not print_trial_metrics_summary(root):
            self.metrics_collector.print_summary()
        self._saved = metrics_saved and logger_saved

    def _wait_for_joint_states(self, rate):
        rospy.loginfo("Waiting for joint states ...")
        while not self.robot_interface.ready() and not rospy.is_shutdown():
            self.robot_interface.brake()
            rate.sleep()
        if rospy.is_shutdown():
            raise rospy.ROSInterruptException("shutdown while waiting for joints")
        rospy.loginfo("ACT teleop received joint states.")

    def _wait_for_start(self, rate):
        maybe_prompt_trial_metadata(self.logger)
        self.start_button_interface.reset_button()
        self.end_button_interface.reset_button()
        enter_pressed = threading.Event()

        if sys.stdin.isatty():
            print("----- Press Square or Enter to start. Circle ends the trial. -----")

            def _wait_enter():
                try:
                    input()
                    enter_pressed.set()
                except EOFError:
                    pass

            threading.Thread(target=_wait_enter, daemon=True).start()
        elif self.start_button_interface.ready():
            print("----- Press Square to start. Circle ends the trial. -----")
        else:
            print("----- Non-interactive start: skipping prompt -----")
            return

        while not self.ctrl_c and not rospy.is_shutdown():
            if (
                trial_button_action(
                    waiting=True,
                    start_latched=self.start_button_interface.button == 1,
                    end_latched=self.end_button_interface.button == 1,
                )
                == "start"
            ):
                self.start_button_interface.reset_button()
                self.end_button_interface.reset_button()
                return
            if enter_pressed.is_set():
                self.end_button_interface.reset_button()
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

    def _predict_chunk(self, q: np.ndarray) -> np.ndarray:
        """Return a full unnormalized action chunk ``(H, nu)`` of joint velocities."""
        torch = self._torch
        obs = {
            "observation.environment_state": torch.as_tensor(
                np.asarray(q, dtype=np.float32).reshape(1, -1)
            )
        }
        with torch.no_grad():
            obs = self.preprocessor(obs)
            chunk = self.policy.predict_action_chunk(obs)  # (1, H, nu), normalized
            actions = [self.postprocessor(chunk[:, i]) for i in range(chunk.shape[1])]
        chunk = torch.cat(actions, dim=0).detach().cpu().numpy().astype(float)
        if chunk.ndim != 2 or chunk.shape[1] != self.nu:
            raise ValueError(f"ACT chunk must be (H, {self.nu}), got {chunk.shape}")
        return chunk

    def _compute_v_cmd(self, cycle_period):
        del cycle_period  # unused for ACT
        q = np.asarray(self.robot_interface.q, dtype=float).reshape(-1)[: self.dof]
        v_cmd = self.runner.step(q)
        # Logging placeholders for unused teleop fields.
        enabled = True
        desired_base_vel = np.zeros(3, dtype=float)
        desired_ee_vel = np.zeros(6, dtype=float)
        joy_axes = np.zeros(6, dtype=float)
        joy_buttons = np.zeros(16, dtype=float)
        return (
            v_cmd,
            enabled,
            "act",
            desired_base_vel,
            desired_ee_vel,
            joy_axes,
            joy_buttons,
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

    def _populate_metrics(self):
        data = self.logger.data
        twists = None
        if (
            data.get("q") is not None
            and data.get("u_cmd") is not None
            and len(data["q"])
        ):
            twists = commanded_ee_twists(
                data["q"],
                data["u_cmd"],
                lambda q: spatial_jacobian(self.robot_mdl, q),
            )
        populate_collector_from_log(
            self.metrics_collector,
            data,
            jerk_dt=None,
            state_lb=self.robot_mdl.lb_x,
            state_ub=self.robot_mdl.ub_x,
            base_mask=self.metrics_controller.mpsf_base_mask,
            ee_mask=self.metrics_controller.mpsf_ee_mask,
            commanded_ee_twists=twists,
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
        rospy.loginfo("ACT teleop started (session %s)", self.session_timestamp)
        self._cycle_t0 = time.perf_counter()

        while not self.ctrl_c and not rospy.is_shutdown():
            if (
                trial_button_action(
                    waiting=False,
                    start_latched=self.start_button_interface.button == 1,
                    end_latched=self.end_button_interface.button == 1,
                )
                == "end"
            ):
                self.end_button_interface.reset_button()
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
            ) = self._compute_v_cmd(cycle_period)
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
        rospy.loginfo("ACT teleop stopped")


if __name__ == "__main__":
    rospy.init_node("controller_act_teleop")
    ActTeleopROSNode().run()
