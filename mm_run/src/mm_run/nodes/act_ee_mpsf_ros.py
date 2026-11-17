#!/usr/bin/env python3
"""ACT EE → MPSF: policy outputs desired_ee_vel (6); MPSF executes.

Thin subclass of ``MPSFControllerROSNode``. Logged ``desired_ee_vel`` is already
in the MPC EE-vel convention, so policy actions are written straight into
``references["desired_velocity"]["ee_velocity"]`` (no second teleop frame map).

``--env-state`` selects what is sent as LeRobot ``observation.environment_state``:
``obs_duplicate`` (default, env=q), ``ee_pose`` (6D), or ``zeros`` (width-1 constant).
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
from pathlib import Path

import numpy as np
import rospy
from scipy.spatial.transform import Rotation as Rot

from mm_run.nodes.mpsf_ros import MPSFControllerROSNode
from mm_utils.act_eval import action_width_from_config
from mm_utils.act_policy_runtime import ActChunkRunner, ActPolicySubprocess
from mm_utils.teleop_joy import STICKS_ACTIVE_PARAM
from mm_utils.teleop_session_logging import apply_act_run, apply_teleop_backend


def _checkpoint_config(checkpoint: str) -> dict:
    return json.loads((Path(checkpoint) / "config.json").read_text())


def _nu_from_checkpoint(checkpoint: str) -> int:
    return int(action_width_from_config(_checkpoint_config(checkpoint)))


def _n_action_steps_from_checkpoint(checkpoint: str) -> int:
    cfg = _checkpoint_config(checkpoint)
    n = int(cfg.get("n_action_steps", 0) or 0)
    if n < 1:
        raise ValueError(
            f"checkpoint missing positive n_action_steps: {checkpoint}/config.json"
        )
    return n


class ActEeMpsfROSNode(MPSFControllerROSNode):
    """MPSF teleop references replaced by ACT ``desired_ee_vel`` chunks."""

    def __init__(self):
        argv = rospy.myargv(argv=sys.argv)
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument(
            "--n-action-steps",
            type=int,
            default=12,
            help="ACT execute horizon (frozen: 12; 0=from checkpoint n_action_steps)",
        )
        parser.add_argument(
            "--nu",
            type=int,
            default=0,
            help="ACT action width (0=from checkpoint; 6=desired_ee_vel; 3=xyz)",
        )
        parser.add_argument(
            "--env-state",
            default="obs_duplicate",
            choices=("obs_duplicate", "ee_pose", "zeros"),
            help="ACT observation.environment_state source",
        )
        parser.add_argument(
            "--boundary-blend",
            type=float,
            default=0.5,
            help="Weight on previous chunk leftover at refresh in [0,1] (frozen: 0.5)",
        )
        parser.add_argument(
            "--boundary-blend-mode",
            default="decay",
            choices=("constant", "decay", "head"),
            help="How boundary_blend weight is applied (frozen: decay)",
        )
        parser.add_argument(
            "--boundary-blend-head",
            type=int,
            default=3,
            help="For mode=head: number of steps at the seam to blend",
        )
        parser.add_argument(
            "--temporal-ensemble-coeff",
            default="none",
            help="Paper-style temporal ensembling coeff (none=off; requires na=1)",
        )
        known, remaining = parser.parse_known_args(argv[1:])
        # Parent argparse rejects unknown flags; strip ACT-only args first.
        sys.argv = [argv[0]] + remaining

        self._act_checkpoint = known.checkpoint
        self._act_n_action_steps = (
            int(known.n_action_steps)
            if int(known.n_action_steps) > 0
            else _n_action_steps_from_checkpoint(known.checkpoint)
        )
        self._act_nu = (
            int(known.nu)
            if int(known.nu) > 0
            else _nu_from_checkpoint(known.checkpoint)
        )
        self._act_env_state = str(known.env_state)
        self._act_boundary_blend = float(known.boundary_blend)
        self._act_boundary_blend_mode = str(known.boundary_blend_mode)
        self._act_boundary_blend_head = int(known.boundary_blend_head)
        te_raw = str(known.temporal_ensemble_coeff).strip().lower()
        self._act_temporal_ensemble_coeff = (
            None if te_raw in ("", "none", "null", "off") else float(te_raw)
        )
        if self._act_temporal_ensemble_coeff is not None:
            # Paper / LeRobot path: every-tick infer + online exp average.
            self._act_n_action_steps = 1
            self._act_boundary_blend = 0.0
        if self._act_nu < 1:
            raise ValueError(f"--nu must be >= 1, got {self._act_nu}")
        if self._act_n_action_steps < 1:
            raise ValueError(
                f"--n-action-steps must be >= 1, got {self._act_n_action_steps}"
            )

        super().__init__()
        apply_teleop_backend(self.logger, "act_ee")

        # Policy drives EE refs after Square; joy sticks are ignored.
        self.teleop_enabled = True
        self.teleop_control_mode = "ee"
        self._teleop_enable_active = True
        self._update_mpsf_masks()
        self.controller.set_teleop_effort_mode("ee")
        rospy.set_param(STICKS_ACTIVE_PARAM, True)
        self._sticks_active_param = True
        self._latest_ee_pose = np.zeros(6, dtype=float)

        self._policy = ActPolicySubprocess(self._act_checkpoint, nu=self._act_nu)
        atexit.register(self._policy.close)
        self.runner = ActChunkRunner(
            self._predict_chunk,
            n_action_steps=self._act_n_action_steps,
            boundary_blend=self._act_boundary_blend,
            boundary_blend_mode=self._act_boundary_blend_mode,
            boundary_blend_head=self._act_boundary_blend_head,
            temporal_ensemble_coeff=self._act_temporal_ensemble_coeff,
        )
        apply_act_run(
            self.logger,
            self._act_checkpoint,
            n_action_steps=self.runner.n_action_steps,
            boundary_blend=self.runner.boundary_blend,
            boundary_blend_mode=self.runner.boundary_blend_mode,
            boundary_blend_head=self.runner.boundary_blend_head,
            nu=self._act_nu,
            env_state=self._act_env_state,
            temporal_ensemble_coeff=self.runner.temporal_ensemble_coeff,
        )
        te_s = (
            "off"
            if self.runner.temporal_ensemble_coeff is None
            else f"{self.runner.temporal_ensemble_coeff:g}"
        )
        rospy.loginfo(
            "ACT-EE MPSF: checkpoint=%s nu=%d n_action_steps=%d "
            "boundary_blend=%.3f mode=%s head=%d temporal_ensemble=%s "
            "env_state=%s%s",
            self._act_checkpoint,
            self._act_nu,
            self.runner.n_action_steps,
            self.runner.boundary_blend,
            self.runner.boundary_blend_mode,
            self.runner.boundary_blend_head,
            te_s,
            self._act_env_state,
            " (pad angular vel=0)" if self._act_nu == 3 else "",
        )

    def _predict_chunk(self, q: np.ndarray) -> np.ndarray:
        if self._act_env_state == "ee_pose":
            return self._policy.predict_chunk(q, env_state=self._latest_ee_pose)
        if self._act_env_state == "zeros":
            return self._policy.predict_chunk(q, env_state=np.zeros(1, dtype=float))
        return self._policy.predict_chunk(q)

    def _ee_pose_for_policy(self, q: np.ndarray) -> np.ndarray:
        """Match logged ``ee_pose`` (xyz + rpy); same sources as mpc_ros logging."""
        if self.use_vicon_tool_data:
            ee_pose = np.zeros(6, dtype=float)
            ee_pose[:3] = np.asarray(
                self.vicon_tool_interface.position, dtype=float
            ).reshape(3)
            ee_pose[3:] = Rot.from_quat(
                np.asarray(self.vicon_tool_interface.orientation, dtype=float)
            ).as_euler("xyz")
            return ee_pose
        ee_pos, ee_quat = self.robot.getEE(q)
        return np.hstack(
            [
                np.asarray(ee_pos, dtype=float).reshape(3),
                Rot.from_quat(np.asarray(ee_quat, dtype=float)).as_euler("xyz"),
            ]
        )

    def _wait_for_start(self, rate):
        super()._wait_for_start(rate)
        self.runner.reset()
        self._teleop_enable_active = True
        rospy.set_param(STICKS_ACTIVE_PARAM, True)
        self._sticks_active_param = True
        rospy.loginfo("ACT-EE MPSF: trial started; feeding policy EE velocities")

    def update_references(self, references, robot_states, cycle_period):
        del cycle_period  # policy runs open-loop in time; MPSF uses its own dt
        self.teleop_control_mode = "ee"
        self.controller.set_teleop_effort_mode("ee")
        self.controller.mpsf_base_mask = np.array([0, 0, 0], dtype=bool)
        self.controller.mpsf_ee_mask = np.array([1, 1, 1, 1, 1, 1], dtype=bool)

        q = np.asarray(robot_states[0], dtype=float).reshape(-1)
        if self._act_env_state == "ee_pose":
            self._latest_ee_pose = self._ee_pose_for_policy(q)
        try:
            action = np.asarray(self.runner.step(q), dtype=float).reshape(-1)
        except Exception as exc:  # noqa: BLE001 - keep robot safe on infer errors
            rospy.logerr_throttle(1.0, "ACT-EE predict failed: %s", exc)
            action = np.zeros(self._act_nu, dtype=float)

        if action.size == 6:
            desired_ee_vel = action.copy()
        elif action.size == 3:
            desired_ee_vel = np.zeros(6, dtype=float)
            desired_ee_vel[:3] = action
        else:
            raise ValueError(f"ACT-EE action width must be 3 or 6, got {action.shape}")

        # Match logged frame: already MPC EE-vel convention (no teleop remap).
        desired_velocity = {"ee_velocity": desired_ee_vel}
        self._current_desired_base_vel = None
        self._current_desired_ee_vel = desired_ee_vel
        references["desired_velocity"] = desired_velocity
        self._teleop_enable_active = True
        if not self._sticks_active_param:
            self._sticks_active_param = True
            rospy.set_param(STICKS_ACTIVE_PARAM, True)

    def shutdownhook(self):
        try:
            self._policy.close()
        except Exception:
            pass
        super().shutdownhook()


if __name__ == "__main__":
    rospy.init_node("controller_ros_act_ee_mpsf")
    node = ActEeMpsfROSNode()
    node.run()
    node._save_metrics()
