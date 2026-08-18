"""High-rate ``cmd_vel`` from ``mm_run/MpcPlan`` + optional joint P tracking."""

import argparse
import sys
import threading
import time

import numpy as np
import rospy
from mobile_manipulation_central.ros_interface import MobileManipulatorROSInterface
from rosgraph_msgs.msg import Clock

from mm_control.robot import MobileManipulator3D
from mm_run.msg import MpcPlan
from mm_utils import parsing
from mm_utils.mpc_plan_tracking import build_plan_interpolators, low_level_velocity_step


class LowLevelCmdNode:
    def __init__(self):
        argv = rospy.myargv(argv=sys.argv)
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c", "--config", required=True, help="Path to configuration file."
        )
        parser.add_argument(
            "--ctrl_config",
            type=str,
            help="controller config overlay",
        )
        parser.add_argument(
            "--planner_config",
            type=str,
            help="planner config overlay",
        )
        args = parser.parse_args(argv[1:])

        config = parsing.load_config(args.config)
        if args.ctrl_config and args.ctrl_config != "default":
            ctrl_overlay = parsing.load_config(args.ctrl_config)
            config = parsing.recursive_dict_update(config, ctrl_overlay)
        if args.planner_config and args.planner_config != "default":
            planner_overlay = parsing.load_config(args.planner_config)
            config = parsing.recursive_dict_update(config, planner_overlay)

        self.ctrl_config = config["controller"]
        self.sim_dt = float(config["simulation"]["timestep"])
        self._mpc_dt_cfg = float(self.ctrl_config["dt"])
        self.cmd_vel_type = self.ctrl_config["cmd_vel_type"]
        self.rate_hz = float(self.ctrl_config["cmd_vel_pub_rate"])

        ll = self.ctrl_config.get("low_level_tracking", {})
        self.ll_enabled = bool(ll.get("enabled", False))
        self.kp = np.asarray(ll.get("kp", []), dtype=float).reshape(-1)
        self.kp_arg = (
            self.kp
            if (self.ll_enabled and self.kp.size > 0 and np.any(self.kp != 0.0))
            else None
        )

        self.robot_mdl = MobileManipulator3D(self.ctrl_config)
        self.lb_u_full = np.asarray(self.robot_mdl.lb_u, dtype=float).reshape(-1)
        self.ub_u_full = np.asarray(self.robot_mdl.ub_u, dtype=float).reshape(-1)
        self.nu = int(self.ctrl_config["robot"]["dims"]["u"])

        self.robot_interface = MobileManipulatorROSInterface()
        self.cmd_vel = np.zeros(self.nu, dtype=float)

        self.lock = threading.Lock()
        self.interps = None
        self.plan_stamp = None
        self.horizon_s = 0.0

        rospy.Subscriber("mpc_plan", MpcPlan, self._on_plan, queue_size=1)

        self._plan_count = 0
        self._timer_count = 0
        self._prev_clock_stamp = None
        self._use_clock_drive = rospy.get_param("/use_sim_time", False)
        self.timer = None
        if self._use_clock_drive:
            # One cmd_vel update per /clock tick matches experiment.py (one low_level step
            # per Bullet step). rospy.Timer can drift vs sim time and destabilize contacts.
            rospy.Subscriber("/clock", Clock, self._on_clock, queue_size=1)
        else:
            dt_pub = 1.0 / self.rate_hz
            ds = int(dt_pub)
            dns = int((dt_pub - ds) * 1e9)
            self.timer = rospy.Timer(rospy.Duration(ds, dns), self._on_timer)

    def _reject_plan(self, reason: str):
        rospy.logwarn_throttle(
            5.0,
            "low_level_cmd_node: invalid MpcPlan (%s); clearing plan and zeroing cmd.",
            reason,
        )
        with self.lock:
            self.interps = None
            self.plan_stamp = None
            self.horizon_s = 0.0
            self.cmd_vel.fill(0.0)

    def _on_plan(self, msg: MpcPlan):
        N, nu, dof = int(msg.N), int(msg.nu), int(msg.dof)
        if nu != self.nu:
            self._reject_plan("nu %d != configured nu %d" % (nu, self.nu))
            return
        if len(msg.u_flat) != N * nu or len(msg.v_flat) != (N + 1) * nu:
            self._reject_plan("u_flat/v_flat length mismatch for N/nu")
            return
        if len(msg.q_flat) != (N + 1) * dof:
            self._reject_plan("q_flat length mismatch for N/dof")
            return

        mpc_dt = float(msg.mpc_dt)
        if not np.isfinite(mpc_dt) or mpc_dt <= 0.0:
            self._reject_plan("mpc_dt is not finite and positive")
            return

        try:
            u_bar = np.asarray(msg.u_flat, dtype=float).reshape(N, nu)
            v_bar = np.asarray(msg.v_flat, dtype=float).reshape(N + 1, nu)
            q_bar = np.asarray(msg.q_flat, dtype=float).reshape(N + 1, dof)
        except Exception as exc:
            self._reject_plan("reshape failed: %s" % exc)
            return

        if not (
            np.isfinite(u_bar).all()
            and np.isfinite(v_bar).all()
            and np.isfinite(q_bar).all()
        ):
            self._reject_plan("non-finite values in u_flat, v_flat, or q_flat")
            return

        if self._plan_count == 0 and abs(mpc_dt - self._mpc_dt_cfg) > 1e-9:
            rospy.logwarn(
                "MpcPlan.mpc_dt (%s) != controller.dt (%s); using message value.",
                mpc_dt,
                self._mpc_dt_cfg,
            )
        try:
            interps = build_plan_interpolators(
                mpc_dt, u_bar, v_bar, q_bar, self.cmd_vel_type
            )
        except Exception as exc:
            self._reject_plan("build_plan_interpolators failed: %s" % exc)
            return

        with self.lock:
            self.interps = interps
            self.plan_stamp = msg.header.stamp
            self.horizon_s = float(N) * mpc_dt

        self._plan_count += 1
        if self._plan_count == 1:
            rospy.loginfo(
                "low_level_cmd_node: first MpcPlan (N=%d nu=%d dof=%d mpc_dt=%.4f).",
                N,
                nu,
                dof,
                mpc_dt,
            )

    def _on_clock(self, msg: Clock):
        stamp = msg.clock
        if self._prev_clock_stamp is None:
            sim_dt = self.sim_dt
        else:
            sim_dt = stamp.to_sec() - self._prev_clock_stamp.to_sec()
            if sim_dt <= 0.0 or sim_dt > 10.0 * self.sim_dt:
                sim_dt = self.sim_dt
        self._prev_clock_stamp = stamp
        self._run_cmd_step(stamp, sim_dt)

    def _on_timer(self, event):
        self._run_cmd_step(rospy.Time.now(), self.sim_dt)

    def _run_cmd_step(self, now: rospy.Time, sim_dt: float):
        wall_t0 = time.perf_counter()
        self._timer_count += 1

        t_elapsed = None
        stale = "n/a"
        out = np.zeros(self.nu, dtype=float)
        try:
            q = np.asarray(self.robot_interface.q, dtype=float).reshape(-1)
            lb_u, ub_u = self.lb_u_full[: self.nu], self.ub_u_full[: self.nu]

            with self.lock:
                interps = self.interps
                plan_stamp = self.plan_stamp
                horizon_s = self.horizon_s

                if interps is None or plan_stamp is None:
                    self.cmd_vel.fill(0.0)
                    out = self.cmd_vel.copy()
                    t_elapsed = None
                    stale = False
                else:
                    t_elapsed = max(0.0, (now - plan_stamp).to_sec())
                    if t_elapsed >= horizon_s:
                        self.cmd_vel.fill(0.0)
                        out = self.cmd_vel.copy()
                        stale = True
                    else:
                        dof = interps.dof
                        q_meas = q[:dof] if dof > 0 else q
                        self.cmd_vel[:] = low_level_velocity_step(
                            self.cmd_vel,
                            t_elapsed,
                            sim_dt,
                            interps,
                            q_meas,
                            self.kp_arg,
                            lb_u,
                            ub_u,
                        )
                        out = self.cmd_vel.copy()
                        stale = False
        except Exception as exc:
            rospy.logerr_throttle(
                5.0,
                "low_level_cmd_node: cmd step error (%s); braking.",
                exc,
            )
            with self.lock:
                self.interps = None
                self.plan_stamp = None
                self.horizon_s = 0.0
                self.cmd_vel.fill(0.0)
            out = np.zeros(self.nu, dtype=float)
            t_elapsed = None
            stale = "n/a"

        self.robot_interface.publish_cmd_vel(out)

        wall_dt_ms = (time.perf_counter() - wall_t0) * 1000.0
        ctrl_started = rospy.get_param("/controller_started", False)
        wait_hint = ""
        if self._plan_count == 0 and not ctrl_started:
            wait_hint = (
                "\n  (no MpcPlan yet: mpc_ros still initializing or before start)"
            )
        drive = "clock" if self._use_clock_drive else "timer"
        rospy.loginfo_throttle(
            0.5,
            "\n----- low_level_cmd_node -----\n"
            "  drive=%s  wall_cb_ms=%.3f  sim_dt_cfg=%.4f  pub_rate_hz=%.1f"
            "  plans_rx=%d  ticks=%d\n"
            "  t_elapsed=%s  stale=%s  |cmd|=%.4f%s",
            drive,
            wall_dt_ms,
            self.sim_dt,
            self.rate_hz,
            self._plan_count,
            self._timer_count,
            "n/a" if t_elapsed is None else f"{t_elapsed:.4f}",
            stale if t_elapsed is not None else "n/a",
            float(np.linalg.norm(out)),
            wait_hint,
        )


def main():
    rospy.init_node("low_level_cmd_node")
    LowLevelCmdNode()
    rospy.spin()


if __name__ == "__main__":
    main()
