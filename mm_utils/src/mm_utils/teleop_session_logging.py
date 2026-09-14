"""Shared teleop session logging: paths, DataLogger field helpers, rosbag recorder."""

from __future__ import annotations

import datetime
import signal
import subprocess
from pathlib import Path

import numpy as np

EXPERIMENT_TIMESTAMP_PARAM = "/experiment_timestamp"
SESSION_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

DEFAULT_CURATED_BAG_TOPICS = [
    "/bluetooth_teleop/joy",
    "/ridgeback/joint_states",
    "/ur10/joint_states",
    "/ridgeback/cmd_vel",
    "/ur10/cmd_vel",
    "/mpc_plan",
    "/tf",
    "/tf_static",
    "/Robotiq3FGripperRobotOutput",
    "/Robotiq3FGripperRobotInput",
]


def session_root(base_directory, session_timestamp):
    return Path(base_directory) / str(session_timestamp)


def format_session_timestamp(when=None):
    """Wall-time stamp used as the results session folder name."""
    when = when or datetime.datetime.now()
    return when.strftime(SESSION_TIMESTAMP_FORMAT)


def resolve_experiment_timestamp(create=True):
    """Return ``/experiment_timestamp``, creating it only if missing.

    Never overwrites an existing value so sim and control share one folder even
    when one node finishes ``__init__`` much later (e.g. PyBullet GUI).
    """
    import rospy

    if rospy.has_param(EXPERIMENT_TIMESTAMP_PARAM):
        return str(rospy.get_param(EXPERIMENT_TIMESTAMP_PARAM))
    if not create:
        return None
    ts = format_session_timestamp()
    rospy.set_param(EXPERIMENT_TIMESTAMP_PARAM, ts)
    return ts


def clear_experiment_timestamp():
    """Drop the shared stamp so the next launch does not reuse this folder."""
    import rospy

    if not rospy.has_param(EXPERIMENT_TIMESTAMP_PARAM):
        return
    try:
        rospy.delete_param(EXPERIMENT_TIMESTAMP_PARAM)
    except KeyError:
        pass


def vicon_object_topic(name):
    name = str(name)
    return f"/vicon/{name}/{name}"


def build_curated_bag_topics(
    tool_vicon_name,
    base_vicon_name="ThingBase_Fed",
    plan_topic="/mpc_plan",
):
    topics = list(DEFAULT_CURATED_BAG_TOPICS)
    topics[topics.index("/mpc_plan")] = str(plan_topic)
    topics.append(vicon_object_topic(base_vicon_name))
    topics.append(vicon_object_topic(tool_vicon_name))
    return topics


def build_rosbag_record_cmd(bag_path, bag_all, topics):
    bag_path = Path(bag_path)
    cmd = ["rosbag", "record", "-O", str(bag_path)]
    if bag_all:
        cmd.append("-a")
    else:
        cmd.extend(list(topics))
    return cmd


class TrialBagRecorder:
    def __init__(
        self,
        session_root,
        record_bag,
        bag_all,
        tool_vicon_name,
        base_vicon_name="ThingBase_Fed",
        plan_topic="/mpc_plan",
        popen_fn=subprocess.Popen,
    ):
        self.session_root = Path(session_root)
        self.record_bag = bool(record_bag)
        self.bag_all = bool(bag_all)
        self.tool_vicon_name = tool_vicon_name
        self.base_vicon_name = base_vicon_name
        self.plan_topic = plan_topic
        self._popen_fn = popen_fn
        self._proc = None
        self.bag_path = None
        self.last_error = None

    def start(self):
        if not self.record_bag or self._proc is not None:
            return
        bag_dir = self.session_root / "bag"
        try:
            bag_dir.mkdir(parents=True, exist_ok=True)
            name = "trial_all.bag" if self.bag_all else "trial.bag"
            self.bag_path = bag_dir / name
            topics = build_curated_bag_topics(
                self.tool_vicon_name,
                base_vicon_name=self.base_vicon_name,
                plan_topic=self.plan_topic,
            )
            cmd = build_rosbag_record_cmd(self.bag_path, self.bag_all, topics)
            self._proc = self._popen_fn(cmd)
            self.last_error = None
        except Exception as exc:  # noqa: BLE001 — never block teleop
            self.last_error = exc
            self._proc = None
            self.bag_path = None

    def stop(self):
        if self._proc is None:
            return
        try:
            self._proc.send_signal(signal.SIGINT)
            self._proc.wait(timeout=10)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None


def pad_joy_buttons(buttons, length=16):
    out = np.zeros(length, dtype=float)
    b = np.asarray(buttons, dtype=float).reshape(-1)
    n = min(length, b.size)
    out[:n] = b[:n]
    return out


def mode_to_int(mode):
    if mode == "base":
        return 0
    if mode == "ee":
        return 1
    return -1


def commanded_ee_twist(J, u_cmd):
    J = np.asarray(J, dtype=float).reshape(6, -1)
    u_cmd = np.asarray(u_cmd, dtype=float).reshape(-1)
    return (J @ u_cmd).reshape(6)


def append_teleop_sample(logger, **kwargs):
    logger.append("ts", float(kwargs["ts"]))
    logger.append("q", np.asarray(kwargs["q"], dtype=float).reshape(-1))
    logger.append("v", np.asarray(kwargs["v"], dtype=float).reshape(-1))
    logger.append("base_pose", np.asarray(kwargs["base_pose"], dtype=float).reshape(3))
    logger.append("base_vel", np.asarray(kwargs["base_vel"], dtype=float).reshape(3))
    logger.append("ee_pose", np.asarray(kwargs["ee_pose"], dtype=float).reshape(6))
    logger.append("ee_vel", np.asarray(kwargs["ee_vel"], dtype=float).reshape(6))
    logger.append("joy_axes", np.asarray(kwargs["joy_axes"], dtype=float).reshape(6))
    logger.append("joy_buttons", pad_joy_buttons(kwargs["joy_buttons"]))
    logger.append("teleop_mode", mode_to_int(kwargs["teleop_mode"]))
    logger.append("teleop_enabled", 1.0 if kwargs["teleop_enabled"] else 0.0)
    logger.append(
        "desired_base_vel",
        np.asarray(kwargs["desired_base_vel"], dtype=float).reshape(3),
    )
    logger.append(
        "desired_ee_vel", np.asarray(kwargs["desired_ee_vel"], dtype=float).reshape(6)
    )
    logger.append("u_cmd", np.asarray(kwargs["u_cmd"], dtype=float).reshape(-1))
    logger.append("cycle_period", float(kwargs["cycle_period"]))
