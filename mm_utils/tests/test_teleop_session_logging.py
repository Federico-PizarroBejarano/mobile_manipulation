import signal
from pathlib import Path

import numpy as np

from mm_utils.logging import DataLogger
from mm_utils.teleop_session_logging import (
    TrialBagRecorder,
    append_teleop_sample,
    build_curated_bag_topics,
    build_rosbag_record_cmd,
    clear_experiment_timestamp,
    commanded_ee_twist,
    format_session_timestamp,
    pad_joy_buttons,
    resolve_experiment_timestamp,
    session_root,
    vicon_object_topic,
)


class FakeProc:
    def __init__(self):
        self.pid = 4242
        self.sent = []
        self.waited = False

    def send_signal(self, sig):
        self.sent.append(sig)

    def wait(self, timeout=None):
        self.waited = True
        return 0


def test_session_root_joins_timestamp():
    assert session_root(Path("/tmp/logs"), "2026-09-11_12-00-00") == Path(
        "/tmp/logs/2026-09-11_12-00-00"
    )


def test_format_session_timestamp():
    when = __import__("datetime").datetime(2026, 9, 11, 16, 44, 59)
    assert format_session_timestamp(when) == "2026-09-11_16-44-59"


def test_resolve_experiment_timestamp_does_not_overwrite(monkeypatch):
    store = {"/experiment_timestamp": "existing_stamp"}

    class FakeRospy:
        @staticmethod
        def has_param(name):
            return name in store

        @staticmethod
        def get_param(name):
            return store[name]

        @staticmethod
        def set_param(name, value):
            store[name] = value

        @staticmethod
        def delete_param(name):
            del store[name]

    monkeypatch.setitem(__import__("sys").modules, "rospy", FakeRospy)
    # Force re-import path used inside resolve
    assert resolve_experiment_timestamp(create=True) == "existing_stamp"
    assert store["/experiment_timestamp"] == "existing_stamp"


def test_resolve_experiment_timestamp_creates_when_missing(monkeypatch):
    store = {}

    class FakeRospy:
        @staticmethod
        def has_param(name):
            return name in store

        @staticmethod
        def get_param(name):
            return store[name]

        @staticmethod
        def set_param(name, value):
            store[name] = value

    monkeypatch.setitem(__import__("sys").modules, "rospy", FakeRospy)
    ts = resolve_experiment_timestamp(create=True)
    assert ts == store["/experiment_timestamp"]
    assert len(ts) >= 15


def test_clear_experiment_timestamp(monkeypatch):
    store = {"/experiment_timestamp": "x"}

    class FakeRospy:
        @staticmethod
        def has_param(name):
            return name in store

        @staticmethod
        def delete_param(name):
            del store[name]

    monkeypatch.setitem(__import__("sys").modules, "rospy", FakeRospy)
    clear_experiment_timestamp()
    assert "/experiment_timestamp" not in store


def test_vicon_object_topic():
    assert (
        vicon_object_topic("ThingContainer") == "/vicon/ThingContainer/ThingContainer"
    )


def test_build_rosbag_record_cmd_curated():
    topics = ["/bluetooth_teleop/joy", "/mpc_plan"]
    cmd = build_rosbag_record_cmd(
        Path("/tmp/bag/trial.bag"), bag_all=False, topics=topics
    )
    assert cmd[:3] == ["rosbag", "record", "-O"]
    assert cmd[3] == "/tmp/bag/trial.bag"
    assert "/bluetooth_teleop/joy" in cmd
    assert "-a" not in cmd


def test_build_rosbag_record_cmd_all():
    cmd = build_rosbag_record_cmd(
        Path("/tmp/bag/trial_all.bag"), bag_all=True, topics=[]
    )
    assert cmd == ["rosbag", "record", "-O", "/tmp/bag/trial_all.bag", "-a"]


def test_curated_topics_include_joy_and_vicon():
    topics = build_curated_bag_topics("ThingContainer", base_vicon_name="ThingBase_Fed")
    assert "/bluetooth_teleop/joy" in topics
    assert "/ridgeback/cmd_vel" in topics
    assert "/vicon/ThingBase_Fed/ThingBase_Fed" in topics
    assert "/vicon/ThingContainer/ThingContainer" in topics


def test_curated_topics_replace_default_plan_topic():
    topics = build_curated_bag_topics(
        "ThingContainer",
        plan_topic="/robot/controller/mpc_plan",
    )
    assert "/robot/controller/mpc_plan" in topics
    assert "/mpc_plan" not in topics


def test_trial_bag_recorder_start_curated(tmp_path):
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return FakeProc()

    rec = TrialBagRecorder(
        session_root=tmp_path,
        record_bag=True,
        bag_all=False,
        tool_vicon_name="ThingContainer",
        popen_fn=fake_popen,
    )
    rec.start()
    assert rec.bag_path == tmp_path / "bag" / "trial.bag"
    assert (tmp_path / "bag").is_dir()
    cmd, kwargs = calls[0]
    assert cmd[0:4] == ["rosbag", "record", "-O", str(rec.bag_path)]
    assert "-a" not in cmd
    assert "/bluetooth_teleop/joy" in cmd
    rec.stop()


def test_trial_bag_recorder_uses_remapped_plan_topic(tmp_path):
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append(cmd)
        return FakeProc()

    rec = TrialBagRecorder(
        session_root=tmp_path,
        record_bag=True,
        bag_all=False,
        tool_vicon_name="ThingContainer",
        plan_topic="/robot/controller/mpc_plan",
        popen_fn=fake_popen,
    )
    rec.start()

    assert "/robot/controller/mpc_plan" in calls[0]
    assert "/mpc_plan" not in calls[0]


def test_trial_bag_recorder_stop_sends_sigint(tmp_path):
    proc = FakeProc()

    def fake_popen(cmd, **kwargs):
        return proc

    rec = TrialBagRecorder(
        session_root=tmp_path,
        record_bag=True,
        bag_all=True,
        tool_vicon_name="ThingContainer",
        popen_fn=fake_popen,
    )
    rec.start()
    assert rec.bag_path.name == "trial_all.bag"
    rec.stop()
    assert proc.sent == [signal.SIGINT]
    assert proc.waited is True


def test_trial_bag_recorder_disabled_is_noop(tmp_path):
    def boom(*a, **k):
        raise AssertionError("should not start")

    rec = TrialBagRecorder(
        session_root=tmp_path,
        record_bag=False,
        bag_all=False,
        tool_vicon_name="ThingContainer",
        popen_fn=boom,
    )
    rec.start()
    rec.stop()
    assert rec.bag_path is None


def test_pad_joy_buttons():
    np.testing.assert_array_equal(
        pad_joy_buttons([1, 0, 1]), np.array([1, 0, 1] + [0] * 13)
    )


def test_commanded_ee_twist():
    J = np.eye(6, 9)
    u = np.arange(9, dtype=float)
    np.testing.assert_allclose(commanded_ee_twist(J, u), u[:6])


def test_append_teleop_sample_shapes(tmp_path):
    cfg = {"logging": {"log_dir": str(tmp_path)}}
    logger = DataLogger(cfg, name="control")
    append_teleop_sample(
        logger,
        ts=1.0,
        q=np.zeros(9),
        v=np.zeros(9),
        base_pose=np.zeros(3),
        base_vel=np.zeros(3),
        ee_pose=np.zeros(6),
        ee_vel=np.zeros(6),
        joy_axes=np.zeros(6),
        joy_buttons=[0, 1],
        teleop_mode="base",
        teleop_enabled=True,
        desired_base_vel=np.array([0.1, 0, 0]),
        desired_ee_vel=np.zeros(6),
        u_cmd=np.zeros(9),
        cycle_period=0.05,
    )
    assert len(logger.data["ts"]) == 1
    assert logger.data["joy_buttons"][0].shape == (16,)
    assert logger.data["teleop_mode"][0] == 0


def test_trial_bag_recorder_popen_failure_sets_error(tmp_path):
    def fail(*a, **k):
        raise OSError("no rosbag")

    rec = TrialBagRecorder(
        session_root=tmp_path,
        record_bag=True,
        bag_all=False,
        tool_vicon_name="ThingContainer",
        popen_fn=fail,
    )
    rec.start()
    assert rec.last_error is not None
    assert rec.bag_path is None
