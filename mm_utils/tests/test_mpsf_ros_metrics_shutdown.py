"""MPSF ROS metrics persistence on shutdown (Task 6 follow-up)."""

import ast
from pathlib import Path
from types import SimpleNamespace

from mm_utils.teleop_session_logging import session_root

MPSF_ROS = Path(__file__).resolve().parents[2] / "mm_run/src/mm_run/nodes/mpsf_ros.py"
MPC_ROS = MPSF_ROS.with_name("mpc_ros.py")
DIRECT_TELEOP_ROS = MPSF_ROS.with_name("direct_teleop_ros.py")


def test_mpsf_ros_shutdown_metrics_wiring():
    source = MPSF_ROS.read_text()
    assert "def shutdownhook(self):" in source
    assert "super().shutdownhook()" in source
    assert "self._save_metrics()" in source
    assert "node._save_metrics()" in source
    assert "_metrics_saved" in source
    assert "session_root(self.logger.base_directory, self.session_timestamp)" in source
    assert '/ "metrics"' in source or '/"metrics"' in source
    assert "node._save_metrics()" in source


def test_metrics_runtime_wiring_uses_comparable_dt_and_disabled_intent():
    mpsf_source = MPSF_ROS.read_text()
    direct_source = DIRECT_TELEOP_ROS.read_text()
    mpc_source = MPC_ROS.read_text()

    assert "dt=1.0 / self._mpc_loop_hz" in mpsf_source
    assert "metric_desired_base" in mpsf_source
    assert "metric_desired_ee" in mpsf_source
    assert "dt=cycle_period" in direct_source
    assert 'plan_topic=rospy.resolve_name("mpc_plan")' in direct_source
    assert 'plan_topic=rospy.resolve_name("mpc_plan")' in mpc_source
    vicon_branch = mpc_source.split("if self.use_vicon_tool_data:", 1)[1].split(
        "ee_euler =", 1
    )[0]
    assert "ee_vel = np.zeros(6)" not in vicon_branch


def test_save_metrics_guard_skips_second_call(tmp_path):
    saved_paths = []

    class FakeCollector:
        def save(self, metrics_dir):
            saved_paths.append(metrics_dir)

        def print_summary(self):
            pass

    node = SimpleNamespace(
        _metrics_saved=False,
        logger=SimpleNamespace(base_directory=tmp_path, session_timestamp="unused"),
        session_timestamp="2026-09-11_12-00-00",
        metrics_collector=FakeCollector(),
    )

    module = ast.parse(MPSF_ROS.read_text())
    class_node = next(
        item
        for item in module.body
        if isinstance(item, ast.ClassDef) and item.name == "MPSFControllerROSNode"
    )
    method_node = next(
        item
        for item in class_node.body
        if isinstance(item, ast.FunctionDef) and item.name == "_save_metrics"
    )
    method_module = ast.fix_missing_locations(
        ast.Module(body=[method_node], type_ignores=[])
    )
    namespace = {
        "session_root": session_root,
        "rospy": SimpleNamespace(logerr=lambda *args: None),
    }
    exec(compile(method_module, str(MPSF_ROS), "exec"), namespace)

    namespace["_save_metrics"](node)
    namespace["_save_metrics"](node)

    assert len(saved_paths) == 1
    assert saved_paths[0].name == "metrics"
