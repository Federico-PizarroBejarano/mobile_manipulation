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

    assert "metrics_collector.update" not in mpsf_source
    assert "metrics_collector.update" not in direct_source
    assert "jerk_dt=1.0 / self._mpc_loop_hz" in mpsf_source
    assert "jerk_dt=None" in direct_source
    assert "commanded_ee_twists" in direct_source
    assert "commanded_ee_twists" in mpsf_source
    assert "base_mask=self.controller.mpsf_base_mask" not in mpsf_source
    assert "ee_mask=self.controller.mpsf_ee_mask" not in mpsf_source
    assert 'plan_topic=rospy.resolve_name("mpc_plan")' in direct_source
    assert 'plan_topic=rospy.resolve_name("mpc_plan")' in mpc_source
    vicon_branch = mpc_source.split("if self.use_vicon_tool_data:", 1)[1].split(
        "ee_euler =", 1
    )[0]
    assert "ee_vel = np.zeros(6)" not in vicon_branch


def test_save_metrics_guard_skips_second_call(tmp_path):
    saved_paths = []
    control_saves = []

    class FakeCollector:
        def save(self, metrics_dir):
            saved_paths.append(metrics_dir)

        def print_summary(self):
            pass

    class FakeLogger:
        def __init__(self):
            self.base_directory = tmp_path
            self.config = {"logging": {}}

        def save(self, session_timestamp):
            control_saves.append(session_timestamp)

    populated = []
    node = SimpleNamespace(
        _metrics_saved=False,
        logger=FakeLogger(),
        session_timestamp="2026-09-11_12-00-00",
        metrics_collector=FakeCollector(),
        _populate_metrics=lambda: populated.append(True),
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
        "rospy": SimpleNamespace(
            logerr=lambda *args: None,
            logwarn=lambda *args: None,
            loginfo=lambda *args: None,
        ),
        "getattr": getattr,
        "apply_path_coverage_to_metrics": lambda *a, **k: None,
        "print_trial_metrics_summary": lambda *a, **k: False,
    }
    exec(compile(method_module, str(MPSF_ROS), "exec"), namespace)

    namespace["_save_metrics"](node)
    namespace["_save_metrics"](node)

    assert len(saved_paths) == 1
    assert saved_paths[0].name == "metrics"
    assert control_saves == ["2026-09-11_12-00-00"]
    assert populated == [True]


def test_save_metrics_flushes_control_log_before_path_coverage():
    """Circle-end path: control logger must be saved inside _save_metrics."""
    source = MPSF_ROS.read_text()
    idx_pop = source.find("self._populate_metrics()")
    idx_log = source.find("self.logger.save(session_timestamp=self.session_timestamp)")
    idx_metrics = source.find('self.metrics_collector.save(root / "metrics")')
    idx_cov = source.find("apply_path_coverage_to_metrics(root, pcfg)")
    idx_print = source.find("print_trial_metrics_summary(root)")
    assert idx_pop != -1 and idx_log != -1 and idx_metrics != -1
    assert idx_cov != -1 and idx_print != -1
    assert idx_pop < idx_log < idx_metrics < idx_cov < idx_print


def test_mpc_run_flushes_control_log_at_trial_end():
    source = MPC_ROS.read_text()
    assert "Failed to save control log at trial end" in source
    module = ast.parse(source)
    class_node = next(
        item
        for item in module.body
        if isinstance(item, ast.ClassDef) and item.name == "ControllerROSNode"
    )
    run_node = next(
        item
        for item in class_node.body
        if isinstance(item, ast.FunctionDef) and item.name == "run"
    )
    run_src = ast.get_source_segment(source, run_node)
    assert run_src is not None
    assert "self.logger.save(session_timestamp=self.session_timestamp)" in run_src
    assert "if not self._deploy_logging:" in run_src
    assert "if self.controller.record_horizon_log:" in run_src


def test_deploy_gates_horizon_logging():
    root = Path(__file__).resolve().parents[2]
    mpc = (root / "mm_control/src/mm_control/MPC.py").read_text()
    experiment = (root / "mm_run/src/mm_run/scripts/experiment.py").read_text()
    assert "if self.record_horizon_log:" in mpc
    assert "controller.record_horizon_log" in experiment
    assert 'logging_profile(config) != "deploy"' in experiment
