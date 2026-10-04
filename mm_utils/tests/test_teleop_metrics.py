import numpy as np

from mm_utils.metrics import (
    MPSFMetricsCollector,
    compute_corrections,
    populate_collector_from_log,
)


class Ctrl:
    mpsf_base_mask = np.ones(3)
    mpsf_ee_mask = np.ones(6)
    base_mask = np.ones(3)
    ee_mask = np.ones(6)
    log = {}
    constraints = []


def update_collector(c, u, **kwargs):
    c.update(
        references={},
        states={
            "base": {"pose": np.zeros(3), "velocity": np.zeros(3)},
            "EE": {"pose": np.zeros(6), "velocity": np.zeros(6)},
        },
        u=np.asarray(u, dtype=float),
        desired_base_vel=None,
        desired_ee_vel=None,
        controller=Ctrl(),
        robot_states=(np.zeros(9), np.zeros(9)),
        sim_timestep=0.05,
        **kwargs,
    )


def test_intent_correction_direct_ee_uses_commanded_twist():
    c = MPSFMetricsCollector()
    desired = np.array([0.1, 0, 0, 0, 0, 0], dtype=float)
    commanded = np.array([0.05, 0, 0, 0, 0, 0], dtype=float)
    u = np.zeros(9)

    c.update(
        references={},
        states={
            "base": {"pose": np.zeros(3), "velocity": np.zeros(3)},
            "EE": {"pose": np.zeros(6), "velocity": np.zeros(6)},
        },
        u=u,
        desired_base_vel=None,
        desired_ee_vel=desired,
        controller=Ctrl(),
        robot_states=(np.zeros(9), np.zeros(9)),
        sim_timestep=0.05,
        teleop_mode="ee",
        teleop_enabled=True,
        commanded_ee_twist=commanded,
        measured_ee_vel=np.zeros(6),
    )
    expected = compute_corrections(desired, commanded)
    assert c.metrics["intent_correction_ee"][-1] == expected
    assert c.metrics["ee_corrections"][-1] == expected


def test_jerk_is_aligned_and_uses_optional_dt():
    c = MPSFMetricsCollector()
    update_collector(c, np.zeros(9), dt=0.2)
    update_collector(c, np.ones(9), dt=0.2)

    assert c.metrics["jerks"] == [0.0, 15.0]
    assert len(c.metrics["jerks"]) == len(c.metrics["control_efforts"])


def test_metrics_use_singular_teleop_filter_keys():
    c = MPSFMetricsCollector()
    update_collector(c, np.zeros(9), teleop_mode="base", teleop_enabled=True)

    assert c.metrics["teleop_mode"] == [0]
    assert c.metrics["teleop_enabled"] == [1.0]
    assert "teleop_modes" not in c.metrics
    assert "teleop_enableds" not in c.metrics

    c_rl = MPSFMetricsCollector()
    update_collector(c_rl, np.zeros(9), teleop_mode="rl", teleop_enabled=True)
    assert c_rl.metrics["teleop_mode"] == [2]


def test_save_writes_npz_and_summary(tmp_path):
    c = MPSFMetricsCollector()
    c.metrics["control_efforts"].append(1.5)
    c.metrics["jerks"].append(0.2)
    c.metrics["intent_correction_base"].append(0.1)
    out = c.save(tmp_path / "metrics")
    assert (out / "metrics.npz").is_file()
    assert (out / "summary.txt").is_file()
    data = np.load(out / "metrics.npz", allow_pickle=True)
    assert float(data["mean_control_effort"]) == 1.5


def _sample_log(mode, enabled, u, desired_ee, desired_base=None, ee_vel=None):
    n = len(u)
    if desired_base is None:
        desired_base = np.zeros((n, 3))
    if ee_vel is None:
        ee_vel = np.zeros((n, 6))
    return {
        "u_cmd": np.asarray(u, dtype=float),
        "desired_base_vel": np.asarray(desired_base, dtype=float),
        "desired_ee_vel": np.asarray(desired_ee, dtype=float),
        "base_vel": np.zeros((n, 3)),
        "ee_vel": np.asarray(ee_vel, dtype=float),
        "teleop_mode": np.full(n, mode, dtype=float),
        "teleop_enabled": np.full(n, enabled, dtype=float),
        "cycle_period": np.full(n, 0.2),
        "q": np.zeros((n, 3)),
        "v": np.zeros((n, 3)),
    }


def test_offline_summary_matches_live_ee_correction_and_nominal_jerk():
    desired = np.array([0.1, 0, 0, 0, 0, 0], dtype=float)
    commanded = np.array([0.05, 0, 0, 0, 0, 0], dtype=float)
    u = np.vstack([np.zeros(9), np.ones(9)])
    log = _sample_log(
        mode=1, enabled=1.0, u=u, desired_ee=np.vstack([desired, desired])
    )
    log["q"] = np.zeros((2, 3))
    collector = MPSFMetricsCollector()
    populate_collector_from_log(
        collector,
        log,
        jerk_dt=0.2,
        commanded_ee_twists=np.vstack([commanded, commanded]),
        state_lb=-np.ones(6),
        state_ub=np.ones(6),
    )
    expected = compute_corrections(desired, commanded)
    assert collector.metrics["ee_corrections"] == [expected, expected]
    assert collector.metrics["base_corrections"] == [0.0, 0.0]
    assert collector.metrics["jerks"] == [0.0, 15.0]
    assert "self" not in collector.metrics["constraint_violations"][0]
    assert "control" not in collector.metrics["constraint_violations"][0]


def test_offline_summary_zeros_inactive_subspace_and_keeps_debug_constraints():
    u = np.zeros((1, 9))
    u[0, 3] = 0.4
    desired_ee = np.zeros((1, 6))
    desired_ee[0, 0] = 0.4
    log = _sample_log(mode=0, enabled=1.0, u=u, desired_ee=desired_ee)
    log["mpc_self_constraints"] = [np.array([-0.2, 0.02])]
    log["mpc_u_bars"] = [np.array([[0.0, 0.0], [1.2, 0.0]])]
    log["v"] = np.array([[0.0, 0.0, 2.0]])
    collector = MPSFMetricsCollector()
    populate_collector_from_log(
        collector,
        log,
        jerk_dt=0.05,
        state_lb=-np.ones(6),
        state_ub=np.ones(6),
        input_lb=np.array([-1.0, -1.0]),
        input_ub=np.array([1.0, 1.0]),
    )
    assert collector.metrics["ee_corrections"] == [0.0]
    assert collector.metrics["base_corrections"] == [0.0]
    step = collector.metrics["constraint_violations"][0]
    assert abs(step["self"]["max"] - 0.02) < 1e-9
    assert step["self"]["violations"] == 1
    assert step["control"]["violations"] == 1
    assert abs(step["control"]["max"] - 0.2) < 1e-9
    assert step["state"]["violations"] == 1


def test_offline_cycle_period_jerk_and_rl_counts_both_subspaces():
    u = np.vstack([np.zeros(9), np.ones(9)])
    desired_ee = np.tile(np.array([0.2, 0, 0, 0, 0, 0]), (2, 1))
    desired_base = np.tile(np.array([0.1, 0, 0]), (2, 1))
    log = _sample_log(
        mode=2, enabled=1.0, u=u, desired_ee=desired_ee, desired_base=desired_base
    )
    log["cycle_period"] = np.array([0.0, 0.1])
    collector = MPSFMetricsCollector()
    populate_collector_from_log(collector, log, jerk_dt=None)
    assert collector.metrics["base_corrections"][1] == compute_corrections(
        desired_base[1], u[1, :3]
    )
    assert collector.metrics["ee_corrections"][1] == compute_corrections(
        desired_ee[1], u[1, 3:9]
    )
    assert collector.metrics["jerks"][1] == 30.0
