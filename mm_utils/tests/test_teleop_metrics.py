import numpy as np

from mm_utils.metrics import MPSFMetricsCollector, compute_corrections


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
