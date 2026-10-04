import numpy as np
import pytest

from mm_utils.path_coverage import (
    apply_path_coverage_to_metrics,
    compute_paths_coverage,
    compute_polyline_coverage,
    load_ee_xyz_from_control_npz,
    parse_path_coverage_config,
    polyline_arc_length,
    print_trial_metrics_summary,
    sample_polyline,
    upsert_path_coverage_summary,
)


def test_sample_polyline_includes_endpoints_and_spacing():
    waypoints = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    samples = sample_polyline(waypoints, sample_spacing=0.25)
    assert samples.shape[1] == 3
    np.testing.assert_allclose(samples[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(samples[-1], [1.0, 0.0, 0.0])
    # spacing ~0.25 along 1 m segment → 0, 0.25, 0.50, 0.75, 1.0
    assert len(samples) == 5


def test_full_coverage_when_ee_on_polyline():
    waypoints = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    ee = np.linspace([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 21)
    cov = compute_polyline_coverage(ee, waypoints, epsilon=0.01, sample_spacing=0.05)
    assert cov == pytest.approx(1.0)


def test_zero_coverage_when_ee_far():
    waypoints = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    ee = np.array([[0.0, 10.0, 0.0], [1.0, 10.0, 0.0]], dtype=float)
    cov = compute_polyline_coverage(ee, waypoints, epsilon=0.05, sample_spacing=0.1)
    assert cov == 0.0


def test_partial_coverage_first_half_only():
    waypoints = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    ee = np.linspace([0.0, 0.0, 0.0], [0.4, 0.0, 0.0], 10)
    cov = compute_polyline_coverage(ee, waypoints, epsilon=0.05, sample_spacing=0.1)
    assert 0.3 < cov < 0.7


def test_zero_length_polyline_returns_zero():
    waypoints = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=float)
    ee = np.array([[1.0, 2.0, 3.0]], dtype=float)
    assert (
        compute_polyline_coverage(ee, waypoints, epsilon=0.1, sample_spacing=0.01)
        == 0.0
    )


def test_epsilon_boundary_is_inclusive():
    # Single-segment path; EE exactly epsilon away from x=0.5 sample
    waypoints = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    samples = sample_polyline(waypoints, sample_spacing=0.5)
    # pick mid sample at 0.5
    mid = samples[len(samples) // 2]
    ee = mid + np.array([0.0, 0.05, 0.0])
    assert (
        compute_polyline_coverage(
            ee.reshape(1, 3), waypoints, epsilon=0.05, sample_spacing=0.5
        )
        > 0.0
    )
    assert (
        compute_polyline_coverage(
            ee.reshape(1, 3), waypoints, epsilon=0.049, sample_spacing=0.5
        )
        < 1.0
    )


def test_parse_legacy_waypoints_ok():
    paths, eps, ds = parse_path_coverage_config(
        {
            "waypoints": [[0, 0, 0], [1, 0, 0]],
            "epsilon": 0.03,
            "sample_spacing": 0.01,
        }
    )
    assert len(paths) == 1
    assert paths[0][0] == "path"
    assert paths[0][1].shape == (2, 3)
    assert eps == 0.03
    assert ds == 0.01


def test_parse_multi_paths_ok():
    paths, eps, ds = parse_path_coverage_config(
        {
            "epsilon": 0.1,
            "sample_spacing": 0.01,
            "paths": [
                {"name": "table1", "waypoints": [[0, 0, 0], [1, 0, 0]]},
                {"name": "table2", "waypoints": [[0, 1, 0], [1, 1, 0]]},
            ],
        }
    )
    assert eps == 0.1 and ds == 0.01
    assert [n for n, _ in paths] == ["table1", "table2"]
    assert paths[0][1].shape == (2, 3)


def test_parse_path_coverage_config_rejects_bad():
    with pytest.raises(ValueError):
        parse_path_coverage_config(
            {"waypoints": [[0, 0, 0]], "epsilon": 0.03, "sample_spacing": 0.01}
        )
    with pytest.raises(ValueError):
        parse_path_coverage_config(
            {
                "waypoints": [[0, 0, 0], [1, 0, 0]],
                "epsilon": 0.0,
                "sample_spacing": 0.01,
            }
        )
    with pytest.raises(ValueError):
        parse_path_coverage_config({"epsilon": 0.1, "sample_spacing": 0.01})


def test_multi_path_coverage_is_arc_length_weighted():
    # Path A length 1, fully covered; path B length 3, uncovered → overall 0.25
    paths = [
        ("a", np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)),
        ("b", np.array([[0.0, 1.0, 0.0], [3.0, 1.0, 0.0]], dtype=float)),
    ]
    ee = np.linspace([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 21)
    overall, per = compute_paths_coverage(ee, paths, epsilon=0.05, sample_spacing=0.1)
    assert per["a"] == pytest.approx(1.0)
    assert per["b"] == pytest.approx(0.0)
    assert overall == pytest.approx(0.25)
    assert polyline_arc_length(paths[0][1]) == pytest.approx(1.0)
    assert polyline_arc_length(paths[1][1]) == pytest.approx(3.0)


def test_multi_path_ignores_gap_between_paths():
    # Transition between tables is not a scored segment.
    paths = [
        ("t1", np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)),
        ("t2", np.array([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]], dtype=float)),
    ]
    ee_t1 = np.linspace([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 21)
    ee_t2 = np.linspace([0.0, 2.0, 0.0], [1.0, 2.0, 0.0], 21)
    ee = np.vstack([ee_t1, ee_t2])
    overall, per = compute_paths_coverage(ee, paths, epsilon=0.05, sample_spacing=0.1)
    assert per["t1"] == pytest.approx(1.0)
    assert per["t2"] == pytest.approx(1.0)
    assert overall == pytest.approx(1.0)


def test_load_ee_xyz_from_control_npz(tmp_path):
    npz = tmp_path / "data.npz"
    ee = np.array([[0.0, 0.0, 0.7, 0, 0, 0], [0.1, 0.0, 0.7, 0, 0, 0]], dtype=float)
    np.savez_compressed(npz, ee_pose=ee)
    out = load_ee_xyz_from_control_npz(npz)
    np.testing.assert_allclose(out, ee[:, :3])


def test_upsert_replaces_existing_line():
    text = (
        "EXPERIMENT\nPath coverage: 10.0%\nPath coverage (table1): 1.0%\nMean Jerk: 1\n"
    )
    out = upsert_path_coverage_summary(text, 0.873, {"table1": 1.0, "table2": 0.5})
    assert "Path coverage: 87.3%" in out
    assert "Path coverage (table1): 100.0%" in out
    assert "Path coverage (table2): 50.0%" in out
    assert out.count("Path coverage:") == 1
    assert "Path coverage (table1): 1.0%" not in out


def test_apply_path_coverage_writes_metrics(tmp_path):
    session = tmp_path / "trial"
    control = session / "control"
    metrics = session / "metrics"
    control.mkdir(parents=True)
    metrics.mkdir(parents=True)
    ee = np.linspace([0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], 50)
    np.savez_compressed(control / "data.npz", ee_pose=ee)
    (metrics / "summary.txt").write_text("EXPERIMENT METRICS SUMMARY\n")
    np.savez_compressed(metrics / "metrics.npz", mean_jerk=0.1)

    cfg = {
        "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "epsilon": 0.05,
        "sample_spacing": 0.1,
    }
    cov = apply_path_coverage_to_metrics(session, cfg)
    assert cov == pytest.approx(1.0)
    data = np.load(metrics / "metrics.npz", allow_pickle=True)
    assert float(data["path_coverage"]) == pytest.approx(1.0)
    assert float(data["path_coverage_path"]) == pytest.approx(1.0)
    assert "Path coverage: 100.0%" in (metrics / "summary.txt").read_text()


def test_apply_multi_path_writes_per_path_metrics(tmp_path):
    session = tmp_path / "trial"
    control = session / "control"
    metrics = session / "metrics"
    control.mkdir(parents=True)
    metrics.mkdir(parents=True)
    ee = np.linspace([0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], 50)
    np.savez_compressed(control / "data.npz", ee_pose=ee)
    (metrics / "summary.txt").write_text("SUMMARY\n")
    np.savez_compressed(metrics / "metrics.npz", mean_jerk=0.1)

    cfg = {
        "epsilon": 0.05,
        "sample_spacing": 0.1,
        "paths": [
            {"name": "table1", "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
            {"name": "table2", "waypoints": [[0.0, 5.0, 0.0], [1.0, 5.0, 0.0]]},
        ],
    }
    cov = apply_path_coverage_to_metrics(session, cfg)
    assert cov == pytest.approx(0.5)
    data = np.load(metrics / "metrics.npz", allow_pickle=True)
    assert float(data["path_coverage_table1"]) == pytest.approx(1.0)
    assert float(data["path_coverage_table2"]) == pytest.approx(0.0)
    text = (metrics / "summary.txt").read_text()
    assert "Path coverage (table1): 100.0%" in text
    assert "Path coverage (table2): 0.0%" in text


def test_apply_skips_when_ee_pose_empty(tmp_path):
    session = tmp_path / "trial"
    control = session / "control"
    metrics = session / "metrics"
    control.mkdir(parents=True)
    metrics.mkdir(parents=True)
    np.savez_compressed(control / "data.npz", ee_pose=np.zeros((0, 6)))
    np.savez_compressed(metrics / "metrics.npz", mean_jerk=0.1)

    cfg = {
        "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "epsilon": 0.05,
        "sample_spacing": 0.1,
    }
    assert apply_path_coverage_to_metrics(session, cfg) is None
    data = np.load(metrics / "metrics.npz", allow_pickle=True)
    assert "path_coverage" not in data.files
    assert float(data["mean_jerk"]) == pytest.approx(0.1)


def test_apply_skips_when_control_missing(tmp_path):
    session = tmp_path / "trial"
    session.mkdir()
    cfg = {
        "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "epsilon": 0.05,
        "sample_spacing": 0.1,
    }
    assert apply_path_coverage_to_metrics(session, cfg) is None


def test_print_trial_metrics_summary_includes_path_coverage(tmp_path, capsys):
    session = tmp_path / "trial"
    metrics = session / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "summary.txt").write_text(
        "EXPERIMENT METRICS SUMMARY\nPath coverage: 87.3%\nPath coverage (table1): 100.0%\n"
    )
    assert print_trial_metrics_summary(session) is True
    out = capsys.readouterr().out
    assert "EXPERIMENT METRICS SUMMARY" in out
    assert "Path coverage: 87.3%" in out
    assert "Path coverage (table1): 100.0%" in out


def test_print_trial_metrics_summary_missing_file(tmp_path):
    assert print_trial_metrics_summary(tmp_path / "missing") is False
