# mm_utils/tests/test_il_episode.py

import numpy as np
import pytest

from mm_utils.il_episode import enabled_mask, extract_episode, filter_enabled, infer_fps


def test_enabled_mask_drops_zeros():
    teleop_enabled = np.array([0.0, 1.0, 1.0, 0.0])
    assert enabled_mask(teleop_enabled).tolist() == [False, True, True, False]


def test_filter_enabled_keeps_aligned_rows():
    data = {
        "q": np.arange(12, dtype=float).reshape(4, 3),
        "u_cmd": np.arange(8, dtype=float).reshape(4, 2),
        "teleop_enabled": np.array([0.0, 1.0, 1.0, 0.0]),
    }
    out = filter_enabled(data, ("q", "u_cmd"))
    assert out["q"].shape == (2, 3)
    assert out["u_cmd"].shape == (2, 2)
    np.testing.assert_array_equal(out["q"][0], data["q"][1])


def test_infer_fps_from_median_cycle_period():
    cycle_period = np.array([0.02, 0.021, 0.019, 0.02])
    assert infer_fps(cycle_period, None) == 50


def test_extract_episode_writes_float32_and_rejects_short(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    n = 10
    np.savez_compressed(
        control / "data.npz",
        q=np.zeros((n, 9), dtype=float),
        u_cmd=np.ones((n, 9), dtype=float),
        teleop_enabled=np.array([0.0] + [1.0] * 5 + [0.0] * 4),
        cycle_period=np.full(n, 0.02),
    )
    with pytest.raises(ValueError, match="too short"):
        extract_episode(control / "data.npz", min_enabled_seconds=0.5)

    np.savez_compressed(
        control / "data.npz",
        q=np.random.randn(40, 9),
        u_cmd=np.random.randn(40, 9),
        teleop_enabled=np.array([0.0] * 5 + [1.0] * 30 + [0.0] * 5),
        cycle_period=np.full(40, 0.02),
    )
    ep = extract_episode(control / "data.npz", min_enabled_seconds=0.5)
    assert ep["q"].shape == (30, 9)
    assert ep["u_cmd"].dtype == np.float32
    assert ep["fps"] == 50
    assert ep["n_frames"] == 30
