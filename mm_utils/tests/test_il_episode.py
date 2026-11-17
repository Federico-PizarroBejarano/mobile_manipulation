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
    assert ep["observation"].shape == (30, 9)
    assert ep["environment_state"].shape == (30, 9)
    np.testing.assert_array_equal(ep["environment_state"], ep["observation"])
    assert ep["action"].dtype == np.float32
    assert ep["action"].shape == (30, 9)
    assert ep["obs"] == "q"
    assert ep["env_state"] == "obs_duplicate"
    assert ep["action_name"] == "u_cmd"
    assert ep["fps"] == 50
    assert ep["n_frames"] == 30


def test_extract_episode_desired_ee_vel(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    n = 40
    desired = np.random.randn(n, 6)
    np.savez_compressed(
        control / "data.npz",
        q=np.random.randn(n, 9),
        desired_ee_vel=desired,
        teleop_enabled=np.array([0.0] * 5 + [1.0] * 30 + [0.0] * 5),
        cycle_period=np.full(n, 0.02),
    )
    ep = extract_episode(
        control / "data.npz",
        action="desired_ee_vel",
        min_enabled_seconds=0.5,
    )
    assert ep["action"].shape == (30, 6)
    np.testing.assert_allclose(ep["action"][0], desired[5], rtol=1e-5)

    ep_xyz = extract_episode(
        control / "data.npz",
        action="desired_ee_vel_xyz",
        min_enabled_seconds=0.5,
    )
    assert ep_xyz["action"].shape == (30, 3)
    np.testing.assert_allclose(ep_xyz["action"][0], desired[5, :3], rtol=1e-5)


def test_extract_episode_env_state_ee_pose(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    n = 40
    ee_pose = np.random.randn(n, 6)
    np.savez_compressed(
        control / "data.npz",
        q=np.random.randn(n, 9),
        ee_pose=ee_pose,
        desired_ee_vel=np.random.randn(n, 6),
        teleop_enabled=np.array([0.0] * 5 + [1.0] * 30 + [0.0] * 5),
        cycle_period=np.full(n, 0.02),
    )
    ep = extract_episode(
        control / "data.npz",
        action="desired_ee_vel",
        env_state="ee_pose",
        min_enabled_seconds=0.5,
    )
    assert ep["observation"].shape == (30, 9)
    assert ep["environment_state"].shape == (30, 6)
    assert ep["env_state"] == "ee_pose"
    np.testing.assert_allclose(ep["environment_state"][0], ee_pose[5], rtol=1e-5)


def test_extract_episode_env_state_zeros(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    n = 40
    np.savez_compressed(
        control / "data.npz",
        q=np.random.randn(n, 9),
        desired_ee_vel=np.random.randn(n, 6),
        teleop_enabled=np.array([0.0] * 5 + [1.0] * 30 + [0.0] * 5),
        cycle_period=np.full(n, 0.02),
    )
    ep = extract_episode(
        control / "data.npz",
        action="desired_ee_vel",
        env_state="zeros",
        min_enabled_seconds=0.5,
    )
    assert ep["environment_state"].shape == (30, 1)
    assert ep["env_state"] == "zeros"
    np.testing.assert_array_equal(ep["environment_state"], np.zeros((30, 1)))
