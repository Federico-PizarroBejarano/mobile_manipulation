# mm_utils/src/mm_utils/il_episode.py
"""Extract ACT training episodes from teleop control/data.npz logs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

# Obs / action / env_state selectors for export and train. ``key`` is the npz
# field; ``width`` is the exported vector length; optional ``cols`` slices
# columns. ``obs_duplicate`` copies observation.state into environment_state.
OBS_SPECS = {
    "q": {"key": "q", "width": 9},
}

ACTION_SPECS = {
    "u_cmd": {"key": "u_cmd", "width": 9},
    "desired_ee_vel": {"key": "desired_ee_vel", "width": 6},
    "desired_ee_vel_xyz": {"key": "desired_ee_vel", "width": 3, "cols": (0, 3)},
}

ENV_STATE_SPECS = {
    "obs_duplicate": None,
    "ee_pose": {"key": "ee_pose", "width": 6},
    # Constant zeros; width 1 minimizes the unused ENV pathway (ACT still requires it).
    "zeros": {"width": 1, "constant": 0.0},
}


def resolve_obs_spec(name: str) -> dict:
    if name not in OBS_SPECS:
        raise ValueError(f"unknown --obs {name!r}; choose from {tuple(OBS_SPECS)}")
    return OBS_SPECS[name]


def resolve_action_spec(name: str) -> dict:
    if name not in ACTION_SPECS:
        raise ValueError(
            f"unknown --action {name!r}; choose from {tuple(ACTION_SPECS)}"
        )
    return ACTION_SPECS[name]


def resolve_env_state_spec(name: str):
    if name not in ENV_STATE_SPECS:
        raise ValueError(
            f"unknown --env-state {name!r}; choose from {tuple(ENV_STATE_SPECS)}"
        )
    return ENV_STATE_SPECS[name]


def load_control_npz(
    path: Path, *, required_keys: Sequence[str]
) -> Dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as raw:
        data = {k: raw[k] for k in raw.files}
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"{path} missing keys {missing}")
    return data


def enabled_mask(teleop_enabled: np.ndarray) -> np.ndarray:
    return np.asarray(teleop_enabled, dtype=float).reshape(-1) > 0.5


def filter_enabled(
    data: Mapping[str, np.ndarray], keys: Sequence[str]
) -> Dict[str, np.ndarray]:
    mask = enabled_mask(data["teleop_enabled"])
    out = {}
    for key in keys:
        arr = np.asarray(data[key])
        if arr.shape[0] != mask.shape[0]:
            raise ValueError(
                f"{key} length {arr.shape[0]} != teleop_enabled length {mask.shape[0]}"
            )
        out[key] = arr[mask]
    return out


def infer_fps(
    cycle_period: Optional[np.ndarray],
    ts: Optional[np.ndarray],
    default_fps: float = 50.0,
) -> int:
    if cycle_period is not None:
        cp = np.asarray(cycle_period, dtype=float).reshape(-1)
        cp = cp[np.isfinite(cp) & (cp > 1e-6)]
        if cp.size:
            return max(1, int(round(1.0 / float(np.median(cp)))))
    if ts is not None:
        t = np.asarray(ts, dtype=float).reshape(-1)
        if t.size >= 2:
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 1e-6)]
            if dt.size:
                return max(1, int(round(1.0 / float(np.median(dt)))))
    return max(1, int(round(float(default_fps))))


def _take_columns(arr: np.ndarray, width: int, cols: Optional[Tuple[int, int]]):
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {arr.shape}")
    if cols is not None:
        start, end = cols
        arr = arr[:, start:end]
    if arr.shape[1] != width:
        raise ValueError(f"expected width {width}, got shape {arr.shape}")
    return arr


def extract_episode(
    control_npz: Path,
    *,
    obs: str = "q",
    action: str = "u_cmd",
    env_state: str = "obs_duplicate",
    min_enabled_seconds: float = 0.5,
    default_fps: float = 50.0,
) -> Dict[str, object]:
    obs_spec = resolve_obs_spec(obs)
    action_spec = resolve_action_spec(action)
    env_spec = resolve_env_state_spec(env_state)
    obs_key = str(obs_spec["key"])
    action_key = str(action_spec["key"])
    required = ["teleop_enabled", obs_key, action_key]
    filter_keys = [obs_key, action_key]
    if env_spec is not None and "key" in env_spec:
        env_key = str(env_spec["key"])
        required.append(env_key)
        filter_keys.append(env_key)
    data = load_control_npz(control_npz, required_keys=tuple(required))
    filtered = filter_enabled(data, tuple(dict.fromkeys(filter_keys)))
    fps = infer_fps(
        data.get("cycle_period"),
        data.get("ts"),
        default_fps=default_fps,
    )
    observation = _take_columns(
        filtered[obs_key],
        int(obs_spec["width"]),
        obs_spec.get("cols"),
    )
    action_arr = _take_columns(
        filtered[action_key],
        int(action_spec["width"]),
        action_spec.get("cols"),
    )
    n = int(observation.shape[0])
    if env_spec is None:
        environment_state = observation
    elif "constant" in env_spec:
        environment_state = np.full(
            (n, int(env_spec["width"])),
            float(env_spec["constant"]),
            dtype=np.float32,
        )
    else:
        environment_state = _take_columns(
            filtered[str(env_spec["key"])],
            int(env_spec["width"]),
            env_spec.get("cols"),
        )
    if action_arr.shape[0] != n:
        raise ValueError("observation/action length mismatch after filter")
    if environment_state.shape[0] != n:
        raise ValueError("observation/env_state length mismatch after filter")
    duration = n / float(fps)
    if duration < float(min_enabled_seconds):
        raise ValueError(
            f"episode too short: {duration:.3f}s enabled < {min_enabled_seconds}s"
        )
    return {
        "observation": np.asarray(observation, dtype=np.float32),
        "environment_state": np.asarray(environment_state, dtype=np.float32),
        "action": np.asarray(action_arr, dtype=np.float32),
        "obs": obs,
        "env_state": env_state,
        "action_name": action,
        "fps": int(fps),
        "n_frames": n,
    }


def find_control_npzs(root: Path) -> Iterable[Path]:
    """Yield ``**/control/data.npz`` under ``root`` (sorted)."""
    root = Path(root)
    return sorted(root.glob("**/control/data.npz"))
