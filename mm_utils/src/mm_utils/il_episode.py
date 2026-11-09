# mm_utils/src/mm_utils/il_episode.py
"""Extract ACT training episodes from teleop control/data.npz logs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

REQUIRED_KEYS = ("q", "u_cmd", "teleop_enabled")


def load_control_npz(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as raw:
        data = {k: raw[k] for k in raw.files}
    missing = [k for k in REQUIRED_KEYS if k not in data]
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


def extract_episode(
    control_npz: Path,
    *,
    min_enabled_seconds: float = 0.5,
    default_fps: float = 50.0,
) -> Dict[str, object]:
    data = load_control_npz(control_npz)
    filtered = filter_enabled(data, ("q", "u_cmd"))
    fps = infer_fps(
        data.get("cycle_period"),
        data.get("ts"),
        default_fps=default_fps,
    )
    n = int(filtered["q"].shape[0])
    if filtered["u_cmd"].shape[0] != n:
        raise ValueError("q/u_cmd length mismatch after filter")
    if filtered["q"].shape[1] != 9 or filtered["u_cmd"].shape[1] != 9:
        raise ValueError(
            f"expected q,u_cmd width 9, got {filtered['q'].shape}, {filtered['u_cmd'].shape}"
        )
    duration = n / float(fps)
    if duration < float(min_enabled_seconds):
        raise ValueError(
            f"episode too short: {duration:.3f}s enabled < {min_enabled_seconds}s"
        )
    return {
        "q": np.asarray(filtered["q"], dtype=np.float32),
        "u_cmd": np.asarray(filtered["u_cmd"], dtype=np.float32),
        "fps": int(fps),
        "n_frames": n,
    }


def find_control_npzs(root: Path) -> Iterable[Path]:
    """Yield ``**/control/data.npz`` under ``root`` (sorted)."""
    root = Path(root)
    return sorted(root.glob("**/control/data.npz"))
