"""Polyline path-coverage scoring from logged EE positions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PathSpec = Tuple[str, np.ndarray]  # (name, waypoints Mx3)


def sample_polyline(waypoints: np.ndarray, sample_spacing: float) -> np.ndarray:
    """Sample a polyline uniformly by arc length.

    Includes the start of the path and always includes the final endpoint.
    Returns shape ``(0, 3)`` if total length is 0.
    """
    waypoints = np.asarray(waypoints, dtype=float).reshape(-1, 3)
    if waypoints.shape[0] < 2 or sample_spacing <= 0:
        return np.zeros((0, 3), dtype=float)

    diffs = np.diff(waypoints, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total = float(np.sum(seg_lengths))
    if total <= 0.0:
        return np.zeros((0, 3), dtype=float)

    # Cumulative length at each vertex
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    # Arc-length query points: 0, ds, 2ds, ... and always total
    n_steps = int(np.floor(total / sample_spacing))
    arcs = [i * sample_spacing for i in range(n_steps + 1)]
    if arcs[-1] < total - 1e-12:
        arcs.append(total)
    arcs = np.asarray(arcs, dtype=float)

    samples = np.zeros((len(arcs), 3), dtype=float)
    for i, s in enumerate(arcs):
        # Find segment containing arc length s
        idx = int(np.searchsorted(cum, s, side="right") - 1)
        idx = min(max(idx, 0), len(seg_lengths) - 1)
        if seg_lengths[idx] <= 0.0:
            samples[i] = waypoints[idx]
            continue
        t = (s - cum[idx]) / seg_lengths[idx]
        samples[i] = waypoints[idx] + t * (waypoints[idx + 1] - waypoints[idx])
    return samples


def polyline_arc_length(waypoints: np.ndarray) -> float:
    """Total arc length of a polyline (0 if degenerate)."""
    waypoints = np.asarray(waypoints, dtype=float).reshape(-1, 3)
    if waypoints.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))


def compute_polyline_coverage(
    ee_xyz: np.ndarray,
    waypoints: np.ndarray,
    epsilon: float,
    sample_spacing: float,
) -> float:
    """Fraction of arc-length samples within ``epsilon`` (3D) of any EE pose."""
    ee_xyz = np.asarray(ee_xyz, dtype=float).reshape(-1, 3)
    if ee_xyz.shape[0] == 0 or epsilon < 0:
        return 0.0
    samples = sample_polyline(waypoints, sample_spacing)
    if samples.shape[0] == 0:
        return 0.0

    # For each sample, min distance to any EE point
    # samples (N,1,3) - ee (1,M,3) → (N,M)
    d = np.linalg.norm(samples[:, None, :] - ee_xyz[None, :, :], axis=2)
    covered = np.min(d, axis=1) <= float(epsilon)
    return float(np.count_nonzero(covered) / covered.size)


def compute_paths_coverage(
    ee_xyz: np.ndarray,
    paths: Sequence[PathSpec],
    epsilon: float,
    sample_spacing: float,
) -> Tuple[float, Dict[str, float]]:
    """Arc-length-weighted coverage over independent polylines.

    Returns
    -------
    overall : float
        ``sum(length_i * coverage_i) / sum(length_i)``, or 0 if total length is 0.
    per_path : dict
        Coverage in ``[0, 1]`` keyed by path name.
    """
    ee_xyz = np.asarray(ee_xyz, dtype=float).reshape(-1, 3)
    per_path: Dict[str, float] = {}
    total_len = 0.0
    weighted = 0.0
    for name, waypoints in paths:
        cov = compute_polyline_coverage(ee_xyz, waypoints, epsilon, sample_spacing)
        per_path[str(name)] = cov
        length = polyline_arc_length(waypoints)
        if length > 0.0:
            weighted += length * cov
            total_len += length
    overall = float(weighted / total_len) if total_len > 0.0 else 0.0
    return overall, per_path


_PATH_COVERAGE_LINE_PREFIX = "Path coverage"


def _sanitize_metric_key(name: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
    key = key.strip("_") or "path"
    return key


def _parse_waypoints(raw: Any, label: str) -> np.ndarray:
    waypoints = np.asarray(raw, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[0] < 2 or waypoints.shape[1] != 3:
        raise ValueError(f"{label} must be an array of shape (M, 3) with M >= 2")
    return waypoints


def parse_path_coverage_config(
    cfg: Dict[str, Any],
) -> Tuple[List[PathSpec], float, float]:
    """Parse path_coverage config into paths, epsilon, sample_spacing.

    Accepts either:
    - ``paths: [{name, waypoints}, ...]`` (preferred; table-to-table hops omitted), or
    - legacy ``waypoints: [...]`` (treated as a single path named ``path``).
    """
    if not isinstance(cfg, dict):
        raise ValueError("path_coverage config must be a dict")

    epsilon = float(cfg.get("epsilon"))
    sample_spacing = float(cfg.get("sample_spacing"))
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not np.isfinite(sample_spacing) or sample_spacing <= 0:
        raise ValueError("sample_spacing must be > 0")

    paths: List[PathSpec] = []
    raw_paths = cfg.get("paths")
    if raw_paths is not None:
        if not isinstance(raw_paths, (list, tuple)) or len(raw_paths) < 1:
            raise ValueError("paths must be a non-empty list")
        for i, entry in enumerate(raw_paths):
            if not isinstance(entry, dict):
                raise ValueError(f"paths[{i}] must be a dict with waypoints")
            name = str(entry.get("name") or f"path{i}")
            wps = _parse_waypoints(entry.get("waypoints"), f"paths[{i}].waypoints")
            paths.append((name, wps))
    elif "waypoints" in cfg:
        wps = _parse_waypoints(cfg.get("waypoints"), "waypoints")
        paths.append(("path", wps))
    else:
        raise ValueError("path_coverage requires 'paths' or legacy 'waypoints'")

    return paths, epsilon, sample_spacing


def load_ee_xyz_from_control_npz(control_npz: Path) -> np.ndarray:
    control_npz = Path(control_npz)
    with np.load(control_npz, allow_pickle=True) as data:
        if "ee_pose" not in data.files:
            raise KeyError(f"ee_pose missing in {control_npz}")
        ee_pose = np.asarray(data["ee_pose"], dtype=float)
    if ee_pose.ndim == 1:
        ee_pose = ee_pose.reshape(1, -1)
    if ee_pose.shape[1] < 3:
        raise ValueError(f"ee_pose must have at least 3 columns, got {ee_pose.shape}")
    return ee_pose[:, :3]


def format_path_coverage_summary_line(coverage: float) -> str:
    return f"{_PATH_COVERAGE_LINE_PREFIX}: {100.0 * float(coverage):.1f}%"


def format_path_coverage_summary_lines(
    coverage: float,
    per_path: Optional[Dict[str, float]] = None,
) -> List[str]:
    lines = [format_path_coverage_summary_line(coverage)]
    if per_path:
        for name, cov in per_path.items():
            lines.append(
                f"{_PATH_COVERAGE_LINE_PREFIX} ({name}): {100.0 * float(cov):.1f}%"
            )
    return lines


def upsert_path_coverage_summary(
    summary_text: str,
    coverage: float,
    per_path: Optional[Dict[str, float]] = None,
) -> str:
    lines = [
        ln
        for ln in summary_text.splitlines()
        if not ln.startswith(_PATH_COVERAGE_LINE_PREFIX)
    ]
    lines.extend(format_path_coverage_summary_lines(coverage, per_path))
    return "\n".join(lines) + (
        "\n" if summary_text.endswith("\n") or not summary_text else "\n"
    )


def write_path_coverage_metrics(
    metrics_dir: Path,
    coverage: float,
    per_path: Optional[Dict[str, float]] = None,
) -> None:
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    npz_path = metrics_dir / "metrics.npz"
    payload: Dict[str, Any] = {}
    if npz_path.is_file():
        with np.load(npz_path, allow_pickle=True) as data:
            payload.update({k: data[k] for k in data.files})
    # Drop stale per-path keys from a previous config shape.
    for key in list(payload.keys()):
        if key.startswith("path_coverage_"):
            del payload[key]
    payload["path_coverage"] = float(coverage)
    if per_path:
        for name, cov in per_path.items():
            payload[f"path_coverage_{_sanitize_metric_key(name)}"] = float(cov)
    np.savez_compressed(npz_path, **payload)

    summary_path = metrics_dir / "summary.txt"
    prev = summary_path.read_text() if summary_path.is_file() else ""
    summary_path.write_text(upsert_path_coverage_summary(prev, coverage, per_path))


def apply_path_coverage_to_metrics(
    session_root: Path,
    path_coverage_cfg: Dict[str, Any],
) -> Optional[float]:
    """Load control EE poses, score coverage, merge into metrics. Skip if no npz."""
    session_root = Path(session_root)
    control_npz = session_root / "control" / "data.npz"
    if not control_npz.is_file():
        return None
    try:
        ee_xyz = load_ee_xyz_from_control_npz(control_npz)
    except KeyError:
        return None
    if ee_xyz.shape[0] == 0:
        return None
    paths, epsilon, sample_spacing = parse_path_coverage_config(path_coverage_cfg)
    coverage, per_path = compute_paths_coverage(ee_xyz, paths, epsilon, sample_spacing)
    write_path_coverage_metrics(session_root / "metrics", coverage, per_path)
    return coverage


def print_trial_metrics_summary(session_root: Path) -> bool:
    """Print ``metrics/summary.txt`` (includes path coverage if already applied).

    Returns True if the file existed and was printed.
    """
    summary_path = Path(session_root) / "metrics" / "summary.txt"
    if not summary_path.is_file():
        return False
    text = summary_path.read_text()
    print(text, end="" if text.endswith("\n") else "\n")
    return True
