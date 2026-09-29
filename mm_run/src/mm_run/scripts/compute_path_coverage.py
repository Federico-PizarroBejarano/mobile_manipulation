"""Compute wipe path coverage from a teleop trial control/data.npz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from mm_utils import parsing
from mm_utils.path_coverage import (
    compute_paths_coverage,
    load_ee_xyz_from_control_npz,
    parse_path_coverage_config,
    write_path_coverage_metrics,
)


def _resolve_session_and_npz(
    session: Path | None, npz: Path | None
) -> tuple[Path, Path]:
    if npz is not None:
        npz = Path(npz)
        session = npz.parent.parent if session is None else Path(session)
        return session, npz
    if session is None:
        raise SystemExit("Provide --session and/or --npz")
    session = Path(session)
    return session, session / "control" / "data.npz"


def _load_path_coverage_cfg(path: Path) -> dict[str, Any]:
    try:
        config = parsing.load_config(str(path))
    except Exception:
        import yaml

        config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping, got {type(config).__name__}")
    pcfg = config.get("logging", {}).get("path_coverage")
    if not isinstance(pcfg, dict):
        raise ValueError("Config has no logging.path_coverage block")
    return dict(pcfg)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, help="Trial session directory")
    p.add_argument("--npz", type=Path, help="Path to control/data.npz")
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment YAML containing logging.path_coverage",
    )
    p.add_argument("--epsilon", type=float, default=None, help="Override epsilon (m)")
    p.add_argument(
        "--sample-spacing", type=float, default=None, help="Override sample spacing (m)"
    )
    p.add_argument(
        "--write-metrics",
        action="store_true",
        help="Merge path_coverage into session metrics.npz + summary.txt",
    )
    args = p.parse_args(argv)

    session, control_npz = _resolve_session_and_npz(args.session, args.npz)
    if not control_npz.is_file():
        print(f"Missing control log: {control_npz}", file=sys.stderr)
        return 1

    try:
        pcfg = _load_path_coverage_cfg(args.config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.epsilon is not None:
        pcfg["epsilon"] = args.epsilon
    if args.sample_spacing is not None:
        pcfg["sample_spacing"] = args.sample_spacing

    try:
        paths, epsilon, sample_spacing = parse_path_coverage_config(pcfg)
        ee_xyz = load_ee_xyz_from_control_npz(control_npz)
    except (ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    coverage, per_path = compute_paths_coverage(ee_xyz, paths, epsilon, sample_spacing)
    print(f"Path coverage: {100.0 * coverage:.1f}%")
    for name, cov in per_path.items():
        print(f"Path coverage ({name}): {100.0 * cov:.1f}%")

    if args.write_metrics:
        write_path_coverage_metrics(session / "metrics", coverage, per_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
