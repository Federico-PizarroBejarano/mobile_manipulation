"""Export per-trial user-study metrics from teleop session logs.

Only sessions under the configured user names are included. Edit
``DEFAULT_USERS`` or pass ``--users`` to change that list.

Known bad trials are listed in ``config/mpsf_experiments/user_study_corrections.csv``.
Rows with action ``override`` replace ``duration_s`` and ``base_path_length_m``
after the raw log is read. Rows with action ``flag`` are printed and left
unchanged. Each export copies that file next to the metrics CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import numpy as np

# Users included by default. Pass --users to override.
DEFAULT_USERS = ("mo", "ali", "wenda", "raghav", "sebas", "sat", "expert")

DEFAULT_TASKS = ("teleop_tray", "teleop_table_wipe")

COLUMNS = (
    "task",
    "user",
    "backend",
    "trial",
    "session",
    "duration_s",
    "path_coverage",
    "path_coverage_table1",
    "path_coverage_table2",
    "state_violation_count",
    "state_violation_steps",
    "state_violation_max",
    "mean_jerk",
    "mode_switch_count",
    "base_path_length_m",
    "collisions",
    "protective_stops",
    "singularities",
    "bottle_drops",
)

# Sheet names that do not match the results folder.
_INCIDENT_USER_NAMES = {
    "mohammed": "mo",
    "mo": "mo",
    "ali": "ali",
    "wenda": "wenda",
    "sebastian": "sebas",
    "sebas": "sebas",
    "raghav": "raghav",
    "sat": "sat",
    "satoshi": "sat",
    "expert": "expert",
}
_INCIDENT_TASKS = {"t": "tray", "w": "table_wipe"}
_INCIDENT_BACKENDS = {"m": "mpsf", "d": "direct"}
_INCIDENT_EVENT_PATTERNS = (
    ("bottle_drops", re.compile(r"fall|bottle", re.I)),
    ("protective_stops", re.compile(r"pro(?:tective)?[\s-]*stop", re.I)),
    ("singularities", re.compile(r"singu", re.I)),
    ("collisions", re.compile(r"collis|colls", re.I)),
)

_BASE_EE_MODES = (0, 1)


def default_results_dir() -> Path | None:
    candidate = Path(__file__).resolve().parents[3] / "results"
    if (candidate / "mpsf_experiments").is_dir():
        return candidate
    return None


def task_label(task_dir_name: str) -> str:
    prefix = "teleop_"
    if task_dir_name.startswith(prefix):
        return task_dir_name[len(prefix) :]
    return task_dir_name


def _scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0]
    raise ValueError(f"expected a scalar, got shape {arr.shape}")


def trial_duration_s(ts: np.ndarray) -> float:
    ts = np.asarray(ts, dtype=float).reshape(-1)
    if ts.size < 2:
        return 0.0
    return float(ts[-1] - ts[0])


def mode_switch_count(modes: np.ndarray) -> int:
    """Count base↔ee changes. The initial mode is not a switch."""
    count = 0
    prev = None
    for raw in np.asarray(modes).reshape(-1):
        mode = int(raw)
        if mode not in _BASE_EE_MODES:
            continue
        if prev is not None and mode != prev:
            count += 1
        prev = mode
    return count


def base_path_length_m(base_pose: np.ndarray) -> float:
    """Arc length of base x,y. Yaw is ignored."""
    xy = np.asarray(base_pose, dtype=float)
    if xy.ndim == 1:
        xy = xy.reshape(-1, 3)
    if xy.shape[0] < 2:
        return 0.0
    steps = np.diff(xy[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(steps, axis=1)))


def state_violations(samples) -> tuple[int, int, float] | None:
    """Return (count, n_steps, max) for logged state-bound violations."""
    samples = np.asarray(samples, dtype=object).reshape(-1)
    if samples.size == 0:
        return None
    count = 0
    max_v = -np.inf
    n = 0
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        state = sample.get("state")
        if not isinstance(state, dict):
            continue
        n += 1
        count += int(state.get("violations", 0) > 0)
        max_v = max(max_v, float(state.get("max", -np.inf)))
    if n == 0:
        return None
    return count, n, float(max_v)


def _optional_float(payload, key: str) -> float | None:
    if key not in payload.files:
        return None
    value = payload[key]
    if np.asarray(value).size == 0:
        return None
    return float(_scalar(value))


def _mean_jerk(payload) -> float | None:
    if "mean_jerk" in payload.files:
        return _optional_float(payload, "mean_jerk")
    if "jerks" not in payload.files:
        return None
    jerks = np.asarray(payload["jerks"], dtype=float).reshape(-1)
    if jerks.size == 0:
        return None
    return float(np.mean(jerks))


def _fmt(value, digits: int) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def session_row(task: str, user: str, session_dir: Path) -> dict[str, str]:
    control_path = session_dir / "control" / "data.npz"
    with np.load(control_path, allow_pickle=True) as control:
        backend = str(_scalar(control["teleop_backend"]))
        trial = int(_scalar(control["trial"]))
        duration = trial_duration_s(control["ts"])
        switches = mode_switch_count(control["teleop_mode"])
        path_length = base_path_length_m(control["base_pose"])

    coverage = None
    coverage_table1 = None
    coverage_table2 = None
    violations = None
    jerk = None
    metrics_path = session_dir / "metrics" / "metrics.npz"
    if metrics_path.is_file():
        with np.load(metrics_path, allow_pickle=True) as metrics:
            coverage = _optional_float(metrics, "path_coverage")
            coverage_table1 = _optional_float(metrics, "path_coverage_table1")
            coverage_table2 = _optional_float(metrics, "path_coverage_table2")
            if "constraint_violations" in metrics.files:
                violations = state_violations(metrics["constraint_violations"])
            jerk = _mean_jerk(metrics)
    else:
        print(f"Missing metrics: {metrics_path}", file=sys.stderr)

    if violations is None:
        vio_count, vio_steps, vio_max = "", "", ""
    else:
        vio_count = str(violations[0])
        vio_steps = str(violations[1])
        vio_max = _fmt(violations[2], 6)

    return {
        "task": task,
        "user": user,
        "backend": backend,
        "trial": str(trial),
        "session": session_dir.name,
        "duration_s": _fmt(duration, 3),
        "path_coverage": _fmt(coverage, 4),
        "path_coverage_table1": _fmt(coverage_table1, 4),
        "path_coverage_table2": _fmt(coverage_table2, 4),
        "state_violation_count": vio_count,
        "state_violation_steps": vio_steps,
        "state_violation_max": vio_max,
        "mean_jerk": _fmt(jerk, 4),
        "mode_switch_count": str(switches),
        "base_path_length_m": _fmt(path_length, 4),
    }


def iter_sessions(results_dir: Path, tasks: tuple[str, ...], users: tuple[str, ...]):
    root = Path(results_dir) / "mpsf_experiments"
    for task_dir_name in tasks:
        task_dir = root / task_dir_name
        if not task_dir.is_dir():
            print(f"Missing task directory: {task_dir}", file=sys.stderr)
            continue
        label = task_label(task_dir_name)
        for user in users:
            user_dir = task_dir / user
            if not user_dir.is_dir():
                print(
                    f"No sessions for user {user!r} in {task_dir_name}", file=sys.stderr
                )
                continue
            for control_npz in sorted(user_dir.rglob("control/data.npz")):
                yield label, user, control_npz.parent.parent


def collect_rows(results_dir: Path, tasks: tuple[str, ...], users: tuple[str, ...]):
    rows = [
        session_row(task, user, session_dir)
        for task, user, session_dir in iter_sessions(results_dir, tasks, users)
    ]
    rows.sort(
        key=lambda row: (
            row["task"],
            row["user"],
            row["backend"],
            int(row["trial"]),
            row["session"],
        )
    )
    return rows


_OVERRIDE_FIELDS = ("duration_s", "base_path_length_m")


def default_corrections_path() -> Path:
    """Tracked list of bad trials. ``results/`` is gitignored, so this is not."""
    return (
        Path(__file__).resolve().parents[3]
        / "results"
        / "mpsf_experiments"
        / "user_study_corrections.csv"
    )


def _trial_key(task, user, backend, trial, session):
    return (str(task), str(user), str(backend), str(int(trial)), str(session))


def load_corrections(path: Path):
    """Return ``(overrides, flags)`` keyed by task, user, backend, trial, session."""
    overrides = {}
    flags = []
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"action", "task", "user", "backend", "trial", "session", "note"}
        missing -= set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for line_no, raw in enumerate(reader, start=2):
            action = (raw.get("action") or "").strip()
            if not action:
                continue
            if action not in ("override", "flag"):
                raise ValueError(f"{path}:{line_no}: unknown action {action!r}")
            key = _trial_key(
                raw["task"], raw["user"], raw["backend"], raw["trial"], raw["session"]
            )
            note = (raw.get("note") or "").strip()
            if action == "override":
                values = {
                    field: (raw.get(field) or "").strip() for field in _OVERRIDE_FIELDS
                }
                if not any(values.values()):
                    raise ValueError(
                        f"{path}:{line_no}: override has no replacement values"
                    )
                overrides[key] = (values, note)
            else:
                flags.append((key, note))
    return overrides, flags


def apply_corrections(rows, overrides, flags):
    """Replace listed fields. Return unmatched override and flag keys."""
    present = {
        _trial_key(
            row["task"], row["user"], row["backend"], row["trial"], row["session"]
        )
        for row in rows
    }
    for row in rows:
        key = _trial_key(
            row["task"], row["user"], row["backend"], row["trial"], row["session"]
        )
        if key not in overrides:
            continue
        values, _note = overrides[key]
        for field, value in values.items():
            if value:
                row[field] = value
    missed_overrides = [key for key in overrides if key not in present]
    missed_flags = [key for key, _note in flags if key not in present]
    return missed_overrides, missed_flags


def _parse_incident_note(text):
    """Count the four recorded incident types in one trial note.

    Clauses that are not one of those types (for example a backwards wipe)
    are returned separately so they can be printed.
    """
    counts = {name: 0 for name, _pattern in _INCIDENT_EVENT_PATTERNS}
    ignored = []
    for clause in re.split(r"[,;]", text or ""):
        clause = clause.strip()
        if not clause:
            continue
        number = re.search(r"\d+", clause)
        count = int(number.group()) if number else 1
        matched = False
        for name, pattern in _INCIDENT_EVENT_PATTERNS:
            if pattern.search(clause):
                counts[name] += count
                matched = True
                break
        if not matched:
            ignored.append(clause)
    return counts, ignored


def _order_tokens(text, mapping):
    parts = [part.strip().lower() for part in re.split(r"\s*->\s*", text.strip())]
    if len(parts) != 2 or any(part not in mapping for part in parts):
        raise ValueError(f"Expected two codes joined by '->', got {text!r}")
    return [mapping[part] for part in parts]


def load_incidents(path: Path):
    """Map ``(user, task, backend, chronological_trial)`` to incident counts.

    Trial 1/2/3 in the sheet is the order the conditions were run, which is
    session-timestamp order. It is not always the ``trial`` field in the log
    (Mo's MPSF wipes are stored as trials 2, 3, and 4).
    """
    incidents = {}
    ignored_notes = []
    with Path(path).open(newline="") as fh:
        rows = list(csv.reader(fh))
    for raw in rows:
        if len(raw) < 4:
            continue
        name = raw[1].strip()
        task_order = raw[2].strip()
        approach_order = raw[3].strip()
        if not re.fullmatch(r"[WwTt]\s*->\s*[WwTt]", task_order):
            continue
        if not name:
            continue
        user = _INCIDENT_USER_NAMES.get(name.lower())
        if user is None:
            raise ValueError(f"Unknown user name in {path}: {name!r}")
        tasks = _order_tokens(task_order, _INCIDENT_TASKS)
        backends = _order_tokens(approach_order, _INCIDENT_BACKENDS)
        cells = raw[4:16]
        if len(cells) < 12:
            cells = cells + [""] * (12 - len(cells))
        conditions = (
            (tasks[0], backends[0]),
            (tasks[0], backends[1]),
            (tasks[1], backends[0]),
            (tasks[1], backends[1]),
        )
        for condition_index, (task, backend) in enumerate(conditions):
            for trial_index in range(3):
                note = cells[condition_index * 3 + trial_index].strip()
                counts, ignored = _parse_incident_note(note)
                key = (user, task, backend, trial_index + 1)
                if key in incidents:
                    raise ValueError(f"Duplicate incident row for {key}")
                incidents[key] = counts
                for clause in ignored:
                    ignored_notes.append((key, clause))
    return incidents, ignored_notes


def apply_incidents(rows, incidents):
    """Fill incident columns using chronological trial order within each condition.

    Users who do not appear in the notes at all (the expert) are filled with
    zeros. A user who does appear, but is missing a condition, is left blank.
    """
    noted_users = {key[0] for key in incidents}
    groups = {}
    for row in rows:
        groups.setdefault((row["user"], row["task"], row["backend"]), []).append(row)
    missing = []
    unlisted = set()
    zeros = {name: 0 for name, _pattern in _INCIDENT_EVENT_PATTERNS}
    for key, group in groups.items():
        group.sort(key=lambda row: row["session"])
        for index, row in enumerate(group, start=1):
            if key[0] not in noted_users:
                counts = zeros
                unlisted.add(key[0])
            else:
                counts = incidents.get((key[0], key[1], key[2], index))
            if counts is None:
                missing.append((*key, row["session"]))
                for name in zeros:
                    row[name] = ""
                continue
            for name, value in counts.items():
                row[name] = str(value)
    return missing, sorted(unlisted)


def write_csv(rows, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=default_results_dir(),
        help="Directory that contains mpsf_experiments/ (default: mm_run/results)",
    )
    parser.add_argument(
        "--users",
        nargs="+",
        default=list(DEFAULT_USERS),
        help=f"User folder names to include (default: {' '.join(DEFAULT_USERS)})",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help=f"Task folder names (default: {' '.join(DEFAULT_TASKS)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path (default: <results>/mpsf_experiments/user_study_metrics.csv)",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=default_corrections_path(),
        help=(
            "CSV of override and flag rows "
            "(default: config/mpsf_experiments/user_study_corrections.csv)"
        ),
    )
    args = parser.parse_args(argv)

    if args.results is None or not (Path(args.results) / "mpsf_experiments").is_dir():
        print(
            "Pass --results pointing at the mm_run results directory.", file=sys.stderr
        )
        return 1

    output = args.output
    if output is None:
        output = Path(args.results) / "mpsf_experiments" / "user_study_metrics.csv"

    corrections_path = Path(args.corrections)
    if not corrections_path.is_file():
        print(f"Missing corrections file: {corrections_path}", file=sys.stderr)
        return 1

    try:
        overrides, flags = load_corrections(corrections_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    users = tuple(args.users)
    tasks = tuple(args.tasks)
    rows = collect_rows(Path(args.results), tasks, users)
    missed_overrides, missed_flags = apply_corrections(rows, overrides, flags)
    incidents_path = output.with_name("user_study_incidents.csv")
    if not incidents_path.is_file():
        print(f"Missing incident notes: {incidents_path}", file=sys.stderr)
        return 1
    try:
        incidents, ignored_notes = load_incidents(incidents_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    missing_incidents, unlisted_users = apply_incidents(rows, incidents)
    write_csv(rows, output)

    beside = output.with_name("user_study_corrections.csv")
    if corrections_path.resolve() != beside.resolve():
        beside.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(corrections_path, beside)

    print(f"Wrote {len(rows)} trials to {output}")
    print(
        f"Applied {len(overrides) - len(missed_overrides)} corrections from {corrections_path}"
    )
    for key in missed_overrides:
        print(f"Correction did not match a trial: {key}", file=sys.stderr)
    for key in missed_flags:
        print(f"Flag did not match a trial: {key}", file=sys.stderr)
    if flags:
        print(f"Flags ({len(flags) - len(missed_flags)}):")
        for key, note in flags:
            if key in missed_flags:
                continue
            task, user, backend, trial, session = key
            print(f"  {task} {user} {backend} trial {trial} {session}: {note}")
    if unlisted_users:
        print(
            "No incident notes for "
            + ", ".join(unlisted_users)
            + "; those trials were filled with zeros."
        )
    if ignored_notes:
        print("Incident notes that are not collisions, stops, singularities, or drops:")
        for key, clause in ignored_notes:
            user, task, backend, trial = key
            print(f"  {user} {task} {backend} run {trial}: {clause}")
    if missing_incidents:
        print("No incident row for these trials (columns left blank):", file=sys.stderr)
        for item in missing_incidents:
            print(f"  {item}", file=sys.stderr)
    if missed_overrides or missed_flags or missing_incidents:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
