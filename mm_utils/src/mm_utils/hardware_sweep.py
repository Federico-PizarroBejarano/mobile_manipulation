"""Cell lists and trial-folder bookkeeping for hardware ACT sweeps.

Trials land in ``<results>/<log_dir>/<logging_sub_folder>/<timestamp>/``.
A cell is done when that folder holds ``trials`` timestamp directories.
Deleting a timestamp directory is how a bad attempt is dropped; the next
launch of the sweep fills the gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

TIMESTAMP_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

TABLE_WIPE_LOG_DIR = "mpsf_experiments/teleop_table_wipe"
DECAY_SWEEP = "decay_blend"
MODEL_SWEEP = "models"

DECAY_BLENDS = (0.0, 0.3, 0.5, 0.7)
# Full 25-demo checkpoints only. Rank blends by smoothness at the chunk
# refresh (velocity jump), not by path coverage, which saturates on these runs.
DECAY_REFERENCE_RUNS = (
    ("act_il_direct_ucmd", "act"),
    ("act_il_mpc_ucmd", "act"),
    ("act_il_mpc_ee", "act_ee"),
)
DECAY_TRIALS = 3
MODEL_TRIALS = 5

_MODEL_SIZES = ("n5", "n10", "n15", "n20", "")
_MODEL_FAMILIES = (
    ("act_il_direct_ucmd", "act"),
    ("act_il_mpc_ucmd", "act"),
    ("act_il_mpc_ee", "act_ee"),
)


@dataclass(frozen=True)
class SweepCell:
    """One hardware condition: a checkpoint, a teleop backend, and a folder."""

    name: str
    teleop: str
    run_name: str
    logging_sub_folder: str
    boundary_blend: float = 0.5
    boundary_blend_mode: str = "decay"
    n_action_steps: int = 12
    temporal_ensemble_coeff: str = "none"
    trials: int = 3

    def checkpoint(self, outputs_dir: Path) -> Path:
        return (
            Path(outputs_dir)
            / self.run_name
            / "checkpoints"
            / "last"
            / "pretrained_model"
        )


def blend_cell_name(blend: float) -> str:
    return "blend_%s" % ("%g" % float(blend))


def model_run_name(prefix: str, size: str) -> str:
    return prefix if size == "" else "%s_%s" % (prefix, size)


def decay_blend_cells(
    blends: Sequence[float] = DECAY_BLENDS,
    reference_runs: Sequence[tuple] = DECAY_REFERENCE_RUNS,
    trials: int = DECAY_TRIALS,
) -> List[SweepCell]:
    """Decay-blend grid on the three full 25-demo checkpoints.

    Temporal ensemble stays off. Compare smoothness at the action-chunk
    refresh; path coverage will not separate these blends.
    """
    cells = []
    for run_name, teleop in reference_runs:
        for blend in blends:
            blend_name = blend_cell_name(blend)
            name = "%s/%s" % (run_name, blend_name)
            cells.append(
                SweepCell(
                    name=name,
                    teleop=teleop,
                    run_name=run_name,
                    logging_sub_folder="%s/%s/%s" % ("sweeps", DECAY_SWEEP, name),
                    boundary_blend=float(blend),
                    trials=int(trials),
                )
            )
    return cells


def model_cells(
    boundary_blend: float = 0.5, trials: int = MODEL_TRIALS
) -> List[SweepCell]:
    """Five sizes of direct-ucmd, mpc-ucmd, and mpc-ee. ``last`` checkpoint each."""
    cells = []
    for prefix, teleop in _MODEL_FAMILIES:
        for size in _MODEL_SIZES:
            run_name = model_run_name(prefix, size)
            cells.append(
                SweepCell(
                    name=run_name,
                    teleop=teleop,
                    run_name=run_name,
                    logging_sub_folder="%s/%s/%s" % ("sweeps", MODEL_SWEEP, run_name),
                    boundary_blend=float(boundary_blend),
                    trials=int(trials),
                )
            )
    return cells


def trial_dirs(cell_dir: Path) -> List[Path]:
    """Timestamp session folders directly inside a cell directory."""
    cell_dir = Path(cell_dir)
    if not cell_dir.is_dir():
        return []
    found = [
        p for p in cell_dir.iterdir() if p.is_dir() and TIMESTAMP_DIR_RE.match(p.name)
    ]
    return sorted(found, key=lambda p: p.name)


def new_trial_dir(cell_dir: Path, before: Iterable[Path]) -> Optional[Path]:
    """The session folder created by the launch that just exited, if any."""
    previous = {Path(p).resolve() for p in before}
    created = [p for p in trial_dirs(cell_dir) if p.resolve() not in previous]
    if not created:
        return None
    return max(created, key=lambda p: p.name)


def after_trial_action(line: str) -> Optional[str]:
    """Map an operator line to keep, collision, delete, skip, or quit.

    Empty input keeps the trial. ``c`` keeps it and marks a collision.
    ``d`` deletes it and repeats the cell. ``s`` leaves the shortfall for a
    later resume and moves to the next cell. ``q`` resets the robot and stops.
    Anything else is a re-prompt.
    """
    key = str(line).strip().lower()
    if key in ("", "k", "keep"):
        return "keep"
    if key in ("c", "collision"):
        return "collision"
    if key in ("d", "delete"):
        return "delete"
    if key in ("s", "skip"):
        return "skip"
    if key in ("q", "quit"):
        return "quit"
    return None
