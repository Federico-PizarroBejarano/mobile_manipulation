"""Receding-horizon execution of ACT action chunks."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Callable, Optional

import numpy as np


class ActTemporalEnsembler:
    """Paper / LeRobot temporal ensembling (Algorithm 2 of ACT).

    ``w_i = exp(-coeff * i)`` with ``i=0`` the oldest prediction in the online
    average. ``coeff=0.01`` is the original ACT default (favors older). Negative
    coeff favors newer. Requires one ``predict_chunk`` per control tick.
    """

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int):
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.chunk_size = int(chunk_size)
        self.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
        i = np.arange(self.chunk_size, dtype=float)
        self.ensemble_weights = np.exp(-self.temporal_ensemble_coeff * i)
        self.ensemble_weights_cumsum = np.cumsum(self.ensemble_weights)
        self.reset()

    def reset(self) -> None:
        self.ensembled_actions = None  # (remaining, nu)
        self.ensembled_actions_count = None  # (remaining,)

    def update(self, actions: np.ndarray) -> np.ndarray:
        """Ingest a new ``(chunk_size, nu)`` chunk; return the next action."""
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[0] != self.chunk_size:
            raise ValueError(
                f"actions must be ({self.chunk_size}, nu), got {actions.shape}"
            )
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.copy()
            self.ensembled_actions_count = np.ones(self.chunk_size, dtype=int)
        else:
            c = self.ensembled_actions_count
            w_old = self.ensemble_weights_cumsum[c - 1]
            w_new = self.ensemble_weights[c]
            w_sum = self.ensemble_weights_cumsum[c]
            self.ensembled_actions = (
                self.ensembled_actions * w_old[:, None] + actions[:-1] * w_new[:, None]
            ) / w_sum[:, None]
            self.ensembled_actions_count = np.minimum(c + 1, self.chunk_size)
            self.ensembled_actions = np.concatenate(
                [self.ensembled_actions, actions[-1:]], axis=0
            )
            self.ensembled_actions_count = np.concatenate(
                [self.ensembled_actions_count, np.array([1], dtype=int)]
            )
        action = self.ensembled_actions[0].copy()
        self.ensembled_actions = self.ensembled_actions[1:]
        self.ensembled_actions_count = self.ensembled_actions_count[1:]
        return action


class ActChunkRunner:
    """Open-loop ACT chunk execution with optional boundary blend on refresh.

    ``boundary_blend`` in ``[0, 1]`` is the peak weight on the *previous* chunk's
    leftover at refresh. ``0`` is a hard switch. Still one ``predict_chunk`` per
    refresh.

    Modes:
      - ``constant``: weight ``w`` on old for all ``n_action_steps``
      - ``decay``: weight ``w * (1 - i/n_action_steps)`` (stronger at the seam)
      - ``head``: weight ``w`` only on the first ``boundary_blend_head`` steps

    If ``temporal_ensemble_coeff`` is set (paper-style), forces every-tick infer
    (``n_action_steps`` must be 1) and ignores boundary blend.
    """

    def __init__(
        self,
        predict_chunk: Callable[[np.ndarray], np.ndarray],
        *,
        n_action_steps: int,
        boundary_blend: float = 0.0,
        boundary_blend_mode: str = "constant",
        boundary_blend_head: int = 3,
        temporal_ensemble_coeff: Optional[float] = None,
    ):
        if n_action_steps < 1:
            raise ValueError("n_action_steps must be >= 1")
        w = float(boundary_blend)
        if not (0.0 <= w <= 1.0):
            raise ValueError(f"boundary_blend must be in [0, 1], got {w}")
        mode = str(boundary_blend_mode)
        if mode not in ("constant", "decay", "head"):
            raise ValueError(
                f"boundary_blend_mode must be constant|decay|head, got {mode!r}"
            )
        head = int(boundary_blend_head)
        if head < 1:
            raise ValueError(f"boundary_blend_head must be >= 1, got {head}")
        te = None if temporal_ensemble_coeff is None else float(temporal_ensemble_coeff)
        if te is not None:
            if int(n_action_steps) != 1:
                raise ValueError(
                    "temporal_ensemble_coeff requires n_action_steps=1 "
                    f"(got {n_action_steps})"
                )
            if w > 0.0:
                raise ValueError(
                    "temporal_ensemble_coeff cannot combine with boundary_blend>0"
                )
        self.predict_chunk = predict_chunk
        self.n_action_steps = int(n_action_steps)
        self.boundary_blend = w
        self.boundary_blend_mode = mode
        self.boundary_blend_head = head
        self.temporal_ensemble_coeff = te
        self._ensembler: Optional[ActTemporalEnsembler] = None
        self._chunk = None
        self._raw_chunk = None
        self._idx = 0

    def reset(self) -> None:
        self._chunk = None
        self._raw_chunk = None
        self._idx = 0
        if self._ensembler is not None:
            self._ensembler.reset()

    def _weight_at(self, i: int, n: int) -> float:
        w = self.boundary_blend
        if self.boundary_blend_mode == "constant":
            return w
        if self.boundary_blend_mode == "decay":
            if n <= 1:
                return w
            return w * (1.0 - float(i) / float(n))
        # head
        return w if i < self.boundary_blend_head else 0.0

    def _blend_with_leftover(self, new_chunk: np.ndarray) -> np.ndarray:
        """Blend new[i] with previous raw[n_action_steps + i] (same absolute time)."""
        if self._raw_chunk is None or self.boundary_blend <= 0.0:
            return new_chunk
        prev = self._raw_chunk
        out = np.asarray(new_chunk, dtype=float).copy()
        n = min(int(self.n_action_steps), int(out.shape[0]))
        for i in range(n):
            wi = self._weight_at(i, n)
            if wi <= 0.0:
                continue
            j = int(self.n_action_steps) + i
            if j >= prev.shape[0]:
                break
            out[i] = (1.0 - wi) * out[i] + wi * prev[j]
        return out

    def step(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if self.temporal_ensemble_coeff is not None:
            chunk = np.asarray(self.predict_chunk(q), dtype=float)
            if chunk.ndim != 2:
                raise ValueError(
                    f"predict_chunk must return (H, nu), got {chunk.shape}"
                )
            if self._ensembler is None:
                self._ensembler = ActTemporalEnsembler(
                    self.temporal_ensemble_coeff, chunk.shape[0]
                )
            elif self._ensembler.chunk_size != chunk.shape[0]:
                raise ValueError(
                    f"chunk length changed {self._ensembler.chunk_size} → "
                    f"{chunk.shape[0]}"
                )
            return self._ensembler.update(chunk)

        if self._chunk is None or self._idx >= self.n_action_steps:
            chunk = np.asarray(self.predict_chunk(q), dtype=float)
            if chunk.ndim != 2:
                raise ValueError(
                    f"predict_chunk must return (H, nu), got {chunk.shape}"
                )
            if chunk.shape[0] < self.n_action_steps:
                raise ValueError(
                    f"chunk length {chunk.shape[0]} < n_action_steps {self.n_action_steps}"
                )
            blended = self._blend_with_leftover(chunk)
            self._raw_chunk = chunk
            self._chunk = blended
            self._idx = 0
        action = self._chunk[self._idx].copy()
        self._idx += 1
        return action


def default_mm_il_python() -> str:
    """Interpreter for ACT train/infer (conda env ``mm-il``, torch+cu118)."""
    return os.environ.get(
        "MM_IL_PYTHON",
        os.path.expanduser("~/miniconda3/envs/mm-il/bin/python"),
    )


class ActPolicySubprocess:
    """JSONL client to ``python -m mm_utils.act_policy_worker`` under ``mm-il``."""

    def __init__(
        self,
        checkpoint: str,
        *,
        nu: int,
        python_executable: Optional[str] = None,
    ):
        self.nu = int(nu)
        self.checkpoint = checkpoint
        py = python_executable or default_mm_il_python()
        if not os.path.isfile(py):
            raise FileNotFoundError(f"mm-il python not found at {py}; set MM_IL_PYTHON")
        self._proc = subprocess.Popen(
            [
                py,
                "-u",
                "-m",
                "mm_utils.act_policy_worker",
                "--checkpoint",
                checkpoint,
                "--nu",
                str(self.nu),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        ready = self._readline(timeout_label="worker startup")
        if not ready.get("ok") or not ready.get("ready"):
            self.close()
            raise RuntimeError(
                f"ACT worker failed to start: {ready.get('error', ready)}"
            )

    def _readline(self, *, timeout_label: str) -> dict:
        """Read the next JSON object from the worker, skipping banner lines."""
        if self._proc.stdout is None:
            raise RuntimeError("ACT worker has no stdout")
        while True:
            line = self._proc.stdout.readline()
            if not line:
                err = ""
                if self._proc.stderr is not None:
                    err = self._proc.stderr.read()
                raise RuntimeError(
                    f"ACT worker exited during {timeout_label} "
                    f"(code={self._proc.poll()}): {err.strip()}"
                )
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            return json.loads(line)

    def predict_chunk(
        self, q: np.ndarray, env_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self._proc.poll() is not None:
            raise RuntimeError(f"ACT worker died (code={self._proc.returncode})")
        assert self._proc.stdin is not None
        payload = {
            "cmd": "predict",
            "q": np.asarray(q, dtype=float).reshape(-1).tolist(),
        }
        if env_state is not None:
            payload["env_state"] = (
                np.asarray(env_state, dtype=float).reshape(-1).tolist()
            )
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        resp = self._readline(timeout_label="predict")
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "predict failed"))
        chunk = np.asarray(resp["chunk"], dtype=float)
        if chunk.ndim != 2 or chunk.shape[1] != self.nu:
            raise ValueError(f"ACT chunk must be (H, {self.nu}), got {chunk.shape}")
        return chunk

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self._proc.stdin.flush()
        except BrokenPipeError:
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    def __del__(self):  # noqa: D105
        try:
            self.close()
        except Exception:
            pass
