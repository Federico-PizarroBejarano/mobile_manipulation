"""Receding-horizon execution of ACT action chunks."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Callable, Optional

import numpy as np


class ActChunkRunner:
    def __init__(
        self, predict_chunk: Callable[[np.ndarray], np.ndarray], *, n_action_steps: int
    ):
        if n_action_steps < 1:
            raise ValueError("n_action_steps must be >= 1")
        self.predict_chunk = predict_chunk
        self.n_action_steps = int(n_action_steps)
        self._chunk = None
        self._idx = 0

    def reset(self) -> None:
        self._chunk = None
        self._idx = 0

    def step(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
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
            self._chunk = chunk
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

    def predict_chunk(self, q: np.ndarray) -> np.ndarray:
        if self._proc.poll() is not None:
            raise RuntimeError(f"ACT worker died (code={self._proc.returncode})")
        assert self._proc.stdin is not None
        payload = {
            "cmd": "predict",
            "q": np.asarray(q, dtype=float).reshape(-1).tolist(),
        }
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
