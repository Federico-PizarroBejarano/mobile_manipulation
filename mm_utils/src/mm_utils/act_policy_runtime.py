"""Receding-horizon execution of ACT action chunks."""

from __future__ import annotations

from typing import Callable

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
