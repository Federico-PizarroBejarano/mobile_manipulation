# mm_utils/tests/test_act_policy_runtime.py
import numpy as np

from mm_utils.act_policy_runtime import ActChunkRunner


def test_act_chunk_runner_consumes_then_refreshes():
    calls = []

    def predict_chunk(q):
        calls.append(np.asarray(q).copy())
        # chunk of 4 actions, each = q + offset
        base = np.asarray(q, dtype=float).reshape(9)
        return np.stack([base + i for i in range(4)], axis=0)

    runner = ActChunkRunner(predict_chunk, n_action_steps=2)
    q0 = np.zeros(9)
    a0 = runner.step(q0)
    a1 = runner.step(q0 + 1.0)  # still using first chunk; should not call predict again
    assert len(calls) == 1
    np.testing.assert_allclose(a0, np.zeros(9))
    np.testing.assert_allclose(a1, np.ones(9))
    a2 = runner.step(q0 + 2.0)  # refresh
    assert len(calls) == 2
    np.testing.assert_allclose(a2, np.full(9, 2.0))
