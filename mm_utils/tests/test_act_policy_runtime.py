# mm_utils/tests/test_act_policy_runtime.py
import numpy as np
import pytest

from mm_utils.act_policy_runtime import ActChunkRunner, ActTemporalEnsembler


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


def test_boundary_blend_mixes_leftover_on_refresh():
    # First chunk: rows 0..3 are constant 0,1,2,3 (broadcast over nu=2)
    # Second chunk: rows are 10,11,12,13
    chunks = [
        np.arange(4, dtype=float).reshape(4, 1) * np.ones((1, 2)),
        (10 + np.arange(4, dtype=float)).reshape(4, 1) * np.ones((1, 2)),
    ]
    idx = {"i": 0}

    def predict_chunk(_q):
        c = chunks[idx["i"]]
        idx["i"] += 1
        return c

    runner = ActChunkRunner(predict_chunk, n_action_steps=2, boundary_blend=0.5)
    runner.step(np.zeros(2))
    runner.step(np.zeros(2))
    # Refresh: new[0]=10 blended with old leftover old[2]=2 → 6
    a = runner.step(np.zeros(2))
    np.testing.assert_allclose(a, np.full(2, 6.0))
    # new[1]=11 blended with old[3]=3 → 7
    b = runner.step(np.zeros(2))
    np.testing.assert_allclose(b, np.full(2, 7.0))


def test_boundary_blend_rejects_bad_weight():
    with pytest.raises(ValueError):
        ActChunkRunner(lambda q: np.zeros((2, 1)), n_action_steps=1, boundary_blend=1.5)


def test_boundary_blend_decay_and_head():
    chunks = [
        np.arange(4, dtype=float).reshape(4, 1) * np.ones((1, 2)),
        (10 + np.arange(4, dtype=float)).reshape(4, 1) * np.ones((1, 2)),
    ]

    def make_predict():
        idx = {"i": 0}

        def predict_chunk(_q):
            c = chunks[idx["i"]]
            idx["i"] += 1
            return c

        return predict_chunk

    # decay: i=0 → w=0.5, blend 10 with 2 → 6; i=1 → w=0.25, blend 11 with 3 → 9
    decay = ActChunkRunner(
        make_predict(),
        n_action_steps=2,
        boundary_blend=0.5,
        boundary_blend_mode="decay",
    )
    decay.step(np.zeros(2))
    decay.step(np.zeros(2))
    np.testing.assert_allclose(decay.step(np.zeros(2)), np.full(2, 6.0))
    np.testing.assert_allclose(decay.step(np.zeros(2)), np.full(2, 9.0))

    # head=1: only first step blended
    head = ActChunkRunner(
        make_predict(),
        n_action_steps=2,
        boundary_blend=0.5,
        boundary_blend_mode="head",
        boundary_blend_head=1,
    )
    head.step(np.zeros(2))
    head.step(np.zeros(2))
    np.testing.assert_allclose(head.step(np.zeros(2)), np.full(2, 6.0))
    np.testing.assert_allclose(head.step(np.zeros(2)), np.full(2, 11.0))


def test_temporal_ensembler_matches_offline_average():
    # Two overlapping preds for the same absolute time: first chunk[1]=2,
    # second chunk[0]=8. With coeff=0, equal weights → mean 5 when second fires.
    ens = ActTemporalEnsembler(0.0, chunk_size=2)
    a0 = ens.update(np.array([[1.0, 1.0], [2.0, 2.0]]))
    np.testing.assert_allclose(a0, np.full(2, 1.0))
    a1 = ens.update(np.array([[8.0, 8.0], [9.0, 9.0]]))
    np.testing.assert_allclose(a1, np.full(2, 5.0))


def test_temporal_ensemble_runner_infers_every_step():
    calls = {"n": 0}

    def predict_chunk(_q):
        calls["n"] += 1
        v = float(calls["n"])
        return np.array([[v, v], [v + 10, v + 10]], dtype=float)

    runner = ActChunkRunner(
        predict_chunk, n_action_steps=1, temporal_ensemble_coeff=0.01
    )
    runner.step(np.zeros(2))
    runner.step(np.zeros(2))
    runner.step(np.zeros(2))
    assert calls["n"] == 3


def test_temporal_ensemble_requires_na1_and_no_boundary():
    with pytest.raises(ValueError, match="n_action_steps=1"):
        ActChunkRunner(
            lambda q: np.zeros((2, 1)),
            n_action_steps=2,
            temporal_ensemble_coeff=0.01,
        )
    with pytest.raises(ValueError, match="boundary_blend"):
        ActChunkRunner(
            lambda q: np.zeros((2, 1)),
            n_action_steps=1,
            boundary_blend=0.2,
            temporal_ensemble_coeff=0.01,
        )
