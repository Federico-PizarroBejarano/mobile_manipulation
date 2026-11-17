# mm_utils/tests/test_act_eval.py

import json

import pytest

from mm_utils.act_eval import (
    aggregate_eval_trials,
    infer_teleop_from_action_width,
    infer_teleop_from_checkpoint,
    parse_metrics_summary,
    resolve_pretrained_model,
    write_eval_overlay,
)


def test_infer_teleop_from_action_width():
    assert infer_teleop_from_action_width(9) == "act"
    assert infer_teleop_from_action_width(6) == "act_ee"
    assert infer_teleop_from_action_width(3) == "act_ee"
    with pytest.raises(ValueError):
        infer_teleop_from_action_width(7)


def test_infer_teleop_from_checkpoint(tmp_path):
    model = tmp_path / "pretrained_model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"output_features": {"action": {"type": "ACTION", "shape": [6]}}})
    )
    teleop, nu = infer_teleop_from_checkpoint(model)
    assert teleop == "act_ee"
    assert nu == 6


def test_resolve_pretrained_model(tmp_path):
    model = tmp_path / "checkpoints" / "last" / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    assert resolve_pretrained_model(tmp_path, "last") == model.resolve()
    with pytest.raises(ValueError):
        resolve_pretrained_model(tmp_path, "../last")


def test_parse_metrics_summary():
    text = """
EXPERIMENT METRICS SUMMARY
Constraint Violations Summary:
  state: max=-0.1, violations=0/233, per_dim_max=[-1]
Path coverage: 97.8%
Path coverage (table1): 100.0%
Path coverage (table2): 94.9%
"""
    parsed = parse_metrics_summary(text)
    assert parsed["path_coverage"] == pytest.approx(0.978)
    assert parsed["path_coverage_by_name"]["table1"] == pytest.approx(1.0)
    assert parsed["path_coverage_by_name"]["table2"] == pytest.approx(0.949)
    assert parsed["state_violations"] == 0


def test_aggregate_and_overlay(tmp_path):
    trials = [
        {"path_coverage": 0.5, "trial": 1},
        {"path_coverage": 1.0, "trial": 2},
        {"path_coverage": None, "trial": 3},
    ]
    agg = aggregate_eval_trials(trials)
    assert agg["n_trials"] == 3
    assert agg["n_with_coverage"] == 2
    assert agg["path_coverage_mean"] == pytest.approx(0.75)
    overlay = write_eval_overlay(tmp_path / "o.yaml", log_dir=tmp_path / "logs")
    text = overlay.read_text()
    assert "prompt_trial_metadata: false" in text
    assert str((tmp_path / "logs").resolve()) in text
    overlay_rate = write_eval_overlay(
        tmp_path / "o_rate.yaml", log_dir=tmp_path / "logs2", ctrl_rate=10
    )
    text_rate = overlay_rate.read_text()
    assert "ctrl_rate: 10" in text_rate
