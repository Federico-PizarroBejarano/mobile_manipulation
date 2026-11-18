from pathlib import Path

from mm_utils.hardware_sweep import (
    ENSEMBLE_COEFFS,
    ENSEMBLE_RUNS,
    after_trial_action,
    blend_cell_name,
    decay_blend_cells,
    ensemble_coeff_cells,
    model_cells,
    new_trial_dir,
    trial_dirs,
)


def test_trial_dirs_keep_only_timestamps(tmp_path):
    (tmp_path / "2026-11-16_11-18-05").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "2026-11-16_11-21-46").mkdir()
    (tmp_path / "readme.txt").write_text("x")
    names = [p.name for p in trial_dirs(tmp_path)]
    assert names == ["2026-11-16_11-18-05", "2026-11-16_11-21-46"]
    assert trial_dirs(tmp_path / "missing") == []


def test_new_trial_dir_is_the_one_just_created(tmp_path):
    old = tmp_path / "2026-11-16_11-18-05"
    old.mkdir()
    before = trial_dirs(tmp_path)
    created = tmp_path / "2026-11-16_11-21-46"
    created.mkdir()
    assert new_trial_dir(tmp_path, before) == created
    assert new_trial_dir(tmp_path, trial_dirs(tmp_path)) is None


def test_after_trial_action():
    assert after_trial_action("") == "keep"
    assert after_trial_action("c") == "collision"
    assert after_trial_action("d") == "delete"
    assert after_trial_action("S") == "skip"
    assert after_trial_action("q") == "quit"
    assert after_trial_action("nope") is None


def test_decay_cells_are_the_blend_grid():
    cells = decay_blend_cells()
    assert len(cells) == 12
    assert {c.trials for c in cells} == {3}
    by_run = {}
    for cell in cells:
        by_run.setdefault((cell.run_name, cell.teleop), []).append(cell.boundary_blend)
    assert by_run[("act_il_direct_ucmd", "act")] == [0.0, 0.3, 0.5, 0.7]
    assert by_run[("act_il_mpc_ucmd", "act")] == [0.0, 0.3, 0.5, 0.7]
    assert by_run[("act_il_mpc_ee", "act_ee")] == [0.0, 0.3, 0.5, 0.7]
    ee = [c for c in cells if c.run_name == "act_il_mpc_ee" and c.boundary_blend == 0.3]
    assert ee[0].logging_sub_folder == "sweeps/decay_blend/act_il_mpc_ee/blend_0.3"
    assert blend_cell_name(0.5) == "blend_0.5"
    ckpt = ee[0].checkpoint(Path("/outputs"))
    assert ckpt == Path("/outputs/act_il_mpc_ee/checkpoints/last/pretrained_model")


def test_model_cells_cover_fifteen_runs():
    cells = model_cells(boundary_blend=0.3)
    assert len(cells) == 15
    by_teleop = {}
    for cell in cells:
        by_teleop.setdefault(cell.teleop, []).append(cell.run_name)
        assert cell.boundary_blend == 0.3
        assert cell.n_action_steps == 12
        assert cell.trials == 5
        assert cell.logging_sub_folder == "sweeps/models/%s" % cell.run_name
    assert by_teleop["act"] == [
        "act_il_direct_ucmd_n5",
        "act_il_direct_ucmd_n10",
        "act_il_direct_ucmd_n15",
        "act_il_direct_ucmd_n20",
        "act_il_direct_ucmd",
        "act_il_mpc_ucmd_n5",
        "act_il_mpc_ucmd_n10",
        "act_il_mpc_ucmd_n15",
        "act_il_mpc_ucmd_n20",
        "act_il_mpc_ucmd",
    ]
    assert by_teleop["act_ee"] == [
        "act_il_mpc_ee_n5",
        "act_il_mpc_ee_n10",
        "act_il_mpc_ee_n15",
        "act_il_mpc_ee_n20",
        "act_il_mpc_ee",
    ]


def test_model_cells_horizon_override():
    cells = model_cells(n_action_steps=25)
    assert {cell.n_action_steps for cell in cells} == {25}


def test_ensemble_coeff_cells_force_horizon_one_and_no_blend():
    cells = ensemble_coeff_cells()
    assert len(cells) == len(ENSEMBLE_RUNS) * len(ENSEMBLE_COEFFS)
    assert {c.trials for c in cells} == {2}
    assert {c.n_action_steps for c in cells} == {1}
    assert {c.boundary_blend for c in cells} == {0.0}
    assert {c.temporal_ensemble_coeff for c in cells} == {
        "0.01",
        "0.05",
        "0.1",
        "-0.05",
    }
    assert [c.run_name for c in cells[::4]] == [run for run, _ in ENSEMBLE_RUNS]
    first = cells[0]
    assert first.run_name == "act_il_mpc_ucmd_n15"
    assert first.teleop == "act"
    assert first.temporal_ensemble_coeff == "0.01"
    assert first.logging_sub_folder == (
        "sweeps/ensemble/act_il_mpc_ucmd_n15/coeff_0.01"
    )
    ee = [
        c
        for c in cells
        if c.run_name == "act_il_mpc_ee_n10" and c.temporal_ensemble_coeff == "0.1"
    ]
    assert len(ee) == 1
    assert ee[0].teleop == "act_ee"
    assert ee[0].logging_sub_folder == "sweeps/ensemble/act_il_mpc_ee_n10/coeff_0.1"
    neg = [
        c
        for c in cells
        if c.run_name == "act_il_mpc_ucmd_n15" and c.temporal_ensemble_coeff == "-0.05"
    ]
    assert len(neg) == 1
    assert neg[0].logging_sub_folder == (
        "sweeps/ensemble/act_il_mpc_ucmd_n15/coeff_-0.05"
    )
