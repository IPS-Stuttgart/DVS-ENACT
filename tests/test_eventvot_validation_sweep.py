import argparse
import importlib.util
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eventvot_validation_sweep.py"
SCRIPTS_DIR = SCRIPT_PATH.parent


def _load_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "run_eventvot_validation_sweep_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_validation_fixture(root: Path) -> tuple[Path, Path]:
    split_root = root / "validating Subset"
    sequence_dir = split_root / "seq1"
    sequence_dir.mkdir(parents=True)
    (split_root / "list.txt").write_text("seq1\n", encoding="utf-8")
    (sequence_dir / "groundtruth.txt").write_text(
        "1 1 10 10\n1 1 10 10\n",
        encoding="utf-8",
    )
    (sequence_dir / "absent.txt").write_text("1\n1\n", encoding="utf-8")
    result_root = root / "base_results"
    result_root.mkdir()
    (result_root / "seq1.txt").write_text(
        "1 1 10 10\n1 1 10 10\n",
        encoding="utf-8",
    )
    return split_root, result_root


def _parse_sweep_args(module, tmp_path: Path, result_root: Path, *extra: str):
    return module.build_parser().parse_args(
        [
            "--eventvot-root",
            str(tmp_path),
            "--base-results",
            str(result_root),
            "--output-root",
            str(tmp_path / "out"),
            "--split",
            "val",
            *extra,
        ],
    )


def _single_refiner_grid_args(module, tmp_path: Path, result_root: Path, *extra: str):
    return _parse_sweep_args(
        module,
        tmp_path,
        result_root,
        "--refinement-blend",
        "0.1",
        "--search-expansion-factor",
        "1.1",
        "--max-events",
        "64",
        "--min-events",
        "3",
        "--event-activity-floor",
        "0.0",
        "--inactive-activity-threshold",
        "0.1",
        "--measurement-noise-variance",
        "1.0",
        *extra,
    )


def test_eventvot_validation_metrics_match_perfect_boxes(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    split_root, result_root = _write_validation_fixture(tmp_path)

    metrics = module.evaluate_eventvot_results(split_root, result_root, ["seq1"])

    assert math.isclose(metrics["sr_auc"], 20.0 / 21.0)
    assert metrics["pr_auc"] == 1.0
    assert metrics["pr_20"] == 1.0
    assert metrics["npr_auc"] == 1.0
    assert metrics["npr_020"] == 1.0


def test_validation_sweep_refuses_test_split(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    args = argparse.Namespace(
        split="test",
        allow_test_split=False,
        eventvot_root=tmp_path,
        base_results=tmp_path / "base",
        output_root=tmp_path / "out",
    )

    with pytest.raises(SystemExit, match="Refusing to tune"):
        module.run_sweep(args)


def test_validation_sweep_dry_run_uses_requested_grid(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    _split_root, result_root = _write_validation_fixture(tmp_path)
    args = _parse_sweep_args(
        module,
        tmp_path,
        result_root,
        "--max-configs",
        "2",
        "--dry-run",
    )

    payload = module.run_sweep(args)

    assert payload["summary"]["dry_run"]
    assert payload["summary"]["config_count"] == 2
    assert payload["summary"]["completed_config_count"] == 0
    assert (tmp_path / "out" / "validation_sweep_summary.json").exists()


def test_validation_sweep_acceptance_grid_parses_dispatch_strings(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    _split_root, result_root = _write_validation_fixture(tmp_path)
    args = _single_refiner_grid_args(
        module,
        tmp_path,
        result_root,
        "--min-accept-used-events",
        "10,20",
        "--min-accept-candidate-iou",
        "0.85 0.95",
        "--dry-run",
    )

    grid = module.iter_parameter_grid(args)
    payload = module.run_sweep(args)

    assert len(grid) == 4
    assert payload["summary"]["config_count"] == 4
    assert sorted({config["min_accept_used_events"] for config in grid}) == [10, 20]
    assert sorted({config["min_accept_candidate_iou"] for config in grid}) == [
        0.85,
        0.95,
    ]
    assert all(config["max_accept_center_shift_ratio"] == 0.25 for config in grid)


def test_validation_sweep_optional_gates_default_to_none(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    _split_root, result_root = _write_validation_fixture(tmp_path)
    args = _single_refiner_grid_args(module, tmp_path, result_root, "--dry-run")

    grid = module.iter_parameter_grid(args)
    acceptance = module.acceptance_config_from_config(grid[0])

    assert len(grid) == 1
    for key in module.OPTIONAL_ACCEPTANCE_GRID_KEYS:
        assert grid[0][key] is None
    assert acceptance.min_raw_candidate_iou is None
    assert acceptance.min_raw_candidate_area_ratio is None
    assert acceptance.max_raw_candidate_area_ratio is None
    assert acceptance.max_raw_center_shift_ratio is None
    assert acceptance.min_polarity_consistency_fraction is None
    assert acceptance.min_mean_event_polarity_weight is None
    assert acceptance.max_quadratic_form_per_active_measurement is None
    assert acceptance.min_active_fraction is None


def test_validation_sweep_optional_gates_can_be_enabled(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    _split_root, result_root = _write_validation_fixture(tmp_path)
    args = _single_refiner_grid_args(
        module,
        tmp_path,
        result_root,
        "--min-raw-candidate-iou",
        "0.50",
        "--min-polarity-consistency-fraction",
        "0.25",
        "--dry-run",
    )

    config = module.iter_parameter_grid(args)[0]
    acceptance = module.acceptance_config_from_config(config)

    assert config["min_raw_candidate_iou"] == 0.50
    assert config["min_polarity_consistency_fraction"] == 0.25
    assert acceptance.min_raw_candidate_iou == 0.50
    assert acceptance.min_polarity_consistency_fraction == 0.25


def test_validation_sweep_required_gate_rejects_none_token(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    _split_root, result_root = _write_validation_fixture(tmp_path)
    args = _parse_sweep_args(
        module,
        tmp_path,
        result_root,
        "--min-accept-used-events",
        "none",
        "--dry-run",
    )

    with pytest.raises(ValueError, match="cannot be disabled"):
        module.iter_parameter_grid(args)


def test_validation_sweep_config_id_changes_for_every_grid_key(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    _split_root, result_root = _write_validation_fixture(tmp_path)
    args = _single_refiner_grid_args(module, tmp_path, result_root, "--dry-run")
    config = module.iter_parameter_grid(args)[0]
    config_id = module.make_config_id(1, config)

    assert "_h" in config_id
    for key in module.CONFIG_ID_KEYS:
        changed_config = dict(config)
        if changed_config[key] is None:
            changed_config[key] = 0.125
        elif key in module.INT_GRID_KEYS:
            changed_config[key] = int(changed_config[key]) + 1
        else:
            value = float(changed_config[key])
            changed_config[key] = 42.0 if not math.isfinite(value) else value + 0.125

        assert module.make_config_id(1, changed_config) != config_id, key


def test_validation_sweep_acceptance_config_comes_from_grid(monkeypatch):
    module = _load_module(monkeypatch)
    config = {
        "min_accept_used_events": 40,
        "min_accept_active_measurements": 8,
        "min_accept_mean_activity": 0.30,
        "min_accept_candidate_iou": 0.95,
        "min_accept_area_ratio": 0.80,
        "max_accept_area_ratio": 1.10,
        "max_accept_center_shift_ratio": 0.05,
        "min_raw_candidate_iou": 0.50,
        "min_raw_candidate_area_ratio": 0.25,
        "max_raw_candidate_area_ratio": 2.00,
        "max_raw_center_shift_ratio": 0.40,
        "min_polarity_consistency_fraction": 0.60,
        "min_mean_event_polarity_weight": -0.25,
        "max_quadratic_form_per_active_measurement": 8.0,
        "min_active_fraction": 0.10,
    }

    acceptance = module.acceptance_config_from_config(config)

    assert acceptance.min_used_event_count == 40
    assert acceptance.min_active_measurement_count == 8
    assert acceptance.min_mean_event_activity == 0.30
    assert acceptance.min_candidate_iou == 0.95
    assert acceptance.min_candidate_area_ratio == 0.80
    assert acceptance.max_candidate_area_ratio == 1.10
    assert acceptance.max_center_shift_ratio == 0.05
    assert acceptance.min_raw_candidate_iou == 0.50
    assert acceptance.min_raw_candidate_area_ratio == 0.25
    assert acceptance.max_raw_candidate_area_ratio == 2.00
    assert acceptance.max_raw_center_shift_ratio == 0.40
    assert acceptance.min_polarity_consistency_fraction == 0.60
    assert acceptance.min_mean_event_polarity_weight == -0.25
    assert acceptance.max_quadratic_form_per_active_measurement == 8.0
    assert acceptance.min_active_fraction == 0.10
