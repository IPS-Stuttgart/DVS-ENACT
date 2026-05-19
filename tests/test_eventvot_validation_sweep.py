import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
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


def _single_config_grid_args() -> tuple[str, ...]:
    return (
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


def test_eventvot_curve_average_keeps_all_zero_sequences(monkeypatch):
    module = _load_module(monkeypatch)
    perfect_curve = np.ones_like(module.OVERLAP_THRESHOLDS, dtype=float)
    failed_curve = np.zeros_like(module.OVERLAP_THRESHOLDS, dtype=float)

    averaged = module.mean_curves([perfect_curve, failed_curve])

    assert np.allclose(averaged, 0.5)


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
    args = _parse_sweep_args(
        module,
        tmp_path,
        result_root,
        *_single_config_grid_args(),
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
    assert all(config["refinement_mode"] == "box" for config in grid)


def test_validation_sweep_projection_grid_parses_dispatch_strings(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    _split_root, result_root = _write_validation_fixture(tmp_path)
    args = _parse_sweep_args(
        module,
        tmp_path,
        result_root,
        *_single_config_grid_args(),
        "--refinement-mode",
        "box size-only width-only height-only",
        "--projection-width-blend",
        "none,0.10",
        "--projection-height-blend",
        "none,0.10",
        "--projection-size-smoothing",
        "none,0.50",
        "--projection-size-deadband-ratio",
        "none,0.05",
        "--projection-confidence-field",
        "none mean_event_activity",
        "--projection-confidence-floor",
        "none 0.10",
        "--projection-confidence-ceiling",
        "none 0.50",
        "--projection-no-clip",
        "--dry-run",
    )

    grid = module.iter_parameter_grid(args)
    payload = module.run_sweep(args)

    assert len(grid) == 64
    assert payload["summary"]["config_count"] == 64
    assert sorted({config["refinement_mode"] for config in grid}) == [
        "box",
        "height-only",
        "size-only",
        "width-only",
    ]
    assert sorted(
        {
            config["projection_width_blend"]
            for config in grid
            if config["projection_width_blend"] is not None
        }
    ) == [0.10]
    assert sorted(
        {
            config["projection_size_smoothing"]
            for config in grid
            if config["projection_size_smoothing"] is not None
        }
    ) == [0.50]
    assert sorted(
        {
            config["projection_size_deadband_ratio"]
            for config in grid
            if config["projection_size_deadband_ratio"] is not None
        }
    ) == [0.05]
    assert sorted(
        {
            config["projection_confidence_field"]
            for config in grid
            if config["projection_confidence_field"] is not None
        }
    ) == ["mean_event_activity"]
    assert all(config["projection_no_clip"] for config in grid)


def test_validation_sweep_optional_acceptance_defaults_disable_gates(
    tmp_path,
    monkeypatch,
):
    module = _load_module(monkeypatch)
    _split_root, result_root = _write_validation_fixture(tmp_path)
    args = _parse_sweep_args(
        module,
        tmp_path,
        result_root,
        *_single_config_grid_args(),
        "--dry-run",
    )

    config = module.iter_parameter_grid(args)[0]
    acceptance = module.acceptance_config_from_config(config)

    assert config["min_raw_candidate_iou"] is None
    assert config["min_raw_candidate_area_ratio"] is None
    assert config["max_raw_candidate_area_ratio"] is None
    assert config["max_raw_center_shift_ratio"] is None
    assert config["min_polarity_consistency_fraction"] is None
    assert config["min_mean_event_polarity_weight"] is None
    assert config["max_quadratic_form_per_active_measurement"] is None
    assert config["min_active_fraction"] is None
    assert acceptance.min_raw_candidate_iou is None
    assert acceptance.min_raw_candidate_area_ratio is None
    assert acceptance.max_raw_candidate_area_ratio is None
    assert acceptance.max_raw_center_shift_ratio is None
    assert acceptance.min_polarity_consistency_fraction is None
    assert acceptance.min_mean_event_polarity_weight is None
    assert acceptance.max_quadratic_form_per_active_measurement is None
    assert acceptance.min_active_fraction is None


def test_validation_sweep_optional_none_token_can_be_swept_with_values(monkeypatch):
    module = _load_module(monkeypatch)

    values = module.parse_sweep_values(
        ("none,0.25 off 0.5",),
        cast=float,
        argument_name="--min-active-fraction",
        allow_none=True,
    )

    assert values == [None, 0.25, 0.5]


def test_validation_sweep_default_gates_accept_missing_polarity_metrics(
    tmp_path,
    monkeypatch,
):
    module = _load_module(monkeypatch)
    _split_root, result_root = _write_validation_fixture(tmp_path)
    args = _parse_sweep_args(
        module,
        tmp_path,
        result_root,
        *_single_config_grid_args(),
        "--disable-event-polarity",
        "--dry-run",
    )
    config = module.iter_parameter_grid(args)[0]
    acceptance = module.acceptance_config_from_config(config)
    refinement_module = sys.modules["run_eventvot_refinement"]

    decision = refinement_module.evaluate_refinement_acceptance(
        np.array([10.0, 10.0, 20.0, 20.0]),
        _FakeResult([10.0, 10.0, 20.0, 20.0]),
        acceptance,
    )

    assert decision.accepted
    assert "polarity_consistency_fraction_missing" not in decision.rejection_reasons
    assert "mean_event_polarity_weight_missing" not in decision.rejection_reasons


def test_validation_sweep_config_id_changes_for_every_grid_key(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    _split_root, result_root = _write_validation_fixture(tmp_path)
    args = _parse_sweep_args(
        module,
        tmp_path,
        result_root,
        *_single_config_grid_args(),
        "--dry-run",
    )
    config = module.iter_parameter_grid(args)[0]
    config_id = module.make_config_id(1, config)

    assert "_h" in config_id
    for key in module.CONFIG_ID_KEYS:
        changed_config = dict(config)
        value = changed_config[key]
        if key in module.INT_GRID_KEYS:
            changed_config[key] = int(value) + 1
        elif key in module.STRING_GRID_KEYS:
            if key == "projection_confidence_field":
                changed_config[key] = "mean_event_activity"
                changed_config["projection_confidence_floor"] = 0.1
                changed_config["projection_confidence_ceiling"] = 0.5
            else:
                changed_config[key] = "size-only" if value != "size-only" else "box"
        elif key in module.BOOL_GRID_KEYS:
            changed_config[key] = not value
        elif value is None:
            changed_config[key] = 0.125
        else:
            numeric_value = float(value)
            changed_config[key] = (
                42.0 if not math.isfinite(numeric_value) else numeric_value + 0.125
            )

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
        "min_raw_candidate_iou": 0.40,
        "min_raw_candidate_area_ratio": 0.60,
        "max_raw_candidate_area_ratio": 1.40,
        "max_raw_center_shift_ratio": 0.20,
        "min_polarity_consistency_fraction": 0.75,
        "min_mean_event_polarity_weight": 0.10,
        "max_quadratic_form_per_active_measurement": 2.0,
        "min_active_fraction": 0.50,
    }

    acceptance = module.acceptance_config_from_config(config)

    assert acceptance.min_used_event_count == 40
    assert acceptance.min_active_measurement_count == 8
    assert acceptance.min_mean_event_activity == 0.30
    assert acceptance.min_candidate_iou == 0.95
    assert acceptance.min_candidate_area_ratio == 0.80
    assert acceptance.max_candidate_area_ratio == 1.10
    assert acceptance.max_center_shift_ratio == 0.05
    assert acceptance.min_raw_candidate_iou == 0.40
    assert acceptance.min_raw_candidate_area_ratio == 0.60
    assert acceptance.max_raw_candidate_area_ratio == 1.40
    assert acceptance.max_raw_center_shift_ratio == 0.20
    assert acceptance.min_polarity_consistency_fraction == 0.75
    assert acceptance.min_mean_event_polarity_weight == 0.10
    assert acceptance.max_quadratic_form_per_active_measurement == 2.0
    assert acceptance.min_active_fraction == 0.50


def test_validation_sweep_make_refiner_wraps_projection_mode(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    _split_root, result_root = _write_validation_fixture(tmp_path)
    args = _parse_sweep_args(
        module,
        tmp_path,
        result_root,
        *_single_config_grid_args(),
        "--refinement-mode",
        "size-only",
        "--projection-width-blend",
        "0.10",
        "--projection-height-blend",
        "0.10",
        "--projection-size-smoothing",
        "0.50",
        "--projection-size-deadband-ratio",
        "0.05",
        "--projection-confidence-field",
        "mean_event_activity",
        "--projection-confidence-floor",
        "0.10",
        "--projection-confidence-ceiling",
        "0.50",
    )
    config = module.iter_parameter_grid(args)[0]

    refiner = module.make_refiner(config, args)

    assert refiner.refinement_mode == "size-only"
    assert refiner.projection_width_blend == 0.10
    assert refiner.projection_height_blend == 0.10
    assert refiner.projection_size_smoothing == 0.50
    assert refiner.projection_size_deadband_ratio == 0.05
    assert refiner.projection_confidence_field == "mean_event_activity"
    assert refiner.projection_confidence_floor == 0.10
    assert refiner.projection_confidence_ceiling == 0.50


class _FakeResult:
    fallback_reason = None
    used_event_count = 12
    active_measurement_count = 3
    mean_event_activity = 0.20
    mean_event_polarity_weight = None
    polarity_consistency_fraction = None
    quadratic_form = None

    def __init__(self, xywh):
        self._xywh = tuple(float(value) for value in xywh)

    def as_xywh(self):
        return self._xywh

    @property
    def refined_bbox(self):
        x, y, width, height = self._xywh
        return {
            "x_min": x,
            "y_min": y,
            "width": width,
            "height": height,
        }
