import importlib.util
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eventvot_replay_projection_sweep.py"
SCRIPTS_DIR = SCRIPT_PATH.parent


def _load_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "run_eventvot_replay_projection_sweep_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_replay_fixture(root: Path) -> Path:
    base_results = root / "base"
    base_results.mkdir()
    base_result_file = base_results / "seq1.txt"
    base_result_file.write_text(
        "10\t10\t20\t20\n10\t10\t20\t20\n",
        encoding="utf-8",
    )
    diagnostics_json = root / "diagnostics.json"
    diagnostics_json.write_text(
        json.dumps(
            {
                "options": {"split": "test"},
                "refiner_config": {"image_width": 100, "image_height": 100},
                "acceptance_config": {
                    "min_used_event_count": 10,
                    "min_active_measurement_count": 3,
                    "min_mean_event_activity": 0.1,
                    "min_candidate_iou": 0.0,
                    "min_candidate_area_ratio": 0.0,
                    "max_candidate_area_ratio": 10.0,
                    "max_center_shift_ratio": 10.0,
                },
                "sequences": [
                    {
                        "sequence": "seq1",
                        "base_result_file": str(base_result_file),
                        "frames": [
                            {
                                "frame_index": 0,
                                "fallback_reason": "initial_frame",
                                "refiner_output_xywh": [10.0, 10.0, 20.0, 20.0],
                            },
                            {
                                "frame_index": 1,
                                "fallback_reason": None,
                                "refiner_output_xywh": [11.0, 10.0, 20.0, 20.0],
                                "refined_bbox": {
                                    "x_min": 20.0,
                                    "y_min": 5.0,
                                    "x_max": 60.0,
                                    "y_max": 35.0,
                                },
                                "used_event_count": 32,
                                "active_measurement_count": 16,
                                "mean_event_activity": 0.8,
                            },
                        ],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return diagnostics_json


def _write_eventvot_fixture(root: Path) -> Path:
    split_root = root / "Testing Subset"
    sequence_dir = split_root / "seq1"
    sequence_dir.mkdir(parents=True)
    (split_root / "list.txt").write_text("seq1\n", encoding="utf-8")
    (sequence_dir / "groundtruth.txt").write_text(
        "10\t10\t20\t20\n20\t5\t40\t30\n",
        encoding="utf-8",
    )
    (sequence_dir / "absent.txt").write_text("1\n1\n", encoding="utf-8")
    return root


def _parse_sweep_args(module, diagnostics_json: Path, output_root: Path, *extra):
    return module.build_parser().parse_args(
        [
            "--diagnostics-json",
            str(diagnostics_json),
            "--output-root",
            str(output_root),
            "--skip-evaluation",
            *extra,
        ]
    )


def test_replay_projection_sweep_help_runs_as_script():
    help_text = subprocess.check_output(
        (sys.executable, str(SCRIPT_PATH), "--help"),
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )

    assert "Sweep replay-output policies" in help_text
    assert "--diagnostics-json" in help_text
    assert "--replay-output-mode" in help_text
    assert "--replay-output-size-clamp-ratio" in help_text
    assert "--replay-output-center-smoothing" in help_text
    assert "--replay-output-motion-smoothing" in help_text
    assert "--replay-output-confidence-field" in help_text
    assert "--min-raw-candidate-iou" in help_text
    assert "--min-active-fraction" in help_text
    assert "--max-temporal-center-shift-ratio" in help_text
    assert "--max-temporal-size-change-ratio" in help_text
    assert "--max-motion-prediction-error-ratio" in help_text
    assert "--max-rejected-center-hold-frames" in help_text
    assert "--rejected-center-hold-decay" in help_text
    assert "--max-rejected-center-hold-support-score" in help_text
    assert "--sequence-metrics-csv" in help_text


def test_replay_projection_sweep_grid_parses_dispatch_strings(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    args = module.build_parser().parse_args(
        [
            "--diagnostics-json",
            str(tmp_path / "diagnostics.json"),
            "--output-root",
            str(tmp_path / "out"),
            "--skip-evaluation",
            "--replay-output-mode",
            "box scale-only",
            "--replay-output-blend",
            "none,0.25",
            "--replay-output-size-clamp-ratio",
            "none 0.20",
            "--replay-output-center-smoothing",
            "none 0.50",
            "--replay-output-motion-smoothing",
            "none 0.25",
        ]
    )

    configs = module.iter_projection_grid(args)

    assert len(configs) == 32
    assert sorted({config.mode for config in configs}) == ["box", "scale-only"]
    assert sorted({config.blend for config in configs}, key=lambda value: value or -1) == [
        None,
        0.25,
    ]
    assert sorted(
        {config.size_clamp_ratio for config in configs},
        key=lambda value: value or -1,
    ) == [None, 0.20]
    assert sorted(
        {config.center_smoothing for config in configs},
        key=lambda value: value or -1,
    ) == [None, 0.50]
    assert sorted(
        {config.motion_smoothing for config in configs},
        key=lambda value: value or -1,
    ) == [None, 0.25]


def test_replay_projection_sweep_acceptance_grid_parses_dispatch_strings(
    tmp_path,
    monkeypatch,
):
    module = _load_module(monkeypatch)
    args = module.build_parser().parse_args(
        [
            "--diagnostics-json",
            str(tmp_path / "diagnostics.json"),
            "--output-root",
            str(tmp_path / "out"),
            "--skip-evaluation",
            "--replay-output-mode",
            "box center-only",
            "--min-accept-used-events",
            "diagnostic 20",
            "--min-raw-candidate-iou",
            "diagnostic none 0.50",
            "--max-temporal-center-shift-ratio",
            "diagnostic 0.75",
            "--max-motion-prediction-error-ratio",
            "diagnostic 0.50",
            "--max-rejected-center-hold-frames",
            "diagnostic 2",
            "--rejected-center-hold-decay",
            "diagnostic 0.50",
            "--max-rejected-center-hold-support-score",
            "diagnostic 0.25",
        ]
    )

    configs = module.iter_sweep_grid(args)

    assert len(configs) == 384
    assert sorted({config.output_projection.mode for config in configs}) == [
        "box",
        "center-only",
    ]
    overrides = [config.acceptance_overrides for config in configs]
    assert {} in overrides
    assert {"min_used_event_count": 20} in overrides
    assert {"min_raw_candidate_iou": None} in overrides
    assert {"min_raw_candidate_iou": 0.50} in overrides
    assert {
        "min_used_event_count": 20,
        "min_raw_candidate_iou": 0.50,
    } in overrides
    assert {"max_temporal_center_shift_ratio": 0.75} in overrides
    assert {"max_motion_prediction_error_ratio": 0.50} in overrides
    assert {"max_rejected_center_hold_frames": 2} in overrides
    assert {"rejected_center_hold_decay": 0.50} in overrides
    assert {"max_rejected_center_hold_support_score": 0.25} in overrides


def test_replay_projection_sweep_rewrites_result_files(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    diagnostics_json = _write_replay_fixture(tmp_path)
    output_root = tmp_path / "sweep"
    args = _parse_sweep_args(
        module,
        diagnostics_json,
        output_root,
        "--replay-output-mode",
        "box center-only",
        "--replay-output-blend",
        "1.0",
    )

    payload = module.run_projection_sweep(args)

    assert payload["summary"]["config_count"] == 2
    assert payload["summary"]["completed_config_count"] == 2
    assert payload["summary"]["best_config_id"] is not None
    assert len(payload["top_configs"]) == 2
    assert (output_root / "projection_sweep_metrics.csv").exists()
    assert (output_root / "projection_sweep_summary.json").exists()
    result_files = sorted((output_root / "results").glob("*/seq1.txt"))
    assert len(result_files) == 2
    rewritten = [np.loadtxt(path) for path in result_files]
    assert any(
        np.allclose(boxes[1], np.array([20.0, 5.0, 40.0, 30.0]))
        for boxes in rewritten
    )
    assert any(
        np.allclose(boxes[1], np.array([30.0, 10.0, 20.0, 20.0]))
        for boxes in rewritten
    )


def test_replay_projection_sweep_can_sweep_acceptance_gates(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    diagnostics_json = _write_replay_fixture(tmp_path)
    output_root = tmp_path / "sweep"
    args = _parse_sweep_args(
        module,
        diagnostics_json,
        output_root,
        "--replay-output-mode",
        "box",
        "--replay-output-blend",
        "1.0",
        "--min-accept-active-measurements",
        "diagnostic 999",
    )

    payload = module.run_projection_sweep(args)

    assert payload["summary"]["config_count"] == 2
    rows = payload["top_configs"]
    assert sorted(row["accepted_refinement_count"] for row in rows) == [0, 1]
    assert {row["acceptance_min_active_measurement_count"] for row in rows} == {
        3,
        999,
    }
    assert any(row["acceptance_overrides"] == "{}" for row in rows)
    assert any(
        row["acceptance_overrides"] == '{"min_active_measurement_count":999}'
        for row in rows
    )


def test_replay_projection_sweep_reports_baseline_deltas(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    diagnostics_json = _write_replay_fixture(tmp_path)
    eventvot_root = _write_eventvot_fixture(tmp_path / "eventvot")
    output_root = tmp_path / "sweep"
    args = module.build_parser().parse_args(
        [
            "--diagnostics-json",
            str(diagnostics_json),
            "--eventvot-root",
            str(eventvot_root),
            "--output-root",
            str(output_root),
            "--replay-output-mode",
            "box",
            "--replay-output-blend",
            "1.0",
            "--rank-metric",
            "delta_sr_auc",
        ]
    )

    payload = module.run_projection_sweep(args)

    assert payload["baseline_metrics"] is not None
    assert payload["summary"]["rank_metric"] == "delta_sr_auc"
    assert payload["summary"]["best_delta_sr_auc"] > 0.0
    assert payload["summary"]["best_rank_metric"] == payload["summary"]["best_delta_sr_auc"]
    assert payload["summary"]["sequence_metrics_csv"] is not None
    top_config = payload["top_configs"][0]
    assert top_config["delta_sr_auc"] > 0.0
    assert top_config["delta_pr_auc"] > 0.0
    assert top_config["delta_npr_auc"] > 0.0
    assert top_config["mean_iou"] > payload["baseline_metrics"]["mean_iou"]
    assert top_config["sequence_delta_sr_positive_count"] == 1
    assert top_config["sequence_delta_sr_negative_count"] == 0
    assert top_config["best_sequence_delta_sr_sequence"] == "seq1"
    assert top_config["worst_sequence_delta_sr_sequence"] == "seq1"

    sequence_metrics_csv = output_root / "projection_sweep_sequence_metrics.csv"
    assert sequence_metrics_csv.exists()
    with sequence_metrics_csv.open(newline="", encoding="utf-8") as handle:
        sequence_rows = list(csv.DictReader(handle))
    assert len(sequence_rows) == 1
    assert sequence_rows[0]["config_id"] == top_config["config_id"]
    assert sequence_rows[0]["sequence"] == "seq1"
    assert float(sequence_rows[0]["delta_sr_auc"]) > 0.0
    assert float(sequence_rows[0]["delta_pr_auc"]) > 0.0
    assert int(sequence_rows[0]["accepted_refinement_count"]) == 1
