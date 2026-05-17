import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "report_eventvot_refinement_diagnostics.py"
SCRIPTS_DIR = SCRIPT_PATH.parent


def _load_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "report_eventvot_refinement_diagnostics_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_sequence(split_root: Path, name: str) -> None:
    sequence_dir = split_root / name
    sequence_dir.mkdir(parents=True)
    (sequence_dir / "groundtruth.txt").write_text(
        "1 1 10 10\n1 1 10 10\n1 1 10 10\n",
        encoding="utf-8",
    )
    (sequence_dir / "absent.txt").write_text("1\n1\n1\n", encoding="utf-8")


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    split_root = root / "test"
    split_root.mkdir()
    (split_root / "list.txt").write_text("seq1\nseq2\n", encoding="utf-8")
    _write_sequence(split_root, "seq1")
    _write_sequence(split_root, "seq2")

    base_results = root / "base"
    refined_results = root / "refined"
    base_results.mkdir()
    refined_results.mkdir()

    perfect = "1 1 10 10\n1 1 10 10\n1 1 10 10\n"
    shifted = "1 1 10 10\n3 1 10 10\n3 1 10 10\n"
    worse = "1 1 10 10\n5 1 10 10\n5 1 10 10\n"

    (base_results / "seq1.txt").write_text(perfect, encoding="utf-8")
    (refined_results / "seq1.txt").write_text(shifted, encoding="utf-8")
    (base_results / "seq2.txt").write_text(worse, encoding="utf-8")
    (refined_results / "seq2.txt").write_text(perfect, encoding="utf-8")

    diagnostics = {
        "options": {
            "eventvot_root": str(root),
            "split": "test",
            "base_results": str(base_results),
            "resolved_output_results": str(refined_results),
        },
        "eventvot_evaluator": {"tracking_result_dir": str(refined_results)},
        "sequences": [
            {
                "sequence": "seq1",
                "frame_count": 3,
                "accepted_refinement_count": 2,
                "refiner_success_frame_count": 2,
                "fallback_counts": {"refined": 2, "initial_frame": 1},
                "acceptance_counts": {"accepted": 2, "initial_frame": 1},
                "frames": [
                    {"frame_index": 0},
                    {
                        "frame_index": 1,
                        "candidate_iou": 0.9,
                        "raw_candidate_iou": 0.8,
                        "center_shift_ratio": 0.1,
                        "raw_center_shift_ratio": 0.2,
                        "candidate_area_ratio": 1.0,
                        "raw_candidate_area_ratio": 1.0,
                        "used_event_count": 20,
                        "event_count": 25,
                        "active_measurement_count": 10,
                        "quadratic_form": 40.0,
                    },
                    {
                        "frame_index": 2,
                        "candidate_iou": 0.8,
                        "raw_candidate_iou": 0.7,
                        "center_shift_ratio": 0.1,
                        "raw_center_shift_ratio": 0.2,
                        "candidate_area_ratio": 1.0,
                        "raw_candidate_area_ratio": 1.0,
                        "used_event_count": 10,
                        "event_count": 12,
                        "active_measurement_count": 5,
                        "quadratic_form": 10.0,
                    },
                ],
            },
            {
                "sequence": "seq2",
                "frame_count": 3,
                "accepted_refinement_count": 1,
                "refiner_success_frame_count": 2,
                "fallback_counts": {"refined": 2, "initial_frame": 1},
                "acceptance_counts": {
                    "accepted": 1,
                    "raw_candidate_iou": 1,
                    "initial_frame": 1,
                },
                "frames": [
                    {"frame_index": 0},
                    {
                        "frame_index": 1,
                        "candidate_iou": 0.95,
                        "raw_candidate_iou": 0.95,
                        "center_shift_ratio": 0.05,
                        "raw_center_shift_ratio": 0.05,
                        "candidate_area_ratio": 1.0,
                        "raw_candidate_area_ratio": 1.0,
                        "active_fraction": 0.8,
                        "quadratic_form_per_active_measurement": 1.5,
                        "used_event_count": 20,
                        "event_count": 30,
                        "active_measurement_count": 16,
                    },
                ],
            },
        ],
    }
    diagnostics_path = root / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    return base_results, refined_results, diagnostics_path


def test_refinement_diagnostic_report_writes_sequence_rows(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    base_results, refined_results, diagnostics_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "report"

    args = module.build_parser().parse_args(
        [
            "--diagnostics-json",
            str(diagnostics_path),
            "--eventvot-root",
            str(tmp_path),
            "--base-results",
            str(base_results),
            "--refined-results",
            str(refined_results),
            "--split",
            "test",
            "--output-dir",
            str(output_dir),
        ]
    )

    payload = module.run_report(args)

    assert payload["summary"]["sequence_count"] == 2
    assert (output_dir / "per_sequence_deltas.csv").exists()
    assert (output_dir / "acceptance_reasons.csv").exists()
    assert (output_dir / "diagnostic_correlations.csv").exists()
    assert (output_dir / "worst_sequences.md").exists()
    assert (output_dir / "best_sequences.md").exists()

    rows = {row["sequence"]: row for row in payload["per_sequence"]}
    assert rows["seq1"]["delta_sr"] < 0.0
    assert rows["seq2"]["delta_sr"] > 0.0
    assert np.isclose(rows["seq1"]["accepted_refinement_rate"], 1.0)
    assert np.isclose(rows["seq2"]["accepted_refinement_rate"], 0.5)
    assert np.isclose(rows["seq1"]["mean_active_fraction"], 0.5)
    assert rows["seq2"]["mean_raw_candidate_iou"] == 0.95

    reason_rows = {row["reason"]: row for row in payload["acceptance_reasons"]}
    assert reason_rows["accepted"]["count"] == 3
    assert reason_rows["raw_candidate_iou"]["count"] == 1


def test_refinement_diagnostic_report_infers_paths_from_diagnostics(
    tmp_path,
    monkeypatch,
):
    module = _load_module(monkeypatch)
    _base_results, _refined_results, diagnostics_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "report"

    args = module.build_parser().parse_args(
        [
            "--diagnostics-json",
            str(diagnostics_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    payload = module.run_report(args)

    assert payload["summary"]["sequence_count"] == 2
    assert payload["summary"]["base_results"].endswith("base")
    assert payload["summary"]["refined_results"].endswith("refined")
