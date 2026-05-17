import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eventvot_acceptance_replay.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_eventvot_acceptance_replay_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_acceptance_replay_raw_gate_rejects_bad_unblended_update():
    module = _load_module()
    frame = {
        "frame_index": 1,
        "fallback_reason": None,
        "refiner_output_xywh": [10.5, 10.0, 20.0, 20.0],
        "refined_bbox": {
            "x_min": 80.0,
            "y_min": 80.0,
            "x_max": 100.0,
            "y_max": 100.0,
        },
        "used_event_count": 32,
        "active_measurement_count": 16,
        "mean_event_activity": 0.8,
        "polarity_consistency_fraction": 0.9,
        "mean_event_polarity_weight": 1.0,
        "quadratic_form": 4.0,
    }

    accepted_without_raw_gate = module.evaluate_frame_acceptance(
        np.array([10.0, 10.0, 20.0, 20.0]),
        frame,
        module.ReplayAcceptanceConfig(),
    )
    rejected_with_raw_gate = module.evaluate_frame_acceptance(
        np.array([10.0, 10.0, 20.0, 20.0]),
        frame,
        module.ReplayAcceptanceConfig(min_raw_candidate_iou=0.10),
    )

    assert accepted_without_raw_gate.accepted
    assert not rejected_with_raw_gate.accepted
    assert rejected_with_raw_gate.rejection_reasons == ("raw_candidate_iou",)
    assert rejected_with_raw_gate.candidate_iou > 0.9
    assert rejected_with_raw_gate.raw_candidate_iou == 0.0


def test_acceptance_replay_rewrites_result_file_from_diagnostics(tmp_path):
    module = _load_module()
    base_results = tmp_path / "base"
    base_results.mkdir()
    base_result_file = base_results / "recording_0001.txt"
    base_result_file.write_text(
        "10\t10\t20\t20\n11\t10\t20\t20\n12\t10\t20\t20\n",
        encoding="utf-8",
    )
    diagnostics_json = tmp_path / "diagnostics.json"
    diagnostics_json.write_text(
        json.dumps(
            {
                "options": {"split": "test"},
                "acceptance_config": {
                    "min_used_event_count": 10,
                    "min_active_measurement_count": 3,
                    "min_mean_event_activity": 0.1,
                    "min_candidate_iou": 0.6,
                    "min_candidate_area_ratio": 0.5,
                    "max_candidate_area_ratio": 1.5,
                    "max_center_shift_ratio": 0.25,
                },
                "sequences": [
                    {
                        "sequence": "recording_0001",
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
                                "refiner_output_xywh": [11.5, 10.0, 20.0, 20.0],
                                "refined_bbox": {
                                    "x_min": 11.5,
                                    "y_min": 10.0,
                                    "x_max": 31.5,
                                    "y_max": 30.0,
                                },
                                "used_event_count": 32,
                                "active_measurement_count": 16,
                                "mean_event_activity": 0.8,
                            },
                            {
                                "frame_index": 2,
                                "fallback_reason": None,
                                "refiner_output_xywh": [12.2, 10.0, 20.0, 20.0],
                                "refined_bbox": {
                                    "x_min": 80.0,
                                    "y_min": 80.0,
                                    "x_max": 100.0,
                                    "y_max": 100.0,
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
    output_results = tmp_path / "replayed"

    payload = module.run(
        module.EventVOTAcceptanceReplayOptions(
            diagnostics_json=diagnostics_json,
            output_results=output_results,
            skip_evaluation=True,
            acceptance_config=module.ReplayAcceptanceConfig(min_raw_candidate_iou=0.10),
        )
    )

    replayed = np.loadtxt(output_results / "recording_0001.txt")
    np.testing.assert_allclose(
        replayed,
        np.array(
            [
                [10.0, 10.0, 20.0, 20.0],
                [11.5, 10.0, 20.0, 20.0],
                [12.0, 10.0, 20.0, 20.0],
            ]
        ),
    )
    assert payload["summary"]["accepted_refinement_count"] == 1
    assert payload["summary"]["acceptance_counts"] == {
        "accepted": 1,
        "initial_frame": 1,
        "raw_candidate_iou": 1,
    }
    assert (output_results / "acceptance_replay_summary.json").exists()


def test_acceptance_replay_help_runs_as_script():
    help_text = subprocess.check_output(
        (sys.executable, str(SCRIPT_PATH), "--help"),
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )

    assert "Replay EventVOT DVS-ENACT" in help_text
    assert "--diagnostics-json" in help_text
    assert "--min-raw-candidate-iou" in help_text
    assert "--min-polarity-consistency-fraction" in help_text
    assert "--max-quadratic-form-per-active-measurement" in help_text
