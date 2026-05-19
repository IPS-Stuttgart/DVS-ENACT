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


def test_acceptance_replay_excludes_raw_event_projection_modes():
    module = _load_module()

    assert "event-boundary-center" not in module.REPLAY_OUTPUT_MODES
    assert "event-centroid-center" not in module.REPLAY_OUTPUT_MODES
    assert "event-edge-center" not in module.REPLAY_OUTPUT_MODES
    assert "event-paired-edge-center" not in module.REPLAY_OUTPUT_MODES


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


def test_acceptance_replay_can_reblend_raw_refinement_before_gating():
    module = _load_module()
    frame = {
        "frame_index": 1,
        "fallback_reason": None,
        "refiner_output_xywh": [11.0, 10.0, 20.0, 20.0],
        "refined_bbox": {
            "x_min": 18.0,
            "y_min": 10.0,
            "x_max": 38.0,
            "y_max": 30.0,
        },
        "used_event_count": 32,
        "active_measurement_count": 16,
        "mean_event_activity": 0.8,
    }
    candidate = np.array([10.0, 10.0, 20.0, 20.0])

    diagnostic = module.frame_projected_output_xywh(
        candidate,
        frame,
        module.ReplayOutputProjectionConfig(),
    )
    reblended = module.frame_projected_output_xywh(
        candidate,
        frame,
        module.ReplayOutputProjectionConfig(mode="box", blend=0.25),
    )
    decision = module.evaluate_frame_acceptance(
        candidate,
        frame,
        module.ReplayAcceptanceConfig(max_center_shift_ratio=0.05),
        module.ReplayOutputProjectionConfig(mode="box", blend=0.25),
    )

    np.testing.assert_allclose(diagnostic, np.array([11.0, 10.0, 20.0, 20.0]))
    np.testing.assert_allclose(reblended, np.array([12.0, 10.0, 20.0, 20.0]))
    assert not decision.accepted
    assert decision.rejection_reasons == ("center_shift_ratio",)


def test_acceptance_replay_can_project_reblended_output_center_only():
    module = _load_module()
    frame = {
        "frame_index": 1,
        "fallback_reason": None,
        "refiner_output_xywh": [11.0, 10.0, 20.0, 20.0],
        "refined_bbox": {
            "x_min": 20.0,
            "y_min": 10.0,
            "x_max": 60.0,
            "y_max": 30.0,
        },
        "used_event_count": 32,
        "active_measurement_count": 16,
        "mean_event_activity": 0.8,
    }

    projected = module.frame_projected_output_xywh(
        np.array([10.0, 10.0, 20.0, 20.0]),
        frame,
        module.ReplayOutputProjectionConfig(mode="center-only", blend=0.5),
    )

    np.testing.assert_allclose(projected, np.array([20.0, 10.0, 20.0, 20.0]))


def test_acceptance_replay_can_project_width_only():
    module = _load_module()
    frame = {
        "frame_index": 1,
        "fallback_reason": None,
        "refiner_output_xywh": [0.0, 0.0, 40.0, 60.0],
        "refined_bbox": {
            "x_min": 0.0,
            "y_min": 0.0,
            "x_max": 40.0,
            "y_max": 60.0,
        },
        "used_event_count": 32,
        "active_measurement_count": 16,
        "mean_event_activity": 0.8,
    }

    projected = module.frame_projected_output_xywh(
        np.array([10.0, 10.0, 20.0, 30.0]),
        frame,
        module.ReplayOutputProjectionConfig(mode="width-only"),
    )

    np.testing.assert_allclose(projected, np.array([0.0, 10.0, 40.0, 30.0]))


def test_acceptance_replay_smooths_replayed_projected_size():
    module = _load_module()
    frame = {
        "frame_index": 2,
        "fallback_reason": None,
        "refiner_output_xywh": [0.0, 0.0, 40.0, 40.0],
        "refined_bbox": {
            "x_min": 0.0,
            "y_min": 0.0,
            "x_max": 40.0,
            "y_max": 40.0,
        },
        "used_event_count": 32,
        "active_measurement_count": 16,
        "mean_event_activity": 0.8,
    }

    projected = module.frame_projected_output_xywh(
        np.array([10.0, 10.0, 20.0, 20.0]),
        frame,
        module.ReplayOutputProjectionConfig(
            mode="size-only",
            blend=1.0,
            size_smoothing=0.5,
        ),
        previous_projected_size=np.array([20.0, 20.0]),
    )

    np.testing.assert_allclose(projected, np.array([5.0, 5.0, 30.0, 30.0]))


def test_acceptance_replay_confidence_weights_replayed_projection():
    module = _load_module()
    frame = {
        "frame_index": 1,
        "fallback_reason": None,
        "refiner_output_xywh": [20.0, 10.0, 20.0, 20.0],
        "refined_bbox": {
            "x_min": 20.0,
            "y_min": 10.0,
            "x_max": 40.0,
            "y_max": 30.0,
        },
        "used_event_count": 32,
        "active_measurement_count": 16,
        "mean_event_activity": 0.25,
    }

    projected = module.frame_projected_output_xywh(
        np.array([10.0, 10.0, 20.0, 20.0]),
        frame,
        module.ReplayOutputProjectionConfig(
            mode="box",
            confidence_field="mean_event_activity",
            confidence_floor=0.0,
            confidence_ceiling=0.5,
        ),
    )

    np.testing.assert_allclose(projected, np.array([15.0, 10.0, 20.0, 20.0]))


def test_acceptance_replay_size_deadband_filters_small_size_changes():
    module = _load_module()
    frame = {
        "frame_index": 1,
        "fallback_reason": None,
        "refiner_output_xywh": [9.0, 15.0, 102.0, 60.0],
        "refined_bbox": {
            "x_min": 9.0,
            "y_min": 15.0,
            "x_max": 111.0,
            "y_max": 75.0,
        },
        "used_event_count": 32,
        "active_measurement_count": 16,
        "mean_event_activity": 0.8,
    }

    projected = module.frame_projected_output_xywh(
        np.array([10.0, 20.0, 100.0, 50.0]),
        frame,
        module.ReplayOutputProjectionConfig(
            mode="size-only",
            size_deadband_ratio=0.05,
        ),
    )

    np.testing.assert_allclose(projected, np.array([10.0, 15.0, 100.0, 60.0]))


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


def test_acceptance_replay_run_uses_output_projection_config(tmp_path):
    module = _load_module()
    base_results = tmp_path / "base"
    base_results.mkdir()
    base_result_file = base_results / "recording_0001.txt"
    base_result_file.write_text(
        "10\t10\t20\t20\n11\t10\t20\t20\n",
        encoding="utf-8",
    )
    diagnostics_json = tmp_path / "diagnostics.json"
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
                                "refiner_output_xywh": [12.0, 10.0, 20.0, 20.0],
                                "refined_bbox": {
                                    "x_min": 21.0,
                                    "y_min": 10.0,
                                    "x_max": 61.0,
                                    "y_max": 30.0,
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
            output_projection_config=module.ReplayOutputProjectionConfig(
                mode="center-only",
                blend=0.5,
            ),
        )
    )

    replayed = np.loadtxt(output_results / "recording_0001.txt")
    np.testing.assert_allclose(
        replayed,
        np.array(
            [
                [10.0, 10.0, 20.0, 20.0],
                [21.0, 10.0, 20.0, 20.0],
            ]
        ),
    )
    assert payload["output_projection_config"] == {
        "mode": "center-only",
        "blend": 0.5,
        "size_smoothing": None,
        "size_deadband_ratio": None,
        "confidence_field": None,
        "confidence_floor": None,
        "confidence_ceiling": None,
        "image_width": 100.0,
        "image_height": 100.0,
    }


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
    assert "--replay-output-mode" in help_text
    assert "--replay-output-blend" in help_text
    assert "--replay-output-size-smoothing" in help_text
    assert "--replay-output-size-deadband-ratio" in help_text
    assert "--replay-output-confidence-field" in help_text
