import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


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


def _diagnostic_frame(
    output_xywh,
    *,
    frame_index=1,
    mean_event_activity=0.8,
    used_event_count=32,
    active_measurement_count=16,
    polarity_consistency_fraction=None,
):
    x, y, width, height = (float(value) for value in output_xywh)
    frame = {
        "frame_index": frame_index,
        "fallback_reason": None,
        "refiner_output_xywh": [x, y, width, height],
        "refined_bbox": {
            "x_min": x,
            "y_min": y,
            "x_max": x + width,
            "y_max": y + height,
        },
        "used_event_count": used_event_count,
        "active_measurement_count": active_measurement_count,
        "mean_event_activity": mean_event_activity,
    }
    if polarity_consistency_fraction is not None:
        frame["polarity_consistency_fraction"] = polarity_consistency_fraction
    return frame


def _write_replay_diagnostics(
    path: Path,
    base_result_file: Path,
    frames: list[dict],
    *,
    acceptance_config: dict | None = None,
    refiner_config: dict | None = None,
) -> None:
    payload = {
        "options": {"split": "test"},
        "acceptance_config": acceptance_config or {},
        "sequences": [
            {
                "sequence": "recording_0001",
                "base_result_file": str(base_result_file),
                "frames": frames,
            }
        ],
    }
    if refiner_config is not None:
        payload["refiner_config"] = refiner_config
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _replay_center_hold_sequence(module, frames, **config_kwargs):
    base_boxes = np.array(
        [
            [8.0, 8.0, 10.0, 10.0],
            [9.0, 8.0, 10.0, 10.0],
            [10.0, 8.0, 10.0, 10.0],
        ]
    )
    config = module.ReplayAcceptanceConfig(
        max_rejected_center_hold_frames=1,
        rejected_center_hold_decay=1.0,
        **config_kwargs,
    )
    return module.replay_sequence_boxes("seq1", base_boxes, frames, config)


def test_acceptance_replay_excludes_raw_event_projection_modes():
    module = _load_module()

    assert "event-centroid-center" not in module.REPLAY_OUTPUT_MODES


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


def test_acceptance_replay_event_support_score_gate_rejects_weak_support():
    module = _load_module()
    decision = module.evaluate_frame_acceptance(
        np.array([10.0, 10.0, 20.0, 20.0]),
        _diagnostic_frame(
            [10.0, 10.0, 20.0, 20.0],
            used_event_count=64,
            active_measurement_count=3,
            mean_event_activity=0.20,
        ),
        module.ReplayAcceptanceConfig(min_event_support_score=0.50),
    )

    assert not decision.accepted
    assert decision.rejection_reasons == ("event_support_score",)


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


def test_acceptance_replay_can_project_scale_only():
    module = _load_module()
    frame = {
        "frame_index": 1,
        "fallback_reason": None,
        "refiner_output_xywh": [0.0, 0.0, 40.0, 20.0],
        "refined_bbox": {
            "x_min": 0.0,
            "y_min": 0.0,
            "x_max": 40.0,
            "y_max": 20.0,
        },
        "used_event_count": 32,
        "active_measurement_count": 16,
        "mean_event_activity": 0.8,
    }

    projected = module.frame_projected_output_xywh(
        np.array([0.0, 0.0, 10.0, 20.0]),
        frame,
        module.ReplayOutputProjectionConfig(mode="scale-only"),
    )

    np.testing.assert_allclose(projected, np.array([-5.0, -10.0, 20.0, 40.0]))


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


def test_acceptance_replay_smooths_replayed_projected_center():
    module = _load_module()
    frame = {
        "frame_index": 2,
        "fallback_reason": None,
        "refiner_output_xywh": [30.0, 10.0, 20.0, 20.0],
        "refined_bbox": {
            "x_min": 30.0,
            "y_min": 10.0,
            "x_max": 50.0,
            "y_max": 30.0,
        },
        "used_event_count": 32,
        "active_measurement_count": 16,
        "mean_event_activity": 0.8,
    }

    projected = module.frame_projected_output_xywh(
        np.array([10.0, 10.0, 20.0, 20.0]),
        frame,
        module.ReplayOutputProjectionConfig(
            mode="box",
            center_smoothing=0.5,
        ),
        previous_projected_center=np.array([20.0, 20.0]),
    )

    np.testing.assert_allclose(projected, np.array([20.0, 10.0, 20.0, 20.0]))


def test_acceptance_replay_smooths_replayed_center_with_base_motion():
    module = _load_module()
    frame = {
        "frame_index": 2,
        "fallback_reason": None,
        "refiner_output_xywh": [30.0, 0.0, 20.0, 20.0],
        "refined_bbox": {
            "x_min": 30.0,
            "y_min": 0.0,
            "x_max": 50.0,
            "y_max": 20.0,
        },
        "used_event_count": 32,
        "active_measurement_count": 16,
        "mean_event_activity": 0.8,
    }

    projected = module.frame_projected_output_xywh(
        np.array([10.0, 0.0, 20.0, 20.0]),
        frame,
        module.ReplayOutputProjectionConfig(
            mode="center-only",
            motion_smoothing=0.5,
        ),
        previous_projected_center=np.array([12.0, 10.0]),
        previous_candidate_center=np.array([10.0, 10.0]),
    )

    np.testing.assert_allclose(projected, np.array([21.0, 0.0, 20.0, 20.0]))


def test_acceptance_replay_confidence_weights_replayed_projection():
    module = _load_module()
    frame = _diagnostic_frame([20.0, 10.0, 20.0, 20.0], mean_event_activity=0.25)

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
    frame = _diagnostic_frame([9.0, 15.0, 102.0, 60.0])

    projected = module.frame_projected_output_xywh(
        np.array([10.0, 20.0, 100.0, 50.0]),
        frame,
        module.ReplayOutputProjectionConfig(
            mode="size-only",
            size_deadband_ratio=0.05,
        ),
    )

    np.testing.assert_allclose(projected, np.array([10.0, 15.0, 100.0, 60.0]))


def test_acceptance_replay_center_deadband_filters_small_center_shifts():
    module = _load_module()
    frame = _diagnostic_frame([12.0, 21.0, 100.0, 50.0])

    projected = module.frame_projected_output_xywh(
        np.array([10.0, 20.0, 100.0, 50.0]),
        frame,
        module.ReplayOutputProjectionConfig(
            mode="box",
            center_deadband_ratio=0.03,
        ),
    )

    np.testing.assert_allclose(projected, np.array([10.0, 20.0, 100.0, 50.0]))


def test_acceptance_replay_center_clamp_caps_center_shift():
    module = _load_module()
    frame = _diagnostic_frame([10.0, 0.0, 3.0, 4.0])

    projected = module.frame_projected_output_xywh(
        np.array([0.0, 0.0, 3.0, 4.0]),
        frame,
        module.ReplayOutputProjectionConfig(
            mode="box",
            center_clamp_ratio=1.0,
        ),
    )

    np.testing.assert_allclose(projected, np.array([5.0, 0.0, 3.0, 4.0]))


def test_acceptance_replay_size_clamp_caps_size_change():
    module = _load_module()
    frame = _diagnostic_frame([0.0, 5.0, 140.0, 80.0])

    projected = module.frame_projected_output_xywh(
        np.array([10.0, 20.0, 100.0, 50.0]),
        frame,
        module.ReplayOutputProjectionConfig(
            mode="size-only",
            size_clamp_ratio=0.20,
        ),
    )

    np.testing.assert_allclose(projected, np.array([0.0, 15.0, 120.0, 60.0]))


def test_acceptance_replay_temporal_gates_reject_output_shocks():
    module = _load_module()
    frame = _diagnostic_frame([30.0, 10.0, 40.0, 20.0])
    candidate = np.array([30.0, 10.0, 40.0, 20.0])

    def decide(previous_output, *, center_gate, size_gate):
        return module.evaluate_frame_acceptance(
            candidate,
            frame,
            module.ReplayAcceptanceConfig(
                min_candidate_iou=0.0,
                max_center_shift_ratio=None,
                max_temporal_center_shift_ratio=center_gate,
                max_temporal_size_change_ratio=size_gate,
            ),
            previous_output_xywh=np.asarray(previous_output, dtype=float),
        )

    jump_decision = decide([0.0, 10.0, 20.0, 20.0], center_gate=0.50, size_gate=2.0)
    size_decision = decide([30.0, 10.0, 20.0, 20.0], center_gate=2.0, size_gate=0.50)

    assert not jump_decision.accepted
    assert jump_decision.rejection_reasons == ("temporal_center_shift_ratio",)
    assert jump_decision.temporal_center_shift_ratio > 0.50
    assert not size_decision.accepted
    assert size_decision.rejection_reasons == ("temporal_size_change_ratio",)
    assert size_decision.temporal_size_change_ratio == 1.0


def test_acceptance_replay_motion_prediction_gate_rejects_inconsistent_update():
    module = _load_module()
    frame = {
        "frame_index": 1,
        "fallback_reason": None,
        "refiner_output_xywh": [40.0, 10.0, 20.0, 20.0],
        "refined_bbox": {
            "x_min": 40.0,
            "y_min": 10.0,
            "x_max": 60.0,
            "y_max": 30.0,
        },
        "used_event_count": 32,
        "active_measurement_count": 16,
        "mean_event_activity": 0.8,
    }

    decision = module.evaluate_frame_acceptance(
        np.array([10.0, 10.0, 20.0, 20.0]),
        frame,
        module.ReplayAcceptanceConfig(
            min_candidate_iou=0.0,
            max_center_shift_ratio=None,
            max_motion_prediction_error_ratio=0.50,
        ),
        previous_candidate_xywh=np.array([0.0, 10.0, 20.0, 20.0]),
        previous_output_xywh=np.array([0.0, 10.0, 20.0, 20.0]),
    )

    assert not decision.accepted
    assert decision.rejection_reasons == ("motion_prediction_error_ratio",)
    assert decision.motion_prediction_error_ratio > 0.50


def test_acceptance_replay_can_hold_rejected_center_correction():
    module = _load_module()
    frames = [
        {
            "frame_index": 0,
            "fallback_reason": "initial_frame",
            "refiner_output_xywh": [8.0, 8.0, 10.0, 10.0],
        },
        _diagnostic_frame([11.0, 8.0, 10.0, 10.0], frame_index=1),
        _diagnostic_frame(
            [99.0, 99.0, 10.0, 10.0],
            frame_index=2,
            mean_event_activity=0.0,
            used_event_count=0,
            active_measurement_count=0,
        ),
    ]

    replayed, counts, decisions = _replay_center_hold_sequence(module, frames)

    np.testing.assert_allclose(
        replayed,
        np.array(
            [
                [8.0, 8.0, 10.0, 10.0],
                [11.0, 8.0, 10.0, 10.0],
                [12.0, 8.0, 10.0, 10.0],
            ]
        ),
    )
    assert counts["accepted"] == 1
    assert counts["held_rejected_center"] == 1
    assert decisions[1]["held_rejected_center_correction"]
    assert decisions[1]["rejected_center_hold_age"] == 1


def test_acceptance_replay_support_gate_blocks_center_hold():
    module = _load_module()
    frames = [
        {"frame_index": 0},
        _diagnostic_frame(
            [11.0, 8.0, 10.0, 10.0],
            frame_index=1,
            mean_event_activity=1.0,
            used_event_count=64,
            active_measurement_count=64,
            polarity_consistency_fraction=1.0,
        ),
        _diagnostic_frame(
            [99.0, 99.0, 10.0, 10.0],
            frame_index=2,
            mean_event_activity=1.0,
            used_event_count=64,
            active_measurement_count=64,
            polarity_consistency_fraction=1.0,
        ),
    ]

    replayed, counts, decisions = _replay_center_hold_sequence(
        module,
        frames,
        max_rejected_center_hold_support_score=0.25,
    )

    np.testing.assert_allclose(replayed[2], np.array([10.0, 8.0, 10.0, 10.0]))
    assert counts["accepted"] == 1
    assert counts["held_rejected_center"] == 0
    assert not decisions[1]["held_rejected_center_correction"]
    assert decisions[1]["event_support_score"] == pytest.approx(1.0)


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
    _write_replay_diagnostics(
        diagnostics_json,
        base_result_file,
        [
            {
                "frame_index": 0,
                "fallback_reason": "initial_frame",
                "refiner_output_xywh": [10.0, 10.0, 20.0, 20.0],
            },
            _diagnostic_frame([11.5, 10.0, 20.0, 20.0], frame_index=1),
            {
                **_diagnostic_frame([12.2, 10.0, 20.0, 20.0], frame_index=2),
                "refined_bbox": {
                    "x_min": 80.0,
                    "y_min": 80.0,
                    "x_max": 100.0,
                    "y_max": 100.0,
                },
            },
        ],
        acceptance_config={
            "min_used_event_count": 10,
            "min_active_measurement_count": 3,
            "min_mean_event_activity": 0.1,
            "min_candidate_iou": 0.6,
            "min_candidate_area_ratio": 0.5,
            "max_candidate_area_ratio": 1.5,
            "max_center_shift_ratio": 0.25,
        },
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
    projected_frame = _diagnostic_frame([12.0, 10.0, 20.0, 20.0])
    projected_frame["refined_bbox"] = {
        "x_min": 21.0,
        "y_min": 10.0,
        "x_max": 61.0,
        "y_max": 30.0,
    }
    _write_replay_diagnostics(
        diagnostics_json,
        base_result_file,
        [
            {
                "frame_index": 0,
                "fallback_reason": "initial_frame",
                "refiner_output_xywh": [10.0, 10.0, 20.0, 20.0],
            },
            projected_frame,
        ],
        acceptance_config={
            "min_used_event_count": 10,
            "min_active_measurement_count": 3,
            "min_mean_event_activity": 0.1,
            "min_candidate_iou": 0.0,
            "min_candidate_area_ratio": 0.0,
            "max_candidate_area_ratio": 10.0,
            "max_center_shift_ratio": 10.0,
        },
        refiner_config={"image_width": 100, "image_height": 100},
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
        "center_smoothing": None,
        "motion_smoothing": None,
        "center_clamp_ratio": None,
        "center_deadband_ratio": None,
        "size_clamp_ratio": None,
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
    assert "--min-event-support-score" in help_text
    assert "--max-temporal-center-shift-ratio" in help_text
    assert "--max-temporal-size-change-ratio" in help_text
    assert "--max-motion-prediction-error-ratio" in help_text
    assert "--max-rejected-center-hold-frames" in help_text
    assert "--rejected-center-hold-decay" in help_text
    assert "--max-rejected-center-hold-support-score" in help_text
    assert "--replay-output-mode" in help_text
    assert "--replay-output-blend" in help_text
    assert "--replay-output-size-smoothing" in help_text
    assert "--replay-output-center-smoothing" in help_text
    assert "--replay-output-motion-smoothing" in help_text
    assert "--replay-output-center-clamp-ratio" in help_text
    assert "--replay-output-center-deadband-ratio" in help_text
    assert "--replay-output-size-clamp-ratio" in help_text
    assert "--replay-output-size-deadband-ratio" in help_text
    assert "--replay-output-confidence-field" in help_text
