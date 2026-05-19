"""Multi-hypothesis EventVOT refinement runner."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from dvs_enact import DVSContourRefiner
from eventvot_multihypothesis_scoring import (  # pylint: disable=import-error
    EventVOTHypothesis,
    EventVOTMultiHypothesisConfig,
    build_candidate_refiner_configs,
    default_multihypothesis_config,
    evaluate_frame_hypotheses,
    select_frame_hypothesis,
)
from run_eventvot_refinement import (  # pylint: disable=import-error
    EventVOTAcceptanceConfig,
    EventVOTRefinementOptions,
    _save_timing_file,
    _validate_sequence_frame_count,
    bbox_dict_to_xywh,
    default_diagnostics_path,
    default_eventvot_refiner,
    existing_output_result_is_complete,
    find_sequence_event_csv,
    iter_eventvot_frame_windows,
    load_xywh_result_file,
    register_tracker_in_config,
    resolve_base_result_file,
    resolve_eventvot_split_root,
    resolve_output_result_file,
    resolve_output_results_root,
    resolve_sequence_names,
    save_xywh_result_file,
    select_sequence_chunk,
    summarize_skipped_sequence,
    summarize_sequence_results,
    xywh_to_diagnostic_bbox,
)


def run_multihypothesis(
    options: EventVOTRefinementOptions,
    *,
    refiner: DVSContourRefiner | None = None,
    multi_hypothesis_config: EventVOTMultiHypothesisConfig | None = None,
) -> dict[str, Any]:
    """Run multi-hypothesis refinement over one or more EventVOT result files."""
    refiner = refiner or default_eventvot_refiner()
    multi_hypothesis_config = (
        multi_hypothesis_config or default_multihypothesis_config()
    )
    split_root = resolve_eventvot_split_root(options.eventvot_root, options.split)
    output_root = resolve_output_results_root(
        options.output_results,
        tracker_name=options.tracker_name,
    )
    names = resolve_sequence_names(
        split_root,
        options.base_results,
        requested_sequences=options.sequences,
    )
    names = select_sequence_chunk(
        names,
        sequence_index=options.sequence_index,
        sequence_count=options.sequence_count,
    )
    summaries = [
        refine_sequence_multihypothesis(
            name,
            split_root / name,
            resolve_base_result_file(options.base_results, name),
            resolve_output_result_file(options.base_results, output_root, name),
            refiner,
            event_column_order=options.event_column_order,
            acceptance_config=options.acceptance_config,
            multi_hypothesis_config=multi_hypothesis_config,
            skip_existing=options.skip_existing,
        )
        for name in names
    ]

    config_tracker_updated = False
    if options.config_tracker_path is not None:
        if options.tracker_name is None:
            raise ValueError("--config-tracker requires --tracker-name")
        config_tracker_updated = register_tracker_in_config(
            options.config_tracker_path,
            options.tracker_name,
        )

    payload = {
        "schema_version": 2,
        "description": "EventVOT multi-hypothesis DVS-ENACT refinement.",
        "options": {
            **asdict(options),
            "eventvot_root": str(options.eventvot_root),
            "base_results": str(options.base_results),
            "output_results": str(options.output_results),
            "resolved_output_results": str(output_root),
            "diagnostics_json": str(options.diagnostics_json)
            if options.diagnostics_json is not None
            else None,
            "config_tracker_path": str(options.config_tracker_path)
            if options.config_tracker_path is not None
            else None,
        },
        "eventvot_evaluator": {
            "tracker_name": options.tracker_name,
            "tracking_result_dir": str(output_root),
            "config_tracker_updated": config_tracker_updated,
        },
        "refiner_config": asdict(refiner.config),
        "acceptance_config": asdict(options.acceptance_config),
        "multi_hypothesis_config": asdict(multi_hypothesis_config),
        "candidate_refiner_count": len(
            build_candidate_refiner_configs(refiner.config, multi_hypothesis_config),
        ),
        "summary": summarize_sequence_results(summaries),
        "sequences": summaries,
    }
    diagnostics_path = options.diagnostics_json or default_diagnostics_path(output_root)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def refine_sequence_multihypothesis(
    sequence_name: str,
    sequence_dir: Path,
    base_result_file: Path,
    output_result_file: Path,
    refiner: DVSContourRefiner,
    *,
    event_column_order: str = "auto",
    acceptance_config: EventVOTAcceptanceConfig | None = None,
    multi_hypothesis_config: EventVOTMultiHypothesisConfig | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    acceptance_config = acceptance_config or EventVOTAcceptanceConfig()
    multi_hypothesis_config = (
        multi_hypothesis_config or default_multihypothesis_config()
    )
    base_boxes = load_xywh_result_file(base_result_file)
    frame_count = int(base_boxes.shape[0])
    _validate_sequence_frame_count(sequence_dir, sequence_name, frame_count)
    if skip_existing:
        complete, output_boxes = existing_output_result_is_complete(
            output_result_file,
            base_boxes,
        )
        if complete and output_boxes is not None:
            return summarize_skipped_sequence(
                sequence_name,
                sequence_dir,
                None,
                base_result_file,
                output_result_file,
                base_boxes,
                output_boxes,
            )

    event_csv = find_sequence_event_csv(sequence_dir, sequence_name)
    configs = build_candidate_refiner_configs(refiner.config, multi_hypothesis_config)
    refiners = [
        refiner if config == refiner.config else DVSContourRefiner(config)
        for config in configs
    ]
    refined_boxes = np.array(base_boxes, dtype=float, copy=True)
    timings = np.zeros(frame_count, dtype=float)
    frames: list[dict[str, Any]] = [_initial_frame_record(base_boxes[0])]

    for frame_index, event_window in iter_eventvot_frame_windows(
        event_csv,
        frame_count,
        event_column_order=event_column_order,
    ):
        hypotheses = evaluate_frame_hypotheses(
            base_boxes[frame_index],
            base_boxes[frame_index - 1],
            event_window,
            refiners,
            acceptance_config,
            previous_output_xywh=refined_boxes[frame_index - 1],
        )
        selected = select_frame_hypothesis(hypotheses)
        timings[frame_index] = sum(
            hypothesis.elapsed_seconds for hypothesis in hypotheses
        )
        refiner_output = np.asarray(selected.result.as_xywh(), dtype=float)
        refined_boxes[frame_index] = (
            refiner_output
            if selected.decision.accepted
            else np.asarray(base_boxes[frame_index], dtype=float)
        )
        frames.append(
            _frame_record(
                frame_index,
                refined_boxes[frame_index],
                selected,
                hypotheses,
                timings[frame_index],
            ),
        )

    save_xywh_result_file(output_result_file, refined_boxes)
    _save_timing_file(output_result_file, timings)
    fallback_counts = Counter(
        "refined" if frame["fallback_reason"] is None else frame["fallback_reason"]
        for frame in frames
    )
    acceptance_counts = Counter(
        "accepted" if frame["accept_refinement"] else frame["rejection_reasons"][0]
        for frame in frames
    )
    selected_counts = Counter(
        int(frame["selected_hypothesis_index"]) for frame in frames[1:]
    )
    used_event_counts = [int(frame["used_event_count"]) for frame in frames[1:]]
    accepted_count = sum(1 for frame in frames if frame["accept_refinement"])
    return {
        "sequence": sequence_name,
        "sequence_dir": str(sequence_dir),
        "event_csv": str(event_csv),
        "base_result_file": str(base_result_file),
        "output_result_file": str(output_result_file),
        "frame_count": frame_count,
        "refined_frame_count": int(accepted_count),
        "accepted_refinement_count": int(accepted_count),
        "held_rejected_center_count": 0,
        "refiner_success_frame_count": sum(
            1 for frame in frames if frame["fallback_reason"] is None
        ),
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "acceptance_counts": dict(sorted(acceptance_counts.items())),
        "selected_hypothesis_counts": dict(sorted(selected_counts.items())),
        "candidate_refiner_count": len(refiners),
        "mean_used_event_count": float(np.mean(used_event_counts))
        if used_event_counts
        else 0.0,
        "total_refinement_seconds": float(np.sum(timings)),
        "frames": frames,
    }


def _initial_frame_record(box_xywh: np.ndarray) -> dict[str, Any]:
    box = np.asarray(box_xywh, dtype=float)
    return {
        "frame_index": 0,
        "fallback_reason": "initial_frame",
        "candidate_bbox": xywh_to_diagnostic_bbox(box),
        "output_bbox": xywh_to_diagnostic_bbox(box),
        "refiner_output_bbox": xywh_to_diagnostic_bbox(box),
        "output_xywh": box.tolist(),
        "refiner_output_xywh": box.tolist(),
        "event_count": 0,
        "used_event_count": 0,
        "active_measurement_count": 0,
        "accept_refinement": False,
        "rejection_reasons": ["initial_frame"],
        "candidate_iou": 1.0,
        "candidate_area_ratio": 1.0,
        "center_shift_ratio": 0.0,
        "raw_candidate_iou": 1.0,
        "raw_candidate_area_ratio": 1.0,
        "raw_center_shift_ratio": 0.0,
        "temporal_center_shift_ratio": None,
        "temporal_size_change_ratio": None,
        "motion_prediction_error_ratio": None,
        "held_rejected_center_correction": False,
        "rejected_center_hold_age": 0,
        "rejected_center_hold_decay": 1.0,
        "active_fraction": None,
        "quadratic_form_per_active_measurement": None,
        "selected_hypothesis_index": -1,
        "selected_hypothesis_score": None,
        "hypothesis_count": 0,
        "hypotheses": [],
    }


def _frame_record(
    frame_index: int,
    output_xywh: np.ndarray,
    selected: EventVOTHypothesis,
    hypotheses: list[EventVOTHypothesis],
    elapsed_seconds: float,
) -> dict[str, Any]:
    frame_record = selected.result.to_dict()
    refiner_output = np.asarray(selected.result.as_xywh(), dtype=float)
    frame_record.update(
        {
            "frame_index": int(frame_index),
            "accept_refinement": selected.decision.accepted,
            "rejection_reasons": list(selected.decision.rejection_reasons),
            "candidate_iou": float(selected.decision.candidate_iou),
            "candidate_area_ratio": float(selected.decision.candidate_area_ratio),
            "center_shift_ratio": float(selected.decision.center_shift_ratio),
            "raw_candidate_iou": float(selected.decision.raw_candidate_iou),
            "raw_candidate_area_ratio": float(
                selected.decision.raw_candidate_area_ratio,
            ),
            "raw_center_shift_ratio": float(selected.decision.raw_center_shift_ratio),
            "temporal_center_shift_ratio": (
                selected.decision.temporal_center_shift_ratio
            ),
            "temporal_size_change_ratio": (
                selected.decision.temporal_size_change_ratio
            ),
            "motion_prediction_error_ratio": (
                selected.decision.motion_prediction_error_ratio
            ),
            "active_fraction": selected.decision.active_fraction,
            "quadratic_form_per_active_measurement": (
                selected.decision.quadratic_form_per_active_measurement
            ),
            "refiner_output_bbox": frame_record.get("output_bbox"),
            "refiner_output_xywh": refiner_output.astype(float).tolist(),
            "output_bbox": xywh_to_diagnostic_bbox(output_xywh),
            "output_xywh": np.asarray(output_xywh, dtype=float).tolist(),
            "elapsed_seconds": float(elapsed_seconds),
            "held_rejected_center_correction": False,
            "rejected_center_hold_age": 0,
            "rejected_center_hold_decay": 1.0,
            "selected_hypothesis_index": int(selected.index),
            "selected_hypothesis_score": float(selected.score),
            "hypothesis_count": len(hypotheses),
            "hypotheses": [
                _serialize_hypothesis(hypothesis) for hypothesis in hypotheses
            ],
        },
    )
    return frame_record


def _serialize_hypothesis(hypothesis: EventVOTHypothesis) -> dict[str, Any]:
    return {
        "index": int(hypothesis.index),
        "accepted": bool(hypothesis.decision.accepted),
        "rejection_reasons": list(hypothesis.decision.rejection_reasons),
        "score": float(hypothesis.score),
        "elapsed_seconds": float(hypothesis.elapsed_seconds),
        "output_xywh": np.asarray(hypothesis.result.as_xywh(), dtype=float).tolist(),
        "raw_refined_xywh": bbox_dict_to_xywh(hypothesis.result.refined_bbox)
        .astype(float)
        .tolist(),
        "used_event_count": int(hypothesis.result.used_event_count),
        "active_measurement_count": int(hypothesis.result.active_measurement_count),
        "mean_event_activity": hypothesis.result.mean_event_activity,
        "polarity_consistency_fraction": (
            hypothesis.result.polarity_consistency_fraction
        ),
        "fallback_reason": hypothesis.result.fallback_reason,
        "candidate_iou": float(hypothesis.decision.candidate_iou),
        "candidate_area_ratio": float(hypothesis.decision.candidate_area_ratio),
        "center_shift_ratio": float(hypothesis.decision.center_shift_ratio),
        "temporal_center_shift_ratio": hypothesis.decision.temporal_center_shift_ratio,
        "temporal_size_change_ratio": hypothesis.decision.temporal_size_change_ratio,
        "motion_prediction_error_ratio": (
            hypothesis.decision.motion_prediction_error_ratio
        ),
        "refiner_config": _compact_config(hypothesis.refiner_config),
    }


def _compact_config(config: Any) -> dict[str, Any]:
    return {
        "refinement_blend": float(config.refinement_blend),
        "search_expansion_factor": float(config.search_expansion_factor),
        "max_events": config.max_events,
        "measurement_noise_variance": float(config.measurement_noise_variance),
        "event_activity_floor": float(config.event_activity_floor),
        "inactive_activity_threshold": float(config.inactive_activity_threshold),
        "use_event_polarity": bool(config.use_event_polarity),
    }
