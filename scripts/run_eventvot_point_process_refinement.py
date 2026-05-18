"""Refine EventVOT results with a point-process likelihood acceptance gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_eventvot_refinement import (  # noqa: E402
    EventVOTAcceptanceConfig,
    EventVOTRefinementOptions,
    _acceptance_config_from_args,
    _add_refiner_arguments,
    _refiner_from_args,
    _resolve_cli_config_tracker_path,
    _resolve_cli_output_results,
    _save_timing_file,
    _validate_sequence_frame_count,
    bbox_dict_to_xywh,
    default_diagnostics_path,
    default_eventvot_refiner,
    evaluate_refinement_acceptance,
    existing_output_result_is_complete,
    find_sequence_event_csv,
    iter_eventvot_frame_windows,
    load_requested_sequence_names,
    load_xywh_result_file,
    register_tracker_in_config,
    resolve_base_result_file,
    resolve_eventvot_split_root,
    resolve_output_result_file,
    resolve_output_results_root,
    resolve_sequence_names,
    save_xywh_result_file,
    select_sequence_chunk,
    summarize_sequence_results,
    summarize_skipped_sequence,
    xywh_to_diagnostic_bbox,
)
from dvs_enact import DVSContourRefiner, EventLikelihoodConfig  # noqa: E402
from dvs_enact.refinement_likelihood import (  # noqa: E402
    BBoxEventLikelihoodConfig,
    RefinementLikelihoodComparison,
    compare_refinement_likelihood,
)
from dvs_enact.refiner import crop_events_to_bbox  # noqa: E402


@dataclass(frozen=True)
class EventVOTPointProcessGateConfig:
    """Point-process likelihood gate for EventVOT post-hoc refinements."""

    enabled: bool = True
    min_delta_log_likelihood: float = 0.0
    min_delta_log_likelihood_per_event: float | None = None
    samples_per_edge: int = 24
    spatial_sigma_px: float = 2.0
    foreground_rate: float = 1.0
    background_rate: float = 1e-4
    activity_floor: float = 0.05
    min_intensity: float = 1e-12
    include_expected_count: bool = True
    normalize_kernel: bool = True

    def __post_init__(self) -> None:
        if self.samples_per_edge <= 0:
            raise ValueError("samples_per_edge must be positive")
        EventLikelihoodConfig(
            spatial_sigma_px=self.spatial_sigma_px,
            foreground_rate=self.foreground_rate,
            background_rate=self.background_rate,
            activity_floor=self.activity_floor,
            min_intensity=self.min_intensity,
        )

    def likelihood_config(self) -> BBoxEventLikelihoodConfig:
        """Return the box-likelihood configuration used by the gate."""
        return BBoxEventLikelihoodConfig(
            likelihood=EventLikelihoodConfig(
                spatial_sigma_px=self.spatial_sigma_px,
                foreground_rate=self.foreground_rate,
                background_rate=self.background_rate,
                activity_floor=self.activity_floor,
                min_intensity=self.min_intensity,
                include_expected_count=self.include_expected_count,
                normalize_kernel=self.normalize_kernel,
            ),
            samples_per_edge=self.samples_per_edge,
        )


@dataclass(frozen=True)
class EventVOTPointProcessRefinementOptions(EventVOTRefinementOptions):
    """EventVOT refinement options with a point-process likelihood gate."""

    point_process_gate: EventVOTPointProcessGateConfig = field(
        default_factory=EventVOTPointProcessGateConfig
    )


def refine_sequence(
    sequence_name: str,
    sequence_dir: Path,
    base_result_file: Path,
    output_result_file: Path,
    refiner: DVSContourRefiner,
    *,
    event_column_order: str = "auto",
    acceptance_config: EventVOTAcceptanceConfig | None = None,
    point_process_gate: EventVOTPointProcessGateConfig | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Refine one sequence and accept only likelihood-improving boxes."""
    acceptance_config = acceptance_config or EventVOTAcceptanceConfig()
    point_process_gate = point_process_gate or EventVOTPointProcessGateConfig()
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
                base_result_file,
                output_result_file,
                base_boxes,
                output_boxes,
            )

    event_csv = find_sequence_event_csv(sequence_dir, sequence_name)
    refined_boxes = np.array(base_boxes, dtype=float, copy=True)
    timings = np.zeros(frame_count, dtype=float)
    frames: list[dict[str, Any]] = [_initial_frame_record(base_boxes, refined_boxes)]

    for frame_index, event_window in iter_eventvot_frame_windows(
        event_csv,
        frame_count,
        event_column_order=event_column_order,
    ):
        started = time.perf_counter()
        result = refiner.refine(
            base_boxes[frame_index],
            event_window,
            previous_candidate_bbox=base_boxes[frame_index - 1],
        )
        timings[frame_index] = time.perf_counter() - started
        refiner_output = np.asarray(result.as_xywh(), dtype=float)
        geometry_decision = evaluate_refinement_acceptance(
            base_boxes[frame_index],
            result,
            acceptance_config,
        )
        likelihood_comparison = _score_point_process_refinement(
            base_boxes[frame_index],
            refiner_output,
            event_window,
            result,
            point_process_gate,
        )
        point_process_reasons = _point_process_rejection_reasons(
            likelihood_comparison,
            point_process_gate,
        )
        rejection_reasons = list(geometry_decision.rejection_reasons)
        rejection_reasons.extend(point_process_reasons)
        accept_refinement = geometry_decision.accepted and not point_process_reasons
        refined_boxes[frame_index] = (
            refiner_output
            if accept_refinement
            else np.asarray(base_boxes[frame_index], dtype=float)
        )
        frames.append(
            _frame_record(
                frame_index,
                result,
                refiner_output,
                refined_boxes[frame_index],
                geometry_decision,
                rejection_reasons,
                accept_refinement,
                timings[frame_index],
                likelihood_comparison,
            )
        )

    save_xywh_result_file(output_result_file, refined_boxes)
    _save_timing_file(output_result_file, timings)
    return _sequence_summary(
        sequence_name,
        sequence_dir,
        event_csv,
        base_result_file,
        output_result_file,
        frame_count,
        frames,
        timings,
    )


def run(
    options: EventVOTPointProcessRefinementOptions,
    refiner: DVSContourRefiner | None = None,
) -> dict[str, Any]:
    """Run point-process-scored refinement over EventVOT result files."""
    refiner = refiner or default_eventvot_refiner()
    split_root = resolve_eventvot_split_root(options.eventvot_root, options.split)
    output_root = resolve_output_results_root(
        options.output_results,
        tracker_name=options.tracker_name,
    )
    sequence_names = resolve_sequence_names(
        split_root,
        options.base_results,
        requested_sequences=options.sequences,
    )
    sequence_names = select_sequence_chunk(
        sequence_names,
        sequence_index=options.sequence_index,
        sequence_count=options.sequence_count,
    )

    summaries = []
    for sequence_name in sequence_names:
        sequence_dir = split_root / sequence_name
        base_result_file = resolve_base_result_file(options.base_results, sequence_name)
        output_result_file = resolve_output_result_file(
            options.base_results,
            output_root,
            sequence_name,
        )
        summaries.append(
            refine_sequence(
                sequence_name,
                sequence_dir,
                base_result_file,
                output_result_file,
                refiner,
                event_column_order=options.event_column_order,
                acceptance_config=options.acceptance_config,
                point_process_gate=options.point_process_gate,
                skip_existing=options.skip_existing,
            )
        )

    config_tracker_updated = False
    if options.config_tracker_path is not None:
        if options.tracker_name is None:
            raise ValueError("--config-tracker requires --tracker-name")
        config_tracker_updated = register_tracker_in_config(
            options.config_tracker_path,
            options.tracker_name,
        )

    payload = {
        "schema_version": 1,
        "description": (
            "Post-hoc EventVOT result refinement with an additional "
            "contour-conditioned point-process likelihood acceptance gate."
        ),
        "options": _json_ready_options(options, output_root),
        "eventvot_evaluator": {
            "tracker_name": options.tracker_name,
            "tracking_result_dir": str(output_root),
            "config_tracker_updated": config_tracker_updated,
        },
        "refiner_config": asdict(refiner.config),
        "acceptance_config": asdict(options.acceptance_config),
        "point_process_gate": asdict(options.point_process_gate),
        "summary": summarize_sequence_results(summaries),
        "sequences": summaries,
    }
    diagnostics_path = options.diagnostics_json or default_diagnostics_path(output_root)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refine EventVOT xywh tracker results with DVS-ENACT and a "
            "point-process likelihood gate."
        ),
    )
    parser.add_argument("--eventvot-root", type=Path, required=True)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--output-results", type=Path)
    parser.add_argument("--eventvot-toolkit-root", type=Path)
    parser.add_argument("--tracker-name")
    parser.add_argument("--config-tracker", type=Path)
    parser.add_argument("--update-config-tracker", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--sequence-list", action="append", default=[])
    parser.add_argument("--sequence-file", type=Path, action="append", default=[])
    parser.add_argument("--sequence-index", type=int)
    parser.add_argument("--sequence-count", type=int)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument(
        "--event-column-order",
        default="auto",
        choices=("auto", "xypt", "txyp", "xytp", "yxpt", "yxpt5", "yxt"),
    )
    parser.add_argument("--diagnostics-json", type=Path)
    _add_refiner_arguments(parser)
    _add_point_process_arguments(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(
        _options_from_args(args),
        refiner=_refiner_from_args(args),
    )
    print(json.dumps(payload["summary"], indent=2))
    return 0


def _initial_frame_record(
    base_boxes: np.ndarray,
    refined_boxes: np.ndarray,
) -> dict[str, Any]:
    return {
        "frame_index": 0,
        "fallback_reason": "initial_frame",
        "candidate_bbox": xywh_to_diagnostic_bbox(base_boxes[0]),
        "output_bbox": xywh_to_diagnostic_bbox(refined_boxes[0]),
        "refiner_output_bbox": xywh_to_diagnostic_bbox(refined_boxes[0]),
        "output_xywh": refined_boxes[0].astype(float).tolist(),
        "refiner_output_xywh": refined_boxes[0].astype(float).tolist(),
        "event_count": 0,
        "used_event_count": 0,
        "active_measurement_count": 0,
        "accept_refinement": False,
        "rejection_reasons": ["initial_frame"],
        "point_process_likelihood": None,
    }


def _frame_record(
    frame_index: int,
    result: Any,
    refiner_output: np.ndarray,
    output_box: np.ndarray,
    geometry_decision: Any,
    rejection_reasons: list[str],
    accept_refinement: bool,
    elapsed_seconds: float,
    likelihood_comparison: RefinementLikelihoodComparison | None,
) -> dict[str, Any]:
    frame_record = result.to_dict()
    refiner_output_bbox = frame_record.get("output_bbox")
    frame_record.update(
        {
            "frame_index": int(frame_index),
            "accept_refinement": bool(accept_refinement),
            "rejection_reasons": rejection_reasons,
            "candidate_iou": float(geometry_decision.candidate_iou),
            "candidate_area_ratio": float(geometry_decision.candidate_area_ratio),
            "center_shift_ratio": float(geometry_decision.center_shift_ratio),
            "raw_candidate_iou": float(geometry_decision.raw_candidate_iou),
            "raw_candidate_area_ratio": float(geometry_decision.raw_candidate_area_ratio),
            "raw_center_shift_ratio": float(geometry_decision.raw_center_shift_ratio),
            "active_fraction": geometry_decision.active_fraction,
            "quadratic_form_per_active_measurement": (
                geometry_decision.quadratic_form_per_active_measurement
            ),
            "refiner_output_bbox": refiner_output_bbox,
            "refiner_output_xywh": refiner_output.astype(float).tolist(),
            "output_bbox": xywh_to_diagnostic_bbox(output_box),
            "output_xywh": output_box.astype(float).tolist(),
            "elapsed_seconds": float(elapsed_seconds),
            "point_process_likelihood": (
                None if likelihood_comparison is None else likelihood_comparison.to_dict()
            ),
        }
    )
    return frame_record


def _sequence_summary(
    sequence_name: str,
    sequence_dir: Path,
    event_csv: Path,
    base_result_file: Path,
    output_result_file: Path,
    frame_count: int,
    frames: list[dict[str, Any]],
    timings: np.ndarray,
) -> dict[str, Any]:
    fallback_counts = Counter(
        "refined" if frame["fallback_reason"] is None else frame["fallback_reason"]
        for frame in frames
    )
    acceptance_counts = Counter(
        "accepted" if frame["accept_refinement"] else frame["rejection_reasons"][0]
        for frame in frames
    )
    refiner_success_frame_count = sum(
        1 for frame in frames if frame["fallback_reason"] is None
    )
    accepted_refinement_count = sum(1 for frame in frames if frame["accept_refinement"])
    used_event_counts = [int(frame["used_event_count"]) for frame in frames[1:]]
    return {
        "sequence": sequence_name,
        "sequence_dir": str(sequence_dir),
        "event_csv": str(event_csv),
        "base_result_file": str(base_result_file),
        "output_result_file": str(output_result_file),
        "frame_count": frame_count,
        "refined_frame_count": int(accepted_refinement_count),
        "accepted_refinement_count": int(accepted_refinement_count),
        "refiner_success_frame_count": int(refiner_success_frame_count),
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "acceptance_counts": dict(sorted(acceptance_counts.items())),
        "mean_used_event_count": float(np.mean(used_event_counts))
        if used_event_counts
        else 0.0,
        "total_refinement_seconds": float(np.sum(timings)),
        "frames": frames,
    }


def _score_point_process_refinement(
    base_xywh: np.ndarray,
    refined_xywh: np.ndarray,
    event_window: Any,
    result: Any,
    gate: EventVOTPointProcessGateConfig,
) -> RefinementLikelihoodComparison | None:
    if not gate.enabled:
        return None
    score_events = crop_events_to_bbox(event_window, result.search_bbox)
    return compare_refinement_likelihood(
        base_xywh,
        refined_xywh,
        score_events,
        result.event_velocity,
        gate.likelihood_config(),
        bbox_format="xywh",
        image_area=_bbox_area(result.search_bbox),
    )


def _point_process_rejection_reasons(
    comparison: RefinementLikelihoodComparison | None,
    gate: EventVOTPointProcessGateConfig,
) -> list[str]:
    if not gate.enabled or comparison is None:
        return []
    reasons: list[str] = []
    if comparison.delta_log_likelihood < gate.min_delta_log_likelihood:
        reasons.append("point_process_log_likelihood")
    if gate.min_delta_log_likelihood_per_event is not None:
        per_event = comparison.delta_log_likelihood_per_event
        if per_event is None or per_event < gate.min_delta_log_likelihood_per_event:
            reasons.append("point_process_log_likelihood_per_event")
    return reasons


def _bbox_area(bbox: dict[str, Any]) -> float:
    xywh = bbox_dict_to_xywh(bbox)
    return float(max(0.0, xywh[2]) * max(0.0, xywh[3]))


def _json_ready_options(
    options: EventVOTPointProcessRefinementOptions,
    output_root: Path,
) -> dict[str, Any]:
    return {
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
    }


def _add_point_process_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--disable-point-process-gate",
        action="store_true",
        help="Disable the point-process likelihood gate.",
    )
    parser.add_argument("--point-process-min-delta-log-likelihood", type=float, default=0.0)
    parser.add_argument("--point-process-min-delta-log-likelihood-per-event", type=float)
    parser.add_argument("--point-process-samples-per-edge", type=int, default=24)
    parser.add_argument("--point-process-spatial-sigma-px", type=float, default=2.0)
    parser.add_argument("--point-process-foreground-rate", type=float, default=1.0)
    parser.add_argument("--point-process-background-rate", type=float, default=1e-4)
    parser.add_argument("--point-process-activity-floor", type=float, default=0.05)
    parser.add_argument("--point-process-min-intensity", type=float, default=1e-12)
    parser.add_argument("--point-process-disable-expected-count", action="store_true")
    parser.add_argument("--point-process-disable-kernel-normalization", action="store_true")


def _point_process_gate_from_args(
    args: argparse.Namespace,
) -> EventVOTPointProcessGateConfig:
    return EventVOTPointProcessGateConfig(
        enabled=not args.disable_point_process_gate,
        min_delta_log_likelihood=args.point_process_min_delta_log_likelihood,
        min_delta_log_likelihood_per_event=(
            args.point_process_min_delta_log_likelihood_per_event
        ),
        samples_per_edge=args.point_process_samples_per_edge,
        spatial_sigma_px=args.point_process_spatial_sigma_px,
        foreground_rate=args.point_process_foreground_rate,
        background_rate=args.point_process_background_rate,
        activity_floor=args.point_process_activity_floor,
        min_intensity=args.point_process_min_intensity,
        include_expected_count=not args.point_process_disable_expected_count,
        normalize_kernel=not args.point_process_disable_kernel_normalization,
    )


def _options_from_args(args: argparse.Namespace) -> EventVOTPointProcessRefinementOptions:
    sequence_names = load_requested_sequence_names(
        args.sequence,
        args.sequence_list,
        args.sequence_file,
    )
    return EventVOTPointProcessRefinementOptions(
        **_common_option_kwargs(args, sequence_names),
        point_process_gate=_point_process_gate_from_args(args),
    )


def _common_option_kwargs(
    args: argparse.Namespace,
    sequence_names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "eventvot_root": args.eventvot_root,
        "base_results": args.base_results,
        "output_results": _resolve_cli_output_results(args),
        "split": args.split,
        "sequences": sequence_names,
        "sequence_index": args.sequence_index,
        "sequence_count": args.sequence_count,
        "tracker_name": args.tracker_name,
        "skip_existing": not args.no_skip_existing,
        "event_column_order": args.event_column_order,
        "diagnostics_json": args.diagnostics_json,
        "config_tracker_path": _resolve_cli_config_tracker_path(args),
        "acceptance_config": _acceptance_config_from_args(args),
    }


if __name__ == "__main__":
    raise SystemExit(main())
