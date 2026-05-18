"""Replay EventVOT DVS-ENACT acceptance policies from diagnostics.

The expensive part of an EventVOT DVS-ENACT run is computing candidate
refinements. This utility reuses the per-frame diagnostics JSON from one such
run and cheaply rewrites result files under a different accept/reject policy.

It is intended for validation-set policy tuning: add stricter confidence gates,
evaluate the replayed result, then lock one policy before touching the held-out
test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from functools import partial
from operator import gt, lt
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_eventvot_refinement import (  # noqa: E402
    area_ratio_xywh,
    box_iou_xywh,
    center_shift_ratio_xywh,
    load_xywh_result_file,
    resolve_base_result_file,
    resolve_eventvot_split_root,
    save_xywh_result_file,
    size_change_ratio_xywh,
)
from run_eventvot_refinement_modes import (  # noqa: E402
    PROJECTION_CONFIDENCE_FIELDS,
    REFINEMENT_MODES,
    project_refinement_output,
    validate_projection_confidence_weighting,
)
from run_eventvot_validation_sweep import evaluate_eventvot_results  # noqa: E402

REPLAY_OUTPUT_MODES = ("diagnostic", *REFINEMENT_MODES)


@dataclass(frozen=True)
class ReplayAcceptanceConfig:
    """Acceptance policy for replaying DVS-ENACT EventVOT diagnostics."""

    enabled: bool = True
    min_used_event_count: int | None = 10
    min_active_measurement_count: int | None = 3
    min_mean_event_activity: float | None = 0.10
    min_candidate_iou: float | None = 0.60
    min_candidate_area_ratio: float | None = 0.50
    max_candidate_area_ratio: float | None = 1.50
    max_center_shift_ratio: float | None = 0.25

    # Raw-update gates inspect the unblended DVS box. They catch updates that
    # look safe only because the final EventVOT output was heavily blended with
    # the base tracker box.
    min_raw_candidate_iou: float | None = None
    min_raw_candidate_area_ratio: float | None = None
    max_raw_candidate_area_ratio: float | None = None
    max_raw_center_shift_ratio: float | None = None

    # DVS confidence gates use diagnostics already emitted by DVSContourRefiner.
    min_polarity_consistency_fraction: float | None = None
    min_mean_event_polarity_weight: float | None = None
    max_quadratic_form_per_active_measurement: float | None = None
    min_active_fraction: float | None = None
    max_temporal_center_shift_ratio: float | None = None
    max_temporal_size_change_ratio: float | None = None


@dataclass(frozen=True)
class ReplayOutputProjectionConfig:
    """How replay should turn stored diagnostics into output boxes.

    ``diagnostic`` preserves the previously emitted ``refiner_output_xywh``.
    Other modes rebuild the output from the stored raw DVS refinement, which is
    useful for cheap validation sweeps over smaller blends or projection modes
    without recomputing the expensive EventVOT refinements.
    """

    mode: str = "diagnostic"
    blend: float | None = None
    size_smoothing: float | None = None
    center_smoothing: float | None = None
    center_clamp_ratio: float | None = None
    center_deadband_ratio: float | None = None
    size_clamp_ratio: float | None = None
    size_deadband_ratio: float | None = None
    confidence_field: str | None = None
    confidence_floor: float | None = None
    confidence_ceiling: float | None = None
    image_width: float | None = None
    image_height: float | None = None


@dataclass(frozen=True)
class ReplayAcceptanceDecision:
    """Replay decision and diagnostics for one frame."""

    accepted: bool
    rejection_reasons: tuple[str, ...]
    candidate_iou: float
    candidate_area_ratio: float
    center_shift_ratio: float
    raw_candidate_iou: float
    raw_candidate_area_ratio: float
    raw_center_shift_ratio: float
    temporal_center_shift_ratio: float | None
    temporal_size_change_ratio: float | None
    active_fraction: float | None
    quadratic_form_per_active_measurement: float | None


@dataclass(frozen=True)
class EventVOTAcceptanceReplayOptions:
    """Inputs and outputs for one acceptance replay run."""

    diagnostics_json: Path
    output_results: Path
    eventvot_root: Path | None = None
    base_results: Path | None = None
    split: str | None = None
    sequences: tuple[str, ...] = ()
    summary_json: Path | None = None
    decisions_csv: Path | None = None
    skip_evaluation: bool = False
    acceptance_config: ReplayAcceptanceConfig | None = None
    output_projection_config: ReplayOutputProjectionConfig = field(
        default_factory=ReplayOutputProjectionConfig
    )
    config_overrides: dict[str, Any] = field(default_factory=dict)


def main() -> int:
    args = build_parser().parse_args()
    payload = run(options_from_args(args))
    print(json.dumps(payload["summary"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay EventVOT DVS-ENACT accept/reject decisions from an existing "
            "diagnostics JSON without recomputing refinements."
        )
    )
    parser.add_argument("--diagnostics-json", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--eventvot-root", type=Path)
    parser.add_argument("--base-results", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--decisions-csv", type=Path)
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only rewrite result files; do not compute EventVOT metrics.",
    )
    parser.add_argument(
        "--disable-conservative-gates",
        action="store_true",
        help="Accept every non-fallback refinement during replay.",
    )
    parser.add_argument(
        "--replay-output-mode",
        choices=REPLAY_OUTPUT_MODES,
        default="diagnostic",
        help=(
            "How to rebuild accepted output boxes from diagnostics. "
            "'diagnostic' preserves refiner_output_xywh. 'box', "
            "'center-only', and 'size-only' re-project from the stored raw "
            "DVS refinement, optionally using --replay-output-blend."
        ),
    )
    parser.add_argument(
        "--replay-output-blend",
        type=float,
        help=(
            "Optional blend from the base tracker box toward the stored raw "
            "DVS refinement before applying --replay-output-mode. Requires "
            "--replay-output-mode other than 'diagnostic'."
        ),
    )
    parser.add_argument(
        "--replay-output-size-smoothing",
        type=float,
        help=(
            "Optional temporal size smoothing for replayed projected outputs. "
            "The value is the weight of the previous accepted replay width/height."
        ),
    )
    parser.add_argument(
        "--replay-output-center-smoothing",
        type=float,
        help=(
            "Optional temporal center smoothing for replayed projected outputs. "
            "The value is the weight of the previous accepted replay center."
        ),
    )
    parser.add_argument(
        "--replay-output-size-deadband-ratio",
        type=float,
        help=(
            "Optional per-axis size deadband relative to the base width/height. "
            "Replayed size changes smaller than this ratio are ignored."
        ),
    )
    parser.add_argument(
        "--replay-output-center-deadband-ratio",
        type=float,
        help=(
            "Optional center-shift deadband relative to the base-box diagonal. "
            "Replayed center shifts smaller than this ratio are ignored."
        ),
    )
    parser.add_argument(
        "--replay-output-center-clamp-ratio",
        type=float,
        help=(
            "Optional center-shift clamp relative to the base-box diagonal. "
            "Replayed center shifts larger than this ratio are capped."
        ),
    )
    parser.add_argument(
        "--replay-output-size-clamp-ratio",
        type=float,
        help=(
            "Optional per-axis size clamp relative to the base width/height. "
            "Replayed size changes larger than this ratio are capped."
        ),
    )
    parser.add_argument(
        "--replay-output-confidence-field",
        choices=PROJECTION_CONFIDENCE_FIELDS,
        help=(
            "Optional frame diagnostic used to shrink replayed corrections "
            "toward the base tracker when confidence is weak."
        ),
    )
    parser.add_argument(
        "--replay-output-confidence-floor",
        type=float,
        help="Confidence value that maps replay correction strength to zero.",
    )
    parser.add_argument(
        "--replay-output-confidence-ceiling",
        type=float,
        help="Confidence value that maps replay correction strength to one.",
    )
    _add_policy_arguments(parser)
    return parser


def _add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-accept-used-events", dest="min_used_event_count", type=int)
    parser.add_argument(
        "--min-accept-active-measurements",
        dest="min_active_measurement_count",
        type=int,
    )
    parser.add_argument(
        "--min-accept-mean-activity",
        dest="min_mean_event_activity",
        type=float,
    )
    parser.add_argument("--min-accept-candidate-iou", dest="min_candidate_iou", type=float)
    parser.add_argument(
        "--min-accept-area-ratio",
        dest="min_candidate_area_ratio",
        type=float,
    )
    parser.add_argument(
        "--max-accept-area-ratio",
        dest="max_candidate_area_ratio",
        type=float,
    )
    parser.add_argument(
        "--max-accept-center-shift-ratio",
        dest="max_center_shift_ratio",
        type=float,
    )
    parser.add_argument("--min-raw-candidate-iou", type=float)
    parser.add_argument("--min-raw-area-ratio", dest="min_raw_candidate_area_ratio", type=float)
    parser.add_argument("--max-raw-area-ratio", dest="max_raw_candidate_area_ratio", type=float)
    parser.add_argument(
        "--max-raw-center-shift-ratio",
        dest="max_raw_center_shift_ratio",
        type=float,
    )
    parser.add_argument("--min-polarity-consistency-fraction", type=float)
    parser.add_argument("--min-mean-event-polarity-weight", type=float)
    parser.add_argument("--max-quadratic-form-per-active-measurement", type=float)
    parser.add_argument("--min-active-fraction", type=float)
    parser.add_argument("--max-temporal-center-shift-ratio", type=float)
    parser.add_argument("--max-temporal-size-change-ratio", type=float)


def options_from_args(args: argparse.Namespace) -> EventVOTAcceptanceReplayOptions:
    override_keys = set(asdict(ReplayAcceptanceConfig()))
    overrides = {
        key: getattr(args, key)
        for key in override_keys
        if hasattr(args, key) and getattr(args, key) is not None
    }
    if args.disable_conservative_gates:
        overrides["enabled"] = False
    return EventVOTAcceptanceReplayOptions(
        diagnostics_json=args.diagnostics_json,
        output_results=args.output_results,
        eventvot_root=args.eventvot_root,
        base_results=args.base_results,
        split=args.split,
        sequences=tuple(args.sequence),
        summary_json=args.summary_json,
        decisions_csv=args.decisions_csv,
        skip_evaluation=args.skip_evaluation,
        output_projection_config=output_projection_config_from_args(args),
        config_overrides=overrides,
    )


def run(options: EventVOTAcceptanceReplayOptions) -> dict[str, Any]:
    """Replay an acceptance policy and optionally evaluate EventVOT metrics."""

    diagnostics = json.loads(options.diagnostics_json.read_text(encoding="utf-8"))
    config = options.acceptance_config or acceptance_config_from_diagnostics(
        diagnostics,
        options.config_overrides,
    )
    output_projection = output_projection_config_from_diagnostics(
        options.output_projection_config,
        diagnostics,
    )
    selected_summaries = select_sequence_summaries(
        diagnostics.get("sequences", []),
        options.sequences,
    )
    if not selected_summaries:
        raise ValueError("No sequence diagnostics selected for replay")

    options.output_results.mkdir(parents=True, exist_ok=True)
    decisions: list[dict[str, Any]] = []
    sequence_outputs: list[dict[str, Any]] = []
    aggregate_counts: Counter[str] = Counter()
    sequence_names: list[str] = []

    for sequence_summary in selected_summaries:
        sequence_name = str(sequence_summary["sequence"])
        sequence_names.append(sequence_name)
        base_result_file = resolve_replay_base_result_file(
            sequence_summary,
            options.base_results,
        )
        base_boxes = load_xywh_result_file(base_result_file)
        replayed_boxes, sequence_counts, sequence_decisions = replay_sequence_boxes(
            sequence_name,
            base_boxes,
            sequence_summary.get("frames", []),
            config,
            output_projection,
        )
        save_xywh_result_file(options.output_results / f"{sequence_name}.txt", replayed_boxes)
        aggregate_counts.update(sequence_counts)
        decisions.extend(sequence_decisions)
        sequence_outputs.append(
            {
                "sequence": sequence_name,
                "base_result_file": str(base_result_file),
                "output_result_file": str(options.output_results / f"{sequence_name}.txt"),
                "frame_count": int(base_boxes.shape[0]),
                "acceptance_counts": dict(sorted(sequence_counts.items())),
            }
        )

    metrics = None
    split = options.split or diagnostics.get("options", {}).get("split", "test")
    if not options.skip_evaluation:
        if options.eventvot_root is None:
            raise ValueError("--eventvot-root is required unless --skip-evaluation is used")
        split_root = resolve_eventvot_split_root(options.eventvot_root, str(split))
        metrics = evaluate_eventvot_results(split_root, options.output_results, sequence_names)

    if options.decisions_csv is not None:
        write_decisions_csv(options.decisions_csv, decisions)

    summary = {
        "sequence_count": len(sequence_outputs),
        "frame_count": sum(output["frame_count"] for output in sequence_outputs),
        "accepted_refinement_count": int(aggregate_counts.get("accepted", 0)),
        "acceptance_counts": dict(sorted(aggregate_counts.items())),
        "metrics": metrics,
        "output_results": str(options.output_results),
    }
    payload = {
        "schema_version": 1,
        "description": (
            "EventVOT DVS-ENACT acceptance replay from stored per-frame "
            "diagnostics. Result files contain the base tracker boxes except "
            "where the replay policy accepts refiner_output_xywh."
        ),
        "split": split,
        "diagnostics_json": str(options.diagnostics_json),
        "acceptance_config": asdict(config),
        "output_projection_config": asdict(output_projection),
        "summary": summary,
        "sequences": sequence_outputs,
    }
    summary_json = options.summary_json or options.output_results / "acceptance_replay_summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def acceptance_config_from_diagnostics(
    diagnostics: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> ReplayAcceptanceConfig:
    """Build a replay policy from the original diagnostics plus CLI overrides."""

    values = asdict(ReplayAcceptanceConfig())
    original = diagnostics.get("acceptance_config", {})
    for key in values:
        if key in original:
            values[key] = original[key]
    for key, value in (overrides or {}).items():
        if key not in values:
            raise ValueError(f"Unknown acceptance-policy field: {key}")
        values[key] = value
    return ReplayAcceptanceConfig(**values)


def output_projection_config_from_args(
    args: argparse.Namespace,
) -> ReplayOutputProjectionConfig:
    """Return replay-output projection options from CLI arguments."""

    config = ReplayOutputProjectionConfig(
        mode=args.replay_output_mode,
        blend=args.replay_output_blend,
        size_smoothing=args.replay_output_size_smoothing,
        center_smoothing=args.replay_output_center_smoothing,
        center_clamp_ratio=args.replay_output_center_clamp_ratio,
        center_deadband_ratio=args.replay_output_center_deadband_ratio,
        size_clamp_ratio=args.replay_output_size_clamp_ratio,
        size_deadband_ratio=args.replay_output_size_deadband_ratio,
        confidence_field=args.replay_output_confidence_field,
        confidence_floor=args.replay_output_confidence_floor,
        confidence_ceiling=args.replay_output_confidence_ceiling,
    )
    validate_output_projection_config(config)
    return config


def output_projection_config_from_diagnostics(
    config: ReplayOutputProjectionConfig,
    diagnostics: dict[str, Any],
) -> ReplayOutputProjectionConfig:
    """Fill replay-output image bounds from diagnostics when available."""

    refiner_config = diagnostics.get("refiner_config", {})
    image_width = config.image_width
    if image_width is None:
        image_width = _optional_float(refiner_config.get("image_width"))
    image_height = config.image_height
    if image_height is None:
        image_height = _optional_float(refiner_config.get("image_height"))
    resolved = ReplayOutputProjectionConfig(
        mode=config.mode,
        blend=config.blend,
        size_smoothing=config.size_smoothing,
        center_smoothing=config.center_smoothing,
        center_clamp_ratio=config.center_clamp_ratio,
        center_deadband_ratio=config.center_deadband_ratio,
        size_clamp_ratio=config.size_clamp_ratio,
        size_deadband_ratio=config.size_deadband_ratio,
        confidence_field=config.confidence_field,
        confidence_floor=config.confidence_floor,
        confidence_ceiling=config.confidence_ceiling,
        image_width=image_width,
        image_height=image_height,
    )
    validate_output_projection_config(resolved)
    return resolved


def validate_output_projection_config(config: ReplayOutputProjectionConfig) -> None:
    """Raise ``ValueError`` for invalid replay-output projection settings."""

    if config.mode not in REPLAY_OUTPUT_MODES:
        raise ValueError(
            f"Unsupported replay output mode {config.mode!r}; "
            f"expected one of {', '.join(REPLAY_OUTPUT_MODES)}"
        )
    if config.blend is not None:
        if config.mode == "diagnostic":
            raise ValueError("--replay-output-blend requires a projected output mode")
        if not 0.0 <= float(config.blend) <= 1.0:
            raise ValueError("--replay-output-blend must be between 0 and 1")
    if config.size_smoothing is not None:
        if config.mode == "diagnostic":
            raise ValueError(
                "--replay-output-size-smoothing requires a projected output mode"
            )
        if not 0.0 <= float(config.size_smoothing) <= 1.0:
            raise ValueError("--replay-output-size-smoothing must be between 0 and 1")
    if config.center_smoothing is not None:
        if config.mode == "diagnostic":
            raise ValueError(
                "--replay-output-center-smoothing requires a projected output mode"
            )
        if not 0.0 <= float(config.center_smoothing) <= 1.0:
            raise ValueError("--replay-output-center-smoothing must be between 0 and 1")
    if config.size_deadband_ratio is not None:
        if config.mode == "diagnostic":
            raise ValueError(
                "--replay-output-size-deadband-ratio requires a projected output mode"
            )
        if float(config.size_deadband_ratio) < 0.0:
            raise ValueError(
                "--replay-output-size-deadband-ratio must be non-negative"
            )
    if config.size_clamp_ratio is not None:
        if config.mode == "diagnostic":
            raise ValueError(
                "--replay-output-size-clamp-ratio requires a projected output mode"
            )
        if float(config.size_clamp_ratio) < 0.0:
            raise ValueError("--replay-output-size-clamp-ratio must be non-negative")
    if config.center_deadband_ratio is not None:
        if config.mode == "diagnostic":
            raise ValueError(
                "--replay-output-center-deadband-ratio requires a projected output mode"
            )
        if float(config.center_deadband_ratio) < 0.0:
            raise ValueError(
                "--replay-output-center-deadband-ratio must be non-negative"
            )
    if config.center_clamp_ratio is not None:
        if config.mode == "diagnostic":
            raise ValueError(
                "--replay-output-center-clamp-ratio requires a projected output mode"
            )
        if float(config.center_clamp_ratio) < 0.0:
            raise ValueError("--replay-output-center-clamp-ratio must be non-negative")
    validate_projection_confidence_weighting(
        config.confidence_field,
        config.confidence_floor,
        config.confidence_ceiling,
    )
    if config.confidence_field is not None and config.mode == "diagnostic":
        raise ValueError("--replay-output-confidence-field requires a projected mode")


def select_sequence_summaries(
    sequence_summaries: Iterable[dict[str, Any]],
    requested_sequences: tuple[str, ...],
) -> list[dict[str, Any]]:
    summaries = list(sequence_summaries)
    if not requested_sequences:
        return summaries
    requested = set(requested_sequences)
    selected = [summary for summary in summaries if str(summary.get("sequence")) in requested]
    missing = sorted(requested.difference(str(summary.get("sequence")) for summary in selected))
    if missing:
        raise ValueError(f"Requested sequences are missing from diagnostics: {missing}")
    return selected


def resolve_replay_base_result_file(
    sequence_summary: dict[str, Any],
    base_results: Path | None,
) -> Path:
    sequence_name = str(sequence_summary["sequence"])
    if base_results is not None:
        return resolve_base_result_file(base_results, sequence_name)
    base_result_file = sequence_summary.get("base_result_file")
    if not base_result_file:
        raise ValueError(
            f"{sequence_name}: diagnostics do not include base_result_file; "
            "--base-results is required"
        )
    path = Path(str(base_result_file))
    if not path.exists():
        raise FileNotFoundError(
            f"{sequence_name}: base result file from diagnostics does not exist: {path}. "
            "Pass --base-results to use a local result directory."
        )
    return path


def replay_sequence_boxes(
    sequence_name: str,
    base_boxes: np.ndarray,
    frames: list[dict[str, Any]],
    config: ReplayAcceptanceConfig,
    output_projection: ReplayOutputProjectionConfig | None = None,
) -> tuple[np.ndarray, Counter[str], list[dict[str, Any]]]:
    """Return replayed boxes, aggregate counts, and per-frame decision records."""

    replayed_boxes = np.array(base_boxes, dtype=float, copy=True)
    counts: Counter[str] = Counter()
    decision_records: list[dict[str, Any]] = []
    output_projection = output_projection or ReplayOutputProjectionConfig()
    previous_accepted_projected_center: np.ndarray | None = None
    previous_accepted_projected_size: np.ndarray | None = None

    if not frames:
        counts["missing_frame_diagnostics"] += int(base_boxes.shape[0])
        return replayed_boxes, counts, decision_records

    for frame in frames:
        frame_index = int(frame.get("frame_index", -1))
        if frame_index < 0 or frame_index >= base_boxes.shape[0]:
            continue
        if frame_index == 0:
            counts["initial_frame"] += 1
            continue

        decision = evaluate_frame_acceptance(
            base_boxes[frame_index],
            frame,
            config,
            output_projection,
            previous_projected_center=previous_accepted_projected_center,
            previous_projected_size=previous_accepted_projected_size,
            previous_output_xywh=replayed_boxes[frame_index - 1],
        )
        reason_key = "accepted" if decision.accepted else decision.rejection_reasons[0]
        counts[reason_key] += 1
        if decision.accepted:
            replayed_output = frame_projected_output_xywh(
                base_boxes[frame_index],
                frame,
                output_projection,
                previous_projected_center=previous_accepted_projected_center,
                previous_projected_size=previous_accepted_projected_size,
            )
            replayed_boxes[frame_index] = replayed_output
            if output_projection.mode != "diagnostic":
                if (
                    output_projection.center_smoothing is not None
                    and output_projection.mode in {"box", "center-only"}
                ):
                    previous_accepted_projected_center = (
                        replayed_output[:2] + 0.5 * replayed_output[2:]
                    ).copy()
                if (
                    output_projection.size_smoothing is not None
                    and output_projection.mode != "center-only"
                ):
                    previous_accepted_projected_size = replayed_output[2:].copy()
        decision_records.append(
            {
                "sequence": sequence_name,
                "frame_index": frame_index,
                **decision_to_dict(decision),
            }
        )

    return replayed_boxes, counts, decision_records


def evaluate_frame_acceptance(
    candidate_xywh: np.ndarray,
    frame: dict[str, Any],
    config: ReplayAcceptanceConfig,
    output_projection: ReplayOutputProjectionConfig | None = None,
    *,
    previous_projected_center: np.ndarray | None = None,
    previous_projected_size: np.ndarray | None = None,
    previous_output_xywh: np.ndarray | None = None,
) -> ReplayAcceptanceDecision:
    """Evaluate one stored DVS-ENACT refinement under a replay policy."""

    candidate = np.asarray(candidate_xywh, dtype=float)
    output_projection = output_projection or ReplayOutputProjectionConfig()
    proposed = frame_projected_output_xywh(
        candidate,
        frame,
        output_projection,
        previous_projected_center=previous_projected_center,
        previous_projected_size=previous_projected_size,
    )
    raw_proposed = frame_raw_refined_xywh(frame, fallback=proposed)

    candidate_iou = box_iou_xywh(candidate, proposed)
    candidate_area_ratio = area_ratio_xywh(candidate, proposed)
    center_shift_ratio = center_shift_ratio_xywh(candidate, proposed)
    raw_candidate_iou = box_iou_xywh(candidate, raw_proposed)
    raw_candidate_area_ratio = area_ratio_xywh(candidate, raw_proposed)
    raw_center_shift_ratio = center_shift_ratio_xywh(candidate, raw_proposed)
    temporal_center_shift_ratio = (
        center_shift_ratio_xywh(previous_output_xywh, proposed)
        if previous_output_xywh is not None
        else None
    )
    temporal_size_change_ratio = (
        size_change_ratio_xywh(previous_output_xywh, proposed)
        if previous_output_xywh is not None
        else None
    )

    used_event_count = _optional_int(frame.get("used_event_count"))
    active_measurement_count = _optional_int(frame.get("active_measurement_count"))
    active_fraction = _active_fraction(active_measurement_count, used_event_count)
    quadratic_per_active = _quadratic_per_active(
        frame.get("quadratic_form"),
        active_measurement_count,
    )

    rejection_reasons: list[str] = []
    fallback_reason = frame.get("fallback_reason")
    if fallback_reason is not None:
        rejection_reasons.append(f"fallback:{fallback_reason}")

    if not _finite_xywh(proposed):
        rejection_reasons.append("refiner_output_invalid")
    if not _finite_xywh(raw_proposed):
        rejection_reasons.append("raw_refiner_output_invalid")

    if config.enabled:
        _append_min_count_gate(
            rejection_reasons,
            "used_event_count",
            used_event_count,
            config.min_used_event_count,
        )
        _append_min_count_gate(
            rejection_reasons,
            "active_measurement_count",
            active_measurement_count,
            config.min_active_measurement_count,
        )
        _append_min_float_gate(
            rejection_reasons,
            "mean_event_activity",
            frame.get("mean_event_activity"),
            config.min_mean_event_activity,
            missing_reason="mean_event_activity_missing",
        )
        _append_min_float_gate(
            rejection_reasons,
            "candidate_iou",
            candidate_iou,
            config.min_candidate_iou,
        )
        _append_min_float_gate(
            rejection_reasons,
            "candidate_area_ratio",
            candidate_area_ratio,
            config.min_candidate_area_ratio,
        )
        _append_max_float_gate(
            rejection_reasons,
            "candidate_area_ratio",
            candidate_area_ratio,
            config.max_candidate_area_ratio,
        )
        _append_max_float_gate(
            rejection_reasons,
            "center_shift_ratio",
            center_shift_ratio,
            config.max_center_shift_ratio,
        )
        _append_min_float_gate(
            rejection_reasons,
            "raw_candidate_iou",
            raw_candidate_iou,
            config.min_raw_candidate_iou,
        )
        _append_min_float_gate(
            rejection_reasons,
            "raw_candidate_area_ratio",
            raw_candidate_area_ratio,
            config.min_raw_candidate_area_ratio,
        )
        _append_max_float_gate(
            rejection_reasons,
            "raw_candidate_area_ratio",
            raw_candidate_area_ratio,
            config.max_raw_candidate_area_ratio,
        )
        _append_max_float_gate(
            rejection_reasons,
            "raw_center_shift_ratio",
            raw_center_shift_ratio,
            config.max_raw_center_shift_ratio,
        )
        _append_min_float_gate(
            rejection_reasons,
            "polarity_consistency_fraction",
            frame.get("polarity_consistency_fraction"),
            config.min_polarity_consistency_fraction,
            missing_reason="polarity_consistency_fraction_missing",
        )
        _append_min_float_gate(
            rejection_reasons,
            "mean_event_polarity_weight",
            frame.get("mean_event_polarity_weight"),
            config.min_mean_event_polarity_weight,
            missing_reason="mean_event_polarity_weight_missing",
        )
        _append_max_float_gate(
            rejection_reasons,
            "quadratic_form_per_active_measurement",
            quadratic_per_active,
            config.max_quadratic_form_per_active_measurement,
            missing_reason="quadratic_form_per_active_measurement_missing",
        )
        _append_min_float_gate(
            rejection_reasons,
            "active_fraction",
            active_fraction,
            config.min_active_fraction,
            missing_reason="active_fraction_missing",
        )
        _append_max_float_gate(
            rejection_reasons,
            "temporal_center_shift_ratio",
            temporal_center_shift_ratio,
            config.max_temporal_center_shift_ratio,
            missing_reason="temporal_center_shift_ratio_missing",
        )
        _append_max_float_gate(
            rejection_reasons,
            "temporal_size_change_ratio",
            temporal_size_change_ratio,
            config.max_temporal_size_change_ratio,
            missing_reason="temporal_size_change_ratio_missing",
        )

    return ReplayAcceptanceDecision(
        accepted=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
        candidate_iou=float(candidate_iou),
        candidate_area_ratio=float(candidate_area_ratio),
        center_shift_ratio=float(center_shift_ratio),
        raw_candidate_iou=float(raw_candidate_iou),
        raw_candidate_area_ratio=float(raw_candidate_area_ratio),
        raw_center_shift_ratio=float(raw_center_shift_ratio),
        temporal_center_shift_ratio=temporal_center_shift_ratio,
        temporal_size_change_ratio=temporal_size_change_ratio,
        active_fraction=active_fraction,
        quadratic_form_per_active_measurement=quadratic_per_active,
    )


def decision_to_dict(decision: ReplayAcceptanceDecision) -> dict[str, Any]:
    payload = asdict(decision)
    payload["rejection_reasons"] = list(decision.rejection_reasons)
    return payload


def frame_refiner_output_xywh(frame: dict[str, Any]) -> np.ndarray:
    if "refiner_output_xywh" in frame:
        return np.asarray(frame["refiner_output_xywh"], dtype=float)
    if "output_xywh" in frame:
        return np.asarray(frame["output_xywh"], dtype=float)
    if "output_bbox" in frame:
        return bbox_dict_to_xywh(frame["output_bbox"])
    raise ValueError(f"Frame {frame.get('frame_index')} lacks refiner output diagnostics")


def frame_projected_output_xywh(
    candidate_xywh: np.ndarray,
    frame: dict[str, Any],
    output_projection: ReplayOutputProjectionConfig,
    *,
    previous_projected_center: np.ndarray | None = None,
    previous_projected_size: np.ndarray | None = None,
) -> np.ndarray:
    """Return the accepted replay box under the selected output projection."""

    diagnostic_output = frame_refiner_output_xywh(frame)
    if output_projection.mode == "diagnostic":
        return diagnostic_output

    candidate = np.asarray(candidate_xywh, dtype=float)
    raw_refined = frame_raw_refined_xywh(frame, fallback=diagnostic_output)
    source_output = diagnostic_output
    if output_projection.blend is not None:
        source_output = blend_xywh(candidate, raw_refined, float(output_projection.blend))
    return project_refinement_output(
        candidate,
        source_output,
        refinement_mode=output_projection.mode,
        raw_refined_xywh=raw_refined,
        previous_projected_center=previous_projected_center,
        previous_projected_size=previous_projected_size,
        projection_size_smoothing=output_projection.size_smoothing,
        projection_center_smoothing=output_projection.center_smoothing,
        projection_center_clamp_ratio=output_projection.center_clamp_ratio,
        projection_center_deadband_ratio=output_projection.center_deadband_ratio,
        projection_size_clamp_ratio=output_projection.size_clamp_ratio,
        projection_size_deadband_ratio=output_projection.size_deadband_ratio,
        projection_confidence_value=frame_projection_confidence_value(
            frame,
            output_projection.confidence_field,
        ),
        projection_confidence_floor=output_projection.confidence_floor,
        projection_confidence_ceiling=output_projection.confidence_ceiling,
        image_width=output_projection.image_width,
        image_height=output_projection.image_height,
    )


def frame_projection_confidence_value(
    frame: dict[str, Any],
    field: str | None,
) -> float | None:
    """Read a scalar projection-confidence diagnostic from replay frame data."""
    if field is None:
        return None
    if field == "active_fraction":
        active_count = _optional_int(frame.get("active_measurement_count"))
        used_count = _optional_int(frame.get("used_event_count"))
        return _active_fraction(active_count, used_count)
    return _optional_float(frame.get(field))


def blend_xywh(candidate_xywh: np.ndarray, raw_refined_xywh: np.ndarray, blend: float) -> np.ndarray:
    """Blend linearly from the base tracker box toward the raw DVS refinement."""

    candidate = np.asarray(candidate_xywh, dtype=float)
    raw_refined = np.asarray(raw_refined_xywh, dtype=float)
    return (1.0 - blend) * candidate + blend * raw_refined


def frame_raw_refined_xywh(
    frame: dict[str, Any],
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    refined_bbox = frame.get("refined_bbox")
    if isinstance(refined_bbox, dict):
        return bbox_dict_to_xywh(refined_bbox)
    return np.asarray(fallback, dtype=float)


def bbox_dict_to_xywh(bbox: dict[str, Any]) -> np.ndarray:
    lower = {str(key).lower(): float(value) for key, value in bbox.items()}
    if {"x", "y", "width", "height"}.issubset(lower):
        return np.asarray(
            [lower["x"], lower["y"], lower["width"], lower["height"]],
            dtype=float,
        )
    if {"x_min", "y_min", "width", "height"}.issubset(lower):
        return np.asarray(
            [lower["x_min"], lower["y_min"], lower["width"], lower["height"]],
            dtype=float,
        )
    if {"x_min", "y_min", "x_max", "y_max"}.issubset(lower):
        return np.asarray(
            [
                lower["x_min"],
                lower["y_min"],
                lower["x_max"] - lower["x_min"],
                lower["y_max"] - lower["y_min"],
            ],
            dtype=float,
        )
    raise ValueError(f"Unsupported bbox diagnostic fields: {sorted(bbox)}")


def write_decisions_csv(path: Path, decisions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not decisions:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(decisions[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in decisions:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def _append_min_count_gate(
    reasons: list[str],
    reason: str,
    value: int | None,
    threshold: int | None,
) -> None:
    if threshold is None:
        return
    if value is None or value < int(threshold):
        reasons.append(reason)


def _append_float_gate(
    reasons: list[str],
    reason: str,
    value: Any,
    threshold: float | None,
    *,
    is_rejected: Callable[[float, float], bool],
    missing_reason: str | None = None,
) -> None:
    if threshold is None:
        return
    numeric = _optional_float(value)
    if numeric is None:
        reasons.append(missing_reason or reason)
    elif is_rejected(numeric, float(threshold)):
        reasons.append(reason)


_append_min_float_gate = partial(_append_float_gate, is_rejected=lt)
_append_max_float_gate = partial(_append_float_gate, is_rejected=gt)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _active_fraction(
    active_measurement_count: int | None,
    used_event_count: int | None,
) -> float | None:
    if active_measurement_count is None or used_event_count is None or used_event_count <= 0:
        return None
    return float(active_measurement_count / used_event_count)


def _quadratic_per_active(
    quadratic_form: Any,
    active_measurement_count: int | None,
) -> float | None:
    quadratic = _optional_float(quadratic_form)
    if quadratic is None or active_measurement_count is None or active_measurement_count <= 0:
        return None
    return float(quadratic / active_measurement_count)


def _finite_xywh(box: np.ndarray) -> bool:
    values = np.asarray(box, dtype=float)
    return values.shape == (4,) and np.all(np.isfinite(values)) and bool(np.all(values[2:] > 0.0))


if __name__ == "__main__":
    raise SystemExit(main())
