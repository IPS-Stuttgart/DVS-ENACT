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
)
from run_eventvot_validation_sweep import evaluate_eventvot_results  # noqa: E402


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
        config_overrides=overrides,
    )


def run(options: EventVOTAcceptanceReplayOptions) -> dict[str, Any]:
    """Replay an acceptance policy and optionally evaluate EventVOT metrics."""

    diagnostics = json.loads(options.diagnostics_json.read_text(encoding="utf-8"))
    config = options.acceptance_config or acceptance_config_from_diagnostics(
        diagnostics,
        options.config_overrides,
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
) -> tuple[np.ndarray, Counter[str], list[dict[str, Any]]]:
    """Return replayed boxes, aggregate counts, and per-frame decision records."""

    replayed_boxes = np.array(base_boxes, dtype=float, copy=True)
    counts: Counter[str] = Counter()
    decision_records: list[dict[str, Any]] = []

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

        decision = evaluate_frame_acceptance(base_boxes[frame_index], frame, config)
        reason_key = "accepted" if decision.accepted else decision.rejection_reasons[0]
        counts[reason_key] += 1
        if decision.accepted:
            replayed_boxes[frame_index] = frame_refiner_output_xywh(frame)
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
) -> ReplayAcceptanceDecision:
    """Evaluate one stored DVS-ENACT refinement under a replay policy."""

    candidate = np.asarray(candidate_xywh, dtype=float)
    proposed = frame_refiner_output_xywh(frame)
    raw_proposed = frame_raw_refined_xywh(frame, fallback=proposed)

    candidate_iou = box_iou_xywh(candidate, proposed)
    candidate_area_ratio = area_ratio_xywh(candidate, proposed)
    center_shift_ratio = center_shift_ratio_xywh(candidate, proposed)
    raw_candidate_iou = box_iou_xywh(candidate, raw_proposed)
    raw_candidate_area_ratio = area_ratio_xywh(candidate, raw_proposed)
    raw_center_shift_ratio = center_shift_ratio_xywh(candidate, raw_proposed)

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

    return ReplayAcceptanceDecision(
        accepted=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
        candidate_iou=float(candidate_iou),
        candidate_area_ratio=float(candidate_area_ratio),
        center_shift_ratio=float(center_shift_ratio),
        raw_candidate_iou=float(raw_candidate_iou),
        raw_candidate_area_ratio=float(raw_candidate_area_ratio),
        raw_center_shift_ratio=float(raw_center_shift_ratio),
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


def _append_min_float_gate(
    reasons: list[str],
    reason: str,
    value: Any,
    threshold: float | None,
    *,
    missing_reason: str | None = None,
) -> None:
    _append_float_gate(
        reasons,
        reason,
        value,
        threshold,
        is_rejected=lambda numeric, floor: numeric < floor,
        missing_reason=missing_reason,
    )


def _append_max_float_gate(
    reasons: list[str],
    reason: str,
    value: Any,
    threshold: float | None,
    *,
    missing_reason: str | None = None,
) -> None:
    _append_float_gate(
        reasons,
        reason,
        value,
        threshold,
        is_rejected=lambda numeric, ceiling: numeric > ceiling,
        missing_reason=missing_reason,
    )


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
