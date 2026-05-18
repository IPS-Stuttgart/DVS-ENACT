"""Replay EventVOT results with DVS-ENACT confidence and short-term memory.

The normal EventVOT refinement adapter uses DVS-ENACT as a one-frame box
refiner: either the current DVS contour update passes conservative gates and
replaces the base tracker box, or the base box is kept. This utility reuses the
same diagnostics JSON but turns DVS-ENACT into a confidence and memory signal:

* confident DVS updates are trusted directly and stored as a correction memory;
* weak or event-silent frames may reuse the last trusted DVS correction for a
  few compatible frames with exponential decay;
* incompatible jumps, stale memories, and low-confidence direct updates fall
  back to the base tracker.

No contours are recomputed. The script is therefore intended for validation-set
experiments after one expensive ``run_eventvot_refinement.py`` pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_eventvot_acceptance_replay import (  # noqa: E402
    frame_refiner_output_xywh,
    frame_raw_refined_xywh,
    resolve_replay_base_result_file,
    select_sequence_summaries,
)
from run_eventvot_refinement import (  # noqa: E402
    area_ratio_xywh,
    box_iou_xywh,
    center_shift_ratio_xywh,
    load_xywh_result_file,
    resolve_eventvot_split_root,
    save_xywh_result_file,
)
from run_eventvot_validation_sweep import evaluate_eventvot_results  # noqa: E402


@dataclass(frozen=True)
class ConfidenceMemoryConfig:
    """Policy for direct DVS trust and short-term correction memory."""

    direct_confidence_threshold: float = 0.65
    memory_confidence_threshold: float = 0.0
    max_memory_age: int = 5
    memory_decay: float = 0.75
    direct_alpha: float = 1.0
    memory_alpha: float = 0.75

    reference_used_event_count: float = 30.0
    reference_active_measurement_count: float = 6.0
    reference_mean_event_activity: float = 0.25

    min_direct_candidate_iou: float = 0.60
    min_direct_area_ratio: float = 0.50
    max_direct_area_ratio: float = 1.50
    max_direct_center_shift_ratio: float = 0.25

    min_memory_base_iou: float = 0.05
    max_memory_base_center_shift_ratio: float = 0.80
    max_memory_scale_change_ratio: float = 2.00

    clip_to_image: bool = True
    image_width: int | None = None
    image_height: int | None = None


@dataclass(frozen=True)
class FrameConfidence:
    """DVS confidence score and interpretable components for one frame."""

    score: float
    components: dict[str, float]
    candidate_iou: float
    candidate_area_ratio: float
    center_shift_ratio: float
    raw_candidate_iou: float
    raw_candidate_area_ratio: float
    raw_center_shift_ratio: float
    fallback_reason: str | None


@dataclass
class CorrectionMemory:
    """Last trusted DVS correction relative to the base tracker state."""

    frame_index: int
    base_xywh: np.ndarray
    trusted_xywh: np.ndarray
    confidence: float


@dataclass(frozen=True)
class EventVOTConfidenceMemoryOptions:
    """Inputs and outputs for one confidence-memory replay run."""

    diagnostics_json: Path
    output_results: Path
    eventvot_root: Path | None = None
    base_results: Path | None = None
    split: str | None = None
    sequences: tuple[str, ...] = ()
    summary_json: Path | None = None
    decisions_csv: Path | None = None
    skip_evaluation: bool = False
    config: ConfidenceMemoryConfig | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)


def main() -> int:
    args = build_parser().parse_args()
    payload = run(options_from_args(args))
    print(json.dumps(payload["summary"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay EventVOT result files using stored DVS-ENACT diagnostics as "
            "a confidence and short-term correction-memory signal."
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
    _add_policy_arguments(parser)
    return parser


def _add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--direct-confidence-threshold", type=float)
    parser.add_argument("--memory-confidence-threshold", type=float)
    parser.add_argument("--max-memory-age", type=int)
    parser.add_argument("--memory-decay", type=float)
    parser.add_argument("--direct-alpha", type=float)
    parser.add_argument("--memory-alpha", type=float)
    parser.add_argument("--reference-used-event-count", type=float)
    parser.add_argument("--reference-active-measurement-count", type=float)
    parser.add_argument("--reference-mean-event-activity", type=float)
    parser.add_argument("--min-direct-candidate-iou", type=float)
    parser.add_argument("--min-direct-area-ratio", type=float)
    parser.add_argument("--max-direct-area-ratio", type=float)
    parser.add_argument("--max-direct-center-shift-ratio", type=float)
    parser.add_argument("--min-memory-base-iou", type=float)
    parser.add_argument("--max-memory-base-center-shift-ratio", type=float)
    parser.add_argument("--max-memory-scale-change-ratio", type=float)
    parser.add_argument(
        "--no-clip-to-image",
        dest="clip_to_image",
        action="store_false",
        default=None,
        help="Do not clip replayed boxes to the refiner image extent.",
    )
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--image-height", type=int)


def options_from_args(args: argparse.Namespace) -> EventVOTConfidenceMemoryOptions:
    override_keys = set(asdict(ConfidenceMemoryConfig()))
    overrides = {
        key: getattr(args, key)
        for key in override_keys
        if hasattr(args, key) and getattr(args, key) is not None
    }
    return EventVOTConfidenceMemoryOptions(
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


def run(options: EventVOTConfidenceMemoryOptions) -> dict[str, Any]:
    """Create confidence-memory EventVOT result files and optional metrics."""

    diagnostics = json.loads(options.diagnostics_json.read_text(encoding="utf-8"))
    config = options.config or confidence_memory_config_from_diagnostics(
        diagnostics,
        options.config_overrides,
    )
    selected_summaries = select_sequence_summaries(
        diagnostics.get("sequences", []),
        options.sequences,
    )
    if not selected_summaries:
        raise ValueError("No sequence diagnostics selected for confidence-memory replay")

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
        output_result_file = options.output_results / f"{sequence_name}.txt"
        save_xywh_result_file(output_result_file, replayed_boxes)
        aggregate_counts.update(sequence_counts)
        decisions.extend(sequence_decisions)
        sequence_outputs.append(
            {
                "sequence": sequence_name,
                "base_result_file": str(base_result_file),
                "output_result_file": str(output_result_file),
                "frame_count": int(base_boxes.shape[0]),
                "action_counts": dict(sorted(sequence_counts.items())),
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
        "direct_trust_count": int(aggregate_counts.get("direct", 0)),
        "memory_count": int(aggregate_counts.get("memory", 0)),
        "base_fallback_count": int(aggregate_counts.get("base", 0)),
        "action_counts": dict(sorted(aggregate_counts.items())),
        "metrics": metrics,
        "output_results": str(options.output_results),
    }
    payload = {
        "schema_version": 1,
        "description": (
            "EventVOT confidence-memory replay from stored DVS-ENACT diagnostics. "
            "Confident DVS updates are trusted directly; subsequent compatible "
            "frames may reuse the last trusted correction with age decay."
        ),
        "split": split,
        "diagnostics_json": str(options.diagnostics_json),
        "confidence_memory_config": asdict(config),
        "summary": summary,
        "sequences": sequence_outputs,
    }
    summary_json = options.summary_json or (
        options.output_results / "confidence_memory_summary.json"
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def confidence_memory_config_from_diagnostics(
    diagnostics: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> ConfidenceMemoryConfig:
    """Build a confidence-memory policy from diagnostics plus CLI overrides."""

    values = asdict(ConfidenceMemoryConfig())
    refiner_config = diagnostics.get("refiner_config", {})
    if values["image_width"] is None and refiner_config.get("image_width") is not None:
        values["image_width"] = int(refiner_config["image_width"])
    if values["image_height"] is None and refiner_config.get("image_height") is not None:
        values["image_height"] = int(refiner_config["image_height"])
    for key, value in (overrides or {}).items():
        if key not in values:
            raise ValueError(f"Unknown confidence-memory field: {key}")
        values[key] = value
    return ConfidenceMemoryConfig(**values)


def replay_sequence_boxes(
    sequence_name: str,
    base_boxes: np.ndarray,
    frames: Iterable[dict[str, Any]],
    config: ConfidenceMemoryConfig,
) -> tuple[np.ndarray, Counter[str], list[dict[str, Any]]]:
    """Return replayed boxes, action counts, and per-frame decision records."""

    replayed_boxes = np.array(base_boxes, dtype=float, copy=True)
    frames_by_index = {
        int(frame.get("frame_index", -1)): frame
        for frame in frames
        if int(frame.get("frame_index", -1)) >= 0
    }
    counts: Counter[str] = Counter()
    decisions: list[dict[str, Any]] = []
    memory: CorrectionMemory | None = None

    for frame_index in range(base_boxes.shape[0]):
        candidate = np.asarray(base_boxes[frame_index], dtype=float)
        frame = frames_by_index.get(frame_index)
        if frame_index == 0:
            counts["initial_frame"] += 1
            decisions.append(
                _decision_record(sequence_name, frame_index, "initial_frame", candidate)
            )
            continue
        if frame is None:
            counts["missing_frame_diagnostics"] += 1
            decisions.append(
                _decision_record(
                    sequence_name,
                    frame_index,
                    "missing_frame_diagnostics",
                    candidate,
                )
            )
            continue

        confidence = compute_frame_confidence(candidate, frame, config)
        action = "base"
        reason = "low_direct_confidence"
        output_box = candidate
        memory_age: int | None = None
        memory_strength: float | None = None

        direct_reasons = direct_rejection_reasons(confidence, frame, config)
        if not direct_reasons:
            trusted_box = _blend_xywh(
                candidate,
                frame_refiner_output_xywh(frame),
                config.direct_alpha,
            )
            output_box = _clip_xywh(trusted_box, config)
            if _finite_xywh(output_box):
                action = "direct"
                reason = "direct_confident"
                memory = CorrectionMemory(
                    frame_index=frame_index,
                    base_xywh=candidate.copy(),
                    trusted_xywh=output_box.copy(),
                    confidence=confidence.score,
                )
            else:
                action = "base"
                reason = "direct_output_invalid"
                output_box = candidate
        else:
            memory_candidate = build_memory_box(
                memory,
                candidate,
                base_boxes[frame_index - 1],
                frame_index,
                confidence,
                config,
            )
            if memory_candidate is not None:
                output_box, memory_age, memory_strength = memory_candidate
                action = "memory"
                reason = "memory_reused"

        replayed_boxes[frame_index] = output_box
        counts[action] += 1
        decisions.append(
            {
                "sequence": sequence_name,
                "frame_index": int(frame_index),
                "action": action,
                "reason": reason,
                "direct_rejection_reasons": list(direct_reasons),
                "confidence": float(confidence.score),
                "confidence_components": confidence.components,
                "candidate_iou": float(confidence.candidate_iou),
                "candidate_area_ratio": float(confidence.candidate_area_ratio),
                "center_shift_ratio": float(confidence.center_shift_ratio),
                "raw_candidate_iou": float(confidence.raw_candidate_iou),
                "raw_candidate_area_ratio": float(confidence.raw_candidate_area_ratio),
                "raw_center_shift_ratio": float(confidence.raw_center_shift_ratio),
                "fallback_reason": confidence.fallback_reason,
                "memory_age": memory_age,
                "memory_strength": memory_strength,
                "output_xywh": output_box.astype(float).tolist(),
            }
        )

    return replayed_boxes, counts, decisions


def compute_frame_confidence(
    candidate_xywh: np.ndarray,
    frame: dict[str, Any],
    config: ConfidenceMemoryConfig,
) -> FrameConfidence:
    """Score whether the stored DVS update is trustworthy on this frame."""

    candidate = np.asarray(candidate_xywh, dtype=float)
    proposed = frame_refiner_output_xywh(frame)
    raw_proposed = frame_raw_refined_xywh(frame, fallback=proposed)
    candidate_iou = box_iou_xywh(candidate, proposed)
    candidate_area_ratio = area_ratio_xywh(candidate, proposed)
    center_shift_ratio = center_shift_ratio_xywh(candidate, proposed)
    raw_candidate_iou = box_iou_xywh(candidate, raw_proposed)
    raw_candidate_area_ratio = area_ratio_xywh(candidate, raw_proposed)
    raw_center_shift_ratio = center_shift_ratio_xywh(candidate, raw_proposed)
    fallback_reason = _optional_str(frame.get("fallback_reason"))

    components = {
        "event_evidence": _saturating_fraction(
            _optional_float(frame.get("used_event_count")),
            config.reference_used_event_count,
        ),
        "active_measurements": _saturating_fraction(
            _optional_float(frame.get("active_measurement_count")),
            config.reference_active_measurement_count,
        ),
        "normal_flow_activity": _saturating_fraction(
            _optional_float(frame.get("mean_event_activity")),
            config.reference_mean_event_activity,
        ),
        "geometry": _geometry_confidence(
            candidate_iou,
            candidate_area_ratio,
            center_shift_ratio,
            config,
        ),
    }
    polarity_confidence = _polarity_confidence(frame)
    if polarity_confidence is not None:
        components["polarity"] = polarity_confidence

    score = _weighted_confidence(components)
    if fallback_reason is not None or not _finite_xywh(proposed):
        score = 0.0
    return FrameConfidence(
        score=float(score),
        components={key: float(value) for key, value in components.items()},
        candidate_iou=float(candidate_iou),
        candidate_area_ratio=float(candidate_area_ratio),
        center_shift_ratio=float(center_shift_ratio),
        raw_candidate_iou=float(raw_candidate_iou),
        raw_candidate_area_ratio=float(raw_candidate_area_ratio),
        raw_center_shift_ratio=float(raw_center_shift_ratio),
        fallback_reason=fallback_reason,
    )


def direct_rejection_reasons(
    confidence: FrameConfidence,
    frame: dict[str, Any],
    config: ConfidenceMemoryConfig,
) -> tuple[str, ...]:
    """Return direct-trust rejection reasons for a frame."""

    reasons: list[str] = []
    if confidence.fallback_reason is not None:
        reasons.append(f"fallback:{confidence.fallback_reason}")
    if confidence.score < config.direct_confidence_threshold:
        reasons.append("confidence")
    if not _finite_xywh(frame_refiner_output_xywh(frame)):
        reasons.append("refiner_output_invalid")
    if confidence.candidate_iou < config.min_direct_candidate_iou:
        reasons.append("candidate_iou")
    if confidence.candidate_area_ratio < config.min_direct_area_ratio:
        reasons.append("candidate_area_ratio")
    if confidence.candidate_area_ratio > config.max_direct_area_ratio:
        reasons.append("candidate_area_ratio")
    if confidence.center_shift_ratio > config.max_direct_center_shift_ratio:
        reasons.append("center_shift_ratio")
    return tuple(reasons)


def build_memory_box(
    memory: CorrectionMemory | None,
    candidate_xywh: np.ndarray,
    previous_candidate_xywh: np.ndarray,
    frame_index: int,
    confidence: FrameConfidence,
    config: ConfidenceMemoryConfig,
) -> tuple[np.ndarray, int, float] | None:
    """Apply the last trusted DVS correction if it is fresh and compatible."""

    if memory is None or config.max_memory_age <= 0:
        return None
    age = int(frame_index - memory.frame_index)
    if age <= 0 or age > config.max_memory_age:
        return None
    if confidence.score < config.memory_confidence_threshold:
        return None
    if not memory_is_motion_compatible(
        previous_candidate_xywh,
        candidate_xywh,
        config,
    ):
        return None

    strength = float(config.memory_alpha * (config.memory_decay ** max(age - 1, 0)))
    if strength <= 0.0:
        return None
    memory_box = apply_memory_correction(candidate_xywh, memory, strength)
    memory_box = _clip_xywh(memory_box, config)
    if not _finite_xywh(memory_box):
        return None
    return memory_box, age, strength


def memory_is_motion_compatible(
    previous_candidate_xywh: np.ndarray,
    candidate_xywh: np.ndarray,
    config: ConfidenceMemoryConfig,
) -> bool:
    """Check whether base-tracker motion is smooth enough to carry memory."""

    previous = np.asarray(previous_candidate_xywh, dtype=float)
    current = np.asarray(candidate_xywh, dtype=float)
    if not _finite_xywh(previous) or not _finite_xywh(current):
        return False
    if box_iou_xywh(previous, current) < config.min_memory_base_iou:
        return False
    if center_shift_ratio_xywh(previous, current) > config.max_memory_base_center_shift_ratio:
        return False
    ratios = current[2:] / np.maximum(previous[2:], 1e-9)
    ratios = np.maximum(ratios, 1.0 / np.maximum(ratios, 1e-9))
    return bool(np.all(ratios <= config.max_memory_scale_change_ratio))


def apply_memory_correction(
    candidate_xywh: np.ndarray,
    memory: CorrectionMemory,
    strength: float,
) -> np.ndarray:
    """Apply a decayed center/scale correction from the last trusted frame."""

    candidate = np.asarray(candidate_xywh, dtype=float)
    base_center = _center_xywh(memory.base_xywh)
    trusted_center = _center_xywh(memory.trusted_xywh)
    current_center = _center_xywh(candidate)
    center_delta = trusted_center - base_center
    size_ratio = memory.trusted_xywh[2:] / np.maximum(memory.base_xywh[2:], 1e-9)
    log_size_ratio = np.log(np.maximum(size_ratio, 1e-9))
    output_size = candidate[2:] * np.exp(float(strength) * log_size_ratio)
    output_center = current_center + float(strength) * center_delta
    return np.asarray(
        [
            output_center[0] - 0.5 * output_size[0],
            output_center[1] - 0.5 * output_size[1],
            output_size[0],
            output_size[1],
        ],
        dtype=float,
    )


def write_decisions_csv(path: Path, decisions: list[dict[str, Any]]) -> None:
    """Write per-frame confidence/memory decisions for validation analysis."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not decisions:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in decisions for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in decisions:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def _decision_record(
    sequence_name: str,
    frame_index: int,
    action: str,
    output_xywh: np.ndarray,
) -> dict[str, Any]:
    return {
        "sequence": sequence_name,
        "frame_index": int(frame_index),
        "action": action,
        "reason": action,
        "direct_rejection_reasons": [],
        "confidence": 0.0,
        "confidence_components": {},
        "memory_age": None,
        "memory_strength": None,
        "output_xywh": np.asarray(output_xywh, dtype=float).tolist(),
    }


def _weighted_confidence(components: dict[str, float]) -> float:
    weights = {
        "event_evidence": 0.25,
        "active_measurements": 0.25,
        "normal_flow_activity": 0.20,
        "geometry": 0.30,
        "polarity": 0.10,
    }
    numerator = 0.0
    denominator = 0.0
    for name, value in components.items():
        weight = weights.get(name, 0.0)
        numerator += weight * float(np.clip(value, 0.0, 1.0))
        denominator += weight
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _geometry_confidence(
    candidate_iou: float,
    candidate_area_ratio: float,
    center_shift_ratio: float,
    config: ConfidenceMemoryConfig,
) -> float:
    area_score = _area_compatibility(candidate_area_ratio)
    shift_reference = max(2.0 * config.max_direct_center_shift_ratio, 1e-9)
    shift_score = 1.0 - float(center_shift_ratio) / shift_reference
    return float(np.clip(candidate_iou * area_score * np.clip(shift_score, 0.0, 1.0), 0.0, 1.0))


def _polarity_confidence(frame: dict[str, Any]) -> float | None:
    values = [
        _optional_float(frame.get("polarity_consistency_fraction")),
        _optional_float(frame.get("mean_event_polarity_weight")),
    ]
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return float(np.clip(np.mean(finite_values), 0.0, 1.0))


def _area_compatibility(area_ratio: float) -> float:
    if not math.isfinite(area_ratio) or area_ratio <= 0.0:
        return 0.0
    return float(np.clip(min(area_ratio, 1.0 / area_ratio), 0.0, 1.0))


def _saturating_fraction(value: float | None, reference: float) -> float:
    if value is None or not math.isfinite(reference) or reference <= 0.0:
        return 0.0
    return float(np.clip(float(value) / float(reference), 0.0, 1.0))


def _blend_xywh(base_xywh: np.ndarray, target_xywh: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * np.asarray(base_xywh, dtype=float) + alpha * np.asarray(
        target_xywh,
        dtype=float,
    )


def _clip_xywh(box_xywh: np.ndarray, config: ConfidenceMemoryConfig) -> np.ndarray:
    box = np.asarray(box_xywh, dtype=float).copy()
    if not config.clip_to_image:
        return box
    if config.image_width is None or config.image_height is None:
        return box
    width = max(float(config.image_width), 1.0)
    height = max(float(config.image_height), 1.0)
    x1 = float(np.clip(box[0], 0.0, width))
    y1 = float(np.clip(box[1], 0.0, height))
    x2 = float(np.clip(box[0] + box[2], 0.0, width))
    y2 = float(np.clip(box[1] + box[3], 0.0, height))
    if x2 <= x1 or y2 <= y1:
        return box
    return np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=float)


def _center_xywh(box_xywh: np.ndarray) -> np.ndarray:
    box = np.asarray(box_xywh, dtype=float)
    return np.asarray([box[0] + 0.5 * box[2], box[1] + 0.5 * box[3]], dtype=float)


def _finite_xywh(box_xywh: np.ndarray) -> bool:
    values = np.asarray(box_xywh, dtype=float)
    return values.shape == (4,) and np.all(np.isfinite(values)) and bool(np.all(values[2:] > 0.0))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
