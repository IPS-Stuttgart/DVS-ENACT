"""Replay EventVOT DVS-ENACT diagnostics as a confidence/memory gate.

This utility is complementary to ``run_eventvot_acceptance_replay.py``. It still
uses the stored DVS-ENACT per-frame diagnostics as confidence evidence, but when
DVS evidence rejects a tracker update for configured geometry-disagreement
reasons, the output can fall back to a short-term memory box instead of simply
writing the external tracker box.
"""

from __future__ import annotations

import argparse
import json
import re
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
    ReplayAcceptanceConfig,
    acceptance_config_from_diagnostics,
    decision_to_dict,
    evaluate_frame_acceptance,
    frame_refiner_output_xywh,
    resolve_replay_base_result_file,
    select_sequence_summaries,
    write_decisions_csv,
)
from run_eventvot_refinement import (  # noqa: E402
    load_xywh_result_file,
    register_tracker_in_config,
    resolve_eventvot_split_root,
    resolve_output_results_root,
    save_xywh_result_file,
    xywh_to_diagnostic_bbox,
)
from run_eventvot_validation_sweep import evaluate_eventvot_results  # noqa: E402

DEFAULT_GATE_REJECTION_REASONS = (
    "candidate_iou",
    "candidate_area_ratio",
    "center_shift_ratio",
    "raw_candidate_iou",
    "raw_candidate_area_ratio",
    "raw_center_shift_ratio",
)


@dataclass(frozen=True)
class ReplayConfidenceGateConfig:
    """Policy for memory fallback during confidence-gate replay."""

    enabled: bool = True
    motion_model: str = "hold"
    gate_rejection_reasons: tuple[str, ...] = DEFAULT_GATE_REJECTION_REASONS
    max_consecutive_memory_frames: int = 5
    apply_box_refinement: bool = True


@dataclass(frozen=True)
class EventVOTConfidenceGateReplayOptions:
    """Inputs and outputs for one confidence-gate replay run."""

    diagnostics_json: Path
    output_results: Path
    eventvot_root: Path | None = None
    base_results: Path | None = None
    split: str | None = None
    sequences: tuple[str, ...] = ()
    summary_json: Path | None = None
    decisions_csv: Path | None = None
    skip_evaluation: bool = False
    tracker_name: str | None = None
    config_tracker_path: Path | None = None
    acceptance_config: ReplayAcceptanceConfig | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    gate_config: ReplayConfidenceGateConfig = field(
        default_factory=ReplayConfidenceGateConfig
    )


def main() -> int:
    args = build_parser().parse_args()
    payload = run(options_from_args(args))
    print(json.dumps(payload["summary"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay stored EventVOT DVS-ENACT diagnostics as a confidence/memory "
            "gate without recomputing refinements."
        )
    )
    parser.add_argument("--diagnostics-json", type=Path, required=True)
    parser.add_argument("--output-results", type=Path)
    parser.add_argument("--eventvot-toolkit-root", type=Path)
    parser.add_argument("--tracker-name")
    parser.add_argument("--config-tracker", type=Path)
    parser.add_argument("--update-config-tracker", action="store_true")
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
        help="Treat every non-fallback DVS-ENACT output as confident.",
    )
    _add_policy_arguments(parser)
    _add_gate_arguments(parser)
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
    parser.add_argument(
        "--min-raw-area-ratio",
        dest="min_raw_candidate_area_ratio",
        type=float,
    )
    parser.add_argument(
        "--max-raw-area-ratio",
        dest="max_raw_candidate_area_ratio",
        type=float,
    )
    parser.add_argument(
        "--max-raw-center-shift-ratio",
        dest="max_raw_center_shift_ratio",
        type=float,
    )
    parser.add_argument("--min-polarity-consistency-fraction", type=float)
    parser.add_argument("--min-mean-event-polarity-weight", type=float)
    parser.add_argument("--max-quadratic-form-per-active-measurement", type=float)
    parser.add_argument("--min-active-fraction", type=float)


def _add_gate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--disable-confidence-gate",
        action="store_true",
        help="Disable memory fallback and only replay accepted refinements/base boxes.",
    )
    parser.add_argument(
        "--confidence-only",
        action="store_true",
        help=(
            "Use DVS-ENACT only as confidence evidence. Confident frames keep the "
            "external tracker box instead of writing DVS-refined coordinates."
        ),
    )
    parser.add_argument(
        "--gate-motion-model",
        choices=("hold", "constant_velocity"),
        default="hold",
        help="Memory prediction used when a frame is gated.",
    )
    parser.add_argument(
        "--max-consecutive-memory-frames",
        type=int,
        default=5,
        help="Maximum consecutive memory-fallback frames before passing base boxes.",
    )
    parser.add_argument(
        "--gate-rejection-reason",
        action="append",
        default=[],
        help=(
            "Acceptance rejection reason that should trigger memory fallback. Can "
            "be repeated or comma-separated. Defaults to geometry-disagreement "
            "reasons such as candidate_iou and center_shift_ratio."
        ),
    )


def options_from_args(args: argparse.Namespace) -> EventVOTConfidenceGateReplayOptions:
    output_results = _resolve_cli_output_results(args)
    override_keys = set(asdict(ReplayAcceptanceConfig()))
    overrides = {
        key: getattr(args, key)
        for key in override_keys
        if hasattr(args, key) and getattr(args, key) is not None
    }
    if args.disable_conservative_gates:
        overrides["enabled"] = False
    return EventVOTConfidenceGateReplayOptions(
        diagnostics_json=args.diagnostics_json,
        output_results=output_results,
        eventvot_root=args.eventvot_root,
        base_results=args.base_results,
        split=args.split,
        sequences=tuple(args.sequence),
        summary_json=args.summary_json,
        decisions_csv=args.decisions_csv,
        skip_evaluation=args.skip_evaluation,
        tracker_name=args.tracker_name,
        config_tracker_path=_resolve_cli_config_tracker_path(args),
        config_overrides=overrides,
        gate_config=ReplayConfidenceGateConfig(
            enabled=not args.disable_confidence_gate,
            motion_model=args.gate_motion_model,
            gate_rejection_reasons=parse_gate_rejection_reasons(
                args.gate_rejection_reason
            ),
            max_consecutive_memory_frames=args.max_consecutive_memory_frames,
            apply_box_refinement=not args.confidence_only,
        ),
    )


def run(options: EventVOTConfidenceGateReplayOptions) -> dict[str, Any]:
    """Replay stored diagnostics with a confidence/memory-gating policy."""
    _validate_gate_config(options.gate_config)
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
        raise ValueError("No sequence diagnostics selected for confidence-gate replay")

    output_root = resolve_output_results_root(
        options.output_results,
        tracker_name=options.tracker_name,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    config_tracker_updated = False
    if options.config_tracker_path is not None:
        if options.tracker_name is None:
            raise ValueError("--config-tracker requires --tracker-name")
        config_tracker_updated = register_tracker_in_config(
            options.config_tracker_path,
            options.tracker_name,
        )

    image_width = float(diagnostics.get("refiner_config", {}).get("image_width", 1280.0))
    image_height = float(diagnostics.get("refiner_config", {}).get("image_height", 720.0))

    decisions: list[dict[str, Any]] = []
    sequence_outputs: list[dict[str, Any]] = []
    aggregate_action_counts: Counter[str] = Counter()
    aggregate_rejection_counts: Counter[str] = Counter()
    sequence_names: list[str] = []

    for sequence_summary in selected_summaries:
        sequence_name = str(sequence_summary["sequence"])
        sequence_names.append(sequence_name)
        base_result_file = resolve_replay_base_result_file(
            sequence_summary,
            options.base_results,
        )
        base_boxes = load_xywh_result_file(base_result_file)
        replayed_boxes, summary, sequence_decisions = replay_sequence_with_memory(
            sequence_name,
            base_boxes,
            sequence_summary.get("frames", []),
            config,
            options.gate_config,
            image_width=image_width,
            image_height=image_height,
        )
        output_file = output_root / f"{sequence_name}.txt"
        save_xywh_result_file(output_file, replayed_boxes)
        aggregate_action_counts.update(summary["confidence_gate_action_counts"])
        aggregate_rejection_counts.update(summary["rejection_counts"])
        decisions.extend(sequence_decisions)
        sequence_outputs.append(
            {
                "sequence": sequence_name,
                "base_result_file": str(base_result_file),
                "output_result_file": str(output_file),
                **summary,
            }
        )

    metrics = None
    split = options.split or diagnostics.get("options", {}).get("split", "test")
    if not options.skip_evaluation:
        if options.eventvot_root is None:
            raise ValueError("--eventvot-root is required unless --skip-evaluation is used")
        split_root = resolve_eventvot_split_root(options.eventvot_root, str(split))
        metrics = evaluate_eventvot_results(split_root, output_root, sequence_names)

    if options.decisions_csv is not None:
        write_decisions_csv(options.decisions_csv, decisions)

    summary = {
        "sequence_count": len(sequence_outputs),
        "frame_count": int(sum(item["frame_count"] for item in sequence_outputs)),
        "output_changed_frame_count": int(
            sum(item["output_changed_frame_count"] for item in sequence_outputs)
        ),
        "accepted_refinement_count": int(
            sum(item["accepted_refinement_count"] for item in sequence_outputs)
        ),
        "dvs_confident_frame_count": int(
            sum(item["dvs_confident_frame_count"] for item in sequence_outputs)
        ),
        "memory_gate_count": int(
            sum(item["memory_gate_count"] for item in sequence_outputs)
        ),
        "confidence_gate_action_counts": dict(sorted(aggregate_action_counts.items())),
        "rejection_counts": dict(sorted(aggregate_rejection_counts.items())),
        "metrics": metrics,
        "output_results": str(output_root),
    }
    payload = {
        "schema_version": 1,
        "description": (
            "EventVOT DVS-ENACT confidence/memory-gate replay from stored "
            "per-frame diagnostics. Geometry-inconsistent DVS evidence can "
            "hold or extrapolate a short-term memory box instead of blindly "
            "passing the external tracker output."
        ),
        "split": split,
        "diagnostics_json": str(options.diagnostics_json),
        "eventvot_evaluator": {
            "tracker_name": options.tracker_name,
            "tracking_result_dir": str(output_root),
            "config_tracker_updated": config_tracker_updated,
        },
        "acceptance_config": asdict(config),
        "confidence_gate_config": asdict(options.gate_config),
        "summary": summary,
        "sequences": sequence_outputs,
    }
    summary_json = options.summary_json or output_root / "confidence_gate_replay_summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def replay_sequence_with_memory(
    sequence_name: str,
    base_boxes: np.ndarray,
    frames: list[dict[str, Any]],
    config: ReplayAcceptanceConfig,
    gate_config: ReplayConfidenceGateConfig,
    *,
    image_width: float,
    image_height: float,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Return memory-gated boxes, summary counts, and per-frame decisions."""
    replayed_boxes = np.array(base_boxes, dtype=float, copy=True)
    indexed_frames = {
        int(frame.get("frame_index", -1)): frame
        for frame in frames
        if int(frame.get("frame_index", -1)) >= 0
    }
    action_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    decisions: list[dict[str, Any]] = []
    memory_box = np.asarray(base_boxes[0], dtype=float).copy()
    previous_memory_box: np.ndarray | None = None
    consecutive_memory_frames = 0

    for frame_index in range(int(base_boxes.shape[0])):
        if frame_index == 0:
            action_counts["initial_frame"] += 1
            continue
        frame = indexed_frames.get(frame_index)
        if frame is None:
            action_counts["missing_frame_diagnostics"] += 1
            previous_memory_box, memory_box = memory_box.copy(), base_boxes[frame_index]
            consecutive_memory_frames = 0
            continue

        decision = evaluate_frame_acceptance(base_boxes[frame_index], frame, config)
        rejection_counts.update(decision.rejection_reasons)
        memory_prediction = predict_memory_box(
            memory_box,
            previous_memory_box,
            gate_config.motion_model,
            image_width,
            image_height,
        )
        gate_reasons = configured_gate_rejection_reasons(decision, gate_config)
        blocked_reasons: tuple[str, ...] = ()

        if decision.accepted and gate_config.apply_box_refinement:
            output_xywh = frame_refiner_output_xywh(frame)
            action = "accepted_refinement"
            memory_gate_applied = False
            accepted_refinement = True
        elif decision.accepted:
            output_xywh = np.asarray(base_boxes[frame_index], dtype=float)
            action = "accepted_base"
            memory_gate_applied = False
            accepted_refinement = False
        elif not gate_config.enabled:
            output_xywh = np.asarray(base_boxes[frame_index], dtype=float)
            action = "base_passthrough"
            memory_gate_applied = False
            accepted_refinement = False
            blocked_reasons = ("disabled",)
        elif consecutive_memory_frames >= gate_config.max_consecutive_memory_frames:
            output_xywh = np.asarray(base_boxes[frame_index], dtype=float)
            action = "base_passthrough"
            memory_gate_applied = False
            accepted_refinement = False
            blocked_reasons = ("max_consecutive_memory_frames",)
        elif gate_reasons:
            output_xywh = memory_prediction
            action = f"memory_{gate_config.motion_model}"
            memory_gate_applied = True
            accepted_refinement = False
        else:
            output_xywh = np.asarray(base_boxes[frame_index], dtype=float)
            action = "base_passthrough"
            memory_gate_applied = False
            accepted_refinement = False
            blocked_reasons = ("no_configured_gate_rejection_reason",)

        replayed_boxes[frame_index] = output_xywh
        action_counts[action] += 1
        previous_memory_box, memory_box = memory_box.copy(), output_xywh.copy()
        consecutive_memory_frames = (
            consecutive_memory_frames + 1 if memory_gate_applied else 0
        )
        decisions.append(
            {
                "sequence": sequence_name,
                "frame_index": frame_index,
                **decision_to_dict(decision),
                "dvs_confident": bool(decision.accepted),
                "accepted_refinement": bool(accepted_refinement),
                "confidence_gate_action": action,
                "memory_gate_applied": bool(memory_gate_applied),
                "gate_rejection_reasons": list(gate_reasons),
                "gate_blocked_reasons": list(blocked_reasons),
                "memory_prediction_xywh": memory_prediction.astype(float).tolist(),
                "memory_prediction_bbox": xywh_to_diagnostic_bbox(memory_prediction),
                "output_xywh": output_xywh.astype(float).tolist(),
                "output_bbox": xywh_to_diagnostic_bbox(output_xywh),
                "consecutive_memory_frames": int(consecutive_memory_frames),
            }
        )

    changed_frame_count = int(
        np.any(~np.isclose(replayed_boxes, base_boxes, rtol=1e-6, atol=1e-6), axis=1).sum()
    )
    summary = {
        "frame_count": int(base_boxes.shape[0]),
        "output_changed_frame_count": changed_frame_count,
        "accepted_refinement_count": int(action_counts.get("accepted_refinement", 0)),
        "dvs_confident_frame_count": int(
            action_counts.get("accepted_refinement", 0)
            + action_counts.get("accepted_base", 0)
        ),
        "memory_gate_count": int(
            action_counts.get("memory_hold", 0)
            + action_counts.get("memory_constant_velocity", 0)
        ),
        "confidence_gate_action_counts": dict(sorted(action_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }
    return replayed_boxes, summary, decisions


def configured_gate_rejection_reasons(
    decision: Any,
    gate_config: ReplayConfidenceGateConfig,
) -> tuple[str, ...]:
    """Return rejection reasons configured to trigger memory fallback."""
    return tuple(
        reason
        for reason in decision.rejection_reasons
        if reason in gate_config.gate_rejection_reasons
    )


def predict_memory_box(
    memory_box: np.ndarray,
    previous_memory_box: np.ndarray | None,
    motion_model: str,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Predict the short-term memory box used by memory fallback."""
    if motion_model == "hold" or previous_memory_box is None:
        prediction = np.asarray(memory_box, dtype=float).copy()
    elif motion_model == "constant_velocity":
        prediction = np.asarray(memory_box, dtype=float) + (
            np.asarray(memory_box, dtype=float) - np.asarray(previous_memory_box, dtype=float)
        )
    else:
        raise ValueError(f"Unsupported gate motion model: {motion_model}")
    return clip_xywh_to_image(prediction, image_width, image_height)


def clip_xywh_to_image(
    box_xywh: np.ndarray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Clip a predicted xywh box to the image extent."""
    box = np.asarray(box_xywh, dtype=float).copy()
    if box.shape != (4,):
        raise ValueError(f"Expected an xywh box with shape (4,), got {box.shape}")
    width = max(1e-6, float(box[2]))
    height = max(1e-6, float(box[3]))
    if image_width > 0.0:
        width = min(width, float(image_width))
        box[0] = float(np.clip(box[0], 0.0, max(0.0, image_width - width)))
    if image_height > 0.0:
        height = min(height, float(image_height))
        box[1] = float(np.clip(box[1], 0.0, max(0.0, image_height - height)))
    box[2] = width
    box[3] = height
    return box


def parse_gate_rejection_reasons(items: Iterable[str]) -> tuple[str, ...]:
    """Parse repeated or comma/whitespace-separated gate rejection reasons."""
    reasons: list[str] = []
    for item in items:
        reasons.extend(token for token in re.split(r"[\s,]+", item.strip()) if token)
    if not reasons:
        return DEFAULT_GATE_REJECTION_REASONS
    unique: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason in seen:
            continue
        unique.append(reason)
        seen.add(reason)
    return tuple(unique)


def _resolve_cli_output_results(args: argparse.Namespace) -> Path:
    if args.eventvot_toolkit_root is not None:
        return args.eventvot_toolkit_root
    if args.output_results is not None:
        return args.output_results
    raise SystemExit("--output-results or --eventvot-toolkit-root is required")


def _resolve_cli_config_tracker_path(args: argparse.Namespace) -> Path | None:
    if args.config_tracker is not None:
        return args.config_tracker
    if args.update_config_tracker:
        if args.eventvot_toolkit_root is None:
            raise SystemExit("--update-config-tracker requires --eventvot-toolkit-root")
        if args.tracker_name is None:
            raise SystemExit("--update-config-tracker requires --tracker-name")
        return args.eventvot_toolkit_root / "utils" / "config_tracker.m"
    return None


def _validate_gate_config(gate_config: ReplayConfidenceGateConfig) -> None:
    if gate_config.motion_model not in {"hold", "constant_velocity"}:
        raise ValueError(f"Unsupported gate motion model: {gate_config.motion_model}")
    if gate_config.max_consecutive_memory_frames < 0:
        raise ValueError("max_consecutive_memory_frames must be non-negative")


if __name__ == "__main__":
    raise SystemExit(main())
