"""Refine EventVOT tracker result files with the DVS-ENACT contour refiner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dvs_enact import (
    DVSContourRefiner,
    DVSContourRefinerConfig,
    EventBatch,
    empty_event_batch,
)


@dataclass(frozen=True)
class EventVOTAcceptanceConfig:
    """Conservative gates for accepting post-hoc EventVOT refinements."""

    enabled: bool = True
    min_used_event_count: int = 10
    min_active_measurement_count: int = 3
    min_mean_event_activity: float = 0.10
    min_candidate_iou: float = 0.60
    max_candidate_area_ratio: float = 1.50


@dataclass(frozen=True)
class EventVOTAcceptanceDecision:
    """Decision record for one candidate/refined EventVOT box pair."""

    accepted: bool
    rejection_reasons: tuple[str, ...]
    candidate_iou: float
    candidate_area_ratio: float


@dataclass(frozen=True)
class EventVOTRefinementOptions:
    """Filesystem and event-parsing options for EventVOT post-processing."""

    eventvot_root: Path
    base_results: Path
    output_results: Path
    split: str = "test"
    sequences: tuple[str, ...] = ()
    tracker_name: str | None = None
    event_column_order: str = "auto"
    diagnostics_json: Path | None = None
    config_tracker_path: Path | None = None
    acceptance_config: EventVOTAcceptanceConfig = field(
        default_factory=EventVOTAcceptanceConfig
    )


def default_eventvot_refiner() -> DVSContourRefiner:
    """Return the recommended EventVOT refiner configuration."""
    return DVSContourRefiner(
        DVSContourRefinerConfig(
            input_bbox_format="xywh",
            output_bbox_format="xywh",
            image_width=1280,
            image_height=720,
            search_expansion_factor=1.25,
            max_events=128,
            min_events=3,
            use_event_polarity=True,
            refinement_blend=0.25,
        )
    )


def load_xywh_result_file(path: Path) -> np.ndarray:
    """Load an EventVOT/HDETrack result file as an ``N x 4`` xywh array."""
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        values = _parse_numeric_tokens(line)
        if len(values) < 4:
            raise ValueError(f"Result row has fewer than four columns in {path}: {line}")
        rows.append([float(value) for value in values[:4]])
    if not rows:
        raise ValueError(f"No result boxes found in {path}")
    return np.asarray(rows, dtype=float)


def save_xywh_result_file(path: Path, boxes: np.ndarray) -> None:
    """Write xywh boxes in a format accepted by the EventVOT evaluator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(boxes, dtype=float), delimiter="\t", fmt="%.6f")


def read_eventvot_event_time_span(
    event_csv: Path,
    *,
    event_column_order: str = "auto",
) -> tuple[int, int, int]:
    """Return first timestamp, last timestamp, and parsed event count."""
    first_ts: int | None = None
    last_ts: int | None = None
    count = 0
    for timestamp, _x, _y, _polarity in iter_eventvot_events(
        event_csv,
        event_column_order=event_column_order,
    ):
        if first_ts is None:
            first_ts = timestamp
        last_ts = timestamp
        count += 1
    if first_ts is None or last_ts is None:
        raise ValueError(f"No parseable events found in {event_csv}")
    return int(first_ts), int(last_ts), int(count)


def iter_eventvot_frame_windows(
    event_csv: Path,
    frame_count: int,
    *,
    event_column_order: str = "auto",
) -> Iterator[tuple[int, EventBatch]]:
    """Yield raw events between frame ``k - 1`` and frame ``k``.

    EventVOT raw CSVs do not ship with a separate timestamp file in the expected
    benchmark layout. The official conversion scripts split each sequence-long
    stream into evenly spaced temporal bins, so this adapter reconstructs frame
    timestamps by linearly spacing the raw event time range over the result
    length. Frame 0 is the external tracker's initialization and is not yielded.
    """
    if frame_count <= 1:
        return
    first_ts, last_ts, _event_count = read_eventvot_event_time_span(
        event_csv,
        event_column_order=event_column_order,
    )
    if last_ts <= first_ts:
        for frame_index in range(1, frame_count):
            yield frame_index, empty_event_batch()
        return

    frame_times = np.linspace(float(first_ts), float(last_ts), frame_count)
    current_frame = 1
    timestamps: list[int] = []
    xs: list[int] = []
    ys: list[int] = []
    polarities: list[int] = []

    for timestamp, x_value, y_value, polarity in iter_eventvot_events(
        event_csv,
        event_column_order=event_column_order,
    ):
        while current_frame < frame_count - 1 and timestamp >= frame_times[current_frame]:
            yield current_frame, _event_batch_from_lists(timestamps, xs, ys, polarities)
            timestamps, xs, ys, polarities = [], [], [], []
            current_frame += 1

        if timestamp < frame_times[current_frame - 1]:
            continue
        if current_frame == frame_count - 1 and timestamp > frame_times[current_frame]:
            continue
        timestamps.append(int(timestamp))
        xs.append(int(x_value))
        ys.append(int(y_value))
        polarities.append(int(polarity))

    while current_frame < frame_count:
        yield current_frame, _event_batch_from_lists(timestamps, xs, ys, polarities)
        timestamps, xs, ys, polarities = [], [], [], []
        current_frame += 1


def iter_eventvot_events(
    event_csv: Path,
    *,
    event_column_order: str = "auto",
) -> Iterator[tuple[int, int, int, int]]:
    """Stream EventVOT raw events as ``timestamp, x, y, polarity`` tuples."""
    with event_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            values = _numeric_row(row)
            if values is None:
                continue
            schema = _resolve_event_schema(event_csv, values, event_column_order)
            parsed = _parse_event_row(values, schema)
            if parsed is not None:
                yield parsed


def refine_sequence(
    sequence_name: str,
    sequence_dir: Path,
    base_result_file: Path,
    output_result_file: Path,
    refiner: DVSContourRefiner,
    *,
    event_column_order: str = "auto",
    acceptance_config: EventVOTAcceptanceConfig | None = None,
) -> dict[str, Any]:
    """Refine one EventVOT sequence result file."""
    acceptance_config = acceptance_config or EventVOTAcceptanceConfig()
    event_csv = find_sequence_event_csv(sequence_dir, sequence_name)
    base_boxes = load_xywh_result_file(base_result_file)
    frame_count = int(base_boxes.shape[0])
    _validate_sequence_frame_count(sequence_dir, sequence_name, frame_count)

    refined_boxes = np.array(base_boxes, dtype=float, copy=True)
    timings = np.zeros(frame_count, dtype=float)
    frames: list[dict[str, Any]] = [
        {
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
            "candidate_iou": 1.0,
            "candidate_area_ratio": 1.0,
        }
    ]

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
        decision = evaluate_refinement_acceptance(
            base_boxes[frame_index],
            result,
            acceptance_config,
        )
        refined_boxes[frame_index] = (
            refiner_output
            if decision.accepted
            else np.asarray(base_boxes[frame_index], dtype=float)
        )
        frame_record = result.to_dict()
        refiner_output_bbox = frame_record.get("output_bbox")
        frame_record.update(
            {
                "frame_index": int(frame_index),
                "accept_refinement": decision.accepted,
                "rejection_reasons": list(decision.rejection_reasons),
                "candidate_iou": float(decision.candidate_iou),
                "candidate_area_ratio": float(decision.candidate_area_ratio),
                "refiner_output_bbox": refiner_output_bbox,
                "refiner_output_xywh": refiner_output.astype(float).tolist(),
                "output_bbox": xywh_to_diagnostic_bbox(refined_boxes[frame_index]),
                "output_xywh": refined_boxes[frame_index].astype(float).tolist(),
                "elapsed_seconds": float(timings[frame_index]),
            }
        )
        frames.append(frame_record)

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


def run(options: EventVOTRefinementOptions, refiner: DVSContourRefiner | None = None) -> dict:
    """Run DVS-ENACT refinement over one or more EventVOT result files."""
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
            "Post-hoc EventVOT result refinement: external tracker xywh boxes "
            "plus raw EventVOT events between frame k-1 and frame k are passed "
            "through DVSContourRefiner."
        ),
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
        "summary": summarize_sequence_results(summaries),
        "sequences": summaries,
    }
    diagnostics_path = options.diagnostics_json or default_diagnostics_path(output_root)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def summarize_sequence_results(sequence_summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    sequence_summaries = list(sequence_summaries)
    fallback_counts: Counter[str] = Counter()
    acceptance_counts: Counter[str] = Counter()
    for summary in sequence_summaries:
        fallback_counts.update(summary["fallback_counts"])
        acceptance_counts.update(summary["acceptance_counts"])
    frame_count = sum(int(summary["frame_count"]) for summary in sequence_summaries)
    refined_frame_count = sum(
        int(summary["refined_frame_count"]) for summary in sequence_summaries
    )
    accepted_refinement_count = sum(
        int(summary["accepted_refinement_count"]) for summary in sequence_summaries
    )
    refiner_success_frame_count = sum(
        int(summary["refiner_success_frame_count"]) for summary in sequence_summaries
    )
    return {
        "sequence_count": len(sequence_summaries),
        "frame_count": int(frame_count),
        "refined_frame_count": int(refined_frame_count),
        "accepted_refinement_count": int(accepted_refinement_count),
        "refiner_success_frame_count": int(refiner_success_frame_count),
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "acceptance_counts": dict(sorted(acceptance_counts.items())),
    }


def evaluate_refinement_acceptance(
    candidate_xywh: np.ndarray,
    result: Any,
    config: EventVOTAcceptanceConfig | None = None,
) -> EventVOTAcceptanceDecision:
    """Return whether a DVS-ENACT refinement should replace the base box."""
    config = config or EventVOTAcceptanceConfig()
    refined_xywh = np.asarray(result.as_xywh(), dtype=float)
    candidate_iou = box_iou_xywh(candidate_xywh, refined_xywh)
    candidate_area_ratio = area_ratio_xywh(candidate_xywh, refined_xywh)
    if not config.enabled:
        rejection_reasons = () if result.fallback_reason is None else ("fallback_reason",)
        return EventVOTAcceptanceDecision(
            accepted=result.fallback_reason is None,
            rejection_reasons=rejection_reasons,
            candidate_iou=candidate_iou,
            candidate_area_ratio=candidate_area_ratio,
        )

    rejection_reasons: list[str] = []
    if result.fallback_reason is not None:
        rejection_reasons.append(f"fallback:{result.fallback_reason}")
    if int(result.used_event_count) < config.min_used_event_count:
        rejection_reasons.append("used_event_count")
    if int(result.active_measurement_count) < config.min_active_measurement_count:
        rejection_reasons.append("active_measurement_count")
    if result.mean_event_activity is None:
        rejection_reasons.append("mean_event_activity_missing")
    elif float(result.mean_event_activity) < config.min_mean_event_activity:
        rejection_reasons.append("mean_event_activity")
    if candidate_iou < config.min_candidate_iou:
        rejection_reasons.append("candidate_iou")
    if candidate_area_ratio > config.max_candidate_area_ratio:
        rejection_reasons.append("candidate_area_ratio")

    return EventVOTAcceptanceDecision(
        accepted=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
        candidate_iou=candidate_iou,
        candidate_area_ratio=candidate_area_ratio,
    )


def box_iou_xywh(first_xywh: np.ndarray, second_xywh: np.ndarray) -> float:
    """Return IoU for two ``x,y,width,height`` boxes."""
    first = np.asarray(first_xywh, dtype=float)
    second = np.asarray(second_xywh, dtype=float)
    first_area = _box_area_xywh(first)
    second_area = _box_area_xywh(second)
    if first_area <= 0.0 or second_area <= 0.0:
        return 0.0
    first_x2 = first[0] + first[2]
    first_y2 = first[1] + first[3]
    second_x2 = second[0] + second[2]
    second_y2 = second[1] + second[3]
    intersection_width = max(0.0, min(first_x2, second_x2) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def area_ratio_xywh(reference_xywh: np.ndarray, proposed_xywh: np.ndarray) -> float:
    """Return proposed-box area divided by reference-box area."""
    reference_area = _box_area_xywh(np.asarray(reference_xywh, dtype=float))
    proposed_area = _box_area_xywh(np.asarray(proposed_xywh, dtype=float))
    if reference_area <= 0.0:
        return math.inf
    return float(proposed_area / reference_area)


def xywh_to_diagnostic_bbox(box_xywh: np.ndarray) -> dict[str, float]:
    """Return a JSON-friendly bbox dictionary for an ``xywh`` box."""
    box = np.asarray(box_xywh, dtype=float)
    return {
        "x_min": float(box[0]),
        "y_min": float(box[1]),
        "width": float(box[2]),
        "height": float(box[3]),
        "x_max": float(box[0] + box[2]),
        "y_max": float(box[1] + box[3]),
    }


def _box_area_xywh(box_xywh: np.ndarray) -> float:
    return float(max(0.0, float(box_xywh[2])) * max(0.0, float(box_xywh[3])))


def resolve_eventvot_split_root(eventvot_root: Path, split: str) -> Path:
    """Resolve the EventVOT split directory from common extraction layouts."""
    subset_names = {
        "test": "Testing Subset",
        "train": "Training Subset",
        "val": "validating Subset",
        "validating": "validating Subset",
    }
    candidates = [
        eventvot_root / split,
        eventvot_root / split.lower(),
        eventvot_root,
    ]
    if split.lower() in subset_names:
        candidates.insert(2, eventvot_root / subset_names[split.lower()])
    for candidate in candidates:
        if candidate.exists() and (
            (candidate / "list.txt").exists()
            or any(path.is_dir() for path in candidate.iterdir())
        ):
            return candidate
    raise FileNotFoundError(f"Could not resolve EventVOT split '{split}' below {eventvot_root}")


def resolve_sequence_names(
    split_root: Path,
    base_results: Path,
    *,
    requested_sequences: tuple[str, ...] = (),
) -> list[str]:
    if requested_sequences:
        return list(requested_sequences)
    if base_results.is_file():
        return [base_results.stem]
    if (split_root / "list.txt").exists():
        return [
            line.strip()
            for line in (split_root / "list.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if base_results.exists():
        return [
            path.stem
            for path in sorted(base_results.glob("*.txt"))
            if not _is_auxiliary_result_file(path)
        ]
    return sorted(path.name for path in split_root.iterdir() if path.is_dir())


def resolve_base_result_file(base_results: Path, sequence_name: str) -> Path:
    result_file = base_results if base_results.is_file() else base_results / f"{sequence_name}.txt"
    if not result_file.exists():
        raise FileNotFoundError(f"Base result file does not exist: {result_file}")
    return result_file


def resolve_output_results_root(
    output_results: Path,
    *,
    tracker_name: str | None = None,
) -> Path:
    """Resolve the directory that should contain per-sequence result files."""
    if tracker_name is None:
        return output_results
    if "'" in tracker_name:
        raise ValueError("tracker_name must not contain single quotes")
    if output_results.suffix:
        raise ValueError("--tracker-name requires --output-results to be a directory")

    tracker_dir = f"{tracker_name}_tracking_result"
    if output_results.name == tracker_dir:
        return output_results
    if output_results.name == "eventvot_tracking_results":
        return output_results / tracker_dir
    if output_results.name == "EventVOT_eval_toolkit":
        return output_results / "eventvot_tracking_results" / tracker_dir
    if (output_results / "utils" / "config_tracker.m").exists():
        return output_results / "eventvot_tracking_results" / tracker_dir
    return output_results / tracker_dir


def resolve_output_result_file(
    base_results: Path,
    output_results: Path,
    sequence_name: str,
) -> Path:
    if base_results.is_file() and output_results.suffix:
        return output_results
    return output_results / f"{sequence_name}.txt"


def find_sequence_event_csv(sequence_dir: Path, sequence_name: str) -> Path:
    preferred = sequence_dir / f"{sequence_name}.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(
        path
        for path in sequence_dir.glob("*.csv")
        if path.name.lower() not in {"groundtruth.csv", "absent.csv"}
    )
    if not candidates:
        raise FileNotFoundError(f"No raw EventVOT CSV found in {sequence_dir}")
    return candidates[0]


def default_diagnostics_path(output_results: Path) -> Path:
    if output_results.suffix:
        return output_results.with_name(output_results.stem + "_diagnostics.json")
    return output_results / "eventvot_refinement_summary.json"


def register_tracker_in_config(
    config_tracker_path: Path,
    tracker_name: str,
    *,
    publish: str = "xxx",
) -> bool:
    """Add a tracker entry to EventVOT ``config_tracker.m`` if missing."""
    if "'" in tracker_name or "'" in publish:
        raise ValueError("tracker_name and publish must not contain single quotes")
    text = config_tracker_path.read_text(encoding="utf-8")
    tracker_pattern = re.compile(
        r"struct\('name',\s*'" + re.escape(tracker_name) + r"'",
    )
    if tracker_pattern.search(text):
        return False

    lines = text.splitlines(keepends=True)
    insert_at = None
    for index, line in enumerate(lines):
        if re.match(r"\s*}\s*;", line):
            insert_at = index
            break
    if insert_at is None:
        raise ValueError(f"Could not find tracker cell-array terminator in {config_tracker_path}")

    newline = "\n" if lines and lines[0].endswith("\n") else "\r\n"
    entry = (
        f"                    struct('name', '{tracker_name}',"
        f"           'publish', '{publish}');{newline}"
    )
    lines.insert(insert_at, entry)
    config_tracker_path.write_text("".join(lines), encoding="utf-8")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine EventVOT xywh tracker result files with DVS-ENACT.",
    )
    parser.add_argument("--eventvot-root", type=Path, required=True)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--output-results", type=Path)
    parser.add_argument(
        "--eventvot-toolkit-root",
        type=Path,
        help=(
            "Optional EventVOT_eval_toolkit root. With --tracker-name, refined "
            "results are written below eventvot_tracking_results/."
        ),
    )
    parser.add_argument(
        "--tracker-name",
        help=(
            "Official evaluator tracker name, for example HDETrackV2_DVSENACT. "
            "When set, results are written to <tracker_name>_tracking_result/."
        ),
    )
    parser.add_argument(
        "--config-tracker",
        type=Path,
        help="Optional EventVOT utils/config_tracker.m path to update.",
    )
    parser.add_argument(
        "--update-config-tracker",
        action="store_true",
        help=(
            "Update <eventvot-toolkit-root>/utils/config_tracker.m with "
            "--tracker-name."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Sequence name to refine. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--event-column-order",
        default="auto",
        choices=("auto", "xypt", "txyp", "xytp", "yxpt", "yxpt5", "yxt"),
        help=(
            "Raw-event CSV column order. Use auto for the EventVOT conversion "
            "script conventions."
        ),
    )
    parser.add_argument("--diagnostics-json", type=Path)
    _add_refiner_arguments(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_results = _resolve_cli_output_results(args)
    config_tracker_path = _resolve_cli_config_tracker_path(args)
    refiner = _refiner_from_args(args)
    payload = run(
        EventVOTRefinementOptions(
            eventvot_root=args.eventvot_root,
            base_results=args.base_results,
            output_results=output_results,
            split=args.split,
            sequences=tuple(args.sequence),
            tracker_name=args.tracker_name,
            event_column_order=args.event_column_order,
            diagnostics_json=args.diagnostics_json,
            config_tracker_path=config_tracker_path,
            acceptance_config=_acceptance_config_from_args(args),
        ),
        refiner=refiner,
    )
    print(json.dumps(payload["summary"], indent=2))
    return 0


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


def _add_refiner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image-width", type=float, default=1280.0)
    parser.add_argument("--image-height", type=float, default=720.0)
    parser.add_argument("--search-expansion-factor", type=float, default=1.25)
    parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument("--min-events", type=int, default=3)
    parser.add_argument("--refinement-blend", type=float, default=0.25)
    parser.add_argument(
        "--disable-event-polarity",
        action="store_true",
        help="Ignore event polarity during DVS-ENACT refinement.",
    )
    parser.add_argument(
        "--disable-conservative-gates",
        action="store_true",
        help="Write every non-fallback DVS-ENACT refinement instead of guarded output.",
    )
    parser.add_argument("--min-accept-used-events", type=int, default=10)
    parser.add_argument("--min-accept-active-measurements", type=int, default=3)
    parser.add_argument("--min-accept-mean-activity", type=float, default=0.10)
    parser.add_argument("--min-accept-candidate-iou", type=float, default=0.60)
    parser.add_argument("--max-accept-area-ratio", type=float, default=1.50)


def _refiner_from_args(args: argparse.Namespace) -> DVSContourRefiner:
    return DVSContourRefiner(
        DVSContourRefinerConfig(
            input_bbox_format="xywh",
            output_bbox_format="xywh",
            image_width=args.image_width,
            image_height=args.image_height,
            search_expansion_factor=args.search_expansion_factor,
            max_events=args.max_events,
            min_events=args.min_events,
            use_event_polarity=not args.disable_event_polarity,
            refinement_blend=args.refinement_blend,
        )
    )


def _acceptance_config_from_args(args: argparse.Namespace) -> EventVOTAcceptanceConfig:
    return EventVOTAcceptanceConfig(
        enabled=not args.disable_conservative_gates,
        min_used_event_count=args.min_accept_used_events,
        min_active_measurement_count=args.min_accept_active_measurements,
        min_mean_event_activity=args.min_accept_mean_activity,
        min_candidate_iou=args.min_accept_candidate_iou,
        max_candidate_area_ratio=args.max_accept_area_ratio,
    )


def _event_batch_from_lists(
    timestamps: list[int],
    xs: list[int],
    ys: list[int],
    polarities: list[int],
) -> EventBatch:
    if not timestamps:
        return empty_event_batch()
    return EventBatch(
        ts=np.asarray(timestamps, dtype=np.int64),
        x=np.asarray(xs, dtype=np.int32),
        y=np.asarray(ys, dtype=np.int32),
        p=np.asarray(polarities, dtype=np.int8),
    )


def _validate_sequence_frame_count(
    sequence_dir: Path,
    sequence_name: str,
    result_frame_count: int,
) -> None:
    image_dir = sequence_dir / "img"
    if not image_dir.exists():
        return
    image_count = len(
        [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg"}
        ]
    )
    if image_count != result_frame_count:
        raise ValueError(
            f"{sequence_name}: result has {result_frame_count} rows but img/ "
            f"contains {image_count} frames"
        )


def _save_timing_file(result_file: Path, timings: np.ndarray) -> None:
    timing_file = result_file.with_name(f"{result_file.stem}_time.txt")
    np.savetxt(timing_file, np.asarray(timings, dtype=float), delimiter="\t", fmt="%.9f")


def _resolve_event_schema(
    event_csv: Path,
    values: list[float],
    event_column_order: str,
) -> str:
    if event_column_order != "auto":
        return event_column_order
    name = event_csv.name
    parts = name.split("_")
    third_from_end = parts[-3] if len(parts) >= 3 else ""
    if "_E_" in name and len(third_from_end) > 2 and len(values) >= 3:
        return "yxt"
    if "_EI_" in name and len(third_from_end) > 2 and len(values) >= 5:
        return "yxpt5"
    if len(values) >= 4:
        return "xypt"
    if len(values) >= 3:
        return "yxt"
    raise ValueError(f"Could not infer EventVOT event column order for {event_csv}")


def _parse_event_row(values: list[float], schema: str) -> tuple[int, int, int, int] | None:
    if schema == "xypt" and len(values) >= 4:
        x_value, y_value, polarity, timestamp = values[:4]
    elif schema == "txyp" and len(values) >= 4:
        timestamp, x_value, y_value, polarity = values[:4]
    elif schema == "xytp" and len(values) >= 4:
        x_value, y_value, timestamp, polarity = values[:4]
    elif schema == "yxpt" and len(values) >= 4:
        y_value, x_value, polarity, timestamp = values[:4]
    elif schema == "yxpt5" and len(values) >= 5:
        y_value, x_value, polarity, timestamp = values[0], values[1], values[3], values[4]
    elif schema == "yxt" and len(values) >= 3:
        y_value, x_value, timestamp = values[:3]
        polarity = 1.0
    else:
        return None

    if not all(math.isfinite(value) for value in (timestamp, x_value, y_value, polarity)):
        return None
    return (
        int(timestamp),
        int(x_value),
        int(y_value),
        1 if float(polarity) > 0.0 else 0,
    )


def _numeric_row(row: list[str]) -> list[float] | None:
    tokens = _split_tokens(",".join(row))
    if not tokens:
        return None
    try:
        return [float(token) for token in tokens]
    except ValueError:
        return None


def _parse_numeric_tokens(line: str) -> list[float]:
    return [float(token) for token in _split_tokens(line)]


def _split_tokens(line: str) -> list[str]:
    return [token for token in re.split(r"[\s,]+", line.strip()) if token]


def _is_auxiliary_result_file(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.endswith("_time") or stem.endswith("_all_boxes") or stem.endswith("_all_scores")


if __name__ == "__main__":
    raise SystemExit(main())
