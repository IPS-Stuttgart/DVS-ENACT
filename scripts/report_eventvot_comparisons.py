"""Report EventVOT tracker-vs-refined comparisons and attribute-level gains."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from report_utils import write_csv
from run_eventvot_refinement import resolve_eventvot_split_root, resolve_sequence_names
from run_eventvot_validation_sweep import (
    evaluate_eventvot_sequence,
    mean_curves,
    summarize_eventvot_curves,
)

DEFAULT_TRACKERS = (
    ("HDETrackV2", "HDETrackV2_tracking_result"),
    ("HDETrackV2 + DVS-ENACT", "HDETrackV2_DVSENACT_tracking_result"),
    ("OSTrack-event", "OSTrack-event_tracking_result"),
    ("OSTrack-event + DVS-ENACT", "OSTrack-event_DVSENACT_tracking_result"),
    ("DVS-ENACT-only", "DVS-ENACT-only_tracking_result"),
)
DEFAULT_PAIRS = (
    ("HDETrackV2", "HDETrackV2 + DVS-ENACT"),
    ("OSTrack-event", "OSTrack-event + DVS-ENACT"),
)
ATTRIBUTE_NAMES = (
    "Camera Motion",
    "Deformation",
    "Mild Occlusion",
    "Heavy Occlusion",
    "Full Occlusion",
    "Low Illumination",
    "Out-of-View",
    "Fast Motion",
    "Background Clutter",
    "No Motion",
    "Background Object Motion",
    "Similar Interfering Objects",
    "Scale Variation",
    "Small Target",
)
DEFAULT_HIGHLIGHT_ATTRIBUTES = (
    "Fast Motion",
    "Background Clutter",
    "Scale Variation",
)


@dataclass(frozen=True)
class TrackerSpec:
    """One EventVOT tracker result directory to evaluate."""

    name: str
    result_dir: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the EventVOT paper comparison table and attribute-gain report."
        )
    )
    parser.add_argument("--eventvot-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument(
        "--tracker",
        action="append",
        default=[],
        help=(
            "Tracker spec as Display Name=path/to/result_dir. Can be repeated. "
            "When omitted, the standard five paper rows are read from --result-root."
        ),
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help=(
            "Pair spec as Base Tracker=Refined Tracker. Defaults to HDETrackV2 "
            "and OSTrack-event DVS-ENACT comparisons."
        ),
    )
    parser.add_argument(
        "--fps",
        action="append",
        default=[],
        help="Optional FPS override as Tracker Name=value.",
    )
    parser.add_argument(
        "--auto-fps-from-timing-files",
        action="store_true",
        help=(
            "Estimate FPS from per-sequence <sequence>_time.txt files. These "
            "files are produced by the DVS-ENACT refinement pass and therefore "
            "measure refinement-only throughput, not end-to-end tracker-plus-"
            "refinement throughput. Prefer explicit --fps overrides for paper "
            "tables."
        ),
    )
    parser.add_argument("--attribute-root", type=Path)
    parser.add_argument("--eventvot-toolkit-root", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--min-attribute-sequences", type=int, default=2)
    parser.add_argument(
        "--highlight-attribute",
        action="append",
        default=[],
        help="Attribute to include in the claim-support summary.",
    )
    parser.add_argument(
        "--table-csv",
        type=Path,
        help="Defaults to <output-root>/eventvot_paper_table.csv.",
    )
    parser.add_argument(
        "--table-md",
        type=Path,
        help="Defaults to <output-root>/eventvot_paper_table.md.",
    )
    parser.add_argument(
        "--pairwise-csv",
        type=Path,
        help="Defaults to <output-root>/eventvot_pairwise_gains.csv.",
    )
    parser.add_argument(
        "--attribute-csv",
        type=Path,
        help="Defaults to <output-root>/eventvot_attribute_gains.csv.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Defaults to <output-root>/eventvot_comparison_report.json.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_report(args)
    print(json.dumps(payload["summary"], indent=2))
    return 0


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate requested EventVOT trackers and write paper-ready reports."""
    split_root = resolve_eventvot_split_root(args.eventvot_root, args.split)
    tracker_specs = resolve_tracker_specs(args)
    if not tracker_specs:
        raise ValueError("No tracker result directories were found to report")
    sequence_names = resolve_sequence_names(
        split_root,
        tracker_specs[0].result_dir,
        requested_sequences=tuple(args.sequence),
    )
    fps_overrides = parse_fps_overrides(args.fps)
    auto_fps_from_timing_files = bool(
        getattr(args, "auto_fps_from_timing_files", False),
    )
    attribute_root = resolve_attribute_root(args)
    attribute_map = (
        load_attribute_map(attribute_root, sequence_names)
        if attribute_root is not None
        else {}
    )

    tracker_payloads: dict[str, dict[str, Any]] = {}
    table_rows: list[dict[str, Any]] = []
    for spec in tracker_specs:
        tracker_payload = evaluate_tracker(
            split_root,
            spec.result_dir,
            sequence_names,
            attribute_map,
            args.min_attribute_sequences,
        )
        fps, fps_source, fps_note = resolve_tracker_fps(
            spec,
            sequence_names,
            fps_overrides,
            auto_fps_from_timing_files,
        )
        tracker_payload["fps"] = fps
        tracker_payload["fps_source"] = fps_source
        tracker_payload["fps_note"] = fps_note
        tracker_payloads[spec.name] = tracker_payload
        table_rows.append(make_table_row(spec.name, tracker_payload))

    pairs = resolve_pairs(args.pair, tracker_payloads)
    pairwise_rows = make_pairwise_rows(table_rows, pairs)
    attribute_rows = make_attribute_gain_rows(tracker_payloads, pairs)

    args.output_root.mkdir(parents=True, exist_ok=True)
    table_csv = args.table_csv or args.output_root / "eventvot_paper_table.csv"
    table_md = args.table_md or args.output_root / "eventvot_paper_table.md"
    pairwise_csv = args.pairwise_csv or args.output_root / "eventvot_pairwise_gains.csv"
    attribute_csv = args.attribute_csv or args.output_root / "eventvot_attribute_gains.csv"
    summary_json = args.summary_json or args.output_root / "eventvot_comparison_report.json"

    write_csv(table_csv, table_rows)
    write_markdown_table(table_md, table_rows)
    write_csv(pairwise_csv, pairwise_rows)
    write_csv(attribute_csv, attribute_rows)

    highlights = args.highlight_attribute or list(DEFAULT_HIGHLIGHT_ATTRIBUTES)
    payload = {
        "schema_version": 1,
        "description": (
            "EventVOT comparison report. Table columns SR, PR, and NPR are AUC "
            "scores over the official success, precision, and normalized "
            "precision curves. Pairwise gains isolate the post-hoc DVS-ENACT "
            "physics layer. FPS values are explicit overrides unless fps_source "
            "states refinement_time_files."
        ),
        "summary": {
            "split": args.split,
            "sequence_count": len(sequence_names),
            "tracker_count": len(table_rows),
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "pairwise_csv": str(pairwise_csv),
            "attribute_csv": str(attribute_csv),
            "attribute_root": str(attribute_root) if attribute_root is not None else None,
            "fps_reporting": {
                "explicit_override_trackers": sorted(fps_overrides),
                "auto_fps_from_timing_files": auto_fps_from_timing_files,
                "timing_file_warning": (
                    "Per-sequence *_time.txt files produced by DVS-ENACT "
                    "refinement are refinement-only timings, not end-to-end FPS."
                ),
            },
        },
        "claim_support": make_claim_support(
            pairwise_rows,
            attribute_rows,
            highlights,
        ),
        "table": table_rows,
        "pairwise_gains": pairwise_rows,
        "attribute_gains": attribute_rows,
        "trackers": tracker_payloads,
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def resolve_tracker_specs(args: argparse.Namespace) -> list[TrackerSpec]:
    if args.tracker:
        specs = []
        for item in args.tracker:
            name, path_text = parse_key_value(item, "--tracker")
            result_dir = Path(path_text)
            if not result_dir.is_absolute() and args.result_root is not None:
                result_dir = args.result_root / result_dir
            if result_dir.exists() or args.allow_missing:
                specs.append(TrackerSpec(name=name, result_dir=result_dir))
        return specs
    if args.result_root is None:
        raise ValueError("--result-root is required when --tracker is omitted")
    specs = [
        TrackerSpec(name=name, result_dir=args.result_root / directory_name)
        for name, directory_name in DEFAULT_TRACKERS
    ]
    if args.allow_missing:
        return [spec for spec in specs if spec.result_dir.exists()]
    return specs


def resolve_pairs(
    pair_specs: list[str],
    tracker_payloads: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    if pair_specs:
        return [parse_key_value(item, "--pair") for item in pair_specs]
    return [
        pair
        for pair in DEFAULT_PAIRS
        if pair[0] in tracker_payloads and pair[1] in tracker_payloads
    ]


def evaluate_tracker(
    split_root: Path,
    result_dir: Path,
    sequence_names: list[str],
    attribute_map: dict[str, np.ndarray],
    min_attribute_sequences: int,
) -> dict[str, Any]:
    sequence_metrics: dict[str, dict[str, Any]] = {}
    for sequence_name in sequence_names:
        result_file = result_dir / f"{sequence_name}.txt"
        if not result_file.exists():
            raise FileNotFoundError(f"Missing EventVOT result file: {result_file}")
        sequence_metrics[sequence_name] = evaluate_eventvot_sequence(
            split_root / sequence_name,
            result_file,
        )
    overall = aggregate_sequence_metrics(sequence_metrics, sequence_names)
    attributes = evaluate_attributes(
        sequence_metrics,
        sequence_names,
        attribute_map,
        min_attribute_sequences,
    )
    return {
        "result_dir": str(result_dir),
        "overall": overall,
        "attributes": attributes,
    }


def aggregate_sequence_metrics(
    sequence_metrics: dict[str, dict[str, Any]],
    selected_sequences: list[str],
) -> dict[str, Any]:
    success_curves = [sequence_metrics[name]["success_curve"] for name in selected_sequences]
    precision_curves = [
        sequence_metrics[name]["precision_curve"] for name in selected_sequences
    ]
    normalized_precision_curves = [
        sequence_metrics[name]["normalized_precision_curve"]
        for name in selected_sequences
    ]
    mean_ious = [float(sequence_metrics[name]["mean_iou"]) for name in selected_sequences]
    evaluated_frame_count = sum(
        int(sequence_metrics[name]["evaluated_frame_count"]) for name in selected_sequences
    )
    success_curve = mean_curves(success_curves)
    precision_curve = mean_curves(precision_curves)
    normalized_precision_curve = mean_curves(normalized_precision_curves)
    return summarize_eventvot_curves(
        sequence_count=len(selected_sequences),
        evaluated_frame_count=evaluated_frame_count,
        success_curve=success_curve,
        precision_curve=precision_curve,
        normalized_precision_curve=normalized_precision_curve,
        mean_ious=mean_ious,
    )


def evaluate_attributes(
    sequence_metrics: dict[str, dict[str, Any]],
    sequence_names: list[str],
    attribute_map: dict[str, np.ndarray],
    min_attribute_sequences: int,
) -> dict[str, Any]:
    if not attribute_map:
        return {}
    attributes: dict[str, Any] = {}
    for attribute_index, attribute_name in enumerate(ATTRIBUTE_NAMES):
        selected = [
            sequence_name
            for sequence_name in sequence_names
            if sequence_name in attribute_map
            and attribute_index < attribute_map[sequence_name].shape[0]
            and float(attribute_map[sequence_name][attribute_index]) > 0.0
        ]
        if len(selected) < min_attribute_sequences:
            continue
        attributes[attribute_name] = aggregate_sequence_metrics(
            sequence_metrics,
            selected,
        )
    return attributes


def make_table_row(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    overall = payload["overall"]
    return {
        "tracker": name,
        "sr": overall["sr_auc"],
        "pr": overall["pr_auc"],
        "npr": overall["npr_auc"],
        "fps": payload["fps"],
        "fps_source": payload.get("fps_source", "unknown"),
        "fps_note": payload.get("fps_note", ""),
        "pr_20": overall["pr_20"],
        "npr_020": overall["npr_020"],
        "mean_iou": overall["mean_iou"],
        "evaluated_frame_count": overall["evaluated_frame_count"],
    }


def make_pairwise_rows(
    table_rows: list[dict[str, Any]],
    pairs: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows_by_name = {row["tracker"]: row for row in table_rows}
    gain_rows = []
    for base_name, refined_name in pairs:
        if base_name not in rows_by_name or refined_name not in rows_by_name:
            continue
        base = rows_by_name[base_name]
        refined = rows_by_name[refined_name]
        gain_rows.append(
            {
                "base_tracker": base_name,
                "refined_tracker": refined_name,
                "sr_delta": refined["sr"] - base["sr"],
                "pr_delta": refined["pr"] - base["pr"],
                "npr_delta": refined["npr"] - base["npr"],
                "fps_delta": optional_delta(refined["fps"], base["fps"]),
                "fps_ratio": optional_ratio(refined["fps"], base["fps"]),
            }
        )
    return gain_rows


def make_attribute_gain_rows(
    tracker_payloads: dict[str, dict[str, Any]],
    pairs: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for base_name, refined_name in pairs:
        if base_name not in tracker_payloads or refined_name not in tracker_payloads:
            continue
        base_attributes = tracker_payloads[base_name]["attributes"]
        refined_attributes = tracker_payloads[refined_name]["attributes"]
        for attribute_name in ATTRIBUTE_NAMES:
            if attribute_name not in base_attributes or attribute_name not in refined_attributes:
                continue
            base = base_attributes[attribute_name]
            refined = refined_attributes[attribute_name]
            rows.append(
                {
                    "base_tracker": base_name,
                    "refined_tracker": refined_name,
                    "attribute": attribute_name,
                    "sequence_count": refined["sequence_count"],
                    "sr_base": base["sr_auc"],
                    "sr_refined": refined["sr_auc"],
                    "sr_delta": refined["sr_auc"] - base["sr_auc"],
                    "pr_delta": refined["pr_auc"] - base["pr_auc"],
                    "npr_delta": refined["npr_auc"] - base["npr_auc"],
                }
            )
    return sorted(
        rows,
        key=lambda row: (row["base_tracker"], -float(row["sr_delta"]), row["attribute"]),
    )


def make_claim_support(
    pairwise_rows: list[dict[str, Any]],
    attribute_rows: list[dict[str, Any]],
    highlight_attributes: list[str],
) -> dict[str, Any]:
    pair_by_base = {row["base_tracker"]: row for row in pairwise_rows}
    hde_pair = pair_by_base.get("HDETrackV2")
    hde_attribute_rows = [
        row
        for row in attribute_rows
        if row["base_tracker"] == "HDETrackV2"
        and row["attribute"] in set(highlight_attributes)
    ]
    return {
        "hde_track_v2_overall_sr_improved": (
            bool(hde_pair["sr_delta"] > 0.0) if hde_pair is not None else None
        ),
        "hde_track_v2_fps_ratio": (
            hde_pair.get("fps_ratio") if hde_pair is not None else None
        ),
        "highlight_attributes": hde_attribute_rows,
        "positive_highlight_attribute_count": sum(
            1 for row in hde_attribute_rows if row["sr_delta"] > 0.0
        ),
    }


def resolve_attribute_root(args: argparse.Namespace) -> Path | None:
    candidates = []
    if args.attribute_root is not None:
        candidates.append(args.attribute_root)
    if args.eventvot_toolkit_root is not None:
        candidates.append(args.eventvot_toolkit_root / "annos" / "att")
    candidates.extend(
        [
            args.eventvot_root / "annos" / "att",
            args.eventvot_root / "EventVOT_eval_toolkit" / "annos" / "att",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_attribute_map(attribute_root: Path, sequence_names: list[str]) -> dict[str, np.ndarray]:
    attributes = {}
    for sequence_name in sequence_names:
        path = attribute_root / f"{sequence_name}.txt"
        if not path.exists():
            continue
        values = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if stripped:
                values.append(float(stripped))
        attributes[sequence_name] = np.asarray(values, dtype=float)
    return attributes


def resolve_tracker_fps(
    spec: TrackerSpec,
    sequence_names: list[str],
    fps_overrides: dict[str, float],
    use_timing_files: bool,
) -> tuple[float | None, str, str]:
    """Resolve FPS and its provenance for one tracker row.

    DVS-ENACT refinement writes ``<sequence>_time.txt`` files containing only
    post-hoc refinement elapsed times. Treating those files as automatic FPS for
    a row named ``Tracker + DVS-ENACT`` silently reports refinement-only
    throughput as if it were end-to-end tracker-plus-refinement throughput. The
    report therefore uses explicit ``--fps`` overrides by default and labels any
    opt-in timing-file estimate as refinement-only.
    """
    if spec.name in fps_overrides:
        return (
            float(fps_overrides[spec.name]),
            "override",
            "Explicit --fps override; intended for end-to-end paper-table FPS.",
        )
    if not use_timing_files:
        return (
            None,
            "not_reported",
            (
                "Automatic *_time.txt FPS is disabled because those files are "
                "DVS-ENACT refinement-only timings; pass --fps for end-to-end "
                "tracker throughput or --auto-fps-from-timing-files for a "
                "labeled refinement-only estimate."
            ),
        )
    fps = estimate_fps(spec.result_dir, sequence_names)
    if fps is None:
        return None, "missing", "No usable per-sequence *_time.txt files found."
    return fps, "refinement_time_files", "Computed from refinement-only *_time.txt files."


def estimate_fps(result_dir: Path, sequence_names: list[str]) -> float | None:
    total_time = 0.0
    total_frames = 0
    for sequence_name in sequence_names:
        timing_file = result_dir / f"{sequence_name}_time.txt"
        if not timing_file.exists():
            continue
        timings = np.loadtxt(timing_file, dtype=float)
        timings = np.atleast_1d(timings)
        time_sum = float(np.sum(timings))
        if time_sum <= 0.0:
            continue
        total_time += time_sum
        total_frames += int(timings.shape[0])
    if total_time <= 0.0 or total_frames == 0:
        return None
    return float(total_frames / total_time)


def parse_fps_overrides(items: list[str]) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in (parse_key_value(item, "--fps") for item in items)
    }


def parse_key_value(item: str, flag_name: str) -> tuple[str, str]:
    if "=" not in item:
        raise ValueError(f"{flag_name} expects NAME=VALUE, got: {item}")
    key, value = item.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise ValueError(f"{flag_name} expects non-empty NAME=VALUE, got: {item}")
    return key, value


def optional_delta(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return float(first - second)


def optional_ratio(first: float | None, second: float | None) -> float | None:
    if first is None or second is None or second <= 0.0:
        return None
    return float(first / second)


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Tracker | SR | PR | NPR | FPS | FPS source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['tracker']} | "
            f"{format_metric(row['sr'])} | "
            f"{format_metric(row['pr'])} | "
            f"{format_metric(row['npr'])} | "
            f"{format_fps(row['fps'])} | "
            f"{format_fps_source(row)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric(value: float) -> str:
    return f"{float(value):.3f}"


def format_fps(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.1f}"


def format_fps_source(row: dict[str, Any]) -> str:
    source = str(row.get("fps_source", "unknown"))
    if source == "override":
        return "override"
    if source == "refinement_time_files":
        return "refinement-only"
    if source == "not_reported":
        return "not reported"
    if source == "missing":
        return "missing"
    return source.replace("_", " ")


if __name__ == "__main__":
    raise SystemExit(main())
