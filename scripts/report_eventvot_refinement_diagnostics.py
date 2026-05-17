"""Report per-sequence EventVOT deltas with DVS-ENACT refinement diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from run_eventvot_refinement import (
    resolve_base_result_file,
    resolve_eventvot_split_root,
    resolve_sequence_names,
)
from run_eventvot_validation_sweep import evaluate_eventvot_sequence

DIAGNOSTIC_FIELDS = (
    "accepted_refinement_rate",
    "refiner_success_rate",
    "mean_candidate_iou",
    "mean_raw_candidate_iou",
    "mean_center_shift_ratio",
    "mean_raw_center_shift_ratio",
    "mean_candidate_area_ratio",
    "mean_raw_candidate_area_ratio",
    "mean_active_fraction",
    "mean_quadratic_form_per_active_measurement",
    "mean_used_event_count",
    "mean_event_count",
    "mean_active_measurement_count",
)
DELTA_FIELDS = ("delta_sr", "delta_pr", "delta_npr", "delta_mean_iou")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join EventVOT per-sequence deltas with DVS-ENACT diagnostics."
    )
    parser.add_argument("--diagnostics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eventvot-root", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--base-results", type=Path)
    parser.add_argument("--refined-results", type=Path)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def main() -> int:
    payload = run_report(build_parser().parse_args())
    print(json.dumps(payload["summary"], indent=2))
    return 0


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    diagnostics = json.loads(args.diagnostics_json.read_text(encoding="utf-8"))
    options = diagnostics.get("options", {})
    evaluator = diagnostics.get("eventvot_evaluator", {})
    eventvot_root = args.eventvot_root or Path(options.get("eventvot_root", ""))
    split = args.split or str(options.get("split", "test"))
    base_results = args.base_results or Path(options.get("base_results", ""))
    refined_results = (
        args.refined_results
        or Path(evaluator.get("tracking_result_dir") or options.get("resolved_output_results", ""))
    )
    if not eventvot_root or not base_results or not refined_results:
        raise ValueError("Could not infer eventvot/base/refined paths; pass them explicitly.")

    split_root = resolve_eventvot_split_root(eventvot_root, split)
    sequences = resolve_report_sequences(args, diagnostics, split_root, base_results)
    diag_by_sequence = {
        str(item["sequence"]): item
        for item in diagnostics.get("sequences", [])
        if item.get("sequence")
    }
    rows = [
        make_sequence_row(
            sequence,
            split_root,
            base_results,
            refined_results,
            diag_by_sequence.get(sequence, {}),
        )
        for sequence in sequences
    ]

    acceptance_rows = make_acceptance_rows(rows)
    correlation_rows = make_correlation_rows(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_sequence_csv = args.output_dir / "per_sequence_deltas.csv"
    acceptance_csv = args.output_dir / "acceptance_reasons.csv"
    correlation_csv = args.output_dir / "diagnostic_correlations.csv"
    worst_md = args.output_dir / "worst_sequences.md"
    best_md = args.output_dir / "best_sequences.md"
    summary_json = args.output_dir / "refinement_diagnostic_report.json"

    write_csv(per_sequence_csv, rows)
    write_csv(acceptance_csv, acceptance_rows)
    write_csv(correlation_csv, correlation_rows)
    write_sequence_md(worst_md, sorted(rows, key=lambda r: fnum(r["delta_sr"])), "Worst sequences by SR delta", args.top_k)
    write_sequence_md(best_md, sorted(rows, key=lambda r: fnum(r["delta_sr"]), reverse=True), "Best sequences by SR delta", args.top_k)

    payload = {
        "schema_version": 1,
        "summary": {
            "sequence_count": len(rows),
            "base_results": str(base_results),
            "refined_results": str(refined_results),
            "per_sequence_csv": str(per_sequence_csv),
            "acceptance_csv": str(acceptance_csv),
            "correlation_csv": str(correlation_csv),
            "worst_sequences_md": str(worst_md),
            "best_sequences_md": str(best_md),
            "mean_delta_sr": safe_mean(row["delta_sr"] for row in rows),
            "mean_delta_pr": safe_mean(row["delta_pr"] for row in rows),
            "mean_delta_npr": safe_mean(row["delta_npr"] for row in rows),
            "positive_delta_sr_sequence_count": sum(1 for row in rows if fnum(row["delta_sr"]) > 0.0),
            "negative_delta_sr_sequence_count": sum(1 for row in rows if fnum(row["delta_sr"]) < 0.0),
        },
        "acceptance_reasons": acceptance_rows,
        "diagnostic_correlations": correlation_rows,
        "per_sequence": rows,
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def resolve_report_sequences(args: argparse.Namespace, diagnostics: dict[str, Any], split_root: Path, base_results: Path) -> list[str]:
    if args.sequence:
        return list(dict.fromkeys(name.strip() for name in args.sequence if name.strip()))
    diagnostic_sequences = [str(item["sequence"]) for item in diagnostics.get("sequences", []) if item.get("sequence")]
    if diagnostic_sequences:
        return diagnostic_sequences
    return resolve_sequence_names(split_root, base_results)


def make_sequence_row(sequence: str, split_root: Path, base_results: Path, refined_results: Path, diag: dict[str, Any]) -> dict[str, Any]:
    base = scalar_metrics(evaluate_eventvot_sequence(split_root / sequence, resolve_base_result_file(base_results, sequence)))
    refined = scalar_metrics(evaluate_eventvot_sequence(split_root / sequence, resolve_base_result_file(refined_results, sequence)))
    diag_summary = summarize_diagnostics(diag)
    return {
        "sequence": sequence,
        "frame_count": diag.get("frame_count"),
        "evaluated_frame_count": refined["evaluated_frame_count"],
        "base_sr": base["sr"],
        "refined_sr": refined["sr"],
        "delta_sr": refined["sr"] - base["sr"],
        "base_pr": base["pr"],
        "refined_pr": refined["pr"],
        "delta_pr": refined["pr"] - base["pr"],
        "base_npr": base["npr"],
        "refined_npr": refined["npr"],
        "delta_npr": refined["npr"] - base["npr"],
        "base_mean_iou": base["mean_iou"],
        "refined_mean_iou": refined["mean_iou"],
        "delta_mean_iou": refined["mean_iou"] - base["mean_iou"],
        **diag_summary,
        "fallback_counts": json.dumps(diag.get("fallback_counts", {}), sort_keys=True),
        "acceptance_counts": json.dumps(diag.get("acceptance_counts", {}), sort_keys=True),
    }


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    return {
        "sr": float(np.mean(metrics["success_curve"])),
        "pr": float(np.mean(metrics["precision_curve"])),
        "npr": float(np.mean(metrics["normalized_precision_curve"])),
        "mean_iou": float(metrics["mean_iou"]),
        "evaluated_frame_count": int(metrics["evaluated_frame_count"]),
    }


def summarize_diagnostics(diag: dict[str, Any]) -> dict[str, Any]:
    frame_count = int(diag.get("frame_count") or 0)
    refinable = max(0, frame_count - 1)
    frames = [frame for frame in diag.get("frames", []) if int(frame.get("frame_index", 0)) > 0]
    return {
        "accepted_refinement_count": int(diag.get("accepted_refinement_count") or 0),
        "accepted_refinement_rate": rate(diag.get("accepted_refinement_count") or 0, refinable),
        "refiner_success_frame_count": int(diag.get("refiner_success_frame_count") or 0),
        "refiner_success_rate": rate(diag.get("refiner_success_frame_count") or 0, refinable),
        "mean_candidate_iou": mean_frame(frames, "candidate_iou"),
        "mean_raw_candidate_iou": mean_frame(frames, "raw_candidate_iou"),
        "mean_center_shift_ratio": mean_frame(frames, "center_shift_ratio"),
        "mean_raw_center_shift_ratio": mean_frame(frames, "raw_center_shift_ratio"),
        "mean_candidate_area_ratio": mean_frame(frames, "candidate_area_ratio"),
        "mean_raw_candidate_area_ratio": mean_frame(frames, "raw_candidate_area_ratio"),
        "mean_active_fraction": mean_active_fraction(frames),
        "mean_quadratic_form_per_active_measurement": mean_quadratic_per_active(frames),
        "mean_used_event_count": mean_frame(frames, "used_event_count"),
        "mean_event_count": mean_frame(frames, "event_count"),
        "mean_active_measurement_count": mean_frame(frames, "active_measurement_count"),
    }


def make_acceptance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_frames = sum(int(row.get("frame_count") or 0) for row in rows)
    counts: Counter[str] = Counter()
    deltas: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for reason, count in json.loads(row.get("acceptance_counts") or "{}").items():
            counts[reason] += int(count)
            if int(count) > 0:
                deltas[reason].append(float(row["delta_sr"]))
    return [
        {
            "reason": reason,
            "count": count,
            "frame_fraction": rate(count, total_frames),
            "sequence_count": len(deltas[reason]),
            "mean_delta_sr_for_sequences": safe_mean(deltas[reason]),
            "median_delta_sr_for_sequences": safe_median(deltas[reason]),
        }
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def make_correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for diagnostic in DIAGNOSTIC_FIELDS:
        for target in DELTA_FIELDS:
            r_value, n = pearson([row.get(diagnostic) for row in rows], [row.get(target) for row in rows])
            result.append({"diagnostic": diagnostic, "target": target, "pearson_r": r_value, "sample_count": n})
    return sorted(result, key=lambda row: (-1 if row["pearson_r"] is None else -abs(float(row["pearson_r"])), row["diagnostic"], row["target"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sequence_md(path: Path, rows: list[dict[str, Any]], title: str, top_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "| Rank | Sequence | ΔSR | ΔPR | ΔNPR | Accepted rate | Mean raw IoU | Acceptance counts |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[:top_k], start=1):
        lines.append(
            f"| {rank} | {row['sequence']} | {fmt_signed(row['delta_sr'])} | {fmt_signed(row['delta_pr'])} | "
            f"{fmt_signed(row['delta_npr'])} | {fmt_float(row.get('accepted_refinement_rate'))} | "
            f"{fmt_float(row.get('mean_raw_candidate_iou'))} | `{row.get('acceptance_counts', '{}')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_frame(frames: list[dict[str, Any]], key: str) -> float | None:
    return safe_mean(frame.get(key) for frame in frames)


def mean_active_fraction(frames: list[dict[str, Any]]) -> float | None:
    values = []
    for frame in frames:
        value = frame.get("active_fraction")
        if value is None:
            used = opt_float(frame.get("used_event_count"))
            active = opt_float(frame.get("active_measurement_count"))
            value = None if used is None or used <= 0.0 or active is None else active / used
        if finite(value):
            values.append(float(value))
    return safe_mean(values)


def mean_quadratic_per_active(frames: list[dict[str, Any]]) -> float | None:
    values = []
    for frame in frames:
        value = frame.get("quadratic_form_per_active_measurement")
        if value is None:
            q = opt_float(frame.get("quadratic_form"))
            active = opt_float(frame.get("active_measurement_count"))
            value = None if q is None or active is None or active <= 0.0 else q / active
        if finite(value):
            values.append(float(value))
    return safe_mean(values)


def pearson(xs: list[Any], ys: list[Any]) -> tuple[float | None, int]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys, strict=True) if finite(x) and finite(y)]
    if len(pairs) < 2:
        return None, len(pairs)
    x_arr = np.asarray([x for x, _y in pairs], dtype=float)
    y_arr = np.asarray([y for _x, y in pairs], dtype=float)
    if float(np.std(x_arr)) <= 0.0 or float(np.std(y_arr)) <= 0.0:
        return None, len(pairs)
    return float(np.corrcoef(x_arr, y_arr)[0, 1]), len(pairs)


def safe_mean(values: Any) -> float | None:
    vals = [float(v) for v in values if finite(v)]
    return None if not vals else float(np.mean(vals))


def safe_median(values: Any) -> float | None:
    vals = [float(v) for v in values if finite(v)]
    return None if not vals else float(median(vals))


def rate(numerator: Any, denominator: Any) -> float:
    den = opt_float(denominator) or 0.0
    return 0.0 if den <= 0.0 else float(numerator) / den


def opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def finite(value: Any) -> bool:
    val = opt_float(value)
    return val is not None and math.isfinite(val)


def fnum(value: Any) -> float:
    val = opt_float(value)
    return float("nan") if val is None else val


def fmt_signed(value: Any) -> str:
    val = opt_float(value)
    return "n/a" if val is None else f"{val:+.6f}"


def fmt_float(value: Any) -> str:
    val = opt_float(value)
    return "n/a" if val is None else f"{val:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
