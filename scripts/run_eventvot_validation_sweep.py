"""Tune EventVOT DVS-ENACT refinement parameters on the validation subset."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from dvs_enact import DVSContourRefiner, DVSContourRefinerConfig
from run_eventvot_refinement import (
    EventVOTAcceptanceConfig,
    EventVOTRefinementOptions,
    add_acceptance_arguments,
    load_xywh_result_file,
    resolve_base_result_file,
    resolve_eventvot_split_root,
    resolve_output_results_root,
    resolve_sequence_names,
    run as run_eventvot_refinement,
)

DEFAULT_REFINEMENT_BLEND = (0.10, 0.25, 0.50, 1.00)
DEFAULT_SEARCH_EXPANSION_FACTOR = (1.10, 1.25, 1.50)
DEFAULT_MAX_EVENTS = (64, 128, 256, 512)
DEFAULT_MIN_EVENTS = (3, 10, 20)
DEFAULT_EVENT_ACTIVITY_FLOOR = (0.00, 0.02, 0.05)
DEFAULT_INACTIVE_ACTIVITY_THRESHOLD = (0.02, 0.05, 0.10)
DEFAULT_MEASUREMENT_NOISE_VARIANCE = (1.0, 4.0, 9.0)

OVERLAP_THRESHOLDS = np.arange(0.0, 1.0001, 0.05)
ERROR_THRESHOLDS = np.arange(0.0, 51.0, 1.0)
NORMALIZED_ERROR_THRESHOLDS = ERROR_THRESHOLDS / 100.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a validation-only EventVOT hyperparameter sweep for guarded "
            "DVS-ENACT post-hoc refinement."
        )
    )
    parser.add_argument("--eventvot-root", type=Path, required=True)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--allow-test-split",
        action="store_true",
        help="Allow --split test. Intended only for final evaluation, not tuning.",
    )
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument(
        "--event-column-order",
        default="auto",
        choices=("auto", "xypt", "txyp", "xytp", "yxpt", "yxpt5", "yxt"),
    )
    parser.add_argument("--tracker-prefix", default="HDETrackV2_DVSENACT_VAL")
    parser.add_argument("--image-width", type=float, default=1280.0)
    parser.add_argument("--image-height", type=float, default=720.0)
    parser.add_argument("--disable-event-polarity", action="store_true")
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        help="Defaults to <output-root>/validation_sweep_metrics.csv.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Defaults to <output-root>/validation_sweep_summary.json.",
    )
    parser.add_argument(
        "--refinement-blend",
        type=float,
        nargs="+",
        default=list(DEFAULT_REFINEMENT_BLEND),
    )
    parser.add_argument(
        "--search-expansion-factor",
        type=float,
        nargs="+",
        default=list(DEFAULT_SEARCH_EXPANSION_FACTOR),
    )
    parser.add_argument("--max-events", type=int, nargs="+", default=list(DEFAULT_MAX_EVENTS))
    parser.add_argument("--min-events", type=int, nargs="+", default=list(DEFAULT_MIN_EVENTS))
    parser.add_argument(
        "--event-activity-floor",
        type=float,
        nargs="+",
        default=list(DEFAULT_EVENT_ACTIVITY_FLOOR),
    )
    parser.add_argument(
        "--inactive-activity-threshold",
        type=float,
        nargs="+",
        default=list(DEFAULT_INACTIVE_ACTIVITY_THRESHOLD),
    )
    parser.add_argument(
        "--measurement-noise-variance",
        type=float,
        nargs="+",
        default=list(DEFAULT_MEASUREMENT_NOISE_VARIANCE),
    )
    add_acceptance_arguments(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_sweep(args)
    print(json.dumps(payload["summary"], indent=2))
    return 0


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run the validation sweep and write ranked metrics."""
    if _is_test_split(args.split) and not args.allow_test_split:
        raise SystemExit(
            "Refusing to tune on EventVOT test split. Use --split val/validating, "
            "or pass --allow-test-split only for final held-out evaluation."
        )

    split_root = resolve_eventvot_split_root(args.eventvot_root, args.split)
    sequences = resolve_sequence_names(
        split_root,
        args.base_results,
        requested_sequences=tuple(args.sequence),
    )
    configs = list(iter_parameter_grid(args))
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.metrics_csv or args.output_root / "validation_sweep_metrics.csv"
    summary_json = args.summary_json or args.output_root / "validation_sweep_summary.json"

    base_metrics = evaluate_eventvot_results(split_root, args.base_results, sequences)
    rows: list[dict[str, Any]] = []
    if not args.dry_run:
        for config_index, config in enumerate(configs, start=1):
            config_id = make_config_id(config_index, config)
            tracker_name = f"{args.tracker_prefix}_{config_id}"
            result_parent = args.output_root / "eventvot_tracking_results"
            diagnostics_json = args.output_root / "diagnostics" / f"{config_id}.json"
            refiner_payload = run_eventvot_refinement(
                EventVOTRefinementOptions(
                    eventvot_root=args.eventvot_root,
                    base_results=args.base_results,
                    output_results=result_parent,
                    split=args.split,
                    sequences=tuple(sequences),
                    tracker_name=tracker_name,
                    event_column_order=args.event_column_order,
                    diagnostics_json=diagnostics_json,
                    acceptance_config=acceptance_config_from_args(args),
                ),
                refiner=make_refiner(config, args),
            )
            result_dir = resolve_output_results_root(
                result_parent,
                tracker_name=tracker_name,
            )
            metrics = evaluate_eventvot_results(split_root, result_dir, sequences)
            rows.append(
                make_result_row(
                    config_id,
                    tracker_name,
                    config,
                    metrics,
                    refiner_payload["summary"],
                )
            )
            write_sweep_outputs(
                rows,
                configs,
                base_metrics,
                metrics_csv,
                summary_json,
                args.top_k,
            )

    payload = write_sweep_outputs(
        rows,
        configs,
        base_metrics,
        metrics_csv,
        summary_json,
        args.top_k,
        dry_run=args.dry_run,
    )
    return payload


def iter_parameter_grid(args: argparse.Namespace) -> list[dict[str, float | int]]:
    keys = (
        "refinement_blend",
        "search_expansion_factor",
        "max_events",
        "min_events",
        "event_activity_floor",
        "inactive_activity_threshold",
        "measurement_noise_variance",
    )
    values = (
        args.refinement_blend,
        args.search_expansion_factor,
        args.max_events,
        args.min_events,
        args.event_activity_floor,
        args.inactive_activity_threshold,
        args.measurement_noise_variance,
    )
    return [
        dict(zip(keys, combination, strict=True))
        for combination in itertools.product(*values)
    ]


def make_refiner(
    config: dict[str, float | int],
    args: argparse.Namespace,
) -> DVSContourRefiner:
    return DVSContourRefiner(
        DVSContourRefinerConfig(
            input_bbox_format="xywh",
            output_bbox_format="xywh",
            image_width=args.image_width,
            image_height=args.image_height,
            search_expansion_factor=float(config["search_expansion_factor"]),
            max_events=int(config["max_events"]),
            min_events=int(config["min_events"]),
            event_activity_floor=float(config["event_activity_floor"]),
            inactive_activity_threshold=float(config["inactive_activity_threshold"]),
            measurement_noise_variance=float(config["measurement_noise_variance"]),
            use_event_polarity=not args.disable_event_polarity,
            refinement_blend=float(config["refinement_blend"]),
        )
    )


def acceptance_config_from_args(args: argparse.Namespace) -> EventVOTAcceptanceConfig:
    return EventVOTAcceptanceConfig(
        min_used_event_count=args.min_accept_used_events,
        min_active_measurement_count=args.min_accept_active_measurements,
        min_mean_event_activity=args.min_accept_mean_activity,
        min_candidate_iou=args.min_accept_candidate_iou,
        min_candidate_area_ratio=args.min_accept_area_ratio,
        max_candidate_area_ratio=args.max_accept_area_ratio,
        max_center_shift_ratio=args.max_accept_center_shift_ratio,
    )


def make_result_row(
    config_id: str,
    tracker_name: str,
    config: dict[str, float | int],
    metrics: dict[str, Any],
    refiner_summary: dict[str, Any],
) -> dict[str, Any]:
    frame_count = int(refiner_summary["frame_count"])
    sequence_count = int(refiner_summary["sequence_count"])
    accepted_count = int(refiner_summary["accepted_refinement_count"])
    refinable_frame_count = max(0, frame_count - sequence_count)
    return {
        "config_id": config_id,
        "tracker_name": tracker_name,
        **config,
        "sr_auc": metrics["sr_auc"],
        "pr_auc": metrics["pr_auc"],
        "pr_20": metrics["pr_20"],
        "npr_auc": metrics["npr_auc"],
        "npr_020": metrics["npr_020"],
        "mean_iou": metrics["mean_iou"],
        "evaluated_frame_count": metrics["evaluated_frame_count"],
        "frame_count": frame_count,
        "accepted_refinement_count": accepted_count,
        "refiner_success_frame_count": int(refiner_summary["refiner_success_frame_count"]),
        "acceptance_rate": (
            float(accepted_count / refinable_frame_count)
            if refinable_frame_count
            else 0.0
        ),
    }


def write_sweep_outputs(
    rows: list[dict[str, Any]],
    configs: list[dict[str, float | int]],
    base_metrics: dict[str, Any],
    metrics_csv: Path,
    summary_json: Path,
    top_k: int,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["sr_auc"]),
            -float(row["pr_auc"]),
            -float(row["npr_auc"]),
        ),
    )
    if rows:
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(ranked)

    payload = {
        "schema_version": 1,
        "description": (
            "Validation-only EventVOT hyperparameter sweep. Rank by SR AUC; "
            "PR/NPR and acceptance rate are secondary diagnostics."
        ),
        "summary": {
            "dry_run": dry_run,
            "config_count": len(configs),
            "completed_config_count": len(rows),
            "best_config_id": ranked[0]["config_id"] if ranked else None,
            "best_sr_auc": ranked[0]["sr_auc"] if ranked else None,
            "metrics_csv": str(metrics_csv),
        },
        "base_metrics": base_metrics,
        "top_configs": ranked[:top_k],
        "grid": configs,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def evaluate_eventvot_results(
    split_root: Path,
    result_root: Path,
    sequence_names: list[str],
) -> dict[str, Any]:
    """Evaluate EventVOT xywh result files with the official curve definitions."""
    success_curves = []
    precision_curves = []
    normalized_precision_curves = []
    mean_ious = []
    evaluated_frame_count = 0
    for sequence_name in sequence_names:
        sequence_dir = split_root / sequence_name
        result_file = resolve_base_result_file(result_root, sequence_name)
        sequence_metrics = evaluate_eventvot_sequence(sequence_dir, result_file)
        success_curves.append(sequence_metrics["success_curve"])
        precision_curves.append(sequence_metrics["precision_curve"])
        normalized_precision_curves.append(
            sequence_metrics["normalized_precision_curve"]
        )
        mean_ious.append(sequence_metrics["mean_iou"])
        evaluated_frame_count += int(sequence_metrics["evaluated_frame_count"])

    success_curve = mean_nonzero_curves(success_curves)
    precision_curve = mean_nonzero_curves(precision_curves)
    normalized_precision_curve = mean_nonzero_curves(normalized_precision_curves)
    metrics = summarize_eventvot_curves(
        sequence_count=len(sequence_names),
        evaluated_frame_count=evaluated_frame_count,
        success_curve=success_curve,
        precision_curve=precision_curve,
        normalized_precision_curve=normalized_precision_curve,
        mean_ious=mean_ious,
    )
    metrics["absent_handling"] = (
        "mirrors official toolkit: absent.txt is inverted before filtering"
    )
    return metrics


def summarize_eventvot_curves(
    *,
    sequence_count: int,
    evaluated_frame_count: int,
    success_curve: np.ndarray,
    precision_curve: np.ndarray,
    normalized_precision_curve: np.ndarray,
    mean_ious: list[float],
) -> dict[str, Any]:
    """Return scalar EventVOT metrics and serialized official curves."""
    return {
        "sequence_count": int(sequence_count),
        "evaluated_frame_count": int(evaluated_frame_count),
        "sr_auc": float(np.mean(success_curve)),
        "pr_auc": float(np.mean(precision_curve)),
        "pr_20": float(precision_curve[20]),
        "npr_auc": float(np.mean(normalized_precision_curve)),
        "npr_020": float(normalized_precision_curve[20]),
        "mean_iou": float(np.mean(mean_ious)) if mean_ious else 0.0,
        "success_curve": success_curve.astype(float).tolist(),
        "precision_curve": precision_curve.astype(float).tolist(),
        "normalized_precision_curve": normalized_precision_curve.astype(float).tolist(),
        "overlap_thresholds": OVERLAP_THRESHOLDS.astype(float).tolist(),
        "error_thresholds": ERROR_THRESHOLDS.astype(float).tolist(),
        "normalized_error_thresholds": NORMALIZED_ERROR_THRESHOLDS.astype(float).tolist(),
    }


def evaluate_eventvot_sequence(sequence_dir: Path, result_file: Path) -> dict[str, Any]:
    groundtruth = load_numeric_matrix(sequence_dir / "groundtruth.txt", min_columns=4)[:, :4]
    absent = load_numeric_matrix(sequence_dir / "absent.txt", min_columns=1)[:, 0]
    results = load_xywh_result_file(result_file)
    if results.shape[0] < groundtruth.shape[0]:
        raise ValueError(
            f"{result_file} has {results.shape[0]} rows, but "
            f"{sequence_dir / 'groundtruth.txt'} has {groundtruth.shape[0]}"
        )
    if results.shape[0] != groundtruth.shape[0]:
        results = results[: groundtruth.shape[0], :]
    if absent.shape[0] < groundtruth.shape[0]:
        raise ValueError(f"{sequence_dir / 'absent.txt'} is shorter than groundtruth")
    absent = absent[: groundtruth.shape[0]]
    results = replace_invalid_results(results, groundtruth)
    results[0, :] = groundtruth[0, :]

    official_absent = 1.0 - absent
    present_mask = ~np.isclose(official_absent, 1.0)
    filtered_results = results[present_mask, :]
    filtered_groundtruth = groundtruth[present_mask, :]
    len_all = float(groundtruth.shape[0]) + np.finfo(float).eps
    if filtered_groundtruth.size == 0:
        return {
            "success_curve": np.zeros_like(OVERLAP_THRESHOLDS),
            "precision_curve": np.zeros_like(ERROR_THRESHOLDS),
            "normalized_precision_curve": np.zeros_like(NORMALIZED_ERROR_THRESHOLDS),
            "mean_iou": 0.0,
            "evaluated_frame_count": 0,
        }

    overlaps = coverage_errors(filtered_results, filtered_groundtruth)
    center_errors = center_distance_errors(
        filtered_results,
        filtered_groundtruth,
        normalized=False,
    )
    normalized_center_errors = center_distance_errors(
        filtered_results,
        filtered_groundtruth,
        normalized=True,
    )
    success_curve = np.asarray(
        [np.sum(overlaps > threshold) / len_all for threshold in OVERLAP_THRESHOLDS],
        dtype=float,
    )
    precision_curve = np.asarray(
        [np.sum(center_errors <= threshold) / len_all for threshold in ERROR_THRESHOLDS],
        dtype=float,
    )
    normalized_precision_curve = np.asarray(
        [
            np.sum(normalized_center_errors <= threshold) / len_all
            for threshold in NORMALIZED_ERROR_THRESHOLDS
        ],
        dtype=float,
    )
    valid_overlap = overlaps[overlaps >= 0.0]
    return {
        "success_curve": success_curve,
        "precision_curve": precision_curve,
        "normalized_precision_curve": normalized_precision_curve,
        "mean_iou": float(np.mean(valid_overlap)) if valid_overlap.size else 0.0,
        "evaluated_frame_count": int(filtered_groundtruth.shape[0]),
    }


def replace_invalid_results(results: np.ndarray, groundtruth: np.ndarray) -> np.ndarray:
    sanitized = np.array(results, dtype=float, copy=True)
    for index in range(1, groundtruth.shape[0]):
        result = sanitized[index, :]
        annotation = groundtruth[index, :]
        invalid = (
            np.any(np.isnan(result))
            or np.any(~np.isfinite(result))
            or not np.all(np.isreal(result))
            or result[2] <= 0.0
            or result[3] <= 0.0
        )
        if invalid and not np.any(np.isnan(annotation)):
            sanitized[index, :] = sanitized[index - 1, :]
    return sanitized


def coverage_errors(results: np.ndarray, groundtruth: np.ndarray) -> np.ndarray:
    valid_mask = np.sum(groundtruth > 0.0, axis=1) == 4
    overlaps = -np.ones(groundtruth.shape[0], dtype=float)
    if np.any(valid_mask):
        overlaps[valid_mask] = rect_iou_xywh(
            results[valid_mask, :],
            groundtruth[valid_mask, :],
        )
    return overlaps


def center_distance_errors(
    results: np.ndarray,
    groundtruth: np.ndarray,
    *,
    normalized: bool,
) -> np.ndarray:
    result_centers = np.column_stack(
        (
            results[:, 0] + (results[:, 2] - 1.0) / 2.0,
            results[:, 1] + (results[:, 3] - 1.0) / 2.0,
        )
    )
    gt_centers = np.column_stack(
        (
            groundtruth[:, 0] + (groundtruth[:, 2] - 1.0) / 2.0,
            groundtruth[:, 1] + (groundtruth[:, 3] - 1.0) / 2.0,
        )
    )
    if normalized:
        result_centers[:, 0] = result_centers[:, 0] / groundtruth[:, 2]
        result_centers[:, 1] = result_centers[:, 1] / groundtruth[:, 3]
        gt_centers[:, 0] = gt_centers[:, 0] / groundtruth[:, 2]
        gt_centers[:, 1] = gt_centers[:, 1] / groundtruth[:, 3]
    errors = np.sqrt(np.sum((result_centers - gt_centers) ** 2, axis=1))
    valid_mask = np.sum(groundtruth > 0.0, axis=1) == 4
    errors[~valid_mask] = -1.0
    return errors


def rect_iou_xywh(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left_first = first[:, 0]
    bottom_first = first[:, 1]
    right_first = left_first + first[:, 2] - 1.0
    top_first = bottom_first + first[:, 3] - 1.0
    left_second = second[:, 0]
    bottom_second = second[:, 1]
    right_second = left_second + second[:, 2] - 1.0
    top_second = bottom_second + second[:, 3] - 1.0
    intersection = np.maximum(
        0.0,
        np.minimum(right_first, right_second) - np.maximum(left_first, left_second) + 1.0,
    ) * np.maximum(
        0.0,
        np.minimum(top_first, top_second) - np.maximum(bottom_first, bottom_second) + 1.0,
    )
    area_first = first[:, 2] * first[:, 3]
    area_second = second[:, 2] * second[:, 3]
    return intersection / (area_first + area_second - intersection)


def mean_nonzero_curves(curves: list[np.ndarray]) -> np.ndarray:
    if not curves:
        return np.zeros(1, dtype=float)
    stacked = np.vstack(curves)
    nonzero = stacked[np.sum(stacked, axis=1) > np.finfo(float).eps, :]
    if nonzero.size == 0:
        return np.zeros(stacked.shape[1], dtype=float)
    return np.mean(nonzero, axis=0)


def load_numeric_matrix(path: Path, *, min_columns: int) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        tokens = [token for token in re.split(r"[\s,]+", line.strip()) if token]
        if not tokens:
            continue
        values = [float(token) for token in tokens]
        if len(values) < min_columns:
            raise ValueError(f"{path} row has fewer than {min_columns} columns: {line}")
        rows.append(values)
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    return np.asarray(rows, dtype=float)


def make_config_id(index: int, config: dict[str, float | int]) -> str:
    return (
        f"cfg{index:04d}"
        f"_rb{tag_number(config['refinement_blend'])}"
        f"_se{tag_number(config['search_expansion_factor'])}"
        f"_mx{int(config['max_events'])}"
        f"_mn{int(config['min_events'])}"
        f"_af{tag_number(config['event_activity_floor'])}"
        f"_it{tag_number(config['inactive_activity_threshold'])}"
        f"_rn{tag_number(config['measurement_noise_variance'])}"
    )


def tag_number(value: float | int) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _is_test_split(split: str) -> bool:
    normalized = split.strip().lower().replace("_", "").replace("-", "")
    return normalized in {"test", "testing", "testset", "testingsubset"}


if __name__ == "__main__":
    raise SystemExit(main())
