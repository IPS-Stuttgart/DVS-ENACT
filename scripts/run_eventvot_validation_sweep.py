"""Tune EventVOT DVS-ENACT refinement parameters on the validation subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    load_xywh_result_file,
    resolve_base_result_file,
    resolve_eventvot_split_root,
    resolve_output_results_root,
    resolve_sequence_names,
    run as run_eventvot_refinement,
    validate_rejected_center_hold_config,
)
from run_eventvot_refinement_modes import (
    PROJECTION_CONFIDENCE_FIELDS,
    ProjectedOutputRefiner,
    REFINEMENT_MODES,
)

DEFAULT_REFINEMENT_BLEND = (0.10, 0.25, 0.50, 1.00)
DEFAULT_SEARCH_EXPANSION_FACTOR = (1.10, 1.25, 1.50)
DEFAULT_MAX_EVENTS = (64, 128, 256, 512)
DEFAULT_MIN_EVENTS = (3, 10, 20)
DEFAULT_EVENT_ACTIVITY_FLOOR = (0.00, 0.02, 0.05)
DEFAULT_INACTIVE_ACTIVITY_THRESHOLD = (0.02, 0.05, 0.10)
DEFAULT_MEASUREMENT_NOISE_VARIANCE = (1.0, 4.0, 9.0)

SweepValue = float | int | str | bool | None
NONE_SWEEP_TOKENS = {"none", "null", "off", "disabled", "disable"}

REFINER_GRID_KEYS = (
    "refinement_blend",
    "search_expansion_factor",
    "max_events",
    "min_events",
    "event_activity_floor",
    "inactive_activity_threshold",
    "measurement_noise_variance",
)
PROJECTION_GRID_KEYS = (
    "refinement_mode",
    "projection_width_blend",
    "projection_height_blend",
    "projection_no_clip",
    "projection_size_smoothing",
    "projection_center_smoothing",
    "projection_motion_smoothing",
    "projection_center_clamp_ratio",
    "projection_center_deadband_ratio",
    "projection_size_clamp_ratio",
    "projection_size_deadband_ratio",
    "projection_confidence_field",
    "projection_confidence_floor",
    "projection_confidence_ceiling",
    "projection_min_raw_width_ratio",
    "projection_max_raw_width_ratio",
    "projection_min_raw_height_ratio",
    "projection_max_raw_height_ratio",
)
ACCEPTANCE_GRID_KEYS = (
    "min_accept_used_events",
    "min_accept_active_measurements",
    "min_accept_mean_activity",
    "min_accept_candidate_iou",
    "min_accept_area_ratio",
    "max_accept_area_ratio",
    "max_accept_center_shift_ratio",
    "min_raw_candidate_iou",
    "min_raw_candidate_area_ratio",
    "max_raw_candidate_area_ratio",
    "max_raw_center_shift_ratio",
    "min_polarity_consistency_fraction",
    "min_mean_event_polarity_weight",
    "max_quadratic_form_per_active_measurement",
    "min_active_fraction",
    "max_temporal_center_shift_ratio",
    "max_temporal_size_change_ratio",
    "max_motion_prediction_error_ratio",
    "max_rejected_center_hold_frames",
    "rejected_center_hold_decay",
)
OPTIONAL_ACCEPTANCE_GRID_KEYS = {
    "min_raw_candidate_iou",
    "min_raw_candidate_area_ratio",
    "max_raw_candidate_area_ratio",
    "max_raw_center_shift_ratio",
    "min_polarity_consistency_fraction",
    "min_mean_event_polarity_weight",
    "max_quadratic_form_per_active_measurement",
    "min_active_fraction",
    "max_temporal_center_shift_ratio",
    "max_temporal_size_change_ratio",
    "max_motion_prediction_error_ratio",
}
STRING_GRID_KEYS = {"refinement_mode", "projection_confidence_field"}
BOOL_GRID_KEYS = {"projection_no_clip"}
OPTIONAL_FLOAT_GRID_KEYS = {
    "projection_width_blend",
    "projection_height_blend",
    "projection_size_smoothing",
    "projection_center_smoothing",
    "projection_motion_smoothing",
    "projection_center_clamp_ratio",
    "projection_center_deadband_ratio",
    "projection_size_clamp_ratio",
    "projection_size_deadband_ratio",
    "projection_confidence_floor",
    "projection_confidence_ceiling",
    "projection_min_raw_width_ratio",
    "projection_max_raw_width_ratio",
    "projection_min_raw_height_ratio",
    "projection_max_raw_height_ratio",
}
INT_GRID_KEYS = {
    "max_events",
    "min_events",
    "min_accept_used_events",
    "min_accept_active_measurements",
    "max_rejected_center_hold_frames",
}
CONFIG_ID_KEYS = REFINER_GRID_KEYS + PROJECTION_GRID_KEYS + ACCEPTANCE_GRID_KEYS

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
    add_projection_sweep_arguments(parser)
    add_acceptance_sweep_arguments(parser)
    return parser


def add_projection_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    """Add output-projection sweep arguments."""
    parser.add_argument(
        "--refinement-mode",
        nargs="+",
        default=("box",),
        help=(
            "Output projection modes to evaluate. 'box' keeps the full "
            "DVS-ENACT box update, 'center-only' keeps the base size, and "
            "'size-only', 'scale-only', 'width-only', and 'height-only' keep "
            "the base center."
        ),
    )
    parser.add_argument(
        "--projection-width-blend",
        nargs="+",
        default=("none",),
        help=(
            "Optional projection width blends. Use 'none' to use the normal "
            "--refinement-blend output width."
        ),
    )
    parser.add_argument(
        "--projection-height-blend",
        nargs="+",
        default=("none",),
        help=(
            "Optional projection height blends. Use 'none' to use the normal "
            "--refinement-blend output height."
        ),
    )
    parser.add_argument(
        "--projection-no-clip",
        action="store_true",
        help="Reject projected outputs that would be clipped by image bounds.",
    )
    parser.add_argument(
        "--projection-size-smoothing",
        nargs="+",
        default=("none",),
        help=(
            "Optional temporal size-smoothing values. Use 'none' to disable. "
            "A value of 0 uses the current projection; 1 holds the previous "
            "accepted projected size."
        ),
    )
    parser.add_argument(
        "--projection-center-smoothing",
        nargs="+",
        default=("none",),
        help=(
            "Optional temporal center-smoothing values. Use 'none' to disable. "
            "A value of 0 uses the current projection; 1 holds the previous "
            "accepted projected center."
        ),
    )
    parser.add_argument(
        "--projection-motion-smoothing",
        nargs="+",
        default=("none",),
        help=(
            "Optional motion-compensated center-smoothing values. Use 'none' "
            "to disable. A value of 0 uses the current projection; 1 follows "
            "the previous accepted projected center advanced by base-tracker motion."
        ),
    )
    parser.add_argument(
        "--projection-size-deadband-ratio",
        nargs="+",
        default=("none",),
        help=(
            "Optional per-axis size deadband values relative to base width/height. "
            "Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--projection-center-deadband-ratio",
        nargs="+",
        default=("none",),
        help=(
            "Optional center-shift deadband values relative to base-box diagonal. "
            "Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--projection-center-clamp-ratio",
        nargs="+",
        default=("none",),
        help=(
            "Optional center-shift clamp values relative to base-box diagonal. "
            "Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--projection-size-clamp-ratio",
        nargs="+",
        default=("none",),
        help=(
            "Optional per-axis size clamp values relative to base width/height. "
            "Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--projection-confidence-field",
        nargs="+",
        default=("none",),
        help=(
            "Optional confidence fields for adaptive projection strength. "
            f"Choices: {', '.join(PROJECTION_CONFIDENCE_FIELDS)}; use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--projection-confidence-floor",
        nargs="+",
        default=("none",),
        help="Confidence values that map projected correction strength to zero.",
    )
    parser.add_argument(
        "--projection-confidence-ceiling",
        nargs="+",
        default=("none",),
        help="Confidence values that map projected correction strength to one.",
    )
    parser.add_argument("--projection-min-raw-width-ratio", nargs="+", default=("none",))
    parser.add_argument("--projection-max-raw-width-ratio", nargs="+", default=("none",))
    parser.add_argument("--projection-min-raw-height-ratio", nargs="+", default=("none",))
    parser.add_argument("--projection-max-raw-height-ratio", nargs="+", default=("none",))


def add_acceptance_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    """Add acceptance-gate sweep arguments.

    Each value accepts either repeated shell tokens (``--arg 0.1 0.2``) or a
    single comma/whitespace-separated workflow-dispatch string
    (``--arg "0.1,0.2"``). Optional acceptance gates also accept ``none``,
    ``null``, or ``off`` to leave the corresponding guard disabled. Defaults are
    singletons so the historical refiner grid size is unchanged unless the
    caller explicitly sweeps gates.
    """
    parser.add_argument(
        "--min-accept-used-events",
        nargs="+",
        default=("10",),
        help="Acceptance sweep values for the minimum used event count.",
    )
    parser.add_argument(
        "--min-accept-active-measurements",
        nargs="+",
        default=("3",),
        help="Acceptance sweep values for the minimum active measurements.",
    )
    parser.add_argument(
        "--min-accept-mean-activity",
        nargs="+",
        default=("0.10",),
        help="Acceptance sweep values for the minimum mean event activity.",
    )
    parser.add_argument(
        "--min-accept-candidate-iou",
        nargs="+",
        default=("0.60",),
        help="Acceptance sweep values for the minimum base/refined IoU.",
    )
    parser.add_argument(
        "--min-accept-area-ratio",
        nargs="+",
        default=("0.50",),
        help="Acceptance sweep values for the minimum refined/base area ratio.",
    )
    parser.add_argument(
        "--max-accept-area-ratio",
        nargs="+",
        default=("1.50",),
        help="Acceptance sweep values for the maximum refined/base area ratio.",
    )
    parser.add_argument(
        "--max-accept-center-shift-ratio",
        nargs="+",
        default=("0.25",),
        help=(
            "Acceptance sweep values for the maximum center shift normalized "
            "by the base-box diagonal."
        ),
    )
    parser.add_argument(
        "--min-raw-candidate-iou",
        nargs="+",
        default=("none",),
        help="Acceptance sweep values for minimum raw/base IoU; use 'none' to disable.",
    )
    parser.add_argument(
        "--min-raw-candidate-area-ratio",
        nargs="+",
        default=("none",),
        help="Acceptance sweep values for minimum raw/base area ratio; use 'none' to disable.",
    )
    parser.add_argument(
        "--max-raw-candidate-area-ratio",
        nargs="+",
        default=("none",),
        help="Acceptance sweep values for maximum raw/base area ratio; use 'none' to disable.",
    )
    parser.add_argument(
        "--max-raw-center-shift-ratio",
        nargs="+",
        default=("none",),
        help="Acceptance sweep values for maximum raw/base center shift ratio; use 'none' to disable.",
    )
    parser.add_argument(
        "--min-polarity-consistency-fraction",
        nargs="+",
        default=("none",),
        help="Acceptance sweep values for minimum polarity consistency; use 'none' to disable.",
    )
    parser.add_argument(
        "--min-mean-event-polarity-weight",
        nargs="+",
        default=("none",),
        help="Acceptance sweep values for minimum mean event polarity weight; use 'none' to disable.",
    )
    parser.add_argument(
        "--max-quadratic-form-per-active-measurement",
        nargs="+",
        default=("none",),
        help=(
            "Acceptance sweep values for max quadratic-form-per-active-measurement; "
            "use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--min-active-fraction",
        nargs="+",
        default=("none",),
        help="Acceptance sweep values for minimum active fraction; use 'none' to disable.",
    )
    parser.add_argument(
        "--max-temporal-center-shift-ratio",
        nargs="+",
        default=("none",),
        help=(
            "Acceptance sweep values for max frame-to-frame center shift ratio; "
            "use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--max-temporal-size-change-ratio",
        nargs="+",
        default=("none",),
        help=(
            "Acceptance sweep values for max frame-to-frame size change ratio; "
            "use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--max-motion-prediction-error-ratio",
        nargs="+",
        default=("none",),
        help=(
            "Acceptance sweep values for max error from previous output plus "
            "base-tracker motion; use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--max-rejected-center-hold-frames",
        nargs="+",
        default=("0",),
        help=(
            "Output-policy sweep values for how many rejected frames may reuse "
            "the last accepted center correction. 0 disables the hold."
        ),
    )
    parser.add_argument(
        "--rejected-center-hold-decay",
        nargs="+",
        default=("1.0",),
        help=(
            "Output-policy sweep values for the per-frame center-hold decay. "
            "1 keeps the full offset; 0 drops it immediately."
        ),
    )


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
                    acceptance_config=acceptance_config_from_config(config),
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


def iter_parameter_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    keys = REFINER_GRID_KEYS + PROJECTION_GRID_KEYS + ACCEPTANCE_GRID_KEYS
    projection_values = projection_value_lists_from_args(args)
    acceptance_values = acceptance_value_lists_from_args(args)
    values = tuple(
        getattr(args, key)
        if key in REFINER_GRID_KEYS
        else projection_values[key]
        if key in PROJECTION_GRID_KEYS
        else acceptance_values[key]
        for key in keys
    )
    configs: list[dict[str, Any]] = []
    for combination in itertools.product(*values):
        config: dict[str, Any] = {}
        for key, value in zip(keys, combination, strict=True):
            config[key] = normalize_config_value(key, value)
        if has_incomplete_projection_blend(config):
            continue
        if has_incomplete_projection_confidence(config):
            continue
        validate_projection_config(config)
        configs.append(config)
    if not configs:
        raise ValueError("Validation sweep grid did not contain any valid configs")
    return configs


def projection_value_lists_from_args(args: argparse.Namespace) -> dict[str, list[Any]]:
    return {
        "refinement_mode": parse_refinement_mode_values(args.refinement_mode),
        "projection_width_blend": parse_sweep_values(
            args.projection_width_blend,
            cast=float,
            argument_name="--projection-width-blend",
            allow_none=True,
        ),
        "projection_height_blend": parse_sweep_values(
            args.projection_height_blend,
            cast=float,
            argument_name="--projection-height-blend",
            allow_none=True,
        ),
        "projection_no_clip": [bool(args.projection_no_clip)],
        "projection_size_smoothing": parse_sweep_values(
            args.projection_size_smoothing,
            cast=float,
            argument_name="--projection-size-smoothing",
            allow_none=True,
        ),
        "projection_center_smoothing": parse_sweep_values(
            args.projection_center_smoothing,
            cast=float,
            argument_name="--projection-center-smoothing",
            allow_none=True,
        ),
        "projection_motion_smoothing": parse_sweep_values(
            args.projection_motion_smoothing,
            cast=float,
            argument_name="--projection-motion-smoothing",
            allow_none=True,
        ),
        "projection_size_deadband_ratio": parse_sweep_values(
            args.projection_size_deadband_ratio,
            cast=float,
            argument_name="--projection-size-deadband-ratio",
            allow_none=True,
        ),
        "projection_size_clamp_ratio": parse_sweep_values(
            args.projection_size_clamp_ratio,
            cast=float,
            argument_name="--projection-size-clamp-ratio",
            allow_none=True,
        ),
        "projection_center_deadband_ratio": parse_sweep_values(
            args.projection_center_deadband_ratio,
            cast=float,
            argument_name="--projection-center-deadband-ratio",
            allow_none=True,
        ),
        "projection_center_clamp_ratio": parse_sweep_values(
            args.projection_center_clamp_ratio,
            cast=float,
            argument_name="--projection-center-clamp-ratio",
            allow_none=True,
        ),
        "projection_confidence_field": parse_projection_confidence_field_values(
            args.projection_confidence_field
        ),
        "projection_confidence_floor": parse_sweep_values(
            args.projection_confidence_floor,
            cast=float,
            argument_name="--projection-confidence-floor",
            allow_none=True,
        ),
        "projection_confidence_ceiling": parse_sweep_values(
            args.projection_confidence_ceiling,
            cast=float,
            argument_name="--projection-confidence-ceiling",
            allow_none=True,
        ),
        "projection_min_raw_width_ratio": parse_sweep_values(
            args.projection_min_raw_width_ratio,
            cast=float,
            argument_name="--projection-min-raw-width-ratio",
            allow_none=True,
        ),
        "projection_max_raw_width_ratio": parse_sweep_values(
            args.projection_max_raw_width_ratio,
            cast=float,
            argument_name="--projection-max-raw-width-ratio",
            allow_none=True,
        ),
        "projection_min_raw_height_ratio": parse_sweep_values(
            args.projection_min_raw_height_ratio,
            cast=float,
            argument_name="--projection-min-raw-height-ratio",
            allow_none=True,
        ),
        "projection_max_raw_height_ratio": parse_sweep_values(
            args.projection_max_raw_height_ratio,
            cast=float,
            argument_name="--projection-max-raw-height-ratio",
            allow_none=True,
        ),
    }


def parse_refinement_mode_values(raw_values: list[str] | tuple[str, ...]) -> list[str]:
    """Parse repeated or comma/whitespace-separated projection mode values."""
    values: list[str] = []
    for raw_value in raw_values:
        for token in re.split(r"[\s,]+", str(raw_value).strip()):
            if not token:
                continue
            if token not in REFINEMENT_MODES:
                expected = ", ".join(REFINEMENT_MODES)
                raise ValueError(
                    f"Invalid value for --refinement-mode: {token!r}; "
                    f"expected one of {expected}"
                )
            values.append(token)
    if not values:
        raise ValueError("--refinement-mode must contain at least one value")
    return list(dict.fromkeys(values))


def parse_projection_confidence_field_values(
    raw_values: list[str] | tuple[str, ...],
) -> list[str | None]:
    """Parse repeated or comma/whitespace-separated projection confidence fields."""
    values: list[str | None] = []
    for raw_value in raw_values:
        for token in re.split(r"[\s,]+", str(raw_value).strip()):
            if not token:
                continue
            normalized = token.strip().lower()
            if normalized in NONE_SWEEP_TOKENS:
                values.append(None)
                continue
            if token not in PROJECTION_CONFIDENCE_FIELDS:
                expected = ", ".join(PROJECTION_CONFIDENCE_FIELDS)
                raise ValueError(
                    f"Invalid value for --projection-confidence-field: {token!r}; "
                    f"expected one of {expected} or none"
                )
            values.append(token)
    if not values:
        raise ValueError("--projection-confidence-field must contain at least one value")
    return list(dict.fromkeys(values))


def acceptance_value_lists_from_args(args: argparse.Namespace) -> dict[str, list[SweepValue]]:
    return {
        "min_accept_used_events": parse_sweep_values(
            args.min_accept_used_events,
            cast=int,
            argument_name="--min-accept-used-events",
        ),
        "min_accept_active_measurements": parse_sweep_values(
            args.min_accept_active_measurements,
            cast=int,
            argument_name="--min-accept-active-measurements",
        ),
        "min_accept_mean_activity": parse_sweep_values(
            args.min_accept_mean_activity,
            cast=float,
            argument_name="--min-accept-mean-activity",
        ),
        "min_accept_candidate_iou": parse_sweep_values(
            args.min_accept_candidate_iou,
            cast=float,
            argument_name="--min-accept-candidate-iou",
        ),
        "min_accept_area_ratio": parse_sweep_values(
            args.min_accept_area_ratio,
            cast=float,
            argument_name="--min-accept-area-ratio",
        ),
        "max_accept_area_ratio": parse_sweep_values(
            args.max_accept_area_ratio,
            cast=float,
            argument_name="--max-accept-area-ratio",
        ),
        "max_accept_center_shift_ratio": parse_sweep_values(
            args.max_accept_center_shift_ratio,
            cast=float,
            argument_name="--max-accept-center-shift-ratio",
        ),
        "min_raw_candidate_iou": parse_sweep_values(
            args.min_raw_candidate_iou,
            cast=float,
            argument_name="--min-raw-candidate-iou",
            allow_none=True,
        ),
        "min_raw_candidate_area_ratio": parse_sweep_values(
            args.min_raw_candidate_area_ratio,
            cast=float,
            argument_name="--min-raw-candidate-area-ratio",
            allow_none=True,
        ),
        "max_raw_candidate_area_ratio": parse_sweep_values(
            args.max_raw_candidate_area_ratio,
            cast=float,
            argument_name="--max-raw-candidate-area-ratio",
            allow_none=True,
        ),
        "max_raw_center_shift_ratio": parse_sweep_values(
            args.max_raw_center_shift_ratio,
            cast=float,
            argument_name="--max-raw-center-shift-ratio",
            allow_none=True,
        ),
        "min_polarity_consistency_fraction": parse_sweep_values(
            args.min_polarity_consistency_fraction,
            cast=float,
            argument_name="--min-polarity-consistency-fraction",
            allow_none=True,
        ),
        "min_mean_event_polarity_weight": parse_sweep_values(
            args.min_mean_event_polarity_weight,
            cast=float,
            argument_name="--min-mean-event-polarity-weight",
            allow_none=True,
        ),
        "max_quadratic_form_per_active_measurement": parse_sweep_values(
            args.max_quadratic_form_per_active_measurement,
            cast=float,
            argument_name="--max-quadratic-form-per-active-measurement",
            allow_none=True,
        ),
        "min_active_fraction": parse_sweep_values(
            args.min_active_fraction,
            cast=float,
            argument_name="--min-active-fraction",
            allow_none=True,
        ),
        "max_temporal_center_shift_ratio": parse_sweep_values(
            args.max_temporal_center_shift_ratio,
            cast=float,
            argument_name="--max-temporal-center-shift-ratio",
            allow_none=True,
        ),
        "max_temporal_size_change_ratio": parse_sweep_values(
            args.max_temporal_size_change_ratio,
            cast=float,
            argument_name="--max-temporal-size-change-ratio",
            allow_none=True,
        ),
        "max_motion_prediction_error_ratio": parse_sweep_values(
            args.max_motion_prediction_error_ratio,
            cast=float,
            argument_name="--max-motion-prediction-error-ratio",
            allow_none=True,
        ),
        "max_rejected_center_hold_frames": parse_sweep_values(
            args.max_rejected_center_hold_frames,
            cast=int,
            argument_name="--max-rejected-center-hold-frames",
        ),
        "rejected_center_hold_decay": parse_sweep_values(
            args.rejected_center_hold_decay,
            cast=float,
            argument_name="--rejected-center-hold-decay",
        ),
    }


def parse_sweep_values(
    raw_values: list[str] | tuple[str, ...],
    *,
    cast,
    argument_name: str,
    allow_none: bool = False,
) -> list[SweepValue]:
    """Parse repeated or comma/whitespace-separated sweep values."""
    values: list[SweepValue] = []
    for raw_value in raw_values:
        for token in re.split(r"[\s,]+", str(raw_value).strip()):
            if not token:
                continue
            if allow_none and token.strip().lower() in NONE_SWEEP_TOKENS:
                values.append(None)
                continue
            try:
                values.append(cast(token))
            except ValueError as error:
                raise ValueError(
                    f"Invalid value for {argument_name}: {token!r}"
                ) from error
    if not values:
        raise ValueError(f"{argument_name} must contain at least one value")

    unique: list[SweepValue] = []
    seen: set[SweepValue] = set()
    for value in values:
        if value in seen:
            continue
        unique.append(value)
        seen.add(value)
    return unique


def normalize_config_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in INT_GRID_KEYS:
        return int(value)
    if key in STRING_GRID_KEYS:
        return str(value)
    if key in BOOL_GRID_KEYS:
        return bool(value)
    return float(value)


def validate_projection_config(config: dict[str, Any]) -> None:
    if has_incomplete_projection_blend(config):
        raise ValueError("projection width/height blends must both be set or both be none")
    if has_incomplete_projection_confidence(config):
        raise ValueError(
            "projection confidence field, floor, and ceiling must all be set or all be none"
        )
    for name in OPTIONAL_FLOAT_GRID_KEYS:
        value = config[name]
        if value is not None and float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    smoothing = config["projection_size_smoothing"]
    if smoothing is not None and float(smoothing) > 1.0:
        raise ValueError("projection_size_smoothing must be between 0 and 1")
    center_smoothing = config["projection_center_smoothing"]
    if center_smoothing is not None and float(center_smoothing) > 1.0:
        raise ValueError("projection_center_smoothing must be between 0 and 1")
    motion_smoothing = config["projection_motion_smoothing"]
    if motion_smoothing is not None and float(motion_smoothing) > 1.0:
        raise ValueError("projection_motion_smoothing must be between 0 and 1")
    deadband = config["projection_size_deadband_ratio"]
    if deadband is not None and float(deadband) < 0.0:
        raise ValueError("projection_size_deadband_ratio must be non-negative")
    size_clamp = config["projection_size_clamp_ratio"]
    if size_clamp is not None and float(size_clamp) < 0.0:
        raise ValueError("projection_size_clamp_ratio must be non-negative")
    center_deadband = config["projection_center_deadband_ratio"]
    if center_deadband is not None and float(center_deadband) < 0.0:
        raise ValueError("projection_center_deadband_ratio must be non-negative")
    center_clamp = config["projection_center_clamp_ratio"]
    if center_clamp is not None and float(center_clamp) < 0.0:
        raise ValueError("projection_center_clamp_ratio must be non-negative")
    confidence_floor = config["projection_confidence_floor"]
    confidence_ceiling = config["projection_confidence_ceiling"]
    if confidence_floor is not None and confidence_ceiling is not None:
        if confidence_floor >= confidence_ceiling:
            raise ValueError("projection_confidence_floor must be less than ceiling")
    _validate_min_max_pair(
        config,
        "projection_min_raw_width_ratio",
        "projection_max_raw_width_ratio",
    )
    _validate_min_max_pair(
        config,
        "projection_min_raw_height_ratio",
        "projection_max_raw_height_ratio",
    )


def _validate_min_max_pair(config: dict[str, Any], min_key: str, max_key: str) -> None:
    minimum = config[min_key]
    maximum = config[max_key]
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{min_key} must not exceed {max_key}")


def has_incomplete_projection_blend(config: dict[str, Any]) -> bool:
    return (config["projection_width_blend"] is None) != (
        config["projection_height_blend"] is None
    )


def has_incomplete_projection_confidence(config: dict[str, Any]) -> bool:
    values = (
        config["projection_confidence_field"],
        config["projection_confidence_floor"],
        config["projection_confidence_ceiling"],
    )
    return any(value is None for value in values) and any(
        value is not None for value in values
    )


def make_refiner(
    config: dict[str, SweepValue],
    args: argparse.Namespace,
) -> Any:
    refiner = DVSContourRefiner(
        DVSContourRefinerConfig(
            input_bbox_format="xywh",
            output_bbox_format="xywh",
            event_crop_coordinate_mode="half_open",
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
    if (
        config["refinement_mode"] == "box"
        and config["projection_size_smoothing"] is None
        and config["projection_center_smoothing"] is None
        and config["projection_motion_smoothing"] is None
        and config["projection_center_clamp_ratio"] is None
        and config["projection_center_deadband_ratio"] is None
        and config["projection_size_clamp_ratio"] is None
        and config["projection_size_deadband_ratio"] is None
        and config["projection_confidence_field"] is None
    ):
        return refiner
    return ProjectedOutputRefiner(
        refiner,
        refinement_mode=str(config["refinement_mode"]),
        projection_width_blend=config["projection_width_blend"],
        projection_height_blend=config["projection_height_blend"],
        projection_no_clip=bool(config["projection_no_clip"]),
        projection_size_smoothing=config["projection_size_smoothing"],
        projection_center_smoothing=config["projection_center_smoothing"],
        projection_motion_smoothing=config["projection_motion_smoothing"],
        projection_center_clamp_ratio=config["projection_center_clamp_ratio"],
        projection_center_deadband_ratio=config["projection_center_deadband_ratio"],
        projection_size_clamp_ratio=config["projection_size_clamp_ratio"],
        projection_size_deadband_ratio=config["projection_size_deadband_ratio"],
        projection_confidence_field=config["projection_confidence_field"],
        projection_confidence_floor=config["projection_confidence_floor"],
        projection_confidence_ceiling=config["projection_confidence_ceiling"],
        projection_min_raw_width_ratio=config["projection_min_raw_width_ratio"],
        projection_max_raw_width_ratio=config["projection_max_raw_width_ratio"],
        projection_min_raw_height_ratio=config["projection_min_raw_height_ratio"],
        projection_max_raw_height_ratio=config["projection_max_raw_height_ratio"],
    )


def acceptance_config_from_config(config: dict[str, SweepValue]) -> EventVOTAcceptanceConfig:
    validate_rejected_center_hold_config(
        int(config["max_rejected_center_hold_frames"]),
        float(config["rejected_center_hold_decay"]),
    )
    return EventVOTAcceptanceConfig(
        min_used_event_count=int(config["min_accept_used_events"]),
        min_active_measurement_count=int(config["min_accept_active_measurements"]),
        min_mean_event_activity=float(config["min_accept_mean_activity"]),
        min_candidate_iou=float(config["min_accept_candidate_iou"]),
        min_candidate_area_ratio=float(config["min_accept_area_ratio"]),
        max_candidate_area_ratio=float(config["max_accept_area_ratio"]),
        max_center_shift_ratio=float(config["max_accept_center_shift_ratio"]),
        min_raw_candidate_iou=optional_float_config_value(
            config,
            "min_raw_candidate_iou",
        ),
        min_raw_candidate_area_ratio=optional_float_config_value(
            config,
            "min_raw_candidate_area_ratio",
        ),
        max_raw_candidate_area_ratio=optional_float_config_value(
            config,
            "max_raw_candidate_area_ratio",
        ),
        max_raw_center_shift_ratio=optional_float_config_value(
            config,
            "max_raw_center_shift_ratio",
        ),
        min_polarity_consistency_fraction=optional_float_config_value(
            config,
            "min_polarity_consistency_fraction",
        ),
        min_mean_event_polarity_weight=optional_float_config_value(
            config,
            "min_mean_event_polarity_weight",
        ),
        max_quadratic_form_per_active_measurement=optional_float_config_value(
            config,
            "max_quadratic_form_per_active_measurement",
        ),
        min_active_fraction=optional_float_config_value(config, "min_active_fraction"),
        max_temporal_center_shift_ratio=optional_float_config_value(
            config,
            "max_temporal_center_shift_ratio",
        ),
        max_temporal_size_change_ratio=optional_float_config_value(
            config,
            "max_temporal_size_change_ratio",
        ),
        max_motion_prediction_error_ratio=optional_float_config_value(
            config,
            "max_motion_prediction_error_ratio",
        ),
        max_rejected_center_hold_frames=int(config["max_rejected_center_hold_frames"]),
        rejected_center_hold_decay=float(config["rejected_center_hold_decay"]),
    )


def optional_float_config_value(
    config: dict[str, SweepValue],
    key: str,
) -> float | None:
    """Return an optional float gate value, preserving disabled ``None``."""
    value = config.get(key)
    if value is None:
        return None
    return float(value)


def make_result_row(
    config_id: str,
    tracker_name: str,
    config: dict[str, SweepValue],
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
    configs: list[dict[str, SweepValue]],
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


def make_config_id(index: int, config: dict[str, SweepValue]) -> str:
    """Return a stable, cache-safe identifier for a complete sweep config.

    The readable tags intentionally summarize only the historically most useful
    parameters. The hash prefix is computed from every refiner and acceptance
    grid key, including gates that are not printed below, so changing any tuned
    parameter changes the output directory name and diagnostics filename.
    """
    config_hash = make_config_hash(config)
    return (
        f"cfg{index:04d}_h{config_hash}"
        f"_rb{tag_number(config['refinement_blend'])}"
        f"_se{tag_number(config['search_expansion_factor'])}"
        f"_mx{int(config['max_events'])}"
        f"_mn{int(config['min_events'])}"
        f"_af{tag_number(config['event_activity_floor'])}"
        f"_it{tag_number(config['inactive_activity_threshold'])}"
        f"_rn{tag_number(config['measurement_noise_variance'])}"
        f"_au{int(config['min_accept_used_events'])}"
        f"_aa{int(config['min_accept_active_measurements'])}"
        f"_act{tag_number(config['min_accept_mean_activity'])}"
        f"_ai{tag_number(config['min_accept_candidate_iou'])}"
        f"_ar{tag_number(config['min_accept_area_ratio'])}"
        f"_ax{tag_number(config['max_accept_area_ratio'])}"
        f"_cs{tag_number(config['max_accept_center_shift_ratio'])}"
        f"_pm{tag_text(config['refinement_mode'])}"
    )


def make_config_hash(config: dict[str, SweepValue]) -> str:
    """Hash all sweep parameters that affect refinement or acceptance."""
    payload = {
        key: canonical_config_value(key, config[key])
        for key in CONFIG_ID_KEYS
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def canonical_config_value(key: str, value: Any) -> float | int | str | bool | None:
    if value is None:
        return None
    if key in INT_GRID_KEYS:
        return int(value)
    if key in STRING_GRID_KEYS:
        return str(value)
    if key in BOOL_GRID_KEYS:
        return bool(value)
    if value is None:
        return None
    numeric_value = float(value)
    if np.isposinf(numeric_value):
        return "inf"
    if np.isneginf(numeric_value):
        return "-inf"
    if np.isnan(numeric_value):
        raise ValueError(f"{key} cannot be NaN in a validation-sweep config")
    return numeric_value


def tag_number(value: float | int) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def tag_text(value: Any) -> str:
    return str(value).replace("-", "").replace("_", "")


def _is_test_split(split: str) -> bool:
    normalized = split.strip().lower().replace("_", "").replace("-", "")
    return normalized in {"test", "testing", "testset", "testingsubset"}


if __name__ == "__main__":
    raise SystemExit(main())
