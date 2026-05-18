"""Sweep EventVOT replay policies from existing diagnostics.

This script is the cheap companion to ``run_eventvot_validation_sweep.py``.  It
does not recompute DVS-ENACT refinements.  Instead, it reuses an existing
diagnostics JSON and tries different replay-output projection and acceptance
policies, then optionally evaluates the rewritten EventVOT result files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_eventvot_acceptance_replay import (  # noqa: E402
    EventVOTAcceptanceReplayOptions,
    REPLAY_OUTPUT_MODES,
    ReplayAcceptanceConfig,
    ReplayOutputProjectionConfig,
    run as run_acceptance_replay,
    validate_output_projection_config,
)
from run_eventvot_refinement_modes import PROJECTION_CONFIDENCE_FIELDS  # noqa: E402

NONE_SWEEP_TOKENS = {"none", "null", "off", "disabled", "disable"}
DIAGNOSTIC_SWEEP_TOKENS = {"diagnostic", "original", "default", "keep"}
DIAGNOSTIC_VALUE = object()


@dataclass(frozen=True)
class ReplaySweepConfig:
    """One replay-sweep configuration."""

    output_projection: ReplayOutputProjectionConfig
    acceptance_overrides: dict[str, Any]


def main() -> int:
    args = build_parser().parse_args()
    payload = run_projection_sweep(args)
    print(json.dumps(payload["summary"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep replay-output policies from an EventVOT DVS-ENACT "
            "diagnostics JSON without recomputing refinements."
        )
    )
    parser.add_argument("--diagnostics-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eventvot-root", type=Path)
    parser.add_argument("--base-results", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    add_projection_grid_arguments(parser)
    add_acceptance_grid_arguments(parser)
    return parser


def add_projection_grid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--replay-output-mode", nargs="+", default=("diagnostic",))
    parser.add_argument("--replay-output-blend", nargs="+", default=("none",))
    parser.add_argument("--replay-output-size-smoothing", nargs="+", default=("none",))
    parser.add_argument(
        "--replay-output-center-clamp-ratio",
        nargs="+",
        default=("none",),
    )
    parser.add_argument(
        "--replay-output-center-deadband-ratio",
        nargs="+",
        default=("none",),
    )
    parser.add_argument(
        "--replay-output-size-clamp-ratio",
        nargs="+",
        default=("none",),
    )
    parser.add_argument(
        "--replay-output-size-deadband-ratio",
        nargs="+",
        default=("none",),
    )
    parser.add_argument(
        "--replay-output-confidence-field",
        nargs="+",
        default=("none",),
    )
    parser.add_argument(
        "--replay-output-confidence-floor",
        nargs="+",
        default=("none",),
    )
    parser.add_argument(
        "--replay-output-confidence-ceiling",
        nargs="+",
        default=("none",),
    )


def add_acceptance_grid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-accept-used-events",
        dest="min_used_event_count",
        nargs="+",
        default=("diagnostic",),
    )
    parser.add_argument(
        "--min-accept-active-measurements",
        dest="min_active_measurement_count",
        nargs="+",
        default=("diagnostic",),
    )
    parser.add_argument(
        "--min-accept-mean-activity",
        dest="min_mean_event_activity",
        nargs="+",
        default=("diagnostic",),
    )
    parser.add_argument(
        "--min-accept-candidate-iou",
        dest="min_candidate_iou",
        nargs="+",
        default=("diagnostic",),
    )
    parser.add_argument(
        "--min-accept-area-ratio",
        dest="min_candidate_area_ratio",
        nargs="+",
        default=("diagnostic",),
    )
    parser.add_argument(
        "--max-accept-area-ratio",
        dest="max_candidate_area_ratio",
        nargs="+",
        default=("diagnostic",),
    )
    parser.add_argument(
        "--max-accept-center-shift-ratio",
        dest="max_center_shift_ratio",
        nargs="+",
        default=("diagnostic",),
    )
    parser.add_argument(
        "--min-raw-candidate-iou",
        nargs="+",
        default=("diagnostic",),
    )
    parser.add_argument(
        "--min-raw-area-ratio",
        dest="min_raw_candidate_area_ratio",
        nargs="+",
        default=("diagnostic",),
    )
    parser.add_argument(
        "--max-raw-area-ratio",
        dest="max_raw_candidate_area_ratio",
        nargs="+",
        default=("diagnostic",),
    )
    parser.add_argument(
        "--max-raw-center-shift-ratio",
        dest="max_raw_center_shift_ratio",
        nargs="+",
        default=("diagnostic",),
    )
    parser.add_argument(
        "--min-polarity-consistency-fraction",
        nargs="+",
        default=("diagnostic",),
    )
    parser.add_argument(
        "--min-mean-event-polarity-weight",
        nargs="+",
        default=("diagnostic",),
    )
    parser.add_argument(
        "--max-quadratic-form-per-active-measurement",
        nargs="+",
        default=("diagnostic",),
    )
    parser.add_argument(
        "--min-active-fraction",
        nargs="+",
        default=("diagnostic",),
    )


def run_projection_sweep(args: argparse.Namespace) -> dict[str, Any]:
    configs = list(iter_sweep_grid(args))
    if args.max_configs is not None:
        configs = configs[: args.max_configs]
    if not configs:
        raise ValueError("Replay policy sweep grid did not contain any configs")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs, start=1):
        config_id = make_sweep_config_id(config_index, config)
        result_dir = output_root / "results" / config_id
        summary_json = output_root / "summaries" / f"{config_id}.json"
        payload = run_acceptance_replay(
            EventVOTAcceptanceReplayOptions(
                diagnostics_json=args.diagnostics_json,
                output_results=result_dir,
                eventvot_root=args.eventvot_root,
                base_results=args.base_results,
                split=args.split,
                sequences=tuple(args.sequence),
                summary_json=summary_json,
                skip_evaluation=args.skip_evaluation,
                output_projection_config=config.output_projection,
                config_overrides=config.acceptance_overrides,
            )
        )
        rows.append(make_result_row(config_id, config, result_dir, payload))
        write_sweep_outputs(rows, configs, args)

    return write_sweep_outputs(rows, configs, args)


def iter_sweep_grid(args: argparse.Namespace) -> list[ReplaySweepConfig]:
    configs: list[ReplaySweepConfig] = []
    for output_projection in iter_projection_grid(args):
        for acceptance_overrides in iter_acceptance_override_grid(args):
            configs.append(
                ReplaySweepConfig(
                    output_projection=output_projection,
                    acceptance_overrides=acceptance_overrides,
                )
            )
    return configs


def iter_projection_grid(args: argparse.Namespace) -> list[ReplayOutputProjectionConfig]:
    value_lists = {
        "mode": parse_mode_values(args.replay_output_mode),
        "blend": parse_sweep_values(
            args.replay_output_blend,
            argument_name="--replay-output-blend",
        ),
        "size_smoothing": parse_sweep_values(
            args.replay_output_size_smoothing,
            argument_name="--replay-output-size-smoothing",
        ),
        "center_clamp_ratio": parse_sweep_values(
            args.replay_output_center_clamp_ratio,
            argument_name="--replay-output-center-clamp-ratio",
        ),
        "center_deadband_ratio": parse_sweep_values(
            args.replay_output_center_deadband_ratio,
            argument_name="--replay-output-center-deadband-ratio",
        ),
        "size_clamp_ratio": parse_sweep_values(
            args.replay_output_size_clamp_ratio,
            argument_name="--replay-output-size-clamp-ratio",
        ),
        "size_deadband_ratio": parse_sweep_values(
            args.replay_output_size_deadband_ratio,
            argument_name="--replay-output-size-deadband-ratio",
        ),
        "confidence_field": parse_confidence_field_values(
            args.replay_output_confidence_field
        ),
        "confidence_floor": parse_sweep_values(
            args.replay_output_confidence_floor,
            argument_name="--replay-output-confidence-floor",
        ),
        "confidence_ceiling": parse_sweep_values(
            args.replay_output_confidence_ceiling,
            argument_name="--replay-output-confidence-ceiling",
        ),
    }
    keys = tuple(value_lists)
    configs: list[ReplayOutputProjectionConfig] = []
    for values in _product(*(value_lists[key] for key in keys)):
        raw = dict(zip(keys, values, strict=True))
        if has_incomplete_confidence_config(raw):
            continue
        config = ReplayOutputProjectionConfig(**raw)
        try:
            validate_output_projection_config(config)
        except ValueError:
            continue
        configs.append(config)
    return configs


def parse_mode_values(raw_values: list[str] | tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for raw_value in raw_values:
        for token in split_tokens(raw_value):
            if token not in REPLAY_OUTPUT_MODES:
                expected = ", ".join(REPLAY_OUTPUT_MODES)
                raise ValueError(
                    f"Invalid value for --replay-output-mode: {token!r}; "
                    f"expected one of {expected}"
                )
            values.append(token)
    return unique_preserve_order(values)


def parse_confidence_field_values(
    raw_values: list[str] | tuple[str, ...],
) -> list[str | None]:
    values: list[str | None] = []
    for raw_value in raw_values:
        for token in split_tokens(raw_value):
            normalized = token.lower()
            if normalized in NONE_SWEEP_TOKENS:
                values.append(None)
            elif token in PROJECTION_CONFIDENCE_FIELDS:
                values.append(token)
            else:
                expected = ", ".join(PROJECTION_CONFIDENCE_FIELDS)
                raise ValueError(
                    f"Invalid value for --replay-output-confidence-field: {token!r}; "
                    f"expected one of {expected} or none"
                )
    return unique_preserve_order(values)


def parse_sweep_values(
    raw_values: list[str] | tuple[str, ...],
    *,
    argument_name: str,
) -> list[float | None]:
    values: list[float | None] = []
    for raw_value in raw_values:
        for token in split_tokens(raw_value):
            if token.lower() in NONE_SWEEP_TOKENS:
                values.append(None)
                continue
            try:
                values.append(float(token))
            except ValueError as error:
                raise ValueError(
                    f"Invalid value for {argument_name}: {token!r}"
                ) from error
    return unique_preserve_order(values)


ACCEPTANCE_GRID_SPECS = (
    ("min_used_event_count", "--min-accept-used-events", int),
    ("min_active_measurement_count", "--min-accept-active-measurements", int),
    ("min_mean_event_activity", "--min-accept-mean-activity", float),
    ("min_candidate_iou", "--min-accept-candidate-iou", float),
    ("min_candidate_area_ratio", "--min-accept-area-ratio", float),
    ("max_candidate_area_ratio", "--max-accept-area-ratio", float),
    ("max_center_shift_ratio", "--max-accept-center-shift-ratio", float),
    ("min_raw_candidate_iou", "--min-raw-candidate-iou", float),
    ("min_raw_candidate_area_ratio", "--min-raw-area-ratio", float),
    ("max_raw_candidate_area_ratio", "--max-raw-area-ratio", float),
    ("max_raw_center_shift_ratio", "--max-raw-center-shift-ratio", float),
    ("min_polarity_consistency_fraction", "--min-polarity-consistency-fraction", float),
    ("min_mean_event_polarity_weight", "--min-mean-event-polarity-weight", float),
    (
        "max_quadratic_form_per_active_measurement",
        "--max-quadratic-form-per-active-measurement",
        float,
    ),
    ("min_active_fraction", "--min-active-fraction", float),
)


def iter_acceptance_override_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    value_lists = {
        key: parse_acceptance_sweep_values(
            getattr(args, key),
            argument_name=argument_name,
            cast=cast,
        )
        for key, argument_name, cast in ACCEPTANCE_GRID_SPECS
    }
    keys = tuple(value_lists)
    configs: list[dict[str, Any]] = []
    acceptance_fields = set(asdict(ReplayAcceptanceConfig()))
    for values in _product(*(value_lists[key] for key in keys)):
        raw = dict(zip(keys, values, strict=True))
        overrides = {
            key: value
            for key, value in raw.items()
            if value is not DIAGNOSTIC_VALUE
        }
        unknown_keys = sorted(set(overrides) - acceptance_fields)
        if unknown_keys:
            raise ValueError(f"Unknown acceptance-policy fields: {unknown_keys}")
        configs.append(overrides)
    return configs


def parse_acceptance_sweep_values(
    raw_values: list[str] | tuple[str, ...],
    *,
    argument_name: str,
    cast: type,
) -> list[Any]:
    values: list[Any] = []
    for raw_value in raw_values:
        for token in split_tokens(raw_value):
            normalized = token.lower()
            if normalized in DIAGNOSTIC_SWEEP_TOKENS:
                values.append(DIAGNOSTIC_VALUE)
                continue
            if normalized in NONE_SWEEP_TOKENS:
                values.append(None)
                continue
            try:
                values.append(cast(token))
            except ValueError as error:
                raise ValueError(
                    f"Invalid value for {argument_name}: {token!r}"
                ) from error
    return unique_preserve_order(values)


def split_tokens(raw_value: Any) -> list[str]:
    return [
        token
        for token in re.split(r"[\s,]+", str(raw_value).strip())
        if token
    ]


def unique_preserve_order(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    if not unique:
        raise ValueError("sweep argument must contain at least one value")
    return unique


def has_incomplete_confidence_config(config: dict[str, Any]) -> bool:
    values = (
        config["confidence_field"],
        config["confidence_floor"],
        config["confidence_ceiling"],
    )
    return any(value is None for value in values) and any(
        value is not None for value in values
    )


def make_projection_config_id(
    index: int,
    config: ReplayOutputProjectionConfig,
) -> str:
    return make_sweep_config_id(
        index,
        ReplaySweepConfig(
            output_projection=config,
            acceptance_overrides={},
        ),
    )


def make_sweep_config_id(
    index: int,
    config: ReplaySweepConfig,
) -> str:
    payload = json.dumps(
        sweep_config_to_dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"replay{index:04d}_h{digest}_pm{tag_text(config.output_projection.mode)}"


def make_result_row(
    config_id: str,
    config: ReplaySweepConfig,
    result_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    summary = payload["summary"]
    metrics = summary.get("metrics") or {}
    acceptance_config = payload.get("acceptance_config") or {}
    return {
        "config_id": config_id,
        "output_results": str(result_dir),
        "acceptance_overrides": json.dumps(
            config.acceptance_overrides,
            sort_keys=True,
            separators=(",", ":"),
        ),
        **asdict(config.output_projection),
        **{
            f"acceptance_{key}": value
            for key, value in acceptance_config.items()
        },
        "sequence_count": summary["sequence_count"],
        "frame_count": summary["frame_count"],
        "accepted_refinement_count": summary["accepted_refinement_count"],
        "sr_auc": metrics.get("sr_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "npr_auc": metrics.get("npr_auc"),
        "mean_iou": metrics.get("mean_iou"),
    }


def write_sweep_outputs(
    rows: list[dict[str, Any]],
    configs: list[ReplaySweepConfig],
    args: argparse.Namespace,
) -> dict[str, Any]:
    ranked = rank_rows(rows)
    metrics_csv = args.metrics_csv or args.output_root / "projection_sweep_metrics.csv"
    summary_json = args.summary_json or args.output_root / "projection_sweep_summary.json"
    if rows:
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(ranked)

    payload = {
        "schema_version": 1,
        "description": (
            "Replay policy sweep from existing EventVOT DVS-ENACT diagnostics."
        ),
        "summary": {
            "config_count": len(configs),
            "completed_config_count": len(rows),
            "best_config_id": ranked[0]["config_id"] if ranked else None,
            "best_sr_auc": ranked[0].get("sr_auc") if ranked else None,
            "metrics_csv": str(metrics_csv),
        },
        "top_configs": ranked[: args.top_k],
        "grid": [sweep_config_to_dict(config) for config in configs],
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            _descending_metric(row.get("sr_auc")),
            _descending_metric(row.get("pr_auc")),
            _descending_metric(row.get("npr_auc")),
            -int(row.get("accepted_refinement_count") or 0),
        ),
    )


def _descending_metric(value: Any) -> float:
    if value is None:
        return float("inf")
    return -float(value)


def tag_text(value: Any) -> str:
    return str(value).replace("-", "").replace("_", "")


def sweep_config_to_dict(config: ReplaySweepConfig) -> dict[str, Any]:
    return {
        "output_projection": asdict(config.output_projection),
        "acceptance_overrides": config.acceptance_overrides,
    }


def _product(*value_lists: list[Any]):
    if not value_lists:
        yield ()
        return
    first, *rest = value_lists
    for value in first:
        for suffix in _product(*rest):
            yield (value, *suffix)


if __name__ == "__main__":
    raise SystemExit(main())
