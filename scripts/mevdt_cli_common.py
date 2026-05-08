"""Shared helpers for MEVDT artifact-export command-line scripts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from dvs_enact import TrackerComparisonConfig


def add_mevdt_comparison_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\Uni-Data\MEVDT-one-sequence"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../2026-05-DVS-ENACT-Paper"),
    )
    parser.add_argument("--event-csv")
    parser.add_argument("--label-file")
    parser.add_argument("--max-events-per-window", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--min-events-per-window", type=int, default=3)
    parser.add_argument(
        "--disable-event-polarity",
        action="store_true",
        help="Ignore event polarity and use unsigned normal-flow activity only.",
    )
    parser.add_argument("--polarity-mismatch-weight", type=float, default=0.25)
    parser.add_argument(
        "--polarity-contrast-sign",
        choices=("infer", "positive", "negative", "none"),
        default="infer",
    )


def tracker_config_from_options(options: Mapping[str, object]) -> TrackerComparisonConfig:
    return TrackerComparisonConfig(
        max_events_per_window=int(options.get("max_events_per_window", 64)),
        max_windows=_optional_int(options.get("max_windows", 500)),
        min_events_per_window=int(options.get("min_events_per_window", 3)),
        use_event_polarity=not bool(options.get("disable_event_polarity", False)),
        polarity_mismatch_weight=float(options.get("polarity_mismatch_weight", 0.25)),
        polarity_contrast_sign=_polarity_contrast_sign(
            options.get("polarity_contrast_sign", "infer")
        ),
    )


def _polarity_contrast_sign(value: object) -> float | str | None:
    value = str(value).lower()
    if value == "none":
        return None
    if value == "positive":
        return 1.0
    if value == "negative":
        return -1.0
    return "infer"


def option_string(options: Mapping[str, object], name: str) -> str | None:
    value = options.get(name)
    return str(value) if value is not None else None


def write_evidence_json(output_root: Path, file_name: str, payload: dict) -> Path:
    data_dir = output_root / "data" / "paper_evidence"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / file_name
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def parse_and_print(run_function: Callable[..., dict[str, Any]]) -> None:
    parser = argparse.ArgumentParser()
    add_mevdt_comparison_arguments(parser)
    result = run_function(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2))


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)
