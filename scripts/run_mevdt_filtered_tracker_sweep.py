"""Run filtered MEVDT tracker comparison and a compact parameter sweep."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from dvs_enact import (
    TrackerComparisonConfig,
    WindowFilterConfig,
    compare_trackers_on_labels,
    read_event_csv,
    read_tracking_labels,
    select_mevdt_event_and_label_files,
)
from dvs_enact.mevdt import MEVDT_DATASET_URL, MEVDT_DOI


def _path_for_payload(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.relative_to(dataset_root))
    except ValueError:
        return str(path)


def _sweep_configs(base_config: TrackerComparisonConfig) -> list[tuple[str, TrackerComparisonConfig]]:
    return [
        ("default_filtered", base_config),
        ("low_shape_variance", replace(base_config, shape_variance=5.0)),
        (
            "high_measurement_noise",
            replace(base_config, measurement_noise_variance=16.0),
        ),
        ("high_radial_noise", replace(base_config, radial_noise_variance=9.0)),
        (
            "low_activity_gate",
            replace(
                base_config,
                event_activity_floor=0.01,
                inactive_activity_threshold=0.0,
            ),
        ),
        (
            "aggressive_activity_gate",
            replace(
                base_config,
                event_activity_floor=0.05,
                inactive_activity_threshold=0.25,
            ),
        ),
    ]


def _compact_result(name: str, result: dict) -> dict:
    return {
        "name": name,
        "tracker_parameters": result["tracker_parameters"],
        "summary": result["summary"],
    }


def run(
    dataset_root: Path,
    output_root: Path,
    event_csv: str | None = None,
    label_file: str | None = None,
    max_events_per_window: int = 64,
    max_windows: int | None = 500,
    min_events_per_window: int = 3,
) -> dict:
    base_config = TrackerComparisonConfig(
        max_events_per_window=max_events_per_window,
        max_windows=max_windows,
        min_events_per_window=min_events_per_window,
    )
    window_filter = WindowFilterConfig()
    selected_event, selected_label = select_mevdt_event_and_label_files(
        dataset_root,
        event_csv=event_csv,
        label_file=label_file,
    )
    labels = read_tracking_labels(selected_label)
    if not labels:
        raise ValueError(f"No tracking labels parsed from {selected_label}")
    events = read_event_csv(selected_event)

    sweep_results = []
    default_result = None
    for name, config in _sweep_configs(base_config):
        result = compare_trackers_on_labels(
            labels,
            events,
            config=config,
            window_filter=window_filter,
        )
        if default_result is None:
            default_result = result
        sweep_results.append(_compact_result(name, result))

    payload = {
        "dataset": {
            "name": "MEVDT",
            "url": MEVDT_DATASET_URL,
            "doi": MEVDT_DOI,
            "dataset_root": str(dataset_root),
            "event_csv": _path_for_payload(selected_event, dataset_root),
            "label_file": _path_for_payload(selected_label, dataset_root),
            "association": "label-assisted",
        },
        "window_filter": default_result["window_filter"],
        "parsed_sequence": default_result["parsed_sequence"],
        "filtered_default_comparison": default_result,
        "sweep": sweep_results,
    }

    data_dir = output_root / "data" / "paper_evidence"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "mevdt_filtered_tracker_sweep.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "output": str(output_path),
        "filtered_default_summary": default_result["summary"],
        "sweep": sweep_results,
    }


def main():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                event_csv=args.event_csv,
                label_file=args.label_file,
                max_events_per_window=args.max_events_per_window,
                max_windows=args.max_windows,
                min_events_per_window=args.min_events_per_window,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
