"""Run label-assisted MEVDT baseline-vs-DVS-ENACT tracker comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dvs_enact import TrackerComparisonConfig, compare_mevdt_tracker_sequence


def run(
    dataset_root: Path,
    output_root: Path,
    event_csv: str | None = None,
    label_file: str | None = None,
    max_events_per_window: int = 64,
    max_windows: int | None = 500,
    min_events_per_window: int = 3,
) -> dict:
    config = TrackerComparisonConfig(
        max_events_per_window=max_events_per_window,
        max_windows=max_windows,
        min_events_per_window=min_events_per_window,
    )
    payload = compare_mevdt_tracker_sequence(
        dataset_root,
        event_csv=event_csv,
        label_file=label_file,
        config=config,
    )

    data_dir = output_root / "data" / "paper_evidence"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "mevdt_tracker_comparison.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output_path), "summary": payload["summary"]}


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
