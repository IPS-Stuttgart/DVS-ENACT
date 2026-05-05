"""Run MEVDT side-support diagnostics for DVS-ENACT validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dvs_enact import (
    MEVDT_DATASET_URL,
    MEVDT_DOI,
    compute_bbox_event_diagnostics,
    find_event_csv_files,
    find_tracking_label_files,
    read_event_csv,
    read_tracking_labels,
    summarize_diagnostics,
    summarize_loaded_sequence,
)


def _resolve_path(files: list[Path], requested: str | None, kind: str) -> Path:
    if requested is not None:
        path = Path(requested)
        if not path.exists():
            raise FileNotFoundError(f"{kind} file does not exist: {path}")
        return path
    if not files:
        raise FileNotFoundError(f"No {kind} files found")
    return files[0]


def _path_for_payload(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.relative_to(dataset_root))
    except ValueError:
        return str(path)


def run(
    dataset_root: Path,
    output_root: Path,
    event_csv: str | None = None,
    label_file: str | None = None,
    max_events: int | None = None,
    band_fraction: float = 0.15,
    max_windows: int | None = 500,
) -> dict:
    event_files = find_event_csv_files(dataset_root)
    label_files = find_tracking_label_files(dataset_root)
    selected_event_file = _resolve_path(event_files, event_csv, "event CSV")
    selected_label_file = _resolve_path(label_files, label_file, "tracking label")

    labels = read_tracking_labels(selected_label_file)
    if not labels:
        raise ValueError(f"No tracking labels parsed from {selected_label_file}")
    events = read_event_csv(selected_event_file, max_events=max_events)
    parsed_sequence = summarize_loaded_sequence(labels, events)
    diagnostics = compute_bbox_event_diagnostics(
        labels,
        events,
        band_fraction=band_fraction,
    )
    if max_windows is not None:
        diagnostics = diagnostics[:max_windows]
    summary = summarize_diagnostics(diagnostics)
    payload = {
        "dataset": {
            "name": "MEVDT",
            "url": MEVDT_DATASET_URL,
            "doi": MEVDT_DOI,
            "dataset_root": str(dataset_root),
            "event_csv": _path_for_payload(selected_event_file, dataset_root),
            "label_file": _path_for_payload(selected_label_file, dataset_root),
        },
        "parameters": {
            "max_events": max_events,
            "band_fraction": band_fraction,
            "max_windows": max_windows,
        },
        "parsed_sequence": parsed_sequence,
        "summary": summary,
        "diagnostics": [item.to_dict() for item in diagnostics],
    }

    data_dir = output_root / "data" / "paper_evidence"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "mevdt_support_diagnostics.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output_path), "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../2026-05-DVS-ENACT-Paper"),
    )
    parser.add_argument("--event-csv")
    parser.add_argument("--label-file")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--band-fraction", type=float, default=0.15)
    parser.add_argument("--max-windows", type=int, default=500)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                event_csv=args.event_csv,
                label_file=args.label_file,
                max_events=args.max_events,
                band_fraction=args.band_fraction,
                max_windows=args.max_windows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
