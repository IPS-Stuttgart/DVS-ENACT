"""Run label-assisted MEVDT baseline-vs-DVS-ENACT tracker comparison."""

from __future__ import annotations

from pathlib import Path

from dvs_enact import compare_mevdt_tracker_sequence
from mevdt_cli_common import (
    option_string,
    parse_and_print,
    tracker_config_from_options,
    write_evidence_json,
)


def run(
    dataset_root: Path,
    output_root: Path,
    **options: object,
) -> dict:
    payload = compare_mevdt_tracker_sequence(
        dataset_root,
        event_csv=option_string(options, "event_csv"),
        label_file=option_string(options, "label_file"),
        config=tracker_config_from_options(options),
    )

    output_path = write_evidence_json(
        output_root,
        "mevdt_tracker_comparison.json",
        payload,
    )
    return {"output": str(output_path), "summary": payload["summary"]}


def main():
    parse_and_print(run)


if __name__ == "__main__":
    main()
