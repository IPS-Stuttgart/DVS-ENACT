"""Run the complete DVS-ENACT paper artifact export pipeline.

This script is intentionally a thin orchestrator around the existing per-stage
experiment scripts. It standardizes one paper-facing command and writes a
manifest that records which artifacts were generated, skipped, or failed.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import run_mevdt_support_diagnostics
import run_mevdt_tracker_comparison
import run_synthetic_count_likelihood
import run_synthetic_cube_activity
import run_synthetic_tracker_comparison


StageFunction = Callable[..., dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _json_safe(value: Any) -> Any:
    """Convert common scientific/Python objects into JSON-safe values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    # Numpy scalars and arrays expose item/tolist without importing numpy here.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def _run_stage(
    name: str,
    function: StageFunction,
    *args: Any,
    enabled: bool = True,
    skip_reason: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    started_at = _utc_now()
    if not enabled:
        return {
            "name": name,
            "status": "skipped",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "reason": skip_reason or "stage disabled",
        }

    try:
        result = function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - manifest should capture all stage failures.
        return {
            "name": name,
            "status": "failed",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }

    return {
        "name": name,
        "status": "succeeded",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "result": _json_safe(result),
    }


def _write_manifest(output_root: Path, manifest: dict[str, Any], manifest_name: str) -> Path:
    evidence_dir = output_root / "data" / "paper_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = evidence_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    dataset_root = args.dataset_root.resolve() if args.dataset_root is not None else None
    include_mevdt = dataset_root is not None and not args.skip_mevdt
    repo_root = _repo_root()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "repository": {
            "name": "DVS-ENACT",
            "root": str(repo_root),
            "head": _git_head(repo_root),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "parameters": {
            "output_root": str(output_root),
            "dataset_root": str(dataset_root) if dataset_root is not None else None,
            "event_csv": args.event_csv,
            "label_file": args.label_file,
            "skip_mevdt": args.skip_mevdt,
            "max_windows": args.max_windows,
            "max_events": args.max_events,
            "band_fraction": args.band_fraction,
            "max_events_per_window": args.max_events_per_window,
            "min_events_per_window": args.min_events_per_window,
            "synthetic_n_steps": args.synthetic_n_steps,
            "synthetic_events_per_window": args.synthetic_events_per_window,
            "synthetic_total_events": args.synthetic_total_events,
            "seed": args.seed,
        },
        "stages": [],
    }

    stage_calls: list[tuple[str, StageFunction, tuple[Any, ...], dict[str, Any], bool, str | None]] = [
        (
            "synthetic_cube_activity",
            run_synthetic_cube_activity.run,
            (output_root,),
            {},
            True,
            None,
        ),
        (
            "synthetic_count_likelihood",
            run_synthetic_count_likelihood.run,
            (output_root, args.synthetic_total_events, args.seed),
            {},
            True,
            None,
        ),
        (
            "synthetic_tracker_comparison",
            run_synthetic_tracker_comparison.run,
            (),
            {
                "output_root": output_root,
                "n_steps": args.synthetic_n_steps,
                "events_per_window": args.synthetic_events_per_window,
                "seed": args.seed,
                "max_events_per_window": args.max_events_per_window,
            },
            True,
            None,
        ),
        (
            "mevdt_support_diagnostics",
            run_mevdt_support_diagnostics.run,
            (),
            {
                "dataset_root": dataset_root,
                "output_root": output_root,
                "event_csv": args.event_csv,
                "label_file": args.label_file,
                "max_events": args.max_events,
                "band_fraction": args.band_fraction,
                "max_windows": args.max_windows,
            },
            include_mevdt,
            "No --dataset-root supplied, or --skip-mevdt was set.",
        ),
        (
            "mevdt_tracker_comparison",
            run_mevdt_tracker_comparison.run,
            (),
            {
                "dataset_root": dataset_root,
                "output_root": output_root,
                "event_csv": args.event_csv,
                "label_file": args.label_file,
                "max_events_per_window": args.max_events_per_window,
                "max_windows": args.max_windows,
                "min_events_per_window": args.min_events_per_window,
            },
            include_mevdt,
            "No --dataset-root supplied, or --skip-mevdt was set.",
        ),
    ]

    for name, function, positional, keyword, enabled, skip_reason in stage_calls:
        record = _run_stage(
            name,
            function,
            *positional,
            enabled=enabled,
            skip_reason=skip_reason,
            **keyword,
        )
        manifest["stages"].append(record)
        print(f"[{record['status']}] {name}", file=sys.stderr)
        if record["status"] == "failed" and not args.keep_going:
            break

    manifest["finished_at"] = _utc_now()
    manifest["summary"] = {
        "succeeded": sum(stage["status"] == "succeeded" for stage in manifest["stages"]),
        "skipped": sum(stage["status"] == "skipped" for stage in manifest["stages"]),
        "failed": sum(stage["status"] == "failed" for stage in manifest["stages"]),
    }
    manifest_path = _write_manifest(output_root, manifest, args.manifest_name)
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate canonical DVS-ENACT artifacts for the paper repository.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../2026-05-DVS-ENACT-Paper"),
        help="Directory where data/paper_evidence and figures are written.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional MEVDT extraction root. If omitted, MEVDT stages are skipped.",
    )
    parser.add_argument("--event-csv", help="Optional explicit MEVDT event CSV path.")
    parser.add_argument("--label-file", help="Optional explicit MEVDT label file path.")
    parser.add_argument(
        "--skip-mevdt",
        action="store_true",
        help="Skip MEVDT stages even if --dataset-root is supplied.",
    )
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--band-fraction", type=float, default=0.15)
    parser.add_argument("--max-events-per-window", type=int, default=64)
    parser.add_argument("--min-events-per-window", type=int, default=3)
    parser.add_argument("--synthetic-n-steps", type=int, default=40)
    parser.add_argument("--synthetic-events-per-window", type=int, default=64)
    parser.add_argument("--synthetic-total-events", type=int, default=240)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--manifest-name",
        default="artifact_manifest.json",
        help="Manifest file name below data/paper_evidence/.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failed stage, but still exit non-zero if any stage failed.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = run_pipeline(args)
    print(json.dumps(manifest, indent=2))
    return 1 if manifest["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
