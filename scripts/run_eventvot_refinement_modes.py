"""Run EventVOT DVS-ENACT refinement with alternative output projection modes.

This script is a small compatibility wrapper around ``run_eventvot_refinement``.
It intentionally avoids duplicating EventVOT parsing/evaluation code.  The main
use case is a conservative ``center-only`` mode: DVS-ENACT may correct the box
center, while the external tracker's width and height are retained.  This is a
low-risk refinement mode for strong trackers such as HDETrackV2, where noisy
scale changes can hurt SR even when the event contour update is locally useful.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Any

import numpy as np

from dvs_enact import DVSContourRefiner, DVSRefinementResult

import run_eventvot_refinement as base

REFINEMENT_MODES = ("box", "center-only")


class ProjectedOutputRefiner:
    """Wrap a DVSContourRefiner and project its output before acceptance.

    ``run_eventvot_refinement`` accepts or rejects the ``as_xywh()`` output of
    the supplied refiner.  By replacing the output box in the refinement result,
    this wrapper makes the existing guarded acceptance logic evaluate the actual
    projected output rather than the raw full-box contour update.
    """

    def __init__(self, refiner: DVSContourRefiner, *, refinement_mode: str):
        validate_refinement_mode(refinement_mode)
        self.refiner = refiner
        self.config = refiner.config
        self.refinement_mode = refinement_mode

    def refine(self, candidate_bbox: Any, events: Any, **kwargs: Any) -> DVSRefinementResult:
        result = self.refiner.refine(candidate_bbox, events, **kwargs)
        if self.refinement_mode == "box" or result.fallback_reason is not None:
            return result

        candidate_xywh = bbox_dict_to_xywh(result.candidate_bbox)
        refiner_xywh = np.asarray(result.as_xywh(), dtype=float)
        projected_xywh = project_refinement_output(
            candidate_xywh,
            refiner_xywh,
            refinement_mode=self.refinement_mode,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
        )
        return replace(
            result,
            output_bbox=xywh_to_bbox_dict(
                projected_xywh,
                bbox_format=self.config.output_bbox_format,
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    """Return the base EventVOT parser extended with refinement modes."""
    parser = base.build_parser()
    parser.description = (
        "Refine EventVOT xywh tracker result files with DVS-ENACT and an "
        "optional conservative output projection mode."
    )
    parser.add_argument(
        "--refinement-mode",
        choices=REFINEMENT_MODES,
        default="box",
        help=(
            "Output projection mode. 'box' preserves the current full-box "
            "DVS-ENACT update. 'center-only' keeps the base-track width/height "
            "and transfers only the DVS-ENACT center correction."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Run projected-output EventVOT refinement for parsed CLI arguments."""
    validate_refinement_mode(args.refinement_mode)
    refiner = ProjectedOutputRefiner(
        base._refiner_from_args(args),
        refinement_mode=args.refinement_mode,
    )
    sequence_names = base.load_requested_sequence_names(
        args.sequence,
        args.sequence_list,
        args.sequence_file,
    )
    return base.run(
        base.EventVOTRefinementOptions(
            eventvot_root=args.eventvot_root,
            base_results=args.base_results,
            output_results=base._resolve_cli_output_results(args),
            split=args.split,
            sequences=sequence_names,
            sequence_index=args.sequence_index,
            sequence_count=args.sequence_count,
            tracker_name=args.tracker_name,
            skip_existing=not args.no_skip_existing,
            event_column_order=args.event_column_order,
            diagnostics_json=args.diagnostics_json,
            config_tracker_path=base._resolve_cli_config_tracker_path(args),
            acceptance_config=base._acceptance_config_from_args(args),
        ),
        refiner=refiner,
    )


def project_refinement_output(
    candidate_xywh: np.ndarray,
    refiner_output_xywh: np.ndarray,
    *,
    refinement_mode: str,
    image_width: float | None = None,
    image_height: float | None = None,
) -> np.ndarray:
    """Return the candidate replacement box for the selected refinement mode."""
    validate_refinement_mode(refinement_mode)
    candidate = np.asarray(candidate_xywh, dtype=float).reshape(4)
    refined = np.asarray(refiner_output_xywh, dtype=float).reshape(4)

    if refinement_mode == "box":
        output = np.array(refined, dtype=float, copy=True)
    else:
        refined_center = refined[:2] + 0.5 * refined[2:]
        output = np.array(
            [
                refined_center[0] - 0.5 * candidate[2],
                refined_center[1] - 0.5 * candidate[3],
                candidate[2],
                candidate[3],
            ],
            dtype=float,
        )

    return clip_xywh_box(output, image_width=image_width, image_height=image_height)


def clip_xywh_box(
    box_xywh: np.ndarray,
    *,
    image_width: float | None = None,
    image_height: float | None = None,
) -> np.ndarray:
    """Clip an xywh box to image bounds while preserving valid extents."""
    output = np.asarray(box_xywh, dtype=float).reshape(4).copy()
    output[2:] = np.maximum(output[2:], 0.0)
    x_min, y_min = output[:2]
    x_max, y_max = output[:2] + output[2:]

    if image_width is not None:
        width = float(image_width)
        x_min = min(max(x_min, 0.0), width)
        x_max = min(max(x_max, 0.0), width)
    if image_height is not None:
        height = float(image_height)
        y_min = min(max(y_min, 0.0), height)
        y_max = min(max(y_max, 0.0), height)

    return np.array(
        [
            x_min,
            y_min,
            max(0.0, x_max - x_min),
            max(0.0, y_max - y_min),
        ],
        dtype=float,
    )


def bbox_dict_to_xywh(bbox: dict[str, float]) -> np.ndarray:
    """Convert a DVSRefinementResult bbox dictionary to an xywh array."""
    return np.array(
        [bbox["x_min"], bbox["y_min"], bbox["width"], bbox["height"]],
        dtype=float,
    )


def xywh_to_bbox_dict(box_xywh: np.ndarray, *, bbox_format: str = "xywh") -> dict[str, float]:
    """Return an output_bbox dictionary compatible with DVSRefinementResult."""
    box = np.asarray(box_xywh, dtype=float).reshape(4)
    x_min, y_min, width, height = box.tolist()
    x_max = x_min + width
    y_max = y_min + height
    normalized = {
        "x_min": float(x_min),
        "y_min": float(y_min),
        "x_max": float(x_max),
        "y_max": float(y_max),
        "width": float(width),
        "height": float(height),
        "area": float(max(width, 0.0) * max(height, 0.0)),
        "center_x": float(x_min + 0.5 * width),
        "center_y": float(y_min + 0.5 * height),
    }
    if bbox_format == "xywh":
        return {**normalized, "x": normalized["x_min"], "y": normalized["y_min"]}
    if bbox_format == "xyxy":
        return normalized
    raise ValueError("bbox_format must be 'xywh' or 'xyxy'")


def validate_refinement_mode(refinement_mode: str) -> None:
    """Raise ValueError for unsupported refinement output modes."""
    if refinement_mode not in REFINEMENT_MODES:
        raise ValueError(
            f"Unsupported refinement mode {refinement_mode!r}; "
            f"expected one of {', '.join(REFINEMENT_MODES)}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
