"""Run EventVOT DVS-ENACT refinement with alternative output projection modes.

This script is a small compatibility wrapper around ``run_eventvot_refinement``.
It intentionally avoids duplicating EventVOT parsing/evaluation code.  The main
use cases are conservative projection modes for strong trackers such as
HDETrackV2: ``center-only`` lets DVS-ENACT correct the box center while retaining
the external tracker's size, and ``size-only`` lets DVS-ENACT correct the box
size while retaining the external tracker's center.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Any

import numpy as np

from dvs_enact import DVSContourRefiner, DVSRefinementResult

import run_eventvot_refinement as base

REFINEMENT_MODES = ("box", "center-only", "size-only")


class ProjectedOutputRefiner:
    """Wrap a DVSContourRefiner and project its output before acceptance.

    ``run_eventvot_refinement`` accepts or rejects the ``as_xywh()`` output of
    the supplied refiner.  By replacing the output box in the refinement result,
    this wrapper makes the existing guarded acceptance logic evaluate the actual
    projected output rather than the raw full-box contour update.
    """

    def __init__(
        self,
        refiner: DVSContourRefiner,
        *,
        refinement_mode: str,
        projection_width_blend: float | None = None,
        projection_height_blend: float | None = None,
        projection_no_clip: bool = False,
        projection_min_raw_width_ratio: float | None = None,
        projection_max_raw_width_ratio: float | None = None,
        projection_min_raw_height_ratio: float | None = None,
        projection_max_raw_height_ratio: float | None = None,
    ):
        validate_refinement_mode(refinement_mode)
        validate_projection_blends(projection_width_blend, projection_height_blend)
        validate_projection_ratio_gates(
            projection_min_raw_width_ratio,
            projection_max_raw_width_ratio,
            projection_min_raw_height_ratio,
            projection_max_raw_height_ratio,
        )
        self.refiner = refiner
        self.config = refiner.config
        self.refinement_mode = refinement_mode
        self.projection_width_blend = projection_width_blend
        self.projection_height_blend = projection_height_blend
        self.projection_no_clip = projection_no_clip
        self.projection_min_raw_width_ratio = projection_min_raw_width_ratio
        self.projection_max_raw_width_ratio = projection_max_raw_width_ratio
        self.projection_min_raw_height_ratio = projection_min_raw_height_ratio
        self.projection_max_raw_height_ratio = projection_max_raw_height_ratio

    def refine(self, candidate_bbox: Any, events: Any, **kwargs: Any) -> DVSRefinementResult:
        result = self.refiner.refine(candidate_bbox, events, **kwargs)
        if self.refinement_mode == "box" or result.fallback_reason is not None:
            return result

        candidate_xywh = bbox_dict_to_xywh(result.candidate_bbox)
        refiner_xywh = np.asarray(result.as_xywh(), dtype=float)
        raw_refined_xywh = bbox_dict_to_xywh(result.refined_bbox)
        unclipped_projected_xywh = project_refinement_output(
            candidate_xywh,
            refiner_xywh,
            refinement_mode=self.refinement_mode,
            raw_refined_xywh=raw_refined_xywh,
            projection_width_blend=self.projection_width_blend,
            projection_height_blend=self.projection_height_blend,
        )
        projected_xywh = clip_xywh_box(
            unclipped_projected_xywh,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
        )
        projection_rejections = projection_rejection_reasons(
            candidate_xywh,
            raw_refined_xywh,
            unclipped_projected_xywh,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
            projection_no_clip=self.projection_no_clip,
            projection_min_raw_width_ratio=self.projection_min_raw_width_ratio,
            projection_max_raw_width_ratio=self.projection_max_raw_width_ratio,
            projection_min_raw_height_ratio=self.projection_min_raw_height_ratio,
            projection_max_raw_height_ratio=self.projection_max_raw_height_ratio,
        )
        fallback_reason = (
            None
            if not projection_rejections
            else f"projection:{','.join(projection_rejections)}"
        )
        return replace(
            result,
            output_bbox=xywh_to_bbox_dict(
                projected_xywh,
                bbox_format=self.config.output_bbox_format,
            ),
            fallback_reason=fallback_reason,
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
            "and transfers only the DVS-ENACT center correction. 'size-only' "
            "keeps the base-track center and transfers only the DVS-ENACT "
            "width/height correction."
        ),
    )
    parser.add_argument(
        "--projection-width-blend",
        type=float,
        help=(
            "Optional width blend for size-only mode. When supplied together "
            "with --projection-height-blend, size-only projection blends from "
            "the raw DVS-ENACT refined width instead of the already blended "
            "--refinement-blend output."
        ),
    )
    parser.add_argument(
        "--projection-height-blend",
        type=float,
        help=(
            "Optional height blend for size-only mode. Must be supplied "
            "together with --projection-width-blend."
        ),
    )
    parser.add_argument(
        "--projection-no-clip",
        action="store_true",
        help="Reject projected refinement outputs that would be clipped by image bounds.",
    )
    parser.add_argument("--projection-min-raw-width-ratio", type=float)
    parser.add_argument("--projection-max-raw-width-ratio", type=float)
    parser.add_argument("--projection-min-raw-height-ratio", type=float)
    parser.add_argument("--projection-max-raw-height-ratio", type=float)
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
        projection_width_blend=args.projection_width_blend,
        projection_height_blend=args.projection_height_blend,
        projection_no_clip=args.projection_no_clip,
        projection_min_raw_width_ratio=args.projection_min_raw_width_ratio,
        projection_max_raw_width_ratio=args.projection_max_raw_width_ratio,
        projection_min_raw_height_ratio=args.projection_min_raw_height_ratio,
        projection_max_raw_height_ratio=args.projection_max_raw_height_ratio,
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
    raw_refined_xywh: np.ndarray | None = None,
    projection_width_blend: float | None = None,
    projection_height_blend: float | None = None,
    image_width: float | None = None,
    image_height: float | None = None,
) -> np.ndarray:
    """Return the candidate replacement box for the selected refinement mode."""
    validate_refinement_mode(refinement_mode)
    validate_projection_blends(projection_width_blend, projection_height_blend)
    candidate = np.asarray(candidate_xywh, dtype=float).reshape(4)
    refined = np.asarray(refiner_output_xywh, dtype=float).reshape(4)

    if refinement_mode == "box":
        output = np.array(refined, dtype=float, copy=True)
    elif refinement_mode == "center-only":
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
    else:
        candidate_center = candidate[:2] + 0.5 * candidate[2:]
        if projection_width_blend is not None and projection_height_blend is not None:
            if raw_refined_xywh is None:
                raise ValueError(
                    "raw_refined_xywh is required when projection blends are supplied"
                )
            raw_refined = np.asarray(raw_refined_xywh, dtype=float).reshape(4)
            width = (1.0 - projection_width_blend) * candidate[2]
            width += projection_width_blend * raw_refined[2]
            height = (1.0 - projection_height_blend) * candidate[3]
            height += projection_height_blend * raw_refined[3]
            projected_size = np.array([width, height], dtype=float)
        else:
            projected_size = refined[2:]
        output = np.array(
            [
                candidate_center[0] - 0.5 * projected_size[0],
                candidate_center[1] - 0.5 * projected_size[1],
                projected_size[0],
                projected_size[1],
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


def projection_rejection_reasons(
    candidate_xywh: np.ndarray,
    raw_refined_xywh: np.ndarray,
    projected_xywh: np.ndarray,
    *,
    image_width: float | None = None,
    image_height: float | None = None,
    projection_no_clip: bool = False,
    projection_min_raw_width_ratio: float | None = None,
    projection_max_raw_width_ratio: float | None = None,
    projection_min_raw_height_ratio: float | None = None,
    projection_max_raw_height_ratio: float | None = None,
) -> tuple[str, ...]:
    """Return projection-policy rejection reasons for a candidate output."""
    candidate = np.asarray(candidate_xywh, dtype=float).reshape(4)
    raw_refined = np.asarray(raw_refined_xywh, dtype=float).reshape(4)
    projected = np.asarray(projected_xywh, dtype=float).reshape(4)
    reasons: list[str] = []

    if projection_no_clip and image_width is not None and image_height is not None:
        if (
            projected[0] < 0.0
            or projected[1] < 0.0
            or projected[0] + projected[2] > float(image_width)
            or projected[1] + projected[3] > float(image_height)
        ):
            reasons.append("projection_clip")

    raw_width_ratio = raw_refined[2] / max(candidate[2], 1e-9)
    raw_height_ratio = raw_refined[3] / max(candidate[3], 1e-9)
    _append_projection_min_ratio(
        reasons,
        "projection_raw_width_ratio",
        raw_width_ratio,
        projection_min_raw_width_ratio,
    )
    _append_projection_max_ratio(
        reasons,
        "projection_raw_width_ratio",
        raw_width_ratio,
        projection_max_raw_width_ratio,
    )
    _append_projection_min_ratio(
        reasons,
        "projection_raw_height_ratio",
        raw_height_ratio,
        projection_min_raw_height_ratio,
    )
    _append_projection_max_ratio(
        reasons,
        "projection_raw_height_ratio",
        raw_height_ratio,
        projection_max_raw_height_ratio,
    )
    return tuple(reasons)


def _append_projection_min_ratio(
    reasons: list[str],
    name: str,
    value: float,
    minimum: float | None,
) -> None:
    if minimum is not None and value < minimum:
        reasons.append(name)


def _append_projection_max_ratio(
    reasons: list[str],
    name: str,
    value: float,
    maximum: float | None,
) -> None:
    if maximum is not None and value > maximum:
        reasons.append(name)


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


def validate_projection_blends(
    projection_width_blend: float | None,
    projection_height_blend: float | None,
) -> None:
    """Raise ValueError for unsupported independent size projection blends."""
    if (projection_width_blend is None) != (projection_height_blend is None):
        raise ValueError(
            "--projection-width-blend and --projection-height-blend must be supplied together"
        )
    for name, value in (
        ("projection_width_blend", projection_width_blend),
        ("projection_height_blend", projection_height_blend),
    ):
        if value is None:
            continue
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")


def validate_projection_ratio_gates(
    projection_min_raw_width_ratio: float | None,
    projection_max_raw_width_ratio: float | None,
    projection_min_raw_height_ratio: float | None,
    projection_max_raw_height_ratio: float | None,
) -> None:
    """Raise ValueError for invalid raw-size projection ratio gates."""
    pairs = (
        (
            "projection_min_raw_width_ratio",
            projection_min_raw_width_ratio,
            "projection_max_raw_width_ratio",
            projection_max_raw_width_ratio,
        ),
        (
            "projection_min_raw_height_ratio",
            projection_min_raw_height_ratio,
            "projection_max_raw_height_ratio",
            projection_max_raw_height_ratio,
        ),
    )
    for min_name, min_value, max_name, max_value in pairs:
        for name, value in ((min_name, min_value), (max_name, max_value)):
            if value is None:
                continue
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if min_value is not None and max_value is not None:
            if float(min_value) > float(max_value):
                raise ValueError(f"{min_name} must not exceed {max_name}")


if __name__ == "__main__":
    raise SystemExit(main())
