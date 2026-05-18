"""Run EventVOT DVS-ENACT refinement with alternative output projection modes.

This script is a small compatibility wrapper around ``run_eventvot_refinement``.
It intentionally avoids duplicating EventVOT parsing/evaluation code.  The main
use cases are conservative projection modes for strong trackers such as
HDETrackV2: ``center-only`` lets DVS-ENACT correct the box center while retaining
the external tracker's size, ``size-only`` lets DVS-ENACT correct the box size
while retaining the external tracker's center, and ``width-only``/``height-only``
let validation sweeps keep just the useful size axis.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Any

import numpy as np

from dvs_enact import DVSContourRefiner, DVSRefinementResult

import run_eventvot_refinement as base

REFINEMENT_MODES = ("box", "center-only", "size-only", "width-only", "height-only")
PROJECTION_CONFIDENCE_FIELDS = (
    "mean_event_activity",
    "active_fraction",
    "polarity_consistency_fraction",
    "mean_event_polarity_weight",
)


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
        projection_size_smoothing: float | None = None,
        projection_confidence_field: str | None = None,
        projection_confidence_floor: float | None = None,
        projection_confidence_ceiling: float | None = None,
        projection_min_raw_width_ratio: float | None = None,
        projection_max_raw_width_ratio: float | None = None,
        projection_min_raw_height_ratio: float | None = None,
        projection_max_raw_height_ratio: float | None = None,
    ):
        validate_refinement_mode(refinement_mode)
        validate_projection_blends(projection_width_blend, projection_height_blend)
        validate_projection_size_smoothing(projection_size_smoothing)
        validate_projection_confidence_weighting(
            projection_confidence_field,
            projection_confidence_floor,
            projection_confidence_ceiling,
        )
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
        self.projection_size_smoothing = projection_size_smoothing
        self.projection_confidence_field = projection_confidence_field
        self.projection_confidence_floor = projection_confidence_floor
        self.projection_confidence_ceiling = projection_confidence_ceiling
        self.projection_min_raw_width_ratio = projection_min_raw_width_ratio
        self.projection_max_raw_width_ratio = projection_max_raw_width_ratio
        self.projection_min_raw_height_ratio = projection_min_raw_height_ratio
        self.projection_max_raw_height_ratio = projection_max_raw_height_ratio
        self._previous_accepted_projected_size: np.ndarray | None = None

    def reset_state(self) -> None:
        """Reset temporal projection state at sequence boundaries."""
        self._previous_accepted_projected_size = None

    def refine(self, candidate_bbox: Any, events: Any, **kwargs: Any) -> DVSRefinementResult:
        result = self.refiner.refine(candidate_bbox, events, **kwargs)
        if (
            self.refinement_mode == "box"
            and self.projection_size_smoothing is None
            and self.projection_confidence_field is None
        ) or result.fallback_reason is not None:
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
            previous_projected_size=self._previous_accepted_projected_size,
            projection_size_smoothing=self.projection_size_smoothing,
            projection_confidence_value=projection_confidence_value(
                result,
                self.projection_confidence_field,
            ),
            projection_confidence_floor=self.projection_confidence_floor,
            projection_confidence_ceiling=self.projection_confidence_ceiling,
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

    def observe_refinement_decision(
        self,
        _candidate_bbox: Any,
        result: DVSRefinementResult,
        accepted: bool,
    ) -> None:
        """Record accepted projected size for the next frame."""
        if (
            not accepted
            or self.projection_size_smoothing is None
            or self.refinement_mode == "center-only"
            or result.fallback_reason is not None
        ):
            return
        self._previous_accepted_projected_size = np.asarray(
            result.as_xywh(),
            dtype=float,
        )[2:].copy()


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
            "width/height correction. 'width-only' and 'height-only' transfer "
            "only one DVS-ENACT size axis."
        ),
    )
    parser.add_argument(
        "--projection-width-blend",
        type=float,
        help=(
            "Optional width blend for size projection modes. When supplied "
            "together with --projection-height-blend, projection blends from "
            "the raw DVS-ENACT refined width instead of the already blended "
            "--refinement-blend output."
        ),
    )
    parser.add_argument(
        "--projection-height-blend",
        type=float,
        help=(
            "Optional height blend for size projection modes. Must be supplied "
            "together with --projection-width-blend."
        ),
    )
    parser.add_argument(
        "--projection-no-clip",
        action="store_true",
        help="Reject projected refinement outputs that would be clipped by image bounds.",
    )
    parser.add_argument(
        "--projection-size-smoothing",
        type=float,
        help=(
            "Optional temporal size smoothing for projected outputs. The value "
            "is the weight of the previous accepted projected width/height; "
            "0 uses the current projection and 1 holds the previous size."
        ),
    )
    parser.add_argument(
        "--projection-confidence-field",
        choices=PROJECTION_CONFIDENCE_FIELDS,
        help=(
            "Optional diagnostic field used to shrink projected corrections "
            "toward the base tracker when DVS confidence is weak."
        ),
    )
    parser.add_argument(
        "--projection-confidence-floor",
        type=float,
        help="Confidence value that maps projected correction strength to zero.",
    )
    parser.add_argument(
        "--projection-confidence-ceiling",
        type=float,
        help="Confidence value that maps projected correction strength to one.",
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
        projection_size_smoothing=args.projection_size_smoothing,
        projection_confidence_field=args.projection_confidence_field,
        projection_confidence_floor=args.projection_confidence_floor,
        projection_confidence_ceiling=args.projection_confidence_ceiling,
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
    previous_projected_size: np.ndarray | None = None,
    projection_size_smoothing: float | None = None,
    projection_confidence_value: float | None = None,
    projection_confidence_floor: float | None = None,
    projection_confidence_ceiling: float | None = None,
    image_width: float | None = None,
    image_height: float | None = None,
) -> np.ndarray:
    """Return the candidate replacement box for the selected refinement mode."""
    validate_refinement_mode(refinement_mode)
    validate_projection_blends(projection_width_blend, projection_height_blend)
    validate_projection_size_smoothing(projection_size_smoothing)
    validate_projection_confidence_bounds(
        projection_confidence_floor,
        projection_confidence_ceiling,
    )
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
    elif refinement_mode in {"size-only", "width-only", "height-only"}:
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
        projected_size = project_size_axes(
            candidate[2:],
            projected_size,
            refinement_mode=refinement_mode,
        )
        output = np.array(
            [
                candidate_center[0] - 0.5 * projected_size[0],
                candidate_center[1] - 0.5 * projected_size[1],
                projected_size[0],
                projected_size[1],
            ],
            dtype=float,
        )
    else:
        raise AssertionError(f"Unhandled refinement mode: {refinement_mode}")

    output = smooth_projected_size(
        candidate,
        output,
        refinement_mode=refinement_mode,
        previous_projected_size=previous_projected_size,
        projection_size_smoothing=projection_size_smoothing,
    )
    output = apply_projection_confidence_weighting(
        candidate,
        output,
        confidence_value=projection_confidence_value,
        confidence_floor=projection_confidence_floor,
        confidence_ceiling=projection_confidence_ceiling,
    )
    return clip_xywh_box(output, image_width=image_width, image_height=image_height)


def apply_projection_confidence_weighting(
    candidate_xywh: np.ndarray,
    projected_xywh: np.ndarray,
    *,
    confidence_value: float | None,
    confidence_floor: float | None,
    confidence_ceiling: float | None,
) -> np.ndarray:
    """Shrink projected corrections toward the candidate box using confidence."""
    if confidence_floor is None and confidence_ceiling is None:
        return np.asarray(projected_xywh, dtype=float).reshape(4).copy()
    validate_projection_confidence_bounds(confidence_floor, confidence_ceiling)
    candidate = np.asarray(candidate_xywh, dtype=float).reshape(4)
    projected = np.asarray(projected_xywh, dtype=float).reshape(4)
    strength = projection_confidence_strength(
        confidence_value,
        confidence_floor,
        confidence_ceiling,
    )
    return candidate + strength * (projected - candidate)


def projection_confidence_strength(
    confidence_value: float | None,
    confidence_floor: float | None,
    confidence_ceiling: float | None,
) -> float:
    """Return correction strength in [0, 1] for a diagnostic confidence value."""
    validate_projection_confidence_bounds(confidence_floor, confidence_ceiling)
    if confidence_floor is None and confidence_ceiling is None:
        return 1.0
    if confidence_floor is None or confidence_ceiling is None:
        raise ValueError("projection confidence floor and ceiling must both be set")
    if confidence_value is None:
        return 0.0
    value = float(confidence_value)
    if not np.isfinite(value):
        return 0.0
    strength = (value - float(confidence_floor)) / (
        float(confidence_ceiling) - float(confidence_floor)
    )
    return float(np.clip(strength, 0.0, 1.0))


def projection_confidence_value(result: Any, field: str | None) -> float | None:
    """Read a scalar projection-confidence diagnostic from a refinement result."""
    if field is None:
        return None
    validate_projection_confidence_field(field)
    if field == "active_fraction":
        used_event_count = int(getattr(result, "used_event_count", 0) or 0)
        active_measurement_count = int(
            getattr(result, "active_measurement_count", 0) or 0
        )
        if used_event_count <= 0:
            return None
        return float(active_measurement_count / used_event_count)
    value = getattr(result, field, None)
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def smooth_projected_size(
    candidate_xywh: np.ndarray,
    projected_xywh: np.ndarray,
    *,
    refinement_mode: str,
    previous_projected_size: np.ndarray | None,
    projection_size_smoothing: float | None,
) -> np.ndarray:
    """Blend projected width/height toward the previous accepted projected size."""
    validate_refinement_mode(refinement_mode)
    validate_projection_size_smoothing(projection_size_smoothing)
    output = np.asarray(projected_xywh, dtype=float).reshape(4).copy()
    if (
        projection_size_smoothing is None
        or previous_projected_size is None
        or refinement_mode == "center-only"
    ):
        return output

    previous_size = np.asarray(previous_projected_size, dtype=float).reshape(2)
    smoothing = float(projection_size_smoothing)
    smoothed_size = np.array(output[2:], dtype=float, copy=True)
    size_axes = projected_size_axes(refinement_mode)
    smoothed_size[size_axes] = (
        (1.0 - smoothing) * output[2:][size_axes]
        + smoothing * previous_size[size_axes]
    )
    smoothed_size = np.maximum(smoothed_size, 0.0)
    if refinement_mode in {"size-only", "width-only", "height-only"}:
        candidate = np.asarray(candidate_xywh, dtype=float).reshape(4)
        center = candidate[:2] + 0.5 * candidate[2:]
    else:
        center = output[:2] + 0.5 * output[2:]
    output[:2] = center - 0.5 * smoothed_size
    output[2:] = smoothed_size
    return output


def project_size_axes(
    candidate_size: np.ndarray,
    projected_size: np.ndarray,
    *,
    refinement_mode: str,
) -> np.ndarray:
    """Return size with only the axes selected by refinement mode updated."""
    validate_refinement_mode(refinement_mode)
    candidate = np.asarray(candidate_size, dtype=float).reshape(2)
    projected = np.asarray(projected_size, dtype=float).reshape(2)
    if refinement_mode == "size-only":
        return np.array(projected, dtype=float, copy=True)
    if refinement_mode == "width-only":
        return np.array([projected[0], candidate[1]], dtype=float)
    if refinement_mode == "height-only":
        return np.array([candidate[0], projected[1]], dtype=float)
    raise ValueError(f"{refinement_mode!r} is not a size projection mode")


def projected_size_axes(refinement_mode: str) -> np.ndarray:
    """Return width/height indices modified by a projection mode."""
    validate_refinement_mode(refinement_mode)
    if refinement_mode in {"box", "size-only"}:
        return np.array([0, 1], dtype=int)
    if refinement_mode == "width-only":
        return np.array([0], dtype=int)
    if refinement_mode == "height-only":
        return np.array([1], dtype=int)
    return np.array([], dtype=int)


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


def validate_projection_size_smoothing(projection_size_smoothing: float | None) -> None:
    """Raise ValueError for invalid temporal size smoothing factors."""
    if projection_size_smoothing is None:
        return
    if not 0.0 <= float(projection_size_smoothing) <= 1.0:
        raise ValueError("projection_size_smoothing must be between 0 and 1")


def validate_projection_confidence_field(field: str | None) -> None:
    """Raise ValueError for unsupported projection confidence fields."""
    if field is None:
        return
    if field not in PROJECTION_CONFIDENCE_FIELDS:
        expected = ", ".join(PROJECTION_CONFIDENCE_FIELDS)
        raise ValueError(
            f"Unsupported projection confidence field {field!r}; expected one of {expected}"
        )


def validate_projection_confidence_bounds(
    floor: float | None,
    ceiling: float | None,
) -> None:
    """Raise ValueError for invalid confidence-to-strength bounds."""
    if floor is None and ceiling is None:
        return
    if floor is None or ceiling is None:
        raise ValueError("projection confidence floor and ceiling must both be set")
    if not np.isfinite(float(floor)) or not np.isfinite(float(ceiling)):
        raise ValueError("projection confidence floor and ceiling must be finite")
    if float(floor) >= float(ceiling):
        raise ValueError("projection confidence floor must be less than ceiling")


def validate_projection_confidence_weighting(
    field: str | None,
    floor: float | None,
    ceiling: float | None,
) -> None:
    """Raise ValueError for invalid confidence-weighted projection settings."""
    validate_projection_confidence_field(field)
    validate_projection_confidence_bounds(floor, ceiling)
    if field is None and (floor is not None or ceiling is not None):
        raise ValueError(
            "projection confidence field is required when floor/ceiling are set"
        )
    if field is not None and (floor is None or ceiling is None):
        raise ValueError(
            "projection confidence floor and ceiling are required when field is set"
        )


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
