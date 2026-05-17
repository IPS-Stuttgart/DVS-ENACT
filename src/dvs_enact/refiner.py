"""Post-hoc DVS-ENACT contour refinement for external trackers.

This module is benchmark-agnostic. An external tracker proposes a candidate
bounding box; the refiner crops events around that proposal, runs one
DVS-ENACT contour update, and returns a refined box plus diagnostics for
ablations of the form ``Tracker X`` versus ``Tracker X + DVS-ENACT``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mevdt import BoundingBox, EventBatch, empty_event_batch
from .trackers import DVSFullSCGPTracker

BBoxInput = BoundingBox | Mapping[str, float] | Sequence[float] | np.ndarray


@dataclass(frozen=True)
class DVSContourRefinerConfig:
    """Configuration for post-hoc contour refinement."""

    n_base_points: int = 32
    search_expansion_factor: float = 1.25
    max_events: int | None = 128
    min_events: int = 3
    min_event_velocity: float = 1e-6
    input_bbox_format: str = "xyxy"
    output_bbox_format: str = "xyxy"
    image_width: float | None = None
    image_height: float | None = None
    event_activity_floor: float = 0.05
    inactive_activity_threshold: float = 0.05
    use_event_polarity: bool = True
    polarity_mismatch_weight: float = 0.25
    polarity_contrast_sign: float | str | None = "infer"
    measurement_noise_variance: float = 4.0
    radial_noise_variance: float = 1.0
    shape_variance: float = 25.0
    kinematic_position_variance: float = 1e-3
    kinematic_orientation_variance: float = 1e-4
    refinement_blend: float = 1.0
    bbox_grid_points: int = 128
    event_selection_mode: str = "boundary"
    event_selection_angular_bins: int = 16

    def __post_init__(self) -> None:
        if self.n_base_points <= 0:
            raise ValueError("n_base_points must be positive")
        if self.search_expansion_factor <= 0.0:
            raise ValueError("search_expansion_factor must be positive")
        if self.max_events is not None and self.max_events <= 0:
            raise ValueError("max_events must be positive when provided")
        if self.min_events <= 0:
            raise ValueError("min_events must be positive")
        if self.min_event_velocity < 0.0:
            raise ValueError("min_event_velocity must be non-negative")
        if not 0.0 <= self.refinement_blend <= 1.0:
            raise ValueError("refinement_blend must be in [0, 1]")
        if self.event_selection_mode not in {"chronological", "boundary"}:
            raise ValueError("event_selection_mode must be 'chronological' or 'boundary'")
        if self.event_selection_angular_bins <= 0:
            raise ValueError("event_selection_angular_bins must be positive")


@dataclass(frozen=True)
class DVSRefinementResult:
    """Post-hoc refinement output and diagnostics."""

    candidate_bbox: dict[str, float]
    search_bbox: dict[str, float]
    refined_bbox: dict[str, float]
    output_bbox: dict[str, float]
    event_velocity: list[float]
    event_count: int
    used_event_count: int
    active_measurement_count: int
    mean_event_activity: float | None
    mean_event_polarity_weight: float | None
    polarity_consistency_fraction: float | None
    polarity_contrast_sign: float | None
    fallback_reason: str | None
    quadratic_form: float | None

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.output_bbox["x_min"],
            self.output_bbox["y_min"],
            self.output_bbox["x_max"],
            self.output_bbox["y_max"],
        )

    def as_xywh(self) -> tuple[float, float, float, float]:
        return (
            self.output_bbox["x_min"],
            self.output_bbox["y_min"],
            self.output_bbox["width"],
            self.output_bbox["height"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_bbox": self.candidate_bbox,
            "search_bbox": self.search_bbox,
            "refined_bbox": self.refined_bbox,
            "output_bbox": self.output_bbox,
            "event_velocity": self.event_velocity,
            "event_count": self.event_count,
            "used_event_count": self.used_event_count,
            "active_measurement_count": self.active_measurement_count,
            "mean_event_activity": self.mean_event_activity,
            "mean_event_polarity_weight": self.mean_event_polarity_weight,
            "polarity_consistency_fraction": self.polarity_consistency_fraction,
            "polarity_contrast_sign": self.polarity_contrast_sign,
            "fallback_reason": self.fallback_reason,
            "quadratic_form": self.quadratic_form,
        }


class DVSContourRefiner:
    """Refine external-tracker boxes using one DVS-ENACT contour update."""

    def __init__(self, config: DVSContourRefinerConfig | None = None):
        self.config = config or DVSContourRefinerConfig()

    def refine(
        self,
        candidate_bbox: BBoxInput,
        events: EventBatch,
        *,
        previous_candidate_bbox: BBoxInput | None = None,
        event_velocity: Sequence[float] | np.ndarray | None = None,
        timestamp_ns: int | None = None,
    ) -> DVSRefinementResult:
        """Return a DVS-refined version of an external tracker candidate box."""
        del timestamp_ns  # Reserved for future continuous-time benchmark adapters.
        candidate = bbox_to_dict(
            candidate_bbox,
            bbox_format=self.config.input_bbox_format,
        )
        search_bbox = expand_bbox(
            candidate,
            self.config.search_expansion_factor,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
        )
        cropped = crop_events_to_bbox(events, search_bbox)
        sampled = select_refinement_events(
            cropped,
            candidate,
            self.config.max_events,
            mode=self.config.event_selection_mode,
            angular_bins=self.config.event_selection_angular_bins,
        )
        velocity = self._event_velocity(
            candidate,
            previous_candidate_bbox,
            event_velocity,
        )

        if sampled.count < self.config.min_events:
            return self._fallback_result(
                candidate,
                search_bbox,
                velocity,
                int(cropped.count),
                int(sampled.count),
                "low_event_count",
            )
        if float(np.linalg.norm(velocity)) < self.config.min_event_velocity:
            return self._fallback_result(
                candidate,
                search_bbox,
                velocity,
                int(cropped.count),
                int(sampled.count),
                "low_event_velocity",
            )

        tracker = self._make_tracker(candidate)
        measurements = np.column_stack((sampled.x.astype(float), sampled.y.astype(float)))
        polarities = sampled.p if self.config.use_event_polarity else None
        tracker.update(
            measurements,
            event_velocity=velocity,
            event_polarities=polarities,
            polarity_mismatch_weight=self.config.polarity_mismatch_weight,
            polarity_contrast_sign=self.config.polarity_contrast_sign,
        )
        refined = tracker_bbox_to_dict(tracker, n=self.config.bbox_grid_points)
        output = blend_bboxes(candidate, refined, self.config.refinement_blend)
        output = clip_bbox(
            output,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
        )
        return DVSRefinementResult(
            candidate_bbox=candidate,
            search_bbox=search_bbox,
            refined_bbox=refined,
            output_bbox=format_bbox_dict(output, self.config.output_bbox_format),
            event_velocity=velocity.astype(float).tolist(),
            event_count=int(cropped.count),
            used_event_count=int(sampled.count),
            active_measurement_count=len(tracker.last_active_measurement_indices or []),
            mean_event_activity=_mean_array(tracker.last_event_activities),
            mean_event_polarity_weight=_mean_array(
                tracker.last_event_polarity_weights
            ),
            polarity_consistency_fraction=_mean_boolean(
                tracker.last_event_polarity_consistencies
            ),
            polarity_contrast_sign=tracker.last_polarity_contrast_sign,
            fallback_reason=None,
            quadratic_form=_optional_float(tracker.last_quadratic_form),
        )

    def _make_tracker(self, bbox: dict[str, float]) -> DVSFullSCGPTracker:
        shape_state = rectangle_radial_shape(
            bbox["width"],
            bbox["height"],
            self.config.n_base_points,
        )
        return DVSFullSCGPTracker(
            self.config.n_base_points,
            kinematic_state=np.array([bbox["center_x"], bbox["center_y"], 0.0]),
            kinematic_covariance=np.diag(
                [
                    self.config.kinematic_position_variance,
                    self.config.kinematic_position_variance,
                    self.config.kinematic_orientation_variance,
                ]
            ),
            shape_state=shape_state,
            shape_covariance=self.config.shape_variance
            * np.eye(self.config.n_base_points),
            velocities=False,
            measurement_noise=self.config.measurement_noise_variance * np.eye(2),
            radial_noise_variance=self.config.radial_noise_variance,
            event_activity_floor=self.config.event_activity_floor,
            inactive_activity_threshold=self.config.inactive_activity_threshold,
            polarity_mismatch_weight=self.config.polarity_mismatch_weight,
            polarity_contrast_sign=self.config.polarity_contrast_sign,
        )

    def _event_velocity(
        self,
        candidate: dict[str, float],
        previous_candidate_bbox: BBoxInput | None,
        event_velocity: Sequence[float] | np.ndarray | None,
    ) -> np.ndarray:
        if event_velocity is not None:
            velocity = np.asarray(event_velocity, dtype=float)
            if velocity.shape != (2,):
                raise ValueError("event_velocity must have shape (2,)")
            return velocity
        if previous_candidate_bbox is None:
            return np.zeros(2, dtype=float)
        previous = bbox_to_dict(
            previous_candidate_bbox,
            bbox_format=self.config.input_bbox_format,
        )
        return np.array(
            [
                candidate["center_x"] - previous["center_x"],
                candidate["center_y"] - previous["center_y"],
            ],
            dtype=float,
        )

    def _fallback_result(
        self,
        candidate: dict[str, float],
        search_bbox: dict[str, float],
        velocity: np.ndarray,
        event_count: int,
        used_event_count: int,
        reason: str,
    ) -> DVSRefinementResult:
        return DVSRefinementResult(
            candidate_bbox=candidate,
            search_bbox=search_bbox,
            refined_bbox=candidate,
            output_bbox=format_bbox_dict(candidate, self.config.output_bbox_format),
            event_velocity=velocity.astype(float).tolist(),
            event_count=event_count,
            used_event_count=used_event_count,
            active_measurement_count=0,
            mean_event_activity=None,
            mean_event_polarity_weight=None,
            polarity_consistency_fraction=None,
            polarity_contrast_sign=None,
            fallback_reason=reason,
            quadratic_form=None,
        )


def bbox_to_dict(bbox: BBoxInput, *, bbox_format: str = "xyxy") -> dict[str, float]:
    """Normalize a bounding box into xyxy, size, and center fields."""
    if isinstance(bbox, BoundingBox):
        x_min, y_min, x_max, y_max = bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max
    elif isinstance(bbox, Mapping):
        lower = {str(key).lower(): float(value) for key, value in bbox.items()}
        if {"x_min", "y_min", "x_max", "y_max"}.issubset(lower):
            x_min, y_min, x_max, y_max = (
                lower["x_min"],
                lower["y_min"],
                lower["x_max"],
                lower["y_max"],
            )
        elif {"x", "y", "width", "height"}.issubset(lower):
            x_min, y_min = lower["x"], lower["y"]
            x_max, y_max = x_min + lower["width"], y_min + lower["height"]
        else:
            raise ValueError("bbox mapping must contain xyxy or xywh fields")
    else:
        values = np.asarray(bbox, dtype=float)
        if values.shape != (4,):
            raise ValueError("bbox sequence must have shape (4,)")
        if bbox_format == "xyxy":
            x_min, y_min, x_max, y_max = values.tolist()
        elif bbox_format == "xywh":
            x_min, y_min, width, height = values.tolist()
            x_max, y_max = x_min + width, y_min + height
        else:
            raise ValueError("bbox_format must be 'xyxy' or 'xywh'")
    if x_max < x_min or y_max < y_min:
        raise ValueError("bbox must have non-negative width and height")
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    return {
        "x_min": float(x_min),
        "y_min": float(y_min),
        "x_max": float(x_max),
        "y_max": float(y_max),
        "width": width,
        "height": height,
        "area": width * height,
        "center_x": float(x_min + 0.5 * width),
        "center_y": float(y_min + 0.5 * height),
    }


def format_bbox_dict(bbox: dict[str, float], bbox_format: str) -> dict[str, float]:
    """Return bbox dict in the requested external format plus diagnostics."""
    normalized = bbox_to_dict(bbox)
    if bbox_format == "xyxy":
        return normalized
    if bbox_format == "xywh":
        return {
            **normalized,
            "x": normalized["x_min"],
            "y": normalized["y_min"],
        }
    raise ValueError("bbox_format must be 'xyxy' or 'xywh'")


def expand_bbox(
    bbox: BBoxInput,
    expansion_factor: float,
    *,
    image_width: float | None = None,
    image_height: float | None = None,
) -> dict[str, float]:
    """Expand a bbox around its center and optionally clip to image bounds."""
    if expansion_factor <= 0.0:
        raise ValueError("expansion_factor must be positive")
    normalized = bbox_to_dict(bbox)
    width = normalized["width"] * float(expansion_factor)
    height = normalized["height"] * float(expansion_factor)
    expanded = _bbox_from_center_extent(
        normalized["center_x"],
        normalized["center_y"],
        width,
        height,
    )
    return clip_bbox(expanded, image_width=image_width, image_height=image_height)


def clip_bbox(
    bbox: BBoxInput,
    *,
    image_width: float | None = None,
    image_height: float | None = None,
) -> dict[str, float]:
    normalized = bbox_to_dict(bbox)
    x_min, y_min, x_max, y_max = (
        normalized["x_min"],
        normalized["y_min"],
        normalized["x_max"],
        normalized["y_max"],
    )
    if image_width is not None:
        x_min = min(max(x_min, 0.0), float(image_width))
        x_max = min(max(x_max, 0.0), float(image_width))
    if image_height is not None:
        y_min = min(max(y_min, 0.0), float(image_height))
        y_max = min(max(y_max, 0.0), float(image_height))
    return bbox_to_dict((x_min, y_min, max(x_min, x_max), max(y_min, y_max)))


def crop_events_to_bbox(events: EventBatch, bbox: BBoxInput) -> EventBatch:
    """Return events inside a bbox."""
    normalized = bbox_to_dict(bbox)
    if events.count == 0:
        return empty_event_batch()
    mask = (
        (events.x >= normalized["x_min"])
        & (events.x <= normalized["x_max"])
        & (events.y >= normalized["y_min"])
        & (events.y <= normalized["y_max"])
    )
    return EventBatch(
        ts=events.ts[mask],
        x=events.x[mask],
        y=events.y[mask],
        p=events.p[mask],
    )


def select_refinement_events(
    events: EventBatch,
    candidate_bbox: BBoxInput,
    max_events: int | None,
    *,
    mode: str = "boundary",
    angular_bins: int = 16,
) -> EventBatch:
    """Select events for one post-hoc contour update.

    Chronological subsampling preserves temporal coverage. Boundary-aware
    selection is more conservative for strong base trackers: it prefers events
    close to the candidate contour, balances them by angular sector, and then
    restores chronological order before the update. This reduces the chance that
    background events inside the expanded search box dominate the SCGP update.
    """
    if mode == "chronological":
        return subsample_events_chronologically(events, max_events)
    if mode == "boundary":
        return subsample_events_near_bbox_boundary(
            events,
            candidate_bbox,
            max_events,
            angular_bins=angular_bins,
        )
    raise ValueError("mode must be 'chronological' or 'boundary'")


def subsample_events_near_bbox_boundary(
    events: EventBatch,
    bbox: BBoxInput,
    max_events: int | None,
    *,
    angular_bins: int = 16,
) -> EventBatch:
    """Deterministically subsample events close to a candidate-box contour."""
    if max_events is not None and max_events <= 0:
        raise ValueError("max_events must be positive when provided")
    if angular_bins <= 0:
        raise ValueError("angular_bins must be positive")
    if events.count == 0 or max_events is None or events.count <= max_events:
        return events

    normalized = bbox_to_dict(bbox)
    distances = event_distance_to_bbox_boundary(events, normalized)
    angles = np.arctan2(
        events.y.astype(float) - normalized["center_y"],
        events.x.astype(float) - normalized["center_x"],
    )
    bins = np.floor((angles + np.pi) / (2.0 * np.pi) * angular_bins).astype(np.int64)
    bins = np.clip(bins, 0, angular_bins - 1)

    selected: list[int] = []
    per_bin_quota = max(1, int(np.ceil(max_events / angular_bins)))
    for bin_index in range(angular_bins):
        candidates = np.flatnonzero(bins == bin_index)
        if candidates.size == 0:
            continue
        local_order = np.lexsort((events.ts[candidates], distances[candidates]))
        for index in candidates[local_order[:per_bin_quota]]:
            selected.append(int(index))

    if len(selected) < max_events:
        already_selected = np.zeros(events.count, dtype=bool)
        if selected:
            already_selected[np.asarray(selected, dtype=np.int64)] = True
        remaining = np.flatnonzero(~already_selected)
        global_order = np.lexsort((events.ts[remaining], distances[remaining]))
        fill_count = max_events - len(selected)
        selected.extend(int(index) for index in remaining[global_order[:fill_count]])

    selected = _deduplicate_indices(selected)[:max_events]
    selected_array = np.asarray(selected, dtype=np.int64)
    selected_array = selected_array[np.argsort(events.ts[selected_array], kind="stable")]
    return _event_batch_subset(events, selected_array)


def event_distance_to_bbox_boundary(events: EventBatch, bbox: BBoxInput) -> np.ndarray:
    """Return each event's distance to the nearest point on a bbox boundary."""
    normalized = bbox_to_dict(bbox)
    x = events.x.astype(float)
    y = events.y.astype(float)
    x_min = normalized["x_min"]
    x_max = normalized["x_max"]
    y_min = normalized["y_min"]
    y_max = normalized["y_max"]

    inside = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    inside_distance = np.minimum.reduce(
        (
            np.abs(x - x_min),
            np.abs(x - x_max),
            np.abs(y - y_min),
            np.abs(y - y_max),
        )
    )
    clipped_x = np.clip(x, x_min, x_max)
    clipped_y = np.clip(y, y_min, y_max)
    outside_distance = np.hypot(x - clipped_x, y - clipped_y)
    return np.where(inside, inside_distance, outside_distance)


def subsample_events_chronologically(
    events: EventBatch,
    max_events: int | None,
) -> EventBatch:
    """Deterministically subsample events while preserving temporal coverage."""
    if max_events is not None and max_events <= 0:
        raise ValueError("max_events must be positive when provided")
    if events.count == 0 or max_events is None or events.count <= max_events:
        return events
    order = np.argsort(events.ts, kind="stable")
    positions = np.linspace(0, events.count - 1, max_events, dtype=np.int64)
    selected = order[positions]
    return _event_batch_subset(events, selected)


def blend_bboxes(
    base_bbox: BBoxInput,
    refined_bbox: BBoxInput,
    blend: float,
) -> dict[str, float]:
    """Interpolate between base and refined xyxy boxes."""
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0, 1]")
    base = bbox_to_dict(base_bbox)
    refined = bbox_to_dict(refined_bbox)
    return bbox_to_dict(
        tuple(
            (1.0 - blend) * base[key] + blend * refined[key]
            for key in ("x_min", "y_min", "x_max", "y_max")
        )
    )


def rectangle_radial_shape(width: float, height: float, n_base_points: int) -> np.ndarray:
    """Return star-convex radial samples for an axis-aligned rectangle."""
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive")
    angles = np.linspace(0.0, 2.0 * np.pi, n_base_points, endpoint=False)
    half_width = 0.5 * float(width)
    half_height = 0.5 * float(height)
    cos_abs = np.abs(np.cos(angles))
    sin_abs = np.abs(np.sin(angles))
    eps = 1e-12
    x_limits = np.divide(
        half_width,
        cos_abs,
        out=np.full_like(angles, np.inf),
        where=cos_abs > eps,
    )
    y_limits = np.divide(
        half_height,
        sin_abs,
        out=np.full_like(angles, np.inf),
        where=sin_abs > eps,
    )
    return np.minimum(x_limits, y_limits)


def tracker_bbox_to_dict(tracker: DVSFullSCGPTracker, n: int) -> dict[str, float]:
    estimate = tracker.get_bounding_box(n=n)
    center = np.asarray(estimate["center_xy"], dtype=float)
    dimension = np.asarray(estimate["dimension"], dtype=float)
    return _bbox_from_center_extent(
        center[0],
        center[1],
        max(float(dimension[0]), 0.0),
        max(float(dimension[1]), 0.0),
    )


def _event_batch_subset(events: EventBatch, indices: np.ndarray) -> EventBatch:
    return EventBatch(
        ts=events.ts[indices],
        x=events.x[indices],
        y=events.y[indices],
        p=events.p[indices],
    )


def _deduplicate_indices(indices: Iterable[int]) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for index in indices:
        if index in seen:
            continue
        selected.append(index)
        seen.add(index)
    return selected


def _bbox_from_center_extent(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> dict[str, float]:
    width = max(float(width), 0.0)
    height = max(float(height), 0.0)
    return bbox_to_dict(
        (
            float(center_x) - 0.5 * width,
            float(center_y) - 0.5 * height,
            float(center_x) + 0.5 * width,
            float(center_y) + 0.5 * height,
        )
    )


def _mean_array(values) -> float | None:
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if values.size else None


def _mean_boolean(values) -> float | None:
    if values is None:
        return None
    filtered = [float(value) for value in values if value is not None]
    return float(np.mean(filtered)) if filtered else None


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)
