"""Point-process likelihood scores for DVS-ENACT refinements."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .event_likelihood import (
    ContourSample,
    EventLikelihoodConfig,
    EventLikelihoodTerms,
    event_batch_log_likelihood_terms,
)
from .mevdt import EventBatch
from .refiner import BBoxInput, bbox_to_dict


@dataclass(frozen=True)
class BBoxEventLikelihoodConfig:
    """Configuration for scoring one bounding box against an event window."""

    likelihood: EventLikelihoodConfig = field(
        default_factory=lambda: EventLikelihoodConfig(activity_floor=0.05)
    )
    samples_per_edge: int = 24

    def __post_init__(self) -> None:
        if self.samples_per_edge <= 0:
            raise ValueError("samples_per_edge must be positive")


@dataclass(frozen=True)
class BBoxEventLikelihoodScore:
    """Point-process likelihood terms for one scored bounding box."""

    bbox: dict[str, float]
    terms: EventLikelihoodTerms

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "terms": asdict(self.terms),
        }


@dataclass(frozen=True)
class RefinementLikelihoodComparison:
    """Likelihood comparison between a base and a refined bbox."""

    base: BBoxEventLikelihoodScore
    refined: BBoxEventLikelihoodScore
    delta_log_likelihood: float
    delta_log_likelihood_per_event: float | None

    @property
    def refined_is_better(self) -> bool:
        """Return whether the refined box has strictly higher likelihood."""
        return self.delta_log_likelihood > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "refined": self.refined.to_dict(),
            "delta_log_likelihood": float(self.delta_log_likelihood),
            "delta_log_likelihood_per_event": self.delta_log_likelihood_per_event,
            "refined_is_better": self.refined_is_better,
        }


def bbox_contour_sample(
    bbox: BBoxInput,
    *,
    bbox_format: str = "xyxy",
    samples_per_edge: int = 24,
) -> ContourSample:
    """Return a contour sample on the boundary of an axis-aligned bbox."""
    if samples_per_edge <= 0:
        raise ValueError("samples_per_edge must be positive")
    normalized = bbox_to_dict(bbox, bbox_format=bbox_format)
    width = normalized["width"]
    height = normalized["height"]
    if width <= 0.0 or height <= 0.0:
        raise ValueError("bbox must have positive width and height")

    x_min = normalized["x_min"]
    x_max = normalized["x_max"]
    y_min = normalized["y_min"]
    y_max = normalized["y_max"]
    xs = x_min + (np.arange(samples_per_edge, dtype=float) + 0.5) * (
        width / samples_per_edge
    )
    ys = y_min + (np.arange(samples_per_edge, dtype=float) + 0.5) * (
        height / samples_per_edge
    )

    top = np.column_stack((xs, np.full(samples_per_edge, y_min)))
    right = np.column_stack((np.full(samples_per_edge, x_max), ys))
    bottom = np.column_stack((xs[::-1], np.full(samples_per_edge, y_max)))
    left = np.column_stack((np.full(samples_per_edge, x_min), ys[::-1]))
    points = np.vstack((top, right, bottom, left))
    normals = np.vstack(
        (
            np.tile((0.0, -1.0), (samples_per_edge, 1)),
            np.tile((1.0, 0.0), (samples_per_edge, 1)),
            np.tile((0.0, 1.0), (samples_per_edge, 1)),
            np.tile((-1.0, 0.0), (samples_per_edge, 1)),
        )
    )
    weights = np.concatenate(
        (
            np.full(samples_per_edge, width / samples_per_edge),
            np.full(samples_per_edge, height / samples_per_edge),
            np.full(samples_per_edge, width / samples_per_edge),
            np.full(samples_per_edge, height / samples_per_edge),
        )
    )
    angles = np.arctan2(
        points[:, 1] - normalized["center_y"],
        points[:, 0] - normalized["center_x"],
    )
    return ContourSample(points=points, normals=normals, weights=weights, angles=angles)


def event_batch_xy(events: EventBatch) -> np.ndarray:
    """Return event coordinates as an ``N x 2`` float array."""
    if events.count == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack((events.x.astype(float), events.y.astype(float)))


def score_bbox_event_likelihood(
    bbox: BBoxInput,
    events: EventBatch,
    velocity: np.ndarray | list[float] | tuple[float, float],
    config: BBoxEventLikelihoodConfig | None = None,
    *,
    bbox_format: str = "xyxy",
    batch_duration: float | None = None,
    image_area: float | None = None,
) -> BBoxEventLikelihoodScore:
    """Score a bbox by the contour-conditioned event point-process likelihood."""
    config = config or BBoxEventLikelihoodConfig()
    normalized = bbox_to_dict(bbox, bbox_format=bbox_format)
    contour = bbox_contour_sample(
        normalized,
        samples_per_edge=config.samples_per_edge,
    )
    terms = event_batch_log_likelihood_terms(
        event_batch_xy(events),
        contour,
        np.asarray(velocity, dtype=float),
        config.likelihood,
        batch_duration=batch_duration,
        image_area=image_area,
    )
    return BBoxEventLikelihoodScore(bbox=normalized, terms=terms)


def compare_refinement_likelihood(
    base_bbox: BBoxInput,
    refined_bbox: BBoxInput,
    events: EventBatch,
    velocity: np.ndarray | list[float] | tuple[float, float],
    config: BBoxEventLikelihoodConfig | None = None,
    *,
    bbox_format: str = "xyxy",
    batch_duration: float | None = None,
    image_area: float | None = None,
) -> RefinementLikelihoodComparison:
    """Compare base and refined bboxes using the same event window."""
    config = config or BBoxEventLikelihoodConfig()
    base_score = score_bbox_event_likelihood(
        base_bbox,
        events,
        velocity,
        config,
        bbox_format=bbox_format,
        batch_duration=batch_duration,
        image_area=image_area,
    )
    refined_score = score_bbox_event_likelihood(
        refined_bbox,
        events,
        velocity,
        config,
        bbox_format=bbox_format,
        batch_duration=batch_duration,
        image_area=image_area,
    )
    delta = refined_score.terms.log_likelihood - base_score.terms.log_likelihood
    event_count = max(base_score.terms.event_count, refined_score.terms.event_count)
    per_event = None if event_count <= 0 else float(delta / event_count)
    return RefinementLikelihoodComparison(
        base=base_score,
        refined=refined_score,
        delta_log_likelihood=float(delta),
        delta_log_likelihood_per_event=per_event,
    )
