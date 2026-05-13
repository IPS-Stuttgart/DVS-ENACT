"""Label-assisted MEVDT tracker comparison utilities."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from pyrecest.filters.gprhm_tracker import FullSCGPTracker

from .mevdt import (
    MEVDT_DATASET_URL,
    MEVDT_DOI,
    BoundingBox,
    EventBatch,
    empty_event_batch,
    find_event_csv_files,
    find_tracking_label_files,
    read_event_csv,
    read_tracking_labels,
    summarize_loaded_sequence,
)
from .trackers import DVSFullSCGPTracker


@dataclass(frozen=True)
class TrackerComparisonConfig:
    """Parameters for the label-assisted MEVDT tracker comparison."""

    n_base_points: int = 32
    max_events_per_window: int = 64
    min_events_per_window: int = 3
    max_windows: int | None = 500
    event_cloud_min_extent_fraction: float = 0.25
    event_cloud_min_extent_px: float = 2.0
    event_activity_floor: float = 0.05
    inactive_activity_threshold: float = 0.05
    use_event_polarity: bool = True
    polarity_mismatch_weight: float = 0.25
    polarity_contrast_sign: float | str | None = "infer"
    collapse_threshold: float = 0.75
    measurement_noise_variance: float = 4.0
    radial_noise_variance: float = 1.0
    shape_variance: float = 25.0
    kinematic_position_variance: float = 1e-3
    kinematic_orientation_variance: float = 1e-4
    bbox_grid_points: int = 128


@dataclass(frozen=True)
class WindowFilterConfig:
    """Geometry filters for label-assisted MEVDT comparison windows."""

    image_width: float = 240.0
    image_height: float = 180.0
    border_margin_px: float = 1.0
    min_width: float = 8.0
    min_height: float = 8.0
    min_area: float = 100.0
    max_width_change_fraction: float = 0.25
    max_height_change_fraction: float = 0.25
    trim_track_ends: int = 3


def rectangle_radial_shape(
    width: float,
    height: float,
    n_base_points: int = 32,
    orientation: float = 0.0,
) -> np.ndarray:
    """Return star-convex radial samples for an axis-aligned rectangle."""
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive")
    if n_base_points <= 0:
        raise ValueError("n_base_points must be positive")

    angles = np.linspace(0.0, 2.0 * np.pi, n_base_points, endpoint=False) - orientation
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


def window_filter_reasons(
    current: BoundingBox,
    following: BoundingBox,
    track_window_index: int,
    track_window_count: int,
    config: WindowFilterConfig,
) -> list[str]:
    """Return reasons why a label window should be excluded."""
    if config.border_margin_px < 0.0:
        raise ValueError("border_margin_px must be non-negative")
    if config.trim_track_ends < 0:
        raise ValueError("trim_track_ends must be non-negative")
    reasons: list[str] = []
    boxes = (current, following)

    if (
        track_window_index < config.trim_track_ends
        or track_window_index >= track_window_count - config.trim_track_ends
    ):
        reasons.append("track_end_trim")

    for box in boxes:
        if (
            box.x_min <= config.border_margin_px
            or box.y_min <= config.border_margin_px
            or box.x_max >= config.image_width - config.border_margin_px
            or box.y_max >= config.image_height - config.border_margin_px
        ):
            reasons.append("border_touch")
            break

    if any(box.width < config.min_width or box.height < config.min_height for box in boxes):
        reasons.append("small_box")
    if any(box.area < config.min_area for box in boxes):
        reasons.append("small_area")

    width_change = abs(following.width - current.width) / max(
        current.width,
        following.width,
        1e-12,
    )
    height_change = abs(following.height - current.height) / max(
        current.height,
        following.height,
        1e-12,
    )
    if width_change > config.max_width_change_fraction:
        reasons.append("width_change")
    if height_change > config.max_height_change_fraction:
        reasons.append("height_change")
    return reasons


def subsample_events_chronologically(
    events: EventBatch,
    max_events: int | None,
) -> EventBatch:
    """Return a deterministic timestamp-ordered event subset."""
    if max_events is not None and max_events <= 0:
        raise ValueError("max_events must be positive when provided")
    if events.count == 0:
        return events

    order = np.argsort(events.ts, kind="stable")
    if max_events is None or events.count <= max_events:
        selected = order
    else:
        positions = np.linspace(0, events.count - 1, max_events, dtype=np.int64)
        selected = order[positions]
    return EventBatch(
        ts=events.ts[selected],
        x=events.x[selected],
        y=events.y[selected],
        p=events.p[selected],
    )


def events_for_label_window(
    events: EventBatch,
    current: BoundingBox,
    following: BoundingBox,
) -> EventBatch:
    """Select events in the current label box between consecutive label times."""
    if current.timestamp_ns is None or following.timestamp_ns is None:
        return empty_event_batch()
    mask = (
        (events.ts >= current.timestamp_ns)
        & (events.ts < following.timestamp_ns)
        & (events.x >= current.x_min)
        & (events.x <= current.x_max)
        & (events.y >= current.y_min)
        & (events.y <= current.y_max)
    )
    return EventBatch(
        ts=events.ts[mask],
        x=events.x[mask],
        y=events.y[mask],
        p=events.p[mask],
    )


def bbox_to_dict(bbox: BoundingBox) -> dict[str, float]:
    """Serialize a labeled box with xyxy and extent fields."""
    return {
        "x_min": float(bbox.x_min),
        "y_min": float(bbox.y_min),
        "x_max": float(bbox.x_max),
        "y_max": float(bbox.y_max),
        "width": float(bbox.width),
        "height": float(bbox.height),
        "area": float(bbox.area),
        "center_x": float(bbox.center[0]),
        "center_y": float(bbox.center[1]),
    }


def _bbox_from_center_extent(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> dict[str, float]:
    width = max(float(width), 0.0)
    height = max(float(height), 0.0)
    return {
        "x_min": float(center_x - 0.5 * width),
        "y_min": float(center_y - 0.5 * height),
        "x_max": float(center_x + 0.5 * width),
        "y_max": float(center_y + 0.5 * height),
        "width": width,
        "height": height,
        "area": width * height,
        "center_x": float(center_x),
        "center_y": float(center_y),
    }


def event_cloud_centroid_bbox(
    events: EventBatch,
    reference_bbox: BoundingBox | dict[str, float],
    *,
    min_extent_fraction: float = 0.25,
    min_extent_px: float = 2.0,
    min_event_count: int = 1,
) -> tuple[dict[str, float], dict]:
    """Estimate a heuristic event-cloud bbox from current-window events.

    The center is the event-coordinate centroid. The extent is the event-cloud
    span with conservative lower bounds from the current label extent. This is a
    deterministic label-window heuristic, not an autonomous learned tracker.
    """
    if min_extent_fraction < 0.0:
        raise ValueError("min_extent_fraction must be non-negative")
    if min_extent_px < 0.0:
        raise ValueError("min_extent_px must be non-negative")
    if min_event_count <= 0:
        raise ValueError("min_event_count must be positive")

    reference = _bbox_parts(reference_bbox)
    xs = events.x.astype(float)
    ys = events.y.astype(float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]
    finite_event_count = int(xs.size)
    fallback = {
        "center_source": "reference_bbox",
        "extent_source": "reference_bbox",
        "fallback_reason": "low_event_count",
        "event_count": int(events.count),
        "finite_event_count": finite_event_count,
        "raw_event_width": None,
        "raw_event_height": None,
        "min_width": None,
        "min_height": None,
    }
    if events.count < min_event_count:
        return dict(reference), fallback

    if finite_event_count < min_event_count:
        fallback["fallback_reason"] = "low_finite_event_count"
        return dict(reference), fallback

    raw_width = float(np.max(xs) - np.min(xs))
    raw_height = float(np.max(ys) - np.min(ys))
    reference_width = max(float(reference["width"]), 0.0)
    reference_height = max(float(reference["height"]), 0.0)
    min_width = max(
        min_extent_px,
        min_extent_fraction * reference_width,
    )
    min_height = max(
        min_extent_px,
        min_extent_fraction * reference_height,
    )
    if reference_width > 0.0:
        min_width = min(min_width, reference_width)
    if reference_height > 0.0:
        min_height = min(min_height, reference_height)

    width = max(raw_width, min_width)
    height = max(raw_height, min_height)
    if reference_width > 0.0:
        width = min(width, reference_width)
    if reference_height > 0.0:
        height = min(height, reference_height)

    return (
        _bbox_from_center_extent(
            float(np.mean(xs)),
            float(np.mean(ys)),
            width,
            height,
        ),
        {
            "center_source": "event_cloud_centroid",
            "extent_source": "event_cloud_span_with_reference_minimum",
            "fallback_reason": None,
            "event_count": int(events.count),
            "finite_event_count": finite_event_count,
            "raw_event_width": raw_width,
            "raw_event_height": raw_height,
            "min_width": float(min_width),
            "min_height": float(min_height),
        },
    )


def estimated_tracker_bbox(tracker, n: int = 128) -> dict[str, float]:
    """Return a tracker contour bounding box in the same xyxy shape as labels."""
    estimate = tracker.get_bounding_box(n=n)
    center = np.asarray(estimate["center_xy"], dtype=float)
    dimension = np.asarray(estimate["dimension"], dtype=float)
    width = max(float(dimension[0]), 0.0)
    height = max(float(dimension[1]), 0.0)
    return _bbox_from_center_extent(center[0], center[1], width, height)


def _bbox_parts(bbox: BoundingBox | dict[str, float]) -> dict[str, float]:
    return bbox_to_dict(bbox) if isinstance(bbox, BoundingBox) else bbox


def bbox_iou(
    estimated_bbox: BoundingBox | dict[str, float],
    target_bbox: BoundingBox | dict[str, float],
) -> float:
    """Compute axis-aligned bbox IoU."""
    estimated = _bbox_parts(estimated_bbox)
    target = _bbox_parts(target_bbox)
    x_min = max(estimated["x_min"], target["x_min"])
    y_min = max(estimated["y_min"], target["y_min"])
    x_max = min(estimated["x_max"], target["x_max"])
    y_max = min(estimated["y_max"], target["y_max"])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    union = estimated["area"] + target["area"] - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def bbox_metrics(
    estimated_bbox: BoundingBox | dict[str, float],
    target_bbox: BoundingBox | dict[str, float],
    velocity: Iterable[float],
    collapse_threshold: float = 0.75,
) -> dict:
    """Compute bbox comparison metrics for one tracker window."""
    estimated = _bbox_parts(estimated_bbox)
    target = _bbox_parts(target_bbox)
    velocity = np.asarray(list(velocity), dtype=float)
    center_error = math.hypot(
        estimated["center_x"] - target["center_x"],
        estimated["center_y"] - target["center_y"],
    )
    width_ratio = estimated["width"] / target["width"] if target["width"] > 0 else None
    height_ratio = (
        estimated["height"] / target["height"] if target["height"] > 0 else None
    )
    area_ratio = estimated["area"] / target["area"] if target["area"] > 0 else None
    if abs(float(velocity[0])) >= abs(float(velocity[1])):
        inactive_axis = "height"
        inactive_axis_ratio = height_ratio
    else:
        inactive_axis = "width"
        inactive_axis_ratio = width_ratio
    collapsed = (
        inactive_axis_ratio is not None
        and inactive_axis_ratio < collapse_threshold
    )
    return {
        "bbox_iou": bbox_iou(estimated, target),
        "center_error_px": float(center_error),
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "area_ratio": area_ratio,
        "inactive_axis": inactive_axis,
        "inactive_axis_ratio": inactive_axis_ratio,
        "collapsed": bool(collapsed),
    }


def _make_tracker(tracker_cls, initial_label: BoundingBox, config: TrackerComparisonConfig):
    shape_state = rectangle_radial_shape(
        initial_label.width,
        initial_label.height,
        n_base_points=config.n_base_points,
    )
    kinematic_state = np.array(
        [initial_label.center[0], initial_label.center[1], 0.0],
        dtype=float,
    )
    tracker_kwargs = {
        "kinematic_state": kinematic_state,
        "kinematic_covariance": np.diag(
            [
                config.kinematic_position_variance,
                config.kinematic_position_variance,
                config.kinematic_orientation_variance,
            ]
        ),
        "shape_state": shape_state,
        "shape_covariance": config.shape_variance * np.eye(config.n_base_points),
        "velocities": False,
        "measurement_noise": config.measurement_noise_variance * np.eye(2),
        "radial_noise_variance": config.radial_noise_variance,
    }
    if tracker_cls is DVSFullSCGPTracker:
        tracker_kwargs.update(
            {
                "event_activity_floor": config.event_activity_floor,
                "inactive_activity_threshold": config.inactive_activity_threshold,
                "polarity_mismatch_weight": config.polarity_mismatch_weight,
                "polarity_contrast_sign": config.polarity_contrast_sign,
            }
        )
    return tracker_cls(config.n_base_points, **tracker_kwargs)


def _recenter_tracker(tracker, label: BoundingBox) -> None:
    tracker.state[:2] = label.center
    tracker.state[2] = 0.0
    tracker._sync_state_views()


def _event_measurements(events: EventBatch) -> np.ndarray:
    return np.column_stack((events.x.astype(float), events.y.astype(float)))


def _labels_by_track(labels: Iterable[BoundingBox]) -> dict[int, list[BoundingBox]]:
    labels_by_track: dict[int, list[BoundingBox]] = {}
    for label in labels:
        labels_by_track.setdefault(label.track_id, []).append(label)
    return {
        track_id: sorted(track_labels, key=lambda label: label.frame)
        for track_id, track_labels in labels_by_track.items()
    }


def _mean_optional(values: Iterable[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(filtered)) if filtered else None


def _aggregate_tracker_metrics(windows: list[dict], tracker_key: str) -> dict:
    metrics = [window[tracker_key]["metrics"] for window in windows]
    collapse_count = sum(1 for item in metrics if item["collapsed"])
    return {
        "mean_bbox_iou": _mean_optional(item["bbox_iou"] for item in metrics),
        "mean_center_error_px": _mean_optional(
            item["center_error_px"] for item in metrics
        ),
        "mean_width_ratio": _mean_optional(item["width_ratio"] for item in metrics),
        "mean_height_ratio": _mean_optional(item["height_ratio"] for item in metrics),
        "mean_area_ratio": _mean_optional(item["area_ratio"] for item in metrics),
        "mean_inactive_axis_ratio": _mean_optional(
            item["inactive_axis_ratio"] for item in metrics
        ),
        "collapse_count": int(collapse_count),
        "collapse_fraction": float(collapse_count / len(metrics)) if metrics else 0.0,
    }


def _tracker_metric_delta(value: dict, reference: dict) -> dict:
    return {
        "mean_bbox_iou": _optional_delta(
            value["mean_bbox_iou"],
            reference["mean_bbox_iou"],
        ),
        "mean_center_error_px": _optional_delta(
            value["mean_center_error_px"],
            reference["mean_center_error_px"],
        ),
        "mean_inactive_axis_ratio": _optional_delta(
            value["mean_inactive_axis_ratio"],
            reference["mean_inactive_axis_ratio"],
        ),
        "collapse_count": value["collapse_count"] - reference["collapse_count"],
    }


def _summarize_comparison(
    windows: list[dict],
    windows_considered: int,
    skipped_low_event_windows: int,
    skipped_missing_timestamp_windows: int,
    skipped_filter_windows: int,
    filter_skip_reasons: dict[str, int],
) -> dict:
    constant_position = _aggregate_tracker_metrics(windows, "constant_position")
    event_cloud_centroid = _aggregate_tracker_metrics(windows, "event_cloud_centroid")
    baseline = _aggregate_tracker_metrics(windows, "baseline")
    dvs_enact = _aggregate_tracker_metrics(windows, "dvs_enact")
    return {
        "windows_considered": int(windows_considered),
        "windows_evaluated": len(windows),
        "skipped_low_event_windows": int(skipped_low_event_windows),
        "skipped_missing_timestamp_windows": int(skipped_missing_timestamp_windows),
        "skipped_filter_windows": int(skipped_filter_windows),
        "filter_skip_reasons": dict(sorted(filter_skip_reasons.items())),
        "constant_position": constant_position,
        "event_cloud_centroid": event_cloud_centroid,
        "baseline": baseline,
        "dvs_enact": dvs_enact,
        "event_cloud_centroid_minus_constant_position": _tracker_metric_delta(
            event_cloud_centroid,
            constant_position,
        ),
        "baseline_minus_constant_position": _tracker_metric_delta(
            baseline,
            constant_position,
        ),
        "baseline_minus_event_cloud_centroid": _tracker_metric_delta(
            baseline,
            event_cloud_centroid,
        ),
        "dvs_enact_minus_constant_position": _tracker_metric_delta(
            dvs_enact,
            constant_position,
        ),
        "dvs_enact_minus_event_cloud_centroid": _tracker_metric_delta(
            dvs_enact,
            event_cloud_centroid,
        ),
        "dvs_enact_minus_baseline": _tracker_metric_delta(dvs_enact, baseline),
    }


def _optional_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value - baseline)


def compare_trackers_on_labels(
    labels: Iterable[BoundingBox],
    events: EventBatch,
    config: TrackerComparisonConfig | None = None,
    window_filter: WindowFilterConfig | None = None,
) -> dict:
    """Run the label-assisted vanilla-SCGP vs DVS-ENACT comparison."""
    config = config or TrackerComparisonConfig()
    labels = list(labels)
    labels_by_track = _labels_by_track(labels)
    result_windows: list[dict] = []
    windows_considered = 0
    skipped_low_event_windows = 0
    skipped_missing_timestamp_windows = 0
    skipped_filter_windows = 0
    filter_skip_reasons: dict[str, int] = {}

    for track_id in sorted(labels_by_track):
        track_labels = labels_by_track[track_id]
        if len(track_labels) < 2:
            continue
        baseline_tracker = None
        dvs_tracker = None
        track_window_count = len(track_labels) - 1
        for track_window_index, (current, following) in enumerate(
            zip(track_labels[:-1], track_labels[1:], strict=False)
        ):
            if config.max_windows is not None and windows_considered >= config.max_windows:
                break
            windows_considered += 1
            if window_filter is not None:
                reasons = window_filter_reasons(
                    current,
                    following,
                    track_window_index,
                    track_window_count,
                    window_filter,
                )
                if reasons:
                    skipped_filter_windows += 1
                    for reason in reasons:
                        filter_skip_reasons[reason] = (
                            filter_skip_reasons.get(reason, 0) + 1
                        )
                    baseline_tracker = None
                    dvs_tracker = None
                    continue
            if current.timestamp_ns is None or following.timestamp_ns is None:
                skipped_missing_timestamp_windows += 1
                baseline_tracker = None
                dvs_tracker = None
                continue
            window_events = events_for_label_window(events, current, following)
            sampled_events = subsample_events_chronologically(
                window_events,
                config.max_events_per_window,
            )
            if sampled_events.count < config.min_events_per_window:
                skipped_low_event_windows += 1
                baseline_tracker = None
                dvs_tracker = None
                continue

            if baseline_tracker is None:
                baseline_tracker = _make_tracker(FullSCGPTracker, current, config)
            if dvs_tracker is None:
                dvs_tracker = _make_tracker(DVSFullSCGPTracker, current, config)
            _recenter_tracker(baseline_tracker, current)
            _recenter_tracker(dvs_tracker, current)
            measurements = _event_measurements(sampled_events)
            velocity = following.center - current.center
            baseline_tracker.update(measurements)
            event_polarities = sampled_events.p if config.use_event_polarity else None
            dvs_tracker.update(
                measurements,
                event_velocity=velocity,
                event_polarities=event_polarities,
            )

            baseline_bbox = estimated_tracker_bbox(
                baseline_tracker,
                n=config.bbox_grid_points,
            )
            dvs_bbox = estimated_tracker_bbox(dvs_tracker, n=config.bbox_grid_points)
            constant_position_bbox = bbox_to_dict(current)
            event_cloud_bbox, event_cloud_metadata = event_cloud_centroid_bbox(
                sampled_events,
                constant_position_bbox,
                min_extent_fraction=config.event_cloud_min_extent_fraction,
                min_extent_px=config.event_cloud_min_extent_px,
                min_event_count=config.min_events_per_window,
            )
            target_bbox = bbox_to_dict(following)
            result_windows.append(
                {
                    "track_id": int(track_id),
                    "frame": int(current.frame),
                    "next_frame": int(following.frame),
                    "timestamp_ns": int(current.timestamp_ns),
                    "next_timestamp_ns": int(following.timestamp_ns),
                    "event_count": int(window_events.count),
                    "used_event_count": int(sampled_events.count),
                    "center_velocity_px_per_frame": velocity.astype(float).tolist(),
                    "reference_bbox": constant_position_bbox,
                    "target_bbox": target_bbox,
                    "constant_position": {
                        "bbox": constant_position_bbox,
                        "metrics": bbox_metrics(
                            constant_position_bbox,
                            target_bbox,
                            velocity,
                            collapse_threshold=config.collapse_threshold,
                        ),
                    },
                    "event_cloud_centroid": {
                        "bbox": event_cloud_bbox,
                        "metadata": event_cloud_metadata,
                        "metrics": bbox_metrics(
                            event_cloud_bbox,
                            target_bbox,
                            velocity,
                            collapse_threshold=config.collapse_threshold,
                        ),
                    },
                    "baseline": {
                        "bbox": baseline_bbox,
                        "metrics": bbox_metrics(
                            baseline_bbox,
                            target_bbox,
                            velocity,
                            collapse_threshold=config.collapse_threshold,
                        ),
                    },
                    "dvs_enact": {
                        "bbox": dvs_bbox,
                        "used_event_polarity": bool(config.use_event_polarity),
                        "active_measurement_count": len(
                            dvs_tracker.last_active_measurement_indices or []
                        ),
                        "polarity_contrast_sign": (
                            None
                            if dvs_tracker.last_polarity_contrast_sign is None
                            else float(dvs_tracker.last_polarity_contrast_sign)
                        ),
                        "mean_event_activity": float(
                            np.mean(np.asarray(dvs_tracker.last_event_activities))
                        )
                        if dvs_tracker.last_event_activities is not None
                        else None,
                        "metrics": bbox_metrics(
                            dvs_bbox,
                            target_bbox,
                            velocity,
                            collapse_threshold=config.collapse_threshold,
                        ),
                    },
                }
            )
        if config.max_windows is not None and windows_considered >= config.max_windows:
            break

    return {
        "parsed_sequence": summarize_loaded_sequence(labels, events),
        "tracker_parameters": asdict(config),
        "window_filter": asdict(window_filter) if window_filter is not None else None,
        "baseline_descriptions": {
            "constant_position": (
                "Predicts the next box as unchanged from the current label."
            ),
            "event_cloud_centroid": (
                "Heuristic event-cloud baseline: centers the prediction on the "
                "current-window event centroid and uses event-cloud span with "
                "conservative current-label extent lower bounds. It is not an "
                "autonomous learned tracker."
            ),
            "baseline": (
                "Vanilla PyRecEst FullSCGPTracker updated with the "
                "current-window events."
            ),
            "dvs_enact": (
                "DVSFullSCGPTracker updated with the current-window events and "
                "normal-flow activity support."
            ),
        },
        "summary": _summarize_comparison(
            result_windows,
            windows_considered,
            skipped_low_event_windows,
            skipped_missing_timestamp_windows,
            skipped_filter_windows,
            filter_skip_reasons,
        ),
        "windows": result_windows,
    }


def select_mevdt_event_and_label_files(
    dataset_root: str | Path,
    event_csv: str | Path | None = None,
    label_file: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve matching MEVDT event and tracking-label files."""
    root = Path(dataset_root)
    if event_csv is not None:
        selected_event = Path(event_csv)
        if not selected_event.exists():
            raise FileNotFoundError(f"event CSV file does not exist: {selected_event}")
    else:
        event_files = find_event_csv_files(root)
        if not event_files:
            raise FileNotFoundError(f"No event CSV files found below {root}")
        selected_event = event_files[0]

    if label_file is not None:
        selected_label = Path(label_file)
        if not selected_label.exists():
            raise FileNotFoundError(f"tracking label file does not exist: {selected_label}")
        return selected_event, selected_label

    label_files = find_tracking_label_files(root)
    if not label_files:
        raise FileNotFoundError(f"No tracking label files found below {root}")
    sequence_id = selected_event.name.replace("_events.csv", "")
    matching = [path for path in label_files if sequence_id in path.name]
    candidates = matching if matching else label_files
    return selected_event, sorted(candidates, key=_label_preference_key)[0]


def _label_preference_key(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if path.suffix.lower() == ".json" and "coco" in name:
        rank = 0
    elif "custom24" in name:
        rank = 1
    elif "mot24" in name:
        rank = 2
    elif path.suffix.lower() == ".json":
        rank = 3
    else:
        rank = 4
    return rank, str(path)


def _path_for_payload(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.relative_to(dataset_root))
    except ValueError:
        return str(path)


def compare_mevdt_tracker_sequence(
    dataset_root: str | Path,
    event_csv: str | Path | None = None,
    label_file: str | Path | None = None,
    config: TrackerComparisonConfig | None = None,
    window_filter: WindowFilterConfig | None = None,
) -> dict:
    """Load one MEVDT sequence and run the tracker comparison."""
    dataset_root = Path(dataset_root)
    selected_event, selected_label = select_mevdt_event_and_label_files(
        dataset_root,
        event_csv=event_csv,
        label_file=label_file,
    )
    labels = read_tracking_labels(selected_label)
    if not labels:
        raise ValueError(f"No tracking labels parsed from {selected_label}")
    events = read_event_csv(selected_event)
    comparison = compare_trackers_on_labels(
        labels,
        events,
        config=config,
        window_filter=window_filter,
    )
    return {
        "dataset": {
            "name": "MEVDT",
            "url": MEVDT_DATASET_URL,
            "doi": MEVDT_DOI,
            "dataset_root": str(dataset_root),
            "event_csv": _path_for_payload(selected_event, dataset_root),
            "label_file": _path_for_payload(selected_label, dataset_root),
            "association": "label-assisted",
        },
        **comparison,
    }
