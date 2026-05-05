"""Controlled synthetic tracker-level benchmarks for DVS-ENACT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .mevdt import BoundingBox, EventBatch
from .mevdt_comparison import (
    TrackerComparisonConfig,
    compare_trackers_on_labels,
)
from .synthetic import EDGE_ORDER

EDGE_NORMALS = {
    "left": np.array([-1.0, 0.0]),
    "right": np.array([1.0, 0.0]),
    "top": np.array([0.0, -1.0]),
    "bottom": np.array([0.0, 1.0]),
}


@dataclass(frozen=True)
class SyntheticRectangleSequenceConfig:
    """Parameters for a controlled moving-rectangle event sequence."""

    n_steps: int = 40
    width: float = 40.0
    height: float = 24.0
    start_center: tuple[float, float] = (80.0, 80.0)
    velocity: tuple[float, float] = (1.0, 0.0)
    events_per_window: int = 64
    background_activity: float = 0.0
    visible_edges: tuple[str, ...] | None = None
    measurement_noise_std: float = 0.0
    timestamp_step_ns: int = 1_000_000
    seed: int = 0
    track_id: int = 1


def normal_flow_edge_probabilities(
    velocity: Iterable[float],
    background_activity: float = 0.0,
    visible_edges: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Return edge probabilities from rectangle normal-flow activity."""
    if background_activity < 0.0:
        raise ValueError("background_activity must be non-negative")
    if visible_edges is not None:
        invalid_edges = set(visible_edges) - set(EDGE_ORDER)
        if invalid_edges:
            raise ValueError(f"Unknown visible edge labels: {sorted(invalid_edges)}")
        if not visible_edges:
            raise ValueError("visible_edges must not be empty")
    velocity = np.asarray(list(velocity), dtype=float)
    velocity_norm = float(np.linalg.norm(velocity))
    if velocity_norm <= 1e-12:
        weights = np.ones(len(EDGE_ORDER), dtype=float)
    else:
        weights = np.array(
            [
                abs(float(EDGE_NORMALS[edge] @ velocity)) / velocity_norm
                + float(background_activity)
                for edge in EDGE_ORDER
            ],
            dtype=float,
        )
    if visible_edges is not None:
        visibility = np.array([edge in visible_edges for edge in EDGE_ORDER], dtype=float)
        weights = weights * visibility
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones(len(EDGE_ORDER), dtype=float)
        if visible_edges is not None:
            weights = weights * np.array(
                [edge in visible_edges for edge in EDGE_ORDER],
                dtype=float,
            )
    probabilities = weights / float(np.sum(weights))
    return {
        edge: float(probability)
        for edge, probability in zip(EDGE_ORDER, probabilities, strict=True)
    }


def synthetic_rectangle_labels(
    config: SyntheticRectangleSequenceConfig,
) -> list[BoundingBox]:
    """Return ground-truth labels for the controlled rectangle sequence."""
    if config.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if config.width <= 0.0 or config.height <= 0.0:
        raise ValueError("width and height must be positive")
    velocity = np.asarray(config.velocity, dtype=float)
    start_center = np.asarray(config.start_center, dtype=float)
    labels = []
    for step in range(config.n_steps + 1):
        center = start_center + step * velocity
        labels.append(
            BoundingBox(
                frame=step,
                track_id=config.track_id,
                x_min=float(center[0] - 0.5 * config.width),
                y_min=float(center[1] - 0.5 * config.height),
                x_max=float(center[0] + 0.5 * config.width),
                y_max=float(center[1] + 0.5 * config.height),
                timestamp_ns=step * config.timestamp_step_ns,
                class_label="synthetic_rectangle",
            )
        )
    return labels


def _sample_edge_points(
    bbox: BoundingBox,
    edge: str,
    count: int,
    rng: np.random.Generator,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    if edge in {"left", "right"}:
        xs = np.full(count, bbox.x_min if edge == "left" else bbox.x_max, dtype=float)
        ys = rng.uniform(bbox.y_min, bbox.y_max, size=count)
    else:
        xs = rng.uniform(bbox.x_min, bbox.x_max, size=count)
        ys = np.full(count, bbox.y_min if edge == "top" else bbox.y_max, dtype=float)
    if noise_std > 0.0:
        xs = xs + rng.normal(0.0, noise_std, size=count)
        ys = ys + rng.normal(0.0, noise_std, size=count)
    return np.clip(xs, bbox.x_min, bbox.x_max), np.clip(ys, bbox.y_min, bbox.y_max)


def generate_synthetic_rectangle_events(
    labels: list[BoundingBox],
    config: SyntheticRectangleSequenceConfig,
) -> tuple[EventBatch, dict[str, int]]:
    """Generate contour events for consecutive synthetic rectangle labels."""
    if config.events_per_window <= 0:
        raise ValueError("events_per_window must be positive")
    if config.measurement_noise_std < 0.0:
        raise ValueError("measurement_noise_std must be non-negative")
    probabilities = normal_flow_edge_probabilities(
        config.velocity,
        background_activity=config.background_activity,
        visible_edges=config.visible_edges,
    )
    probability_vector = np.array([probabilities[edge] for edge in EDGE_ORDER])
    rng = np.random.default_rng(config.seed)
    timestamps: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    polarities: list[np.ndarray] = []
    edge_counts = {edge: 0 for edge in EDGE_ORDER}
    velocity = np.asarray(config.velocity, dtype=float)

    for current, following in zip(labels[:-1], labels[1:], strict=False):
        per_edge_counts = rng.multinomial(config.events_per_window, probability_vector)
        window_ts = np.linspace(
            current.timestamp_ns,
            following.timestamp_ns - 1,
            config.events_per_window,
            dtype=np.int64,
        )
        rng.shuffle(window_ts)
        cursor = 0
        for edge, count in zip(EDGE_ORDER, per_edge_counts, strict=True):
            count = int(count)
            edge_counts[edge] += count
            edge_xs, edge_ys = _sample_edge_points(
                current,
                edge,
                count,
                rng,
                config.measurement_noise_std,
            )
            edge_ts = window_ts[cursor : cursor + count]
            cursor += count
            polarity = 1 if float(EDGE_NORMALS[edge] @ velocity) >= 0.0 else 0
            timestamps.append(edge_ts)
            xs.append(np.rint(edge_xs).astype(np.int32))
            ys.append(np.rint(edge_ys).astype(np.int32))
            polarities.append(np.full(count, polarity, dtype=np.int8))

    if not timestamps:
        return (
            EventBatch(
                ts=np.array([], dtype=np.int64),
                x=np.array([], dtype=np.int32),
                y=np.array([], dtype=np.int32),
                p=np.array([], dtype=np.int8),
            ),
            edge_counts,
        )
    ts = np.concatenate(timestamps)
    order = np.argsort(ts, kind="stable")
    return (
        EventBatch(
            ts=ts[order],
            x=np.concatenate(xs)[order],
            y=np.concatenate(ys)[order],
            p=np.concatenate(polarities)[order],
        ),
        edge_counts,
    )


def generate_synthetic_rectangle_sequence(
    config: SyntheticRectangleSequenceConfig,
) -> tuple[list[BoundingBox], EventBatch, dict]:
    """Generate labels, events, and generation metadata for one sequence."""
    labels = synthetic_rectangle_labels(config)
    events, edge_counts = generate_synthetic_rectangle_events(labels, config)
    edge_probabilities = normal_flow_edge_probabilities(
        config.velocity,
        background_activity=config.background_activity,
        visible_edges=config.visible_edges,
    )
    return (
        labels,
        events,
        {
            "config": asdict(config),
            "edge_probabilities": edge_probabilities,
            "edge_counts": edge_counts,
        },
    )


def run_synthetic_tracker_benchmark(
    scenarios: dict[str, SyntheticRectangleSequenceConfig],
    tracker_config: TrackerComparisonConfig | None = None,
) -> dict:
    """Run vanilla-SCGP vs DVS-ENACT on controlled synthetic scenarios."""
    tracker_config = tracker_config or TrackerComparisonConfig(max_windows=None)
    scenario_results = []
    for name, sequence_config in scenarios.items():
        labels, events, generation = generate_synthetic_rectangle_sequence(
            sequence_config
        )
        comparison = compare_trackers_on_labels(
            labels,
            events,
            config=tracker_config,
        )
        scenario_results.append(
            {
                "scenario": name,
                "generation": generation,
                **comparison,
            }
        )
    return {
        "description": (
            "Controlled synthetic rectangle tracker benchmark with known extent "
            "and normal-flow contour event generation."
        ),
        "tracker_parameters": asdict(tracker_config),
        "scenarios": scenario_results,
    }
