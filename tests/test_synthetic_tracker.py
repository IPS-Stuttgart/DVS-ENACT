import numpy as np
import pytest

import pyrecest.backend
from dvs_enact import (
    SyntheticRectangleSequenceConfig,
    TrackerComparisonConfig,
    generate_synthetic_rectangle_sequence,
    normal_flow_edge_probabilities,
    run_synthetic_tracker_benchmark,
    synthetic_rectangle_labels,
)


def test_normal_flow_edge_probabilities_horizontal_side_only():
    probabilities = normal_flow_edge_probabilities([1.0, 0.0])

    assert probabilities["left"] == pytest.approx(0.5)
    assert probabilities["right"] == pytest.approx(0.5)
    assert probabilities["top"] == pytest.approx(0.0)
    assert probabilities["bottom"] == pytest.approx(0.0)


def test_normal_flow_edge_probabilities_respects_visible_edges():
    probabilities = normal_flow_edge_probabilities(
        [1.0, 0.0],
        visible_edges=("left",),
    )

    assert probabilities["left"] == pytest.approx(1.0)
    assert probabilities["right"] == pytest.approx(0.0)
    assert probabilities["top"] == pytest.approx(0.0)
    assert probabilities["bottom"] == pytest.approx(0.0)


def test_synthetic_rectangle_labels_have_constant_extent_and_velocity():
    config = SyntheticRectangleSequenceConfig(
        n_steps=2,
        width=8.0,
        height=4.0,
        start_center=(10.0, 20.0),
        velocity=(2.0, 1.0),
        timestamp_step_ns=100,
    )

    labels = synthetic_rectangle_labels(config)

    assert len(labels) == 3
    assert labels[0].width == 8.0
    assert labels[0].height == 4.0
    assert labels[1].center.tolist() == [12.0, 21.0]
    assert labels[2].timestamp_ns == 200


def test_generate_synthetic_rectangle_sequence_side_only_counts():
    config = SyntheticRectangleSequenceConfig(
        n_steps=3,
        velocity=(1.0, 0.0),
        events_per_window=20,
        background_activity=0.0,
        seed=3,
    )

    _labels, events, metadata = generate_synthetic_rectangle_sequence(config)

    assert events.count == 60
    assert metadata["edge_counts"]["left"] + metadata["edge_counts"]["right"] == 60
    assert metadata["edge_counts"]["top"] == 0
    assert metadata["edge_counts"]["bottom"] == 0


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Synthetic tracker benchmark uses numpy tracker assertions",
)
def test_run_synthetic_tracker_benchmark_returns_window_metrics():
    sequence_config = SyntheticRectangleSequenceConfig(
        n_steps=2,
        velocity=(1.0, 0.0),
        events_per_window=8,
        background_activity=0.0,
        seed=4,
    )
    tracker_config = TrackerComparisonConfig(
        n_base_points=8,
        max_events_per_window=8,
        max_windows=None,
        bbox_grid_points=32,
    )

    payload = run_synthetic_tracker_benchmark(
        {"tiny": sequence_config},
        tracker_config=tracker_config,
    )

    scenario = payload["scenarios"][0]
    assert scenario["scenario"] == "tiny"
    assert scenario["summary"]["windows_considered"] == 2
    assert scenario["summary"]["windows_evaluated"] == 2
    assert "inactive_axis_ratio" in scenario["windows"][0]["baseline"]["metrics"]
