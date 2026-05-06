import numpy as np
import numpy.testing as npt
import pytest

import pyrecest.backend
from dvs_enact import (
    BoundingBox,
    EventBatch,
    TrackerComparisonConfig,
    WindowFilterConfig,
    bbox_metrics,
    compare_trackers_on_labels,
    rectangle_radial_shape,
    subsample_events_chronologically,
    window_filter_reasons,
)


def test_rectangle_radial_shape_matches_axis_aligned_rectangle_axes():
    radial = rectangle_radial_shape(width=4.0, height=2.0, n_base_points=4)

    npt.assert_allclose(radial, np.array([2.0, 1.0, 2.0, 1.0]), atol=1e-12)


def test_subsample_events_chronologically_is_deterministic():
    events = EventBatch(
        ts=np.array([30, 10, 20, 40, 50], dtype=np.int64),
        x=np.array([3, 1, 2, 4, 5], dtype=np.int32),
        y=np.array([0, 0, 0, 0, 0], dtype=np.int32),
        p=np.array([1, 1, 1, 1, 1], dtype=np.int8),
    )

    sampled = subsample_events_chronologically(events, max_events=3)

    assert sampled.ts.tolist() == [10, 30, 50]
    assert sampled.x.tolist() == [1, 3, 5]


def test_bbox_metrics_uses_motion_inactive_axis():
    target = {
        "x_min": 0.0,
        "y_min": 0.0,
        "x_max": 10.0,
        "y_max": 10.0,
        "width": 10.0,
        "height": 10.0,
        "area": 100.0,
        "center_x": 5.0,
        "center_y": 5.0,
    }
    estimated = {
        "x_min": 0.0,
        "y_min": 2.0,
        "x_max": 10.0,
        "y_max": 8.0,
        "width": 10.0,
        "height": 6.0,
        "area": 60.0,
        "center_x": 5.0,
        "center_y": 5.0,
    }

    horizontal = bbox_metrics(estimated, target, velocity=[2.0, 0.0])
    vertical = bbox_metrics(estimated, target, velocity=[0.0, 2.0])

    assert horizontal["inactive_axis"] == "height"
    assert horizontal["inactive_axis_ratio"] == 0.6
    assert horizontal["collapsed"]
    assert vertical["inactive_axis"] == "width"
    assert vertical["inactive_axis_ratio"] == 1.0
    assert not vertical["collapsed"]


def test_window_filter_reasons_report_geometry_failures():
    current = BoundingBox(0, 1, 0.0, 10.0, 6.0, 20.0, timestamp_ns=0)
    following = BoundingBox(1, 1, 2.0, 10.0, 20.0, 40.0, timestamp_ns=10)
    config = WindowFilterConfig(
        min_width=8.0,
        min_height=8.0,
        min_area=80.0,
        max_width_change_fraction=0.25,
        max_height_change_fraction=0.25,
        trim_track_ends=1,
    )

    reasons = window_filter_reasons(
        current,
        following,
        track_window_index=0,
        track_window_count=5,
        config=config,
    )

    assert "track_end_trim" in reasons
    assert "border_touch" in reasons
    assert "small_box" in reasons
    assert "small_area" in reasons
    assert "width_change" in reasons
    assert "height_change" in reasons


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="MEVDT comparison fixture uses numpy tracker assertions",
)
def test_compare_trackers_on_labels_returns_valid_payload():
    labels = [
        BoundingBox(0, 1, 0.0, 0.0, 10.0, 10.0, timestamp_ns=0),
        BoundingBox(1, 1, 2.0, 0.0, 12.0, 10.0, timestamp_ns=10),
    ]
    events = EventBatch(
        ts=np.array([1, 2, 3, 4], dtype=np.int64),
        x=np.array([0, 10, 0, 10], dtype=np.int32),
        y=np.array([2, 2, 8, 8], dtype=np.int32),
        p=np.array([1, 1, 1, 1], dtype=np.int8),
    )
    config = TrackerComparisonConfig(
        n_base_points=8,
        max_events_per_window=4,
        max_windows=1,
        bbox_grid_points=32,
    )

    payload = compare_trackers_on_labels(labels, events, config=config)

    assert payload["parsed_sequence"]["label_count"] == 2
    assert payload["summary"]["windows_considered"] == 1
    assert payload["summary"]["windows_evaluated"] == 1
    assert "constant_position" in payload["summary"]
    assert "baseline_minus_constant_position" in payload["summary"]
    assert "dvs_enact_minus_constant_position" in payload["summary"]
    window = payload["windows"][0]
    assert window["used_event_count"] == 4
    assert window["constant_position"]["bbox"] == window["reference_bbox"]
    constant_metrics = window["constant_position"]["metrics"]
    assert constant_metrics["bbox_iou"] == pytest.approx(2.0 / 3.0)
    assert constant_metrics["center_error_px"] == pytest.approx(2.0)
    assert constant_metrics["inactive_axis_ratio"] == pytest.approx(1.0)
    assert "bbox_iou" in window["baseline"]["metrics"]
    assert "inactive_axis_ratio" in window["dvs_enact"]["metrics"]


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="MEVDT comparison fixture uses numpy tracker assertions",
)
def test_compare_trackers_on_labels_reports_filter_skips():
    labels = [
        BoundingBox(0, 1, 0.0, 0.0, 10.0, 10.0, timestamp_ns=0),
        BoundingBox(1, 1, 2.0, 0.0, 12.0, 10.0, timestamp_ns=10),
    ]
    events = EventBatch(
        ts=np.array([1, 2, 3, 4], dtype=np.int64),
        x=np.array([0, 10, 0, 10], dtype=np.int32),
        y=np.array([2, 2, 8, 8], dtype=np.int32),
        p=np.array([1, 1, 1, 1], dtype=np.int8),
    )
    config = TrackerComparisonConfig(n_base_points=8, max_windows=1)
    window_filter = WindowFilterConfig(border_margin_px=1.0, trim_track_ends=0)

    payload = compare_trackers_on_labels(
        labels,
        events,
        config=config,
        window_filter=window_filter,
    )

    assert payload["summary"]["windows_considered"] == 1
    assert payload["summary"]["windows_evaluated"] == 0
    assert payload["summary"]["skipped_filter_windows"] == 1
    assert payload["summary"]["filter_skip_reasons"]["border_touch"] == 1
