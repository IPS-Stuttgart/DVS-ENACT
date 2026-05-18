import numpy as np

from dvs_enact.event_likelihood import EventLikelihoodConfig
from dvs_enact.mevdt import EventBatch
from dvs_enact.refinement_likelihood import (
    BBoxEventLikelihoodConfig,
    bbox_contour_sample,
    compare_refinement_likelihood,
)


def test_bbox_contour_sample_tracks_rectangle_perimeter():
    contour = bbox_contour_sample(
        [10.0, 20.0, 30.0, 10.0],
        bbox_format="xywh",
        samples_per_edge=4,
    )

    assert contour.points.shape == (16, 2)
    assert contour.normals.shape == (16, 2)
    assert contour.weights.shape == (16,)
    assert np.isclose(np.sum(contour.weights), 80.0)


def test_refined_box_with_boundary_events_scores_higher():
    events = EventBatch(
        ts=np.arange(4, dtype=np.int64),
        x=np.full(4, 30, dtype=np.int32),
        y=np.array([4, 8, 12, 16], dtype=np.int32),
        p=np.ones(4, dtype=np.int8),
    )
    config = BBoxEventLikelihoodConfig(
        likelihood=EventLikelihoodConfig(
            spatial_sigma_px=1.0,
            background_rate=1e-9,
            activity_floor=0.05,
            include_expected_count=False,
        ),
        samples_per_edge=8,
    )

    comparison = compare_refinement_likelihood(
        [0.0, 0.0, 20.0, 20.0],
        [10.0, 0.0, 20.0, 20.0],
        events,
        [1.0, 0.0],
        config,
        bbox_format="xywh",
    )

    assert comparison.delta_log_likelihood > 0.0
    assert comparison.delta_log_likelihood_per_event is not None
    assert comparison.refined_is_better
    assert comparison.to_dict()["refined"]["terms"]["event_count"] == 4
