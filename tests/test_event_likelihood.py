import unittest

import numpy as np

from dvs_enact.event_likelihood import (
    ContourSample,
    EventLikelihoodConfig,
    contour_event_intensity,
    event_batch_log_likelihood,
    expected_event_count,
    normal_flow_activities,
)


def _rectangle_contour(width: float, height: float) -> ContourSample:
    half_width = 0.5 * width
    half_height = 0.5 * height
    return ContourSample(
        points=np.array(
            [
                [-half_width, 0.0],
                [half_width, 0.0],
                [0.0, half_height],
                [0.0, -half_height],
            ],
            dtype=float,
        ),
        normals=np.array(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            dtype=float,
        ),
        weights=np.array([height, height, width, width], dtype=float),
    )


class TestEventLikelihood(unittest.TestCase):
    def test_horizontal_motion_activates_only_vertical_sides(self):
        contour = _rectangle_contour(width=4.0, height=2.0)

        activities = normal_flow_activities(contour.normals, np.array([1.0, 0.0]))

        np.testing.assert_allclose(activities, np.array([1.0, 1.0, 0.0, 0.0]))

    def test_active_edge_events_have_higher_intensity(self):
        contour = _rectangle_contour(width=4.0, height=2.0)
        config = EventLikelihoodConfig(
            spatial_sigma_px=0.4,
            foreground_rate=10.0,
            background_rate=1e-3,
        )

        intensities = contour_event_intensity(
            np.array(
                [
                    [2.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=float,
            ),
            contour,
            np.array([1.0, 0.0]),
            config,
        )

        self.assertGreater(intensities[0], 100.0 * intensities[1])

    def test_inactive_side_length_does_not_change_expected_count(self):
        config = EventLikelihoodConfig(foreground_rate=2.0, background_rate=0.0)
        narrow = _rectangle_contour(width=1.0, height=2.0)
        wide = _rectangle_contour(width=8.0, height=2.0)

        narrow_foreground, _ = expected_event_count(
            narrow,
            np.array([1.0, 0.0]),
            config,
        )
        wide_foreground, _ = expected_event_count(
            wide,
            np.array([1.0, 0.0]),
            config,
        )

        self.assertAlmostEqual(narrow_foreground, wide_foreground)

    def test_likelihood_prefers_correct_width_for_two_sided_support(self):
        config = EventLikelihoodConfig(
            spatial_sigma_px=0.5,
            foreground_rate=10.0,
            background_rate=1e-3,
            include_expected_count=False,
        )
        events = np.array(
            [
                [-2.0, -0.2],
                [-2.0, 0.2],
                [2.0, -0.2],
                [2.0, 0.2],
            ],
            dtype=float,
        )

        correct = event_batch_log_likelihood(
            events,
            _rectangle_contour(width=4.0, height=2.0),
            np.array([1.0, 0.0]),
            config,
        )
        collapsed = event_batch_log_likelihood(
            events,
            _rectangle_contour(width=1.0, height=2.0),
            np.array([1.0, 0.0]),
            config,
        )

        self.assertGreater(correct, collapsed)


if __name__ == "__main__":
    unittest.main()
