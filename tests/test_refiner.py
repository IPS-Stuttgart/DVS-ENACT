import unittest

import numpy as np
import pytest

# pylint: disable=no-name-in-module,no-member
pyrecest_backend = pytest.importorskip("pyrecest.backend")
dvs_enact = pytest.importorskip("dvs_enact")

DVSContourRefiner = dvs_enact.DVSContourRefiner
DVSContourRefinerConfig = dvs_enact.DVSContourRefinerConfig
EventBatch = dvs_enact.EventBatch
refiner_bbox_to_dict = dvs_enact.refiner_bbox_to_dict
refiner_crop_events_to_bbox = dvs_enact.refiner_crop_events_to_bbox
refiner_event_distance_to_bbox_boundary = (
    dvs_enact.refiner_event_distance_to_bbox_boundary
)
refiner_expand_bbox = dvs_enact.refiner_expand_bbox
refiner_select_refinement_events = dvs_enact.refiner_select_refinement_events


@unittest.skipIf(
    pyrecest_backend.__backend_name__ != "numpy",
    reason="DVS contour refiner tests currently use NumPy event batches",
)
class TestDVSContourRefiner(unittest.TestCase):
    def _events(self):
        return EventBatch(
            ts=np.array([0, 1, 2, 3], dtype=np.int64),
            x=np.array([9, 10, 15, 21], dtype=np.int32),
            y=np.array([9, 10, 15, 20], dtype=np.int32),
            p=np.array([1, 0, 1, 0], dtype=np.int8),
        )

    def test_bbox_helpers_support_xywh_and_expansion(self):
        bbox = refiner_bbox_to_dict((20.0, 30.0, 30.0, 40.0), bbox_format="xywh")

        self.assertEqual(bbox["x_min"], 20.0)
        self.assertEqual(bbox["x_max"], 50.0)
        self.assertEqual(bbox["width"], 30.0)

        expanded = refiner_expand_bbox(bbox, 2.0, image_width=100.0, image_height=100.0)
        self.assertEqual(expanded["width"], 60.0)
        self.assertEqual(expanded["height"], 80.0)

    def test_crop_events_to_candidate_search_region(self):
        events = self._events()

        cropped = refiner_crop_events_to_bbox(events, (8.0, 8.0, 16.0, 16.0))

        self.assertEqual(cropped.count, 3)
        np.testing.assert_array_equal(cropped.x, np.array([9, 10, 15]))

    def test_crop_events_uses_half_open_right_and_bottom_edges(self):
        events = EventBatch(
            ts=np.array([0, 1, 2, 3], dtype=np.int64),
            x=np.array([10, 19, 20, 15], dtype=np.int32),
            y=np.array([10, 15, 15, 20], dtype=np.int32),
            p=np.array([1, 1, 1, 1], dtype=np.int8),
        )

        cropped = refiner_crop_events_to_bbox(
            events,
            {"x": 10.0, "y": 10.0, "width": 10.0, "height": 10.0},
        )

        self.assertEqual(cropped.count, 2)
        np.testing.assert_array_equal(cropped.x, np.array([10, 19]))

    def test_boundary_event_selection_prefers_contour_events(self):
        events = EventBatch(
            ts=np.array([0, 1, 2, 3, 4], dtype=np.int64),
            x=np.array([10, 20, 30, 20, 50], dtype=np.int32),
            y=np.array([20, 20, 20, 30, 50], dtype=np.int32),
            p=np.array([1, 1, 0, 0, 1], dtype=np.int8),
        )

        selected = refiner_select_refinement_events(
            events,
            (10.0, 10.0, 30.0, 30.0),
            2,
            mode="boundary",
            angular_bins=4,
        )

        self.assertEqual(selected.count, 2)
        np.testing.assert_array_equal(selected.ts, np.array([0, 2]))
        np.testing.assert_array_equal(selected.x, np.array([10, 30]))

    def test_normal_flow_event_selection_prefers_motion_active_sides(self):
        events = EventBatch(
            ts=np.array([0, 1, 2, 3], dtype=np.int64),
            x=np.array([20, 20, 10, 30], dtype=np.int32),
            y=np.array([10, 30, 20, 20], dtype=np.int32),
            p=np.array([1, 1, 1, 1], dtype=np.int8),
        )

        selected = refiner_select_refinement_events(
            events,
            (10.0, 10.0, 30.0, 30.0),
            2,
            mode="normal_flow",
            angular_bins=1,
            event_velocity=np.array([1.0, 0.0]),
            use_event_polarity=False,
        )

        self.assertEqual(selected.count, 2)
        np.testing.assert_array_equal(selected.ts, np.array([2, 3]))
        np.testing.assert_array_equal(selected.x, np.array([10, 30]))

    def test_boundary_distance_handles_inside_and_outside_events(self):
        events = EventBatch(
            ts=np.array([0, 1, 2], dtype=np.int64),
            x=np.array([10, 20, 35], dtype=np.int32),
            y=np.array([20, 20, 20], dtype=np.int32),
            p=np.array([1, 1, 1], dtype=np.int8),
        )

        distances = refiner_event_distance_to_bbox_boundary(
            events,
            (10.0, 10.0, 30.0, 30.0),
        )

        np.testing.assert_allclose(distances, np.array([0.0, 10.0, 5.0]))

    def test_refiner_falls_back_without_motion(self):
        refiner = DVSContourRefiner(DVSContourRefinerConfig(min_events=1))

        result = refiner.refine((8.0, 8.0, 16.0, 16.0), self._events())

        self.assertEqual(result.fallback_reason, "low_event_velocity")
        self.assertEqual(result.as_xyxy(), (8.0, 8.0, 16.0, 16.0))

    def test_refiner_runs_with_previous_candidate_velocity(self):
        refiner = DVSContourRefiner(
            DVSContourRefinerConfig(
                min_events=1,
                search_expansion_factor=2.0,
                use_event_polarity=True,
            )
        )

        result = refiner.refine(
            (9.0, 8.0, 17.0, 16.0),
            self._events(),
            previous_candidate_bbox=(8.0, 8.0, 16.0, 16.0),
        )

        self.assertIsNone(result.fallback_reason)
        self.assertEqual(result.used_event_count, 3)
        self.assertEqual(len(result.as_xyxy()), 4)
        self.assertIsNotNone(result.mean_event_activity)


if __name__ == "__main__":
    unittest.main()
