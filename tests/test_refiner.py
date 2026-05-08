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
refiner_expand_bbox = dvs_enact.refiner_expand_bbox


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
        self.assertEqual(expanded["width"], 55.0)
        self.assertEqual(expanded["height"], 80.0)

    def test_crop_events_to_candidate_search_region(self):
        events = self._events()

        cropped = refiner_crop_events_to_bbox(events, (8.0, 8.0, 16.0, 16.0))

        self.assertEqual(cropped.count, 3)
        np.testing.assert_array_equal(cropped.x, np.array([9, 10, 15]))

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
