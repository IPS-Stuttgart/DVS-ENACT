import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eventvot_refinement_modes.py"
BASE_SCRIPT_DIR = REPO_ROOT / "scripts"


def _load_module(monkeypatch):
    monkeypatch.syspath_prepend(str(BASE_SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "run_eventvot_refinement_modes_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_center_only_projection_keeps_candidate_size(monkeypatch):
    module = _load_module(monkeypatch)

    projected = module.project_refinement_output(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([20.0, 10.0, 50.0, 60.0]),
        refinement_mode="center-only",
        image_width=200.0,
        image_height=200.0,
    )

    np.testing.assert_allclose(projected, np.array([30.0, 20.0, 30.0, 40.0]))


def test_event_centroid_center_projection_uses_median_event_center(monkeypatch):
    module = _load_module(monkeypatch)
    candidate = np.array([10.0, 20.0, 30.0, 40.0])
    events = SimpleNamespace(
        count=3,
        x=np.array([30.0, 40.0, 1000.0]),
        y=np.array([50.0, 70.0, 1000.0]),
    )

    centroid_box = module.event_center_box(
        candidate,
        events,
        {"x_min": 0.0, "x_max": 100.0, "y_min": 0.0, "y_max": 100.0},
    )
    projected = module.project_refinement_output(
        candidate,
        np.array([99.0, 99.0, 10.0, 10.0]),
        refinement_mode="event-centroid-center",
        event_center_xywh=centroid_box,
    )

    np.testing.assert_allclose(projected, np.array([20.0, 40.0, 30.0, 40.0]))


def test_event_boundary_center_projection_ignores_interior_events(monkeypatch):
    module = _load_module(monkeypatch)
    candidate = np.array([10.0, 20.0, 30.0, 40.0])
    events = SimpleNamespace(
        count=5,
        x=np.array([12.0, 42.0, 26.0, 25.0, 1000.0]),
        y=np.array([38.0, 42.0, 22.0, 40.0, 1000.0]),
    )

    center_box = module.event_center_box(
        candidate,
        events,
        {"x_min": 0.0, "x_max": 100.0, "y_min": 0.0, "y_max": 100.0},
        boundary_weighted=True,
    )
    projected = module.project_refinement_output(
        candidate,
        np.array([99.0, 99.0, 10.0, 10.0]),
        refinement_mode="event-boundary-center",
        event_center_xywh=center_box,
    )

    np.testing.assert_allclose(projected, np.array([11.0, 18.0, 30.0, 40.0]))


def test_event_edge_center_projects_boundary_events_to_box_center(monkeypatch):
    module = _load_module(monkeypatch)
    candidate = np.array([10.0, 20.0, 30.0, 40.0])
    events = SimpleNamespace(
        count=4,
        x=np.array([41.0, 42.0, 26.0, 25.0]),
        y=np.array([44.0, 45.0, 62.0, 62.0]),
    )

    center_box = module.event_center_box(
        candidate,
        events,
        {"x_min": 0.0, "x_max": 100.0, "y_min": 0.0, "y_max": 100.0},
        boundary_weighted=True,
        edge_inferred=True,
    )
    projected = module.project_refinement_output(
        candidate,
        np.array([99.0, 99.0, 10.0, 10.0]),
        refinement_mode="event-edge-center",
        event_center_xywh=center_box,
    )

    np.testing.assert_allclose(projected, np.array([11.0, 22.0, 30.0, 40.0]))


def test_event_paired_edge_center_requires_opposite_edge_evidence(monkeypatch):
    module = _load_module(monkeypatch)
    candidate = np.array([10.0, 20.0, 30.0, 40.0])
    events = SimpleNamespace(
        count=3,
        x=np.array([41.0, 26.0, 25.0]),
        y=np.array([44.0, 22.0, 62.0]),
    )

    single_axis_box = module.event_center_box(
        candidate,
        events,
        {"x_min": 0.0, "x_max": 100.0, "y_min": 0.0, "y_max": 100.0},
        boundary_weighted=True,
        edge_inferred=True,
        require_paired_edges=True,
    )
    no_pair_box = module.event_center_box(
        candidate,
        SimpleNamespace(count=1, x=np.array([41.0]), y=np.array([44.0])),
        {"x_min": 0.0, "x_max": 100.0, "y_min": 0.0, "y_max": 100.0},
        boundary_weighted=True,
        edge_inferred=True,
        require_paired_edges=True,
    )
    projected = module.project_refinement_output(
        candidate,
        np.array([99.0, 99.0, 10.0, 10.0]),
        refinement_mode="event-paired-edge-center",
        event_center_xywh=single_axis_box,
    )

    np.testing.assert_allclose(projected, np.array([10.0, 22.0, 30.0, 40.0]))
    assert no_pair_box is None


def test_size_only_projection_keeps_candidate_center(monkeypatch):
    module = _load_module(monkeypatch)

    projected = module.project_refinement_output(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([20.0, 10.0, 50.0, 60.0]),
        refinement_mode="size-only",
        image_width=200.0,
        image_height=200.0,
    )

    np.testing.assert_allclose(projected, np.array([0.0, 10.0, 50.0, 60.0]))


def test_width_only_projection_keeps_candidate_center_and_height(monkeypatch):
    module = _load_module(monkeypatch)

    projected = module.project_refinement_output(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([20.0, 10.0, 50.0, 60.0]),
        refinement_mode="width-only",
        image_width=200.0,
        image_height=200.0,
    )

    np.testing.assert_allclose(projected, np.array([0.0, 20.0, 50.0, 40.0]))


def test_height_only_projection_keeps_candidate_center_and_width(monkeypatch):
    module = _load_module(monkeypatch)

    projected = module.project_refinement_output(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([20.0, 10.0, 50.0, 60.0]),
        refinement_mode="height-only",
        image_width=200.0,
        image_height=200.0,
    )

    np.testing.assert_allclose(projected, np.array([10.0, 10.0, 30.0, 60.0]))


def test_size_only_projection_supports_independent_size_blends(monkeypatch):
    module = _load_module(monkeypatch)

    projected = module.project_refinement_output(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([12.0, 22.0, 31.0, 41.0]),
        refinement_mode="size-only",
        raw_refined_xywh=np.array([20.0, 10.0, 50.0, 60.0]),
        projection_width_blend=0.25,
        projection_height_blend=0.50,
        image_width=200.0,
        image_height=200.0,
    )

    np.testing.assert_allclose(projected, np.array([7.5, 15.0, 35.0, 50.0]))


def test_size_only_projection_smooths_previous_accepted_size(monkeypatch):
    module = _load_module(monkeypatch)

    projected = module.project_refinement_output(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([0.0, 10.0, 50.0, 60.0]),
        refinement_mode="size-only",
        previous_projected_size=np.array([20.0, 20.0]),
        projection_size_smoothing=0.5,
        image_width=200.0,
        image_height=200.0,
    )

    np.testing.assert_allclose(projected, np.array([7.5, 20.0, 35.0, 40.0]))


def test_width_only_projection_smooths_only_width(monkeypatch):
    module = _load_module(monkeypatch)

    projected = module.project_refinement_output(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([0.0, 10.0, 50.0, 60.0]),
        refinement_mode="width-only",
        previous_projected_size=np.array([20.0, 80.0]),
        projection_size_smoothing=0.5,
        image_width=200.0,
        image_height=200.0,
    )

    np.testing.assert_allclose(projected, np.array([7.5, 20.0, 35.0, 40.0]))


def test_size_deadband_ignores_tiny_axis_changes(monkeypatch):
    module = _load_module(monkeypatch)

    projected = module.project_refinement_output(
        np.array([10.0, 20.0, 100.0, 50.0]),
        np.array([9.0, 15.0, 102.0, 60.0]),
        refinement_mode="size-only",
        projection_size_deadband_ratio=0.05,
    )

    np.testing.assert_allclose(projected, np.array([10.0, 15.0, 100.0, 60.0]))


def test_box_size_deadband_preserves_projected_center(monkeypatch):
    module = _load_module(monkeypatch)

    projected = module.project_refinement_output(
        np.array([10.0, 20.0, 100.0, 50.0]),
        np.array([20.0, 30.0, 102.0, 51.0]),
        refinement_mode="box",
        projection_size_deadband_ratio=0.05,
    )

    np.testing.assert_allclose(projected, np.array([21.0, 30.5, 100.0, 50.0]))


def test_box_projection_smooths_size_around_projected_center(monkeypatch):
    module = _load_module(monkeypatch)

    projected = module.project_refinement_output(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([20.0, 10.0, 50.0, 60.0]),
        refinement_mode="box",
        previous_projected_size=np.array([30.0, 40.0]),
        projection_size_smoothing=0.5,
    )

    np.testing.assert_allclose(projected, np.array([25.0, 15.0, 40.0, 50.0]))


def test_projection_confidence_weighting_shrinks_update(monkeypatch):
    module = _load_module(monkeypatch)

    projected = module.project_refinement_output(
        np.array([10.0, 10.0, 20.0, 20.0]),
        np.array([20.0, 10.0, 20.0, 20.0]),
        refinement_mode="box",
        projection_confidence_value=0.25,
        projection_confidence_floor=0.0,
        projection_confidence_ceiling=0.5,
    )

    np.testing.assert_allclose(projected, np.array([15.0, 10.0, 20.0, 20.0]))


def test_projection_confidence_weighting_missing_value_keeps_candidate(monkeypatch):
    module = _load_module(monkeypatch)

    projected = module.project_refinement_output(
        np.array([10.0, 10.0, 20.0, 20.0]),
        np.array([20.0, 10.0, 20.0, 20.0]),
        refinement_mode="box",
        projection_confidence_value=None,
        projection_confidence_floor=0.0,
        projection_confidence_ceiling=0.5,
    )

    np.testing.assert_allclose(projected, np.array([10.0, 10.0, 20.0, 20.0]))


def test_box_projection_preserves_refiner_output(monkeypatch):
    module = _load_module(monkeypatch)

    projected = module.project_refinement_output(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([20.0, 10.0, 50.0, 60.0]),
        refinement_mode="box",
    )

    np.testing.assert_allclose(projected, np.array([20.0, 10.0, 50.0, 60.0]))


def test_center_only_projection_clips_to_image(monkeypatch):
    module = _load_module(monkeypatch)

    projected = module.project_refinement_output(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([-20.0, -20.0, 10.0, 10.0]),
        refinement_mode="center-only",
        image_width=100.0,
        image_height=100.0,
    )

    assert np.all(projected[:2] >= 0.0)
    assert projected[0] + projected[2] <= 100.0
    assert projected[1] + projected[3] <= 100.0


def test_refinement_mode_validation_rejects_unknown_mode(monkeypatch):
    module = _load_module(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported refinement mode"):
        module.project_refinement_output(
            np.array([0.0, 0.0, 10.0, 10.0]),
            np.array([0.0, 0.0, 10.0, 10.0]),
            refinement_mode="diagonal-only",
        )


def test_projection_blends_must_be_supplied_together(monkeypatch):
    module = _load_module(monkeypatch)

    with pytest.raises(ValueError, match="supplied together"):
        module.project_refinement_output(
            np.array([0.0, 0.0, 10.0, 10.0]),
            np.array([0.0, 0.0, 10.0, 10.0]),
            refinement_mode="size-only",
            projection_width_blend=0.25,
        )


def test_projection_policy_rejects_clipped_outputs(monkeypatch):
    module = _load_module(monkeypatch)

    reasons = module.projection_rejection_reasons(
        np.array([10.0, 10.0, 20.0, 20.0]),
        np.array([10.0, 10.0, 20.0, 20.0]),
        np.array([-1.0, 10.0, 20.0, 20.0]),
        image_width=100.0,
        image_height=100.0,
        projection_no_clip=True,
    )

    assert reasons == ("projection_clip",)


def test_projection_policy_rejects_raw_height_shrink(monkeypatch):
    module = _load_module(monkeypatch)

    reasons = module.projection_rejection_reasons(
        np.array([10.0, 10.0, 20.0, 20.0]),
        np.array([10.0, 10.0, 20.0, 18.0]),
        np.array([10.0, 10.0, 20.0, 20.0]),
        projection_min_raw_height_ratio=1.0,
    )

    assert reasons == ("projection_raw_height_ratio",)


def test_help_exposes_refinement_mode(monkeypatch):
    module = _load_module(monkeypatch)

    help_text = module.build_parser().format_help()

    assert "--refinement-mode" in help_text
    assert "center-only" in help_text
    assert "event-boundary-center" in help_text
    assert "event-centroid-center" in help_text
    assert "event-edge-center" in help_text
    assert "event-paired-edge-center" in help_text
    assert "size-only" in help_text
    assert "width-only" in help_text
    assert "height-only" in help_text
    assert "--projection-width-blend" in help_text
    assert "--projection-height-blend" in help_text
    assert "--projection-no-clip" in help_text
    assert "--projection-size-smoothing" in help_text
    assert "--projection-size-deadband-ratio" in help_text
    assert "--projection-confidence-field" in help_text
    assert "--projection-min-raw-height-ratio" in help_text
