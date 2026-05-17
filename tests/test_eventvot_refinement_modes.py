import importlib.util
import sys
from pathlib import Path

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
            refinement_mode="size-only",
        )


def test_help_exposes_refinement_mode(monkeypatch):
    module = _load_module(monkeypatch)

    help_text = module.build_parser().format_help()

    assert "--refinement-mode" in help_text
    assert "center-only" in help_text
