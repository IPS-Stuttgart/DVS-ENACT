import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eventvot_point_process_refinement.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_eventvot_point_process_refinement_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    split_root = root / "test"
    sequence_dir = split_root / "recording_0001"
    image_dir = sequence_dir / "img"
    image_dir.mkdir(parents=True)
    for index in range(2):
        (image_dir / f"{index:04d}.png").write_bytes(b"")
    (split_root / "list.txt").write_text("recording_0001\n", encoding="utf-8")
    (sequence_dir / "recording_0001.csv").write_text(
        "\n".join(
            [
                "19,10,1,0",
                "19,12,1,10",
                "19,14,1,20",
                "19,16,1,30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    base_results = root / "base_results"
    base_results.mkdir()
    (base_results / "recording_0001.txt").write_text(
        "8\t8\t10\t10\n9\t8\t10\t10\n",
        encoding="utf-8",
    )
    output_results = root / "refined_results"
    return split_root, base_results, output_results


def test_point_process_gate_rejects_lower_likelihood_refinement(tmp_path):
    module = _load_module()
    _split_root, base_results, output_results = _write_fixture(tmp_path)
    refiner = _FakeRefiner(module, [_FakeResult([11.0, 8.0, 10.0, 10.0])])

    payload = module.run(
        module.EventVOTPointProcessRefinementOptions(
            eventvot_root=tmp_path,
            base_results=base_results,
            output_results=output_results,
            split="test",
            point_process_gate=module.EventVOTPointProcessGateConfig(
                spatial_sigma_px=1.0,
                background_rate=1e-9,
                include_expected_count=False,
                samples_per_edge=8,
            ),
        ),
        refiner=refiner,
    )

    refined = np.loadtxt(output_results / "recording_0001.txt")
    np.testing.assert_allclose(
        refined,
        np.array(
            [
                [8.0, 8.0, 10.0, 10.0],
                [9.0, 8.0, 10.0, 10.0],
            ]
        ),
    )
    frame = payload["sequences"][0]["frames"][1]
    assert not frame["accept_refinement"]
    assert frame["rejection_reasons"] == ["point_process_log_likelihood"]
    assert frame["point_process_likelihood"]["delta_log_likelihood"] < 0.0
    assert payload["summary"]["acceptance_counts"]["point_process_log_likelihood"] == 1


def test_point_process_refinement_help_runs_as_script():
    help_text = subprocess.check_output(
        (sys.executable, str(SCRIPT_PATH), "--help"),
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )

    assert "point-process" in help_text
    assert "--point-process-min-delta-log-likelihood" in help_text
    assert "--point-process-samples-per-edge" in help_text
    assert "--disable-point-process-gate" in help_text


class _FakeRefiner:
    def __init__(self, module, results):
        self.config = module.default_eventvot_refiner().config
        self._results = list(results)

    def refine(self, _candidate, _events, *, previous_candidate_bbox=None):
        del previous_candidate_bbox
        return self._results.pop(0)


class _FakeResult:
    fallback_reason = None
    used_event_count = 12
    active_measurement_count = 3
    mean_event_activity = 0.20
    mean_event_polarity_weight = None
    polarity_consistency_fraction = None
    polarity_contrast_sign = None
    quadratic_form = None
    event_velocity = [1.0, 0.0]
    event_count = 4
    search_bbox = {
        "x_min": 6.5,
        "y_min": 5.5,
        "x_max": 21.5,
        "y_max": 20.5,
        "width": 15.0,
        "height": 15.0,
    }

    def __init__(self, xywh):
        self._xywh = tuple(float(value) for value in xywh)

    def as_xywh(self):
        return self._xywh

    @property
    def refined_bbox(self):
        x, y, width, height = self._xywh
        return {
            "x_min": x,
            "y_min": y,
            "width": width,
            "height": height,
        }

    @property
    def output_bbox(self):
        return self.refined_bbox

    def to_dict(self):
        payload = {"candidate_bbox": {}, "search_bbox": self.search_bbox}
        for key in (
            "fallback_reason",
            "refined_bbox",
            "output_bbox",
            "event_velocity",
            "event_count",
            "used_event_count",
            "active_measurement_count",
            "mean_event_activity",
            "mean_event_polarity_weight",
            "polarity_consistency_fraction",
            "polarity_contrast_sign",
            "quadratic_form",
        ):
            value = getattr(self, key)
            payload[key] = value() if callable(value) else value
        return payload
