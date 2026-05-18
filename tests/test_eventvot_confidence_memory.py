import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eventvot_confidence_memory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_eventvot_confidence_memory_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_base_results(root: Path) -> Path:
    base_results = root / "base_results"
    base_results.mkdir()
    (base_results / "recording_0001.txt").write_text(
        "\n".join(
            [
                "8\t10\t10\t10",
                "10\t10\t10\t10",
                "12\t10\t10\t10",
                "40\t10\t10\t10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return base_results


def _frame(frame_index, base_xywh, refiner_xywh, **overrides):
    frame = {
        "frame_index": frame_index,
        "fallback_reason": None,
        "used_event_count": 60,
        "active_measurement_count": 8,
        "mean_event_activity": 0.40,
        "refiner_output_xywh": list(refiner_xywh),
        "output_xywh": list(refiner_xywh),
        "refined_bbox": {
            "x_min": refiner_xywh[0],
            "y_min": refiner_xywh[1],
            "width": refiner_xywh[2],
            "height": refiner_xywh[3],
        },
        "candidate_iou": 1.0,
        "candidate_area_ratio": 1.0,
        "center_shift_ratio": 0.0,
        "output_bbox": {
            "x_min": refiner_xywh[0],
            "y_min": refiner_xywh[1],
            "width": refiner_xywh[2],
            "height": refiner_xywh[3],
        },
        "candidate_bbox": {
            "x_min": base_xywh[0],
            "y_min": base_xywh[1],
            "width": base_xywh[2],
            "height": base_xywh[3],
        },
    }
    frame.update(overrides)
    return frame


def _write_diagnostics(root: Path, base_results: Path) -> Path:
    diagnostics = {
        "schema_version": 1,
        "options": {"split": "test"},
        "refiner_config": {"image_width": 1280, "image_height": 720},
        "sequences": [
            {
                "sequence": "recording_0001",
                "base_result_file": str(base_results / "recording_0001.txt"),
                "frames": [
                    _frame(0, [8.0, 10.0, 10.0, 10.0], [8.0, 10.0, 10.0, 10.0]),
                    _frame(1, [10.0, 10.0, 10.0, 10.0], [11.0, 10.0, 10.0, 10.0]),
                    _frame(
                        2,
                        [12.0, 10.0, 10.0, 10.0],
                        [12.0, 10.0, 10.0, 10.0],
                        fallback_reason="low_event_count",
                        used_event_count=1,
                        active_measurement_count=0,
                        mean_event_activity=0.0,
                    ),
                    _frame(
                        3,
                        [40.0, 10.0, 10.0, 10.0],
                        [40.0, 10.0, 10.0, 10.0],
                        fallback_reason="low_event_count",
                        used_event_count=1,
                        active_measurement_count=0,
                        mean_event_activity=0.0,
                    ),
                ],
            }
        ],
    }
    diagnostics_json = root / "diagnostics.json"
    diagnostics_json.write_text(json.dumps(diagnostics), encoding="utf-8")
    return diagnostics_json


def test_eventvot_confidence_memory_trusts_direct_update_then_reuses_memory(tmp_path):
    module = _load_module()
    base_results = _write_base_results(tmp_path)
    diagnostics_json = _write_diagnostics(tmp_path, base_results)
    output_results = tmp_path / "confidence_memory_results"
    decisions_csv = tmp_path / "decisions.csv"

    payload = module.run(
        module.EventVOTConfidenceMemoryOptions(
            diagnostics_json=diagnostics_json,
            output_results=output_results,
            skip_evaluation=True,
            decisions_csv=decisions_csv,
        )
    )

    replayed = np.loadtxt(output_results / "recording_0001.txt")
    np.testing.assert_allclose(
        replayed,
        np.array(
            [
                [8.0, 10.0, 10.0, 10.0],
                [11.0, 10.0, 10.0, 10.0],
                [12.75, 10.0, 10.0, 10.0],
                [40.0, 10.0, 10.0, 10.0],
            ]
        ),
    )
    assert payload["summary"]["direct_trust_count"] == 1
    assert payload["summary"]["memory_count"] == 1
    assert payload["summary"]["base_fallback_count"] == 1
    assert payload["summary"]["action_counts"]["initial_frame"] == 1
    assert decisions_csv.exists()

    decisions = [
        row
        for row in decisions_csv.read_text(encoding="utf-8").splitlines()
        if "recording_0001" in row
    ]
    assert any(",direct," in row or "direct" in row for row in decisions)
    assert any(",memory," in row or "memory" in row for row in decisions)


def test_eventvot_confidence_memory_can_disable_carryover(tmp_path):
    module = _load_module()
    base_results = _write_base_results(tmp_path)
    diagnostics_json = _write_diagnostics(tmp_path, base_results)
    output_results = tmp_path / "confidence_memory_results"

    payload = module.run(
        module.EventVOTConfidenceMemoryOptions(
            diagnostics_json=diagnostics_json,
            output_results=output_results,
            skip_evaluation=True,
            config=module.ConfidenceMemoryConfig(max_memory_age=0),
        )
    )

    replayed = np.loadtxt(output_results / "recording_0001.txt")
    np.testing.assert_allclose(
        replayed,
        np.array(
            [
                [8.0, 10.0, 10.0, 10.0],
                [11.0, 10.0, 10.0, 10.0],
                [12.0, 10.0, 10.0, 10.0],
                [40.0, 10.0, 10.0, 10.0],
            ]
        ),
    )
    assert payload["summary"]["direct_trust_count"] == 1
    assert payload["summary"].get("memory_count", 0) == 0
    assert payload["summary"]["base_fallback_count"] == 2


def test_eventvot_confidence_memory_help_runs_as_script():
    help_text = subprocess.check_output(
        (sys.executable, str(SCRIPT_PATH), "--help"),
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )

    assert "confidence and short-term correction-memory signal" in help_text
    assert "--direct-confidence-threshold" in help_text
    assert "--max-memory-age" in help_text
    assert "--memory-decay" in help_text
    assert "--decisions-csv" in help_text
