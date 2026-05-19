import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eventvot_confidence_gate_replay.py"
SCRIPTS_DIR = SCRIPT_PATH.parent


def _load_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "run_eventvot_confidence_gate_replay_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_replay_fixture(root: Path, module) -> tuple[Path, Path, Path]:
    base_results = root / "base_results"
    base_results.mkdir()
    base_file = base_results / "recording_0001.txt"
    base_file.write_text(
        "8\t8\t10\t10\n9\t8\t10\t10\n10\t8\t10\t10\n",
        encoding="utf-8",
    )
    diagnostics_json = root / "diagnostics.json"
    diagnostics = {
        "acceptance_config": asdict(module.ReplayAcceptanceConfig()),
        "refiner_config": {"image_width": 1280.0, "image_height": 720.0},
        "options": {"split": "test"},
        "sequences": [
            {
                "sequence": "recording_0001",
                "base_result_file": str(base_file),
                "frames": [
                    _frame(0, [8.0, 8.0, 10.0, 10.0], fallback_reason="initial_frame"),
                    _frame(1, [9.25, 8.0, 10.0, 10.0]),
                    _frame(2, [40.0, 8.0, 10.0, 10.0]),
                ],
            }
        ],
    }
    diagnostics_json.write_text(json.dumps(diagnostics), encoding="utf-8")
    return diagnostics_json, base_results, root / "gated_results"


def _frame(index: int, xywh, *, fallback_reason=None):
    return {
        "frame_index": index,
        "fallback_reason": fallback_reason,
        "used_event_count": 12,
        "active_measurement_count": 3,
        "mean_event_activity": 0.20,
        "mean_event_polarity_weight": None,
        "polarity_consistency_fraction": None,
        "quadratic_form": None,
        "refiner_output_xywh": list(xywh),
        "refined_bbox": {
            "x_min": xywh[0],
            "y_min": xywh[1],
            "width": xywh[2],
            "height": xywh[3],
        },
    }


def test_confidence_gate_replay_holds_memory_on_geometry_rejection(
    tmp_path,
    monkeypatch,
):
    module = _load_module(monkeypatch)
    diagnostics_json, base_results, output_results = _write_replay_fixture(
        tmp_path,
        module,
    )

    payload = module.run(
        module.EventVOTConfidenceGateReplayOptions(
            diagnostics_json=diagnostics_json,
            base_results=base_results,
            output_results=output_results,
            skip_evaluation=True,
            gate_config=module.ReplayConfidenceGateConfig(motion_model="hold"),
        )
    )

    gated = np.loadtxt(output_results / "recording_0001.txt")
    np.testing.assert_allclose(
        gated,
        np.array(
            [
                [8.0, 8.0, 10.0, 10.0],
                [9.25, 8.0, 10.0, 10.0],
                [9.25, 8.0, 10.0, 10.0],
            ]
        ),
    )
    assert payload["summary"]["accepted_refinement_count"] == 1
    assert payload["summary"]["memory_gate_count"] == 1
    assert payload["summary"]["confidence_gate_action_counts"]["memory_hold"] == 1
    sequence = payload["sequences"][0]
    assert sequence["confidence_gate_action_counts"]["accepted_refinement"] == 1


def test_confidence_only_keeps_base_on_confident_frames(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    diagnostics_json, base_results, output_results = _write_replay_fixture(
        tmp_path,
        module,
    )

    payload = module.run(
        module.EventVOTConfidenceGateReplayOptions(
            diagnostics_json=diagnostics_json,
            base_results=base_results,
            output_results=output_results,
            skip_evaluation=True,
            gate_config=module.ReplayConfidenceGateConfig(
                apply_box_refinement=False,
                motion_model="hold",
            ),
        )
    )

    gated = np.loadtxt(output_results / "recording_0001.txt")
    np.testing.assert_allclose(
        gated,
        np.array(
            [
                [8.0, 8.0, 10.0, 10.0],
                [9.0, 8.0, 10.0, 10.0],
                [9.0, 8.0, 10.0, 10.0],
            ]
        ),
    )
    assert payload["summary"]["accepted_refinement_count"] == 0
    assert payload["summary"]["dvs_confident_frame_count"] == 1
    assert payload["summary"]["memory_gate_count"] == 1


def test_confidence_gate_replay_limits_consecutive_memory_frames(
    tmp_path,
    monkeypatch,
):
    module = _load_module(monkeypatch)
    diagnostics_json, base_results, output_results = _write_replay_fixture(
        tmp_path,
        module,
    )
    diagnostics = json.loads(diagnostics_json.read_text(encoding="utf-8"))
    diagnostics["sequences"][0]["frames"][1] = _frame(1, [40.0, 8.0, 10.0, 10.0])
    diagnostics["sequences"][0]["frames"][2] = _frame(2, [40.0, 8.0, 10.0, 10.0])
    diagnostics_json.write_text(json.dumps(diagnostics), encoding="utf-8")

    payload = module.run(
        module.EventVOTConfidenceGateReplayOptions(
            diagnostics_json=diagnostics_json,
            base_results=base_results,
            output_results=output_results,
            skip_evaluation=True,
            gate_config=module.ReplayConfidenceGateConfig(
                max_consecutive_memory_frames=1,
                motion_model="hold",
            ),
        )
    )

    gated = np.loadtxt(output_results / "recording_0001.txt")
    np.testing.assert_allclose(
        gated,
        np.array(
            [
                [8.0, 8.0, 10.0, 10.0],
                [8.0, 8.0, 10.0, 10.0],
                [10.0, 8.0, 10.0, 10.0],
            ]
        ),
    )
    assert payload["summary"]["memory_gate_count"] == 1
    assert payload["summary"]["confidence_gate_action_counts"]["base_passthrough"] == 1


def test_confidence_gate_replay_help_mentions_gate_options(monkeypatch):
    module = _load_module(monkeypatch)
    help_text = module.build_parser().format_help()

    assert "confidence/memory" in help_text
    assert "--confidence-only" in help_text
    assert "--min-event-support-score" in help_text
    assert "--gate-motion-model" in help_text
    assert "--gate-rejection-reason" in help_text
    assert "--max-consecutive-memory-frames" in help_text
