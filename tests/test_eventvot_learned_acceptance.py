import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eventvot_learned_acceptance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_eventvot_learned_acceptance_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_toy_validation_split(tmp_path):
    eventvot_root = tmp_path / "eventvot"
    sequence_dir = eventvot_root / "validating Subset" / "recording_0001"
    sequence_dir.mkdir(parents=True)
    (eventvot_root / "validating Subset" / "list.txt").write_text(
        "recording_0001\n",
        encoding="utf-8",
    )
    sequence_dir.joinpath("groundtruth.txt").write_text(
        "10\t10\t20\t20\n12\t10\t20\t20\n40\t40\t20\t20\n",
        encoding="utf-8",
    )
    # Mirrors the convention used by evaluate_eventvot_sequence(): 1 means
    # present after the official absent-vector inversion.
    sequence_dir.joinpath("absent.txt").write_text("1\n1\n1\n", encoding="utf-8")

    base_results = tmp_path / "base"
    base_results.mkdir()
    base_results.joinpath("recording_0001.txt").write_text(
        "10\t10\t20\t20\n30\t10\t20\t20\n40\t40\t20\t20\n",
        encoding="utf-8",
    )
    return eventvot_root, base_results


def _refinement_frame(frame_index, output_xywh, *, fallback=None, diagnostics=None):
    frame = {
        "frame_index": frame_index,
        "fallback_reason": fallback,
        "refiner_output_xywh": output_xywh,
    }
    if diagnostics:
        frame.update(diagnostics)
    if fallback is None:
        x, y, width, height = output_xywh
        frame["refined_bbox"] = {
            "x_min": x,
            "y_min": y,
            "x_max": x + width,
            "y_max": y + height,
        }
    return frame


def _write_toy_diagnostics(tmp_path, base_result_file):
    diagnostics_json = tmp_path / "diagnostics.json"
    confident_update = {
        "event_count": 500,
        "used_event_count": 128,
        "active_measurement_count": 96,
        "mean_event_activity": 0.85,
        "polarity_consistency_fraction": 0.95,
        "mean_event_polarity_weight": 0.90,
        "quadratic_form": 8.0,
    }
    weak_update = {
        "event_count": 20,
        "used_event_count": 8,
        "active_measurement_count": 1,
        "mean_event_activity": 0.05,
        "polarity_consistency_fraction": 0.10,
        "mean_event_polarity_weight": -0.50,
        "quadratic_form": 60.0,
    }
    sequence = {
        "sequence": "recording_0001",
        "base_result_file": str(base_result_file),
        "frames": [
            _refinement_frame(
                0,
                [10.0, 10.0, 20.0, 20.0],
                fallback="initial_frame",
            ),
            _refinement_frame(
                1,
                [12.0, 10.0, 20.0, 20.0],
                diagnostics=confident_update,
            ),
            _refinement_frame(
                2,
                [80.0, 80.0, 20.0, 20.0],
                diagnostics=weak_update,
            ),
        ],
    }
    diagnostics_json.write_text(
        json.dumps(
            {"options": {"split": "val"}, "sequences": [sequence]},
            indent=2,
        ),
        encoding="utf-8",
    )
    return diagnostics_json


def test_learned_acceptance_trains_and_replays_toy_validation_split(tmp_path):
    module = _load_module()
    eventvot_root, base_results = _write_toy_validation_split(tmp_path)
    diagnostics_json = _write_toy_diagnostics(
        tmp_path,
        base_results / "recording_0001.txt",
    )
    output_results = tmp_path / "learned_results"

    payload = module.run(
        module.LearnedAcceptanceOptions(
            diagnostics_json=diagnostics_json,
            output_results=output_results,
            eventvot_root=eventvot_root,
            base_results=base_results,
            split="val",
            skip_evaluation=True,
            epochs=400,
            threshold_count=9,
        )
    )

    replayed = np.loadtxt(output_results / "recording_0001.txt")
    np.testing.assert_allclose(
        replayed,
        np.array(
            [
                [10.0, 10.0, 20.0, 20.0],
                [12.0, 10.0, 20.0, 20.0],
                [40.0, 40.0, 20.0, 20.0],
            ]
        ),
    )
    assert payload["training_summary"]["positive_count"] == 1
    assert payload["training_summary"]["negative_count"] == 1
    assert payload["summary"]["accepted_refinement_count"] == 1
    assert payload["summary"]["acceptance_counts"] == {
        "accepted": 1,
        "initial_frame": 1,
        "learned_probability": 1,
    }
    assert (output_results / "learned_policy.json").exists()
    assert (output_results / "learned_acceptance_summary.json").exists()


def test_learned_acceptance_can_load_saved_policy(tmp_path):
    module = _load_module()
    eventvot_root, base_results = _write_toy_validation_split(tmp_path)
    diagnostics_json = _write_toy_diagnostics(
        tmp_path,
        base_results / "recording_0001.txt",
    )
    first_output = tmp_path / "first"
    first_payload = module.run(
        module.LearnedAcceptanceOptions(
            diagnostics_json=diagnostics_json,
            output_results=first_output,
            eventvot_root=eventvot_root,
            base_results=base_results,
            split="val",
            skip_evaluation=True,
            epochs=400,
            threshold_count=9,
        )
    )

    second_output = tmp_path / "second"
    second_payload = module.run(
        module.LearnedAcceptanceOptions(
            diagnostics_json=diagnostics_json,
            output_results=second_output,
            base_results=base_results,
            split="val",
            skip_evaluation=True,
            load_policy_json=first_output / "learned_policy.json",
        )
    )

    np.testing.assert_allclose(
        np.loadtxt(second_output / "recording_0001.txt"),
        np.loadtxt(first_output / "recording_0001.txt"),
    )
    assert second_payload["training_summary"] is None
    assert second_payload["summary"]["accepted_refinement_count"] == first_payload["summary"][
        "accepted_refinement_count"
    ]


def test_learned_acceptance_help_runs_as_script():
    help_text = subprocess.check_output(
        (sys.executable, str(SCRIPT_PATH), "--help"),
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )

    assert "validation-learned EventVOT" in help_text
    assert "--diagnostics-json" in help_text
    assert "--policy-json" in help_text
    assert "--load-policy-json" in help_text
    assert "--objective" in help_text
