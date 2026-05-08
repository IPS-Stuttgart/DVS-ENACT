import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eventvot_refinement.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_eventvot_refinement_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_eventvot_fixture(root: Path) -> tuple[Path, Path, Path]:
    split_root = root / "test"
    sequence_dir = split_root / "recording_0001"
    image_dir = sequence_dir / "img"
    image_dir.mkdir(parents=True)
    for index in range(3):
        (image_dir / f"{index:04d}.png").write_bytes(b"")
    (split_root / "list.txt").write_text("recording_0001\n", encoding="utf-8")
    (sequence_dir / "recording_0001.csv").write_text(
        "\n".join(
            [
                "x,y,p,t",
                "10,10,1,0",
                "11,10,0,10",
                "12,11,1,20",
                "13,12,1,30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    base_results = root / "base_results"
    base_results.mkdir()
    (base_results / "recording_0001.txt").write_text(
        "8\t8\t10\t10\n9\t8\t10\t10\n10\t8\t10\t10\n",
        encoding="utf-8",
    )
    output_results = root / "refined_results"
    return split_root, base_results, output_results


def test_eventvot_refinement_writes_xywh_results_and_diagnostics(tmp_path):
    module = _load_module()
    _split_root, base_results, output_results = _write_eventvot_fixture(tmp_path)

    payload = module.run(
        module.EventVOTRefinementOptions(
            eventvot_root=tmp_path,
            base_results=base_results,
            output_results=output_results,
            split="test",
        ),
        refiner=module.DVSContourRefiner(
            module.DVSContourRefinerConfig(
                input_bbox_format="xywh",
                output_bbox_format="xywh",
                image_width=1280,
                image_height=720,
                min_events=99,
            )
        ),
    )

    refined = np.loadtxt(output_results / "recording_0001.txt")
    np.testing.assert_allclose(
        refined,
        np.array(
            [
                [8.0, 8.0, 10.0, 10.0],
                [9.0, 8.0, 10.0, 10.0],
                [10.0, 8.0, 10.0, 10.0],
            ]
        ),
    )
    assert (output_results / "recording_0001_time.txt").exists()
    assert (output_results / "eventvot_refinement_summary.json").exists()
    assert payload["summary"]["sequence_count"] == 1
    assert payload["summary"]["frame_count"] == 3
    assert payload["summary"]["fallback_counts"]["initial_frame"] == 1
    assert payload["summary"]["fallback_counts"]["low_event_count"] == 2


def test_eventvot_refinement_writes_official_tracker_result_layout(tmp_path):
    module = _load_module()
    _split_root, base_results, _output_results = _write_eventvot_fixture(tmp_path)
    toolkit_root = tmp_path / "EventVOT_eval_toolkit"
    config_tracker = toolkit_root / "utils" / "config_tracker.m"
    config_tracker.parent.mkdir(parents=True)
    config_tracker.write_text(
        "\n".join(
            [
                "function trackers = config_tracker()",
                "    trackers = {",
                "                    struct('name', 'Existing',           'publish', 'xxx');",
                "                      } ;",
                "end",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = module.run(
        module.EventVOTRefinementOptions(
            eventvot_root=tmp_path,
            base_results=base_results,
            output_results=toolkit_root,
            split="test",
            tracker_name="HDETrackV2_DVSENACT",
            config_tracker_path=config_tracker,
        ),
        refiner=module.DVSContourRefiner(
            module.DVSContourRefinerConfig(
                input_bbox_format="xywh",
                output_bbox_format="xywh",
                min_events=99,
            )
        ),
    )

    official_result = (
        toolkit_root
        / "eventvot_tracking_results"
        / "HDETrackV2_DVSENACT_tracking_result"
        / "recording_0001.txt"
    )
    assert official_result.exists()
    assert payload["eventvot_evaluator"]["tracker_name"] == "HDETrackV2_DVSENACT"
    assert payload["eventvot_evaluator"]["config_tracker_updated"]
    config_text = config_tracker.read_text(encoding="utf-8")
    assert "struct('name', 'HDETrackV2_DVSENACT'" in config_text

    updated_again = module.register_tracker_in_config(
        config_tracker,
        "HDETrackV2_DVSENACT",
    )
    assert not updated_again
    assert config_tracker.read_text(encoding="utf-8").count("HDETrackV2_DVSENACT") == 1


def test_eventvot_event_window_iterator_uses_between_frame_intervals(tmp_path):
    module = _load_module()
    _split_root, _base_results, _output_results = _write_eventvot_fixture(tmp_path)
    event_csv = tmp_path / "test" / "recording_0001" / "recording_0001.csv"

    windows = list(module.iter_eventvot_frame_windows(event_csv, 3))

    assert [frame_index for frame_index, _window in windows] == [1, 2]
    assert windows[0][1].ts.tolist() == [0, 10]
    assert windows[1][1].ts.tolist() == [20, 30]


def test_eventvot_refinement_help_runs_as_script():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Refine EventVOT xywh tracker result files" in result.stdout
    assert "--eventvot-root" in result.stdout
    assert "--tracker-name" in result.stdout
    assert "--event-column-order" in result.stdout
