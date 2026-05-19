import importlib.util
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "report_eventvot_comparisons.py"
SCRIPTS_DIR = SCRIPT_PATH.parent


def _load_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "report_eventvot_comparisons_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fps_defaults_to_not_reported_even_when_timing_files_exist(
    tmp_path,
    monkeypatch,
):
    module = _load_module(monkeypatch)
    result_dir = tmp_path / "Tracker_DVSENACT_tracking_result"
    result_dir.mkdir()
    (result_dir / "seq1_time.txt").write_text("0.10\n0.20\n", encoding="utf-8")
    spec = module.TrackerSpec("Tracker + DVS-ENACT", result_dir)

    fps, source, note = module.resolve_tracker_fps(
        spec,
        ["seq1"],
        {},
        use_timing_files=False,
    )

    assert fps is None
    assert source == "not_reported"
    assert "refinement-only" in note
    assert "--fps" in note


def test_fps_timing_file_estimate_is_opt_in_and_labeled_refinement_only(
    tmp_path,
    monkeypatch,
):
    module = _load_module(monkeypatch)
    result_dir = tmp_path / "Tracker_DVSENACT_tracking_result"
    result_dir.mkdir()
    (result_dir / "seq1_time.txt").write_text("0.10\n0.20\n", encoding="utf-8")
    spec = module.TrackerSpec("Tracker + DVS-ENACT", result_dir)

    fps, source, note = module.resolve_tracker_fps(
        spec,
        ["seq1"],
        {},
        use_timing_files=True,
    )

    assert math.isclose(fps, 2.0 / 0.30)
    assert source == "refinement_time_files"
    assert "refinement-only" in note


def test_fps_override_takes_precedence_over_timing_files(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    result_dir = tmp_path / "Tracker_DVSENACT_tracking_result"
    result_dir.mkdir()
    (result_dir / "seq1_time.txt").write_text("0.10\n0.20\n", encoding="utf-8")
    spec = module.TrackerSpec("Tracker + DVS-ENACT", result_dir)

    fps, source, note = module.resolve_tracker_fps(
        spec,
        ["seq1"],
        {"Tracker + DVS-ENACT": 12.5},
        use_timing_files=True,
    )

    assert fps == 12.5
    assert source == "override"
    assert "--fps" in note


def test_table_row_preserves_fps_provenance(monkeypatch):
    module = _load_module(monkeypatch)
    row = module.make_table_row(
        "Tracker + DVS-ENACT",
        {
            "overall": {
                "sr_auc": 0.1,
                "pr_auc": 0.2,
                "npr_auc": 0.3,
                "pr_20": 0.4,
                "npr_020": 0.5,
                "mean_iou": 0.6,
                "evaluated_frame_count": 7,
            },
            "fps": None,
            "fps_source": "not_reported",
            "fps_note": "not end-to-end",
        },
    )

    assert row["fps"] is None
    assert row["fps_source"] == "not_reported"
    assert row["fps_note"] == "not end-to-end"
