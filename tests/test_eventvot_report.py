import importlib.util
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


def _write_sequence(sequence_dir: Path, box_line: str) -> None:
    sequence_dir.mkdir(parents=True)
    (sequence_dir / "groundtruth.txt").write_text(
        f"1 1 10 10\n{box_line}\n",
        encoding="utf-8",
    )
    (sequence_dir / "absent.txt").write_text("1\n1\n", encoding="utf-8")


def _write_result(result_dir: Path, sequence_name: str, box_line: str) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / f"{sequence_name}.txt").write_text(
        f"1 1 10 10\n{box_line}\n",
        encoding="utf-8",
    )


def _write_report_fixture(root: Path) -> tuple[Path, Path, Path]:
    split_root = root / "test"
    (split_root / "list.txt").parent.mkdir(parents=True)
    (split_root / "list.txt").write_text("seq_fast\nseq_scale\n", encoding="utf-8")
    _write_sequence(split_root / "seq_fast", "1 1 10 10")
    _write_sequence(split_root / "seq_scale", "1 1 10 10")

    result_root = root / "eventvot_tracking_results"
    hde_base = result_root / "HDETrackV2_tracking_result"
    hde_refined = result_root / "HDETrackV2_DVSENACT_tracking_result"
    for sequence_name in ("seq_fast", "seq_scale"):
        _write_result(hde_base, sequence_name, "4 1 10 10")
        _write_result(hde_refined, sequence_name, "1 1 10 10")
    (hde_refined / "seq_fast_time.txt").write_text("0\n0.01\n", encoding="utf-8")
    (hde_refined / "seq_scale_time.txt").write_text("0\n0.01\n", encoding="utf-8")

    attribute_root = root / "annos" / "att"
    attribute_root.mkdir(parents=True)
    fast_attrs = ["0"] * 14
    scale_attrs = ["0"] * 14
    fast_attrs[7] = "1"
    fast_attrs[8] = "1"
    scale_attrs[8] = "1"
    scale_attrs[12] = "1"
    (attribute_root / "seq_fast.txt").write_text("\n".join(fast_attrs) + "\n")
    (attribute_root / "seq_scale.txt").write_text("\n".join(scale_attrs) + "\n")
    return split_root, result_root, attribute_root


def test_eventvot_report_writes_table_and_attribute_gains(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    _split_root, result_root, attribute_root = _write_report_fixture(tmp_path)
    output_root = tmp_path / "report"
    args = module.build_parser().parse_args(
        [
            "--eventvot-root",
            str(tmp_path),
            "--result-root",
            str(result_root),
            "--output-root",
            str(output_root),
            "--tracker",
            f"HDETrackV2={result_root / 'HDETrackV2_tracking_result'}",
            "--tracker",
            "HDETrackV2 + DVS-ENACT="
            f"{result_root / 'HDETrackV2_DVSENACT_tracking_result'}",
            "--pair",
            "HDETrackV2=HDETrackV2 + DVS-ENACT",
            "--attribute-root",
            str(attribute_root),
            "--min-attribute-sequences",
            "1",
            "--fps",
            "HDETrackV2=100",
        ]
    )

    payload = module.run_report(args)

    assert payload["summary"]["tracker_count"] == 2
    assert output_root.joinpath("eventvot_paper_table.md").exists()
    assert payload["table"][0]["tracker"] == "HDETrackV2"
    assert payload["table"][0]["fps"] == 100.0
    assert payload["table"][1]["tracker"] == "HDETrackV2 + DVS-ENACT"
    assert payload["table"][1]["fps"] > 0.0
    assert payload["pairwise_gains"][0]["sr_delta"] > 0.0

    attribute_names = {row["attribute"] for row in payload["attribute_gains"]}
    assert "Fast Motion" in attribute_names
    assert "Background Clutter" in attribute_names
    assert payload["claim_support"]["hde_track_v2_overall_sr_improved"]
    assert payload["claim_support"]["positive_highlight_attribute_count"] >= 1


def test_eventvot_report_help_runs(monkeypatch):
    module = _load_module(monkeypatch)
    parser = module.build_parser()

    help_text = parser.format_help()

    assert "EventVOT paper comparison table" in help_text
    assert "--tracker" in help_text
    assert "--pair" in help_text
    assert "--attribute-root" in help_text
