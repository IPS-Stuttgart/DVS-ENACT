import argparse
import importlib.util
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eventvot_validation_sweep.py"
SCRIPTS_DIR = SCRIPT_PATH.parent


def _load_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "run_eventvot_validation_sweep_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_validation_fixture(root: Path) -> tuple[Path, Path]:
    split_root = root / "validating Subset"
    sequence_dir = split_root / "seq1"
    sequence_dir.mkdir(parents=True)
    (split_root / "list.txt").write_text("seq1\n", encoding="utf-8")
    (sequence_dir / "groundtruth.txt").write_text(
        "1 1 10 10\n1 1 10 10\n",
        encoding="utf-8",
    )
    (sequence_dir / "absent.txt").write_text("1\n1\n", encoding="utf-8")
    result_root = root / "base_results"
    result_root.mkdir()
    (result_root / "seq1.txt").write_text(
        "1 1 10 10\n1 1 10 10\n",
        encoding="utf-8",
    )
    return split_root, result_root


def test_eventvot_validation_metrics_match_perfect_boxes(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    split_root, result_root = _write_validation_fixture(tmp_path)

    metrics = module.evaluate_eventvot_results(split_root, result_root, ["seq1"])

    assert math.isclose(metrics["sr_auc"], 20.0 / 21.0)
    assert metrics["pr_auc"] == 1.0
    assert metrics["pr_20"] == 1.0
    assert metrics["npr_auc"] == 1.0
    assert metrics["npr_020"] == 1.0


def test_validation_sweep_refuses_test_split(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    args = argparse.Namespace(
        split="test",
        allow_test_split=False,
        eventvot_root=tmp_path,
        base_results=tmp_path / "base",
        output_root=tmp_path / "out",
    )

    with pytest.raises(SystemExit, match="Refusing to tune"):
        module.run_sweep(args)


def test_validation_sweep_dry_run_uses_requested_grid(tmp_path, monkeypatch):
    module = _load_module(monkeypatch)
    _split_root, result_root = _write_validation_fixture(tmp_path)
    args = module.build_parser().parse_args(
        [
            "--eventvot-root",
            str(tmp_path),
            "--base-results",
            str(result_root),
            "--output-root",
            str(tmp_path / "out"),
            "--split",
            "val",
            "--max-configs",
            "2",
            "--dry-run",
        ]
    )

    payload = module.run_sweep(args)

    assert payload["summary"]["dry_run"]
    assert payload["summary"]["config_count"] == 2
    assert payload["summary"]["completed_config_count"] == 0
    assert (tmp_path / "out" / "validation_sweep_summary.json").exists()
