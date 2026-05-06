import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_paper_artifacts.py"
SCRIPTS_DIR = SCRIPT_PATH.parent


def _load_exporter_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "export_paper_artifacts_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exporter_imports_stage_modules(monkeypatch):
    module = _load_exporter_module(monkeypatch)

    parser = module.build_parser()
    args = parser.parse_args(["--skip-mevdt", "--output-root", "paper-smoke"])

    assert args.skip_mevdt
    assert callable(module.run_pipeline)


def test_exporter_help_runs_as_script():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Generate canonical DVS-ENACT artifacts" in result.stdout
    assert "--skip-mevdt" in result.stdout
    assert "--synthetic-n-steps" in result.stdout
