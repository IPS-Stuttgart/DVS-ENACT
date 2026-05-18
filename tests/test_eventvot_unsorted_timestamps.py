import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eventvot_refinement.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_eventvot_refinement_unsorted_timestamp_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_eventvot_frame_windows_sort_unsorted_xypt_events_stably(tmp_path):
    module = _load_module()
    event_csv = tmp_path / "events.csv"
    event_csv.write_text(
        "\n".join(
            [
                "12,21,1,20",
                "10,20,1,0",
                "13,22,0,30",
                "11,20,0,10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    windows = list(
        module.iter_eventvot_frame_windows(
            event_csv,
            3,
            event_column_order="xypt",
        )
    )

    assert [frame_index for frame_index, _window in windows] == [1, 2]
    assert windows[0][1].ts.tolist() == [0, 10]
    assert windows[0][1].x.tolist() == [10, 11]
    assert windows[0][1].y.tolist() == [20, 20]
    assert windows[0][1].p.tolist() == [1, 0]
    assert windows[1][1].ts.tolist() == [20, 30]
    assert windows[1][1].x.tolist() == [12, 13]
    assert windows[1][1].y.tolist() == [21, 22]
    assert windows[1][1].p.tolist() == [1, 0]
