import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eventvot_refinement.py"


def _load_module(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "run_eventvot_refinement_under_timestamp_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_xypt_events(path: Path, timestamps: list[int]) -> None:
    path.write_text(
        "".join(f"1,2,1,{timestamp}\n" for timestamp in timestamps),
        encoding="utf-8",
    )


def test_event_windows_use_sidecar_frame_timestamps(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    sequence_dir = tmp_path / "seq"
    sequence_dir.mkdir()
    (sequence_dir / "frame_timestamps.txt").write_text(
        "0\n10\n50\n100\n",
        encoding="utf-8",
    )
    event_csv = sequence_dir / "seq.csv"
    _write_xypt_events(event_csv, [5, 20, 30, 70, 90])

    frame_timestamps, source = module.resolve_eventvot_frame_timestamps(
        sequence_dir,
        "seq",
        4,
    )
    assert source.endswith("frame_timestamps.txt")

    windows = list(
        module.iter_eventvot_frame_windows(
            event_csv,
            4,
            frame_timestamps=frame_timestamps,
        )
    )

    assert [frame_index for frame_index, _batch in windows] == [1, 2, 3]
    assert [batch.ts.tolist() for _frame_index, batch in windows] == [
        [5],
        [20, 30],
        [70, 90],
    ]


def test_frame_timestamp_root_overrides_sequence_layout(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    sequence_dir = tmp_path / "seq"
    sequence_dir.mkdir()
    timestamp_root = tmp_path / "timestamps"
    timestamp_root.mkdir()
    (timestamp_root / "seq.csv").write_text(
        "0,100\n1,125\n2,200\n",
        encoding="utf-8",
    )

    frame_timestamps, source = module.resolve_eventvot_frame_timestamps(
        sequence_dir,
        "seq",
        3,
        frame_timestamps_root=timestamp_root,
    )

    assert source.endswith("seq.csv")
    np.testing.assert_allclose(frame_timestamps, np.array([100.0, 125.0, 200.0]))


def test_sequential_image_names_are_not_treated_as_timestamps(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    image_dir = tmp_path / "seq" / "img"
    image_dir.mkdir(parents=True)
    for index in range(1, 4):
        (image_dir / f"{index:06d}.png").write_bytes(b"")

    assert module._load_frame_timestamps_from_image_names(image_dir, 3) is None


def test_uniform_fallback_warns_when_timestamps_are_missing(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    event_csv = tmp_path / "seq.csv"
    _write_xypt_events(event_csv, [0, 10, 30])

    with pytest.warns(RuntimeWarning, match="falling back to uniform"):
        windows = list(module.iter_eventvot_frame_windows(event_csv, 3))

    assert [batch.ts.tolist() for _frame_index, batch in windows] == [[0], [10, 30]]
