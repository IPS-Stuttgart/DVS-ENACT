from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eventvot_refinement.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_eventvot_refinement_window_sorting_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_unsorted_xypt_events(path: Path, *, include_header: bool) -> None:
    rows = [
        "1,1,1,20",
        "2,2,0,0",
        "3,3,1,10",
        "4,4,0,5",
    ]
    if include_header:
        rows.insert(0, "x,y,p,t")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _assert_expected_sorted_windows(module, event_csv: Path) -> None:
    windows = list(
        module.iter_eventvot_frame_windows(
            event_csv,
            frame_count=3,
            event_column_order="xypt",
        )
    )

    assert [frame_index for frame_index, _batch in windows] == [1, 2]
    first_batch = windows[0][1]
    second_batch = windows[1][1]

    np.testing.assert_array_equal(first_batch.ts, np.array([0, 5], dtype=np.int64))
    np.testing.assert_array_equal(first_batch.x, np.array([2, 4], dtype=np.int32))
    np.testing.assert_array_equal(first_batch.y, np.array([2, 4], dtype=np.int32))
    np.testing.assert_array_equal(first_batch.p, np.array([0, 0], dtype=np.int8))

    np.testing.assert_array_equal(second_batch.ts, np.array([10, 20], dtype=np.int64))
    np.testing.assert_array_equal(second_batch.x, np.array([3, 1], dtype=np.int32))
    np.testing.assert_array_equal(second_batch.y, np.array([3, 1], dtype=np.int32))
    np.testing.assert_array_equal(second_batch.p, np.array([1, 1], dtype=np.int8))


def test_eventvot_numpy_window_splitter_sorts_unsorted_timestamps(tmp_path: Path) -> None:
    module = _load_module()
    event_csv = tmp_path / "events.csv"
    _write_unsorted_xypt_events(event_csv, include_header=False)

    _assert_expected_sorted_windows(module, event_csv)


def test_eventvot_streaming_window_splitter_sorts_unsorted_timestamps(
    tmp_path: Path,
) -> None:
    module = _load_module()
    event_csv = tmp_path / "events_with_header.csv"
    _write_unsorted_xypt_events(event_csv, include_header=True)

    _assert_expected_sorted_windows(module, event_csv)


def test_eventvot_time_span_uses_min_max_for_unsorted_timestamps(
    tmp_path: Path,
) -> None:
    module = _load_module()
    event_csv = tmp_path / "events.csv"
    _write_unsorted_xypt_events(event_csv, include_header=False)

    assert module.read_eventvot_event_time_span(
        event_csv,
        event_column_order="xypt",
    ) == (0, 20, 4)
