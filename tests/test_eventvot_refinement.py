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
    base_results = root / "base_results"
    base_results.mkdir()
    _write_eventvot_fixture_sequence(split_root, base_results, "recording_0001")
    output_results = root / "refined_results"
    return split_root, base_results, output_results


def _write_eventvot_fixture_sequence(
    split_root: Path,
    base_results: Path,
    sequence_name: str,
) -> None:
    sequence_dir = split_root / sequence_name
    image_dir = sequence_dir / "img"
    image_dir.mkdir(parents=True)
    for index in range(3):
        (image_dir / f"{index:04d}.png").write_bytes(b"")
    existing_sequences = []
    list_path = split_root / "list.txt"
    if list_path.exists():
        existing_sequences = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if sequence_name not in existing_sequences:
        existing_sequences.append(sequence_name)
    list_path.write_text("\n".join(existing_sequences) + "\n", encoding="utf-8")
    (sequence_dir / f"{sequence_name}.csv").write_text(
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
    (base_results / f"{sequence_name}.txt").write_text(
        "8\t8\t10\t10\n9\t8\t10\t10\n10\t8\t10\t10\n",
        encoding="utf-8",
    )


def _run_fake_refinement(
    module,
    tmp_path: Path,
    base_results: Path,
    output_results: Path,
    result_boxes: list[list[float]],
    **config_kwargs,
) -> dict:
    return module.run(
        module.EventVOTRefinementOptions(
            eventvot_root=tmp_path,
            base_results=base_results,
            output_results=output_results,
            split="test",
        ),
        refiner=_FakeRefiner(
            module,
            [_FakeResult(box) for box in result_boxes],
            **config_kwargs,
        ),
    )


def _assert_recording_result(
    output_results: Path,
    expected_rows: list[list[float]],
) -> None:
    np.testing.assert_allclose(
        np.loadtxt(output_results / "recording_0001.txt"),
        np.asarray(expected_rows, dtype=float),
    )


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


def test_eventvot_refinement_skips_complete_existing_result(tmp_path):
    module = _load_module()
    _split_root, base_results, output_results = _write_eventvot_fixture(tmp_path)
    first_refiner = _FakeRefiner(
        module,
        [
            _FakeResult([9.5, 8.0, 10.0, 10.0]),
            _FakeResult([10.0, 8.0, 10.0, 10.0]),
        ],
    )
    module.run(
        module.EventVOTRefinementOptions(
            eventvot_root=tmp_path,
            base_results=base_results,
            output_results=output_results,
            split="test",
        ),
        refiner=first_refiner,
    )
    assert (output_results / "recording_0001_cache_manifest.json").exists()

    payload = module.run(
        module.EventVOTRefinementOptions(
            eventvot_root=tmp_path,
            base_results=base_results,
            output_results=output_results,
            split="test",
        ),
        refiner=_FailingRefiner(module),
    )

    summary = payload["summary"]
    sequence = payload["sequences"][0]
    assert summary["skipped_existing_output_count"] == 1
    assert summary["fallback_counts"]["skipped_existing_output"] == 3
    assert sequence["skipped_existing_output"]
    assert sequence["accepted_refinement_count"] == 1
    assert sequence["frames"] == []
    assert (output_results / "recording_0001_time.txt").exists()


def test_eventvot_refinement_recomputes_unmanifested_existing_result(tmp_path):
    module = _load_module()
    _split_root, base_results, output_results = _write_eventvot_fixture(tmp_path)
    output_results.mkdir()
    (output_results / "recording_0001.txt").write_text(
        "8\t8\t10\t10\n99\t99\t10\t10\n10\t8\t10\t10\n",
        encoding="utf-8",
    )
    payload = _run_fake_refinement(
        module,
        tmp_path,
        base_results,
        output_results,
        [[9.25, 8.0, 10.0, 10.0], [10.0, 8.0, 10.0, 10.0]],
    )

    _assert_recording_result(
        output_results,
        [
            [8.0, 8.0, 10.0, 10.0],
            [9.25, 8.0, 10.0, 10.0],
            [10.0, 8.0, 10.0, 10.0],
        ],
    )
    assert payload["summary"]["skipped_existing_output_count"] == 0
    assert (output_results / "recording_0001_cache_manifest.json").exists()


def test_eventvot_refinement_recomputes_when_resume_config_changes(tmp_path):
    module = _load_module()
    _split_root, base_results, output_results = _write_eventvot_fixture(tmp_path)
    module.run(
        module.EventVOTRefinementOptions(
            eventvot_root=tmp_path,
            base_results=base_results,
            output_results=output_results,
            split="test",
        ),
        refiner=_FakeRefiner(
            module,
            [
                _FakeResult([9.25, 8.0, 10.0, 10.0]),
                _FakeResult([10.0, 8.0, 10.0, 10.0]),
            ],
        ),
    )

    payload = module.run(
        module.EventVOTRefinementOptions(
            eventvot_root=tmp_path,
            base_results=base_results,
            output_results=output_results,
            split="test",
        ),
        refiner=_FakeRefiner(
            module,
            [
                _FakeResult([9.75, 8.0, 10.0, 10.0]),
                _FakeResult([10.0, 8.0, 10.0, 10.0]),
            ],
            max_events=64,
        ),
    )

    refined = np.loadtxt(output_results / "recording_0001.txt")
    np.testing.assert_allclose(refined[1], np.array([9.75, 8.0, 10.0, 10.0]))
    assert payload["summary"]["skipped_existing_output_count"] == 0


def test_eventvot_refinement_selects_sequence_chunk(tmp_path):
    module = _load_module()
    split_root, base_results, output_results = _write_eventvot_fixture(tmp_path)
    _write_eventvot_fixture_sequence(split_root, base_results, "recording_0002")
    _write_eventvot_fixture_sequence(split_root, base_results, "recording_0003")

    payload = module.run(
        module.EventVOTRefinementOptions(
            eventvot_root=tmp_path,
            base_results=base_results,
            output_results=output_results,
            split="test",
            sequence_index=1,
            sequence_count=2,
        ),
        refiner=module.DVSContourRefiner(
            module.DVSContourRefinerConfig(
                input_bbox_format="xywh",
                output_bbox_format="xywh",
                min_events=99,
            )
        ),
    )

    assert payload["summary"]["sequence_count"] == 1
    assert [sequence["sequence"] for sequence in payload["sequences"]] == [
        "recording_0002"
    ]
    assert not (output_results / "recording_0001.txt").exists()
    assert (output_results / "recording_0002.txt").exists()
    assert not (output_results / "recording_0003.txt").exists()


def test_eventvot_sequence_selection_supports_lists_files_and_shards(tmp_path):
    module = _load_module()
    sequence_file = tmp_path / "sequences.txt"
    sequence_file.write_text(
        "\n".join(
            [
                "# comment",
                "recording_0003",
                "recording_0004, recording_0005",
                "recording_0001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    requested = module.load_requested_sequence_names(
        ["recording_0001"],
        ["recording_0002 recording_0003"],
        [sequence_file],
    )

    assert requested == (
        "recording_0001",
        "recording_0002",
        "recording_0003",
        "recording_0004",
        "recording_0005",
    )
    assert module.select_sequence_chunk(
        requested,
        sequence_index=1,
        sequence_count=2,
    ) == ["recording_0002", "recording_0004"]


def test_eventvot_refinement_uses_conservative_acceptance_gates(tmp_path):
    module = _load_module()
    _split_root, base_results, output_results = _write_eventvot_fixture(tmp_path)
    payload = _run_fake_refinement(
        module,
        tmp_path,
        base_results,
        output_results,
        [[9.25, 8.0, 10.0, 10.0], [40.0, 8.0, 10.0, 10.0]],
    )

    _assert_recording_result(
        output_results,
        [
            [8.0, 8.0, 10.0, 10.0],
            [9.25, 8.0, 10.0, 10.0],
            [10.0, 8.0, 10.0, 10.0],
        ],
    )
    summary = payload["summary"]
    assert summary["accepted_refinement_count"] == 1
    assert summary["refiner_success_frame_count"] == 2
    assert summary["acceptance_counts"]["accepted"] == 1
    assert summary["acceptance_counts"]["candidate_iou"] == 1
    frames = payload["sequences"][0]["frames"]
    assert frames[1]["accept_refinement"]
    assert not frames[2]["accept_refinement"]
    assert frames[2]["rejection_reasons"] == ["candidate_iou", "center_shift_ratio"]


def test_eventvot_acceptance_rejects_large_center_shift_and_area_change():
    module = _load_module()
    config = module.EventVOTAcceptanceConfig(
        min_candidate_iou=0.0,
        min_candidate_area_ratio=0.75,
        max_candidate_area_ratio=1.25,
        max_center_shift_ratio=0.25,
    )

    shifted = module.evaluate_refinement_acceptance(
        np.array([10.0, 10.0, 20.0, 20.0]),
        _FakeResult([20.0, 10.0, 20.0, 20.0]),
        config,
    )
    shrunk = module.evaluate_refinement_acceptance(
        np.array([10.0, 10.0, 20.0, 20.0]),
        _FakeResult([10.0, 10.0, 10.0, 10.0]),
        config,
    )
    grown = module.evaluate_refinement_acceptance(
        np.array([10.0, 10.0, 20.0, 20.0]),
        _FakeResult([10.0, 10.0, 30.0, 30.0]),
        config,
    )

    assert not shifted.accepted
    assert shifted.rejection_reasons == ("center_shift_ratio",)
    assert not shrunk.accepted
    assert shrunk.rejection_reasons == ("candidate_area_ratio",)
    assert not grown.accepted
    assert grown.rejection_reasons == ("candidate_area_ratio",)
    assert module.center_shift_ratio_xywh(
        np.array([10.0, 10.0, 20.0, 20.0]),
        np.array([20.0, 10.0, 20.0, 20.0]),
    ) > 0.25


def test_eventvot_acceptance_rejects_raw_refinement_jump():
    module = _load_module()
    config = module.EventVOTAcceptanceConfig(
        min_candidate_iou=0.0,
        min_raw_candidate_iou=0.50,
    )

    decision = module.evaluate_refinement_acceptance(
        np.array([10.0, 10.0, 20.0, 20.0]),
        _FakeResult([10.5, 10.0, 20.0, 20.0], raw_xywh=[100.0, 100.0, 20.0, 20.0]),
        config,
    )

    assert not decision.accepted
    assert decision.rejection_reasons == ("raw_candidate_iou",)
    assert decision.candidate_iou > 0.90
    assert decision.raw_candidate_iou == 0.0


def test_eventvot_acceptance_rejects_temporal_output_shocks():
    module = _load_module()

    jump_decision = module.evaluate_refinement_acceptance(
        np.array([30.0, 10.0, 40.0, 20.0]),
        _FakeResult([30.0, 10.0, 40.0, 20.0]),
        module.EventVOTAcceptanceConfig(
            min_candidate_iou=0.0,
            max_center_shift_ratio=10.0,
            max_temporal_center_shift_ratio=0.50,
            max_temporal_size_change_ratio=2.0,
        ),
        previous_output_xywh=np.array([0.0, 10.0, 20.0, 20.0]),
    )
    size_decision = module.evaluate_refinement_acceptance(
        np.array([30.0, 10.0, 40.0, 20.0]),
        _FakeResult([30.0, 10.0, 40.0, 20.0]),
        module.EventVOTAcceptanceConfig(
            min_candidate_iou=0.0,
            max_center_shift_ratio=10.0,
            max_temporal_center_shift_ratio=2.0,
            max_temporal_size_change_ratio=0.50,
        ),
        previous_output_xywh=np.array([30.0, 10.0, 20.0, 20.0]),
    )

    assert not jump_decision.accepted
    assert jump_decision.rejection_reasons == ("temporal_center_shift_ratio",)
    assert jump_decision.temporal_center_shift_ratio > 0.50
    assert not size_decision.accepted
    assert size_decision.rejection_reasons == ("temporal_size_change_ratio",)
    assert size_decision.temporal_size_change_ratio == 1.0


def test_eventvot_acceptance_rejects_base_motion_inconsistent_update():
    module = _load_module()

    config = module.EventVOTAcceptanceConfig(
        min_candidate_iou=0.0,
        max_center_shift_ratio=10.0,
        max_motion_prediction_error_ratio=0.50,
    )
    candidate = np.array([10.0, 10.0, 20.0, 20.0])
    previous_boxes = {
        "previous_candidate_xywh": np.array([0.0, 10.0, 20.0, 20.0]),
        "previous_output_xywh": np.array([0.0, 10.0, 20.0, 20.0]),
    }

    decision = module.evaluate_refinement_acceptance(
        candidate,
        _FakeResult([40.0, 10.0, 20.0, 20.0]),
        config,
        **previous_boxes,
    )

    assert not decision.accepted
    assert decision.rejection_reasons == ("motion_prediction_error_ratio",)
    assert decision.motion_prediction_error_ratio > 0.50


def test_eventvot_refinement_can_hold_rejected_center_correction(tmp_path):
    module = _load_module()
    split_root, base_results, output_results = _write_eventvot_fixture(tmp_path)
    sequence_dir = split_root / "recording_0001"
    output_file = output_results / "recording_0001.txt"
    low_event_result = _FakeResult([99.0, 99.0, 10.0, 10.0])
    low_event_result.used_event_count = 0
    low_event_result.active_measurement_count = 0

    summary = module.refine_sequence(
        "recording_0001",
        sequence_dir,
        base_results / "recording_0001.txt",
        output_file,
        _FakeRefiner(
            module,
            [
                _FakeResult([11.0, 8.0, 10.0, 10.0]),
                low_event_result,
            ],
        ),
        event_column_order="xypt",
        acceptance_config=module.EventVOTAcceptanceConfig(
            max_rejected_center_hold_frames=1,
            rejected_center_hold_decay=1.0,
        ),
    )

    refined = np.loadtxt(output_file)
    np.testing.assert_allclose(
        refined,
        np.array(
            [
                [8.0, 8.0, 10.0, 10.0],
                [11.0, 8.0, 10.0, 10.0],
                [12.0, 8.0, 10.0, 10.0],
            ]
        ),
    )
    assert summary["accepted_refinement_count"] == 1
    assert summary["held_rejected_center_count"] == 1
    assert summary["frames"][2]["held_rejected_center_correction"]
    assert summary["frames"][2]["rejected_center_hold_age"] == 1


def test_eventvot_event_window_iterator_uses_between_frame_intervals(tmp_path):
    module = _load_module()
    _split_root, _base_results, _output_results = _write_eventvot_fixture(tmp_path)
    event_csv = tmp_path / "test" / "recording_0001" / "recording_0001.csv"

    windows = list(module.iter_eventvot_frame_windows(event_csv, 3))

    assert [frame_index for frame_index, _window in windows] == [1, 2]
    assert windows[0][1].ts.tolist() == [0, 10]
    assert windows[1][1].ts.tolist() == [20, 30]


def test_eventvot_time_span_reads_min_max_and_count(tmp_path):
    module = _load_module()
    _split_root, _base_results, _output_results = _write_eventvot_fixture(tmp_path)
    event_csv = tmp_path / "test" / "recording_0001" / "recording_0001.csv"

    assert module.read_eventvot_event_time_span(event_csv) == (0, 30, 4)


def test_eventvot_frame_windows_use_numpy_for_xypt_csv(tmp_path):
    module = _load_module()
    event_csv = tmp_path / "events.csv"
    event_csv.write_text(
        "\n".join(
            [
                "10,20,1,0",
                "11,20,0,10",
                "12,21,1,20",
                "13,22,1,30",
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
    assert windows[1][1].ts.tolist() == [20, 30]
    assert windows[1][1].p.tolist() == [1, 1]


def test_eventvot_split_root_resolves_nested_dropbox_layout(tmp_path):
    module = _load_module()
    sequence_dir = tmp_path / "EventVOT" / "test" / "test" / "recording_0001"
    (sequence_dir / "img").mkdir(parents=True)
    (sequence_dir / "recording_0001.csv").write_text("x,y,p,t\n", encoding="utf-8")

    assert module.resolve_eventvot_split_root(tmp_path / "EventVOT", "test") == (
        tmp_path / "EventVOT" / "test" / "test"
    )


def test_eventvot_refinement_help_runs_as_script():
    help_text = subprocess.check_output(
        (sys.executable, str(SCRIPT_PATH), "--help"),
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )

    assert "Refine EventVOT xywh tracker result files" in help_text
    assert "--eventvot-root" in help_text
    assert "--tracker-name" in help_text
    assert "--sequence-list" in help_text
    assert "--sequence-file" in help_text
    assert "--sequence-index" in help_text
    assert "--sequence-count" in help_text
    assert "--no-skip-existing" in help_text
    assert "--event-column-order" in help_text
    assert "--event-activity-floor" in help_text
    assert "--inactive-activity-threshold" in help_text
    assert "--measurement-noise-variance" in help_text
    assert "--min-accept-used-events" in help_text
    assert "--min-accept-area-ratio" in help_text
    assert "--max-accept-center-shift-ratio" in help_text
    assert "--min-raw-candidate-iou" in help_text
    assert "--min-active-fraction" in help_text
    assert "--max-temporal-center-shift-ratio" in help_text
    assert "--max-temporal-size-change-ratio" in help_text
    assert "--max-motion-prediction-error-ratio" in help_text
    assert "--max-rejected-center-hold-frames" in help_text
    assert "--rejected-center-hold-decay" in help_text


class _FakeRefiner:
    def __init__(self, module, results, **config_kwargs):
        self.config = module.DVSContourRefinerConfig(
            input_bbox_format="xywh",
            output_bbox_format="xywh",
            **config_kwargs,
        )
        self._results = list(results)

    def refine(self, _candidate, _events, *, previous_candidate_bbox=None):
        del previous_candidate_bbox
        return self._results.pop(0)


class _FailingRefiner:
    def __init__(self, module):
        self.config = module.DVSContourRefinerConfig(
            input_bbox_format="xywh",
            output_bbox_format="xywh",
        )

    def refine(self, _candidate, _events, *, previous_candidate_bbox=None):
        del previous_candidate_bbox
        raise AssertionError("existing complete result should have been skipped")


class _FakeResult:
    fallback_reason = None
    used_event_count = 12
    active_measurement_count = 3
    mean_event_activity = 0.20
    mean_event_polarity_weight = None
    polarity_consistency_fraction = None
    quadratic_form = None

    def __init__(self, xywh, *, raw_xywh=None):
        self._xywh = tuple(float(value) for value in xywh)
        self._raw_xywh = tuple(float(value) for value in (raw_xywh or xywh))

    def as_xywh(self):
        return self._xywh

    @property
    def refined_bbox(self):
        x, y, width, height = self._raw_xywh
        return {
            "x_min": x,
            "y_min": y,
            "width": width,
            "height": height,
        }

    def to_dict(self):
        return {
            "fallback_reason": self.fallback_reason,
            "used_event_count": self.used_event_count,
            "active_measurement_count": self.active_measurement_count,
            "mean_event_activity": self.mean_event_activity,
            "mean_event_polarity_weight": self.mean_event_polarity_weight,
            "polarity_consistency_fraction": self.polarity_consistency_fraction,
            "quadratic_form": self.quadratic_form,
            "refined_bbox": self.refined_bbox,
        }
