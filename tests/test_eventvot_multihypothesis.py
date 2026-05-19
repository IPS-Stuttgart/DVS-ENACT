import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

# Ensure the script can import its sibling run_eventvot_refinement module.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("dvs_enact")

import eventvot_multihypothesis_scoring as mh  # noqa: E402
from dvs_enact import DVSContourRefinerConfig  # noqa: E402
from run_eventvot_refinement import EventVOTAcceptanceDecision  # noqa: E402


@dataclass(frozen=True)
class DummyResult:
    used_event_count: int
    mean_event_activity: float | None
    polarity_consistency_fraction: float | None
    fallback_reason: str | None = None


def _decision(accepted: bool) -> EventVOTAcceptanceDecision:
    return EventVOTAcceptanceDecision(
        accepted=accepted,
        rejection_reasons=() if accepted else ("used_event_count",),
        candidate_iou=0.9,
        candidate_area_ratio=1.0,
        center_shift_ratio=0.02,
        raw_candidate_iou=0.9,
        raw_candidate_area_ratio=1.0,
        raw_center_shift_ratio=0.02,
        temporal_center_shift_ratio=None,
        temporal_size_change_ratio=None,
        motion_prediction_error_ratio=None,
        active_fraction=0.5,
        quadratic_form_per_active_measurement=1.0,
    )


def test_axiswise_candidate_configs_keep_base_first_and_deduplicate():
    base = DVSContourRefinerConfig(
        input_bbox_format="xywh",
        output_bbox_format="xywh",
        refinement_blend=0.25,
        max_events=128,
        measurement_noise_variance=4.0,
    )
    config = mh.EventVOTMultiHypothesisConfig(
        refinement_blends=(0.25, 0.35),
        max_events=(128, 256),
        measurement_noise_variances=(4.0, 8.0),
        max_hypotheses_per_frame=16,
    )

    candidates = mh.build_candidate_refiner_configs(base, config)

    assert candidates[0] == base
    assert len(candidates) == 4
    assert any(candidate.refinement_blend == 0.35 for candidate in candidates)
    assert any(candidate.max_events == 256 for candidate in candidates)
    assert any(candidate.measurement_noise_variance == 8.0 for candidate in candidates)


def test_grid_candidate_configs_respect_hypothesis_cap():
    base = DVSContourRefinerConfig(refinement_blend=0.25, max_events=128)
    config = mh.EventVOTMultiHypothesisConfig(
        refinement_blends=(0.15, 0.25, 0.35),
        max_events=(64, 128, 256),
        combine_values=True,
        max_hypotheses_per_frame=5,
    )

    candidates = mh.build_candidate_refiner_configs(base, config)

    assert len(candidates) == 5
    assert candidates[0] == base


def test_selection_prefers_best_accepted_hypothesis_over_rejected_score():
    base = DVSContourRefinerConfig()
    accepted_weaker = mh.EventVOTHypothesis(
        index=0,
        refiner_config=base,
        result=DummyResult(
            used_event_count=10,
            mean_event_activity=0.1,
            polarity_consistency_fraction=0.5,
        ),
        decision=_decision(True),
        score=1.0,
        elapsed_seconds=0.01,
    )
    accepted_stronger = mh.EventVOTHypothesis(
        index=1,
        refiner_config=base,
        result=DummyResult(
            used_event_count=20,
            mean_event_activity=0.4,
            polarity_consistency_fraction=0.8,
        ),
        decision=_decision(True),
        score=2.0,
        elapsed_seconds=0.01,
    )
    rejected = mh.EventVOTHypothesis(
        index=2,
        refiner_config=base,
        result=DummyResult(
            used_event_count=100,
            mean_event_activity=1.0,
            polarity_consistency_fraction=1.0,
        ),
        decision=_decision(False),
        score=10.0,
        elapsed_seconds=0.01,
    )

    selected = mh.select_frame_hypothesis(
        [accepted_weaker, rejected, accepted_stronger],
    )

    assert selected.index == 1


def test_score_penalizes_rejected_hypotheses():
    result = DummyResult(
        used_event_count=100,
        mean_event_activity=1.0,
        polarity_consistency_fraction=1.0,
    )

    accepted_score = mh.score_refinement_hypothesis(result, _decision(True))
    rejected_score = mh.score_refinement_hypothesis(result, _decision(False))

    assert accepted_score > rejected_score
