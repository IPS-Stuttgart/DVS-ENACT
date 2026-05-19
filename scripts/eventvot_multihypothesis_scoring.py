"""Candidate construction and scoring for EventVOT multi-hypothesis refinement."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from itertools import product
from typing import Any

import numpy as np

from dvs_enact import DVSContourRefiner, DVSContourRefinerConfig, EventBatch
from run_eventvot_refinement import (  # pylint: disable=import-error
    EventVOTAcceptanceConfig,
    EventVOTAcceptanceDecision,
    evaluate_refinement_acceptance,
)


@dataclass(frozen=True)
class EventVOTMultiHypothesisConfig:
    """Candidate-generation controls for per-frame hypothesis selection."""

    enabled: bool = True
    refinement_blends: tuple[float, ...] = ()
    search_expansion_factors: tuple[float, ...] = ()
    max_events: tuple[int | None, ...] = ()
    measurement_noise_variances: tuple[float, ...] = ()
    event_activity_floors: tuple[float, ...] = ()
    inactive_activity_thresholds: tuple[float, ...] = ()
    include_without_event_polarity: bool = False
    combine_values: bool = False
    max_hypotheses_per_frame: int = 24

    def __post_init__(self) -> None:
        if self.max_hypotheses_per_frame <= 0:
            raise ValueError("max_hypotheses_per_frame must be positive")


@dataclass(frozen=True)
class EventVOTHypothesis:
    """One evaluated DVS-ENACT refinement hypothesis."""

    index: int
    refiner_config: DVSContourRefinerConfig
    result: Any
    decision: EventVOTAcceptanceDecision
    score: float
    elapsed_seconds: float


def default_multihypothesis_config() -> EventVOTMultiHypothesisConfig:
    return EventVOTMultiHypothesisConfig(
        refinement_blends=(0.15, 0.25, 0.35),
        search_expansion_factors=(1.15, 1.25, 1.40),
        max_events=(64, 128, 256),
        measurement_noise_variances=(2.0, 4.0, 8.0),
        event_activity_floors=(0.03, 0.05, 0.08),
    )


def evaluate_frame_hypotheses(
    candidate_xywh: np.ndarray,
    previous_candidate_xywh: np.ndarray,
    event_window: EventBatch,
    candidate_refiners: Iterable[DVSContourRefiner],
    acceptance_config: EventVOTAcceptanceConfig,
    *,
    previous_output_xywh: np.ndarray | None = None,
) -> list[EventVOTHypothesis]:
    hypotheses = []
    for index, candidate_refiner in enumerate(candidate_refiners):
        started = time.perf_counter()
        result = candidate_refiner.refine(
            candidate_xywh,
            event_window,
            previous_candidate_bbox=previous_candidate_xywh,
        )
        elapsed = time.perf_counter() - started
        decision = evaluate_refinement_acceptance(
            candidate_xywh,
            result,
            acceptance_config,
            previous_candidate_xywh=previous_candidate_xywh,
            previous_output_xywh=previous_output_xywh,
        )
        hypotheses.append(
            EventVOTHypothesis(
                index,
                candidate_refiner.config,
                result,
                decision,
                score_refinement_hypothesis(result, decision),
                elapsed,
            ),
        )
    if not hypotheses:
        raise ValueError("No DVS-ENACT hypotheses configured")
    return hypotheses


def select_frame_hypothesis(
    hypotheses: Iterable[EventVOTHypothesis],
) -> EventVOTHypothesis:
    candidates = list(hypotheses)
    accepted = [candidate for candidate in candidates if candidate.decision.accepted]
    return max(
        accepted or candidates,
        key=lambda candidate: (candidate.score, -candidate.index),
    )


def score_refinement_hypothesis(
    result: Any,
    decision: EventVOTAcceptanceDecision,
) -> float:
    score = 0.0
    score += 2.0 * _finite_or_default(result.mean_event_activity, 0.0)
    score += 1.5 * _finite_or_default(decision.active_fraction, 0.0)
    score += 0.75 * _finite_or_default(result.polarity_consistency_fraction, 0.5)
    score += 0.05 * math.log1p(max(0, int(result.used_event_count)))
    score += 0.75 * _finite_or_default(decision.candidate_iou, 0.0)
    score -= 0.75 * _finite_or_default(decision.center_shift_ratio, 0.0)
    score -= 0.25 * _abs_log_ratio(decision.candidate_area_ratio)
    score -= 0.02 * _finite_or_default(
        decision.quadratic_form_per_active_measurement,
        0.0,
    )
    if result.fallback_reason is not None:
        score -= 20.0
    if not decision.accepted:
        score -= 100.0 + len(decision.rejection_reasons)
    return float(score)


def build_candidate_refiner_configs(
    base_config: DVSContourRefinerConfig,
    multi_config: EventVOTMultiHypothesisConfig,
) -> tuple[DVSContourRefinerConfig, ...]:
    if not multi_config.enabled:
        return (base_config,)
    value_grid = {
        "refinement_blend": _candidate_float_values(
            base_config.refinement_blend,
            multi_config.refinement_blends,
        ),
        "search_expansion_factor": _candidate_float_values(
            base_config.search_expansion_factor,
            multi_config.search_expansion_factors,
        ),
        "max_events": _candidate_int_values(
            base_config.max_events,
            multi_config.max_events,
        ),
        "measurement_noise_variance": _candidate_float_values(
            base_config.measurement_noise_variance,
            multi_config.measurement_noise_variances,
        ),
        "event_activity_floor": _candidate_float_values(
            base_config.event_activity_floor,
            multi_config.event_activity_floors,
        ),
        "inactive_activity_threshold": _candidate_float_values(
            base_config.inactive_activity_threshold,
            multi_config.inactive_activity_thresholds,
        ),
        "use_event_polarity": (
            base_config.use_event_polarity,
            not base_config.use_event_polarity,
        )
        if multi_config.include_without_event_polarity
        else (base_config.use_event_polarity,),
    }
    configs = (
        _grid_configs(base_config, value_grid)
        if multi_config.combine_values
        else _axiswise_configs(base_config, value_grid)
    )
    return tuple(configs[: multi_config.max_hypotheses_per_frame])


def _axiswise_configs(
    base_config: DVSContourRefinerConfig,
    value_grid: dict[str, tuple[Any, ...]],
) -> list[DVSContourRefinerConfig]:
    configs = [base_config]
    for field_name, values in value_grid.items():
        for value in values:
            variant = replace(base_config, **{field_name: value})
            if variant not in configs:
                configs.append(variant)
    return configs


def _grid_configs(
    base_config: DVSContourRefinerConfig,
    value_grid: dict[str, tuple[Any, ...]],
) -> list[DVSContourRefinerConfig]:
    configs: list[DVSContourRefinerConfig] = []
    names = tuple(value_grid)
    for values in product(*(value_grid[name] for name in names)):
        variant = replace(base_config, **dict(zip(names, values, strict=True)))
        if variant not in configs:
            configs.append(variant)
    return configs


def _candidate_float_values(
    base: float,
    values: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(_dedupe_float_values((base, *(values or (base,)))))


def _candidate_int_values(
    base: int | None,
    values: tuple[int | None, ...],
) -> tuple[int | None, ...]:
    result: list[int | None] = []
    for value in (base, *(values or (base,))):
        if value not in result:
            result.append(value)
    return tuple(result)


def _dedupe_float_values(values: Iterable[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        value = float(value)
        if not any(math.isclose(value, existing) for existing in result):
            result.append(value)
    return result


def _finite_or_default(value: float | None, default: float) -> float:
    if value is None:
        return default
    value = float(value)
    return value if math.isfinite(value) else default


def _abs_log_ratio(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        return 10.0
    return abs(math.log(value))
