"""Shared EventVOT event-support diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

EVENT_SUPPORT_FULL_STRENGTH_EVENT_COUNT = 64.0


def event_support_score(source: Any) -> float:
    """Return a conservative [0, 1] confidence score from DVS diagnostics."""
    activity = finite_unit_interval(
        diagnostic_value(source, "mean_event_activity"),
        default=0.0,
    )
    active_fraction = active_measurement_fraction(source) or 0.0
    event_count_strength = used_event_count_strength(source)
    components = [activity, active_fraction, event_count_strength]

    polarity_consistency = finite_unit_interval(
        diagnostic_value(source, "polarity_consistency_fraction"),
        default=None,
    )
    if polarity_consistency is not None:
        components.append(polarity_consistency)

    clipped = np.clip(np.asarray(components, dtype=float), 0.0, 1.0)
    return float(np.prod(clipped) ** (1.0 / clipped.size))


def active_measurement_fraction(source: Any) -> float | None:
    """Return active measurements per used event, when event support exists."""
    used_event_count = int(diagnostic_value(source, "used_event_count", 0) or 0)
    active_measurement_count = int(
        diagnostic_value(source, "active_measurement_count", 0) or 0
    )
    if used_event_count <= 0:
        return None
    return float(
        np.clip(active_measurement_count / used_event_count, 0.0, 1.0)
    )


def used_event_count_strength(source: Any) -> float:
    """Return a bounded confidence contribution from the used event count."""
    used_event_count = int(diagnostic_value(source, "used_event_count", 0) or 0)
    if used_event_count <= 0:
        return 0.0
    return float(
        min(1.0, used_event_count / EVENT_SUPPORT_FULL_STRENGTH_EVENT_COUNT)
    )


def finite_unit_interval(value: Any, *, default: float | None) -> float | None:
    """Return a finite scalar clipped to [0, 1], or default when unavailable."""
    if value is None:
        return default
    numeric = float(value)
    if not np.isfinite(numeric):
        return default
    return float(np.clip(numeric, 0.0, 1.0))


def diagnostic_value(source: Any, name: str, default: Any = None) -> Any:
    """Read a diagnostic from either a refinement result object or a dict."""
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)
