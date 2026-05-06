"""Run controlled synthetic tracker-level comparisons for DVS-ENACT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dvs_enact import (
    SyntheticRectangleSequenceConfig,
    TrackerComparisonConfig,
    run_synthetic_tracker_benchmark,
)


def _default_scenarios(n_steps: int, events_per_window: int, seed: int):
    return {
        "horizontal_side_only": SyntheticRectangleSequenceConfig(
            n_steps=n_steps,
            velocity=(1.0, 0.0),
            events_per_window=events_per_window,
            background_activity=0.0,
            seed=seed,
        ),
        "vertical_side_only": SyntheticRectangleSequenceConfig(
            n_steps=n_steps,
            velocity=(0.0, 1.0),
            events_per_window=events_per_window,
            background_activity=0.0,
            seed=seed + 1,
        ),
        "horizontal_left_only": SyntheticRectangleSequenceConfig(
            n_steps=n_steps,
            velocity=(1.0, 0.0),
            events_per_window=events_per_window,
            background_activity=0.0,
            visible_edges=("left",),
            seed=seed + 2,
        ),
        "horizontal_weak_inactive_leak": SyntheticRectangleSequenceConfig(
            n_steps=n_steps,
            velocity=(1.0, 0.0),
            events_per_window=events_per_window,
            background_activity=0.05,
            seed=seed + 3,
        ),
    }


def _tracker_series(scenario: dict, tracker_key: str, metric: str) -> list[float]:
    return [
        window[tracker_key]["metrics"][metric]
        for window in scenario["windows"]
        if window[tracker_key]["metrics"][metric] is not None
    ]


def _write_figures(payload: dict, output_root: Path) -> dict[str, str]:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    scenarios = payload["scenarios"]
    fig, axes = plt.subplots(
        len(scenarios),
        1,
        figsize=(7.0, 2.1 * len(scenarios)),
        sharex=True,
        sharey=True,
    )
    if len(scenarios) == 1:
        axes = [axes]
    for axis, scenario in zip(axes, scenarios, strict=True):
        constant_position = _tracker_series(
            scenario,
            "constant_position",
            "inactive_axis_ratio",
        )
        baseline = _tracker_series(scenario, "baseline", "inactive_axis_ratio")
        dvs_enact = _tracker_series(scenario, "dvs_enact", "inactive_axis_ratio")
        axis.plot(
            constant_position,
            label="constant position",
            color="#54a24b",
            linewidth=1.8,
        )
        axis.plot(baseline, label="vanilla SCGP", color="#4c78a8", linewidth=1.8)
        axis.plot(dvs_enact, label="DVS-ENACT", color="#f58518", linewidth=1.8)
        axis.axhline(
            payload["tracker_parameters"]["collapse_threshold"],
            color="#777777",
            linestyle="--",
            linewidth=1.0,
        )
        axis.set_title(scenario["scenario"])
        axis.set_ylabel("inactive-axis ratio")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("window")
    axes[0].legend(frameon=False, ncols=3, loc="lower left")
    fig.tight_layout()
    ratio_path = figure_dir / "synthetic_tracker_inactive_axis_ratio.png"
    fig.savefig(ratio_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    labels = [scenario["scenario"] for scenario in scenarios]
    x = np.arange(len(labels))
    width = 0.26
    constant_position = [
        scenario["summary"]["constant_position"]["collapse_fraction"]
        for scenario in scenarios
    ]
    baseline = [
        scenario["summary"]["baseline"]["collapse_fraction"]
        for scenario in scenarios
    ]
    dvs_enact = [
        scenario["summary"]["dvs_enact"]["collapse_fraction"]
        for scenario in scenarios
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.bar(
        x - width,
        constant_position,
        width,
        label="constant position",
        color="#54a24b",
    )
    ax.bar(x, baseline, width, label="vanilla SCGP", color="#4c78a8")
    ax.bar(x + width, dvs_enact, width, label="DVS-ENACT", color="#f58518")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("collapse fraction")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    collapse_path = figure_dir / "synthetic_tracker_collapse_fraction.png"
    fig.savefig(collapse_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "inactive_axis_ratio": str(ratio_path),
        "collapse_fraction": str(collapse_path),
    }


def run(
    output_root: Path,
    n_steps: int,
    events_per_window: int,
    seed: int,
    max_events_per_window: int,
) -> dict:
    tracker_config = TrackerComparisonConfig(
        max_events_per_window=max_events_per_window,
        max_windows=None,
        measurement_noise_variance=4.0,
        radial_noise_variance=1.0,
        shape_variance=25.0,
    )
    payload = run_synthetic_tracker_benchmark(
        _default_scenarios(n_steps, events_per_window, seed),
        tracker_config=tracker_config,
    )

    data_dir = output_root / "data" / "paper_evidence"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "synthetic_tracker_comparison.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    figures = _write_figures(payload, output_root)
    return {
        "json": str(json_path),
        "figures": figures,
        "summary": {
            scenario["scenario"]: scenario["summary"]
            for scenario in payload["scenarios"]
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../2026-05-DVS-ENACT-Paper"),
    )
    parser.add_argument("--n-steps", type=int, default=40)
    parser.add_argument("--events-per-window", type=int, default=64)
    parser.add_argument("--max-events-per-window", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                output_root=args.output_root,
                n_steps=args.n_steps,
                events_per_window=args.events_per_window,
                seed=args.seed,
                max_events_per_window=args.max_events_per_window,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
