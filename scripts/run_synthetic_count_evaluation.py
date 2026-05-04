"""Run a synthetic normal-flow event-count likelihood evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dvs_enact import (
    EDGE_ORDER,
    count_negative_log_likelihood,
    simulate_rectangle_event_counts,
)


def _serializable_simulation(name, simulation):
    normal_flow_nll = count_negative_log_likelihood(
        simulation.observed_counts,
        simulation.normal_flow_probabilities,
    )
    uniform_nll = count_negative_log_likelihood(
        simulation.observed_counts,
        simulation.uniform_probabilities,
    )
    return {
        "scenario": name,
        "velocity": simulation.velocity.tolist(),
        "observed_counts": simulation.observed_counts,
        "normal_flow_probabilities": simulation.normal_flow_probabilities,
        "uniform_probabilities": simulation.uniform_probabilities,
        "normal_flow_nll": normal_flow_nll,
        "uniform_nll": uniform_nll,
        "nll_improvement": uniform_nll - normal_flow_nll,
    }


def run(output_root: Path, total_events: int, seed: int) -> dict:
    scenarios = {
        "horizontal": np.array([1.0, 0.0]),
        "vertical": np.array([0.0, 1.0]),
        "diagonal": np.array([1.0, 1.0]),
    }
    results = [
        _serializable_simulation(
            name,
            simulate_rectangle_event_counts(
                velocity,
                total_events=total_events,
                background_activity=1e-3,
                seed=seed + index,
            ),
        )
        for index, (name, velocity) in enumerate(scenarios.items())
    ]

    data_dir = output_root / "data" / "paper_evidence"
    figure_dir = output_root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    json_path = data_dir / "synthetic_count_likelihood.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), sharey=True)
    for axis, result in zip(axes, results, strict=True):
        counts = [result["observed_counts"][edge] for edge in EDGE_ORDER]
        axis.bar(EDGE_ORDER, counts, color="#4c78a8")
        axis.set_title(result["scenario"])
        axis.tick_params(axis="x", rotation=35)
        axis.set_ylim(0, total_events)
    axes[0].set_ylabel("events")
    fig.suptitle("Synthetic DVS event counts by rectangle edge")
    fig.tight_layout()
    counts_path = figure_dir / "synthetic_count_likelihood_counts.png"
    fig.savefig(counts_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x = np.arange(len(results))
    width = 0.36
    ax.bar(
        x - width / 2.0,
        [result["normal_flow_nll"] for result in results],
        width,
        label="normal-flow",
        color="#59a14f",
    )
    ax.bar(
        x + width / 2.0,
        [result["uniform_nll"] for result in results],
        width,
        label="uniform",
        color="#e15759",
    )
    ax.set_xticks(x, [result["scenario"] for result in results])
    ax.set_ylabel("count NLL")
    ax.legend(frameon=False)
    fig.tight_layout()
    nll_path = figure_dir / "synthetic_count_likelihood_nll.png"
    fig.savefig(nll_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "json": str(json_path),
        "count_figure": str(counts_path),
        "nll_figure": str(nll_path),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../2026-05-DVS-ENACT-Paper"),
    )
    parser.add_argument("--total-events", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.total_events, args.seed), indent=2))


if __name__ == "__main__":
    main()
