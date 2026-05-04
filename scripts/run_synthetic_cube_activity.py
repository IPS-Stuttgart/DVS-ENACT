"""Generate a small normal-flow activity result for the paper repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dvs_enact import activity_profile, rectangle_contour_samples


def summarize_by_edge(edge_labels, activities):
    labels = np.array(edge_labels)
    return {
        edge: float(np.mean(activities[labels == edge]))
        for edge in ["left", "right", "top", "bottom"]
    }


def run(output_root: Path) -> dict:
    contour = rectangle_contour_samples(width=2.0, height=1.0, samples_per_edge=60)
    velocities = {
        "horizontal": np.array([1.0, 0.0]),
        "vertical": np.array([0.0, 1.0]),
        "diagonal": np.array([1.0, 1.0]),
    }
    summaries = {
        name: summarize_by_edge(
            contour.edge_labels,
            activity_profile(contour.normals, velocity),
        )
        for name, velocity in velocities.items()
    }

    data_dir = output_root / "data" / "paper_evidence"
    figure_dir = output_root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    json_path = data_dir / "synthetic_cube_activity.json"
    json_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6), sharex=True, sharey=True)
    for axis, (name, velocity) in zip(axes, velocities.items(), strict=True):
        activities = activity_profile(contour.normals, velocity)
        scatter = axis.scatter(
            contour.points[:, 0],
            contour.points[:, 1],
            c=activities,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=22,
        )
        axis.set_title(name)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1.15, 1.15)
        axis.set_ylim(-0.65, 0.65)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="normal-flow activity")
    figure_path = figure_dir / "synthetic_cube_activity.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "json": str(json_path),
        "figure": str(figure_path),
        "summaries": summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../2026-05-DVS-ENACT-Paper"),
        help="Root of the paper repository for small generated artifacts.",
    )
    args = parser.parse_args()
    result = run(args.output_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
