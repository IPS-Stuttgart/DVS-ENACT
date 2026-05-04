# DVS-ENACT

Event-based Normal-flow Active Contour Tracking for extended objects observed by
Dynamic Vision Sensor (DVS) / event-camera measurements.

This repository is for implementation and evaluation code. Small generated
results, figures, notes, and manuscript text belong in
`FlorianPfaff/2026-05-DVS-ENACT-Paper`.

## Core Idea

DVS measurements are not uniformly sampled from an object's extent. They are
mostly generated at brightness contours, and only where the apparent motion has
a component along the local contour normal. DVS-ENACT models this with an
activity term

```text
activity(phi) = |n(phi)^T v| / ||v||
```

where `n(phi)` is the contour normal and `v` is the image-plane velocity. This
keeps inactive boundaries from being interpreted as missing shape evidence.

## Repository Roles

- `src/dvs_enact`: reusable tracking and active-contour code.
- `scripts`: repeatable evaluation scripts.
- `tests`: focused unit tests for the DVS-specific modeling assumptions.

## MEVDT Real-Data Validation

MEVDT is the first real-data target. It provides stationary DAVIS 240c traffic
sequences, sequence-long event CSV files, and vehicle tracking annotations with
track IDs and bounding boxes.

Dataset page: <https://deepblue.lib.umich.edu/data/concern/data_sets/bc386k045>

DOI: <https://doi.org/10.7302/d5k3-9150>

Download and extract `MEVDT.zip` outside git, for example into
`data/raw/MEVDT`. The repository ignores `data/` and `downloads/`.

```powershell
$env:PYTHONPATH = "src;../PyRecEst/src"
python scripts/run_mevdt_support_diagnostics.py --dataset-root data/raw/MEVDT
```

The diagnostic uses labels/tracks for object association and rough bounding-box
extent. It measures whether events inside each track box concentrate on motion-
active side bands and whether the event cloud collapses relative to the labeled
box. MEVDT boxes are treated as rough validation signals, not precise contour
ground truth.

## Development

```powershell
$env:PYTHONPATH = "src;../PyRecEst/src"
python -m pytest
```

The `../PyRecEst/src` entry is useful for local development in this workspace.
For normal installation, use the package dependencies in `pyproject.toml`.
