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

## Synthetic Tracker Benchmark

The controlled tracker-level benchmark generates a moving rectangle with known
ground-truth extent and normal-flow contour events.

```powershell
$env:PYTHONPATH = "src;../PyRecEst/src"
python scripts/run_synthetic_tracker_comparison.py
```

It compares vanilla PyRecEst `FullSCGPTracker` with `DVSFullSCGPTracker` on
side-pair and one-sided synthetic event support before moving back to noisier
real-data validation.

## MEVDT Real-Data Validation

MEVDT is the first real-data target. It provides stationary DAVIS 240c traffic
sequences, sequence-long event CSV files, and vehicle tracking annotations with
track IDs and bounding boxes.

Dataset page: <https://deepblue.lib.umich.edu/data/concern/data_sets/bc386k045>

DOI: <https://doi.org/10.7302/d5k3-9150>

Download and extract `MEVDT.zip` outside git, for example into
`D:\Uni-Data\MEVDT-one-sequence` for a single-sequence probe or a full
`D:\Uni-Data\MEVDT` extraction. The repository ignores `data/`, `downloads/`,
and `outputs/`.

For a compact first probe from the archive:

```powershell
New-Item -ItemType Directory -Force -Path D:\Uni-Data\MEVDT-one-sequence
tar -xf D:\Uni-Data\MEVDT.zip -C D:\Uni-Data\MEVDT-one-sequence `
  sequences/test/Scene_A/1581956422501835936/1581956422501835936_events.csv `
  labels/tracking_labels/test/Scene_A/1581956422501835936/1581956422501835936-coco.json `
  labels/tracking_labels/test/Scene_A/1581956422501835936/1581956422501835936-custom24.txt `
  labels/tracking_labels/test/Scene_A/1581956422501835936/1581956422501835936-mot24.txt
```

```powershell
$env:PYTHONPATH = "src;../PyRecEst/src"
python scripts/run_mevdt_support_diagnostics.py --dataset-root D:\Uni-Data\MEVDT-one-sequence
```

The diagnostic uses labels/tracks for object association and rough bounding-box
extent. It measures whether events inside each track box concentrate on motion-
active side bands and whether the event cloud collapses relative to the labeled
box. It currently uses sequence-long `*_events.csv` files rather than the
pre-windowed `.aedat` samples. MEVDT boxes are treated as rough validation
signals, not precise contour ground truth.

The first label-assisted tracker comparison uses the same extracted sequence:

```powershell
$env:PYTHONPATH = "src;../PyRecEst/src"
python scripts/run_mevdt_tracker_comparison.py --dataset-root D:\Uni-Data\MEVDT-one-sequence
```

This compares vanilla PyRecEst `FullSCGPTracker` with `DVSFullSCGPTracker` on
the same label-cropped event windows. It is a stress test of extent stability,
not an autonomous tracking benchmark.

## Development

```powershell
$env:PYTHONPATH = "src;../PyRecEst/src"
python -m pytest
```

The `../PyRecEst/src` entry is useful for local development in this workspace.
For normal installation, use the package dependencies in `pyproject.toml`.
