# DVS-ENACT

Event-based Normal-flow Active Contour Tracking for extended objects observed by
Dynamic Vision Sensor (DVS) / event-camera measurements.

This repository is for implementation and evaluation code. The separate paper
repository has not moved with the code repository; small generated results,
figures, notes, and manuscript text belong in
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

## Post-hoc Tracker Refinement

`DVSContourRefiner` is the lightweight integration point for comparing an
external tracker against the same tracker plus DVS-ENACT physics. The external
tracker still proposes the box; DVS-ENACT only crops events around that proposal
and performs one polarity-aware contour update.

```python
from dvs_enact import DVSContourRefiner, DVSContourRefinerConfig

refiner = DVSContourRefiner(
    DVSContourRefinerConfig(
        search_expansion_factor=1.25,
        max_events=128,
        use_event_polarity=True,
    )
)

# candidate_bbox and previous_bbox are external-tracker outputs. event_window is
# the EventBatch between the previous and current benchmark timestamps.
result = refiner.refine(
    candidate_bbox,
    event_window,
    previous_candidate_bbox=previous_bbox,
)
refined_bbox_xyxy = result.as_xyxy()
```

This supports ablations such as `Tracker X` versus `Tracker X + DVS-ENACT`
without changing the external tracker architecture. `result.to_dict()` includes
the cropped event count, active-measurement count, mean normal-flow activity,
polarity-consistency diagnostics, and fallback reason when refinement is skipped.

For EventVOT-style single-object tracking result files, use the dedicated
adapter. It preserves the official `x,y,width,height` result convention:

```powershell
python scripts/run_eventvot_refinement.py `
  --eventvot-root D:\Uni-Data\EventVOT `
  --base-results path\to\hdetrack_eventvot `
  --eventvot-toolkit-root path\to\EventVOT_eval_toolkit `
  --tracker-name HDETrackV2_DVSENACT `
  --update-config-tracker `
  --split test
```

This writes official evaluator files below
`eventvot_tracking_results/HDETrackV2_DVSENACT_tracking_result/` and adds the
tracker to `utils/config_tracker.m`.

The EventVOT adapter is conservative by default. It only replaces the base
tracker box when DVS-ENACT has no fallback reason, at least 10 used events, at
least 3 active measurements, mean activity of at least 0.10, IoU of at least
0.60 with the base box, and refined/base area ratio of at most 1.50. Rejected
frames keep the base tracker box and are recorded in the diagnostics JSON.

Tune EventVOT refinement parameters on the validation subset only:

```powershell
python scripts/run_eventvot_validation_sweep.py `
  --eventvot-root D:\Uni-Data\EventVOT `
  --base-results path\to\hdetrack_eventvot_validation `
  --output-root outputs\eventvot-validation-sweep `
  --split val
```

The sweep refuses the test split unless `--allow-test-split` is supplied for a
final held-out evaluation. It ranks configurations by validation SR AUC, with
PR, NPR, and refinement acceptance rate written as secondary diagnostics.

After locking the validation-selected configuration and running the held-out
EventVOT test set, generate the paper comparison table and attribute-level
gain report:

```powershell
python scripts/report_eventvot_comparisons.py `
  --eventvot-root D:\Uni-Data\EventVOT `
  --result-root path\to\EventVOT_eval_toolkit\eventvot_tracking_results `
  --eventvot-toolkit-root path\to\EventVOT_eval_toolkit `
  --output-root outputs\eventvot-paper-report `
  --split test
```

By default this expects result directories for `HDETrackV2`,
`HDETrackV2 + DVS-ENACT`, `OSTrack-event`, `OSTrack-event + DVS-ENACT`, and
`DVS-ENACT-only`. The report writes the main `SR/PR/NPR/FPS` table, pairwise
strong-tracker-to-refined gains, and attribute-level gains for EventVOT
challenge factors.

## Benchmark Roadmap

EventVOT is the first benchmark target for tracker-plus-physics claims. Only
after reproducing the official EventVOT baseline and reporting the held-out
strong-tracker-to-refined comparison should the adapter be ported to VisEvent
or FELT. The next-stage guardrails are documented in
[`docs/tracking_benchmark_roadmap.md`](docs/tracking_benchmark_roadmap.md).

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

<!-- markdownlint-disable MD013 -->
```powershell
New-Item -ItemType Directory -Force -Path D:\Uni-Data\MEVDT-one-sequence
tar -xf D:\Uni-Data\MEVDT.zip -C D:\Uni-Data\MEVDT-one-sequence `
  sequences/test/Scene_A/1581956422501835936/1581956422501835936_events.csv `
  labels/tracking_labels/test/Scene_A/1581956422501835936/1581956422501835936-coco.json `
  labels/tracking_labels/test/Scene_A/1581956422501835936/1581956422501835936-custom24.txt `
  labels/tracking_labels/test/Scene_A/1581956422501835936/1581956422501835936-mot24.txt
```
<!-- markdownlint-enable MD013 -->

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

The filtered MEVDT sweep removes boundary-touching, tiny, rapidly changing, and
track-end windows before rerunning the comparison across a compact parameter
sweep:

```powershell
$env:PYTHONPATH = "src;../PyRecEst/src"
python scripts/run_mevdt_filtered_tracker_sweep.py --dataset-root D:\Uni-Data\MEVDT-one-sequence
```

## Development

```powershell
$env:PYTHONPATH = "src;../PyRecEst/src"
python -m pytest
```

The `../PyRecEst/src` entry is useful for local development in this workspace.
For normal installation, use the package dependencies in `pyproject.toml`.
