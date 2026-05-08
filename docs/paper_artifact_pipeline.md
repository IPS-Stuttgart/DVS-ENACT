# Paper artifact pipeline

This document defines the canonical bridge between `IPS-Stuttgart/DVS-ENACT`
and the separate private `FlorianPfaff/2026-05-DVS-ENACT-Paper` repository.

The code repository owns experiment execution. The paper repository should
consume generated JSON evidence and figures, rather than carrying duplicate
experiment logic.

## Local synthetic-only export

From the `DVS-ENACT` repository root:

```powershell
$env:PYTHONPATH = "src;../PyRecEst/src"
python scripts/export_paper_artifacts.py --skip-mevdt
```

This writes to `../2026-05-DVS-ENACT-Paper` by default and produces:

```text
data/paper_evidence/artifact_manifest.json
data/paper_evidence/synthetic_cube_activity.json
data/paper_evidence/synthetic_count_likelihood.json
data/paper_evidence/synthetic_tracker_comparison.json
figures/synthetic_cube_activity.png
figures/synthetic_count_likelihood_counts.png
figures/synthetic_count_likelihood_nll.png
figures/synthetic_tracker_inactive_axis_ratio.png
figures/synthetic_tracker_collapse_fraction.png
```

## Local full export with MEVDT

```powershell
$env:PYTHONPATH = "src;../PyRecEst/src"
python scripts/export_paper_artifacts.py `
  --dataset-root D:\Uni-Data\MEVDT-one-sequence `
  --output-root ..\2026-05-DVS-ENACT-Paper `
  --max-windows 500
```

The MEVDT stages add:

```text
data/paper_evidence/mevdt_support_diagnostics.json
data/paper_evidence/mevdt_tracker_comparison.json
```

Use `--event-csv` and `--label-file` when a specific sequence/annotation pair
must be fixed for the paper.

## GitHub Actions export

The workflow `.github/workflows/export-paper-artifacts.yml` is manual-only
(`workflow_dispatch`) and runs on a Linux self-hosted runner.

Use synthetic-only mode first:

```text
include_mevdt = false
```

For real-data runs, place the extracted MEVDT data on the runner and set:

```text
include_mevdt = true
dataset_root = /absolute/path/to/MEVDT-one-sequence
```

When `include_mevdt = true` and `dataset_root` is empty, the workflow downloads
the MEVDT WebDAV share with rclone into:

```text
$HOME/.cache/datasets/MEVDT
```

The workflow expects these GitHub secrets to be configured:

```text
MEVDT_WEBDAV_URL
MEVDT_DATA_KEY
MEVDT_DATA_PASSWORD
```

The real-data workflow also runs the filtered MEVDT tracker sweep and writes:

```text
data/paper_evidence/mevdt_filtered_tracker_sweep.json
data/paper_evidence/evaluation_summary.json
```

The workflow uploads `paper-output/data/paper_evidence` and
`paper-output/figures` as a GitHub Actions artifact. It does not try to push
generated files into the private paper repository automatically. That keeps the
data-producing workflow auditable and avoids coupling the public code repository
to private-repository credentials.

## Manifest contract

Every run writes `data/paper_evidence/artifact_manifest.json`. The manifest
contains:

- repository head SHA when available,
- Python/runtime information,
- all command parameters,
- one record per stage,
- `succeeded`, `skipped`, and `failed` counts.

The paper repository should cite or commit the manifest alongside generated
figures and evidence JSON whenever paper results are refreshed.

## Recommended next scientific work

After this pipeline is merged, the highest-value next tasks are:

1. Add a constant-position/previous-contour baseline.
2. Add an event-cloud centroid baseline.
3. Add qualitative MEVDT contour overlays for the Results section.
4. Add a failure-mode figure covering low support, clutter, and aperture-like
   ambiguity.
