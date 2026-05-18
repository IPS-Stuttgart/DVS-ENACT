# EventVOT confidence/memory-gate replay

`run_eventvot_confidence_gate_replay.py` reuses an existing EventVOT
DVS-ENACT diagnostics JSON and rewrites tracker result files without recomputing
contour refinements. It is intended for the setting where DVS-ENACT is used as a
confidence and short-term memory signal, not only as a coordinate refiner.

The script first evaluates the stored DVS-ENACT candidate with the same replay
acceptance policy used by `run_eventvot_acceptance_replay.py`. Confident frames
can either write the DVS-refined box or, with `--confidence-only`, keep the
external tracker's box. Rejected frames normally pass the external tracker's box
through, but selected geometry-disagreement rejection reasons can trigger a
short-term memory prediction instead.

## Typical validation workflow

First run the normal EventVOT refiner and keep its diagnostics JSON. Then replay
the diagnostics as a confidence/memory gate on the validation split:

```powershell
python scripts/run_eventvot_confidence_gate_replay.py `
  --diagnostics-json outputs\eventvot-validation\HDETrackV2_DVSENACT_diagnostics.json `
  --eventvot-root D:\Uni-Data\EventVOT `
  --base-results path\to\hdetrack_eventvot_validation `
  --output-results outputs\eventvot-confidence-gate-validation `
  --split val
```

For a pure confidence/memory ablation that never writes DVS-refined coordinates,
add `--confidence-only`:

```powershell
python scripts/run_eventvot_confidence_gate_replay.py `
  --diagnostics-json outputs\eventvot-validation\HDETrackV2_DVSENACT_diagnostics.json `
  --eventvot-root D:\Uni-Data\EventVOT `
  --base-results path\to\hdetrack_eventvot_validation `
  --output-results outputs\eventvot-confidence-only-validation `
  --split val `
  --confidence-only
```

Only lock a policy on validation data before applying it to the held-out test
split.

## Main controls

- `--confidence-only`: use DVS-ENACT only as confidence evidence; confident
  frames keep the external tracker box.
- `--gate-motion-model hold|constant_velocity`: choose how the memory box is
  predicted when a frame is gated.
- `--max-consecutive-memory-frames`: prevent long-term drift by limiting how many
  consecutive frames may use memory fallback.
- `--gate-rejection-reason`: configure which acceptance rejection reasons trigger
  memory fallback. The default is limited to geometry-disagreement reasons such
  as `candidate_iou`, `candidate_area_ratio`, and `center_shift_ratio`.

The output directory contains EventVOT-compatible `x,y,width,height` result files
plus `confidence_gate_replay_summary.json`. Use `--decisions-csv` to store the
per-frame gate decisions for diagnostic attribution.
