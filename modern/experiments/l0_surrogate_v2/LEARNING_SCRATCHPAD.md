# L0 surrogate v2 learning scratchpad

Policy: `COMMITTED`, isolated to the v2 experiment.

## Preregistration guardrails — 2026-09-02

- [user] Preserve v1 unchanged as failed development evidence.
- [user] Commit and push the protocol before any final assessment execution.
- [user] Acquisition may consume training observations and input-only signals,
  never calibration or assessment labels/scales.
- [self] Run all campaigns to exactly 96 rows and load final assessment once,
  only after selection, model, and calibration hashes are frozen.
- [self] Coverage must be evaluated independently for each output and each
  interpolation/boundary/OOD/overall scope across all three group-split
  replicates.
- [tool] Use PowerShell statement separators compatible with this host; do not
  use `&&`.
