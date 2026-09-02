# L1a field-surrogate v3 devlog

## Preregistration

- Created an isolated branch/worktree from current `origin/feat/sota-foundation`.
- Preserved the main workspace, accepted/shared packages, FYP, and prior
  experiments.
- Separated future fine-label materialization by method, calibration, and
  assessment freeze boundaries.
- Reused the physical source at both grids through conservative dual-cell
  integration instead of widening the coarse source.
- Added geometry alignment, weighted adaptive-rank POD, transformed scalar
  discrepancy models, staged counters, dependency locking and strict terminal
  validation.

## Execution

- Initial pushed preregistration `1f53e7b` was amended before lock creation
  because dependency-lock reconstruction compared runtime tuples with sealed
  JSON lists. Final pushed preregistration is
  `98bba6344b8422c918ab091eb593d09bd693b143`.
- The single lock-claimed execution completed all 144 candidate+method coarse
  solves and all 144 corresponding fine solves.
- Method selection then failed during the first model fit with
  `NameError: name 'math' is not defined` in the preregistered runner.
- No calibration or assessment fields were materialized; both future-label
  access counters remained zero.
- The exclusive lock is retained. No code patch or lock-claimed rerun was
  performed.
