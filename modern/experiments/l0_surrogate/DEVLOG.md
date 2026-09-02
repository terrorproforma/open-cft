# L0 surrogate experiment development log

## 2026-09-02 02:36 AEST

- Added a hash-pinned campaign using the accepted L0 sweep, exact ARD
  Matérn-5/2 GP, variance calibrator, and active-learning candidate scorer.
- Added whole-group interpolation, boundary, OOD, and calibration partitions;
  a fixed-sequence comparator; bounded 32-to-96-row learning curves; strict
  model, benchmark, campaign, dataset, and runtime artifacts.
- Ran the real campaign: 96 rows, budget exhausted, RMSE/worst-error gates
  passed, interval-coverage gate failed. Active error was materially lower than
  the fixed comparator. No thresholds were changed.
- Added tests for predeclaration identity, split leakage, model reload hashes,
  exact-GP bounds, selected-index uniqueness, honest pass/fail derivation, and
  full deterministic replay.
- Runtime validation recorded by this entry: campaign completed in 50.25 s
  diagnostic wall time; focused tests and broader compatibility checks are
  recorded after completion below.

## 2026-09-02 validation completion

- Focused experiment tests: 6 passed in 52.47 s, including a complete
  deterministic campaign replay.
- Compatible accepted surrogate/active-learning tests: 134 passed in 0.66 s.
- `python -m compileall -q experiments/l0_surrogate
  tests/experiments/l0_surrogate src`: passed.
- `git diff --exit-code -- FYP` and `git status --short -- FYP`: clean.
- No packages were installed and no commit was created.
