# DEVLOG — wall-loss geometry surrogate v2

## 2026-09-03 — design and shakedown

- Branch `exp/wall-loss-geometry-surrogate-v2` from `origin/feat/sota-foundation` @ `bfe123d4`.
- Reuse policy: v1's `data.py` / `models.py` / `predictor.py` / `experiment.py` are imported and
  hash-bound (code contract scope); v2 adds `features.py` (derived features), the mixture
  candidate, the gbt baseline, the dispatching predictor, the learning curve and the v1 comparison.
- Feature probe on the 96 designs: `chamber_length == stage_count * pitch` exactly, axis cusp count
  == stage count, null count == stage count + 1 (all excluded as exact duplicates). Polarity varies
  (40 / 56). Mirror ratios span 4 .. 2672 -> log10. Continuous features have >= 69 distinct values.
- Fit-role stage counts 13 / 29 / 8 -> the mixture rule (>= 8 per count) qualifies all three counts.
- Method-selection probe (before the protocol was frozen; informational): stgp-logit pooled RMSE
  0.045, knn 0.050, mean 0.056, gbt grid 0.057-0.065 mean-over-outputs. The pooled P(wall) has a
  small spread across designs, so the 2x-baseline gate is intrinsically hard; kept verbatim anyway.
- In-sample ridge (v1's tautology check (c)) with 32 coefficients on 50 designs falls BELOW the
  binomial floor for cell1 / pooled without any tautology -> replaced by leave-one-out ridge
  (declared in the protocol before the shakedown).
- Shakedown (namespace `wall-loss-geometry-surrogate-v2:shakedown-partition`, seed 900002):
  `accepted_result`, bundle validated, all structural gates pass, replay 1e-15, 21 s. Informational
  science numbers are recorded in `shakedown.json` only and are not evidence.

## 2026-09-03 — preregistration, execution, result

- Code/tests commit `21118507`; `prepare` wrote `partitions.json` byte-identical to v1's (semantic
  hash checked against the protocol's `v1_binding`); preregistration commit `503bf87f`, pushed.
- One detached execution from a clean worktree at `503bf87f` (CPU, 8 threads, `.venv-sota`):
  development ~40 s, assessment single-use, learning curve 3 seeds × 4 sizes. Result commit
  `a2b503be` (`results/` force-added past the ignore rule, as in v1). Terminal
  `assessment_rejection`, `rejected_surrogate`, `not_usable_as_mdo_v2_input_rejected_surrogate`.
- Gates: pooled 0.0337 PASS; cells floor-corrected 0.0836 FAIL; 2× baseline 0.99× FAIL (ridge
  0.0334); coverage 0.8125 PASS; extrapolation pooled 0.1027 (reported gate 0.10 missed by 0.003).
  Structural: all pass (determinism bit-exact, replay 1e-15, LOO ridge above floor).
- v1 comparison on the identical 16 designs: pooled 0.0562 → 0.0337, cells 0.1332 → 0.0904,
  every output improved. Tree baseline did not beat the GP (0.0368 vs 0.0337 pooled).
- Learning curve flat from 30 designs (0.0455 → 0.0449 at 50); the power-law "23 designs to 0.05"
  is a statement about the method-selection role, not the assessment role, and is reported as such.
- Dashboard: `generate_wall_loss_geometry_surrogate_v2_dashboard.py` + template (v1 bundle
  verifier and per-design recomputation imported from the v1 generator), 8 tests; the v1 numbers
  on the dashboard are read from v2's hash-bound `v1-comparison.json` and cross-checked in the test
  against v1's committed `assessment.json`. Headless Edge and Chrome: 0 console errors,
  screenshots under `%TEMP%\wlgs2-dashboard`.
