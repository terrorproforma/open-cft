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
