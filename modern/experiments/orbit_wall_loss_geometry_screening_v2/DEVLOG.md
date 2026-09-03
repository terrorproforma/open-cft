# Devlog (orbit wall-loss geometry screening v2)

## 2026-09-03 - code, tests, shakedown

- Modules: `sobol.py` (scrambled Sobol, dependency-free), `cells.py` (catalogue binding, launch
  cells, strata, launches, allocation rule, control selection, estimators), `designs.py` (v1
  re-solve pipeline reused by import, refined for every design; P2 row via the v4 adapter),
  `experiment.py` (plan, authorities, per-design worker, allocation replay, gates, dataset, CSV),
  `consumer.py` (v1 consumer reused; hash-bound v1 dataset loader), `run.py`, `protocol.json`.
- Shakedowns: 1 `prebundle_failure` (uint64 seeds in canonical JSON), 2 `runtime_failure`
  (launch-id grammar), 3 `runtime_failure` (orbit_mc v1.7 Wilson ordering defect at n = 6), 4
  `accepted_result`: 4 designs, 14 cells, 34 cases, 6 topped-up / 8 saturated cells, allocation
  replay passed, 40 control launches with zero discordance, bundle validated (578 artifacts,
  1.5 MB), 113 s. Timing projection: expected 8819 s wall (72 % top-up, 1.35 contention, 12
  workers), worst case 11164 s.
- Tests: `modern/tests/experiments/orbit_wall_loss_geometry_screening_v2` (43 passed, results
  tests skipped before execution).
