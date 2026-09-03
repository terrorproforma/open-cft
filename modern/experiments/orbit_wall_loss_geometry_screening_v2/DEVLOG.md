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

## 2026-09-03 - preregistration, execution, post-hoc manifest, record

- Commits: `a7a884bf` code/tests (rebased onto `beb4772c`), `cef1ee59` preregistration
  (authorities, design authorities, shakedown record; path-isolated), pushed; detached run
  worktree `uni-project-orbit-geo2-run` at `cef1ee59`; `execute` 21:03-22:29 AEST (fields
  ~8 min, orbits 70 min, assessment 80 min; the projection was 147 min because only 31 % of the
  cells were topped up against the planned 72 %).
- Runtime failure AFTER the terminal record: `manifest.json` publication hit EMFILE (16,957
  files vs the 8192-descriptor CRT cap). Recovery module
  `cft_revival.experiment_runtime.recovery` + `MAX_PINNED_DESCRIPTORS` cap in the finalizer +
  4 tests; manifest published post hoc (`876dc7e1...`), `validate` -> `accepted_result`,
  16,968 artifacts; results commit `26029b72` (results only). Disclosure:
  `POSTHOC_FINALIZATION.md`.
- Results tests: 51 passed (incl. 7 post-execution). Dashboard
  `modern/visualization/wall-loss-geometry-screening-v2.html` (776,625 B) + 4 tests; headless
  screenshots under `%TEMP%\owlgs2_dashboard\`.
