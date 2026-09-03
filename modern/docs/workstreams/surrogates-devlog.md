# Surrogate Workstream Devlog

## 2026-09-01

- Established a new runtime only under `cft_revival.surrogates`; no dependency
  on the existing optimization package was introduced.
- Confirmed NumPy 2.5.2 is available in the working environment, then retained
  and tested a standard-library Cholesky path so NumPy remains optional.
- Implemented finite validation, reversible input/output normalization, exact
  Matérn-5/2 scalar GPs, independent multi-output fitting, deterministic bounded
  likelihood selection, and a bounded Cholesky jitter ladder.
- Added per-observation heteroskedastic noise as measurement variance. It stays
  distinct from latent predictive variance and discrepancy.
- Added an AR1 two-fidelity model with a bounded scale and an independent
  discrepancy GP. Documented the variance-independence approximation.
- Added strict JSON persistence with training/schema hashes and deterministic
  refit verification instead of pickle.
- Added grouped spatial holdout, RMSE/MAE/worst-case/coverage reporting,
  residual-quantile variance calibration, and normalized OOD distances.
- Added a lazy BoTorch training-data contract without installing torch,
  GPyTorch, or BoTorch.
- Added fixed-mesh POD coefficient surrogates. Rejected a neural-operator label
  because no neural operator was implemented or evidenced.
- Trained/evaluated the executable benchmark test on deterministic hypothetical
  L0 outputs and preserved explicit non-physical interpretation.
- Added adversarial coverage for duplicate points, clustered ill-conditioning,
  large output scales, nonfinite values, negative variances, hash tampering,
  deterministic reload, fixed-mesh mismatch, and pure-Python factorization.
- Final focused result: 16 surrogate tests passed; surrogate source/tests and
  the full modern tree compiled.
- The final importlib-mode repository run reported 445 passes, one optional
  extension skip, and two unrelated failures in `tests/plasma/test_solver.py`
  where residuals narrowly missed that workstream's configured tolerance.
  Surrogate-owned tests remained green. The default pytest importer also has
  unrelated same-basename collection collisions in unowned test directories.
- The compatible remainder, run in importlib mode with only that external
  failing plasma solver module excluded, passed 460 tests with one optional
  extension skip.
- Ruff and mypy executables were not installed in this environment, so their
  checks were not represented as completed.
- `git diff -- FYP` was empty, and scoped status listed only the allowed new
  surrogate source, test, specification, and workstream-document paths.

## 2026-09-01 audit closure

- Replaced partial GP persistence with artifact schema v2. The canonical model
  hash now covers raw/normalized data, affine bounds, noise, kernel/version,
  ARD vector/mode, hyperparameter policy, fitted values, full/selected jitter,
  calibration probability/provenance, OOD policy, and output semantics.
- Made strict loaders reject unknown keys, unsupported policy constants,
  duplicate keys, nonfinite JSON, hash mismatch, and deterministic-refit drift.
  Added composed AR1 and POD artifact hashes.
- Replaced scalar-only length scale fitting with true ARD vectors and retained
  explicit isotropic mode; added anisotropic recovery and round-trip tests.
- Added `SurrogateError` and finite guards around prediction, intervals,
  calibration, OOD, POD and metric aggregation, including extreme extrapolation.
- Added explicit nominal probability, coverage target/tolerance/acceptance,
  sample count, and separate calibration-fit/held-out-assessment labels.
- Closed split leakage by transitive caller-group plus exact-coordinate grouping.
  Signed zero canonicalizes identically; nonzero near-coordinate tolerance is
  rejected pending unit-aware upstream grouping.
- Added backend-independent mean-only rank-0 POD for zero-energy fields,
  deterministic serialization, orthonormal-basis validation, and fixed-mesh
  signed-zero identity tests.
- Froze one authoritative L0 software benchmark with source-config, dataset,
  split, configuration and complete-result hashes. Previous unhashed exploratory
  metrics are explicitly superseded.
- Audit verification: 37 surrogate tests passed. The authoritative L0 benchmark
  test passed twice with identical frozen hashes and metrics; the 25 exact-GP
  serialization/property tests passed; full `src` and `tests` compileall passed.
- The concurrent full repository snapshot had 632 passes, one skip, 14 failures
  and 10 errors in unowned coupling/fields/plasma/visualization paths. After
  excluding those known external files, one unowned magnetics extreme-scale
  tolerance failed by `3.5e-114`. The remaining compatible suite passed 586
  tests with one optional-extension skip.

## 2026-09-02 acceptance-semantics closure

- Split immutable software evidence from model quality:
  `software_reproducibility_passed` now requires all four expected hashes,
  while `model_quality_passed` requires independent RMSE and coverage gates.
- Replaced the broad `0.30` coverage smoke tolerance with a predeclared `0.05`
  absolute tolerance and minimum held-out sample count of 30. The current
  12-point assessment is explicitly `assessment_limited=true`; neither output's
  coverage is accepted.
- Added one output-independent engineering gate:
  range-normalized RMSE over the complete hashed dataset must be at most 5%.
  Thrust is `22.687%`; Isp is `20.660%`; both fail without threshold tuning.
- Updated the benchmark configuration/result hashes and froze exact false
  quality statuses. “Authoritative” now means reproducible only, and the
  benchmark explicitly requires surrogate improvement.
- Final acceptance verification: all 37 surrogate tests passed; the pinned
  benchmark passed twice with exact hashes; compileall and surrogate JSON
  parsing passed. Concurrent unowned validation tests changed constructor
  contracts during the run, so the stable compatible remainder excluded the
  already documented external failures plus that validation directory and
  passed 571 tests with one optional-extension skip.

## 2026-09-03 - wall-loss geometry surrogate v1 (first surrogate campaign on physically varying data)

- `modern/experiments/wall_loss_geometry_surrogate_v1`: preregistered one-shot
  GP surrogate of the geometry wall-loss screening dataset (96 designs, 11
  declared inputs, per-cell/pooled wall-hit and reflection probabilities with
  KNOWN binomial noise). Candidates: package `ExactGP` (logit/direct), BoTorch
  `SingleTaskGP` fixed-noise Matern-5/2 ARD (logit/direct), BoTorch
  `MultiTaskGP` ICM over the four cells; baselines mean / k-NN(3) / ridge.
  Roles frozen before any fit (50 fit / 10 method-selection / 10 calibration /
  16 assessment + 10 top-decile chamber-length extrapolation hold-out).
- Recorded outcome (`b400d924`, terminal `assessment_rejection`,
  `rejected_surrogate`): selected `botorch-icm-logit`; pooled P(wall) RMSE
  0.0562 (gate 0.05; ridge baseline 0.0546, global mean 0.0553 - the
  surrogate did NOT beat the trivial baselines); floor-corrected cell RMSE
  0.129 (gate 0.05); 90 % coverage 0.80 (gate met at the boundary, after a
  3.34x variance inflation from the calibration role); extrapolation pooled
  RMSE 0.093 (reported gate 0.10 met), coverage 0.84. All structural gates
  passed (dataset/Git binding, single-use labels, no tautology, bit-exact
  determinism, predictor-contract replay 3e-14, code contract).
- Reading: with 50 fit designs in 11-D and step discontinuities in the
  design -> geometry map (`stage_count_selector`, `exit_length_fraction`
  dominate the permutation importance), the GP has no advantage over a
  linear model for the pooled probability and cannot resolve the cell-level
  structure (label sd 0.18-0.25, floor 0.035). Method selection on 10
  designs was unstable (the shakedown partition selected `botorch-stgp-direct`).
- The package still has no ICM/LMC kernel; the coregionalised candidate used
  BoTorch and its posterior is reproduced by the numpy predictor contract.
