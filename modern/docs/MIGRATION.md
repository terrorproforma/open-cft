# Migration and Traceability

## MATLAB-to-modern map

- `CFTOpt.m` → `models.DesignPoint`, future `optimization/problem.py`.
  Decision ranges and objective directions are captured; external SAEA
  integration is not translated.
- `Performance_est.m` → `kernels.calculate_performance`,
  `pipeline.evaluate_design`. Only dimensional post-processing is translated;
  categorical failure codes will become typed statuses.
- `FEMMrun.m` → `backends.MagneticFieldBackend` and planned isolated Windows
  FEMM worker. Geometry generation is not yet translated.
- `cusp_prob.m` → `backends.FemmExportBackend`,
  `kernels.legacy_cusp_fields`, `kernels.cusp_arrival_probabilities`, and
  `native/src/kernels.cpp`; Phase 2A adds the optional batch implementation in
  `warp_backend.py`.
- `HEMP_solver.m` → `backends.PlasmaBackend`. Implementation is intentionally
  blocked pending equation verification.
- `Power_B_EQs.m` → planned C++ residual/Jacobian module after re-derivation.
- `boundaries.m` → `DesignPoint` handles geometry only; plasma-variable bounds
  await the corrected equation specification.
- `params.m` → `config/default.json` for implemented settings; optimizer
  settings await selection of a maintained Python optimizer.
- `buildSurrogates.m` → planned versioned dataset/training pipeline.
- `SensitivityAnalysis_Surr_rev.m` and
  `MDO_sensitivity_analysis_CFTOpt_4objectives.m` → planned reproducible
  sensitivity notebook/report operating on versioned models.
- `plotParetoOptwithColourDots.m` → planned plotting/report module using
  explicit physical objective values rather than stored sign inversions.
- `importfile.m` → `backends._read_femm_plot`.
- `findLocalMin.m`, `plasmaCalc.m`, `vline.m` → no migration until a current
  production requirement is established.

## What Phase 1 preserves

- eight-variable ordering and declared ranges;
- geometric ordering and 0.01 mm legacy clearance checks;
- legacy FEMM export filenames and two-header tolerant parsing;
- hard-coded cusp windows and p1..p4 reversal, visibly isolated as
  compatibility behavior;
- constants and SI conversions used by performance post-processing;
- objective quantities in physical sign (optimization sign conversion is not
  embedded in domain results).

## What Phase 1 deliberately changes

- invalid/missing/non-finite data raises a typed error rather than producing
  zeros or NaNs;
- cusp probability uses the analytic closed form instead of quadrature;
- no globals, current working directory dependence, or implicit unit order;
- FEMM cannot be configured as non-serialized;
- unverified plasma equations fail closed.

## Phase 2A additions

- Added publication provenance and equation-level source links without
  assuming the current snapshot generated the paper's results.
- Added a true Warp kernel that launches one loss-cone calculation per batch
  element on `cpu` or `cuda:N`, with the same host validation as scalar Python.
- Added deterministic CPU/CUDA parity tests and an explicitly
  non-authoritative smoke timing command.
- Kept the C++ and dependency-free paths unchanged; Warp is an optional
  package extra.
- Did not translate disputed plasma signs, logical constraints, sensitivity
  analysis, or FEMM into a pretend GPU solve.

## Recommended next phase

1. Obtain the exact Kornfeld paper/equations, MATLAB/FEMM versions, optimizer
   library, original data files, and at least 5 archived successful and failed
   runs.
2. Write an equation ledger: one row per unknown/residual with symbol, unit,
   source equation, expected scale, sign, and boundary.
3. Repair a copy of the residual model in the modern tree only. Pass
   `Ua`, `Ia`, probabilities, and constants explicitly; remove logical
   residuals; normalize equations; expose residual diagnostics.
4. In parallel, establish the manufactured-solution and FEMM-profile fixtures
   for a 2D axisymmetric magnetostatic solver described in `ARCHITECTURE.md`.
5. Implement the CPU C++ residual and Jacobian only after equation sign-off.
6. Build a serialized FEMM worker and immutable golden dataset. Compare field
   profiles and cusp probabilities, not only final objectives.
7. Select an optimizer after defining continuous feasibility margins and
   expensive-evaluation policy. Do not restore 12-way FEMM parallelism on one
   desktop.

## Exit criteria for Phase 2

- domain expert signs off the equation ledger;
- corrected Python reference and C++ implementation agree at random valid
  states;
- solver passes conservation and residual thresholds on golden cases;
- every failure is classified without converting it to a fake zero objective;
- FEMM artifacts are provenance-linked and stale-file tests pass;
- baseline objective differences are explained rather than tuned away.
