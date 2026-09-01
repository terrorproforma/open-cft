# Migration and Traceability

## MATLAB-to-modern map

- `CFTOpt.m` → `models.DesignPoint` for legacy compatibility and
  `optimization.Design`/campaign spec v1.4 for the new multi-fidelity boundary.
  Decision ranges and objective directions are captured; external SAEA
  behavior and archived objective values are not treated as a correctness
  oracle.
- `Performance_est.m` → `kernels.calculate_performance`,
  `pipeline.evaluate_design`. Only dimensional post-processing is translated;
  categorical failure codes will become typed statuses.
- `FEMMrun.m` → `backends.MagneticFieldBackend` and planned isolated Windows
  FEMM worker. Geometry generation is not yet translated.
- `cusp_prob.m` → `backends.FemmExportBackend`,
  `kernels.legacy_cusp_fields`, `kernels.cusp_arrival_probabilities`, and
  `native/src/kernels.cpp`; Phase 2A adds the optional batch implementation in
  `warp_backend.py`.
- `HEMP_solver.m` → `backends.PlasmaBackend` remains the compatibility
  quarantine. The accepted `plasma` package is a corrected, source-ledgered
  global-discharge numerical foundation, not a claim that the historical
  solver or a predictive CFT model has been reconstructed.
- `Power_B_EQs.m` → planned C++ residual/Jacobian module after re-derivation.
- `boundaries.m` → `DesignPoint` handles geometry only; plasma-variable bounds
  await the corrected equation specification.
- `params.m` → `config/default.json` for implemented settings; optimizer
  settings await selection of a maintained Python optimizer.
- `buildSurrogates.m` → the accepted `surrogates` runtime supplies versioned
  identities, dependency-free GP/POD/multifidelity foundations, optional
  NumPy/Torch boundaries, and held-out metrics. The present quality benchmark
  failed its acceptance gates and therefore does not replace the legacy or
  high-fidelity models.
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

## Integrated foundation additions

- Added `physics.XenonOperatingPoint` and
  `physics.evaluate_performance` as an independent L0 conservation layer. It
  does not call `pipeline.evaluate_design` or the quarantined legacy plasma
  residual.
- Added checked hypothetical point/sweep configs and JSON result schemas. The
  deterministic sweep can run on Python, Warp CPU, or Warp CUDA and always
  computes CPU-reference parity.
- Added a strict campaign spec validator and dependency-free deterministic
  initial-design command. BoTorch is not required for schema validation,
  scheduling records, Pareto operations, or design generation.
- Kept `DesignPoint`, `L0XenonOperatingPoint`, and `OptimizationDesign`
  distinct at the shared package boundary.
- Recorded the first 8,192-point RTX 5090 L0 sweep in `FIRST_RESULTS.md`;
  timing is explicitly uncontrolled and no physical-accuracy or speedup claim
  is made.
- Added the independently accepted L1a axisymmetric equivalent-current FDM
  solver, deterministic three-design artifact bundle, and browser-tested
  schema-1.1 viewer. This is not FEM or a material-aware field solution.
- Added independent magnetics, coupling, global-plasma, prescribed-field
  hybrid, reduced PIC, surrogate, active-learning, and validation/evidence
  packages. Their contracts are not merged into one domain model, and optional
  Warp/NumPy/Torch-family packages remain outside base import.
- Added evidence-gated paper source and deterministic generated table inputs.
  Paper L1--L3 result gates remain closed; local PDF/build outputs are ignored.

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
7. Runtime-verify the optional BoTorch/GPyTorch and pymoo boundaries, calibrate
   source costs, and freeze F3-verified hypervolume policy before running an
   optimization benchmark. Do not restore 12-way FEMM parallelism on one
   desktop.

## Exit criteria for Phase 2

- domain expert signs off the equation ledger;
- corrected Python reference and C++ implementation agree at random valid
  states;
- solver passes conservation and residual thresholds on golden cases;
- every failure is classified without converting it to a fake zero objective;
- FEMM artifacts are provenance-linked and stale-file tests pass;
- baseline objective differences are explained rather than tuned away.
