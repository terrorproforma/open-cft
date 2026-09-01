# Axisymmetric Field Workstream Development Log

## 2026-09-01 — L1a implementation

### Implemented

- Added immutable SI-explicit domain, source-band, solver, diagnostics, and
  field-map models under the independent `cft_revival.fields` package.
- Derived and implemented the `psi=r A_phi` constant-permeability equation as
  a conservative, symmetric, node-centred second-order FDM operator.
- Added dependency-free Python Jacobi-PCG and real Warp 1.14 float64
  matrix-free kernels for CPU and CUDA.
- Required separately recomputed true-residual acceptance, finite publication,
  bounded iteration count, residual history, and typed validation/failure.
- Added second-order field recovery, analytic axis limit, flux-reconstruction
  identity diagnostics, manufactured-solution convergence, and backend parity.
- Added deterministic versioned artifacts for three explicitly hypothetical
  equivalent-current geometries and a dashboard-facing JSON contract.

### Verification

- Focused field tests: 13 passed, including Warp CPU and RTX 5090 CUDA.
- Manufactured `psi` order: 2.00468 then 2.00116.
- Manufactured recovered-field order: 2.00420 then 2.00591.
- Three-design Python/Warp CPU/CUDA maximum scale-relative field difference:
  below `2.34e-14`.
- Published designs: 259-285 PCG iterations, true relative residual below
  `9.67e-11`, max flux-reconstruction identity defect below `2.78e-15 T/m`.
- Dual-cell source transfer conserved signed requested ampere-turns to within
  `1.28e-11 A-turn` per band (`2.73e-11 A-turn` per complete design) in the
  generated designs.

### Scope controls

- Did not edit shared package, CLI, README, shared devlog/scratchpad,
  visualization, optimization, existing physics, FYP, or Git state.
- Did not install dependencies, benchmark speed, commit, or push.
- Inspected Warp FEM examples but did not call this structured-grid method FEM.
- Deferred permanent magnets, material interfaces, nonlinear B-H, open
  boundaries, plasma response, FEMM parity, and MFEM comparison.

## 2026-09-01 — L1a audit hardening

### Corrected

- Replaced permissive artifact checks with closed recursive runtime schemas and
  versioned JSON contracts. Unknown keys, wrong types, nonfinite/nonmonotonic
  coordinates, shape damage, false convergence, residual-policy violations,
  and corrupted `|B|` are rejected.
- Added non-recursive canonical payload SHA-256, filename-bound raw file
  SHA-256 sidecars, manifest artifact file/payload anchors, safe plain-path
  resolution, and tamper/path-substitution tests.
- Enforced explicit interior dual-cell source support and at least two grid
  spacings per band dimension. Boundary-clipped and underresolved bands now
  fail before sampling; artifacts report area/centroid/touched-cell transfer
  errors and signed ampere-turn balance.
- Added finite positive span/spacing/spacing-squared/inverse-spacing and
  binary64 coordinate-collapse gates.
- Added degenerate near-zero topology plus distinct sign-changing, isolated,
  plateau, and outer-boundary-minimum states.
- Changed false recursive PCG crossings to bounded true-residual restarts with
  published restart/stagnation diagnostics.
- Renamed the former divergence diagnostic to a flux-reconstruction identity,
  explicitly removing any independent-validation claim.
- Removed the fields/physics `test_spec_ledgers.py` module-name collision by
  renaming the fields-owned test module.

### Verification

- Added adversarial regressions for every reproduced audit defect.
- Regenerated all three deterministic artifacts, payload hashes, raw file hash
  sidecars, and the hash-anchored manifest.
- Focused fields suite: 39 passed.
- Fields plus the 21 Git-tracked prior test modules: 287 passed with one
  unchanged optional-extension skip in both normal and importlib modes.
- Fields plus physics collection: 125 passed in both modes, directly proving
  the duplicate-basename collection defect is gone.
- All-design Python/Warp CPU/RTX 5090 parity remained below `2.34e-14`.
- Native CTest 1/1 and compileall passed; FYP remained unchanged.
- The repository-wide suite reached 664 passed and one skip, but the unowned
  visualization workstream is pinned to the superseded manifest schema/hash,
  and concurrently added coupling/plasma tests each had one unrelated failure.
  Those paths were not edited.

## 2026-09-02 — Final solver-owned boundary correction

- Replaced raw minimum-width subtraction with one shared radial/axial
  half-ULP endpoint/target envelope. `[0.04,0.06]` at `0.01 m` spacing now
  passes as exactly two represented spacings; moving either upper endpoint one
  binary64 value inward fails.
- Kept represented resolution explicit through radial/axial nodes and total
  dual cells touched in each source-discretization record.
- Added `FieldArtifactValidationError`, a typed
  `FieldValidationError`/`ValueError` subclass, for every artifact and manifest
  contract failure.
- Wrapped huge-integer `float()` overflow and JSON decoder integer-limit
  errors; no raw `OverflowError` escapes strict parsing.
- Added exact-boundary, nextafter-narrower, huge in-memory number, and
  5,000-digit JSON integer regressions.
- Final focused fields suite: 42 passed; dedicated adversarial/artifact
  selection: 30 passed.
- Opposed-cusp parity spot: Warp CPU `1.79e-14`, RTX 5090 CUDA `1.35e-14`
  maximum scale-relative field difference; both completed 259 iterations with
  true relative residual below `8.98e-11`.
