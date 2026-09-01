# Magnetics Workstream Development Log

## 2026-09-01 — Material/source foundation

### Implemented

- Added an independent `cft_revival.magnetics` package without importing or
  editing the active axisymmetric solver.
- Added immutable SI validation, typed failures, vectors, and canonical JSON.
- Added linear isotropic permeability with analytic inverse and energy.
- Added monotone PCHIP tabulated B-H interpolation, odd symmetry, explicit
  error/tangent extrapolation, deterministic inversion, exact coenergy
  integration, energy, differential permeability, and secant permeability.
- Added a SmCo-like linear recoil model with explicit remanence/coercivity
  temperature coefficients and a closed validity interval.
- Added axisymmetric uniform magnetization and equivalent bound-current
  descriptions, including physical handling of a region touching the axis.
- Added material-region, oriented Maxwell interface, demagnetization-screen,
  finite-box open-boundary, and axisymmetric worker hand-off contracts.
- Added a machine-readable equation/data/policy ledger.
- Added only authored synthetic checked examples; no vendor data was copied.

### Verification loop

- Checked constant-permeability analytic forward/inverse and energy limits.
- Checked all PCHIP knots, odd symmetry, dense monotonicity, positive
  differential permeability, finite-difference derivatives, inverse
  consistency, and exact energy/coenergy identity.
- Checked temperature coefficients, recoil line, magnetization magnitude,
  equivalent surface-current signs, zero uniform bound volume current, and
  Maxwell interface signs.
- Checked demagnetization and open-boundary safe/warning/invalid branches.
- Checked nonfinite, malformed, extreme, mutable, dangling-reference, and
  out-of-validity rejection.
- Checked deterministic serialization and runtime/ledger example agreement.

Final command results are recorded after the verification phase below.

### Scope controls

- Owned only new paths under `modern/src/cft_revival/magnetics/`,
  `modern/tests/magnetics/`, `modern/spec/magnetics/`, and
  `modern/docs/workstreams/magnetics-*`.
- Did not edit the active fields paths, shared files, other workstreams, FYP,
  or Git history.
- Did not install dependencies, benchmark performance, commit, or push.
- Made no throughput, acceleration, or production-performance claim.

### Verification results

- Focused magnetics suite: `33 passed`.
- Full `modern/src` and `modern/tests` byte-compilation: passed.
- Repository-wide default pytest command: collection blocked before test
  execution by pre-existing duplicate module basenames
  (`tests/coupling/test_spec.py` vs `tests/optimization/test_spec.py`, and
  `tests/pic/test_warp_backend.py` vs `tests/test_warp_backend.py`).
- An earlier repository-wide pytest with isolated importlib collection passed
  (`427 passed, 1 skipped`). A final rerun after concurrent workstreams added
  tests collected a larger suite and reported `2 failed, 465 passed, 1
  skipped`; both failures are outside this workstream in
  `tests/plasma/test_solver.py`. The skip is the existing optional unbuilt
  pybind11 extension.
- `mypy` and `ruff` are not installed in the environment; they were not
  installed because this workstream forbids dependency installation.
- FYP diff/status output was empty.
- Ownership status contains only the new magnetics paths declared above.

## 2026-09-01 — Audit remediation

### Corrected

- Replaced dimensional raw-coefficient PCHIP evaluation with interval-local
  decimal scaling, Bernstein/de Casteljau value evaluation, positive
  Fritsch-Carlson/Hyman-limited tangents, and a stable derivative form.
- Accepted valid abrupt monotone curves that the earlier endpoint rule rejected.
- Replaced global fixed-step inversion with interval-local safeguarded
  Newton/bisection and relative/ULP bracket termination.
- Replaced cancellation-prone `H B - coenergy` energy evaluation with direct
  positive `H dB` integration; retained direct `B dH` coenergy integration.
- Made vector normalization overflow-safe by scaling components before taking
  the local norm; nonrepresentable magnitude queries now fail explicitly.
- Added explicit mutually exclusive recoil-remanence and equivalent-current
  permanent-magnet authorities, typed material compatibility, exact recoil
  permeability matching, and source/region binding.
- Added axis regularity, nonzero-radius current-sheet, radial-span, orientation,
  finite thickness, and finite surface-area guards.
- Added duplicate material, region, interface, and source ID rejection.
- Added signed-zero canonicalization, canonical SHA-256 content digests,
  strict closed-schema deserialization, duplicate-key rejection, derived-field
  recomputation, and tamper detection.

### Audit verification loop

- Added deterministic scale/property tests spanning subnormal, `1e-300`-class
  derivatives, tiny/huge reciprocal scales, abrupt monotone shapes, knot
  neighborhoods, and seeded varied interval slopes.
- Added overflow-safe vector direction and typed magnitude tests.
- Added authority, double-counting, mismatched-host, geometry, duplicate-ID,
  cross-process digest, strict round-trip, and tamper tests.

### Audit verification results

- Focused magnetics suite including seeded property tests: `48 passed`.
- Full `modern/src` and `modern/tests` compileall: passed.
- Compatible repository suite (`--import-mode=importlib`): `673 passed, 1
  skipped, 14 failed, 10 errors`. Every failure/error is in concurrently
  changing `tests/coupling/` or `tests/visualization/` and their axisymmetric
  field artifacts; no magnetics test failed.
- Python source/test line-length scan at the configured 100 columns: clean.
- Final FYP and owned-path status checks are recorded at handoff.

## 2026-09-02 — Final acceptance remediation

### Corrected

- Removed binary64 conversion of normalized inversion brackets. Locally linear
  intervals now use direct Decimal physical-unit inversion; nonlinear solves
  retain Decimal candidates and stop only when reconstructed physical-`H`
  endpoints are equal or adjacent binary64 values.
- Preserved exact minimum-subnormal inversion over
  `(0,0)->(binary64_max,binary64_max)` and retained direct nonnegative
  energy/coenergy underflow semantics.
- Replaced independently supplied equivalent-source magnetization with a
  derived property of the typed permanent-magnet material, validated
  temperature, and normalized direction.
- Added handoff material-parameter identity checks and documented
  `32*epsilon` magnetization component tolerance.
- Updated strict deserialization to resolve the declared permanent-magnet
  material before source construction and to recompute source magnetization.

### Added verification

- Fraction-oracle tests for `H=5e-324`, `H=1e-323`, minimum normal, ordinary,
  and maximum-neighbour binary64 values.
- Nonlinear endpoint-near monotone physical-bracket tests for the first 32
  subnormal flux values.
- Explicit `T=1 K`, mismatched source material, rehashed `M=1 A/m`, and
  rehashed invalid-temperature rejection tests.

### Final acceptance results

- Focused magnetics suite including binary64 oracle/property tests:
  `52 passed`.
- Compatible complete repository suite (`--import-mode=importlib`):
  `802 passed, 1 skipped`; the skip is the existing optional unbuilt pybind11
  extension.
- Full `modern/src` and `modern/tests` compileall: passed.
- Python source/test line-length scan at the configured 100 columns: clean.
- FYP and final ownership status are checked separately at handoff.
