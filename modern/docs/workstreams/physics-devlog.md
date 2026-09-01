# Physics Workstream Development Log

## 2026-09-01 — Verified L0 implementation

### Implemented

- Added immutable SI-explicit xenon operating-point, mass-flow, charge-state,
  mass-utilization, beam/divergence, and electrical-boundary models.
- Added typed validation, optional-dependency, and device errors.
- Implemented dependency-free particle, mass, charge, work-energy, momentum,
  specific-impulse, kinetic-power, and boundary-specific efficiency relations.
- Added conservation diagnostics and structured applicability warnings.
- Added one-launch NVIDIA Warp float64 batching with explicit CPU/CUDA
  selection and no hot-section host scalar round trips.
- Added a regular axisymmetric uniform-field manufactured solution.
- Added a machine-readable equation ledger and separately labeled 2020
  external regression fixtures.
- Added analytic, endpoint, invalid-domain, conservation, deterministic batch,
  field-regularity, optional dependency/device, and Warp parity tests.

### Deliberate exclusions

- Did not edit `FYP/` or any shared modern source, test, configuration, CLI,
  package initializer, architecture, README, devlog, or scratchpad file.
- Did not implement disputed Kornfeld residual signs.
- Did not infer or calibrate beam-current, divergence, plasma, cathode, or wall
  closures.
- Did not claim that a FEM or PIC solver exists.
- Did not install dependencies or run an uncontrolled-load speed comparison.

### Verification record

- Focused `python -m pytest tests/physics -q`: 44 passed.
- Full `python -m pytest`: 95 passed, 1 skipped. The skip is the unchanged
  optional pybind11 extension test; the full suite includes parallel
  optimization work present in the working tree.
- Physics Warp parity ran on Warp 1.14.0 `cpu` and RTX 5090 `cuda:0`.
- `python -m compileall -q src tests`: passed.
- Unchanged native `ctest --test-dir build --output-on-failure`: 1/1 passed.
- `git diff --exit-code -- FYP`: passed; preserved legacy files are unchanged.
- Ruff was not available in the installed environment and was not installed.
  A direct line-length scan and `git diff --check` passed.

## 2026-09-01 — Acceptance-defect correction

### Corrected

- Replaced strict, backend-dependent PPU subtraction with an initial shared
  adjacent-binary64 boundary contract. Later 1024-case reconstruction testing
  showed that legitimate regrouping can span two ULPs; see the correction
  below.
- Restricted running points to positive mass flow and voltage, imposed the
  documented `Xe2+ <= 0.01 c` nonrelativistic gate, and reject every
  non-representable derived state with `PhysicsValidationError`.
- Replaced decimal fraction tolerances with stable summation, a two-ULP bound
  at unity, and explicit normalization of representation noise.
- Reformed thrust, current, acceleration speed, and especially specific
  impulse to avoid avoidable overflow, underflow, and `0*inf` pathways.
- Unified reference/Warp scalar, shape, ragged, and mismatched-length batch
  validation.
- Replaced absolute-error-dominated parity with exact-zero, relative, and
  Decimal-oracle assertions for tiny observables.

### Added regression gates

- PPU exact/adjacent/outside boundaries over `1e-200` to `1e200`, including
  sign-bit checks on Python, Warp CPU, and RTX 5090 CUDA.
- Extreme finite input rejection and finite-publication checks.
- Fraction normalization and material sum-error tests.
- `V=mdot=1e-300` high-precision specific-impulse and nonzero-current checks.
- Scalar, 2-D, ragged, and inconsistent sequence batch failures using only
  typed physics errors.

### Verification record

- Focused physics suite: 68 passed, including Warp 1.14 float64 execution on
  `cpu` and RTX 5090 `cuda:0`.
- Full Python suite after the parallel optimization work stabilized:
  137 passed, 1 unchanged optional-pybind11 skip.
- `python -m compileall -q src tests`: passed.
- Unchanged native CTest: 1/1 passed.
- FYP diff: clean.
- No benchmark, dependency installation, commit, or push was performed.

## 2026-09-01 — Remaining acceptance corrections

- Canonicalized required PPU load with `fsum`, accepted legitimate regrouping
  within four scale-aware ULPs, and snapped that band to one effective budget.
- Added requested/effective PPU input and boundary-adjustment metadata. Loss
  and efficiency now use the same effective input.
- Represented zero-denominator efficiencies as `None`; no `0/0` path reports a
  fabricated numeric efficiency.
- Added exponent-separated products/ratios and separated square roots so
  representable custom-species values survive intermediate underflow/overflow.
- Replaced rounded fraction-sum admission with an exact rational sum of the
  binary64 inputs before deterministic normalization.
- Applied the finite-publication contract to manufactured vector potentials.
- Added all-1024 caller reconstruction, cutoff-neighbor, canonical-budget,
  undefined-ratio, max-mass/max-flow Decimal-oracle, exact-fraction-cutoff,
  and field-overflow tests across Python, Warp CPU, and RTX 5090 CUDA.

### Verification record

- Focused physics suite: 79 passed.
- Dedicated PPU/extreme/subnormal adversarial selection: 22 passed on the
  dependency-free reference, Warp CPU, and RTX 5090 CUDA paths.
- Physics plus unchanged core tests: 112 passed, 1 unchanged optional-pybind11
  skip.
- Full suite was executed while the parallel optimization workstream was
  changing. Physics and shared/core tests passed; optimization campaign tests
  were not stable at the final observation and remain outside this workstream.
- `compileall`: passed; unchanged native CTest: 1/1 passed; FYP diff: clean.
- No install, benchmark, commit, or push was performed.

## 2026-09-01 — Manufactured-field range correction

- Replaced `(0.5*B0)*r` with a `frexp`/`ldexp` half-product that combines
  mantissas and exponents before final binary64 rounding.
- Preserved the reviewed `B0=5e-324`, `r=float_max` result
  `4.4408920985006257e-16 T m`, plus the huge-field/tiny-radius symmetry.
- Added exact Decimal-oracle, normal, signed, minimum-subnormal, canonical
  positive-zero, and true-overflow regression cases.
- Verification: 86 focused physics tests passed; physics plus unchanged core
  tests passed 119 with one unchanged optional-pybind11 skip; compileall and
  native CTest 1/1 passed; FYP diff remained clean.
