# Hybrid Workstream Development Log

## 2026-09-01 — Prescribed-field L2 first slice

### Implemented

- Added immutable SI xenon species and weighted particle state for Xe, Xe+,
  and Xe2+.
- Added dependency-free Boris advance, zero-field drift, periodic/reflecting/
  absorbing box policies, and conservative wall exchange.
- Added periodic cell-centred 1-D CIC number, charge, current, momentum, and
  kinetic-energy deposition with cell-volume normalization.
- Added stateless `splitmix64-counter-v1` random draws keyed by seed, particle
  ID, step, stream, and draw.
- Added elastic and charge-exchange operators against an explicitly synthetic
  constant cross-section fixture, including prescribed-neutral reservoir
  exchange.
- Added a fluid-electron closure protocol, isothermal quasineutral verification
  fixture, and conservative source exchange. Electric field and anomalous
  mobility remain unresolved rather than inferred.
- Added optional NVIDIA Warp float64 Boris and CIC kernels with explicit CPU/
  CUDA selection and finite-publication checks.
- Added canonical-JSON SHA-256 checkpoint and provenance contracts.
- Added a three-particle, four-step manufactured run bounded to at most 32
  steps; it is labeled as a verification fixture.
- Added machine-readable equation and checkpoint specifications plus
  formulation and next-stage roadmap documentation.

### Verification record

- Focused `python -m pytest tests/hybrid -ra`: 23 passed with no skips. This
  exercised Python reference, Warp CPU, and Warp CUDA paths.
- Explicit CUDA smoke for Boris plus CIC:
  `test_warp_pusher_and_deposition_match_reference[cuda:0]`: 1 passed.
- Default full-suite collection found a concurrent-workstream basename
  collision between `tests/pic/test_warp_backend.py` and the existing
  `tests/test_warp_backend.py`; no PIC/shared file was changed.
- Full `python -m pytest --import-mode=importlib`: 471 passed, 1 unchanged
  optional-pybind11 skip.
- `python -m compileall -q src tests`: passed.
- `git diff --exit-code -- FYP` and `git status --short -- FYP`: clean.
- Ownership status showed additions only under the four permitted hybrid path
  families. No tracked file was modified.
- Ruff and mypy were not installed in the environment and were not installed.
  Direct 100-character scans over hybrid source, tests, and docs passed.

### Scope controls

- No dependencies were installed and no benchmark was run.
- No shared fields, coupling, PIC, package, FYP, repository memory, or other
  workstream file was edited.
- No commit or push was performed.
- No self-consistent field, electron-energy, wall, plume, or calibrated
  thruster claim is made.

## 2026-09-01 — Hybrid audit corrections

### Corrected

- Declared and enforced standard leapfrog state
  `x^n,v^(n-1/2),E^n,B^n`; synchronous state now requires explicit
  initialization and is produced only by an explicit diagnostic
  synchronization helper.
- Added Boris half-kick electric-work diagnostics consistent with half-level
  kinetic energy.
- Restricted charge exchange to resonant Xe+; Xe2+ charge exchange now fails
  before sampling until products and source ledgers exist.
- Canonicalized collision traversal by unique particle ID and changed
  probability/momentum/energy reductions to stable summation.
- Required unique IDs before collision, deposition, advance, or checkpoint;
  `alive` is now strictly boolean.
- Preserved complete custom species identifier, symbol, charge state, mass,
  and charge through checkpoint roundtrip.
- Closed every checkpoint object, validated finite typed fields, UTC ISO
  timestamps, notes, provenance, RNG identity, and velocity staggering.
- Clarified that the embedded SHA-256 detects accidental corruption but does
  not authenticate a payload that an actor can edit and rehash.
- Made empty Warp pushes return `()` and empty Warp deposition return canonical
  zero moments on both CPU and CUDA without launching a zero-sized kernel.

### Added audit gates

- Constant-electric-field analytic displacement, synchronized work-energy, and
  half-level step-work identities.
- Long gyro boundedness and nondissipation.
- Xe2+ CX rejection plus Xe+ species/charge/source conservation.
- Exact aggregate collision-source invariance under input permutation.
- Duplicate RNG identity rejection across operators.
- Validly rehashed extra fields, invalid scalar types, nonfinite values,
  timestamps, notes, and staggering.
- Validly rehashed mutation acceptance to demonstrate integrity is not
  authenticity.
- Empty Warp CPU/CUDA parity.

### Verification

- Focused `python -m pytest tests/hybrid -q`: 41 passed.
- Explicit Warp normal/empty parity on `cpu` and `cuda:0`: 4 passed with no
  skips.
- Checkpoint/Warp file: 18 passed with no skips.
- Compatible default suite excluding only the concurrently inconsistent
  axisymmetric visualization artifact test: 681 passed, 1 unchanged optional
  pybind11 skip.
- Compatible importlib suite with the same exclusion: 682 passed, 1 unchanged
  optional pybind11 skip.
- Unexcluded default suite reached all 41 hybrid tests successfully, then
  reported 6 failures and 10 setup errors in the unrelated axisymmetric
  visualization test because its manifest digest/content no longer matched
  its generator contract: 679 passed and 1 unchanged skip.
- `python -m compileall -q src tests`: passed.
- Hybrid source/test/docs direct line-length scans: passed.
- Final FYP and owned-path status checks are recorded at handoff.
- No install, benchmark, commit, push, or out-of-scope edit was performed.

## 2026-09-02 — Checkpoint and identity hardening

### Corrected

- Replaced float coercion with actual-real validation in public SI models.
- Added checkpoint-only JSON numeric validation that accepts only finite
  built-in parser `int`/`float` values and rejects bool/string/float-like
  representations before model construction.
- Removed arbitrary xenon charge behavior. Xe, Xe+, and Xe2+ now derive
  `q=z e`; a supplied or checkpointed charge must match exactly. Custom
  identifier and mass remain supported.
- Tightened particle IDs and every public counter-RNG seed/counter to built-in
  unsigned 64-bit integers, explicitly rejecting booleans and floats.
- Added recursive JSON `object_pairs_hook` parsing that rejects duplicate keys
  at top-level or any nested object before digest/schema validation.
- Updated checkpoint schema with strict charge-state alternatives and numeric/
  duplicate-key parser contracts.

### Added gates

- String values in checkpoint time, `dt`, weight, position, velocity, species
  mass, and charge fields.
- Float-like model objects, boolean/float particle IDs, and boolean/float RNG
  counters in every counter position.
- Zero/arbitrary ion charge and nonzero neutral charge definitions.
- Rehashed checkpoint charge mismatch.
- Raw top-level and nested duplicate JSON member names.

### Verification

- Focused default mode: 70 passed.
- Focused importlib mode: 70 passed.
- Checkpoint/Warp test file: 22 passed.
- Explicit normal/empty Warp parity on `cpu` and `cuda:0`: 4 passed with no
  skips.
- Compatible default suite excluding only the concurrently inconsistent
  axisymmetric visualization artifact test: 769 passed, 1 unchanged optional
  pybind11 skip.
- Compatible importlib suite with the same exclusion: 779 passed, 1 unchanged
  optional pybind11 skip.
- `python -m compileall -q src tests`: passed.
- Hybrid source/test/docs 100-character scans: passed.
- Final ownership and FYP checks are recorded at handoff.
- No install, benchmark, commit, push, or out-of-scope edit was performed.
