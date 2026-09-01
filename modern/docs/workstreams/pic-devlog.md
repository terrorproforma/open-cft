# PIC Workstream Development Log

## 2026-09-01 — Independent verified L3 foundation

### Implemented

- Added immutable SI-explicit grid, species, solver, and simulation
  configuration contracts plus validated mutable particle state.
- Added conservative periodic CIC deposition/gather and a mean-zero
  finite-difference Poisson CG solve with recomputed true residual and explicit
  nonconvergence.
- Added electrostatic leapfrog and standalone 3-V Boris pushers.
- Added deterministic elastic MCC with probability admission, synthetic
  traceable tables, interpolation, rate diagnostics, and source-hash policy.
- Added a reduced integrated CPU step, charge/energy/solver diagnostics,
  stability reports, hashed JSON checkpoints, and runtime provenance.
- Added genuine optional Warp float64 CIC and gather/push kernels.
- Added future WarpX/PICMI and LXCat protocol boundaries without introducing
  either dependency.
- Added machine-readable scope/equation ledgers and scientific tests.

### Deliberate exclusions

- Did not implement axisymmetric geometry, walls, sheaths, ionization,
  cathodes, open boundaries, calibrated xenon physics, or predictive CFT
  outputs.
- Did not claim an integrated Warp solver: only deposition and gather/push
  kernels have Warp parity.
- Did not install WarpX/PICMI or AMReX after confirming they were absent.
- Did not edit shared, hybrid, field, other workstream, FYP, or Git-controlled
  paths; did not commit, push, or run throughput benchmarks.

### Environment assessment

- Python 3.12.10.
- Warp 1.14.0 installed with `cpu` and RTX 5090 `cuda:0`.
- `pywarpx`/WarpX and `amrex` imports unavailable.
- NumPy and pytest available; no package installation performed.

### Verification record

- Focused `python -m pytest tests/pic -q`: 19 passed.
- `python -m compileall -q src/cft_revival/pic tests/pic`: passed.
- Warp float64 CIC/gather/push parity passed on `cpu` and RTX 5090 `cuda:0`.
- A separate small CUDA charge/deposition smoke passed; no throughput timing
  or benchmark was run.
- Root `git diff --exit-code -- FYP`: passed.
- Protected FYP, hybrid, and field path diff query returned no changes.
- Branch remained `feat/sota-foundation`; status showed only the authorized new
  `pic` and `pic-*` paths for this workstream.
- Ruff and mypy were not installed, and were not installed for this task.

## 2026-09-01 — Audit defect corrections

### Corrected

- Replaced overflow-prone Poisson squared norms with max-scaled L2 norms and
  normalized-RHS CG. Source, iterate, physical reconstruction, true residual,
  and tolerance publication all require finite representable values.
- Replaced centred nodal field recovery with operator-consistent face fields.
  Added symmetric face-to-node particle gather, explicit area-aware charge and
  energy dimensions, and time-centred leapfrog energy diagnostics.
- Made electrostatic/Boris pushes and MCC proposals transactional. MCC now
  commits particles, RNG, and cumulative counters only after all particles and
  events validate.
- Enforced plasma-frequency and particle-Courant gates before, after push, and
  after collision in `PICStepper.step`; MCC probability is enforced before
  publication. All stage outputs are finite-validated and failures are typed.
- Replaced checkpoint v1 hash-only validation with a closed v2 schema,
  canonical finite JSON, typed reconstruction, identity, time/stagger,
  code-revision, backend/device/runtime, MCC RNG/hash/config/counter checks.
- Enriched provenance with code revision, backend/device, runtime identity,
  staggering, optional dependency versions/availability, and reduced claim.
- Renamed the PIC-owned Warp test module to avoid default-collection basename
  collisions.

### Added regression gates

- Alternating `+/-1e298` rejection and representable `+/-1e296` Poisson solve.
- Low-mode face amplitude, Nyquist preservation, Poisson energy identity,
  periodic self-force, and deposition/gather adjointness.
- Late-particle MCC rollback, push rollback, pre/post stability rollback, and
  nonfinite injected/mutated-state typed failures.
- Rehashed extra-key, empty/ragged particle, identity, time, stagger, runtime,
  code/backend, RNG, cross-section-hash, and MCC-counter checkpoint damage.

### Verification record

- Focused PIC suite: 45 passed before final documentation-only updates.
- Compileall for PIC source/tests: passed.
- Warp float64 parity ran through focused tests on CPU and RTX 5090 `cuda:0`;
  a separate area-aware staggered CUDA smoke passed.
- Default full suite: 663 passed, 1 skipped, 8 failed, 10 errors.
- Importlib full suite: 664 passed, 1 skipped, 8 failed, 10 errors.
- Full-suite failures were outside PIC and changed with concurrent work:
  coupling plus magnetics/plasma ledger tests, and axisymmetric visualization
  manifest hash/schema drift. No out-of-scope fixes were attempted.
- Final default focused PIC: 46 passed.
- Final importlib-mode focused PIC: 46 passed.
- `python -m compileall -q src tests`: passed.
- Final root FYP diff and protected hybrid/field diff query: clean.
- Final status on `feat/sota-foundation` showed only authorized new PIC paths
  for this workstream. No install, benchmark, commit, or push was performed.

## 2026-09-02 — Final PIC-local acceptance corrections

### Corrected

- `stability_report` now validates all mutable position/3V arrays, equal
  dimensions, particle bounds, and current grid/species/config scalars before
  the zero-density branch. Nonfinite metrics raise typed PIC errors.
- Replaced direct charge/volume arithmetic with exponent-separated
  product-ratio evaluation. Nonzero charge whose volumetric density cannot be
  represented is rejected instead of deposited as zero.
- Added independent CPU and Warp integrated-charge publication gates against a
  robust represented-charge calculation.
- Corrected the energy-refinement claim: acceptance is `coarse > 1.5*fine`;
  the current deterministic fixture observation is about `1.93`, with no
  general convergence-order or heating-bound claim.

### Added regressions

- NaN-mutated position, each velocity component, species weight, and ragged
  state rejection even at zero supplied density.
- Finite stable zero-density metrics.
- `area=1e300`, `charge=1e-300` typed underflow rejection before CPU/Warp work.
- Accepted normal/subnormal boundary densities preserve represented charge on
  Python, Warp CPU, and Warp CUDA.

### Verification record

- Default focused PIC: 63 passed.
- Importlib-mode focused PIC: 63 passed.
- Warp parity tests covered CPU and RTX 5090 `cuda:0`; a separate accepted
  subnormal extreme-area CUDA charge smoke passed exact reconstruction.
- `python -m compileall -q src tests`: passed.
- Compatible repository suite: 785 passed, 1 skipped, 3 out-of-scope coupling
  failures (field artifact schema expectation drift and a time-sensitive stale
  evidence fixture).
- Root FYP diff: clean.
- No dependency installation, benchmark, commit, push, or out-of-scope edit.
