# Target Architecture

## Design goals

1. Preserve `FYP/` as read-only historical evidence.
2. Make units, assumptions, inputs, outputs, solver status, and provenance
   explicit.
3. Separate orchestration from deterministic numerical work.
4. Establish parity tests before optimization or acceleration.
5. Fail closed when physics is unknown or an external solve is incomplete.

## Layers

### Python API and CLI

`cft_revival.models` owns immutable, typed boundary objects. Unit suffixes are
part of every field name; serialization will include a schema version.
`pipeline.evaluate_design` orchestrates backends without knowing whether they
are FEMM, a surrogate, CPU C++, or CUDA. The CLI currently validates configs,
tests the translated kernel, and inspects legacy FEMM exports.

Future Python modules should own experiment manifests, optimizer adapters,
cache keys, structured logging, dataset generation, and property/regression
tests. Optimizers must receive continuous constraint margins separately from
failure/status enums.

### C++17 numerical core

CMake builds `cft_kernels`; pybind11 exposes it as `cft_revival._native`.
`native/src/kernels.cpp` contains the first defensible translation: the
closed-form cusp loss-cone probability. Python has a behaviorally identical
fallback so development and parity tests do not require a compiler.

Subsequent C++ translations require:

- an equation specification with symbol, unit, source, and sign for every term;
- normalized residuals and analytic/automatic Jacobians;
- explicit parameters rather than globals;
- status containing convergence reason and per-equation residuals;
- golden comparisons against independently checked reference cases.

### Backend contracts

`MagneticFieldBackend` returns centreline/wall profiles with provenance.
`PlasmaBackend` returns 30 values plus convergence and residual norm.
`UnimplementedPlasmaBackend` intentionally prevents accidental use of the
unverified MATLAB equations.

`FemmExportBackend` is the first compatibility adapter. It reads the exact
legacy filenames but does not drive FEMM. The production FEMM adapter should be
a separate Windows worker process that:

1. accepts an immutable design manifest;
2. owns one FEMM instance behind an inter-process lock;
3. uses a per-run temporary directory;
4. closes FEMM in guaranteed cleanup;
5. validates both profiles and atomically publishes a completion manifest;
6. returns explicit timeout/automation/mesh/solve/export failure categories.

FEMM evaluations should be serialized per Windows desktop/COM endpoint.
Parallel optimization can still prepare jobs and run non-FEMM work; scaling
FEMM requires isolated VMs/workers, not threads sharing one application.

## CUDA and RTX 5090 plan

FEMM's Windows application automation, geometry setup, meshing, and individual
CPU solve do not become GPU workloads by wrapping them in CUDA. Initial GPU
work is therefore conditional.

Good eventual GPU candidates:

- batched evaluation/Jacobians of a verified algebraic plasma residual;
- large parameter sweeps and Monte Carlo uncertainty propagation;
- training and inference for a field surrogate generated from validated FEMM
  samples;
- a purpose-built batched axisymmetric magnetostatic solver, if its
  discretization and nonlinear material model are independently validated.

Poor candidates:

- launching FEMM or issuing `mi_*`/`mo_*` commands;
- tiny four-element probability calculations;
- single 30-variable nonlinear solves where transfer/launch overhead dominates.

### Phase 2A Warp backend

The optional `warp_backend.py` is a genuine batch kernel and toolchain proof,
not a FEMM replacement. It runs the verified closed-form loss-cone relation on
Warp `cpu` or `cuda:N`, after applying the scalar Python input contract on the
host. The dependency-free Python and C++ paths remain available.

The verified local stack is Warp 1.14.0, Warp CUDA Toolkit 12.9, driver CUDA
13.2, and an RTX 5090 32 GiB reported as `sm_120`. A standalone `nvcc` is not
on `PATH` and is not required by this Warp kernel. GPU occupancy was already
100% during inspection, so Phase 2A records only parity and smoke timing; it
makes no speedup claim.

Phases for Blackwell support:

1. **CPU correctness:** scalar Python/C++ parity, equation-level tests, FEMM
   golden datasets.
2. **Batch API:** structure-of-arrays inputs and deterministic multi-design CPU
   benchmark.
3. **Surrogate:** versioned training data, held-out field/profile error,
   uncertainty estimate, out-of-domain rejection.
4. **CUDA backend:** compile for the installed CUDA toolkit's supported
   Blackwell architecture, never hard-code an architecture before toolchain
   detection; compare CPU/GPU outputs and gradients.
5. **Replacement field solve (optional):** validate mesh convergence, linear
   materials, nonlinear B-H curves, boundaries, and profile outputs against
   FEMM and analytical cases.

## Next field-solver milestone: axisymmetric magnetostatics

The next real GPU milestone is a 2D `(r,z)` finite-element magnetostatic solver
that reproduces the axisymmetric problem assembled by `FEMMrun.m`. This is a
separate numerical project with the following required formulation and gates.

### Governing formulation

- Solve the magnetostatic curl-curl equation
  `curl(nu curl(A)) = J + magnetization source` with
  `A = A_phi(r,z) e_phi` and the axisymmetric `2*pi*r` weak-form weight.
- Recover fields with
  `B_r = -dA_phi/dz` and
  `B_z = (1/r) d(r A_phi)/dr`.
- Enforce regularity `A_phi=0` on `r=0`; handle the `1/r` terms analytically in
  the weak form, never by evaluating a singular point formula.
- Start with linear piecewise permeability and prescribed permanent-magnet
  coercive field/magnetization. Add nonlinear iron B-H curves only after the
  linear solver and Jacobian pass verification.
- Represent the three alternating SmCo ring magnets and their axial
  magnetization, pure-iron guides/shield, aluminium/BN/air regions, and open
  boundary from the exact legacy geometry. Material numbers must be exported
  from the referenced FEMM 4.2 library rather than guessed.
- Treat the exterior first as a successively enlarged truncated domain with a
  documented boundary condition. Implement an infinite/asymptotic boundary
  only after its FEMM formula and units are reproduced.

### Verification ladder

1. **Manufactured solution:** choose smooth `A_phi=r*f(r,z)` so axis regularity
   is exact, derive `J_phi`, and demonstrate expected `L2(A)` and magnetic
   field convergence rates on systematic mesh refinement.
2. **Simple analytical cases:** uniform-field/long-solenoid and linear
   permanent-magnet fixtures with energy and symmetry checks.
3. **Material interfaces:** verify normal `B` and tangential `H` transmission
   and magnetic-energy consistency.
4. **Legacy geometry, linear materials:** use identical dimensions,
   magnetization directions, mesh-refinement sequence, and outer domain.
5. **FEMM profile parity:** compare all 200 centreline/wall samples, not only
   four window means. Gate maximum/relative profile error, cusp locations,
   integrated magnetic energy, and derived p1-p4 separately.
6. **Nonlinear iron:** import and version the FEMM B-H table; verify nonlinear
   residual, iteration convergence, hysteresis assumptions, and mesh
   independence.
7. **Batch/GPU:** only after single-case parity, measure multi-design
   throughput including assembly, solve, transfers, profile extraction, and
   failed-solve handling.

### MFEM C++/CUDA versus Warp FEM

- **MFEM candidate:** production-oriented C++ mesh/finite-element assembly,
  established partial assembly and CUDA backends, and stronger access to
  scalable preconditioners. It fits the existing CMake/pybind11 core but adds a
  substantial native dependency and Blackwell-compatible build requirement.
- **Warp FEM candidate:** fastest route to a Python-controlled CUDA prototype,
  easy manufactured-source kernels, and compatibility with the proven local
  Warp stack. Axisymmetric curl-curl spaces, robust open boundaries, nonlinear
  B-H materials, and production preconditioning require explicit validation;
  maturity must not be assumed.
- **Decision gate:** implement the same linear manufactured problem and one
  legacy-geometry case in small spikes. Select only after comparing numerical
  convergence, profile parity, solver robustness, memory, build
  reproducibility, and end-to-end batch throughput. A scalar cusp kernel does
  not influence this choice.

## Correctness gates

- Cusp kernel: use the rationalized
  `0.5*r/(1+sqrt(1-r))` form for `r=B_low/B_high`; require exact endpoint
  behavior at `r=0,1`, relative error <= `1e-14` in the tiny-ratio regime, and
  Python/C++/Warp CPU/CUDA parity under scale-appropriate absolute/relative
  tolerances. The subtractive form is prohibited because it returns zero once
  `1-r` rounds to one.
- Plasma solve: all normalized equation residuals <= a documented tolerance;
  conservation closure reported independently; no status-only acceptance.
- FEMM adapter: exact design/config hash, two valid profiles, monotonic sample
  coordinate, no stale file reuse.
- Surrogate: held-out metrics per profile and derived cusp probability;
  uncertainty-calibrated rejection; never silently replace high-uncertainty
  FEMM calls.
- Performance: dimensional-analysis tests, finite outputs, efficiency in
  `[0,1]`, and golden cases approved by a domain reviewer.
- GPU: representative batch speedup includes transfers and preprocessing;
  target at least 3x end-to-end over optimized multithreaded CPU before making
  CUDA the default. Numerical error must remain within the CPU correctness
  budget.

## Benchmark protocol

Record hardware, compiler/toolkit, optimization flags, batch size, warm-up,
median and p95 latency, throughput, peak memory, and numerical error. Benchmark
separately:

- FEMM wall-clock per serialized solve;
- profile extraction/cusp calculation;
- plasma residual and Jacobian batches;
- optimizer end-to-end throughput;
- surrogate inference and fallback rate.

No GPU claim is meaningful until profiling shows a dominant batchable kernel.
