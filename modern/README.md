# CFT/HEMP Project Revival

This directory is a clean-room modernization foundation for the MATLAB files in
`../FYP`. The originals are intentionally unchanged.

The modern foundation provides:

- typed Python domain and configuration models with explicit unit suffixes;
- validation of the legacy design-space and geometry constraints;
- magnetic-field and plasma-solver backend contracts;
- a reader for existing FEMM text exports;
- a tested Python implementation of the legacy cusp-arrival calculation;
- an optional C++17/pybind11 implementation of the same numerical kernel;
- an optional NVIDIA Warp CPU/CUDA batch implementation of that kernel;
- an SI-explicit Xe/Xe+/Xe2+ L0 conservation model with Python and optional
  Warp CPU/CUDA batch paths, complete power-boundary diagnostics, and
  manufactured uniform-field fixtures;
- immutable multi-fidelity optimization records, constrained Pareto ranking,
  pending-aware replayable campaigns, guardrails, budgets/retries, and
  dependency-free shifted-Halton initial designs;
- accepted, independent foundations for L1a axisymmetric FDM fields, corrected
  global plasma numerics, magnetic materials/sources, topology-aware coupling,
  prescribed-field hybrid kernels, reduced electrostatic PIC kernels,
  surrogates, active learning, and validation/evidence records;
- deterministic L1a artifacts and a self-contained axisymmetric result viewer;
- a small CLI, audit, traceability map, migration plan, and correctness gates.

These foundations retain separate meanings and claim boundaries. L1a is
constant-permeability equivalent-current finite-box FDM, not FEM. The plasma,
hybrid, and PIC slices are not accepted predictive L2/L3 CFT models; the
surrogate held-out quality benchmark did not pass; and material-aware field
results remain screening-only. The legacy nonlinear equation system still has
confirmed state/constraint defects and is not a reconstruction oracle.

## Quick start

From the repository root, no third-party runtime dependency is required for
the Python fallback:

```powershell
cd modern
$env:PYTHONPATH = "$PWD\src"
python -m pytest
python -m cft_revival validate-config config/default.json
python -m cft_revival cusp-probability --low-t 0.02 --high-t 0.2
python -m cft_revival l0-evaluate config/l0-representative-point.json
python -m cft_revival validate-campaign-spec spec/optimization/campaign-v1.json
python -m cft_revival generate-initial-design spec/optimization/campaign-v1.json --count 32 --seed 7
python -m cft_revival validate-axisymmetric-results examples/axisymmetric/results/manifest-l1a-v1.json
```

The `PYTHONPATH` assignment is required for a fresh checkout because this
package uses a `src/` layout. It runs the core CLI without installing the
package or any optional native/GPU/model dependencies. On POSIX shells, use
`export PYTHONPATH="$PWD/src"` after `cd modern`.

With the optional Warp dependency already installed, run the checked 8,192
point L0 CUDA sweep or the older cusp parity/timing smoke:

```powershell
cd modern
$env:PYTHONPATH = "$PWD\src"
python -m cft_revival l0-sweep config/l0-deterministic-sweep.json --device cuda:0 --output $env:TEMP\cft-l0-sweep.json
python -m cft_revival benchmark-cusp --device cuda:0 --batch-size 65536 --gpu-busy
```

Both commands mark timing non-authoritative. The L0 artifact includes every
input/result, complete diagnostics, output ranges, and full-batch Python
reference parity. See `docs/FIRST_RESULTS.md`. This is a CUDA execution result
for reduced conservation equations, not a plasma/FEMM speed or physical
accuracy claim.

Build and test the dependency-free C++ kernel:

```powershell
cd modern
cmake -S . -B build -DCFT_BUILD_PYTHON=OFF -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Building the optional Python native extension additionally requires a C++17
toolchain plus the declared `pybind11` and `scikit-build-core` build
dependencies. It is not required for the no-install CLI path above.

See `docs/AUDIT.md`, `docs/REFERENCES.md`, `docs/ARCHITECTURE.md`, and
`docs/MIGRATION.md` before translating more physics. Workstream-level evidence
and integration instructions remain under `docs/workstreams/`.

Optional packages are metadata-only and imported lazily:

- `.[gpu]`: NVIDIA Warp;
- `.[numerics]`: NumPy acceleration for supported numerical helpers;
- `.[optimization]`: PyTorch, BoTorch, GPyTorch, and pymoo.

The accepted core requires none of these packages. The optimization model
adapter remains runtime-unverified until those versions are tested deliberately
in an isolated environment.

Open `visualization/first-results.html`,
`visualization/geometry-designs.html`, or
`visualization/axisymmetric-results.html` directly in a browser. Their
generators, JavaScript checks, and offline scans are deterministic test gates.
The manuscript under `../paper/` has separate policy, generated-table, and
two-clean-build checks; its L1--L3 evidence gates remain closed.

The L0 point/sweep schemas and optimization campaign v1.4 schema are closed and
versioned: unknown fields, duplicate JSON keys, non-finite numbers, malformed
types/ranges, and contradictory policy values are rejected with typed errors.
