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
- a small CLI, audit, traceability map, migration plan, and correctness gates.

It does **not** claim to reproduce a calibrated optimizer or plasma solver. The
legacy nonlinear equation system has confirmed state/constraint defects and is
quarantined until equations are checked against the cited source.

## Quick start

No third-party runtime dependency is required for the Python fallback:

```powershell
cd modern
python -m pytest
python -m cft_revival validate-config config/default.json
python -m cft_revival cusp-probability --low-t 0.02 --high-t 0.2
python -m cft_revival l0-evaluate config/l0-representative-point.json
python -m cft_revival validate-campaign-spec spec/optimization/campaign-v1.json
python -m cft_revival generate-initial-design spec/optimization/campaign-v1.json --count 32 --seed 7
```

With the optional Warp dependency already installed, run the checked 8,192
point L0 CUDA sweep or the older cusp parity/timing smoke:

```powershell
python -m cft_revival l0-sweep config/l0-deterministic-sweep.json --device cuda:0 --output $env:TEMP\cft-l0-sweep.json
python -m cft_revival benchmark-cusp --device cuda:0 --batch-size 65536 --gpu-busy
```

Both commands mark timing non-authoritative. The L0 artifact includes every
input/result, complete diagnostics, output ranges, and full-batch Python
reference parity. See `docs/FIRST_RESULTS.md`. This is a CUDA execution result
for reduced conservation equations, not a plasma/FEMM speed or physical
accuracy claim.

When running outside pytest, either install the package or expose `src`:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m cft_revival validate-config config/default.json
```

Build and test the dependency-free C++ kernel:

```powershell
cmake -S . -B build -DCFT_BUILD_PYTHON=OFF -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The Python extension additionally needs `pybind11` and `scikit-build-core`:

```powershell
python -m pip install .
```

See `docs/AUDIT.md`, `docs/REFERENCES.md`, `docs/ARCHITECTURE.md`, and
`docs/MIGRATION.md` before translating more physics. Workstream-level evidence
and integration instructions remain under `docs/workstreams/`.

Optional packages are metadata-only and imported lazily:

- `.[gpu]`: NVIDIA Warp;
- `.[optimization]`: PyTorch, BoTorch, GPyTorch, and pymoo.

The accepted core requires none of these packages. The optimization model
adapter remains runtime-unverified until those versions are tested deliberately
in an isolated environment.
