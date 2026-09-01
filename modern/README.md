# CFT/HEMP Project Revival

This directory is a clean-room modernization foundation for the MATLAB files in
`../FYP`. The originals are intentionally unchanged.

Phase 1 provides:

- typed Python domain and configuration models with explicit unit suffixes;
- validation of the legacy design-space and geometry constraints;
- magnetic-field and plasma-solver backend contracts;
- a reader for existing FEMM text exports;
- a tested Python implementation of the legacy cusp-arrival calculation;
- an optional C++17/pybind11 implementation of the same numerical kernel;
- an optional NVIDIA Warp CPU/CUDA batch implementation of that kernel;
- a small CLI, audit, traceability map, migration plan, and correctness gates.

It does **not** claim to reproduce the complete optimizer or plasma solver. The
legacy nonlinear equation system has confirmed state/constraint defects and is
quarantined until equations are checked against the cited source.

## Quick start

No third-party runtime dependency is required for the Python fallback:

```powershell
cd modern
python -m pytest
python -m cft_revival validate-config config/default.json
python -m cft_revival cusp-probability --low-t 0.02 --high-t 0.2
```

With the optional Warp dependency already installed, run a parity/timing smoke
command:

```powershell
python -m cft_revival benchmark-cusp --device cuda:0 --batch-size 65536 --gpu-busy
```

The command reports end-to-end timing but marks it non-authoritative. It is a
CUDA toolchain proof for one verified batch kernel, not a plasma or FEMM
acceleration claim.

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
`docs/MIGRATION.md` before translating more physics.
