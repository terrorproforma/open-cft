# Development Log

## 2026-09-01 — Phase 1 foundation

### Investigated

- Read and traced every MATLAB file under `FYP/`.
- Mapped decision variables, 30 plasma unknowns, globals, objective signs,
  solver flags, FEMM automation/export flow, plotting/surrogate dependencies,
  and I/O conventions.
- Separated confirmed implementation defects from physics requiring source
  verification in `docs/AUDIT.md`.

### Implemented

- Added Python package/CLI/config metadata under `modern/`.
- Added immutable unit-explicit models and validation for design geometry,
  constants, profiles, plasma results, and app configuration.
- Added magnetic/plasma backend interfaces, safe unimplemented plasma backend,
  and legacy FEMM export parser.
- Translated the cusp loss-cone probability to analytic Python and C++17
  kernels with pybind11 bindings.
- Translated performance post-processing behind a requirement for a validated
  converged plasma solution.
- Added focused Python and C++ tests and architecture/migration documents.

### Deliberate deferrals

- No legacy source was edited.
- No external optimizer, surrogate, FEMM automation, or plasma residual was
  guessed.
- CUDA was designed as a future batched backend, not added before profiling and
  correctness baselines.

### Verification record

- `python -m pytest`: 11 passed in 0.51 s.
- `python -m compileall -q src tests`: passed.
- `python -m cft_revival validate-config config/default.json`: passed with
  resolved output path.
- Cusp CLI for `B_low=0.2 T`, `B_high=1.0 T`: returned
  `0.052786404500042072`.
- CMake 4.3.2 + GNU C++ 16.1.0 configured and built with
  `CFT_BUILD_PYTHON=OFF`; CTest: 1/1 passed.
- Optional extension configuration is blocked because no
  `pybind11Config.cmake`/`pybind11-config.cmake` is installed.
- `pybind11`, `scikit-build-core`, `ruff`, and `mypy` are not installed, so the
  wheel build and those two static analyzers were not run. No packages or
  machine-wide toolchains were installed.
- No CUDA compiler (`nvcc`) is available; CUDA was not built or benchmarked.

## 2026-09-01 — Phase 2A Warp and publication reconciliation

### Evidence reconciled

- Retrieved ISTS 2017-b-32 from the official archive and recorded its
  citation, file hash, equation links, and evidence classes in
  `docs/REFERENCES.md`.
- Documented publication/snapshot differences: 3 versus 4 objectives, 100
  versus 50 generations, 8 claimed versus 5 reported variables, failed 5%
  surrogate quality versus Sobol use, sensitivity prose/table conflict, and
  inconsistent S1 power.
- Kept the ISTS paper as source traceability, not proof for the complete
  Kornfeld residual set or the exact publication-run source revision.

### Implemented

- Added `warp_backend.py` with a real float64 Warp kernel for batched
  cusp-arrival probability on `cpu`, `cuda`, `cuda:N`, or `auto`.
- Reused the scalar validation contract for finite fields,
  `B_low >= 0`, `B_high > 0`, and `B_low <= B_high`.
- Added a dependency-free analytic reference function and preserved optional
  C++ dispatch.
- Added deterministic Warp CPU/CUDA edge and random-batch parity tests.
- Added `benchmark-cusp`, reporting device, batch, warmup/repeats, end-to-end
  timing scope, sample outputs, maximum error, and a mandatory
  non-authoritative timing label.
- Specified the 2D axisymmetric magnetostatic weak form, permanent-magnet and
  material handling, manufactured solutions, mesh convergence, FEMM profile
  parity, nonlinear iron gate, and MFEM-versus-Warp-FEM decision criteria.

### Verification record

- Environment: Warp 1.14.0; Warp CUDA Toolkit 12.9; driver CUDA 13.2;
  RTX 5090 32 GiB, `sm_120`; standalone `nvcc` absent.
- `python -m pytest`: 19 passed, including Warp CPU and CUDA tests.
- `python -m compileall -q src tests`: passed.
- Native CMake rebuild: no compilation work needed; CTest 1/1 passed.
- Warp CPU smoke, batch 32,768, 1 warmup, 3 measured runs:
  maximum absolute error `0.0`.
- Warp CUDA smoke with the same batch/run counts:
  maximum absolute error `0.0`.
- Timing was printed by the CLI but is not accepted as benchmark evidence.
  The GPU had been reported at 100% utilization and no speedup claim was made.
- No packages or machine configuration were changed.

## 2026-09-01 — Phase 2A numerical stability correction

### Independent verification finding

- The initially used expression `0.5*(1-sqrt(1-r))` catastrophically cancels
  when `r=B_low/B_high` is tiny. At `r=1e-18`, binary64 rounds `1-r` to one
  and the expression incorrectly returns zero instead of approximately
  `2.5e-19`.
- The prior random-batch maximum-error result did not exercise this regime and
  was therefore normal-scale parity evidence only.
- Warp batch emptiness used `if not sequence`, which is ambiguous for
  multi-element NumPy arrays and raises before validation.

### Correction

- Python, C++17, and Warp now use the algebraically equivalent stable form
  `0.5*r/(1+sqrt(1-r))`.
- Validation still requires finite fields, `B_low >= 0`, `B_high > 0`, and
  `B_low <= B_high`; `r=0` returns exactly 0 and `r=1` exactly 0.5.
- Warp uses explicit lengths and optional one-dimensional shape checks. It
  validates values in place and passes the original sequence/NumPy array to
  Warp, avoiding intermediate tuple copies.
- Added tiny-ratio relative checks, infinities, one- and two-dimensional NumPy
  arrays, missing-Warp behavior, unavailable devices, and an optional native
  extension test.

### Verification record

- `python -m pytest`: 32 passed, 1 skipped. The skip is the pybind11 extension
  test because that optional extension is not installed/built.
- `python -m compileall -q src tests`: passed.
- Native C++ rebuilt successfully; CTest 1/1 passed, including `r=1e-18` and
  infinity rejection.
- Direct Warp CPU and `cuda:0` smoke values for
  `r=[0,1e-30,1e-18,0.2,1]` were
  `[0,2.5e-31,2.5e-19,0.052786404500042065,0.5]` on both devices.
- Maximum observed relative difference from the Python reference over nonzero
  smoke values was `0.0` on both Warp CPU and CUDA.
- No speed benchmark was run and no package or machine configuration changed.

## 2026-09-01 — Signed-zero canonicalization

- Accepted `B_low=-0.0` previously propagated a negative zero through Python,
  C++ and Warp. All implementations now return explicit positive `0.0` when
  the validated low field/ratio compares equal to zero.
- The branch is exact-zero only; tiny positive values continue through the
  stable rationalized formula.
- Added Python/native/Warp CPU/CUDA sign-bit checks and retained a positive
  subnormal regression (`r=2e-323` gives `5e-324`).
- Full verification: 33 passed, 1 optional pybind11-extension test skipped;
  `compileall` passed; native CTest 1/1 passed; focused Warp CPU/CUDA sign and
  subnormal checks passed.
- No performance benchmark, installation, machine change, or `FYP` edit was
  performed.

## 2026-09-01 — Public repository preparation

### Changed

- Added root public-repository metadata: `README.md`, MIT `LICENSE`,
  `CITATION.cff`, and a root `.gitignore`.
- Updated `modern/pyproject.toml` from proprietary placeholder metadata to MIT
  licensing and the project author's name.
- Excluded builds, caches, outputs, FEMM exports/results, compiled artifacts,
  virtual environments, local PDFs/decks, and AppleDouble metadata.
- Preserved all 16 historical `FYP/*.m` source files without edits.

### Validation

- Candidate scans found no likely tokens, credentials, private keys, or
  user-home absolute paths.
- `python -m pytest`: 33 passed, 1 optional pybind11-extension test skipped.
- `python -m compileall -q src tests`: passed.
- Configuration validation: passed.
- CMake configure/build: passed; CTest: 1/1 passed.
- Focused Warp CPU/CUDA parity and tiny-ratio smoke: 4 passed.
- No benchmarks, dependency installations, or machine configuration changes
  were performed.

### Follow-up

- The repository still intentionally lacks a validated complete plasma,
  magnetostatic, or optimization workflow; public documentation states these
  limitations explicitly.

## 2026-09-01 — Physics/optimization foundation integration

### Integrated

- Retained the accepted 86-test physics and 54-test optimization workstreams,
  their machine-readable specs, and detailed workstream reports.
- Added domain-explicit shared exports:
  `L0XenonOperatingPoint`/`evaluate_l0_performance` and
  `OptimizationDesign`/`OptimizationCampaign`. Legacy `DesignPoint` remains
  unchanged and distinct.
- Added checked L0 operating-point and deterministic sweep workflows with
  complete JSON inputs, outputs, diagnostics, model-fidelity labels, aggregate
  ranges, and full CPU-reference parity.
- Added strict campaign spec v1.4 validation and dependency-free initial
  design manifests. Fixed the discovered zero-count initial-design edge case.
- Added optional metadata for Warp and lazy
  Torch/BoTorch/GPyTorch/pymoo model boundaries without installing packages.
- Added hypothetical representative point and 8,192-point sweep configs;
  neither uses the 2017 archived objectives nor treats 2020 fixtures as fitted
  values.
- Reconciled architecture, migration, provenance, roadmap, citation, workstream
  development history, and learning guardrails.

### First result

- Executed 8,192 points on Warp 1.14.0 `cuda:0`, NVIDIA GeForce RTX 5090
  (`sm_120`, 32,607 MiB), CUDA Toolkit 12.9/driver CUDA 13.2.
- Full-batch Python parity passed all 26 published numeric fields within the
  documented binary64 gates; no point failed.
- Axial thrust spanned `0.00188384225`–`0.0513183291 N`; Isp
  `799.268670`–`2726.81617 s`; beam power
  `17.2688569`–`786.015592 W`.
- Maximum relative particle, mass, current, and beam-power conservation
  residuals were `2.36837e-16`, `3.24198e-16`, `4.10551e-16`, and
  `4.43207e-16`.
- End-to-end CUDA time was 0.634302 s (12,914.99 points/s) and the separate
  Python reference construction was 0.141245 s. Load/clocks were uncontrolled,
  so no speedup or slowdown claim was accepted.

### Verification

- Full Python suite: 184 passed, one expected optional pybind11 skip.
- `python -m compileall -q src tests`: passed.
- CMake configure/build and native CTest: 1/1 passed.
- Focused Warp CPU/CUDA tests: 26 passed.
- Full 8,192-point Warp CPU sweep: full Python-reference parity passed.
- L0 point/sweep JSON, campaign spec v1.4, and 256-design manifest validation:
  passed.
- `git diff --check`: passed; `git diff --exit-code -- FYP`: passed.
- Ruff was unavailable in the existing environment and was not installed;
  compileall, tests, diff checks, and direct line-length review were used.
- No dependencies, system packages, drivers, or machine configuration changed.

Detailed workstream timelines remain in `docs/workstreams/physics-devlog.md`
and `docs/workstreams/optimization-devlog.md`.

## 2026-09-01 — Integration defect corrections

### Corrected

- Made every documented src-layout CLI block runnable from the repository root
  without installing the package: `cd modern`, then
  `$env:PYTHONPATH="$PWD\src"`. Added the POSIX equivalent and removed the
  unnecessary package-install command from the core path.
- Replaced permissive campaign-spec parsing with a closed v1.4 schema:
  recursive allowlists, exact objective transform/order, typed/ranged
  acquisition fractions summing to one, policy cross-checks, exact stopping
  gates, and null-until-verified benchmark results.
- Replaced permissive L0 config parsing with closed point/sweep schemas,
  strict duplicate/non-finite JSON loading, exact input/range fields, bounded
  batch sizes, and pre-evaluation device validation.
- Added adversarial tests for unknown/missing fields, malformed types/ranges,
  bad devices, wrong direction transforms, invalid acquisition fractions,
  contradictory retry/stopping policy, duplicate keys, and NaN/Infinity.
- Added subprocess smoke coverage for documented no-install core commands and
  a guard requiring every PowerShell CLI block to set the source path.

### Verification

- Full Python suite: 232 passed, one expected optional pybind11 skip.
- Strict schema/documentation selection: 52 passed.
- `python -m compileall -q src tests`: passed.
- Native CTest: 1/1 passed.
- Exact documented no-install config, cusp, L0 point, campaign validation, and
  32-design commands: passed from `modern/` with only `PYTHONPATH` set.
- Focused Warp CPU/CUDA parity: 26 passed; documented cusp selection: 4 passed.
- Repeated 8,192-point RTX 5090 CUDA sweep: zero parity mismatches and output
  ranges exactly matched `docs/FIRST_RESULTS.md`.
- JSON syntax checks, `git diff --check`, and `git diff --exit-code -- FYP`:
  passed. No dependencies were installed.

## 2026-09-02 — Accepted workstream integration

### Integrated

- Added independently scoped packages/specs/tests for L1a axisymmetric FDM
  fields, corrected global plasma numerics, magnetics, coupling,
  prescribed-field hybrid, reduced electrostatic PIC, surrogates, active
  learning, and validation/evidence.
- Added the deterministic three-design L1a artifact bundle and schema-1.1
  offline viewer. The shared CLI now validates the manifest, sidecars, and
  referenced artifacts while preserving every existing command.
- Added lazy shared module exports. Base import remains independent of
  Warp, NumPy, Torch, BoTorch, and GPyTorch; optional metadata now includes a
  NumPy numerical-acceleration extra.
- Integrated paper manuscript/evidence/generated sources and deterministic
  policy/build scripts while keeping `paper/build/` and PDFs ignored.
- Preserved the publication's `MDO (original)` label in the authorized 2020
  external fixture; “corrected low-fidelity” remains editorial interpretation.

### Claim boundaries

- L1a is linear-vacuum, constant-permeability, equivalent-current,
  structured-grid FDM—not FEM or material-aware production fields.
- Global plasma, hybrid, and PIC workstreams are numerically verified
  foundations, not accepted predictive L2/L3 CFT models.
- The surrogate held-out benchmark did not satisfy its quality gates.
- Paper L1--L3 result gates remain closed. No human coauthor, contribution,
  affiliation, or corresponding-author approval was invented.
- The still-screening `material_fields` implementation, tests, specs, examples,
  and workstream reports were excluded from integration.

### Integration verification

- Accepted tree, excluding the explicitly screening `tests/material_fields`:
  860 passed and one optional pybind11 skip in both default and importlib
  collection modes.
- Full all-path runs in both modes reached 876 passed and one skip, with one
  screening-only failure: a concurrently changed `material_fields` artifact
  reports a stale implementation hash. No excluded path was changed to mask it.
- Accepted Warp CPU/RTX 5090 CUDA smoke selection: 94 passed across cusp, L0,
  L1a fields, hybrid, and PIC paths.
- Visualization generation and deterministic/offline/JavaScript gates:
  60 passed. L0, geometry v1.1, and L1a HTML regenerated byte-current.
- Python compileall passed; native CMake build and CTest passed 1/1.
- Paper table generation, policy checks, and 11 adversarial tests passed. Two
  clean PDF builds were byte-identical with SHA-256
  `bd35afd2a075ef83b1368db56169e398c180f95d801389fd04eecd86377a31f2`;
  the PDF and build records remain ignored.
- Staged diff/secret/home-path/build-artifact scans and both working/staged FYP
  diffs passed.

## 2026-09-03 — PIC-2D xenon cross-section data file

### Investigated

- LXCat (`nl.lxcat.net`) serves data only through interactive sessions and the
  NIST BEB ionisation database has no Xe entry; LoKI-B, EDIPIC-2D, and Starfish
  ship no e–Xe tables. WarpX-data has Magboltz-7.1 Xe tables but they are
  1.9–2.2 MB 0.01 eV resamples ending at 750 eV with a single excitation level.
- `lanl/ThunderBoltz` commits a verbatim LXCat export (21 May 2023) containing
  the complete Biagi-v7.1, Biagi, BSR, COP, Hayashi, Morgan, Puech, SIGLO, and
  TRINITI e/Xe sets.

### Implemented

- Added `modern/spec/pic2d/xenon-cross-sections-v1.json` (schema
  `cft.pic2d.xenon-cross-sections.v1`) with elastic momentum transfer, one
  lumped 8.32 eV excitation channel (sum of the four Biagi-v7.1 levels), and
  12.13 eV single ionisation, each on a log grid to exactly 1000 eV, plus the
  `.sha256` sidecar and a canonical-JSON payload hash.
- Added `build_xenon_cross_sections.py`, which rebuilds the JSON byte-identically
  from `sources/lxcat_biagi-v7.1_xe_extract.txt` (35 kB LF-normalised extract of
  the 1.97 MB upstream export, upstream sha256 pinned) and can re-fetch the
  pinned commit with `--refresh-source`.

### Verification record

- Rebuild is byte-identical; independent re-interpolation from the raw CRLF
  upstream agrees to < 5e-6 relative (6-significant-figure rounding).
- Biagi-v7.1 agrees with Hayashi/SIGLO momentum transfer within ~10–25 %,
  ionisation peaks 5.61e-20 m² vs 5.45–5.88e-20 m² across sets, summed
  excitation peak 2.95e-20 m² vs 0.9–3.7e-20 m² across sets.
- Only non-tabulated content is a one-point power-law tail from 965–977 eV to
  1000 eV.

## 2026-09-03 - Test health after the LF pin

- Resolved every red directory left by the CRLF-era checkout without editing
  any `results/` byte or frozen preregistration file; details, root causes and
  audits in `docs/workstreams/test-health-devlog.md`.
- Sweep v2: posthoc EOL audit and a tolerance bound to exactly `protocol.json`
  in `protocol.py` and the v2 dashboard generator.
- Sweep v1: force-added the ignore-masked `results/` that the committed
  dashboard renders.
- Material fields: audit anchored to the `8603a905` blobs, then re-bound the
  v1.4 example artifacts with the workstream's replay-guarded refresh script
  (only hash-binding leaves changed).
- L0 surrogate v3/v4: pre-execution tests made lifecycle-aware.
- `--import-mode=importlib` in `pyproject.toml`; one invocation of
  `python -m pytest tests` gives 1619 passed, 5 documented skips;
  `paper/tests` 19 passed.
