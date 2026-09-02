# Material-fields development log

## 2026-09-02 — L1b linear material milestone

- Added strict adapters over the accepted geometry and magnetics public APIs.
- Preserved geometry/magnetics SHA-256 anchors, original geometry material IDs,
  PM temperatures/polarities, and manufacturing tolerances.
- Added exact axisymmetric region-volume integration and conservative sheet
  current transfer diagnostics.
- Implemented CPU and optional float64 Warp CPU/CUDA matrix-free PCG with
  harmonic interface reluctivity and true-residual publication acceptance.
- Added recoil-remanence source assembly without equivalent-current duplication.
- Added deterministic full/downsampled field, material, source and topology
  artifacts plus a viewer contract.
- Verified uniform-medium L1a reduction, a piecewise-reluctivity manufactured
  discrete solution, polarity antisymmetry, raster conservation, energy
  stationarity, fail-closed extremes, and Warp CPU parity when available.
- Kept nonlinear B-H coupling gated because independent nonlinear energy and
  constitutive-residual evidence is not yet available.

Final evidence: 141 focused/compatible geometry, magnetics, L1a and L1b tests
passed; owned paths compile; manufactured flux convergence orders were
2.00708 and 2.00176. Three CUDA runs on the RTX 5090 agreed with CPU fields to
at most `1.0e-15` scale-relative error. Axis-field mesh changes from 56x112 to
72x144 were 0.211%, 0.220% and 0.304%; padding-expansion changes were 0.918%,
0.721% and 0.478%. Recoil weak-source conservation errors were below
`3.0e-15`. Ruff was unavailable and was not installed.

No dependencies were installed, no uncontrolled benchmarks were run, and no
repository commit was created.

## 2026-09-02 — L1b audit closure

- Replaced axisymmetric-volume source averaging with the correct meridional
  weak-form measure and exact analytical weak-action diagnostics.
- Separated free current, PM bound current and recoil remanence structurally;
  invalid mixed/missing PM authority now fails construction.
- Added overflow-safe harmonic reluctivity on Python and Warp.
- Added closed v1.1 result/viewer schemas and recursive validators. Screening
  artifacts serialize for audit but strict publication validation rejects them.
- Added base plus two 1.5x domain expansions from three characteristic lengths,
  boundary ratios, successive axis/peak changes, independent mesh studies,
  energy/current/source gates, and explicit saturation/demagnetization warnings.
- Preserved all geometry v1.1 regions in provenance, including tapered regions
  omitted from the current public magnetics handoff.
- Re-ran recoil/equivalent comparisons at base and fine meshes. Differences
  remain model/discretization-form evidence and are not normalized away.
- All three regenerated designs are `SCREENING_NOT_ACCEPTED`; see the manifest
  and learning ledger for failed gates.

Final audit verification: 152 compatible geometry/magnetics/L1a/L1b tests and
17 focused L1b tests passed, compileall passed, and all artifact/viewer hashes
and closed contracts revalidated. Warp CPU and RTX 5090 CUDA `Bz` scale-relative
differences remained below `7.7e-16`. FYP has no diff. Ruff remained unavailable
and was not installed.

## 2026-09-02 — L1b corrective hardening

- Removed public construction of publication evidence from artifact generation.
  Artifacts now derive acceptance from embedded raw base, two domain, mesh,
  alignment and two PM model-form runs.
- Bound every observation to config, geometry-v1.1 payload, magnetics payload,
  implementation, backend, grid, domain, coefficient-problem and run SHA-256.
  Validation recomputes raw metrics, warning enums and all 13 gates.
- Derived padding and expansion ratios from run domains/source envelopes;
  duplicate runs and non-strict domain expansions fail closed.
- Replaced acceptance use of sampled maxima with interpolated fixed physical
  stage-centre QoIs. Cellwise and sampled-axis maxima are screening-only.
- Added half-cell volume-fraction face transmissibilities and tensor-product
  first-moment sheet deposition. PM form discrepancy is tracked at base/fine
  grids and must itself converge.
- Added minimum effective feature-cell and alignment gates, exact geometry
  schema/hash provenance, authoritative counts, recursively closed raw-study
  schemas, resealed-gate adversarial tests, and artifact-linked viewer
  validation.

All designs remain `SCREENING_NOT_ACCEPTED`. Boundary, residual, energy,
source-total and weak-action gates pass, but available grids resolve the
thinnest features with only `0.233–0.278` cells. Fixed-QoI mesh changes are
`26.6–27.2%`, alignment changes `3.48–23.4%`, and PM discrepancy changes
`42.4–86.0%`.
The predeclared limits were not tuned. The regenerated study used seven solves
per design and completed in about 46 seconds on the available controlled path.

## 2026-09-02 — L1b mathematical and replay hardening

- Replaced cut-cell coefficient averaging with exact face-normal series
  resistance, including analytic radial `r/nu` integration and exact
  line/polygon crossings for linear tapers.
- Closed the reviewer oracle: the previously arithmetic `67.11` case now
  returns exact `3.883`; interface-position and contrast sweeps agree with an
  independent series oracle to binary64 roundoff.
- Corrected equivalent PM authority to `M=Br/(mu0 mu_r)`. Recoil remanence is
  now deposited through face-integrated `G` fluxes and the same
  gradient/divergence adjoint pair as the operator.
- Added a symmetric far-field dipole Robin condition, while retaining two
  mandatory nested-domain checks.
- Added exact deterministic replay from accepted geometry/magnetics bundles
  and compressed binary64 solution data. Replay rebuilds coefficient/source
  arrays, operator action, true residual, energy, fields and all QoIs; top-level
  anchors now bind run, config, implementation, grid, domain and problem.
- Enforced three cells through every feature by default. The controlled RTX
  runs used 237x919, 283x876 and 261x1047 base grids, with 2x fine grids and
  expansions as large as 596x1815, 541x2169 and 484x1904.

All three v1.2 artifacts remain `SCREENING_NOT_ACCEPTED`. Fine-grid
recoil/equivalent differences now pass at `0.04797%`, `0.00542%` and
`0.01399%`; residual, energy, current, weak-action, resolution and expansion
count/factor gates pass. Publication is blocked by successive-domain QoI
changes of `2.543%`, `3.516%`, and `2.765%`, mesh changes of `1.974%`,
`14.594%`, and `1.601%`, compact/divergent alignment changes of `2.244%` and
`1.378%`, plus the historical boundary ratio of `0.4646%`. No threshold was
loosened and no artifact was promoted.

## 2026-09-02 — L1b re-audit qualification

- Corrected the dipole Robin operator for `psi=C r^2/rho^3`: axial
  `alpha=3|z-zc|/rho^2`, radial `alpha=3r/rho^2-2/r`. Analytic side and corner
  checks now close at roundoff.
- Added exact polygon/control-volume clipping for linear embedded boundaries
  and included geometry-only taper fractions in coefficient/material maps.
- Moved Warp alpha, beta, dot and vector updates fully to device arrays.
  Host reads now occur only every 25 iterations, at true-residual checks and
  final transfers; base-run synchronization counts are 145, 119 and 179 for
  3500, 2850 and 4350 iterations.
- Replay now rejects inconsistent stored relative residuals and requires one
  normalized physical geometry, material registry, solver configuration,
  implementation and backend identity across all seven runs.
- Upgraded bore QoIs to fixed fifth-order tensor-Gauss quadrature with radial
  area weighting. Added a conservative host-memory preflight.
- Raised qualification to twelve cells through every PDE-active material
  feature. Base grids are 670x1153, 516x1034 and 467x1814; largest domain
  grids are 1411x2388, 1070x2142 and 953x3758.

All v1.3 artifacts remain `SCREENING_NOT_ACCEPTED`. Boundary ratios now pass
at `8.71e-6`, `2.86e-6` and `1.10e-5`; PM equivalence, source, residual,
energy, current and resolution gates also pass. Compact fails domain/mesh/
alignment at `2.223%/5.302%/1.353%`; divergent at
`1.829%/2.624%/1.299%`; historical fails domain/mesh at `0.469%/2.344%`
while alignment passes at `0.183%`. No limit was changed and no unverified
P2/FEM claim was added.

## 2026-09-02 — L1b v1.4 recovery checkpoint

- Recovered the interrupted v1.4 implementation without resetting concurrent
  work. Strict role order/cardinality, phase-locked domains, composite
  cell-intersection bore quadrature, all active geometry polygons, third-grid
  actual-h diagnostics, and formal CPU/CUDA evidence are implemented.
- Added resumable per-design generation and bounded replay retention so a
  ten-run bundle does not retain every full-resolution replay field in memory.
- Focused solver/artifact tests passed (`22 passed`) after the recovery edits.
- Historical 467x1814 / 584x2268 / 730x2835 and compact
  670x1153 / 838x1442 / 1047x1802 studies completed. Both remain
  `SCREENING_NOT_ACCEPTED`; their minimum observed order is `0.0`.
- Historical fails expansion-factor (`1.499816 < 1.5`), mesh (`2.344%`) and
  observed-order gates. Compact fails expansion-factor (`1.499487 < 1.5`),
  mesh (`5.278%`), alignment (`1.398%`) and observed-order gates. CPU/CUDA
  field L2 parity passes at `6.83e-15` and `8.86e-10`, respectively.
- The divergent high-resolution rerun completed all solves once, but was
  invalidated when evidence-code files changed concurrently before assembly.
  Bounded retries then stopped at the unchanged memory preflight: available
  physical memory fell to `0.02 GiB`, below the `1.59 GiB` fine-grid estimate.
- The current evidence implementation SHA-256 is
  `9336db181da921d467da714621f818035bf1a8ed3c99dc38f36b1194d84a403f`;
  existing historical/compact v1.4 files still carry older evidence hashes and
  therefore require strict replay/resealing after memory is available.

This is a recovery checkpoint, not a completed artifact campaign. The
divergent file remains v1.3 and no screening result was promoted.

## 2026-09-02 — Interrupted v1.4 recovery boundary

- Recovered actual third-grid runs for historical (730x2835) and compact
  (1047x1802), including exact ten-role cardinality, CPU/CUDA parity and
  `STRUCTURED_GRID_L1B_INSUFFICIENT` evidence. Neither artifact is accepted.
- Diagnosed the divergent 807x1616 raster failure as global-coordinate
  shoelace cancellation: a fully covered `1.12438578895061e-8 m2` cell
  exceeded itself by `1.3336e-20 m2`. Local-coordinate polygon area evaluation
  fixes the original preregistered grid without relaxing overlap tolerances.
- Several overlapping recovery generators changed evidence code during active
  runs. Their partial outputs were not merged. Redundant full/historical and
  duplicate divergent processes were stopped; no material-field process
  remains.
- The final divergent solve could not be resumed safely: the memory preflight
  reported only `0.01–0.29 GiB` free and rejected a `1.02 GiB` raster. The
  checked-in artifact directory therefore remains an explicitly incomplete
  recovery state: historical/compact are v1.4 candidates requiring strict
  replay under the final code, while divergent remains the last v1.3 screening
  artifact and must not be combined into a v1.4 manifest.
- Focused code tests passed 26 of 27; the sole failure is the intentional strict
  artifact replay detecting the stale raw-problem contract. Compileall, spec
  JSON parsing, whitespace and FYP checks passed. Acceptance limits were not
  changed and every extant result remains `SCREENING_NOT_ACCEPTED`.

## 2026-09-02 — L1b v1.4 contract and qualification correction

- Removed the universal-positive/SPD Robin claim. The radial logarithmic
  coefficient is allowed to be negative near corners; analytic side/corner
  checks, an exact small-grid minimum-eigenvalue audit, converged PCG and
  positive magnetic energy now provide the stated numerical evidence.
- Closed publication evidence to exactly one base, fine, third-grid and
  alignment run, two ordered domain expansions, two PM-form runs, and one
  ordered CPU/CUDA parity pair. Added normalized solver-config identity,
  phase-lock, parity, positive-energy and observed-order gates.
- Bore averages now use composite cell-intersection Gauss integration over
  the represented piecewise-bilinear field. Every PDE-active geometry polygon
  is enumerated in feature-resolution evidence.
- Preregistered third grids are compact 1047x1802, divergent 807x1616, and
  historical 730x2835. Richardson values are diagnostic only; order below 1.5
  marks the structured-grid method insufficient.
- RTX qualification could not be completed on this host. Three attempts were
  OS-terminated under memory pressure; staged logging localized the historical
  failure to construction of the 584x2268 fine problem. The corrected
  preflight now reports only 0.08 GiB safe working memory versus 1.62 GiB
  estimated even for the 467x1814 base and fails before allocation. Existing
  v1.3 artifacts remain screening evidence and must not be promoted.

## 2026-09-02 — Frozen bundle recovery blocked safely

- Added atomic per-role raw-run checkpoints with inner-hash and sidecar
  verification. Python objects are collected and the CUDA device is
  synchronized with a zero Warp mempool release threshold after every role.
- Bound each campaign to frozen geometry, evidence, CPU solver, CUDA solver,
  and generator identities. The only CPU solve remains the formal CPU parity
  role.
- Strict validation rejects all extant artifacts: historical and compact do
  not satisfy the closed v1.4 acceptance object, while divergent has the older
  unsupported schema. The mixed manifest and its sidecar were removed; a
  fail-closed blocker notice remains beside the individual screening files.
- No simulation was started. Free physical memory was 0.427–0.747 GiB with
  85.3% CPU and 100% CUDA utilization from other workstreams. The conservative
  per-role estimates range from 1.021 GiB for divergent base to 6.840 GiB for
  historical domain-2; the 40% safety policy requires 17.100 GiB free physical
  memory for the complete frozen campaign.
- The complete material-fields suite reports 26 passed and one expected stale
  artifact failure. Focused Robin, cut-cell, and composite-quadrature
  regressions report six passed; compilation and whitespace checks pass.

## 2026-09-02 — L1b v1.4 memory-limited screening closure

- Preserved the local-coordinate cut-cell polygon-area correction and
  regenerated all three designs under one frozen v1.4 evidence identity.
- The host preflight initially found only 0.29--0.40 GiB free RAM while
  unrelated work owned the memory; no process was killed. Preregistered bases
  historical 467x1814, compact 670x1153 and divergent 516x1034 were not run.
- Used the largest common ten-role plan below the unconditional 64 MiB raster
  bound: 60x120 base, 75x150 fine, 83x166 third, 80x160 parity, and
  phase-locked expansions no larger than 127x250 (62.75 MiB estimated).
- Every artifact records high-resolution qualification as `NOT_EVALUATED`
  with `HOST_MEMORY_LIMIT`. All 18 publication gates are now explicitly
  `NOT_EVALUATED`; measured reduced-grid values and unchanged thresholds remain
  separately labelled `MEASURED_REDUCED_RESOURCE_ONLY`.

All designs remain `SCREENING_NOT_ACCEPTED` and
`STRUCTURED_GRID_L1B_INSUFFICIENT`. No reduced result passes or fails a
publication gate because the high-resolution prerequisite is absent. Strict
replay rejects the legacy boolean shape and any forged reduced-resource
`PASS`; viewers expose only tri-state mesh status, and the P2 comparison
handoff carries no structured gate fields. Numerical simulation arrays and
gate thresholds were unchanged; only evidence metadata and hashes were
regenerated. P2/body-fitted FEM remains the qualification handoff.

Final strict replay passed for all 30 embedded runs. The material-fields suite
passed 34 tests; compileall, schema parsing, whitespace, viewer/bundle replay
and the traceable-FYP compatibility check passed. Final payload hashes are
compact `7579f1602c75cdf6773b24279dd7621dd9c290dd9c2205b875da0962e5c7ed67`,
divergent `da7ef3f3660f1b2e6f6ea3ac9840bed23b1f177efc9168f75d3c90f5c5f12966`,
historical `d91f4dd8b86251ec4294948b7df4e8a700ec362e45dc920752fca4592039a860`,
and manifest
`32ce64983a03fc7278be1ceaa7bf20fb73e9b04926786530a1801148893ba134`.

## 2026-09-03 — Implementation digests re-bound after the LF pin

- Defect: after the repo-wide `eol=lf` pin, strict validation refused all
  three v1.4 artifacts with `raw run hash binding failed` (four failures in
  `tests/material_fields/test_spec_ledgers.py`). The recorded
  `implementation_sha256` / `evidence_implementation_sha256` values were the
  SHA-256 of the **CRLF** working-tree bytes of the seven bound source files at
  the generating commit `8603a905`; Git stores LF and the digests differ.
- Audit (`examples/material_fields/POSTHOC_AUDIT.md`,
  `audit_implementation_eol.py`, `tests/material_fields/test_posthoc_eol_audit.py`):
  hashing the `8603a905` blobs with `\n` → `\r\n` reproduces all three
  recorded digests exactly; the live source is those blobs byte-for-byte; every
  artifact was byte-exact against its sidecar and payload seal. The source
  content that produced the evidence is unchanged.
- Resolution: a CRLF tolerance cannot live in verification code here because
  `acceptance.py`, `replay.py` and `numerics.py` are themselves hashed. Ran
  `refresh_artifact_metadata.py` once (strict replay of all 30 embedded runs
  passed). Structural diff before/after: only `implementation_sha256`,
  `evidence_implementation_sha256`, `run_sha256`, `base_run_sha256`,
  `cpu_run_sha256`/`cuda_run_sha256`, payload seals and manifest file/payload
  digests changed; raw solutions, problems, diagnostics, all 18 gates and
  summaries are identical. A second run reproduced the same bytes.
- Fixed `refresh_artifact_metadata.write` to emit sidecars with
  `newline="\n"` (it wrote CRLF on Windows, the orbit_mc 1.6 pattern).
- New payload digests: compact
  `a6b69d03bde3626f31858b2b914ef8b6d2d9bdc26a3925371b7914a300f60da0`,
  divergent `cf703753108f36a0d175e0540262b367d034080141634a062c8820c582194a06`,
  historical `dc1ab5ed462fd34271db64866fb45097767ce64c0f0129bcae69591a68244dcf`,
  manifest `eba362d8b18f46f8e5254eceecb4092c0e12b35673a011014a3751c43113ae7c`.
  Recorded implementation digests are now the LF digests `ef17d161…`
  (evidence), `6ced73da…` (warp), `2ce98ebd…` (python). Status remains
  `SCREENING_NOT_ACCEPTED`; `tests/material_fields`: 40 passed.
