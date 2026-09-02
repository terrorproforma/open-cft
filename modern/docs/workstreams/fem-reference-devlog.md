# FEM reference devlog

## 2026-09-02 — implementation

- Confirmed installed numerical capabilities without changing the environment:
  NumPy 2.5.2 and Warp are available; SciPy, Gmsh, meshio, and triangle are not.
- Chose a required CPU-only NumPy implementation with native CSR/IC(0)-PCG.
- Added a deterministic conforming strip mesher for all accepted rectangular
  and linear-taper polygon boundaries, strict P2 edge topology, region tags,
  mesh validation, degree-five triangle quadrature, edge quadrature, sparse
  assembly, corrected Robin terms, solve diagnostics, field/QoI evaluation,
  and canonical hashes.
- Consumed geometry and magnetics through their public models and
  `to_magnetics_handoff`; no accepted/shared path was edited.
- Kept the formulation in \(\psi=rA_\phi\), while using regular P2
  \(A_\phi\) trial functions so that \(\psi=O(r^2)\) at the axis.

## 2026-09-02 — verification

- Smooth manufactured refinements 4/8/16 produced relative sampled errors
  `3.196e-3`, `3.827e-4`, and `4.851e-5`; observed orders were `3.062` and
  `2.980` (P2 sampled-L2 expectation: 3).
- Aligned and oblique piecewise-reluctivity cases with an `11:1` jump had
  relative solution errors `1.47e-12` and `7.61e-12`. Their true residual
  ratios were below `1.8e-11`. These exact cases exercise continuity with a
  derivative jump rather than hiding the interface in a smooth coefficient.
- The analytic dipole-Robin case at refinement 10 had relative solution error
  `4.54e-3`; refinement 12 passes the focused `<3e-3` check.
- Recoil-remanence and equivalent-sheet PM solutions differed by
  `1.80e-12` in relative max norm. Full polarity reversal negated the solution
  to reported roundoff, and homogeneous physical source/energy actions agreed
  below `1e-12`.
- A uniform-medium current band cross-check against L1a at three fixed axis
  points had maximum relative difference `1.240e-3`.

## 2026-09-02 — three-design campaign

The reproducible campaign used global `(radial, axial)` divisions `(12,24)`,
`(20,40)`, and `(32,64)`, plus every exact geometry breakpoint and polygon
track, with dipole-Robin padding factor `1.5`.

- Campaign wall time: `62.89 s`.
- Finest meshes: `9,847` to `10,613` P2 DOFs and `4,820` to `5,198`
  triangles.
- Finest per-run assembly: `1.27–1.40 s`; solve: `6.34–7.69 s`.
- Finest iterations: `171–187`.
- Estimated peak numerical working set: `3.15–3.40 MB`.
- Finest true-residual ratios: below `1.8e-9`.
- Homogeneous energy/source action mismatch: at most `4.4e-13`.

The one-percent all-QoI gate was **not reached**:

- historical envelope trailing maximum change: `3.66%`;
- compact stack trailing maximum change: `19.28%`;
- divergent stack trailing maximum change: `21.94%`.

Axis point derivatives dominate those maxima. Trailing bore-average changes
were also not all below one percent: historical reached up to `2.43%`, compact
up to `6.11%`, and divergent up to `1.86%`. No interface-cell maximum was used
for acceptance.

Against the published L1b structured-grid fixed/bore summaries at the finest
campaign level:

- historical differences span `0.82%–11.14%`;
- compact differences span `1.44%–15.15%`;
- divergent differences span `4.55%–13.18%`.

These are unresolved numerical discrepancies, not evidence that either method
is physically validated. The conforming results support the diagnosis that
fixed/bore values still need finer graded meshes and matched finite-domain
studies before using the FEM reference to replace L1b.

Artifacts and viewer contracts are under
`examples/fem_reference/artifacts/`; all include mesh, solution, provenance,
quality, convergence, and L1b comparison hashes. `replay_artifacts.py`
recomputes file, payload, mesh, solution, and viewer anchors.

## Verification commands

```text
python -m pytest tests/fem_reference -q
python -m compileall -q src/cft_revival/fem_reference tests/fem_reference examples/fem_reference
$env:PYTHONPATH='src'; python examples/fem_reference/replay_artifacts.py
git diff --check
git diff --exit-code -- FYP
```

Focused plus compatible tests passed `34/34`; full `src`, `tests`, and FEM
example compileall passed; three artifacts replayed; whitespace and FYP checks
passed. Ruff was unavailable and was not installed. No install, commit, FYP
edit, accepted-package edit, or GPU claim was made.

## 2026-09-02 — audit correction implementation

- Replaced the mislabeled bore-wall line sample with the exact L1b-compatible
  axisymmetric volume average
  `2/(R^2*dz) integral psi(R,z) dz`. The trace is split at every triangle
  crossing and integrated piecewise at P2 order. The old result survives only
  as `bore_wall_line_average`.
- Added weighted quadratic axis patch recovery. Axis values and wall-line
  values are diagnostic; finite-radius, finite-length volume averages are the
  acceptance QoIs.
- Replaced independent remeshing with a graded initial body-fitted mesh and
  deterministic nested midpoint refinement. Parent mesh hashes and element
  parent IDs prove ancestry.
- Added residual, constitutive normal-flux-jump, and normalized QoI proxy
  indicators with deterministic Dörfler `theta=0.5` marking.
- Manufactured verification now uses integrated axisymmetric L2 and energy
  norms and computes order from measured `h`: L2 orders are `2.998/2.998`;
  energy orders are `1.983/1.995`.
- Strengthened mesh hashing and replay to include every vertex, P2 node,
  triangle, edge, midpoint DOF and coordinate, element DOF, region/interface
  pair, boundary owner, parent ID, protected coordinate, index range, shape,
  and recomputed quality value. A topology edit still fails after resealing.
- Added fail-closed solver-control validation. Relative tolerance must be
  finite and strictly in `(0,1)`, absolute tolerance finite and nonnegative,
  and maximum iterations a positive non-boolean integer. PCG acceptance always
  recomputes the true residual; energy stationarity remains diagnostic.
- Vectorized native CSR products and used Jacobi-PCG for large references while
  retaining IC(0)-PCG for smaller verification systems. Both remain NumPy CPU
  paths.

The divergent audit demonstrated why a QoI line must not be a physical mesh
constraint: at the straight-to-taper transition it creates an unavoidable
`9.46 deg` wedge. Piecewise trace integration removed that artificial edge;
all three initial physical meshes exceed the `20 deg` target and reject any
adaptive level below `10 deg`.

## 2026-09-02 — timeout recovery and bounded strict-gradation campaign

- Recovered the completed audit corrections without resetting concurrent
  work. The focused baseline was `22/22` passing before recovery edits.
- Changed marking from one bulk set over a summed indicator to the union of
  separate residual, normal-flux-jump, and QoI-proxy Dörfler sets. Each
  component captures at least `theta=0.5`; observed marked fractions were
  `0.614–0.781`, `0.504–0.703`, and `0.5000–0.5006`, respectively.
- Added parent-level gradation closure. It promotes coarse neighbors to
  longest-edge or red refinement before constructing the child mesh. All six
  solved meshes pass adjacent area-size growth `<=1.3` with actual maxima
  `1.118–1.150`; minimum angles remain `32.89–38.66 deg`.
- Tightened campaign solves to a recomputed true-residual target of `2e-10`.
  Actual ratios were `1.956e-10–1.995e-10`.
- Recorded actual bore and source-feature `h`, made both fixed-axis and bore
  L1b comparisons explicitly identical physical QoIs with their distinct
  evaluation methods, and added six hash-sealed per-level checkpoints.

Strict `1.3` gradation below the binary-bisection factor `sqrt(2)` propagates
through the nearly uniform body-fitted hardware mesh. The first adaptive step
therefore becomes global red closure:

- historical: `66,987 -> 266,869` P2 DOFs;
- compact: `91,319 -> 363,965` P2 DOFs;
- divergent: `79,421 -> 316,489` P2 DOFs.

A second such step has an upper bound above the deterministic `400,000`-DOF
campaign limit, so it was not launched. This preserves the unchanged
gradation and `<1%` acceptance gates rather than weakening them. The one
available successive bore-volume change is already below `0.046%` for every
design, but two successive changes and a three-level observed order are
required; all three studies therefore remain screening-only with order
reported as unavailable.

Finest identical-QoI FEM/L1b differences are:

- historical: `0.102–1.364%`;
- compact: `0.145–2.939%`;
- divergent: `0.043–0.845%`.

The bounded campaign took `1,131.65 s`; the largest solve used `363,965` P2
DOFs, `181,328` triangles, `3,430` iterations, `118,895,560` estimated working
bytes, and `202.41 s` assembly-plus-solve time. Final verification passed
`23/23` focused tests, compiled all owned Python paths, replayed three
artifacts plus six checkpoints, passed whitespace checks, and left `FYP`
unchanged.

## 2026-09-02 — estimator, sparse-memory, and authority audit

- Replaced the midpoint expression `length * jump^2` with the standard
  `h_e * integral_e jump^2 ds`, using three-point edge Gauss quadrature. A
  constant jump now scales exactly as `length^2 * jump^2`.
- On manufactured levels `4/8/16`, estimator norms decrease
  `0.128385 -> 0.0310158 -> 0.00763442`; energy-error effectivities remain
  bounded and trend `10.063 -> 9.614 -> 9.431`. QoI-marked mean-distance
  ratios fall `0.605 -> 0.168 -> 0.00108`, confirming deterministic
  localization.
- Replaced assembly row dictionaries with a preallocated topology COO-key
  pass and direct numerical CSR pass. On the unchanged `4,959`-DOF,
  `54,520`-nonzero probe, `tracemalloc` peak fell from `6,810,648` to
  `2,914,493` bytes (`57.2%`); assembly time changed from `3.98` to `5.33 s`.
  The slower small-case assembly is retained because the memory scaling is
  deterministic and removes Python-object amplification.
- Added direct-parent P2 prolongation. On the small two-level probe, cold/warm
  solves took `40/37` iterations and differed by `9.53e-12` in relative
  solution norm.
- Reworked initial meshing so the hardware geometry/material/QoI support uses
  the fine scale while a conservative geometric radial size field grades the
  exterior. Initial design meshes use `64,203/82,967/75,357` P2 DOFs,
  minimum angles `33.07/27.67/32.49 deg`, and adjacent area-size growth
  `1.118/1.174/1.150`; topology remains deterministic.
- Raised the policy ceiling to `1,500,000` P2 DOFs without changing accuracy
  gates. Third-level execution now requires explicit opt-in, exactly one
  design, and at least `8,589,934,592` free bytes. The live preflight reported
  only `12,083,200` bytes free, so no third-level run was launched.
- Added a separate, currently unmet Robin domain-expansion requirement:
  padding `0.5/1.0/1.5`, fixed phase-matched source/QoI local `h`, and
  successive QoI change below one percent.
- Schema `1.2` artifacts recompute QoIs, residual/actions, comparisons,
  actual-`h` changes/orders, and status from bound mesh/solution/config/code
  evidence. Rehashed QoI and status attacks fail. Existing large `1.1`
  artifacts are explicitly legacy integrity-only screening records.
- Migrated all six checkpoints to chained file/mesh ancestry and anchored
  every file/payload hash in the sealed campaign manifest. No self-hash is
  treated as external authority.
- Final bounded verification passed `31/31` FEM tests in `87.18 s`, including
  matrix/solution/QoI parity, estimator trends, graded-mesh quality,
  fail-closed resource policy, authoritative artifact attacks, and legacy
  checkpoint replay. Compileall, whitespace checks, and the `FYP` no-diff
  check passed. No install, commit, or heavy campaign run was performed.

## 2026-09-02 — final adaptive-readiness blockers

- Replaced prolongation point location with direct child-parent P2 basis
  evaluation through `element_parent_ids`. A four-refinement scaling probe
  covered `693/2,665/10,449/41,377` fine DOFs in
  `9.95/48.60/224.67/759.08 ms`; normalized costs remained
  `5.18/6.33/7.31/6.18 us` per six-DOF child-element evaluation. Correctness
  is checked against an exactly representable quadratic while global element
  location is monkeypatched to fail.
- Promoted new artifacts to schema `1.3`. Every adaptive checkpoint now embeds
  a complete bound artifact: problem and source-quadrature configuration,
  full mesh/topology, solver controls, and solution arrays. Canonical file and
  ancestry hashes remain manifest-anchored; each numerical array additionally
  binds little-endian dtype, shape, C order, and byte SHA-256. Acceptance loads
  these checkpoints and independently recomputes residuals, QoIs, local `h`,
  changes, orders, comparisons, domain evidence, and final gates. Resealed
  summary/order and dtype attacks fail.
- Domain replay now derives extents, QoI-region `h`, and source-region local
  `h` from each checkpoint's bound grid. Padding is bound inside its checkpoint;
  nonfinite/nonpositive resolution, nonfinite/nonnested extents, phase mismatch,
  or NaN QoIs fail before the one-percent gate.
- Added a calibrated allocation model for COO keys, sort workspace,
  unique masks/keys, conservative CSR, mesh/Krylov temporaries, and allocator
  overhead. It applies `1.75x` plus `256 MiB`. On the 4,959-DOF/2,408-triangle
  probe, `tracemalloc` was `2,914,469` bytes, retained RSS increased
  `675,840` bytes, modeled memory was `273,318,699` bytes, and the effective
  guarded requirement was `478,307,723` bytes.
- Projected third-level model requirements are `2,335,293,702` bytes
  (historical), `3,014,970,630` bytes (compact), and `2,682,651,174` bytes
  (divergent). The unchanged strict `8 GiB` floor dominates all three, so each
  revised per-design execution trigger is `8,589,934,592` currently available
  bytes. The latest probe found only `319,053,824` bytes; no heavy run started.
- RAM is queried immediately before every guarded level and again inside
  assembly. A failed level preflight writes a sealed resource-abort checkpoint
  referencing the current mesh and previous checkpoint before raising.
- Final clean verification passed `34/34` FEM tests in `69.29 s`, including
  forged evidence, four-level prolongation scaling, memory calibration/RSS,
  legacy campaign replay, and all prior manufactured/interface/PM checks.
  Compileall, JSON specification parsing, and the guarded campaign CLI passed.
  No install, commit, FYP edit, or heavy run was performed.

## 2026-09-02 — cap, chain, and shared-guard closure

- Moved positive integer/non-boolean dimension, `1,500,000` P2-DOF, and
  projected triangle/boundary topology enforcement into the common allocation
  guard. Initial mesh and refinement compute topology before constructing
  vertex/child arrays; assembly, solve, artifact, checkpoint, replay, and
  validation independently call the same guard. Tests accept exactly
  `1,500,000` DOFs with sufficient simulated RAM and reject `1,500,001` and
  `3,000,000` before allocation.
- Added typed `ResourceBlockedError` with `NOT_EVALUATED`. Publication uses
  guarded atomic temporary-file replacement. The calibrated estimate now
  includes sixfold checkpoint parse/decompression/serialization buffer
  expansion and a scale-dependent reserve up to `256 MiB`. Readiness work
  below both `100,000` P2 DOFs and `64 MiB` serialized state remains
  dimension/cap/topology checked while the live physical-RAM gate applies to
  heavy allocation.
- Replaced complete checkpoint JSON arrays with compressed NumPy sidecars.
  Metadata and sidecars are separately SHA-256 bound; ZIP central-directory
  uncompressed sizes are guarded before lazy loading, and each decompressed
  array replays its little-endian dtype, shape, C-order, and byte hash.
- Bound every adaptive/domain checkpoint to one authority root containing
  artifact schema/classification, design, geometry, magnetics, config, code,
  and base-problem identities. Ordered anchors bind parent checkpoint/mesh and
  common final run/mesh identities. A complete three-level schema-1.3
  synthetic sidecar chain replays successfully; a resealed internally valid
  checkpoint declaring an unrelated design is rejected. Forced zero-RAM
  replay returns typed `NOT_EVALUATED`.
- Revised projected assembly-only requirements are `2,331,914,567` bytes
  (historical), `3,014,970,630` bytes (compact), and `2,682,651,174` bytes
  (divergent). The unchanged `8,589,934,592`-byte third-level floor remains
  the effective trigger for every design.
- Final verification passed `36/36` FEM readiness tests in `90.79 s`,
  including exact-cap/over-cap, successful binary-chain replay, foreign-chain,
  zero-RAM replay, forged evidence, prolongation scaling, and legacy artifact
  coverage. No heavy third-level campaign was launched.

## 2026-09-02 — verified-header and finalization closure

- Changed third-level startup low-RAM rejection to typed
  `ResourceBlockedError(status="NOT_EVALUATED")`, matching level, topology,
  and cap failures while preserving the strict `8 GiB` floor.
- Removed anchor dimensions from checkpoint guard sizing. Metadata is bounded
  to `8 MiB`; ZIP central-directory entries and NPY headers are inspected first
  to derive actual P2 DOFs, triangles, boundary counts, dtype, shape, order,
  and uncompressed bytes. The guard uses those values, then requires exact
  anchor equality. A resealed understated-count attack now fails.
- Added direct bound-artifact identity checks for artifact schema,
  classification, design, geometry, magnetics, config, normalized problem,
  run, implementation, and acceptance code. Adaptive problems must match the
  top-level problem; each domain problem is independently anchor-bound while
  sharing the top authority root. An internally valid foreign-geometry domain
  checkpoint is rejected.
- Preliminary checkpoints now use bounded metadata plus compressed binary
  sidecars immediately. Finalization calls the same streamed-hash,
  header-verified, resource-guarded loader and bounded metadata summary; it no
  longer performs full `read_bytes()` or unguarded JSON parsing. Legacy
  preliminary JSON over `8 MiB` is maximum-topology resource-gated before
  streaming parse/migration, and returns typed `NOT_EVALUATED` under low RAM.
- Final verification passed `37/37` FEM readiness tests in `86.12 s`,
  including typed startup blocking, understated-header counts, successful
  sidecar-chain replay, foreign bound/domain identity, guarded legacy
  finalization, and all earlier manufactured/interface/PM checks.

## 2026-09-02 — guarded third-level numerical qualification

- Resource preflight reported `36,713,521,152` available bytes before launch.
  All designs ran one at a time with explicit third-level opt-in; no guard was
  relaxed and no hardware-validation claim is made.
- Historical-envelope adaptive levels used `64,203 / 255,733 / 1,020,777`
  P2 DOFs and `31,832 / 127,328 / 509,312` triangles. Minimum angle remained
  `33.071 deg`, adjacent area-size growth `1.118`, and peak modeled numerical
  working set was `334,053,096` bytes. Finest bore QoIs were
  `-0.278884373 / 0.213149874 / -0.325274222 T`; observed orders were
  `1.1035 / 1.3387 / -0.2005`. Both adaptive changes were below one percent
  and the domain gate passed, but the negative stage-3 order keeps this design
  screening-only.
- Compact adaptive levels used `82,967 / 119,575 / 330,581` P2 DOFs and
  `41,162 / 59,466 / 164,648` triangles. Minimum angle was `27.672 deg`,
  maximum growth `1.296`, and peak working set `107,949,256` bytes. Finest
  bore QoIs were `-0.138711205 / 0.138699923 / -0.140722471 /
  0.138197945 / -0.220578826 T`; orders were `-0.6307 / 2.0683 / 1.0497 /
  2.4643 / -5.2817`. Adaptive and domain change gates passed, but stages 1
  and 5 have negative order, so this remains screening-only.
- Divergent adaptive levels used `75,357 / 300,233 / 1,198,545` P2 DOFs and
  `37,380 / 149,520 / 598,080` triangles. Minimum angle was `32.486 deg`,
  growth `1.150`, and peak working set `392,258,344` bytes. Finest bore QoIs
  were `-0.115623310 / 0.110014291 / -0.109022075 / 0.171895655 T`;
  observed orders were `1.3483 / 1.3534 / 1.3776 / 1.3955`. Both adaptive
  changes were below one percent, all orders were positive, and the
  phase-matched domain gate passed. This design achieves numerical P2
  qualification only.
- Domain-study DOFs for padding `0.5 / 1.0 / 1.5` were
  `64,203 / 123,435 / 199,253` (historical),
  `82,967 / 165,633 / 291,573` (compact), and
  `75,357 / 141,597 / 245,597` (divergent). Maximum successive domain changes
  were respectively `7.600e-4 / 6.384e-5`, `8.355e-4 / 4.515e-5`, and
  `2.665e-3 / 3.289e-4`; every domain gate passed.
- Final artifact hashes are `ea5c6db8...01e9d` (historical),
  `cdc12666...7a63` (compact), and `6c326120...f133` (divergent). Independent
  replay returned exact mesh/solution hashes and zero psi replay error for all
  three.
