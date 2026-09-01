# Coupling Workstream Devlog

## 2026-09-01 — Robust field/topology coupling

### Changed

- Added structural protocols and immutable records for generic axisymmetric
  maps, SI/provenance validation, deterministic map/model-policy hashing, and
  centreline/interpolated-wall profiles.
- Added signed-null and magnitude-extremum detection with linear/quadratic
  interpolation, endpoint/plateau/tie policies, preserved alternative
  candidates, evidence confidence, and topology-midplane segmentation.
- Added stable mirror/loss-cone relations with independent uncertainty and
  monotonic endpoint bounds.
- Added a JSON-safe coupling record and narrow global-solver projection
  carrying map, field-model, source, and coupling-model hashes.
- Added analytic/noisy/resampled/adversarial tests, a closed record schema,
  equation ledger, formulation, and integration instructions.

### Validation

- `python -m compileall -q src/cft_revival/coupling`: passed.
- Initial focused suite before schema coverage:
  `python -m pytest -q tests/coupling`: 23 passed.
- Final focused suite:
  `python -m pytest -q tests/coupling`: 32 passed.
- Complete tracked baseline suite with isolated import mode: 248 passed and
  one expected optional-pybind11 skip.
- Full shared-tree isolated run reached 468 passed/one skip before one
  concurrent plasma tolerance failure. A later shared-tree run reached 467
  passed/one skip with two concurrent magnetics API/test signature failures.
  These paths are outside coupling ownership.
- `python -m compileall -q src tests`: passed.
- Coupling specification JSON parsing: passed.
- `git diff --exit-code -- FYP`: passed; `git diff --check`: passed.
- Ruff and mypy were unavailable in the existing environment and were not
  installed.

### Risks and follow-ups

- Confidence is an evidence heuristic, not calibrated uncertainty.
- Component uncertainty currently uses caller-supplied absolute/relative
  magnitude bounds and assumes independent low/high errors.
- Integration into active physics/plasma code is intentionally left to those
  owning workstreams; this change does not claim a plasma closure.

## 2026-09-01 — Coupling audit correction (v2)

### Corrected

- Replaced raw structural-map record construction with an opaque accepted
  evidence token. Token issuance now requires exact artifact bytes, a strict
  adapter protocol, accepted schema/model level, artifact/map/source-binding
  SHA-256 verification, complete implementation identities, UTC freshness,
  and finite converged diagnostics under both declared tolerances.
- Added typed `resolved`, `ambiguous`, `no_topology`, and `degenerate` states.
  Boundary extrema are separate diagnostics and require explicit promotion.
  Tolerance chaining, one-tesla tie scaling, and uncertainty-submerged ripple
  segmentation were removed.
- Hardened convex interpolation, signed-root fractions, midpoint/quadratic
  interpolation, profile integration, uncertainty arithmetic, and mirror
  ratios against overflow/underflow and nonfinite publication.
- Added independent/residual correlation, covariance in T², shared additive
  common-mode uncertainty, PSD validation, covariance-aware delta propagation,
  and conservative shared-mode interval bounds.
- Expanded solver rows and the canonical v2 record hash across artifact/source
  evidence, schema/model/code/config/backend/adapter identity, diagnostics,
  timestamp/freshness, inner/wall roles and radii, uncertainty, intervals,
  covariance, confidence, candidates, and segments.
- Replaced the v1 schema/ledger with closed v2 specifications and documented
  the future exact L1a adapter without importing concurrent field internals.

### Validation

- Coupling default import mode: 64 passed.
- Coupling importlib mode: 64 passed.
- Complete Git-tracked baseline, default mode: 248 passed, one expected
  optional-pybind11 skip.
- Complete Git-tracked baseline, importlib mode: 248 passed, one expected
  optional-pybind11 skip.
- `python -m compileall -q src tests`: passed.
- Full concurrent tree probe: 681 passed and one optional skip, with seven
  failures/ten setup errors in concurrently changing surrogate and
  axisymmetric visualization/manifest workstreams.
- `git diff --exit-code -- FYP`: passed. No dependency was installed.

### Deferred integration

- The exact L1a byte loader remains intentionally deferred until its hardened
  artifact schema stabilizes. `coupling-integration.md` lists the required
  duplicate-key/nonfinite/schema/hash/binding/diagnostic tests.

## 2026-09-02 — Final acceptance hardening

### Corrected

- Replaced the dataclass/seal token with a non-dataclass, immutable,
  private-factory wrapper. Ordinary construction and `dataclasses.replace`
  now fail.
- Added deterministic reverification to every `build_coupling_record`: exact
  retained bytes, canonical map, artifact/map/source binding hashes, current
  schema/model/adapter contract, implementation identities, diagnostics,
  timestamp/freshness, map validity, and derived profile role.
- Stored a separate domain-separated snapshot-invariant digest in the private
  wrapper so replacement with a different valid-looking identity also fails.
- Changed the default/current L1a artifact schema to 1.1.0. Version 1.0 now
  requires an explicit 1.0-to-1.1 migration contract and an allowlisted adapter
  ID; direct/default adapters cannot accept it.
- Added adapter contract ID/version/input/normalized schema/migration fields to
  coupling records, canonical hashes, solver rows, and the closed schema.
- Rewrote correlated ratio variance as a squared relative-error difference
  plus a non-negative `(1-rho)` term. Perfectly correlated proportional errors
  now cancel exactly or within the tested four-ULP bound.
- Added adversarial private-snapshot replacement tests for bytes, map, map
  hash, source binding, timestamp, convergence, and residual tolerance, plus
  post-issuance staleness and migration-policy tests.

### Validation

- Coupling default import mode: 76 passed.
- Coupling importlib mode: 76 passed.
- Complete Git-tracked baseline default mode: 248 passed, one expected
  optional-pybind11 skip.
- Complete Git-tracked baseline importlib mode: 248 passed, one expected
  optional-pybind11 skip.
- `python -m compileall -q src tests`: passed.
- `git diff --exit-code -- FYP` and coupling-only scope check: passed.
- No dependencies installed; no commit or push performed.

## 2026-09-02 — Physically meaningful flux-surface v3

### Changed

- Added private-factory v3 evidence over exact `r,z,ψ,Br,Bz` bytes and bound
  artifact/source/geometry/material/mesh/domain/model/code/config/backend
  identity with build-time diagnostics and freshness reverification.
- Added dependency-free marching squares, saddle disambiguation, metric
  connectivity, closure/boundary diagnostics, bilinear ψ residual checks, and
  explicit rejection of same-z/different-ψ sample pairs.
- Separated finite-box endpoint zeros from geometry-identified interior
  cusps. Added mandatory caller evidence for full/downsampled/enlarged-domain
  cell-count and cusp-position stability.
- Added preregistered per-cell flux quantiles and preserved every local
  connected contour component as a surface distribution.
- Added bounded field/interpolation/surface propagation and nominal-value
  suppression for uncertainty-dominated surfaces.
- Added electron gyroradius/field-scale-length gating. Exact/unresolved nulls,
  missing energy inputs, and nonadiabatic surfaces cannot publish mirror
  probabilities.
- Moved v2 same-z behavior behind deprecated `screening_proxy`; the package
  root accepted builder and solver projection now use only v3 types.
- Added v3 schema/equation ledger and manufactured island, dipole, X-point,
  endpoint, stability, closure, evidence-forgery, nonadiabatic, uncertainty,
  and extreme-binary64 tests.

### Validation

- Coupling default mode from `modern`: 88 passed.
- Coupling importlib mode: 88 passed.
- Compatible non-experiment importlib suite: 956 passed, one expected optional
  pybind11 skip.
- Full importlib probe: 1049 passed, one skip, with four legacy
  `l1a_plasma_coupling` failures/errors caused by the intentional root-API
  removal, plus ten unrelated existing experiment-result/manifest failures.
  A second probe excluding those consumers reached 1012 passes and only ten
  four-cell visualization setup errors from an existing dataset hash mismatch.
- `python -m compileall -q modern/src/cft_revival/coupling
  modern/tests/coupling`, `git diff --check`, and
  `git diff --exit-code -- FYP`: passed.
- The root-invoked default suite initially had three `tests` package import
  errors; rerunning from the correct `modern` project root passed.
- No package installation, commit, push, FYP edit, or non-coupling source edit
  was performed.

## 2026-09-02 — V3 coupling audit closure

### Corrected

- Replaced vertex-only field checks with adaptive quadratic-on-segment
  certificates carrying outward ULP margins, extrema bounds, gradient bounds,
  refinement diagnostics, and fail-closed interior-null detection.
- Corrected alternating-sign marching-squares cells with the scaled bilinear
  asymptotic determinant and explicit exact-saddle policy.
- Replaced greedy contour chaining with deterministic edge graphs and simple
  cycle validation against retraces, duplicate edges/vertices, branches,
  self-intersections, and boundary contact.
- Added one hash-visible outcome per preregistered quantile. Cell and record
  acceptance are now atomic; no successful subset can reach solver inputs.
- Applied positive finite `coverage_factor` to complete uncertainty bounds and
  added scaled overflow-safe arithmetic plus nonrelativistic energy validity.
- Expanded primary and all stability-case identities with artifact/binding,
  schema/model, source/geometry/material/mesh/domain, implementation/backend,
  timestamp, and freshness fields.
- Added stable extreme/subnormal root interpolation and kept v2 behavior only
  through the warning-emitting, non-projectable screening proxy.

### Validation

- Coupling default mode: 100 passed.
- Coupling importlib mode: 100 passed.
- Compatible suite excluding separately owned experiments/material-fields:
  948 passed with one expected optional-pybind11 skip.
- Broader non-experiment probe: 968 passed and one skip; its only two failures
  were concurrent `material_fields` schema/publication-identity defects outside
  coupling ownership.
- Coupling compileall, JSON schema/ledger parsing, `git diff --check`, and
  `git diff --exit-code -- FYP` passed.
- No installs, commits, pushes, FYP edits, or out-of-scope source edits were
  made.
