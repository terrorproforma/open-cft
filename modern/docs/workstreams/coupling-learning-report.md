# Coupling Learning Scratchpad

File policy: `COMMITTED` workstream record. This coupling-prefixed file is used
instead of the repository-wide default because this task owns only new
coupling paths.

## 2026-09-01 — Topology coupling implementation

- [user] Keep all writes under the new coupling source, test, spec, and
  coupling-prefixed workstream documentation paths. Preflight guardrail:
  inspect shared/field/physics code read-only and verify the FYP diff at
  handoff.
- [self] Existing `FieldMap` data has SI-suffixed attributes but no complete
  consumer freshness/model-hash contract. Require separate
  `FieldProvenance`; do not infer trust from object type or filename.
- [self] A map-content hash and field-model hash answer different questions.
  Hash coordinates/components exactly so resampling changes map identity while
  preserving the separately supplied model identity.
- [self] A signed `B_z` crossing alone is not a vector null off-axis. Gate the
  interpolated crossing with `B_r` and retain wall magnitude extrema
  separately.
- [self] Exact nulls are also magnitude minima. Select the null as the segment
  representative but preserve the coincident minimum as an alternative with
  its evidence; do not overwrite equal candidates.
- [self] An ideal zero-field mirror ratio is mathematically unbounded but
  nonfinite JSON would violate the artifact boundary. Encode the ratio as
  `None`/`null` and retain the finite low/high ratio and zero loss probability.
- [self] The loss-cone derivative diverges at `B_low/B_high=1`. Use first-order
  propagation only below that endpoint and bounded monotonic intervals at the
  endpoint.
- [tool] The project test path uses `modern/src` via `PYTHONPATH`; no install is
  needed. Initial focused execution passed 23 tests before spec tests were
  added.
- [self] A non-package pytest directory imports test modules by basename.
  Naming the first spec test `test_spec.py` collided with the optimization
  workstream during full collection. Rename workstream tests distinctly
  (`test_coupling_spec.py`) before claiming compatibility.
- [tool] Concurrent untracked plasma/magnetics/PIC workstreams can change full
  collection and execution during this branch. Verify the owned suite, then
  run the complete Git-tracked test list separately to distinguish coupling
  regressions from moving external failures.
- [tool] Ruff and mypy are declared optional but absent locally. Respect the
  no-install constraint and record compileall, tests, JSON parsing, and manual
  line-length checks instead of silently installing tools.

## Guardrails for follow-up work

- Never restore fixed axial windows or positional p1-p4 reversal in the new
  coupling API.
- Never collapse tied/symmetric candidates unless an explicit tie policy is
  recorded in the coupling-model hash.
- A confidence score is numerical evidence, not a calibrated probability.
- A topology-derived isotropic loss-cone probability is not a validated plasma
  transport closure.
- Any future covariance-aware propagation must version the equation ledger,
  schema, and coupling-model hash together.

## 2026-09-01 — Audit correction loop

- [user] Structural protocols are interoperability contracts, not evidence of
  acceptance. A raw map plus self-declared hashes must never reach
  `build_coupling_record`; only a coupling-sealed token produced from exact
  bytes and an accepted format adapter may cross that boundary.
- [self] Checking `diagnostics.converged` alone repeated the original
  status-only acceptance defect. Require finite absolute and relative
  residuals, each under its separately declared tolerance, before issuing the
  token.
- [self] A global magnitude scale can turn a huge but nonzero field into a
  false null, and `left*right<0` overflows. Strict sign comparisons plus a
  scaled-sum root fraction are required.
- [self] Endpoint extrema are useful diagnostics but not interior topology.
  Store them under `boundary_extrema`; only an explicit policy may promote a
  boundary minimum.
- [self] Adjacent tolerance comparisons are not transitive. Plateau grouping
  must constrain total run span to prevent tolerance chaining.
- [self] A one-tesla tie floor collapsed physically distinct sub-nT
  candidates. Tie tolerance now uses candidate-local scale and an explicit
  absolute T floor.
- [self] Shared additive uncertainty is not two independent adverse errors.
  Delta propagation includes covariance/common-mode cancellation; interval
  propagation moves the common term with the same sign in low/high fields
  while retaining conservative independent extremes.
- [self] Canonical identity must cover acceptance evidence and consumption
  geometry, not only the map and algorithm. The v2 record hash includes exact
  artifact/source bindings, all implementation identities, diagnostics,
  freshness, roles/radii, uncertainty, intervals, and covariance.
- [user] The L1a artifact schema is concurrently changing. Keep its future
  loader behind `AcceptedArtifactAdapter`; document exact requirements but do
  not import unstable field implementation details now.

## 2026-09-02 — Final acceptance corrections

- [user] A frozen dataclass plus a copyable seal is not an evidence invariant.
  Make ordinary construction/replacement unavailable, but rely on complete
  deterministic reverification at every build rather than a process-secret
  object identity claim.
- [self] Token issuance validation becomes stale the moment time advances.
  Build must recheck timestamp/freshness and diagnostics as well as hashes;
  tests now advance reference time after valid issuance and mutate private
  snapshots to prove failure.
- [self] Revalidating only hash syntax cannot detect replacement with another
  valid-looking identity. Keep a separate domain-separated invariant digest
  outside the replaceable snapshot and recompute it before detailed checks.
- [self] Artifact schema acceptance and migration are different policies. The
  default is direct L1a 1.1 to 1.1. L1a 1.0 requires an explicit migration
  contract and adapter-ID allowlist; adding 1.0 to a generic accepted set is
  prohibited.
- [self] The direct covariance expansion subtracts nearly equal terms at
  `rho=1`. Rewrite it as a squared relative-error difference plus
  `2*r_L*r_H*(1-rho)` so proportional errors cancel exactly or within a
  measured binary64 ULP bound.
- [self] Python private names do not provide hostile-code security. State the
  limitation and distinguish API misuse resistance from signatures, process
  isolation, or native-memory integrity.
