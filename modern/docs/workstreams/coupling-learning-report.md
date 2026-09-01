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

## 2026-09-02 — Flux-surface acceptance corrections

- [user] Same axial coordinate is not a magnetic field-line identity. Accepted
  mirror extrema must be sampled on one connected constant-ψ component; axis
  `ψ=0` and wall `ψ!=0` are a prohibited physical comparison.
- [user] Exact magnetic nulls invalidate the first-adiabatic-invariant mirror
  model. Keep nulls as separatrix geometry, never as a roundoff `B_low`, and
  require particle-energy/gyroradius scale evidence before publication.
- [self] A topology count from one finite mesh is not acceptance evidence.
  Bind full, downsampled, and enlarged-domain cases and fail closed on count or
  cusp-location instability.
- [self] Marching-squares output needs physics-facing diagnostics beyond a list
  of points: ψ residual, connectivity gap, closure, finite-box contact, and
  component identity must survive into the record.
- [self] Flux-surface choice is an analysis degree of freedom. Quantiles and
  cell IDs must be preregistered, strictly interior, and hashed into the record;
  preserve component distributions instead of selecting a favorable line.
- [self] Near nonlinear limits, monotone bounded propagation is identifiable
  and fail-closed. If uncertainty overwhelms the nominal estimate, omit the
  nominal probability rather than presenting false precision.
- [tool] Windows PowerShell in this environment does not accept `&&`; use
  sequential commands guarded by `$LASTEXITCODE`.

## 2026-09-02 — V3 contour audit corrections

- [user] Nonzero contour vertices do not prove a regular field line. Under
  bilinear interpolation, both field components can cross zero inside an edge;
  certify the full segment with adaptive derivative bounds before publishing.
- [self] The arithmetic cell-centre sign is not the marching-squares
  asymptotic decider. Use the scaled bilinear determinant and test rotations,
  global sign inversion, and exact ties against analytically known pairings.
- [user] Quantile preregistration is atomic. Recording only traced surfaces
  loses absent quantiles and permits successful-subset bias; every requested
  quantile needs a retained outcome and one failure blocks the entire cell.
- [self] Endpoint chaining can disguise retraces and figure-eight paths as
  closure. Build an edge graph, then independently revalidate retained points
  for repeated vertices/edges, branching, intersections, and boundary contact.
- [self] A coverage factor carried only in metadata is ineffective. Multiply
  the complete uncertainty bound by it before ratio propagation and test that
  changing it widens intervals and can change acceptance.
- [self] Complete stability identity means all three artifacts, not only three
  map hashes. Carry each artifact/binding/provenance/implementation/freshness
  identity into the record hash.
- [self] Opposite extreme signed values must never use `right-left`; a
  ratio-scaled absolute-magnitude interpolation locates both ±1e308 and
  subnormal midpoint roots.

## 2026-09-02 — HEMP wall-cusp v4 recovery loop

- [user] A HEMP cusp is a stable wall-normal/radial field maximum with
  wall-intersecting bundles; requiring an X-point, O-point, null, or closed
  contour would preserve the wrong v3 topology contract.
- [self] A map-set invariant over only three map hashes does not preserve
  provenance. Exact artifacts with identical field arrays can carry different
  implementation, adapter, diagnostic, or freshness identities, so the v4
  fingerprint covers every accepted-evidence field.
- [self] “Refined” cannot mean one dimension grew while another shrank.
  Require componentwise nondecreasing sample counts, one strict increase, and
  the same domain. “Enlarged” must contain every primary bound before extending
  one.
- [self] Path coordinates alone do not bind an orbit claim to a field map.
  Include full-map hash and ψ label in the trajectory identity, then require
  returned orbit claims to echo both path hash and sample ID.
- [self] If uncertainty removes the positive `B_low` lower bound, computing a
  gyroradius with a numerical floor creates false adiabatic evidence. Preserve
  diagnostics but type the path `uncertainty_dominated` and publish no nominal
  probability.
- [self] A mutable criterion status is not validation evidence. Status changes
  only after exact artifact verification against criterion version,
  development-set identity, preregistration hash, disjointness, all-case
  success, diagnostics, code hash, and freshness.
- [self] Solver projection is a second trust boundary. Recompute the canonical
  record hash and recheck held-out, cusp, cell, map, orbit, and stability gates
  instead of trusting a replaceable `resolved` enum.
- [tool] The initial root invocation reproduced the known `tests` package
  collection issue. Running from `modern/` established the useful baseline:
  100 passing tests and seven local v4/spec defects.
- [tool] The configured `learning-scratchpad-loop` and `devlog-loop` skill
  definitions were unavailable in this workspace. Their intended continuous
  effects were maintained directly in this coupling-only learning record and
  devlog.

## 2026-09-02 — V4 audit-defect learning loop

- [user] A held-out claim is not verifiable merely because an adapter returns
  `disjoint=True`. Preserve exact development and held-out case/family sets,
  recompute their manifest hashes and intersections, and require explicit
  complete outcomes.
- [self] “This record was evaluated” needs content membership, not a family
  label. Bind the evaluated case and family to the exact primary/refined/
  enlarged map-hash tuple carried by one held-out outcome.
- [self] Orbit convergence is implementation-dependent. Preregister adapter,
  model, and convergence policy IDs and versions together with all code/config
  hashes; echoed claims must equal that identity exactly.
- [self] Adjacent-sample prominence tends to zero under refinement even for a
  fixed analytic cusp. A physical support window, barrier-aware side minima,
  quadratic peak interpolation, and physical separation remain meaningful as
  81/161/321 grids change.
- [self] Checking a wall crossing only after RK4 is too late because midpoint
  or endpoint stages may already have sampled outside the dielectric channel.
  Guard every stage, halve before crossing, and interpolate only the final
  tolerance-bounded remaining arclength.
- [self] A stability branch for unequal cusp counts is dead code if map
  preprocessing raises first. Per-map assessment must retain zero/extra cusps,
  empty/partial cells, typed status, and role-specific reason so the
  cross-map record can fail atomically and diagnostically.
- [self] Validation freshness is part of preregistration, not a verifier-only
  default. Record and recheck maximum age and future skew at projection.

## 2026-09-02 — V4 projection trust-boundary loop

- [user] A canonical hash proves that a summary is internally serialized as
  claimed; it does not prove that the summary came from accepted evidence.
- [self] Projection authority must retain the original accepted map artifacts
  and held-out bytes privately. Rebuilding at every explicit evaluation time
  turns freshness, diagnostics, provenance, and all physical outcomes back
  into verified facts instead of caller-editable claims.
- [self] A field-value hash cannot distinguish identical arrays generated by
  another model, implementation, configuration, backend, mesh, timestamp, or
  nonconverged solve. Held-out membership therefore needs complete,
  role-ordered evidence fingerprints in addition to map hashes.
- [self] Atomic projection is stricter than a resolved top-level enum. Every
  preregistered cell, seed, direction, and orbit sample must remain present;
  every path must reach the physical wall with bounded error; extrema must
  remain on that exact hashed path; and lower, nominal, and upper probability
  values must all be finite and ordered. One missing nominal value blocks the
  complete projection.
- [tool] Fixed-clock adversarial tests canonically rehashed invalid summaries,
  advanced only the map clock while held-out evidence remained fresh, and
  substituted model/code/config provenance without changing field arrays.
