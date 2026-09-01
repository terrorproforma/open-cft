# Coupling v2 Integration and Later L1a Adapter

## Current accepted-evidence workflow

Record construction is deliberately two-stage:

```python
evidence = verify_accepted_field_artifact(
    artifact_bytes,
    accepted_l1a_adapter,
    MapValidationPolicy(maximum_age_s=3600.0),
)
record = build_coupling_record(
    evidence,
    wall_radius_m=wall_radius_m,
    uncertainty_model=UncertaintyModel(
        absolute_independent_sigma_t=absolute_sigma_t,
        relative_independent_sigma=relative_sigma,
        common_mode_sigma_t=shared_sigma_t,
        residual_correlation=rho,
    ),
)
```

Do not create `AcceptedFieldEvidence` directly and do not add a raw-map
overload. A validation/topology failure rejects that design; it never permits
filesystem-order fallback to an older artifact.

The evidence wrapper is intentionally not a dataclass and ordinary
construction/replacement raises. This is not authentication against hostile
Python running in the same process. The critical guarantee is deterministic:
every build recomputes the complete invariant from retained immutable bytes
and canonical map values, checks a separate snapshot-invariant digest, and
rechecks freshness at build time.

`global_solver_inputs(record)` emits rows only when topology is `resolved`.
Degenerate, no-topology, and ambiguous records remain serializable evidence
but project to no solver rows. Each resolved row carries the canonical record
hash, all field/artifact/source identities, profile roles/radii, backend/
adapter identity, provenance/freshness, diagnostics, probability interval,
covariance/correlation, and confidence.

## Exact L1a 1.1 adapter requirements

The supported/default artifact schema is now
`cft-axisymmetric-field-map/1.1.0`. The future exact loader should live in its
owning integration workstream. It may depend on the stable artifact parser,
but coupling must continue to depend only on `AcceptedArtifactAdapter`,
`AdapterVersionContract`, and `AcceptedArtifactClaims`.

The adapter must:

1. accept immutable bytes, not a path or already-decoded mutable dictionary;
2. parse JSON with duplicate-key and NaN/Infinity rejection;
3. enforce the exact stabilized schema recursively, including unknown-field
   rejection, radial-major layout, SI unit declarations, and explicit `L1a`;
4. extract `r_m`, `z_m`, `b_r_t`, and `b_z_t` without importing solver classes;
5. recompute the artifact SHA-256 from the exact bytes;
6. recompute the coupling canonical map hash from extracted binary64 values;
7. recompute the source/map/artifact binding hash rather than trusting an
   embedded digest;
8. require source hash, field-model ID/hash, implementation code hash, complete
   config hash, backend ID/version, and UTC generation timestamp;
9. map stabilized diagnostics into `SolverDiagnosticsEvidence`, including
   converged, absolute/relative residuals, their declared tolerances, and
   iterations;
10. reject a status-only convergence claim, missing diagnostic, nonfinite
    value, residual over tolerance, stale timestamp, or identity mismatch;
11. expose a stable adapter ID and SHA-256 code/version identity; and
12. declare an adapter contract ID/version with input schema 1.1, normalized
    schema 1.1, `L1a`, and `is_migration=False`; and
13. add cross-tests proving one-byte artifact tampering, map mutation, source
    substitution, stale reuse, and diagnostic weakening are rejected.

Schema 1.0 is rejected unless a separately validated migration adapter declares
`1.0 -> 1.1`, `is_migration=True`, and its adapter ID is explicitly listed in
the acceptance policy. Do not treat a policy schema tuple as migration
validation.

## Plasma/global solver adapter requirements

The owning solver workstream should:

- accept only coupling schema `cft-field-plasma-coupling/2.0.0`;
- verify/recompute `record_hash` before use;
- map by physical `segment_id`/`z`, never positional p1-p4 names;
- reject or explicitly route non-resolved topology;
- apply confidence and probability-interval policy before solving;
- include `record_hash`, `field_map_hash`, `artifact_hash`, and
  `coupling_model_hash` in cache/result identity; and
- state whether loss-cone probability is an input, prior, or closure.

The isotropic loss cone remains a geometric descriptor, not calibrated plasma
transport.

## No-install verification

From `modern/`:

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest -q tests/coupling
python -m pytest -q --import-mode=importlib tests/coupling
python -m compileall -q src/cft_revival/coupling tests/coupling
```

No field, plasma, GPU, native, or schema-validation package is required.
