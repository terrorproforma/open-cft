# Coupling v4 Integration and Held-Out Promotion

## Current v4 workflow

All three maps must first pass `verify_canonical_field_v12_artifact`: exact
non-empty artifact bytes, current direct schema
`cft-axisymmetric-field-map/1.2.0`, authoritative canonical-byte/reload
round trip under `field-json-sorted-utf8-signed-zero-v2`, model level `L1a`,
SI `m/Wb/T`, canonical ψ/Br/Bz hash, complete
source/geometry/material/mesh/domain/model/code/config/backend/adapter
identity, fresh timestamp, and converged residual diagnostics.

```python
primary = verify_canonical_field_v12_artifact(
    primary_bytes,
    primary_binding,
    reference_time_utc=evaluation_time,
)
# Repeat for refined and enlarged.
maps = verify_v4_map_set(primary, refined, enlarged)
development = build_cft_coupling_record(
    maps,
    geometry=geometry,
    registrations=registrations,
    validation_registration=validation_registration,
    orbit_adapter=orbit_adapter,
    cusp_policy=cusp_policy,
    trace_policy=trace_policy,
    axial_policy=axial_policy,
    stability_policy=stability_policy,
    uncertainty_model=uncertainty_model,
)
preregistration_hash = cft_preregistration_hash(
    geometry=development.geometry,
    registrations=development.registrations,
    validation_registration=development.validation_registration,
    three_map_hashes=(
        development.stability.primary.identity.full_map_hash,
        development.stability.refined.identity.full_map_hash,
        development.stability.enlarged.identity.full_map_hash,
    ),
    three_map_evidence_fingerprints=development.evidence_fingerprints,
    orbit_identity=development.orbit_identity,
    cusp_policy=development.cusp_policy,
    trace_policy=development.trace_policy,
    axial_policy=development.axial_policy,
    stability_policy=development.stability_policy,
    uncertainty_model=development.uncertainty_model,
)
held_out = verify_held_out_validation(
    exact_validation_artifact_bytes,
    held_out_adapter,
    reference_time_utc=evaluation_time,
    policy=validation_registration.policy,
)
record = build_cft_coupling_record(
    maps,
    geometry=geometry,
    registrations=registrations,
    validation_registration=validation_registration,
    orbit_adapter=orbit_adapter,
    cusp_policy=cusp_policy,
    trace_policy=trace_policy,
    axial_policy=axial_policy,
    stability_policy=stability_policy,
    uncertainty_model=uncertainty_model,
    held_out_validation_evidence=held_out,
    reference_time_utc=evaluation_time,
)
accepted_projection = accept_cft_projection(
    record,
    maps,
    held_out_validation_evidence=held_out,
    orbit_adapter=orbit_adapter,
    reference_time_utc=evaluation_time,
)
rows = cft_solver_inputs(
    accepted_projection,
    reference_time_utc=evaluation_time,
)
```

The refined map must contain at least as many samples in each coordinate,
strictly increase at least one count, and preserve the primary domain exactly.
The enlarged map must contain the complete primary domain and strictly extend
at least one bound. All maps must share artifact schema, model level, source,
geometry, material, field model, implementation, configuration, backend,
adapter contract/code, and validation policy. Mesh, domain, artifact, binding,
and full-map hashes remain map-specific and are all retained.

Schema v1.1 remains an explicit historical read-only format. It cannot enter a
new v4 map set directly. A separately generated canonical v1.2 target may
declare its v1.1 origin only by passing both exact legacy bytes and a canonical
`cft-axisymmetric-serialization-migration/1.0.0` manifest. Coupling recomputes
both file/payload identities, requires one unique source-to-target entry, and
binds migration-manifest and source-artifact hashes into the evidence
fingerprint, record, held-out outcome, and projection rows.

Freeze both development and held-out manifests, evaluated case/family, exact
three-map hashes and complete role-ordered evidence fingerprints, required
outcome count and case-to-family registrations,
cell order, seed coordinates, both
integration directions, electron energy/pitch samples, physical prominence
support/separation, wall-event tolerance, ψ drift, interpolation/path errors,
uncertainty coverage/dominance, axial-core thresholds, cross-map tolerances,
freshness/future skew, and complete orbit adapter/model/convergence
IDs/versions/code/config hashes plus validation adapter/code/config hashes
before generating validation artifacts.

## Held-out validation prerequisites

The adapter-verified validation artifact must provide:

1. criterion ID `cft-hemp-wall-cusp-v4` and version `4.0.0`;
2. the exact frozen 56-case development manifest;
3. explicit held-out case and geometry-family IDs plus recomputable manifest
   hash;
4. one explicit `(case, family, three-map hashes, three complete evidence
   fingerprints, passed)` outcome for every held-out case;
5. evaluated case/family membership, map hashes, and fingerprints matching one
   outcome;
6. the exact `cft_preregistration_hash` generated before evaluation;
7. exact validation artifact, code, and configuration SHA-256 identities;
8. finite converged diagnostics; and
9. a timezone-aware timestamp under the preregistered maximum age and future
   skew;
10. three direct canonical v1.2 fingerprints using normalized signed-zero and
    preserved finite-subnormal semantics; and
11. for any declared migration, exact v1.1 source bytes and a uniquely matching
    canonical migration manifest.

Coupling recomputes both manifest hashes and case/family set disjointness;
there are no trusted `disjoint` or `all_passed` booleans. The 56 characterized
cases are development evidence only. Relabeling the criterion status,
providing a successful subset, reusing stale evidence, changing one
preregistered choice, swapping orbit implementation/configuration, or
evaluating map hashes outside the held-out manifest rejects promotion.
Development records and caller-rehashed summary records are never projection
authority. `cft_solver_inputs` accepts only the opaque result of
`accept_cft_projection` and requires an explicit timezone-aware evaluation
clock. At every call it reopens the retained map artifacts and held-out bytes,
rechecks both freshness policies and diagnostics, reruns all cusp/cell/path/
orbit/uncertainty gates, and requires the rebuilt record to equal the accepted
record. A missing nominal probability, non-wall termination, invalid path, or
failed atomic seed/direction/sample blocks every row. After promotion it emits
one row per primary-map
seed/direction path, carrying same-line field extrema and locations, bounded
mirror probability, orbit ordering/invariant results, all three map hashes,
field provenance, and held-out manifest/artifact/code/config/preregistration
identity plus the canonical projection record hash.

## Historical v3 integration

### Accepted v3 workflow

```python
evidence = verify_v3_field_artifact(artifact_bytes, psi_capable_l1a_adapter)
stability = verify_v3_topology_stability(
    evidence,
    downsampled_evidence,
    enlarged_domain_evidence,
    maximum_cusp_shift_m=preregistered_shift_m,
)
record = build_coupling_record(
    evidence,
    stability_evidence=stability,
    cell_registrations=(
        CellRegistration("cell-1", (0.25, 0.50, 0.75)),
        CellRegistration("cell-2", (0.25, 0.50, 0.75)),
        CellRegistration("cell-3", (0.25, 0.50, 0.75)),
        CellRegistration("cell-4", (0.25, 0.50, 0.75)),
    ),
    electron_inputs=ElectronAdiabaticInputs(
        kinetic_energy_ev=electron_energy_ev,
        perpendicular_energy_fraction=perpendicular_fraction,
        maximum_gyroradius_to_scale_length=epsilon_max,
    ),
)
```

The v3 adapter must expose exact radial-major `psi_wb`, `b_r_t`, and `b_z_t`
arrays and direct current-schema `L1a` claims. Canonical identity includes all
five map arrays plus artifact, source, geometry, material, mesh, domain,
field-model, code, config, backend, and adapter hashes/identities. A migration
adapter is not accepted on the v3 physical path.

The stability study is evidence, not a request to downsample internally. The
caller supplies separately generated, content-identified full-resolution,
downsampled, and enlarged-domain cases. Each case declares mesh dimensions,
domain bounds, interior cusp coordinates, and cell count. The full case hash
must equal the accepted map. Downsample dimensions must actually be smaller;
the enlarged case must actually extend at least one domain bound. All cell
counts must equal the observed interior-cusp count and every cusp must remain
within the preregistered shift tolerance.

## Requirements for preregistered four-cell search v2

Before evaluating any design candidate, freeze:

1. the exact L1a 1.1 ψ/Br/Bz adapter and all geometry, material, source, model,
   code, config, backend, mesh, and domain identities;
2. a four-cell hypothesis requiring exactly four stable geometry-identified
   interior cusps after finite-box endpoint zeros are excluded;
3. full, independently generated downsampled, and enlarged-domain evidence,
   including a numerical cusp-shift tolerance and a no-count-change rule;
4. cell IDs/order and strictly interior flux quantiles for every cell;
5. marching-squares ψ/connectivity/closure tolerances and minimum contour
   resolution;
6. field, interpolation, and surface uncertainty bounds plus the threshold at
   which nominal probability is suppressed;
7. electron energy/perpendicular-energy assumptions and a justified
   `rho_e/L_B` acceptance threshold; and
8. acceptance logic requiring every reported mirror pair to be extrema along
   one connected constant-ψ component, with no same-z axis/wall fallback.

A candidate with three/five cells on any study, a moved cusp beyond tolerance,
an open contour, a null-reaching surface, missing electron inputs, or
uncertainty-dominated probability is retained as diagnostics but is not an
accepted four-cell plasma-coupling result.

### Four-cell-v2 audit readiness

The coupling package is ready for a preregistered search only when the search
adapter supplies three independently accepted current-schema artifacts and
uses `verify_v3_topology_stability`. The frozen registration must additionally
declare:

- every per-cell quantile as one atomic set (no successful-subset projection);
- `segment_bound_absolute_tolerance_t`,
  `segment_bound_relative_tolerance`, and `segment_max_depth`;
- the exact-saddle policy, normally `reject`;
- graph closure/connectivity tolerance and a prohibition on retraced or
  self-intersecting cycles;
- `coverage_factor` and the uncertainty-dominance threshold; and
- nonrelativistic electron-energy applicability plus the `rho_e/L_B` gate.

Readiness fails if any quantile outcome is absent, any contour certificate is
nonregular, any stability case omits its artifact/binding/implementation/
freshness identity, or any arithmetic bound is not finitely representable.
The deprecated screening proxy remains available for diagnostics, but its
legacy record is rejected by the root v3 `global_solver_inputs`.

## Deprecated v2 compatibility

Legacy calls move to
`cft_revival.coupling.screening_proxy.build_screening_proxy`. This explicit
namespace emits a deprecation warning. Its same-z outputs are not
`V3CouplingRecord` objects and `global_solver_inputs` rejects them by schema and
topology type. No accepted search should use the proxy as fallback.

## Historical v2 evidence workflow

## Frozen v2 accepted-evidence workflow

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

## Historical exact L1a 1.1 adapter requirements

The frozen v2 accepted-evidence schema remains
`cft-axisymmetric-field-map/1.1.0`. These requirements are historical and do
not authorize new v4.2 held-out promotion. V2 continues to depend only on
`AcceptedArtifactAdapter`, `AdapterVersionContract`, and
`AcceptedArtifactClaims`.

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
