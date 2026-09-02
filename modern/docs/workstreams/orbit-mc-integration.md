# Orbit Monte Carlo integration protocol

## Ensemble API

The public sequence is:

1. Construct `PsiBicubicField` from canonical `r_m`, `z_m`, `psi_wb`, and the
   explicit material map. Retain canonical Br/Bz only as a checked reference.
   Record its dense/certified maximum ratio; construction fails with
   `NOT_EVALUATED` when the preregistered certificate-tightness floor is not met.
2. Construct immutable launches with `build_launch_ensemble`.
3. Freeze `frozen_batch_manifest` under `UNWEIGHTED_BINOMIAL`, including exact
   launch order and normalized equal weights for every contiguous batch.
4. Declare `OrbitConfig` with wall and domain geometry, physical time and path
   limits, global-max-B rotation bound, event tolerance, and gamma guard.
5. Run `run_ensemble`; changing execution partition must not change result identity.
6. Run primary/refined/enlarged map and N/2N/4N timestep campaigns with exactly
   the same launch identities.
7. Build the structural `result_artifact`, then call `write_artifact` with the
   external field/config/launch/policy authorities and actual field/config
   objects. Sealing occurs only after deterministic replay.
8. Use the returned opaque `VerifiedOrbitEvidence` (or obtain one through
   `load_and_verify_artifact`) to generate `coupling_v42_handoff`.

`load_artifact` intentionally returns `UnverifiedOrbitArtifact`. It is not a
mapping and cannot feed coupling/publication APIs. Structural validity is not
campaign evidence.

## Coupling v4.2 and plasma_network

The handoff schema is
`cft-revival-orbit-mc-coupling-v4.2/1.3.0`. It identifies the target
`cft-field-plasma-coupling/4.2.0` and supplies:

- direct dielectric-wall loss probability;
- binomial standard uncertainty and Wilson 95% interval;
- trial count;
- full result artifact and deterministic result hashes; and
- the status `export_only_pending_consumer_integration`.

No public coupling-v4.2/plasma-network consumer currently accepts this object.
It is export-only pending a separately reviewed adapter. Running independent
coupling schema or plasma-network tests does not exercise this export.

## Preregistered wall-loss validation

Before inspecting outcomes, freeze:

- energy, pitch, flux-surface/position and directional strata and weights;
- at least eight uniformly spaced deterministic gyrophases per stratum;
- primary, refined-resolution, and enlarged-domain canonical map hashes;
- N/2N/4N global-max-B timestep policies;
- physical wall/domain geometry and time/path limits;
- event, energy, interpolation, map, timestep, backend-parity, and incomplete
  outcome thresholds; and
- one genuinely held-out geometry family.

Before scheduling any batches, call `preflight_campaign(launches, field,
config)`. It fails closed on empty/duplicate launch authority, initial
wall/domain violations, invalid launch fields, underdeclared field maxima, and
timestep rotation violations. Passing preflight does not run particles or
authorize a campaign outcome.

For the first campaign, every declared gyrophase and both parallel directions
within the campaign estimand are equal-weight samples. Counts, proportions, and
Wilson intervals therefore use the unweighted binomial contract directly.
No importance, post-stratification, or unequal design weights are permitted.

Publication requires all manufactured tests, exact wall-event checks,
field-interpolator checks, timestep and cross-map probability convergence,
stable gyrophase reduction, zero numerical/incomplete outcomes, and complete
hash/provenance replay. The registered numerical defaults are in
`modern/spec/orbit_mc/validation-protocol-v1.json`.

The direct wall probability must change by no more than 0.01 successively over
the registered map/timestep refinements, with overlapping Wilson intervals.
This is a predeclared numerical gate, not experimental validation. The
held-out geometry family must then pass the same frozen protocol.

## Checkpoint and artifact policy

The v1.4 checkpoint contains the complete immutable launch authority and frozen
weighted batch manifest, but its results are a strict unique subset. Coverage
must equal all launches in completed batches plus an optional exact prefix of
one current batch; pending IDs are the exact complement. Arbitrary batch IDs,
out-of-order partial batches, inconsistent counters, dropped prior results, and
duplicate launches fail. `merge_checkpoint_results` verifies monotone coverage,
unchanged completed evidence, and the previous-checkpoint hash.

Validation and reload require externally trusted campaign, launch-manifest,
batch-manifest, and policy hashes plus the certificate-tightness floor and file
SHA-256. Consequently coherent internal rehashing cannot weaken the 0.001 floor
or replace campaign membership.

In v1.3, estimator policy and identity are included in batch and campaign
authority. Only `UNWEIGHTED_BINOMIAL` exists. Every launch in the declared
campaign estimand must have equal normalized weight; weighted/stratified
estimators are unsupported and fail closed because no weighted uncertainty
contract has been accepted.

In v1.4, the frozen batch-manifest SHA-256 is mandatory external authority at
every capability boundary: deterministic replay, artifact write, verified
load, checkpoint construction/finalization, and coupling handoff. Each API
compares it exactly with `identities.batch_manifest_sha256` (or checkpoint
authority) before publishing bytes or returning verified evidence. The hash
includes estimator policy and canonical batch order. Repartitioning one batch
into two therefore changes authority and fails even if every internal hash is
coherently recomputed; input ordering that canonicalizes to the same frozen
manifest retains the same hash.
Final artifact reload likewise requires an external file SHA-256; a mutable
sidecar is not treated as authority. Writes use atomic replacement.
The final artifact embeds launches and orbit records and binds field, config,
code, launches, and results identities. Runtime loading applies the closed
schema plus semantic replay: termination enums, endpoint/limit consistency,
phase/cycle bins, state vectors, energy envelopes, transit fractions, μ
diagnostics, probabilities, counts, and all hashes. JSON rejects nonfinite
numbers.

Every v1.2 result embeds a final-step event witness: geometry/config policy,
before/after position and velocity, event fraction, all candidate fractions,
elapsed/path counters, reflection bracket/root, gamma evidence, and
field/config/policy identities. Runtime validation recomputes wall/domain,
time/path, reflection, priority, endpoint, and counter semantics.
`validate_result_replay` additionally reruns each deterministic orbit against
externally bound launch/config/field objects when those artifacts are available.

The v1.5 result/checkpoint contracts add the snapped `event_position_m` and a
closed `event_resolution`. A `tolerance_close_fraction_zero` wall/domain
witness is accepted only for a positive attempted timestep, a strictly-inside
start within tolerance, outward attempted motion, the correct snapped surface,
and unchanged zero-fraction counters. Initial boundary/outside states remain
`INITIAL_STATE_INVALID`.

### v1.6 witness fields (2026-09-03)

Contract versions: `cft-revival-orbit-mc-result/1.6.0`,
`cft-revival-orbit-mc-checkpoint/1.6.0`,
`cft-revival-orbit-mc-validation-protocol/1.6.0`;
`cft-revival-orbit-mc-coupling-v4.2/1.3.0` is unchanged;
`cft_revival.orbit_mc.__version__ == "1.6.0"`.

The `eventWitness` object gains three required 3-vectors (the key set is
closed, so a 1.5 witness is rejected as "event witness is not closed"):

| key | meaning | failure witness |
| --- | --- | --- |
| `event_velocity_m_per_s` | Boris velocity at the event fraction; identical to `result.final_velocity_m_per_s` | `[0,0,0]` |
| `step_magnetic_midpoint_t` | B used by the final step's push (start B on the zero-fraction path) | `[0,0,0]` |
| `step_electric_midpoint_v_per_m` | E used by the final step's push | `[0,0,0]` |

Validator additions in `_validate_event_witness` for physical events:
`relativistic_boris_push(v_start, E_mid, B_mid, step_dt)` must reproduce
`step_end_velocity_m_per_s`; `_event_velocity(...)` (f=0 → start, f=1 → end,
else push of `f*step_dt`) must reproduce `event_velocity_m_per_s`; both to
`64·eps·|v|` absolute; `final_velocity_m_per_s` must equal
`event_velocity_m_per_s` exactly; midpoint vectors must be finite and
`|B_mid| <= result.maximum_b_t·(1+64 eps)`. Failure witnesses must carry
exact zero vectors in the three new keys. Protocol gates added:
`event_velocity_definition`, `require_witness_midpoint_fields`,
`require_boris_replayed_event_velocity`,
`maximum_event_velocity_replay_relative_difference` (64 eps).

Downstream consumers (v4 experiment) must: expect the 1.6.0 schema constants
from `SCHEMA_VERSION`/`CHECKPOINT_VERSION`; not hand-build witnesses without
the three keys; expect `maximum_relative_energy_error` to be a real
integrator diagnostic (0.0 on the pure-B campaign field, so a 1e-10 gate is
now satisfiable); and expect `final_velocity_m_per_s` to differ from a 1.5
artifact by O(θ²|v|) (up to ~1e3 m/s on primary-N) while terminations, step
counts, event fractions and event positions agree to roundoff.

No self-consistent E, collision, space-charge, sheath, plasma-response or PIC
claim may be inferred from a successful handoff.
