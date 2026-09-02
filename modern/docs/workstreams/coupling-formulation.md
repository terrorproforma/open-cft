# Verified Field-to-Plasma Coupling v4

## Source-backed HEMP/CFT contract

Version 4 models HEMP topology from the magnetic field at the dielectric
channel wall, not from the v3 closed-contour topology. A cusp candidate is a
quadratically interpolated local maximum of wall-normal `|B_r(r_wall,z)|`.
Topographic prominence uses minima reached on both sides within a
preregistered physical support window, and nearby ripple candidates are
suppressed under a physical separation distance. These definitions do not
shrink with mesh spacing. A physical cusp must also
receive the preregistered minimum number of field-line endpoints within its
axial plane tolerance on every primary, refined, and enlarged map.

Consecutive stable wall-cusp planes define a cell. The cell core must pass
pointwise, passing-fraction, and mean `|B_z|/|B|` thresholds, representing the
predominantly wall-parallel/axial field between cusps. X-points, O-points,
magnetic nulls, and closed constant-ψ islands may be retained as diagnostics,
but none is required to define a wall cusp or cell. This follows the HEMP
descriptions in:

- Kornfeld et al., *Physics and Evolution of HEMP-Thrusters*,
  IEPC-2007-108, https://electricrocket.org/IEPC/IEPC-2007-108.pdf;
- Koch et al., *The HEMPT Concept*, IEPC-2011-236,
  https://electricrocket.org/IEPC/IEPC-2011-236.pdf; and
- Dannenmayer et al., *Applied Sciences* 13 (2023) 3491,
  https://doi.org/10.3390/app13063491.

Each preregistered seed is integrated in both directions along
`d(r,z)/ds=±B/|B|` until the declared dielectric wall is intersected. Hardware
coordinates, plasma axial limits, and computational-box boundaries are not
wall intersections. RK stages are clipped/adapted before they sample beyond
the wall; the final wall event is interpolated only when its remaining
arclength estimate is within the preregistered wall tolerance. The path hash
binds map content, seed, direction, ψ label,
and all retained trajectory points. `B_low` and `B_high`, their locations, the
field-scale bound, and every orbit claim therefore refer to that one exact
connected trajectory; equal-z values from different field lines cannot be
combined.

Publication is atomic over every cell, seed, direction, energy, and pitch
sample on all three maps. A near-null segment, ψ drift, non-wall termination,
weak cusp bundle, non-axial core, unstable cusp/path/cell metric, missing or
nonconverged orbit evidence, an adapter/model/code/config/convergence identity
different from preregistration, excessive `rho_e/L_B`, excessive
magnetic-moment variation, or uncertainty removing the positive field bound suppresses the
mirror probability and solver projection. The guiding-centre basis is Cary
and Brizard, *Rev. Mod. Phys.* 81 (2009) 693,
https://doi.org/10.1103/RevModPhys.81.693.

V4.2 accepts field maps only from the authoritative field artifact
`1.2.0` byte path. `reload_field_artifact_bytes(...,
allow_legacy_v1_1=False)` must return the map and
`field_artifact_canonical_bytes` must reproduce the exact input bytes.
Canonicalization maps both IEEE-754 zero signs to `+0.0` while preserving every
finite nonzero binary64 value, including subnormals. Coupling applies the same
zero normalization to its labelled binary map hash, then binds the field
payload hash and canonicalization ID into each evidence fingerprint.

Field artifact `1.1.0` remains historical/read-only and cannot directly enter
a new held-out map set. If a current canonical v1.2 artifact was regenerated
from v1.1, exact legacy bytes and a canonical serialization-migration manifest
must uniquely match the old/new file and payload hashes. Their hashes are
retained in the map fingerprint and v4.2 record.

The assessed 56-case characterization is frozen as an immutable development
manifest of its exact case IDs, geometry-family ID, and recomputed manifest
hash. It remains `development_non_validation`: its IDs or hash cannot promote
a v4 record. Held-out evidence similarly carries exact case/family IDs and a
recomputed manifest hash. Coupling computes case and family disjointness,
requires one passing outcome for every held-out case, and verifies that the
evaluated case/family and exact primary/refined/enlarged map hashes are one of
those outcomes. Membership also requires role-ordered fingerprints over exact
artifact bytes and canonical field values plus schema/model/code/config/
backend/geometry/material/source/mesh/domain/timestamp, finite convergence
diagnostics, adapter contract, and map-validation policy. No caller-supplied
disjoint/all-passed boolean is accepted.

The preregistration hash covers both manifests, evaluated membership, all
three map hashes and complete fingerprints, geometry, ordered cells/seeds,
both directions,
energy/pitch samples, every numerical policy, required outcome IDs,
freshness/future skew, and complete orbit adapter/model/code/config/
convergence identity and versions, plus validation adapter/code/config
identity. The canonical record is an integrity digest, not projection
authority. Projection requires a privately constructed
`AcceptedCFTProjection` retaining the accepted raw map and held-out evidence;
each call rebuilds at an explicit evaluation time and rechecks freshness,
diagnostics, membership, cusp/cell/path/orbit status, wall termination,
same-line extrema, complete atomic registrations, ordered finite probability
bounds, and positive finite uncertainty coverage before any row is emitted.

The package root now names v4 as the current contract. Historical APIs remain
explicit: v2 is `build_screening_proxy`, and v3 is
`build_closed_contour_record`/`closed_contour_solver_inputs`. Their distinct
types and schema versions are rejected by `cft_solver_inputs`.

## Historical v3 accepted physical contract

The accepted path is now `verify_v3_field_artifact(...)` followed by
`build_closed_contour_record(...)`. Its field input is a radial-major axisymmetric
`psi_wb, b_r_t, b_z_t` map. A mirror sample is admissible only when every point
belongs to one connected marching-squares component of one constant-ψ level.
Equal axial coordinate is not a field-line identity: in particular, axis
`ψ=0` and wall `ψ!=0` are never compared as a physical mirror pair.

Marching squares linearly intersects cell edges, resolves four-edge saddle
cases with a cell-centre asymptotic decider, joins endpoints under an explicit
metric tolerance, and re-evaluates ψ by bilinear interpolation. Each contour
records closure, finite-box contact, endpoint connectivity gap, and maximum ψ
residual. Open/truncated, disconnected, or ψ-inconsistent contours cannot
publish solver inputs.

Magnetic zeros on `z_min`, `z_max`, or the outer radial truncation are
`BoundaryNullDiagnostic` values only. They are excluded before segmentation.
The symmetry axis itself is not treated as a finite-box boundary. Interior
axis or off-axis zeros identify separatrix/cusp geometry, but are not inserted
as `B_low`. Cell count and cusp locations must agree across caller-supplied
full-resolution, downsampled, and enlarged-domain studies. A count change or
cusp drift over the preregistered tolerance is typed as ambiguous and rejected.

Each stable cusp defines a cell bounded by midplanes to adjacent interior
cusps. The caller supplies increasing, strictly interior flux quantiles before
evaluation. The implementation traces all local connected components at every
quantile and preserves their mirror distributions; it does not collapse them
to one wall proxy.

For a connected surface, `B_low=min_s |B|` and `B_high=max_s |B|`. Field,
interpolation, and surface errors produce conservative extrema bounds, which
are transformed monotonically through
`x=B_low/B_high` and `p=x/(1+sqrt(1-x))`. Delta-method uncertainty is not used
near a null or nonlinear endpoint. If bounds include an invalid high field, or
the probability interval dominates the nominal value by the declared factor,
status is `uncertainty_dominated` and the nominal probability is omitted.

Exact/unresolved nulls are nonregular points. Mirror publication requires
electron energy and perpendicular-energy fraction, then evaluates
`rho_e=sqrt(2 m_e E_perp)/(e B_low)`, `L_B=B/|dB/ds|`, and the preregistered
gate `rho_e/L_B <= epsilon_max`. Missing inputs, a null, nonfinite arithmetic,
or a failed gate omits the nominal mirror probability. This is the standard
guiding-centre ordering `rho/L << 1`, not a claim that `0.1` is universal.
Sources: Cary and Brizard, *Rev. Mod. Phys.* 81, 693 (2009),
https://doi.org/10.1103/RevModPhys.81.693; Brizard, *Phys. Plasmas* 24,
042115 (2017), https://doi.org/10.1063/1.4981217; Brizard and Markowski,
*Phys. Plasmas* 29, 022101 (2022), https://doi.org/10.1063/5.0078786.

V3 identity hashes exact artifact bytes and canonical binary64
`r,z,ψ,Br,Bz`, then binds source, geometry, material, mesh, domain, model,
code, config, backend, and adapter identities. Build-time reverification also
rechecks diagnostics and freshness. Python object privacy remains an API
integrity measure, not hostile-process security.

## V3 audit hardening: segment certificates and atomic cells

Contour vertices are not sufficient evidence that `|B|>0`. Along each
marching-squares edge, bilinear `Br` and `Bz` become quadratic functions of the
edge parameter. The implementation reconstructs those quadratics from
endpoint/midpoint samples and recursively bounds each interval with
`|B(mid)| - h sup|dB/dt|`, including an explicit ULP margin. Refinement stops
only after a positive null-floor lower bound and declared extrema tolerance
are both certified. A zero crossing between nonzero vertices, a near-null
interval, nonconverged bound, or nonrepresentable derivative invalidates the
surface and suppresses mirror publication.

`L_B` now uses the certified lower field bound and the contour-wide upper
gradient bound. This is deliberately conservative. The nonrelativistic
electron gyroradius model rejects energies at or above the electron rest
energy rather than silently applying the wrong kinematics.

Ambiguous marching-squares cells use the bilinear asymptotic decider
`Q=q00*q11-q10*q01` after common scaling. `Q>0` pairs edges `(0,1)` and
`(2,3)`; `Q<0` pairs `(0,3)` and `(1,2)`. An exact saddle is rejected by
default. The only alternatives are explicit `pair_01_23` or `pair_03_12`
policies, which are record-hashed.

Segments are assembled as an undirected edge graph. Before physics use, every
closed contour is revalidated from retained points: exactly one traversal per
edge, degree two, one terminal closure, no repeated nonterminal vertex,
duplicate/reversed edge, retrace, branch, self-intersection, or finite-domain
contact. Thus a 13-point path that disguises an 8-unique-vertex retrace is not
a valid flux surface.

Cell acceptance is atomic over the full preregistration. Every requested
quantile gets a `FluxQuantileOutcome`, including no-contour and trace-error
cases. A cell is valid only when every quantile has at least one local
component and every component passes topology, segment-null, adiabatic, and
uncertainty gates. One failure makes the cell and complete record ambiguous;
solver projection emits no rows.

Uncertainty bounds now multiply the complete declared field/interpolation/
surface error by finite positive `coverage_factor`. Overflow-safe scaled sums
and products either produce finite bounds or `numerically_invalid`; no
overflow path publishes a nominal value. Coverage, registrations, every
quantile outcome/certificate, validation/freshness policy, `field_model_id`,
and complete artifact/map/source/geometry/material/mesh/domain/model/code/
config/backend/adapter identities for all three stability maps are included
in canonical record identity.

## Deprecated v2 screening proxy

The old same-z axis/wall algorithm remains only as
`cft_revival.coupling.screening_proxy.build_screening_proxy`. It emits a
`DeprecationWarning`, returns the legacy diagnostic type, and cannot pass the
v3 schema/type checks in `global_solver_inputs`. It is suitable only for rough
sensitivity screening.

## V2 audit history

## Claim boundary

`cft_revival.coupling` converts accepted axisymmetric magnetic-field evidence
into topology descriptors for a future plasma/global solver. It does not solve
plasma transport or calibrate wall losses.

Record construction no longer accepts a structural map plus caller-declared
strings. `build_coupling_record` requires an opaque `AcceptedFieldEvidence`
token issued by `verify_accepted_field_artifact`. The verifier accepts exact
immutable artifact bytes and an `AcceptedArtifactAdapter`; without both, it
fails closed.

`AcceptedFieldEvidence` is a non-dataclass immutable wrapper with private
factory construction, so ordinary calls and `dataclasses.replace` cannot mint
or alter it. This is API integrity, not a Python security boundary. Every
`build_coupling_record` call reopens its private snapshot and recomputes the
artifact hash, canonical map hash, source/map/artifact binding, schema/model/
adapter contract, all identities, diagnostics, timestamp/freshness, map
geometry, and derived profile role. Thus even a test using private internals to
replace and reinsert a snapshot fails unless all deterministic evidence is
mutually consistent.

The wrapper stores a separate domain-separated invariant hash over the
accepted snapshot metadata and policy. Build recomputes it before detailed
checks, so replacing a valid identity with a different valid-looking hash is
also detected; format validation alone would not catch that mutation.

The adapter protocol intentionally imports no active field implementation. A
format-specific adapter parses and schema-checks bytes, then returns
`AcceptedArtifactClaims`. Coupling independently verifies:

- an accepted explicit artifact schema and model level;
- `SHA256(artifact_bytes)` against the claimed artifact hash;
- canonical labelled binary64 map bytes against the map-content hash;
- a domain-separated SHA-256 binding of map, source, and artifact hashes;
- SI metres/tesla and cylindrical-axisymmetric coordinates;
- model, code, config, backend, and adapter identities;
- a timezone-aware timestamp under maximum-age/future-skew policy; and
- finite diagnostics with `converged is True`, absolute residual not above its
  declared tolerance, and relative residual not above its declared tolerance.

The current direct adapter contract is
`cft-axisymmetric-field-map/1.1.0 -> 1.1.0`, model level `L1a`. Version 1.0 is
rejected by default. It can enter only through an adapter declaring a 1.0 to
1.1 migration contract whose adapter ID is explicitly allowlisted in
`validated_migration_adapter_ids`. Adapter contract ID/version, input,
normalized output, and migration status are carried into every record.

The adapter remains a deliberate trust root. Python reflection, native memory
mutation, or importing private module keys can bypass object encapsulation;
this package does not claim process isolation, signatures, or hostile-code
security. Deterministic build-time recomputation protects against ordinary
construction/copy/replacement mistakes and inconsistent forged snapshots.

## Profile roles and map validation

All radial coordinates must be non-negative, even when an axis is optional.
The innermost row is `centreline` only when its radius is exactly zero.
Otherwise it is `inner_radial_profile`; the role and sampled radius survive in
descriptors, model identity, canonical record identity, and solver rows.

The requested wall radius is linearly interpolated between radial rows using a
convex interpolation that avoids forming an overflowing `right-left`.
Nonfinite, inverted, duplicate, stale, undersampled, shape-invalid, or
axis-irregular maps fail typed.

## Topology states and gates

Every profile descriptor has one of:

- `resolved`: at least one interior null/minimum passes all gates;
- `ambiguous`: candidates exist but uncertainty/confidence is insufficient;
- `no_topology`: no interior null/minimum exists; or
- `degenerate`: vector components and magnitude are constant within policy.

Local magnitude extrema are interior by default. Strict boundary extrema are
reported separately as `boundary_minimum`/`boundary_maximum` diagnostics and
cannot create a full-domain segment unless
`allow_boundary_minima_as_cusps=True`.

Plateau runs are bounded by their total value span, not chained adjacent
differences. Policies select midpoint, bounds, or typed rejection. Equal tied
candidates are preserved by default. Optional tie selection uses

`max(tie_absolute_tolerance_t, tie_relative_tolerance*local_candidate_scale)`,

never an implicit one-tesla floor.

A minimum must pass relative prominence, prominence-to-one-sigma, candidate
confidence, and final segment confidence gates. Unsupported ripple candidates
remain in the record with confidence; they never silently define cells. If any
selected segment fails its gate, the result is wholly `ambiguous`, not a
partial topology.

## Safe interpolation and roots

Signed roots test strict opposite signs without multiplying the values.
For `a=|Bz_i|`, `b=|Bz_(i+1)|`, the root fraction is evaluated as:

- `(a/b)/(1+a/b)` when `a<=b`;
- `1/(1+b/a)` otherwise.

This avoids overflow in `a+b` for values near `1e308`. Same-sign huge fields
cannot be nulls. Midpoints use half-scaled addition. Quadratic extrema normalize
both axial span and field magnitude. If finite samples imply an unrepresentable
spacing, integral, tolerance, uncertainty, or mirror ratio, a typed error is
returned rather than infinity, NaN, or `AssertionError`.

## Covariance-aware loss cone

The uncertainty model separates:

- independent/residual absolute sigma in T;
- relative independent sigma;
- shared additive common-mode sigma in T;
- residual correlation `rho` in `[-1,1]`; and
- a positive interval coverage factor.

For `x=B_low/B_high`, relative residual errors
`r_L=sigma_Li/B_low`, `r_H=sigma_Hi/B_high`, shared sigma `sigma_c`, and high
field `H`, the stable form is

`var(x) = x^2 [(r_L-r_H)^2 + 2 r_L r_H (1-rho)]`

`         + ((1-x) sigma_c/H)^2`.

This algebra is non-negative without subtracting nearly equal squared terms.
At `rho=1`, proportional low/high residual errors cancel exactly when their
computed relative errors are equal, and otherwise leave only the squared
relative-error difference (bounded by tested binary64 ULPs). The final term
correctly cancels equal additive common-mode motion when `x=1`. Direct callers
may supply covariance in T²; it is converted to a correlation and rejected
unless the covariance matrix is positive semidefinite. Supplying both
covariance and correlation requires consistency. Zero covariance recovers the
independent calculation.

Probability uses the cancellation-resistant expression

`p = 0.5*x/(1+sqrt(1-x))`.

Delta uncertainty uses `dp/dx=1/(4 sqrt(1-x))` below the singular endpoint.
Bounds combine adverse independent low/high errors with the same signed
additive common-mode shift in numerator and denominator, evaluate both common
directions, and take the enclosing interval. Residual correlation is omitted
from interval tightening, intentionally retaining a conservative bound.

## Identity

The v2 canonical record hash covers every field except itself: artifact/map/
source/binding hashes, schema/model/code/config/backend/adapter identities,
generation time and freshness policy, complete diagnostics, inner/wall roles
and radii, uncertainty/covariance outputs, candidates, confidence, and
segments. Different wall radii, inner roles, or uncertainty assumptions also
change the coupling-model hash.

Machine-readable definitions are in
`modern/spec/coupling/equation-ledger-v2.json` and
`modern/spec/coupling/coupling-record-v2.schema.json`.
