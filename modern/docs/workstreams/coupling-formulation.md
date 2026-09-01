# Verified Field-to-Plasma Coupling v2

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
