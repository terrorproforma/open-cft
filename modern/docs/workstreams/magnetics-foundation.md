# Magnetic Material and Source Foundation

## Scope

This independent package defines solver-agnostic magnetic constitutive, source,
interface, validity, and publication contracts in SI units. It does not modify
or silently extend the active axisymmetric field solver.

Implemented:

- isotropic constant permeability;
- monotone, invertible, single-valued tabulated B-H response;
- differential and secant permeability;
- exact segment integration for energy/coenergy;
- SmCo-like permanent-magnet remanence, intrinsic coercivity, recoil
  permeability, and linear temperature coefficients over an explicit interval;
- axisymmetric magnetization and equivalent bound volume/surface currents;
- material-region and Maxwell interface jump contracts;
- demagnetization screening warnings;
- finite-box open-boundary acceptance criteria; and
- deterministic, versioned solver hand-off payloads.

Explicitly out of scope:

- hysteresis, minor loops, magnetic loss, and rate dependence;
- irreversible demagnetization prediction;
- eddy currents;
- a nonlinear or material-interface field solve;
- certification of any vendor material grade; and
- treating a finite homogeneous boundary as an exact open boundary.

## Constitutive conventions

All public values are SI. `H` is A/m, `B` is T, permeability is H/m,
magnetization is A/m, temperature is K, and energy density is J/m³.

For a linear isotropic material,

`B = mu0 mu_r H`.

For a tabulated material, the input is a strictly increasing first-quadrant
sequence beginning at `(0, 0)`. The runtime extends this as the odd,
single-valued relation `B(-H)=-B(H)`. It uses Fritsch-Carlson monotone
piecewise cubic Hermite interpolation with a positive Hyman limiter. Each
interval is normalized by its own `delta H` and `delta B`, then evaluated in
Bernstein/de Casteljau form. Dimensional raw polynomial coefficients are never
formed. Valid abrupt curves receive a positive shape-preserving endpoint
tangent rather than being rejected merely because the unmodified endpoint
formula would be zero.

The extrapolation policy is mandatory:

- `error` rejects `|H|` or `|B|` outside the table;
- `linear_tangent` extends with the terminal differential permeability.

No interpolation branch is a hysteresis model. The curve has exactly one `B`
for each `H`.

`mu_diff=dB/dH`. `mu_sec=B/H`, with the origin limit set to the initial
differential permeability. The package integrates each cubic exactly:

`w_co(H)=integral_0^H B(h) dh`,

and independently integrates `H dB` for magnetic energy. This direct positive
form avoids catastrophic cancellation while retaining the single-valued
Legendre relation as a verification identity:

`w(B)=H B-w_co(H)`.

This energy construction is applicable only to the conservative,
single-valued law represented here.

The inverse first locates the containing `B` interval, solves only its
normalized Hermite coordinate with safeguarded Newton/bisection, and stops on
physical-`H` binary64 adjacency. Locally linear intervals use direct extended-
range `delta B * delta H / delta B_interval` inversion, so a normalized target
smaller than the minimum binary64 subnormal is never rounded away. Nonlinear
brackets and candidates remain Decimal until the final correctly rounded
physical field conversion. It does not bisect the complete global range for a
fixed iteration count.

Energy and coenergy are accumulated in extended range. They return `0.0` only
when the true nonnegative result is below binary64's minimum subnormal; a true
overflow remains a typed failure.

## Permanent-magnet convention

The SmCo-like model is a linear recoil description:

`Br(T)=Br_ref[1+alpha_Br(T-T_ref)]`,

`Hci(T)=Hci_ref[1+alpha_Hci(T-T_ref)]`,

`B_parallel=Br(T)+mu0 mu_recoil H_parallel`.

Evaluation outside the declared temperature interval is rejected. A uniform
source uses `M=Br/mu0` in its supplied magnetization direction. This is a source
mapping, not proof that the local operating point remains reversible.

Equivalent-current sources store the referenced typed permanent-magnet model,
temperature, and unit direction; `magnetization_a_per_m` is derived and cannot
be supplied independently. Handoff validation requires exact material
parameter identity, a temperature inside the material interval, and
`M=Br(T)/mu0` component agreement within relative tolerance
`32*epsilon = 7.105427357601002e-15` (zero components must be exact).

Every hand-off selects exactly one permanent-magnet authority:

- `recoil_remanence_constitutive`: the SmCo-like recoil law and direction are
  authoritative in the material region, with no equivalent current source;
- `equivalent_bound_current`: explicit bound sheets/volume current are
  authoritative, and the host region is linear with `mu_r` exactly matching
  the permanent magnet's recoil `mu_r`.

Mixing these authorities for one permanent-magnet material is rejected.

The demagnetization assessment projects local `H` opposite the magnetization
direction and compares it with `Hci(T)`. It reports safe-screen, warning,
invalid, or temperature-indeterminate status. This remains a screening check:
the caller must supply credible local worst-case field and temperature data,
and full irreversible/load-line behaviour is not represented.

## Axisymmetric bound-current convention

Coordinates are right-handed cylindrical `(r, phi, z)` with
`partial_phi=0` and meridional `M=M_r e_r+M_z e_z`.

`J_bound_phi=(curl M)_phi=partial_z M_r-partial_r M_z`,

`K_bound_phi=(M cross n)_phi=M_z n_r-M_r n_z`.

Uniform magnetization therefore has zero interior bound volume current and
surface sheets on physical region faces. A solid region touching `r=0` emits
no degenerate inner cylindrical sheet and cannot carry radial magnetization,
which is not regular on the axis. Explicit constant-radius sheets require
strictly positive radius; constant-`z` sheets require nonnegative ordered
radial spans. Every sheet area and normalized orientation must be finite and
nondegenerate. Bound current and free current are different semantics and must
not be double-counted.

## Material and interface contracts

Each `MaterialRegionContract` binds one immutable constitutive-law identifier
to meridional bounds and an integer assignment priority. An integration worker
must reject ambiguous equal-priority overlap.

For the interface normal from the minus region to the plus region,

`n dot (B_plus-B_minus)=0`,

`[n cross (H_plus-H_minus)]_phi=K_free_phi`.

The runtime provides residual evaluation with this sign convention. Permanent
magnet bound current is not inserted into `K_free_phi`.

## Open-boundary domain policy

`OpenBoundaryDomainPolicy` does not implement an infinite element. It accepts a
finite truncation only when all required evidence passes:

1. minimum radial and axial padding measured in source characteristic lengths;
2. sufficiently small maximum boundary field relative to the interior peak;
3. at least the required number of repeated domain expansions; and
4. converged quantities of interest on the trailing expansions.

Passing these checks supports a finite-domain convergence statement only.
Mesh convergence and domain convergence are separate requirements.

## Axisymmetric solver hand-off

`AxisymmetricMaterialProblemContract` version `1.0.0` is the integration seam.
It serializes:

- constitutive-law dictionaries;
- material regions and priorities;
- oriented interfaces and free-current jumps;
- uniform magnetization sources and explicit equivalent sheets; and
- the open-boundary evidence policy.

The receiving worker must:

- assign each integration point deterministically;
- preserve free/bound current semantics;
- enforce unique material, region, interface, and source identifiers;
- enforce one compatible permanent-magnet authority and recoil permeability;
- use differential permeability in a consistent nonlinear linearization;
- independently recompute algebraic and constitutive residuals;
- reject nonfinite states; and
- retain domain-expansion inputs and quantity-of-interest changes in any
  published artifact.

This package intentionally has no import from `cft_revival.fields`; integration
is through the serialized contract rather than a hidden solver dependency.

`serialize_handoff` wraps canonical content with a SHA-256 digest.
Canonicalization sorts keys, forbids nonfinite values, and maps both signs of
zero to `0.0`. `deserialize_handoff` accepts only canonical JSON for the closed
versioned schema, rejects duplicate/extra/missing keys and unknown
discriminators, verifies the digest, reconstructs typed objects, and recomputes
all derived fields before accepting the payload. Rehashed source temperature,
direction, or magnetization changes are therefore rejected even when the
attacker recomputes the outer digest.

## Data provenance

The repository contains only two authored synthetic checked examples:

- `synthetic-soft-magnetic-example-v1`, for interpolation and energy checks;
- `synthetic-smco-like-example-v1`, for temperature, recoil, source, and
  demagnetization checks.

They are explicitly marked synthetic, unmeasured, and not representative of a
named grade. No third-party vendor curve is redistributed. Production use must
supply a traceable licensed or redistributable dataset and preserve its
temperature, direction, batch, and measurement metadata.
