# Axisymmetric Magnetostatics L1a Formulation

## Scope and claim

L1a is a real two-dimensional `(r,z)` magnetostatic field solver for explicit
axisymmetric azimuthal equivalent currents in a constant-permeability domain.
It is a structured-grid, matrix-free finite-difference method (FDM), not FEM.
It does not model permanent-magnet constitutive behaviour, nonlinear iron,
material interfaces, plasma response, or an exact open boundary.

## Coordinates, potential, and SI units

Use right-handed cylindrical coordinates `(r,phi,z)`, assume
`partial/partial(phi)=0`, and set

`A = A_phi(r,z) e_phi`, `psi = r A_phi`.

`A_phi` has units T m and `psi` has units Wb. The represented source is
`J_phi` in A/m², permeability is `mu` in H/m, and recovered fields are tesla.
For a regular Cartesian vector field near the axis,

`psi(r,z) = a(z) r² + O(r⁴)`.

Consequently `psi(0,z)=0`, `B_r(0,z)=0`, and `B_z(0,z)=2a(z)`.

## PDE and source-sign derivation

The curl of the azimuthal vector potential gives

`B_r = -(1/r) partial_z psi`,

`B_z =  (1/r) partial_r psi`.

Ampere's law for constant permeability is `curl(B)=mu J`. Its azimuthal
component is

`(curl B)_phi = partial_z B_r - partial_r B_z = mu J_phi`.

Substitution gives

`-(1/r)[partial_rr psi - (1/r)partial_r psi + partial_zz psi] = mu J_phi`,

or equivalently

`partial_rr psi - (1/r)partial_r psi + partial_zz psi = -mu r J_phi`.

The solver uses the symmetric positive form

`-div[(1/r) grad psi] = mu J_phi`.

This establishes the sign: positive `J_phi` produces a positive right-hand
side in the implemented positive operator. The manufactured solution in the
verification suite independently substitutes into both forms and checks this
sign numerically.

## Source and boundary convention

Each source is a rectangular annular/axial band strictly away from the axis.
Signed ampere-turns are smeared over its meridional cross-section:

`J_phi = polarity * ampere_turns / [(r_outer-r_inner)(z_max-z_min)]`.

This is an explicit equivalent coil/current-sheet approximation. It is not a
mapping from a magnet grade, remanence, coercivity, or magnetization vector.
Grid nodes receive dual-cell overlap averages rather than centre-point
inclusion. A band must lie fully inside the interior support
`r=[dr/2,R-dr/2]`, `z=[z_min+dz/2,z_max-dz/2]` and span at least two grid
spacings in each direction. Violations fail before sampling with an actionable
typed error; omitted boundary volume is never silently compressed into the
remaining cells. The transfer preserves `sum(J_phi dr dz)` and reports area,
centroid, and touched radial/axial dual-node counts as the represented
resolution per band.

The two-spacing test uses the summed half-ULP uncertainty envelope of the two
geometry endpoints and the represented `2*spacing` target. Thus decimal input
`[0.04,0.06]` at `spacing=0.01` passes even though direct subtraction rounds
slightly below `0.02`; moving `0.06` one representable value toward `0.04`
falls outside the envelope and fails. The identical rule applies in `r` and
`z`.

L1a imposes `psi=0` on the axis and on the finite outer box (`r=R`, `z=z_min`,
`z=z_max`). The axis condition is exact regularity. The other three conditions
are finite-domain truncations and can bias fields if sources are too close to
the box; they are not called open boundaries.

## Discretization and solve

Unknowns are node-centred interior values. Radial `1/r` coefficients are
evaluated at cell faces, producing the conservative second-order operator

`-[c_(i+1/2)(psi_(i+1)-psi_i)-c_(i-1/2)(psi_i-psi_(i-1))]/dr²`

plus the axial `-(1/r_i) partial_zz psi` term, where
`c_(i+1/2)=1/r_(i+1/2)`. This matrix is symmetric positive definite under the
homogeneous boundary conditions.

The pure-Python reference and Warp CPU/CUDA paths use binary64,
Jacobi-preconditioned conjugate gradient, a bounded iteration count, and
absolute/relative tolerances. A recurrence crossing is insufficient:
the implementation reapplies the operator and accepts `converged=True` only
if the separately recomputed algebraic residual meets
`max(atol, rtol*initial_residual)`. If recurrence and true residual disagree,
PCG restarts from the true residual within the remaining iteration budget.
Restart count is bounded and published; exhaustion is a typed
stagnation/nonconvergence state. Nonfinite input, output, residual,
non-positive PCG curvature, wrong shapes, unavailable devices, and
nonconvergence are typed failures.

Domain construction also rejects nonfinite/zero derived span, spacing,
spacing-squared, inverse spacing, and binary64 coordinate collapse before any
operator division. Fields use centred second-order derivatives away from boundaries. On-axis,
`B_z=(16 psi(dr)-psi(2dr))/(6 dr²)` applies the regular even expansion and
exactly recovers its `r²` and cancels its `r⁴` terms. The reported cylindrical
mixed-derivative metric is a **flux-reconstruction identity check** computed
from `B_r` and `B_z` reconstructed from the same `psi`. It detects inconsistent
reconstruction but is not independent validation of the PDE or Maxwell
divergence. Outer-boundary sensitivity still requires domain-size convergence.

## Artifact integrity and topology

The field-map and manifest contracts are closed at every represented object:
unknown keys, wrong scalar/list types, nonfinite or nonmonotonic coordinates,
shape errors, failed/excess-residual solves, and inconsistent serialized
`|B|=hypot(B_r,B_z)` are rejected. Each artifact and manifest contains a
canonical payload SHA-256 over compact sorted-key UTF-8 JSON excluding only
its own top-level `integrity` object. Raw file bytes are separately protected
by filename-bound `.sha256` sidecars. Manifest entries anchor both hashes for
each plain-filename artifact, avoiding recursive hashes and path substitution.
Every runtime contract failure raises `FieldArtifactValidationError`, a
documented `FieldValidationError`/`ValueError` subclass. This includes Python
`float()` overflow from huge JSON integers and JSON decoder integer-limit
failures; raw `OverflowError` is never exposed.

Topology uses `max(1e-15 T, 1e-10*|B|max, 32 ulp(|B|max))` as its candidate
tolerance. A map with `|B|max <= 1e-14 T` is explicitly
`degenerate_near_zero_field` and publishes no isolated nulls or plateaus.
Nondegenerate axis candidates are classified as sign-changing samples,
sign-changing interpolants, isolated samples, or near-zero plateaus. Finite-box
boundary minima are reported separately and never promoted to axis nulls.

## Manufactured solution

For `R=domain radius`, `L=z_max-z_min`, `k=pi/L`, and arbitrary flux scale `C`,

`psi=C[(r/R)²-(r/R)⁴] sin(k(z-z_min))`.

It is regular and zero on every boundary. The analytic source is

`J_phi=(C/mu)[8r/R⁴+k²(r/R²-r³/R⁴)] sin(k(z-z_min))`.

The exact fields are

`B_r=-C k(r/R²-r³/R⁴) cos(k(z-z_min))`,

`B_z=C(2/R²-4r²/R⁴) sin(k(z-z_min))`.

The checked 16x32, 32x64, and 64x128 interval grids show approximately
second-order convergence for both `psi` and recovered `B`; exact values are
recorded in the workstream report and tests.

## Warp FEM inspection decision

Warp 1.14's `example_diffusion.py` demonstrates scalar weak-form assembly and
`example_magnetostatics.py` demonstrates a three-dimensional H(curl)
curl-curl solve. Neither inspected example establishes the singular
axisymmetric `1/r` weight, axis trace/regularity, or this source convention.
The fastest defensible implementation is therefore the explicit auditable
FDM above with real Warp matrix-free kernels. A later FEM implementation must
be independently cross-verified and must not retroactively relabel L1a.
