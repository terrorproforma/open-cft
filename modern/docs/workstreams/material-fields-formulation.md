# L1b material-aware axisymmetric formulation

This workstream advances the accepted L1a vacuum solver to piecewise linear
magnetic materials. Results are hypothetical design simulations, not validated
hardware predictions.

## Equation and sign ledger

For axisymmetric `A=A_phi e_phi`, define `psi=r A_phi`. Then

- `B_r=-(1/r) partial_z psi`
- `B_z=(1/r) partial_r psi`
- `H=nu(B-Br)` in a recoil permanent magnet and `H=nu B` elsewhere.

Using `(curl H)_phi=partial_z H_r-partial_r H_z=J_free_phi` gives

`-div((nu/r) grad psi) = J_free_phi - div(nu Br_z, -nu Br_r)`.

For test functions with zero trace only at the symmetry axis (and the derived
outer Robin term retained),

`integral (nu/r) grad(psi).grad(v) =
 integral J_free_phi v + integral (nu Br_z,-nu Br_r).grad(v)`.

This fixes the PM polarity sign. A positive axial remanence contributes through
the radial component `G_r=nu Br_z`. Reversing all remanence vectors reverses the
complete field.

## Interfaces

The accepted jump conditions are `n.(B+ - B-)=0` and
`(n cross (H+ - H-))_phi=K_free_phi`. Bound PM current is not free current.
For uniform axial magnetization, interior `curl M` is zero and only the radial
faces carry `K_bound_phi=M_z n_r`; inner and outer sheets have opposite signs.

The finite-volume stencil places `nu_f/r_f` on each radial connection and
`nu_f/r_i` on each axial connection. It now integrates the PDE resistance
itself, not an area-averaged coefficient:

- radial: `nu_f=r_f Delta_r / integral(r/nu dr)`;
- axial: `nu_f=Delta_z / integral(1/nu dz)`.

Rectangular crossings provide analytic subsegments. Linear tapered/oblique
boundaries use exact line/polygon crossings, which is exact for their linear
edges and at least second-order geometrically. The independent reviewer oracle
that gave arithmetic `67.11` now gives the required series value `3.883`;
interface-position and contrast sweeps agree to binary64 roundoff.
Uniform `nu` still reduces algebraically to L1a after multiplying by `mu`.

Exactly one PM authority is legal:

1. recoil remanence in `H=nu_rec(B-Br)`, or
2. equivalent `J_b=curl M`, `K_b=M cross n` with
   `M=Br/(mu0 mu_r)` in a linear recoil host.

Adapters reject neither/both authority and therefore prevent remanence/current
double counting.

## Numerical acceptance

Jacobi-preconditioned matrix-free PCG records recurrence history but accepts a
solution only after a freshly recomputed true residual meets the configured
absolute/relative tolerance. Nonfinite coefficients, source values, iterates,
detected PCG curvature breakdown, and unconverged publication fail closed.
The Robin elimination is not claimed universally SPD: its radial logarithmic
coefficient is negative near sufficiently distant rectangular corners. Small
manufactured grids are therefore checked by an exact minimum-eigenvalue audit,
while publication additionally requires converged PCG and positive magnetic
energy. Warp CPU/CUDA uses the same float64 series-resistance operator.

The finite outer box uses the exact logarithmic derivative of the leading
axisymmetric dipole `psi=C r^2/rho^3`. On either axial side,
`alpha=3|z-zc|/rho^2`; on the outer radial side,
`alpha=3r/rho^2-2/r`, in `partial_n psi+alpha psi=0`. Both normal conditions
are checked independently at rectangular corners, including negative radial
coefficients. The condition is eliminated symmetrically into the adjacent
stencil; the axis remains `psi=0`.

Nonlinear iron is explicitly gated. The accepted tabulated B-H law is not
activated until a safeguarded nonlinear iteration has independent constitutive,
algebraic, and energy residual verification.

## Audit-closed publication policy

Material cut cells are averaged with the meridional `dr dz` measure appearing
in the weak form. Axisymmetric `2 pi r dr dz` volume remains a geometry
diagnostic and is not used as a surrogate source-conservation test. Recoil
sources are checked by exact analytical actions of `G.grad(v)` for constant,
radial, axial and mixed basis gradients. Equivalent sheets are checked against
the exact `integral K_b v ds` actions for `v={1,r,z,rz}`.

`RasterizedMaterialProblem` stores free current and PM bound current in
different arrays. Recoil authority requires remanence and a zero PM-current
array. Equivalent-current authority requires zero remanence and a nonzero PM
bound-current array. Construction rejects either double counting or a missing
authoritative PM source.

Publication state is never caller supplied. Every run embeds a deterministic
zlib/base64 binary64 raw solution plus solver diagnostics, accepted geometry
and magnetics bundles, raster evidence and coefficient/source digests, bound to
configuration, geometry payload, magnetics payload, implementation, backend,
grid, domain, problem and run SHA-256 values. Replay reconstructs coefficients
and sources from the accepted bundles, reapplies the operator, recomputes true
residual, energy, `Br/Bz`, fixed/per-stage/bore-averaged QoIs, warnings and
gates under the exact implementation revision. Each gate has an explicit
`PASS|FAIL|NOT_EVALUATED` status, measured value, unchanged threshold and
diagnostic status. Resealing altered statuses or self-hashes cannot promote
evidence. Every run must share a normalized physical
geometry identity, material registry and solver configuration identity.
Qualification studies share one backend; a separately enumerated, hash-bound
CPU/CUDA pair is the sole backend exception and supplies the parity gate.
A separate evidence-implementation digest binds adapters, solvers, replay,
acceptance and artifact validation code under the v1.4 schema revision.

Publication requires the accepted magnetics thresholds: base padding of at
least three characteristic lengths, exactly the required two successive 1.5x
expansions, boundary/peak at most `1e-3`, and successive fixed-physical-QoI
change at most `1e-3`. Domain grids preserve base `dr,dz` and extend each side
by integer cells, making every interior coordinate phase locked. Mesh and
grid-alignment gates track interpolated `Bz` at each stage centre and composite
cell-intersection Gauss integration of each piecewise-bilinear cell with
radial-area weighting. Raw cellwise and sampled-axis maxima remain explicitly
screening-only. Qualification requires twelve effective cells through every
PDE-active geometry polygon. Three preregistered grids supply observed order
and Richardson diagnostics; Richardson values never determine acceptance, and
observed order below `1.5` marks the structured-grid L1b method insufficient.
True residual, positive-energy definiteness, CPU/CUDA parity, energy,
source/current action and PM-form discrepancy-convergence gates also apply.
One failure makes evaluated high-resolution evidence
`SCREENING_NOT_ACCEPTED`. When the high-resolution prerequisite is absent,
every publication gate is `NOT_EVALUATED`; reduced-resource measurements
remain diagnostics and cannot imply threshold satisfaction. P2 comparison
handoffs carry QoI values only, never structured-grid gate statuses or fields.

The geometry integration binds the exact
`cft_revival.geometry.axisymmetric_cft/1.1.0` schema and canonical payload hash
through the current public adapter.
Rectangular handoff regions and geometry-only linear taper polygons participate
in exact line clipping. Every geometry region remains enumerated in provenance.
Regenerate all artifacts if the final v1.1 topology contract changes.
