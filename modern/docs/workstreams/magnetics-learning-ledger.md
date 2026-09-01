# Magnetics Learning Ledger

## Evidence and decisions

### A B-H table needs a declared mathematical meaning

A list of points is insufficient: interpolation, behaviour outside the table,
symmetry, invertibility, and the zero-field secant limit all affect a nonlinear
solve. The contract therefore requires first-quadrant total `B(H)` data,
strictly increasing axes, an origin point, odd extension, monotone PCHIP, and
an explicit extrapolation enum.

PCHIP avoids overshoot but its unmodified endpoint formula can produce a zero
tangent for valid abrupt data. Rejecting that curve confuses a formula
limitation with invalid measurements. The revised construction gives such an
endpoint a small positive secant-relative tangent, then applies a shared-knot
Hyman limiter. Positive normalized endpoint slopes whose sum is at most three
give a strictly positive Bernstein-form derivative throughout each interval.

### Normalize before evaluating constitutive polynomials

Raw dimensional cubic coefficients can overflow or cancel even when every
input, output, and physical derivative is representable. Computing PCHIP
slopes in extended-range decimal arithmetic and evaluating each interval in
its own dimensionless coordinates avoids that failure. Direct positive
integrals of `B dH` and `H dB` also avoid subtracting nearly equal `H B` and
coenergy values. Underflowed nonnegative energy is reported as zero; a
nonrepresentable overflow is a typed failure.

Global fixed-step bisection wastes precision when a table spans many decades.
Locating the `B` interval first and solving its normalized coordinate with a
safeguarded Newton/bisection bracket permits relative/ULP termination. Forward
`B` quantization can still condition the recovered `H`, so tests use both ULP
and relative error bounds.

The first normalized implementation still converted normalized bracket ends to
binary64. For a `(0,0)->(max,max)` interval, the valid target corresponding to
`B=5e-324` is about `2^-2098`; both normalized bracket values became `0.0`, and
halving the Decimal target lost the physical endpoint-near value. The fixed
algorithm directly inverts linear intervals in extended range and measures
nonlinear stopping in reconstructed physical `H`, while retaining Decimal
brackets/candidates. Fraction-based binary64 oracles now prove exact recovery
of the first subnormals.

### Differential and secant permeability serve different purposes

`B/H` describes the ray from the origin; `dB/dH` describes the local tangent.
They coincide only for a linear law. A Newton or tangent-stiffness
linearization needs the latter. The origin secant is defined by the analytic
limit `dB/dH|0`, avoiding division by zero and an arbitrary epsilon.

### Energy checks expose interpolation/integration mistakes

For the single-valued law, exact cubic integration gives coenergy and the
Legendre relation gives energy. Checking
`w + w_co = H B` catches segment-index, sign, extrapolation, and integration
errors. This does not make hysteretic material conservative; hysteresis is
excluded precisely because it requires path/state information and loss.

### Permanent-magnet source and demagnetization are separate contracts

`M=Br/mu0` maps remanence to a uniform source at a valid temperature. It does
not prove that the solved operating point remains on the recoil line.
Projecting local reverse `H` against `Hci(T)` is useful as a conservative
screen, but intrinsic coercivity alone does not model knee shape, geometry
load line, irreversible state, manufacturing variation, or local thermal
extrema. Results therefore carry explicit limitations and become
indeterminate outside the parameter validity interval.

The prior hand-off still allowed recoil remanence and equivalent bound currents
to describe the same magnet. Authority now is an enum, not prose. A recoil
region carries the permanent-magnet material and direction with no source. An
equivalent-current source instead targets a linear host whose `mu_r` exactly
matches the permanent magnet's recoil `mu_r`. Mixed authority is rejected.

Storing an arbitrary magnetization vector alongside only a material ID left a
second consistency hole. The source now owns the typed permanent-magnet model,
temperature, and direction, and derives `M=Br(T)/mu0` as a property. Handoff
validation compares the complete material parameters and uses a documented
`32*epsilon` component tolerance. Strict deserialization resolves the source
against the declared material and recomputes magnetization, so changing `M` to
`1 A/m` or temperature to `1 K` cannot be legitimized by rehashing the payload.

### Bound-current signs should be derived, not inferred from sketches

With right-handed `(e_r,e_phi,e_z)`,
`(curl M)_phi=partial_z M_r-partial_r M_z` and
`(M cross n)_phi=M_z n_r-M_r n_z`. Applying this formula independently to all
four rectangle faces yields opposite sheet signs on opposing faces. The axis
of a solid body is not a finite-area inner cylindrical surface.

Radial magnetization is also not regular at `r=0`: the cylindrical radial unit
vector has no unique direction there. Axis-touching regions therefore permit
axial but not radial magnetization. Current-sheet constructors independently
reject zero-radius cylinders, negative radial spans, non-axis-aligned normals,
and nonfinite areas.

### Vector and persistence primitives need adversarial scale checks

Computing a vector norm before normalization can overflow for two finite
maximum components and divide both by infinity, yielding `(0,0)`. Scaling by
the largest component first gives the correct unit direction. A later request
for the true nonrepresentable magnitude raises a typed error.

Deterministic JSON alone does not detect corruption. Handoff persistence now
canonicalizes signed zero, hashes canonical UTF-8 bytes with SHA-256, rejects
duplicate keys and all schema extensions, verifies the digest, reconstructs
typed models, and compares recomputed derived content. A second Python process
is used to verify digest stability rather than assuming same-process equality.

### Interface current semantics prevent double counting

The Maxwell jump uses free surface current. A permanent magnet represented by
remanence or equivalent bound sheets must not also appear as free current.
The solver hand-off states this directly and keeps oriented minus/plus region
identifiers with the normal so sign conventions survive serialization.

### An open-boundary claim requires convergence evidence

A large-looking finite box is not evidence of an open boundary. Padding and a
small field at the truncation surface are useful diagnostics, but quantities
of interest must also stabilize under repeated domain expansion. The policy
records all three gates and deliberately limits the claim to finite-domain
convergence. Mesh refinement remains an independent study.

### Data provenance is part of numerical validity

Named grade data without a clear redistribution right should not be embedded.
The checked examples are authored synthetic curves with explicit warnings and
machine-readable `is_synthetic`/`measured=false` labels. A later production
dataset should preserve source, license, batch, measurement method,
temperature, orientation, and uncertainty rather than replacing those facts
with a familiar material name.

## Deferred integration questions

- Which nonlinear residual and continuation strategy will the axisymmetric
  worker adopt for high permeability contrast?
- How will curved and overlapping region geometry be represented beyond the
  rectangular foundation contract?
- Will permanent magnets be integrated by recoil constitutive law, bound
  sheets, or one verified equivalent formulation? Exactly one must be
  authoritative in a solve.
- Which infinite-element, mapped-boundary, or boundary-integral method will be
  independently verified against domain expansion?
- Which licensed measured B-H and demagnetization curves are valid for the
  actual temperatures, directions, and batches in the intended design?
