# Geometry learning ledger

## Evidence classification must survive generation

Legacy code can place fixed dimensions, comments, and optimizer variables next
to one another. Treating them all as measured dimensions would manufacture
false provenance. The configuration evidence records therefore classify each
statement as traceable, assumption, or limitation.

## A hardware analogy needs a physics firewall

The useful TWT resemblance is an annular PPM/pole focusing stack. Extending the
analogy to slow-wave RF amplification or beam-wave power transfer would be
physically wrong for a CFT plasma propulsion model. The distinction is checked
as a required limitation note, not left only in narrative documentation.

## Region validity is cross-field, not schema-only

JSON Schema can close object shape and unit constants, but cannot reliably
prove tapered-region non-overlap, alternating references, tolerance stacks, or
pitch equality. Strict constructors perform those invariants and verify the
canonical payload digest.

## Nominal cusp count is not solved topology

An alternating stack has `N-1` intended inter-stage cusp locations, but
materials, boundaries, and end effects determine actual magnetic nulls. The
descriptor is named and documented as nominal geometry information; topology
must be obtained from a field solve.

## Permanent-magnet authority must be exclusive

Recoil remanence and equivalent bound current are two representations of the
same magnetization. Supplying both doubles the source. Authority must therefore
be serialized into a plan whose ID binds configuration and authority; the
adapter has no authority override. Changing authority creates a distinct plan,
while L1a previews carry the plan ID and remain structurally non-authoritative.

## Serialized ownership outranks convenience defaults

An adapter that silently replaces an unresolved permanent magnet with a
synthetic SmCo model can also turn a bad copper/material reference into a
plausible-looking solve. Requiring an exact caller-supplied registry makes
missing IDs, wrong kinds, and permeability mismatch fail before handoff.

## Clearance equality is not margin

A nominal gap equal to the thermal requirement leaves no tolerance margin.
Compliance now requires a strict positive excess after both-sided tolerance.
Descriptors publish the exact subtraction instead of snapping a nearby
binary64 value to the requirement.

## Complete surfaces require segmentation

A long shield surface can adjoin several magnets and pole pieces. One
interface per unsplit region side loses adjacency. The topology builder splits
surfaces at every matching neighbor boundary and emits reciprocal unit-normal
segments for inner, outer, upstream, and downstream surfaces. A divergent
outer slope of `1/6` yields approximately `(0.9864,-0.1644)`.

## Integrity requires loading, not only writing

Hash sidecars alone do not prevent rehashed substitutions. The strict bundle
loader reconstructs geometry, viewer data, SVG, descriptors, and dimensions,
then checks the manifest names and complete directory set. This binds bytes to
meaning rather than merely proving that bytes were hashed.

## Integrity is not publisher authenticity

A SHA-256 sidecar can be replaced alongside a modified file. It detects
accidental change and tampering only when the expected digest is trusted.
Allowlisting generator identity/version and the exact claim limit adds a
semantic boundary against consistently rehashed overclaiming, but it does not
replace signatures or an authenticated release channel.

## A stage center is not a stage envelope

Consistent pitch and magnet centers do not stop an entire coherent stack from
being translated outside the chamber. Serialized stage envelopes now bind
magnet and `pole_after` extents to chamber coordinates, enforce exact
pole/magnet/next-stage adjacency, and form one connected axial stack.

## Shape filtering must never drop authority

Filtering accepted handoff regions to rectangles is safe for nonmagnetic
geometry only when documented. Applying that filter to permanent magnets can
silently remove sources. Geometry v1.1 therefore rejects tapered PMs and the
adapter checks PM-region and source counts independently. L1b must add an exact
polygonal/frustum PM representation before this restriction can be lifted.

## Tapers expose adapter fidelity limits

The solver-neutral contract preserves a linear taper exactly. The accepted
magnetics v1 contract uses rectangular bounds; because BN and plasma are both
`mu_r=1` in this screening map, omitting those tapered regions does not change
that magnetic constitutive assignment. A future material/mesh adapter should
accept polygons before dielectric or plasma material contrast is introduced.

## Deterministic graphics are testable artifacts

Plain SVG supports exact region polygons, metadata hashes, labels, and
polarity without a CAD dependency. Stable formatting plus raw-byte sidecars
makes regeneration reviewable. A partial DXF exporter would create more risk
than value, so it was not added.
