# Parametric axisymmetric CFT geometry formulation

## Scope and physics boundary

This workstream defines geometry, material ownership, manufacturing metadata,
solver handoffs, and deterministic visual artifacts. The phrase “TWT-inspired
PPM stack” means only an alternating annular permanent-magnet and pole-piece
focusing stack. Travelling-wave-tube RF slow-wave interaction, electron-beam
gain, RF power extraction, and termination physics are not models of cusped
field thruster plasma propulsion and are absent from the code.

Every supplied variant is classified
`hypothetical_not_optimized_not_build_qualified`. No descriptor predicts
thrust, efficiency, specific impulse, lifetime, thermal state, or achievable
magnetic gradient.

## Coordinates and regions

The model uses SI metres in the right-handed cylindrical meridional half-plane
`(r,z)`, with `r >= 0`. A region is either a rectangular annulus or a linear
taper annulus represented by inner and outer radius at both axial ends. The
solid of revolution volume is evaluated exactly as the difference of two
conical frusta.

Regions carry unique IDs, owner IDs, role, material ID, bounds, and optional
magnet polarity. Plasma/channel regions touch the regular symmetry axis.
Permanent magnets are annular and use axial magnetization only, so no radial
magnetization singularity is introduced at `r=0`.

Material ownership is authoritative serialized data. Region roles resolve to
closed material kinds, and every permanent-magnet region must reference a
`permanent_magnet` material. A magnetics adapter requires a caller-supplied
registry keyed by the exact serialized material IDs. It rejects missing,
renamed, permeability-inconsistent, or wrong-kind entries and never substitutes
a synthetic magnet for copper or an unknown material.

The chamber is split into injector, straight channel, and optional divergent
exit regions. The anode is an upstream annulus. Cathode and neutralizer are
deliberately external, non-axisymmetric metadata and are not revolved into the
2D model.

## PPM sequence and geometry checks

Stage centers are strictly ordered and separated by one common pitch. Adjacent
stage polarity must alternate `+z,-z,+z,...` or its sign reversal. Magnet
spans must leave positive pole-piece gaps. Constructors enforce:

- finite SI values and positive thicknesses;
- deterministic `(z,r,id)` region ordering;
- unique material, region, stage, external-component, and evidence IDs;
- valid references and explicit material ownership;
- stage centers consistent with magnet-region centers to a small ULP bound;
- unique and complete magnet/pole references;
- explicit stage `z_min_m/z_max_m` envelopes inside the chamber;
- each magnet contained by its stage envelope and each non-terminal
  `pole_after` exactly adjacent from magnet end to stage end;
- connected stage envelopes, with each next magnet beginning at the previous
  pole end;
- exact chamber-fluid and dielectric-wall coverage from `z=0` to chamber end;
- ULP-checked divergent channel/wall endpoints, thickness, and slope;
- no positive-area overlap in the meridional plane;
- strictly positive nominal clearance above thermal/minimum requirements after
  both-sided tolerance stacks;
- tolerances smaller than half the minimum manufacturable thickness;
- a complete TWT/CFT claim-boundary note.

JSON is closed and canonicalized with sorted keys, compact separators, UTF-8,
and finite values. SHA-256 covers the payload after removing only the
top-level `integrity` object. JSON artifacts use a byte-exact no-trailing-
newline policy; only ASCII SHA-256 sidecars end with one newline.

## Solver adapters

`GeometryAdapter.solver_neutral_contract()` retains every region, including
linear tapers, and exposes neutral material, interface, magnetization, and
external-component descriptors. Every region emits oriented inner, outer,
`z_min`, and `z_max` surface segments. Curved-in-section taper lines use exact
unit normals; an outer line `r(z)` uses `(1,-dr/dz)/sqrt(1+(dr/dz)^2)`.

`to_magnetics_handoff(geometry, material_registry=...)` targets accepted
`AxisymmetricMaterialProblemContract 1.0.0`. Authority is not a call-time
switch. It is bound into the geometry's machine-readable plan ID:

1. recoil-remanence constitutive regions with no equivalent-current sources; or
2. linear recoil-permeability host regions plus typed equivalent-bound-current
   sources with no recoil-remanence regions.

The accepted v1 magnetic region primitive is rectangular. Every rectangular
serialized region is passed with the exact registry material. Tapered exit
regions remain exact in the solver-neutral contract and are omitted from the
rectangular-only accepted handoff until the material-aware worker supports
polygonal regions.

Permanent-magnet regions are stricter: geometry v1.1 rejects every
non-rectangular PM before publication. The adapter independently checks this
again and proves that the authoritative PM region count equals the handed-off
PM region/source count. No PM stage may disappear through shape filtering.
For each rectangular PM, clearance is evaluated against both endpoints of
every axially overlapping straight or tapered dielectric segment.

`to_l1a_current_equivalent_preview()` smears the two cylindrical bound-current
sheets of each axially magnetized annulus into thin `AzimuthalCurrentBand`
objects inside `L1aCurrentEquivalentPreview`. The wrapper is permanently
`authoritative=false`, carries the source representation-plan ID, and is not an
`AxisymmetricMaterialProblemContract`. This prevents preview data from being
passed as a solver-authoritative material handoff.

## Geometric descriptors

The descriptor set includes exact channel and active volumes, inlet/exit area,
pitch, nominal cusp count (`stage_count - 1`), minimum radial and axial gaps,
SmCo mass estimate, envelope, manufacturing warnings, and an ordered SI
design-variable vector. The 8300 kg/m³ SmCo density is explicitly an assumed
representative value, not a selected grade.

Before publication, every derived area, volume, gap, envelope coordinate,
design variable, and mass must be finite and representable. Positive region
volume underflow, nonzero-density mass underflow, and overflow raise
`GeometryValidationError`. Published gaps are the exact binary64 subtraction;
they are never rounded into compliance.

## Artifact verification

`load_artifact_bundle()` closes and verifies the manifest, safe plain
filenames, duplicate-free canonical JSON, raw file hashes, payload hashes,
sidecars, geometry roundtrips, viewer projection, SVG regeneration,
descriptors, dimensions, substitutions, and absence of unmanifested files.
It also allowlists the generator identity/version and requires the exact
hypothetical/non-performance claim limit.

Hashes and sidecars provide integrity relative to the supplied sidecar bytes;
they are not signatures and do not establish publisher authenticity. The
semantic allowlist prevents a consistently rehashed bundle from changing its
generator declaration or evidence boundary, but trusted distribution still
requires an authenticated channel or future signature.
