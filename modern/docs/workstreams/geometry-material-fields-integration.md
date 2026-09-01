# Geometry contract change notice for material_fields

Status: integration action required.

The geometry audit hardening introduces the following contract changes for the
material-aware axisymmetric field worker:

1. Geometry schema is now
   `cft_revival.geometry.axisymmetric_cft/1.1.0`.
   Solver-neutral and viewer contracts are likewise 1.1.0.
2. `MaterialDefinition.category` is a closed material kind. Permanent-magnet
   regions resolve only to `permanent_magnet` definitions.
3. `to_magnetics_handoff()` no longer accepts an authority override and no
   longer creates a synthetic SmCo model. Callers must supply
   `material_registry`, keyed by exact serialized IDs. Missing IDs, key/ID
   disagreement, PM/non-PM type mismatch, and scalar permeability mismatch are
   fatal.
4. `permanent_magnet_plan` is serialized and hashed. Its plan ID binds the
   configuration ID and exactly one solver-authoritative authority:
   recoil-remanence or equivalent bound current.
5. `interface_topology()` emits deterministic, segmented inner/outer/upstream/
   downstream surfaces with region-to-region, ambient, or symmetry-axis
   adjacency. Normals are unit vectors outward from `region_id`. For an outer
   taper with `dr/dz=1/6`, the normal is approximately
   `(radial=0.9863939, axial=-0.1643990)`.
6. The accepted rectangular magnetics contract receives every rectangular
   serialized region with its supplied registry material and complete
   material-interface segments. Tapered regions remain in the solver-neutral
   contract; the worker must add polygonal-region support before claiming
   material-aware divergent-wall fidelity.
7. `to_l1a_current_equivalent_preview()` now returns
   `L1aCurrentEquivalentPreview`, not a tuple and not a material handoff. It is
   permanently `authoritative=false`.
8. JSON artifact bytes are canonical compact UTF-8 with no trailing newline.
   Sidecars alone retain one ASCII newline.
9. Stages now serialize axial envelopes. Every magnet is contained by its
   stage/chamber, and each non-terminal pole exactly joins its magnet to the
   next stage.
10. Geometry v1.1 rejects non-rectangular PMs and the adapter requires PM count
    equality. See `geometry-l1b-migration-notice.md` for the polygonal/frustum
    PM gates required before L1b can lift this restriction.

Recommended integration sequence:

1. Build the traceable magnetics registry in `material_fields`.
2. Load geometry through `deserialize_geometry()` or
   `load_artifact_bundle()`.
3. Consume `GeometryAdapter(...).solver_neutral_contract()` for exact polygon
   and interface topology.
4. Call `to_magnetics_handoff(..., material_registry=registry)` only when
   rectangular projection is acceptable.
5. Reject preview wrappers at every solver-authoritative entrypoint.
6. Add one divergent-wall regression using the normal above and one registry
   mismatch regression before accepting this contract version.
