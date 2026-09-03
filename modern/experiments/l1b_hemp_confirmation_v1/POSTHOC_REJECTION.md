# L1b HEMP confirmation v1 - recorded development rejection

- Preregistration commit `b9449ee5` ("preregister L1b HEMP confirmation v1"); one detached
  execution from `uni-project-l1b-hemp-run` (CPU only, one worker, GPU untouched); result commit
  `978c71be` (results/ only, 104 files, terminal state `development_rejection`).
- Design stage 2738 s: 13/15 designs resolved (both P2 levels converged at 2e-10, sampled,
  characterized, compared). Designs `l1a-gs-v3-028-f012c0bf33` and `l1a-gs-v3-048-aabacb3a59`
  failed at `resolve` BEFORE any solve: "level 0 mesh violates the minimum-angle rejection gate"
  (10 deg, inherited from the fem_reference qualification campaign).
- Root cause (diagnosed after the run on the same geometry, no evidence changed): geometric
  near-coincidences of the body-fitted structured mesher, not resolution.
  - 028: exit taper 0.254 mm long and 0.83 mm high -> three 5.3 deg sliver triangles in the
    exit dielectric polygon (5.3 / 8.0 / 4.9 deg at 3 / 4 / 5 feature elements).
  - 048: injector end (0.996 mm) and first magnet edge 0.045 mm apart -> the mandatory axial
    coordinate set inherits a 0.045 mm interval; 13,816 anisotropic 9.3 deg triangles (mostly
    anode / injector zone). `improve_mesh_quality` exceeds the 1.5M DOF policy cap on both.
- Why the shakedown did not catch it: it exercised three real designs (015, 036, 106); the mesh
  survey before the freeze built all 15 level-0 meshes but never evaluated the angle gate, which
  lives inside the solve path. Lesson (LEARNING_SCRATCHPAD.md): preflight EVERY design through
  every cheap fail-closed gate, not a sample.
- No assessment, gates, verdict or dashboard exist for v1. The 13 resolved records are valid
  design-level evidence but were never assessed; nothing from them enters any estimand.
- Successor: `experiments/l1b_hemp_confirmation_v1_1` (angle gate 5 deg with per-level sliver
  disclosure; whole-set mesh preflight in the shakedown, verified by prepare and execute; the
  two rejected designs added to the shakedown set; every other declaration identical).
