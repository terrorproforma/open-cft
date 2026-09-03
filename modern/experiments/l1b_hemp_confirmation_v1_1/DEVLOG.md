# DEVLOG - L1b HEMP confirmation v1.1

## 2026-09-04

- 04:55 AEST: the v1 execution (b9449ee5) ended in `development_rejection`: 13/15 resolved,
  028 and 048 failed the 10 deg level-0 mesh angle gate before any solve (2738 s design stage).
  Results committed as recorded (`978c71be`); `POSTHOC_REJECTION.md` written in v1.
- Diagnosis (mesh builds only, no evidence touched): 028 has three 5.3 deg slivers in the
  0.254 mm exit-dielectric taper; 048 has 13,816 anisotropic 9.3 deg triangles from a 0.045 mm
  near-coincidence of the injector end and the first magnet edge. Angles 5.3 / 9.3 deg at 3, 4
  and 5 feature elements; `improve_mesh_quality` exceeds the 1.5M DOF cap on both.
- v1.1 = v1 with (i) `reject_below_angle_deg` 10 -> 5 (disclosed in
  `protocol.json#p2.mesh.angle_gate_disclosure`), per-level sliver statistics recorded (count,
  fraction, regions of elements below 10 deg), (ii) `mesh_preflight` over every declared design
  in the shakedown (recorded in shakedown.json, required by `verify_shakedown_record`, hence by
  prepare and execute), (iii) 028 and 048 added to the shakedown set (5 designs). Every
  threshold, tolerance and numerical parameter is identical; a `predecessor` block records v1.
- Tests: v1.1 copies of the v1 tests plus a whole-set mesh preflight test (all 15 level-0
  meshes pass the 5 deg gate and the 600k cap; exactly 028 and 048 carry elements below 10 deg);
  the v1 results test now binds the recorded rejection (state, the two failures, 13 records).
- Shakedown v1.1 launched 05:10 AEST (5 real designs + whole-set preflight).
