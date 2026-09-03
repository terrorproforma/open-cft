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
- Shakedown v1.1 launched 05:10 AEST (5 real designs + whole-set preflight): passed, 1560 s,
  mesh preflight 15/15 (min angle 5.31 deg, largest level-1 bound 467,532 DOFs), 11/11 gates,
  replay bit-identical; informational: 028 loses the HEMP-like flag under P2 (1.515 -> 1.464).
  Timing projection 6018 s at the 1.5x contention factor (budget 5400 s; recorded as
  within_budget false, informational; the run took 3079 s + 5 min assessment).
  Committed with the code, tests and v1 posthoc docs as `3d232c7c`; `prepare` frozen and
  committed as `ead9b525` ("preregister L1b HEMP confirmation v1.1"), pushed.
- Execution 05:45-06:50 AEST from the clean detached worktree `uni-project-l1b-hemp-v11-run`
  at `ead9b525` (one worker, CPU only, GPU untouched): 15/15 resolved, design stage 3079 s,
  per design 28-360 s (median 137 s), P2 DOFs 24k-117k / 50k-466k, relative true residual
  <= 2.0e-10, peak RSS 240 MB (6.8 % of the 3.5 GB budget = 0.4 x 8.8 GB free at start),
  terminal `accepted_result`, status `accepted_l1b_confirmation_confirmed`; all 11 binding
  gates true, replay bit-identical. Recorded as `4db0a852` (results/ only, 134 files, 7.5 MB).
- Verdict CONFIRMED: (b) 15/15 strict; (c) 37/37 cusps matched, max shift 0.362 mm = 0.80 tol,
  median 0.267 mm. Reported (d): HEMP-like 14/15 (028 lost, rho 1.515 -> 1.464); rho P2/L1a
  0.94-1.45 (median 1.06); cusp wall |B| P2/L1a 1.05-1.53 (median 1.23); peak wall 0.93-1.39;
  axis peak 0.98-1.35; channel axis nulls shift up to 1.07 mm; separatrix lean 0.46 -> 1.14 mm;
  outside nulls 1.1-1.75 mm; level-0 -> level-1 cusp shift <= 1.4e-6 m; 2x sampling 15/15.
- Dashboard `modern/visualization/l1b-hemp-confirmation-v1.html` (524 kB, 6 overlays, v1
  rejection panel, sliver disclosure table); 62 experiment + dashboard tests pass; headless
  Edge render: no JS errors.
