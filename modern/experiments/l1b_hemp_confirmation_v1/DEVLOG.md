# DEVLOG - L1b HEMP confirmation v1

## 2026-09-04

- Read the sweep-v3 result (15/128 HEMP-like, L1b/P2 confirmation queued), the v3.1 topology
  definition, the fem_reference P2 solver / adapters / adaptivity / resource policy, and the v4
  P2 adapter (bucketed point location).
- Probe (design 000, 4 feature elements): level 0 40,659 DOFs 7.7 s, level 1 161,813 DOFs
  49.1 s (Jacobi-PCG, 2114 iterations, residual 2e-10), peak RSS 114 MB. Host: 4-13 GB free
  physical RAM while the PIC plume run (PID 53824) holds the GPU; budget rule 0.4 x free at
  start; DOF cap 600k (fem_reference policy cap 1.5M).
- Mesh survey of all 15 designs: (8,4) 40k-188k level-0 DOFs, (8,3) 24k-117k. Frozen: bore 8,
  feature 3, padding 0.5, two levels.
- Implemented: `p2_fields` (two-level solve under the RAM guard, vectorised regular-grid P2
  sampler, sampling grids), `designs` (sealed catalogue / manifest binding, sweep-v3 rebuild with
  identity proof, L1a reference extraction), `experiment` (plans, comparison, gates (a)/(b)/(c),
  reported (d), dataset, CSV, runtime callbacks), `run` (lifecycle, BLAS threads pinned), tests
  (synthetic sampler on an exact quadratic P2 field, comparison / verdict logic, protocol and
  sealed-source binding, real-input preflight: rebuild + level-0 mesh under the cap for two
  designs, lifecycle-aware results tests).
- Single-design probe (015, 3 feature elements): level 0 90,023 DOFs 19 s, level 1 358,773
  DOFs 119 s, RSS 143 MB; characterization of the three maps 60 s; 3 cusps on both maps,
  shifts <= 0.17 mm (0.37 tolerance), wall |B| ratio 1.15-1.21, rho 1.63 -> 1.78. Two defects
  found and fixed before the shakedown: the L1a reference lacked the per-cell wall maxima
  (peak wall |B| ratio was None) and the pooled axis-null match was dominated by the end nulls
  (channel / outside populations now reported separately).
- ruff: clean under the classic default rule set (E4, E7, E9, F); ruff 0.16.6's broader
  defaults leave the template's `except Exception` (recorded-never-hidden) and `noqa: E402`
  patterns, shared with the accepted campaigns.
