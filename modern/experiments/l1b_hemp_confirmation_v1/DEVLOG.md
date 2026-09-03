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
- Shakedown 1 (015 / 036 / 106, non-evidentiary): passed first run, 932 s, bundle validated,
  determinism replay bit-identical. It showed that the channel AXIS nulls move by up to 1.1 mm
  under iron (036: 4.34 / 7.46 mm -> 5.41 / 6.39 mm) while the WALL cusps move <= 0.35 mm, so the
  comparison now records the sorted channel-null shifts and the axis-null-to-cusp lean of both
  maps (a code change; shakedown re-run). Shakedown 2: passed, 902 s, all 11 integrity gates
  true, informational verdict CONFIRMED on the three designs, peak RSS 183 MB (4.1 % of the
  4.45 GB budget), projection 4926 s wall for 15 + 1 designs (budget 5400 s). Committed with the
  code and tests as `3e19575b`; `prepare` frozen and committed as `b9449ee5` ("preregister L1b
  HEMP confirmation v1"), pushed to origin/exp/l1b-hemp-confirmation-v1.
- Execution launched 04:01 AEST from the clean detached worktree `uni-project-l1b-hemp-run` at
  `b9449ee5` (one worker, CPU only; the PIC plume process had already ended by itself, the GPU
  was never used).
- 04:55 AEST: terminal state `development_rejection` - 13/15 designs resolved, designs 028 and
  048 failed the 10 deg level-0 mesh angle gate BEFORE any solve (geometric slivers of the
  body-fitted mesher, 5.3 / 9.3 deg at every feature count). Recorded as `978c71be` (results/
  only); `POSTHOC_REJECTION.md` written; the campaign continues as `l1b_hemp_confirmation_v1_1`
  (5 deg gate disclosed + whole-set mesh preflight). No verdict exists for v1.
