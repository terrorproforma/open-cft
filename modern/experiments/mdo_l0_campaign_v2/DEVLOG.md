# MDO L0 campaign v2 devlog

## 2026-09-03 16:45 AEST -- single execution recorded (`accepted_result`)

- Preregistration `99914dc2` (tests/code `19c91a90`), executed once from the clean
  detached worktree `uni-project-mdo-v2-run`; lock acquired 05:16:27Z; result commit
  `a003f766` (148 files under `results/` only, 7.4 MB); manifest
  `ca3b58ce21eedb8ef094a3d73894b508fe8c438183fa02620d05f759541f7b1f`.
- All 12 binding gates passed (integrity gates; acceptance != efficacy). 1440 evaluations,
  91 infeasible (all in the qLogNEHVI runs, which probe the beam-current boundary).
  Import scope = hash scope (28 files); NSGA-III 0 duplicates in 3/3 runs; labels rebuilt
  from arguments; largest hypervolume roundoff decrease 0.0.
- Dense reference 96 x 1024 = 98,304 evaluations in 54 s on 12 workers: robust HV
  1.9073e-3 (front 48 points on designs 46, 49, 50, 73, 94), nominal HV 5.498e-3; 77 of
  96 designs have a per-design robust HV < 1e-9 under CL-1.
- Final robust HV (fraction of the dense reference): qLogNEHVI 9.269e-4 (0.49) /
  2.159e-3 (1.13) / 2.151e-3 (1.13); NSGA-III 5.864e-4 (0.31) / 6.435e-4 (0.34) /
  4.652e-4 (0.24); LHS 1.184e-4 (0.06) / 1.983e-4 (0.10) / 2.692e-4 (0.14). BO beats LHS
  3/3 and NSGA-III 3/3 (counts, not significance). Seed 101 converged on design 50 and
  never found 49; seeds 202/303 exceeded the dense reference by refining the operating
  point of design 49.
- Pooled robust front: 96 points on catalogue designs 49, 50, 94 (pooled P(wall) 0.375,
  0.430, 0.379 -- the three lowest-loss designs in the screening; all 5-stage, divergent
  exit, L 20-29 mm, first polarity -1/-1/+1). Nominal front 86 points on 49, 50, 74, 94;
  75 shared (Jaccard 0.70).
- CL-1 vs CL-2: the CL-2 front (50 points, 25 catalogue designs, HV 0.045, 459 of 1240
  pooled designs infeasible) shares 0 designs with the CL-1 front (Jaccard 0.0): the
  recorded front depends entirely on the declared closure.
- Width sensitivity (CL-1): w = 1/4 -> front 15 (Jaccard 0.03, designs 30, 44, 46, 49, 50,
  94; common-set front identical up to ties); w = 4 -> 91 (Jaccard 0.82, designs 49, 50,
  94); point estimate -> 94 (Jaccard 0.79, designs 49, 50, 74, 94). Unlike v1's uniform
  prior, the width rescales every design's CVaR multiplier differently, so the common-set
  fronts are NOT invariant for w = 4 / point.
- Timing: qLogNEHVI 1637 / 1784 / 1394 s per run (candidate stage 1109 / 1215 / 946 s,
  refinement 435 / 462 / 350 s, GP fits 91 / 105 / 95 s; all 128 refinements accepted in
  every seed); NSGA-III ~3 s; LHS ~2 s; assessment 4886 s; whole execution ~83 min wall
  under 100 % CPU load from concurrent agents (protocol sizing said ~50 min BO).

## 2026-09-03 (AEST) -- design, shakedowns, preregistration

- Probed the BoTorch mixed-space optimisers on CPU under heavy contention (12 foreign
  workers + PIC run): `optimize_acqf_mixed_alternating` q=8 ~105 s per batch at reduced
  settings; `optimize_acqf_discrete` over 96 x 8 candidates ~40-50 s + per-member
  refinement ~15 s. The NEHVI partitioning dominates, not the candidate count. Declared
  the exhaustive-categorical candidate stage + continuous refinement (protocol
  `optimizers.qlognehvi.why_this_optimiser`).
- pymoo 0.6.2 NSGA-III accepts `Choice`/`Real` mixed variables with
  `MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination())`; probe:
  160 evaluations, 160 unique, shared initial population honoured.
- Pure-Python regularised incomplete beta + quantile agree with scipy to 1e-13 / 1e-14;
  the 96 x 64 per-design sample takes ~2.5 s.
- Shakedown 1 (40 evaluations/run, seeds 900101/900202, dense 96 x 32 in 5 s on 12
  workers): every stage ran; `hypervolume_monotone` FAILED on a -2.1e-16 relative
  decrease (nsga3:900101, evaluation 34): a new nondominated point with negligible
  exclusive volume changed the slicing order. Gate now tolerates relative decreases
  <= 1e-12 and records the largest decrease (protocol updated before the freeze).
- 73 of 96 designs have at least one saturated cell (128/128 wall hits), so their CL-1
  survival is ~1e-4..1e-6 and they contribute ~nothing to the hypervolume; the shakedown
  fronts were carried by catalogue designs 49, 68, 73, 74, 90 (dense: 49, 73, 90).
