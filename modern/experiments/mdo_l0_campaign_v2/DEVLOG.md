# MDO L0 campaign v2 devlog

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
