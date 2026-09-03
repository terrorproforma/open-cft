# Optimization Workstream Devlog

## 2026-09-01

- Established new, isolated optimization package and tests on
  `feat/sota-foundation`; no shared or FYP paths changed.
- Chose immutable dataclasses and canonical JSON/SHA-256 IDs for dependency-free
  persistence and reproducibility.
- Modelled failure as a separate terminal result after rejecting objective
  penalty encoding as semantically unsafe.
- Added deterministic constrained Pareto fronts with mixed objective directions.
- Added event-sourced asynchronous ask/tell with pending, duplicate, count,
  equivalent-cost, F3 quota, promotion, and stopping gates.
- Implemented shifted Halton because no validated Sobol dependency is available;
  documented the sequence accurately and included geometry challenge points.
- Added grouped splits to prevent cross-fidelity or cross-seed leakage.
- Checked current official BoTorch documentation (0.18.1) for qLogNEHVI and
  qLogNParEGO import paths and pending/constraint constructor semantics.
- Kept all torch-family imports lazy and added deterministic missing-dependency
  tests because optional packages were intentionally not installed.
- Added campaign policy and fair benchmark definitions without benchmark claims.

## 2026-09-01 independent-review hardening

- Made all boundary collections defensive tuple copies and validated nested
  records, enum values, finite scalars, costs, uncertainties, schemas, and
  replicate agreement.
- Expanded design identity with bounds/units and evaluation identity with
  replicate count, outcome schemas, and result-defining context.
- Added explicit all-maximize BoTorch output/reference transformation
  `(+thrust, +efficiency, +Isp, -power)` and strict-negative constraint helpers.
- Implemented qLogNParEGO candidate lists through `optimize_acqf_list`; removed
  reliance on an unimplemented sequential-greedy claim.
- Added objective comparison scales/tolerances and dimensionless normalized
  constraint violation. Replaced products of marginal probabilities with the
  conservative requirement that every marginal clear the configured threshold.
- Required promotion sources to be recorded, successful, same-design,
  lower-fidelity, schema-identical, and currently Pareto/feasibility eligible.
  Already-F3 observations cannot generate redundant validation.
- Reworked replay around typed exact schemas, configuration identity, event hash
  chaining, and the same live `ask`/`tell` transition checks.
- Added explicit stopping evidence for calibration, guardrails, acquisition
  convergence, verified hypervolume, and scheduler iteration-policy compliance.
- Corrected boundary challenge provenance so it is never labelled shifted Halton.
- Added nonfinite, mutation, collision, randomized identity, malformed replay,
  tampering, promotion, tolerance, and transformation tests.

## 2026-09-01 integration-blocker closure

- Split F3 successful-validation targets from attempts: 12 initial slots, four
  retry-only slots, 16 total attempts, one retry per lineage, and a cost ceiling
  that can fund configured initial budgets plus retries.
- Added explicit retry requests tied to terminal failed evaluation keys. Retries
  preserve design, fidelity, schemas, replicate count, context, and promotion
  lineage; duplicate/concurrent source jobs across seeds are rejected.
- Expanded stopping diagnostics with F3 attempts, successes, failures, retry
  capacity, retryable lineages, separate mandatory-count and successful-fraction
  gates, and terminal validation exhaustion.
- Defined the successful fraction as successful F3 observations divided by all
  successful completed observations; failed and pending jobs are excluded from
  the ratio but failures retain cost and attempt charges.
- Added `ModelOutputLayout`: objective indices feed
  `IdentityMCMultiOutputObjective`; disjoint constraint indices feed only
  strict-negative constraint callables.
- Replaced arbitrary MultiTaskGP `train_Yvar` use with per-output
  `StratifiedStandardize` and inferred noise. Known task-varying noise now raises
  `UnsupportedTaskNoiseError` with the per-source SingleTaskGP alternative.
- Made JSON parsing reject NaN/Infinity and wrapped non-string payloads, list
  observations, and nested malformed values as `CampaignError`.
- Rejected duplicate design-variable names before design identity is available.

## 2026-09-01 final acceptance closure

- Prohibited retries for `SolverFailure.retryable=False`.
- Required finite positive charged cost for every failed F3 result and every
  retry result, whether successful or failed.
- Added hash-chained pre-execution `reject` events as explicit zero-cost,
  non-attempt terminal transitions.
- Re-keyed promotion deduplication from source observation ID to canonical
  design + target fidelity + campaign/model/code/schema/result-context lineage.
  Different low-fidelity seeds now share one F3 promotion lineage.
- Preserved the original source observation ID alongside lineage identity so
  retry provenance remains auditable.
- Added model-noise transformation: objective variances remain unchanged while
  each constraint variance divides by `violation_scale²`.
- Added recursive finite-number validation immediately after JSON decoding,
  before configuration/event hashing; exponent overflow now raises
  `CampaignError`.
- Added adversarial tests for nonretryable failures, unpaid failures/retries,
  zero-cost rejection replay, cross-seed promotion deduplication, multi-scale
  constraint noise, and nested overflow values.

## 2026-09-01 context/persistence acceptance closure

- Added an explicit Pareto comparable-context identity covering campaign spec,
  source fidelity and complete information-source semantics, output schemas,
  result context, and model/code revision.
- Restricted promotion dominance/ranking to observations with identical
  comparable-context IDs. Cross-context high scores can no longer suppress an
  otherwise eligible promotion.
- Reused comparable context inside promotion lineage, then added canonical
  design and F3 target. Same-context repeated seeds still deduplicate.
- Switched strict JSONL decoding to an `object_pairs_hook` that rejects duplicate
  keys at top-level or arbitrary nesting before dictionary construction.
- Added cross-context isolation, same-context dominance, canonical replay, and
  top-level/nested duplicate-key tests.

## 2026-09-03 first in-repo BoTorch execution; MDO L0 campaign v1 preregistration

- `botorch_adapter.load_api` resolved `optimize_acqf_list` from `botorch.optim`;
  BoTorch 0.18.1 only exports it from `botorch.optim.optimize`. The adapter
  had never been executed against an installed BoTorch ("execution remains
  unverified"); the first call from the campaign runner failed with
  `OptionalDependencyError`. Fixed the lookup.
- `build_qlognehvi` gained an optional `sampler` argument: qLogNEHVI caches
  Cholesky base samples at construction, so assigning `.sampler` afterwards
  raises a shape error in `SobolQMCNormalSampler._update_base_samples`.
- Campaign `modern/experiments/mdo_l0_campaign_v1` (protocol frozen, shakedown
  passed on the `.venv-sota` runtime: torch 2.13.0+cu130, BoTorch 0.18.1,
  GPyTorch 1.15.2, pymoo 0.6.2): three operating-point design variables, seven
  uncertain inputs with the cusp probabilities as declared uncertain inputs
  (mirror formula falsified by wall-loss v4), closure CL-1, CVaR robust
  objectives, constrained qLogNEHVI vs NSGA-III vs LHS at 96 evaluations per
  run, three seeds. Geometry radii excluded (no geometry-to-L0 map).
- Measured under the concurrent PIC GPU run: GP fit + acquisition on `cuda:0`
  20-40x slower than cpu for these tiny models; the campaign declares cpu.
- `spec/optimization/campaign-v1.json#benchmark.results` stays null (no F3
  verification); the instance index `spec/optimization/mdo-l0-campaign-v1.json`
  points at the campaign and receives the recorded bundle pointer after the
  single execution.

## 2026-09-03 MDO L0 campaign v1 recorded (first optimiser run on the new physics)

- Preregistration `4898d0fd`, result `c553124b` on `exp/mdo-l0-campaign-v1`;
  terminal `accepted_result`, 8/8 binding gates, 864 evaluations, 28 min.
- Robust hypervolume at 96 evaluations: qLogNEHVI 0.003863/0.003877/0.003860,
  NSGA-III 0.002926/0.003505/0.003271, LHS 0.002844/0.003213/0.002804 (seeds
  101/202/303); BO beats random 3/3 and NSGA-III 3/3; BO seed std 9.2e-6; the
  8192-point dense reference reaches 0.003798, so BO attains 1.02x of it with
  96 evaluations. The predeclared design-set invariance to the cusp prior
  holds on the common feasible set. Robust vs nominal fronts: 114 vs 62
  designs, 24 shared.
- This is an optimiser-comparison and evaluation-chain result on the L0
  model under closure CL-1 and declared priors. It is not a thruster
  performance result and `campaign-v1.json#benchmark.results` stays null.
- Dashboard `modern/visualization/mdo-l0-campaign-v1.html` (generator
  `generate_mdo_l0_campaign_v1_dashboard.py`, 6 tests).

## 2026-09-03 MDO L0 campaign v2: the corrected L0 model over the screened design catalogue

- Design space = the 96 screened sweep-v2 designs (categorical catalogue index with
  sealed geometry and accepted-2N wall-hit counts from
  `orbit_wall_loss_geometry_screening_v1`) x v1's operating point (Ua, Ia, mdot). No
  surrogate (both geometry surrogates were rejected); each design's per-cell P(wall)
  enters CL-1 directly through its Jeffreys Beta posterior (frozen 64-row QMC sample per
  design, v1's unit rows). Classification
  `l0_model_optimisation_over_screened_design_catalogue_with_test_particle_wall_loss_closure_not_thruster_performance`;
  CL-1 identifies the collisionless test-particle wall-hit probability with the per-cusp
  survival factor, so a design wins under the declared closure only.
- Optimisers on the mixed space: BoTorch qLogNEHVI over `MixedSingleTaskGP` models
  (CategoricalKernel on the index, Matern-5/2 ARD on the operating point) with an
  exhaustive categorical candidate stage (`optimize_acqf_discrete`, 96 x 8 LHS points)
  plus per-member continuous refinement (`optimize_acqf`, fixed catalogue feature);
  pymoo NSGA-III with `Choice`/`Real` variables and
  `MixedVariableDuplicateElimination`; two-stage LHS over the catalogue. 160
  evaluations per run (32 shared initial), seeds 101/202/303, 1440 total; dense
  reference 96 x 1024 evaluated in parallel by design (12 workers, 54 s).
- The six v1 audit disclosures are closed by protocol fields and binding gates: F9
  result commit = results/ only; F10 explicit import-bound hash scope + gate
  `code_hash_scope_matches_imports` + fresh-interpreter import-trace test; F22 no rounded
  probabilities; F26 gate semantics declared (integrity, not efficacy; seed counts on
  every efficacy statement); F27 duplicate elimination + gate; F28 labels generated from
  the arguments / fitted objects + gate `labels_consistent`.
- Tests/code `19c91a90`, preregistration `99914dc2`, result `a003f766` on
  `exp/mdo-l0-campaign-v2`; terminal `accepted_result`, 12/12 binding gates, 1440
  evaluations (91 infeasible), ~83 min wall under full CPU contention.
- Robust hypervolume at 160 evaluations (v1 frame, fraction of the 98,304-point dense
  reference 1.9073e-3): qLogNEHVI 9.269e-4 / 2.159e-3 / 2.151e-3 (0.49 / 1.13 / 1.13),
  NSGA-III 5.864e-4 / 6.435e-4 / 4.652e-4, LHS 1.184e-4 / 1.983e-4 / 2.692e-4; BO beats
  LHS 3/3 and NSGA-III 3/3 (counts; three seeds carry no significance). Pooled robust
  front: 96 points on catalogue designs 49, 50, 94 (the three lowest screening P(wall),
  0.375-0.430; all 5-stage, divergent exit); nominal front on 49, 50, 74, 94 (75 shared,
  Jaccard 0.70). CL-2 (pooled survival) front shares 0 designs with the CL-1 front; the
  posterior width moves the front from 15 points (w = 1/4) to 91-94 (w = 4, point).
- This is an optimiser-comparison and evaluation-chain result under a declared closure;
  no thruster-performance claim; `campaign-v1.json#benchmark.results` stays null.
- Dashboard `modern/visualization/mdo-l0-campaign-v2.html` (generator
  `generate_mdo_l0_campaign_v2_dashboard.py`, 7 tests) with a v1-versus-v2 panel.
