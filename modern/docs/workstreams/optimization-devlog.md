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
