# Optimization Workstream Report

## Delivered interfaces

- `domain`: deeply immutable designs, schemas, information sources, objectives,
  continuous constraints, stochastic requests, observations, failures,
  provenance, replicate checks, and collision-resistant result identity.
- `pareto`: constrained mixed-direction dominance, filtering, ranking, and
  robust/chance-feasible promotion metadata.
- `campaign`: pending-aware asynchronous ask/tell, duplicate and budget gates,
  eligibility-checked F3 promotion, retry-only reserved capacity, one in-flight
  job per design/F3/model-context lineage, retryable-only paid retries,
  zero-cost non-attempt rejection events, complete count/fraction/exhaustion diagnostics, and
  typed hash-chained JSONL replay through live transition invariants. Duplicate
  JSON object keys and nonfinite values are rejected recursively before hashing.
- `sampling`: shifted-Halton initial designs, boundary challenges, and grouped
  train/validation splitting.
- `guardrails`: OOD distance, uncertainty, invariant, and F3 reevaluation gates.
- `botorch_adapter`: optional exact/fixed-noise and source-task GP construction,
  explicit `(+,+,+,-)` objective/reference transforms, strict-negative normalized
  constraint convention, objective-only acquisition selectors, separate
  constraint-output callables, `Yvar/s²` constraint-noise scaling with unchanged
  objective noise, task-stratified MultiTaskGP, typed rejection of
  unsupported differing task noise, qLogNEHVI, and implemented
  `optimize_acqf_list` qLogNParEGO batching.

The campaign specification defines the current 8-variable, 4-objective problem,
source budgets, iteration mix, promotion/stopping gates, and fair benchmark
protocol. Its benchmark `results` field is deliberately null.

Pareto promotion comparisons are isolated by a comparable-context identity:
campaign spec, fidelity and information-source semantics, objective/constraint
schemas, result context, and model/code revision must all match. Canonical design
deduplication remains active inside each such context.

## Integration instructions

Shared files were intentionally not edited. The integrating workstream should:

1. Add an `optimization` optional dependency group in `modern/pyproject.toml`
   with mutually compatible pinned ranges for torch, BoTorch, and GPyTorch after
   testing in an isolated environment.
2. Decide whether symbols should be re-exported by the existing
   `cft_revival/__init__.py`; direct imports from `cft_revival.optimization`
   already work.
3. Add CLI commands only after defining job-runner and artifact-store contracts.
   The CLI should serialize `EvaluationRequest` and return `Observation`; it
   must not contain optimizer logic.
4. Link this report and the machine-readable campaign spec from shared
   architecture/README documentation.
5. Connect physics evaluators by mapping each source result to explicit
   `ObjectiveValue`, `ConstraintValue`, `SolverFailure`, and `Provenance`.
6. Supply measured source costs and uncertainty calibration; the checked-in
   equivalent-cost values are initial planning units, not empirical results.
7. Implement verified hypervolume calculation with one frozen reference point
   and objective normalization policy before executing benchmarks.
8. Runtime-test the BoTorch adapter against the chosen versions, including
   tensor shapes, outcome transforms, constraint callables, pending points,
   and sequential batch optimization.
9. The scheduler must calculate and supply truthful booleans for surrogate
   calibration, promoted-candidate guardrails, acquisition convergence, and
   compliance with the configured 8–16 cheap/medium plus 1–4 high-fidelity
   iteration mix. `stopping_diagnostics` rejects missing/non-boolean evidence.
10. Persist and display F3 attempt count, success count, failure count, retry-only
    capacity, retryable lineage count, fixed success target, and successful-F3 /
    all-successful fraction separately. Never relabel a paid failure as free.

## Known limitations

- No torch-family dependency exists in this environment, so optional model and
  acquisition execution is not verified.
- MORBO, NSGA-III, MOEA-D, and a validated Sobol benchmark runner are specified,
  not implemented here.
- The dependency-free core does not fit a surrogate or calculate hypervolume.
- JSONL persistence is provided as a deterministic payload; durable locking and
  atomic object-store writes belong to the integration/job infrastructure.
- Event hashes make accidental or untrusted-log tampering evident but are not a
  digital signature; use signed/ACL-protected storage for adversarial operators.
- BoTorch execution remains unverified without optional dependencies. In
  particular, task-stratified transform and selector tensor shapes require an
  integration test once compatible packages are deliberately installed.
