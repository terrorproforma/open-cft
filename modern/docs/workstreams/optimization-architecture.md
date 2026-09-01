# Optimization Workstream Architecture

## Scope

This workstream owns only `cft_revival/optimization`, its focused tests, campaign
specification, and `optimization-*` reports. It does not alter the legacy FYP or
shared modern package surfaces.

## Data model and API

`domain.py` is the serialization boundary. Every caller-owned iterable is copied
to a tuple and every nested record/value is validated before identity is exposed.
Design SHA-256 IDs include ordered names, bounds, units, and exact finite values.
An evaluation key additionally includes fidelity, seed, requested replicate count,
objective/constraint schemas, and result-defining context.
F0 is corrected analytical, F1 is fields/reduced, F2 is hybrid, and F3 is
PIC/experiment. Source records keep expected equivalent-F3 cost and uncertainty.

An `Observation` is immutable and either:

- successful, with numeric objectives and continuous constraint values; or
- failed, with `SolverFailure` and no objective or constraint values.

This prevents failed solves from contaminating a Pareto front as artificial
objective values or signed category constraints. Provenance has its own stable ID.

`pareto.py` transforms minimize objectives by sign and then uses constrained
dominance. Feasible designs dominate infeasible designs; infeasible designs are
ordered by total positive continuous residual. Nondominated fronts are stable
because candidates are sorted by immutable observation ID. Per-objective
scale-aware absolute/relative tolerances prevent roundoff-only dominance.
Constraint violations are normalized by declared physical scales before
aggregation. Promotion metadata records nominal, two-sigma robust, and
per-constraint chance feasibility. Every marginal probability must clear the
threshold; no independence assumption or product of marginals is used.

`campaign.py` provides asynchronous `ask`/`tell`. Pending and completed keys block
duplicates. Per-fidelity counts, committed expected cost, and observed charged
cost are enforced. F3 has 12 initial attempts plus four retry-only slots, a hard
16-attempt limit, and one retry per failed evaluation/promotion lineage. Failed
F3 attempts and every retry require finite positive charged cost. A retry also
requires `SolverFailure.retryable=True`, references a terminal failure, and
preserves every result-defining input. Zero-cost pre-execution rejection is a
separate `reject` event that removes pending work without becoming an attempt.
Promotion lineage hashes canonical design, F3 target, campaign/model/code/schema
and result context—not source seed or observation ID. Only one pending or
completed promotion may exist per lineage. Canonical JSON captures every
ask/tell/reject in a SHA-256 hash chain. Replay validates typed event schemas,
configuration identity, hashes, ordering, budget/cost/pending/duplicate policy,
promotion provenance, and canonical live reapplication. A complete prefix ending
with pending work is valid; malformed/truncated records fail closed. Parsing uses
an object-pairs hook that rejects duplicate keys recursively before Python can
collapse them, followed by recursive finite-number validation.

Pareto eligibility uses an explicit comparable-context hash containing campaign
spec ID, fidelity, the complete information-source record, objective/constraint
schemas, result context, and source model/code revision. Only observations with
the same hash can dominate one another. Promotion lineage then adds canonical
design and target fidelity, preserving design-level duplicate prevention within
that context while isolating unrelated models/configurations.

`sampling.py` implements a deterministic Cranley-Patterson-shifted Halton
sequence. It is intentionally not described as Sobol. Boundary, center,
alternating-corner, and one-factor boundary challenges precede low-discrepancy
points. Boundary points carry `boundary-challenge` provenance; only actual
sequence points carry `shifted-halton` provenance. Grouped splitting hashes
design IDs, keeping every fidelity and seed for one design in the same partition.

`guardrails.py` is independent of any GP package. It gates normalized
nearest-training distance, combined uncertainty, and named conservation/invariant
flags. Emulator standard error and physical/model discrepancy stay as separate
fields and combine only in quadrature for a gate. Every non-F3 promoted candidate
is rejected pending highest-fidelity reevaluation.

## Algorithm policy

The machine-readable policy is `spec/optimization/campaign-v1.json`.
It allocates initial counts F0=256, F1=96, F2=32, F3=12, with four reserved F3
retry attempts and an equivalent-F3 cost ceiling of 19. An iteration requests
8–16 cheap/medium evaluations and 1–4 high-fidelity evaluations. The acquisition
portfolio is constrained qLogNEHVI, MORBO-style trust regions, and constrained
qLogNParEGO fallback. Promotion requires rank-zero status, robust feasibility,
95% probability for every individual constraint, guardrail passage, and F3
verification.

Stopping keeps two validation gates separate: (1) successful F3 count is at least
12, and (2) successful F3 observations divided by all successful completed
observations is at least 3%. Failures and pending jobs are excluded from that
fraction, but failures remain charged attempts. Diagnostics expose attempts,
successes, failures, retry capacity, retryable lineages, and exhausted validation.
The attempt numerator counts every issued F3 job, including pending jobs; success
and failure counts include terminal observations only.
Normal completion also requires no pending work, acquisition convergence,
calibration, guardrails, iteration-policy compliance, and stalled F3-verified
hypervolume. Cost ceiling or exhausted bounded validation is terminal.

## Optional BoTorch boundary

`botorch_adapter.py` has no eager optional imports. Its documented plan is:

1. independent exact `SingleTaskGP` models with per-observation `train_Yvar` for
   measured heteroskedastic noise;
2. one `MultiTaskGP` per output with source fidelity as the task feature and
   `StratifiedStandardize` over that task feature;
3. an explicit residual discrepancy component, never silently merged with
   emulator posterior variance;
4. constrained, pending-aware qLogNEHVI batches;
5. qLogNParEGO batches built with one acquisition/random scalarization per
   candidate and executed through `optimize_acqf_list`.

Physical outputs are explicitly transformed to BoTorch's all-maximize convention:
`(+thrust, +efficiency, +Isp, -anode power)`. Physical reference points receive
the same transform. Adapter constraint callbacks use BoTorch's strict `< 0`
feasibility convention; normalized physical equality is shifted by a documented
small negative epsilon.

The model output layout places objectives first and continuous constraints second.
`IdentityMCMultiOutputObjective` receives objective indices only; strict-negative
constraint callables receive only the separate constraint indices. Constraint
means divide by `violation_scale`; their observation variances divide by its
square. Objective variances are unchanged, including minimized objectives.
BoTorch 0.18.1
does not support differing known noise across `MultiTaskGP` tasks. The adapter
therefore raises `UnsupportedTaskNoiseError` for any supplied task `train_Yvar`;
supported choices are inferred task noise with stratified standardization or
per-source fixed-noise `SingleTaskGP` models.

Imports and constructor names were checked against official BoTorch 0.18.1 docs:
`botorch.acquisition.multi_objective.logei.qLogNoisyExpectedHypervolumeImprovement`
and `botorch.acquisition.multi_objective.parego.qLogNParEGO`. The official guide
states qLogNParEGO uses a new random scalarization per candidate and
`optimize_acqf_list` for greedy batches. No runtime claim is made: torch,
BoTorch, and GPyTorch were not installed.

## Anti-patterns

- Do not run direct expensive NSGA-II as the production strategy.
- Do not call shifted Halton “Sobol.”
- Do not mix objective directions without an explicit transform.
- Do not impute solver failure with penalties masquerading as measurements.
- Do not split train/validation rows independently across fidelities or seeds.
- Do not promote a surrogate Pareto point without F3/highest-source validation.
- Do not report surrogate hypervolume as verified hypervolume.
- Do not combine emulator error and model inadequacy into one unexplained sigma.
- Do not publish benchmark values without equivalent-cost, common-F3 verification.
