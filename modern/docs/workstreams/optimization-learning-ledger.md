# Optimization Workstream Learning Ledger

## Decisions retained

- A reproducible campaign needs stable identity at three levels: design,
  design/fidelity/seed evaluation, and complete observation/provenance.
- Feasibility has several non-interchangeable meanings. Nominal feasibility,
  robust sigma-margin feasibility, and per-constraint chance feasibility must
  remain visible. Multiplying marginal probabilities falsely assumes
  independence; every marginal now passes the threshold independently.
- Async BO needs pending designs in both orchestration and acquisition. Preventing
  duplicate completed jobs alone does not prevent duplicate in-flight expense.
- Cost has committed and spent views. Pending source estimates belong in
  committed cost; completed records retain actually charged cost.
- High-fidelity quota is a stopping gate, while promoted Pareto candidates create
  explicit mandatory high-fidelity jobs. Both are needed.
- Measurement noise (`train_Yvar`), emulator uncertainty, and source/model
  discrepancy answer different questions and must not share an unlabeled field.
- Row-wise random splitting is invalid for multi-fidelity replicates because one
  design can leak through a different source or seed.
- Verified hypervolume must use high-fidelity outcomes. Surrogate hypervolume can
  guide acquisition but is not benchmark evidence.

## Items requiring later evidence

- Calibrate equivalent-F3 source costs from wall-clock/resource accounting.
- Determine source-task versus autoregressive discrepancy performance by grouped
  validation and F3 predictive coverage.
- Select MORBO trust-region settings for this eight-dimensional geometry.
- Validate reference point and normalization before any cross-strategy
  hypervolume comparison.
- Runtime-check BoTorch 0.18.1-compatible versions and GPU/CPU reproducibility.
- Establish whether F3 means PIC, experiment, or a typed subtype in a given run;
  mixing them under one source without discrepancy metadata may be invalid.

## Review-derived lessons

- `frozen=True` is shallow: list-valued constructor inputs remain mutable unless
  copied, and nested schema semantics must participate in identity.
- A stochastic evaluation key must include replicate count and solver/model
  context, not only design, source, and seed.
- Multi-objective BoTorch assumes maximization. Direction handling in reporting
  is insufficient; training outcomes and reference points must be transformed.
- Physical equality (`residual == 0`) and BoTorch's strict-negative feasibility
  convention need an explicit normalized epsilon policy.
- Pareto comparison requires declared scales and tolerances. Exact float
  comparison can turn numerical dust into a new front.
- Constraint violation totals are meaningful only after each residual is made
  dimensionless using a declared scale.
- Replay is a security boundary. Parsing records directly into internal maps
  bypasses live limits; replay must reapply normal transitions and verify an
  identity/hash chain.
- Stopping evidence is not implied by cost accounting. Calibration, guardrails,
  acquisition convergence, verified hypervolume, and scheduler mix compliance
  must each be supplied and visible.

## Integration-blocker lessons

- A successful-validation quota is not an attempt budget. If both are 12, one
  paid failure makes success impossible. Retry capacity must be reserved,
  globally bounded, lineage bounded, and costed in advance.
- Retry identity is not just a new seed. It must reference a terminal failure and
  preserve the failed request plus any promotion-source lineage.
- A minimum F3 fraction and a fixed F3 count answer different questions. Their
  numerators, denominators, success semantics, and gate states must remain
  separate in diagnostics.
- A constrained GP output tensor needs a declared schema. Objectives and
  constraints may share a model but cannot share acquisition selector indices.
- BoTorch 0.18.1 warns that MultiTaskGP known observation noise does not support
  differing noise levels across tasks. Rejecting that path is safer than silently
  fitting a warning-prone likelihood; task-stratified standardization with
  inferred noise is the supported source-task path.
- Python's JSON parser accepts NaN and Infinity unless `parse_constant` rejects
  them. Persistence safety requires strict JSON before domain validation.
- Duplicate variable names make ordered schemas ambiguous even if bounds differ;
  uniqueness is a prerequisite for stable design identity.

## Final acceptance lessons

- Retryable is a semantic gate, not descriptive metadata. A terminal failure
  marked nonretryable must never consume reserved retry capacity.
- A zero-cost preflight rejection is not a solver attempt. Conflating it with a
  failed observation corrupts cost, attempt, retry, and exhaustion diagnostics.
- Promotion identity cannot use stochastic source observation ID: repeated seeds
  of one design would bypass duplicate prevention. Lineage must bind design,
  target, campaign/model identity, schemas, and result context while retaining
  the chosen source ID as provenance.
- Normalizing a noisy constraint by `s` transforms variance by `s²`. Objective
  direction sign changes do not alter objective variance.
- JSON grammar acceptance does not guarantee finite IEEE values: `1e999` can
  decode to infinity. Recursive finite validation must precede all canonical
  hashing and event construction.

## Context and duplicate-key lessons

- Pareto dominance is meaningful only among outputs generated under the same
  model, configuration, fidelity/source semantics, and output schema. Filtering
  only by fidelity and objective names still permits cross-model contamination.
- Comparable-context identity and promotion-lineage identity are related but
  distinct: context defines who may compete; lineage adds design and target to
  define who may schedule one promotion.
- Standard `json.loads` silently keeps the last duplicate object key. Duplicate
  detection must happen in `object_pairs_hook`; checking the resulting mapping is
  too late, including for nested tampering.
