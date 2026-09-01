# Active-Learning Learning Ledger

## 2026-09-01 retained lessons

- [self] Independence is strongest at a narrow structural boundary. Posterior and
  campaign adapters expose moments and accounting without importing either
  implementation.
- [self] Epistemic uncertainty, aleatoric uncertainty, and source discrepancy are not
  interchangeable. Combining them requires labels and a stated covariance
  assumption; absent covariance is not evidence of zero physical correlation.
- [self] Cross-fidelity correction requires paired designs. Unpaired source averages
  confound model discrepancy with where each source happened to sample.
- [self] Cheap information should be normalized by cost, but cost normalization alone
  can starve expensive truth. Remaining scheduler capacity must explicitly
  reserve the mandatory highest-fidelity quota.
- [self] Pending posterior means plus spatial repulsion are useful as a lightweight
  asynchronous approximation, but are not conditional GP fantasies. The method
  name must preserve that limitation in every score record.
- [self] Multiplying marginal chance-constraint probabilities adds an unsupported
  independence assumption. Promotion conservatively requires every marginal to
  clear the configured probability.
- [self] Nominal, sigma-robust, and chance feasibility answer different questions and
  should all remain visible in promotion evidence.
- [self] Monte Carlo reproducibility requires a local seeded generator. Using module
  global random state allows unrelated callers to change results.
- [self] CVaR needs an explicit upper or lower tail. A bare “CVaR” field is ambiguous
  when objectives mix maximize and minimize semantics.
- [self] Calibration must be measured against held-out known truth, not training
  residuals. Coverage, interval width, and finite-sample error are complementary.
- [self] A successful highest-fidelity quota is not the same as an attempt count.
  Active-learning selection reserves slots; campaign execution remains
  responsible for failures, retries, and charged cost.
- [self] Stopping uses F3-verified hypervolume, not surrogate hypervolume. Calibration,
  pending work, scheduler compliance, convergence, and promotion guardrails are
  independent gates under optimization schema v1.4.

## Guardrails for future work

- Keep framework tensors, GP fitting, campaign mutation, and legacy fixtures
  outside this package.
- Do not claim exact asynchronous fantasization unless a posterior adapter adds
  a separately reviewed conditional-sampling contract.
- Add covariance only through an explicit validated representation; do not
  silently alter the independent-variance formula.
- Keep synthetic closed-form truth separate from source outputs and never
  promote legacy output to an oracle.
- Do not publish benchmark conclusions from these analytical contract tests.

## Session writeback

- [user] Ownership was restricted to four new active-learning path families;
  the usual `.cursor/memory` scratchpad and devlog locations were therefore not
  created or edited.
- [self] Implemented and tested the requested algorithms instead of limiting the
  workstream to an architecture proposal.
- [tool] Windows PowerShell in this environment rejected `&&`; subsequent
  independent checks used separate tool calls or PowerShell-compatible
  separators.
- [tool] Ruff and mypy were absent. The no-install requirement took precedence;
  compileall, focused tests, full tests, import scans, line-length scans, and
  diff checks supplied the available verification.
- [self] Renamed source accounting from “completed” to “successful” during
  hardening so failed highest-fidelity attempts cannot satisfy a success quota.
- [self] Closed-form truth, constant declared source bias, and constructed
  Gaussian quantiles made source correction and coverage tests independently
  checkable without legacy fixtures.
- [self] Open risk: pending highest-fidelity jobs count as reserved potential
  quota completions; execution-layer retry capacity remains responsible for
  failed jobs.

## 2026-09-01 audit hardening

- [self] Validation must precede all promotion logic, even when the comparison
  front is empty. Vacuous `all()` and `any()` results are unsafe unless the
  candidate, directions, comparisons, constraints, and probabilities have
  already passed the domain boundary.
- [self] Finite inputs do not guarantee finite arithmetic. Corrected means,
  squared uncertainty, weighted sums, source-cost division, and hypervolume
  ratios each need scaling, bounded monotonic transforms, or checked failure.
- [self] Ranking ties must include an explicit canonical key. Relying on
  `max()` alone makes the winner depend on proposal iteration order.
- [self] Empirical expected shortfall is an integral over empirical mass, not
  the mean of every sample on one side of an interpolated quantile. A partial
  boundary observation must contribute fractional mass; equal values do not
  grant extra mass.
- [self] Paired discrepancy residual variance has two roles that cannot share
  one field: residual heterogeneity remains predictive spread, while variance
  divided by pair count is uncertainty in the fitted mean bias.
- [self] Python comparisons such as `value < 0` do not enforce integer type.
  Booleans and positive fractions pass surprisingly far unless integer
  construction is centralized.
- [self] Calibration coverage across in-domain and OOD data is not one
  exchangeable statistic. Every diagnostic now carries its stratum, sample
  count, binomial standard error, confidence level, and Wilson interval.
- [self] Generic pytest import mode keys test modules by basename. Unique
  active-learning-prefixed filenames prevent collisions without changing
  repository-wide pytest behavior.
- [tool] A repository-wide run during concurrent work can fail for missing or
  internally inconsistent files in unowned workstreams. Preserve that evidence,
  verify the owned plus direct integration suites in both import modes, and
  declare every exclusion used for the compatible broad run.
- [self] Exact rational arithmetic is an effective independent oracle for
  finite empirical-tail tests because it avoids reproducing the implementation's
  floating-point boundary calculations.

## 2026-09-01 final UQ ordering closure

- [self] Saturating uncertainty components separately before averaging changes
  the declared variance model. Independent SDs must first combine by
  quadrature; only the resulting total may be normalized.
- [self] A simple `x / (1 + x)` transform numerically saturates to exactly one
  for large finite `x`, destroying order. A scaled-hypot/log-domain evaluation
  of `log1p(x) / (1 + log1p(x))` remains bounded while retaining useful order
  across extreme finite magnitudes.
- [self] Objective uncertainty is dimensional. A declared positive output scale
  is required to make the normalized signal invariant when units are rescaled.
- [self] `hasattr(distribution, "sample")` is not an adapter contract: the
  attribute may be non-callable, have the wrong signature, raise internally,
  or return a container/nonfinite value. Validate callable construction and
  wrap the full invocation-plus-scalar conversion boundary.
- [self] Passing the propagation-local `random.Random` directly to custom
  adapters preserves deterministic seeds without allowing module-global random
  state to affect draws.

## 2026-09-02 sampler scalar closure

- [self] Successful `float(value)` conversion is not proof that an adapter
  returned a numeric scalar: text, bytes, booleans, and arbitrary `__float__`
  objects all coerce. Validate the semantic type before conversion.
- [self] The dependency-light boundary can use `numbers.Real` to admit standard
  real scalars and NumPy real scalar classes that register with the numeric
  tower, without importing NumPy. Explicitly exclude bool because it subclasses
  int; NumPy bool is naturally outside `numbers.Real`.
- [self] Decimal and complex values are deliberately outside the v1.2 sampler
  contract. Conversions, rounding policy, and complex semantics must not enter
  tolerance propagation implicitly.
