# Active-Learning Workstream Devlog

## 2026-09-01

- Created an isolated, dependency-free `cft_revival.active_learning` package.
- Added structural posterior and campaign-record adapters without editing or
  importing existing optimization/surrogate implementations.
- Added explicit epistemic, aleatoric, and discrepancy variance decomposition;
  paired additive source-bias estimation; and uncertainty-preserving correction.
- Added candidate scoring across predicted improvement, conservative marginal
  feasibility, discrepancy, and uncertainty, normalized by source cost.
- Added mandatory highest-fidelity slot reservation and impossible-quota
  detection.
- Added the explicitly labelled deterministic
  `asynchronous-posterior-mean-fantasy-approximation` with pending-mean
  incumbent updates and spatial duplicate repulsion.
- Added manufacturing/operating normal, uniform, and triangular tolerances;
  deterministic seeded Monte Carlo propagation; quantiles; and both CVaR tails.
- Added nominal, sigma-robust, chance-constrained, nondominated promotion with
  mandatory highest-fidelity reevaluation metadata.
- Added predictive-coverage diagnostics and the complete optimization v1.4
  stopping-gate vocabulary plus hard-cost/validation-exhaustion overrides.
- Added a machine-readable contract and closed-form synthetic multi-fidelity
  tests. Analytical functions are truth; declared biased source functions are
  not. No legacy outputs or benchmark claims are used.
- Validation performed after final hardening:
  `python -m pytest tests/active_learning -q` passed 15 tests;
  `python -m pytest -q` passed 372 tests with one expected optional-extension
  skip; and `python -m compileall -q src/cft_revival/active_learning
  tests/active_learning` passed.
- `git diff -- FYP` and `git status --short -- FYP` were empty.
- Ruff and mypy were unavailable in the environment. They were not installed,
  in accordance with the no-install constraint.
- No dependencies were installed and no commit or push was performed.

## Follow-ups and limits

- Exact joint or conditional GP fantasization remains intentionally outside the
  dependency-light adapter.
- Physical covariance, tolerance distributions, and equivalent source costs
  require campaign-specific evidence before production use.
- Synthetic tests validate algorithms and contracts, not physical accuracy or
  comparative optimization performance.

## 2026-09-01 audit defect closure

- Moved finite, dimensional, and exact-direction validation ahead of all Pareto
  promotion logic. Empty comparison fronts remain valid only after the
  candidate domain passes; malformed vectors, directions, probabilities, and
  constraint records fail closed as `ActiveLearningError`.
- Replaced overflow-prone acquisition arithmetic with bounded objective-relative
  improvement, bounded magnitude signals, scaled weight aggregation, checked
  corrected means, and monotonic bounded cost normalization
  `utility / (utility + cost)`. Added canonical score/rank/name tie-breaking.
- Reimplemented empirical CVaR as finite-sample expected shortfall with mass
  `1/n` per observation and fractional boundary mass. Exact `Fraction`-based
  oracle tests cover the requested `[0,1,2,3]` lower-40% result `0.375`,
  upper-tail symmetry, duplicate values, full-tail means, and invalid edges.
- Split predictive discrepancy into paired-residual heterogeneity spread and
  estimated mean-bias standard error. Both remain explicit and combine with
  existing model terms in quadrature under the documented zero-covariance
  assumption.
- Centralized strict integer construction for source ranks/counts, quotas,
  slots, draws, seeds, discrepancy pair counts, calibration counts, and
  stopping counts/windows. Booleans and integer-valued floats are rejected.
- Wrapped malformed posterior adapter returns and shape/type failures in
  `ActiveLearningError`.
- Extended calibration with scalar sample count, declared `in-domain` or `ood`
  stratum, binomial standard error, confidence level, and Wilson interval.
- Renamed all four owned test modules with unique `test_active_learning_*`
  basenames; no global pytest configuration changed.
- Final focused tests: 74 passed in normal mode and 74 passed in importlib mode.
- Final optimization plus active-learning tests: 150 passed in normal mode and
  150 passed in importlib mode.
- Compatible broad suite, excluding unrelated concurrently incomplete
  coupling, surrogate, PIC, magnetics, visualization, hybrid, and plasma test
  workstreams: 380 passed, one optional pybind11 skip in each import mode.
- Unrestricted full collection was also attempted. It is currently blocked
  outside this ownership by missing `coupling.validation` and
  `surrogates.multifidelity`, a PIC/root test basename collision, inconsistent
  magnetics/plasma work, and stale axisymmetric visualization manifest hashes.
- `compileall`, JSON specification parsing, line-length/import scans,
  `git diff --check`, and FYP diff/status checks passed.
- Installed nothing and created no commit or push.

## 2026-09-01 final acquisition/distribution closure

- Replaced separately saturated SD components in acquisition with per-objective
  independent quadrature over epistemic, aleatoric, model-discrepancy, and
  mean-bias-estimation SDs.
- Added positive declared objective uncertainty scales and one stable monotonic
  transform of total SD. Scaled `hypot` and log-domain evaluation remain finite
  for extreme values and preserve ordering under declared unit rescaling.
- Applied the same declared scale to the separate discrepancy acquisition
  signal so uncertainty-only candidate scores remain invariant under equivalent
  output-unit changes.
- Strengthened tolerance distribution construction from attribute presence to
  callable `sample(rng)`. Wrapped lookup, signature, invocation, adapter-raised,
  non-scalar, invalid-type, and nonfinite return failures as
  `ActiveLearningError`.
- Added deterministic custom-adapter tests proving propagation-local seeds
  reproduce identical samples and differing seeds change them.
- Added the requested ordering regression: epistemic-only SD 1.0 ranks above
  four independent SDs 0.2, whose total SD is 0.4.
- Focused matrix passed 86 tests in normal and importlib modes; optimization
  plus active learning passed 162 tests in each mode; compatible broad suites
  passed 394 with one expected optional-extension skip in each mode.
- Final checks include compileall, JSON specification parsing, line-length and
  dependency-import scans, FYP diff/status, and branch/ownership status.
- No dependency installation, commit, or push was performed.

## 2026-09-02 sampler real-scalar closure

- Tightened v1.2 tolerance sampler returns to finite, non-boolean
  `numbers.Real` values checked before float conversion.
- Accepted standard int/float/Fraction and optional NumPy integer/floating
  scalars registered with `numbers.Real`, without importing NumPy in the
  package. Rejected bool/NumPy bool, string, bytes, bytearray, Decimal, complex,
  containers, and custom `__float__`-only objects.
- Added valid int/float and available NumPy-real tests plus adversarial scalar
  type tests. All failures remain `ActiveLearningError`; no raw `TypeError`
  escapes.
- Focused tests passed 96 in normal and importlib modes. Optimization plus
  active learning passed 172 in each mode.
- Compatible broad tests passed 361 with one expected optional-extension skip
  in each mode, excluding the previously documented incomplete workstreams and
  validation, which concurrently developed an unrelated
  `EvidenceRecord.independence_identity` fixture mismatch.
- Compileall, JSON specification parsing, line-length/dependency scans, FYP
  diff/status, and ownership checks passed.
- Installed nothing and created no commit or push.
