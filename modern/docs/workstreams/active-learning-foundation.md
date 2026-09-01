# Active-Learning and UQ Foundation

## Scope and independence

`cft_revival.active_learning` is a standard-library-only layer. It does not
import optimization, campaign, surrogate, legacy, shared, or FYP code. A model
integrates by implementing `PosteriorAdapter.predict(design, source)`. An event
log or database integrates by implementing `CampaignRecordAdapter.counts()` and
`pending_designs()`. These are structural protocols, so no inheritance or
framework dependency is required.

The package does not train a GP. It consumes posterior moments from any GP,
ensemble, analytical model, or external service that can satisfy the adapter.
The objective moments supplied to acquisition must use an all-maximize
orientation. Direction conversion remains an adapter responsibility.

## Uncertainty and discrepancy

Each objective carries epistemic and aleatoric standard deviations, predictive
model-discrepancy spread, and uncertainty in the estimated mean discrepancy
bias. With no declared covariance, the package combines their independent
variances:

`variance_total = variance_epistemic + variance_aleatoric +
variance_discrepancy_spread + variance_bias_estimation`

It never invents covariance. `estimate_additive_discrepancy` uses paired designs
only and estimates `highest_output - lower_output`. Bias is retained as a
separate discrepancy correction rather than silently rewriting model output.
Paired residual sample variance is retained as irreducible/model-discrepancy
heterogeneity. That variance divided by pair count is separately retained as
uncertainty in the estimated mean bias. Existing and fitted terms combine in
quadrature under the explicit zero-covariance assumption.

## Candidate and fidelity selection

The candidate score contains all four required signals:

1. positive predicted improvement relative to the incumbent;
2. the minimum marginal Gaussian feasibility probability;
3. absolute discrepancy correction plus discrepancy uncertainty;
4. total labelled predictive uncertainty.

Predicted improvement is feasibility weighted. Extreme finite signals are
scaled into bounded dimensionless components before weighted aggregation.
For each objective, acquisition first combines epistemic, aleatoric,
model-discrepancy, and mean-bias-estimation SDs by quadrature. It then applies
one transform to the total:

`u = log1p(total_sd / declared_scale) /
     (1 + log1p(total_sd / declared_scale))`

The declared positive scale makes differing output units comparable. A scaled
`hypot` plus log-domain evaluation preserves total-uncertainty ordering without
overflow or separate component saturation. Under the independent-component
covariance policy, one SD of 1.0 therefore ranks above four SDs of 0.2, whose
total is 0.4.
Cost normalization uses `utility / (utility + cost)`, which is monotonic in
`utility / cost` for positive cost and remains finite. Corrected-mean overflow,
nonfinite components, and nonfinite final scores fail before ranking. Ties use
descending score, descending source rank, then ascending source name, making
selection independent of input order. Weights and component scores remain
visible in records, avoiding an opaque acquisition scalar.

`select_fidelity` reserves enough remaining slots to satisfy the mandatory
highest-fidelity quota. When the number of remaining slots equals the quota
deficit, selection is forced to the highest source. An already impossible
schedule is rejected instead of silently violating the quota.

## Asynchronous pending approximation

The pending method is explicitly labelled
`asynchronous-posterior-mean-fantasy-approximation`. Pending posterior means
temporarily update the incumbent and a radial penalty repels duplicate nearby
work. This is deterministic and dependency-light. It is not exact conditional
GP fantasization and must not be reported as such.

## Tolerances and risk

Manufacturing and operating tolerance variables are explicit records. Normal,
uniform, and triangular distributions sample from a local `random.Random`
instance initialized by a required non-negative integer seed. Monte Carlo
propagation returns immutable samples, means, interpolated empirical quantiles,
and lower and upper empirical CVaR. CVaR is finite-sample expected shortfall:
each observation has mass `1/n`, a partial boundary observation contributes
fractional mass, and the supplied probability is tail mass for either tail.

Custom tolerance distributions must expose callable `sample(rng)` and return
one finite, non-boolean `numbers.Real` scalar. Type is checked before float
conversion. Standard int/float/Fraction values and optional NumPy
integer/floating scalars registered with `numbers.Real` are accepted. Boolean
and NumPy boolean values, strings, bytes, bytearrays, Decimal, complex,
containers, and objects that merely implement `__float__` are rejected.
Construction rejects missing, inaccessible, or non-callable methods.
Propagation wraps bad signatures, adapter exceptions, non-scalar shapes,
invalid types, and NaN/Inf as `ActiveLearningError`; raw adapter `TypeError`
does not escape. Every call receives only the local campaign-seeded
`random.Random`, preserving deterministic replay.

Promotion requires all of:

- a nondominated objective vector under declared maximize/minimize directions;
- nominal residual feasibility (`residual <= 0`);
- sigma-robust feasibility (`mean + k * standard_deviation <= 0`);
- every marginal feasibility probability above its threshold, without a
  product-of-marginals or independence assumption.

An eligible lower-fidelity result still requires highest-fidelity reevaluation.
Before any feasibility or dominance operation, promotion validates a finite,
non-empty candidate objective vector, exact `maximize`/`minimize` directions,
and every finite comparison vector at the same dimension. Empty comparison
sets are valid, but do not bypass candidate/direction validation. Malformed
directions, probabilities, constraints, and NaN/Inf fail closed.

## Calibration and stopping

Calibration uses held-out known truth within exactly one declared `in-domain`
or `ood` stratum. It reports scalar sample count, nominal versus observed
marginal Gaussian interval coverage, interval width, absolute coverage error,
expected/max calibration error, binomial standard error, and a Wilson interval
at the declared confidence level. Unlike regimes cannot be silently aggregated.
A calibration check is evidence; it is not inferred from model fit.

Counts, quotas, source ranks, draws, seeds, discrepancy pair counts, and
stopping windows accept real Python integers only. Booleans, integer-valued
floats, negative values, and zero where positivity is required are rejected.
Malformed posterior adapter returns and shape/type failures are wrapped as
`ActiveLearningError` rather than leaking implementation exceptions.

`StoppingPolicyV14` reproduces the optimization v1.4 gate names and defaults:
12 successful F3 observations, 3% successful F3 fraction, at most 0.5%
F3-verified hypervolume improvement over five iterations, no pending jobs,
calibration checked, promotion guardrails passed, acquisition converged, and
external scheduler-policy evidence supplied. All gates must pass. The hard
equivalent-F3 cost ceiling and validation exhaustion are terminal overrides.
Only verified hypervolume history may enter that gate.

## Verification boundary

Tests use closed-form parabolas with a known Pareto interval, an analytical
feasibility residual, and declared source biases. They verify source choice,
paired bias correction, constructed predictive coverage, robust Pareto
promotion, deterministic propagation, and stopping behavior. Legacy outputs
are never used as truth. These tests establish contract behavior only; they are
not benchmark or physical-performance claims.
