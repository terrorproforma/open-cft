# Surrogate Learning Ledger

## Decisions retained

- Numerical stability is part of the public model contract. Normalization,
  bounded hyperparameters, Cholesky, a visible jitter ladder, and explicit
  failure are preferable to an opaque inverse or unbounded optimizer.
- Duplicate coordinates are legitimate when observations have independent
  noise. Exact duplicates with zero declared noise still require numerical
  jitter; the diagnostic records that intervention.
- Heteroskedastic observation noise, latent emulator uncertainty, cross-fidelity
  discrepancy, and physical/model-form error are different quantities.
- Deterministic hyperparameter search is modest but reloadable and testable.
  It must not be marketed as globally optimal likelihood fitting.
- Multi-output independence is a baseline architecture. Correlated-output
  kernels require evidence and substantially more conditioning machinery.
- AR1 co-kriging is useful only when the fidelity relationship is reasonably
  linear after scaling. Grouped held-out comparison against independent models
  should decide whether it is retained for a data source.
- Distance-based OOD checks identify geometric novelty, not epistemic truth.
  High predictive variance and OOD distance should be surfaced together.
- Coverage on the same rows used to fit a variance multiplier is not a
  calibration result. Calibration and final assessment need separate labels.
- POD is a low-risk field interface only on a fixed mesh. Mesh transfer,
  conservation, boundary behavior, and localized-feature preservation remain
  future validation work.
- Serialization should authenticate semantics and data, not Python object
  memory. Canonical hashes plus deterministic refit make model drift visible.

## Evidence still required

- Controlled comparisons of the bounded grid policy against a trusted optimizer
  on representative source data.
- Calibration and OOD thresholds stratified by output, fidelity, and operational
  region.
- AR1 versus independent-fidelity and source-task baselines under identical
  grouped spatial splits.
- Decision-specific error limits before any surrogate is allowed to skip a
  higher-fidelity evaluation.
- POD rank selection using held-out field norms and local quantities of interest,
  not retained energy alone.
- Runtime BoTorch interoperability tests in a deliberately provisioned optional
  environment.
- Higher-fidelity or experimental evidence before making physical-prediction or
  state-of-the-art claims.

## Audit-derived lessons

- A training-data hash is not a model identity. Executable identity must include
  transforms, bounds, kernel/version, search policy, fitted parameters, jitter,
  calibration, OOD behavior, and output semantics.
- A hash detects accidental or untrusted in-transit modification only when the
  expected hash is trusted separately; it is not a signature. Deterministic
  reconstruction additionally detects unsupported or internally inconsistent
  policy payloads.
- Scalar length scales hide anisotropy even after unit-box normalization. ARD
  needs one validated scale per dimension and an explicit isotropic alternative.
- Coverage in `[0,1]` is merely a valid fraction. Acceptance requires a nominal
  level, target, tolerance, sample count, and an assessment partition that was
  not used to fit calibration.
- Caller labels are insufficient leakage controls. Exact coordinates must join
  partition groups transitively, including the binary64 equivalence of `-0.0`
  and `+0.0`.
- Generic near-coordinate tolerances are unsafe without units and
  per-dimension scales. Exact matching plus upstream unit-aware grouping is the
  current fail-closed policy.
- Zero snapshot energy has no identifiable modal direction. A mean-only rank-0
  representation is deterministic across linear-algebra backends.
- Extrapolation with finite inputs can still overflow intermediate arithmetic.
  Public APIs must either use stable limits/scaled norms or raise typed errors;
  finite input validation alone is insufficient.
- Benchmark numbers without dataset/split/config hashes are exploratory and
  must not share identity with a later authoritative reproduction.
- Reproducibility is orthogonal to quality. Exact hash reproduction can pass
  while every model-quality gate fails; one status must never imply the other.
- A broad coverage tolerance selected to admit an observed result is not an
  acceptance rule. Coverage needs a predeclared deviation tolerance and enough
  independent held-out samples; otherwise assessment remains limited.
- Range-normalized RMSE makes a single predeclared threshold portable across
  differently scaled outputs. The scale itself must be identified—in this
  benchmark, each output's range over the complete hashed dataset.
- Small-sample uncertainty cannot be repaired by relabelling assessment as
  calibration. Calibration-fit data, held-out assessment, and minimum evidence
  size remain separate contracts.

## Lessons from wall-loss geometry surrogate v1 (2026-09-03)

- Known label noise must enter the GP as fixed variance AND the accuracy
  gate must be read against the binomial floor; with 128-launch cell labels
  the floor (0.035) is 70 % of a 0.05 gate.
- A ridge/global-mean baseline chosen on the assessment role is a strong
  honest yardstick: the GP lost to it on the pooled probability with 50
  fit designs in 11 dimensions.
- Method selection on 10 designs is unstable between partitions; report the
  full candidate table, not only the winner.
- Discontinuous design -> geometry maps (selectors, minimum lengths) should
  be declared as irreducible error for a stationary kernel, or the inputs
  should be the realised geometry.
- Deciding a shakedown by its structural gates only keeps an honest negative
  science preview from reading as a broken pipeline.
