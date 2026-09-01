# Standalone Surrogate Runtime

## Boundary

`cft_revival.surrogates` is a real model runtime and does not import or depend on
`cft_revival.optimization`. Its mandatory runtime is the Python standard
library. NumPy is detected and used for Cholesky and SVD acceleration when
present; tested standard-library Cholesky and POD fallbacks remain available.

The first training target is the deterministic hypothetical L0 generator. That
checks interpolation, software stability, scaling, and uncertainty plumbing. It
does not make L0 measured data, a high-fidelity label, or physical truth.

## Models

- `ExactGP` fits scalar Matérn-5/2 exact Gaussian processes after unit-box input
  scaling and output standardization. Deterministic bounded likelihood fitting
  supports true per-dimension ARD vectors by default and an explicit isotropic
  mode. Every Matérn distance divides each affine-normalized coordinate by its
  validated length scale.
- Cholesky factorization follows the declared normalized-variance jitter ladder
  `0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4`, then fails explicitly. Per-row
  heteroskedastic measurement variance is supported and is never relabelled as
  emulator uncertainty.
- `IndependentMultiOutputGP` uses one scalar GP per output. It makes no hidden
  output-independence calibration claim; the independence is the model
  structure.
- `TwoFidelityAR1` implements
  `high(x) = rho * low(x) + discrepancy(x)`. `rho` is bounded, and the
  discrepancy has its own exact GP. Its variance sum states the posterior-
  independence approximation in diagnostics.
- `Prediction` validates finite mean/non-negative variance and carries nominal
  interval probability and uncertainty semantics. `VarianceCalibrator` records
  `calibration-fit`; metrics record `held-out-assessment`. Coverage is accepted
  only against a declared target/tolerance and minimum assessment size.
  Benchmark model quality separately requires range-normalized RMSE under one
  predeclared output-independent threshold. Exact expected hashes control only
  `software_reproducibility_passed`, never model quality.
- `OODDetector` returns nearest-training distance and domain-excess distance in
  fitted unit-box coordinates. The flag is a review/guardrail signal, not an
  oracle.

## Persistence and identity

Exact-GP JSON v2 hashes the complete executable artifact: schema; raw and
normalized rows; affine bounds; noise; kernel family/version; ARD mode and
vector; signal and likelihood; full/selected jitter; calibration scale,
probability and provenance; OOD policy; and output semantics. Loading rejects
unknown keys, duplicate keys, nonfinite values, unsupported policy constants,
hash mismatch, and any difference from deterministic reconstruction. AR1
artifacts compose the two complete GP identities with bounded-rho semantics.

## Optional BoTorch boundary

`BoTorchTrainingData` exposes framework-neutral `n x d`, `n x 1`, and `n x 1`
training arrays plus hashes. `to_torch()` imports torch lazily and fails with an
actionable error if absent. This is a data contract, not an installed BoTorch
model and not a promise that acquisition behavior matches this runtime.

## Fixed-mesh fields

`PODFieldSurrogate` computes POD modes over snapshots on one coordinate-hashed
mesh, fits exact GPs to modal coefficients, and reconstructs mean and pointwise
variance. Constant/zero-energy snapshots use a serialized mean-only rank-0
representation that is identical with and without NumPy. Mesh hashes canonicalize
signed zero, and prediction rejects a different hash.

## Leakage and numerical failure policy

Spatial splitting takes the transitive closure of caller group IDs and exact
canonical coordinates, so coincident designs cannot cross partitions even when
mislabelled. Signed zero is exact-coordinate equivalent. Near-coordinate
tolerance is deliberately `0.0`; a nonzero tolerance is rejected because units
and per-dimension risk scales cannot be inferred safely.

Finite extreme extrapolation either reaches the stable zero-covariance kernel
limit or raises `SurrogateValidationError`. Public prediction, calibration,
interval, OOD, field, and metric APIs never return NaN or infinity.

The machine-readable contract is
`modern/spec/surrogates/runtime-v1.json`.
