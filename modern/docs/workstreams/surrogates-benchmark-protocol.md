# Surrogate Benchmark Protocol

## Interpretation

This benchmark measures interpolation of deterministic hypothetical L0 output.
It is not physical validation, higher-fidelity agreement, or state-of-the-art
evidence.

## Leakage and uncertainty rules

1. Canonicalize signed zero, then take transitive closure over caller group IDs
   and exact coordinates before partitioning.
2. Use `grouped-spatial-coordinate-closure-v2` with exact tolerance `0.0`.
   Near-coordinate grouping needs declared unit-aware scales and must happen
   upstream; the generic splitter rejects a nonzero tolerance.
3. Keep calibration-fit rows distinct from held-out assessment rows.
4. Carry nominal interval probability through every prediction and report.
5. Report RMSE, full-hashed-dataset output range, range-normalized RMSE, MAE,
   worst-case absolute error, empirical coverage, target, tolerance,
   acceptance, and assessment sample count per output.
6. Identify dataset, split, model configuration, and complete result separately
   with canonical SHA-256 hashes.

## Authoritative reproducibility check

“Authoritative” means only that this exact software result is immutable and
reproducible from its hashes. It does not mean the model passes quality gates.

The one checked result uses:

- base config `modern/config/l0-deterministic-sweep.json`;
- overrides `batch_size=48`, L0 generator `seed=31`;
- ARD exact-GP runtime `cft-surrogate-exact-gp/2.0.0`;
- spatial validation fraction `0.25`, split seed `9`;
- 90% nominal/target coverage with a predeclared `0.05` absolute tolerance;
- minimum coverage-assessment size `n=30`, while this check has only `n=12`;
- one output-independent RMSE gate:
  `RMSE / full-hashed-dataset output range <= 0.05`;
- no calibration-fit rows, explicitly labelled
  `uncalibrated-held-out-assessment`.

- source config:
  `194b21ec5d40e87a6ac9a934025bd0dcc848c29d367aad4fac920c9d3677def2`;
- dataset:
  `2647de548f3fe3ca3cc070eeca9fac187b877904bc729061ccc42c6c5ab77006`;
- split:
  `f44fdd4b2498ebc73157c91cc9bbbc4ed33e77fc8753825217b92f1ba43c4360`;
- configuration:
  `100834899102d9d6a19c410f82096e72b70c36612db5fe718f9ad70185e6fc7f`;
- complete benchmark:
  `ccf730a32bbb0bd82399e8728e36ffe24b9e2d8a051c2040703713d436c8a566`.

Results on 12 held-out points:

- axial thrust: RMSE `0.007894429684864392 N`, MAE
  `0.006534945269753859 N`, worst absolute error
  `0.01590895031552436 N`, dataset range `0.034796449298740074 N`,
  normalized RMSE `0.22687457611228848`, coverage `0.75`;
- specific impulse: RMSE `337.7674123910935 s`, MAE
  `270.7299834538546 s`, worst absolute error `782.2225334223967 s`,
  dataset range `1634.9201412627206 s`, normalized RMSE
  `0.20659566413453131`, coverage `11/12`.

Exact statuses:

- `software_reproducibility_passed=true`;
- `assessment_limited=true`;
- thrust `rmse_accepted=false`, `coverage_accepted=false`,
  `model_quality_passed=false`;
- Isp `rmse_accepted=false`, `coverage_accepted=false`,
  `model_quality_passed=false`;
- overall `model_quality_passed=false`.

Isp's point coverage is close to nominal, but the assessment has fewer than 30
held-out rows, so coverage cannot pass. Thrust also misses the 5-percentage-point
coverage tolerance. Both normalized RMSE values greatly exceed the predeclared
5% gate. Calibration remains absent and separate from assessment. Surrogate
improvement and a larger held-out assessment are required.
