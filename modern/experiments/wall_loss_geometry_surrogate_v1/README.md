# Wall-loss geometry surrogate v1

**Classification: `SURROGATE_OF_SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`.** A
Gaussian-process surrogate of the first wall-loss-vs-geometry dataset
(`modern/experiments/orbit_wall_loss_geometry_screening_v1`, label
`SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`): eleven geometry-v1.1 design
variables in, collisionless test-particle wall-hit probability per axial cell
(4 × 128 launches), pooled (512 launches) and pooled reflection probability out,
with calibrated 90 % intervals. It is a surrogate of a **screening** dataset on
L1a linear-vacuum fields — not physical-orbit evidence, not a performance model,
and it validates neither the orbit model nor the field model.

## Why this campaign is not a repeat of the 19 failed surrogate campaigns

| Audited failure (l0_surrogate v1–v9, l1a_field_surrogate v1–v10) | Here |
| --- | --- |
| No physically varying training data | 96 accepted designs × orbit integration in 96 re-solved fields |
| Tautological target (v9: target = algebra of inputs) | targets are orbit termination counts; `no_tautology` gate (count ratios, single-input affine R² < 0.99, ridge stays above the binomial floor) |
| Held-out sets not frozen / pooled-row conformal ranks | `partitions.json` frozen at preregistration and hash-bound; roles fit / method-selection / calibration / assessment / extrapolation; assessment and extrapolation labels are read exactly once, after the predictor is written (runtime access records prove the order) |
| Gates in normalised units, tuned after the fact | gates in probability units, written before the shakedown, `thresholds_tunable_after_execution: false` |
| Unknown noise treated as free hyperparameter | binomial noise is known per label and enters the GP as fixed heteroscedastic variance (logit or direct working space) |
| No trivial baseline | global mean, k-NN(3) and ridge, with the best one chosen on the assessment role itself |

## Design

- **Inputs**: the sweep's eleven declared variables (verified non-degenerate: ≥ 72
  distinct values each, 96 distinct tuples). Known step discontinuities of the
  design → geometry map (stage count, polarity, exit minimum length) are declared
  as irreducible model error.
- **Partition** (seeded, stratified over the non-dominated flag): the 10 longest
  chambers (top decile, all five-stage) are the EXTRAPOLATION hold-out; the other
  86 split 50 fit / 10 method-selection / 10 calibration / 16 assessment.
- **Candidates** (selected on the method-selection role by mean P-unit RMSE):
  package `ExactGP` (ARD Matérn-5/2, grid hyperparameters) on logit and direct
  targets; BoTorch `SingleTaskGP` with `train_Yvar` and the dimension-scaled
  Matérn-5/2 prior on logit and direct targets; BoTorch `MultiTaskGP` ICM over the
  four cells (logit). The package has no ICM kernel, so that variant is BoTorch.
- **Calibration**: one variance scale from the 50 standardised calibration
  residuals (`VarianceCalibrator`, 90 %); intervals are observation intervals
  (latent + known binomial noise at the predicted mean).
- **Binding gates**: pooled-P RMSE ≤ 0.05; floor-corrected cell RMSE ≤ 0.05;
  ≥ 2× the best baseline (pooled P); 90 % coverage in [0.80, 0.97]; no
  tautology; determinism replay; dataset/Git binding; frozen partition +
  single-use labels; predictor-contract replay; code contract.
  **Reported**: extrapolation RMSE ≤ 0.10 and coverage, raw cell RMSE, cell 2×
  ratio, reflection output, active-learning add-on.
- **Predictor contract** `results/artifacts/predictor.json` + `predictor.py`
  (numpy only): physical inputs → per-output probability with latent and
  observation 90 % intervals and interpolation-scope flags; every prediction
  carries both labels. The contract reproduces the fitted library posteriors
  within 1e-9 (gate).

## Lifecycle

```
# from modern/, .venv-sota interpreter, cpu only (torch 8 threads)
python -m experiments.wall_loss_geometry_surrogate_v1.run shakedown  # real data, shakedown partition namespace
python -m experiments.wall_loss_geometry_surrogate_v1.run prepare    # writes partitions.json + authorities.json
# commit "preregister wall-loss geometry surrogate v1", push, then from a clean detached worktree
python -m experiments.wall_loss_geometry_surrogate_v1.run execute
python -m experiments.wall_loss_geometry_surrogate_v1.run validate
```

Dashboard: `modern/visualization/wall-loss-geometry-surrogate-v1.html`.
Tests: `modern/tests/experiments/wall_loss_geometry_surrogate_v1` (system
Python runs the dependency-free subset; the venv runs everything).

## Recorded result (single execution at `b602d147`, recorded in `b400d924`)

**Terminal state `assessment_rejection` - `rejected_surrogate`.** Three of the
four science gates failed on the single-use assessment role; every structural
gate passed. Numbers as recorded in `results/artifacts/`:

| gate | value | threshold | result |
| --- | --- | --- | --- |
| pooled P(wall) RMSE | 0.0562 (floor-corrected 0.0525) | <= 0.05 | FAIL |
| cell P(wall) RMSE, floor-corrected | 0.129 (raw 0.133, floor 0.034) | <= 0.05 | FAIL |
| best baseline / surrogate (pooled) | 0.97x (ridge 0.0546; mean 0.0553; k-NN 0.0681) | >= 2.0x | FAIL |
| 90 % coverage (80 intervals) | 0.800 (64 / 80) | [0.80, 0.97] | PASS |
| extrapolation pooled RMSE (reported) | 0.093 | <= 0.10 | met |
| extrapolation coverage (reported) | 0.84 | [0.80, 0.97] | met |
| structural gates (6) | binding, single-use labels, no tautology, bit-exact determinism, predictor replay 3e-14, code contract | - | PASS |

Selected candidate `botorch-icm-logit` (method-selection mean RMSE 0.1035 vs
0.1047 / 0.1067 / 0.1375 / 0.158 for the others); calibration inflated the GP
variance 3.34x. Permutation importance: `stage_count_selector` (0.18) and
`exit_length_fraction` (0.16) dominate, `stage_pitch_m` 0.03, everything else
< 0.004 - the step discontinuities of the design -> geometry map, not smooth
magnet parameters, move the cell probabilities. The surrogate is NOT usable
as an MDO v2 input beyond the trivial baselines; `predictor.json` is published
as the recorded (rejected) contract for audit only.
