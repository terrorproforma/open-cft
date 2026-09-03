# Wall-loss geometry surrogate v2 (derived physical features)

Classification `SURROGATE_OF_SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`: a surrogate of the
SCREENING dataset of `orbit_wall_loss_geometry_screening_v1` (96 accepted L1a sweep-v2 designs,
512 launches each, 4 axial cells). Not physical-orbit evidence, not a performance model.

## What v2 changes against v1 (and nothing else)

v1 (`wall_loss_geometry_surrogate_v1`, recorded `rejected_surrogate` at `b400d924`) fitted GPs on the
eleven raw design selectors; its permutation importance put almost all signal on the
step-discontinuous selectors (`stage_count_selector`, `exit_length_fraction`), which a stationary GP
cannot represent. v2 keeps v1's roles and gates **verbatim** and changes:

1. **Inputs = 31 derived physical features** (`features.py`, hash-bound): channel straight length
   `L = exit_start - injector_length`, wall radius, exit start / length / realised fraction / outer
   radius ratio, stage count (integer + one-hot), pitch, polarity, magnet dimensions, the L1a
   field descriptors recorded in the sweep (bore max |B|, centreline peak / mid |Bz|, log mirror
   ratios, gradient RMS, log field energy, boundary-to-peak ratio) and each cell's distance to
   the nearest axis |Bz| peak and Bz null in pitches. Every feature is a deterministic arithmetic
   function of the committed dataset row; **derived, not fitted**. Provenance per feature is in
   `features.FEATURE_TABLE` and published as `artifacts/features.json`.
2. **Candidates**: (a) BoTorch SingleTaskGP, Matérn-5/2 ARD, fixed binomial noise, logit and
   direct; (b) a per-stage-count GP mixture (one GP per stage count with ≥ 8 fit designs — the fit
   role has 13 / 29 / 8 — sharing the prior family and unit box, all-count GP as fallback, explicit
   dispatch in the predictor contract).
3. **Baselines**: mean, k-NN(3), ridge, **and gradient-boosted trees** (scikit-learn, deterministic,
   grid chosen on the method-selection role). If the tree beats the GP, that is the finding.
4. **Partition**: v1's `partitions.json` inherited by hash (same seed namespace, seed, counts;
   byte-identical file), so the 16 assessment and 10 extrapolation designs are identical to v1's.
5. **Reported, non-gated**: paired v1-vs-v2 comparison on those identical designs; learning curve
   on fit sizes 20 / 30 / 40 / 50 (3 seeds, scored on the method-selection role) with a log-log
   extrapolation of the designs needed for pooled RMSE 0.05.
6. **No-tautology check (c)** is leave-one-out ridge (32 coefficients on 50 designs overfit in-sample).

Gates (binding, v1 verbatim): pooled RMSE ≤ 0.05; floor-corrected cell RMSE ≤ 0.05; ≥ 2× best
baseline (pooled); 90 % coverage in [0.80, 0.97]; plus the structural gates. Extrapolation ≤ 0.10
reported. Acceptance marks `predictor.json` (via `artifacts/predictor-status.json` and
`campaign-result.json`) `usable_as_mdo_v2_input_with_screening_label`; rejection records why and
what v3 would need.

## Lifecycle

```
python -m experiments.wall_loss_geometry_surrogate_v2.run shakedown   # real data, shakedown namespace, non-evidentiary
python -m experiments.wall_loss_geometry_surrogate_v2.run prepare     # refuses without a passing shakedown; partitions.json == v1's
# commit "preregister wall-loss geometry surrogate v2", push; then from a clean detached worktree:
python -m experiments.wall_loss_geometry_surrogate_v2.run execute
python -m experiments.wall_loss_geometry_surrogate_v2.run validate
```

CPU only (`.venv-sota`, 8 threads). Dashboard: `modern/visualization/wall-loss-geometry-surrogate-v2.html`.

## Result (recorded `a2b503be`, terminal `assessment_rejection`, `rejected_surrogate`)

Selected on the method-selection role: `botorch-stgp-logit` (mean-over-outputs RMSE 0.0637 vs
direct 0.0644, stage mixture 0.0835; baselines ridge 0.0512, gbt 0.0566, k-NN 0.0878, mean 0.1616).

Assessment role, same 16 designs as v1:

| quantity | v1 (`b400d924`) | v2 | ridge | gbt | gate |
|---|---|---|---|---|---|
| pooled P(wall) RMSE | 0.0562 | **0.0337** | 0.0334 | 0.0368 | ≤ 0.05 **pass** |
| cells RMSE raw / floor-corrected | 0.1332 / 0.129 | **0.0904 / 0.0836** | 0.0778 | 0.0906 | ≤ 0.05 **fail** |
| best-baseline ratio (pooled) | 0.97× | 0.99× (ridge) | – | – | ≥ 2× **fail** |
| 90 % coverage (gated outputs) | 0.800 | 0.8125 | – | – | [0.80, 0.97] **pass** |
| extrapolation pooled RMSE / coverage | 0.093 / 0.84 | 0.1027 / 0.80 | 0.0557 | 0.1563 | ≤ 0.10 reported, missed |

v2 improves on v1 for every output on the identical designs (pooled −0.0225, cells −0.0428; v2
closer on 8 / 16 pooled and 11–12 / 16 for cells 1–3), passes the pooled gate v1 failed, but fails
the cell gate and the 2× baseline gate: ridge on the derived features is as good as the GP
(0.0334), and the tree baseline does not beat the GP (pooled 0.0368, cells 0.0906) — the remaining
error is not the step structure. Variance scale 1.86 (v1 3.34). All structural gates passed.

Learning curve (pooled RMSE on the method-selection role, 3 seeds): 20 → 0.0527, 30 → 0.0455,
40 → 0.0448, 50 → 0.0449 — flat beyond 30 designs (log-log slope −0.18). Features that matter
(permutation, ARD and tree importance agree): the per-cell cusp / null distances in pitches,
`stage_pitch_m`, `stage_count_is_5`; the magnet and field-magnitude descriptors carry little.

`predictor.json` is written and replays (1e-15) but is marked
`not_usable_as_mdo_v2_input_rejected_surrogate`. What v3 would need (from `campaign-result.json`):
lower label noise (more launches per design; the cell floor 0.035 is 70 % of the gate) rather than
more designs of the same kind; a 2× gate restated against the binomial floor instead of a ratio to
a mean baseline whose pooled RMSE is already 0.055; and a cell-level model that uses the cusp /
null distances directly (they are the signal in all three importance measures).

Dashboard: `modern/visualization/wall-loss-geometry-surrogate-v2.html` (v1 vs v2 panel, learning
curve, feature table with provenance; tests in `modern/tests/visualization/`).
