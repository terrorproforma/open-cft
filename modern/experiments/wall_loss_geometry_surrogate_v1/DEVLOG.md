# Devlog — wall-loss geometry surrogate v1

## 2026-09-03 (AEST) — build, shakedown, preregistration

- Read the screening dataset (96 designs, 11 declared inputs, per-cell and
  pooled counts at accepted-2N), the nine L0 and two in-repo L1a surrogate
  campaigns, the v9 tautology audit, `cft_revival.surrogates` and the MDO v1
  lifecycle template. Binomial floors measured on the data: cells 0.035 (94 of
  384 cell labels are exactly 1.0), pooled 0.020; across-design sd: pooled
  0.100, cells 0.18–0.25.
- Wrote `protocol.json` (gates verbatim from the brief, no later edits),
  `data.py` (binding by byte hash + Git blob at `ab7c2897`, count-ratio check,
  Haldane–Anscombe logit / Laplace direct noise, seeded stratified partition
  with the top-decile chamber-length extrapolation cluster), `models.py`
  (five candidates exported to one GP contract block format, three baselines,
  permutation importance, uncertainty-sampling add-on), `predictor.py`
  (numpy-only consumer), `experiment.py` (callbacks, single-use label reads
  recorded through `before_expensive(kind="label")`, gates), `run.py`.
- BoTorch 0.18.1 facts used: `SingleTaskGP` defaults to an RBF kernel, so the
  Matérn-5/2 ARD kernel is passed explicitly via
  `get_covar_module_with_dim_scaled_prior(use_rbf_kernel=False)`; `MultiTaskGP`
  accepts `train_Yvar` and exposes the task covariance through
  `covar_module.kernels[1].covar_matrix`; both posteriors reproduce in numpy to
  ~1e-14.
- Shakedown run 1 (shakedown partition, 40 s): every stage ran, bundle
  validated, determinism bit-exact, predictor replay 5e-15; the terminal state
  was `assessment_rejection` because the shakedown decision used the science
  gates. Fixed: the shakedown plan is decided by the structural gates only
  (science gates informational), the evidentiary plan by all binding gates.
- Shakedown run 2 on the final code: see `shakedown.json` (informational
  numbers must not be read as the result).

## 2026-09-03 (AEST) - preregistration, execution, record

- Commits: `aa9349a9` code/protocol/tests, `b602d147` preregistration
  (shakedown.json, partitions.json, authorities.json; rebased once onto
  `c32dd780` before the push, hash scope unchanged), `b400d924` record.
- Shakedown 2 (final code, 37 s): `accepted_result` on structural gates;
  informational science gates: pooled 0.040 PASS, cells 0.077 FAIL, 2x 1.36
  FAIL, coverage 0.95 PASS; selected `botorch-stgp-direct`.
- Execution (detached worktree `uni-project-wl-surrogate-run` at `b602d147`,
  cpu, 8 threads, 169 s): `assessment_rejection` / `rejected_surrogate`;
  selected `botorch-icm-logit`; see README "Recorded result".
- Dashboard `modern/visualization/wall-loss-geometry-surrogate-v1.html`
  (generator + template + 7 tests); headless Edge: 0 JS errors, screenshots
  under `%TEMP%\wls_probe\shots\`.
