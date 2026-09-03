# Learning scratchpad — wall-loss geometry surrogate v1

- [self] Decide the gate list and thresholds BEFORE the shakedown and never
  edit them afterwards; the shakedown record binds the protocol hash so any
  later edit is visible.
- [self] A shakedown whose science gates fail is still a passed shakedown of
  the CODE PATH; decide the shakedown plan by structural gates only, or every
  honest negative preview looks like a broken pipeline.
- [tool] BoTorch 0.18.1 `SingleTaskGP` defaults to RBF, not Matérn-5/2; pass
  `get_covar_module_with_dim_scaled_prior(ard_num_dims, use_rbf_kernel=False)`.
  Without the dimension-scaled prior, plain `ScaleKernel(MaternKernel)` sends
  irrelevant length-scales to ~4000 on 50 points.
- [tool] `MultiTaskGP` in 0.18.1 has no `task_covar_module`; the ICM task
  covariance is `covar_module.kernels[1].covar_matrix` and the means are
  `mean_module.base_means[i].constant`.
- [self] With 128 launches per cell the label noise floor (0.035) is 70 % of a
  0.05 RMSE gate; declare the floor and report floor-corrected RMSE beside the
  raw value, do not loosen the gate.
- [self] k-NN(3) is a strong baseline for the pooled probability in 11-D with
  50 fit points (a 2x margin over it is demanding); say so before the run.
- [tool] Everything written through `context.write_json` must be plain JSON
  types: numpy scalars/bools and tuples are rejected or tagged by the
  canonicaliser (`_plain()` before every write).
