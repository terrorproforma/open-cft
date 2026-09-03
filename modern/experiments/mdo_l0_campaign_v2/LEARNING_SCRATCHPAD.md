# MDO L0 campaign v2 learning scratchpad

- [self] Read the v1 audit's "what a v2 must change" list BEFORE designing: every item
  became either a protocol field or a binding gate (F9 result-commit policy, F10
  import-bound scope + gate + fresh-interpreter test, F22 rule-computed probabilities,
  F26 gate semantics, F27 duplicate elimination + gate, F28 arg-derived labels + gate).
- [self] Exact hypervolume of a growing set is monotone only in exact arithmetic; a
  strict `>=` gate on floating-point curves is a time bomb (tripped at -2e-16). Gate with
  a declared roundoff tolerance and record the largest decrease.
- [tool] BoTorch 0.18.1 `SingleTaskGP`/`MixedSingleTaskGP` default to RBF via
  `get_covar_module_with_dim_scaled_prior`; pass `use_rbf_kernel=False` for Matern-5/2
  and read the label from the fitted kernel classes rather than hard-coding it.
- [tool] qLogNEHVI cost on 4 objectives is dominated by the box decomposition per MC
  sample; halving the candidate set barely helps, halving MC samples helps ~20 %.
- [self] Under CL-1 with per-design posteriors the robust/nominal ratio is constant
  WITHIN a design and varies ACROSS designs; check separability per design, not pooled.
- [tool] Quantile inversion accuracy must be asserted in x (bracket by ulps), not in u:
  steep CDFs (Beta(0.5, 511.5)) give u-residuals of 1e-12 at x-accuracy of one ulp.
