# Learning scratchpad — wall-loss geometry surrogate v2

- A check that was calibrated for 11 inputs (in-sample ridge above the binomial floor) is not
  dimension-free: with 31 features and 50 designs it fails for overfitting, not tautology. Any
  "algebraic reconstruction" check must be out-of-sample (leave-one-out) to stay meaningful.
- Sharing a single fit-role unit box across the mixture's per-stage-count GPs keeps the predictor
  contract to ONE normaliser; the dispatch is a pure routing rule on a physical feature.
- A paired comparison against a prior campaign must tolerate a shakedown whose roles differ from
  the prior one (pair on the shared designs, label it), or the shakedown crashes in assessment.
- Realised lengths (pitch, magnet radial thickness) legitimately coincide with raw design variables;
  the distinguishing property of a derived feature is "deterministic from the committed row", not
  "different name".
- Fixing the input representation moved the pooled RMSE from 0.056 to 0.034 for EVERY model,
  including ridge; the gain was in the features, not the kernel. A "≥ 2× best baseline" gate cannot
  be met when the baseline shares the good features — the gate measures the model family, and the
  informative question became "how far above the binomial floor" instead.
- A learning curve that flattens at 30 of 50 designs says the next unit of evidence is label
  precision (launches per design), not design count; report the power-law extrapolation with that
  caveat rather than as a headline.
- When a dashboard shows another campaign's numbers, read them from THIS campaign's hash-bound
  comparison artifact and cross-check against the other campaign's committed file in the test; do
  not read the other bundle at render time.
- `Get-ChildItem` output can be swallowed in this shell; `cmd /c dir` is the reliable existence
  check before concluding a headless browser produced nothing.
