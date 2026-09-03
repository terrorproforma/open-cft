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
