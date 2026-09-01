# L0 surrogate v4 results

## Identity and status

Scientific identity is valid:

- observed HEAD and protocol commit:
  `e29762ec1f06b4bd29b09c411e552396710d0aed`;
- commit existed and was present on `origin/feat/sota-foundation`;
- protocol tree hash:
  `29e64569ed7073493089c6eca4b44cde7c4035f926fc7f0d64b49876863a7ce5`;
- every committed v4 source/test blob matched the working tree;
- no intervening v4 protocol commit or untracked v4 protocol file existed.

The run is valid but **failed the predeclared numerical gates**. No active or
fixed-baseline replicate passed every output and stratum.

## Per-replicate overall metrics

Percentages below are range-normalized RMSE and worst absolute error.

### Group split 2026090201

- Active thrust: RMSE 2.164%, worst 8.757%, coverage 0.849 — failed coverage.
- Active Isp: RMSE 3.620%, worst 10.649%, coverage 0.821 — failed coverage.
- Baseline thrust: RMSE 5.420%, worst 29.197%, coverage 0.883.
- Baseline Isp: RMSE 8.305%, worst 34.175%, coverage 0.901.
- Active interpolation passed both outputs. Boundary coverage failed both;
  OOD coverage was 0.982 for thrust and 0.838 for Isp.

### Group split 2026090202

- Active thrust: RMSE 2.410%, worst 13.078%, coverage 0.834 — failed coverage.
- Active Isp: RMSE 2.764%, worst 9.051%, coverage 0.932 — passed overall.
- Baseline thrust: RMSE 6.209%, worst 30.657%, coverage 0.971.
- Baseline Isp: RMSE 5.440%, worst 20.611%, coverage 0.890.
- Active boundary thrust coverage was 0.760; active interpolation and OOD
  passed both outputs.

### Group split 2026090203

- Active thrust: RMSE 2.221%, worst 13.821%, coverage 0.915 — passed overall.
- Active Isp: RMSE 3.438%, worst 10.605%, coverage 0.841 — failed coverage.
- Baseline thrust: RMSE 7.086%, worst 36.434%, coverage 0.824.
- Baseline Isp: RMSE 10.277%, worst 31.654%, coverage 0.797.
- Active boundary thrust and interpolation coverage failed. OOD thrust
  coverage was 0.980; OOD Isp failed narrowly on RMSE at 5.072% and had 0.790
  coverage.

## Interpretation

At equal 96-row budgets, active selection substantially reduced point error
relative to the fixed sequence. The rejection is driven primarily by unstable
per-stratum split-conformal coverage, plus one marginal active OOD-Isp RMSE
failure. Thresholds and models were not changed after observing v3 or v4.

Run manifest:
`9230e1b5c288379c79e3f91efa41af96e0ec4e2710bac0d5d68455d70c15c688`.
Wall time was 185.894 s and is diagnostic only.

These results concern deterministic L0 software emulation only and provide no
physical-accuracy evidence.
