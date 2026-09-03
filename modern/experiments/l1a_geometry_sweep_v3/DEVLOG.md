# DEVLOG - L1a geometry sweep v3

## 2026-09-03

- Read the TWT/PPM review (beb4772c), sweep v2, cusp topology search v3.1 and the L1a
  field/geometry packages; measured the CPU solve cost (1.8 s at 80x144, 18 s at 2x).
- Corner feasibility solves (six corners of the proposed box) all pass the six sweep-v2
  metric gates: boundary_to_peak_ratio <= 0.0212, topology_confidence >= 0.868,
  manufacturing margins >= 50 um. Bounds fixed: r_w <= 4.2 mm, L >= 3.4 mm,
  clearance <= 1.6 mm, thickness <= 5.0 mm.
- Implemented: pure-Python scrambled Sobol (Joe-Kuo direction numbers, LMS + shift,
  SHA-256 bits), v3 builder (sweep-v2 rules, 2^-40 m length quantum), descriptors (Koch
  rho in four readings, I_1 prediction, wall harmonics, profiles), campaign mechanics on
  the shared ExperimentRuntime, v3 catalogue (schema 1.1.0), lifecycle runner, tests.
- Sobol sample: 128 designs, 43/43/42 with 3/4/5 stages, 6 inside the sweep-v2 box,
  51 predicted HEMP-like (x_w >= 1.9373), 17 of them five-stage.
- Single-design smoke (l1a-gs-v3-030, l1a-gs-v2-000): 28 s per design; all gates true.
