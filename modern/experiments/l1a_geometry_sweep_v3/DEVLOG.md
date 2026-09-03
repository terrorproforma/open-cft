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
- Shakedown (9 real designs, non-evidentiary): passed first run, 86 s, bundle validated,
  replay bit-identical; projected 1498 s at 6 workers. Committed with the code
  (`b04d5935`); `prepare` frozen and committed as `1923ef76` ("preregister L1a
  geometry sweep v3"), pushed to origin/exp/l1a-geometry-sweep-v3.
- Execution (detached worktree at 1923ef76, 6 CPU workers, GPU untouched): 224/224
  designs resolved, design stage 1576 s + assessment 101 s (29 min wall), terminal state
  `accepted_result`, status `accepted_l1a_sweep_v3`, all 11 binding gates true; recorded
  as `2cfe8223` ("record L1a geometry sweep v3 result", 980 files, 49 MB).
- Headline (reported): 15/128 Sobol designs HEMP-like (11.7 %); 51 predicted by
  I_1(x_w) >= 1.5 -> 15 realised, 36 predicted-only (28 fail only at their end cusps),
  0 realised-only; accuracy 0.72. rho_cons on I_1(x_w): slope 0.689, R^2 0.39, 70 % of
  365 cusps within +-25 %; rho / I_1 median 0.80 (end cusps) and 0.87 (interior cusps).
  Realised threshold x* = 2.34 (r_w / L = 0.745; smallest realised HEMP-like x_w 2.25)
  vs the I_1 prediction 1.94 (0.617). Five-stage four-cusp HEMP-like: 2 (005, 106).
  Sweep-v2 region (96 + 6): 0 HEMP-like, max rho 0.993. rho_wall < 1 for all 642 cusps.
  Held-out 96/96 (QoIs, 479 axis nulls to 27 um, 4 stored maps). All 224 stable; rho
  resolution sensitivity median 0.8 %, max 8 %.
- Dashboard `modern/visualization/l1a-geometry-sweep-v3.html` (641 kB, 6 cusp maps);
  headless Edge screenshots at 1440 px and 390 px under %TEMP%\sweepv3.
