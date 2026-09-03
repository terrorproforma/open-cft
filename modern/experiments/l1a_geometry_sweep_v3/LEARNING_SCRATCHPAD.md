# Learning scratchpad - L1a geometry sweep v3

## Established before execution

- The design criterion of the device (Koch rho) was never reachable in the sweep-v2
  parameterisation; v3 varies the ratio the HEMP literature optimises (r_w / L) instead
  of adding more of the same variables.
- The wall-cusp definition is imported from cusp topology search v3.1, never
  re-implemented; the numerical parameters are asserted equal to the v3.1 protocol.
- rho is reported in four readings because Koch's paper does not state the radius of the
  downstream field; the conservative reading (stronger adjacent axis peak) is the
  classifier and the others are reported.
- The hypothesis (rho tracks I_1(x_w); threshold x* = 1.937318) is a REPORTED test, not
  a gate; the binding gates are integrity only.
- [self] 22 of the 128 raw Sobol value sets broke the geometry v1.1 ULP identity
  `(r_w + d) - r_w == d` by rounding (the sum crosses 2^-8 m). The sweep-v2 inward-ULP
  walk cannot repair a chamber-radius/dielectric pair, so v3 represents every radial
  length as a multiple of 2^-40 m; requested and represented values are both recorded.
- [self] Two-dimensional Sobol projections are balanced only for t = 0 pairs; a test that
  demands 8x16 balance for arbitrary dimension pairs is wrong, not the sampler.
- The bounds were fixed after six corner solves for the sweep-v2 gates (boundary ratio
  <= 0.021 vs 0.05); no rho value was consulted.

## Learned from the accepted result

- [self] The single-harmonic I_1(x_w) is an UPPER envelope of the realised Koch ratio,
  not its value: rho / I_1 sits at 0.80 (end cusps) and 0.87 (interior cusps) because
  the finite stack's end field raises the adjacent axis peaks; the HEMP-like threshold
  moves from r_w / L = 0.617 to about 0.75. Report end and interior cusps separately.
- [self] 28 of the 36 "predicted but not realised" designs fail only at their end
  cusps: the anode-side and exit-side cells see the un-cancelled end field. A HEMP-like
  five-stage four-cusp stack needs x_w >= 2.6 here, and only 2/42 reached it.
- [tool] `validate_bundle` pins the root identity; it passes only in the worktree that
  ran the execution. Byte-level verification through the manifest is the portable check.

## Claim guardrail

L1a linear-vacuum screening only. The declared soft-iron pole pieces are source-free
vacuum in the field; for r_w / L > 0.5 an L1b/P2 confirmation is queued, not run. No
plasma, mirror-probability, thrust or efficiency claim is permitted.
