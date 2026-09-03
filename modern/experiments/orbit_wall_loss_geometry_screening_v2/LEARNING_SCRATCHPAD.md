# Learning scratchpad (orbit wall-loss geometry screening v2)

- [tool] orbit_mc v1.7 sealing (`_validate_probability`) and `coupling_v42_handoff` require
  `lower <= p <= upper` verbatim; `wilson_interval(0, n).lower` is a positive round-off for 734 of
  the first 4000 n (incl. 384) and `wilson_interval(n, n).upper` is `1 - ulp` for 1238 (incl. 512,
  640). Shakedown 3 died on a 6-launch control case with zero timeouts. Fix without touching the
  frozen package: one case per (design, cell, 128-launch block); control 16 / 64; the plan
  constructors refuse inexact sizes.
- [tool] orbit_mc v1.7 validates launch ids against `<campaign>:E..:P..:X..:D+-1:G..`; the Sobol
  index went into `G`, the cell index into `X`.
- [tool] canonical JSON integers are signed 64-bit: uint64 Sobol seeds are stored as decimal strings.
- [self] The catalogue keys sweep designs by their sweep case id, not by the sweep's long design id.
- [self] The catalogue's `accepted_field_identity_sha256` is v3.1's own hash scheme and differs from
  v1's field identity for all 96 designs; carry both, do not assert equality.
- [self] One anode-side cell (`l1a-gs-v2-088`, 0.16 mm) has its midpoint inside the injector zone;
  flag, never move a preregistered launch plane by hand.
- [self] Shakedown on 3 sweep designs + P2 hinted that interior cells launched at the mid-plane
  between cusps at 0.65-0.825 r_w hit the wall 16/16; the dataset must report the per-cell
  structure by position class before any pooled value (v1 lesson kept).
