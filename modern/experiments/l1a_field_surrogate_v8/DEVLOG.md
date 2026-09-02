# L1a field-surrogate v8 devlog

## 2026-09-02

- Created the isolated v8 branch/worktree from patched runtime commit
  `b46e263`; main, accepted packages, FYP, and v1-v7 remain untouched.
- Used the immutable v7 method-freeze and terminal artifacts as the sole
  scientific development evidence. Prior partitions are referenced only to
  enforce coordinate exclusion.
- Expanded input-only screening to 1,024 rows and balanced every role across
  stage count, stratum, and polarity with 162/216/270 maximin prefixes.
- Added a strict alignment roundtrip gate and stage-local joint POD objective
  with energy, unweighted axial-window, and explicit axis-Bz channels.
- Made source error input-only and mirrors/gradients reconstruction-derived.
  Scalar discrepancy targets require candidate-only suitability evidence.
- Replaced correlation ARD heuristics with candidate-only grouped CV and
  marginal-likelihood length selection. Qualified field models use dedicated
  low-field bases and singular-value-whitened coefficients.
- Persist all 324 development checkpoints into the immutable result bundle
  with runtime artifact hashes so development metrics can be regenerated.
- Added an atomic persistent Git-common-dir attempt claim using the accepted
  runtime's pinned filesystem primitive before runtime construction.
- Synthetic science and all five shared-runtime terminal states passed without
  a real field solve or label access.
- Patched runtime `b46e263` globally sorts manifest inventory before sealing.
- Final preparation screened 1,024 rows (1,017 valid, 105 corrected, 7
  rejected), froze 432 with zero v1-v7 overlap, proved exact 18-cell role and
  budget-prefix balance, and passed all five synthetic terminal states without
  real solver or label access.
