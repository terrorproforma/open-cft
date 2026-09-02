# L1a field-surrogate v9 devlog

## 2026-09-02

- Created the isolated v9 branch/worktree from patched runtime commit
  `b46e263`; main, accepted packages, FYP, and v1-v8 remain untouched.
- Used the immutable v8 method-freeze and terminal artifacts as the sole
  scientific development evidence. Prior partitions are referenced only to
  enforce coordinate exclusion.
- Expanded input-only screening to 1,024 rows and balanced every role across
  stage count, stratum, and polarity with 162/216/270 maximin prefixes.
- Replaced inverted alignment with a lossless polarity-canonical residual on
  the original physical fine grid. Decode reads only the primary block.
- Added cylindrical-energy, overlapping tapered landmark, and axis-Bz
  null/extrema channels as basis objectives without introducing output seams.
- Candidate-only grouped physical L2/energy/topology gates choose the smallest
  passing rank from 64/96/128/192 before any method oracle or coefficient fit.
- All field QoIs derive from reconstructed physical fields; source error
  remains input-only. Qualified models retain low-field basis features,
  whitening, deterministic ARD, and per-mode regularization.
- Persist all 324 development checkpoints into the immutable result bundle
  with runtime artifact hashes so development metrics can be regenerated.
- Added an atomic persistent Git-common-dir attempt claim using the accepted
  runtime's pinned filesystem primitive before runtime construction.
- Synthetic science and all five shared-runtime terminal states passed without
  a real field solve or label access.
- Patched runtime `b46e263` globally sorts manifest inventory before sealing.
- Final preparation screened 1,024 rows (1,018 valid, 105 corrected, 6
  rejected), froze 432 with zero v1-v8 overlap and exact role/prefix balance,
  and passed all five synthetic terminal states without real solver access.
