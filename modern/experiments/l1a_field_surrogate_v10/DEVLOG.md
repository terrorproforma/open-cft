# L1a field-surrogate v10 devlog

## 2026-09-02

- Created the isolated v10 branch/worktree from patched runtime commit
  `b46e263`; main, accepted packages, FYP, and v1-v9 remain untouched.
- Used the immutable v9 method-freeze and terminal artifacts as the sole
  scientific development evidence. Prior partitions are referenced only to
  enforce coordinate exclusion.
- Expanded input-only screening to 1,024 rows and balanced every role across
  stage count, stratum, and polarity with 162/216/270 maximin prefixes.
- Added deterministic full-residual local interpolation over lossless physical
  snapshots with geometry and coarse-field descriptors.
- Preregistered pooled/stage-specific models, 8/16 neighbours, and
  Wendland-C2/inverse-distance kernels with stable duplicate/OOD handling and
  neighbour distance/spread uncertainty.
- Hyperparameters use three true held-out candidate folds; every balanced cell
  contributes absent validation rows before method access.
- Added independent bulk Br, bulk Bz, axis-Bz, and overlapping landmark-patch
  comparator declarations without neural or augmented-SVD coupling.
- Persist all 324 development checkpoints into the immutable result bundle
  with runtime artifact hashes so development metrics can be regenerated.
- Added an atomic persistent Git-common-dir attempt claim using the accepted
  runtime's pinned filesystem primitive before runtime construction.
- Synthetic science and all five shared-runtime terminal states passed without
  a real field solve or label access.
- Patched runtime `b46e263` globally sorts manifest inventory before sealing.
- Final preparation screened 1,024 rows (1,018 valid, 103 corrected, 6
  rejected), froze 432 with zero v1-v9 overlap and exact role/prefix balance,
  and passed all five synthetic terminal states without real solver access.
