# L1a field-surrogate v7 devlog

## 2026-09-02

- Created the isolated v7 branch/worktree from accepted runtime commit
  `231873d`; main, accepted packages, FYP, and v1-v6 remain untouched.
- Used the immutable v6 method-freeze and terminal artifacts as the sole
  scientific development evidence. Prior partitions are referenced only to
  enforce coordinate exclusion.
- Expanded input-only screening to 768 rows and froze 224 candidate plus 48
  method/calibration/assessment rows with balanced 16-row strata.
- Replaced mixed/high-only families with observed-coarse-only scalar and field
  models. Added per-output transformed discrepancies, regularized mirror
  transforms, source grid phases, fitted constant means, standardized outputs,
  and ARD Mahalanobis Matérn-5/2 kernels.
- Added polarity canonicalization, all-stage piecewise alignment, stage-count
  POD bases, coarse-energy normalization, budget-specific rank/retention
  limits, observed low-field modal coefficient features, and a mandatory
  projection-oracle gate before coefficient fitting.
- Persisted all 272 development checkpoints into the immutable result bundle
  with runtime artifact hashes so development metrics can be regenerated.
- Added an atomic persistent Git-common-dir attempt claim using the accepted
  runtime's pinned filesystem primitive before runtime construction.
- Synthetic science and all five shared-runtime terminal states passed without
  a real field solve or label access.
- Preparation screened 768 rows (761 valid, 71 corrected, 7 rejected), froze
  368 unique rows with zero v1-v6 coordinate overlap, and completed all 11
  scientific path groups with no unresolved static names.
