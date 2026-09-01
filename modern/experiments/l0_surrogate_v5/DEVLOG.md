# L0 surrogate v5 devlog

## 2026-09-02 preregistration

- Defined a 16,384-row position-scrambled, digitally shifted radical-inverse
  design with fixed seed and exact input-only partitions.
- Proved zero exact surrogate-coordinate intersection with all v3/v4
  calibration and assessment identities.
- Added separate method-selection and final-calibration roles, each with at
  least 96 rows and eight independent groups per stratum.
- Recorded a successful synthetic serialization preflight with zero physics
  evaluations and zero assessment accesses.
- Added detached Git binding, complete imported dependency subtree hashes,
  and an atomic retained one-execution lock.
