# L0 surrogate v3 learning scratchpad

Policy: `COMMITTED`, experiment-local.

## 2026-09-02 preregistration guardrails

- [user] Keep v2 immutable and bind its failure identity into v3 provenance.
- [self] Test actual model save/reload through missing and deep parent paths
  before freezing the successor.
- [self] Every artifact write uses a same-directory temporary file, flush,
  fsync and atomic replace; failures remove temporary files.
- [self] The synthetic preflight covers all three replicate output layouts and
  single-use assessment without loading real v3 assessment labels.
- [self] V3 changes serialization only. Selection, budgets, splits, conformal
  rules and gates remain inherited bit-for-bit from v2.
