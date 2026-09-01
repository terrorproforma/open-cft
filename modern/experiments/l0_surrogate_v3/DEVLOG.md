# L0 surrogate v3 development log

## 2026-09-02 — preregistration

- Bound immutable v2 predeclaration, partition and failure hashes.
- Added root-confined atomic JSON/model serialization with parent creation,
  model hash reload, permission-failure propagation and temporary cleanup.
- Added mandatory environment/config/disk/path and three-replicate synthetic
  preflight. No real v3 assessment labels are accessed by preflight.
- Real execution remains locked until this source, its tests, input-only
  partitions and preflight record are committed and pushed.
