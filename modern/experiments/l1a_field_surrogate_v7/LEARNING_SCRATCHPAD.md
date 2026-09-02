# L1a field-surrogate v7 learning scratchpad

## Pre-registration observations

- [user] Preserve v1-v6 and use only v6 development evidence for v7.
- [self] Coordinate exclusion is an integrity dependency, not scientific
  tuning; the dependency lock distinguishes prior partitions from the sole v6
  method-freeze development source.
- [user] A field coefficient model cannot qualify until its projection oracle
  passes L2, energy, and topology gates.
- [self] Enforcing that rule before fitting coefficients also makes rejection
  decomposition explicit: basis capacity versus projection versus regression.
- [user] Development evidence must independently regenerate every metric.
- [self] Hash-only transient cache inventories are insufficient after cleanup;
  publish immutable copies of all candidate/method checkpoints through
  `RunContext.write_blob`.
- [user] Prevent duplicate attempts across worktrees.
- [self] Result-root locks are worktree-local. A persistent exclusive claim in
  the Git common directory closes that namespace race while leaving shared
  runtime ownership of result lifecycle and cleanup intact.
- [self] Stage-wise POD requires stage-wise routing at method, calibration,
  assessment, and latency inference; a global basis would silently violate the
  preregistration even if its metrics looked favorable.
