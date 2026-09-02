# L1a field-surrogate v8 learning scratchpad

## Pre-registration observations

- [user] Preserve v1-v7 and use only v7 development evidence for v8.
- [self] Coordinate exclusion is an integrity dependency, not scientific
  tuning; the dependency lock distinguishes prior partitions from the sole v7
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
- [user] Balance stage count, input stratum, and polarity in every role and
  every candidate budget prefix.
- [self] Eighteen stage/stratum/polarity cells make all three budgets exact
  round boundaries, so deterministic cell-local maximin ordering preserves
  balance without future-label access.
- [user] Source error is input-only while mirrors and gradients derive from
  reconstructed output.
- [self] Keeping these derivation paths outside scalar fitting removes both an
  avoidable surrogate and label leakage from model-family selection.
- [self] V7's finalization exposed an insertion-order assumption. Patched
  runtime `b46e263` sorts inventory globally before manifest validation.
