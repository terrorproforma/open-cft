# L1a field-surrogate v6 devlog

## 2026-09-02

- Created `exp/l1a-field-surrogate-v6` in an isolated worktree from accepted
  shared-runtime commit `231873d`; main, accepted packages, FYP, and v1-v5
  evidence remain untouched.
- Preserved v5's immutable `prebundle_failure`: its callback rechecked global
  cleanliness after runtime-owned output creation and rejected that output.
- Moved detached HEAD, commit subject/path isolation, dependency closure,
  remote identity, and one repository-wide clean check before
  `ExperimentRuntime` construction and immutable attestation.
- Prebundle now consumes the frozen attestation/closure. Runtime drift rejects
  all tracked/staged changes and foreign untracked paths while allowing only
  exact v6 result/cache descendants.
- Retained the complete v5 scientific callbacks, shared-runtime five-state
  synthetic matrix, sequential checkpoints, and strict preregistered gates.
- Introduced seed `20260907` and explicit v1-v5 coordinate exclusion.
- Preparation passed with 508/512 valid geometries (37 corrected, 4 rejected),
  240 frozen rows, zero prior overlap, 11 scientific groups/554 executions,
  all five synthetic terminal states, and no real solver/label access.
- Focused v6 plus accepted runtime tests passed 138 with 2 skips; source
  compilation and protected FYP diff/status checks passed.
