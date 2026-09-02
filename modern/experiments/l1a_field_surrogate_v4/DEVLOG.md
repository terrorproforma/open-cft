# L1a field-surrogate v4 devlog

## 2026-09-02

- Created `exp/l1a-field-surrogate-v4` in an isolated worktree from
  `origin/feat/sota-foundation`; the main worktree and v1/v2/v3 remain untouched.
- Retained the v2 production geometry constructor and actual `nextafter` path,
  while introducing a fresh v4 seed and explicit v1/v2/v3 coordinate exclusion.
- Added the missing `math` import and a runtime synthetic preflight over the
  same checkpoint, model, metric, selection, conformal, topology and latency
  functions used by the sealed runner.
- Added AST/symbol-table undefined-global checking, pre-read access counters,
  sidecar-protected phase inventories and unconditional `.working` cleanup.
- Added experiment-local Git attributes so JSON, sidecars, Python and Markdown
  are checked out as LF on Windows; NPZ remains binary.
