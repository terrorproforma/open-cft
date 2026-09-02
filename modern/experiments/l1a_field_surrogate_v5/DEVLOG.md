# L1a field-surrogate v5 devlog

## 2026-09-02

- Created `exp/l1a-field-surrogate-v5` in an isolated worktree from
  `origin/feat/sota-foundation` at accepted runtime commit `231873d`; the main
  worktree, accepted packages, FYP, and v1-v4 branches remain untouched.
- Preserved v4's immutable prebundle failure (zero solver/label access) and
  moved v5 production root preflight, lock, atomic artifacts, cache cleanup,
  failure handling, and all five terminal states to `ExperimentRuntime`.
- Retained the v2 production geometry constructor and actual `nextafter` path,
  while introducing seed `20260905` and explicit v1/v2/v3/v4 coordinate
  exclusion.
- Added the missing `math` import and a runtime synthetic preflight over the
  same checkpoint, model, metric, selection, conformal, topology and latency
  functions used by the sealed runner.
- Added AST/symbol-table undefined-global checking, pre-read access counters,
  sidecar-protected phase inventories and unconditional `.working` cleanup.
- Added experiment-local Git attributes so JSON, sidecars, Python and Markdown
  are checked out as LF on Windows; NPZ remains binary.
- Preparation passed with 510/512 valid raw geometries (37 corrected, 2
  rejected), 240 frozen rows, zero prior-coordinate intersection, 11
  scientific path groups/554 path executions, and all five runtime states.
- Focused v5 plus accepted runtime tests passed 133 with 2 skips; full
  `compileall` passed and FYP diff/status remained empty. Repository-wide
  pytest with `--import-mode=importlib` reached 1221 passes, 5 skips, 31
  failures and 37 errors in pre-existing experiment/visualization evidence;
  those accepted out-of-scope paths were not modified.
