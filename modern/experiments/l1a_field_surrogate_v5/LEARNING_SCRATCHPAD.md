# L1a field-surrogate v5 learning scratchpad

## Pre-registration observations

- [user] V1-v4 failures are immutable evidence. V5 must use a fresh isolated
  branch, seed, and coordinate set without modifying accepted packages or FYP.
- [self] A compile-only check cannot detect a missing module global such as
  `math`; retain AST/symbol-table checks and invoke production scientific paths.
- Synthetic coverage is meaningful only when it uses the production functions.
  V5 hashes those callable sources and records every required path combination.
- A solver-call count alone cannot prove staged access. V5 records solver
  attempts, completed materializations, and checkpoint/label reads separately
  for candidate, method, calibration and assessment roles.
- A failure bundle needs durable facts that survive cache deletion. Each solve
  phase therefore seals numerical records and a checkpoint hash inventory
  before the shared runtime performs its unconditional managed-cache cleanup.
- Sidecar byte identity must be tested after Git checkout on Windows, not merely
  after local writing. The detached execution checkout is the required fresh
  worktree test.
- [tool] Shared `ExperimentRuntime` creates a missing result parent/root and
  owns all lock, atomic-pair, cache-cleanup, failure, and terminal behavior.
  Production callbacks must only use `RunContext` for durable result writes.
- [self] `compileall` reports missing paths without a nonzero exit when invoked
  from the repository root with modern-relative paths omitted. Run it from
  `modern/` (or prefix every target with `modern/`) and inspect its output.
- [self] `compileall -f` rewrites tracked legacy `__pycache__` files even with
  `PYTHONDONTWRITEBYTECODE=1`. After compilation, restore those exact generated
  v1 paths and verify status contains only the owned v5 experiment/test trees.
