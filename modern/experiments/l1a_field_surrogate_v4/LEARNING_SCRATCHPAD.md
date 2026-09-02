# L1a field-surrogate v4 learning scratchpad

## Pre-registration observations

- A compile-only check cannot detect a missing module global such as `math`.
  V4 combines AST parsing, symbol-table global resolution and real invocation of
  the affected production paths.
- Synthetic coverage is meaningful only when it uses the production functions.
  V4 hashes those callable sources and records every required path combination.
- A solver-call count alone cannot prove staged access. V4 records solver
  attempts, completed materializations, and checkpoint/label reads separately
  for candidate, method, calibration and assessment roles.
- A failure bundle needs durable facts that survive cache deletion. Each solve
  phase therefore seals numerical records and a checkpoint hash inventory
  before the cache is removed in `finally`.
- Sidecar byte identity must be tested after Git checkout on Windows, not merely
  after local writing. The detached execution checkout is the required fresh
  worktree test.
