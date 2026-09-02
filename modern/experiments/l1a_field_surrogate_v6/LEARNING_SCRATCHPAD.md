# L1a field-surrogate v6 learning scratchpad

## Pre-registration observations

- [user] Preserve all v1-v5 outcomes and modify only the isolated v6
  experiment/test paths.
- [self] V5 attested cleanliness before runtime construction but repeated an
  unconditional global status check inside prebundle. Runtime-created output
  therefore invalidated its own attempt before any solve.
- [user] Verify detached/commit/dependency/remote/global cleanliness exactly
  once before constructing the runtime, then freeze it into an immutable
  `ExecutionAttestation`.
- [user] Any retained runtime drift check must reject all tracked or staged
  changes globally and allow untracked files only below exact runtime-owned
  v6 result/cache roots.
- [self] Status allowlists must compare exact root equality or `root + "/"`;
  prefix-only matching would incorrectly permit sibling names such as
  `results-sibling`.
- [tool] Shared runtime output is path-identity bound. Validate a committed
  result bundle in its original detached execution worktree.
- [self] Avoid `compileall -f` across tracked legacy `__pycache__`; use
  `py_compile` with bytecode disabled or restore only verified generated paths.
