# L0 surrogate v4 learning scratchpad

Policy: `COMMITTED`, experiment-local.

## 2026-09-02 guardrails

- [user] Preserve v2/v3 immutable failures and keep v4 science unchanged.
- [self] Never type or infer a commit SHA for execution. Resolve HEAD in code,
  prove commit existence and remote ancestry, then record it automatically.
- [self] The preregistration commit is identified by exact subject and
  exact-path isolation; any later v4 path change invalidates binding.
- [self] Unrelated pushed commits may intervene because they cannot alter the
  committed v4 blob tree; unpushed HEAD or dirty/untracked v4 files are rejected.
