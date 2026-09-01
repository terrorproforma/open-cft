# L1a field-surrogate v1 devlog

## Preregistration

- Created only new experiment and test paths in an isolated detached worktree.
- Reused accepted geometry v1.1, L1a field, AR1/GP/POD, sampling, and
  validation foundations without modifying them.
- Fixed a fresh seed and deterministic reorder so assessment boundary/OOD
  challenges cannot leak into candidate or method-selection roles.
- Preflight is input/identity/model-contract only and accesses no solver labels.

## Execution

- Pending the pushed preregistration commit and retained exclusive lock.
