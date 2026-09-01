# L1a field-surrogate v1 devlog

## Preregistration

- Created only new experiment and test paths in an isolated detached worktree.
- Reused accepted geometry v1.1, L1a field, AR1/GP/POD, sampling, and
  validation foundations without modifying them.
- Fixed a fresh seed and deterministic reorder so assessment boundary/OOD
  challenges cannot leak into candidate or method-selection roles.
- Preflight is input/identity/model-contract only and accesses no solver labels.

## Execution

- Preregistration commit `6e8b74f2cefc2f4ed4e4745e6e9bc91580a09af7`
  was pushed before execution.
- The exclusive detached RTX 5090 run was claimed once and retained its lock.
- Low and paired high solves proceeded until fresh method-selection geometry
  index 69. Accepted geometry v1.1 rejected that input with
  `GeometryValidationError: divergent wall slopes are not continuous`.
- No patch or rerun was performed. Model fitting, conformal calibration,
  single-use assessment, latency comparison, and representative prediction
  generation were not reached. The prospective experiment is terminally
  rejected under its zero-failure gate.
