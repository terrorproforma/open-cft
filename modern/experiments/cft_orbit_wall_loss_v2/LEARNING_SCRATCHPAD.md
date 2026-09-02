# Learning scratchpad

## File policy

- `COMMITTED`: this experiment-local scratchpad is part of preregistration.

## Preflight guardrails

- [user] Use only the divergent-exit design with committed
  `NUMERICAL_P2_QUALIFIED` evidence.
- [user] Preserve main, accepted packages and FYP; isolate every change under
  this experiment directory and a new worktree.
- [user] Run one pushed, clean, detached preregistration exactly once through
  shared `experiment_runtime`; retain a Git-common lock and never patch/rerun.
- [user] Freeze launch strata and equal weights before outcomes; process
  sequential batches and bind the launch and batch hashes externally.
- [user] Treat coupling as export-only and make no PIC, self-consistent plasma,
  experiment, hardware, mirror-formula or total-performance claim.
- [self] The accepted orbit event geometry is cylindrical. Restrict physical
  wall-hit authority to the straight dielectric section and classify radial
  exit in the divergent section as plasma-subdomain escape.
- [self] Build regular ψ grids from bound P2 `A_phi`, derive B consistently
  through `PsiBicubicField`, and reject any sample whose owning triangle is not
  a declared homogeneous plasma region.
- [tool] PowerShell uses `;` for sequential commands; do not rely on `&&`.

## Session entry — 2026-09-02

- Task: preregister and execute audit-corrected CFT full-orbit wall-loss v2.
- Working pattern: verify exact manifest/checkpoint/sidecar Git and SHA-256
  authorities before P2 sampling; run manufactured orbit/CUDA gates before
  outcome access.
- Risk checkpoint: accepted `orbit_mc` supports a constant cylindrical wall,
  not the 18–24 mm sloped exit wall. The preregistration states this limitation
  explicitly rather than misclassifying a 2 mm crossing as physical wall loss.
- [tool] Orbit launch seeds are unsigned 64-bit values while shared runtime JSON
  accepts signed 64-bit integers. The single v2 payload builder encodes seeds
  as decimal strings, preserves tuple positions for canonical tagging, and
  execution compares exact preregistered bytes before typed reconstruction.
- [self] Never compare a raw tuple-bearing record to the parsed canonical JSON
  tag mapping. Never pass parsed tagged mappings back to `canonical_bytes`;
  reserved tags deliberately make that a rejected double encoding.
- [user] V1 accessed zero P2 fields and zero outcomes. Reuse its launch grid
  only with explicit disclosure and fresh v2 campaign/launch/seed identities.
- Outcome status: pending the clean detached single execution.
