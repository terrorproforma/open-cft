# Devlog

## 2026-09-02 — v7 prospective orbit preregistration

- Created the isolated v7 worktree from runtime `b46e263` and merged immutable
  v1-v6 history through rejected v6 result `876c3a9d`.
- Disclosed v6 only as algorithm-development evidence; v7 thresholds and
  samples are prospective and frozen from manufactured/prior development data.
- Declared eight new five-stage cases over 5.8/7.2 mm pitch and 11.5/11.9 mm
  radius; all v1-v6 IDs, coordinates, and families are excluded.
- Replaced component-wise B interpolation with a shared ψ-gradient sampler,
  regular-axis limit, and verified within-cell divergence identity.
- Added exact-time nested N/2N/4N midpoint-Boris integration with timestep
  scale set by the maximum declared or path-encountered cyclotron frequency.
- Split numerical convergence from physical μ/rho/LB/curvature adiabaticity;
  raw instantaneous and gyro-averaged μ are preserved on failed gates.
- Added phase-aligned guiding-centre trajectory/state errors, energy drift,
  mirror detection, cross-map convergence, and sample-level reason aggregates.
- Added uniform, gradient, and mirror manufactured preflight; exact-time and
  mirror tests pass without held-out access.
- Distinguished uncertainty-not-evaluable from invalid uncertainty bounds and
  report axial-core failures separately.
- Removed redundant post-publication bundle validation; all remaining
  validation paths approve the result-local `.gitattributes` placeholder.
- V7 plus targeted coupling/field tests passed 72/72; full compileall and
  native CMake/CTest passed 1/1.
- Broader runtime/field/coupling verification passed 346 tests with one
  Windows symlink skip. Two unrelated checked-out example hash tests fail
  because host `core.autocrlf=true` rewrites files whose sidecars bind LF bytes;
  accepted example files were not modified.
- Foundation package/spec/pyproject and FYP diffs are empty.
- Detached one-shot execution remains pending after preregistration push.
