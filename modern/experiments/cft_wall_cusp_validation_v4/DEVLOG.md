# Devlog

## 2026-09-02 — v4 preregistration implementation

- Based the isolated branch on foundation commit `231873d` and merged the
  immutable v3 failure history without changing accepted packages, specs, FYP,
  or the main workspace.
- Declared an eight-case family disjoint from development and all accessed
  v1-v3 cases and coordinates.
- Integrated coupling record v4.2 with direct canonical field artifact v1.2
  verification and complete primary/refined/enlarged evidence fingerprints.
- Replaced experiment-local lifecycle code with shared runtime callbacks for
  roots, lock, cache, sidecar-bound atomic artifacts, counters, access records,
  terminal state, and terminal bundle validation.
- Added map-based Boris orbit checks for magnetic moment, energy, pitch, mirror
  state, and timestep convergence. Kept path-length convergence independent.
- Added synthetic serializer coverage for signed zero, subnormals, all public
  coupling diagnostic dataclasses, manufactured records, and orbit diagnostics.
- Froze all development thresholds before preregistration; prior held-out
  outcomes are explicitly excluded from tuning.
- Focused runtime/coupling/field/v4 verification passed 296 tests with one
  Windows symlink privilege skip; the v4 suite passed 6/6.
- Full repository collection required import isolation, then reached 1216
  passes, 4 skips, 31 failures and 37 setup errors in unrelated pre-existing
  result/visualization artifacts. No v4 test failed.
- Full source/test/experiment compileall passed. Native CMake/CTest passed 1/1.
  FYP and all foundation package/spec diffs remain empty. Ruff and mypy are not
  installed in the no-install environment.
- Actual pre-access device closure passed on RTX 5090, compute capability 12.0,
  Warp 1.14.0.
- Sole clean-detached execution remains pending.
