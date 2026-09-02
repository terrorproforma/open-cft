# Devlog

## 2026-09-02 — v5 preregistration implementation

- Created isolated branch/worktree from foundation `231873d`; merged immutable
  v1-v4 experiment history, including v4 terminal failure `64fcafe`, without
  changing the main workspace, accepted packages/specs, or FYP.
- Diagnosed v4's sole execution failure: its experiment-local `map_policy()`
  inherited the shared v1.1 `current_artifact_schema` default.
- Made all eleven `MapValidationPolicy` fields explicit for current field v1.2,
  L1a only, no migrations, exact sample/axis tolerances, and one-hour freshness.
- Added static/runtime preflight rejection for implicit policy construction or
  any v5 legacy field reload.
- Added a pre-held-out manufactured solve using production `solve_map`, exact
  canonical v1.2 bytes, shared-runtime atomic storage, strict reload,
  `CanonicalFieldV12Adapter`, three-map v4 set, and coupling v4.2 record.
- Declared a fresh eight-case 5/9-stage family at 5.4/6.9 mm pitch and 10.3 mm
  chamber radius. It explicitly excludes every disclosed v1-v4 access,
  including `wcval-v4-s04-p0-r0-neg`.
- Preserved the shared runtime lifecycle, sequential assessment, map-based
  Boris orbit checks, replay, freshness, three-map, path, uncertainty, source,
  opaque projection, and all-case promotion gates.
- Focused runtime/coupling/field/v5 verification passed 298 tests with one
  Windows symlink privilege skip; the v5 suite passed 8/8, including the actual
  three-map production preflight on CUDA.
- Full Python source/test/v5 compileall passed. Native CMake/CTest passed 1/1.
  Foundation package/spec/pyproject and FYP diffs remain empty.
- Thresholds remain unchanged from v4 and no prior held-out result was a tuning
  input. Sole clean-detached execution remains pending.
