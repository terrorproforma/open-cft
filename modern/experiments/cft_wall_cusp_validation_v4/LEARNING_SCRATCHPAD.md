# Learning scratchpad

## 2026-09-02 — preregistration construction

- `COMMITTED`: this scratchpad is experiment-local evidence.
- The latest foundation branch does not contain v3 because v3 diverged before
  runtime commit `231873d`; preserve both histories with a merge before the v4
  preregistration commit.
- V1 accessed `wcval-f1-s04-p0-r0-neg`; v2 accessed no held-out case; v3
  accessed `wcval-v3-s03-p0-r0-neg` and failed before materializing its first
  canonical field artifact. V4 excludes all IDs and coordinate tuples.
- The v3 artifact failure came from hashing non-authoritative generic JSON
  bytes. V4 uses only field v1.2 canonical file bytes and reloads those exact
  bytes through `CanonicalFieldV12Binding`.
- Keep protocol-specific numerical work in callbacks; shared
  `cft_revival.experiment_runtime` owns every lifecycle and filesystem action.
- A Boris particle trajectory measures magnetic moment, energy, pitch, and
  terminal state from map samples. Path polyline convergence remains a separate
  gate and must not stand in for orbit conservation.
- Do not inspect or tune from v4 held-out outcomes after the preregistration
  commit. One clean detached execution is authoritative even if negative.
- A 6.8 mm pitch failed strict binary64 stage-spacing preflight for eight
  stages. Before preregistration and without field access, the independent
  geometry declaration was corrected to representable 6.7 mm.
- Runtime, coupling, field-v1.2, and v4 tests pass together. The broader
  repository has unrelated stale/missing result-artifact failures; preserve
  them rather than changing accepted or prior-experiment paths.
