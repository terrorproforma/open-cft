# Learning scratchpad

## 2026-09-02 — v5 preregistration construction

- `COMMITTED`: this scratchpad is experiment-local evidence.
- [user] Preserve all v1-v4 negative outcomes and exclude every accessed case
  and coordinate; v4 accessed `wcval-v4-s04-p0-r0-neg`.
- [self] V4 passed unit tests but failed its sole run because omitted
  `MapValidationPolicy.current_artifact_schema` silently inherited shared v1.1.
  Future experiment policies must spell every dataclass field and statically
  reject incomplete constructors.
- [self] Testing only a synthetic serializer was insufficient. Before held-out
  access, exercise the exact production solve/policy/adapter path through
  persisted bytes, strict reload, v4 map-set construction, and record creation.
- [user] Direct v1.2 evidence must prove input and normalized schema v1.2, L1a,
  non-migration status, absent migration metadata, canonicalization v2, and
  byte equality; any legacy reload in v5 is forbidden.
- [tool] The latest feature foundation remains `231873d`; preserving v4 requires
  merging remote result commit `64fcafe` into the isolated v5 worktree.
- [self] Stable pre-access geometry probing found 5.4 and 6.9 mm representable
  for both 5- and 9-stage stacks. These coordinates are fresh relative to all
  disclosed v1-v4 accesses and development values.
- [user] Once preregistered, run exactly once from clean detached HEAD and record
  the terminal outcome directly; do not patch or rerun.
