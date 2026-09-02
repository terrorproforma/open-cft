# Devlog

## 2026-09-02 — v6 preregistration implementation

- Created an isolated worktree from patched foundation `b46e263` and merged the
  immutable v1-v5 history, including v5 result `d249a0b`.
- V5 correctly reached its first scientific 5-versus-6 cusp rejection, then
  failed because topology data had already been wrapped in reserved
  `__cft_type__` envelopes before `RunContext.write_json`.
- Added one plain-domain callback writer, explicit boundary-null and map
  assessment serializers, and static rejection of pre-canonicalized payloads.
- Added a production callback matrix covering resolved, ambiguous zero-cell and
  zero-orbit, nonempty boundary-null, and full assessment-rejection payloads.
- Numerical rejections now atomically write topology and outcome checkpoints;
  failed cases are not overwritten and the assessment returns `Decision(False)`.
- Kept the v1.2 policy, production three-map roundtrip, Boris orbit adapter,
  coupling v4.2 scientific thresholds, replay, freshness and promotion gates.
- Declared eight fresh five-stage cases over 5.6/7.0 mm pitch and
  10.7/11.1 mm radius. All retain six required cusps and exclude v1-v5 access.
- Uses runtime `b46e263` globally sorted inventory, distinct
  `preflight/production-fields/` paths, and result-local `* -text`.
- V6 suite passed 10/10, including a patched-runtime integration test that
  finalized a five-of-six cusp outcome as a valid `assessment_rejection`.
- Focused runtime/coupling/field/v6 verification passed 304 tests with one
  Windows symlink privilege skip. Full Python compileall passed.
- Native CMake/CTest passed 1/1. Foundation package/spec/pyproject and FYP
  diffs remain empty.
- Sole clean-detached execution remains pending.
