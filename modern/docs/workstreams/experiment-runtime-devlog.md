# Experiment runtime devlog

## 2026-09-02 10:59 UTC+10 — reusable fail-closed lifecycle

- Added `cft_revival.experiment_runtime` with one strict typed canonical
  serializer for persisted/hash bytes, stable producer IDs, safe root
  preflight, immutable exclusive lock, atomic artifact pairs, managed cache,
  phase/counter/access records, five terminal bundle states, full inventory,
  last-written completion manifest, diagnostics, and replay validation.
- Added machine-readable execution-lock/manifest schemas and an exact runtime
  state/event transition ledger under `modern/spec/experiment_runtime`.
- Added successor integration guidance for L1a-surrogate v5 and wall-cusp
  validation, plus architecture, failure-matrix, and learning-loop records.
- Added injected coverage for missing/occupied/reparse/unknown paths;
  permission/create/write/zero-write/short-write/disk-full/replace/fsync/delete
  failures; same/different/malformed and concurrent locks; cache states and
  cleanup failures; hard-crash/finalization points; LF/CRLF; all terminal states;
  fake solver/label access ordering; inventory, sidecar, and tamper rejection.

Validation:

- `python -m pytest tests/experiment_runtime -q`: 58 passed, 1 skipped in
  7.06 s. The skipped real directory-symlink case requires Windows privileges;
  injected reparse and unknown-entry tests passed.
- `python -m compileall -q src/cft_revival/experiment_runtime
  tests/experiment_runtime`: passed.
- Repository-wide default pytest mode: collection blocked before execution by
  pre-existing duplicate module basenames in experiment test directories
  (`test_protocol.py`, `test_preregistration.py`, and others).
- Repository-wide importlib mode with fail-fast: 329 passed, 1 skipped, then a
  pre-existing `l0_surrogate_v3` test failed because its real results path
  already exists. The owned runtime suite had already passed in that run.
- `git diff --check`: passed; Git emitted pre-existing LF/CRLF warnings for
  tracked axisymmetric example results.
- `git diff --exit-code -- FYP` and `git status --short -- FYP`: passed/clean.
- Owned-source/test trailing-whitespace and 100-column scans: clean.
- Ruff and mypy were unavailable (`No module named ruff` / `mypy`) and were not
  installed.

No dependency was installed, no commit or Git metadata was changed, no FYP or
existing experiment/shared export was edited, and no real solver or held-out
label backend was run.

Follow-up:

- L1a-surrogate v5 and the next wall-cusp validation must provide real
  clean/detached Git attestation and wrap every production solver/label access
  with `RunContext.before_expensive`.
- Preserve any lock-only, partial-pair, stale-temp, or no-manifest attempt as
  immutable failure evidence; do not patch or rerun it.

## 2026-09-02 — independent-audit hardening

- Replaced mutable-path recursive enumeration with
  `NtQueryDirectoryFile`/`scandir(dirfd)` and used pinned parent/root identities
  for creation, reads, inventory, publication, and cache deletion.
- Added Windows `NtCreateFile` relative opens with no reparse following,
  `FILE_WRITE_THROUGH`, explicit file flush, handle-relative publication, and
  final-path plus volume/file-ID checks. The manifest records that ordinary
  Windows directory handles do not provide POSIX-equivalent metadata fsync.
- Made manifest finalization two-stage: private canonical candidate, complete
  production semantic validation while files are deny-write sealed, then sole
  completion-marker publication with no post-publication semantic validator.
- Added closed production contracts for lock, terminal, manifest, sidecar,
  transition, counter, access, decision, tagged values, exact booleans,
  bounded integers, UTC microseconds, hashes, and nested keys.
- Added recursive file/directory inventory, undeclared/empty-directory denial,
  actual junction/reparse and injected special-entry rejection, bidirectional
  cache/result alias checks, tuple/list separation, reserved Windows path
  rejection, and unstable-producer rejection.
- Moved the post-lock runtime into an immediate outer `try/finally`, including
  `BaseException` paths, so simulated hard crashes release pinned handles while
  retaining the immutable lock and incomplete evidence.
- Expanded tests through production lifecycle operations and spawned complete
  runtimes, including parent/probe/lock/temp/publish/flush/delete failures,
  process races, process death, candidate denials, exact malformed-lock
  classification, handle-only enumeration, and fresh-checkout attributes.

Final validation:

- `python -m pytest tests/experiment_runtime -q`: 123 passed, 1 skipped in
  25.45 s.
- `python -m pytest tests/experiment_runtime -q --import-mode=importlib`:
  123 passed, 1 skipped in 25.44 s.
- The single skip is the privilege-dependent directory-symlink test. Actual
  Windows junction root and inner-inventory attack tests both passed.
- Spawned process subset: 3 passed (exclusive lock race, full-runtime race,
  and crash-after-lock).
- `python -m compileall -q src/cft_revival/experiment_runtime
  tests/experiment_runtime`: passed.
- Runtime spec tests: 2 passed. The optional `jsonschema` package is not
  installed, so no dependency was added; production validation is implemented
  directly and the specs were strict-JSON parsed/contract-compared.
- Local Git attribute checks report `eol: lf` for owned source, tests, and
  specs. `git diff --exit-code -- FYP` and `git status --short -- FYP` passed.
- Owned Python trailing-whitespace and 100-column scans passed.

Successor readiness: pass for lifecycle integration. L1a-surrogate v5 and the
wall-cusp successor still must supply their real clean Git/device attestation,
protocol decisions, and solver/label callbacks. Windows power-loss durability
remains explicitly bounded to write-through temporary creation, file flush,
and handle-relative rename without a POSIX-equivalent directory fsync claim.

## 2026-09-02 — final local QA closure

- Enforced a bijection between ordinary artifacts and sidecars before candidate
  manifest creation and during replay validation. Orphans, missing pairs,
  duplicate declared references, filename/contract mismatches, malformed
  contracts, and sidecar-of-sidecar names now fail closed.
- Added a full lifecycle `orphan.json.sha256.json` regression proving the lock
  remains and no manifest becomes visible, plus duplicate, mismatch,
  sidecar-of-sidecar, and post-publication replay regressions.
- Windows durability metadata now names both tolerated directory
  `FlushFileBuffers` outcomes: `ERROR_INVALID_HANDLE` and
  `ERROR_ACCESS_DENIED`. They remain bounded limitations and never imply
  POSIX-equivalent directory durability.
- A parallel verification run exposed a transient process race where one
  preflight observed another process's disappearing write probe. Placeholder
  reads now normalize that race to `FilesystemSafetyError`, preserving the
  intended fail-closed behavior.

Final QA:

- Default focused pytest: 128 passed, 1 privilege-dependent symlink skip.
- Importlib focused pytest: 128 passed, 1 privilege-dependent symlink skip.
- Process subset: 3 passed; specs: 2 passed; compileall: passed.
- LF attribute and FYP checks passed; no install, commit, push, real solver, or
  held-out access occurred.
