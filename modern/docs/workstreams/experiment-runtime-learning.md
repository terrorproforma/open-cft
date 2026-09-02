# Experiment runtime learning scratchpad

File policy: `COMMITTED`. This workstream-local file is used because the task
explicitly forbids edits outside new `experiment-runtime-*` documentation and
the new runtime source/spec/test trees.

## Retained guardrails

- [user] Own only the new experiment-runtime paths. Existing experiments,
  shared exports, FYP, and Git state are concurrent/read-only.
- [user] Do not install dependencies, commit, or run real held-out experiments.
- [user] A one-attempt experiment must fail closed on serializer, output,
  lock, cache, and finalization failures.
- [self] Hash exactly the canonical persisted bytes. A separate pretty
  serializer creates an avoidable identity split.
- [self] Do not glob `*.json` without excluding `.sha256.json`; sidecars are
  JSON too and otherwise corrupt sequence validation.
- [self] A one-byte writability probe cannot exercise short-write handling.
  Probe data must be long enough for an injected short write to occur.
- [self] Do not canonicalize already-tagged parsed JSON through the Python
  typed-value encoder. Verify its strict sorted compact JSON bytes directly,
  because `__cft_type__` is intentionally reserved for encoder output.
- [self] A cache that failed validation must not be deleted as if it were owned.
  Cleanup only after successful cache preparation.

## 2026-09-02 session

- Task: build the reusable fail-closed experiment lifecycle before the next
  L1a surrogate or wall-cusp attempt.
- [tool] Windows denied creation of a directory symlink in the test process.
  Keep the real symlink test conditional and always run injected
  reparse/unknown-kind coverage.
- [self] Initial transition replay included JSON sidecars and saw duplicate or
  missing sequence numbers. The validator now filters sidecar metadata and
  independently inventories it.
- [self] Initial fake backends made the same sidecar-glob mistake. Their
  assertions now inspect only primary access records.
- [self] Cleanup-only failure originally needed an explicit state transition.
  Runtime and schema now permit an accepted assessment to become
  `runtime_failure` when final cleanup fails.
- [self] Check `Enum` before `int` in the typed encoder because `IntEnum`
  subclasses `int`; the opposite order silently loses the enum type.
- [self] Cleanup can fail after a valid development or assessment rejection,
  not only after acceptance. Those rejection-to-runtime-failure edges must
  remain in both executable and machine-readable transition contracts.
- Worked: injectable OS boundaries exercised create/write/short-write/disk-full/
  fsync/replace/delete failures without monkeypatching experiment logic.
- Worked: unique atomic event/counter/access artifacts avoid mutable JSONL tails
  and make every prefix independently diagnosable.
- Follow-up: successor experiments must supply real clean/detached Git checks,
  protocol-specific decisions, and actual solver/label backends; the runtime
  deliberately does not infer those claims.

## 2026-09-02 independent-audit closure

- [self] Pinning a directory handle is insufficient if recursive enumeration
  still calls `Path.iterdir`; an attacker can redirect what is inventoried.
  Inventory and cache cleanup now enumerate through `NtQueryDirectoryFile` on
  Windows and `scandir(dirfd)` on POSIX.
- [self] Windows directory handles need `FILE_LIST_DIRECTORY`, not only
  `FILE_READ_ATTRIBUTES`, for handle-relative enumeration.
- [self] A `BaseException` intentionally bypasses terminal finalization. The
  outer handle/cache `finally` therefore must begin immediately after lock
  acquisition and enclose all post-lock work, not only the ordinary exception
  path.
- [self] Existing-lock classification must validate all exact keys, tagged UTC
  form, exact booleans, producer path, canonical bytes, and schema version
  before comparing attempt identity. Otherwise malformed same-looking locks
  are misclassified.
- [self] Candidate validation needs deny-write file seals through manifest
  publication. Semantic validation happens only against an invisible override;
  no post-publication semantic check can turn a published completion marker
  into a newly rejected attempt.
- [self] Windows uses `NtCreateFile(FILE_WRITE_THROUGH)`, explicit file flush,
  and handle-relative rename. Ordinary directory handles do not provide a
  POSIX-equivalent metadata fsync on this host, so the manifest records that
  bounded limitation.
- [tool] Actual junction creation and inner-junction inventory attacks worked
  without elevated symlink privilege and were rejected as reparse points.
- [tool] Both focused pytest import modes reached the same preserved baseline
  before final audit expansion; final counts are recorded in the devlog.

## 2026-09-02 final local QA

- [self] Listing every sidecar is not the same as proving an artifact/sidecar
  bijection. Validate declared references first, reject duplicate targets,
  then require filename-to-contract equality and exactly one sidecar for every
  ordinary artifact.
- [self] A sidecar whose basename is itself a sidecar must be rejected before
  generic orphan handling; otherwise the diagnostic hides the prohibited
  recursive metadata shape.
- [self] Windows directory `FlushFileBuffers` commonly reports either
  `ERROR_INVALID_HANDLE` or `ERROR_ACCESS_DENIED`. Both are tolerated only as
  named bounded limitations; neither supports a POSIX-equivalent durability
  claim.
- [tool] Parallel full-suite execution exposed a legitimate preflight race:
  another process's probe can disappear between handle enumeration and open.
  Normalize that transient read failure to `FilesystemSafetyError`; never let
  it escape as an unclassified OS error or continue toward lock acquisition.

## 2026-09-02 inventory ordering closure

- [self] Recursive traversal order is not canonical lexical path order. In a
  same-stem layout, `x.json` sorts before `x/nested.json` even though a
  depth-first walk visits the `x/` subtree first.
- [self] Normalize raw inventory exactly once before constructing entries:
  reject duplicate paths, then globally sort by path. Candidate construction
  and replay validation must call that same function so their ordering cannot
  drift.
- [self] A duplicate-path manifest and a unique-but-unsorted manifest are
  different contract defects and need separate diagnostics. Filesystem
  traversal itself may be unsorted and should be canonicalized, not rejected.
