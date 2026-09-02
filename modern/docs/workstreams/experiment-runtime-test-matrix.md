# Experiment runtime failure matrix

The matrix uses only temporary directories, injected filesystem adapters, a
fake solver, and a fake label backend. It never invokes a production solver or
reads held-out evidence.

## Serialization and identity

- Repeatability, key ordering, exact persisted/hash bytes, UTF-8, aware UTC
  normalization, relative path tagging, bytes/base64, enum, and dataclass tags.
- Rejection of naive time, absolute/traversing paths, non-string keys, reserved
  tag collisions, unsupported sets, NaN, infinity, duplicate JSON keys, and
  non-finite JSON constants.
- Stable `relative-file:qualname` identity and rejection of local/lambda
  producers.
- Tuple/list separation, drive-relative Windows paths, mixed separators, exact
  booleans, bounded counters, and closed nested type tags.
- LF and CRLF JSON checkout forms canonicalize identically. LF and CRLF
  transcript blobs remain intentionally byte-distinct and both bytes are
  included and hashed.

## Root preflight

- Nested missing parents, existing empty root, and an exact approved placeholder.
- Target occupied by file or directory link. The Windows symlink test skips
  only when the host denies link creation; injected reparse and unknown kinds
  always run.
- Existing terminal manifest, execution lock, stale atomic temporary, partial
  sidecar, changed placeholder, and unknown content.
- Injected parent creation denial, exclusive-create denial, write error,
  zero-progress write, short write, disk full, file fsync, target-probe delete,
  and directory fsync.
- Successful short writes are completed in a loop; successful preflight leaves
  no probe file.
- Actual Windows junction rejection, pinned-root move detection, injected
  handle identity change, a visible-path swap during an artifact write, and
  handle-only recursion proven by forbidding `Path.iterdir`.

## Lock and atomic persistence

- Same-attempt, different-attempt, and malformed existing locks all block.
- Thread and spawned-process concurrent `O_EXCL` acquisitions produce exactly
  one winner.
- Two spawned complete runtimes targeting the same result root produce one
  accepted bundle and one blocked attempt.
- Full existing-lock schema validation precedes same-attempt classification;
  missing, extra, wrong-version, and bool-as-int fields classify as malformed.
- Lock fsync failure retains the lock and blocks a later acquisition.
- Exact canonical artifact bytes plus valid semantic/byte sidecar.
- Stale fixed temporary detection.
- Data/sidecar replace interruption and directory-fsync interruption produce
  diagnosable partial pairs and/or stale temporaries, never a manifest.
- Injected create, write, zero-write, disk-full, and file-fsync failures cannot
  produce false completion.
- Post-finalization mutation invalidates sidecars/inventory.
- A spawned process exiting immediately after lock acquisition leaves a lock
  and no manifest.

## Cache lifecycle

- Absent cache creation, empty cache initialization, valid populated cache
  reuse, normal cleanup, unmarked populated rejection, malformed marker
  rejection, and cleanup deletion denial.
- Cleanup denial with an existing experiment exception retains that exception
  as primary and writes a separate secondary failure artifact.
- Cleanup denial as the only failure becomes a primary `runtime_failure`.
- Result/cache identity and canonical-path overlap is rejected in both ancestor
  directions; the cleanup protected-root guard is independently injected.

## Phase and terminal lifecycle

- Valid bundles for `prebundle_failure`, `runtime_failure`,
  `development_rejection`, `assessment_rejection`, and `accepted_result`.
- Counter snapshot, access record, and phase-start transition exist before the
  fake solver or fake label backend runs.
- Assessment access remains zero after prebundle/development rejection.
- Label counters match label-kind access artifacts.
- Transition sequences, predecessor/state agreement, monotonic counters, lock
  attestation, artifact inventory, and transcript bytes are replay-validated.
- A simulated hard crash after an artifact write preserves the lock, removes
  the managed cache through the outer `finally`, and leaves no manifest.
- Manifest replace failure leaves a stale manifest temporary and no false
  completion.
- A complete attempt cannot be reused as a second attempt.
- Candidate-manifest semantic/counter denial, empty/unknown/reparse inventory
  entries, actual inner junctions, injected special entries, malformed nested
  access types, and tampered closed schemas all leave no manifest.
- Full lifecycle orphan-sidecar, duplicate-reference, mismatched-contract, and
  sidecar-of-sidecar regressions prove artifact/sidecar bijection before
  manifest publication; replay validation independently rejects a newly added
  orphan.
- A full accepted lifecycle combines `x/...` with sibling `x.json` and proves
  the candidate and replay inventory are globally path-sorted. Malformed
  unsorted and duplicate-path manifests receive distinct contract diagnostics;
  duplicate raw inventory paths leave no manifest.
- Candidate validation observes no visible manifest, holds existing files
  deny-write, and is the sole semantic validation call; no post-publication
  rejection path exists.
- Every required directory is explicitly bound in the manifest, and local
  `.gitattributes` resolve owned Python/JSON files to LF on fresh checkout.
- The production lifecycle itself injects missing-parent, probe create/write/
  zero-write/disk-full/file-flush/delete, lock flush, temporary create/write/
  disk-full/flush, publication, directory-flush, and short-write behavior.
- Windows durability metadata tests require both `ERROR_INVALID_HANDLE` and
  `ERROR_ACCESS_DENIED` to be named as tolerated bounded directory-flush
  limitations alongside the non-POSIX durability claim.

## Commands

Run from `modern`:

```text
python -m pytest tests/experiment_runtime -q
python -m pytest tests/experiment_runtime -q --import-mode=importlib
python -m pytest tests/experiment_runtime/test_audit_hardening.py -q
python -m compileall -q src/cft_revival/experiment_runtime tests/experiment_runtime
```

Repository safety checks run from the repository root:

```text
git diff --check
git diff --exit-code -- FYP
git status --short -- FYP
```

The dated devlog records the observed outcomes rather than predicted results.
