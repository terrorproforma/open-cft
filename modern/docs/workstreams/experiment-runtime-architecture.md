# Fail-closed experiment runtime

`cft_revival.experiment_runtime` is the reusable lifecycle boundary for
preregistered open-cft experiments. It owns result-root safety, one-attempt
locking, typed serialization, phase records, working-cache cleanup, artifact
identity, terminal classification, and replay-independent validation. It does
not own experiment physics or acceptance thresholds.

## Public API

- `ExecutionAttestation` freezes attempt, commit, command, host, device, and
  clean-worktree claims. Invalid or dirty attestations are rejected before any
  lock is acquired.
- `ExperimentRuntime` performs root preflight, acquires the immutable lock,
  starts the outer cleanup `try/finally`, invokes the three lifecycle callbacks,
  writes the terminal record, inventories every byte, and writes the manifest
  last.
- `RuntimeCallbacks(prebundle, development, assessment)` separates operations
  which must occur before development labels, development-only selection, and
  one-use assessment.
- `Decision(accepted, payload)` represents a gate outcome. A false development
  decision never invokes assessment.
- `RunContext.before_expensive(...)` atomically advances counters and writes an
  access record before a solver, label service, or other backend is called.
- `RunContext.write_json` and `write_blob` are the only normal artifact paths.
  `write_transcript` requires the transcript bytes; there is no digest-only
  transcript API.
- `validate_bundle` checks the immutable lock, canonical bytes, every sidecar,
  complete inventory, transition grammar, counters, access order, and terminal
  state without rerunning a solver.
- `diagnose_bundle` identifies missing roots, stale temporary files, partial
  pairs, orphan sidecars, missing locks, and missing completion manifests.

## Canonical value policy

The identity and persistence byte stream is exactly
`cft-typed-canonical-json-v1`: UTF-8, sorted keys, compact separators, no NaN or
infinity, and no trailing newline. Maps require string keys and reserve
`__cft_type__`.

- aware datetimes are normalized to UTC with six fractional digits and `Z`;
  naive datetimes are rejected;
- paths must be normalized, non-empty, relative, traversal-free components and
  are encoded as portable POSIX paths;
- bytes are tagged base64 rather than coerced to text;
- tuples carry a type tag while lists remain JSON arrays, so they cannot hash
  identically;
- enums and dataclasses carry explicit class and type tags;
- drive-relative Windows paths, mixed separators, trailing separators, absolute
  paths, traversal, lambdas, and nested/local producers are rejected;
- unsupported objects and non-finite floats are rejected;
- producer identity is `relative-file:qualname`; lambdas, local functions,
  missing source, and source outside the declared source root are rejected.

The SHA-256 semantic identity is the SHA-256 of those same persisted bytes.
There is no second serializer whose behavior can drift.

## Filesystem contract

Preflight runs before lock acquisition. Every existing ancestor and target must
be a real directory, never a file, symlink, junction/reparse point, or special
entry. The immediate parent and result root remain pinned for the complete
attempt. Their volume/file identity and canonical final path are checked before
and after every mutation.

On Windows, directory handles use `CreateFileW` with
`FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS` and omit
`FILE_SHARE_DELETE`. Descendant directories/files are opened or created
relative to those handles with `NtCreateFile`; publication renames the still
open, no-delete-sharing source handle with `NtSetInformationFile`. Recursive
inventory uses `NtQueryDirectoryFile` on the pinned handle, never
`Path.iterdir`. This remains confined to the pinned directory even when modern
Windows POSIX rename semantics move the visible path; the final-path/identity
check then fails the attempt.

On POSIX, directories use `openat`/dirfd plus `O_NOFOLLOW`; recursive
enumeration uses `scandir(dirfd)`, and files use dirfd operations, `O_TMPFILE`,
`linkat(AT_EMPTY_PATH)`, and directory `fsync`.
Platforms/filesystems lacking the required primitives fail closed.

Missing nested parents are created relative to pinned handles. An existing
result root must be empty or exactly match approved placeholder files and
directories. A durable exclusive probe is created in the pinned target,
written, flushed, removed handle-relatively, and directory-synced where the
platform supports it.

The lock uses `O_CREAT | O_EXCL` and canonical bytes. It is never replaced or
deleted. A same-attempt, different-attempt, or malformed existing lock is
classified but always blocks execution. A failure while writing or syncing the
new lock leaves it in place, preventing an unsafe retry.

Normal artifacts use same-directory temporary handles, full-write loops, file
flush, handle-relative atomic publication, and directory flush. Data and
`<name>.sha256.json` are both prepared before either publication. A crash
between publications creates a detectable partial pair and cannot create a
valid manifest.

Windows temporary files use `NtCreateFile(FILE_WRITE_THROUGH)`, file bytes are
flushed, and handle-relative rename is used. Directory
`FlushFileBuffers` is attempted. `ERROR_ACCESS_DENIED` and
`ERROR_INVALID_HANDLE` are the two explicitly tolerated bounded outcomes for
ordinary directory handles. The manifest records both names and does not claim
POSIX power-loss semantics.
POSIX records file and containing-directory `fsync`.

`manifest.json` has no sidecar because it is itself the sole completion marker.
The complete candidate bytes are built privately, all artifact files are held
deny-write/delete on Windows (read-locked and read-only on POSIX), and
`validate_bundle(..., manifest_override=candidate)` validates schemas,
semantics, inventory, counters, access, transitions, identity, and hashes while
no completion file exists. Only a passing candidate is atomically published.
No semantic validator runs after publication.

Inventory walks every entry recursively and records both required directories
and files. Links, junctions, special entries, stale temporaries, undeclared or
empty directories, unknown files, and partial pairs fail before manifest
publication. The artifact/sidecar graph is bijective: every ordinary artifact
has exactly one valid `<artifact>.sha256.json`, and each sidecar's contract
names that exact existing artifact. Orphans, duplicate references, contract
mismatches, sidecar-of-sidecar names, and missing pairs are rejected both for
private candidates and replayed bundles. The lock, completion manifest, and
approved placeholders are explicit non-sidecar contracts.

## State machine

The first-class terminal bundle states are:

1. `prebundle_failure`: preparation/cache/prebundle failed before valid
   development execution;
2. `runtime_failure`: development, assessment, lifecycle, or cleanup failed;
3. `development_rejection`: a valid prospective run failed development-only
   selection and assessment was not accessed;
4. `assessment_rejection`: development froze successfully and the one-use
   assessment failed its preregistered gates;
5. `accepted_result`: development and assessment both passed.

Every phase start is an atomic transition plus counter/access record written
before the callback. State validation rejects decreasing counters, missing or
duplicate sequence numbers, assessment after development rejection, mismatched
terminal predecessors, and success-only assumptions. The machine-readable
contract is `modern/spec/experiment_runtime/state-machine-v1.json`.

## Working cache and failures

The cache is outside the result root and contains a canonical run marker.
Absent and empty caches are initialized; a correctly marked populated cache can
be reused; unmarked, malformed, wrong-run, linked, or non-directory caches are
rejected. Cleanup is in the outer `finally` immediately following lock
acquisition. If experiment code already failed, its type/message remains the
primary error and cleanup failure is recorded separately. If cleanup is the
only failure, it becomes the primary `runtime_failure`.

Overlap is rejected in both ancestor directions using normalized canonical
paths before writing and pinned volume/file IDs plus final paths after cache
preparation. Secure cleanup rechecks this relationship and deletes only through
the pinned cache hierarchy; it cannot target the result tree.

## Closed record schemas

Production validation—not only tests—requires exact keys, schema and
canonicalization versions, enum values, exact booleans (never integer `0/1`),
bounded signed integers, finite floats, canonical UTC/path/type tags, valid
SHA-256 strings, and exact nested transition/counter/access/terminal/decision
shapes. Existing locks receive full schema and canonical-byte validation before
same/different-attempt classification; any malformed field classifies the lock
as `malformed` and still blocks.
