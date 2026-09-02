# Successor experiment integration

New experiments should import the package directly; do not copy lifecycle code
into an experiment. Keep protocol-specific thresholds, solver construction, and
result interpretation in the successor experiment.

## Common launcher shape

```python
from pathlib import Path

from cft_revival.experiment_runtime import (
    Decision,
    ExecutionAttestation,
    ExperimentRuntime,
    RootPolicy,
    RuntimeCallbacks,
)


def run_successor() -> str:
    runtime = ExperimentRuntime(
        experiment_id=PROTOCOL["experiment_id"],
        result_root=EXPERIMENT_DIR / "results",
        cache_root=EXPERIMENT_DIR / ".working-cache",
        attestation=ExecutionAttestation(
            attempt=1,
            commit=verified_detached_head(),
            command="python -m experiments.successor.run",
            host=verified_host(),
            device=PROTOCOL["solver"]["device"],
            clean_worktree=verified_clean_worktree(),
        ),
        producer=run_successor,
        source_root=MODERN_ROOT,
        root_policy=RootPolicy(
            approved_placeholders={"README.md": b"Reserved for one run.\n"},
            allow_empty_existing=True,
        ),
    )
    outcome = runtime.run(
        RuntimeCallbacks(prebundle, development, assessment)
    )
    return outcome.state.value
```

`verified_clean_worktree()` must perform the real Git check and return false on
any tracked or untracked change. The runtime validates the attestation; it does
not invent it. Keep the cache outside `results`. Do not catch
`ExistingLockError` to retry.

Every solver or label call follows this order:

```python
context.before_expensive(
    case.case_id,
    kind="solver",  # or "label"
    details={"role": role, "partition": partition_id},
)
value = backend(case)
context.write_json(f"records/{case.case_id}-{role}.json", value)
```

The access call must be immediately before the expensive call. If a backend
returns stdout/stderr, persist the actual bytes with `write_transcript`; never
put an unbound transcript digest in a JSON claim.

## L1a-surrogate v5 successor

Use these callback boundaries:

- `prebundle`: copy the preregistration with `write_json`, record the complete
  dependency/blob inventory, validate all role partitions, prove development
  and assessment IDs disjoint, run the serializer matrix, and record zero
  solver/label access. Return only after all pre-run checks pass.
- `development`: before each low/high solver or development label access, call
  `before_expensive`. Fit candidate models, select the method using
  development-only rows, calibrate uncertainty using its declared development
  partition, and atomically persist model/freeze records. Return
  `Decision(False, ...)` when method-selection or calibration gates fail; this
  produces `development_rejection` and makes assessment unreachable.
- `assessment`: first call `before_expensive(..., kind="label")` for each
  single-use held-out label batch. Evaluate the already-frozen model and gates;
  do not refit, recalibrate, or modify thresholds. Return `Decision(False, ...)`
  for a valid negative result or `Decision(True, ...)` only when every
  preregistered gate passes.

Replace v1-style mutable `write_json` helpers, ad-hoc `.sha256` text files,
experiment-local lock code, and success-only validators with the runtime store.
Representatives, numerical records, models, and reports are ordinary
sidecar-bound artifacts. Keep every accessed assessment label blob or canonical
record in the bundle.

Recommended call-site assertions:

- `terminal.json.counts.assessment_access_count == 0` for development
  rejection;
- `label_access_count` equals the number of label-kind access records;
- the selected-model and calibration hashes appear as ordinary inventory links,
  not as claims to absent files;
- the returned state, not file presence such as `final-assessment.json`, drives
  the process exit/report text.

## Wall-cusp validation successor

Use the runtime around the next held-out attempt rather than finalizing an
external worker into a mutable lock.

- `prebundle`: persist protocol, exact Git dependency closure, accepted coupling
  identity, runtime/device evidence, held-out/development disjointness proof,
  and manufactured serializer/orbit preflight. This phase must not access any
  held-out map.
- `development`: run only manufactured or frozen development checks. Persist
  the acceptance-policy freeze and return development rejection if these checks
  do not support an assessment attempt.
- `assessment`: before each case and before each primary/refined/enlarged solve,
  write access/counter records. Persist map artifacts immediately, then
  prerecord, replay, topology, orbit, and projection evidence as each becomes
  available. A numerical criterion miss is `assessment_rejection`; an exception,
  device error, malformed backend result, or lifecycle I/O error is
  `runtime_failure`.

The callback should derive candidate/resolved map, cusp, cell, path, orbit, and
projection counts from available phase records for every state. Never construct
held-out outcomes only inside a success branch and then assume they exist in
failure reporting. Capture actual worker/backend stdout and stderr bytes if
subprocesses are retained.

For both successors, a process crash can leave a lock, sidecar pair, or
temporary file without a manifest. That is an immutable incomplete attempt, not
permission to patch or rerun. Use `diagnose_bundle` for the incident record and
preserve the directory.

Successor launchers must treat only a replay-valid `manifest.json` as complete.
The runtime validates a private candidate before publication, so a schema,
counter, access, inventory, identity, or semantic denial leaves no manifest.
On Windows the manifest records `ERROR_INVALID_HANDLE` and
`ERROR_ACCESS_DENIED` as the two tolerated bounded directory
`FlushFileBuffers` outcomes. Neither is POSIX-equivalent directory metadata
fsync, and this limitation must not be rewritten as a full power-loss
durability claim in experiment reports.
