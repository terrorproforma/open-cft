# Post-hoc manifest publication of the single execution (disclosure)

**What happened.** The one preregistered execution (`cef1ee59`, detached worktree
`uni-project-orbit-geo2-run`, `python -m experiments.orbit_wall_loss_geometry_screening_v2.run
execute`, started 2026-09-03 21:03 AEST) ran every phase to completion: prebundle, field
re-solves for 97 designs, 1105 orbit cases (104,832 orbits), allocation replay, gates, dataset,
consumer record. The shared runtime wrote `terminal.json` (state `accepted_result`, payload
status `accepted_screening_dataset`) and the `terminal` transition at 22:29 AEST, then failed in
its own last step, the publication of `manifest.json`:

```
File "...\cft_revival\experiment_runtime\lifecycle.py", line 607, in _finalize_manifest
    sealed = self.store.seal_files(
File "...\cft_revival\experiment_runtime\filesystem.py", line 513, in seal_files
    descriptors.append(self.ops.open_read(parent, path.name))
File "...\cft_revival\experiment_runtime\platformfs.py", line 724, in open_file_read
    descriptor = msvcrt.open_osfhandle(
OSError: [Errno 24] Too many open files
```

`_finalize_manifest` pins every file of the bundle (one read descriptor each) while the
candidate manifest is validated. The Windows C runtime allows 8192 low-level descriptors; this
bundle has 16,957 files (1105 cases x summary / endpoints / orbit sidecar / handoff with their
hash sidecars, 1889 access records, 97 fields and field-evidence files). No bundle of this size
could have been published by the runtime as it was, independently of the campaign's outcome.

**What was NOT affected.** Every artifact was written atomically with its `.sha256.json` sidecar
before the terminal record; the transition log ends at `terminal`; the counters and access
records are complete; `terminal.json` and its sidecar are durable. The stderr of the process is
preserved verbatim in the run log (`%TEMP%\owlgs2_execute.err`, copied to
`%TEMP%\owlgs2_execute_stderr_preserved.txt`); its first part shows only the Warp backend
initialisation of the CPU parity check.

**What was done.** No orbit was re-integrated and no experiment code was changed (the
experiment code hash bound at preregistration is untouched). A fail-closed recovery path was
added to the runtime, `cft_revival.experiment_runtime.recovery.finalize_unpublished_attempt`
(committed together with this file), which:

1. refuses if a manifest exists, if the lock or the terminal record is missing, if the transition
   log does not end at `terminal`, or if the terminal state disagrees with the last transition;
2. rebuilds the inventory with the runtime's own `_inventory` (every file byte-hashed, sidecar
   bijection and every sidecar contract verified; any tampered file raises);
3. hashes the on-disk transition log, terminal record and lock exactly as `_finalize_manifest`
   does, assembles the manifest with the same schema, validates the candidate through the
   runtime's own `validate_bundle(manifest_override=...)` (unpinned) and publishes it.

It was run once, from the branch worktree's code, against the run worktree's results root:

```
python -m cft_revival.experiment_runtime.recovery <run worktree>\modern\experiments\orbit_wall_loss_geometry_screening_v2\results
artifact_count 16968, file_count 16957, transition_count 9, state accepted_result
manifest_byte_sha256 876dc7e1ca76b33d1975a51c7fe749e2e271ab0d42ecdcaf158ecfa31fa0a30c
terminal_byte_sha256 a495d12bc83241e6c2b84623b2e0c75e760176b9c0796854724aea467e195b6a
```

`python -m experiments.orbit_wall_loss_geometry_screening_v2.run validate` (the frozen v2 code
in the run worktree, unchanged runtime validator) then returned `accepted_result` with 16,968
artifacts. The results commit `26029b72` contains only `results/`.

**Runtime hardening.** `ExperimentRuntime._finalize_manifest` now pins at most
`MAX_PINNED_DESCRIPTORS = 4096` files during candidate validation (files beyond the cap stay
unpinned; the validation reads every file's bytes regardless), so future bundles of this size
publish inside the locked attempt. Tests: `modern/tests/experiment_runtime/test_recovery.py`
(simulated EMFILE at publication -> unpublished complete attempt -> recovery publishes a manifest
whose inventory equals an uninterrupted run's; refusals for published / crashed / tampered
bundles; the descriptor cap).

**How to read the bundle.** The manifest's content is what the locked attempt would have
published (same inventory function, same hashes, same root identity); the only difference is
that it was written by the recovery function after the process had died, which this file, the
results-commit message, the README, the campaign devlog and the dashboard disclose. The
paper-side consequence: any admission of this dataset must cite this disclosure and must not
describe the bundle as published inside the locked attempt.
