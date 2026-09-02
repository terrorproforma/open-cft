# Test-health devlog

Repository-wide test-health work: defects that make the committed suite red on
a fresh checkout without any scientific change. Evidence under `results/`,
frozen preregistration files and `FYP/` are never edited by this workstream;
each entry names the audit that proves what was tolerated or re-bound.

## 2026-09-03 — CRLF-era hash defects and single-invocation collection

Context. The repository was checked out for weeks with `core.autocrlf=true`;
commit `fab0eccc` pinned `* text=auto eol=lf`. Several artifacts had recorded
SHA-256 digests of CRLF working-tree bytes while Git stores LF, so byte checks
failed on every LF checkout. A per-directory sweep of `modern/tests` found the
red directories below; everything else was green. Branch
`fix/test-health-crlf-era`, worktree separate from the main tree.

### 1. `l1a_geometry_sweep_v2` (+ `l1a_field_surrogate_v1`, `l1a_field_surrogate_v2`, v2 visualization)

- Failure: collection error `ValueError: invalid SHA-256 sidecar for
  protocol.json` from `experiments/l1a_geometry_sweep_v2/protocol.py:73`
  (raised at import via `experiment.PROTOCOL = load_protocol()`), which also
  broke collection of the two field-surrogate experiments that import the
  sweep; the v2 dashboard test errored with `protocol file SHA-256 mismatch`.
- Root cause: `protocol.json.sha256` was frozen at preregistration commit
  `092f5fae` on a CRLF checkout and records `64b2c58c…`, the digest of the
  7924 CRLF bytes; Git stores the 7790-byte LF form (`2a5ba9e4…`). The
  immutable bundle repeats `64b2c58c…` as `protocol_file_sha256`. The sealed
  payload digest `da319f22…` (EOL-independent) recomputes; all 34 result files
  are byte-exact. The field-surrogate sidecars are byte-exact.
- Resolution (pattern a+b): `experiments/l1a_geometry_sweep_v2/POSTHOC_AUDIT.md`,
  `audit_sidecar_eol.py` (read-only), `tests/experiments/l1a_geometry_sweep_v2/test_posthoc_audit.py`;
  `protocol.py` gains `EOL_AUDITED_SIDECARS` / `eol_equivalent_digest` and
  `verify_sidecar` accepts, for exactly `protocol.json`, the recorded digest
  iff the LF bytes hash to the audited LF digest and their CRLF transform
  reproduces the recorded one (returning the recorded digest, the identity the
  bundle binds); `visualization/generate_dashboard.py` applies the same rule
  in `_verify_file` (`AUDITED_PROTOCOL_LF_SHA256`). The committed HTML is
  byte-identical (`3c3f5aea…`). Frozen `protocol.json`, its sidecar, the
  experiment `DEVLOG.md`/`LEARNING_SCRATCHPAD.md` (preregistered paths) and
  `results/` are untouched.
- Files: `modern/experiments/l1a_geometry_sweep_v2/{protocol.py,POSTHOC_AUDIT.md,audit_sidecar_eol.py,visualization/generate_dashboard.py}`,
  `modern/tests/experiments/l1a_geometry_sweep_v2/test_posthoc_audit.py`.

### 2. `l1a_geometry_sweep_visualization`

- Failure: 10 errors, `build_payload` could not read
  `experiments/l1a_geometry_sweep/results/manifest.json`.
- Root cause: root `.gitignore` `Results/` matches `results/`
  case-insensitively on Windows; the v1 results were never committed and only
  the producing checkout held them, while the dashboard HTML, generator and
  tests that verify them were committed.
- Resolution: the 38 files (9.6 MB; 19 artifacts + 19 sidecars) were copied
  byte-for-byte from the producing checkout and force-added, as for the v2
  bundle (7.9 MB), `four_cell_topology_search/results/` and the orbit
  wall-loss bundles. All artifacts are byte-exact against their sidecars and
  the generator's pinned digests (`manifest.json` `eb73fc69…`, `dataset.json`
  `2fb0c19d…`); ten CRLF-written sidecars are stored LF by the pin, which
  changes no attested digest. `l1a_plasma_coupling/results/` is referenced by
  no test (its tests use `examples/axisymmetric/results` and `tmp_path`) and
  stays untracked.
- Files: `modern/experiments/l1a_geometry_sweep/results/**` (new),
  `modern/experiments/l1a_geometry_sweep/DEVLOG.md`.

### 3. `material_fields` (`test_spec_ledgers.py`, 4 failures)

- Failure: `MaterialFieldValidationError: raw run hash binding failed`
  (`cft_revival/material_fields/acceptance.py:1081`).
- Root cause: the v1.4 artifacts bind each raw run to
  `implementation_sha256` / `evidence_implementation_sha256`, SHA-256 over the
  bytes of seven `material_fields` source files
  (`numerics._implementation_sha256`). Generated at `8603a905` on a CRLF
  checkout, the recorded digests (`d229f62d…`, `dc988f4b…`, `734cff6a…`) are
  the CRLF hashes; hashing the `8603a905` blobs (identical to `HEAD`) with
  `\n`→`\r\n` reproduces all three exactly. Not a real binding defect.
- Resolution: pattern (a) audit
  (`examples/material_fields/POSTHOC_AUDIT.md`, `audit_implementation_eol.py`,
  `tests/material_fields/test_posthoc_eol_audit.py`, anchored to the
  `8603a905` Git blobs). Pattern (b) is structurally impossible here: the
  comparison sites and the hash function live in the hashed files, so any
  tolerance changes the bytes being hashed. Used the workstream's own
  `refresh_artifact_metadata.py` (strict replay of all 30 embedded runs); the
  before/after structural diff changed only hash-binding leaves
  (`implementation_sha256`, `evidence_implementation_sha256`, `run_sha256`,
  `base_run_sha256`, parity run digests, payload seals, manifest digests) —
  raw solutions, problems, diagnostics, gates and summaries identical; a
  second run reproduced the same bytes. Fixed the refresh script to write
  sidecars with `newline="\n"`. No `cft_revival/material_fields` source file
  changed. Isolated in its own commit for easy review/revert.
- Files: `modern/examples/material_fields/{artifacts/**,POSTHOC_AUDIT.md,audit_implementation_eol.py,refresh_artifact_metadata.py}`,
  `modern/tests/material_fields/test_posthoc_eol_audit.py`,
  `modern/docs/workstreams/material-fields-{devlog,learning}.md`.

### 4. `l0_surrogate_v3` and `l0_surrogate_v4` (2 failures each)

- Failure: `test_preflight_never_loads_real_v3_rows_or_assessment` /
  `test_preflight_does_not_load_real_rows` raised `results path already
  exists`; `test_real_result_path_and_placeholder_are_empty_before_freeze` /
  `test_empty_result_state_before_execution` asserted `not RESULTS.exists()`.
- Root cause: pre-execution assertions, but both experiments executed
  (`4af90a34`, `2c310750`) and their bundles are committed.
- Resolution (cft_orbit_wall_loss_v4 pattern): when `results/run-manifest.json`
  exists, assert that lock and manifest bind the frozen predeclaration,
  partitions and preflight hashes, that `run_manifest_hash`,
  `replicate_result_hash`, `frozen_hash` and every model hash recompute (v3
  additionally that `provenance-failure.json` binds the manifest), and that
  `preflight`/`execute` refuse without changing a byte under `results/`. The
  blind-preflight property still runs against a scratch results path. v4's
  `DEVLOG.md`/`LEARNING_SCRATCHPAD.md` are commit-bound protocol paths and are
  not edited.
- Files: `modern/tests/experiments/l0_surrogate_v3/test_atomic_pipeline.py`,
  `modern/tests/experiments/l0_surrogate_v4/{test_commit_binding.py,test_protocol.py}`.

### 5. Duplicate-basename collection clash

- Failure: `import file mismatch` for `test_results.py`, `test_dashboard.py`
  and others when more than one experiment directory was collected together.
- Root cause: rootdir-relative module names collide under the default
  `prepend` import mode when test directories lack `__init__.py`.
- Resolution: `--import-mode=importlib` in `modern/pyproject.toml` `addopts`
  (plus `"."` in `pythonpath`); no test file renamed, no `__init__.py` added.
- Files: `modern/pyproject.toml`.

### 6. Verification

- `modern/`: `python -m pytest tests -q` (one invocation):
  **1619 passed, 5 skipped, 0 failed** in 13 min 29 s on `8466c37a`; after
  rebasing onto `62de2ca3` (pic2d, paper claim-matrix admission):
  **1677 passed, 5 skipped, 0 failed** in 13 min 03 s. Skips are pre-existing
  and documented: Windows directory-symlink privilege, orbit v4 overlay-commit
  allowlist (HEAD is not that commit), field-surrogate v1 pre-execution,
  field-surrogate v2 no terminal failure, optional pybind11 extension.
- repo root: `python -m pytest paper/tests -q`: **19 passed, 138 subtests**
  on `8466c37a`; **33 passed, 177 subtests** after the rebase.
- `git ls-files --eol | Select-String 'i/lf\s+w/crlf'` is empty; every new
  tracked text file is UTF-8 LF.
- Not changed: `four_cell_topology_search_v2/results/preregistered-protocol.json.sha256`
  also records a CRLF digest (`ec2e9a73…` vs LF `5c195119…`); it is under
  `results/`, no test checks it, and it is left as-is for its own audit.
