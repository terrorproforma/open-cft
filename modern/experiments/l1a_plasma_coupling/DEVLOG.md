# L1a plasma-coupling development log

## 2026-09-02 02:51 AEST

- Added an experiment-local strict L1a v1.1 adapter that checks strict JSON,
  sidecars, sealed payloads, accepted manifest fields, serialized map hashes,
  source/map binding, producer/config/code identities, provenance and field
  residual gates before coupling v2 can issue accepted evidence.
- Added topology-aware coupling records, uncertainty propagation, exact
  four-cell compatibility gates, hypothetical L0-derived plasma inputs,
  deterministic nine-start plumbing, complete residual/conservation/branch
  serialization, gated L0 screening and sealed dataset/report/manifest output.
- Ran the real experiment. Results: 0 accepted, 3 failed, 0 plasma solves.
  Compact failed inverted-mirror projection; opposed-cusp had only three
  segments; triplet's apparent four segments relied on finite-boundary zeros.
  No state or performance was published.
- Added eight tests for artifact tampering, manifest mismatch, staleness,
  topology mismatch, identity, deterministic replay, fail-closed publication
  and a successful manufactured coupled plasma fixture.
- Validation: focused experiment/coupling/plasma/physics suite passed 206;
  compileall passed for `src`, `tests` and this experiment; FYP diff/status
  remained clean; owned Python files had no lines over 100 characters.
- Full suite with isolated imports reached 897 passed and 1 skipped, with one
  unrelated concurrent material-field artifact/schema failure at
  `tests/material_fields/test_spec_ledgers.py`.
- Follow-up: accepted material/open-boundary fields need four interior,
  geometry-identified cells and calibrated uncertainty/transport evidence
  before a real field-driven global-plasma state can be published.

## 2026-09-03 00:08 AEST

- Break: `tests/experiments/l1a_plasma_coupling` was 3 failed / 3 errors /
  2 passed after commit `00cf29fc`. Three independent causes:
  1. Working-tree line endings. Global `core.autocrlf=true` rewrote every
     accepted file under `examples/axisymmetric/results/` to CRLF when the
     `dbcab646` migration re-materialized them (index `i/lf`, worktree
     `w/crlf`). Exact-bytes SHA-256 sidecars therefore failed first
     (`invalid SHA-256 sidecar for manifest-l1a-v1.json`), and
     `tests/fields/test_axisymmetric_contract_ledgers.py` failed the same way.
     Fixed locally by writing HEAD's blob bytes back verbatim (no content
     change; `git diff` empty after `git update-index --refresh`). A
     `.gitattributes` `eol=lf` pin for these hash-bound files is the durable
     fix and is left as a proposal (outside this task's commit scope).
  2. Serialization pin. The adapter hard-pinned schema
     `cft-axisymmetric-field-map/1.1.0` / manifest `1.1.0` and hand-parsed the
     bytes; origin migrated the accepted set to v1.2
     (`field-json-sorted-utf8-signed-zero-v2`).
  3. Coupling API. `cft_revival.coupling.build_coupling_record`,
     `coupling_record_dict` and `global_solver_inputs` now name the coupling
     v4 CFT builder; the v2 same-z axis/wall comparison this experiment uses is
     re-exported only as the deprecated `build_screening_proxy` (no acceptance
     authority) with serializers in `cft_revival.coupling.records`.
- Fix (`adapter.py`): artifacts load through the public v1.2 loader
  `reload_field_artifact_bytes(allow_legacy_v1_1=False)` plus an explicit
  `field_artifact_canonical_bytes` round-trip; the manifest loads through
  `parse_field_json_bytes(require_canonical_file_bytes=True)` +
  `validate_design_manifest`, and the whole accepted set is cross-validated by
  `validate_design_manifest_file`. Gate order kept: sidecars -> manifest
  entry/file hash -> accepted-set loader -> coupling v2 evidence. Pins:
  `ARTIFACT_SCHEMA`/`MANIFEST_SCHEMA`/`FIELD_CANONICALIZATION` = v1.2 with a
  fail-closed drift check against the package constants; adapter id
  `experiments.l1a-plasma-coupling.accepted-l1a-v1.2`, contract `2.0.0`,
  backend version `artifact-schema-1.2.0`; semantic hashes use
  `canonical_payload_sha256`; `serialization.py` joined the producer code
  hash inputs; `ACCEPTED_MAP_POLICY` names v1.2 because coupling v2's policy
  default still says v1.1 (callers passing that default are pinned to v1.2;
  any other schema is rejected).
- Fix (`experiment.py`): artifact/manifest documents come from the same
  loaders (no `strict_json_bytes` on accepted files); topology projection
  calls `build_screening_proxy` and the `records` serializers; the
  DeprecationWarning is left to propagate. Dataset schema `1.0.0 -> 1.1.0`
  (additive `coupling_policy.projection` block naming the deprecated proxy);
  report claim boundary states both facts. Test: the one direct
  `build_coupling_record` call became `build_screening_proxy`.
- Physics check (v1.1 `dbcab646~1` vs v1.2 worktree): 0 finite-value
  differences in all three artifacts; only 33-35 `field_map.b_r_t` axis
  entries (and one compact `profiles.wall.b_r_t`) changed `-0.0 -> 0.0`,
  plus schema/canonicalization/hash strings. Old (v1.1-derived) vs new
  dataset: 0 non-identity differences in `topology`, `legacy_comparison`,
  `plasma`, `screening_performance`, `status`, `failure_reason`, `summary`
  (still 0 accepted / 3 failed / 0 compatible; same probabilities and mirror
  ratios). No test asserted a frozen hash; no pin was edited.
- Identity mapping (provenance only, not physics). Unchanged: `source_hash`,
  `config_hash`, `coupling_model_hash`, `diagnostics`. Changed:
  `artifact_hash`/`file_sha256` and `payload_sha256` per the migration ledger
  (compact `6510f6ea..->bec5ea78..`, opposed `dbf05208..->6591950a..`,
  triplet `ac5420d9..->86ec001f..`; manifest file `8444389e..->462c46d2..`,
  payload `2c912b84..->c961e560..`); `field_map_hash` (coupling v2 packs
  IEEE bits, so `-0.0 -> +0.0` changes it: compact `c4e19ab8..->6e06b869..`,
  opposed `29c17e92..->80190eec..`, triplet `82068005..->06e8e7ca..`);
  `field_model_hash` `3098be23..->72a17c0a..`; `code_hash`
  `e12602bf..->664496b5..`; `adapter_code_hash` `42e5d436..->0a2eb61c..`;
  therefore `source_map_binding_hash` and `record_hash` (opposed
  `b8b3bade..->03afbb0e..`, triplet `e8e3370b..->8726396b..`).
- Results dir: tests never read `results/` (they replay into `tmp_path`), so
  nothing was force-added. The gitignored local bundle was regenerated so it
  validates against the v1.2 accepted set; the v1.1-derived bundle was kept
  out of the repo for the comparison above.
- Validation: `tests/experiments/l1a_plasma_coupling` 8 passed;
  `tests/fields` 62 passed; `tests/coupling` 134 passed / 9 failed, all in
  `test_coupling_records.py` with `field map is stale under maximum_age_s`:
  `evidence_helpers.NOW` is fixed at 2026-09-01T12:00Z and those tests rebuild
  with the wall clock and the 86 400 s default, so they began failing at
  2026-09-02T12:00Z independent of this change (not touched; reported).
- Follow-ups: pin LF for hash-bound artifacts in `.gitattributes`; port the
  topology stage from the deprecated v2 screening proxy to coupling v4;
  `code_hash` hashes worktree bytes of the fields package, so it differs
  between CRLF and LF checkouts of the same commit.
