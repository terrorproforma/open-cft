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
