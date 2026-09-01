# Validation Workstream Devlog

## 2026-09-01

- Added dependency-free typed/versioned contracts for evidence, quantities,
  experiment uncertainty, stochastic PIC ensembles, and context of use.
- Added canonical JSON/SHA-256 evidence identity, strict JSON parsing, bundle
  hash checking, and duplicate/conflicting evidence detection.
- Added unit-aware point/interval comparisons, conservation gates, convergence
  studies, replicate summaries, analytical/manufactured assessments, grouped
  split audits, context-of-use audits, and deterministic report generation.
- Encoded the corrected low-fidelity, PIC, and PIC-informed 2020 S1 published
  values with DOI provenance and an explicit external-model/not-experimental-
  truth policy.
- Added machine-readable evidence and integration contracts plus an integration
  handoff document.
- The first direct Python invocation could not import the source-layout package;
  setting `PYTHONPATH=src` was the correct no-install workflow.
- Validation at this entry: `python -m pytest tests/validation -q` passed 29
  tests before the combined verification-assessment API and its test were added.
- Final validation rerun: `python -m pytest tests/validation -q` passed 29 tests
  after the combined assessment change (the existing convergence test was
  expanded, so the count remained 29).
- `python -m compileall -q src/cft_revival/validation tests/validation` passed.
- Default full-suite collection was blocked by pre-existing duplicate test
  module basenames. With `--import-mode=importlib`, 436 tests passed, one
  optional extension test skipped, and two pre-existing plasma solver tolerance
  tests failed; no out-of-scope fixes were made.
- `git diff -- FYP` and `git status --short -- FYP` were empty. The owned-path
  status check listed only the requested validation directories and
  `validation-*` documents. No commit or push was performed.

## 2026-09-01 audit-defect closure

- Bumped validation contracts and evidence bundle to semantic version 2.0.0.
- Added closed evidence-kind/source-authority pairings and authority claim and
  credibility ceilings. Simulation/PIC and published outputs cannot assert
  experimental truth or predictive validity.
- Required complete metadata and per-observation uncertainty for experiment
  evidence.
- Bound context-of-use requirements to exact quantities/SI units, evidence and
  independent-group counts, applicability ranges, contexts, and revisions, with
  dimension-specific missing diagnostics.
- Bound conservation/convergence gates to evidence hash/provenance, contexts,
  quantity/unit, revisions, group, and mesh; made spacing unit-bearing and
  relative residual scale mandatory.
- Hardened post-hash bundle parsing for closed nested keys, enums, booleans,
  nulls, finite/range checks, DOI/locator validation, and integrity limitations.
- Required homogeneous PIC ensembles with at least three seeds and replaced the
  1.96 interval with a documented two-sided Student-t policy.
- Made reports recompute partition status from registry identity. Empty,
  insufficient, and leaking evidence cannot report `PASS`.
- Preserved `MDO (original)`, `PIC`, and `MDO (modified)` separately from
  editorial interpretations and added citation/label tests.
- Renamed owned test modules with `test_validation_*` basenames to avoid future
  flat-import collisions without changing global pytest configuration.
- Focused audit suite passed 42 tests before final spec/document checks were
  added. Final verification results follow after execution.
- Final focused normal and importlib runs each passed 43 tests.
- The dedicated schema/hash adversarial file passed 16 tests, including
  maliciously recomputed hashes with unknown keys, booleans, nulls, invalid
  enums/URIs/ranges, duplicate keys, and nonfinite values.
- `compileall` passed for the owned source and tests.
- Bundle/report generation loaded three records with exact labels
  `MDO (original)`, `PIC`, and `MDO (modified)`; the reference-only report
  correctly returned `NOT_EVALUATED` and had deterministic SHA-256
  `40e41ea85c51bd55227059b6e0720117262440794fc68889ec71243e6a2dc539`.
- Normal and importlib full-suite runs produced the same out-of-scope result:
  612 passed, one optional extension skipped, 22 failed, and 10 errors in
  coupling, hybrid, plasma, and visualization workstreams. No validation test
  failed and no out-of-scope repair was attempted.
- FYP diff/status was empty. No dependency installation, commit, or push was
  performed.

## 2026-09-02 remaining v2 acceptance closure

- Replaced assessment input of precomputed conservation results with raw
  content-hash-bound gates. Assessment and verification reporting now recompute
  residual, threshold, normalized value, and status from candidate observations.
- Replaced caller-supplied convergence values with per-level evidence records.
  Each level binds error/spacing observations, content hash, provenance,
  context, revisions, immutable run/design identity, and mesh; order is
  recomputed and the finest content hash must equal the candidate.
- Added immutable design, run-lineage, hardware-article, campaign, and specimen
  identities. Partition and context audits no longer derive independence from
  `group_id`, and context audit checks calibration overlap directly.
- Hardened HTTP(S) authority and DOI validation and wrapped huge integer/float
  binary64 conversion overflow as `EvidenceSerializationError`.
- Corrected empty registry/report semantics from `FAIL` to explicit
  `NOT_EVALUATED`; evaluated identity leakage remains `FAIL`.
- Added forged-residual, changed-finest-level, physical-leakage, URI/DOI, huge
  numeric, and status-semantics adversarial tests.
- Focused suite passed 59 tests before final documentation and verification.
- Final focused normal and importlib runs each passed 59 tests.
- Schema/report-focused checks passed 29 tests; compileall passed.
- Normal and importlib full suites each passed 790 tests with one optional
  extension skip.
- Bundle/report generation loaded the three exact native labels, reported both
  published-reference and empty-registry status as `NOT_EVALUATED`, and produced
  deterministic report SHA-256
  `4d66bf6541aef2dd8cd44abc741fb9ef6aa9dce1089453a3d15748a170d53335`.
- FYP diff/status was empty. No install, commit, push, or out-of-scope edit was
  performed.
