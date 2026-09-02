# Paper workstream devlog

## 2026-09-01 — Reproducible updated-paper scaffold

### Scope

- Worked only in new `paper/` paths and
  `modern/docs/workstreams/paper-*`.
- Used committed revision
  `17a121921671ac6e0012e6def2a1c9e1afe48294` as the result-evidence boundary.
- Did not edit code, shared documentation, `FYP/`, or Git history.

### Added

- Buildable LaTeX manuscript with modernization/reproducibility framing,
  literature lineage, methods, legacy audit, verified L0 model, multi-fidelity
  campaign, V&V/UQ protocol, the first 8,192-point result, limitations, data
  availability, and visible closed L1/L2/L3 sections.
- Machine-readable BibTeX records, claim matrix, gate registry, and
  figure/table generation contract.
- Notation/glossary, author checklist, response-ready supplementary outline,
  and paper workstream instructions.
- Standard-library policy checks for forbidden overclaims, missing/unused
  citations, evidence-source Git blobs, claim markers, gate structure, JSON,
  and generic placeholders.
- A no-install TeX build wrapper and unit tests for the policy checks.
- This paper-specific devlog and `paper-learning-ledger.md`.

### Evidence controls

- Registered all present result claims against committed source documents and
  Git blob identities.
- Marked the optimization benchmark as unevaluated because its committed
  `results` field is null.
- Kept current GPU timing as an uncontrolled diagnostic and prohibited a
  speedup conclusion.
- Required accepted committed manifests before L1, L2, or L3 prose can replace
  evidence-gate blocks.

### Validation

- `python paper/scripts/check_paper.py`: passed claim, citation, gate, source
  identity, JSON, overclaim, and placeholder checks.
- `python -m unittest discover -s paper/tests -v`: 6 tests passed.
- `python -m compileall -q paper/scripts paper/tests`: passed.
- `python paper/scripts/build.py`: passed through the no-install direct
  `pdflatex`/`bibtex` fallback because MiKTeX `latexmk` requires unavailable
  Perl. Final `manuscript.log` has no LaTeX errors, unresolved citations,
  unresolved references, or overfull boxes.
- Built `paper/build/manuscript.pdf`: 7 pages, 242,526 bytes, SHA-256
  `2F5D24CCE1BBEE47906623D83CC30DED6FD800CB01A33971DBA0F19CD03F3116`.
- `git diff --check` on paper-owned paths: passed.
- Protected-path diff check for `FYP/`, modern source/tests, and shared modern
  audit/reference/result/architecture docs: empty.

### Corrections during validation

- The first lint run caught two negated README statements as positive
  overclaims and found that two arXiv DOI fields lacked matching resolver URLs.
  Wording and bibliography records were corrected, and all policy tests pass.
- The initial TeX attempt exposed the missing `amsmath` declaration; the
  package was already installed and was declared explicitly.
- MiKTeX exposed a present-but-unusable `latexmk` because Perl is absent. The
  wrapper now detects this state and performs direct TeX/BibTeX passes without
  installing anything.
- The shared branch advanced concurrently from the evidence snapshot to
  `41bf909` while this workstream ran. Source blobs remained unchanged; the
  checker now requires the frozen evidence revision to be an ancestor of
  current `HEAD`, preserving reproducibility without assuming branch
  immobility.

### Follow-up risks

- A future L0 machine-readable result manifest is needed before replacing the
  manually transcribed range table with generated content.
- Bibliographic metadata should be rechecked against the selected venue's
  style at submission freeze; non-DOI records are explicitly identified.
- Physics and optimization results produced by concurrent agents remain
  inadmissible until committed manifests satisfy the registered gates.

## 2026-09-01 — Substantive audit-defect correction

### Changed

- Moved the evidence freeze to committed dashboard revision
  `41bf909127dc021abe8078fd77a98aa3a6e4cf33`.
- Added a compiled-and-machine-readable manifest schema registry. L1--L3
  acceptance now requires recognized type/version, a JSON manifest in the
  dedicated manifest directory, resolvable ancestry, a committed matching
  manifest blob, required source roles, Git blob plus SHA-256 bindings, and
  gate-specific metric constraints.
- Added `paper/evidence/l0-run-manifest.json`, binding the committed sweep
  config, report, equation ledger, model/reference/Warp/workflow/CLI code,
  dashboard/gallery generators and data, and accepted HTML.
- Recorded raw reported ranges, CUDA and reconstructed-reference residual
  classes, parity/failure counts, dataset identity, and the mandatory
  uncontrolled-timing caveat. The run revision remains explicitly null.
- Replaced detached claim annotations with exact `EvidenceClaim` bodies and
  location/scope authorization. Added adversarial detection for quantitative,
  experimental-accuracy, validation, cross-backend, and GPU/CUDA performance
  prose outside structured claims.
- Replaced the hand-copied L0 table with
  `paper/generated/l0-ranges.tex`, generated from the manifest and committed
  HTML payload. Its deterministic sidecar binds generator, manifest, accepted
  HTML blob/data hashes, output hash, claim ID, evidence revision, and source
  epoch.
- Corrected the implementation wording: Python and Warp share canonical
  preprocessing and are cross-backend implementations.
- Set the manuscript author to Angus Muffatti. Added structured human gates for
  coauthor/order, contributions, affiliations, and corresponding-author data.
- Added deterministic TeX/PDF controls and a two-clean-build verifier. Local
  build products remain excluded by `paper/.gitignore`; source/evidence and
  generated table inputs remain trackable.

### Validation

- Strict policy/citation/artifact check: passed.
- Adversarial unit tests: 11 passed, including fake `deadbeef`, README
  masquerade, wrong manifest type, missing roles/metrics, altered claim text,
  detached IDs, experimental-accuracy variants, CUDA 10x variants, and manual
  table alteration.
- Python compile check for paper scripts/tests: passed.
- Two consecutive clean builds: byte-identical 245,354-byte PDFs, SHA-256
  `bd35afd2a075ef83b1368db56169e398c180f95d801389fd04eecd86377a31f2`.
- Build tools recorded locally: CPython 3.12.10, MiKTeX-pdfTeX 4.23
  (MiKTeX 25.12), and MiKTeX-BibTeX 4.2 (MiKTeX 25.12).
- PDF metadata inspection: title and Angus Muffatti author are fixed; no
  creation/modification dates are emitted.
- Final TeX log: no errors, unresolved citations/references, or overfull boxes.

### Remaining human gates

- Approve coauthor inclusion/order and the final manuscript.
- Approve contribution taxonomy assignments.
- Supply and approve current affiliations.
- Select a corresponding author and approve correspondence details.
- Confirm target-venue disclosure, licensing, and submission-format choices.
## 2026-09-03 — Wall-loss v4 evidence and draft subsection

### Scope

- New paths only: `paper/scripts/generate_wall_loss_v4_evidence.py`,
  `paper/evidence/wall-loss-v4.json`, `paper/generated/wall-loss-v4.tex`,
  `paper/generated/wall-loss-v4.provenance.json`, `paper/sections/`,
  `paper/tests/test_wall_loss_v4_evidence.py`; `manuscript.tex`, `claims.json`
  and `result-gates.json` untouched.
- Evidence source: the accepted `cft_orbit_wall_loss_v4` bundle at commit
  `6922a3cf97d261735266aa1a5a0c0c9683e021ca` (results manifest SHA-256
  `ef3863b0a3ba0a1d74187b05daf81d5d94d3838a7e33ecf82c485dccd162929f`).

### Added

- Standard-library generator that verifies every bundle file against the
  results manifest (tolerating exactly the nine CRLF-recorded
  `artifacts/orbits/<case>.json.sha256` sidecars), cross-checks terminal,
  campaign, gates, summaries and convergence artifacts, and emits macro-only
  TeX with a per-macro artifact/pointer/format trace.
- Draft results subsection "Collisionless full-orbit electron wall loss in the
  divergent-exit field" (method, results with per-case and per-cell tables,
  numerical convergence, boxed model-bounded interpretation) using evidence
  macros only; a standalone driver compiles it.
- Tests: deterministic regeneration, committed outputs current, every macro
  traces to a hashed artifact and reformats identically, derived macros
  recompute, section uses only defined macros and no literal digits, tampered
  bundles rejected, standalone pdflatex compile clean of errors and overfull
  boxes.

### Repaired

- `paper/generated/l0-ranges.provenance.json` regenerated with the repository's
  own `generate_tables.py`: the committed sidecar carried the CRLF-era
  generator and manifest hashes, so `check_paper.py` and two existing tests
  failed on every LF checkout. Table bytes unchanged.

### Validation

- `python paper/scripts/check_paper.py`: passed.
- `python -m unittest discover -s paper/tests`: 19 tests OK.
- `python paper/scripts/build.py`: clean deterministic manuscript build
  (pdflatex/bibtex, MiKTeX installer disabled).
- Standalone section: two pdflatex passes, no errors, no undefined references,
  no overfull boxes; three pages.

## 2026-09-03 — Wall-loss v4 admitted to the claim matrix and manuscript

### Scope

- Worktree `uni-project-paper-v4`, branch `paper/wall-loss-v4-claim` from
  `origin/feat/sota-foundation` (`5b85d2ad`). Paper-owned paths and
  `modern/docs/workstreams/paper-*` only; nothing under
  `modern/experiments/**/results/` or `FYP/` touched.

### Evidence level decision

- The campaign is numerical evidence about a collisionless, prescribed-field,
  test-particle model (`collisionless_prescribed_field_test_particle_wall_loss_not_pic`).
  It is not L1 (no reduced performance model, no L0 mapping, coupling export
  is export-only), not L2 (not coupled), not L3 (not PIC, not experimental).
  The framework's mechanism for accepted numerical evidence is the L0 pattern:
  typed committed manifest, artifacts bound by Git blob and SHA-256 at a
  resolvable revision, exact registered claim text, generated artifacts with
  sidecars. That mechanism is applied through a new gate kind,
  `numerical-campaign` (`GATE-WALL-LOSS-V4`, `opens_level: null`), at the
  verification tier of the V&V protocol; L1--L3 stay closed.

### Added

- `paper/evidence/manifests/wall-loss-v4.json`
  (`paper-test-particle-campaign-manifest` 1.0, level `numerical-campaign`):
  29 source files bound at `6922a3cf` (results manifest, campaign result,
  gates, convergence, protocol, authorities, P2 authority, field-map
  convergence, manufactured gates, coupling export, terminal record, lock,
  shakedown, nine case summaries, two field-evidence records, two transitions,
  and the frozen preregistration protocol/authorities/shakedown), the post-hoc
  audit bound at `258f69b2`, 35 metrics copied from the raw artifact values.
- Gate `GATE-WALL-LOSS-V4` in `result-gates.json` with 20 metric constraints,
  the section binding, heading, revision macro and prohibited inferences;
  L1--L3 gates now declare `kind: physics-level`.
- Claim records CLM-012 (abstract summary), CLM-013 (campaign result, with
  `evidence_level`, `evidence_level_justification`, `bindings`, `non_claims`),
  CLM-014 (generated tables `TAB-WALL-LOSS-V4`), CLM-015 (numerical
  verification), CLM-016 (model-bounded scope limitation), CLM-017 (labelled
  Discussion interpretation: zero reflections, mirror picture unsupported);
  manifest `WALL-LOSS-V4-20260902-4608-V1` registered in `claims.json`.
- `check_paper.py`: flattens `\input{sections/...}` before every prose,
  claim and citation check; recognises `numerical-campaign` gates; validates
  the campaign manifest as a typed gate manifest (committed blob, roles,
  metric constraints) and then runs `_check_wall_loss_campaign` (byte-identical
  regeneration of evidence/TeX/sidecar from the bundle, artifact hashes on
  disk, metric == raw macro value, results tree unchanged, preregistration and
  audit revision chains, section macro-only rule with no literal digits,
  classification macro rendering, registered non-claims present, section and
  macro-file bindings exactly once, `\WallLossEvidenceRevision` spells the
  revision, claim records bound and located); artifact contract now dispatches
  by `generator_module` and accepts a declared `artifact_claim_count`;
  `claims.json` manifest entries are checked against their files.
- `manuscript.tex`: `\input{generated/wall-loss-v4.tex}` in the preamble;
  new `\section{Accepted numerical campaign: collisionless electron wall loss}`
  after the L0 timing status and before the L1 gate, containing
  `\input{sections/wall-loss-v4.tex}`; abstract sentence CLM-012 (macro-bound);
  evidence-boundary text and date line name both revisions; a sentence after
  the L1 gate box stating the campaign leaves it closed; new
  `\section{Discussion}` (interpretation only) with CLM-017 and the open
  question on the multi-cell wall-cusp topology (topology experiments are not
  admitted, so they are not cited as evidence); Limitations, data availability
  and Conclusion updated without new numbers.
- Section: result, verification and interpretation sentences are now exact
  `\EvidenceClaim` bodies; new derived macro `\WlfToleratedSidecars` replaces
  the one number that had been typed as a word; the standalone driver defines
  the claim macros and loads `microtype` like the manuscript.
- Tests: `test_wall_loss_v4_admission.py` (14 adversarial tests: tampered
  metric, wrong level, wrong classification, missing non-claim, unbound or
  relocated claim, revision-macro mismatch, duplicated or misplaced binding,
  evidence-file substitution, heading resolution through flattening, contract
  item); two existing wall-loss tests updated for the admitted state.

### Validation

- Before the commit `python paper/scripts/check_paper.py` reported exactly one
  error, the fail-closed "accepted manifest is not committed at HEAD"; after
  the commit it passed.
- `python -m unittest discover -s paper/tests`: 33 tests OK (19 existing plus
  14 in `test_wall_loss_v4_admission.py`; two wall-loss tests updated).
- `python paper/scripts/build.py`: clean; `paper/build/manuscript.pdf`
  303,672 bytes, SHA-256
  `bdfdba4cc65e9b0a15723dfcaae5116394824a3f0dd3c5d77f65b13062736fde`, 11
  pages, no LaTeX errors, undefined references or overfull boxes.
- `python paper/scripts/verify_reproducible_build.py`: two clean builds
  byte-identical.
- Pages rendered with MiKTeX `pdftoppm` and inspected: the campaign section
  opens on page 5 (Section 7, subsection 7.1), the per-case and per-cell tables
  (Tables 2 and 3) render on page 7, the boxed model-bounded interpretation on
  page 8 ahead of the closed L1 gate, and the Discussion on page 9; the
  abstract sentence and the two-revision date line render on page 1.
- Standalone section driver: two pdflatex passes, no errors, no undefined
  references, no overfull boxes (with `microtype`, as in the manuscript).
- One overfull box appeared while integrating (2.9 pt, the sentence before the
  40-hex L0 revision); fixed by rewording the sentence, not the hash.
