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

## 2026-09-03 — L1a sweep v2 and topology null results admitted to the claim matrix

### Scope

- Worktree `uni-project-paper-topo`, branch `paper/topology-and-sweep-claims`
  from `origin/feat/sota-foundation` (`7a30fc2e`). Paper-owned paths,
  `modern/docs/workstreams/paper-*`, plus one read-only overlay under
  `modern/experiments/four_cell_topology_search_v2` (`POSTHOC_AUDIT.md`,
  `audit_sidecar_eol.py`) and its test; nothing under
  `modern/experiments/**/results/`, no frozen preregistration file and nothing
  under `FYP/` touched.

### Evidence level decision

- New gate kind `numerical-screening` beside `numerical-campaign`: it admits one
  preregistered, single-execution L1a field-only screening study (linear-vacuum
  equivalent-current fields; no permanent-magnet or nonlinear-iron material
  model) at exactly its `recorded_outcome`, one of `accepted-screening`,
  `preregistered-null`, `recorded-characterization`. Gate status `accepted`
  means "admitted as recorded", never that a positive finding is accepted; a
  null is admitted as a null under the frozen cusp/cell definitions and is not
  an existence disproof. `numerical-campaign` was not reused because that kind
  was defined for one accepted campaign on a declared component model with a
  campaign-specific manifest type, whereas these studies screen a design space
  and two of them are nulls that "accepted campaign" would overstate. None of
  the three is the L1 field-resolved reduction (no manufactured solution,
  refinement study, L0 mapping or performance model); `GATE-L1`--`L3` stay
  closed and every gate declares `opens_level: null`.

### Added

- `605be5ce add four_cell_topology_search_v2 posthoc EOL audit`: read-only
  `audit_sidecar_eol.py` (stdlib), `POSTHOC_AUDIT.md`,
  `modern/tests/experiments/four_cell_topology_search_v2/test_posthoc_audit.py`
  (10 tests). Finding: `results/preregistered-protocol.json` (and the
  `protocol_sha256` bound by manifest, dataset and lock) records the CRLF-era
  digest `ec2e9a73…` (10811 bytes) of the frozen protocol whose LF bytes hash
  to `5c195119…` (10580 bytes, 231 lines); canonical payload `bd522269…`
  recomputes; the other 12 artifacts and 13 sidecars are byte-exact.
  Disclosed: `validate_results` refuses the bundle on LF checkouts, and the
  bundle's own GPU replay records 2 of 4 diagnostic passes (`v2-031` 9.42e-6,
  `v2-063` 6.50e-6 against 5e-6) while every field component reproduced.
- `paper/scripts/generate_topology_screening_evidence.py`: one generator for
  the three studies (`ExperimentSpec` per study), verifying each bundle against
  its own manifest (sidecars, sealed canonical payloads, protocol bindings),
  binding it to its results commit and tree, and writing per study the
  evidence file, generated macros with two `\ArtifactClaim` tables, and
  sidecar. Audited LF→CRLF tolerance for exactly `protocol.json` (sweep) and
  `results/preregistered-protocol.json` (four-cell), nothing else.
  Macro prefixes `Swp` (107 macros), `Fcn` (107, including hash-bound lineage
  macros for the superseded proxy search and the two failed coupling-v4
  validations), `Tch` (83).
- Typed manifests `paper/evidence/manifests/{l1a-sweep-v2,four-cell-v2,
  topology-characterization-v1}.json` (`paper-l1a-screening-manifest` 1.0):
  20 / 17 / 35 source files bound by Git blob and SHA-256 at the results
  revisions (`f30cb42e`, `7120e8ed`, `3ce6c546`), frozen protocols, post-hoc
  audits (`9e68df21`, `605be5ce`), six lineage files at their own revisions
  (four-cell only), 34 / 42 / 37 metrics copied from the raw artifact values.
- Gates `GATE-L1A-SWEEP-V2`, `GATE-FOUR-CELL-V2`, `GATE-TOPOLOGY-CHAR-V1` in
  `result-gates.json` with `recorded_outcome`, metric constraints, prohibited
  inferences and `null_semantics`; claims CLM-018 (abstract sentence), CLM-019
  / CLM-022 / CLM-025 (results, with `non_claims`), CLM-020 / CLM-023 / CLM-026
  (tables), CLM-021 / CLM-024 / CLM-027 (scope limitations; CLM-024 carries the
  lineage non-claim), CLM-028 (Discussion interpretation bound to the two
  nulls); contract items `TAB-L1A-SWEEP-V2`, `TAB-FOUR-CELL-V2`,
  `TAB-TOPOLOGY-CHAR-V1`.
- `check_paper.py`: `SCREENING_GATE_KIND`, `SCREENING_OUTCOMES`,
  `SCREENING_METRIC_MACROS`, `SCREENING_POLICY_METRICS`, schema type;
  `_check_topology_screening` (byte-identical regeneration, artifact and
  lineage hashes on disk, audited-EOL recomputation bound to
  `protocol.py::EOL_AUDITED_SIDECARS` / `audit_sidecar_eol.py` digests, metric
  == raw macro with type equality, policy metrics, outcome / model /
  classification agreement across gate, manifest, evidence and generator,
  results tree unchanged, prereg → results → audit → HEAD chains, frozen
  blobs, macro-only section with no literal digit, required table and
  classification macros, ArtifactClaim wrapping, bindings once, revision
  macro, claim records bound and located, lineage entries validated);
  `acceptance_policy.gate_kinds` must define exactly the known kinds; artifact
  renderers now take the contract item; new trackables; new required section.
- `manuscript.tex`: three revision macros; `\input` of the three macro files
  in the preamble; abstract sentence CLM-018; contribution list; evidence
  boundary paragraph; new Section 8 "Preregistered topology screening: sweep
  acceptance and four-cell null result" with three `\input` subsections
  (`sections/l1a-sweep-v2.tex`, `four-cell-v2.tex`,
  `topology-characterization-v1.tex`) between the wall-loss campaign and the
  L1 gate; sentence after the L1 gate box; Discussion paragraph rewritten
  from "open question, not cited" to the macro-bound CLM-028 with the labelled
  interpretation kept; Limitations, data availability and Conclusion updated
  without new numbers; date line points to Section 8 for the three revisions.
- Standalone driver `sections/topology-screening-standalone.tex`; tests
  `test_topology_screening_admission.py` (17) and
  `test_topology_screening_evidence.py` (8); README, author checklist,
  supplementary outline (S4c) and notation updated.

### Validation

- Before the paper commit `check_paper.py` reported exactly the three
  fail-closed "accepted manifest is not committed at HEAD" errors; after
  `ee633e6f` it passed.
- `python -m unittest discover -s paper/tests`: 58 tests OK (33 existing + 25
  new). `pytest tests/experiments/four_cell_topology_search_v2`: 10 passed;
  `l1a_geometry_sweep_v2`, `cft_orbit_wall_loss_v4/test_posthoc_audit.py`,
  `tests/visualization/test_plasma_topology_dashboard.py`: green (one false
  alarm from a `SOURCE_DATE_EPOCH` left in the shell; passed once unset).
- `python paper/scripts/verify_reproducible_build.py`: two clean builds
  byte-identical, `paper/build/manuscript.pdf` 391,734 bytes, SHA-256
  `6b4c6978e56fd5c225a24387f44a84ec080b19f8074733e7d3766b04d34f8701`, 17
  pages, no LaTeX errors, undefined references, warnings or overfull boxes.
- Pages rendered with MiKTeX `pdftoppm -r 70` to `%TEMP%\paper-topo-pages\`
  (`p-01.png` … `p-17.png`) and inspected: abstract sentence p. 1; Section 8
  opens p. 8 with 8.1 (p. 8–9); Table 4 (sweep gates) p. 10, Table 5
  (representatives with per-cell mirror ratios) p. 11; 8.2 p. 10–12 with
  Table 6 (failure taxonomy, `TOPOLOGY_COUNT` / `TOPOLOGY_UNSTABLE` 128 each)
  p. 11 and Table 7 (interior cusps per map) p. 12; 8.3 p. 12–13 with Table 8
  (null classes by zone) and Table 9 (stage relation) p. 13; L1 gate p. 14;
  Discussion p. 14–15 with the macro-bound multi-cell paragraph p. 15.
- Two overfull boxes appeared while integrating (a 148 pt gate table and the
  unbreakable `\texttt{manufacturability}` cell); fixed by ragged-right `p{}`
  columns with a wider gate column, not by editing any number.

## 2026-09-03 — MDO L0 campaign v1 admitted to the claim matrix and manuscript

### Scope

- Worktree `uni-project-paper-mdo`, branch `paper/mdo-l0-v1-claim` from
  `origin/feat/sota-foundation` (`e642f38c`), LF verified (`git ls-files --eol`
  shows `w/crlf` only on three `attr/-text` files of other experiments).
  Paper-owned paths and `modern/docs/workstreams/paper-*` only; nothing under
  `modern/experiments/**/results/`, no frozen preregistration file and nothing
  under `FYP/` touched; no GPU work.

### Evidence level decision

- The `numerical-campaign` kind is reused (second gate `GATE-MDO-L0-V1`,
  `opens_level: null`) because its definition, one accepted preregistered
  campaign about a declared component model, fits: the component model is the
  accepted L0 conservation model under the declared multiplicative
  cusp-survival closure CL-1 and declared uniform priors, executed once
  (8 of 8 binding gates, `accepted_result`). It is optimiser evidence
  (hypervolume per budget, paired comparisons, seed variance, Pareto sizes,
  robust-versus-nominal fronts, sensitivity to the cusp prior), not
  performance evidence; the gate record carries a `kind_justification`, and
  the gate-kind description in `acceptance_policy` now names both campaigns.
  No field solve, geometry variable or L0 mapping exists, so `GATE-L1`--`L3`
  stay closed. A new manifest type `paper-mdo-campaign-manifest` 1.0 was
  needed because required roles and metrics are type-specific.

### Added

- `paper/scripts/generate_mdo_l0_v1_evidence.py`: verifies the sealed bundle
  of `modern/experiments/mdo_l0_campaign_v1` file by file against
  `results/manifest.json` (137 files, every artifact sidecar re-checked; no
  end-of-line tolerance exists or is granted), requires the frozen
  `protocol.json`/`authorities.json`/`shakedown.json` to equal the sealed
  canonical copies, cross-checks the committed results dashboard
  (`modern/visualization/mdo-l0-campaign-v1.html` + generator, pinned
  manifest SHA-256 and revisions, embedded `campaign_result`, seed variance,
  gate blocks and per-run hypervolumes must equal the artifacts), recomputes
  every repeated number (seed means/stds, attained fractions, Jaccard index,
  scenario survivals, objective ranges, pair wins, invariance flags), and
  writes 334 `\Mdo...` macros with three `\ArtifactClaim` tables
  (`\MdoHvTable`, `\MdoRobustNominalTable`, `\MdoScenarioTable`) plus the
  evidence file and sidecar. Derived macros record their derivation and
  inputs; the four-cell solver probe numbers are parsed from the frozen
  protocol's disclosure text with a fixed regular expression.
- Typed manifest `paper/evidence/manifests/mdo-l0-v1.json`: 36 source files
  bound by Git blob and SHA-256 at `c553124b` (results manifest, terminal,
  lock, 20 artifacts, nine run records, two transitions, three frozen files
  whose blobs equal those at `4898d0fd`), a `dashboard` block binding the
  generator and HTML at `e642f38c`, 68 metrics copied from the raw macro
  values plus 10 fixed policy metrics.
- `result-gates.json` gate `GATE-MDO-L0-V1` (35 metric constraints,
  prohibited inferences); `manifest-schemas.json` type; claims CLM-029
  (abstract), CLM-030 (results, with `non_claims`), CLM-031 (tables), CLM-032
  (robust versus nominal), CLM-033 (sensitivity/scenarios, bound to both the
  optimisation and wall-loss manifests: the collisionless wall-hit estimand
  is not the Kornfeld cusp probability; the corrected four-cell solver closed
  only at p = 0 in the recorded probe), CLM-034 (scope limitation),
  CLM-035 (labelled Discussion interpretation, four readings); contract item
  `TAB-MDO-L0-V1` (`artifact_claim_count: 3`).
- `check_paper.py`: `MDO_METRIC_MACROS`, `MDO_POLICY_METRICS`, schema type,
  `_check_mdo_campaign` (byte-identical regeneration, artifact hashes on disk,
  dashboard bound at its revision and equal to the checkout by LF-normalised
  SHA-256, metric == raw with type equality, policy metrics, results tree
  unchanged, prereg → results → dashboard → HEAD chains, frozen blobs,
  macro-only section with no literal digit, classification and closure
  macros, non-claims, bindings once, `\MdoEvidenceRevision`, three
  ArtifactClaims, claim records bound and located), renderer, new required
  section, new trackables.
- `manuscript.tex`: `\MdoEvidenceRevision`, preamble `\input` of the macro
  file, fourth date line, abstract sentence CLM-029, contribution list,
  evidence-boundary paragraph, pointer from the optimisation-protocol
  subsection, new Section 9 "Preregistered robust multi-objective
  optimisation of the L0 model" (intro with the CL-1 formula, then
  `\input{sections/mdo-l0-v1.tex}` with subsection 9.1), sentence after the
  L1 gate box, Discussion paragraph with CLM-035 and a plain future-work
  sentence on the geometry screening, Limitations rewritten (the old "no
  admitted hypervolume result or baseline comparison" sentence was no longer
  true), data availability, Conclusion.
- Section `paper/sections/mdo-l0-v1.tex` (Method, Results, Robust versus
  nominal fronts, Sensitivity to the cusp prior, boxed interpretation);
  standalone driver; tests `test_mdo_l0_v1_admission.py` (17) and
  `test_mdo_l0_v1_evidence.py` (8); README, author checklist, supplementary
  outline (S4d) and notation updated.

### Validation

- Before the commit `check_paper.py` reported exactly the fail-closed
  "accepted manifest is not committed at HEAD"; after commit `e6db2122` it
  passed.
- `python -m unittest discover -s paper/tests`: 83 tests OK (58 existing +
  25 new).
- `python paper/scripts/verify_reproducible_build.py`: two clean builds
  byte-identical, `paper/build/manuscript.pdf` 446,316 bytes, SHA-256
  `e7900c1000e3d48ef02cc6e67e114dce946c9cadfa4cd7820414ee48bde0d4ff`, 22
  pages, no LaTeX errors, undefined references, warnings or overfull boxes.
- Pages rendered with MiKTeX `pdftoppm -r 80` to `%TEMP%\paper-mdo-pages\`
  (`p-01.png` … `p-22.png`) and inspected: abstract sentence and four-line
  date p. 1; Section 9 opens p. 14 with the CL-1 equation and 9.1; Table 10
  (hypervolume per optimiser and seed with mean ± std) p. 16; Table 11
  (robust versus nominal) p. 17 with the boxed interpretation; Table 12
  (priors and scenarios) p. 18 ahead of the L1 gate; Discussion paragraph
  with CLM-035 p. 20.
- One overfull box appeared while integrating (229 pt, the eight-column
  sensitivity table); fixed by ragged-right `p{}` columns for the label and
  probability columns, stacked two-line headers and `\scriptsize`, not by
  dropping any number.

## 2026-09-03 — Four-cell power-balance closure analysis admitted to the claim matrix and manuscript

### Scope

- Worktree `uni-project-paper-closure`, branch `paper/four-cell-closure-claim`
  from `origin/feat/sota-foundation` (`ba6875f6`), LF verified (`git ls-files
  --eol` shows no `w/crlf` under `paper/` or the paper workstream docs).
  Paper-owned paths and `modern/docs/workstreams/paper-*` only; nothing under
  `modern/experiments/**/results/`, no frozen preregistration file and nothing
  under `FYP/` touched (`FYP/Power_B_EQs.m` is read at its blob); no GPU work.

### Evidence level decision

- The admitted object is neither a preregistered campaign nor a screening: it
  is a derivation about the corrected four-cell discharge ledger (28 rows, 25
  unknowns) whose closed form is verified numerically. A new gate kind
  `analytic-consistency` was defined in `result-gates.json`: it admits one
  analytic consistency result about a declared equation set (a derivation whose
  closed form is verified numerically to a stated tolerance and pinned by
  committed tests), opens no physics level, and its status `accepted` means the
  derivation and its numerical verification are admitted as recorded; it
  accepts no correction of the equation set (the proposal stays
  `PROPOSED_NOT_ACCEPTED`) and says nothing about the physical thruster. Gate
  `GATE-FOUR-CELL-CLOSURE-V1` carries a `kind_justification`; manifest type
  `paper-analytic-consistency-manifest` 1.0.
- The typed manifest binds 14 files by Git blob and SHA-256 at the analysis
  commit `266d8a99` and requires them unchanged at `ba6875f6`: the analysis
  document, the ledger, the five `cft_revival.plasma` files, three pinning test
  files, the frozen MDO `protocol.json` (blob equal at preregistration
  `4898d0fd`), `FYP/Power_B_EQs.m` as lineage, `AUDIT.md` and `REFERENCES.md`.
  The executed package in the checkout must equal the bound blobs
  (LF-normalised SHA-256) or the generator refuses; a future change to
  `cft_revival.plasma` therefore requires re-admission at the new revision.

### Added

- `paper/scripts/generate_four_cell_closure_evidence.py`: binds the sources
  (one `git ls-tree` per commit plus one `git cat-file --batch`), verifies the
  executed package, imports `cft_revival.plasma` from `modern/src` of the
  checkout only, and RECOMPUTES the verification: `global_row_closed_form`
  against the full residual over 400 seeded random admissible states (max
  relative difference `\FccClosedFormRelDiff`), the R00–R26 residual on the
  manifold, the anode-fall coefficient at p = 0, the continuation ladder
  p = eps (1,1,1,1) at 300 V / 1 A through the production multistart solver
  (1 start, 600 iterations), the anode-only closures p = (0,0,0,eps) (5
  starts), the published-state misfit on the exact manifold, one relaxed root
  by bisection and the Jacobian rank at every floor. Documented numbers are
  read from the analysis document blob with fixed regular expressions
  (`DOCUMENT_PATTERNS`), from the ledger by JSON pointer and from the frozen
  protocol with the optimisation generator's `PROBE_PATTERN`; the generator
  refuses on any departure beyond `TOLERANCES` (closed form <= 1e-12,
  manifold <= 1e-11, each floor within 25 % of the document, floor/eps spread
  <= 2, anode-only closures converged under 1e-8, misfit within 2 %, relaxed
  depth within 2 %, rank 22 and condition <= 200 at every floor). The 13/80
  probe is read from the frozen protocol, not recomputed, and the document's
  reproduction must agree; differential evolution and the 200 random starts
  are documented only (SciPy, minutes). Recomputed values are recorded to a
  declared number of significant digits. 163 `\Fcc...` macros, two
  `\ArtifactClaim` tables (continuation ladder; global search / relaxed roots /
  Jacobian / misfit / probe with per-row status), evidence file and sidecar.
  Recomputation is cached per process (about 35 s once per process).
- Manifest `paper/evidence/manifests/four-cell-closure.json` (69 metrics from
  the evidence values plus 10 fixed policy metrics, 79 in all), gate record, gate-kind
  description, `manifest-schemas.json` type, claims CLM-036 (abstract),
  CLM-037 (closed form, verification, no admissible root; `non_claims`),
  CLM-038 (sub-region, continuation, global search, Jacobian, probe; bound to
  the MDO manifest too), CLM-039 (tables), CLM-040 (attribution: Kornfeld
  assumption 8, printed anode sign, legacy lines 136–137, audit fixes cancel),
  CLM-041 (proposed correction `PROPOSED_NOT_ACCEPTED`, rank 22 -> 21, solver
  defect fixed), CLM-042 (claim boundary), CLM-043 (Discussion interpretation:
  legacy `lsqnonlin` exit flags 1–3 accepted by status alone, residual norm
  discarded, `TolFun=1e-50`; residual floors read as the reported values;
  interpretation), CLM-044 (Discussion interpretation: mirror picture
  unsupported, four-cell topology undemonstrated, power balance inconsistent
  for interior p; bound to the wall-loss and four-cell manifests); contract
  item `TAB-FOUR-CELL-CLOSURE-V1` (`artifact_claim_count: 2`); policy
  `analytic_consistency_rule`.
- `check_paper.py`: `ANALYTIC_GATE_KIND`, `FOUR_CELL_CLOSURE_METRIC_MACROS`,
  `FOUR_CELL_CLOSURE_POLICY_METRICS`, schema type, `_check_four_cell_closure`
  (byte-identical regeneration = recomputation, source bindings at the analysis
  and verified-tree revisions, frozen protocol blob, executed package on disk
  equal to the bound blobs, artifact blobs, metric == value with type equality,
  policy metrics, classification and correction-status macros, opens no level,
  bindings once, `\ClosureEvidenceRevision`, the displayed closed form in the
  manuscript section with macro-bound coefficient and row index and no typed
  digit, macro-only section, two ArtifactClaims, claim records bound and
  located, interpretations kept out of the results section), renderer,
  required Section 10, trackables, kind handling in `_check_gates`.
- `manuscript.tex`: `\ClosureEvidenceRevision`, preamble `\input`, fifth
  date line, abstract sentence CLM-036, contribution list, evidence-boundary
  paragraph, Section 10 “Consistency of the four-cell power balance” (intro
  with the displayed closed form, `\cite{Kornfeld2007}`, then
  `\input{sections/four-cell-closure.tex}` with subsection 10.1), sentence
  after the L2 gate box, Discussion paragraph with CLM-043 and CLM-044,
  Limitations, data availability, Conclusion.
- Section `paper/sections/four-cell-closure.tex` (Derivation and numerical
  verification; Solution sub-region, continuation and global search;
  Attribution; Proposed correction (not accepted); Scope with the boxed claim
  boundary); standalone driver; tests `test_four_cell_closure_admission.py`
  (19) and `test_four_cell_closure_evidence.py` (9); README, author checklist,
  supplementary outline (S4e) and notation updated.

### Recomputed against the document

- Closed form vs evaluated R27: 2.0e-13 max relative over 400 states
  (document 1.9e-13 over 400); manifold residual 2.8e-13 (< 1e-11).
- Continuation floors (1 start, 600 it): 1.35e-6, 1.36e-5, 1.37e-4, 4.16e-4,
  1.49e-3, 5.78e-3 against the document's 1.28e-6, 1.28e-5, 1.30e-4, 3.99e-4,
  1.43e-3, 5.78e-3 (5 starts): largest departure 6 %, floor/eps spread 1.40,
  every rung `iteration_limit` with R27 dominant, rank 22 of 25, condition
  <= 20. Anode-only p = (0,0,0,eps): 6 of 6 converged, residuals 1e-16 to
  2e-13, phi_4 - Ua = 0.
- Published-state misfit 1.47e-3 (document 1.47e-3; ledger 1.4866e-3);
  relaxed root 1.18 V below the anode with all rows to 2e-16, infeasible
  (document 1.18 V at 300 V / 1 A); anode-fall coefficient 2.0.
- Probe 13/80 from the frozen protocol equals the document's reproduction.
- Legacy blob `8eeca9c6`: the three `+IE` cusp terms sit on line 136 and the
  anode electron term `(x(9)-Ua+x(13))` on line 137; the document names
  “line 137”, so the generator requires the documented line to fall inside the
  two-line span and the paper reports both lines.

### Validation

- Before the commit `check_paper.py` reported exactly the fail-closed
  “accepted manifest is not committed at HEAD”; after commit `bb7c25b2` it
  passed (about 90 s, of which ~35 s is the recomputation).
- `python -m unittest discover -s paper/tests`: 111 tests (83 existing + 28
  new); the only pre-commit failure was that same fail-closed check; 111 OK
  after the commit.
- `python paper/scripts/verify_reproducible_build.py`: two clean builds
  byte-identical, `paper/build/manuscript.pdf` 493,839 bytes, SHA-256
  `6ac978b29ab899092e0427c44bbe5f26f8608190589877b50bd729f16ede8a85`, 27
  pages, no LaTeX errors, undefined references, warnings or overfull boxes.
- Pages rendered with MiKTeX `pdftoppm -r 80` to `%TEMP%\paper-closure-pages\`
  (`p-01.png` … `p-27.png`) and inspected: abstract sentence and five-line
  date p. 1; Section 10 opens p. 18 with the displayed closed form (eq. 8) on
  p. 19; Table 13 (continuation ladder) p. 20; Table 14 (global search) p. 21;
  Discussion paragraph with CLM-043 and CLM-044 p. 24; Limitations p. 25.
- Corrections during integration: nested `$` from a `sci` macro inside a
  caption's math (`\le\FccResidualTolerance`) gave “Missing $ inserted”;
  moved the macro outside math. A 24 pt overfull from a typewriter ledger
  expression with `\allowbreak` only; replaced by `\hspace{0pt plus 1.5pt}`
  after every operator so the line has both break points and stretch.
- The user brief said the legacy solver “accepted exit-flag-4 floors”; the
  audit and `Performance_est.m:91-128` say flags 1–3 are accepted by status
  alone and flag 4 is rejected (`HEMP_solver.m:64` discards the residual norm,
  `TolFun=1e-50`). The claim uses the audit's wording, macro-bound.

## 2026-09-03 - Orbit wall-loss geometry screening v1 admitted to the claim matrix and manuscript

### Scope

- Worktree `C:\Users\Angus\Desktop\projects\uni-project-paper-geo`, branch
  `paper/geometry-screening-claim` from `origin/feat/sota-foundation`
  (`22e2156b`), LF verified (`git ls-files --eol`: no `w/crlf`). Paper-owned
  paths and `modern/docs/workstreams/paper-*` only; `results/**`, the frozen
  preregistration files and `FYP/` untouched; no GPU work.
- Evidence: `modern/experiments/orbit_wall_loss_geometry_screening_v1`,
  preregistered `c86bfca3`, recorded `ab7c2897` (`record orbit wall-loss
  geometry screening v1 result`), merged into `feat/sota-foundation` by
  `22e2156b`. The results tree first exists at `ab7c2897` and the merge adds
  nothing under the experiment or the visualization, so `ab7c2897` is both the
  evidence revision and the dashboard revision (the dashboard HTML was
  regenerated from the sealed bundle in the same commit).

### Evidence level and gate

- Gate `GATE-WALL-LOSS-GEOMETRY-SCREENING-V1`, kind `numerical-screening`,
  `opens_level: null`, at a fourth recorded outcome
  `accepted-screening-dataset` (added to `SCREENING_OUTCOMES`, the kind
  description in `result-gates.json`, notation and README). Justification: the
  sealed campaign status is `accepted_screening_dataset`, a test-particle
  dataset over a design space on L1a screening fields that are not
  P2-qualified; the existing outcomes name a field-only screening, a null and a
  characterization. `accepted` keeps meaning admitted as recorded and never
  reads as the physical-orbit evidence `GATE-WALL-LOSS-V4` admits.
- New manifest type `paper-orbit-screening-manifest` 1.0 (30 required roles,
  78 required metrics = 68 mapped + 10 policy). Manifest
  `paper/evidence/manifests/wall-loss-geometry-screening-v1.json`: 67 source
  files bound by Git blob and SHA-256 at `ab7c2897` (top-level bundle
  artifacts, the four representatives' three summaries, 2N handoff, endpoint
  table, orbit artifact and sidecar, bore field and field evidence, the six
  extreme designs' 2N summaries, the four frozen preregistration files whose
  blobs equal those at `c86bfca3`), dashboard generator/template/HTML bound at
  `ab7c2897` and equal to the checkout by LF-normalised SHA-256, 58 gate
  metric constraints, `posthoc_audit: null` with a note (no audit exists; the
  bundle needs no end-of-line tolerance, orbit_mc 1.7 writes LF sidecars).

### Added

- `paper/scripts/generate_wall_loss_geometry_screening_v1_evidence.py`:
  verifies all 2,835 manifest files byte for byte with their sidecars,
  requires the frozen files to equal the sealed copies and the same blob at
  preregistration and record commits, cross-checks the dataset against every
  per-case summary, handoff hash and orbit sidecar (and the representatives'
  gzipped endpoint tables), recomputes every reported, per-case and per-cell
  Wilson-95 interval operation for operation (exact equality), recomputes the
  convergence flags and the least/most ordering, cross-checks the committed
  dashboard payload (identity, headline, every per-design estimate and
  convergence flag, gate and consumer counts), and writes 271 `\Wlg...`
  macros (121 derived with derivation and inputs, incl. Spearman rank
  correlations), four `\ArtifactClaim` tables (dataset summary and
  convergence; least/most designs with sealed geometry; per-cell distribution
  and saturation; termination classes with escape sub-classes), evidence file
  and sidecar. ~20 s per run.
- Claims CLM-045 (abstract), CLM-046 (execution, wall-hit range/median,
  convergence, extremes; `non_claims`), CLM-047 (tables), CLM-048 (reflections
  in every design, escapes and sub-classes, per-cell means, 94/384 saturated
  at one and 0 at zero; bound to the v4 manifest too), CLM-049 (geometry
  association, stated as observation: extremes' lengths/radii and ranks,
  Spearman rho with length -0.05, radius -0.12, pitch +0.36, stage count
  -0.31, minimum mirror ratio +0.35, mu variation -0.37, reflection -0.79),
  CLM-050 (first consumer of the coupling-v4.2 export; 96/96 handoffs; v4
  export as labelled reference row), CLM-051 (boxed scope), CLM-052
  (Discussion interpretation: the mirror-picture statement is field-specific;
  the screening is the geometry-to-wall-loss bridge at screening tier;
  design-dependent optimisation is future work). CLM-016 amended so the v4
  scope no longer says "no consumer model has ingested it" but records the
  labelled reference-row consumption; `paper/sections/wall-loss-v4.tex` and
  its standalone driver updated accordingly.
- `manuscript.tex`: `\GeometryScreeningEvidenceRevision`, preamble `\input`,
  sixth date line, abstract sentence, contribution list, evidence-boundary
  paragraph, Section 11 "Preregistered wall-loss screening across the
  accepted sweep geometries" with `\input{sections/wall-loss-geometry-screening-v1.tex}`
  (subsection 11.1), sentence after the L1 gate box, Discussion paragraph
  retitled and extended with CLM-052, the MDO paragraph's "planned bridge"
  sentence rewritten (realised at screening tier; consumer optimisation future
  work), Limitations, data availability, Conclusion.
- Section `paper/sections/wall-loss-geometry-screening-v1.tex` (Method;
  Results; Reflections, escapes and cell structure; Geometry and the wall-hit
  probability; Coupling consumer; Scope with the boxed claim boundary) and the
  standalone driver; `check_paper.py` (`GEOMETRY_SCREENING_METRIC_MACROS`,
  `GEOMETRY_SCREENING_POLICY_METRICS`, schema type, `_check_geometry_screening`,
  renderer, required Section 11, trackables, outcome set); tests
  `test_wall_loss_geometry_screening_admission.py` (17) and
  `test_wall_loss_geometry_screening_evidence.py` (11); README, author
  checklist, supplementary outline (S4f) and notation updated.

### Numbers verified against the bundle

- 96 designs (25 non-dominated + 71 extension; 0 excluded), 196 cases,
  100,352 orbits, 6,664 validators / 0 failures, 196/196 sealed and replayed,
  96/96 converged (largest N to 2N change 0.0059, mean 3.5e-4, 83 designs
  unchanged), refined-N sensitivity of the 4 representatives <= 0.0078; energy
  drift 0; 0 timeouts; 0 numerical failures.
- P(wall) 0.375-0.869 (median 0.702, mean 0.697); least 049/094/050, most
  091/021/043 (ordering recomputed and unique). Reflections in all 96 designs:
  32-282 of 512 at 2N (11,268 of 49,152, 22.9 %), 22,904 of 100,352 over
  every case (22.8 %). Escape 0-0.215 (median 0.069; 8 designs without an
  escape): 1,635 anode plane, 1,127 exit plane, 862 divergent radial, 0
  unclassified at 2N. Per-cell means 0.65/0.82/0.77/0.55; 94 of 384
  design-cells at 1.0, none at 0.0. mu-variation medians 0.11-0.47.
- Field provenance: 96/96 identity proven; interpolation rms <= 0.87 %;
  stored-map agreement of the 4 representatives 9.3e-21 Wb / 2.8e-15 T
  against tolerances 1e-15 Wb / 1e-9 T; the refined re-solve and
  cross-resolution diagnostic (<= 0.66 %) exist for the 4 representatives
  ONLY (`include_refined` is true for representatives; the check passes
  vacuously elsewhere). The paper states this; the experiment README's "for
  every design" overstates it.
- Consumer: 96/96 handoffs `consumed_verified_handoff`, 7 checks each; v4
  export consumed as reference row (0.645 [0.602, 0.685], 512 trials,
  `NUMERICAL_P2_QUALIFIED`, not in the screening set).
- The brief's "longer channels / larger wall radius lose least" is NOT a
  population trend: Spearman rho(P_wall, L) = -0.05 and rho(P_wall, r_w) =
  -0.12 over the 96 designs; only the three least-loss designs sit at long
  lengths (ranks 42-92). The claim reports the extremes and the correlations
  as observations and states that none is a design rule.

### Validation

- Before the commit `check_paper.py` reported exactly the fail-closed
  "accepted manifest is not committed at HEAD"; after the first commit
  (`914af6a1`, later amended with these notes) it passed (about 60 s) and
  passed again on the amended commit. `unittest discover -s paper/tests`: 139 tests (111
  existing + 28 new), 139 OK after the commit (the only pre-commit failure was
  that same fail-closed check inside `test_paper_checks`).
- `verify_reproducible_build.py`: two clean builds byte-identical,
  `paper/build/manuscript.pdf` 532,205 bytes, SHA-256
  `67a531f9562f785f2eef7a5c6f053c3f2c4cb2918c4e80c240735939670fd720`, 33 pages,
  no LaTeX errors, undefined references or overfull boxes.
- Pages rendered with MiKTeX `pdftoppm -r 80` to `%TEMP%\paper-geo-pages\`
  (`p-01.png` ... `p-33.png`) and inspected: abstract sentence p. 2; Section
  11 opens p. 22, subsection 11.1 p. 23, Results and Table 15 p. 24-25,
  Tables 16-17 p. 26, Table 18 and the boxed scope p. 27; L1 note p. 28;
  Discussion paragraph with CLM-052 p. 28-29; Limitations p. 30-31.
- One correction during integration: the extremes table overflowed by 6.5 pt
  at `\scriptsize` (caught by the standalone compile); the case and interval
  `p{}` columns were narrowed (2.9 -> 2.7 cm, 2.6 -> 2.45 cm) and
  `\tabcolsep` reduced to 2 pt.
