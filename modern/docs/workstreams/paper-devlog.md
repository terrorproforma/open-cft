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
