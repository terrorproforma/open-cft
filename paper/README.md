# Updated-paper workstream

This directory is a reproducible, evidence-gated manuscript workstream for the
modern CFT revival. It is intentionally separate from the preserved `FYP/`
snapshot and from shared modern documentation.

## Evidence boundary

All present-tense L0 result claims are limited to committed revision
`41bf909127dc021abe8078fd77a98aa3a6e4cf33`. One further result is admitted:
the preregistered collisionless test-particle electron wall-loss campaign
`modern/experiments/cft_orbit_wall_loss_v4`, whose sealed results bundle is
committed at `6922a3cf97d261735266aa1a5a0c0c9683e021ca` (preregistration
`757e365f9f667620c7610663574294c3b71e1f51`, post-hoc audit
`258f69b2f4bc081c6f571251ce2ad76d49ddab0a`) and admitted through the
numerical-campaign gate `GATE-WALL-LOSS-V4`. The checked evidence is enumerated
in `evidence/claims.json`. Concurrent or later work is not publishable merely
because files exist in a working tree: a planned section opens only when its
gate in `evidence/result-gates.json` names an accepted, committed manifest.

The manuscript prohibits classifying L0 as one-dimensional, geometrically
predictive, physically calibrated, or experimentally validated. No comparative
GPU-performance validation exists. L0 is an algebraic, conservation-reduced
operating-point baseline with externally supplied closures. The wall-loss
campaign is classified `collisionless_prescribed_field_test_particle_wall_loss_not_pic`:
it is not particle-in-cell, not self-consistent, not thruster performance, not
validated, and its pooled wall-hit fraction is an equal-weight design average
of a bimodal per-cell result, not a loss rate. It opens none of L1--L3.

## Reproduce checks and build

Only Python's standard library is required for policy checks:

```powershell
python paper/scripts/generate_tables.py
python paper/scripts/check_paper.py
python -m unittest discover -s paper/tests -v
```

The build wrapper installs nothing. It requires existing `pdflatex` and
`bibtex`, cleans `paper/build/`, applies the pinned `SOURCE_DATE_EPOCH`, and
records tool versions. The reproducibility wrapper performs two clean builds
and requires identical PDF SHA-256 values:

```powershell
python paper/scripts/build.py
python paper/scripts/verify_reproducible_build.py
```

The PDF and intermediate files are written below `paper/build/`. If no TeX
engine exists, the wrapper reports that condition after running policy checks;
it does not modify the environment.

## Workstream map

- `manuscript.tex` — buildable article scaffold and current verified result.
- `references.bib` — machine-readable publication metadata and exact DOI
  fields where a DOI exists.
- `evidence/claims.json` — claim-to-evidence matrix pinned to a Git revision.
- `evidence/l0-run-manifest.json` — strict committed-source and accepted-HTML
  binding for the current L0 evidence.
- `evidence/manifest-schemas.json` — recognized manifest types, versions,
  file roles, and required metrics.
- `evidence/result-gates.json` — explicit L1/L2/L3 admission criteria
  (`physics-level` gates) and the accepted `numerical-campaign` gate
  `GATE-WALL-LOSS-V4`.
- `evidence/manifests/wall-loss-v4.json` — typed campaign manifest binding
  every results-bundle file by Git blob and SHA-256, the frozen
  preregistration files, the post-hoc audit, and the metrics that the checker
  compares with the raw artifact values behind the evidence macros.
- `evidence/wall-loss-v4.json` — hash-bound evidence file: every `\Wlf...`
  macro with its artifact path, JSON pointer, formatter and SHA-256.
- `sections/wall-loss-v4.tex` — the admitted results subsection, bound once
  by `\input` from `manuscript.tex`; it renders numbers only through macros.
- `evidence/figure-table-contract.json` — source and provenance contract for
  every planned display item.
- `evidence/submission-gates.json` — author identity and human approval gates.
- `generated/` — trackable deterministic table sources and provenance
  sidecars; only `build/` is locally ignored.
- `notation.md` — notation, fidelity names, and prohibited equivalences.
- `author-checklist.md` — pre-submission evidence and reporting checks.
- `supplementary-outline.md` — response-ready supplement and reviewer package.
- `scripts/` and `tests/` — build, lint, and regression checks.
- `../modern/docs/workstreams/paper-devlog.md` and
  `../modern/docs/workstreams/paper-learning-ledger.md` — paper-specific
  chronological and learning loops.

## Integrating future results

1. Create a recognized JSON manifest under `paper/evidence/manifests/`.
2. Bind every required input/output to its Git blob and SHA-256 at a resolvable
   evidence revision, and satisfy every gate-specific metric.
3. Commit the manifest and bound artifacts before setting the gate to
   `accepted`; the checker rejects uncommitted or ancestry-inconsistent
   acceptance.
4. Add an exact `authorized_tex` claim record and use
   `\EvidenceClaim{ID}{exact registered text}`, or register a generated
   figure/table artifact and sidecar. Detached identifiers do not authorize
   prose.
5. Replace the closed `\EvidenceGate{...}{...}` only after acceptance.
6. Regenerate display artifacts, run policy/adversarial tests, and require two
   byte-identical clean PDF builds.

A preregistered numerical campaign that is not a physics level follows the
same steps with a `numerical-campaign` gate: its typed manifest lives under
`paper/evidence/manifests/`, is committed before the gate is `accepted`,
declares `opens_level: null`, and is cross-checked by a campaign-specific
checker in `check_paper.py`. Section files bound through
`\input{sections/...}` are flattened into the manuscript before every claim,
citation and prose check.

## Admitted numerical campaign: wall-loss v4

`paper/scripts/generate_wall_loss_v4_evidence.py` reads the sealed results
bundle of `modern/experiments/cft_orbit_wall_loss_v4` (verified against
`results/manifest.json`, bound to the committed results revision) and writes
`paper/evidence/wall-loss-v4.json` (every macro value with its artifact path,
JSON pointer, formatter and SHA-256), `paper/generated/wall-loss-v4.tex`
(`\Wlf...` macros plus two generated tables, each wrapped in `\ArtifactClaim`)
and `paper/generated/wall-loss-v4.provenance.json`. The subsection
`paper/sections/wall-loss-v4.tex` renders numbers only through those macros and
states its results, verification facts and scope limits as exact
`\EvidenceClaim` bodies registered in `evidence/claims.json` (the campaign
result, generated-table, verification and scope-limitation records, plus the
abstract summary and the labelled discussion interpretation). It is `\input`
once into `manuscript.tex`; `paper/sections/wall-loss-v4-standalone.tex`
still compiles it on its own.

`check_paper.py` regenerates the three generated files from the bundle at
every run and fails closed if any byte differs, if any evidence artifact hash
differs from the bundle on disk, if any manifest metric differs from the raw
artifact value behind its macro, if the section types a literal digit or uses
an undefined macro, if the classification macro does not render the
classification string, if a registered non-claim is missing from the section,
or if the manuscript's `\WallLossEvidenceRevision` macro does not spell the
manifest revision.

```powershell
python paper/scripts/generate_wall_loss_v4_evidence.py
python -m unittest discover -s paper/tests -p "test_wall_loss*" -v
```
