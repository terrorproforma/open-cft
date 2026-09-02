# Updated-paper workstream

This directory is a reproducible, evidence-gated manuscript workstream for the
modern CFT revival. It is intentionally separate from the preserved `FYP/`
snapshot and from shared modern documentation.

## Evidence boundary

All present-tense result claims are limited to committed revision
`41bf909127dc021abe8078fd77a98aa3a6e4cf33`. The checked evidence is enumerated
in `evidence/claims.json`. Concurrent or later work is not publishable merely
because files exist in a working tree: a planned section opens only when its
gate in `evidence/result-gates.json` names an accepted, committed manifest.

The manuscript prohibits classifying L0 as one-dimensional, geometrically
predictive, physically calibrated, or experimentally validated. No comparative
GPU-performance validation exists. L0 is an algebraic, conservation-reduced
operating-point baseline with externally supplied closures.

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
- `evidence/result-gates.json` — explicit L1/L2/L3 admission criteria.
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
## Draft evidence pending integration: wall-loss v4

`paper/scripts/generate_wall_loss_v4_evidence.py` reads the sealed results
bundle of `modern/experiments/cft_orbit_wall_loss_v4` (verified against
`results/manifest.json`, bound to the committed results revision) and writes
`paper/evidence/wall-loss-v4.json` (every macro value with its artifact path,
JSON pointer, formatter and SHA-256), `paper/generated/wall-loss-v4.tex`
(`\Wlf...` macros plus two generated tables) and
`paper/generated/wall-loss-v4.provenance.json`. The draft subsection
`paper/sections/wall-loss-v4.tex` cites those macros only and is compiled on
its own by `paper/sections/wall-loss-v4-standalone.tex`; it is **not**
`\input` into `manuscript.tex`, whose evidence boundary and claim matrix do
not yet admit the campaign. Integration follows the steps above: a claim
record, a gate or manifest entry, and only then the manuscript binding.

```powershell
python paper/scripts/generate_wall_loss_v4_evidence.py
python -m unittest discover -s paper/tests -p "test_wall_loss*" -v
```
