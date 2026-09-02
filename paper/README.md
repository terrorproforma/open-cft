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
numerical-campaign gate `GATE-WALL-LOSS-V4`. Three preregistered, single-execution
L1a field-only topology-screening studies are admitted at exactly their recorded
outcomes through `numerical-screening` gates: the accepted geometry sweep
`modern/experiments/l1a_geometry_sweep_v2` (results
`f30cb42ec4a8633bf634a3d32ffa5b11f66be97a`, preregistration `092f5fae…`,
post-hoc EOL audit `9e68df21…`; `GATE-L1A-SWEEP-V2`, `accepted-screening`), the
four-cell topology search `modern/experiments/four_cell_topology_search_v2`
(results `7120e8edcb74c02c1df968c730d1f93b3758b4e1`, preregistration
`d6317910…`, post-hoc EOL audit `605be5ce…`; `GATE-FOUR-CELL-V2`,
`preregistered-null`: 0 of 128 candidates stable under the frozen cusp/cell
definition) and the developmental characterization
`modern/experiments/cft_topology_characterization_v1` (results
`3ce6c546194e1d3e943d0b3d0951d03e15e354d9`, preregistration `af88470b…`;
`GATE-TOPOLOGY-CHAR-V1`, `recorded-characterization`: 0 stable eligible cusps
or cells over 56 designs). A null is admitted as a null under its frozen
definitions, never as proof that no such design exists. The checked evidence is
enumerated in `evidence/claims.json`. Concurrent or later work is not
publishable merely because files exist in a working tree: a planned section
opens only when its gate in `evidence/result-gates.json` names an accepted,
committed manifest.

The manuscript prohibits classifying L0 as one-dimensional, geometrically
predictive, physically calibrated, or experimentally validated. No comparative
GPU-performance validation exists. L0 is an algebraic, conservation-reduced
operating-point baseline with externally supplied closures. The wall-loss
campaign is classified `collisionless_prescribed_field_test_particle_wall_loss_not_pic`:
it is not particle-in-cell, not self-consistent, not thruster performance, not
validated, and its pooled wall-hit fraction is an equal-weight design average
of a bimodal per-cell result, not a loss rate. It opens none of L1--L3. The
topology-screening studies use linear-vacuum L1a equivalent-current fields
(no permanent-magnet or nonlinear-iron material model); their axis cusps are
sampled-axis descriptors, their mirror ratios are screening QoIs, and none of
them demonstrates a stable multi-cell wall-cusp topology, claims plasma or
performance content, or opens `GATE-L1`.

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
  (`physics-level` gates), the accepted `numerical-campaign` gate
  `GATE-WALL-LOSS-V4`, and the three `numerical-screening` gates
  `GATE-L1A-SWEEP-V2`, `GATE-FOUR-CELL-V2` and `GATE-TOPOLOGY-CHAR-V1`, each
  carrying its `recorded_outcome`.
- `evidence/manifests/{l1a-sweep-v2,four-cell-v2,topology-characterization-v1}.json`
  — typed screening manifests (`paper-l1a-screening-manifest` 1.0) binding
  every bundle file by Git blob and SHA-256 at the results revision, the frozen
  protocol, the post-hoc EOL audit where one exists, lineage files (four-cell
  only; non-claims), and the metrics the checker compares with the raw
  artifact values behind the evidence macros.
- `evidence/{l1a-sweep-v2,four-cell-v2,topology-characterization-v1}.json` —
  hash-bound evidence files for the `\Swp...`, `\Fcn...` and `\Tch...` macros.
- `sections/{l1a-sweep-v2,four-cell-v2,topology-characterization-v1}.tex` —
  the admitted screening subsections, each bound once by `\input` from
  Section 8 of `manuscript.tex`; macro-only.
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

## Admitted topology screening: sweep v2, four-cell v2 null, characterization v1

`paper/scripts/generate_topology_screening_evidence.py` reads the three sealed
bundles, verifies each against its own manifest (sidecars, sealed canonical
payloads, protocol bindings), binds each to its committed results revision and
writes, per study, `paper/evidence/<key>.json`, `paper/generated/<key>.tex`
(macros plus two generated tables wrapped in `\ArtifactClaim`) and
`paper/generated/<key>.provenance.json`. Two bundles recorded one frozen-protocol
digest on a `core.autocrlf=true` checkout (the sweep's `protocol.json.sha256`
and the four-cell copy `results/preregistered-protocol.json`); the generator and
`check_paper.py` accept exactly those files through the audited rule of their
`POSTHOC_AUDIT.md` (`sha256(bytes.replace(LF, CRLF)) == recorded`, LF digest
as audited, recorded byte count) and require the digests to appear verbatim in
`protocol.py::EOL_AUDITED_SIDECARS` and `audit_sidecar_eol.py`. Every other
byte difference fails. The four-cell evidence additionally hash-binds lineage
records (the superseded proxy search and the two failed coupling-v4 criterion
validations) that the section quotes only inside a registered non-claim.

The `numerical-screening` gate kind admits a study at its recorded outcome:
`accepted-screening` for the sweep, `preregistered-null` for the four-cell
search (0 of 128 candidates stable under the frozen definition; not an
existence disproof) and `recorded-characterization` for the developmental
characterization (0 stable eligible cusps or cells over 56 designs). The
checker requires the outcome to agree between gate, manifest, evidence file and
generator, the section to render numbers only through macros, and the
Discussion claim on the multi-cell topology to be macro-bound.

```powershell
python paper/scripts/generate_topology_screening_evidence.py
python -m unittest discover -s paper/tests -p "test_topology_screening*" -v
```
