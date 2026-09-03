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
definitions, never as proof that no such design exists. One further
`numerical-campaign` gate, `GATE-MDO-L0-V1`, admits the preregistered robust
multi-objective optimisation campaign of the L0 model
`modern/experiments/mdo_l0_campaign_v1` (results
`c553124b7393890d8ee9c6fc022e536c8a1fd35e`, preregistration `4898d0fd…`,
dashboard `e642f38c…`) as optimiser evidence under the declared closure CL-1:
it makes no thruster-performance claim. One `analytic-consistency` gate,
`GATE-FOUR-CELL-CLOSURE-V1`, admits the four-cell power-balance closure
analysis (`modern/docs/workstreams/global-plasma-closure-analysis.md` and
`spec/plasma/equation-ledger.json#global_row_consistency` at
`266d8a99ce75fe35b4870d5d046c9069d7b26c0b`, verified unchanged at `ba6875f6…`):
on the manifold of the corrected ledger the global power row reduces to
`2 (j_e3 (1-p4) + I4)(phi_4 - Ua) + EI (p1 j_e0 + p2 j_e1 + p3 j_e2)`, both
terms are non-negative, so the equation set has no admissible root for any
positive interior cusp probability; the checker recomputes the verification
from the bound `cft_revival.plasma` package at every run, the proposed
correction stays `PROPOSED_NOT_ACCEPTED`, and nothing follows about the
physical thruster. A fourth `numerical-screening` gate,
`GATE-WALL-LOSS-GEOMETRY-SCREENING-V1`, admits the orbit wall-loss geometry
screening `modern/experiments/orbit_wall_loss_geometry_screening_v1` (results
`ab7c28977963822b2ad6eac451d2bafef5185e6c`, preregistration `c86bfca3…`,
dashboard at the same record commit) at a fourth recorded outcome,
`accepted-screening-dataset`: 100,352 collisionless test-particle electron
orbits in the re-solved L1a screening fields of all 96 accepted sweep-v2
designs (P(wall) 0.375–0.869, median 0.702; reflections in every design;
96/96 converged under timestep halving; every handoff consumed by the first
consumer of the coupling export format). The fields are not P2-qualified, so
the dataset is surrogate and optimisation input under its label and never
accepted physical-orbit, plasma or performance evidence. The checked evidence is
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
performance content, or opens `GATE-L1`. The optimisation campaign is
classified
`l0_model_robust_multiobjective_optimisation_under_declared_input_uncertainty_not_thruster_performance`:
its estimands are properties of the optimisers (hypervolume per budget, paired
comparisons, seed variance, Pareto sizes) and of the declared evaluation chain
(robust-versus-nominal fronts, sensitivity to the cusp prior); every number is
conditional on the declared closure and priors; geometry is excluded because no
geometry-to-L0 map survives the audit; it opens none of L1--L3. The closure
analysis is classified
`analytic_consistency_of_the_corrected_four_cell_power_balance_not_thruster_physics`:
it is a statement about an equation set, not about the thruster; the reading
that the legacy performance values were residual-floor artefacts is a labelled
interpretation in the Discussion, and no value of the unavailable legacy run is
claimed or recomputed. The geometry screening is classified
`SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`: its launch cells are protocol
positions, its geometry associations (rank correlations with chamber length,
wall radius, stage pitch and stage count) are observations of one launch
design and not a design rule, its refined-field diagnostic exists for four
representatives only, and no surrogate or optimisation consuming it is
admitted.

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
  (`physics-level` gates), the accepted `numerical-campaign` gates
  `GATE-WALL-LOSS-V4` and `GATE-MDO-L0-V1`, the four
  `numerical-screening` gates `GATE-L1A-SWEEP-V2`, `GATE-FOUR-CELL-V2`,
  `GATE-TOPOLOGY-CHAR-V1` and `GATE-WALL-LOSS-GEOMETRY-SCREENING-V1`, each
  carrying its `recorded_outcome`, and the `analytic-consistency` gate
  `GATE-FOUR-CELL-CLOSURE-V1`.
- `evidence/manifests/wall-loss-geometry-screening-v1.json` — typed screening
  manifest (`paper-orbit-screening-manifest` 1.0) binding the sealed bundle's
  top-level artifacts, the representatives' per-case summaries, handoffs,
  endpoint tables, orbit artifacts and sidecars, bore fields and field
  evidence, the six extreme designs' summaries, the frozen preregistration
  files and the results dashboard by Git blob and SHA-256 at the record
  commit, plus the metrics the checker compares with the raw artifact values
  behind the `\Wlg...` macros.
- `evidence/wall-loss-geometry-screening-v1.json`,
  `generated/wall-loss-geometry-screening-v1.tex`,
  `sections/wall-loss-geometry-screening-v1.tex` — hash-bound evidence file,
  generated macros with four `\ArtifactClaim` tables, and the admitted
  macro-only subsection bound once by `\input` from Section 11 of
  `manuscript.tex`.
- `evidence/manifests/four-cell-closure.json` — typed analysis manifest
  (`paper-analytic-consistency-manifest` 1.0) binding the analysis document,
  the ledger, the five `cft_revival.plasma` files, three pinning test files,
  the frozen MDO protocol, `FYP/Power_B_EQs.m` (lineage), `AUDIT.md` and
  `REFERENCES.md` by Git blob and SHA-256 at the analysis revision, the
  executed package digests, the recomputation protocol and tolerances, and the
  metrics the checker compares with the values behind the `\Fcc...` macros.
- `evidence/four-cell-closure.json`, `generated/four-cell-closure.tex`,
  `sections/four-cell-closure.tex` — evidence file (documented macros bound by
  pointer or fixed pattern; recomputed macros with their protocol), generated
  macros with two `\ArtifactClaim` tables, and the admitted macro-only
  subsection bound once by `\input` from Section 10 of `manuscript.tex`.
- `evidence/manifests/mdo-l0-v1.json` — typed campaign manifest
  (`paper-mdo-campaign-manifest` 1.0) binding every consumed bundle file by
  Git blob and SHA-256 at the results revision, the frozen preregistration
  files, the results dashboard at its own revision, and the metrics the
  checker compares with the raw artifact values behind the `\Mdo...` macros.
- `evidence/mdo-l0-v1.json`, `generated/mdo-l0-v1.tex`,
  `sections/mdo-l0-v1.tex` — hash-bound evidence file, generated macros with
  three `\ArtifactClaim` tables, and the admitted macro-only subsection bound
  once by `\input` from Section 9 of `manuscript.tex`.
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

## Admitted numerical campaign: MDO L0 campaign v1

`paper/scripts/generate_mdo_l0_v1_evidence.py` reads the sealed results bundle
of `modern/experiments/mdo_l0_campaign_v1` (137 files verified byte for byte
against `results/manifest.json`; no end-of-line tolerance exists or is
granted), requires the frozen `protocol.json`, `authorities.json` and
`shakedown.json` to equal the sealed copies, cross-checks the committed results
dashboard (`modern/visualization/mdo-l0-campaign-v1.html` and its generator,
bound at `e642f38c`) against the same bundle, and writes
`paper/evidence/mdo-l0-v1.json` (every `\Mdo...` macro with its artifact path,
JSON pointer, formatter and SHA-256, or its derivation and inputs),
`paper/generated/mdo-l0-v1.tex` (macros plus three tables wrapped in
`\ArtifactClaim`: hypervolume per optimiser and seed, robust versus nominal
fronts, alternative priors and fixed scenarios) and the provenance sidecar.
The subsection `paper/sections/mdo-l0-v1.tex` renders numbers only through
macros; its results, robust-versus-nominal, sensitivity and scope statements
are exact `\EvidenceClaim` bodies (CLM-030, CLM-032, CLM-033, CLM-034), the
abstract sentence is CLM-029 and the labelled Discussion interpretation is
CLM-035. The gate reuses the `numerical-campaign` kind because the campaign is
one accepted campaign on a declared component model (L0 under the declared
closure CL-1); it is optimiser evidence, not performance evidence, and
`opens_level` is null.

`check_paper.py` regenerates the three generated files at every run and fails
closed on any byte difference, any artifact hash mismatch, any manifest metric
that differs (in value or type) from the raw artifact value behind its macro,
any policy metric off its fixed value, a dashboard checkout that differs from
the blob bound at the dashboard revision, a results tree changed since the
evidence revision, a frozen file changed since preregistration, a literal digit
or undefined macro in the section, a classification or closure macro that does
not render its string, a missing registered non-claim, or a
`\MdoEvidenceRevision` macro that does not spell the manifest revision.

```powershell
python paper/scripts/generate_mdo_l0_v1_evidence.py
python -m unittest discover -s paper/tests -p "test_mdo*" -v
```

## Admitted analytic consistency result: four-cell power-balance closure

`paper/scripts/generate_four_cell_closure_evidence.py` binds the analysis
document, the equation ledger, the `cft_revival.plasma` package, the three
pinning test files, the frozen MDO protocol (blob equal at the preregistration
commit) and the legacy `FYP/Power_B_EQs.m` blob at the analysis revision
`266d8a99`, requires the checkout's package to equal the bound blobs
(LF-normalised SHA-256), and then RECOMPUTES the verification with that
package: the closed form `global_row_closed_form` against the full residual
over a 400-state seeded sample (max relative difference recorded as
`\FccClosedFormRelDiff`), the R00--R26 manifold residual, the anode-fall
coefficient, the continuation ladder `p = eps (1,1,1,1)` at 300 V / 1 A through
the production solver (one start, 600 iterations), the anode-only closures
`p = (0,0,0,eps)` (five starts), the published-state misfit, one relaxed root by
bisection and the Jacobian rank at every floor. It refuses to write anything if
a recomputed number departs from the analysis document beyond the declared
tolerance (`TOLERANCES`). The `13/80` probe is read from the frozen MDO protocol
disclosure with the same fixed pattern the optimisation generator uses, and the
document's reproduction must agree; the differential-evolution and
random-start searches are documented values, not recomputed (they need SciPy
and minutes of solver time). Recomputed values are recorded to a declared
number of significant digits because the floors are solver-stall values.

The subsection `paper/sections/four-cell-closure.tex` renders numbers only
through `\Fcc...` macros; its result, sub-region/continuation, attribution,
proposed-correction and scope statements are exact `\EvidenceClaim` bodies
(CLM-037, CLM-038, CLM-040, CLM-041, CLM-042), the abstract sentence is
CLM-036 and the two Discussion interpretations (legacy-study consequence;
the three chain findings read together) are CLM-043 and CLM-044. The closed
form is displayed in the manuscript's Section 10 with the coefficient and row
index bound to macros. The gate kind `analytic-consistency` admits the
derivation and its numerical verification as recorded, opens no physics level
and accepts no correction.

```powershell
python paper/scripts/generate_four_cell_closure_evidence.py
python -m unittest discover -s paper/tests -p "test_four_cell*" -v
```

## Admitted screening dataset: orbit wall-loss geometry screening v1

`paper/scripts/generate_wall_loss_geometry_screening_v1_evidence.py` reads the
sealed results bundle of
`modern/experiments/orbit_wall_loss_geometry_screening_v1` (2,835 files
verified byte for byte against `results/manifest.json`, every artifact paired
with a manifest-bound sidecar; no end-of-line tolerance exists or is granted),
requires the frozen `protocol.json`, `authorities.json`, `shakedown.json` and
`design-authorities.json` to equal the sealed copies and to carry the same
blob at the preregistration and record commits, cross-checks the dataset
against all 196 per-case summaries, handoffs and orbit sidecars (and the
representatives' gzipped endpoint tables), recomputes every reported,
per-case and per-cell Wilson interval operation for operation, cross-checks
the committed results dashboard
(`modern/visualization/wall-loss-geometry-screening-v1.html`, its generator
and template at `ab7c2897`) against the same bundle, and writes
`paper/evidence/wall-loss-geometry-screening-v1.json` (every `\Wlg...` macro
with its artifact path, JSON pointer, formatter and SHA-256, or its
derivation and inputs), `paper/generated/wall-loss-geometry-screening-v1.tex`
(macros plus four tables wrapped in `\ArtifactClaim`: dataset summary and
convergence, least and most wall-loss designs with sealed geometry, per-cell
distribution, termination classes) and the provenance sidecar. Derived macros
include Spearman rank correlations of the wall-hit probability with sealed
geometry and field descriptors; the section states them as observations and
the claim records forbid reading them as a design rule.

The subsection `paper/sections/wall-loss-geometry-screening-v1.tex` renders
numbers only through macros; its results, reflection/escape/cell,
geometry-association, consumer and scope statements are exact
`\EvidenceClaim` bodies (CLM-046, CLM-048, CLM-049, CLM-050, CLM-051), the
abstract sentence is CLM-045 and the labelled Discussion interpretation (the
wall-loss campaign's mirror-picture statement is field-specific; the
screening is the geometry-to-wall-loss bridge at screening tier) is CLM-052.
The gate reuses the `numerical-screening` kind at a fourth outcome value,
`accepted-screening-dataset`, because the study screens a design space on
linear-vacuum fields that are not P2-qualified and its sealed status is a
dataset accepted as screening input; `opens_level` is null. The first
consumer of the coupling-v4.2 export format is recorded here, so the
wall-loss campaign's scope claim (CLM-016) now says the export was ingested
only as a labelled reference row.

`check_paper.py` regenerates the three generated files at every run (which
re-verifies the whole bundle and recomputes the Wilson intervals) and fails
closed on any byte difference, any artifact hash mismatch, any manifest metric
that differs (in value or type) from the raw artifact value behind its macro,
any policy metric off its fixed value, a dashboard checkout that differs from
the blob bound at the dashboard revision, a results tree changed since the
evidence revision, a frozen file changed since preregistration, a recorded
outcome that disagrees anywhere, a literal digit or undefined macro in the
section, a classification, recorded-outcome or campaign-status macro that does
not render its string, a missing registered non-claim, an interpretation claim
inside the results section, or a `\GeometryScreeningEvidenceRevision` macro
that does not spell the manifest revision.

```powershell
python paper/scripts/generate_wall_loss_geometry_screening_v1_evidence.py
python -m unittest discover -s paper/tests -p "test_wall_loss_geometry*" -v
```
