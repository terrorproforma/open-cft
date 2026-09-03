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
accepted physical-orbit, plasma or performance evidence. A third
`numerical-campaign` gate, `GATE-MDO-L0-V2`, admits the catalogue optimisation
campaign `modern/experiments/mdo_l0_campaign_v2` (results
`a003f766c330d4e5648844ba49cdf1c3a3ce3bc1`, preregistration `99914dc2…`,
dashboard `0ea33a7e…`) as optimiser evidence: the L0 model over the 96 screened
sweep designs (categorical catalogue index × operating point) under the
declared closure CL-1 with each design's per-cell test-particle wall-hit
posteriors as per-cusp survival factors, no surrogate (both fitted to the
dataset were rejected), 1440 evaluations, 12/12 binding integrity gates,
qLogNEHVI beating LHS 3/3 and NSGA-III 3/3 (counts, not significance), a pooled
robust front on catalogue designs 49, 50 and 94 (the three lowest screening
P(wall)), a pooled-probability closure whose front shares no design with it
(Jaccard 0.0), and the six disclosures of the v1 post-hoc audit closed by
protocol rules and binding gates. It makes no thruster-performance or design
claim; its design ranking is a property of the declared closure. A fifth
`numerical-screening` gate, `GATE-CUSP-TOPOLOGY-V3-1`, admits the cusp topology
search `modern/experiments/cusp_topology_search_v3_1` (results
`cec47f12f5909c5886424bf5d46ac20ce06f1ac5`, preregistration `1600cfd3…`,
dashboard `9abbd537…`) at a fifth recorded outcome,
`accepted-topology-screening`: the HEMP/DCFT literature definition of a wall
cusp (axis null, separatrix traced to the dielectric, wall cusp at the
intersection, cells between consecutive cusps) evaluated on 281 prescribed field
maps (the 96 re-solved L1a sweep designs, the 128 sealed four-cell candidates,
the 56 characterization cases and the P2-qualified divergent-exit-stack field),
281/281 stable under refinement (largest shift 33 µm), both held-out references
reproduced (56/56 characterization axis roots, 96/96 sweep axis nulls); wall-cusp
histogram 0:6 / 1:140 / 2:36 / 3:56 / 4:25 / 5:6 / 6:6 / 7:6; N−1 wall cusps for
83 of 96 sweep designs with the cusps at the inter-magnet gaps (four wall cusps
in 19, four cells in 47); exactly one wall cusp in every four-cell candidate
(weak even stages leave one axis null, so the frozen four-cell null follows from
the construction of both its definition and its source policy); three P2 cusps
at 6.028/12.000/17.972 mm within 31 µm of the kinetic workstream's axis-null
planes (a reported, ungated consistency reference). Its predecessor
`cusp_topology_search_v3` (preregistration `69159934…`, recorded
`assessment_rejection` at `8cbcdbe6…`, read-only audit `9fa6359a…`) is bound as
lineage and never cited for a number: its held-out reference kept only sealed
axis clusters with centroid radius exactly zero, a recording-layer defect. The
cusps and cells are geometric properties of field maps under a stated
definition, the mirror ratios are field ratios and never probabilities, no
plasma, confinement or performance claim follows, and the frozen-definition
nulls of Section 8 remain true; the Discussion (CLM-028, CLM-044) now says the
frozen wall-null definition was non-standard and that the cells exist under the
literature definition while their plasma physics stays undemonstrated. A sixth
`numerical-screening` gate, `GATE-L1A-SWEEP-V3`, admits the geometry sweep
`modern/experiments/l1a_geometry_sweep_v3` (results
`2cfe8223630fbef6bfe8099a5dcecaf4eb8c6b44`, preregistration `1923ef76…`,
dashboard `44d0c63c…`) at the recorded outcome `accepted-screening`, the outcome
of the sweep v2 whose box it widens: 128 scrambled-Sobol designs on a box that
extends the wall-radius-to-pitch ratio to r_w/L = 1.24 (x_w = πr_w/L up to 3.88)
plus the 96 accepted sweep-v2 designs re-solved as a held-out set, 224/224
resolved and stable, 11/11 binding gates, the six sweep-v2 metric gates verbatim,
the literature wall-cusp definition of the cusp topology search imported unchanged
and the Koch design ratio ρ reported at every wall cusp as a field ratio. 15 of 128
Sobol designs are HEMP-like (ρ ≥ 1.5 at every cusp; 0 of 77 below the
single-harmonic threshold x* = 1.937, 5/30, 4/13 and 6/8 in the three bands above
it), none of the 102 designs of the sweep-v2 region is (largest ρ 0.993), and the
preregistered hypothesis did not hold as preregistered (slope 0.689, R² 0.39, 70 %
in band, accuracy 0.72): I₁(x_w) is an upper envelope of the realised ratio
(ρ/I₁ median 0.80 at the 256 end cusps, 0.87 at the 109 interior cusps; realised
threshold x_w 2.34, r_w/L 0.745). The manifest binds the TWT/PPM literature review
(`beb4772c…`) with its read-only check script and committed output as the
definition and hypothesis source (the launch-position analysis of the geometry
screening's sealed orbits is read from that output in the Discussion), and the
sealed sweep-v2 manifest, the frozen topology-v3.1 protocol, the topology P2
record and the frozen wall-loss protocol as references. The declared iron pole
pieces are vacuum in the field, the material-aware confirmation the protocol
queues was not run, no HEMP-like design is a design recommendation, and the
Discussion (CLM-076) reads the result as interpretation: the legacy design space
could not contain a HEMP-like cusp because its parameterisation never varied the
ratio the HEMP criterion depends on. The same bound analysis re-scopes the
wall-loss campaign's zero reflections (CLM-016, CLM-017, CLM-044, CLM-052): the
P2 field is a PPM mirror field, the campaign's launch planes sit 0.5 mm from the
magnet centres where mirroring is impossible (the review's 7 launch cells within
0.17 pitch of a magnet centre produced 0–1 reflections per 128 against 32–88 for
the 9 cells 0.22–0.48 pitch away), so the zero count is a launch-position result
and the screening's reflections are the mirror reflections the field predicts;
the Limitations add that every electron of these fields is non-adiabatic at the
wall cusps (Mendel α 9.93–1190, ε 0.05–0.75, μ-variation medians ordered by ε),
so a per-cusp loss probability cannot be a loss-cone number. A seventh
`numerical-screening` gate, `GATE-WALL-LOSS-GEOMETRY-SCREENING-V2`, admits the
orbit wall-loss geometry screening v2
`modern/experiments/orbit_wall_loss_geometry_screening_v2` (results
`26029b72222e2b408e87fca3493940b0516b0f5d`, preregistration `cef1ee59…`,
runtime fix and disclosure `bb756418…`, dashboard `eef7ac82…`) at the recorded
outcome `accepted-screening-dataset`, the v1 outcome reused: 104,832
collisionless test-particle electron orbits launched at the midpoints of the 377
separatrix-bounded catalogue cells (cusp topology v3.1 at `cec47f12…`: 96
anode-side partials, 181 interior, 96 exit-side partials of the 96 sweep-v2
designs plus the 4 cells of the P2 row `divergent-exit-stack`, labelled
`P2_QUALIFIED_FIELD_SCREENING_LAUNCH_DESIGN`), 8 strata × scrambled Sobol
(stage 1 = 128 launches per cell, top-up to 512 where the stage-1 Wilson width
exceeds 0.10, replayed by code; 117 cells topped up, 260 saturated), a 1/8 paired
N→2N control (11,648 orbits, 2 discordant, bias −0.9e-4 ± 0.9e-4), 1,105 cases,
16,549 validators / 0 failed, 97/97 sealed, cross-resolution 2× diagnostic for
every design (max rms 1.04 % vs gate 5 %). All 181 interior cells are at
P(wall) = 1.000 (read in the Discussion as the field-maximum result of the PPM
analysis: the mid-planes sit at the magnet-stage centres); anode-side median
0.984 (min 0.307; 34 at one), exit-side median 0.500
(0.248–1.000; the value near one half is a direction split whose wall-reaching
direction equals the last stage's polarity in 82 of 90 divergent-exit designs);
10,407 reflections (11.2 %) in 65 exit-side cells and 1 anode-side cell, none
interior; Jeffreys floor median 0.0055, max 0.0242; 294/377 cells (78 %)
surrogate-ready; v1 vs v2 pooled (launch-weighted) Spearman +0.15, mean Δ +0.038,
|Δ| 0.113, 45 % interval overlap; P2 row 0.605 / 1.0 / 1.0 / 0.170 with 350
reflections in its 28 µm exit-side cell. Three disclosures are admitted with the
dataset and verified by the generator: (i) the results manifest was published
post hoc by a fail-closed runtime recovery after the locked attempt's publication
step hit the Windows CRT 8192-descriptor cap on 16,957 files, with the terminal
record and every sidecar-attested artifact durable and nothing rerun (the
disclosed manifest and terminal hashes must equal the committed bundle's; a 4096
pin cap was added); (ii) the frozen orbit_mc 1.7 launch-id grammar and the
Wilson-exactness defect (734 of the first 4000 case sizes have a positive
zero-count lower bound, 1238 an inexact full-count upper bound; recomputed) that
fixed the 128/16/64 case sizes; (iii) one cell (`l1a-gs-v2-088`, 0.16 mm) with its
midpoint inside the injector zone and 13 sweep cells under 1 mm, flagged not
moved. Every per-cell value is a collisionless geometric wall-access fraction and
never a loss probability; the Discussion (CLM-085) reads the interior saturation
as the exhaustion of the collisionless geometry-to-wall-loss chain as a closure
source, defers the surrogate/optimisation iteration on these labels and names a
kinetic (PIC) design sweep as future work that is not evidence. An eighth
`numerical-screening` gate, `GATE-L1B-HEMP-CONFIRMATION-V1-1`, admits the L1b/P2
material-aware HEMP confirmation v1.1
`modern/experiments/l1b_hemp_confirmation_v1_1` (results
`54cd3e82b7c879110cb7242c5f6210d1ac59fc92`, preregistration `c8692ff2…`, v1.1 code
and v1 post-hoc rejection note `b6125fe7…`, dashboard `560909f7…`; predecessor v1
record `2d8d6705…` / preregistration `fb143eb2…`) at a sixth recorded outcome,
`accepted-material-aware-confirmation` (justified on the gate: the field is a
linear-iron P2 field for every design, so no L1a screening outcome fits): the 15
HEMP-like sweep-v3 designs re-solved with the adaptive quadratic FEM reference
(soft-iron poles and return yoke μr 4000, recoil-remanence magnets μr 1.05, two
nested Dörfler/red levels at residual 2e-10, CPU only) and compared with their
sealed L1a records under the cusp topology v3.1 definition imported verbatim.
Verdict CONFIRMED: gate (a) 30/30 solves converged (level 0 24k–117k, level 1
50k–466k DOFs); gate (b) 15/15 cusp counts unchanged (strict); gate (c) 37/37
cusps matched bijectively, largest shift 0.362 mm = 0.80 of the tolerance
max(r_w/8, L1a dz) = 0.45–0.52 mm (median 0.267 mm), every shift above the P2
level-0→1 discretisation shift (≤ 1.4 µm); reported (d) HEMP-like preserved 14/15
(design 028: ρ 1.515 → 1.464), ρ ratio P2/L1a 0.94–1.45 (median 1.06), wall |B| at
the cusps 1.05–1.53× L1a at equal magnet strength (median 1.23), axis peak
0.98–1.35×, channel axis nulls moved up to 1.07 mm (bijection within the cusp
tolerance in 6/15 designs), separatrix lean 0.46 → 1.14 mm; 3079 s + 305 s on one
CPU worker, peak RSS 240 MB. Three disclosures are admitted and verified by the
generator: (i) the predecessor v1 executed once and ended in
`development_rejection` (13/15 resolved; 028 and 048 failed the inherited 10°
level-0 mesh-angle gate before any solve; no verdict), the v1 → v1.1 protocol
differing at exactly 12 paths (identity, predecessor block, angle gate, shakedown
set and preflight; every tolerance identical); (ii) the relaxed 5° gate with the
sliver record (028: 5.3°, 3 of 29,158 level-0 elements below 10°; 048: 5.6°,
13,816 of 46,582) and the whole-set mesh preflight 15/15; (iii) the five shakedown
designs are evidentiary designs, the shakedown's timing projection (100.3 min at
contention 1.5) exceeded the 90 min budget (the stage took 51.3 min), and the
campaign's own record declared paper admission out of its scope. Every ratio is a
ratio of two field models and never a probability; the Discussion (CLM-093) reads
the result as the topology being a property of the magnet arrangement under
linear iron, with absolute-field or ρ thresholds near 1.5 and the axis-null
positions not robust to the material model; no saturation, plasma,
mirror-probability, thrust, efficiency or design-recommendation claim follows. The
checked evidence is enumerated in `evidence/claims.json`. Concurrent or later work is not
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
representatives only, and no surrogate consuming it is admitted; its successor,
the catalogue-cell screening v2, carries the same classification (plus
`P2_QUALIFIED_FIELD_SCREENING_LAUNCH_DESIGN` for its P2 row), launches from the
catalogue cells, records unit access in every interior cell, and is admitted
with its post-hoc manifest publication disclosed; the one
optimisation consuming the v1 dataset is the catalogue campaign, classified
`l0_model_optimisation_over_screened_design_catalogue_with_test_particle_wall_loss_closure_not_thruster_performance`:
its closure identifies a collisionless test-particle wall-hit probability on a
linear-vacuum screening field with a per-cusp survival factor by declaration
(the v1 scenario analysis showed these are different quantities), a design that
wins there wins under that closure only, its fronts move under the
pooled-probability closure and under rescaled posterior widths, 77 of the 96
designs have negligible own dense hypervolume because a launch cell saturated,
and it opens none of L1--L3.

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
  `GATE-WALL-LOSS-V4`, `GATE-MDO-L0-V1` and `GATE-MDO-L0-V2`, the eight
  `numerical-screening` gates `GATE-L1A-SWEEP-V2`, `GATE-FOUR-CELL-V2`,
  `GATE-TOPOLOGY-CHAR-V1`, `GATE-WALL-LOSS-GEOMETRY-SCREENING-V1`,
  `GATE-CUSP-TOPOLOGY-V3-1`, `GATE-L1A-SWEEP-V3`,
  `GATE-WALL-LOSS-GEOMETRY-SCREENING-V2` and `GATE-L1B-HEMP-CONFIRMATION-V1-1`,
  each carrying its `recorded_outcome`, and the `analytic-consistency` gate
  `GATE-FOUR-CELL-CLOSURE-V1`.
- `evidence/manifests/l1b-hemp-confirmation-v1-1.json` — typed confirmation
  manifest (`paper-material-aware-confirmation-manifest` 1.0) binding every one
  of the sealed bundle's 134 files (dataset, CSV, 15 design records, 15 field
  grids, gates, campaign plan and result, source binding, runtime, lock, terminal
  record, phases, counters, access records, 9 transitions and every sidecar) at
  the record commit and the four frozen preregistration files at the
  preregistration commit; the results dashboard at its revision; the sweep-v3
  catalogue, manifest and design authorities and the frozen cusp-topology-v3.1
  protocol as references at their revisions; the predecessor's whole rejection
  bundle (104 files), its frozen files and `POSTHOC_REJECTION.md` as lineage at
  their revisions; plus the metrics the checker compares with the raw artifact
  values behind the `\Hmc...` macros.
- `evidence/l1b-hemp-confirmation-v1-1.json`,
  `generated/l1b-hemp-confirmation-v1-1.tex`,
  `generated/l1b-hemp-confirmation-v1-1.provenance.json` — the hash-bound
  evidence file, the generated `\Hmc...` macros with four `\ArtifactClaim`
  tables, and the provenance sidecar, regenerated by the checker at every run.
- `sections/l1b-hemp-confirmation-v1-1.tex` — the macro-only results subsection
  of Section 16; `sections/l1b-hemp-confirmation-v1-1-standalone.tex` compiles
  it on its own.
- `evidence/manifests/wall-loss-geometry-screening-v2.json` — typed screening
  manifest (`paper-orbit-cell-screening-manifest` 1.0) binding the sealed
  bundle's top-level artifacts, every transition, the representatives' and the
  P2 row's sample cases (summary, handoff, endpoints, orbit artifact and
  sidecar), fields and field evidence and the frozen preregistration files at the
  record commit; the results dashboard at its revision; the cusp-cell catalogue
  and topology manifest, the v1 dataset and manifest, the wall-loss export and
  the sweep manifest as references at their revisions; `POSTHOC_FINALIZATION.md`,
  the runtime recovery and lifecycle modules and the recovery tests as disclosure
  sources at the disclosure revision; plus the metrics the checker compares with
  the raw artifact values behind the `\Wlh...` macros.
- `evidence/wall-loss-geometry-screening-v2.json`,
  `generated/wall-loss-geometry-screening-v2.tex`,
  `generated/wall-loss-geometry-screening-v2.provenance.json` — the hash-bound
  evidence file, the generated `\Wlh...` macros with five `\ArtifactClaim`
  tables, and the provenance sidecar, regenerated by the checker at every run.
- `sections/wall-loss-geometry-screening-v2.tex` — the macro-only results
  subsection of Section 15; `sections/wall-loss-geometry-screening-v2-standalone.tex`
  compiles it on its own.
- `evidence/manifests/l1a-sweep-v3.json` — typed screening manifest
  (`paper-l1a-regime-screening-manifest` 1.0) binding the sealed bundle's
  top-level artifacts, the HEMP-like and representative designs' records and
  field grids and the frozen preregistration files at the record commit, the
  results dashboard at its revision, the sealed sweep-v2 manifest, the frozen
  topology-v3.1 protocol, the topology P2 record and the frozen wall-loss
  protocol as references at their revisions, the TWT/PPM review with its check
  script and output as the definition and hypothesis source, plus the metrics
  the checker compares with the raw artifact values behind the `\Swt...` macros.
- `evidence/l1a-sweep-v3.json`, `generated/l1a-sweep-v3.tex`,
  `sections/l1a-sweep-v3.tex` — hash-bound evidence file, generated macros with
  four `\ArtifactClaim` tables, and the admitted macro-only subsection bound
  once by `\input` from Section 14 of `manuscript.tex`.
- `evidence/manifests/cusp-topology-v3-1.json` — typed screening manifest
  (`paper-separatrix-topology-screening-manifest` 1.0) binding the sealed
  bundle's top-level artifacts, the 14 representatives' design records and
  field grids and the frozen preregistration files at the record commit, the
  results dashboard at its revision, the predecessor's rejected bundle, frozen
  protocol and read-only post-hoc audit as lineage at their own revisions, the
  sealed characterization and four-cell datasets and the sweep manifest as
  references at their admitted revisions, the literature review as the
  definition source, plus the metrics the checker compares with the raw
  artifact values behind the `\Ctv...` macros.
- `evidence/cusp-topology-v3-1.json`, `generated/cusp-topology-v3-1.tex`,
  `sections/cusp-topology-v3-1.tex` — hash-bound evidence file, generated
  macros with four `\ArtifactClaim` tables, and the admitted macro-only
  subsection bound once by `\input` from Section 13 of `manuscript.tex`.
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
- `evidence/manifests/mdo-l0-v2.json` — typed campaign manifest
  (`paper-mdo-catalogue-campaign-manifest` 1.0) binding every consumed bundle
  file of the catalogue campaign, the frozen preregistration files, the
  screening dataset and manifest behind the catalogue, the prior campaign's
  bundle files read for the comparison table, the prior campaign's post-hoc
  audit and the results dashboard, by Git blob and SHA-256, plus the metrics
  the checker compares with the raw artifact values behind the `\Mdb...`
  macros.
- `evidence/mdo-l0-v2.json`, `generated/mdo-l0-v2.tex`,
  `sections/mdo-l0-v2.tex` — hash-bound evidence file (macros marked with the
  bundle they were read from), generated macros with four `\ArtifactClaim`
  tables, and the admitted macro-only subsection bound once by `\input` from
  Section 12 of `manuscript.tex`.
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
python -m unittest discover -s paper/tests -p "test_mdo_l0_v1*" -v
```

## Admitted numerical campaign: MDO L0 campaign v2 (screened design catalogue)

`paper/scripts/generate_mdo_l0_v2_evidence.py` reads the sealed results bundle
of `modern/experiments/mdo_l0_campaign_v2` (147 files verified byte for byte
against `results/manifest.json`, pinned to the admitted manifest SHA-256; no
end-of-line tolerance exists or is granted) and, for the comparison table, the
sealed bundle of `mdo_l0_campaign_v1` verified the same way and pinned to its
admitted identity. It requires the frozen `protocol.json`, `authorities.json`
and `shakedown.json` to equal the sealed copies; re-verifies the sealed
96-design catalogue against the screening dataset it was drawn from (bytes and
Git blob at the screening record commit `ab7c2897`, manifest entry) and
recomputes every probability, Jeffreys posterior mean, Wilson interval and
nominal survival exactly from the counts; checks with `git diff-tree` that the
results commit adds files under `results/` only and that the preregistration
commit is experiment-path isolated; parses the disclosure list of the v1
post-hoc audit (`POSTHOC_AUDIT.md` at `e9f9af16`, blob equal at HEAD) and
requires it to equal the protocol's `v1_audit_disclosures_closed`; verifies that
the v2 protocol shares v1's reference point, scales, unit rows, robust
formulation and operating domain (the comparison is in one frame); cross-checks
the committed dashboard's extraction of both campaigns; and writes
`paper/evidence/mdo-l0-v2.json` (every `\Mdb...` macro with its artifact path,
JSON pointer, formatter, SHA-256 and bundle, or its derivation and inputs),
`paper/generated/mdo-l0-v2.tex` (macros plus four tables wrapped in
`\ArtifactClaim`: hypervolume per optimiser and seed with the catalogue designs
on each Pareto set; the catalogue designs on the dense-reference robust front
with screening probabilities, sealed geometry and own hypervolume; the closure
and uncertainty-width re-evaluations; the v1-versus-v2 comparison) and the
provenance sidecar. The subsection `paper/sections/mdo-l0-v2.tex` renders
numbers only through macros (short closure names `CL-1`/`CL-2` are macros too);
its results, catalogue-front, closure/width, audit-closure and scope statements
are exact `\EvidenceClaim` bodies (CLM-054, CLM-056, CLM-057, CLM-058,
CLM-059), the abstract sentence is CLM-053 and the labelled Discussion
interpretation (first geometry-dependent optimisation at screening tier;
ranking closure-dependent; saturated cells make most of the sweep space
unreachable under CL-1) is CLM-060. CLM-035 and CLM-052 were amended so that
the earlier "geometry link is open / consuming optimisation is future work"
sentences point at Section 12.

`check_paper.py` shares one checker (`_check_mdo_family`) between the two
optimisation gates and adds, for this one, the prior-campaign, screening-dataset
and post-hoc-audit bindings; it fails closed on any byte difference, artifact
hash mismatch in either bundle, metric that differs (in value or type) from the
raw artifact value behind its macro, policy metric off its fixed value, changed
results tree of either campaign, results commit touching a path outside
`results/`, dataset blob that differs between the screening record commit and
HEAD, audit blob or disclosure list that differs, dashboard checkout that
differs from the blob bound at the dashboard revision, literal digit or
undefined macro in the section, classification/closure/sensitivity-closure or
screening-classification macro that does not render its string, missing
registered non-claim, or a `\MdbEvidenceRevision` macro that does not spell the
manifest revision.

```powershell
python paper/scripts/generate_mdo_l0_v2_evidence.py
python -m unittest discover -s paper/tests -p "test_mdo_l0_v2*" -v
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
wall-loss campaign's zero reflections are a launch-position result and the
screening's reflections are the mirror reflections toward the magnet centres
that the PPM field predicts, per the recorded analysis bound with the sweep-v3
manifest; the screening is the geometry-to-wall-loss bridge at screening tier)
is CLM-052.
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

## Admitted topology screening: cusp topology search v3.1 (literature definition)

`paper/scripts/generate_cusp_topology_v3_1_evidence.py` reads the sealed
results bundle of `modern/experiments/cusp_topology_search_v3_1` (1,211 files
verified byte for byte against `results/manifest.json`, every artifact paired
with a manifest-bound sidecar; no end-of-line tolerance exists or is granted),
requires the frozen `protocol.json`, `authorities.json`, `shakedown.json` and
`design-authorities.json` to equal the sealed copies and to carry the same
blob at the preregistration and record commits, re-derives the headline and
every per-set estimand (histograms, legacy-target fractions, mirror-ratio and
angle distributions, gap distances, held-out counts, P2 consistency) from the
281 per-design rows and refuses on any difference, cross-checks every design
record, gzipped field grid, catalogue entry and CSV row against its row,
recomputes the boundary-ambiguity flags, gap and stage-centre distances and
cell lengths, verifies that every wall-reaching separatrix trace agrees with
its flux root, and cross-checks the committed results dashboard
(`modern/visualization/cusp-topology-search-v3.html`, its generator and
template at `9abbd537`) against the same bundle. It also byte-verifies the
predecessor's recorded `assessment_rejection` bundle
(`cusp_topology_search_v3` at `8cbcdbe6`), reproduces its read-only post-hoc
audit from the sealed characterization-v1 dataset (26 of 206 sealed axis
clusters dropped by the centroid filter, 22 in the channel, exactly the 14
recorded failing cases, 56/56 under the intended filter) and requires the
audit's documented numbers to agree; binds the sealed v1 and v2 datasets and
the sweep manifest as references that must hash to the sealed-source
identities the bundle recorded; binds the literature review that fixed the
definition; and writes `paper/evidence/cusp-topology-v3-1.json` (every
`\Ctv...` macro with its artifact path, JSON pointer, formatter and SHA-256, or
its derivation and inputs), `paper/generated/cusp-topology-v3-1.tex` (macros
plus four tables wrapped in `\ArtifactClaim`: cusp-count histogram per set,
the sweep by stage count, the P2 row against the two recorded references, the
lineage of the recorded rejection) and the provenance sidecar. Derived macros
include the N−1 / N−2 / N+1 design counts of the sweep, the v1 in-channel roots
split into axis clusters and off-axis bilinear roots, and the v2 strength-ratio
range that explains the four-cell null by construction.

The subsection `paper/sections/cusp-topology-v3-1.tex` renders numbers only
through macros (version tokens and the field level included); its results,
sweep-structure, four-cell/characterization, P2, lineage and scope statements
are exact `\EvidenceClaim` bodies (CLM-062, CLM-064, CLM-065, CLM-066,
CLM-067, CLM-068), the abstract sentence is CLM-061 and the two Discussion
interpretations amended by this admission are CLM-028 (the frozen wall-null
definition was non-standard; the same fields carry N−1 wall cusps under the
literature definition; the definition question is settled at screening tier,
the material question stays open) and CLM-044 (cells exist under the
literature definition; their plasma physics remains undemonstrated). The gate
reuses the `numerical-screening` kind at a fifth outcome value,
`accepted-topology-screening`; `opens_level` is null; the shared flag
`stable_multicell_wall_cusp_topology_demonstrated` of the earlier screening
gates is not reused because it was defined against the frozen definition,
which this study does not evaluate.

`check_paper.py` regenerates the three generated files at every run and fails
closed on any byte difference, any artifact hash mismatch, any manifest metric
that differs (in value or type) from the raw artifact value behind its macro,
any policy metric off its fixed value, a dashboard checkout that differs from
the blob bound at the dashboard revision, a results tree of either campaign
changed since its evidence revision, a frozen file changed since
preregistration, a lineage, reference or definition-source file that differs
from its bound blob or from the evidence binding, a lineage block that does
not chain preregistration -> rejection -> audit -> corrected preregistration or
that is declared citable, a recorded outcome that disagrees anywhere, a literal
digit or undefined macro in the section, a classification, P2-classification,
recorded-outcome, campaign-status, field-level or lineage-state macro that does
not render its string, a missing registered non-claim, an interpretation claim
inside the results section, a Discussion amendment (CLM-028, CLM-044) not bound
to the manifest or lacking its literature-definition wording, or a
`\CuspTopologyEvidenceRevision` macro that does not spell the manifest revision.

```powershell
python paper/scripts/generate_cusp_topology_v3_1_evidence.py
python -m unittest discover -s paper/tests -p "test_cusp_topology*" -v
```

## Admitted design-space screening: L1a geometry sweep v3 (HEMP-like regime)

`paper/scripts/generate_l1a_sweep_v3_evidence.py` reads the sealed results bundle
of `modern/experiments/l1a_geometry_sweep_v3` (979 files verified byte for byte
against `results/manifest.json`, every artifact paired with a manifest-bound
sidecar; no end-of-line tolerance exists or is granted), requires the frozen
`protocol.json`, `authorities.json`, `shakedown.json` and `design-authorities.json`
to equal the sealed copies and to carry the same blob at the preregistration and
record commits, re-derives the headline and every per-set estimand (Sobol,
held-out, pooled and sweep-v2 region) including the preregistered hypothesis
statistics from the 224 per-design rows (counts, histograms and medians exactly;
numpy sums recomputed with `math.fsum` within a relative tolerance of 1e-9),
recomputes x_w, every Bessel prediction, every ρ reading and every flag from
their inputs, cross-checks every design record, gzipped field grid, catalogue
entry and CSV row against its row, requires the sealed sweep-v2 manifest to hash
to the identity the bundle recorded and the imported definition parameters to
equal the frozen cusp-topology-v3.1 protocol, cross-checks the committed results
dashboard (`modern/visualization/l1a-geometry-sweep-v3.html`, its generator and
template at `44d0c63c`) against the same bundle, binds the TWT/PPM literature
review, its read-only check script and its committed output at `beb4772c` (the
commit at which the shakedown ran) and derives from that output the
launch-position classes of the geometry screening's launch cells, the wall-loss
campaign's launch offset from the P2 magnet centres (frozen wall-loss protocol
plus the topology screening's P2 record), the Mendel and adiabaticity parameters
and the magnetic-moment medians, and writes `paper/evidence/l1a-sweep-v3.json`
(every `\Swt...` macro with its artifact path, JSON pointer, formatter and
SHA-256, or its derivation and inputs), `paper/generated/l1a-sweep-v3.tex`
(macros plus four tables wrapped in `\ArtifactClaim`: the sweep-v2 box against
the sweep-v3 box; ρ by x_w band against I₁(x_w); the preregistered hypothesis
thresholds beside the observed statistics; the HEMP-like designs) and the
provenance sidecar. Derived macros include the end- and interior-cusp medians of
ρ/I₁, the x_w band counts and the predicted-only designs that fail at end cusps
only.

The subsection `paper/sections/l1a-sweep-v3.tex` renders numbers only through
macros (version tokens, the field level and digit-bearing symbols such as `I_1`
and `R^2` included, through a whitelisted `symbol` formatter); its execution,
HEMP-like-regime, hypothesis, earlier-box and scope statements are exact
`\EvidenceClaim` bodies (CLM-070, CLM-072, CLM-073, CLM-074, CLM-075), the
abstract sentence is CLM-069 and the labelled Discussion interpretation (the
legacy design space could not contain a HEMP-like cusp because the
parameterisation never varied r_w/L into the HEMP band; the material-aware
confirmation is queued and unreported) is CLM-076. The gate reuses the
`numerical-screening` kind at the existing outcome `accepted-screening`, the
outcome of the sweep v2 whose box it widens, because the object is the same kind
of study; the hypothesis is admitted at its recorded outcome (not held as
preregistered; I₁ an upper envelope). The shared flag
`stable_multicell_wall_cusp_topology_demonstrated` is not reused (it was defined
against the frozen definition); the boundary is carried by explicit flags
(`hypothesis_h1_held`/`hypothesis_h2_held` false, `material_aware_confirmation_run`
false, `hemp_like_designs_are_design_recommendations` false, `iron_in_field`
false, `rho_is_probability` false).

`check_paper.py` regenerates the three generated files at every run and fails
closed on any byte difference, any artifact hash mismatch, any manifest metric
that differs (in value or type) from the raw artifact value behind its macro, any
policy metric off its fixed value, a dashboard checkout that differs from the
blob bound at the dashboard revision, a results tree changed since the evidence
revision, a frozen file changed since preregistration, a reference or
definition-source file that differs from its bound blob or from the evidence
binding, a recorded outcome that disagrees anywhere, a hypothesis recorded as
held or a confirmation recorded as run, a literal digit or undefined macro in the
section, a classification, catalogue-label, recorded-outcome, campaign-status,
field-level, confirmation-status or hypothesis-outcome macro that does not render
its string, a missing registered non-claim, an interpretation claim inside the
results section, a hypothesis or earlier-box claim lacking its as-recorded
wording, a Discussion interpretation (CLM-076) not bound to the manifest, or a
`\SweepThreeEvidenceRevision` macro that does not spell the manifest revision.

```powershell
python paper/scripts/generate_l1a_sweep_v3_evidence.py
python -m unittest discover -s paper/tests -p "test_l1a_sweep_v3*" -v
```

## Admitted screening dataset: orbit wall-loss geometry screening v2 (catalogue cells)

`paper/scripts/generate_wall_loss_geometry_screening_v2_evidence.py` reads the
sealed results bundle of
`modern/experiments/orbit_wall_loss_geometry_screening_v2` (16,957 files verified
byte for byte against `results/manifest.json`, every artifact paired with a
manifest-bound sidecar; no end-of-line tolerance exists or is granted), requires
the frozen `protocol.json`, `authorities.json`, `shakedown.json` and
`design-authorities.json` to equal the sealed copies and to carry the same blob
at the preregistration and record commits, requires the record commit to carry
only `results/` and the disclosure commit to change only Markdown under the
experiment, binds the cusp-cell catalogue (`cec47f12`; must hash to
`dataset.catalogue_file_sha256`) and requires every dataset cell to equal its
catalogue cell field by field with the midpoint launch plane, wall area,
length/pitch, short-cell and injector-zone flags recomputed, replays the frozen
two-stage allocation rule from every stage-1 case (stage-1 Wilson width > 0.10 ⇔
topped up ⇔ three stage-2 blocks present), pools the stage counts and recomputes
every per-cell, per-case and per-stratum Wilson interval, both floors and the
readiness flag, replays the paired N→2N control orbit by orbit from the gzipped
endpoint tables (launch keys of the control case matched to their N-step
partners; wall_N, wall_2N and discordance recomputed for every cell), recomputes
the wall-area- and launch-weighted design averages, the pooled control gate with
the experiment's standard error, every headline statistic (position-class
summary, floors, readiness, least/most designs, P2 row) and the v1 comparison
from the bound v1 dataset (`ab7c2897`; Spearman, mean differences, interval
overlap, per-design rows), cross-checks every case summary, handoff, orbit
sidecar and consumer row, verifies every field-evidence record (cross-resolution
for all 97 designs), cross-checks the committed results dashboard
(`modern/visualization/wall-loss-geometry-screening-v2.html`, its generator and
template at `eef7ac82`) against the same bundle, binds `POSTHOC_FINALIZATION.md`,
`experiment_runtime/recovery.py`, `lifecycle.py` and `test_recovery.py` at
`bb756418` and requires the disclosed manifest and terminal hashes, file,
artifact and transition counts, results commit and descriptor arithmetic to
equal the committed bundle and the `MAX_PINNED_DESCRIPTORS` constant to equal
the disclosed cap, recomputes the Wilson-exactness scan behind the case-size
constraint (734 / 1238 of the first 4000), and writes
`paper/evidence/wall-loss-geometry-screening-v2.json` (every `\Wlh...` macro
with its artifact path, JSON pointer, formatter and SHA-256, or its derivation
and inputs), `paper/generated/wall-loss-geometry-screening-v2.tex` (macros plus
five tables wrapped in `\ArtifactClaim`: dataset and allocation summary,
per-cell-class distributions with saturation and floors, reflections and the
paired control with the exit-side direction split, the pooled v1 comparison,
the disclosures) and the provenance sidecar. Derived macros include the
per-class floor medians, the exit-side direction split and its last-polarity
association, the execution wall time from the transition log and the
disclosure values.

The subsection `paper/sections/wall-loss-geometry-screening-v2.tex` renders
numbers only through macros (the time-step policy names `N`/`2N` included); its
execution, cell-class, reflection/control/comparison, floors, disclosure and
scope statements are exact `\EvidenceClaim` bodies (CLM-078, CLM-080, CLM-081,
CLM-082, CLM-083, CLM-084), the abstract sentence is CLM-077, the tables are
CLM-079 and the labelled Discussion interpretation (the collisionless
geometry-to-wall-loss chain is exhausted as a closure source; the
surrogate/optimisation iteration on these labels is deferred; a kinetic sweep is
future work, not evidence) is CLM-085. The gate reuses the `numerical-screening`
kind at the existing outcome `accepted-screening-dataset` (the v1 outcome; the
study is the same kind of object with a different launch design) and justifies
the reuse on the gate; `opens_level` is null. The shared flag
`stable_multicell_wall_cusp_topology_demonstrated` is not reused (it was defined
against the frozen wall-null definition); the boundary is carried by explicit
flags (`manifest_published_posthoc` true, `evidence_durable_before_recovery`
true, `interior_cells_saturated` true, `access_fraction_is_loss_probability`
false, `p2_row_is_v4_replication` false, `pooled_design_values_are_estimands`
false, `catalogue_cells_are_demonstrated_confinement_cells` false,
`surrogate_fitted_or_accepted` false). Section 11's "planned" wording now cites
Section 15, and the topology screening's Limitations sentence names this
screening as the catalogue's first admitted consumer.

`check_paper.py` regenerates the three generated files at every run and fails
closed on any byte difference, any artifact hash mismatch, any manifest metric
that differs (in value or type) from the raw artifact value behind its macro, any
policy metric off its fixed value, a dashboard checkout that differs from the
blob bound at the dashboard revision, a results tree changed since the evidence
revision, a frozen file changed since preregistration, a reference or
disclosure-source file that differs from its bound blob or from the evidence
binding, a disclosure that names other hashes than the committed bundle's, a
results-bundle block that does not disclose the post-hoc publication, a recorded
outcome that disagrees anywhere or a gate that does not justify the reuse, an
interior-saturation, reflection-structure, control or disclosure finding that
does not hold in the evidence, a literal digit or undefined macro in the section,
a classification, P2-label, recorded-outcome or campaign-status macro that does
not render its string, a missing registered non-claim, a claim lacking its
as-recorded wording, an interpretation claim inside the results section, a
Discussion interpretation (CLM-085) not bound to the manifest or lacking its
wording, the superseded planning wording anywhere in the manuscript, Limitations
without the post-hoc publication and saturation statements, or a
`\GeometryScreeningTwoEvidenceRevision` macro that does not spell the manifest
revision.

```powershell
python paper/scripts/generate_wall_loss_geometry_screening_v2_evidence.py
python -m unittest discover -s paper/tests -p "test_wall_loss_geometry_screening_v2*" -v
```

## Admitted material-aware confirmation: L1b/P2 HEMP confirmation v1.1

`paper/scripts/generate_l1b_hemp_confirmation_v1_1_evidence.py` reads the sealed
results bundle of `modern/experiments/l1b_hemp_confirmation_v1_1` (133 manifest
entries plus the manifest verified byte for byte, every artifact paired with a
manifest-bound sidecar, no stray file in the tree; no end-of-line tolerance exists
or is granted) and the byte-verified rejection bundle of its predecessor
`modern/experiments/l1b_hemp_confirmation_v1` (103 entries plus the manifest,
terminal state `development_rejection`), requires the frozen `protocol.json`,
`authorities.json`, `shakedown.json` and `design-authorities.json` of both
campaigns to equal their sealed copies and to carry the same blob at their
preregistration and record commits, requires both record commits to carry only
their `results/` trees (134 and 104 files) and both preregistration commits to
freeze exactly the two authority files, recomputes the sealed experiment-code,
dependency-source and field-pipeline hashes (name, NUL, bytes, NUL per file) from
the Git blobs at the rebased preregistration commit `c8692ff2` and requires them to
equal the sealed values (the bundle's lock names the pre-rebase commit `ead9b525`
of `origin/exp/l1b-hemp-confirmation-v1`, recorded as a string), binds the sweep-v3
catalogue, manifest and design authorities (`2cfe8223`; must hash to the sealed
sources; the declared designs must be exactly the catalogue's HEMP-like Sobol
entries in order and every L1a reference must equal its catalogue entry) and the
frozen cusp-topology-v3.1 protocol (`cec47f12`; the imported numerical parameters
must equal it), and for every design recomputes x_w and r_w/L, the tolerance
max(r_w/8, L1a dz), every matched cusp's shift, tolerance ratio, wall-field ratio,
magnet-strength scaling and ρ ratio, every conservative Koch ratio from the wall
field and the adjacent axis peaks, every HEMP-like flag against the recorded 1.5
threshold, the strict and boundary-tolerant count agreement, the bijection flag,
the channel and outside axis-null populations, the separatrix lean, the level
structure, convergence, DOF cap and mesh-angle gate of every solve (from the design
records' mesh-quality blocks), the material regions (one magnet per stage, one pole
per gap, a return yoke), the discretisation and sampling stability, the field grid
by payload hash and identity, and the agreement-table and CSV rows; then re-derives
gates (b) and (c), the verdict by the predeclared rule, reported (d) and every
headline and estimand distribution; verifies the predecessor's two failures as the
level-0 mesh-angle rejection at the resolve stage with 13 + 2 = 15, proves the
v1 → v1.1 protocol differs at exactly the 12 declared paths with the comparison,
gates, definition import, design set, solver, materials, adaptivity, sampling,
resources, claim boundary and outputs identical, verifies `POSTHOC_REJECTION.md`
(bound at `b6125fe7`) against the v1 bundle by regex (commits, counts, stage,
gate, stage time, absence of a verdict) and requires the two rejected designs'
v1.1 level-0 meshes to fall below the v1 gate; cross-checks the committed
dashboard (`modern/visualization/l1b-hemp-confirmation-v1.html`, generator and
template at `560909f7`) against both bundles; and writes
`paper/evidence/l1b-hemp-confirmation-v1-1.json` (every `\Hmc...` macro with its
artifact path, JSON pointer, formatter and SHA-256, or its derivation and inputs;
lineage macros carry `bundle: lineage`), `paper/generated/l1b-hemp-confirmation-v1-1.tex`
(macros plus four tables wrapped in `\ArtifactClaim`: the per-design agreement
table, the verdict with its gates and reported quantities, the field model and
solve evidence, the disclosures) and the provenance sidecar. Derived macros
include the level and version tokens (`zero`, `one`, `L1a`, `v3.1`, `v3`, `v1`,
`v1.1`), the histograms, the extremes and argmax designs, the shifts above the
stability and discretisation scales, the wall-field rise in percent and the
lineage counts.

The subsection `paper/sections/l1b-hemp-confirmation-v1-1.tex` renders numbers
only through macros (the field level `L1a`, the topology version and the level
names included); its execution, results, reported, disclosure and scope
statements are exact `\EvidenceClaim` bodies (CLM-087, CLM-089, CLM-090, CLM-091,
CLM-092), the abstract sentence is CLM-086, the tables are CLM-088 and the
labelled Discussion interpretation (the separatrix-cusp topology is a property of
the magnet arrangement under linear iron; absolute-field or ρ thresholds near the
design value and the axis-null positions are not robust to the material model;
no design recommendation) is CLM-093. The gate is the `numerical-screening` kind
at a new sixth outcome, `accepted-material-aware-confirmation`, justified on the
gate (reusing `accepted-topology-screening` would state that the fields are L1a
fields with one P2 row, which is false here); the kind's description names the
outcome; `opens_level` is null. The verdict is admitted as recorded and never as a
positive finding; the boundary is carried by explicit policy flags
(`verdict_admitted_as_recorded` true, `positive_finding_accepted` false,
`saturation_or_bh_nonlinearity_modelled` false, `field_is_p2_qualified_chain`
false, `wall_cusp_positions_robust_within_tolerance` true,
`axis_null_positions_robust_within_tolerance` false,
`absolute_field_or_rho_threshold_robust` false,
`hemp_like_flag_preserved_for_every_design` false,
`predecessor_rejection_disclosed` true). The sweep-v3 scope claim (CLM-075) and
Discussion claim (CLM-076) and the topology Discussion claim (CLM-028) now cite
Section 16 instead of a queued, unreported confirmation; CLM-076 and CLM-028 read
confirmation macros and are bound to the manifest.

`check_paper.py` regenerates the three generated files at every run and fails
closed on any byte difference, any artifact hash mismatch in either bundle, a
manifest that does not bind every file of the results tree, any manifest metric
that differs (in value or type) from the raw artifact value behind its macro, any
policy metric off its fixed value, a dashboard checkout that differs from the blob
bound at the dashboard revision, a results tree changed since the evidence
revision, a frozen file changed since preregistration, a reference or lineage file
that differs from its bound blob or from the evidence binding, a lineage that does
not bind the predecessor's whole bundle and rejection note or that is cited for a
number, a recomputed source hash that differs between manifest and evidence, a
recorded outcome that disagrees anywhere, reuses an L1a screening outcome or is
not justified as a sixth outcome, a kind description that does not name the
outcome, a verdict, gate, preservation, non-robustness, wall-field-rise or
lineage finding that does not hold in the evidence, a literal digit or undefined
macro in the section, a classification, topology-label, recorded-outcome,
campaign-status, verdict or version macro that does not render its string, a
missing registered non-claim, a claim lacking its as-recorded wording, an
interpretation claim inside the results section, a Discussion interpretation
(CLM-093) not bound to the manifest or lacking its wording, the superseded
queued-confirmation wording anywhere in the manuscript, CLM-075/076/028 without
the cross-reference, Limitations without the confirmation boundary, or a
`\HempConfirmationEvidenceRevision` macro that does not spell the manifest
revision.

```powershell
python paper/scripts/generate_l1b_hemp_confirmation_v1_1_evidence.py
python -m unittest discover -s paper/tests -p "test_l1b_hemp_confirmation*" -v
```
