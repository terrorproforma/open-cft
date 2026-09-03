# Supplementary material and reviewer-response outline

## S1. Artifact identity and evidence map

- Repository revision and clean/dirty status at submission.
- Claim-to-evidence matrix export.
- Git blob and SHA-256 inventory for source data, manifests, generated figures,
  tables, and logs.
- Build-tool versions and full build/check transcript.
- Explanation of why working-tree and closed-gate material was excluded.

## S2. Legacy snapshot audit

- File inventory and reconstructed call graph.
- Confirmed defects with file/line evidence.
- Suspicious physics items kept separate from confirmed implementation defects.
- Publication/snapshot reconciliation for objectives, generations, variables,
  sensitivity claims, and reported power.
- Missing dependency and data inventory.

## S3. L0 equation and numerical contract

- Complete equation-ledger rendering.
- Constants, units, domains, supplied closures, and omitted phenomena.
- Binary64 fraction admission, finite-output, PPU-boundary, tiny-state, and
  nonrelativistic policies.
- Analytic and manufactured cases.
- Python/Warp CPU/CUDA parity tolerances and full residual definitions.

## S4. First deterministic batch

- Committed result manifest when one becomes available.
- Input-generation algorithm, seed, bounds, and rationale.
- Environment and device identity.
- Per-field aggregate summary and deterministic selected cases.
- Full-batch parity and conservation summary.
- Rejected/failed-point accounting.
- Explicit statement that the bounds are hypothetical.

## S4b. Collisionless test-particle wall-loss campaign v4

- Typed campaign manifest `paper/evidence/manifests/wall-loss-v4.json` and the
  hash-bound evidence file `paper/evidence/wall-loss-v4.json`.
- Preregistration commit, results commit, post-hoc audit, and the sealed
  results bundle with its manifest, lock, transitions and per-orbit artifacts.
- Frozen protocol: launch strata, field maps, timestep policies, estimator,
  binding gates, shakedown disclosure, prior-campaign disclosure.
- Per-case and per-cell tables regenerated from the bundle; convergence
  chains; energy, endpoint, manufactured-order and CPU/CUDA parity facts.
- The sealed publication boundary and classification string, verbatim.
- Explicit statement that the pooled fraction is an equal-weight design
  average and that no mirror-formula estimate is published.

## S4c. Preregistered L1a topology screening (sweep v2, four-cell v2, characterization v1)

- Typed screening manifests `paper/evidence/manifests/{l1a-sweep-v2,
  four-cell-v2,topology-characterization-v1}.json` and the hash-bound
  evidence files `paper/evidence/<key>.json`, each with its `recorded_outcome`.
- Preregistration and results commits, the two post-hoc end-of-line audits
  (`l1a_geometry_sweep_v2/POSTHOC_AUDIT.md`,
  `four_cell_topology_search_v2/POSTHOC_AUDIT.md`) and the sealed bundles
  (manifests, summaries/datasets, reports, locks, frozen protocols,
  representative geometry and field artifacts).
- Frozen definitions verbatim: sweep terminal gates and QoI policy (axis cusp,
  mirror ratio, topology-claim limit); four-cell cusp/cell definition, geometry
  slots, cross-map shift, endpoint exclusion, zero-pass policy; characterization
  eligibility, separatrix and cross-map correspondence rules.
- Tables regenerated from the bundles: sweep gates and representatives;
  four-cell failure taxonomy and interior-cusp counts per map; characterization
  null classes by zone and the empirical stage relation.
- GPU replay outcomes as recorded, including the two four-cell replay
  candidates that exceeded the residual-diagnostic tolerance.
- Explicit statements that the fields are linear-vacuum equivalent-current
  models, that each null holds under its frozen definitions only, that no
  plasma, mirror-probability or performance quantity is claimed, and that the
  superseded proxy search and failed criterion validations are lineage only.

## S4d. Preregistered robust multi-objective optimisation of the L0 model (campaign v1)

- Typed campaign manifest `paper/evidence/manifests/mdo-l0-v1.json` and the
  hash-bound evidence file `paper/evidence/mdo-l0-v1.json`.
- Preregistration, results and dashboard commits; the sealed bundle (manifest,
  lock, transitions, protocol, authorities, code contract, shakedown, frozen
  sample, nine run records with per-evaluation designs and objectives,
  hypervolume curves, Pareto sets, pooled and per-strategy fronts, dense
  reference, sensitivity) and the committed results dashboard whose embedded
  extraction the generator requires to agree with the bundle.
- Frozen protocol verbatim: design variables and excluded legacy radii with
  the recorded reason, uncertain inputs and cusp-prior calibration, closure
  CL-1 (declared, not derived), objectives, constraint, CVaR robust
  formulation and separability expectation, optimiser settings, budget and
  fairness rules, binding and reported gates, shakedown rule, prior-model
  disclosures (F0-only status; corrected four-cell solver probe).
- Tables regenerated from the bundle: hypervolume per optimiser and seed with
  seed variance; robust versus nominal pooled fronts; alternative priors and
  fixed scenarios.
- Explicit statements that the campaign is optimiser evidence under CL-1 and
  the declared priors, not thruster performance, not a design recommendation,
  not an optimiser-superiority claim beyond the recorded budget, seeds and
  model; that the worst case enforced is the worst sampled case; and that the
  collisionless wall-hit probability is not the per-cusp loss probability.

## S4e. Consistency of the corrected four-cell power balance (analytic result)

- Typed analysis manifest `paper/evidence/manifests/four-cell-closure.json`
  and the evidence file `paper/evidence/four-cell-closure.json`.
- Analysis commit `266d8a99` (document, ledger entry
  `global_row_consistency`, diagnostics `potential_parametrized_state` and
  `global_row_closed_form`, pinning tests), verified-tree commit `ba6875f6`,
  the frozen MDO protocol disclosure (probe 13/80) at preregistration
  `4898d0fd`, and the legacy `FYP/Power_B_EQs.m` blob as lineage.
- The derivation verbatim: substitution of R00--R26 into R27, the closed form
  `2 (j_e3 (1-p4) + I4)(phi_4 - Ua) + EI (p1 j_e0 + p2 j_e1 + p3 j_e2)`,
  non-negativity on the admissible region, the solution sub-region
  (p1 = p2 = p3 = 0, any p4, phi_4 = Ua), the source of both terms (Kornfeld
  assumption 8; printed anode electron sign), the cancellation of the audit
  corrections, and the proposed correction with its `PROPOSED_NOT_ACCEPTED`
  status and rank consequence.
- Recomputation protocol and tolerances exactly as declared in the generator
  (seeded 400-state sample, continuation ladder, anode-only closures,
  published-state misfit, relaxed root, Jacobian rank, anode-fall coefficient)
  with the recorded significant digits; the documented-only items
  (differential evolution, random starts, the 80-case probe) marked as such.
- Tables regenerated at every check: the continuation ladder (documented and
  recomputed floors) and the global-search/relaxed-root/Jacobian/misfit table
  with per-row status.
- Explicit statements that the result is about the equation set, not the
  thruster; that the correction is not accepted; that the legacy-study
  consequence is interpretation; and that no value of the legacy run is claimed.

## S4f. Wall-loss geometry screening (accepted screening dataset)

- Typed screening manifest
  `paper/evidence/manifests/wall-loss-geometry-screening-v1.json` and the
  evidence file `paper/evidence/wall-loss-geometry-screening-v1.json`.
- Preregistration commit `c86bfca3` (frozen protocol, authorities, shakedown,
  design authorities), record commit `ab7c2897` (sealed bundle of 2,835 files;
  results dashboard regenerated in the same commit).
- Field provenance: the accepted sweep-v2 fields re-solved on the CPU with the
  accepted L1a solver, identity proven per design (geometry/source/config/case
  hashes, QoI replay within the sweep's tolerances, node-wise agreement of the
  four stored representative maps), interpolation and cross-resolution
  diagnostics with the cross-resolution recorded for the four representatives
  only.
- Launch design verbatim: four cells at 1/8, 3/8, 5/8, 7/8 of the straight
  span, two radii, energies {5, 25} eV, pitches {20, 70} degrees, both
  directions, eight gyrophases (offset 11 pi/96), 512 launches per case,
  timestep policies N and 2N, refined-N for the representatives.
- Per-design records: wall-hit, reflection, escape and timeout probabilities
  with Wilson intervals, convergence flag, per-cell and per-stratum counts,
  escape sub-classes, magnetic-moment diagnostic, tolerance-close share.
- Tables regenerated at every check: dataset summary and convergence, least
  and most wall-loss designs with sealed geometry, per-cell distribution and
  saturation, termination classes; derived Spearman rank correlations with
  chamber length, wall radius, stage pitch, stage count, minimum mirror ratio,
  magnetic-moment variation and reflection probability.
- Coupling-consumer record: every handoff consumed as a verified handoff; the
  wall-loss campaign's export consumed as a labelled reference row absent from
  the screening set.
- Explicit statements that the fields are not P2-qualified and the dataset is
  never accepted physical-orbit, plasma or performance evidence; that launch
  cells are protocol positions; that the geometry associations are
  observations and not a design rule; and that the dataset is surrogate and
  optimisation input only under its label.

## S4g. Preregistered catalogue optimisation of the L0 model over the screened sweep designs (campaign v2)

- Typed campaign manifest `paper/evidence/manifests/mdo-l0-v2.json` and the
  hash-bound evidence file `paper/evidence/mdo-l0-v2.json` (every macro marked
  with the bundle it was read from).
- Preregistration commit `99914dc2` (frozen authorities only), results commit
  `a003f766` (148 files, all under `results/`), dashboard commit `0ea33a7e`;
  the sealed bundle (manifest, lock, transitions, protocol, authorities, code
  contract with its 28-file import-bound hash scope, import-scope record,
  shakedown, sealed catalogue with counts and geometry, catalogue binding,
  frozen unit rows and per-design sample, nine run records, hypervolume curves,
  Pareto sets, pooled and per-strategy fronts, dense reference with per-design
  summaries and separability, sensitivity) and the committed results dashboard
  whose embedded extraction of both campaigns the generator requires to agree
  with the bundles.
- Catalogue provenance: the 96 accepted sweep-v2 designs of the wall-loss
  geometry screening (`ab7c2897`), bound by bytes, Git blob, manifest entry and
  ancestry; per-cell and pooled counts at the accepted-2N timestep; Jeffreys
  posteriors, Wilson intervals and nominal survivals recomputed exactly.
- Frozen protocol verbatim: the categorical catalogue index and the prior
  campaign's operating-point domain, closures CL-1 (campaign; identification
  disclosure) and CL-2 (sensitivity), objectives, reference point, CVaR robust
  formulation and per-design separability expectation, the mixed qLogNEHVI
  (discrete candidate stage over every design, per-member refinement), mixed
  NSGA-III with duplicate elimination and the two-stage LHS, budget 160 per run
  with a 32-point shared initial design, seeds 101/202/303, the 96 x 1024 dense
  reference, the twelve binding integrity gates and their declared semantics
  (acceptance is not efficacy), the reported outcomes, and the closure of the v1
  audit disclosures F9, F10, F22, F26, F27 and F28.
- Tables regenerated at every check: hypervolume per optimiser and seed with the
  catalogue designs on each Pareto set; the dense-reference robust-front designs
  with screening probabilities, sealed geometry and own hypervolume; the
  CL-1/CL-2 and uncertainty-width re-evaluations; the v1-versus-v2 comparison
  in the shared reference frame.
- Explicit statements that the campaign is optimiser evidence under a declared
  identification of a screening wall-hit probability with a per-cusp survival
  factor, not thruster performance, not a design recommendation ("wins under
  this closure only"), not an optimiser-superiority claim beyond the recorded
  budget, seeds and model; that no surrogate is used; that the first Bayesian
  seed never evaluated the design the other seeds' fronts share; that the
  design ranking is closure-dependent; and that 77 of 96 designs have
  negligible own hypervolume because a launch cell saturated.

## S4h. Cusp topology search v3.1 (accepted topology screening under the literature definition)

- Typed screening manifest `paper/evidence/manifests/cusp-topology-v3-1.json`
  (`paper-separatrix-topology-screening-manifest` 1.0) and the hash-bound
  evidence file `paper/evidence/cusp-topology-v3-1.json`.
- Preregistration commit `1600cfd3` (frozen protocol, authorities, shakedown,
  design authorities), record commit `cec47f12` (sealed bundle of 1,211 files:
  topology dataset in JSON and CSV form, cusp-cell catalogue, 281 design
  records and accepted field grids, gates, source binding, campaign plan and
  result, lock and transitions), dashboard commit `9abbd537`.
- Lineage bound at its own revisions and never cited for a number: the
  predecessor `cusp_topology_search_v3` (preregistration `69159934`, recorded
  `assessment_rejection` at `8cbcdbe6`: 8 of 9 binding gates true, the
  held-out gate false for 14 of 56 characterization cases because the
  reference kept only sealed axis clusters with centroid radius exactly zero)
  and its read-only post-hoc audit at `9fa6359a`; the generator byte-verifies
  the rejected bundle and reproduces the audit's counts (26 of 206 sealed axis
  clusters dropped, 22 in the channel, 56/56 under the intended filter) from
  the sealed characterization dataset.
- References bound at their admitted revisions and required to hash to the
  sealed-source identities the bundle recorded: the characterization-v1
  dataset (`3ce6c546`), the four-cell-v2 dataset (`7120e8ed`) and the sweep-v2
  results manifest (`f30cb42e`); the literature review that fixed the
  definition bound at `66879e00`.
- Definition verbatim: axis null (sign change of the on-axis field on the
  axis-regular bicubic flux interpolant, bisection to 1e-12 m, X-type by the
  analytic Jacobian and the frozen characterization classification), separatrix
  (event-aware RK4 field-line trace from the radial eigen-direction to the wall,
  cross-checked against the flux root within 50 um), wall cusp (intersection
  inside the straight dielectric), cells (intervals between consecutive cusps
  plus partials), wall and axis mirror ratios (field ratios), stability on a
  2x refined map within 0.25 mm, boundary ambiguity within 0.25 mm, two binding
  held-out references (sealed characterization axis roots, sealed sweep axis
  nulls) within 0.25 mm.
- Tables regenerated at every check: cusp-count histogram per design set; the
  sweep resolved by stage count (N-1, legacy targets, interior cells, mirror
  ratios, gap distances, angles); the P2 row against the kinetic-workstream
  axis-null planes and the plasma-topology-dashboard wall maxima (reported,
  ungated); the lineage table setting the recorded rejection beside the
  admitted campaign.
- Explicit statements that cusps and cells are geometric properties of
  prescribed field maps under a stated definition; that the mirror ratios are
  field ratios and never probabilities; that no plasma, confinement, wall-loss
  or performance quantity is published; that the L1a sets are not P2-qualified
  and the single P2 row's iron sensitivity was not tested; that the
  frozen-definition nulls of the earlier admissions remain true; that the
  catalogue is a consumer contract under its labels with no admitted consumer;
  and that the predecessor's rejection was a recording-layer defect disclosed
  as lineage.

## S4i. L1a geometry sweep v3 (accepted field-only screening of the HEMP-like regime)

- Typed screening manifest `paper/evidence/manifests/l1a-sweep-v3.json`
  (`paper-l1a-regime-screening-manifest` 1.0) and the hash-bound evidence file
  `paper/evidence/l1a-sweep-v3.json`.
- Preregistration commit `1923ef76` (frozen protocol, authorities, shakedown,
  design authorities), record commit `2cfe8223` (sealed bundle of 979 files:
  sweep dataset in JSON and CSV form, wall-cusp catalogue v3, 224 design
  records and accepted field grids, gates, source binding, campaign plan and
  result, lock and transitions), dashboard commit `44d0c63c`.
- Definition and hypothesis source bound at `beb4772c`: the TWT/PPM literature
  review, its read-only check script and the committed output the script wrote
  (the launch-position analysis of the geometry screening's sealed orbits by
  launch cell, the Mendel and adiabaticity parameters and the magnetic-moment
  medians are derived from that output and cited as a recorded analysis).
- References bound at their admitted revisions: the sweep-v2 results manifest
  (`f30cb42e`; must hash to the sealed-source identity the bundle recorded), the
  frozen cusp-topology-v3.1 protocol (`cec47f12`; the imported definition
  parameters must equal it), the topology screening's P2 design record
  (`cec47f12`; stage centres of the wall-loss field) and the frozen wall-loss
  protocol (`757e365f`; launch planes).
- Design box verbatim: the sweep-v2 rules on a wider box (pitch 3.4-6.5 mm, wall
  radius 1.4-4.2 mm, magnet clearance 0.35-1.6 mm, magnet thickness 2.2-5.0 mm;
  r_w/L 0.215-1.235, x_w 0.68-3.88; the v2 box a strict subset), 128
  scrambled-Sobol designs (seed 20260903) plus the 96 accepted sweep-v2 designs
  re-solved as a held-out set; six sweep-v2 metric gates verbatim; 2x refined
  map per design; Koch ratio in four readings with the conservative reading
  binding and HEMP-like := rho >= 1.5 at every wall cusp; hypothesis H1/H2
  preregistered as reported tests.
- Tables regenerated at every check: the sweep-v2 box against the sweep-v3 box;
  rho by x_w band against I_1(x_w) with predicted and realised HEMP-like counts;
  the preregistered hypothesis thresholds beside the observed statistics (slope
  0.689, R^2 0.39, 70 % in band, accuracy 0.72, end/interior rho/I_1 medians
  0.80/0.87, realised threshold x_w 2.34); the 15 HEMP-like designs with the two
  five-stage four-cusp designs flagged.
- Explicit statements that the ratios are field ratios of linear-vacuum
  screening fields and never probabilities; that the declared iron pole pieces
  are vacuum in the field; that the hypothesis is admitted at its recorded
  outcome (not held as preregistered; I_1 an upper envelope); that the
  material-aware confirmation is queued and unreported; that no HEMP-like
  design is a design recommendation; and that the legacy-design-space reading
  is a labelled Discussion interpretation.

## S5. Controlled performance benchmark

This supplement remains closed until a benchmark manifest records kernel
warm-up, synchronized kernel-only and end-to-end regions, controlled device
state, deterministic batch sizes, repeated trials, dispersion, and parity for
every timed configuration. The current diagnostic timing is excluded from any
speedup argument.

## S6. L1 field-resolved evidence

Opened only by `GATE-L1`. Include governing equations, geometry and materials,
boundary conditions, solver identity, manufactured solutions, mesh/domain
convergence, numerical uncertainty, artifact hashes, and failed cases.

## S7. L2 coupled-model evidence

Opened only by `GATE-L2`. Include closure provenance, coupling algorithm,
interface conservation, spatial/temporal convergence, discrepancy model,
uncertainty calibration, code comparison, and failure taxonomy.

## S8. L3 PIC/experimental evidence

Opened only by `GATE-L3`. Include collision and boundary data, scaling,
convergence, diagnostics, facility conditions, measurement uncertainty,
preregistered and withheld cases, discrepancy assessment, and applicability
limits.

## S9. Optimization and UQ

- Campaign schema and objective-direction transform.
- Initial designs, grouped splits, seeds, pending jobs, and failed evaluations.
- Surrogate diagnostics, calibration, and out-of-domain guardrails.
- Equivalent-F3 accounting and retry ledger.
- Frozen objective normalization and reference point.
- Per-seed F3-verified hypervolume with confidence intervals.
- Same-cost baselines and sensitivity-estimator convergence.

## S10. Reproduction instructions

- No-install core check path.
- Optional environment lock files, if later committed.
- Commands to regenerate every display item.
- Expected output hashes or declared normalized comparisons.
- Platform deviations and known nondeterminism.

## Reviewer-response evidence template

Use one block per substantive reviewer point:

1. **Reviewer point:** quote the point exactly.
2. **Response:** state whether the manuscript changed and why.
3. **Claim impact:** list added, narrowed, removed, or unchanged claim IDs.
4. **New evidence:** list manifest IDs, committed paths, revisions, checks, and
   uncertainty changes.
5. **Manuscript locations:** section and line range in the response revision.
6. **Artifact impact:** figures, tables, supplement items, and data package.
7. **Gate status:** identify any gate opened or explicitly left closed.
8. **Residual limitation:** state what the new evidence still cannot support.

No response should answer a request for stronger conclusions by converting a
planned result into prose without first satisfying its evidence gate.
