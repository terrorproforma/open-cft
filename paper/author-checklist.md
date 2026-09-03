# Author and submission checklist

## Evidence freeze

- [ ] Record the submission Git revision and require a clean tracked tree.
- [ ] Confirm every verified claim ID exists in `evidence/claims.json`.
- [ ] Confirm every evidence source matches its registered Git blob and
  Git-object SHA-256 at a resolvable ancestor revision.
- [ ] Confirm every accepted manifest has a recognized type/schema, all
  required file roles, gate-specific metrics, and a committed matching blob.
- [ ] Exclude untracked, uncommitted, and concurrent-work artifacts.
- [ ] Review closed L1/L2/L3 gates; do not paraphrase them as results.
- [ ] Archive machine-readable manifests and all hashed artifacts.

## Scientific claims

- [ ] Describe L0 as an algebraic conservation-reduced operating-point
  baseline with externally supplied closures.
- [ ] Do not describe L0 as spatial, one-dimensional, or geometry-resolving.
- [ ] Do not claim geometric prediction, physical calibration, measured
  accuracy, or experimental validation from L0.
- [ ] State that cross-backend parity and conservation closure demonstrate
  implementation consistency only.
- [ ] State that Python and Warp share canonical preprocessing; do not call
  them fully independent implementations.
- [ ] Do not claim a GPU speedup from the current uncontrolled diagnostic.
- [ ] Keep historical `total_efficiency` separate from L0 efficiency
  boundaries.
- [ ] Label hypothetical bounds as neither calibration nor uncertainty
  intervals.
- [ ] Report failed, rejected, censored, and out-of-domain cases.
- [ ] Describe the wall-loss campaign only as
  `collisionless_prescribed_field_test_particle_wall_loss_not_pic`: not PIC,
  not self-consistent, not thruster performance, not validated; the pooled
  fraction is an equal-weight design average of a bimodal per-cell result.
- [ ] Keep every number of the wall-loss section a `\Wlf...` macro; never
  type a digit into `paper/sections/wall-loss-v4.tex`.
- [ ] Keep the mirror-picture and multi-cell-topology statements in the
  Discussion labelled as interpretation; the multi-cell statement is
  macro-bound to the admitted four-cell null and characterization null and
  phrased as "not shown stable" / "undemonstrated", never as non-existence.
- [ ] Describe the topology-screening studies as linear-vacuum L1a
  equivalent-current field screening (no permanent-magnet or nonlinear-iron
  material model); report each at its `recorded_outcome` (accepted screening,
  preregistered null, recorded characterization) and report the four-cell GPU
  replay as recorded (2 of 4 diagnostic passes).
- [ ] Keep every number of the screening sections a `\Swp...`, `\Fcn...` or
  `\Tch...` macro; never type a digit into `paper/sections/l1a-sweep-v2.tex`,
  `four-cell-v2.tex` or `topology-characterization-v1.tex`.
- [ ] Quote the superseded four-cell proxy search and the failed coupling-v4
  validations only inside the registered lineage non-claim.
- [ ] Describe the optimisation campaign only as
  `l0_model_robust_multiobjective_optimisation_under_declared_input_uncertainty_not_thruster_performance`:
  optimiser evidence on the L0 model under the declared closure CL-1 and
  declared priors; no thruster-performance, plasma or physical-device claim,
  no design recommendation, no optimiser-superiority claim beyond the recorded
  budget, seeds and model, geometry excluded, benchmark field still null.
- [ ] Keep every number of the optimisation section a `\Mdo...` macro; never
  type a digit into `paper/sections/mdo-l0-v1.tex`.
- [ ] Keep the four Discussion readings of the optimisation campaign labelled
  as interpretation; the fourth now points at the catalogue campaign of
  Section 12 as the recorded successor, never at a demonstrated
  geometry-to-performance map.
- [ ] Describe the catalogue campaign only as
  `l0_model_optimisation_over_screened_design_catalogue_with_test_particle_wall_loss_closure_not_thruster_performance`:
  optimiser evidence on the L0 model over the 96 screened sweep designs under
  the declared closure CL-1 whose per-cusp factors are the screening's
  per-cell test-particle wall-hit posteriors (an identification declared, not
  derived; the v1 scenario analysis showed the quantities differ); no
  surrogate; no thruster-performance, plasma or physical-device claim; no
  design recommendation ("a design that wins here wins under this closure
  only"); no optimiser-superiority claim beyond the recorded budget, seeds and
  model; three seeds give counts, never significance; benchmark field still
  null.
- [ ] Keep every number of the catalogue section a `\Mdb...` macro (including
  the short closure names `CL-1` and `CL-2`); never type a digit into
  `paper/sections/mdo-l0-v2.tex`, and never borrow a `\Mdo...` macro there.
- [ ] Keep the three Discussion readings of the catalogue campaign labelled as
  interpretation: first geometry-dependent optimisation at screening tier under
  a declared identification; ranking closure-dependent (CL-2 shares no design);
  saturated launch cells make most of the sweep space unreachable under CL-1,
  with the kinetic question stated as undecided and not admitted.
- [ ] Describe the four-cell closure analysis only as
  `analytic_consistency_of_the_corrected_four_cell_power_balance_not_thruster_physics`:
  a statement about the corrected equation set (no admissible root for any
  positive interior cusp probability; solutions only with zero interior
  probabilities at the anode potential), never about the physical thruster;
  the proposed correction stays `PROPOSED_NOT_ACCEPTED`.
- [ ] Keep every number of the closure section a `\Fcc...` macro (documented
  values bound to the analysis document, ledger or frozen protocol; recomputed
  values from the bound package); never type a digit into
  `paper/sections/four-cell-closure.tex`, and keep the displayed closed form's
  coefficient and row index as macros.
- [ ] Keep the legacy-study consequence (residual floors accepted by
  `lsqnonlin` exit status) and the three-finding synthesis labelled as
  interpretation in the Discussion; never claim a numerical value of the
  unavailable legacy run.
- [ ] Describe the wall-loss geometry screening only as
  `SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS` admitted at
  `accepted-screening-dataset`: collisionless test particles in re-solved
  L1a screening fields that are not P2-qualified; never accepted
  physical-orbit evidence, never a plasma or performance quantity; launch cells
  are protocol positions; geometry associations are observations, not a design
  rule; surrogate and optimisation input only under its label.
- [ ] Keep every number of the screening section a `\Wlg...` macro; never type
  a digit into `paper/sections/wall-loss-geometry-screening-v1.tex`.
- [ ] Keep the mirror-picture statement of the wall-loss campaign scoped to its
  field (the screening records reflections in every design); the
  design-dependent optimisation that consumes the dataset is admitted in
  Section 12 and its geometry link is a closure-dependent ranking, not a map
  from the screening fields to the operating-point model.
- [ ] Describe the cusp topology search v3.1 only as
  `SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY` (three L1a sets) and
  `P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY` (one P2 row) admitted at
  `accepted-topology-screening`: cusps and cells are geometric properties of
  prescribed field maps under the literature definition; mirror ratios are
  field ratios and never probabilities; no plasma, confinement, wall-loss or
  performance quantity; iron sensitivity of the P2 row untested; the catalogue
  is a consumer contract under its labels with no admitted consumer.
- [ ] Keep every number of the topology section a `\Ctv...` macro (including
  the version tokens `v3.1`, `v3`, `v2`, `v1` and the field level `L1a`); never
  type a digit into `paper/sections/cusp-topology-v3-1.tex`.
- [ ] Cite the predecessor `cusp_topology_search_v3` only as lineage (recorded
  `assessment_rejection`, recording-layer defect of the held-out reference
  extraction); never quote a number from its bundle.
- [ ] Keep the Discussion's multi-cell paragraph labelled as interpretation:
  the frozen wall-null definition was non-standard, the same fields carry the
  N-1 wall cusps under the literature definition, the definition question is
  settled at screening tier, the material question stays open, and whether a
  cell confines a plasma is not decided anywhere in the paper (CLM-028,
  CLM-044).
- [ ] Say that the geometry screening's launch cells are channel fractions and
  not the catalogue cells; a screening launched from the catalogue is future
  work with no result.
- [ ] Describe the L1a geometry sweep v3 only as
  `L1a_FIELD_ONLY_SCREENING_NOT_HARDWARE_VALID` admitted at
  `accepted-screening` (the sweep-v2 outcome): the Koch ratio and every mirror
  descriptor are field ratios of linear-vacuum screening fields and never
  probabilities; the declared iron pole pieces are vacuum in the field; the
  preregistered hypothesis is reported at its recorded outcome (neither H1 nor
  H2 held; I_1(x_w) is an upper envelope); the material-aware confirmation is
  queued, not run; no HEMP-like design is a design recommendation.
- [ ] Keep every number of the sweep-v3 section a `\Swt...` macro (including
  the version tokens `v3`, `v2`, `v3.1`, `v4`, the field levels `L1a`/`L1b` and
  the digit-bearing symbols `I_1`, `b_3/b_1`, `R^2`, `H1`, `H2`); never type a
  digit into `paper/sections/l1a-sweep-v3.tex`.
- [ ] Keep the Discussion's legacy-design-space paragraph labelled as
  interpretation (CLM-076): the legacy parameterisation never varied r_w/L into
  the HEMP band, so in the screening field its design space could not contain a
  HEMP-like cusp; whether the widened-box ratios survive a material-aware field
  is not reported.
- [ ] Cite the TWT/PPM review's launch-position analysis as a recorded analysis
  of the geometry screening's sealed orbits (bound at `beb4772c` through the
  sweep-v3 manifest), never as a result of the sweep or a new orbit result.

## Methods and uncertainty

- [ ] Separate code verification, solution verification, validation, and
  prediction.
- [ ] Report numerical, input, measurement, emulator, and model-discrepancy
  uncertainty separately.
- [ ] Justify input distributions and dependencies before sensitivity analysis.
- [ ] Group train/validation partitions by design across fidelity and seeds.
- [ ] Require highest-available-fidelity reevaluation for promoted designs.
- [ ] Freeze objective normalization and the verified-hypervolume reference
  point before strategy comparison.
- [ ] Use common initial data, cost checkpoints, constraints, failures, and F3
  verification for all optimization baselines.

## Reproducibility

- [ ] Run `python paper/scripts/generate_tables.py`.
- [ ] Run `python paper/scripts/generate_wall_loss_v4_evidence.py` and confirm
  `git status` shows no change to the three generated wall-loss files.
- [ ] Run `python paper/scripts/generate_topology_screening_evidence.py` and
  confirm `git status` shows no change to the nine generated screening files.
- [ ] Run `python paper/scripts/generate_mdo_l0_v1_evidence.py` and confirm
  `git status` shows no change to the three generated optimisation files.
- [ ] Run `python paper/scripts/generate_mdo_l0_v2_evidence.py` (it verifies
  both optimisation bundles, the screening dataset and the v1 audit; a few
  seconds) and confirm `git status` shows no change to the three generated
  catalogue-campaign files.
- [ ] Run `python paper/scripts/generate_four_cell_closure_evidence.py` (it
  recomputes the verification from `cft_revival.plasma`; about half a minute)
  and confirm `git status` shows no change to the three generated closure
  files; any change to the plasma package requires re-admission at the new
  revision.
- [ ] Run `python paper/scripts/generate_wall_loss_geometry_screening_v1_evidence.py`
  (it re-verifies the 2,835-file bundle and recomputes every Wilson interval;
  about twenty seconds) and confirm `git status` shows no change to the three
  generated screening-dataset files.
- [ ] Run `python paper/scripts/generate_cusp_topology_v3_1_evidence.py` (it
  verifies the 1,211-file bundle and the 1,211-file predecessor bundle, re-derives
  every estimand from the rows and reproduces the predecessor's audit; about
  five seconds) and confirm `git status` shows no change to the three generated
  topology files.
- [ ] Run `python paper/scripts/generate_l1a_sweep_v3_evidence.py` (it verifies
  the 979-file bundle, re-derives every estimand and the hypothesis statistics
  from the rows and re-reads the review's committed check output; about five
  seconds) and confirm `git status` shows no change to the three generated
  sweep-v3 files.
- [ ] Run `python paper/scripts/check_paper.py`.
- [ ] Run `python -m unittest discover -s paper/tests -v`.
- [ ] Run `python paper/scripts/verify_reproducible_build.py` and require two
  clean byte-identical PDFs.
- [ ] Confirm PDF author/title metadata and absence of volatile creation dates.
- [ ] Save tool versions and check/build output with the submission artifact.
- [ ] Regenerate figures and tables from their declared contract.
- [ ] Verify figure/table provenance sidecars and output hashes.
- [ ] Check that every cited key exists and every bibliography entry is cited.
- [ ] Check DOI links against publisher or official proceedings records.
- [ ] Confirm non-DOI papers are explicitly marked as such.
- [ ] Confirm data/code availability language matches what is actually
  distributed.

## Authorship, ethics, and release

- [ ] Retain Angus Muffatti as the current known original-project author.
- [ ] Satisfy the structured coauthor, contribution, affiliation, and
  corresponding-author approval gates before submission.
- [ ] Record software and data contributor credit.
- [ ] Check licenses and redistribution rights for legacy material, papers,
  datasets, and figures.
- [ ] Remove machine-specific paths, secrets, credentials, and personal data.
- [ ] Disclose model, surrogate, and AI-assisted workflow use as required by
  the target venue.
- [ ] Select target-venue format and copy-edit only after the evidence freeze.

## Response readiness

- [ ] Build the supplementary package in the order defined by
  `supplementary-outline.md`.
- [ ] Answer reviewer requests through claim IDs and manifests.
- [ ] Narrow or remove claims when requested evidence remains unavailable.
- [ ] Rerun checks and record the revision after every response-round change.
