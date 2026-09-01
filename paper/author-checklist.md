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
