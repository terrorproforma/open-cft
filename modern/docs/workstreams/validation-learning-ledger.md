# Validation Workstream Learning Ledger

## Retained lessons

- A user-selected claim field is not authority. Evidence kind and source
  authority need a closed pairing and an independently enforced claim ceiling.
- PIC remains simulation evidence regardless of fidelity, replicate count, or
  whether a publication reports it. Only an actual experiment with complete
  measurement metadata can carry experimental-truth provenance.
- Context-of-use evidence is multi-dimensional. Matching only record IDs or
  credibility allows the wrong quantity, unit, model revision, operating point,
  or dependent hardware group to satisfy a requirement.
- Conservation and convergence pass/fail values are unsafe when detached from
  the evidence hash, provenance, quantity, units, contexts, revisions, group,
  and mesh identities that produced them.
- Binding a candidate hash to a precomputed conservation result is still unsafe:
  the result values can be fabricated. Store raw residual observations and gate
  policy, then recompute arithmetic and status in assessment and reporting.
- A convergence-study-level binding is necessary. A candidate-level identity
  does not protect coarse/medium errors or a changed finest observation that
  reuses the same record ID and mesh.
- Relative residual tolerances without a declared scale hide normalization.
  Require a positive, dimensionally identical, exact-unit scale.
- Grid spacing is a physical quantity, not a float. Unit-bearing spacing and
  unique mesh IDs prevent unrelated refinement sequences from being combined.
- A normal 1.96 multiplier understates uncertainty for small replicate counts.
  The Student-t interval makes finite-sample policy explicit; fewer than three
  PIC runs is rejected as an ensemble.
- Hash verification and schema validation solve different problems. Verifying
  the hash first catches accidental mutation; validating afterward rejects a
  maliciously rehashed but malformed payload. Neither authenticates origin.
- Source wording and editorial interpretation must be separate. Exact native
  labels remain citable while explanatory language remains visibly editorial.
- Audit objects become stale. Reports should derive status from the current
  content-addressed registry. Empty or insufficient evidence is
  `NOT_EVALUATED`; `FAIL` is reserved for an evaluation that actually ran and
  failed.
- Caller-chosen group labels are not independence evidence. Physical/design/run
  identifiers belong in immutable record identity, and context audit must also
  detect overlap with calibration records.
- URL parsing needs authority validation beyond checking scheme/netloc. DOI
  shape and binary64 overflow need explicit checks and domain error wrapping.
- Prefixing validation test basenames avoids future flat-import collisions
  without changing repository-wide pytest behavior.

## Remaining evidence needs

- Authenticated publication/source distribution if adversarial tampering is in
  scope; unkeyed SHA-256 alone cannot provide it.
- Independently acquired experiment records and uncertainty budgets.
- Pre-registered PIC seed policies and replicate counts for production studies.
- Decision-specific context-of-use thresholds and empirically supported
  applicability ranges.
