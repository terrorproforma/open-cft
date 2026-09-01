# Validation Workstream Integration Contract v2

## Authority before claims

Every record declares both an evidence kind and a source authority. The package
enforces this pairing and applies an authority claim ceiling:

- analytical and manufactured references: verified implementation;
- independent code, simulation/PIC, and published model output: cross-model
  agreement;
- experiment: predictive validity, but only with complete experiment metadata,
  uncertainty for every observation, and explicit measured-truth provenance.

Choosing a more ambitious `maximum_claim` cannot upgrade source authority.
Simulated PIC is never experimental truth. Published PIC remains a published
model benchmark and cannot become validation evidence merely by placing it in a
different partition.

## Context-of-use evidence

A context-of-use ledger names exact required quantities and SI units, minimum
record and independent-group counts, applicability parameters/ranges, model and
result contexts, and model/code revisions. Independence comes from immutable
design, run-lineage, hardware-article, campaign, and specimen identifiers—not
`group_id`. The context audit also compares selected validation evidence against
all calibration records and fails shared physical identities. Missing evidence
is returned as dimension-specific diagnostics. For example, an `elapsed_time`
observation cannot satisfy a `thrust` requirement.

The ledger is NASA-STD-7009B-style evidence bookkeeping. It is not compliance,
qualification, endorsement, or certification; `certification_claimed=true` is
rejected.

## Numerical gates

Conservation gates bind the raw residual observation to an evidence ID and
canonical hash plus provenance, model/result context, model/code revisions,
immutable run/design identity, quantity, exact unit, and mesh. Assessment stores
the gate policy—not a result—and recomputes residual, threshold, normalized
value, and status. A precomputed result cannot be supplied. A relative tolerance
requires an explicit positive same-unit scale.

Every convergence level embeds a complete immutable evidence record. Error and
spacing are looked up from named observations and the level content hash,
provenance, model/result context, revisions, immutable run/design identity,
quantity/unit, and mesh are validated. Orders are recomputed from those bound
observations. The finest level content hash must equal the assessed candidate,
so changing its value while retaining its ID and mesh fails.

## PIC replicates

An ensemble needs at least three unique seeds from one named seed policy.
Authority, model/code/result context, partition, group, mesh, operating
parameters, quantity names, and exact units must be homogeneous. Reported 95%
mean intervals use exact two-sided Student-t critical values through 30 degrees
of freedom and conservative lower-degree-of-freedom table values above that.
This quantifies replicate variation and does not confer experimental authority.

## Persistence and reports

Published bundles verify their declared canonical SHA-256 before applying the
closed nested schema. Duplicate keys, unknown keys, nulls, nonfinite or
binary64-overflowing values, booleans in numeric fields, invalid enums, invalid
ranges, and malformed source locators are rejected. Web sources require an
absolute HTTP(S) host without whitespace, controls, user information, or invalid
authority. DOIs match `10.<4-9 digits>/<nonspace suffix>`.

SHA-256 is integrity/version identity, not authentication. Anyone able to
replace a payload can recompute its unkeyed hash; distribution needs a separate
authenticated channel.

Reports accept only a registry and recompute its bound partition audit. Empty,
reference-only, or one-sided calibration/validation registries are explicitly
`NOT_EVALUATED`; evaluated physical-identity leakage is `FAIL`; only sufficient
disjoint calibration and validation partitions are `PASS`.

Machine-readable contracts:

- `modern/spec/validation/evidence-contract-v2.schema.json`
- `modern/spec/validation/integration-contract-v2.json`
