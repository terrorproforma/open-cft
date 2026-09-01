# Plasma Network Integration Plan

## Geometry handoff

1. Implement `GeometryTopologyAdapter.plasma_topology_snapshot()` in the
   geometry-owned integration layer.
2. Supply uniquely identified cells with contiguous axial order, all
   classified null candidates, and the cathode/anode adjacency.
3. Mark only geometry-supported interior cusps as `INTERIOR_CUSP`; mark
   finite-domain artifacts as `FINITE_BOUNDARY_NULL` with reason/confidence.
4. Carry geometry/material/source/artifact/model/code/schema hashes, ordered
   positions, confidence, complete loss covariance, and nested provenance.
5. Call `build_chain_topology()` and persist `identity_sha256` beside results.

No coordinate or field-magnitude window belongs in the plasma solver.

## Coupling handoff

The coupling layer should provide anode voltage/current and the terminal
arrival probability, select the named anode-ion branch, and preserve source
provenance. Interior probabilities come from topology edges rather than a
four-element field. Existing four-cell records can use
`from_accepted_four_cell()` only as a compatibility bridge.

Consumers must handle a null state. Backend convergence is not a trusted fact;
canonical re-evaluation can reject a candidate because of residual,
conservation, rank, inequality, finite-value, or bound gates.
When `REPRESENT_NULLSPACE` is selected, persist the complete identifiability
record; do not collapse it to a nominal point.

## GPU follow-on

Group points by `cell_count` and topology layout, then pass flat state/input
arrays to a backend implementing `LeastSquaresBackend`. Preserve generated
row order, exact scales, deterministic start identity, and CPU parity tests.
Ragged mixed-N batches require bucketing before kernel launch.

## Promotion gates

- geometry adapter contract and provenance replay;
- generated-ledger identity and schema validation;
- N=4 row parity for both named anode-ion branches;
- N=1...6 residual/Jacobian CPU parity;
- deterministic fail-closed behavior under malformed and nonfinite inputs;
- explicit uncertainty or nullspace representation for rank-deficient output;
- independent evidence before any physical validation or predictive claim.

## Current limitations

The formulation inherits prescribed anode current, empirical energy fractions,
and the unresolved named anode-ion sign alternatives. Mirror ratios are
accepted with uncertainty and provenance but are metadata until an independently
specified closure uses them. No sheath, neutral, mass-flow, plume, lifetime,
facility, or experimental-validation model is introduced.
