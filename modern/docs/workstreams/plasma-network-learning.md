# Plasma Network Learning Ledger

## 2026-09-02

- [user] A geometry-supported one- or two-cusp interior topology must not be
  forced into the accepted four-cell shape.
- [self] Preserving the accepted residual family order gives closed-form
  dimensions: `6N+1` state values and `7N` equations.
- [self] The accepted rank 22 of 25 is not an isolated four-cell accident.
  N=1...6 manufactured Jacobians follow rank `5N+2`, hence right-nullity
  `N-1`, exactly the number of interior cusp edges.
- [self] Strict residual closure does not imply an identifiable point.
  Publication must either require full rank or persist a complete normalized
  right-nullspace basis.
- [self] N=4 parity can be bit exact, including summation order, while still
  deriving all loops and dimensions from topology.
- [self] N=1 requires a real terminal special case: it has no dielectric cusp
  current or cusp-power term, and the final cell is bounded directly by the
  cathode and anode terminals.
- [self] Geometry classification belongs upstream. The solver validates
  incidence and records excluded finite-boundary null IDs but contains no
  coordinate or field-value windows.
- [self] Mirror ratio can be accepted with uncertainty and provenance without
  pretending it closes an equation. It remains metadata until an independent
  closure is specified.
- [self] Homogeneous dynamic layouts provide a practical future GPU boundary;
  mixed-N batches need deterministic bucketing rather than ragged arrays.
- [tool] The requested named learning/devlog skills were not installed as
  callable skills, so their material effects were preserved in owned ledger
  documents without modifying shared configuration.
- [audit] A backend result is adversarial input. A reported zero residual can
  hide a canonical `0.426` row, and a failure status can accompany an exact
  point; neither status nor backend residual belongs in publication logic.
- [audit] Bounds are admissibility constraints, not variable scales. Rank and
  nullspace use fixed `Ua`/`Ia` scaling and recorded tolerances.
- [audit] Nullspace count is insufficient. Orthonormality, independence,
  finite shape, structural rank, and canonical scaled `Jv` must all close.
- [audit] A frozen outer dataclass is not enough when direct construction,
  replacement, or deliberate attribute mutation can create malformed nested
  topology. Every public boundary revalidates the complete chain.
- [audit] A topology hash that omits covariance, excluded-null reasons, or
  executable/source identities is not a replay identity.

## Open risks

- The source model remains conditional on imposed anode voltage/current and
  empirical energy fractions.
- The named anode-ion sign alternatives remain unresolved by source evidence.
- Nullspace representation reports non-identifiability; it does not create
  uncertainty calibration or a unique physical solution.
- A production geometry adapter, GPU backend, and independent physical
  validation evidence remain future integration work.
