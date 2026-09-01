# Four-cell topology search development log

## 2026-09-02

- Declared a 128-case deterministic shifted-Halton search over stage count and
  pitch, stack offset, alternating polarity/strength, chamber and magnet
  radii/thicknesses, wall radius, and radial/upstream/downstream domain padding.
- Added strict geometry v1.1 generation, binary64-stable variable L1a domains,
  Warp field artifacts, exact-byte experiment acceptance, source/map/artifact
  identity binding, coupling-v2 topology records, and CPU parity sampling.
- Derived four-cell gates from coupling-v2 segment semantics and the corrected
  global model: exactly four ordered interior segments, no boundary samples,
  centreline/wall role validity, positive finite non-inverted mirrors,
  confidence/prominence, valid probabilities, and field solver/source/domain
  gates.
- Added unchanged deterministic nine-start global-plasma attempts at three
  predeclared hypothetical operating points for every compatible candidate.
  Nonconverged branches retain diagnostics but cannot publish state,
  conservation, or performance.
- Corrected the source-strength transform after a focused test showed that the
  accepted L1a preview emits an opposite-current inner/outer sheet pair per
  stage. Both sheets now receive one stage-level strength multiplier.
- Final search: 128/128 strict geometries and Warp fields evaluated, 2 passed
  every four-cell and field gate, both converged at all 3 operating points,
  and all 8 CPU parity cases passed. The overlapping failure taxonomy recorded
  68 field-gate, 118 topology-count, 36 boundary-leakage, and 61 mirror-order
  failures.
- Validation: 8 focused experiment tests and 141 focused/compatible tests
  passed; source/tests/experiment compileall passed; FYP status and diff stayed
  clean. Ruff was not installed and was not added.

## 2026-09-02 semantic publication correction

- Reclassified v1 as non-preregistered development evidence invalid for
  physical mirror, identifiable-state, and performance claims.
- Replaced the prior six "converged state/performance publication" labels with
  six rank-22/25 non-identifiable screening-equation residual roots.
  `performance_publication_count` and `identifiable_state_count` are both zero.
- Removed state vectors, power objects, and performance-publication objects
  from plasma outcomes. Preserved their exact numerical values in a separate
  audit-only archive; residual rows, conservation closures, solver diagnostics,
  and Jacobian rank remain in the outcomes.
- Recorded that coupling v2 used a deprecated same-z mirror proxy and
  roundoff-scale null lows. Coupling v3 and a preregistered search v2 are both
  required before physical mirror or performance claims.
- Regenerated only dataset/report/manifest metadata and hashes. No field or
  plasma simulation was rerun, no selection/ranking changed, and all
  representative field/geometry artifacts retained their prior hashes.
- Validation after correction: 9 focused semantic tests and 154
  focused/compatible tests passed; compileall passed; FYP status/diff remained
  clean.
