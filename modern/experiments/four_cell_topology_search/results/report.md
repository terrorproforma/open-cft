# Four-cell topology-targeted L1a search

- Classification: `DEVELOPMENT_EVIDENCE_INVALID_FOR_PHYSICAL_MIRROR_OR_PERFORMANCE_CLAIMS`
- Protocol status: `development_evidence_only`
- Preregistered: `False`
- Physical mirror claims valid: `False`
- Performance claims valid: `False`
- Requested/evaluated: 128/128
- Topology compatible: 2
- Residual-root candidates: 2
- Plasma residual roots: 6
- Identifiable states: 0
- Performance publications: 0
- CPU/Warp parity cases: 8 (failures: 0)

## Best configurations

### four-cell-029-52bb37501f

- Compatible: `True`
- Stages/pitch: `4` / `0.00457338285 m`
- Segments/confidence: `4` / `0.986693066`
- Probabilities: `[7.465510152148198e-18, 6.087321618352463e-17, 1.0806426773395197e-16, 1.2099026421117728e-16]`
- Plasma residual floors: `[1.6653345369377348e-16, 7.401486830834377e-17, 3.3306690738754696e-16]`
- Plasma classification: `non-identifiable screening equations only`

### four-cell-005-8885e09139

- Compatible: `True`
- Stages/pitch: `4` / `0.00482029643 m`
- Segments/confidence: `4` / `0.983505227`
- Probabilities: `[1.3407251653889322e-17, 5.871968472318331e-17, 4.785299685293601e-17, 1.011897268778225e-16]`
- Plasma residual floors: `[1.1102230246251565e-16, 1.4802973661668753e-16, 5.380140777333509e-13]`
- Plasma classification: `non-identifiable screening equations only`

### four-cell-059-80ac2000e2

- Compatible: `False`
- Stages/pitch: `4` / `0.00488202482 m`
- Segments/confidence: `4` / `0.989656877`
- Probabilities: `[3.136447760894617e-18, 0.0002688212600038058, 4.431944098671781e-18, 1.6208541591088415e-16]`
- Plasma residual floors: `[]`
- Plasma classification: `not attempted; incompatible gates`

### four-cell-123-1ed2d0b426

- Compatible: `False`
- Stages/pitch: `4` / `0.00554046103 m`
- Segments/confidence: `4` / `0.988970163`
- Probabilities: `[0.0, 0.0025565470398894842, 1.2627682875258607e-16, 6.711096726558953e-17]`
- Plasma residual floors: `[]`
- Plasma classification: `not attempted; incompatible gates`

### four-cell-093-4518cdf2d5

- Compatible: `False`
- Stages/pitch: `4` / `0.00523181906 m`
- Segments/confidence: `4` / `0.987578336`
- Probabilities: `[0.0020971944245460677, 1.2152255363929254e-17, 2.087397912286995e-16, 0.00592785115374344]`
- Plasma residual floors: `[]`
- Plasma classification: `not attempted; incompatible gates`


## Failure taxonomy

- `BOUNDARY_LEAKAGE`: 36 — a selected cusp depended on a finite-domain boundary sample or margin
- `CONFIDENCE_FAILURE`: 0 — candidate, prominence, segment, or overall confidence failed
- `COUPLING_REJECTED`: 0 — coupling v2 rejected evidence or mirror projection
- `FIELD_GATE_FAILURE`: 68 — field residual, identity, source, or boundary gate failed
- `FIELD_SOLVER_FAILURE`: 0 — Warp field solve or strict L1a artifact validation failed
- `GEOMETRY_INVALID`: 0 — accepted geometry v1.1 construction or strict validation failed
- `MIRROR_INVERTED`: 61 — wall/cusp fields or mirror ratio were inverted, zero, or nonfinite
- `ORDERING_FAILURE`: 0 — segments or cusps were not strictly ordered and contiguous
- `PARITY_FAILURE`: 0 — selected CPU/Warp parity case exceeded the declared scale-relative gates
- `PLASMA_NONCONVERGENCE`: 0 — strict deterministic global-plasma multi-start found no residual root
- `PROBABILITY_INVALID`: 0 — coupling-v2 loss probability was nonfinite or outside [0,1)
- `ROLE_MISMATCH`: 0 — centreline or wall profile role/radius was invalid
- `SOURCE_INVALID`: 0 — L1a current-equivalent source construction failed
- `TOPOLOGY_COUNT`: 118 — resolved coupling topology did not contain exactly four segments
- `TOPOLOGY_STATUS`: 0 — coupling topology was not resolved

## Claim boundary

Version 1 is non-preregistered development evidence. Coupling v2 used
a deprecated same-z mirror proxy and roundoff-scale null lows. Therefore
its nominal mirror ratios/probabilities are not physical mirror results.
The six rank-22/25 outcomes are non-identifiable screening-equation
residual roots, not plasma-state or performance publications. Raw state
and power values are retained only in the audit archive and are prohibited
from physical/performance use. Coupling v3 plus a preregistered search v2
are required before any such claims. Timings are not claimed.
