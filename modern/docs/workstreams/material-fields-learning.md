# Material-fields learning ledger

## Decisions retained

- Solving for `psi=r A_phi` keeps the symmetry-axis condition regular and
  preserves the L1a field reconstruction.
- Reluctivity, not permeability, is the coefficient in the weak form. The
  harmonic face value is required by the series-reluctance interface model.
- Recoil remanence is most safely represented as the weak source vector
  `G=(nu Br_z,-nu Br_r)`; translating it to sheets in the same solve would
  double count the same physical magnetization.
- Exact overlap integration is useful even when the solve uses dominant
  node-centred material IDs: it exposes clipping and geometric representation
  error instead of hiding it.
- Recurrence residuals can drift in finite precision. Artifact acceptance must
  use a freshly applied operator and true residual.

## Remaining learning questions

- Curved interfaces would still need higher-order reconstruction; current
  rectangular and linear tapered/oblique interfaces have exact line clipping.
- Nonlinear B-H activation needs a safeguarded Picard/Newton method with a
  differential-permeability Jacobian and independent energy line search.
- The dipole Robin boundary removes pure zero-box truncation but is not yet
  sufficient: multipole or mapped-infinite-domain work is needed to reach the
  predeclared fixed-QoI domain gate.
- Demagnetization screening needs worst-case local `H` and temperature fields;
  a recoil-line solve alone is not irreversible-demagnetization validation.
- RTX is not a distinct numerical backend in this scalar PDE implementation;
  CUDA parity is the relevant available comparison.

## Audit findings retained

- The weak equation is integrated in meridional `dr dz`, not physical
  `2 pi r dr dz` volume. Using volume weighting changed the PM source action
  and concealed cut-cell bias.
- Boundary decay and domain-QoI convergence remain separate tests. Compact and
  divergent boundary ratios pass (`0.0491%`, `0.0378%`); historical does not
  (`0.4646%`). Worst successive fixed/bore QoI changes are `2.543%`, `3.516%`
  and `2.765%`, all above the predeclared `0.1%`.
- The final controlled study resolves every feature by at least `3.050` cells.
  Base grids are 283x876, 261x1047 and 237x919; 2x fine grids are 566x1752,
  522x2094 and 474x1838. Resolution now passes, but fixed/bore mesh changes are
  still `1.974%`, `14.594%` and `1.601%`; compact/divergent alignment changes
  are `2.244%` and `1.378%`.
- Correcting `M=Br/(mu0 mu_r)` and using the face-adjoint recoil source closes
  the former PM discrepancy. Fine recoil/equivalent gaps are `0.04797%`,
  `0.00542%` and `0.01399%`, all below the unchanged `2%` gate and decreasing
  substantially from their base values.
- Self-hashes authenticate bytes, not physics. The v1.2 evidence path now
  decompresses the bound raw binary64 solution, reconstructs geometry,
  magnetics, coefficients and source, and independently recomputes operator
  action, residual, energy, fields, fixed/bore QoIs, warnings and gates.

## Re-audit findings

- The earlier Robin factor was mathematically incomplete. For
  `psi=C r^2/rho^3`, the `r^2` numerator contributes the essential `-2/r`
  term to radial `alpha`; axial decay carries a factor of three. Correcting
  both reduces worst boundary/peak ratios to `1.10e-5`, but small boundary
  fields still do not imply domain-independent QoIs.
- A controlled low-resolution three-domain comparison gave successive
  fixed-QoI changes of `11.22%/16.22%` for corrected Robin versus
  `11.17%/16.26%` for Dirichlet. The Robin condition is analytically correct
  for the leading dipole but does not cure geometry/grid-alignment error by
  itself.
- Fifth-order fixed-window bore quadrature and twelve-cell active-feature
  grids remove the prior resolution objection. Remaining domain changes are
  `2.223%`, `1.829%` and `0.469%`; mesh changes are `5.302%`, `2.624%` and
  `2.344%`. These are physical/discretization blockers, not algebraic error.
- Device-resident PCG needs device scalar state as well as device vectors.
  Reading each dot product was still a host synchronization. Keeping rho,
  denominator, alpha and beta on-device reduces base synchronization to
  roughly one per 25 iterations plus bounded final checks.
- Linear polygon clipping conserves oblique-region area to roundoff under
  shifted grids. Curved interfaces and a verified body-fitted P2 reference
  remain future work; the divergent result was not promoted without them.
- A common study identity must normalize away only the declared PM
  representation plan and its derived equivalent host. Domain-specific
  handoff hashes remain bound per run, while physical geometry/material,
  configuration and implementation identities remain common.
- Linear iron cannot support a saturation claim, and recoil-line PM fields
  cannot support an irreversible-demagnetization claim. Both warnings are
  machine-readable summary fields.
- Geometry v1.1 topology is bound by exact schema ID and canonical payload
  SHA-256, not a descriptive compatibility string. Every geometry region is
  listed in provenance and every authoritative handoff region, interface and
  PM source is counted.
- Three-grid fixed-QoI sequences are not reliably asymptotic on the structured
  mesh. Historical has zero inferred order for every stage axis/bore QoI;
  compact has positive order only for stage-2 axis/bore (`2.843`/`4.805`) and
  zero for the other eight QoIs. The minimum-order gate therefore correctly
  remains `0.0 < 1.5`; Richardson values cannot promote either result.
- Integer-cell domain extension phase-locks the coordinates, but rounding a
  requested `1.5x` padding can produce a derived ratio just below the unchanged
  gate (`1.499816` historical, `1.499487` compact). The evidence must report
  that failure rather than treating nominal padding inputs as observed ratios.
- Full-resolution replay reports own complete `Br/Bz` matrices. Retaining an
  entire ten-run bundle can exhaust host memory even when CUDA solves fit;
  bounded replay retention and one strict producer validation are required.

## Interrupted v1.4 recovery findings

- Shoelace areas for very small clipped cells must be evaluated after
  translating vertices to a local origin. Global-coordinate products can lose
  enough digits to make a fully covered cell appear larger than itself even
  when the geometry is valid.
- A completed numerical solve is not an artifact checkpoint. Evidence-code
  edits between run observation and sealing invalidate the common identity;
  staged runs need durable raw-observation checkpoints before more large
  structured-grid campaigns are justified.
- Mixed v1.3/v1.4 files are not a screening bundle. Until every design passes
  strict replay under one implementation, retain `SCREENING_NOT_ACCEPTED`,
  treat the structured-grid qualification as incomplete/insufficient, and use
  the body-fitted P2 FEM path for preferred qualification.
- A correct Robin logarithmic derivative does not imply a universally
  positive boundary coefficient or SPD matrix. State only the verified
  small-grid eigenvalue and solution-path energy/curvature evidence.
- Fixed-window quadrature must intersect every field cell and integrate that
  cell's bilinear representation; one global Gauss stencil can miss steep,
  grid-shifted structure.
- Domain sensitivity is interpretable only when `dr`, `dz`, and all retained
  interior coordinates are phase locked. Extending by integer cells separates
  boundary truncation from raster phase.
- High-resolution RTX work is host-memory limited before it is GPU limited:
  rasterization currently owns Python scalar/list working sets. A production
  handoff needs packed NumPy/array storage or direct device rasterization,
  streaming raw-run persistence, and enough host/page-file headroom for at
  least 6.840 GiB for the largest preregistered role plus artifact encoding.
- Page-file capacity is not a substitute for physical headroom in this
  workload. With 120 GiB page file free but less than 0.75 GiB physical free,
  beginning a 1.021 GiB raster would create avoidable thrash and violate the
  40% preflight safety policy.
- Role checkpoints must be atomic and independently hash-verifiable. They
  reduce lifetime and restart cost, but do not make an otherwise unsafe role
  allocation safe; campaign scheduling still needs 17.100 GiB free physical
  memory for the 6.840 GiB historical domain-2 estimate.
- A reduced study can close artifact integrity without answering the
  preregistered qualification question. Bind the ten completed reduced roles,
  mark high-resolution qualification and every publication gate
  `NOT_EVALUATED`, and retain values only as explicitly reduced-resource
  diagnostics. A single failing categorical gate is insufficient because the
  remaining booleans can still be misread as publication passes.
- The artifact full map must come from the exact result that produced the
  checkpointed base run. A second converged CUDA solve may differ at the bit
  level and is correctly rejected by deterministic full-map replay.
- Directory-level staging and replay before replacement prevents a transient
  v1.3/v1.4 manifest from becoming the delivered screening bundle.
- Tri-state status must propagate through schemas, loaders, replay and viewer
  summaries. Rejecting legacy booleans and recomputing status from qualification
  scope prevents a resealed reduced bundle from forging `PASS`.
- A P2 handoff may consume structured-grid QoI values for comparison, but must
  not inherit structured-grid gate statuses; qualification authority remains
  with the independent P2 evidence contract.
