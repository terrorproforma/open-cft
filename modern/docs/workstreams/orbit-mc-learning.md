# Orbit Monte Carlo learning scratchpad

## Retained lessons

- [physics] A duration stated in tesla or derived from a nominal gyroperiod is
  not a physical termination criterion. Wall-loss authority needs first-event
  full orbits continued to wall, reflection, escape, or preregistered physical
  path/time limits.
- [physics] Both physical orbit rotation and accumulated gyro phase use
  \(|q|B/(\gamma m)\). Omitting gamma overcounts complete cycles for
  relativistic particles; a gamma-two orbit exposed this defect.
- [numerics] Interpolating Br and Bz independently needlessly breaks flux
  consistency. Interpolating ψ and differentiating one C1 surface preserves
  the axisymmetric relation; the axis must use the regular
  \(\partial_{rr}\psi\) limit.
- [numerics] Complete gyro averages require phase-weighted bin closure. A
  trailing partial cycle must not be labeled a gyro average.
- [statistics] Deterministic seeds are insufficient if reduction depends on
  task completion order. Sorting by immutable launch identity before both
  execution batches and reduction makes batch size irrelevant.
- [evidence] Wall-hit, reflection, escape, and incomplete counts all need
  uncertainty and provenance. Suppressing numerical/time-limit outcomes biases
  the wall probability.
- [evidence] Loss-cone theory is useful only as a gated asymptotic comparator.
  It cannot become the direct wall-loss authority merely because μ appears
  stable over a short trajectory.
- [GPU] A CUDA availability string is not parity evidence. Run the same
  float64 relativistic push on the actual device and compare its output to the
  CPU reference before making a GPU claim.

## Corrections made during implementation

- Initial helix verification used the nonrelativistic cyclotron frequency for
  the analytic trajectory. The Boris pusher was correct; the analytic
  reference was not. Including launch gamma restored N/2N/4N orders of
  `1.9984/1.9996`.
- The initial Warp charge-to-mass decimal did not match the exact constants
  used by the CPU path and produced a relative discrepancy near `9.5e-13`.
  Binding the kernel literal to the exact declared constants reduced observed
  CPU and CUDA differences to zero.
- The first magnetic-bottle scale required too many interpreted Python steps
  for a focused check. A geometrically similar, stronger-curvature
  divergence-free bottle retained the same analytic mirror relation and
  completed quickly without weakening its 3% registered error gate.
- A floor-based phase-bin boundary could round back to the just-completed
  cycle and produce a zero-width infinite loop. Tracking explicit in-cycle
  progress and finalizing within an ulp-scaled tolerance removed that failure;
  the 30-cycle grad-B test now exercises the corrected path.
- [audit] Checking wall intersections before time/path limits allowed a later
  wall crossing to win. Representing every event as a fraction of the same
  trial segment makes earliest-event ordering explicit and testable.
- [audit] Nodal or caller-supplied maxima are not timestep certificates.
  Bounding every regular-flux polynomial and its derivatives over each whole
  traversable cell closes interior-overshoot and axis gaps.
- [audit] A hash only proves byte identity. Exact Wilson recomputation,
  ordered campaign/launch/result identities, result-count replay, closed
  checkpoints, and externally anchored checkpoint file hashes are required
  to reject internally rehashed semantic lies.
- [audit] The first field scheme sampled varying E at the old position and had
  no second-order varying-field claim. Predictor/midpoint field sampling now
  demonstrates orders `2.004/2.017` on a nonuniform electric field.
- [evidence] An internally self-consistent checkpoint hash is not immutable
  authority. Supplying the expected campaign ID and launch-manifest hash from
  outside the checkpoint is what makes coherent launch substitution detectable.
- [evidence] Closed JSON shape validation is weaker than evidence validation.
  Persisting physical limits in each result allows runtime replay of event,
  time, path, phase/cycle, state, energy, transit, and μ consistency.
- [numerics] A safe polynomial bound can still be too conservative for a
  feasible preregistered run. Dense/bound ratio is a diagnostic only; failing
  its floor must stop preflight without weakening or replacing the certificate.
- [evidence] A valid termination enum plus plausible final state is not event
  evidence. The before/after segment, every competing fraction, frozen geometry
  and counters, and reflection/gamma witnesses are needed to replay selection.
- [evidence] A checkpoint is legitimately partial only relative to a frozen
  batch manifest. Complete-batch unions plus one explicit ordered prefix make
  completed and pending coverage independently recomputable.
- [evidence] Mutable artifact metadata cannot authorize its own acceptance
  floor. Protocol identity and the 0.001 certificate ratio must enter through
  external validator arguments and agree with embedded records.
- [evidence] Witness validation proves local event semantics; deterministic
  replay is stronger when canonical launch/config/field artifacts are present,
  because it also catches plausible changes outside witness invariants.
- [statistics] Wilson intervals describe unweighted Bernoulli counts. Merely
  storing normalized weights does not make Wilson uncertainty valid for an
  unequal design; unequal weights must fail until a rigorous weighted estimator
  and variance contract are separately accepted.
- [evidence] Structural validity is not a capability. Returning a distinct
  non-mapping unverified type prevents accidental use, while an opaque verified
  token records that external authorities and deterministic replay were applied.
- [evidence] Replay must happen before bytes are published, not as an optional
  later audit. Otherwise a structurally valid but dynamically false result can
  acquire a publication hash and leak into downstream integration.
- [evidence] Launch-set authority does not freeze batching. If batch membership
  affects checkpoint completion and resume semantics, its hash must come from
  outside the artifact at replay, sealing, verified load, finalization, and
  handoff. Internal coherent rehashing cannot substitute for that authority.

## Follow-up risks

- Real canonical maps can contain null-adjacent regions where launch basis,
  timestep sufficiency, and μ interpretation require explicit preregistered
  exclusion/failure handling.
- Warp parity qualifies the relativistic pusher inside the shared host event
  loop. Field interpolation and event location remain host operations.
- The v4.2 handoff is export-only. Integration remains pending until a public
  consumer is implemented and exercised.
- A real CFT probability remains unevaluated until launch weights, all three
  map identities, N/2N/4N controls, and a held-out geometry family are frozen
  before running the campaign.
