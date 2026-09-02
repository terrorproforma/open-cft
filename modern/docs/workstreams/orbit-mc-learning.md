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
- [numerics] A geometrically valid crossing can round to fraction zero even
  when the attempted timestep is positive. Multiplying the timestep by that
  fraction erases the motion evidence and can create a boundary stall. Preserve
  the attempted step, type the tolerance-close resolution, snap the endpoint,
  and terminate before another field query.
- [evidence] Fraction zero is not sufficient event evidence. The start must be
  strictly inside and within tolerance, attempted motion must point outward,
  the snapped endpoint must lie on the claimed surface, and earlier candidates
  must remain absent or later under the frozen priority.
- [operations] Campaign preflight should reject invalid launch geometry,
  launch-field states, max-B declarations, and timestep policy before batch
  scheduling. It is a readiness check, not a substitute for deterministic
  result replay.
- [numerics] (2026-09-02, v3 post-mortem) The zero-step stall is not
  triggered by the obvious case. A launch one ulp inside the wall with purely
  radial velocity terminated correctly on v1.4 because the one-ulp corrected
  step landed exactly on the wall. The stall needs a grazing approach angle,
  or a non-uniform field in which the midpoint-corrected segment repeatedly
  lands just inside the wall and converges to it. Reproduce by sweeping angles
  and offsets, not by a single hand-picked launch.
- [numerics] Zero-step failure has two faces in v1.4: the geometric stall
  (`step_limit` witness with `step_dt_s = 0`, rejected by the validator) and
  an unhandled `ZeroDivisionError` when the elapsed time lands exactly on the
  deadline by rounding. Any fix must keep every step's attempted timestep
  positive, not just the geometric candidates.
- [evidence] Snapping the endpoint onto a boundary moves the particle without
  adding path. Every derived ratio whose denominator is path length must bound
  the denominator by the displacement itself, otherwise a first-step snap
  produces an out-of-range `transit_fraction` that the schema rejects. The
  real campaign never showed this because every real orbit had accumulated
  millimetres of path before its snap; only the synthetic step-1 regression
  exposed it.
- [evidence] Roughly 40% of real primary-N orbits (202/512) ended through the
  tolerance-close path. The v3 failure was not a rare edge case; the fix is on
  the campaign's main path and must stay covered by real-field shakedowns.
- [protocol] Energy conservation of completed Boris steps is exact, but the
  linearly interpolated event velocity is a chord of the rotation and reports
  up to 6e-4 relative energy loss at a 0.16 rad rotation policy. A gate of
  1e-10 on `maximum_relative_energy_error` will reject every fractional event;
  decide whether to renormalize interpolated speed or gate on completed steps
  before preregistering v4.
- [operations] A source edit while a campaign is running invalidates later
  checkpoints through `code_identity()`. Treat this as a feature and finish
  all code edits before starting a shakedown or campaign.
- [performance] The Warp backend launches one kernel per push for one
  particle; on CUDA it is ~18x slower than numpy for this workload. The CPU
  reference at ~90 ms/orbit (primary-N) and ~380 ms/orbit (4N) is the campaign
  path; 9 cases × 512 launches is ~30-40 CPU minutes.

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
