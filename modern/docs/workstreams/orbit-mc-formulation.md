# Full-orbit test-particle formulation

## Scope and authority

`cft_revival.orbit_mc` supplies direct electron first-event probabilities for
dielectric-wall impact, confirmed magnetic reflection, and computational-domain
escape. It is a test-particle model. It does not contain or imply a
self-consistent electric field, collisions, space charge, a sheath, plasma
response, or PIC. A loss-cone expression is never the wall-loss authority.

## Launch ensemble

`build_launch_ensemble` forms a deterministic Cartesian product over kinetic
energy (eV), pitch angle, Cartesian SI position with a flux-surface identity,
both parallel directions, and explicit gyrophases. Every launch has a stable
text identity and SHA-256-derived integer seed. Sorting by launch identity
makes batch size and input order irrelevant to reduction.

The local velocity uses the relativistic speed corresponding to the requested
energy and an orthonormal basis around the local magnetic-field direction:

\[
v=v_\parallel\hat b+v_\perp(\cos\phi\,e_1+\sin\phi\,e_2).
\]

## Flux interpolation

`PsiBicubicField` interpolates the regular variable
\(g=(\psi-\psi_\mathrm{axis})/r^2\). Its axis value is an even extrapolation in
\(r^2\), and \(\partial_r g=0\) at the axis. This removes the Cartesian cusp
that results from dividing a generic ψ polynomial by radius. The field is:

\[
B_r=-r\partial_z g,\qquad B_z=2g+r\partial_r g.
\]

Every query cell must have four matching plasma material tags. Interface and
non-plasma cells are quarantined; no C1 claim crosses a material discontinuity.
Each traversable Hermite cell is converted to its power polynomial. Absolute
coefficient sums bound \(g,\partial_r g,\partial_z g\) over the complete unit
cell, giving a conservative cellwise and map-wide \(|B|\) bound. Reference
Br/Bz and maximum values are checked for consistency and may only increase
the operative bound. Runtime field and cyclotron-rotation checks remain
active on every step.

A deterministic 9×9-per-cell diagnostic reports
\(B_{\mathrm{dense}}/B_{\mathrm{certified}}\). It does not certify safety—the
polynomial bound does that—but detects an impractically loose certificate
before orbit work. The preregistered minimum is 0.001; lower ratios fail closed
as `NOT_EVALUATED` and require certificate refinement. This is a feasibility
gate, not a performance claim.

## Full-orbit integration

The CPU reference uses a predictor followed by midpoint field evaluation and
a relativistic momentum-Boris update. This is second order for varying E and
B under the verified smooth-cell assumptions. The timestep uses the certified
map-wide maximum field:

\[
\Delta t\leq\theta_{\max}m_e/(|q_e|B_{\max}).
\]

It is not enlarged in low-field regions. Optional Warp CPU/CUDA kernels
execute the same float64 relativistic push for parity qualification.

Every completed trial step computes candidate fractions for cylindrical wall,
radial/axial domain escape, exact time deadline, remaining path length, and a
reflection root. Reflection uses bisection on \(v_\parallel=0\). The minimum
fraction wins, with deadline/path priority on exact ties. Thus a later wall
crossing cannot mask an earlier timeout and no timeout overshoot is retained.
Launches outside the plasma bore/domain return `INITIAL_STATE_INVALID`.

The v1.5 resolver treats a positive attempted step from strictly inside a
boundary differently from an invalid boundary launch. If the start is within
the frozen event tolerance, motion is outward, and the computed crossing or
corrected endpoint has zero representable progress, the orbit terminates on
that attempted step. The witness records
`event_resolution=tolerance_close_fraction_zero`, a zero event fraction, the
positive attempted `step_dt_s`, the attempted endpoint/direction, and the
boundary-snapped event position. No midpoint, reflection, or next-step field
query is made beyond that boundary. An interior corrected segment with no
representable progress fails immediately before another field query rather
than spinning to the step limit.

Each v1.2 result retains the complete final-step witness. Impact/escape records
contain the segment and frozen wall/domain geometry; reflection records contain
the signed-\(v_\parallel\) bracket and root; deadline/path/step records contain
pre-step counters and every competing candidate fraction; extreme-relativity
records contain observed gamma or \(v^2/c^2\) and the frozen threshold.
Validation regenerates geometric/deadline candidates and applies the same
fraction/priority ordering. Field, config, and protocol-policy identities are
bound to artifact authority. Where canonical field/config objects are
available, deterministic replay compares the complete result bytes.

## Diagnostics

The accumulated local phase is exactly the requested diagnostic

\[
\Phi_g=\int |q_e|\,|B|/(\gamma m_e)\,dt.
\]

Magnetic moment is computed from perpendicular relativistic momentum,
\(\mu=p_\perp^2/(2m_e|B|)\). A gyro average is emitted only after a complete
phase interval of \(2\pi\); a trailing partial cycle is discarded. Results
also retain endpoint, elapsed time, path length, transit fraction, timestep,
energy drift, instantaneous μ variation, backend, and complete failure
taxonomy. Each record additionally carries its configured time limit, path
limit, and event tolerance so persisted timeout and overshoot claims can be
replayed without trusting an unavailable configuration object.

## Statistical evidence

Direct outcomes are Bernoulli counts with Wilson 95% intervals. Wall,
reflection, escape, and incomplete probabilities are all reported. Reduction
is over launch-ID-sorted records and is invariant to deterministic batch
partitioning. The asymptotic loss-cone comparator is exposed only when
\(\rho/L_B\), μ variation, and complete-cycle gates all pass, and its artifact
label states that it is not wall-loss evidence.

The v1.3 estimator policy is the closed enum `UNWEIGHTED_BINOMIAL`. All launch
weights in the campaign estimand must equal \(1/N\) within binary64 tolerance.
The estimator identity hashes the policy, estimand, sorted launch identities,
and normalized weights independently of batch partition or input permutation.
Unequal weighting is rejected: weighted/stratified variance and confidence
interval semantics have not been accepted or implemented.

Artifact structure and witnesses are necessary but not sufficient for
publication. Deterministic replay against externally bound field, config,
launch, and policy authorities is mandatory at write and verified-load
boundaries. Only the opaque verified evidence token can create a coupling
handoff.

The v1.4 campaign authority additionally requires the externally frozen
batch-manifest hash at replay and every verified-evidence boundary. Because the
hash covers estimator policy, exact batch membership, canonical launch order,
and equal weights, batch partition is part of the estimand execution contract
rather than mutable runtime metadata.
