# Independent L3 PIC-MCC Foundation

## Claim boundary

This workstream provides correctness-tested reduced electrostatic PIC-MCC
kernels and a small CPU integration smoke. It is not a generally verified PIC
reference, a CFT discharge prediction, a calibrated xenon model, or an
independent validation of any legacy output.

The integration smoke is periodic Cartesian 1D3V with an explicit transverse
area (default `1 m2`). Its purpose is to establish conservative, testable
kernels before axisymmetric geometry, physical boundaries, xenon chemistry,
cathodes, walls, and measured validation data are admitted.

## Numerical model

Charge density and potential are node-centred on a uniform periodic mesh.
Electric field is face-centred:

`E_(i+1/2) = -(phi_(i+1)-phi_i)/dx`.

This gradient is the one whose adjoint divergence forms the Poisson operator,
so it retains the Nyquist field and satisfies the tested Poisson energy
identity. Particle force gathering first averages adjacent faces symmetrically
to nodes and then applies nodal CIC. That explicit smoothing retains the
periodic single-particle zero-self-force property; it does not claim to retain
Nyquist content in the particle force.
Particles use first-order cloud-in-cell (CIC) shape functions:

`rho_i = sum_p(q_s W_s S_i(x_p))/(A_perp dx)`.

`A_perp` is the explicit positive `Grid1D.transverse_area_m2`; its default is
named as `1 m2`, not hidden. Charge is integrated with `A_perp dx`. Particle,
field, and total energies are all joules over that same represented volume.

The use of the same shape for deposition and gather gives a tested discrete
adjoint identity. Integrated deposited charge is tested against represented
macro-particle charge, including particles adjacent to the periodic seam.
This is a global charge-deposition invariant. A charge-conserving current
deposition and local discrete continuity equation are not yet implemented.

The field solve removes the mean charge required by periodic solvability and
solves

`-D2(phi) = (rho - mean(rho))/epsilon_0`

with binary64 conjugate gradients in the mean-zero subspace. The right-hand
side is normalized before iteration, and residual norms use max scaling before
squaring. Publication requires finite source, iterates, potential, field,
initial norm, tolerance, and an independently recomputed true residual.
`inf <= inf` can therefore never publish convergence. Failure to meet the
finite `max(atol, rtol*r0)` contract either raises `PICConvergenceError` or
returns explicit nonconverged diagnostics.

Electrostatic motion accepts `x^n, v^(n-1/2)` and uses kick-drift leapfrog to
publish `x^(n+1), v^(n+1/2)`; checkpoint velocity is therefore half a step
behind its stored position. A standalone 3-V Boris kernel establishes the
future magnetic-pusher boundary.
The reduced integrated step is deposit → Poisson → gather/kick/drift → optional
MCC → Poisson at the new positions. It operates on a working particle copy and
publishes only after all checks pass. Diagnostics include charge, Poisson
residual, and time-centred energy
`K(v[n+1/2]) + 0.5*(UE[x[n]]+UE[x[n+1]])`, all in joules.

## Collision evidence policy

The elastic MCC probability is

`P = 1 - exp(-n_target sigma(E) |v| dt)`.

The operator is deterministically seeded, linearly interpolates energy tables,
enforces a maximum event probability, and isotropically randomizes direction
while preserving speed under the stationary infinite-mass-target assumption.
It first validates every probability and proposes all events, velocities, RNG
state, and counters against a cloned generator. Only then are particles, RNG,
and counters committed. A late-particle failure leaves all three unchanged.

Only synthetic verification tables are shipped. They test interpolation and a
closed-form binomial event rate; they are not xenon data. A future LXCat parser
must retain the source identity and SHA-256 of the exact downloaded bytes.
Parsed/normalized table hashes do not replace that source-byte hash.

## Verification matrix

- Low Fourier modes verify potential, staggered face-field amplitude and sign.
- A Nyquist mode verifies nonzero face field and the Poisson energy identity.
- Periodic one-particle cases verify resolved self-force is zero.
- Charge integration and the CIC deposition/gather adjoint identity are tested.
- Constant-field acceleration verifies charge-to-mass sign and leapfrog order.
- Magnetic-only Boris motion preserves speed.
- A normalized cold-plasma mode crosses zero at the expected quarter period.
- Reducing `dt` from 0.1 to 0.02 reduces the four-second energy envelope by
  about `1.93x` for the current deterministic fixture. Acceptance requires
  only `coarse_envelope > 1.5*fine_envelope`. This is one refinement trend,
  not a measured convergence order, general heating bound, performance
  benchmark, or claim of exact energy conservation.
- Seed replay and a 20,000-trial synthetic binomial test verify MCC behavior.
- One-iteration multi-mode Poisson explicitly exercises nonconvergence.
- Genuine Warp float64 CIC and gather/push kernels match the Python reference
  on Warp CPU and CUDA when those optional devices are available.

## Stability and admission gates

`PICStepper.step` enforces particle cell crossing `|vx|max*dt/dx` and
`omega_p*dt` before the step, after the push, and after collision scattering.
MCC rejects event probability above its configured limit before publication.
The defaults are 1.0, 0.2, and 0.2 respectively. Cyclotron gating is not
applicable because the integrated step is electrostatic; the standalone Boris
kernel does not imply magnetic integration. These are necessary gates, not
proof that a grid resolves Debye length, sheaths, gyro-motion, or mean free
paths.

`stability_report` revalidates all mutable position and three-velocity arrays,
their equal nonzero dimensions, particle bounds, and current grid/species/config
scalars before evaluating even a zero-density case. It publishes only finite
Courant and plasma-frequency metrics; otherwise it raises a typed PIC error.

Charge deposition computes represented charge and per-particle volumetric
density with mantissa/exponent scaling and one final binary64 rounding.
Accepted nonzero represented charge must produce nonzero finite volumetric
density and pass an integrated-charge publication check on Python and Warp.
Extreme area/charge combinations whose density is below binary64 range are
rejected before particle mutation or Warp launch rather than deposited as zero.

Before physical CFT use, a future workstream must add and verify at least:

1. axisymmetric conservative volumes and axis regularity;
2. charge-conserving current deposition and local discrete continuity;
3. physically justified open/material boundaries and charge accounting;
4. externally sourced, hashed xenon cross sections and collision kinematics;
5. multi-species ionization/source coupling and current closure;
6. mesh/timestep ensembles and numerical-heating budgets;
7. WarpX/PICMI parity against this reference and independent benchmarks;
8. comparison to measurements not used for calibration.

## Adapter and integration status

`PoissonSolver` is geometry-neutral at the orchestrator boundary.
`WarpXPICMIAdapter` defines input construction and one-step result boundaries
without importing WarpX into the reference package. `LXCatParser` requires an
exact source SHA-256. During this implementation WarpX/PICMI and AMReX were not
installed and nothing was installed.

The CPU reduced step has only a small correctness integration smoke. Warp
deposition and face-gather/push have small CPU/CUDA parity checks; no integrated
Warp Poisson step is claimed.

Checkpoint v2 uses a closed schema and canonical finite-JSON SHA-256. Validation
reconstructs typed grid/species/config/particle state, rejects inconsistent or
empty arrays, checks step/time/staggering, and requires code-revision,
backend/device, and runtime identity. Optional MCC state includes validated RNG
state, source-table hash, configuration, and consistent counters. Provenance
records the same revision, runtime, backend/device, staggering, optional
dependency availability/versions, data hash, and reduced-claim boundary.
Field arrays are not checkpointed; face fields are deterministically
recomputed from stored positions at the integer position step.
