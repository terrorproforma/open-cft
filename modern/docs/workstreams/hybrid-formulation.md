# L2 Hybrid First-Slice Formulation

## Claim boundary

This slice is a verification-grade, prescribed-field kinetic heavy-species
model with a fluid-electron closure interface. It advances weighted Xe, Xe+,
and Xe2+ macroparticles in SI units. It is not self-consistent electrostatic
PIC, a calibrated xenon collision model, or a predictive thruster simulation.

The canonical implementation is the dependency-free Python CPU reference.
NVIDIA Warp float64 kernels provide optional CPU/CUDA execution for the Boris
push and moment deposition.

## Particle state and equations

Each macroparticle stores immutable
`(particle_id, species, position_m, velocity_m_per_s, weight, alive,
velocity_time_level)` state. Particle IDs are unique unsigned 64-bit RNG
identities. A species carries its identifier, symbol, charge state, mass, and
physical charge. Custom identifiers and positive finite masses are preserved.
Charge is not customizable: for supported xenon states it is derived as
`q=z e`. A serialized charge is redundant integrity data and must equal that
derived value exactly before a state can reach dynamics or collisions.

The canonical time state is standard PIC leapfrog:

- position is `x^n`;
- stored velocity is `v^(n-1/2)`;
- prescribed fields are `E^n, B^n`.

`VelocityTimeLevel` enforces this distinction. A synchronous `(x^n,v^n)`
particle can only enter the integrator through `initialize_leapfrog`, which
applies a backward half Boris velocity step without moving position.
`synchronize_velocity` applies the corresponding forward half step for
diagnostics and marks its output synchronous; that output cannot be advanced
collided, or deposited until explicitly reinitialized.
The represented particle has mass `w m_Xe` and charge `w z e`; acceleration
uses the physical charge-to-mass ratio `z e / m_Xe`, so macroparticle weight
does not change a trajectory.

For a field held constant over one step,

`dv/dt = (q/m) (E + v x B)`, and `dx/dt = v`.

The pusher uses the standard kick-rotate-kick Boris update:

1. `v- = v^(n-1/2) + (q E^n/m) dt/2`
2. `t = (q B/m) dt/2`, `s = 2t/(1 + |t|^2)`
3. `v' = v- + v- x t`, `v+ = v- + v' x s`
4. `v^(n+1/2) = v+ + (q E^n/m) dt/2`
5. `x^(n+1) = x^n + v^(n+1/2) dt`

With `E=0`, the rotation conserves speed up to binary64 roundoff. With `B=0`,
initialization plus leapfrog gives exact uniform-`E` velocity and displacement
at every integer time. Step work is evaluated over the two electric half-kicks:

`W_E = q E . [(v_old+v-)+(v+ + v_new)] dt/4`.

This is compared to kinetic energy between `v^(n-1/2)` and `v^(n+1/2)`.
Physical `q E . (x^N-x^0)` is compared between synchronized endpoint
energies. With both fields zero, the method reduces to drift. `dt=0` returns
the original leapfrog particle object.

## Boundaries

An axis-aligned box provides three explicit policies:

- periodic: wrap position and exchange no momentum or energy;
- reflecting: mirror overshoot, reverse the normal velocity component, and
  report equal-and-opposite wall momentum exchange;
- absorbing: mark the particle dead and transfer its represented momentum and
  kinetic energy to the wall/background account.

Arbitrary finite overshoot is folded with a period-`2L` map, avoiding iterative
wall crossing loops.

## Conservative moment deposition

The first mesh is deliberately narrow: periodic, one-dimensional,
cell-centred cloud-in-cell with represented transverse area. For each alive
particle, two shape weights satisfy `S_left + S_right = 1`.

The implementation deposits number, charge, all three current components, all
three momentum components, and kinetic energy:

`M_i = sum_p S_ip M_p / cell_volume`.

Therefore `sum_i M_i cell_volume = sum_p M_p` up to summation roundoff.
Particles outside the declared interval are rejected rather than silently
aliased; the upper periodic endpoint is equivalent to the lower endpoint.
Charge is at `x^n`; current and momentum use `v^(n-1/2)`. Duplicate particle
IDs and synchronous velocities are rejected before deposition.

## Collision fixture

For prescribed neutral density `n_n`, relative speed `g`, and cross section
`sigma(g)`,

`nu = n_n sigma(g) g`,

`P(event) = 1 - exp(-nu dt)`.

The included cross section is an explicitly synthetic constant fixture. It
exists to verify event frequency, deterministic execution, and source
accounting; it must not be presented as xenon collision data.

Charge exchange resets the ion marker velocity to the prescribed neutral
velocity for resonant `Xe+ + Xe -> Xe + Xe+`; the marker tracks the outgoing
Xe+, so represented species count and charge remain unchanged. Xe2+ charge
exchange is rejected even at zero collision probability because its physical
products and source accounting are not implemented. Elastic events
isotropically rotate relative velocity at constant speed. In both implemented
cases, every ion momentum and kinetic-energy change is recorded with an equal
and opposite prescribed-reservoir change.

Random values follow `splitmix64-counter-v1` and are pure functions of
`(seed, particle_id, step, stream, draw)`. Traversal order cannot alter events.
Each counter must be a built-in unsigned 64-bit integer; booleans, floats, and
integer-like objects are rejected rather than converted or aliased.
Aggregate collision probability, momentum, and energy reductions are evaluated
in ascending particle-ID order with stable summation, so input permutation
cannot alter source values.

## Fluid-electron interface

`FluidElectronClosure` separates ion kinetics from future electron models. The
only concrete closure is a verification fixture:

`n_e = rho_i/e`, `p_e = n_e k_B T_e`.

It returns neither an electric field nor anomalous mobility. Its
`anomalous_mobility_m2_per_v_s` value is explicitly `None`. Coupling sources
must satisfy

`Delta p_ion + Delta p_e/background = 0`,

`Delta E_ion + Delta E_e/background = 0`.

This avoids inventing anomalous transport or hiding energy/momentum sinks.

## Checkpoint and provenance

`hybrid-checkpoint-v1` stores simulation time, step, `dt`, particle state,
custom species identifier/mass plus derived charge, RNG algorithm/seed, explicit
`x^n,v^(n-1/2),fields^n` staggering, and provenance. The envelope, payload,
RNG, time-level, particle, species, and provenance objects are closed and
typed. Number fields accept only actual finite JSON integer/float parser values;
strings, booleans, bytes, and float-like objects are never coerced. JSON object
members are parsed recursively with duplicate-key rejection before hashing or
schema validation. Loading also rejects extras, malformed UTC timestamps,
mutable/malformed notes, duplicate identities, unsupported staggering, invalid
SI state, or a changed RNG contract.

Canonical sorted compact JSON is hashed with SHA-256. This detects accidental
corruption or mismatch; it is not a signature, MAC, or authenticity proof. An
actor who edits the payload and recomputes the digest can produce an accepted
checkpoint. Adversarial authenticity would require signed provenance or a
trusted external digest, neither of which this slice claims.

The provenance record distinguishes sourced quantities from synthetic
fixtures and permits a null code revision for an uncommitted working tree.
The machine-readable contracts are in `modern/spec/hybrid/`.

## Evidence

- SI defining constants: <https://www.bipm.org/en/measurement-units/si-defining-constants>
- Xenon standard atomic weight: <https://ciaaw.org/xenon.htm>
- Boris update derivation: <https://www.particleincell.com/2011/vxb-rotation/>
- PIC cloud-in-cell overview: <https://www.particleincell.com/2010/es-pic-method/>
- PIC/MCC implementation reference:
  <https://doi.org/10.1016/S0010-4655(02)00728-7>

The URLs support equations and algorithms, not validation of the synthetic
cross section or of a thruster operating point.
