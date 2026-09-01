# Physics Workstream: Verified L0 Foundation

## Scope and status

This workstream implements a conservation-based xenon performance baseline and
manufactured magnetic-field fixtures. It is intentionally isolated under
`cft_revival.physics`. It does not translate the uncertain Kornfeld residual
system, fit a closure, solve a plasma, or claim that a finite-element solver
exists.

The CPU implementation has no third-party runtime dependency. NVIDIA Warp is
optional and evaluates the same equations in float64 on an explicitly selected
`cpu`, `cuda`, or `cuda:N` device.

Python performs shared exponent-aware canonical preprocessing before either
backend. Warp receives stable primitive rates, speeds, powers, and boundary
values, then evaluates the batched species/momentum reductions in one launch.
This is necessary because Warp 1.14 exposes no `frexp`/`ldexp` kernel intrinsics.

## API

The public workstream import is:

```python
from cft_revival.physics import (
    BeamDivergenceFactors,
    ChargeStateFractions,
    MassUtilization,
    PowerBoundaryInputs,
    PropellantMassFlow,
    XenonOperatingPoint,
    evaluate_performance,
)
```

`evaluate_performance(point)` returns particle rates, Xe+/Xe2+ velocities,
undiverged and axial thrust, specific impulse, a complete reported power
budget, conservation residuals, and structured applicability warnings.

The optional batch backend is deliberately imported separately:

```python
from cft_revival.physics.warp_backend import evaluate_performance_warp

batch = evaluate_performance_warp(points, device="cuda:0")
```

The field fixture API is `UniformAxialFieldFixture(B0)`. It supplies
`A_phi=B0*r/2`, analytic `Br=0`, `Bz=B0`, a cylindrical-curl evaluation, and
an axis-regularity residual.

## Model boundaries

- All public dimensional fields include SI units in their names.
- A running operating point requires finite `mdot_Xe > 0` and `Vd > 0`.
  Zero-flow and zero-voltage states are not accepted as thruster operating
  points.
- The L0 nonrelativistic gate requires computed Xe2+ speed to remain at or
  below `0.01 c`. Inputs that overflow any derived scalar are rejected before
  launch or result publication.
- Charge-state values are number fractions of all xenon particles and must sum
  to one across Xe, Xe+, and Xe2+. Preprocessing sums the exact rational values
  of the three binary64 inputs and normalizes only discrepancies within two
  exact ULPs at unity.
- Mass utilization is the accelerated Xe+ plus Xe2+ mass fraction. Because all
  represented xenon states use the same atomic mass, it must equal `f1+f2`.
- `beam_current_fraction_of_anode_current` maps charge-conserving beam current
  to anode current. It is an external input in `(0,1]`.
- `axial_momentum_fraction_of_ion_momentum` applies only to thrust. It is an
  external input in `[0,1]` and does not remove kinetic beam power.
- Anode input is `Vd*Ia`; cathode input is separately reported; thruster
  electrical input is their sum; PPU input is the outer reported boundary.
- Canonical PPU required load uses stable summation. Requested input within
  four scale-aware ULPs is snapped to that effective load and yields canonical
  positive-zero loss on every backend; larger deficits are rejected.
- Power budgets expose requested PPU input, effective PPU input, and their
  boundary adjustment. PPU loss and efficiency both use the effective value.
- A represented zero power denominator yields `None` efficiency. The API never
  invents a numeric efficiency for `0/0`.
- Specific impulse is evaluated from charge-state-weighted exhaust velocity,
  not from separately rounded thrust and mass flow. This preserves
  representable tiny results when thrust itself must underflow.
- Exponent-separated products and ratios preserve representable custom
  mass/flow results, including the maximum-finite mass and flow oracle case.
- Manufactured `A_phi=B0*r/2` uses an exponent-scaled half-product, preserving
  representable normal and subnormal outputs across the accepted binary64
  domain, canonicalizing axis zero to `+0.0`, and rejecting only truly
  nonrepresentable results.
- Efficiencies are named `anode_to_beam`,
  `thruster_electrical_to_beam`, and `ppu_input_to_beam`. The implementation
  does not expose an ambiguous “total efficiency.”

The machine-readable source of truth is
`spec/physics/equation-ledger.json`.

## Permitted accuracy claims

Permitted:

- exact agreement with single-charge analytic constructions;
- binary64 CPU/Warp CPU/CUDA parity within the tested numerical tolerance;
- high-precision-oracle agreement for tiny representable speed, current, and
  specific-impulse states;
- closure of particle, mass, charge-current, kinetic/electrical beam-power,
  and reported power-boundary identities;
- exact manufactured-field agreement and regularity on the symmetry axis.

Not permitted:

- measured thruster performance accuracy;
- predictive accuracy for ionization, plume divergence, beam-current fraction,
  cathode behavior, wall losses, erosion, heating, or facility effects;
- a “total efficiency” claim;
- FEM, PIC, or Kornfeld-model fidelity;
- treating the 2020 S1/PIC/PIC-informed values as fitted truth. They are
  external, cross-model regression evidence only.

## Benchmark plan

Performance benchmarking is deferred until GPU load is controlled. A valid
benchmark should:

1. record Warp, driver, device, Python, and operating-system versions;
2. use a dedicated GPU or report utilization and clocks throughout;
3. compile and warm the kernel before measurement;
4. synchronize around timed regions;
5. report kernel-only and end-to-end transfer-inclusive timings separately;
6. use multiple deterministic batch sizes and repetitions with dispersion;
7. retain float64 parity checks in every run;
8. make no speed assertion from the current shared/uncontrolled GPU session.

## Remaining physics gaps

- sourced, sign-verified global plasma/current/energy balances;
- ionization/excitation, wall, thermalization, and cathode closures;
- topology- or measurement-based beam-current and divergence models;
- field material laws, mesh/domain convergence, and FEMM/Maxwell profile data;
- electrostatic separatrix and plume models;
- PIC-MCC collision datasets, boundaries, scaling, and independent code parity;
- experimental thrust, current, Faraday/RPA/LIF, thermal, and uncertainty data;
- explicit treatment of Xe3+ and higher charge states where relevant.

## Integration instructions

This parallel workstream did not modify shared files. After parallel branches
are reconciled:

1. retain the complete new `physics/` source, test, spec, and workstream-doc
   directories;
2. optionally re-export selected symbols from `cft_revival/__init__.py`; direct
   `cft_revival.physics` imports already work without that edit;
3. link this report and the equation ledger from shared architecture/reference
   documentation;
4. add `tests/physics` to any path-specific CI selection (the existing pytest
   `tests` discovery already includes it);
5. retain the existing optional `gpu` dependency declaration for Warp 1.14;
6. merge the entries from `physics-devlog.md` and
   `physics-learning-ledger.md` into shared ledgers only after parallel work
   completes;
7. do not connect these L0 outputs to the optimizer as validated objectives
   until closure provenance and uncertainty gates are defined.
