# Hybrid Workstream: Remaining Path

The implemented slice is intentionally prescribed-field and small. Advancing
it to a self-consistent L2 model requires separate verification gates rather
than adding empirical terms to the current fixture.

## 1. Self-consistent electrostatics

- Define charge staggering, field staggering, and particle/mesh boundary
  compatibility around the established `x^n, v^(n-1/2), E^n, B^n` contract.
- Add a Poisson operator with explicit gauge and dielectric/interface
  contracts.
- Verify manufactured potential solutions, discrete Gauss law, mesh
  convergence, null spaces, and solver residuals.
- Couple gather and scatter with the same shape functions; verify discrete
  particle-field energy accounting.
- Decide whether field energy uses integer or half levels and add an explicit
  synchronization rule; do not reinterpret stored half-step kinetic energy.
- Keep prescribed-field mode as the regression oracle.

## 2. Electron energy and transport

- Add an electron-energy equation with named flux, Joule heating, collisional
  exchange, ionization, excitation, and wall-loss boundaries.
- Source every rate coefficient and cross-section dataset with interpolation
  and out-of-range policies.
- Verify homogeneous relaxation, conduction manufactured solutions, and total
  ion/electron/field energy exchange.
- Do not add anomalous mobility until a selected model, calibration dataset,
  uncertainty range, and validity domain are documented. `None` remains the
  correct current value.

## 3. Collisions and reactions

- Replace the synthetic constant cross section with versioned, sourced Xe
  elastic and charge-exchange data.
- Add energy-dependent interpolation, threshold handling, null-collision
  majorants, and multi-channel event selection.
- Implement explicit Xe2+ charge-exchange products, particle/species changes,
  and charge/source ledgers before enabling that reaction.
- Represent the partner neutral population dynamically before claiming local
  heavy-species momentum or energy closure.
- Add ionization/recombination only with particle creation/destruction,
  electron-energy, charge, and mass accounting in the same transaction.

## 4. Walls and sheaths

- Define material-tagged absorbing, reflecting, sputtering, and secondary
  emission contracts.
- Couple ion incidence and electron/sheath closures without double-counting
  energy.
- Verify normal incidence, grazing incidence, zero-flux, and closed-box
  conservation before using measured material data.

## 5. Open plume and facility domain

- Add injection/outflow/reservoir boundaries with signed mass, charge,
  momentum, and energy ledgers.
- Establish domain-size, mesh, timestep, particle-count, and boundary
  sensitivity studies.
- Separate free-space plume claims from facility-background predictions.

## 6. Validation and predictive claims

- Freeze provenance-rich input decks and checkpoint compatibility policies.
- Compare against analytic and manufactured cases before cross-code cases.
- Select independent experimental observables and uncertainty budgets before
  calibration.
- Report numerical verification, cross-model comparison, calibration, and
  experimental validation as distinct evidence levels.

No item above is implied by the tiny manufactured run.
