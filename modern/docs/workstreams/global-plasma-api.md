# Global Plasma Workstream API and Limitations

## CPU reference

```python
from cft_revival.plasma import (
    SolverOptions,
    XenonGlobalInputs,
    solve_global_discharge_multistart_cpu,
)

inputs = XenonGlobalInputs(
    anode_voltage_v=1000.0,
    anode_current_a=1.0,
    cusp_arrival_probabilities=(0.060, 0.119, 0.160, 0.254),
)
result = solve_global_discharge_multistart_cpu(
    inputs,
    start_count=5,
    options=SolverOptions(residual_tolerance=1e-8),
)
```

`result.best.state` and `result.best.evaluation` are `None` unless the residual,
finiteness, box-bound, and nonlinear-inequality gates all pass.
`result.attempts` retains every deterministic start and `result.residual_floor`
reports the best observed normalized infinity norm even when no root closes.
Each attempt's diagnostics records a deterministic reason, iteration and
evaluation counts, normalized residual/gradient norms, damping, active bounds,
feasibility, all 28 normalized rows, numerical Jacobian rank, condition
estimate, and finite-publication status. Solver status alone never publishes a
state.

## Residual and batch boundary

`evaluate_plasma_residual_cpu(state, inputs)` returns:

- 28 dimensional residuals in fixed equation order;
- the same 28 equations normalized by current or power scale;
- beam, ionization, excitation, cusp, anode electron loss, signed anode-ion
  exchange, net anode transfer, input, and closure powers;
- explicit interface-current, cusp-current, local-energy, and global-energy
  closures.

`evaluate_plasma_residual_batch_cpu(states, inputs)` accepts nonempty,
equal-length typed sequences and preserves order. Its fixed 25-input/28-output
layout and absence of Python callback semantics are intended as the later
C++/Warp kernel boundary. This workstream does not claim that a C++ or Warp
implementation already exists.

`analytic_jacobian` gives the 28-by-25 normalized chain-rule Jacobian.
`finite_difference_jacobian` is an independent diagnostic checker.
`default_state_bounds`, `constraint_margins`, and `is_feasible` expose the
actual constraints; inequalities are never encoded as `True`/`False`
residuals.

`XenonGlobalInputs` requires `Ua` and `Ia` independently to be finite positive
normal binary64 values. It also rejects tiny voltage domains whose
`voltage^(3/2)` scale underflows, nonrepresentable bounds, and
subnormal/zero/overflowed `Ua*Ia`. Residuals use exactly `Ia` and `Ua*Ia`;
there are no hidden fallback scales. Consequently, minimum-normal `Ua` is
rejected with a derived cathode-scale error; minimum-normal `Ia` is accepted
only when the selected `Ua` keeps the product and every derived scale/bound
valid.

The exact inherited source label is `MDO (original)`. Corrections and editorial
interpretations are separate fields in the row-complete equation ledger.

## Scope limits

This is a conditional global current/power model at specified `Ua`, `Ia`, cusp
probabilities, and empirical energy fractions. It does not consume xenon mass
flow or neutral density and cannot produce validated utilization, thrust,
specific impulse, plume, lifetime, or facility predictions.

The 2020 S1 entry `YEO2020-S1-MDO-ORIGINAL` in
`spec/plasma/external-comparison-2020.json` uses the source-native model label
`MDO (original)`. Any postprocessing interpretation is separate editorial
metadata. All 2020 values are external comparison evidence only and are
prohibited as solver tolerances, fit targets, or experimental validation.
