# Topology-General Plasma Network Formulation

## Scope and evidence boundary

This package generalizes the accepted corrected conditional current/power
residual to a geometry-identified ordered chain. It does not add neutral flow,
collision-rate, thrust, plume, lifetime, or validation physics, and it makes no
physical-performance claim. N=4 parity is a software compatibility statement.

The topology adapter supplies ordered finite positions, cells, classified field
nulls, terminals, confidence, full loss covariance, and provenance. Canonical
identity hashes geometry/material/source/artifact/model/code/schema identities
and every nested semantic field. The solver never searches location windows.
Finite-boundary nulls retain IDs, reasons, confidence, position, and provenance
and cannot become interior cusp edges.

## Graph and dimensions

For `N >= 1`, the chain has `N` cells, `N-1` interior cusp edges, and cathode
and anode terminals. The state is ordered as

`x = [phi_N, Te_N, I_N, je_(N+1), ji_(N+1), jic_(N-1)]`,

so `state_size = 6N+1`. The generated equation families contain:

- one cathode-emission row;
- `N-1` interior electron-continuity rows;
- `N` ionization-source rows;
- `N-1` interior ion-continuity rows and one signed anode-ion row;
- `N-1` thermal-transport rows;
- `N+1` interface-current rows;
- `N-1` cusp-current rows;
- `N` local-energy rows and one global-energy row.

Therefore `residual_size = 7N`. `generate_equation_ledger()` creates one
semantic ID and one row ID for every equation. Ampere rows use exactly
`anode_current_a`; watt rows use exactly `anode_voltage_v*anode_current_a`.
Inequalities remain separate margins.

## Orientation and balances

Electron current is positive from cathode to anode. Ion current is positive
from anode to cathode. Interior cusp ion current is positive out of the axial
ion-current control volume:

`ji[k] - ji[k+1] - I[k] + jic[k] = 0`.

The terminal row retains the signed anode current:

`ji[N-1] - I[N-1] - ji[N] = 0`.

Energy gain is

`dE[0] = phi[0] - phi_cathode + Te_cathode`,

`dE[k] = phi[k] - phi[k-1] + Te[k-1]`.

The corrected excitation-loss, transmitted-current thermal row, cusp loss,
beam power, and named anode electron/ion terms are generated for any `N`.
For `N=1`, there is no interior cusp state or cusp-power term; the two
terminals bound one cell directly.

## Rank and publication

The balance Jacobian has generated structural nullity `N-1` and rank `5N+2`
of `6N+1`. Numerical rank uses fixed physical scales (`Ua` for potential and
temperature, `Ia` for all currents), never state bounds. Every solve records
its configurable relative-rank and nullspace tolerances, condition estimate,
and dimensionless orthonormal right-nullspace basis.

The default policy refuses to publish a rank-deficient state even after strict
residual convergence. `REPRESENT_NULLSPACE` requires exactly the structural
nullity, finite dimensions, orthonormal independence, and canonical scaled
`||Jv||` below tolerance. Unexpected extra deficiency is rejected. No policy
turns uncertainty into an extra equality row.

## Numerical interface

`ScaledQrLmBackend` adapts the accepted variable-scaled pivoted-QR
Levenberg-Marquardt implementation through `LeastSquaresBackend`. The backend
is only an untrusted candidate generator: after every return the wrapper
independently checks vector shape/finiteness, bounds, inequalities, canonical
raw and normalized residuals/scales, conservation, Jacobian, rank, and
nullspace. Backend status and reported residual never authorize publication.
`solve_network_multistart()` is deterministic and retains every attempt.
Residual success, finite values, box bounds, inequalities, and the selected
rank policy are independent publication gates.

`evaluate_residual_batch()` and `analytic_jacobian_batch()` accept homogeneous
fixed-layout batches. These are CPU protocol boundaries suitable for a later
GPU implementation; no GPU implementation or performance claim is included.
