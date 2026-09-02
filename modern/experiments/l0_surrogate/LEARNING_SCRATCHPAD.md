# L0 surrogate experiment learning scratchpad

Policy: `COMMITTED` experiment-local record. Accepted shared scratchpads remain
untouched because this concurrent task owns only new experiment paths.

## 2026-09-02 session

- [user] Use accepted public APIs and the complete deterministic 8,192-point
  L0 sweep only as software-emulation truth; never convert this into a physical
  accuracy claim.
- [user] Predeclare gates and a finite exact-GP budget, then preserve an honest
  failure if the budget exhausts.
- [tool] This Windows PowerShell version rejects `&&`; use separate statements.
- [self] Whole 8-bin spatial groups produced 64 interpolation, 62 boundary,
  52 OOD-corner assessment points, and 61 separate calibration points without
  group or index overlap.
- [self] At 96 rows, active ARD Matérn GPs passed RMSE and worst-error gates and
  beat the fixed sequence, but 90% interval coverage was only 0.624/0.775.
  Interior calibration did not transfer to boundary/OOD strata.
- [self] The accepted nearest-distance OOD detector identified only 2/52 points
  in the intentionally withheld corner. Preserve both the design-based stratum
  and detector output; do not relabel after observing the result.
- [self] For L1/L1b, predeclare calibration by geometry/OOD stratum or a
  conservative cross-stratum method before evaluation, and use paired
  discrepancy acquisition on fixed-mesh/POD field targets.
