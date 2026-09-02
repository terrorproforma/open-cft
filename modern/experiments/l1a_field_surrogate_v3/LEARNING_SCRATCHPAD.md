# L1a field-surrogate v3 learning scratchpad

## Before execution

- V2's 1.6 mm coarse smear changed the low-fidelity physical source and left a
  difficult 42% field discrepancy. V3 conserves the original 0.8 mm bands on
  the coarse dual cells.
- Future-role fields must not exist before their freeze. Staging is enforced by
  separate solve calls and counters, not only by index conventions.
- Geometry alignment is input-only: axis landmarks derive from chamber and
  stage geometry, never field values.
- POD rank is data-adaptive only on candidate labels and fails closed if 99.5%
  retained energy requires more than 64 modes.
- The configured learning-scratchpad-loop and devlog-loop skills remain absent;
  their persistent phase records are maintained directly here and in
  `DEVLOG.md`.

## After execution

- Geometry preprocessing succeeded: 506/512 raw rows were valid, 64 required
  constructor-confirmed correction, six were rejected with complete traces,
  and all 240 frozen rows rebuilt hash-identically.
- Staged access worked through method data generation: only candidate and
  method fields existed when model fitting began.
- The run exposed a preregistration implementation defect, not a scientific
  model result: `_build_snapshots` referenced `math.sqrt` without importing
  `math`.
- Because the lock was already claimed, adding the import and resuming would
  violate the protocol. No model/rank/budget was selected and no predictive,
  topology, coverage, or latency gate can be reported.
- The failure bundle preserves 144/144 low and 144/144 fine completions, one
  model-fit access, and zero calibration/assessment accesses.
