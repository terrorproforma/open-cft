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

- Pending exactly-once evidence.
