# L1a field-surrogate v2 devlog

## Preregistration

- Preserved v1 and all accepted/shared/FYP paths unchanged.
- Moved geometry validation ahead of role partitioning and freezing.
- Uses the production geometry-v1.1 constructor for every endpoint attempt;
  no copied slope predicate determines acceptance.
- Rebuilds all 112 frozen geometries and previews before field access.
- Writes complete sidecar-protected provenance and Git-blob closure before the
  first field solve.

## Execution

- Pushed preregistration `373ffae8d8d08affc9d33401378a81c35dbdf964`.
- A Windows CRLF checkout was rejected during binding before lock creation and
  with zero field/QoI access. Canonical committed LF bytes were restored
  without changing code or history.
- The single lock-claimed RTX 5090 execution completed all 112 coarse and 80
  fine solves. All residual, boundary, source, flux, topology-confidence and
  zero-frozen-failure numerical gates passed.
- All 12 budget/family combinations failed development predictive gates.
  Protocol execution stopped before calibration and assessment; access
  counters remained zero for both.
- No threshold, model, code, protocol, or result was patched and no
  lock-claimed execution was repeated.
