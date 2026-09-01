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

- Pending pushed preregistration and the one authorized RTX 5090 run.
