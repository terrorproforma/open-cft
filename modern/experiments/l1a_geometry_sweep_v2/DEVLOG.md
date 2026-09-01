# Development log — L1a geometry sweep v2

## Preregistration construction

- Created a new v2-only implementation and tests; v1 remains unchanged.
- Encoded the full fixed protocol in a sealed JSON document with a file
  sidecar, including 96 sample cases, six parity indices, seven terminal gates,
  four objective directions, five representative roles and scale-aware replay
  tolerances.
- Added single-use execution locking, exact preregistration-commit binding,
  strict sealed outputs and sidecars, role/artifact separation, primary-field
  caching and no-padding coalescence.
- Added pre-run tests for sampling, all geometry/manufacturing contracts,
  identity hashes, tolerant Pareto semantics and representative coalescence.
- Added post-run tests that independently recompute identities, nondominance,
  roles and every terminal gate from immutable raw results.

## Execution record

Pending the required preregistration commit and push. Results must be written
here only once and must be committed even if terminal acceptance fails.
