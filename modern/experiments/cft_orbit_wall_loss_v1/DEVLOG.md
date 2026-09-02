# Devlog

## 2026-09-02 — Preregistration

- Created isolated campaign authority on
  `exp/cft-orbit-wall-loss-v1` from `25dbeaaff4d3a3c276866fdc3f461302a0631120`.
- Frozen the qualified divergent-exit P2 manifest, result, level-1, level-2,
  padding-1.5 checkpoints, sidecars, meshes, runs and Git blobs.
- Added an experiment-local quadratic-P2 to regular-ψ adapter with explicit
  homogeneous-plasma triangle checks, withheld interpolation error and
  certificate-tightness evidence.
- Frozen 512 equal-weight launches: four cells, two interior position repeats,
  two energies, two pitches, both parallel directions and eight gyrophases.
- Declared all primary/refined/enlarged × N/2N/4N campaigns, Wilson estimands,
  numerical gates, partial/final checkpoint chains, exact replay and
  export-only coupling behavior.
- Restricted wall-hit authority to the straight 2 mm cylindrical dielectric;
  radial exit in the divergent section is a domain escape because accepted
  `orbit_mc` does not implement a sloped wall.
- Synthetic preflight passed with zero P2-field/outcome accesses; CPU and CUDA
  Boris parity were exact, helix orders were 1.9984/1.9996, varying-E orders
  were 2.0042/2.0171, mirror error was 1.8828%, and wall error was zero.
- Preregistration validation passed: 4 campaign tests, 94 orbit tests, and
  132 shared-runtime tests (1 Windows privilege skip).
- Single clean detached execution: pending.
