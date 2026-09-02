# PIC-2D Workstream Development Log

## 2026-09-03 — `cft_revival.pic2d` v0.1.0 and the first development snapshot

### Implemented

- `models.py`: SI contracts (`ChannelGeometry`, `Grid2D`, `Species2D`,
  `ParticleArrays`, `PoissonConfig2D`, `StabilityLimits`, `stability_report`,
  `require_stable`, `BoundaryPotentials`), typed fail-closed errors.
- `mesh.py`: plasma-cell mask (exact straight bore, one-cell stair-step cone),
  node classes, cell-wise finite-volume conductances, shape and geometric node
  volumes, `charge_to_source` ratio, wall nodes, column tops.
- `poisson.py`: masked cylindrical FV operator; exact block-Thomas direct
  solver (default) and Jacobi-PCG with true-residual restarts; dense reference;
  nodal `E`; discrete field energy; electrode induced charge.
- `kernels.py`: bilinear deposition (float64 and `2^-40` fixed point), gather,
  relativistic-momentum Boris in the orbit_mc operation order, Cartesian
  advance with frame rotation, boundary classification, renormalised wall
  surface deposit, stable relativistic kinetic energy.
- `p2_field.py`, `fields.py`: hash-bound P2 evaluator (carried from the orbit
  campaign adapter), regular ψ grid over the channel bounding box,
  `PsiBicubicField` node sampling, analytic uniform/linear-ψ/zero maps,
  `MagneticFieldMap` with content hash.
- `mcc.py`: LXCat table loader with payload-hash verification, uniform σ table
  shared by CPU/GPU, null-collision operator (elastic, lumped 8.32 eV
  excitation, 12.13 eV ionisation with Vahedi–Surendra split, Maxwellian
  ions), tallies and inelastic energy ledger.
- `simulation.py`: `PIC2DConfig`, `InjectionConfig`, `SeedPlasmaConfig`,
  `SimulationState`, CPU reference backend, diagnostics accumulator
  (n_e, n_i, φ, T_e from second moments, ionisation map, wall flux per column,
  exit current per radial bin), series records with currents and ledger,
  runtime `ω_pe Δt` gate.
- `warp_backend.py`: Warp CPU/CUDA kernels (fixed-point deposit, moment
  deposit, push+boundary+surface deposit, MCC, spawn via prefix scan,
  injection, compaction, deterministic reductions), device Jacobi-PCG with
  CUDA graphs, host direct-solve path (default), single per-step statistics
  read, deferred ledger energies.
- `artifacts.py`: canonical JSON/npz with sidecars, checkpoints binding
  config/field/cross-section/code identity, runtime identity.
- Specs: `spec/pic2d/pic2d-foundation-v1.json`, `p2-field-authority-v1.json`,
  `xenon-cross-sections-v1.json` (+ source extract, generator, sidecar).
- Experiment `experiments/pic2d_cft_snapshot_v1` (protocol, runner,
  summariser) and dashboard `visualization/generate_pic2d_cft_snapshot.py`.

### Verification record (this session)

- `pytest tests/pic2d`: 50 passed before the snapshot (38 CPU/verification,
  12 Warp parity across `cpu` and `cuda:0`), plus the dashboard tests once
  results exist.
- Poisson manufactured solution orders 1.9988 / 1.9997 / 1.9999 (8→64 radial
  cells); direct vs dense 2e-11 V, PCG vs dense 3e-11 V at 300 V scale;
  Gauss law closure to 1e-9 with volume and surface charge.
- Boris vs `orbit_mc.relativistic_boris_push`: ≤ 8 ulp. Uniform-field orbit vs
  `orbit_mc.integrate_orbit`: < 1e-6 gyroradii over 3000–4000 steps (fixed
  Δt, θ = 1e-3). P2-field orbit error vs orbit_mc shrinks > 2.5× per grid
  refinement (bilinear O(Δx²)). E×B drift to 0.4 %.
- Deposition: charge conserved to 1e-13 (float) / 1e-11 (fixed point); fixed
  point bit-identical across particle permutation and between numpy and Warp.
- MCC rates at 40 eV on 4e5 electrons within 4σ of `n_g σ v Δt` per process;
  chi-square CPU/GPU agreement.
- Single-electron kinetic energy in pure B: drift < 1e-9 over 500 steps.
  Debye sheath slab test: bulk floats +1.9 T_e above grounded electrodes.
- Checkpoint/resume bitwise on CPU and on Warp CUDA (dynamical state) with
  ledger energies to 1e-9.
- P2 field: certified max |B| 0.3057 T (bounding box), 0.226 T in the bore;
  withheld mid-cell error 3.2e-4 relative RMS; primary vs refined ψ grid 0.25 %.
- GPU throughput (RTX 5090, host direct solve): 6–8 ms/step at 31×241 nodes
  with 2e5–1e6 macro-particles.

### Snapshot runs

- Operating point: 300 V / 0 V, n_g 5e20 m⁻³, 0.1 A at 2 eV injected,
  5e16 m⁻³ seed, Δt 2 ps. All cases were stopped fail-closed by the runtime
  `ω_pe Δt` gate at ~55–60 ns as the peak electron density passed ~1.3e18 m⁻³
  (the discharge current was still rising at ~60 mA; the exit ion beam had not
  formed). Numbers per case are in `results/*/summary.json` and the dashboard.

### Deliberate exclusions

- Did not merge into `feat/sota-foundation`; did not preregister anything;
  did not install packages; did not modify `cft_revival.pic` or other
  workstreams' code.
