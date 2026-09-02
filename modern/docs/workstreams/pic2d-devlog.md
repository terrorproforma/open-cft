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

### Snapshot runs (commit `d58fdca1`)

- Operating point: 300 V / 0 V, n_g 5e20 m⁻³, 0.1 A at 2 eV injected,
  5e16 m⁻³ seed, Δt 2 ps, RTX 5090 (`warp-cuda:0`, host direct field solve),
  two or three cases sharing the GPU at a time (nvidia-smi 99 % utilisation
  in every sample). All four cases were stopped fail-closed by the runtime
  `ω_pe Δt ≤ 0.2` gate at 49–60 ns as the peak node electron density passed
  ~1.4–1.6e18 m⁻³. No plateau was reached: the discharge current was still
  rising and the exit ion beam had not formed. Window = last complete 10 %
  segment (15 000 steps) or the half-full trailing partial.

| case | grid | W | steps | t (ns) | wall (s) | steps/s | final e⁻/Xe⁺ macro | I_d (mA) | I_beam,i (mA) | peak n_e (m⁻³) | ⟨n_e⟩ | φ range (V) | ⟨T_e⟩_n (eV) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coarse-w1e5 | 30×240 | 1e5 | 27 800 | 55.6 | 283 | 98 | 0.70 M / 0.73 M | 67.2 | 0.20 | 1.55e18 | 2.52e17 | −29.5 … 314 | 17.9 |
| coarse-w5e4 | 30×240 | 5e4 | 30 100 | 60.2 | 351 | 86 | 1.66 M / 1.72 M | 73.7 | 0.22 | 1.63e18 | 2.79e17 | −29.6 … 312 | 17.5 |
| fine-w5e4 | 60×480 | 5e4 | 24 500 | 49.0 | 380 | 64 | 1.10 M / 1.15 M | 57.7 | 0.14 | 1.41e18 | 2.08e17 | −23.4 … 314 | 18.4 |
| fine-w2.5e4 | 60×480 | 2.5e4 | 26 800 | 53.6 | 575 | 47 | 2.59 M / 2.70 M | 64.0 | 0.23 | 1.52e18 | 2.36e17 | −23.6 … 313 | 18.0 |

- Between-case relative spread of window averages (manifest): φ_max 0.5 %,
  ⟨T_e⟩ 5 %, peak n_e 8 %, ⟨n_e⟩ 29 %, exit ion current 36 %, discharge
  current 58–74 mA. Wall fluxes: electrons 2.2–2.9e21 m⁻² s⁻¹ peak, ions
  5.2–6.4e20 m⁻² s⁻¹ peak, concentrated near the cusp lines; ionisation rate
  peaks ~1e26 m⁻³ s⁻¹.
- Energy ledger: per 200-step interval the residual is ~5e-9 J against a
  total (K+U) of 2.6–3.8e-7 J (≈ 1.5–2 %) and ~70 % of the interval field work;
  it is dominated by the untracked electrode/injection electrostatic work.
- Debye resolution at the observed peak density (T_e ≈ 18 eV → λ_D ≈ 26 µm)
  is violated on both grids (100 µm, 50 µm cells); the gate that stopped the
  runs is the ω_pe Δt gate, and the Debye ratio is reported, not enforced at
  runtime.
- Dashboard: `modern/visualization/pic2d-cft-snapshot.html` (2.0 MB, 8 tests).
- P2-field orbit vs orbit_mc at Δx = 250/125/62 µm: 8.0e-3, 2.0e-3, 2.8e-4
  gyroradii (600 steps, θ = 0.02).

### Deliberate exclusions

- Did not merge into `feat/sota-foundation`; did not preregister anything;
  did not install packages; did not modify `cft_revival.pic` or other
  workstreams' code.
