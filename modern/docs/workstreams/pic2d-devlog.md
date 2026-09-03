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

## 2026-09-03 — phase 2: review-gate merge, v1 diagnosis, model v1.1, snapshot v2

### Step 1 — merge of phase 1

- `feat/pic-2d-axisymmetric` rebased onto `origin/feat/sota-foundation`
  (6f3e6dd5); `.gitignore` conflict resolved keeping the LF-era rules and the
  pic2d negations; `git ls-files --eol` shows no `i/lf w/crlf` entries.
  `tests/pic2d` (58), `tests/pic`, `tests/orbit_mc`, `tests/visualization` and
  the dashboard tests green; fast-forward push into `feat/sota-foundation` at
  `62de2ca3`.

### Step 2 — why v1 over-densified (`pic2d_cft_snapshot_v1/results/diagnosis.json`)

- Trip cell: every case tripped on an **axis node** in the straight bore
  (coarse-w1e5: node (i=0, j=106), r = 0, z = 10.6 mm, n_e = 3.35e18 m⁻³,
  `ω_pe Δt` = 0.207 at Δt = 2 ps; gate density 3.14e18). The top-5 nodes are all
  on the axis in every case; 21–29 % of the "hot" nodes are axis nodes. The
  window-averaged map peak was 1.5–1.6e18 and the last series sample implied
  2.7–3.0e18: the gate sees ~2.2× the window peak (shot noise on the
  smallest-volume nodes).
- Source vs loss over the run (coarse-w1e5, 55.6 ns): 5.77e10 ions created,
  1.50e9 lost (1.44e9 wall, 4.0e7 anode, 1.8e7 exit) → loss/source = 2.6 %;
  final rates 2.5e18 s⁻¹ created vs 4.2e16 s⁻¹ lost (1.7 %). Electron count
  e-folding time 26.7–27.3 ns in all four cases. Ion transit: 1.14 µs axially at
  300 V, 0.55 µs radially at the Bohm speed for 18 eV — the run ended at 5 % of
  a transit time, so the avalanche had no loss channel yet.
- 0-D equilibrium at the observed T_e = 18 eV (unmagnetised Bohm loss to all
  surfaces, A_loss = 3.6e-4 m², V = 3.5e-7 m³; upper bound on loss, so n_eq is a
  lower bound):

| n_g (m⁻³) | source/loss at 18 eV | T_e for balance (eV) | n_eq (power balance at I_d = 64 mA) | λ_D (µm) | ω_pe (s⁻¹) |
|---|---|---|---|---|---|
| 5e20 | 12.1 | 3.9 | 6.3e17 | 39.7 | 4.5e10 |
| 1e20 | 2.4 | 7.9 | 6.3e17 | 39.7 | 4.5e10 |
| 5e19 | 1.2 | 14.0 | 6.3e17 | 39.7 | 4.5e10 |

  (n_eq from the power balance scales with I_d, not n_g; at 3 mA it is ~1–2e17.)
- Per-step cost, fine-w2.5e4 at 5.4 M macro-particles, before: 40.7 ms/step of
  which host block-Thomas 18 ms, host↔device source/φ copies and per-step
  count/statistics reads the rest of the non-kernel time; push+MCC dominated
  the kernel time through same-address float64 atomics.

### Step 3 — model v1.1 (`modern/spec/pic2d/pic2d-model-v1.1.json`)

- Device block-Thomas solve (`method="device-direct"`) with the true-residual
  contract enforced at each host sync; host reads only every
  `device_sync_steps` (200) steps; per-block tile reductions replace the
  per-particle atomics in push and MCC; ion subcycling `k = 8`; electrode-work
  term `Σ_k V_k (ΔQ_induced,k − q_absorbed,k)` in the ledger.
- After: the same fine case runs at 5.46 ms/step (Poisson 2.3 ms), 1.0 ns per
  particle-step; the v2 coarse cases (0.13–0.27 M particles) at 1.2–2.0 ms/step
  while four cases share the GPU. Single-process numbers per case are in the v2
  manifest (`ms_per_step`).
- Tests (`tests/pic2d/test_pic2d_v11_step.py`): device-direct parity (φ to
  1e-10, positions to 1e-15 m), residual contract, exact integer tallies from
  the tiled kernels, k vs 2k insensitivity, ledger closure (< 15 % of the
  electrode work), provenance of `ion_subcycle`.
- Operating point 300 V, n_g = 1e20 m⁻³, 3 mA at 2 eV; budget n_max = 4e17,
  T_e = 8 eV → λ_D,min 33.2 µm, Δt = 1.5 ps (`ω_pe Δt` 0.054, `Ω_ce Δt` 0.077),
  fine 60×480 cell/λ_D = 1.50, coarse 30×240 3.01 (gate 3.1 by design).
- Not done, stated: neutral depletion (≈1 % over 1.6 µs), ion–neutral
  elastic/CEX (no hash-bound Xe⁺–Xe source in the extract).

### Step 4 — snapshot v2 (`modern/experiments/pic2d_cft_snapshot_v2`)

- Attempt 1 at Δt = 3 ps: both coarse cases stopped fail-closed by the runtime
  gate after 90 200 / 146 800 steps (0.27 / 0.44 µs): instantaneous node peak
  1.4e18 vs window peak 4.7–5.4e17 and mean 1.1e17; I_d 3.5 mA still rising
  (electron count +0.2 % per 200 steps). Protocol changed to Δt = 1.5 ps
  (trip density 5.6e18), min 666 667 steps (1 µs), target 1 066 667 (1.6 µs),
  180 ns averaging windows; the reasoning is recorded in `protocol.json`.
- Attempt 2 (Δt = 1.5 ps, four cases sharing the RTX 5090, wall budget 7200 s
  each; window = last complete or ≥ half-full 180 ns segment):

| case | grid | W | steps | t (µs) / τ_i | wall (s) | ms/step (shared) | stop | I_d (mA) | I_beam,i (mA) | I_wall,i (mA) | peak / mean n_e (m⁻³) | ⟨T_e⟩_n (eV) | φ range (V) | ledger residual / electrode work | plateau (drift I_d, N_e) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coarse-w2.4e5 | 30×240 | 2.4e5 | 621 200 | 0.93 | 2 579 | 4.15 | ω_pe Δt gate | 4.36 | 1.35 | 6.09 | 2.36e18 / 5.71e17 | 11.1 | 0 … 353 | +41 % | no (14 %, 69 %) |
| coarse-w1.2e5 | 30×240 | 1.2e5 | 757 000 | 1.14 | 3 896 | 5.15 | ω_pe Δt gate | 4.70 | 1.68 | 5.79 | 2.32e18 / 5.02e17 | 9.5 | 0 … 332 | +42 % | no (22 %, 64 %) |
| fine-w6e4 | 60×480 | 6e4 | 1 022 000 | 1.53 | 7 212 | 7.06 | wall budget | 5.19 | 2.20 | 5.14 | 2.18e18 / 4.62e17 | 6.4 | −1 … 322 | −13 % | no (12 %, 24 %) |
| fine-w3e4 | 60×480 | 3e4 | 802 000 | 1.20 | 7 219 | 9.00 | wall budget | 4.43 | 1.57 | 3.53 | 1.46e18 / 3.21e17 | 6.4 | −10 … 317 | −18 % | no (9 %, 19 %) |

- Peak macro-particle counts 0.76 / 1.8 / 1.9 / 2.6 M (e⁻); the ms/step above
  are with four processes on one GPU. Single-process from the final
  checkpoints, GPU otherwise idle: 2.04 ms/step at 1.53 M particles (coarse
  grid), 4.34 ms/step at 3.8 M and 5.45 ms/step at 5.3 M (fine grid), i.e.
  1.0–1.3 ns per particle-step over a ~1.2 ms launch-bound floor (~40 launches
  per step on WDDM). The ≤ 1.5 ms/step target at 1–2 M was not met (2.0 ms).
- No plateau: after one ion transit time the ion loss (walls + exit + anode)
  is 10 % (coarse) to 35 % (fine) of the ionisation rate; cumulative ions lost /
  created 0.16 (coarse) and 0.33–0.35 (fine); the electron count still grows
  19–69 % over the trailing 20 % of each run. The window peak density is
  3.7–5.9× the a-priori ceiling n_max = 4e17 and the cells are 3–6 λ_D at the
  end, so the v1.1 resolvability budget is exceeded by the kinetic discharge:
  the 0-D estimate used an unmagnetised Bohm loss to all surfaces (an upper
  bound on the loss), and the magnetised channel loses ions much more slowly.
- Grid heating is visible: the coarse grid runs 1.5–1.7× hotter (9.5–11 vs
  6.4 eV), ionises 3.5× faster at similar density, and its ledger residual is
  +41 % of the electrode work (energy appearing), while the fine grid's is
  −13 to −18 %. The electrode-work term is therefore necessary but not
  sufficient for closure at these resolutions; the residual is reported per
  case.
- Convergence statement: not converged. Relative spread across the four cases
  (window averages): φ_max 11 %, I_d 18 %, peak n_e 43 %, mean n_e 54 %,
  ⟨T_e⟩ 57 %, exit and wall ion currents 50 %. The two fine cases agree better
  with each other (I_d 4.4 vs 5.2 mA, ⟨T_e⟩ 6.4 vs 6.4 eV) than with the coarse
  pair.
- Dashboard `modern/visualization/pic2d-cft-snapshot.html` regenerated (v2
  cases; v1 kept in a "fail-closed development history" panel with the 0-D
  table; a-priori budget panel); 9 dashboard tests. Headless Chrome
  screenshots: `%TEMP%\pic2d-cft-snapshot-v2-desktop.png` (1440×4800) and
  `%TEMP%\pic2d-cft-snapshot-v2-narrow.png` (430×10400).

### Step 5 — commits

- `feat(pic2d): all-GPU step with warm-start PCG and ion subcycling`,
  `feat(pic2d): v1.1 operating-point budget and optional neutral depletion`
  (budget + spec + docs; depletion deferred and stated),
  `chore(pic2d): snapshot v2 results and dashboard`; fast-forward into
  `feat/sota-foundation` after the suites pass (SHAs in the session report).

### Deliberate exclusions (phase 2)

- No neutral depletion, no ion–neutral collisions, no change to the 300 V
  anode; protocol v2 not edited after the runs (its hash is bound into every
  summary); no preregistration; nothing outside `pic2d`, its experiments,
  tests, spec and dashboard touched.

## 2026-09-03 — phase 3: operating point v1.2 and the steady-state runner

### Step 1 — sizing from the measured kinetics (`fine-w6e4`, cross-check `fine-w3e4`)

- Time-resolved (30-sample = 9 ns running means of the 200-step series):

| t (µs) | S (s⁻¹) | L = walls+anode+exit (s⁻¹) | f = L/S | N_i | τ_i,eff = N_i/L (µs) | mean n (m⁻³) |
|---|---|---|---|---|---|---|
| 0.25 | 9.0e16 | 2.7e16 | 0.30 | 2.8e10 | 1.06 | 8.2e16 |
| 0.50 | 7.9e16 | 2.7e16 | 0.35 | 4.2e10 | 1.53 | 1.2e17 |
| 0.75 | 9.1e16 | 2.8e16 | 0.31 | 5.6e10 | 1.99 | 1.6e17 |
| 1.00 | 1.01e17 | 3.0e16 | 0.30 | 7.3e10 | 2.40 | 2.1e17 |
| 1.25 | 1.16e17 | 3.8e16 | 0.33 | 9.2e10 | 2.41 | 2.7e17 |
| 1.50 | 1.35e17 | 4.7e16 | 0.35 | 1.12e11 | 2.40 | 3.2e17 |

- f does not approach 1: trailing-half slope 0.03 (w3e4) – 0.07 (w6e4) per µs.
  τ_i,eff rises while the first ions are in flight and saturates at 2.4 µs
  after 1 µs — the kinetic ion residence time (v2 assumed 1.0 µs from a
  Bohm/free-fall argument). Per-electron ionisation frequency ν_iz = S/N_e =
  1.2e6 s⁻¹ (⟨σv⟩_eff = 1.2e-14 m³/s at ⟨T_e⟩ = 6.4 eV), so ν_iz τ = 2.9 > 1:
  dN_i/dt = (ν_iz − 1/τ) N_i predicts a growth rate of 8e5 s⁻¹; observed 9.0e5
  s⁻¹ (e-folding 1.1 µs) in both fine cases. The v2 discharge was
  super-critical — no static-neutral equilibrium at any density, which is
  exactly why it grew past the budget.
- Equilibrium model: the plasma-electron term cannot balance (it scales with
  N); the beam-driven ionisation a ≈ 5e16 s⁻¹ (S − ν_iz N_e at 0.25 µs, ≈ 2.7
  ionisations per injected 300 V electron, ∝ n_g I_inj) sustains a
  sub-critical discharge with N_eq = a τ / (1 − ν_iz τ). ν_iz τ = 1 at n_g =
  3.4e19 m⁻³; at the 5e19 mark it is still 1.45 (runaway unless T_e drops ~30 %)
  — so yes, a plateau with static neutrals needs n_g < 5e19. Projection
  (τ = 2.4 µs held fixed): n_g = 2e19 → 1.65e17 (3 mA) / 1.1e17 (2 mA); n_g =
  1.5e19 → 9.3e16 (3 mA) / 6.2e16 (2 mA).
- Chosen v1.2: **n_g = 1.5e19 m⁻³, I_inj = 3 mA at 2 eV, 300 V, Δt = 1.5 ps,
  k = 8, fine grid 60×480 only, W = 3e4** → ν_iz τ = 0.44 (2.3× margin on the
  rate coefficient), n_eq ≈ 9.3e16 = 0.23 n_max (1.5e17 / 4.2e17 if the rate
  coefficient is 1.5× / 2× higher), 2.1 M macro-particles at n_eq (33 ppc at
  r = 1 mm), approach time constant τ/(1 − ν_iz τ) = 4.3 µs → the 5 %/20 %
  plateau rule fires at ≈ 2.8 time constants = 12 µs = 8.0 M steps, after the
  required 3 × 2.4 µs. Budget table: `modern/spec/pic2d/pic2d-model-v1.2.json`
  and `experiments/pic2d_cft_steady_state_v1/protocol.json`.

### Step 2 — detached resumable runner (`experiments/pic2d_cft_steady_state_v1/run.py`)

- Chunks of ≤ 40 000 steps; after each chunk the checkpoint directory is
  swapped atomically (`checkpoint-tmp` → `checkpoint`, old copy kept until
  the swap is done), `run_state.json` records the cumulative wall time and
  sessions, and the plateau rule is evaluated over all records (`series.jsonl`,
  reloaded and truncated to the checkpoint step on resume). `status.jsonl` gets
  one line per 200-step sync (t, steps, N_e, N_i, I_d, I_beam,i, peak-node /
  mean n_e, ⟨T_e⟩, max ω_pe Δt, wall, ms/step, plateau drifts). Stops: plateau
  (drift of I_d and N_e < 5 % over the trailing 20 % of elapsed time, ≥ 3
  transit times), 12 h cumulative wall budget, stability gate, `--max-steps`;
  all exit 0 after writing `summary.json`, `series.npz`, `maps.npz`,
  `checkpoint-final.*`.
- `Simulation.load_state` now re-bases the interval bookkeeping (cumulative
  tallies, interval step base, energy/electrode samples): before, the first
  record after a resume divided the interval tallies by the absolute step
  count (5× too small a current in the test) and differenced the cumulative
  ledger against zero.
- Tests `tests/pic2d/test_pic2d_steady_state_runner.py` (8): plateau needs
  both drifts and 3 transits; an exponential approach is declared at 3.0–4.6 τ;
  a ramp never, a 15 %-noisy constant yes; one status line and series record
  per sync, one checkpoint per chunk, summary on stop; interrupted + resumed
  run equals the uninterrupted run bitwise (final arrays, counts, interval
  currents) with a stray post-checkpoint record dropped; `load_state`
  re-basing.
- Launched detached (`Start-Process python -u -m
  experiments.pic2d_cft_steady_state_v1.run run -WindowStyle Hidden`, logs
  `results/run.log|run.err`, PID in `results/run.pid`). After 75 s: 39 200
  steps, 1.92 ms/step at 0.19 M particles (launch-bound floor), ω_pe Δt max
  0.044, ⟨T_e⟩ 7.4 eV, I_d 0.06–0.11 mA, N_i > N_e (seed electrons leave
  first; φ_max 351 V). Results are not committed in this phase.
- GPU note: `nvidia-smi` reports 100 % utilisation with no compute process
  (display clients only, ~107 W) before the launch; the steady-state run is the
  only CUDA compute process.

### Deliberate exclusions (phase 3)

- No coarse grid, no second weight (single case by design); no neutral
  depletion or ion–neutral collisions (depletion ≈ 0.25 %/µs at 1.5e19);
  v2 protocol still references a non-existent
  `operating-point-v1.1.json` (its actual spec is `pic2d-model-v1.1.json`) and
  is left untouched because its hash is bound into the v2 summaries.

## 2026-09-03 — phase 3b: model v1.3 (quasi-steady neutral inventory), finalize command

### Why

The v1.2 reference run (n_g = 1.5e19 static) did not ignite: 2.9 of 3 mA of injected
electrons mirror back to the exit plane before colliding, the seed decays and I_d
settles at a beam-driven floor (~0.02–0.08 mA). The v2 cases at 1e20 avalanched
(ν_iz τ = 2.9). A static neutral background therefore either runs away or never
lights; the physical saturation channel — neutral depletion — has to be in the model
before a plateau is meaningful.

### Model v1.3 — `cft_revival.pic2d.neutrals`

- State: one scalar n_g(t), uniform in space, plus four cumulative atom ledgers
  (fed, ionized, effused, artificial).
- Balance: V dn_g/dt = Q_in − S(t) − c n_g − (V/τ_g)(n_g − n_g*), n_g* = (Q_in − S)/c.
  S is the MCC ionisation tally over the series interval × W / interval (measured,
  not modelled). c = v̄ A_exit/4 with v̄ = √(8kT/πm) = 220.0 m/s at the documented
  300 K (the same temperature that sets the born-ion Maxwellian) and
  A_exit = π(3 mm)² → c = 1.5547e-3 m³/s. Physical time constant V/c = 221 µs.
- The relaxation toward n_g* with τ_g = 30 ns is **artificial** (≪ the 2.4 µs ion
  transit so n_g is quasi-steady on plasma time scales; ≪ 221 µs; 100 series
  intervals so the shot noise of S is averaged). Only the fixed point is physical.
  The artificial term is integrated into its own ledger, so
  fed − ionized − effused − artificial = V Δn_g holds to round-off (test: 1e-9 of
  the inventory per update; the GPU smoke run closed to 7e-17 relative).
- Integration: the linear ODE is solved exactly per interval (S held at its interval
  mean): n₁ = n_∞ + (n₀ − n_∞) e^{−r Δt}, r = c/V + 1/τ_g; the ledgers are the exact
  interval integrals.
- MCC coupling: real collision frequencies use n_g, the null ceiling stays at
  n_g0 = the configured density (the candidate fraction is unchanged; the
  per-candidate real probability scales with n_g/n_g0). `set_neutral_scale` on both
  backends; the Warp kernel already took the density as a per-launch scalar, so the
  GPU path is one multiplication. Scale > 1, feed above the ceiling (Q_in/c > n_g0)
  and exhaustion (n_g < 0) fail closed.
- State plumbing: `SimulationState.neutral`, checkpoint array `neutral`
  (hash-bound; `neutral_keys` in the metadata), `SeriesRecord.neutral`,
  `Simulation.load_state` restores n_g and re-applies the scale; a static checkpoint
  cannot load into an inventory config and vice versa. The config identity carries
  the inventory block only when it is on, so v1.0–v1.2 identities (and the running
  reference run's checkpoint) are unchanged.
- `instantaneous_maps(config, masks, state)`: single-sample n_e/n_i/φ/T_e maps
  deposited from a state (used by `finalize`).

### Runner changes (shared `pic2d_cft_steady_state_v1/run.py`)

- Generic protocol handling (`protocol_budget`, optional
  `operating_point.neutral_inventory`, `protocol_path`), so
  `pic2d_cft_steady_state_v2/run.py` is a thin wrapper.
- Status lines gain n_g, its fixed point, S and the effusion rate; `series.npz`
  gains `neutral_*` arrays; the summary gains a `neutral_inventory` block (final
  density, fixed point, trailing means, ledgers, closure, utilisation).
- Plateau: with the inventory on, n_g drift < 5 % over the trailing 20 % is a third
  tracked quantity (`tracked` lists what was checked).
- `finalize`: summary/maps/series from `checkpoint-latest` and the series history
  without stepping (code-identity check relaxed — nothing is computed;
  `maps_kind = instantaneous_checkpoint`, flux maps zero, stray records dropped,
  `run_state.finalized_from_step`, stop event appended).

### Sizing v1.3 (`pic2d-model-v1.3.json`, v2 protocol)

- n_g0 = 5e19 (top of the 4.5–5e19 range: more margin for ignition) →
  Q_in = 7.77e16 atoms/s = **0.0170 mg/s** (tiny because the exit is a 3 mm hole at
  300 K with no plume region).
- Expected fixed point: the user's ν_iz τ = 1 point n_g ≈ 3.4e19 gives
  S = 2.5e16/s (32 % utilisation) and n_eq ≈ 1.1–1.7e17; the linear v2 coefficients
  (a = 6.03e16·x, ν_iz = 6.12e5·x, τ = 2.4 µs, x = n_g/1e20) give the other end,
  n_g* = 2.3e19, n_e = 2.9e17 (0.73 n_max). Since the reference run showed the
  beam-ionisation coefficient collapsing at low n_g, the linear projection is an
  upper bound on n_e: bracket n_g* ∈ [2.3, 3.4]e19, n_e ∈ [1.1, 2.9]e17.
- W = 6e4 → 1.26 M / 1.72 M / 3.2 M macro-particles at 1.1 / 1.5 / 2.9e17.
- 300 V, 3 mA @ 2 eV kept: the run starts at 5e19 where ν_iz τ = 1.45 > 1, so
  ignition is expected without a hotter seed.
- GPU smoke (2000 steps, sharing the card with the reference run): 1.6 ms/step at
  0.1 M particles, ω_pe Δt 0.065, n_g 5.00e19 → 4.98e19 with S = 4e15/s, ledger
  closure 7e-17.

### Tests

- `test_pic2d_neutral_inventory.py` (9): numbers (v̄, c, feed, mg/s); ledger closure
  per update and cumulative with a varying prescribed S; analytic fixed point,
  approach rate 1/τ_g + c/V, artificial term → 0 at the fixed point; fail-closed
  paths; MCC candidate count unchanged and all three real channels halve at scale
  0.5 within 5σ; Simulation records the inventory and its ledger equals the tally × W;
  checkpoint carries n_g, resume bitwise, tamper breaks the hash; static ↔ inventory
  mismatch rejected; GPU backend receives the scale.
- `test_pic2d_steady_state_runner.py` (+3 → 11): plateau tracks n_g; finalize from a
  "killed" run (checkpoint bytes unchanged, stray record dropped, final == latest);
  v1.3 protocol run (status/series/summary neutral fields, closure < 1e-12, n_g in
  the checkpoint, series ≠ sync interval rejected).
- Full run: `tests/pic2d tests/pic tests/orbit_mc tests/visualization` → 388 passed.

### Reference run closed, v1.3 launched (two attempts)

- Reference run `pic2d_cft_steady_state_v1` (v1.2, static n_g = 1.5e19): terminated
  (PID 49664) right after the step-5 440 000 checkpoint landed (3.4 ion transit times,
  8.16 µs, 4850 s wall), closed with `finalize` (instantaneous checkpoint maps).
  No ignition: 99.8 % of the injected electrons returned to the exit plane, 0.66 %
  reached the anode; N_e fell from the seed to 7.8 k macro-particles
  (n_e mean 6.5e14, peak node 1.1e17 at the injector), I_d floor 0–0.08 mA, plateau
  not declared (N_e drift −16 %, I_d drift −35 % on a ~0 mean). Committed as
  `chore(pic2d): steady-state v1 no-ignition reference results` (summary, run_state,
  status.jsonl, maps/series npz, checkpoint-final metadata; 42 MB series.jsonl,
  particle arrays and logs not tracked).
- `finalize` bug found on first use: it built the config with the CPU Poisson method,
  which is part of the config identity → checkpoint rejected. Fixed: `finalize`
  defaults to the run backend (`--backend`), nothing is stepped.
- v1.3 attempt 1 (`results-attempt1-ng0-5e19-seed1e16/`, PID 49736): n_g0 = 5e19,
  Q_in = 7.77e16 s⁻¹, seed 1e16 @ 5 eV. **No ignition** within 1 µs: S fell
  monotonically 2.9e15 → 1.5e15 s⁻¹ (ν_iz = S/N_e 1.16e6 → 5.6e5 s⁻¹ as the seed
  cooled 7.9 → 5.0 eV), N_e flat at 42–44 k (beam transit population), N_i decaying,
  91 % of the beam returned (96 % late), I_d 0.2 mA, 0.116 ionisations per injected
  electron (v2 at 1e20: 2.7). n_g 5.0 → 4.83 → 4.90e19 (fixed point of the tiny S;
  Q_in was not limiting). Stopped after the 760 000-step checkpoint (1.14 µs),
  finalized, archived.
- Diagnosis: the seed does not sustain itself at 5e19 (no heating; inelastic
  cooling and loss of the hot tail), and without a plasma potential structure the
  300 V beam mirrors back in the cusps before colliding. Comparison with v2
  fine-w6e4 (1e20, seed 5e16): the beam was absorbed from the first 100 ns
  (return 0.9 → 0.26 mA, anode 3.2–3.6 mA, ν_iz 2.8–4.2e6 s⁻¹, T_e rising 8 → 10 eV).
  A 23× jump in ionisations per injected electron for a 2× change in n_g points at
  the seed-built potential structure, not the collision rate.
- v1.3 attempt 2 (`results/`, PID 40636, launched 2026-09-02T23:02Z): seed 5e16 @ 5 eV
  (the v1/v2 snapshot seed) and n_g0 = 5.5e19 (Q_in = 8.55e16 s⁻¹ = 0.0186 mg/s) —
  both documented adjustments; protocol `attempts` block records both. After 3 min
  (0.13 µs): S 1.6e16 s⁻¹ (10× attempt 1), 0.86 ionisations per injected electron,
  beam return 60 %, anode 1.5–2.2 mA, I_d 1.2–1.6 mA, T_e rising 7.6 → 8.1 eV,
  n_g 5.5 → 4.46e19 tracking its fixed point, ω_pe Δt 0.07–0.09, 2.2–2.3 ms/step at
  0.51 M particles. At 0.48 µs: **igniting** — N_e 241 k → 289 k (+20 %, growth
  ~4.5e5 s⁻¹), N_i rising again after the seed-ion dip, S 1.9e16 s⁻¹ (22 % of the
  feed), T_e 8.0–8.6 eV, I_d 1.2 mA, n_g 4.27e19 (fixed point 4.34e19), 2.4 ms/step
  at 0.59 M. Projection: 3 transits (4.8 M steps) ≈ 3 h from launch, 10 transits
  (16 M) ≈ 11–12 h — the 12 h wall budget will likely stop it near 9–10 transits as
  the particle count grows toward ~2.6 M.
- Timing note: 2.1–2.5 ms/step at 0.1–0.6 M particles is the launch-bound floor of
  the Windows WDDM path (the reference run at 8–60 k particles ran at 0.8–0.95).

### Commits (phase 3b)

- `520e6b41` feat(pic2d): v1.3 quasi-steady neutral inventory with conservation tests
- `67b04f87` feat(pic2d): finalize command and steady-state v2 experiment
- `3c9e606c` fix(pic2d): finalize uses the run backend for the config identity
- `cb40f06d` chore(pic2d): steady-state v1 no-ignition reference results
  (rebased over `cc7706b2` docs: final roadmap audit; fast-forwarded into
  feat/sota-foundation)
- attempt-2 protocol/README + this entry: see the commit that carries them.

## 2026-09-03 — Phase 4: v1.3 plateau finalized, convergence pair, steady-state dashboard

### Plateau verification (steady-state v2, attempt 2)

- `run_state.json`: `finished`, `plateau_reached_after_min_transit_times` at step
  5 120 000 (7.68 µs = 3.2 ion transit times), 10 141 s wall in one session (PID 40636),
  1.98 ms/step at the end with 999 k electrons + 1.008 M ions (peak 999 k / 1.008 M).
- Criterion recomputed from `series.npz` with the runner's `evaluate_plateau`
  (identical to the summary): trailing-20 % drifts I_d +0.084 %, N_e **+4.98 %**,
  n_g −0.18 %; threshold 5 %; ≥ 3 transits. The electron-count drift passed by
  0.02 % — the plasma was still slowly densifying (ω_pe Δt rose monotonically
  0.06 → 0.115 over the run). Other drifts over the same window: I_beam −0.9 %,
  N_i +4.9 %, S +0.2 %, φ_max −0.02 %. The criterion held as declared; it is a
  marginal pass and is labelled so everywhere.
- Window-averaged final state (steps 4 800 000–5 120 000, 1600 records):
  I_d 3.444 ± 0.325 mA, I_beam,i 2.291 ± 0.272 mA (0.665 I_d), anode e⁻ 3.494 mA,
  anode Xe⁺ 0.049 mA, wall Xe⁺ 3.724 mA, wall e⁻ 3.727 mA, returned e⁻ 1.843 mA of
  3.000 mA injected. S = 3.934e16 s⁻¹ = 6.30 mA equivalent = 46.0 % of Q_in
  (8.551e16 s⁻¹); 2.10 ionisations per injected electron (v2 at 1e20 static: 2.7;
  attempt 1: 0.12). Ion balance closes: S·e = I_beam + I_wall,i + I_anode,i
  (6.30 ≈ 2.29 + 3.72 + 0.05 = 6.06 mA; the 4 % gap is the still-growing inventory).
- Neutrals: n_g window mean 2.969e19 vs analytic fixed point (Q_in − S)/c =
  2.970e19 (−0.03 %), n_g/n_g0 = 0.540; check Q_in = S + c n_g:
  3.934e16 + 1.5547e-3 × 2.969e19 = 8.55e16 ✓. Cumulative atom ledger: fed
  6.567e11, ionised 2.552e11, effused 4.027e11, artificial 8.698e12 (= the
  inventory drop (5.5 − 2.97)e19 × V, i.e. the artificial term did the depletion
  that physical effusion would take 221 µs to do); closure 0.14 atoms = 7.4e-15 of
  the inventory; max interval residual 1.6e-3 atoms. Trailing artificial rate
  5.3e16 s⁻¹ looks large but is (V/τ_g)(n_g − n_g*) with V/τ_g = 11.4 m³/s: it
  corresponds to n_g − n_g* = 4.7e15 m⁻³ = 0.016 % of n_g (n_g lags the slowly
  rising S by τ_g).
- Plasma: n_e mean 2.13e17 (0.93 × the projected 0-D n_eq of 2.3e17 — the 0-D
  estimate landed), **peak 1.64e18 = 4.1 × n_max** at the node z = 14.30 mm,
  r = 0.70 mm (grid Δr = Δz = 50 µm over the 3 mm exit radius). Cusp planes (B_z
  sign change on axis and at the wall, sampled P2 map): z = 6.0, 12.0, 17.95 mm;
  magnet mid-planes 3, 9, 15, 21 mm. The peak sits in the bottle between the 12.0
  and 17.95 mm cusps near the 15 mm mid-plane (field lines axial, off axis); the
  axial profile of max_r n_e is 7.3e17 at 8 mm, 5.9e17 at 10 mm, 2.7e17 at 12 mm
  (cusp), 1.6e18 at 14 mm, 1.3e18 at 16 mm, 2.4e17 at 18 mm (cusp), 4.8e17 at
  20 mm. Wall ion flux peaks 1.57e21 m⁻² s⁻¹ at z = 12.18 mm (cusp) with secondary
  peaks near 19 and 7 mm — ions leave along the cusp lines; wall ion mean energy
  up to 126 eV at 18.1 mm. Exit ion current density peaks on axis (460 A/m²) and
  falls to 25 A/m² at r = 2.8 mm. ⟨T_e⟩ 7.5 eV from the kinetic energy series,
  8.17 eV density-weighted from the maps, T_e max 59 eV; φ −10.7 … 337 V
  (337 V > 300 V anode: the potential hump that traps the beam electrons).
- Resolvability at the peak node: λ_D(1.64e18, 8.2 eV) = 16.6 µm → Δz/λ_D =
  Δr/λ_D = 3.0 (design 1.5 at n_max, gate 2.0 on the *reference* density, not the
  instantaneous peak — the gate did not trip because it is evaluated at n_max);
  ω_pe Δt 0.108 at the peak, max observed 0.118 (gate 0.2). The mean density is
  inside the budget; the peak region is under-resolved and reported as such (claim
  boundary, dashboard, README).
- Energy ledger: cumulative residual −2.86e-7 J = −4.4 % of the 6.51e-6 J electrode
  work; interval RMS 1.5e-11 J; gross source turnover 9.3e-7 J. Same order as the
  snapshot v2 fine cases (momentum-conserving scheme, ~27 particles per cell at n_eq).
- Remaining physical simplifications (unchanged from v1.3): uniform neutral profile
  despite the dynamic inventory (no depletion where the ionisation peak is — at 46 %
  utilisation this is no longer a small correction), no ion–neutral elastic/CEX
  collisions, no SEE / sputtering, Dirichlet 0 V exit plane with the electron
  injection at fixed current, prescribed static B, electrostatic axisymmetric (no
  azimuthal modes / anomalous transport), ion subcycling k = 8, one-cell cone
  stair-step, 3 mA / 0.019 mg/s operating point far below a real CFT.
- Committed `24ab82f4` chore(pic2d): steady-state v2 plateau results (development):
  summary/status/series/maps/run_state/checkpoint-final metadata for `results/` and
  the attempt-1 archive (needed by the history panel); particle checkpoints,
  series.jsonl and logs untracked. `.gitignore` negation block mirrors v1.

### Convergence pair

- `variants.json` (new; `protocol.json` stays byte-frozen so the finished run keeps
  verifying against its recorded protocol hash) declares `seed-b` (seed 20260904) and
  `w-half` (W = 4.2e4 = 0.7 W). Runner: `--case NAME` merges the variant into
  `case` / `stopping_rule.wall_budget_seconds` and writes to `results-NAME/`;
  `load_variants` / `apply_case` with tests (distinct config identities, base
  protocol untouched, unknown case fails closed).
- W/2 rejected for the budget: the base case ran 2.0 ms/step with 2.0 M particles at
  the plateau (launch-bound floor ~2 ms + ~0.5 ms per M), so W/2 (~4 M at plateau)
  projects to ≥ 3 ms/step × 5.1 M steps = 4.3 h > 3.5 h. 0.7 W (~2.9 M) projects to
  ~2.5 ms/step = 3.5 h for 5.1 M steps, i.e. it reaches the 3-transit floor (4.8 M
  steps, 3.3 h) inside the budget only if the plateau is declared at the first
  eligible checkpoint — marginal, recorded in the variant note; resumable if the
  budget stops it first.
- `seed-b` launched detached 2026-09-03T02:32:41Z (PID 49716,
  `results-seed-b/{run.log,run.err,launch.pid,status.jsonl,checkpoint/}`), 3.5 h
  budget. At 1.01 µs (step 674 200, 1127 s wall): N_e 367.6 k vs 366.0 k for the base
  run at the same step (+0.4 %), N_i 374.3 k vs 372.7 k, I_d 1.38 vs 1.44 mA,
  S 1.78e16 vs 2.28e16 s⁻¹ (interval-noisy), n_g 4.226e19 vs 4.242e19, T_e 7.26 vs
  7.21 eV, ω_pe Δt 0.080 vs 0.083 — igniting on the same trajectory; the seed
  changes nothing visible at this stage. Throughput 1.0 ms/step for the first
  0.35 µs, 2.5–2.6 ms/step from 0.7 µs (the base ran 2.55 at the same step). ETA
  to 4.8 M steps (3 transits) ≈ 2.9 h after launch, to 5.12 M ≈ 3.4 h — inside the
  3.5 h budget if the average stays ≤ 2.5 ms/step (the base averaged 1.98).
  `w-half` is launched only after `seed-b` ends (never two GPU campaigns).

### Dashboard and docs

- `modern/visualization/generate_pic2d_cft_steady_state.py` →
  `pic2d-cft-steady-state.html` (1.8 MB, self-contained, byte-deterministic).
  Headline: v1.3 plateau with a verification table (drifts colour-coded green/amber/
  red, transits, window currents, S and utilisation, n_g vs fixed point, peak/mean
  n_e, T_e, φ, both ledgers, peak node vs cusp planes, Δ/λ_D and ω_pe Δt at the
  peak, particle counts, wall time); time series (counts, currents with a robust
  0.2–99.8 % quantile y-range to clip the seed transient, n_g(t) vs (Q_in − S)/c
  with the n_g0 ceiling, atom rates with the Q_in line, φ range, energy ledger,
  ω_pe Δt with the 0.2 gate and the design value) with the trailing-20 % window
  shaded and the 3-transit floor dotted; plateau-window maps with the cusp planes
  dashed; wall/exit fluxes and the axial max_r n_e profile against the cusps and
  n_max; convergence table (single column until the pair finishes) plus the
  variant status table; neutral ledger and a-priori-vs-outcome budget table;
  history panels: steady-state predecessors (v1.2 reference, v1.3 attempt 1),
  snapshot v2 growth cases, snapshot v1 fail-closed cases (reused from the snapshot
  generator by import). Inputs hash-verified via sidecars; a protocol file that
  drifted from the run's recorded hash fails closed (test); history rows may
  predate the current protocol and carry their own hash with a flag.
- 9 dashboard tests (`test_pic2d_steady_state_dashboard.py`): hash binding, claim
  phrases ("single seed", "under-resolved", "not preregistered", "not validated"),
  determinism and checked-in HTML currency, offline self-containment, controls /
  accessibility fragments, strict JSON round trip, node syntax check, tampering
  rejection (8 mutations), protocol drift / tampered summary rejection.
- Headless Chrome screenshots: `%TEMP%\pic2d-cft-steady-state-desktop.png`
  (1440×5400) and `%TEMP%\pic2d-cft-steady-state-narrow.png` (430×11500).
  Visual review fixed three defects: the currents y-range was dominated by the
  seed-transient spike (0.49 A wall current in the first record), log-axis tick
  labels showed log10 values, and n_g(t) was hidden under the fixed-point trace
  (they coincide to 0.02 %; the fixed point is now drawn first).
- Campaign proposal drafted: `docs/workstreams/pic2d-campaign-v1-proposal.md`.

## 2026-09-03 (Phase 4, part 2): seed-b finished; comparison; w-0.7 launched

### seed-b outcome

- `results-seed-b/`: `wall_clock_budget_reached` at step 4,040,000, t = 6.06 µs =
  2.53 ion transits, wall 12,751 s (3.99 ms/step under heavy CPU contention from
  concurrent agents, vs 2.0 ms/step for the base run). No plateau declaration is
  possible (< 3 transits); its own trailing-20 % drifts were I_d +1.3 %, N_e +7.6 %
  (fails the 5 % gate, still densifying), n_g −1.8 %.
- `finalize --case seed-b` was NOT run: the runner's graceful budget stop already wrote
  the full artifact set with *window-average* maps (steps 3.6–4.0 M = 5.4–6.0 µs);
  `finalize` would have replaced them with instantaneous checkpoint maps (flux and
  ionisation maps zero) and rewritten the stop reason. The runner now refuses to
  re-finalize a run it finished itself unless `--allow-refinalize` is passed (test
  added). `finalize` is for killed/crashed runs only.
- Committed small artifacts: `chore(pic2d): steady-state v2 seed-b results
  (development)` (41ccb1ef); `.gitignore` tracks `results-seed-b/` with the same
  split as the base run.

### seed-b vs base (dashboard `comparisons` block; computed from the hash-verified series)

Window A — common window 4.848–6.06 µs (seed-b's trailing 20 %, the same simulated time
in both runs; both at 2.0–2.5 transits, still slowly densifying):

| quantity | base | seed-b | Δ rel. | SE base / seed-b (30 ns batch means) | z | pure shot-noise σ_rel |
| --- | --- | --- | --- | --- | --- | --- |
| I_d | 3.409 mA | 3.422 mA | +0.40 % | 7.9e-6 / 6.4e-6 A | 1.35 | 0.15 % |
| I_beam,i | 2.268 mA | 2.283 mA | +0.64 % | 5.4e-6 / 7.0e-6 A | 1.64 | 0.19 % |
| I_wall,i | 3.328 mA | 3.350 mA | +0.65 % | 1.5e-5 / 1.6e-5 A | 0.99 | 0.15 % |
| I_wall,e | 3.333 mA | 3.356 mA | +0.72 % | 1.5e-5 / 1.5e-5 A | 1.12 | 0.15 % |
| I_exit,e (returned) | 1.856 mA | 1.856 mA | +0.01 % | 6.3e-6 / 6.2e-6 A | 0.02 | 0.21 % |
| S | 3.858e16 /s | 3.899e16 /s | +1.05 % | 5.7e13 / 5.3e13 | 5.2 | 0.11 % |
| n_g | 3.020e19 | 2.994e19 m⁻³ | −0.83 % | 2.9e16 / 2.8e16 | −6.2 | – |
| N_e (macro) | 915.3 k | 920.5 k | +0.57 % | 3.1e3 / 3.2e3 | 1.18 | 0.10 % |
| N_i (macro) | 923.3 k | 928.8 k | +0.60 % | 3.1e3 / 3.3e3 | 1.23 | 0.10 % |
| ⟨T_e⟩ = (2/3)K/N | 7.635 eV | 7.676 eV | +0.53 % | 0.005 / 0.006 | 5.0 | – |
| φ_max | 339.1 V | 338.5 V | −0.18 % | 0.21 / 0.17 V | −2.2 | – |
| φ_mean | 175.0 V | 175.1 V | +0.06 % | 0.21 / 0.33 V | 0.27 | – |
| φ_min | −9.79 V | −9.64 V | +1.5 % (0.15 V) | 0.09 / 0.11 V | 1.04 | – |
| peak ω_pe Δt | 0.1096 | 0.1094 | −0.15 % | | −0.46 | – |

Reading: every plateau quantity agrees to < 1.1 %; the discharge, beam, wall and returned
currents, the particle counts and the potential range are consistent with the
within-run fluctuation statistics (|z| ≤ 1.6; φ_max marginal at 2.2). S (+1.05 %),
n_g (−0.83 %) and ⟨T_e⟩ (+0.53 %) differ by 5–6 of their batch-means SEs: these are
smooth, integrated quantities whose 30-ns-block SE is tiny (0.1–0.15 %), so the
difference is a genuine small seed-to-seed trajectory offset, not counting noise —
seed-b is ~0.6 % denser and ~0.5 % hotter at the same time, which gives ~1.1 % more
ionisation (dS/S ≈ dN/N + dT/T), and the neutral inventory answers with
Δn_g = −ΔS/c = −2.6e17 (−0.87 %) — exactly the observed −0.83 %. The pure shot-noise
expectation for ~1 M macro-particles (0.10 % for counts, 0.15–0.2 % for a current
averaged over 1.2 µs, 0.11 % for S) is a lower bound the observed spread exceeds by
3–7×: the run-to-run variance is dominated by correlated plasma fluctuations and by
the slow densification phase, not by counting statistics. Conclusion: the two seeds
are statistically consistent at the ≤ 1 % level; the seed-to-seed spread of the
headline quantities is ≈ 0.5–1 %, well inside the 5 % plateau tolerance. With n = 2
this is a spread estimate, not a variance.

Window B — base plateau window 7.2–7.68 µs vs seed-b 4.85–6.06 µs (time-offset, because
seed-b stopped 1.6 µs before the base window): I_d −0.64 %, I_beam −0.38 %, S −0.89 %,
n_g +0.85 %, but N_e −7.4 %, I_wall,i/e −10 %, φ_mean +4.9 %, φ_min +8.4 %,
ω_pe Δt −4.3 %. That is the base run's continued densification between 6 and 7.7 µs:
the discharge and beam currents saturate first (set by the injected 3 mA and the beam
fraction), while the density keeps growing with the extra ionisation going to the
walls (wall currents +10 %) and the bulk potential settling 8 V lower.

Maps (base 7.2–7.68 µs vs seed-b 5.4–6.0 µs, time-offset): n_e mean −4.6 %, peak
−8.2 % (consistent with N_e −7.4 %), rel. L2 10 %; T_e density-weighted +2.7 %, peak
65 vs 59 eV; φ RMS difference 9.5 V; wall ion/electron flux peaks at the same node
(z = 12.175 mm, the downstream cusp plane) with −12.6 % amplitude and 14 % L2 shape
difference; exit ion j_z on axis 306 vs 462 A/m² (−34 %) with the total I_beam agreeing to
0.6 % — the axis cell has the smallest area and is shot-noise dominated (the exit profile
L2 difference of 24 % is concentrated in the inner cells).

### w-0.7 (convergence pair b) launched

- `variants.json`: the case was renamed `w-half` → `w-0.7` before launch (it is
  W × 0.7 = 4.2e4, not W/2; the W/2 projection exceeded the 3.5 h budget). README,
  `.gitignore`, and both test files follow.
- Launched 2026-09-03T07:04:41Z, PID 9856, `results-w-0.7/` (run.log/run.err/launch.pid
  next to the runner's own status.jsonl/series.jsonl/checkpoint/). Config identity
  `fine-w4.2e4-ng0-5.5e19-seed5e16-inventory-w0.7`, seed 20260903, wall budget 12,600 s.
- 3-minute check (step 120,600, t = 0.181 µs): 357 k e⁻ / 384 k ions (base at the same
  step: 249 k / 269 k — the 1/0.7 particle ratio as designed), I_d 1.44 mA, S 1.68e16 /s,
  n_g 4.355e19 (base 4.353e19), ⟨T_e⟩ 8.53 eV (base 8.57), ω_pe Δt 0.070 (base 0.086:
  the peak-cell density estimate is smoother with more particles), 2.41–2.47 ms/step
  (GPU otherwise idle). Ignition trajectory matches the base run.
- ETA: at 2.44 ms/step, 3 transits (4.8 M steps) need 3.17 h more against 3.42 h of budget
  left — reachable (3.27 transits) only if the throughput holds. It will not fully hold:
  the base run ran 2.0 ms/step at 1.0 M particles under contention and w-0.7 will carry
  ≈ 1.4 M; at 2.8–3.0 ms/step the 3-transit mark needs 3.6–3.9 h > budget. Expect a
  budget stop at ≈ 2.6–2.9 transits; it will then be compared over the common window like
  seed-b (the comparison block handles that automatically). Resumable if a budget
  extension is decided.

### Dashboard

- `generate_pic2d_cft_steady_state.py`: `build_case(raw_out=…)` hands the full-resolution
  series/maps to `build_comparison`, which writes the `comparisons` payload block: window
  A (variant trailing 20 % in both runs), window B (base plateau window vs variant
  trailing window when they do not overlap), map comparison (mean/peak/L2, wall-flux peak
  position, exit j_z on axis), method text. Statistics: batch means over 30 ns blocks
  (100 series intervals; longer than the ns fluctuation correlation time), z-scores, pure
  shot-noise reference 1/√(macro events in the window). `validate_payload` requires
  exactly one comparison per finished variant. The convergence panel renders the tables
  with |z| colour coding (green < 2, amber < 3, red); the variant status table shows
  transits and plateau declaration.
- Tests: 10 dashboard tests (new: common-window comparison exists, per-cent agreement,
  shot-noise reference below the observed difference, window B offset, same wall-flux
  peak node) + 13 runner tests (new: re-finalize refused). Screenshot:
  `%TEMP%\pic2d-steady-state-seedb-full.png` (1400×9000) and the convergence crop
  `%TEMP%\pic2d-steady-state-seedb-convergence.png`.
