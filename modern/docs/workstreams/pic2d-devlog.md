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
