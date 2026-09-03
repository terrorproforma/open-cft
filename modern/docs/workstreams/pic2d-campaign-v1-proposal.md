# PIC-2D campaign v1 — proposal for a one-shot preregistered kinetic campaign

**Status: proposal (document only). Nothing here has been run, frozen or admitted.**
The development steady state of `experiments/pic2d_cft_steady_state_v2` (model v1.3,
single seed, plateau at 3.2 ion transit times) is the evidence this proposal is sized
from; it is *not* the campaign. A preregistered campaign is one shot: the protocol
below is frozen (hash) before launch, every gate is decided in advance, and the result
is admitted or rejected by the gates, not by inspection.

**Revision 2 (2026-09-03, after the PIC literature review
`modern/docs/literature/pic-mcc-blockers.md`, 116 verified references, and model v1.4).**
The review's recommendations are folded into the sections below and collected in
Section 8; items already implemented in v1.4 (`pic2d-model-v1.4.json`) are marked
**[v1.4 done]**. The seed-b comparison is in; the W × 0.7 case is running.

**Revision 3 (2026-09-03, model v2.0 plume block).** The W × 0.7 case finished and is
compared in `experiments/pic2d_cft_steady_state_v2/README.md` over the common windows:
seed-b moves I_d by +0.4 % and S by +1.1 %, W × 0.7 moves I_d by +4.7 %, S by −4.5 %,
n_g by +3.9 % and the peak n_e by −12 % — the particle-resolution sensitivity exceeds the
seed spread by 5–10×, so block B keeps its place in the matrix and the v1.3 plateau numbers
carry a ~5 % band (peak density ~10 %). The exit-boundary item of Section
3.4 is now the v2.0 plume block (`pic2d-model-v2.0.json`, marked **[v2.0 done]**); the
plume development run `pic2d_cft_plume_v1` precedes the channel-only v3 run (Section 8,
row 4.4).

## 1. What the development plateau established (sizing evidence)

| quantity | development plateau (v1.3 attempt 2) | consequence for the campaign |
| --- | --- | --- |
| operating point | n_g0 = 5.5e19 m⁻³ (Q_in = 8.55e16 s⁻¹ = 0.0186 mg/s), 3 mA @ 2 eV injection, 300 V, seed 5e16 m⁻³ @ 5 eV | frozen as the campaign point (Section 2.1); ignition is seed-thresholded, so the seed is part of the protocol |
| plateau | I_d 3.44 mA, I_beam,i 2.29 mA (0.67 I_d), S 3.93e16 s⁻¹ (46 % utilisation), n_g 2.97e19 = (Q_in − S)/c to 0.03 % | expected values for the gates; the beam fraction and utilisation are the two headline observables |
| mean n_e | 2.13e17 m⁻³ (0.93 × the 0-D projection) | the 0-D projection is a good *mean* estimator |
| peak n_e | 1.64e18 m⁻³ = 4.1 × n_max, between the 12.0 and 17.95 mm cusps | the grid must be budgeted on the **peak**: cells of 3 λ_D there are not admissible |
| ω_pe Δt | 0.118 max at Δt = 1.5 ps | Δt = 1.5 ps is admissible at the peak (gate 0.2) but leaves 1.7× margin only |
| time to plateau | 3.2 transits (7.7 µs, 5.1 M steps), N_e drift +4.98 % at the first eligible checkpoint, ω_pe Δt still rising | the plateau criterion needs a stricter form (Section 2.4) and ≥ 5 transits |
| throughput | 1.0 ms/step ≤ 0.3 M particles, 2.5 ms/step ≥ 0.35 M, 2.0 ms/step averaged over a 2 M-particle run (WDDM, launch-bound) | GPU-hour estimates in Section 5 use 2.5 ms/step + 0.5 ms per extra M particles + 1.0 ms per extra 2× cells (Poisson block-Thomas scales with the grid) |
| ledgers | energy residual −4.4 % of the electrode work; atom closure 7e-15 | ledger gates in Section 2.5 |
| seed variance (seed-b, common window) | ≤ 1.1 % on I_d, I_beam, S, n_g, N_e; 3–7× the shot-noise floor on the smooth quantities (MCC-driven thermalisation, Turner 2006) | σ_dev in Section 2.3 is 1.1 %, so the seed gate is max(2.2 %, 5 %) = 5 % |
| wall-ion recycling (review blocker 3) | 59.9 % of S absorbed at the walls and lost from the atom balance in v1.3 → gross 46 % overstates the atoms consumed | v1.4 recycles them; the campaign quotes **net** utilisation ((S − R)/Q_in) as the headline and gross alongside; expected fixed point moves to n_g* ≈ 4.1e19 (Section 2.1) |

## 2. What the campaign would freeze

### 2.1 Operating point (identical to the development plateau)

- Xenon feed Q_in = 8.551e16 atoms/s (0.01864 mg/s) = c · 5.5e19 m⁻³ with
  c = v̄ A_exit / 4 at 300 K; MCC null-collision ceiling n_g0 = 5.5e19.
- Electron injection 3.00 mA at 2 eV, flux-weighted half-Maxwellian at the exit plane;
  anode 300 V, exit plane 0 V (or the exit-boundary variant selected in Section 3.4).
- Seed plasma 5e16 m⁻³ at T_e = 5 eV, T_i = 0 (initial condition, not replenished).
- Geometry and field: the divergent-exit channel (bore 2 mm, cone from 18 mm to
  3 mm at 24 mm) and the P2 ψ field map, both by hash.
- Neutral closure **[v1.4 done]**: wall-ion recycling ON (γ = 1.0, wall 400 K, exit ions
  not recycled), so the campaign's expected fixed point is n_g* ≈ 4.1e19 m⁻³ (3.8–4.5e19;
  frozen-S bound 4.49e19), S ≈ 5.4e16 s⁻¹, net utilisation ≈ 25 %, mean n_e ≈ 2.9e17,
  peak ≈ 2.2e18 — the numbers Section 2.2 is sized on. The artificial relaxation is **OFF**
  for the campaign (review blocker 3(e): no precedent); the neutral state then evolves on
  the physical effusion time V/c = 221 µs, which is why Section 2.4 requires the ≥ 50 µs
  development case before the plateau criterion is frozen.
- Justification for not moving to a realistic 0.1–1 A / 0.1–1 mg/s point: the
  resolvable density ceiling at this grid (Section 2.2) is ~1e18–2e18; a real discharge
  at 1e19–1e20 needs 10× finer cells and ≥ 100× the GPU hours. The campaign claims a
  *self-consistent kinetic operating point of the model*, not a thruster operating point.

### 2.2 Grid, Δt, W (derived from the plateau peak)

- Peak budget: n_peak,design = 2.2e18 m⁻³ (the v1.4 recycled-fixed-point estimate; 1.35 ×
  the observed v1.3 1.64e18) at T_e = 7.4 eV (the observed peak-node value) →
  λ_D,min = 13.6 µm → cell ≤ 2 λ_D,min = 27 µm; **grid 100 × 800** (Δr = Δz = 30 µm over
  3 mm × 24 mm; 2.8× the cells of 60 × 480) gives 2.2 λ_D per cell at the design peak, to
  be confirmed ≤ 2.0 by the v3 development run's measured peak (if the v3 peak exceeds
  2.2e18 at 7.4 eV the campaign grid becomes 120 × 960, Δ = 25 µm). This is what the
  Greifswald analogue ran (Brandt et al. 2016: Δx = 2 λ_D at the peak); do not relax to
  3 λ_D anywhere (review blocker 1(d)1).
- **Peak-node Debye gate, fail-closed [v1.4 done]**: `PeakDebyeGateConfig(max_cells_per_debye
  = 2.0, min_macro_particles_at_peak = 32)` on the *instantaneous peak node* (densest node
  holding ≥ 32 macro-electrons) at every series record; not on a reference density (the
  development gate was evaluated at n_max and did not trip at 3 λ_D). The v3 development
  run uses 4.5 on the 60 × 480 grid, declared under-resolved.
- **Grid-heating triad [v1.4 done]** (review blocker 1(d)3; Birdsall and Maron 1980; Ueda
  et al. 1994; Adams et al. 2025): energy-ledger residual / electrode work ≤ 5 % (campaign;
  10 % in v3 development), trailing-window drifts of the dense-cell T_e, S and peak ω_pe Δt
  ≤ 3 % as plateau preconditions and ≥ 25 % as a fail-closed stop after one transit;
  additionally the cross-variant member: T_e and S of the W and grid cases within the
  Section 2.3 gates.
- **20 µm discriminator**: block C (150 × 1200, Δ = 20 µm) is the grid-convergence
  discriminator; the grid claim is accepted only if it agrees within 10 % (Section 2.3).
- Δt = 1.0 ps: ω_pe Δt = 0.080 at n_peak,design (gate 0.2, 2.5× margin), Ω_ce Δt = 0.051
  at |B|max = 0.291 T; ion subcycle k = 8 (ω_pi · 8 Δt = 0.0018 at n_peak,design).
- W = 3.0e4 (W/2 of the development run): particles per cell at the peak ≈
  2.2e18 × (30 µm)² × 2π r / 3e4 ≈ 29 at r = 0.7 mm; total ≈ 2 × 2.9e17 × 3.43e-7 / 3e4
  = 6.6 M macro-particles at the recycled plateau (4.9 M at the v1.3 mean density). (The
  development pair's W × 0.7 case sets the particle-resolution gate, Section 2.3.)
- Sync / series interval 200 steps, checkpoint every 40 000 steps, all as v1.3; CUDA-graph
  step replay ON **[v1.4 done]** (bitwise-identical to direct launches; the v3 run measures
  the 1–3 M particle ms/step).

### 2.3 Seeds and convergence gates (from the development pair)

- **≥ 3 RNG seeds** at the frozen (grid, Δt, W); the campaign reports mean ± sample
  standard deviation of every plateau quantity over the seeds.
- Convergence gates derived from the development pair (`seed-b`, `w-0.7` at 0.7 W):
  the campaign passes only if, for I_d, I_beam,i, S, n_g and mean n_e, the seed-to-seed
  spread (max − min)/mean ≤ max(2 σ_dev, 5 %) where σ_dev is the relative seed spread
  measured by the pair (**seed-b: 1.1 % → the gate is 5 %**), **and** the difference
  between the W and W/2 campaign cases is ≤ max(2 × the pair's W-sensitivity, 5 %). If the
  pair shows a W-sensitivity above 10 % on any headline quantity, the campaign is not
  launched at this W (fail closed: re-size first).
- Reporting (review blocker 5(d)3): mean ± sample standard deviation over the seeds, the
  batch-means standard error with its block length, and the shot-noise floor, as in the
  seed-b comparison; state that the literature offers no seed-variance precedent.
- Convergence statement in the benchmark form of Charoy et al. 2019 (review 5(d)4): same
  simulated window, ppc doubled, 5 % agreement on the time-averaged axial profiles of E_z,
  n_i and T_e in addition to the scalar currents.
- One grid-refinement case (Δ = 20 µm, 150 × 1200) at one seed: mean n_e, I_d, I_beam,i
  within 10 % of the 30 µm result, otherwise the grid claim is rejected.

### 2.4 Plateau criterion (stricter than v1.3)

- Tracked drifts over the trailing 20 % of the simulated time: I_d, N_e, n_g **and**
  the window peak node density and ω_pe Δt max (the development run passed while both
  were still rising).
- Threshold 3 % (v1.3: 5 %), and the criterion must hold at **two consecutive**
  checkpoints (60 ns apart) after **≥ 5 ion transit times** (12 µs at 2.4 µs per transit).
- Spectral check (review blocker 5(d)1): the I_d and n_g power spectra over the plateau
  window show no line above the noise floor below 1 MHz; the plateau is stated to be
  conditional on the neutral closure.
- **≥ 50 µs development case before freezing** (review blockers 3(d)3, 5(d)2): one case at
  the physical neutral time scale (relaxation OFF) and reduced W (≥ 50 µs ≈ 21 transits;
  ~33 M steps at 1.5 ps) to observe whether the closure oscillates. The predator-prey
  estimate for this channel is f ≈ (1/2π)√(v_i v_n)/L ≈ 12 kHz (period ≈ 80 µs ≈ 35
  transits); Kahnfeld et al. 2018 found ~100 kHz for the 51 mm DM3a. The plateau criterion
  above is frozen only if that case does not breathe.
- **Breathing-mode contingency**: if the development case breathes, the campaign
  estimands become **period-averaged means** over ≥ 3 periods (I_d, I_beam, S, n_g, mean
  n_e, wall fluxes) plus the breathing frequency and the peak-to-mean amplitude of I_d and
  n_g; the wall budget per case must cover ≥ 3 periods after the transient (≥ 240 µs at
  12 kHz — ~160 M steps, which is not affordable on this workstation at 1.5 ps; the
  contingency then requires a datacentre GPU or the mass-ratio/self-similarity scaling of
  the HEMP community with its retrieval demonstrated, Section 8). Kahnfeld et al. 2018 is
  then the model-to-model comparison.
- Wall budget 14 h per case; a case stopped by the budget without a declared plateau is
  reported as "no plateau" (not resumed for the campaign statistics; it may be resumed
  as development).

### 2.5 Ledger-closure gates (fail-closed at finalization)

- Energy: |cumulative residual| / electrode work ≤ 5 % over the plateau window and
  interval residual RMS ≤ 1e-10 J (development: 4.4 %, 1.5e-11 J).
- Atoms: cumulative closure ≤ 1e-9 of the inventory (development: 7e-15), with the
  recycled ledger included (fed + recycled − ionised − effused − artificial = V Δn_g)
  **[v1.4 done]**.
- Charge: S · e − (I_beam,i + I_wall,i + I_anode,i) ≤ 5 % of S · e over the window
  (development: 4 %, the growing inventory).
- Poisson true-residual contract 1e-10 |rhs| at every sync (unchanged).
- Grid-heating triad soft bounds hold at finalization (Section 2.2) **[v1.4 done]**.

### 2.6 Claim boundary (fixed before launch)

"Self-consistent kinetic operating point of the axisymmetric electrostatic PIC-MCC model
v2.0 of the divergent-exit CFT channel at 3 mA / 0.019 mg/s: discharge current, ion beam
fraction, propellant utilisation, wall-flux distribution and density structure relative
to the cusps, with seed variance and W/grid sensitivity." Not a thruster performance
prediction; not validated against any experiment; the operating point is far below a
real CFT; the model excludes what Section 3 lists as *not yet added* at launch time.

## 3. What must be added first (blocking; model v2.0)

| addition | why it blocks | scope |
| --- | --- | --- |
| **3.1 ion–neutral collisions (elastic + charge exchange)** | at n_g = 3e19 the Xe⁺–Xe CEX mean free path (σ_CEX ≈ 5e-19 m² at 50–300 eV) is ~7 cm — long against the 24 mm channel but the *thermal* ions born at the peak see 1/(n_g σ) ≈ 5–7 cm too; still, CEX converts fast ions to slow ones that feed the wall flux and changes I_beam by up to the CEX fraction (few %) — the beam fraction is a headline observable so it must be in the model | LXCat/Miller-2002 Xe⁺–Xe tables in the hash-bound cross-section loader; null-collision MCC for ions on the subcycle step; ledger of momentum/energy transferred to neutrals (uniform bath); tests: rate vs n_g, isotropic-elastic energy conservation, CEX swaps velocities |
| **3.2 secondary electron emission** | the wall electron current equals the wall ion current (3.7 mA each) — SEE at 7–60 eV impact on BN/alumina (yield 0.3–1 at these energies) changes the sheath and the electron energy balance by O(1); in a HEMP the wall flux is concentrated at the cusps (Matyash et al. 2010; ours peaks at the 12.2 mm cusp) where the sheath is thinnest and can go space-charge-limited locally | **scaffold [v1.4 done]**: `SEEConfig` with the Vaughan 1989 yield for BN (parameters provisional until digitised from Dunaevsky, Raitses and Fisch 2003 / Tondu, Belhaj and Inguimbert 2011), first crossover and the Hobbs–Wesson 1967 limit as a virtual-wall diagnostic; `enabled = True` fails closed. **Remaining**: emission itself (true-secondary / backscattered split, 1–2 eV half-Maxwellian), the Hobbs–Wesson cap or an inverse-sheath treatment (Campanell, Khrabrov and Kaganovich 2012), surface-charge ledger; report the cusp-local yield and the re-emitted fraction; the campaign's **SEE-off case is the reported sensitivity** (Tavant et al. 2018: spatially averaged mobility roughly unchanged, wall-local sheath changed) |
| **3.3 neutral spatial profile** | 46 % gross utilisation with a *uniform* n_g is inconsistent: the depletion happens where S peaks (between the last two cusps) and the feed enters at the anode (Petronio et al. 2024: ionisation depends on the spatial correlation of n_e and n_g) | wall recycling into the 0-D inventory **[v1.4 done]**; **remaining**: the (r,z) free-molecular view-factor model of Katz and Mikellides 2011 on the plasma mask (anode feed, per-cell ionisation sink from the MCC tallies, diffuse wall re-emission at the wall temperature, exit effusion), cell-wise n_g in the MCC (null ceiling at the anode value); ledger closure per cell; tests: uniform-sink limit recovers the 0-D recycled fixed point |
| **3.4 exit-boundary sensitivity → plume block (review 4.4 / blocker 4d)** | the Dirichlet 0 V exit plane fixes φ where the plume should be free; φ_max = 337 V > anode and the returned-electron current (1.8 mA) depend on it | **[v2.0 done]** `pic2d-model-v2.0.json`: L-shaped domain (channel + 12 × 12 mm plume box at 50 µm = 240 × 720 cells, 78 k unknowns; Brandt et al. 2016: 20 × 5 mm box "still too small"), channel walls and the cone as internal dielectric boundaries, front face = dielectric flange to r_body + **grounded conductor** beyond (declared: HEMP-T pole faces are metal on the chamber reference), far field Dirichlet 0 V on r = R_plume and z = z_max (the literature's practice; Brandt's Neumann variant changed the plume ratios, so it is the block-D sensitivity), off-axis **cathode annulus** (r 4.5–6 mm, z = L + 2–4 mm, 2 eV Maxwellian) emitting the interval's discharge current (continuity rule, clamped 3–15 mA; Szabo 2001 quasi-neutral injection; Charoy et al. 2019) with the exit-plane injection kept as the legacy A/B, two-zone neutrals (channel inventory + cosine-cone effusion shape as the plume MCC factor; no CEX until 3.1), momentum ledger + momentum-flux thrust + Maxwell-stress force closure, j_i(θ), IEDF (peak − U_a against Koch et al. IEPC-2011-236's ≈ −15 V as context), exit-plane potential, acceleration region, Isp, anode efficiency; plume-boundary pile-up gate fail-closed. **Remaining**: the development run `pic2d_cft_plume_v1` (4 h, ~5 ms/step → 1.3–1.9 ion transits per launch; resumable), a 1–1.5 L_channel box (needs a longer FEM solve: the P2 domain ends at z = 36.25 mm), the Neumann-side and floating-face variants as block D |
| 3.5 resolvability gate on the instantaneous peak | the development gate did not trip at 3 λ_D per cell | **[v1.4 done]** `PeakDebyeGateConfig` at every series record, fail-closed, with the particle floor; grid-heating triad recorded and gated by the runner |
| 3.6 finalize-with-window-maps | `finalize` produces instantaneous maps only | device window accumulators persisted in the checkpoint so a budget-stopped case still has window averages |
| 3.7 Bohm-scattering sensitivity hook | an axisymmetric electrostatic code excludes the drift instability by construction; any I_d agreement is fortuitous, imposed or axisymmetric-low-frequency (Cho et al. 2015) | **[v1.4 done]** `AnomalousCollisionConfig(alpha)`: ν_an = α ω_ce isotropic speed-preserving scattering on both backends, tallied; campaign block E at α = 1/64 and 1/16 (Szabo 2001; Brandt et al. 2016; Smirnov, Raitses and Fisch 2004), reported not binding; plus the Cho et al. 2015 effective-mobility diagnostic over the plateau window |

Each addition ships with its typed spec (`pic2d-model-v2.0.json`), fail-closed
validation, tests and a development run before the campaign protocol is frozen.

## 4. Campaign matrix

| block | cases | steps (Δt = 1 ps, ≥ 5 transits + plateau) | particles | ms/step (est.) | hours/case |
| --- | --- | --- | --- | --- | --- |
| A. main | 3 seeds × (100 × 800, W = 3e4) | 12–14 M | ~5 M | ~5.5 | 18–21 |
| B. W sensitivity | 1 seed × W = 1.5e4 | 12–14 M | ~10 M | ~8 | 27–31 |
| C. grid refinement | 1 seed × (150 × 1200, W = 3e4) | 12–14 M | ~5 M | ~8 | 27–31 |
| D. exit boundary | 1 seed × alternative exit (from 3.4) | 12–14 M | ~5 M | ~5.5 | 18–21 |
| E. Bohm sensitivity (review 4.3) | 1 seed × α = 1/64, 1 seed × α = 1/16 | 12–14 M | ~5 M | ~5.5 | 18–21 each |
| F. SEE-off/on (review 4.2; only once 3.2 emission exists) | 1 seed × SEE on | 12–14 M | ~5 M | ~6 | 20–23 |

The 14 h wall budget per case (Section 2.4) is therefore **too small at Δt = 1 ps on this
machine**; either Δt = 1.5 ps (0.12 at the peak; 8–9 M steps; blocks A/D 12–14 h,
B/C 18–20 h) or a faster GPU. Decision to be made and frozen in the protocol; the
proposal recommends Δt = 1.5 ps with the peak gate at 0.15. The ms/step column predates
the v1.4 CUDA-graph step: at ≤ 0.2 M particles the graph removed 65–80 % of the step time
under GPU contention (7.8 → 1.5 ms at 9 k; 8.2 → 3.0 ms at 0.2 M); the v3 run supplies the
1–3 M figure and the table is to be re-priced from it before freezing.

## 5. Estimated GPU hours (this workstation, WDDM, launch-bound)

| item | hours |
| --- | --- |
| development pair (seed-b done 3.5 h; w-0.7 running 3.5 h) | 7 |
| v3 development run (v1.4, recycling, 3.5 h budget) | 3.5 |
| ≥ 50 µs development case at the physical neutral time scale, reduced W (Section 2.4; ~33 M steps at ~1.5–2 ms/step with the graph) | 14–18 |
| model v2.0 additions: development runs to validate 3.1–3.4 (4 × ~3 h) plus the two exit-boundary variants (2 × ~4 h) | 20 |
| block A (3 seeds) at Δt = 1.5 ps | 36–42 |
| block B (W/2) | 18–20 |
| block C (grid 20 µm) | 18–20 |
| block D (exit boundary) | 12–14 |
| block E (Bohm α = 1/64, 1/16) | 24–28 |
| finalization / dashboard regeneration | 1 |
| **total** | **≈ 155–175 GPU-hours (7 days wall, sequential) before the graph re-pricing; block F only once SEE emission exists** |

At Δt = 1.0 ps the total rises by ~50 %. A datacentre GPU (no WDDM launch floor, ~4× the
throughput at these particle counts) or the measured CUDA-graph gain at plateau particle
counts (pending from v3) would bring the 1.5 ps campaign to ~40–60 GPU-hours. If the
breathing contingency of Section 2.4 is triggered, the per-case budget rises by ≥ 10×
and the campaign is not affordable on this workstation without scaling.

## 6. What the campaign does *not* do

- No realistic discharge current or mass flow; no thrust, Isp or efficiency claim.
- No azimuthal physics (electrostatic, axisymmetric): anomalous transport is absent
  by construction and the campaign cannot bound it.
- No self-consistent magnetic field, no wall erosion/sputtering, no neutral wall
  accommodation beyond the 1-D axial model.
- No experimental validation: there is no measured discharge at this operating point.

## 7. Preregistration checklist (to be completed before launch)

1. `pic2d-model-v2.0.json` and `pic2d-campaign-v1-protocol.json` written, validated,
   hashed; the hashes recorded in this document's successor (the protocol README).
2. Development pair result (seed spread, W sensitivity) recorded and the convergence
   gates of Section 2.3 instantiated with numbers.
3. Exit-boundary variant chosen (Section 3.4) with the development evidence cited.
4. Grid/Δt/W and wall budget decided per Section 4; the fail-closed peak gate in place.
5. Seeds listed; results directories named; nothing else runs on the GPU during the
   campaign (sequential cases).
6. Claim boundary text (Section 2.6) frozen verbatim; the dashboard generator for the
   campaign reads only hash-verified artifacts and refuses protocol drift.
7. The ≥ 50 µs physical-neutral development case (Section 2.4) has run and its outcome
   (no breathing / breathing with period) decides between the plateau criterion and the
   period-averaged estimands before anything is frozen.
8. The v3 development run's peak-node cells/λ_D and ms/step have re-priced Sections 2.2
   and 4–5.

## 8. What the literature review changed (mapping to `pic-mcc-blockers.md`)

| review item | proposal change | status |
| --- | --- | --- |
| Blocker 1 — resolve the **peak**, not the mean; Brandt et al. 2016 ran Δx = 2 λ_D; grid heating is not removed by a better Poisson solve (Adams et al. 2025) | peak-node Debye gate fail-closed at 2.0 on the 30 µm grid (4.5 on the development grid, declared); 20 µm discriminator (block C); grid-heating triad recorded and gated; no artificial permittivity in the claim | gate + triad **v1.4 done**; grid pending campaign |
| Blocker 2 — cost; per-launch overhead (NVIDIA 2019); nobody reports convergence after 3 transits | CUDA-graph replay of the whole step (bitwise); budget re-scoped around the literature's "converged" (long time average) and the ≥ 50 µs development case; no implicit-EC / sparse grids for this campaign (documented ppc penalty, no cusp precedent) | graph **v1.4 done**; 1–3 M ms/step pending from v3 |
| Blocker 3 — wall recycling missing (Szabo 2001/2014; Brandt et al. 2016 quote *net* ionisation); no precedent for the artificial relaxation | recycling in the 0-D balance with its ledger; net utilisation as the headline; relaxation default OFF, kept ON only for the v3 development run and recorded; view-factor neutrals for v2.0 | recycling **v1.4 done**; view factors pending |
| Blocker 5 — the HEMP community does not expect a strict steady state (Kahnfeld et al. 2018: ~100 kHz breathing; predator-prey estimate here ~12 kHz); MCC thermalisation makes seed variance exceed shot noise (Turner 2006) | ≥ 50 µs physical-neutral case before freezing; breathing contingency with period-averaged estimands and ≥ 3 periods; spectral check; ≥ 3 seeds with batch-means SE and shot-noise floor; Charoy 2019 profile-convergence form | criterion text updated; case pending |
| 4.2 SEE (Vaughan 1989; Dunaevsky et al. 2003; Tondu et al. 2011; Hobbs–Wesson 1967) | SEE scaffold with BN Vaughan yield, fail-closed enable; SEE-off case is the reported sensitivity | scaffold **v1.4 done**; emission pending |
| 4.3 anomalous transport (Szabo; Brandt; Smirnov et al. 2004; Cho et al. 2015) | Bohm-scattering hook α ∈ {1/64, 1/16} as block E, not binding; effective-mobility diagnostic | hook **v1.4 done** |
| 4.4 / blocker 4d — exit-plane and cathode boundary (Szabo 2001; Charoy et al. 2019; Boeuf and Garrigues 2018; Brandt et al. 2016; Duras et al. 2017) | plume block as model v2.0 (Section 3.4): L-shaped domain with the far field at Dirichlet 0 V, grounded front-face conductor, off-axis cathode annulus on the current-continuity rule, two-zone neutrals, thrust from momentum flux with the Maxwell-stress force closure, j_i(θ) / IEDF / divergence / Isp / anode efficiency as **development numbers**; block D becomes the far-field-Neumann and floating-face variants of the same box; the **channel-only steady-state v3 is deferred behind the plume development run** (the exit-plane closure it would freeze is the one v2.0 replaces) | model **v2.0 done**; development run `pic2d_cft_plume_v1` pending the GPU; block D re-priced at ~5–7 ms/step × 78 k unknowns |
| Blocker 6 — validation targets | **Brandt et al. 2016 numbers are model-to-model context labelled by closure, never validation**: their micro-HEMPT (14 mm × 1.5 mm, 400 V, 0.27 sccm; static DSMC neutrals, Bohm diffusion, SEE) gave anode current 4.3 mA (4.5 measured), net ionisation 24 %, beam 2.5 mA, ~10 V / ~5 V steps at the internal cusps, 160 eV cusp ions, Δx = 2 λ_D, 76 µs to quasi-steady state; our closure differs (2 mm bore, 300 V, 0.019 mg/s, 0-D neutrals with recycling, no Bohm, no SEE), so the campaign reports its own numbers beside theirs with the closure differences listed, and makes no agreement claim | text rule adopted here and in the v3 protocol |
