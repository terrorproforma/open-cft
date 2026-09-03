# PIC-2D campaign v1 — proposal for a one-shot preregistered kinetic campaign

**Status: proposal (document only). Nothing here has been run, frozen or admitted.**
The development steady state of `experiments/pic2d_cft_steady_state_v2` (model v1.3,
single seed, plateau at 3.2 ion transit times) is the evidence this proposal is sized
from; it is *not* the campaign. A preregistered campaign is one shot: the protocol
below is frozen (hash) before launch, every gate is decided in advance, and the result
is admitted or rejected by the gates, not by inspection.

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

## 2. What the campaign would freeze

### 2.1 Operating point (identical to the development plateau)

- Xenon feed Q_in = 8.551e16 atoms/s (0.01864 mg/s) = c · 5.5e19 m⁻³ with
  c = v̄ A_exit / 4 at 300 K; MCC null-collision ceiling n_g0 = 5.5e19.
- Electron injection 3.00 mA at 2 eV, flux-weighted half-Maxwellian at the exit plane;
  anode 300 V, exit plane 0 V (or the exit-boundary variant selected in Section 3.4).
- Seed plasma 5e16 m⁻³ at T_e = 5 eV, T_i = 0 (initial condition, not replenished).
- Geometry and field: the divergent-exit channel (bore 2 mm, cone from 18 mm to
  3 mm at 24 mm) and the P2 ψ field map, both by hash.
- Justification for not moving to a realistic 0.1–1 A / 0.1–1 mg/s point: the
  resolvable density ceiling at this grid (Section 2.2) is ~1e18–2e18; a real discharge
  at 1e19–1e20 needs 10× finer cells and ≥ 100× the GPU hours. The campaign claims a
  *self-consistent kinetic operating point of the model*, not a thruster operating point.

### 2.2 Grid, Δt, W (derived from the plateau peak)

- Peak budget: n_peak,design = 2.0e18 m⁻³ (1.2 × the observed 1.64e18) at T_e = 8 eV →
  λ_D,min = 14.9 µm → cell ≤ 2 λ_D,min = 29.8 µm. **Grid 100 × 800** (Δr = Δz = 30 µm
  over 3 mm × 24 mm; 2.8× the cells of 60 × 480). Fail-closed gate: cell/λ_D ≤ 2.0 on
  the *instantaneous peak node density*, not on a reference density (the development gate
  was evaluated at n_max and did not trip at 3 λ_D).
- Δt = 1.0 ps: ω_pe Δt = 0.080 at n_peak,design (gate 0.2, 2.5× margin), Ω_ce Δt = 0.051
  at |B|max = 0.291 T; ion subcycle k = 8 (ω_pi · 8 Δt = 0.0018 at n_peak,design).
- W = 3.0e4 (W/2 of the development run): particles per cell at the peak ≈
  2.0e18 × (30 µm)² × 2π r / 3e4 ≈ 26 at r = 0.7 mm; total ≈ 2 × 2.13e17 × 3.43e-7 / 3e4
  = 4.9 M macro-particles at the plateau. (The development pair's W × 0.7 case sets the
  particle-resolution gate, Section 2.3.)
- Sync / series interval 200 steps, checkpoint every 40 000 steps, all as v1.3.

### 2.3 Seeds and convergence gates (from the development pair)

- **≥ 3 RNG seeds** at the frozen (grid, Δt, W); the campaign reports mean ± sample
  standard deviation of every plateau quantity over the seeds.
- Convergence gates derived from the development pair (`seed-b`, `w-half` at 0.7 W):
  the campaign passes only if, for I_d, I_beam,i, S, n_g and mean n_e, the seed-to-seed
  spread (max − min)/mean ≤ max(2 σ_dev, 5 %) where σ_dev is the relative seed spread
  measured by the pair, **and** the difference between the W and W/2 campaign cases is
  ≤ max(2 × the pair's W-sensitivity, 5 %). If the pair shows a W-sensitivity above
  10 % on any headline quantity, the campaign is not launched at this W (fail closed:
  re-size first).
- One grid-refinement case (Δ = 20 µm, 150 × 1200) at one seed: mean n_e, I_d, I_beam,i
  within 10 % of the 30 µm result, otherwise the grid claim is rejected.

### 2.4 Plateau criterion (stricter than v1.3)

- Tracked drifts over the trailing 20 % of the simulated time: I_d, N_e, n_g **and**
  the window peak node density and ω_pe Δt max (the development run passed while both
  were still rising).
- Threshold 3 % (v1.3: 5 %), and the criterion must hold at **two consecutive**
  checkpoints (60 ns apart) after **≥ 5 ion transit times** (12 µs at 2.4 µs per transit).
- Wall budget 14 h per case; a case stopped by the budget without a declared plateau is
  reported as "no plateau" (not resumed for the campaign statistics; it may be resumed
  as development).

### 2.5 Ledger-closure gates (fail-closed at finalization)

- Energy: |cumulative residual| / electrode work ≤ 5 % over the plateau window and
  interval residual RMS ≤ 1e-10 J (development: 4.4 %, 1.5e-11 J).
- Atoms: cumulative closure ≤ 1e-9 of the inventory (development: 7e-15).
- Charge: S · e − (I_beam,i + I_wall,i + I_anode,i) ≤ 5 % of S · e over the window
  (development: 4 %, the growing inventory).
- Poisson true-residual contract 1e-10 |rhs| at every sync (unchanged).

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
| **3.2 secondary electron emission** | the wall electron current equals the wall ion current (3.7 mA each) — SEE at 7–60 eV impact on BN/alumina (yield 0.3–1 at these energies) changes the sheath and the electron energy balance by O(1) | Vaughan or Furman–Pivi yield with a fixed material choice (BN, σ_max ≈ 1.0 at ~300 eV, E_1 ≈ 30 eV), emitted at 2 eV half-Maxwellian; surface-charge ledger consistent; fail-closed at yield ≥ 1 (space-charge-limited emission) |
| **3.3 neutral spatial profile** | 46 % utilisation with a *uniform* n_g is inconsistent: the depletion happens where S peaks (between the last two cusps) and the feed enters at the anode | 1-D axial free-molecular model (anode injection, ionisation sink per column from the MCC tallies, exit effusion), cell-wise n_g(z) in the MCC (null ceiling at the anode value); ledger closure per column; tests: uniform-sink limit recovers the 0-D fixed point |
| **3.4 exit-boundary sensitivity** | the Dirichlet 0 V exit plane fixes φ where the plume should be free; φ_max = 337 V > anode and the returned-electron current (1.8 mA) depend on it | two variants run as development *before* the campaign: (a) exit plane extended by a 6 mm plume region with a Neumann side and Dirichlet far plane; (b) floating exit potential set by zero net current; the campaign freezes one, and reports the other as the sensitivity |
| 3.5 resolvability gate on the instantaneous peak | the development gate did not trip at 3 λ_D per cell | `StabilityLimits` evaluated on the window peak node density at every sync (fail-closed) |
| 3.6 finalize-with-window-maps | `finalize` produces instantaneous maps only | device window accumulators persisted in the checkpoint so a budget-stopped case still has window averages |

Each addition ships with its typed spec (`pic2d-model-v2.0.json`), fail-closed
validation, tests and a development run before the campaign protocol is frozen.

## 4. Campaign matrix

| block | cases | steps (Δt = 1 ps, ≥ 5 transits + plateau) | particles | ms/step (est.) | hours/case |
| --- | --- | --- | --- | --- | --- |
| A. main | 3 seeds × (100 × 800, W = 3e4) | 12–14 M | ~5 M | ~5.5 | 18–21 |
| B. W sensitivity | 1 seed × W = 1.5e4 | 12–14 M | ~10 M | ~8 | 27–31 |
| C. grid refinement | 1 seed × (150 × 1200, W = 3e4) | 12–14 M | ~5 M | ~8 | 27–31 |
| D. exit boundary | 1 seed × alternative exit (from 3.4) | 12–14 M | ~5 M | ~5.5 | 18–21 |

The 14 h wall budget per case (Section 2.4) is therefore **too small at Δt = 1 ps on this
machine**; either Δt = 1.5 ps (0.12 at the peak; 8–9 M steps; blocks A/D 12–14 h,
B/C 18–20 h) or a faster GPU. Decision to be made and frozen in the protocol; the
proposal recommends Δt = 1.5 ps with the peak gate at 0.15.

## 5. Estimated GPU hours (this workstation, WDDM, launch-bound)

| item | hours |
| --- | --- |
| development pair still running (seed-b, w-half) | 7 |
| model v2.0 additions: development runs to validate 3.1–3.4 (4 × ~3 h) plus the two exit-boundary variants (2 × ~4 h) | 20 |
| block A (3 seeds) at Δt = 1.5 ps | 36–42 |
| block B (W/2) | 18–20 |
| block C (grid 20 µm) | 18–20 |
| block D (exit boundary) | 12–14 |
| finalization / dashboard regeneration | 1 |
| **total** | **≈ 110–125 GPU-hours (5–6 days wall, sequential)** |

At Δt = 1.0 ps the total rises to ≈ 170–190 GPU-hours. A datacentre GPU (no WDDM
launch floor, ~4× the throughput at these particle counts) would bring the 1.5 ps
campaign to ~30 GPU-hours.

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
