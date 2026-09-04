# pic2d plume development run v1 — model v2.0 (channel + plume box, cathode region, thrust ledger)

**Status: development / screening. Not preregistered, not validated, not a performance
prediction.** One detached, checkpointed, resumable GPU run of the divergent-exit CFT
channel **plus a 12 × 12 mm plume box** at the v2/v3 operating point (300 V,
0.0186 mg/s) with model v2.0 (`modern/spec/pic2d/pic2d-model-v2.0.json`) on top of v1.4
(wall-ion recycling, peak-node Debye gate, grid-heating triad, CUDA-graph step). Built from
the PIC literature review `modern/docs/literature/pic-mcc-blockers.md`, blocker 4d ("plume
block D").

## What v2.0 adds (and what it declares)

| item | choice | why (review / literature) |
| --- | --- | --- |
| domain | L-shaped plasma region: channel (r ≤ r_wall(z), z < 24 mm) + plume box r ≤ 12 mm, 24 ≤ z ≤ 36 mm; uniform 50 × 50 µm, 240 × 720 cells (77,940 plasma cells, 78,228 unknowns) | R_plume = 12 mm = 4 r_exit = 6 r_bore is the return-yoke radius (thruster envelope); **L_plume = 12 mm = 0.5 L_channel is bounded by the P2 FEM domain** (z ≤ 36.25 mm) — the requested 1–1.5 L_channel needs a new FEM solve (declared deviation). Brandt et al. 2016: 20 × 5 mm box behind a 14 × 1.5 mm channel was "still too small"; hence the charge pile-up gate |
| internal boundaries | channel wall, cone and the front-face flange r ∈ (3, 4.4] mm: dielectric with surface charge; anode 300 V; front face r > 4.4 mm: **grounded conductor** | the pole faces / shield are metal on the cathode/chamber reference in the tested HEMP-Ts (Kornfeld et al. 2007; Koch et al. 2011); the 4.0–4.4 mm gap is closed as dielectric |
| far field | Dirichlet 0 V on r = 12 mm and z = 36 mm; crossings counted as beam by species with axial momentum, angle about the aperture centre (90 bins) and ion energy (IEDF, 256 bins to 1.5 U_a) | chamber / neutraliser reference; box-size dependence declared |
| cathode | **attempt 4:** region r 0.5–2.0 mm, z 24.3–25.0 mm — inside the channel's exit flux tube (every traced sample connects to the bore; runner re-traces at launch, `require_channel_connected_fraction: 1.0`), isotropic 2 eV Maxwellian, **current continuity**: emits the previous interval's discharge current, relaxed over 4 intervals, clamped to [3 mA floor, 15 mA]. Attempt 3 used the off-axis neutraliser annulus r 4.5–6.0 mm, z 26–28 mm (Kornfeld 2007) and did not ignite — see the launch log | electrons follow B; the channel's field lines close on the front face within 1.5 mm of the exit (axis null at 25.45 mm), so only that volume is magnetically connected to the bore (review blocker 4d; Brandt 2016 fed electrons from the outer boundary in a model with Bohm transport). Continuity = review 4d variant (c); charge conservation makes the far field current-free in steady state. Legacy exit-plane injection kept as the A/B option |
| ignition gate | S and N_e trailing 0.15 µs means over their 0.05–0.2 µs reference means: ≥ 0.8 / 1.1 at 0.75 µs and ≥ 1.2 / 1.4 at 1.5 µs, else `stop_reason = no_ignition` | calibrated on v1.3 attempt 2 (ignited: 1.07 / 1.29, 1.41 / 1.76), v1.3 attempt 1 (failed: 0.59 / 1.03) and plume attempt 3 (0.23 / 0.83); a failed gate costs ≤ 40 min |
| neutrals | two-zone: channel 0-D inventory with recycling (v1.4) + analytic free-molecular cosine cone from the aperture in the plume (capped at 0.5 at the lip) as the local MCC factor; ion–neutral collisions OFF | review blocker 3 / 4a; CEX (Miller et al. 2002) deferred as a sensitivity flag |
| thrust | (a) momentum flux through the far field per species + cold-gas effusion; (b) −F_on_thruster from the particle ledger (absorbed momentum − field impulse) and the **Maxwell-stress force** on every solid boundary from the field; closure reported | conservation check, not enforced |
| plume diagnostics | j_i(θ) per steradian, 95 % divergence half-angle, IEDF mean/peak and peak − U_a, self-consistent exit-plane axis potential, acceleration region (90 → 10 % of the axis drop), Isp, anode efficiency | Koch et al. 2011: ion-energy peak ≈ U_a − 15 V is the validation-v1 observable (context, not validation) |
| gates | v1.4 gates over the whole domain + plume-boundary charge pile-up gate (25 % of the peak density after 2.4 µs; **v2.0.2, attempt 9+**: max over resolved far-field nodes of the **trailing 400 000-step window average** \|⟨n_i⟩ − ⟨n_e⟩\| / ⟨n_e,peak⟩ read from the same accumulators as `maps.npz` and the frames, a node resolved at ≥ 64 000 accumulated macro-particle-steps = 32 beam-ion crossings; enforced once the window is complete; the unrestricted window statistic `charge_fraction_of_peak_window_raw` and the single-deposit witness `charge_fraction_of_peak_raw` are recorded. Attempts 7–8 ran v2.0.1: the single-deposit statistic on nodes holding ≥ 32 macro-particles in one deposit — unreachable at far-field densities, so the gate was inert; disclosed) | Brandt 2016 box-size finding; attempt 6 was stopped by one macro-ion on the axis corner node, attempt 7 showed the per-deposit floor cannot be met — see the launch log and `spec/pic2d/pic2d-model-v2.0.json` `gates_v2_0` |
| seed | 5e16 m⁻³, 5 eV in the **channel only**; the plume starts empty | a plume seed would be 4.5 M unphysical macro-particles |

## Cost (measured 2026-09-03, RTX 5090, CUDA-graph step)

* Field map by direct P2 node evaluation: 6 s; channel cross-check vs the qualified bicubic
  max 0.0008 T. Max |B| in the plasma region 0.705 T at the pole faces (ω_ce Δt = 0.186 vs
  the 0.2 gate); channel max 0.291 T.
* Host Schur-complement factorisation of the 241 axial-row blocks (721 × 721): **~5 min per
  launch / resume**, 1.0 GB of inverse blocks on the device.
* **4.2 ms/step at 0.55 M particles**; the 482 sequential row-block matvecs of the direct
  solve set a ~3 ms floor. Projected 5–7 ms/step at 4–6 M particles → the 4 h budget reaches
  2.0–2.8 M steps = 3.0–4.2 µs = 1.0–1.4 ion transits (3.1 µs: 2.4 µs channel residence +
  0.7 µs plume crossing). **The run is expected to stop on the wall budget before a plateau
  can be declared (≥ 3 transits ≈ 8–12 h cumulative); it is resumable.**

## Energy-ledger correction (model v2.0.6, post hoc; recorded values unchanged)

Up to model v2.0.5 the energy ledger's `inelastic_loss_j` lacked the macro weight W (found by the external-validation v0 launch-1 diagnosis, 036bd679), so every recorded interval residual was `H - L_inel` - biased NEGATIVE by the inelastic power - where `H = field work + dU - electrode work` is the true numerical energy creation. The sidecar(s) `ledger-corrected.json` (+ `.sha256.json`) were written by `python -m cft_revival.pic2d.ledger_recompute <results-dir>` from the recorded `series.npz` (corrected residual = H per record; `spec/pic2d/pic2d-model-v2.0.json#gates_v2_0.energy_ledger_correction_v2_0_6`); **the recorded series, maps and summaries are unchanged.** Values below: trailing-400 000-step residual / electrode work at the last record, recorded -> corrected.

| attempt | sidecar | windowed recorded -> corrected | cumulative recorded -> corrected | 5 % gate at checkpoints: recorded -> corrected |
|---|---|---|---|---|
| 3 (no ignition) | `results-attempt3-no-ignition/ledger-corrected.json` | +8.5 % -> +17.0 % | +9.1 % -> +21.8 % | 0.66 -> 0.66 us |
| 4 (neutral crash) | `results-attempt4-neutral-crash/ledger-corrected.json` | +12.8 % -> +24.1 % | +9.8 % -> +21.0 % | 0.72 -> 0.66 us |
| 5 | results untracked (`results-attempt5-stale-scale/` never committed) | - | - | - |
| 6 (gate shot noise) | `results-attempt6-gate-shot-noise/ledger-corrected.json` | -0.6 % -> **+11.0 %** | -3.4 % -> +9.0 % | never -> 0.66 us |
| 7 (budget, no plateau) | `results-attempt7-wall-budget-no-plateau/ledger-corrected.json` | +12.1 % -> +28.1 % | +1.4 % -> +14.6 % | 3.24 -> 0.66 us |
| 8 (heating triad stop) | `results-attempt8-grid-heating-triad-stop/ledger-corrected.json` | +41.7 % -> **+67.3 %** | +8.6 % -> +24.0 % | 3.24 -> 0.66 us |

On the corrected statistic every 50 um plume attempt reads >= +11 % in its FIRST complete 400 000-step window (0.66 us: +111 mW of numerical
heating on 856 mW of electrode power on attempt 6) and attempt 8 never reads below +4.1 %: the flux-tube cathode's dense cold emission cloud on
the 50 um grid was never conservative, and the attempt-8 runaway (recorded +54.8 % per 0.4-us segment, corrected end window +67 %) is its
continuation. The v2.0.3 residual-power gate would have stopped attempts 6-8 at 0.66 us instead of 3.24 us. Attempt 7's development thrust
numbers stay non-quotable. Peak-Debye under the v2.0.6 accumulated floor on the final windows: attempt 7 3.509 -> 3.509 at (6, 437), attempt 8
3.608 -> 3.608 at (14, 285) - both past pi under either floor (the runs themselves ran the v2.0.1 single-step gate at 4.5).

## Commands (from `modern/`)

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
python -m experiments.pic2d_cft_plume_v1.run run       # start / resume (same command)
python -m experiments.pic2d_cft_plume_v1.run status
python -m experiments.pic2d_cft_plume_v1.run finalize  # only for an externally stopped run
```

Detached launch (as v1/v2):

```powershell
$res = "experiments\pic2d_cft_plume_v1\results"; New-Item -ItemType Directory -Force $res | Out-Null
Start-Process python -ArgumentList "-u","-m","experiments.pic2d_cft_plume_v1.run","run" `
    -WorkingDirectory $PWD -WindowStyle Hidden `
    -RedirectStandardOutput "$res\run.log" -RedirectStandardError "$res\run.err"
```

Artifacts as in the steady-state runs (`status.jsonl`, `series.jsonl`, `checkpoint/`,
`summary.json`, `maps.npz`, `series.npz`, `run_state.json`). New in v2.0: status lines carry
`thrust` (flux, cold gas, total, balance, closure, Maxwell-stress force, ledger residual,
next cathode current) and `plume` (charge fraction at the far field, exit-plane axis
potential, acceleration region); `series.npz` carries `momentum_*` and `plume_*` arrays;
`maps.npz` carries the full-domain maps plus `plume_ion_current_per_sr_a`,
`plume_ion_counts_per_theta`, `iedf_ion_counts`, `iedf_edges_ev` and `sample_count_e` (the
electron sample count per node of the window, used by the dashboards' sampling mask);
`summary.json` carries the `plume` block (window-averaged thrust with the closure, divergence
half-angle, IEDF mean / peak / peak − U_a, exit-plane potential, acceleration region, Isp,
anode efficiency) with its claim boundary.

## Claim boundary

Development/screening run; single seed, grid and weight; 12 × 12 mm plume box with a
Dirichlet far field (box-size dependent plume ratios); volumetric cathode without cathode
physics; two-zone neutral closure without CEX; no SEE, no anomalous transport; electrostatic
axisymmetric; 50 µm grid resolving the channel peak at 3–4 cells per λ_D (gated at 4.5).
Thrust, Isp, efficiency, divergence and the IEDF are development numbers. Channel-only
steady-state v3 (model v1.4) is deferred until after this run (devlog).

## Launch log

* **Launch 1 (2026-09-03 21:50 AEST, PID 42400)** — stopped fail-closed at step 9600 (14 ns,
  40 s of stepping after the 5 min factorisation): `neutral density 5.503e19 exceeds the
  null-collision ceiling 5.5e19`. Cause: the ceiling sat AT n_g0 = Q/c while the seeded ions
  (1.7e10, never fed) were being recycled at R = 0.6 → 2.0e16 /s against S ≈ 0.8e16 /s; with
  the artificial relaxation (τ = 30 ns) the inventory tracks the rate-based fixed point
  (Q + R − S)/c = 1.11 Q/c and rising. In steady state R ≤ S keeps n* ≤ Q/c; the overshoot is
  a transient artefact of the development relaxation. Fix (protocol revision): the ceiling is
  2 × n_g0 (1.1e20 m⁻³, null-collision probability 1.3e-4 per step) and the inventory starts
  at the declared `neutral_inventory.initial_density_per_m3` = n_g0; the real collision rate
  is unchanged (scale n_g / ceiling). `neutral_inventory.max_density_over_zero_ionization` in
  `summary.json` records the transient. The launch-1 artifacts were discarded (a different
  protocol hash); launch 2 starts fresh.
* **Launch 2 (22:11 AEST, PID 38948)** — stopped by hand during the host factorisation (no
  stepping) when the user asked for whole-run video frames before the development run; its
  results were discarded (protocol hash changes with the `frame_recorder` block).
* **Launch 3 (22:50 AEST, PID 28860; attempt 3)** — the same protocol plus
  `numerics.frame_recorder`. **No ignition**: stopped by hand at step 760 000 (1.14 µs, 53 min
  of stepping at 4.3 ms/step, 38 frames) and finalised from the checkpoint. N_e 241 k → 172 k,
  S 2.9e15 → 5e14 s⁻¹, I_d → 0, I_beam ≤ 0.2 mA, cathode 3.0 mA steady, n_g 6.3e19 → 5.6e19.
  Diagnosis (field-line tracing on the run's own node field, `cft_revival.pic2d.fieldlines`;
  frames): the channel's exit flux tube closes on the front face within 1.5 mm of the exit
  plane (axis null at z = 25.45 mm) — no plume volume beyond z ≈ 25.7 mm connects to the bore;
  the cathode annulus (r 4.5–6 mm, z 26–28 mm) sat on lines running from the pole face at
  r ≈ 5.2 mm to the far field (0 of 24 samples connected). Electron budget of the last
  interval: 2.85 of the 3.01 mA emitted left through the far field, 0.48 mA hit the body face,
  0.06 mA reached the anode; φ along the cathode lines stayed within ±1.4 V of the 0 V
  reference while the axis sat at 105 V (z = 27 mm) and 239 V (exit). The seed (5e16, 5 eV,
  channel) was present (N_e(0) = 241 k). Artifacts kept as the development record
  (`results-attempt3-no-ignition/`, video in its `video/`).
* **Launch 4 (00:07 AEST 2026-09-04, PID 53756; attempt 4)** — cathode region moved onto the
  channel-connected flux tube (r 0.5–2.0 mm, z 24.3–25.0 mm), launch-time connectivity gate
  (24/24 samples connected) and the ignition gate added; frames ON. **Ignited, then the
  neutral inventory crashed**: S 1.4e16 → 7.3e16 s⁻¹, I_d 1.5 → 7.0 mA (the continuity
  cathode followed it 3.0 → 7.05 mA), T_e 6.7 → 12.9 eV, N_e 248 k → 512 k in 0.39 µs — until
  S reached 1.26 × the feed (utilisation 1.26). The artificial 30 ns relaxation then drove
  n_g toward its *negative* fixed point: 5.5e19 → 4.2e18 within one interval, S collapsed
  ×17 in the next frame, the 7 mA of cathode electrons charged the cone to −70 V (z = 21 mm)
  and the discharge did not recover when n_g refilled (S 3e15 at n_g 5.9e19, T_e 11 eV,
  N_e 484 k). Ignition gate at 0.75 µs: S/S_ref 0.11 (min 0.8), N_e/N_ref 1.75 → stopped
  `no_ignition` at step 520 000 (0.78 µs, 40 min, 26 frames). Kept as
  `results-attempt4-neutral-crash/` (video in its `video/`).
* **Launch 5 (00:52 AEST, PID 40140; attempt 5)** — attempt 4 plus a guard that suspends
  the artificial relaxation whenever S > Q + R (no fixed point). Reproduced the crash
  (n_g 7.8e18 at 0.375 µs, S then stuck at 1.4e16 with n_g back at 5.5e19); stopped by hand
  at 0.56 µs once the root cause was found (`results-attempt5-stale-scale/`, untracked).
  **Root cause (both attempts): the v1.4 CUDA-graph step baked the MCC neutral density into
  the captured kernels as a scalar**, so the MCC ionised at the n_g of the last graph capture
  (a particle-array reallocation), not the inventory's n_g: S grew unchecked by the falling
  inventory (hence S > Q and the negative fixed point), and after the crash the MCC kept the
  trough density (S/(n_e n_g) ×100 low at an unchanged T_e distribution — the frames show
  it). Fixed in `warp_backend` (device-resident density, as the emission rate already was;
  regression test fails on the old backend). Unaffected: every v1.3 record (launched before
  the graph commit) and attempt 3's coupling diagnosis (n_g stayed 5.6–6.3e19 there).
* **Launch 6 (01:35 AEST 2026-09-04, PID 53824; attempt 6)** — attempt 5 plus the graph-safe
  MCC density; frames ON. **Ignited cleanly** (connectivity 24/24; ignition gate 0.75 µs: S ratio
  1.42 / N_e ratio 2.00; 1.5 µs: 1.70 / 3.60; S_ref 3.18e16 s⁻¹) and ran 1 644 000 steps = 2.466 µs
  (0.80 transits) in 2.34 h (5.1 ms/step, 1.62 M e⁻ / 1.64 M Xe⁺, 82 frames), then **stopped
  fail-closed by the plume-boundary gate 66 ns after it armed**: `net charge density at the
  far-field nodes is 0.259 of the peak electron density (limit 0.25)`. State at the stop: I_d
  6.3–6.9 mA (still rising, drift +8.6 %), I_beam 0.4–0.9 mA, S 6.9–7.6e16 s⁻¹ (gross utilisation
  0.77–0.85, net ~0.5), n_g 2.77e19 tracking its fixed point (2.5–2.8e19), peak n_e 2.3e18 at
  3.2–3.6 cells/λ_D (gate 4.5), ω_pe Δt 0.134 (gate 0.2), energy residual −3.4 % of the electrode
  work, neutral ledger closed to 0.05 atoms, thrust 11–18 µN (closure −7 … −180 %, far from a
  plateau), φ_exit(axis) 55–77 V, IEDF peak 145 eV, 95 % half-angle 77°.
  **Diagnosis — the gate observable, not the plasma (numerical):** the gate read
  `max |n_i − n_e|` over the 481 far-field nodes from the *single-step* deposit (`charge_maps()`),
  and the axis corner node (0, 720) of the far plane has a bilinear shape volume
  π Δr² Δz / 6 = 6.5e-14 m³, so ONE macro-ion (W = 6e4) deposited there reads 9.2e17 m⁻³ = 0.39 of
  the peak. The final checkpoint reproduces the trigger exactly: 0.66 macro-ions and 0.00
  macro-electrons on node (0, 720) → 6.06e17 m⁻³ = 0.259 of the 2.34e18 peak; the whole far-field
  boundary held 3 macro-electrons and 145 macro-ions (median 0 per node). Over 1.5–2.4 µs, 3 % of
  the 200-step samples exceeded 0.25 (91 records, max 0.64) with lag-1 autocorrelation 0.84 — the
  q_far log column shows linear ramps of ~10 samples (an ion crossing the last 50 µm cell at
  1.5e4 m/s = 3 ns) ending in a drop when the ion leaves, e.g. 0.054 → 0.087 → 0.122 → 0.147 →
  0.184 → 0.224 → 0.259 (stop). The interval-averaged frames (20 000 steps) give max |n_i − n_e| /
  peak = 0.030–0.033 over the far-field nodes and 1e-4 volume-weighted; the 400 000-step window
  maps 0.040: **no sheath on the box boundary**. The far plane does carry a normal Dirichlet-wall
  sheath (axis φ 44 → 0 V over the last ~0.5 mm ≈ 3.5 λ_D at n 3e16, T_e 11 eV; n_i 1e17 vs n_e
  3e16 on the axis 2 mm before the plane), which is the declared chamber-reference boundary, not a
  charge pile-up. No other gate was near its limit. Fix (v2.0.1, tests
  `test_plume_boundary_gate_ignores_single_macro_particle_shot_noise_on_the_axis_corner_node`):
  the gate reads only far-field nodes holding ≥ `min_macro_particles_per_node` = 32
  macro-particles (electrons + ions, bilinear weights) — the same sample-size floor the peak-node
  Debye gate applies to its argmax; the unrestricted maximum, its node and macro count are
  recorded (`charge_fraction_of_peak_raw`, `far_field_raw_max_node`, `far_field_resolved_nodes`).
  The threshold (0.25) and the arming time (2.4 µs) are unchanged. Kept as
  `results-attempt6-gate-shot-noise/` (video in its `video/`: five MP4s + the HTML player).
  Video check — the **ionisation-rate panel is shot-noise dominated by construction**: in the
  last 30 ns frame the 42 423 nodes with ≥ 20 electron samples hold 35 005 macro-ionisation events
  in total, 72.5 % of them zero (grey), 16 % one or two, none ≥ 20 (max 18); one event on a
  50 µm node is ~1e23 m⁻³s⁻¹ (mid-scale), on an axis node 3e25 (the colour-scale top). The
  dashboard's event-count mask from `6bd5e5b0` is **not** in the video renderer (which masks by
  electron samples only) and at N = 20 would blank the panel at this cadence; the frame's rate
  integrates to S = 7.0e16 s⁻¹ (matches the series), so the data are right and the panel needs
  spatial binning (3 × 3 → 9× events) or a ~10-frame rolling average before it can be read
  quantitatively (proposed, not done).
* **Launch 7 (04:19 AEST 2026-09-04, PID 52176; attempt 7)** — attempt 6 plus the resolved-node
  plume-boundary gate (v2.0.1, commit `45edd30e`); frames ON; fresh start (the gate parameter is
  part of the configuration identity, so the attempt-6 checkpoint cannot be resumed). Same seed →
  the first 1.64 M steps should replay attempt 6 (the gate change is diagnostic-only); the
  ignition verdicts fall at 0.75 µs (~45 min of stepping after the ~5 min factorisation) and
  1.5 µs (~90 min). First readings (04:28 AEST, 42 600 steps = 0.064 µs, 4.2–5.9 ms/step):
  connectivity 24/24, N_e 256 k / N_i 283 k, I_d 1.6 → 2.8 mA, S 1.7 → 2.8e16 s⁻¹, n_g 5.5e19,
  ω_pe Δt 0.074, 0.7 cells/λ_D, `q_far=0.000(raw 0.000/0n)` (max raw 0.045, 0 resolved far-field
  nodes, gate not yet armed). **Bitwise replay of attempt 6 confirmed**: all 213 series records at
  common steps agree exactly in N_e, N_i, φ_max, K_e, I_d and n_g (same seed, order-independent
  fixed-point deposit) — the run is deterministic and the gate change touched no dynamics.
* **Attempt 7 outcome (stopped 08:25 AEST 2026-09-04; finalized 09:23 AEST)** — ran the whole
  4 h budget: **2 520 000 steps = 3.780 µs = 1.22 transits, 126 frames, 14 443 s wall**
  (5.73 ms/step average including the 5 min factorisation; 6.35 → 6.7 ms/step from 2.0 to 2.4 M
  electrons; 4.4 ms/step at 0.35 M), `stop_reason = wall_clock_budget_reached` at the step-2 520 000
  checkpoint (recorded wall 14 443 s > 14 400 s). **The finalization crashed after the stop**:
  `write_final_artifacts` wrote `maps.npz` (last completed 400 000-step window, steps 2.0–2.4 M =
  3.0–3.6 µs), `series.npz` and `checkpoint-final.*`, then `write_canonical_json(summary.json)` raised
  `OrbitValidationError: artifact is not canonical finite JSON` (`run.err`, 08:25:21 AEST) and the
  process exited with `run_state.json` still at its checkpoint state (`finished=false`, no stop
  reason). Cause: `_gpu_utilisation()` returned `float('nan')` whenever `nvidia-smi` failed or
  exceeded its 5 s timeout, and the per-minute samples went verbatim into the canonical summary —
  17 of the 238 calls of this run took ≥ 5 s (the 200-step interval after a log print stretched to
  5.0–5.7 s in `status.jsonl`; the median call cost 2.3 s → 557 s = 3.9 % of the budget spent in
  `nvidia-smi` under GPU contention; attempt 6's 139 samples all returned). Fixed in `3b8b577a`
  (`None` samples, sanitised summary, honest `finalization_error` record on any further failure,
  and `finalize --recover-runner-stop`, a fail-closed rebuild of the summary from the runner's own
  stop artifacts that accepts only an evidenced stop reason). Attempt 7 was finalized that way
  (`--stop-reason wall_clock_budget_reached`; `maps.npz` and `checkpoint-final.npz` byte-identical to
  the runner's sidecars; recorded in `run_state.finalization_recovery` and the second session).
  **Replay of attempt 6**: all 8219 common records agree bitwise in N_e, N_i, φ, kinetic/field
  energies, every current, n_g and the cumulative particle tallies; only the host-summed momentum /
  field-work ledgers differ at 1 ULP (non-associative reductions, diagnostic only).
  **Gate (v2.0.1) after arming at 2.4 µs (4601 records)**: resolved statistic 0.000 throughout — no
  far-field node ever held ≥ 32 macro-particles in a single-step deposit (`far_field_resolved_nodes`
  = 0 in every record); the raw statistic exceeded 0.25 in 35 records (0.8 %, max 0.37), the first
  at 2.466 µs = exactly the attempt-6 stop. The 400 000-step window average gives max |n_i − n_e| /
  peak = 0.035 over the 481 far-field nodes (volume-weighted 1e-4) with 105 of them holding ≥ 32
  electron samples over the window: **no charge pile-up, but the v2.0.1 floor is unreachable on a
  single-step deposit at far-field densities** (1e16–1e17 m⁻³ × 6.5e-14–1e-11 m³ node volumes / W =
  0.01–20 macro-particles per node) — the gate no longer false-fires, but it cannot fire at all.
  Proposed v2.0.2 (not done): read the gate from the window/frame accumulators (interval sample
  count ≥ 32), which is the statistic the diagnosis of attempt 6 actually used.
  **No plateau** (rule: I_d, N_e, n_g drifts < 5 % over the trailing 20 % AND ≥ 3 transits): at
  1.22 transits the trailing-20 % (3.02–3.78 µs, 2521 records) drifts are I_d −13.1 %, N_e +21.8 %,
  n_g −4.5 %; triad: S +15.5 % (soft fail), ω_pe Δt +3.9 %, T_e,dense +3.8 %, cumulative energy
  residual +1.44 % of the electrode work (hard bounds clear). Trailing means ± block standard error
  (60 ns blocks): **I_d 5.99 ± 0.06 mA** (per-record scatter 7.8 % = the shot noise of 190
  macro-electrons per 200-step interval, 7.3 %), cathode 5.99 mA (continuity), **S 8.57 ±
  0.11e16 s⁻¹** (gross utilisation 0.94, net 0.52; 6 % of S in the plume box), **N_e 2.20 ± 0.04 M /
  N_i 2.23 M** (2.44 / 2.47 M at the stop, 20.7 % of the ions in the plume box, still growing
  ~+0.6 M µs⁻¹), **n_g 2.54 ± 0.01e19 m⁻³ on its fixed point 2.54e19**, peak n_e 2.95e18 at
  3.64 cells/λ_D (max 3.92; gate 4.5), ω_pe Δt 0.151 (max 0.182; gate 0.2), T_e,dense 9.3 eV,
  φ_exit(axis) 77 ± 2 V (+23 % over the window). The window maps (3.0–3.6 µs): I_d 6.09 mA,
  I_beam 0.94 mA, body-face ion current 0.82 mA, anode ion current 0.06 mA.
  **Thrust / exit-plane diagnostics — development numbers, NOT plateaued (every beam quantity still
  drifts +20 % per trailing window; quoted with statistical uncertainty only)**: far-field momentum
  flux T_flux 19.4 ± 0.4 µN, cold-gas effusion 1.56 µN → **T_total 20.9 ± 0.4 µN = 0.021 mN**
  (statistical ±2 %; 3.0–3.6 µs window 20.5 ± 0.5 µN); momentum-balance thrust −F_on_thruster
  19.6 ± 0.4 µN (window 20.55 ± 0.30) → window closure −7.9 % (per-record closure scatter ±0.33);
  Maxwell-stress force on the solids 1.74 µN. I_beam (far field) 0.96 ± 0.02 mA (+23 %/window;
  58 555 macro-ion crossings in the map window, Poisson 0.4 %), 0.874 mA through the z = 36 mm
  plane and 0.064 mA through the r = 12 mm side; flux-weighted ⟨v_z⟩ 14.8 km s⁻¹ (149 eV); IEDF
  at the far field: mean 184 eV, peak 133 eV (peak − U_a = −167 V; the far plane sits at 23 V),
  10/50/90 % quantiles 119/168/289 eV; half-angles containing 50/90/95 % of the crossings 8° / 29° /
  60°; Isp 112 s; anode efficiency 0.6 % at 1.83 W. Channel exit plane (z = 24 mm, final checkpoint,
  ±0.5 mm slab, instantaneous): net ion current 2.24 mA (±0.2 % counting; 104 k of 165 k
  macro-ions forward-moving), density-weighted mean ion energy 28 eV (⟨v_z⟩ 1.4 km s⁻¹ — the
  exit-plane population is dominated by slow, locally born ions), axial ion momentum flux 29 µN
  (includes the ions that later hit the front face). Acceleration region 90 → 10 % of the axis
  drop: z = 5.0 → 35.8 mm, i.e. the whole channel plus plume (φ_axis 330 V at 3.75 mm, 81 V at
  25 mm, a 99 V hump at 27 mm next to the cathode region, 67 V at 30 mm, 59 V at 33 mm, 23 V at
  35.9 mm).
  **Plume shape (window-average n_i)**: exit-plane axis 6.4e17 m⁻³ (aperture area mean 2.5e17);
  the axis density RISES to 1.0e18 at z = 27 mm (beam focus ~1.5 mm past the axis null at
  25.45 mm), falls to 50 % of the exit-axis value at z = 32.8 mm and is still 14.6 % at the far
  plane: **neither the 10 % nor the 1 % axial contour fits in the 12 mm box** (the declared
  box-size limit). Radially the beam is narrow: r(10 % of the local axis value) = 1.3–1.9 mm and
  r(1 %) = 2.9–3.9 mm at z = 27–36 mm, consistent with the 8° / 29° half-angles; the 95 % angle of
  60° comes from the slow wide-angle wings (plume-born ions and the front-face population).
  Record: `results-attempt7-wall-budget-no-plateau/` (a copy of `results/`, which keeps the
  checkpoint for the resume; tracked: summary, run_state, maps/series npz, checkpoint-final.json,
  status.jsonl); video (renderer v0.2, `--cusps 0.006028 0.012 0.017972`, auto window K = 10 frames
  = 300 ns, median resolved node 35 events, 6.1 % of the plasma nodes resolved carrying 69 % of S)
  in its `video/`: `pic2d-results-attempt7-wall-budget-no-plateau-{n_e_per_m3,n_i_per_m3,phi_v,
  t_e_ev,ionization_rate_per_m3_s}.mp4` + `…-timeseries.html` (untracked). The n_i frames show
  the narrow beam and a broad low-density population whose front is still filling the box at
  3.78 µs — the plume has not reached its inventory.
* **Launch 8 (09:51 AEST 2026-09-04, PID 51256; attempt 8 = RESUME of attempt 7)** — the same
  command (`run`) continues `results/` from the step-2 520 000 checkpoint (3.780 µs) with the
  cumulative wall budget raised on the CLI to **50 400 s** (`--wall-budget-seconds 50400`, i.e.
  +10 h on top of the 14 443 s already spent; recorded per session in `run_state.sessions[].
  wall_budget_seconds` since `e8b3fb7b`; the protocol's 14 400 s is unchanged); frames ON (the
  recorder continues at frame 126); configuration identity unchanged (protocol untouched), package
  code identity `8e33932e` unchanged (the attempt-7 fixes live in the experiment runner, which is
  outside the checkpoint's code hash). **Deterministic replay verified before the launch**: two
  independent resumes of the checkpoint in scratch copies (400 steps each, `--max-steps 2520400`)
  agree bitwise in the full dynamical state (checkpoint arrays: 2 442 334 electrons, 2 469 710
  ions, φ, surface charge, cumulative ledgers; identical npz hash `7b95f12a…`) and in 142 of the
  150 series fields; the 8 that differ are the peak-node sample at 1 ULP (3.029392693028336e18 vs
  …3366e18), and the 400-step window maps differ only in `sample_count_e` / `t_e_ev` at ≤ 2e-15
  relative — atomic float diagnostics, the same class as the attempt-6/7 ledger ULPs; n_e, n_i, φ
  and the ionisation maps are bitwise equal. (Two concurrent host factorisations oversubscribe the
  BLAS threads — 20 min without finishing — run them one at a time: ~4 min each.) Expectation: the
  plateau rule needs ≥ 3 transits = 9.3 µs → +3.68 M steps at 6.7 → ~9 ms/step (the step cost
  grows with the particle count, 4.4 ms at 0.35 M → 6.7 ms at 2.44 M electrons) ≈ 7.5–9 h, so the
  first plateau verdict can fall from ≈ 17:30 AEST; if the drifts (N_e +22 % per window now, plume
  still filling) are not below 5 % by then the run continues to the budget (≈ 20:00 AEST,
  ≈ 10.3–10.8 µs = 3.3–3.5 transits) and attempt 9 would be another resume. Logs: `results/run.log`
  / `run.err` (attempt 7's are kept as `run-attempt7.log` / `.err` / `.pid` and in the record folder).
* **Model v2.0.2 (code only, 2026-09-04, while attempt 8 runs; not a launch)** — the two attempt-7
  follow-ups. (1) **Plume-boundary gate re-based on the window accumulators.** Attempts 7 and 8 run
  v2.0.1, whose per-deposit floor (≥ 32 macro-particles on a node in ONE deposit) is unreachable at
  far-field densities: after arming, `far_field_resolved_nodes` = 0 in all 4601 records, the gated
  statistic read 0.000 throughout while the raw single-deposit statistic exceeded 0.25 in 0.8 % of
  the records and the true window average was 0.035 — the gate could not fire (**inert; disclosed**,
  attempt 8 keeps the v2.0.1 configuration identity to its end). v2.0.2 reads, at every series
  record (the existing host sync; nothing per step), the far-field rows of the device window sums
  Σ_t n_e, Σ_t n_i — the same accumulation `maps.npz` and the 20 000-step frames use — keeps
  cumulative totals across the runner's 400 000-step accumulator resets (host-side carry keyed on
  the backend's reset generation) and a ring of totals per record, and forms the **trailing window of
  ≥ 400 000 accumulated steps (0.6 µs = the averaging window = 20 frames)** as an exact difference
  of two totals (the frame recorder's construction). Gate quantity: max over *resolved* far-field
  nodes of |⟨n_i⟩ − ⟨n_e⟩| / ⟨n_e,peak⟩ (denominator: the record-mean of the instantaneous peak).
  **Floor: ≥ 64 000 accumulated macro-particle-steps per node** = 32 × 2000: a 15 km/s beam ion stays
  on a 50 µm node for 2000 steps (the ~10 series intervals per corner-node crossing seen in attempt 6),
  so A ≥ 32 τ guarantees ≥ 32 independent beam-ion crossings (the peak-Debye 32-particle convention,
  ≤ 18 % shot noise) for every particle at least as fast as the beam; on the attempt-6/7 window maps
  this resolves 77 / 121 of the 481 far-field nodes (the far plane to r = 3.9 / 6.7 mm; the corner node
  at occupancy 0.08–0.10 stays unresolved and its neighbour (1, 720) carries the statistic) and reads
  **0.0249 / 0.0339 of the peak instead of 0.000** — live, 7× below the threshold; a genuine sheath at
  0.25 puts ≥ 4.4 macro-ions on that neighbour at all times and is resolved. Enforced only when armed
  (2.4 µs, unchanged) AND the window is complete, so the first 0.6 µs after a resume are recorded,
  not gated (the window history is not checkpointed). Records: `charge_fraction_of_peak` (gate),
  `far_field_window_steps/_complete`, `far_field_resolved_nodes`, `charge_fraction_of_peak_window_raw`
  (all far-field nodes) and the v2.0.1 single-deposit witness `charge_fraction_of_peak_raw`; log column
  `q_far=<gate>(w<steps>/<resolved>n raw <window raw> dep <deposit>)`; summary
  `plume.charge_fraction_of_peak_window_raw_max`, `far_field_window_steps_final`,
  `far_field_resolved_nodes_final`. Tests: a sustained far-field pile-up trips it once the window is
  complete (and not before); a corner-node ion sitting for a window or crossing the last cell (the
  attempt-6 mechanism) neither trips it nor counts as evidence while the deposit witness exceeds 0.25;
  an exactly quasi-neutral uniform plume reads 0 with every far-field node resolved and the
  accumulated weights equal to the window maps' sample counts (not inert); the window is continuous
  across the runner's resets (bitwise same-seed comparison) and restarts on `load_state`; CPU/Warp
  parity of the window statistic. (2) **`nvidia-smi` off the stepping thread**: attempt 7 spent 557 s
  = 3.9 % of its wall budget in synchronous per-minute calls (17 of 238 hit the 5 s timeout). The
  sampler is now a daemon thread (`gpu_sampler.GpuUtilisationSampler`) at a configurable cadence
  (`--gpu-sample-interval-seconds`, default 300 s); the loop reads the shared last value
  (`gpu=…%` in the log line), samples stay `float | None`, `summary.gpu_utilisation_sampler` records
  the cadence and outcome counts; test: a query that never returns leaves `latest()` and `stop()`
  sub-millisecond. Also: the launch-time provenance line now prints `step_graph: "lazy"` (graphs are
  captured on the first step; it read `false` before) and `true` once captured. Spec
  `pic2d-model-v2.0.json` `gates_v2_0` (v2.0.2 entry, floor/window justification, version history)
  and `protocol.json` `numerics.plume_boundary_gate` (`window_steps`, `min_accumulated_macro_particles_per_node`)
  updated; the configuration identity changes, so v2.0.2 applies to fresh starts (attempt 9+).
* **Attempt 8 outcome (stopped 11:36 AEST 2026-09-04 by the grid-heating triad gate; finalized
  cleanly)** — **3 320 000 steps = 4.980 µs = 1.61 transits, 166 frames, 20 611 s cumulative wall**
  (6 168 s this session at 7.7 ms/step for +800 000 steps), `stop_reason =
  grid_heating_triad_gate_stopped_run`: "ionisation_rate_drift 0.253 exceeds 0.25". The `3b8b577a`
  finalizer worked (two `null` GPU samples in the summary, no `finalization_error`; `run_state`
  finished = true). Triad at the stop (trailing 20 % = 3.98–4.98 µs): **S drift +0.253 (hard limit
  0.25 — the member that tripped)**, T_e,dense +0.155 (soft fail, hard 0.25), ω_pe Δt +0.048,
  cumulative energy residual / electrode work **+0.0857 (limit 0.10; rising 0.005 per 60 ns →
  would have tripped at ≈ 5.15 µs)**. Trajectory of the S drift at the 40 000-step checkpoints:
  +0.155 at the resume (3.78 µs), a dip to +0.131 at 4.14 µs, then monotonic +0.133 (4.20) → 0.171
  (4.50) → 0.210 (4.74) → 0.253 (4.98 µs); T_e,dense drift −0.02 at 4.2 µs → +0.155.
  **Diagnosis — numerical runaway (finite-grid heating), not a discharge approaching a denser
  plateau.** The energy ledger's residual (`ΔE_total − accounted sources`; positive = energy the
  scheme created) per 0.4 µs segment, as a fraction of the electrode work in the segment: −7.6 %
  (0.4–0.8 µs) … −0.5 % (2.0–2.4) → **+2.4 % (2.4–2.8) → +5.8 → +11.3 → +15.3 → +23.5 → +37.0 →
  +54.8 % (4.8–5.0 µs)**; residual power −9 → +45 → +110 → +203 → +258 → +384 → +545 → **+719 mW**
  against 1.37–1.55 W of electrode power. Attempt 8 alone: +0.558 µJ on 1.844 µJ = **30 %** (attempt
  7 cumulative: +0.081 µJ on 5.61 µJ = 1.4 % — the cumulative ratio lagged the segment ratio by
  ~1 µs, which is why the +1.44 % at 3.78 µs looked healthy). Power budget of the last 0.4 µs:
  electrode +1373 mW, accounted physical sources net **−615 mW** (wall + inelastic losses exceed the
  electrode input), residual **+646 mW**, dE_total +30 mW → **47 % of the discharge's energy budget
  was grid heating.** The sign change sits at 2.0–2.4 µs, exactly when the peak-node Δ/λ_D crossed
  ≈ 3.2 (2.96 → 3.23 → 3.37 → 3.58 → 3.66–3.75 in the following segments), i.e. the Birdsall–Langdon
  CIC finite-grid-instability threshold Δx/λ_D ≈ π; the accepted channel-only base plateau
  (`pic2d_cft_steady_state_v2/results`, 3.2 transits) sat at Δ/λ_D = 3.17 at its peak (1.64e18 m⁻³,
  T_e 7.4 eV, node (14, 286)) with a residual that stayed negative and closed to +0.4 % (last 0.4 µs:
  electrode 1034 mW, residual +4 mW). **The declared 4.5 gate is therefore not protective; the two
  runs bracket the heating onset at Δ/λ_D ≈ 3.2 at the peak.** Corroborating signatures over 3.78 →
  4.98 µs (0.2 µs block means): T_e,dense 9.3 → 10.4 eV (+15.5 %), T_e at the peak 10.2 → 11.4 eV,
  mean electron energy K_e/N_e 13.3 → 14.3 eV **while the electrode power fell 22 % (I_d 5.58 →
  4.40 mA)**; the specific ionisation rate S/N_e stayed constant (3.77e10 s⁻¹ per macro-electron)
  while S/(N_e n_g) rose 20 % — hotter electrons, not more neutrals (n_g fell 17 %). Trajectories:
  N_e 2.51 → 3.13 M, N_i 2.53 → 3.17 M (+22 %/window, unchanged rate); I_d 5.58 → 4.40 mA (−30 %);
  I_beam 1.06 → 1.46 mA (+30 %); **S 9.3 → 11.8e16 s⁻¹ (+25 %, accelerating)**; n_g 2.54 → 2.14e19
  tracking its fixed point within 1 % (the 30 ns relaxation is not lagging — the fixed point itself
  slides down as S runs up: gross utilisation 1.02 → 1.31 > 1, sustained only by wall-ion recycling
  4.3 → 6.2e16 s⁻¹ = 73 % of the feed; effusion 4.1 → 3.5e16); peak n_e 3.08 → 3.54e18 at node
  (13–14, 286–290) = r 0.65–0.70 mm, z 14.3–14.5 mm (between cusps 2 and 3, the same node as the
  base plateau's peak at 2.1× its density; 98.2 % of the 4001 records; the other 72 put the argmax
  at z 20.6–21.2 mm, r 0.55–0.65 mm, and those carry every Δ/λ_D > 4.0 reading, max 4.28); Δ/λ_D
  3.70 → 3.75 at the z ≈ 14.4 mm peak (gate 4.5), ω_pe Δt 0.157 → 0.161 (max 0.184; gate 0.2), λ_D at
  the peak 12.9–13.4 µm. Nothing flattens: I_d, S, n_g and N_e all drift faster than in attempt 7.
  **No plateau criterion was met** (1.61 of 3 transits; trailing drifts I_d −29.8 %, N_e +21.9 %,
  n_g −22.2 %; S +25.3 %, T_total +13.8 %, I_beam +30 %). **Usability: the v2.0.x results carry grid
  heating from ≈ 2.4 µs, above 10 % of the electrode power from ≈ 3.2 µs; nothing after the trip —
  and none of the attempt-8 trailing window — is usable for thrust; the attempt-7 development window
  (3.0–3.6 µs, residual 6–15 % of the power) is contaminated at the 10 % level and stays
  non-quotable.** **Why the plume-domain discharge runs at ~6 mA where the channel-only plateau
  gave 3.44 mA:** the data support the cathode closure, not the neutral inventory or the anode
  fall. The base run injected a fixed 3.00 mA at the exit plane of which 1.84 mA left again through
  that plane (net 1.16 mA into the channel); the v2.0 cathode emits the discharge current itself
  (continuity rule, 4.4–6.3 mA) on the channel flux tube, so the electron supply is uncapped and
  4–5× larger, N_e 1.5–3.1 M vs 1.0 M, S 6.6–12e16 vs 3.9e16 s⁻¹. Before the heating onset (2.0–2.4
  µs, residual −0.5 %) I_d was already 6.0 mA with S 6.6e16 (1.7× base) and N_e 1.46 M, so the ~6 mA
  level is physical and belongs to the closure; n_g was lower, not higher (2.8e19 then, 2.1–2.5e19
  later, vs 2.97e19) and φ_max − U_a is +25 V here vs +40 V in the base run (both runs carry a
  potential hump above the anode; no evidence that a lower anode fall drives the difference). The S
  rise beyond ≈ 7e16 s⁻¹ from 2.4 µs on is partly heating-fed.
  **Development thrust / plume numbers (trailing 20 %, mean ± block SE; NOT usable, see above)**:
  T_flux 23.3 ± 0.3 µN + cold gas 1.47 → T_total 24.8 ± 0.3 µN (+13.8 %/window; window maps 4.38–
  4.98 µs: 25.4 µN); −F_on_thruster 17.1 ± 0.4 µN → **closure +0.24 (window +0.28; attempt 7:
  −0.08)** — the momentum balance no longer closes (stored-momentum rate −6.8 µN, far-field
  electrostatic force −2.2 µN); Maxwell-stress force on the solids 1.5 µN; I_beam 1.29 ± 0.03 mA
  (+30 %; 1.278 mA through z = 36 mm, 0.083 mA through r = 12 mm; 85 002 crossings in the window);
  IEDF mean 143 eV, peak 101 eV, 10/50/90 % 70/117/271 eV (attempt 7: 184/133; 119/168/289 —
  slower); half-angles 50/90/95 % = 8°/23°/55°; Isp 139 s; anode efficiency 1.2 % at 1.43 W;
  φ_exit(axis) 72 ± 1 V (−18 %); acceleration 90 → 10 % z = 4.5 → 35.9 mm; axis n_i 9.6e17 at the
  exit → 1.6e18 at z 26.9 mm → 50 % at 33.65 mm → **26 % of the exit value at the far plane**
  (attempt 7: 14.6 % — the box keeps filling; 10 %/1 % contours still outside). Window far-field
  max |n_i − n_e| / peak = 0.076 over 122 resolved nodes (v2.0.2 statistic computed offline from
  `maps.npz`; the run's v2.0.1 gate stayed inert: 0 resolved nodes in every record, raw max 0.64).
  Record: `results-attempt8-grid-heating-triad-stop/` (copy of `results/`; tracked: summary,
  run_state, maps/series npz, checkpoint-final.json, status.jsonl + sidecars); video (renderer
  v0.2, `--cusps 0.006028 0.012 0.017972`, auto K = 10 frames = 300 ns, median resolved node 38.5
  events, 6.5 % of the plasma nodes resolved carrying 73 % of S, 88 % in the last frame) in its
  `video/`: `pic2d-results-attempt8-grid-heating-triad-stop-{n_e_per_m3,n_i_per_m3,phi_v,t_e_ev,
  ionization_rate_per_m3_s}.mp4` + `…-timeseries.html` (untracked).
  **Resolution decision (v2.1 NOT launched).** From the attempt-8 peak (max record n 3.69e18, T_e
  11.1 eV → λ_D 12.9 µm, ω_pe 1.08e11 s⁻¹; trailing mean 3.33e18 / 10.9 eV → 13.4 µm): the declared
  gate with a 20 % margin (Δ/λ_D ≤ 3.6) only asks Δ ≤ 46 µm — the run already heats at 3.4–3.8, so
  the gate must move to the CIC threshold, Δ/λ_D ≤ π; with 20 % margin (≤ 2.51) **Δ ≤ 32.4 µm**
  (33.7 at the trailing mean); Δ ≤ λ_D would need 12.9 µm. ω_pe Δt ≤ 0.16 (20 % under 0.2) needs
  **Δt ≤ 1.48 ps** (1.5 ps gives 0.163 at the max record — marginal; 1.4 ps → 0.152; ω_ce Δt 0.186 →
  0.174; electron Courant 0.36 → 0.53 at 33 µm). Cost with the v2.1 spec model (ms/step = fixed(grid)
  + 0.733 ms per M particles, fixed = 2(n_r+1) launches × 5 µs + inverse-block reads at 1.6 TB/s +
  node kernels; reproduces the 8.2 / 7.08 ms anchors to 1 %; particles × (50 µm/Δ)² at fixed
  particles per cell from the attempt-8 end load 6.43 M; 3 transits; GPU ≈ 5.4 GB + blocks + 0.35
  GB/M particles): v2.1 48 × 12 mm at 50 µm 9.7 ms/step → 20.5 h, 9.5 GB (heats); **at 40 µm
  (Δ/λ_D 3.10 = π, no margin) 300 × 1200, 15.2 ms → 32.2 h, 3.5 GB of blocks, 12.5 GB; at 33.3 µm
  (Δ/λ_D 2.59) 360 × 1440, 22.4 ms/step → 47.5 h (50.9 h at Δt 1.4 ps), 6.0 GB of blocks, ~61 min
  factorisation, 16.6 GB GPU; at 25 µm (1.94) 480 × 1920, 42.7 ms → 90 h, 14.2 GB of blocks, 28.8
  GB GPU.** v2.0 36 × 12 at 33.3 µm: 18.8 ms → 32.4 h, 13.8 GB. Channel-only 3 × 24 mm (5.1 M
  particles assumed) at 33.3 µm: 90 × 720, 9.8 ms → 13.1 h (14.0 h at 1.4 ps), 9.8 GB; at 25 µm
  17.3 ms → 23.1 h, 13.4 GB. Options: (a) smaller Δt alone does not touch Δ/λ_D (ω_pe Δt is within
  its gate); (b) Δ = 33 µm is the resolved choice, Δ = 40 µm has no margin; (c) lower W changes
  statistics only: the finite-grid instability is a property of the grid aliasing, not of the
  particle count (more particles per cell reduce only the shot-noise part of the heating); (d) an implicit or energy-conserving scheme is **not available** in this code (explicit
  Boris + momentum-conserving bilinear deposit only); (e) a lower operating point keeps the 50 µm
  grid inside the threshold if the peak density stays ≤ 1.4e18 m⁻³ at T_e 10 eV (Δ/λ_D ≤ 2.51),
  e.g. a current-limited cathode (clamp the continuity rule at ~3 mA like the base injection) or a
  lower mass flow — a physical trade, not the v2/v3 operating point. **Every combined (v2.1 domain +
  resolved Δ) option exceeds the 30 h budget, so nothing was launched.** Recommendation: first
  recalibrate the peak-Debye gate (hard Δ/λ_D ≤ π, soft 2.5) and add a windowed residual-power gate
  (segment residual ≥ 5 % of the electrode work over the trailing window → stop; the cumulative
  ratio lags by ~1 µs); then run the cheapest resolved case — the channel-only box at 33 µm / 1.4 ps
  (≈ 6–14 h depending on the particle load) as the grid-refinement check of the accepted 3.44 mA
  plateau —   or the plume box at 50 µm at a lower operating point (option e), whose peak density
  must be verified against the recalibrated gate before any thrust number is read.
* **Model v2.0.3 (code only, 2026-09-04 after attempt 8; not a launch)** — the two recalibrations
  the attempt-8 diagnosis asked for, applied to this protocol (fresh starts only: the gate keys are
  in `config_sha256`, identity `f7a4bedd…` CUDA / `e1377abd…` CPU; the v2.0.2 identities
  `1937f379…` / `4c969bff…` are reproduced from this protocol with the window keys stripped and are
  pinned by test) and to the prepared v2.1 protocol. (1) **Peak-node Debye gate in window mode**
  (`PeakDebyeGateConfig(max_cells_per_debye = π, soft_cells_per_debye = 2.5, window_steps = 400000,
  window_snapshot_steps = 40000)`): the GATED statistic is the interval-averaged peak — the densest
  node of the trailing 400 000-step window (the maps/frames accumulation, read at the series-record
  host sync from the same device sums, six node arrays per record, bridged across the runner's window
  resets, ring of cumulative snapshots every 40 000 steps) among nodes with mean occupancy ≥ 32
  macro-electrons, with the window's moment T_e; hard π (the Birdsall–Langdon CIC threshold, the value
  the attempt-8 ledger identified: residual sign change at Δ/λ_D ≈ 3.2, the base plateau on the
  threshold at 3.17) fail-closed once the window is complete; soft 2.5 (20 % margin) recorded and a
  plateau precondition (`plateau.peak_debye_soft_ok`), never a stop; the single-step sample stays
  recorded as the shot-noise witness (`gate_mode "window"`, `gate_enforced false`). On this 50 µm
  protocol at the attempt-8 peak (3.3–3.7e18, 11 eV) the window statistic reads 3.4–3.8 > π: a fresh
  attempt would now stop at the heating onset (~2.4 µs) by design instead of running into it — a
  resolved plume run needs Δ ≈ 33 µm (cost table above) or a lower operating point. (2) **Windowed
  residual-power gate** (`stopping_rule.grid_heating_triad.residual_window_steps = 400000`,
  `windowed_energy_residual_over_electrode_work_max = 0.05`): the trailing-400 000-step ledger residual
  over the electrode work of the same records, ONE-SIDED (positive = energy the scheme created), stops
  the run from the first complete window; the cumulative ratio is recorded as the witness (its 10 %
  bound stays a plateau precondition). Calibration: attempt 8's per-window ratio crossed +5 % at
  ≈ 3.1 µs (the gate would have stopped it there instead of 4.98 µs, ~1.9 µs before the S-drift member
  and ~2 µs before the cumulative bound); the accepted channel-only runs (v2 base / seed-b / W×0.7)
  read −12.7 % → −0.2 % / −1.5 % / −4.2 % (never above +0.4 %), so the one-sided bound is silent on
  every accepted run while a two-sided 5 % bound would have stopped all three before 4 µs. Spec
  `pic2d-model-v2.0.json` `gates_v2_0` (`peak_debye_gate_v2_0_3`, `windowed_energy_residual_gate_v2_0_3`,
  `gate_recalibration_history_v2_0_3`); tests `tests/pic2d/test_pic2d_v203_gates.py` (9) + parity /
  identity / runner pins; tests/pic2d 207 passed. First use: the channel-only refinement campaign
  `experiments/pic2d_cft_steady_state_v4/` (33.3 µm / 1.4 ps / W 2.667e4, preregistered).

## Time-series frames and video

`numerics.frame_recorder = {cadence_steps: 20000, precision: float32}` records, every 30 ns,
the exact interval averages of the full-domain maps (n_e, n_i, φ, T_e from the moments,
ionisation rate, electron sample counts), the wall / exit / far-field flux profiles, j_i(θ) and
the IEDF of the interval, the instantaneous surface charge and the scalar series record at the
frame end (t, I_d, I_beam, S, N_e, N_i, n_g, thrust flux / balance / total, closure, cathode
current). Frames are computed as differences of the window-accumulator sums (nothing is added
to the step), written one compressed `frames/frame-NNNNNN.npz` each (4.9 MB uncompressed on
the 241 × 721 grid, ≈ 100–140 frames in the 4 h budget, 257 frames = 1.26 GB at 7.7 µs — no
downsampling needed) atomically before the checkpoint that follows; resume removes frames past
the checkpoint; the manifest is hash-bound in `summary.json → artifacts.frames`. Details and
tests: `spec/pic2d/pic2d-model-v2.0.json → frame_recorder_v2_0`.

Render (after the run has written `summary.json`; the run's protocol supplies the cusp planes):

```powershell
python visualization\render_pic2d_video.py experiments\pic2d_cft_plume_v1\results `
    --protocol experiments\pic2d_cft_plume_v1\protocol.json --fps 10
# -> results\video\pic2d-pic2d_cft_plume_v1-<map>.mp4 (one per map; ffmpeg on PATH or
#    imageio_ffmpeg in .venv-sota; GIF through Pillow otherwise) and
#    results\video\pic2d-pic2d_cft_plume_v1-timeseries.html (offline player: scrubber, play,
#    map selector, body + cusp planes, synchronised I_d / I_beam / N_e / n_g / thrust)
```

Colour scales are fixed across all frames (log with a floor of max/10⁴ for the densities; full
range for φ; 0–99.5 % for T_e), cells with < 20 electron samples in the interval are grey, the
thruster body dark.

**Ionisation-rate panel (renderer v0.2, after attempt 6).** A 30 ns frame of the ionisation map
is shot noise by construction: the map is `events · W / (V_node · Δt)` with `events` the
bilinear-deposited weight of the interval's macro-ionisations, and in attempt 6 ~73 % of the
electron-resolved nodes held zero events per frame, 17 % one or two, while one event on an axis
node (V ≈ 1e-13 m³) is ≈ 3e25 m⁻³ s⁻¹ and set the colour top. The data are right (the frame maps
integrate to S = 7.0e16 s⁻¹, the series value); the per-frame picture was not. The renderer now
shows a **causal rolling window** of K frames (windowed events = trailing sum, windowed rate =
windowed events · W / (V_node · windowed time) = the time-weighted mean of the frame rates, which
integrates to the window's mean S; partial window over the first K−1 frames), greys out nodes
with **fewer than 20 windowed macro-ionisation events** (the dashboards' event mask of `6bd5e5b0`,
"unresolved", never zero), and uses a **fixed log10 scale between the 0.5th and 99.5th percentile
of the resolved windowed values over the whole run** instead of the run maximum. K defaults to the
smallest window for which the median electron-resolved node with at least one event weight holds
≥ 10 events (`choose_window`; `--iz-window K` overrides; `--iz-min-events`, `--iz-percentiles`).
Window, mask, scale and the per-frame resolved share of S are written into the panel legend and
the player caption. No spatial smoothing. Attempt 6: K = 11 frames = 330 ns, 5.6 % of the plasma
nodes resolved on average carrying 67 % (77 % at the end) of the window's ionisation, scale
1.6e23–2.9e24 m⁻³ s⁻¹ (1.3 decades). Re-render with a suffix so the earlier files stay:

```powershell
python visualization\render_pic2d_video.py experiments\pic2d_cft_plume_v1\results-attempt6-gate-shot-noise `
    --cusps 0.006028 0.012 0.017972 --fps 10 --suffix=-v2
# (the run's protocol.json has since changed hash, so the cusp planes are given explicitly: the
#  P2 wall cusps of the accepted topology v3.1)
```

The other four panels are unchanged (byte-identical MP4s without the cusp overlay).
