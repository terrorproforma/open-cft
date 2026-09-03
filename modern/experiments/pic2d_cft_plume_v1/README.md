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
| gates | v1.4 gates over the whole domain + plume-boundary charge pile-up gate (25 % of the peak density after 2.4 µs, read on far-field nodes holding ≥ 32 macro-particles in the deposit — attempt 7+; the unrestricted single-deposit maximum is recorded as `charge_fraction_of_peak_raw`) | Brandt 2016 box-size finding; the sample-size floor mirrors the peak-node Debye gate (attempt 6 was stopped by one macro-ion on the axis corner node, see the launch log) |
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

Colour scales are fixed across all frames (log with a floor of max/10⁴ for densities and the
ionisation rate; full range for φ; 0–99.5 % for T_e), cells with < 20 electron samples in the
interval are grey, the thruster body dark.
