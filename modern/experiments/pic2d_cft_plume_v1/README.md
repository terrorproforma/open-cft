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
| gates | v1.4 gates over the whole domain + plume-boundary charge pile-up gate (25 % of the peak density after 2.4 µs) | Brandt 2016 box-size finding |
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
* **Launch 4 (attempt 4)** — cathode region moved onto the channel-connected flux tube
  (r 0.5–2.0 mm, z 24.3–25.0 mm), launch-time connectivity gate and the ignition gate added
  (see the table); frames ON; this is the plume development run.

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
