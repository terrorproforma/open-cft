# pic2d steady-state v4 — preregistered grid-refinement check of the v2 base plateau (33.3 µm / 1.4 ps / W 2.667e4, v2.0.3 gates)

**Status: preregistered resolution-convergence study of a development model. Not validated,
not a performance prediction.** One detached, checkpointed GPU run of the divergent-exit CFT
channel at the v2 operating point and closure (model v1.3: quasi-steady 0-D neutral inventory,
no wall-ion recycling, exit-plane injection 3 mA @ 2 eV, seed 5e16 m⁻³ @ 5 eV) on a grid refined
by 1.5 per axis with the model **v2.0.3 gates** (`modern/spec/pic2d/pic2d-model-v2.0.json`
`gates_v2_0`). Its object is the accepted 50 µm plateau of `pic2d_cft_steady_state_v2/results`
(commit `24ab82f4`: I_d 3.44 mA, S 3.93e16 s⁻¹, utilisation 0.46, n_g 2.97e19, peak n_e 1.64e18 at
T_e 7.4 eV, 3.2 transits), which sits at **Δ/λ_D = 3.17 at its peak — on the Birdsall–Langdon CIC
finite-grid-instability threshold π** that plume attempt 8 crossed into runaway
(`pic2d_cft_plume_v1/README.md`, attempt-8 entry). `pic2d_cft_steady_state_v3/` (model v1.4
closure, prepared, never launched) is unrelated to this study and untouched.

## Design (frozen in `protocol.json`)

| item | v2 base (reference) | v4 (this run) |
| --- | --- | --- |
| domain | channel r ≤ 3 mm, 0 ≤ z ≤ 24 mm (bore 2 mm to 18 mm, linear cone to 3 mm) | identical |
| grid | 60 × 480 cells, Δr = Δz = 50 µm | **90 × 720 cells, Δr = Δz = 33.33 µm** (3 mm / 90, 24 mm / 720: bore 60, cone start 540, exit 90 — every geometry line on a grid line); 45 810 plasma cells, 46 469 unknowns, 65 611 nodes |
| Δt | 1.5 ps | **1.4 ps** (ω_pe Δt 0.101 at the v2 peak; electron Courant 0.50 cells at 400 eV; ω_ce Δt 0.072). 200-step interval = 0.28 ns, 40 000-step checkpoint = 56 ns, 400 000-step window = 0.56 µs, one 2.4 µs transit = 1 714 286 steps |
| macro-weight | 6e4 | **26 666.7 = 6e4 / 2.25**: the same macro-particles per cell as the base (grid-refinement at fixed particle resolution); ~2.25 M e⁻ + 2.25 M Xe⁺ at the v2 plateau density, 645 k per species at the seed |
| operating point, closure, seed | 300 V / 0 V, n_g0 5.5e19, Q_in 8.551e16 s⁻¹ (0.0186 mg/s), τ_g 30 ns, injection 3 mA @ 2 eV, seed 5e16 @ 5 eV, RNG seed 20260903 | bit-for-bit identical (`operating_point` block copied; `neutral_inventory` without recycling) |
| gates | ω_pe Δt 0.2, Courant, Poisson contract, inventory bounds; no runtime Debye gate (v1.3) | the same plus **v2.0.3**: window-mode peak-Debye gate hard **π** / soft **2.5** on the 400 000-step interval-averaged peak (snapshots every 40 000 steps, floor 32 macro-electrons mean occupancy); windowed residual-power gate **≥ 5 %** of the electrode work over the trailing 400 000 steps (one-sided); triad drifts soft 0.05 / hard 0.25 after one transit; cumulative residual recorded (witness, 10 % soft) |
| frames | none | ON: 20 000-step (28 ns) interval-average frames, ~1.8 MB/frame uncompressed (≈ 260 frames to 3 transits) — the first frames of a channel-only steady-state run |
| plateau rule | drifts of I_d, N_e, n_g < 5 % over the trailing 20 %, ≥ 3 transits | the same **plus** triad soft bounds and the Debye soft margin (`plateau.peak_debye_soft_ok`) |
| wall budget | 12 h | **24 h (86 400 s)** cumulative, resumable |
| expected Δ/λ_D at the peak | 3.17 (window maps) | **2.11** at the v2 peak (λ_D 15.8 µm); soft 2.5 is reached only if the resolved peak densifies by 40 % at fixed T_e |

## Predeclared acceptance (`stopping_rule.acceptance`, evaluated by `run.py assess`)

* **(a) plateau**: `stop_reason == plateau_reached_after_min_transit_times` under the rule above.
* **(b) residual power**: the trailing-400 000-step ledger residual / electrode work at the stop
  **< +2 %** (heating side; the accepted 50 µm runs read −0.2 % / −1.5 % / −4.2 %, the negative side
  is reported, not judged).
* **(c) convergence vs the 50 µm base** (relative |v4 − v2| / |v2| of the trailing/window
  quantities): **I_d ≤ 10 %, S ≤ 10 %, utilisation ≤ 10 %, n_g ≤ 10 %, I_beam ≤ 10 %, peak n_e ≤ 20 %,
  T_e at the peak ≤ 20 %**. Tolerances = 2× the particle-resolution band of the 50 µm convergence
  pair: seed-b (different RNG seed) moved I_d −0.1 % (window) / −0.9 % (trailing), S −0.8 %, n_g
  +0.7 %, peak n_e −8.2 %, T_e,peak −1.1 %; W×0.7 moved I_d +5.7 %, S −4.6 %, n_g +4.0 %, peak n_e
  −11.9 %, T_e,peak −9.3 %. Because the refinement keeps particles per cell (W ÷ 2.25), grid and
  particle-weight effects are entangled at that level; a W-only variant at 50 µm is the declared
  follow-up if (c) fails.
* **(d) re-classification of the 50 µm result** (one of four recorded outcomes):
  `converged` — (a), (b) and every (c) hold: the v2 plateau is resolution-converged at the 10 / 20 %
  level; `resolution_limited` — (a), (b) hold but a (c) tolerance is exceeded: the v2 plateau is
  **resolution-limited** (a valid recorded outcome; the v4 values supersede the ones that moved);
  `refinement_heating` — (a) but not (b): the 33 µm run heats too, the comparison is not a
  convergence test; `no_plateau` — budget / gate / soft-margin stop: inconclusive at 33 µm within
  24 h, reported with the stop reason and trailing drifts.

## Preflight on the real inputs (`preflight.json`, 2026-09-04, RTX 5090, CUDA-graph step; non-evidentiary)

* Field: direct P2 sample (`p2-field-authority-v1`, role primary) on the 91 × 721 nodes in 2.7 s,
  SHA-256 `c201eb0c…`, max |B| 0.291 T. Mesh: 45 810 plasma cells / 46 469 unknowns / 748 wall
  nodes, plasma volume 3.440e-7 m³ (v2: 3.432e-7; the cone stair-step at the finer resolution).
  A-priori stability gate at 4e17 / 8 eV: ω_pe Δt 0.050, ω_ce Δt 0.072, Δ/λ_D 1.00, Courant 0.50,
  collision probability 6e-5 — clear.
* Host Schur-complement factorisation of the 91 row blocks (721 × 721): **95–107 s** per launch.
* **ms/step (production step, accumulation on)**: **2.54 ms at the seed load** (0.55 M e⁻ + 0.65 M
  Xe⁺) and **4.36 ms at the synthetic plateau load** (2.26 M + 2.26 M, seed 1.75e17), 0.565 ms per
  M particles; 2000 timed steps each after a 200-step warm-up.
* Projection: 3 transits (5 142 858 steps) = **6.2 h** at the plateau load (3.6 h at the seed load);
  the v2 verdict time 7.68 µs (5 485 714 steps) = 6.6 h. The 24 h budget is 3.9× the 3-transit
  time (the attempt-8 cost table had projected 9.8 ms/step and 13–14 h; the channel-only grid has
  91 rows, not 241, so the fixed cost is smaller).
* Memory: device pool +0.74 GB (seed run) / +1.38 GB (loaded run) of 34.2 GB (plus the CUDA
  context and graphs); host peak working set 0.73 GB.
* Window-mode gate live on CUDA at production scale: at 2200 accumulated steps 154 resolved nodes,
  window Δ/λ_D 0.49 (single-step witness 0.49) at the seed.

## Shakedown (`shakedown.json`, `results-shakedown/` not committed; non-evidentiary)

The real protocol with only its cadences shrunk (series 200, checkpoint 4 000, window 40 000,
frames 2 000, Debye window 40 000 / snapshots 4 000, residual window 40 000 — grid, Δt, W, field,
seed and every gate threshold unchanged) run for 100 000 steps (0.14 µs) through the full path:
runner → checkpoints (25) → window resets (2) → frames (50) → `maps.npz` / `series.npz` /
`summary.json` → `assess`. 362 s wall, **2.61 ms/step**, 555 858 e⁻ / 601 897 Xe⁺ at the end.
The window-mode peak-Debye gate was **enforced in 301 of 500 records** (1261 resolved nodes, max
0.58 cells per λ_D, soft margin held), the windowed residual completed in 280 records (−12 % of
the electrode work: the seed-transient value of every accepted run; the hard bound is one-sided),
`plateau` carries `triad_soft_ok` and `peak_debye_soft_ok`, and the assessment stage classified
the run `no_plateau` (0.058 transits) with every (c) quantity computed against the pinned
reference, which it re-derived from the v2 artifacts on disk (consistency check). Early dynamics
match the v2 ignition (I_d 0.7 → 1.3 mA, S 1.5–1.7e16 s⁻¹, n_g 5.5 → 4.46e19 by 0.13 µs).

## Energy-ledger correction (model v2.0.6, post hoc; recorded values unchanged)

Up to model v2.0.5 the energy ledger's `inelastic_loss_j` lacked the macro weight W (found by the external-validation v0 launch-1 diagnosis, 036bd679), so every recorded interval residual was `H - L_inel` - biased NEGATIVE by the inelastic power - where `H = field work + dU - electrode work` is the true numerical energy creation. The sidecar(s) `ledger-corrected.json` (+ `.sha256.json`) were written by `python -m cft_revival.pic2d.ledger_recompute <results-dir>` from the recorded `series.npz` (corrected residual = H per record; `spec/pic2d/pic2d-model-v2.0.json#gates_v2_0.energy_ledger_correction_v2_0_6`); **the recorded series, maps and summaries are unchanged.** Values below: trailing-400 000-step residual / electrode work at the last record, recorded -> corrected.

`results/ledger-corrected.json`: **-7.67 % -> +2.46 %** at the stop (cumulative -9.1 % -> +1.8 %); trajectory +0.6 % (0.62 us) -> +1.0 %
(2.0 us) -> +2.0 % (4.82 us) -> +2.46 % (7.28 us), i.e. 2.2 -> 28 mW of numerical heating power against 0.38 -> 1.14 W of electrode power;
maximum over complete windows +2.46 %; cross-check against the final counts exact to 7.8e-5 (the classical-vs-relativistic threshold
bookkeeping). **Acceptance (b) "windowed residual < +2 %" changes status: recorded PASS -> corrected FAIL.** (The diagnosis' end-state estimate
of ~+1.9 % used the final-record S and I_d; the exact window recomputation gives +2.46 %.) (a) plateau and (c) convergence tolerances are
untouched and the verdict `resolution_limited` (about the 50 um base, which itself reads +13.0 % corrected) stands; the 33 um plateau carries
2.5 % numerical heating power, still rising slowly at the stop, to be disclosed with every quoted value - whether it falls at 25 um is what
`pic2d_cft_steady_state_v5` measures. The hard 5 % residual-power gate never fires on the corrected statistic (2x margin). Peak-Debye under the
v2.0.6 accumulated-particle-step floor (64 000 macro-electron-steps): 2.154 at the same node (20, 429); resolved nodes 19 650 -> 42 130; the
densest axis node (0.38 macro-electrons per step) is resolved and reads 0.79.

## Commands (from `modern/`)

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
python -m experiments.pic2d_cft_steady_state_v4.run preflight       # -> preflight.json (non-evidentiary)
python -m experiments.pic2d_cft_steady_state_v4.run shakedown       # -> shakedown.json, results-shakedown/ (non-evidentiary)
python -m experiments.pic2d_cft_steady_state_v4.run launch --expect-commit <prereg sha>   # clean worktree + exclusive lock + run
python -m experiments.pic2d_cft_steady_state_v4.run status
python -m experiments.pic2d_cft_steady_state_v4.run assess          # -> results/assessment.json (verdict a-d)
```

Detached launch from the dedicated worktree (`uni-project-pic2d-ss3`, checked out at the
preregistration commit):

```powershell
$res = "experiments\pic2d_cft_steady_state_v4\results"; New-Item -ItemType Directory -Force $res | Out-Null
Start-Process python -ArgumentList "-u","-m","experiments.pic2d_cft_steady_state_v4.run","launch","--expect-commit","<prereg sha>" `
    -WorkingDirectory $PWD -WindowStyle Hidden -RedirectStandardOutput "$res\run.log" -RedirectStandardError "$res\run.err"
```

`launch` refuses a dirty worktree, a HEAD that is not `--expect-commit`, a `protocol.json` that
differs from the committed blob, missing `preflight.json` / `shakedown.json`, and an existing
`results/execution-lock.json` (O_EXCL, canonical JSON: commit, protocol and configuration hashes,
host, PID, command, UTC). A wall-budget stop may be resumed with `launch --resume` under the same
lock (new session, same identity, recorded in `run_state.sessions`); nothing else changes after
the freeze. Artifacts as in the v2 run (`status.jsonl`, `series.jsonl`, `checkpoint/`,
`summary.json`, `maps.npz`, `series.npz`, `run_state.json`, `frames/`) plus `assessment.json`;
status lines carry `peak_node.window` (the gated statistic) and
`grid_heating_triad.windowed_energy_residual_over_electrode_work`; the log line shows
`win=<Δ/λ_D>(w<steps>)` and `res_w=<%>`.

## Claim boundary

A grid-refinement check of a development PIC-MCC model under its declared closure: single seed,
one refined grid and one weight; the artificial 30 ns neutral relaxation (only the fixed point is
physical); no ion–neutral collisions, no SEE, no anomalous transport, Dirichlet exit, prescribed
B, electrostatic axisymmetric. The outcome classifies the 50 µm plateau as resolution-converged
or resolution-limited at the predeclared tolerances and nothing more; no experimental validation,
no thruster performance.

## Launch log

* **Launch 1 (2026-09-04 13:11:55 AEST = 03:11:55 UTC, PID 18068)** — the one preregistered
  execution, from the dedicated worktree `uni-project-pic2d-ss3` checked out detached at the
  preregistration commit `392129e5` (origin/feat/sota-foundation head at the launch; the gates
  commit is `ceb9b172`), `launch --expect-commit 392129e5`: clean worktree attested, protocol
  SHA-256 `82d3f281…`, configuration identity `f10772b2…` (warp-cuda), `results/execution-lock.json`
  acquired at 03:11:56 UTC; GPU free before the launch (no compute process, 5.5 GB of desktop apps).
  Setup ≈ 2.5 min (field 3 s, factorisation ≈ 100 s, graph capture on the first step). First readings
  (0.086 µs, 61 200 steps, 306 records): **2.50 ms/step mean** (median 2.47) at the seed load
  (0.55 M e⁻ + 0.61 M Xe⁺), GPU 99 %; I_d 1.25 → 1.46 mA, S 1.4–1.7e16 s⁻¹, n_g 5.5 → 4.54e19 heading
  for its fixed point (the v2 attempt-2 ignition pattern), single-step peak 0.45–0.56 cells/λ_D,
  window statistic 0.52 over 61 200 steps (191 resolved nodes; not yet enforced, window 400 000),
  windowed residual −8.3 % of the electrode work (window incomplete; the seed-transient value of
  every accepted run), ω_pe Δt 0.085–0.089. Expectation from the preflight: the step cost rises to
  ≈ 4.4 ms at the plateau load, so 3 transits (5 142 858 steps) fall at ≈ 5.5–6.2 h of stepping →
  **the first plateau verdict can fall from ≈ 18:45–19:30 AEST (08:45–09:30 UTC)**; the v2 verdict
  time (3.2 transits, 5 485 714 steps) ≈ 19:15–20:00 AEST; **budget end (24 h of stepping) ≈ 13:15
  AEST 2026-09-05 (03:15 UTC)**. Watch `results/status.jsonl` (`peak_node.window.cells_per_debye`,
  `grid_heating_triad.windowed_energy_residual_over_electrode_work`, `plateau`) and the PID; the
  results-only commit (results/, `assessment.json`, .gitignore negations) follows the stop and is
  not made by the launching agent.
* **Finish (2026-09-04 18:13:59 AEST = 08:13:59 UTC, PID 18068, one session)** — stop
  `plateau_reached_after_min_transit_times` at step **5 200 000 = 7.280 µs = 3.033 transits**
  (transit = the protocol's 2.4 µs, the measured v2 ion residence time kept in `budget_v1_3`; ≥ 3
  required = 5 142 858 steps; the rule is evaluated at the 40 000-step checkpoints and the trailing
  drifts first satisfied it at 5.2 M). 18 013 s of stepping (5.00 h; **3.46 ms/step mean**, 2.50 →
  4.1–4.2 ms/step as the particle count grew to 1.99 M e⁻ + 2.00 M Xe⁺), 260 frames (28 ns),
  finalizer clean (no `finalization_error`; 8/60 nvidia-smi samples timed out, NaN-safe), GPU 99 %.
  Trailing-20 % drifts: I_d +3.0 %, N_e +4.9 %, n_g −0.5 % (threshold 5 %); triad soft members
  S +0.6 %, T_e,dense −2.0 %, ω_pe Δt +1.5 %; **`plateau.peak_debye_soft_ok` true**: window
  Δ/λ_D **2.15** at the stop (trailing mean 2.10, 0 of the enforced records above the soft 2.5;
  single-step witness 2.37, run maximum 2.46; hard π never approached; λ_D 15.5 µm at the window
  peak); windowed residual **−7.7 %** of the electrode work (cumulative −9.1 %; cooling side, as
  every accepted 50 µm run); ω_pe Δt 0.096. Last records: I_d 3.7–4.0 mA, I_beam 2.1–2.7 mA,
  S 3.5–3.9e16 s⁻¹, n_g 3.19e19, utilisation 0.41–0.44.
* **Assessment (`results/assessment.json`, `run.py assess`, 08:21 UTC): verdict
  `resolution_limited`.** (a) plateau ✓. (b) windowed residual −7.67 % < +2 % ✓. (c) vs the 50 µm
  base (`24ab82f4`; reference consistency re-derived from the v2 artifacts on disk 7/7):
  I_d **3.801 vs 3.444 mA, +10.35 % (tolerance 10 %) ✗**; I_beam 2.459 vs 2.291 mA, +7.3 % ✓;
  S 3.595e16 vs 3.930e16 s⁻¹, −8.5 % ✓; utilisation 0.420 vs 0.460, −8.5 % ✓; n_g 3.188e19 vs
  2.973e19, +7.2 % ✓; peak n_e **1.287e18 vs 1.637e18 m⁻³, −21.4 % (tolerance 20 %) ✗**; T_e at
  the peak **5.58 vs 7.39 eV, −24.5 % (tolerance 20 %) ✗** — at the same location (window peak
  node (20, 429) = r 0.67 mm, z 14.3 mm; v2 node (14, 286) = r 0.70 mm, z 14.3 mm). Bands of the
  50 µm convergence pair for comparison: seed-b I_d −0.1 %, I_beam +0.7 %, S −0.8 %, n_g +0.7 %,
  peak n_e −8.2 %, T_e,peak −1.1 %; W×0.7 I_d +5.7 %, I_beam +3.6 %, S −4.6 %, n_g +4.0 %, peak n_e
  −11.9 %, T_e,peak −9.3 %. (d) **The 50 µm base plateau is RESOLUTION-LIMITED** at the predeclared
  tolerances: the v4 values supersede the v2 values for I_d, peak n_e and T_e,peak; S, utilisation,
  n_g and I_beam agree within 10 %. Reading (not part of the predeclared rule): the refined run's
  peak is cooler and less dense at the same place while its residual is on the cooling side — the
  50 µm base sat at Δ/λ_D 3.17, where attempt 8 located the CIC heating onset; the v4 shifts point
  in the direction of the W×0.7 shifts at about twice their size, so grid and particle-weight
  effects remain entangled (W changed with the grid), and the protocol's declared W-only variant
  at 50 µm stays open. Whether 33 µm is itself converged is the object of the next ladder point
  (25 µm / 1.0 ps / W 1.5e4, `pic2d_cft_steady_state_v5`). Results-only commit: `results/`
  (summary, assessment, execution lock, run state, status/series/maps, final checkpoint metadata),
  the .gitignore negations; frames (241 MB), the checkpoint arrays, `series.jsonl` and the logs
  stay untracked in the run worktree.
