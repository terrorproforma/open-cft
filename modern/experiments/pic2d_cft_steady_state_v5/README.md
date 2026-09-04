# pic2d steady-state v5 — preregistered 25 µm / 1.0 ps / W 1.5e4 ladder point: is the 33.3 µm (v4) plateau itself converged?

**Status: preregistered resolution-convergence study of a development model. Not validated,
not a performance prediction.** One detached, checkpointed GPU run of the divergent-exit CFT
channel at the v2 operating point and closure (model v1.3: quasi-steady 0-D neutral inventory,
no wall-ion recycling, exit-plane injection 3 mA @ 2 eV, seed 5e16 m⁻³ @ 5 eV) on a grid refined
by 4/3 per axis over the v4 refinement (2 per axis over the base) with the model **v2.0.3 gates**.

Why this rung exists: `pic2d_cft_steady_state_v4` (33.3 µm / 1.4 ps / W 2.667e4, one execution at
`392129e5`, record `0d228ad2`) reached its plateau at 3.03 transits and classified the accepted
50 µm base plateau (`24ab82f4`) **resolution-limited**: I_d 3.801 vs 3.444 mA (+10.35 %, tolerance
10 %), peak n_e 1.287e18 vs 1.637e18 m⁻³ (−21.4 %, tolerance 20 %), T_e at the peak 5.58 vs 7.39 eV
(−24.5 %, tolerance 20 %) at the same location, while S −8.5 %, utilisation −8.5 %, n_g +7.2 % and
I_beam +7.3 % stayed within 10 %; its own windowed residual was −7.7 % (cooling side) and its peak
Δ/λ_D 2.15 (soft 2.5 held). A resolution-limited rung says nothing about the *next* rung: this
experiment asks whether the 33.3 µm values are themselves converged, by the same rule, against the
33.3 µm plateau as the **primary** reference; the 50 µm base is carried as the third ladder column
(same tolerances, reported, **not judged** — the v4 verdict already classified it).

## Design (frozen in `protocol.json`)

| item | v2 base (50 µm) | v4 (33.3 µm, primary reference) | **v5 (this run)** |
| --- | --- | --- | --- |
| domain | channel r ≤ 3 mm, 0 ≤ z ≤ 24 mm (bore 2 mm to 18 mm, linear cone to 3 mm) | identical | identical |
| grid | 60 × 480, Δ 50 µm | 90 × 720, Δ 33.33 µm | **120 × 960, Δr = Δz = 25.00 µm** (3 mm / 120, 24 mm / 960: bore 80, cone start 720, exit 120 — every geometry line on a grid line); 81 480 plasma cells, 82 359 unknowns, 116 281 nodes; plasma volume 3.444e-7 m³ (v4 3.440e-7, v2 3.432e-7: the cone stair-step converging) |
| Δt | 1.5 ps | 1.4 ps | **1.0 ps** (ω_pe Δt 0.064 at the v4 peak, 0.072 at the v2 peak; electron Courant 0.47 cells at 400 eV; ω_ce Δt 0.051 at 0.291 T). Step counts: 200-step interval = 0.2 ns, 40 000-step checkpoint = 40 ns, 400 000-step window = 0.4 µs, 20 000-step frame = 20 ns, one 2.4 µs transit = 2 400 000 steps, 3 transits = 7 200 000 steps |
| macro-weight | 6e4 | 26 666.7 | **15 000 = 6e4 × (25/50)²**: the same macro-particles per cell as both rungs (fixed particle resolution per cell; the total count scales ×4 vs the base: ≈ 3.5 M e⁻ + 3.5 M Xe⁺ at the v4 plateau mean density 1.54e17, 1.15 M per species at the seed) |
| operating point, closure, seed | 300 V / 0 V, n_g0 5.5e19, Q_in 8.551e16 s⁻¹, τ_g 30 ns, injection 3 mA @ 2 eV, seed 5e16 @ 5 eV, RNG seed 20260903 | bit-for-bit | **bit-for-bit** (`operating_point` block copied from v4) |
| gates | v1.3 (no runtime Debye gate) | v2.0.3: window-mode peak-Debye hard π / soft 2.5 on the 400 000-step interval-averaged peak; windowed residual-power ≥ 5 % one-sided; triad drifts | **the same, verbatim** (the step-count cadences of v4 are kept so the runner's window / snapshot / reset mechanics are identical) |
| frames | none | 28 ns | ON, 20 ns (≈ 360 frames to 3 transits, ≈ 3.2 MB/frame uncompressed) |
| plateau rule | drifts of I_d, N_e, n_g < 5 % over the trailing 20 %, ≥ 3 transits (2.4 µs measured v2 ion residence time) | + triad soft bounds + Debye soft margin | the same as v4 |
| expected Δ/λ_D at the peak | 3.17 (its own) | 2.15 (its own) | **1.62 at the v4 peak, 1.58 at the v2 peak**; soft 2.5 needs 2.4× the v4 peak density at fixed T_e |
| wall budget | 12 h | 24 h | set from the measured preflight (below) |

## Predeclared acceptance (`stopping_rule.acceptance`, evaluated by `run.py assess`)

* **(a) plateau**: `stop_reason == plateau_reached_after_min_transit_times` under the rule above.
* **(b) residual power**: trailing-400 000-step ledger residual / electrode work at the stop **< +2 %**
  (one-sided; v4 read −7.7 %, the 50 µm runs +0.4 % / −1.5 % / −4.2 %).
* **(c) convergence vs the 33.3 µm v4 plateau** (relative |v5 − v4| / |v4|): **I_d, S, utilisation, n_g,
  I_beam ≤ 10 %; peak n_e, T_e,peak ≤ 20 %** — the v4 tolerances, so both rungs are judged by one rule.
  The 50 µm base is compared with the same tolerances and reported (`secondary_comparison`), not judged.
* **(d) re-classification of the 33.3 µm result**: `converged` — the 33.3 µm plateau is
  resolution-converged at the 10 / 20 % level and the ladder terminates there (the v4 numbers quotable
  with these bands; the 50 µm base stays resolution-limited); `resolution_limited` — the 33.3 µm plateau
  is *also* resolution-limited, the v5 values supersede the ones that moved, no rung of the ladder is
  converged at this operating point and the declared follow-up is a W-only variant at fixed grid and/or a
  lower operating point (not a fourth rung by default); `refinement_heating` — (a) but not (b);
  `no_plateau` — budget / gate / soft-margin stop, inconclusive.

## Preflight on the real inputs (`preflight.json`, 2026-09-04 19:00 AEST, RTX 5090, CUDA-graph step; non-evidentiary)

* Field: direct P2 sample (`p2-field-authority-v1`, role primary) on the 121 × 961 nodes in 5.1 s,
  SHA-256 `2ff82110…`, max |B| 0.291 T. Mesh: 81 480 plasma cells / 82 359 unknowns / 998 wall nodes,
  plasma volume 3.444e-7 m³. A-priori stability gate at 4e17 / 8 eV: ω_pe Δt 0.036, ω_ce Δt 0.051,
  Δ/λ_D 0.75, Courant 0.47, collision probability 4e-5 — clear.
* Host Schur-complement factorisation of the 121 row blocks (961 × 961): **365 s** per launch (v4:
  95–107 s for 91 × 721²; ×3.2 as expected from rows × n³, on a CPU shared with the other campaign).
* **ms/step (production step, accumulation on) — measured under GPU contention**: **9.71 ms at the
  seed load** (0.99 M e⁻ + 1.15 M Xe⁺) and **17.43 ms at the synthetic plateau load** (4.02 M + 4.02 M,
  seed 1.75e17), 1.34 ms per M particles; 2000 timed steps each after a 200-step warm-up. At the time
  eleven CUDA processes of another campaign (`hybrid_l2_v2`, 11 cases) shared the GPU at 100 %
  utilisation; the `gpu_load_before` nvidia-smi snapshot timed out (5 s) under that load and is
  recorded as `null`. The v4 solo cost was 0.565 ms per M particles, so the contention factor is
  ≈ 2.4 and the numbers above are an upper bound; the solo estimate from the v4-calibrated cost model
  (fixed part 1.86 ms × 1.77 nodes + 0.565 ms/M × 7.1–8.0 M) is 7.3–7.8 ms/step.
* Projection: 3 transits (7 200 000 steps) = **34.9 h at the contended plateau load** (19.4 h at the
  contended seed load); solo ≈ 14.6–15.6 h. The 3.03-transit v4 verdict time (7.28 µs) = 7 280 000
  steps. **Wall budget set to 48 h (172 800 s)** = 1.4× the fully-contended projection, 3.2× the solo
  estimate; the run stops at the plateau, so the budget is a cap.
* Memory: device pool +1.48 GB (seed run) / +2.65 GB (loaded run, 8.0 M particles) of 34.2 GB; host
  peak working set 1.30 GB.
* Expected gate readings on this grid: Δ/λ_D **1.62** at the v4 window peak (1.287e18 m⁻³, 5.58 eV;
  λ_D 15.5 µm) and 1.58 at the v2 peak; ω_pe Δt 0.064 / 0.072.
* The wall budget was written into `protocol.json` after this preflight (the only change), so
  `preflight.json.protocol_sha256` names the pre-budget protocol; the configuration identity
  `config_sha256` (`efb9bb09…`, which excludes the stopping rule) is unchanged and is what the lock,
  the shakedown and the run bind.

## Shakedown (`shakedown.json`, `results-shakedown/` not committed; non-evidentiary)

The real protocol with only its cadences shrunk (series 200, checkpoint 4 000, window 40 000,
frames 2 000, Debye window 40 000 / snapshots 4 000, residual window 40 000 — grid, Δt, W, field,
seed and every gate threshold unchanged, including the final 48 h budget) run for 100 000 steps
(0.10 µs) through the full path: runner → checkpoints (25) → window resets (2) → frames (50) →
`maps.npz` / `series.npz` / `summary.json` → `assess`. 1 099 s wall, **6.98 ms/step** (contention
varied during the run: 4.1–12 ms/step in the log), 981 874 e⁻ / 1 072 331 Xe⁺ at the end. The
window-mode peak-Debye gate was **enforced in 301 of 500 records** (681 resolved nodes, max 0.39
cells per λ_D, soft margin held), the windowed residual completed in 280 records (−11.6 % of the
electrode work: the seed-transient value of every accepted run), `plateau` carries `triad_soft_ok`
and `peak_debye_soft_ok`, and the assessment stage classified the run `no_plateau` (0.042 transits)
with every (c) quantity computed against the pinned 33.3 µm reference (re-derived from the v4
artifacts on disk) and the 50 µm column reported beside it (re-derived from the v2 artifacts).
Early dynamics match the v2 / v4 ignition (I_d 0.7 → 1.3 mA, S 1.5–1.7e16 s⁻¹, n_g 5.5 → 4.6e19 by
0.06 µs, single-step peak 0.3–0.4 cells/λ_D).

## Amendment v5.1 — launch platform moved to the Lambda H100 (2026-09-04 ~22:00 AEST)

**Why.** User directive (21:23 AEST): the full PIC run must execute on the Lambda H100, not on the
local PC. Launch 1 (RTX 5090) was withdrawn at 0.800 µs (launch log below; record
`results-launch1-withdrawn/`, commit `a0235676`). Launch 2 is a **fresh start** on the H100 — one
execution on one GPU model for the record; the launch-1 checkpoint is history and is not resumed
across GPU models (a cross-platform resume would also be a `numerical`, not a bitwise, continuation).

**What changed in `protocol.json`** (block `amendments[0]`, `launch_platform`): the wall budget
(`stopping_rule.wall_budget_seconds` 172 800 → **117 000 s = 32.5 h**, re-derived below, original
derivation kept in the note), `preregistration.one_execution` (the withdrawn launch and the fresh
launch recorded), and the new `launch_platform` block (GPU model, MPS sharing, scheduler job,
withdrawn-launch record). **Nothing else**: grid, Δt, W, operating point, closure, seed, gates,
cadences, plateau rule, acceptance (a)–(d), tolerances, references and claim boundary are untouched
— the configuration identity `efb9bb09…` (which excludes the stopping rule) is byte-for-byte the
preregistered one, and `tests/pic2d/test_pic2d_steady_state_v5.py` still pins it (5/5). The GPU
model was never a declared parameter (the 5090 appears only in the cost model and the budget
derivation); the physics state replays bitwise under MPS (mini-sweep `mps-replay.json`).

### Preflight on the launch box (`preflight.json`, 11:37 UTC, H100 80GB HBM3, CUDA MPS; non-evidentiary)

Run from a scratch worktree at `a0235676` as the **4th MPS client** beside the three running
mini-sweep processes (PIDs 19764 reference / 20079 design 047 / 20189 design 009; design 056 had
ended on its triad gate at 10:52 UTC and freed its slot; GPU 100 % utilised, 3 819 MiB used before
this process). It replaces the RTX 5090 record of 19:00 AEST (kept above for comparison).

* Field: direct P2 sample on the 121 × 961 nodes in 4.0 s, max |B| 0.291 T (the CPU-derived map
  hash differs from the Windows anchor — `8098cab8…` vs `2ff82110…` — as the cross-platform
  binding `0ac8d9b8` predicts; the source identity is what the checkpoints bind). Mesh 81 480 plasma
  cells / 82 359 unknowns, unchanged. Host factorisation **3.1 s** (365 s on the Windows PC).
* **ms/step under MPS-4 (production step, accumulation on, 2 000 timed steps after a 200-step
  warm-up): 6.99 ms at the seed load** (0.99 M e⁻ + 1.15 M Xe⁺) and **10.82 ms at the synthetic
  8.0 M-particle plateau load** (4.02 M + 4.02 M), 0.668 ms per M particles. For comparison the
  withdrawn 5090 launch ran 8.8–9.5 ms/step solo at 1.3 M + 1.3 M, so the contended H100 slot is at
  least as fast as the solo 5090 for this grid (latency-bound small kernels; see the H100 benchmark).
* Projection: 3 transits (7 200 000 steps) = **21.6 h at the contended plateau load** (14.0 h at the
  seed load; 21.9 h to the 7.28 µs v4 verdict time). **Wall budget 117 000 s = 32.5 h = 1.5× the
  contended plateau-load projection.** The contention will change during the run (the sweep
  processes end over the next ~12 h → faster; the external-validation v0 job takes a freed slot →
  a heavier 4th client), so the margin covers a mix, not a fixed rate; the run stops at the plateau,
  so the budget is a cap. Disclosed cost to the sweep: the 4th MPS client shares the saturated
  aggregate (N=4 1.54×, N=8 1.58× of one process), i.e. the three running sweep jobs lose roughly
  a quarter of their per-process speed while this run is the 4th client — the same 4-slot design
  the sweep was launched under.
* Memory: device pool +1.48 GB (seed run) / +2.66 GB (loaded run) of 79 GiB; host peak working set
  1.49 GB. Expected gate readings unchanged (Δ/λ_D 1.62 at the v4 peak, ω_pe Δt 0.064).
* The a-priori stability gate is identical to the 5090 preflight (ω_pe Δt 0.036, ω_ce Δt 0.051,
  Δ/λ_D 0.75, Courant 0.47).

### Shakedown on the launch box (`shakedown.json`, 11:44–11:57 UTC, `a529b457`; non-evidentiary)

The same shrunk-cadence protocol as the 5090 shakedown (with the amended 117 000 s budget in the
stopping rule) run for 100 000 steps from the scratch worktree at the amendment commit, through the
full path: runner → 25 checkpoints → 2 window resets → 50 frames → `maps.npz` / `series.npz` /
`summary.json` → `assess` (reference consistency 7/7 primary, 7/7 secondary). It started as the
**6th GPU client** (three sweep runs + the external-validation v0 shakedown PID 26944, which ended
during the run + this) and finished 787 s later at **7.78 ms/step** (6.9–7.7 in the log at 0.96–
0.98 M e⁻; 10.4 in one interval while the ext-val shakedown ran), 982 052 e⁻ / 1 072 788 Xe⁺ at the
end. Window-mode peak-Debye gate **enforced in 301 of 500 records** (687 resolved nodes, max 0.39
cells per λ_D, soft margin held), windowed residual complete in 280 records (−11.7 % of the
electrode work), assessment `no_plateau` (0.042 transits) with every (c) quantity computed against
the pinned 33.3 µm reference and the 50 µm column reported. Early dynamics as on the 5090 (I_d 0.6 →
1.4 mA, S 1.5–1.8e16 s⁻¹, n_g 5.5 → 4.5e19 by 0.1 µs). Compared with the 5090 shakedown (981 874 e⁻ /
1 072 331 Xe⁺, 301/500, 0.39, −11.6 %): the same gate counts and readings to the stated digits, with
particle counts differing at the 2e-4 level — the P2 field map is CPU-derived and differs at ULP level
between Linux/OpenBLAS and Windows (`0ac8d9b8`), so a cross-platform bitwise replay is not expected
and not claimed; the physics is platform-consistent to the diagnostic precision that matters here.
The `results-shakedown/` directory stays untracked (scratch worktree on the box).

### Code version at the launch (`amendments[1]`) and the re-run shakedown

Between `a529b457` and the launch the branch received **model v2.0.4** (`79e6a670`, another agent):
the runtime ω_pe Δt fail-closed gate (threshold 0.2, unchanged) reads the single-step peak over
*resolved* nodes (electron deposit ≥ the 32-macro-particle floor) and records the raw single-node
peak as a witness; `simulation.py` / `warp_backend.py` changed (one reduction slot, graph-safe),
physics untouched. The branch is linear, so the launch commit contains it; the protocol's declared
gates are unchanged (v2.0.3 window-mode peak-Debye + windowed residual-power), the configuration
identity is unchanged, and launch 1 had read the raw statistic at 0.06–0.08 anyway. So that the
shakedown exercises the code the execution runs, the H100 shakedown was **re-run at the launch
commit** and `shakedown.json` replaced again (numbers in the launch log below and in the record).

## Commands (from `modern/`)

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
python -m experiments.pic2d_cft_steady_state_v5.run preflight       # -> preflight.json (non-evidentiary)
python -m experiments.pic2d_cft_steady_state_v5.run shakedown       # -> shakedown.json, results-shakedown/ (non-evidentiary)
python -m experiments.pic2d_cft_steady_state_v5.run launch --expect-commit <prereg sha>   # clean worktree + exclusive lock + run
python -m experiments.pic2d_cft_steady_state_v5.run status
python -m experiments.pic2d_cft_steady_state_v5.run assess          # -> results/assessment.json (verdict a-d, 33 um primary, 50 um reported)
```

Detached launch from the dedicated run worktree (`uni-project/.worktrees/pic2d-ss5`, checked out
detached at the preregistration commit):

```powershell
$res = "experiments\pic2d_cft_steady_state_v5\results"; New-Item -ItemType Directory -Force $res | Out-Null
Start-Process python -ArgumentList "-u","-m","experiments.pic2d_cft_steady_state_v5.run","launch","--expect-commit","<prereg sha>" `
    -WorkingDirectory $PWD -WindowStyle Hidden -RedirectStandardOutput "$res\run.log" -RedirectStandardError "$res\run.err"
```

`launch` refuses a dirty worktree, a HEAD that is not `--expect-commit`, a `protocol.json` that
differs from the committed blob, missing `preflight.json` / `shakedown.json`, and an existing
`results/execution-lock.json` (O_EXCL). A wall-budget stop may be resumed with `launch --resume`
under the same lock (new session, same identity, recorded in `run_state.sessions`); nothing else
changes after the freeze. The stages are the v4 discipline with this experiment's paths, reference
and assessment schema (`pic2d_cft_steady_state_v4/run.py` is frozen with its executed run and is
imported only for its host helpers).

## Claim boundary

A grid-refinement ladder point of a development PIC-MCC model under its declared closure: single
seed, one grid and one weight (grid and particle-weight effects entangled at the ~5 % level); the
artificial 30 ns neutral relaxation (only the fixed point is physical); no ion–neutral collisions,
no SEE, no anomalous transport, Dirichlet exit, prescribed B, electrostatic axisymmetric. The
outcome classifies the 33.3 µm plateau as resolution-converged or resolution-limited at the
predeclared tolerances and nothing more; no experimental validation, no thruster performance.

## Launch log

* **Launch 1 (2026-09-04 19:29:53 AEST = 09:29:53 UTC, PID 43572)** — the one preregistered
  execution, from the dedicated run worktree `uni-project/.worktrees/pic2d-ss5` checked out detached
  at the preregistration commit `69ff435d` (= origin/feat/sota-foundation head at the launch),
  `launch --expect-commit 69ff435d`: clean worktree attested, protocol SHA-256 `2e81659f…`,
  configuration identity `efb9bb09…` (warp-cuda), `results/execution-lock.json` acquired at
  09:29:54 UTC. GPU **shared** at the launch: the eleven `hybrid_l2_v2` CUDA processes of another
  campaign (launched 18:13 AEST, base case at ~7 400 / 12 000 steps and 0.67 s/step at 19:40) were
  still running; the local run is not slowed permanently, only while they last. Setup ≈ 7 min (field
  5 s, factorisation ≈ 6 min, graph capture on the first step). First readings (0.032 µs, 32 000
  steps, 160 records): **9.2–9.8 ms/step under the contention, 3.97 ms/step in the last interval**
  (seed load 0.96 M e⁻ + 1.13 M Xe⁺); I_d 2.4 → 0.73 mA and I_beam 0.14 → 0.72 mA (the seed dump),
  S 1.4–1.7e16 s⁻¹, n_g 5.5 → 4.83e19 heading for its fixed point (the v2 / v4 ignition pattern),
  single-step peak 0.34–0.50 cells/λ_D, window statistic 0.36–0.38 (not yet enforced, window
  400 000), ω_pe Δt 0.068–0.073. Expectation: at the solo cost (7.3–7.8 ms/step at the plateau load)
  3 transits (7 200 000 steps) fall at ≈ 14.6–15.6 h of stepping → **the first plateau verdict can
  fall from ≈ 10:15–11:15 AEST 2026-09-05 (00:15–01:15 UTC)** if the GPU frees within the first
  hours; at the fully contended preflight cost (17.4 ms/step) ≈ 34.9 h → ≈ 06:30 AEST 2026-09-06;
  **budget end (48 h of stepping) ≈ 19:40 AEST 2026-09-06 (09:40 UTC)**. Watch `results/status.jsonl`
  (`peak_node.window.cells_per_debye`, `grid_heating_triad.windowed_energy_residual_over_electrode_work`,
  `plateau`) and the PID; the results-only commit (results/, `assessment.json` with the 33 µm primary
  and 50 µm secondary columns, .gitignore negations) follows the stop and is not made by the
  launching agent.
* **Launch 1 WITHDRAWN by the user at 2026-09-04 21:26:48 AEST (11:26:48 UTC) — compute moved to
  the cloud (Lambda H100).** User directive (21:23 AEST): the full PIC run belongs on the Lambda
  H100, not on the local PC, which the run made unusable. The shared runner has no clean-stop
  channel (no STOP file, no flag, and Windows has no SIGTERM handler to deliver), so the process
  was ended with `Stop-Process 43572` **at a checkpoint boundary**: the watcher waited for
  `run_state.json` to report `checkpoint_step 800000` (written 11:26:48.849 UTC, i.e. after
  `save_checkpoint_atomic` had completed) and issued the stop 12 ms later. **Last checkpoint: step
  800 000 / t = 0.800 µs** (`checkpoint-latest.json` step 800000, 1 297 563 e⁻ / 1 324 061 Xe⁺,
  arrays SHA-256 `5e978213…` 106 729 656 B, field anchor `c14d313b…`; both sidecars re-verified
  against the bytes after the stop); one further 200-step series record (step 800 200) was written
  before the process died and is not covered by the checkpoint. Stepping wall 6 600 s (1.83 h;
  1.96 h since the lock), 4 001 status records, 40 frames (0–0.80 µs, 20 ns cadence). **No terminal
  state**: `run_state.json` stays `finished: false` with no `stop_reason`; no `summary.json`, no
  `maps.npz`, no assessment. The `finalize --recover-runner-stop` path was deliberately NOT called
  — it accepts only evidenced stop reasons (`wall_clock_budget_reached`, plateau) and none applies;
  nothing here is a result and nothing here is a failure of the protocol, the code or the physics.
  Readings at the stop (v2 / v4 ignition pattern, healthy): I_d 1.1–1.5 mA, I_beam 0.3–0.5 mA, S
  1.5–1.9e16 s⁻¹, n_g 4.44e19 (5.5 → 4.44e19, still falling toward its fixed point), gross
  utilisation 0.19, single-step peak n_e 6.4–6.9e17 at 1.0–1.1 cells/λ_D, **window statistic
  0.70 cells/λ_D (enforced, 11 125 resolved nodes, window peak 3.71e17 at 8.5 eV, node (21, 571))**,
  windowed residual −14.2 % of the electrode work (cooling side; cumulative −13.5 %), ω_pe Δt
  0.061–0.083. Cost: 8.25 ms/step mean over the run, 8.8–9.5 ms/step over the last 2 000 records
  at 1.30 M e⁻ + 1.32 M Xe⁺ with the GPU 100 % ours after the hybrid-l2 processes were stopped
  (~20:5x AEST). Record: the results directory of this launch is archived in the run worktree as
  **`results-launch1-withdrawn/`** (renamed from `results/` after the stop, bytes unchanged, so that
  `results/` stays free for the H100 execution) and tracked in this commit: `execution-lock.json`,
  `run_state.json` (+ sidecar), `checkpoint/checkpoint-latest.json` (+ the `.json`, `.npz` and
  `.field.npz` sha256 sidecars), `status.jsonl`, `series.jsonl`, `run.log`, `run.err`, `run.pid`;
  untracked: `checkpoint/checkpoint-latest.npz` (107 MB), `checkpoint/checkpoint-latest.field.npz`
  (1.9 MB), `frames/` (40 × 1.6 MB). The checkpoint is history only: launch 2 is a FRESH start on
  the H100 (one execution on one GPU model for the record; no cross-GPU resume). Local GPU after
  the stop: `nvidia-smi --query-compute-apps` lists no python / Warp process (only desktop
  applications hold memory).
