# pic2d steady-state v4-fast — preregistered solver qualification: the accepted 33.3 µm plateau replayed under the multigrid field solve (device-mg, 14 cycles) and K = 5 moment sampling

**Status: preregistered solver-qualification replay of a development model. Not a new plateau value,
not a convergence statement, not validated, not a performance prediction.** One detached, checkpointed
run on the Lambda H100 (CUDA MPS client) of the divergent-exit CFT channel at the v2 operating point and
v1.3 closure on the v4 grid — **bit-for-bit the accepted `pic2d_cft_steady_state_v4` protocol** (90 × 720,
Δ 33.33 µm, Δt 1.4 ps, W 26 666.7, seed 20260903, frames ON, v2.0.3 gates, plateau rule) **except two
numerics keys**: `numerics.poisson = {"method": "device-mg", "cycles": 14, "pre_sweeps": 2,
"post_sweeps": 2, "omega": 0.8, "coarsest_max_unknowns": 1024}` (poisson_gmg_v1) and
`numerics.performance.moment_sample_interval = 5` (model v2.0.5). Both enter `config_sha256`:
v4 `f10772b2…` → this run `a6275830…` (a different identity, disclosed; the test pins that the two
`PIC2DConfig.to_dict()` records differ in exactly these two keys).

Why this run exists: `docs/pic2d-performance-audit.md` §9 Class C item 4 — *"Only after [the statistical
replay of the accepted v4 33 µm plateau with the new solver] passes may a preregistration use the solver."*
The multigrid was built, tested and measured (§12; spec `poisson_gmg_v1`): contract on the production
masks with the real v4 maps, one-step φ parity 3.8e-10 V, chaotic divergence record, a 60 000-step
plume-50 same-seed replay inside ±5 % (interval-worst contract ratio 8.2e-5). Model v2.0.5 passed its
Class A / A′ replays (physics bitwise; K = 1 vs 5 T_e-derived statistics ≤ 1.6e-3). What is missing is
this run: the full-protocol replay to 3 transits with acceptance = the recorded seed-b band. A `qualified`
verdict is what lets a later preregistered protocol (the 33 µm plume run) name `device-mg` and K = 5.

## Protocol diff vs v4 (frozen in `protocol.json`)

| key | v4 (`392129e5`, record `0d228ad2`) | **v4-fast (this run)** | in `config_sha256`? |
| --- | --- | --- | --- |
| `numerics.poisson` | `"device block-Thomas direct …"` (string → `PoissonConfig2D(method="device-direct")`, exact factorisation, 0.38 GB inverse blocks, 184 launches / solve) | **object → `device-mg`**: 4-level geometric multigrid (90 × 720 → 46 × 361 → 24 × 181 → 13 × 91; 752 coarsest unknowns inverted densely), operator-dependent transfers, Galerkin coarse operators, fixed 14 V(2,2) damped-Jacobi cycles (ω 0.8), warm start, 278 launches / solve, 23 MB device + 4.5 MB host; **same contract** 1e-10 \|rhs\| on the in-graph true residual every step, `verify()` at every sync, fail-closed | **yes** (`poisson.method`, `poisson.multigrid`) |
| `numerics.performance.moment_sample_interval` | absent (= 1) | **5** (T_e-derived window statistics become 5-step sampled estimators; n_e, φ, fluxes, maps per-step and bitwise K = 1) | **yes** (`moment_sample_interval`) |
| code at the run | `392129e5` (model v2.0.3) | the preregistration commit (model v2.0.5 + poisson_gmg_v1; also v2.0.4 — ω_pe Δt gate on the resolved-node peak, threshold unchanged — and **v2.0.6** — the energy ledger's inelastic loss carries W, so the recorded residual is the true H / electrode work and the 5 % gate reads it; the v2.0.6 *optional* accumulated peak-Debye floor is **not declared**, to keep exactly two identity differences — v4's final window reads 2.154 at the same node under both floors) | no (code identity only) |
| everything else | grid, Δt, W, seed, operating point, v1.3 closure, stability limits, ion subcycle 8, sync / series 200, window 400 000, checkpoint 40 000, frames 20 000, v2.0.3 gates (hard π / soft 2.5 window peak-Debye, 5 % windowed residual power, triad), plateau rule (5 % / 20 % / 3 × 2.4 µs) | **identical** (`test_protocol_is_the_v4_protocol_verbatim_outside_the_declared_fields`) | — |
| `stopping_rule.wall_budget_seconds` | 86 400 (RTX 5090) | **102 100** = 1.5 × the box-measured contended 3-transit estimate (below) | no (stopping rule) |

## Predeclared acceptance (`stopping_rule.acceptance`, evaluated by `run.py assess` against the v4 artifacts on disk)

* **(a) plateau** — `stop_reason == plateau_reached_after_min_transit_times` under the v4 rule (≥ 3 transits;
  trailing-20 % drifts of I_d, N_e, n_g < 5 %; triad soft bounds; peak-Debye soft margin 2.5).
* **(b) residual power — a replay criterion against the v2.0.6-corrected v4 value.** While this experiment
  was being composed, model **v2.0.6** landed (`4b53012d`): every ledger up to v2.0.5 added the inelastic loss
  *without* the macro weight, so every recorded residual read too negative by the inelastic power. The v4 run
  **recorded −7.67 %** at its stop; its post-hoc correction (`results/ledger-corrected.json`, `02013df0`) reads
  **+2.46 %** (28 mW of numerical heating on 1.14 W of electrode power, rising +0.6 % at 0.62 µs → +2.0 % at
  4.82 µs → +2.46 % at 7.28 µs; never near the 5 % hard gate). The replay runs v2.0.6 code, so its series carries
  the corrected statistic natively. (b) therefore reads: **|windowed residual (corrected) − (+2.46 %)| ≤ 1
  percentage point, two-sided** — the replay must reproduce the v4 plateau's heating power; a solver that heats
  *or* cools the discharge by more than 1 pp changed the energy ledger. Band derivation: the 50 µm same-W seed
  pair differs by 1.9 pp at a 12 % heating level (~16 % relative); at the 33 µm level of 2.46 % the seed scatter
  is ~0.4 pp, so 1 pp is ~2.5× the seed-pair spread while a 40 % change of the heating power is still caught.
  The project's plateau acceptance value (< +2 %, kept by `gate_recalibration_v2_0_6`) is **reported, not judged**
  (`project_acceptance_b_below_0p02`): v4 itself fails it at +2.46 %, so a faithful replay is expected to fail it
  too — a statement about the 33 µm grid, not about the solver. The hard 5 % gate (one-sided, corrected
  statistic) stays armed and would end the run (`no_plateau`).
* **(c) replay tolerances vs the v4 plateau** (relative \|fast − v4\| / \|v4\|; the v4 assessment's own
  extraction: `window_currents_a`, `neutral_inventory` trailing means, the densest node of the window maps):

  | quantity | v4 value | seed-b band (same W, other seed) | W×0.7 band | **tolerance** |
  | --- | --- | --- | --- | --- |
  | I_d | 3.801 mA | −0.08 % (window) / −0.9 % (trailing 20 %) | +5.68 % | **≤ 2 %** |
  | I_beam | 2.459 mA | +0.68 % | +3.55 % | **≤ 2 %** |
  | S | 3.595e16 s⁻¹ | −0.80 % | −4.64 % | **≤ 2 %** |
  | utilisation | 0.4204 | −0.80 % | −4.64 % | **≤ 2 %** |
  | n_g | 3.188e19 m⁻³ | +0.73 % | +3.95 % | **≤ 2 %** |
  | peak n_e (window) | 1.287e18 m⁻³ | −8.19 % | −11.89 % | **≤ 10 %** |
  | T_e,peak (window) | 5.577 eV | −1.1 % | −9.3 % | **≤ 3 %** |

  Derivation: the tolerance is the recorded **seed-b** particle-resolution band (the v2 50 µm same-W seed
  pair) rounded up to whole percent with ≥ 2× headroom on the currents / rates (0.7–0.9 % → 2 %), peak n_e
  8.2 % → 10 %, T_e,peak 1.1 % → 3 % (the K = 5 sampling moves T_e-derived statistics by ≤ 1.6e-3 relative).
  Every tolerance lies **below the W×0.7 band** and far below the v4 convergence tolerances (10 / 20 %): a
  pass means the solver + sampling change is indistinguishable from a seed change and well inside a
  particle-weight change. The plume-50 60 000-step replay read I_d +1.0 %, S +1.1 %, n_g −0.11 %, peak n_e
  −0.95 %. The window Δ/λ_D (v4 2.15) is reported beside its v4 value, not judged (it follows from peak n_e and
  T_e,peak).
* **(d) field-solve contract** — the multigrid's 1e-10 \|rhs\| contract (in-graph true residual every step,
  interval maximum read by `verify()` at every 200-step sync) was never missed: the run reached a runner
  terminal state (no `PIC2DConvergenceError` escaped the step loop) **and** `summary.provenance.config.poisson`
  names `device-mg` with 14 cycles / 2+2 sweeps / ω 0.8. Disclosed: the shared runner catches
  `PIC2DStabilityError` only, so a contract miss ends the process *without* a terminal state; the predeclared
  handling is `assess --runner-crash-log <log>` → (d) failed → `not_qualified`. A resume with more cycles would
  be a different identity and is not permitted under this preregistration.
* **(e) verdict** — `qualified` = (a) ∧ (b) ∧ all (c) ∧ (d): poisson_gmg_v1 and K = 5 are admitted to
  preregistered protocols on this grid family (the v4 values stay the quoted plateau; this run is a replay
  record). `not_qualified` = (a) ∧ (b) ∧ (d) with a (c) tolerance exceeded, or (d) failed. `heating` = (a) ∧ ¬(b)
  (the corrected residual power outside the ±1 pp band around v4's +2.46 %, either side — the record says which).
  `no_plateau` = ¬(a) (budget / gate / soft-margin stop / non-ignition).

### Corrected-ledger note (documentary, added 2026-09-05 after the launch; the sealed `protocol.json` is unchanged)

The comparison target now carries a committed post-hoc re-read of its preregistered acceptance on the corrected ledger:
`pic2d_cft_steady_state_v4/results/assessment-corrected-ledger.json` (+ `.sha256.json`; written by
`pic2d_cft_steady_state_v4/assess_corrected_ledger.py`, hash-bound to the sidecar `ledger-corrected.json`, the recorded
`assessment.json`, `summary.json` and `protocol.json`). Its content: v4's recorded verdict `resolution_limited` stands as
recorded; **on the corrected ledger v4's acceptance (b) "< +2 %" FAILS at +2.46 %** (recorded PASS at −7.67 %) and the
predeclared (d) tree gives `refinement_heating`; verdict wording *"plateau reached; convergence vs 50 µm as recorded
(resolution_limited for 50 µm); residual precondition (b) FAILED on the corrected ledger → the 33 µm plateau is itself
heating at +2.5 % of electrode work and is NOT a clean reference; 25 µm (v5) pending"*. Consequences for this campaign,
all consistent with the sealed protocol: (i) (b) is evaluated on the **corrected** statistic — natively here (v2.0.6
code), and `python -m cft_revival.pic2d.ledger_recompute <results> --dry-run` at assess time must report "already
W-scaled record: corrected == recorded" as the ledger cross-check; (ii) `project_acceptance_b_below_0p02` is reported
beside the replay criterion and is expected to read FAIL for a faithful replay, exactly as the target does; (iii) a
`qualified` verdict qualifies the *solver* (device-mg + K = 5 reproduce v4 within the seed-b band) and does **not** upgrade
the 33 µm plateau to a clean (energy-conserving) reference — every value quoted from this replay inherits v4's
disclosure (+2.5 % numerical heating power; not converged; 25 µm pending); (iv) the 5 % hard gate and the project's 2 %
acceptance bound are kept (`gate_recalibration_v2_0_6`). See also
`pic2d_cft_steady_state_v4/NOTE-for-v4-fast-coordinator-corrected-ledger.md`.

## Preflight on the launch box (`preflight.json`, 14:34–14:36 UTC 2026-09-04, H100 80GB HBM3 `GPU-a800b021`, CUDA MPS; non-evidentiary)

Run from a scratch worktree at `e1a24aec` (the pre-commit working tree; `git_head` in the record names it) as
the **third MPS client** beside `ss25-base` (PID 32709) and `sweep-056-launch2` (PID 38282); `sweep-reference`
had just exited; GPU 100 % utilised, 3 425 MiB used before this process. **Both protocols were timed one
after the other under this load** — the honest contended A/B:

| load | **fast (device-mg 14, K = 5)** | v4 (block-Thomas, K = 1) | ratio |
| --- | --- | --- | --- |
| seed (0.55 M e⁻ + 0.65 M Xe⁺), 2 000 steps after 200 warm-up | **11.11 ms/step** | 3.15 ms/step | 3.53 |
| synthetic plateau load 4.5 M particles (2.26 M + 2.26 M) | **13.23 ms/step** | 4.56 ms/step | **2.90** |
| ms per M particles | 0.660 | 0.436 | — |
| solver build / factorisation | 0.6 s (23.2 MB device, 4.5 MB host, 4 levels, 278 launches / solve) | 1.4 s (0.38 GB inverse blocks) | — |
| interval-worst contract ratio over the timed steps | **7.5e-9 / 8.0e-9** (contract 1) | exact | — |
| device pool growth (seed / loaded run) | +0.34 / +0.95 GB | — / +0.37 GB | — |

Reading it honestly: **the fast configuration is 2.9× slower than the v4 solver while contended.** The
multigrid's 278 dependent small launches per solve each wait for an SM share under MPS (38–81 µs vs 3–4 µs
solo, audit §12.3), and K = 5's gain is a solo effect. The audit's solo cost model gives channel-33 multigrid
≈ 1.1 vs block-Thomas 0.97 ms per solve — *not faster on the channel grid, as predicted in §7 / §12* — so this
channel run qualifies the numerics for the **plume** box, where the multigrid is already faster under
contention (v2.1-33: 37.8 vs 40.7 ms/step) and frees 6 GB of inverse blocks and the 18–24 s factorisation.
The solo numbers stay model values: **no solo moment arose** (two preregistered clients ran throughout) and
the requested 3-minute solo probe (block-Thomas vs GMG on channel-33 and plume-v2.1-33) was not run.

Field: direct P2 sample on the 91 × 721 nodes, max \|B\| 0.291 T, source identity `8883f29e…` (the
CPU-derived map hash `abf26c5c…` differs from the Windows anchor, as the cross-platform binding `0ac8d9b8`
predicts). Mesh 45 810 plasma cells / 46 469 unknowns (= v4). A-priori stability gate identical to v4
(ω_pe Δt 0.050, ω_ce Δt 0.072, Δ/λ_D 1.00, Courant 0.50). Expected gate reading at the v4 peak: Δ/λ_D
**2.15** (λ_D 15.5 µm), ω_pe Δt 0.090. Host peak working set 1.07 GB.

Projection: 3 transits (5 142 858 steps) = **18.9 h at the contended plateau-load rate** (15.9 h at the
seed-load rate; 19.1 h to the 7.28 µs v4 verdict time). **Wall budget 102 100 s = 28.4 h = 1.5 × 18.90 h**
(102 086 s rounded up to the next 100 s). The contention will change (sweep-056-launch2 ends in ~6 h,
ss25-base in ~11 h → faster; an external-validation launch 2 may take a freed slot → heavier), so the margin
covers a mix. `preflight.json.protocol_sha256` names the pre-budget protocol (the budget and the cost-model
block were written after the preflight, the only changes); the configuration identity `a6275830…` (which
excludes the stopping rule) is unchanged and is what the lock, the shakedown and the run bind.

## Shakedown on the launch box (`shakedown.json`, `results-shakedown/` not committed; non-evidentiary)

The real protocol with only its cadences shrunk (series 200, checkpoint 4 000, window 40 000, frames 2 000,
Debye window 40 000 / snapshots 4 000, residual window 40 000 — grid, Δt, W, field, seed, **solver, K** and
every gate threshold unchanged, including the final 102 100 s budget) run for 100 000 steps (0.14 µs) from the
scratch worktree on the box as the **third MPS client** (beside PIDs 32709 / 38282) through the full path:
runner → 25 checkpoints → 2 window resets → 50 frames → maps.npz / series.npz / summary.json → ssess
(reference consistency 8/8: the seven plateau quantities against the v4 summary + maps **and** the v4 assessment's
run block, plus the corrected residual against the v2.0.6 sidecar). **Committed record = run 4** (16:03–16:23 UTC,
1 227 s wall, **12.22 ms/step**, git_head 0901138a = the origin head the preregistration commit sits on, i.e.
the v2.0.6 code the execution runs). Runs 1–3 (14:38–15:54 UTC, 11.37 / 11.74 / 11.94 ms/step) ran on e1a24aec
(pre-v2.0.6 code) with earlier revisions of the shakedown *record composition*; all four **replayed the physics
bitwise** — 555 213 e⁻ / 601 315 Xe⁺ at the end, peak window enforced in **301 of 500 records** (1 262 resolved
nodes, max 0.560 cells / λ_D at node (39, 562), soft margin held), windowed residual complete in 280 records,
identical (c) rows to the printed digits — only the float-atomic diagnostics (maps.npz SHA-256, last digits)
differ, the recorded MPS/solo pattern. **What v2.0.6 changed in the record**: the windowed residual of the
seed window reads **+0.06 % of the electrode work** (corrected, native) where runs 1–3 read −12.1 % (the biased
pre-v2.0.6 statistic; v4's own corrected seed window read +0.6 % at 0.62 µs) — the physics is identical, the
ledger is now right. (d) passed (provenance.config.poisson = device-mg / 14 / 2+2 / 0.8, moment_sample_interval
5, config identity 7967ec0… of the shrunk protocol); the K = 5 sampling is visible: the final series record's
window block carries window_moment_samples 8 000 for the 40 000-step window. Assessment 
o_plateau at 0.058
transits with (b) reading −2.4 pp vs the v4 plateau value (the seed window is not the plateau; expected). Early
dynamics follow v4's ignition (I_d 2.0 → 0.8 → 1.3–1.6 mA, I_beam 0.3–0.6 mA, S 1.5–1.9e16 s⁻¹, n_g 5.5 →
4.48e19 by 0.12 µs, single-step peak 0.4–0.7 cells / λ_D, ω_pe Δt 0.025–0.037; v4's own shakedown on the 5090
ended at 555 858 e⁻ / 601 897 Xe⁺ with the same gate counts 301/500 and 280 — the ~1e-3 count difference is the
block-Thomas → multigrid round-off divergence plus the Windows/Linux field-map ULP difference, not a physics
change). The 
esults-shakedown/ directory (139 MB) stays untracked on the box. preflight.json.git_head names
e1a24aec (the preflight was timed before v2.0.6 landed; v2.0.6 adds one W multiplication per sync and no
kernel change, so the timing stands — run 4's 12.22 ms/step vs runs 1–3's 11.4–11.9 is background drift); both
records name the pre-commit working tree, as in v4 / v5 (disclosed non-evidentiary).

## Commands (from `modern/`, `PYTHONPATH=src:.`)

```bash
python -m experiments.pic2d_cft_steady_state_v4_fast.run preflight            # fast AND v4 timed under the same load -> preflight.json
python -m experiments.pic2d_cft_steady_state_v4_fast.run shakedown            # 100k real-input steps through finalize + assess -> shakedown.json
python -m experiments.pic2d_cft_steady_state_v4_fast.run launch --expect-commit <prereg sha> --require-mps
python -m experiments.pic2d_cft_steady_state_v4_fast.run status
python -m experiments.pic2d_cft_steady_state_v4_fast.run assess [--runner-crash-log <log>]   # -> results/assessment.json (verdict a-e)
```

On the box the one execution goes through the scheduler (`tools/cloud/schedule.py launch --only ss33-fast`,
jobs.yaml: detached worktree at the preregistration commit under `<WORK>/jobs/ss33-fast/tree`, tmux,
`CUDA_VISIBLE_DEVICES=0`, MPS client variables, `--expect-commit <sha> --require-mps`). `launch` refuses a
dirty worktree, a HEAD that is not `--expect-commit`, a `protocol.json` that differs from the committed blob,
missing `preflight.json` / `shakedown.json`, a missing MPS pipe (`--require-mps`), any configuration other than
device-mg + K = 5, and an existing `results/execution-lock.json` (O_EXCL). A wall-budget stop may be resumed
with `launch --resume` under the same lock (new session, same identity, recorded in `run_state.sessions`).

## What a `qualified` verdict unlocks

The 33 µm plume run (`pic2d_cft_plume_v2_1`, 360 × 1440, ~9.8 M electrons at the plume load) on the
multigrid: its 6.0 GB of inverse blocks and 18–24 s host factorisation disappear (149 MB of arrays), and the
audit's contended measurement already favours it (37.8 vs 40.7 ms/step). The launch log re-derives its
expected ms/step and hours from this run's measured contended channel-33 rate and the audit's solo model; a
`not_qualified` / `heating` / `no_plateau` verdict keeps every protocol on the block-Thomas solve.

## Claim boundary

A same-seed replay of one accepted development run under two numerics changes; single seed, one solver
configuration, one sampling interval; the criterion is statistical parity against a single-pair
particle-resolution band (the runs diverge chaotically after O(5e3) steps from 1e-10 V field differences).
No new plateau value, no convergence statement, no experimental validation, no thruster performance.

## Launch log

* **Launch 1 (2026-09-04 16:28:46 UTC = 02:28:46 AEST 5 Sep, PID 44430, Lambda H100 80GB HBM3
  `GPU-a800b021`, CUDA MPS, 3 clients)** — the one preregistered execution, through
  `tools/cloud/schedule.py launch --only ss33-fast` (jobs.yaml `6807f041`): detached worktree
  `<WORK>/jobs/ss33-fast/tree` at the preregistration commit **`b09f2b71`** (scheduler prereg check
  `ok` / ancestor of HEAD `6807f041` / protocol `frozen`), `launch --expect-commit b09f2b71… --require-mps`,
  clean worktree attested, protocol SHA-256 `d353baea…`, configuration identity **`a6275830…`** (device-mg
  × 14, K = 5 — recorded in the lock), `results/execution-lock.json` acquired 16:28:47 UTC with **2 other MPS
  clients** (`ss25-base` 32709, `sweep-056-launch2` 38282; the box stays at three PIC clients — `sweep-reference`,
  009, 047 and ext-val v0 had all finished by 14:17 UTC, so no fourth newcomer was displaced), Warp `cuda:0` UUID
  cross-checked against nvidia-smi, tmux `pic-ss33-fast`, `CUDA_VISIBLE_DEVICES=0`, 6 BLAS threads; MPS server
  log: client 44430 ACTIVE, no Xid. Setup: field 2 s, multigrid build < 1 s (no factorisation), graph capture on
  the first step. First readings (0.021 µs, 15 000 steps, 75 records): **12.2–12.3 ms/step at the seed load**
  (0.54 M e⁻ + 0.64 M Xe⁺; the v4 run read 2.50 ms/step here on the solo RTX 5090 and the v4 protocol read
  3.15 ms/step under this very load in the preflight — the contended multigrid penalty as measured), GPU pool
  844 MiB; I_d 2.4 → 0.85 mA and I_beam 0.2 → 0.5 mA (the seed dump), S 1.3–1.7e16 s⁻¹, n_g 5.5 → 4.99e19 falling
  toward its fixed point, single-step peak 0.5–0.7 cells / λ_D, window statistic 0.50 (not yet enforced),
  ω_pe Δt 0.025 — the v4 ignition pattern. **Expectation**: at the contended preflight rates (12.3 seed / 13.2
  plateau) 3 transits (5 142 858 steps; the v4 verdict fell at 5 200 000) take 17.6–19.1 h → **verdict ≈ 10:05–11:35
  UTC 5 Sep (20:05–21:35 AEST)**; `sweep-056-launch2` ends in ~3 h and `ss25-base` in ~8 h, after which the run is
  solo (model ≈ 2.2–2.6 ms/step for this step) and the verdict could fall as early as **≈ 02:30 UTC (12:30 AEST)
  5 Sep**; **budget end (102 100 s of stepping) ≈ 20:50 UTC 5 Sep (06:50 AEST 6 Sep)**. Watch
  `<WORK>/jobs/ss33-fast/run.log`, `…/tree/modern/experiments/pic2d_cft_steady_state_v4_fast/results/status.jsonl`
  (`peak_node.window.cells_per_debye`, `grid_heating_triad.windowed_energy_residual_over_electrode_work` — now the
  corrected statistic — and `plateau`) and `schedule.py status`; the results-only commit (`results/`,
  `assessment.json` with verdict (a)–(e), .gitignore negations) is made from the job worktree after the stop, not
  by the launching agent. If the process ends without a terminal state and `run.log` carries "failed its residual
  contract", run `assess --runner-crash-log <WORK>/jobs/ss33-fast/run.log` (verdict `not_qualified`, (d) failed).
* **Contended vs solo, honestly.** Every number above is a GPU *share*: the fast configuration ran 2.9× slower
  than the v4 solver under the same 3-client load (13.23 vs 4.56 ms/step at the plateau load) because the
  latency-bound multigrid's 278 dependent launches each wait for an SM slot under MPS. The audit's solo model puts
  this step at ≈ 2.2–2.6 ms (multigrid solve ≈ 1.1 vs block-Thomas 0.97 ms — *not* faster on the channel grid,
  as predicted), so the contention factor on this run is ≈ 5–6× (block-Thomas ≈ 2×). **No solo moment arose**
  (two preregistered clients ran throughout the composition window), so the requested 3-minute solo probe of
  block-Thomas vs GMG on channel-33 and plume-v2.1-33 was **not run**; the solo figures stay model values until
  the GPU is idle for ≥ 3 min.
* **What a `qualified` verdict unlocks — the 33 µm plume run re-derived.** `pic2d_cft_plume_v2_1` at 33 µm
  (360 × 1440, ≈ 9.8 M electrons at the plume load, ≈ 7.6 M steps to 3 transits at 1.4 ps): the audit's
  cost model (`0.27 ms + 4.1 µs × launches + 0.30 ms/GB + 0.97 ms per M e⁻`) gives block-Thomas ≈ 17 ms/step
  (724 launches + 6.0 GB of inverse blocks) vs multigrid ≈ 12 ms/step (446 launches, no blocks) before v2.0.5;
  with the v2.0.5 born-ledger fold and K = 5 the per-electron term falls to ≈ 0.4 ms/M (this run's preflight
  measured 0.66 ms/M *contended* vs 0.436 for the v4 code, i.e. the same ordering), so the multigrid step is
  ≈ **6–7 ms/step solo → 13–15 h to 3 transits** (≈ 25 h at the pre-v2.0.5 model, ≈ 45 h on block-Thomas + 6 GB).
  Under MPS contention the same step measured **37.8 ms/step** (audit §12.3, 5 clients; block-Thomas 40.7) and
  this run's ≈ 5× penalty implies ≈ 30–40 ms/step as one of three or four clients → **63–85 h**. Conclusion for
  the scheduler: the 33 µm plume run on the multigrid is a **solo-GPU job (≈ 13–15 h, ≤ 25 h)** or needs its own
  H100; as a 4-slot MPS client it is a 3-day job. A `not_qualified` / `heating` / `no_plateau` verdict keeps every
  protocol on the block-Thomas solve (≈ 45 h solo for that box, 6 GB of inverse blocks, 18–24 s factorisation).
