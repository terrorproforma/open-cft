# PIC-2D anomalous cross-field transport v1 — the preregistered Bohm α-series (roadmap R1, model v2.1.0)

**Status: PREREGISTERED `057841cf`; AMENDMENT 1 recorded (§7, §9); launch 1 (α = 1/16) RECORDED as `results/alpha-1over16/` —
the discharge EXTINGUISHED under the closure (drift-member triad stop at 1.00 transit; `no_plateau`; not relaunched); launches of
`alpha-1over64` → `alpha-0.345` follow under the amendment.** Three one-shot executions of the reference design at 33 µm with the
Bohm-type anomalous transport closure at α ∈ {1/64, 1/16, 0.345}; the α = 0 point of the series is the RECORDED ss-v4 plateau
(`pic2d_cft_steady_state_v4/results`, `0d228ad2`), which is not re-run.

## 1. Why

The physics completeness audit (`modern/docs/pic2d-physics-completeness-audit.md`, `0901138a`, §4.c) ranks anomalous cross-field
electron transport as the structural gap of the 2D axisymmetric model: the E × B drift instabilities that carry it in the real
device cannot exist in (r, z), every HEMPT PIC we could read imposes a Bohm-type closure (Brandt et al. 2016: D⊥ = 0.4 kT_e/eB
"derived from a 3D simulation of a similar thruster model"), and our closure without it produced an ionisation avalanche at Brandt's
operating point (ext-val v0 launch 1). Roadmap R1 is the predeclared α-series on the accepted 33 µm plateau, then the sealed ext-val
`bohm-0.4` run.

## 2. What the hook did, what it does now (the audit of R1, model v2.1.0)

The v1.4 hook (`cft_revival.pic2d.sensitivity`, `warp_backend.bohm_kernel`) already applied the right event statistics: an exact
Poisson probability `1 − exp(−α ω_ce Δt)` per electron per step at the electron's local |B| (bilinear gather, so the probability
vanishes at the nulls and peaks at the cusps), a speed-preserving event (kinetic energy conserved to round-off — no ledger energy
term is owed, the event is elastic), the count tallied in `cumulative["anomalous"]` and the axial momentum handed to the
"turbulent field" in `pz_collisions`, drawn from its own seed stream after the push and before the MCC. It is **not** part of the MCC
null-collision budget: it is a separate exact-Poisson process, statistically equivalent to a joint budget to
O(ν_an Δt × ν_mcc Δt) ≈ 1e-7 (ν_an Δt ≤ 0.026 at 0.29 T / 1.4 ps / α 0.345; ν_mcc Δt ~ 1e-5), which is what keeps the α = 0 path
bitwise identical to the model without the hook.

What it lacked was the reference's **event model**: the v1.4 event redirected the velocity isotropically, which also randomises the
parallel speed — a pitch-angle scattering that feeds the cusp loss cones directly. Brandt et al. 2016 (p. Pb_237) rotate "only the
component of the velocity vector perpendicular to the local magnetic field direction … to ensure that the speed of the electrons
along the magnetic field lines does not change". Model v2.1.0 adds that event (`bohm_perpendicular_rotation`: a Rodrigues rotation
about the local B by a uniform random angle; |v| and v∥ unchanged to round-off, gyro-centre shifted by 2 r_L sin(φ/2)) in both
backends, selected by `numerics.anomalous_collisions.model`; the isotropic model stays as the recorded v1.4 default, so every recorded
identity is unchanged.

Both events give the same cross-field diffusion coefficient for a Maxwellian, D⊥ = (kT_e/eB) · α/(1 + α²) (Green-Kubo: the velocity
autocorrelation of a gyrating electron whose gyro-phase is reset at Poisson rate ν is ⟨v_x²⟩ e^{−νt} cos ω_ce t). The diffusion test
(`tests/pic2d/test_pic2d_v210_anomalous_transport.py`: 24 000–60 000 test electrons gyrating in a uniform 0.05 T with no E, the hook
applied per step, MSD fitted over 20–30 collision times) reproduces it within 5 % for α = 1/16 and 0.345 under BOTH models, and shows
the 0.345 case sits 10.6 % BELOW the naive α kT_e/eB — the audit's exact factor for Brandt's ν = 0.4 ω_ce.

## 3. Design (frozen in `protocols/<case>.json`; the campaign protocol is `protocol.json`)

| block | value | source |
|---|---|---|
| template | ss-v4 protocol: divergent-exit-stack, 90 × 720 cells (33.33 µm), Δt 1.4 ps, W 26 666.7, v1.3 closure (0-D inventory, no recycling), seed 20260903, frames ON (20 000-step cadence) | `pic2d_cft_steady_state_v4/protocol.json`, byte-for-byte except the rows below |
| closure | `bohm_perpendicular_rotation`, ν_an = α ω_ce; α = 1/64 (`alpha-1over64`), 1/16 (`alpha-1over16`), 0.345 (`alpha-0.345`) | audit §4.c; D⊥/(kT_e/eB) = 0.0156 / 0.0623 / 0.308 |
| α = 0 | the recorded ss-v4 plateau: I_d 3.801 mA, I_beam 2.459 mA, S 3.595e16 /s, utilisation 0.420, n_g 3.188e19, peak n_e 1.287e18 at node (20, 429), T_e,peak 5.58 eV, Δ/λ_D 2.15 | `reference_run` (= v5's pinned block; `assess` refuses if the artifacts disagree) |
| α = 0 caveat | its acceptance (b) FAILS on the corrected ledger: +2.46 % of the electrode work in the trailing 400k window (recorded −7.67 % before the v2.0.6 W fix) | `results/ledger-corrected.json` (`02013df0`) |
| gates | v2.0.3 window-mode peak-Debye gate (hard π, soft 2.5) with the v2.0.6 accumulated-particle-step floor 64 000; windowed residual-power gate 5 % on the v2.0.6 W-corrected ledger; triad drift members; runtime ω_pe Δt (v2.0.4 resolved statistic) | thresholds byte-for-byte v4's; the floor and the ledger fix are v2.0.6 |
| gates, AMENDMENT 1 (§9) | triad drift members armed by the model v2.1.1 **settled-once latch** (`grid_heating_triad.drift_members_arming`: ≥ 2.0 transits AND the trailing-20 % I_d drift has read < 0.05 at a 40 000-step checkpoint) instead of at 1.0 transit; the v2.0 **ignition gate** (`stopping_rule.ignition_gate`: N_e/N_ref ≥ 0.6 and S/S_ref ≥ 0.3 at 1.0 µs, ≥ 0.6 / ≥ 0.4 at 2.0 µs; reference window 0.05–0.2 µs) stops an extinguished discharge | stopping-rule keys outside `config_sha256`; identities unchanged; the residual-power and peak-Debye gates untouched |
| diagnostics | K = 5 electron-moment sampling (v2.0.5; physics bitwise, enters `config_sha256`) | `8aca6c3a` |
| budget | 1.5 × the launch-box plateau-load preflight rate × 5 142 858 steps (3 transits), per case (`preflight-<case>.json` → `compose --budget-from-preflight`) | §5 |
| identities (warp-cuda) | α = 0: `f10772b25b03…` (= v4, test-pinned); 1/64 `28ca0391fdb0`; 1/16 `90cf53f1aef1`; 0.345 `8ea882736213` | `config_sha256` |

## 4. Predeclared acceptance and hypotheses (`stopping_rule.acceptance`; `run.py assess`)

Per case: (a) the v4 plateau rule (≥ 3 transits, trailing-20 % drifts of I_d, N_e, n_g < 5 %, triad soft bounds, peak-Debye soft
margin 2.5); (b) corrected windowed residual < +2 % (one-sided). Verdict `plateau_clean` (a ∧ b), `plateau_heating` (a ∧ ¬b — like the
α = 0 reference), `no_plateau`.

Shift table against α = 0 for I_d, I_beam, S, utilisation, n_g, peak n_e, T_e,peak with the recorded particle band (the 50 µm pair's
seed-b / W×0.7 bands: 5.7 / 5.7 / 4.6 / 4.6 / 4.0 / 11.9 / 9.3 %); a shift is CONFIRMING when it has the declared sign and exceeds the
band, CONTRADICTING when it has the opposite sign and exceeds the band, INSIDE THE BAND otherwise. Per-cusp report (planes 6.028 /
12.000 / 17.972 mm, ±1 mm): electron and ion wall current, axis-to-wall potential drop, near-wall T_e, beside the v4 values.

Hypotheses (audit §4.c / R1 — the SIGN is the hypothesis, the magnitude is the measurement):

| quantity | sign | expected at α = 1/16 |
|---|---|---|
| I_d | + | +20 to +60 % |
| S, utilisation | − | −10 to −40 % |
| n_g | + | +5 to +25 % |
| peak n_e | − | −15 to −40 % |
| T_e,peak | − | −5 to −25 % |
| I_beam | − (weak) | −10 to −40 % |
| per-cusp electron wall current | + | up |
| cusp sheath drop | − | down |

Series verdict (`assess --series`): `trend_confirmed` = ≥ 3 of the 4 points reached (a), I_d and peak n_e monotone in the declared
direction over α (reversals inside the band are ties), no monotone quantity contradicting at a reached point; `trend_not_confirmed`
= ≥ 3 points reached and a reversal or a contradiction (a finding about the closure's sign); `inconclusive` = < 3 points reached, or
every I_d / peak n_e shift inside the band. No α is "chosen": α stays a declared closure parameter (audit §6) until an r–θ / z–θ
companion campaign supplies the mobility.

## 5. Launch-box preflight and shakedown (non-evidentiary; H100, 16:48–16:57 UTC 2026-09-04, code `c1508c06`, as the 4th CUDA-MPS client beside ss25-base, sweep-056-launch2 and ss33-fast)

`preflight-<case>.json` (`preflight --case … --gpu-timing`, 2000 timed steps after 200 warm-up, block-Thomas + CUDA-graph step):

| case | α | factorisation | seed load (645 k e⁻) | plateau load (2.26 M e⁻ + 2.26 M i) | ν_an Δt at 0.291 T | device pool | 3 transits at the plateau load | budget (× 1.5, 10-min ceiling) |
|---|---|---|---|---|---|---|---|---|
| alpha-1over16 | 1/16 | 1.7 s | 3.70 ms/step | 4.77 ms/step | 0.0045 | 1.32 GB | 6.81 h | **37 200 s (10.3 h)** |
| alpha-1over64 | 1/64 | 1.7 s | 3.69 ms/step | 4.77 ms/step | 0.0011 | 1.32 GB | 6.82 h | **37 200 s (10.3 h)** |
| alpha-0.345 | 0.345 | 1.7 s | 3.46 ms/step | 4.78 ms/step | 0.0248 | 1.32 GB | 6.83 h | **37 200 s (10.3 h)** |

The Bohm kernel's cost is invisible at this contention (the three plateau-load rates agree to 0.2 %); the seed-load run at α = 0.345
recorded 15.6 M anomalous events over 2200 steps (1/16: 2.9 M; 1/64: 0.72 M) — the rate scales with α as declared. Field
`abf26c5c4fa6` (max |B| 0.291 T), 45 810 plasma cells, ω_ce Δt 0.072, ω_pe Δt 0.050 at the reference density. The rates are 4-client
rates (the preflight itself was the fourth client) and faster than the mini-sweep reference's 6.19 ms/step (pre-v2.0.5 code) — the budget
is the declared 1.5 × measured rule; a wall-budget stop is resumable (new session, same identity, disclosed).

`shakedown-alpha-1over16.json` (`shakedown --case alpha-1over16`; 100 000 steps of the real case with shrunk cadences — series /
sync 200, checkpoint 4000, window 40 000, frames 2000; every gate, the grid, Δt, W, α, field and seed the real ones): 3.81 ms/step,
387 s, 50 frames, `target_steps_reached` at 0.140 µs, 448 764 e⁻ / 551 830 Xe⁺, **110.8 M anomalous events** (rate 1.98e19 /s at the
end); the **v2.0.6 accumulated-floor peak-Debye window was enforced in 301 of 500 records** (max 0.355 cells/λ_D; the resolved set
37 147 nodes where the mean-occupancy floor resolved 0 — the floor is live, not inert); the windowed residual-power window completed
in 280 records (last −0.49 %, cooling side, on the W-corrected ledger); `assess --case` → `no_plateau` (a False, b True) with the shift
table and the per-cusp report formed against the α = 0 reference (reference consistency 7/7 recomputed from the v4 artifacts);
`assess --series` → `inconclusive` (only α = 0 reached); re-finalize from the checkpoint 5.6 s. The 0.14 µs transient values are not
physics and are not quoted. The shakedown ran on the a-priori-budget composition (50 400 s); the sealed protocols differ from it only in
`stopping_rule.wall_budget_seconds` / `_note` (outside `config_sha256`; identities unchanged).

## 6. Commands (from `modern/`, `PYTHONPATH=src:.`)

```
python -m experiments.pic2d_anomalous_transport_v1.run compose [--budget-from-preflight]
python -m experiments.pic2d_anomalous_transport_v1.run preflight --case alpha-1over16 --gpu-timing
python -m experiments.pic2d_anomalous_transport_v1.run shakedown --case alpha-1over16
python -m experiments.pic2d_anomalous_transport_v1.run launch --case alpha-1over16 --expect-commit <prereg sha> --require-mps
python -m experiments.pic2d_anomalous_transport_v1.run status
python -m experiments.pic2d_anomalous_transport_v1.run assess --case alpha-1over16
python -m experiments.pic2d_anomalous_transport_v1.run assess --series
python -m experiments.pic2d_anomalous_transport_v1.diagnose_launch1 [--write]     # §9: the launch-1 triad-stop diagnosis from the record
```

Launch order (one H100 MPS slot each, as the scheduler frees them): `alpha-1over16` → `alpha-1over64` → `alpha-0.345`
(`modern/tools/cloud/jobs.yaml` jobs `at-alpha-1over16`, `at-alpha-1over64`, `at-alpha-0.345`).

## 7. Preregistration and launch log

- Draft `2dcaebbc` (model v2.1.0 code `f1255832`): code + package + tests, no preflight / shakedown records, no launch.
- **PREREGISTERED** at the commit carrying this README, the sealed `protocols/*.json` with the measured budgets (sha256
  `33acb08a…` 1/64, `b59b4402…` 1/16, `a9519acb…` 0.345; campaign `protocol.json`), the three preflight records and the 1/16
  shakedown record (§5). Launch order alpha-1over16 → alpha-1over64 → alpha-0.345, one H100 MPS slot each via
  `tools/cloud/schedule.py` jobs `at-alpha-1over16` / `at-alpha-1over64` / `at-alpha-0.345` (jobs.yaml commit after this one); the
  ext-val `bohm-0.4` launch 2 takes the slot after the first α job.
- PREREG commit **`057841cf`** (pushed; the `--expect-commit` of every case). jobs.yaml `b14a9350`.
- **LAUNCH 1 — `alpha-1over16`: PID 46438, 17:08:39 UTC 2026-09-04 (03:08 AEST 5 Sep)**, `schedule.py launch --only at-alpha-1over16`,
  detached worktree `jobs/at-alpha-1over16/tree` at `057841cf`, execution lock 17:08:42 UTC (clean worktree, protocol `b59b4402…`, config
  `90cf53f1aef1…`, MPS pipe present, Warp UUID = nvidia-smi), `results/alpha-1over16/`. Took the free 4th scheduler slot beside ss25-base
  (PID 32709), sweep-056-launch2 (38282) and ss33-fast (44430); a transient 5th GPU client (another agent's `pic2d_xe_collision_set_v2_shakedown`,
  PID 45799, started 17:01 UTC) was on the device at launch — disclosed, not ours, not signalled. First records: 4.72 ms/step at the 490 k-electron
  seed load (the preflight's 3.70 was measured with 4 clients), I_d 3.8–3.9 mA, S 2.5–2.7e16 /s in the ignition transient. ETA 3 transits ≈ 6.7 h
  at the seed rate (≈ 23:50 UTC / 09:50 AEST 5 Sep), later at the plateau load (4.8–6 ms/step → 7–9 h); budget end 17:08 + 10.3 h ≈ 03:30 UTC
  (13:30 AEST) 5 Sep.
- Next (as written at launch 1): the ext-val `bohm-0.4` launch 2 (amendment 1, `a1065ce4`) takes the next slot that frees (sweep-056-launch2 ETA
  ≈ 21:30 UTC), then `at-alpha-1over64`, then `at-alpha-0.345` (enable in jobs.yaml as slots free; never a 5th scheduler client).
- **LAUNCH 1 STOPPED — 19:01:09 UTC 2026-09-04 (05:01 AEST 5 Sep)**: `grid_heating_triad_gate_stopped_run` at step 1 720 000 = 2.408 µs =
  **1.0033 transits — the first checkpoint after the drift members armed** (`enforced_after_transit_times` 1.0): `ionisation_rate_drift −0.618`
  and `t_e_dense_drift +0.366` exceeded the hard 0.25. 6853 s wall (1.90 h), 3.6–4.8 ms/step, exit 0, finalizer OK (window maps, 86 frames).
  The slot freed itself; the box slot-waiter launched the ext-val `bohm-0.4` launch 2 into it at 19:07 UTC (PID 49403).
- **DIAGNOSIS (§9; `results/alpha-1over16/triad-stop-diagnosis.json`, `diagnose_launch1.py`): the discharge EXTINGUISHED under the closure** —
  not heating (windowed residual +1.15 % of the electrode work, cumulative −0.17 %, accumulated-floor peak 0.48 cells/λ_D), not a
  re-equilibration toward the hypothesised state (I_d 3.1 → 0.06 mA), not a code artefact (the hook ran at ν_an = ω_ce/16 per electron at
  ⟨|B|⟩ 0.15 T; the loss e-fold 0.88 µs matches r_w²/4D⊥ at D⊥ = kT_e/16eB). The T_e,dense member's +0.366 IS a shot-noise reading of an
  undefined statistic (the dense node held < a few macro-electrons after 1.1 µs) — recorded as a weakness of that statistic; the S member's
  −0.618 is the real decay. `assess --case alpha-1over16` → **`no_plateau`** ((a) False, (b) True at +0.0115); the trailing-window shifts are
  those of a dead discharge (I_d −95 %, S −99 %, peak n_e −99 %, n_g +72 % = back at n_g0) and carry no trend contribution by the rule.
- **NOT RELAUNCHED**: the same seed and configuration identity (`90cf53f1aef1…`) replay bitwise into the same extinction; under amendment 1
  the ignition gate would stop the replay at 1.008 µs. Launch 1 IS the case's record (`results/alpha-1over16/`, executed under the
  pre-amendment seal `b59b4402…`). The recorded stop was correct in effect (a dead discharge; 8.4 GPU-hours of the 10.3 h budget saved)
  and mislabelled in cause ("grid heating").
- **AMENDMENT 1 — `protocol.json#amendments[0]`, all three cases re-sealed identically** (`cb8fb8da…` 1/64, `7bfd763b…` 1/16, `7c6f288e…`
  0.345; campaign `f103b6fa…`; pre-amendment seals `33acb08a…` / `b59b4402…` / `a9519acb…` recorded as genealogy; `config_sha256`
  identities `28ca0391` / `90cf53f1` / `8ea88273` UNCHANGED): (i) `grid_heating_triad.drift_members_arming` — model v2.1.1 (`e47ae78a`,
  spec `triad_drift_arming_v2_1_1`): the drift members are enforced only after ≥ 2.0 transits AND once the trailing-20 % I_d drift has read
  < 0.05 at a 40 000-step checkpoint (the discharge has settled once), so a case re-equilibrating to a different state under the closure
  cannot be stopped for moving; the residual-power gate (≥ 5 % from the first complete window) and the accumulated-floor peak-Debye gate
  (π) stay enforced from their windows as the physics protections — they read +1.15 % / 0.48 on the 1/16 run; (ii) `stopping_rule.
  ignition_gate` — calibrated on the accepted 33 µm runs (ss-v4 / 047 / 009 / 056-L2: N_e ratio ≥ 1.31, S ratio ≥ 0.96 at 1.0 µs) and the
  extinguished launch (0.45 / 0.37): a discharge the latch can never arm on is stopped at 1.0 / 2.0 µs instead of running to its budget.
  The amendment commit **`33be2a89`** (record `0916a4f8`, model v2.1.1 `e47ae78a`) is the `--expect-commit` of `alpha-1over64` and
  `alpha-0.345` (jobs.yaml commit after it).
- **Queue after the amendment**: the box slot-waiter `r1-queue` was paused at 19:35 UTC (no case may launch under the 1.0-transit rule) and
  restarted with the amended order `at-alpha-1over64` → `at-alpha-0.345` (one MPS slot each as the scheduler frees one; sweep-056-launch2
  ends ≈ 21:10 UTC → 1/64 ≈ 21:15 UTC / 07:15 AEST); the chained physics-effects queue (`pe-queue`) restarted behind it. 0.345 (D⊥ 5× the
  1/16 value) is expected to extinguish faster — the ignition gate bounds that cost to ≈ 1 h and records the point.

## 8. Claim boundary

Preregistered closure-sensitivity study of a development model on one design, one operating point, one seed per α, against a
reference plateau that itself heats at +2.46 % on the corrected ledger. The outcome is the sign and monotonicity of the closure's
effect; the magnitudes are recorded; every discharge quantity of the 2D axisymmetric model is conditional on the declared α; not
validated against experiment; not a thruster performance prediction. Launch 1 adds a recorded finding of the same standing: at α = 1/16
the v1.3-closure discharge with the exit-plane 3 mA / 2 eV injection and dielectric walls without SEE does not self-sustain (§9) — a
statement about THIS model at THIS operating point, not about the device.

## 9. Launch 1 (α = 1/16) — extinction under the closure; the amendment (2026-09-05 05:00–06:30 AEST)

**What stopped the run.** The v1.4 triad drift members armed at 1.0 transit (`enforced_after_transit_times`), and at the first checkpoint
after arming (step 1 720 000, 2.408 µs, 1.0033 transits) two of them exceeded the hard 0.25: `ionisation_rate_drift −0.618`,
`t_e_dense_drift +0.366` (`omega_pe_dt_drift` undefined — no node held ≥ 32 macro-electrons after 1.1 µs). The physics protections had
nothing to say: windowed residual power **+1.15 %** of the electrode work (2.83e-10 J over 2.46e-8 J; cumulative −0.17 %; never above
+1.2 % at any record), accumulated-floor peak **0.48 cells/λ_D** (α = 0 plateau: 2.15; hard π).

**What the discharge did** (trailing-window drifts at 200 000-step checkpoints and the trajectory are in `triad-stop-diagnosis.json`):

| t (µs) | N_e (macro) | I_d = anode e⁻ (mA) | exit-plane e⁻ return (mA) of 3.0 injected | S (/s) | n_g (m⁻³) | T_e,dense (eV) | α = 0 at the same t: N_e / S |
|---|---|---|---|---|---|---|---|
| 0 (seed) | 6.06e5 | — | — | 7.0e15 | 5.50e19 | 4.6 | 6.06e5 / 7.2e15 |
| 0.1 | 4.62e5 | 3.10 | 0.32 | 2.1e16 | 4.16e19 | 11.4 | 5.51e5 / 1.5e16 |
| 0.5 | 3.20e5 | 2.15 | 1.60 | 1.4e16 | 4.65e19 | 8.9 | 6.54e5 / 1.6e16 |
| 1.0 | 1.89e5 | 0.93 | 2.21 | 6.2e15 | 5.09e19 | 1.2 → ~0 | 7.97e5 / 1.8e16 |
| 1.5 | 1.05e5 | 0.49 | 2.43 | 8.6e14 | 5.43e19 | ~0 | 9.62e5 / 2.1e16 |
| 2.0 | 5.78e4 | 0.14 | 2.99 | 2.9e14 | 5.48e19 | ~0 | 1.14e6 / 2.9e16 |
| 2.408 (stop) | 3.74e4 | 0.06 | 2.87 | 0 | 5.49e19 | 0 | 1.29e6 / 3.2e16 |

N_e fell monotonically from the seed (its maximum is t = 0) with an **e-fold of 0.88 µs** over the last microsecond; the injected 3.0 mA
was trapped at first (exit return 0.32 mA at 0.1 µs, like α = 0) and then returned through the exit plane as the seed plasma's potential
structure decayed (2.9 of 3.0 mA at the stop); the wall electron and ion currents tracked N_e down (4.7 / 5.0 mA at 0.1 µs → 0.14 / 0.21 mA);
n_g returned to the undepleted n_g0 = 5.5e19. The anomalous event rate 2.0e19 /s at 0.1 µs is **1.66e9 /s per electron = ω_ce/16 at
⟨|B|⟩ = 0.15 T** (channel |B| 0.05–0.29 T): the hook ran at its declared rate. The decay time is the cross-field Bohm loss time of the
electron inventory, r_w²/4D⊥ ≈ (3 mm)²/(4 × 2 m²/s) ≈ 1 µs at D⊥ = kT_e/16eB (T_e 5–10 eV, |B| 0.1–0.3 T), shorter than the ~2.5 µs
electron confinement the α = 0 discharge sustains itself with (N_e e/I_d) — the classical Bohm value removes the confinement this discharge
needs, in this model, at this operating point (v1.3 closure, exit-plane 3 mA / 2 eV injection, dielectric walls without SEE).

**Verdict** (`triad-stop-diagnosis.json#verdict`): **(d) extinction under the closure** — none of the three predeclared readings: not (a)
heating, not (b) a benign re-equilibration, and (c) an artefact only in the T_e,dense member (a shot-noise drift of an undefined
statistic — recorded as a weakness of that statistic, not the cause of the stop). Against the hypotheses: **"I_d up +20…+60 % at
α = 1/16" is CONTRADICTED in the strongest form — no self-sustained discharge exists at this α**; the S / peak n_e / T_e "down" signs
are realised as an extinction, not as a shifted plateau. By the predeclared rule the case is `no_plateau` (no trend contribution).
Consequence for the series verdict: `trend_confirmed` now needs both 1/64 and 0.345 to reach (a); if 0.345 (D⊥ 5× the 1/16 value)
also extinguishes, the rule returns `inconclusive` and the recorded finding is stated as such — the Bohm closure at α ≥ 1/16
extinguishes this discharge; the sign hypotheses are not testable there.

**Why the arming rule was amended anyway.** The stop exposed that the drift members' 1.0-transit arming had been calibrated on α = 0
plateaus only (ss-v4 at 1.0 transit: I_d drift +0.116, S +0.10, T_e,dense +0.02) — a case that re-equilibrates to a genuinely different
state under the closure (the 1/64 case is the one this protects) could be stopped for moving while nothing is wrong numerically. Model
v2.1.1 (`e47ae78a`, spec `triad_drift_arming_v2_1_1`, 8 tests) arms the drift members by a settled-once latch: ≥ 2.0 transits AND the
trailing-20 % I_d drift has read < 0.05 (the plateau threshold) at a 40 000-step checkpoint; a pure function of the series, so a resume or
an offline re-read reproduces it. Calibration: ss-v4's latch closes at 2.66 transits (checkpoint 4 560 000, drift +0.049); 047 / 009 /
056-L2 read |I_d drift| < 0.05 from 2.0 transits; no accepted run ever tripped a drift member (so their records are unaffected: the members
are enforced strictly later); plume attempt 8 and the ext-val launch 1 are still stopped by the residual-power member at the same record.
A discharge that never settles is an extinction the latch can never arm on — hence the ignition gate beside it, calibrated on the same
accepted runs (N_e / S ratios over the 0.05–0.2 µs reference at 1.0 µs: ss-v4 1.40 / 1.03, 047 1.31 / 0.96, 009 1.32 / 1.17, 056-L2 1.44 /
0.99; bounds 0.6 / 0.3 → margins ≥ 2.2× / 2.4×; the extinguished launch 0.45 / 0.37 → stopped at 1.008 µs instead of 2.408). Both are
stopping-rule keys outside `config_sha256`: identities unchanged; the residual-power and peak-Debye gates untouched.

**Ext-val `bohm-0.4` (α = 0.345 at Brandt's operating point, launch 2, PID 49403, running under the sealed 1.0-transit arming
`a1065ce4`)**: exposed to the same rule in principle; at 0.43 µs its N_e sits at 0.9× the post-seed reference (5.7e4 → 3.9e4 macro,
neither the α = 0 ext-val avalanche nor yet a decay). Decision: the running process is not touched and its sealed protocol is not amended;
if launch 2 stops on the drift members with a re-equilibrating discharge, a launch 3 is preregistered under an ext-val amendment 2 adopting
the v2.1.1 latch + an ignition gate calibrated for the 20 µm / 2e20 seed transient; if it extinguishes, the stop is the recorded outcome
(as here). Recorded in the ext-val README §13.
