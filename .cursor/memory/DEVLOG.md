# Devlog

## 2026-09-04 02:20 AEST — rollover

- Task summary:
  - Archive-first rollover of the two agent memory files (both untracked; no `git add`, no repo file touched).
    `AGENT_SCRATCHPAD.md` (128,154 bytes, 2141 lines) copied verbatim to
    `.cursor/memory/archive/AGENT_SCRATCHPAD-2026-09-04-0220.md` (SHA-256 `2df5d7fef70d7ad12547fca91c8d4ebf4f844f25400fbc9dd01912c2c7cc3fa3`);
    `DEVLOG.md` (169,180 bytes, 2437 lines) copied verbatim to
    `.cursor/memory/archive/DEVLOG-2026-09-04-0220.md` (SHA-256 `c7183ce743308b28e249d48a4f68df0ec1e54afd91a14ff6de1c091cf917f7a8`). Window covered:
    2026-09-02 05:00 (venv provisioning) -> 2026-09-04 02:30 (screening v2 paper admission; plume attempt 6 running).
- Carried forward:
  - Scratchpad: header + File Policy verbatim; all 67 Retained Lessons bullets verbatim, grouped
    under six sub-headings; a condensed chronological session summary; archive pointer.
  - Devlog: this entry; the condensed chronology below (every commit SHA of the archived devlog appears in it);
    every `Follow-up notes / risks` bullet verbatim under Open follow-ups; archive pointer.
- Validation:
  - Archive copies hash-identical to the originals (SHA-256 above). Scripted checks: every 7+-hex-digit token of
    the archived devlog appears in the new devlog; every archived Retained Lessons bullet appears verbatim in the
    new scratchpad; every archived follow-up bullet appears verbatim in the new devlog; both new files are UTF-8,
    LF, no BOM, 0 CR bytes.

## Condensed chronology 2026-09-02 → 2026-09-04

One line per milestone: time (AEST; landing/merge time from the running log, entry time otherwise) — item —
commit SHA(s) — headline numbers. Detail, validation and per-entry follow-ups: archived devlog.

- 09-02 05:00 — `.venv-sota` isolated ML runtime provisioned (pip 25.0.1 -> 26.2.1; PyTorch 2.13.0+cu130, BoTorch
  0.18.1, GPyTorch 1.15.2, pymoo 0.6.2; CUDA float64 on RTX 5090 sm_120 verified; 12 stale `torch==2.9.1+cu128`
  launcher/child processes stopped, 1.01 GiB partial download removed; `pip check` passed). No commit.
- 09-02 23:30 — roadmap-stall investigation — no commit — orbit wall-loss v1/v2/v3 all died on code (v3: zero-length
  step near the wall -> invalid STEP_LIMIT witness, orbit_mc v1.4 `integrator.py`); main tree `ahead 1, behind 13`,
  84 dirty entries, 33 worktrees; uncommitted v1.5: 102 tests passed in 7.27 s.
- 09-02 23:45 — main-tree reconciliation — local `746462a8` = origin `8603a905` (skipped); 35 new files in 5 commits
  `25dbeaaf..caf3a04c`; tag `backup/main-pre-reconcile-20260902-233116`, stash `main-tree-pre-reconcile-20260902-233116`;
  31 worktrees pruned — 141 dirty entries byte-identical to origin, 12 identical via LFS oid.
- 09-03 00:06 — orbit_mc v1.5 merged — `7cf65053` — real-field shakedown 512/512 validators on primary/refined/enlarged/4N;
  Warp CUDA per-particle path 18x slower -> CPU.
- 09-03 00:11 — l1a_plasma_coupling adapter fixed — `40dcaa4c` (adapter predated serialization v1.2 `dbcab646`) —
  8/8 tests; root `.gitattributes` `* text=auto eol=lf` — `fab0eccc` — 760 files re-smudged to LF.
- 09-03 00:25 — tests/coupling wall-clock time bomb fixed — `4661a7be` — 143 passed (was 134 passed / 9 failed).
- 09-03 00:45 — orbit_mc v1.6 merged — `3ab50ef5` — Boris sub-push event velocity; 1e-10 energy gate 512/512 on all
  four cases; tests/orbit_mc 120.
- 09-03 00:46 — v4 phase 1 scaffold — untracked worktree — shakedown 9/9 cases, 289 validators / 0 failures, 118 s;
  `zip(..., strict=True)` v3 bug found + fixed; 21 v4 tests.
- 09-03 01:20 — wall-loss v4 ACCEPTED — `23d37bee` tests / `757e365f` prereg / `6922a3cf` result — 667 s, 15/15 gates,
  4608 orbits, wall-hit 0.641-0.645, escape 0.355-0.359, reflection 0, changes <= 0.0039.
- 09-03 01:52 — orbit_mc v1.7 LF sidecars + v4 posthoc audit — `cc4bd5e1`, `258f69b2` (ff `6922a3cf..258f69b2`) —
  9 sidecars EOL-only, 378 byte-exact; results tree `447a5cf7` unchanged; tests/orbit_mc 147, coupling 143.
- 09-03 02:06 — plasma topology dashboard — `16670281` — 72 hashed sources, 17 tests; characterization v1 0/0 stable
  cusps over 56 designs (1276 nulls excluded), four-cell v2 0/128, sweep v2 96/0.
- 09-03 02:35 — v4 dashboard + paper evidence — `bc2f8e47`, `ea867bf1`, `5b85d2ad` (from `6922a3cf`; feature branch
  stays `fb4117e4`) — 681,963 B HTML, 130 macros, 13 + 8 tests; exit is 23 mm in the artifacts.
- 09-03 02:50 — stale design gallery fixed — `8466c37a` (from `16670281`, rebased over `bc2f8e47..5b85d2ad`; ff
  `5b85d2ad..8466c37a`) — pin `a4703ac1` (CRLF bytes; full `a4703ac1541539829f47f909d24d01d4996ed1da97a9d86e9e2323e54039fbbf`)
  -> `2d727b1a` (LF blob, also at `41bf9091`; full `2d727b1af7d9be9f35f227cc318beae29af6cbd2fbead28842a4c17d67551b6b`);
  dataset `c0a36ed8` unchanged; visualization 93 passed. Left red: sweep-v2 sidecar CRLF hash `64b2c58c` vs LF blob `2a5ba9e4`.
- 09-03 03:00 — paper: wall-loss v4 admitted — `0fabda2c` -> rebased `6f3e6dd5` (ff `8466c37a..6f3e6dd5`) — gate kind
  `numerical-campaign`, CLM-012..017, 29 files bound at `6922a3cf`; 33 paper tests; PDF 11 pages `bdfdba4c`.
- 09-03 03:09 — PIC-2D phase 1 — `53ac3b02` -> `f44a7399` -> `d58fdca1` -> `dd5f2ff1` — `cft_revival.pic2d`, 58 tests,
  Poisson order 1.999, orbits vs orbit_mc 2.8e-4 gyroradii; snapshot v1 gate-stopped at 49-60 ns.
- 09-03 03:46 — test health closed — `9e68df21..7a30fc2e` — modern/tests 1677 / 0 / 5 in one invocation; pic2d phase 1
  merged `df4b2d77..62de2ca3`.
- 09-03 04:50 — paper: L1a sweep v2 + topology nulls admitted — `605be5ce` (four-cell v2 EOL audit: CRLF digest
  `ec2e9a73` of LF blob `5c195119`, payload `bd522269`), `f171e9ec` (ff `7a30fc2e..f171e9ec`) — gate kind
  `numerical-screening`, CLM-018..028; 58 paper tests; PDF 17 pages `6b4c6978`.
- 09-03 06:53 — PIC-2D phase 2 — `3a42bcd7`, `a0fc4a20`, `1cdaae80` (ff to `1cdaae80`) — 40.7 -> 5.46 ms/step at 5.4 M;
  snapshot v2 no plateau, peak n_e 3.7-5.9x ceiling; 368 tests.
- 09-03 07:20 — PIC phase 3 — `44b7c8dc` — resumable steady-state runner; tau_i,eff 2.4 us, nu_iz*tau 2.9 at n_g 1e20;
  v1.2 at 1.5e19 no ignition (PID 49664).
- 09-03 08:03 — roadmap audit — `cc7706b2` — 63 % (58-68) vs canvas 70 %; suite 1702/0/5; `origin/main` stale at
  `7ca3dc2d`, feat/sota-foundation 97 ahead.
- 09-03 09:18 — PIC v1.3 neutral inventory — `520e6b41`, `67b04f87`, `3c9e606c`, `cb40f06d`, `8babb31e` (ff to
  `8babb31e`) — 388 tests; attempt 2 igniting (PID 40636, n_g0 5.5e19, seed 5e16).
- 09-03 10:13 — MDO L0 v1 ACCEPTED — `fdc6b37d` / `4898d0fd` / `c553124b` / `e642f38c` (ff `8babb31e..e642f38c`) —
  864 evals, 8/8 gates, 28 min; HV 0.003863/0.003877/0.003860 vs dense 0.003798; protocol `09755b85`, source
  `da21671f`, shakedown `8b5a8293`, manifest `2a326f3c`; solver probe 13/80.
- 09-03 10:58 — four-cell closure analysis — `266d8a99` (ff `e642f38c..266d8a99`) — R27 closed form, no root for
  interior p > 0; PAV projection 16/16; 24 tests. Pre-existing failure vs `f80a360f` (topology char v1) not mine.
- 09-03 11:02 — paper: MDO v1 admitted — `9f351776` -> rebased `ba6875f6` (ff `266d8a99..ba6875f6`) — CLM-029..035,
  334 macros; 83 paper tests; PDF 22 pages `e7900c10` (`e7900c1000e3d48ef02cc6e67e114dce946c9cadfa4cd7820414ee48bde0d4ff`).
- 09-03 12:18 — PIC steady-state v2 plateau — no commit yet — 3.2 transits, 7.68 us, 5.12 M steps, 2.8 h; I_d 3.5-4.0
  mA, n_g 2.95e19.
- 09-03 12:26 — paper: closure analysis admitted — `d09ffee2` (ff `ba6875f6..d09ffee2`) — gate kind
  `analytic-consistency`, CLM-036..044, 163 macros; 111 paper tests; PDF 27 pages `6ac978b2`
  (`6ac978b29ab899092e0427c44bbe5f26f8608190589877b50bd729f16ede8a85`).
- 09-03 12:42 — geometry screening v1 RECORDED — `484335c2` / `c86bfca3` / `ce7cb895` / `5f4a6426` / `ab7c2897`, merge
  `22e2156b` (ff `d09ffee2..22e2156b`) — 96 designs, 100 352 orbits, 6664 validators; P(wall) 0.375-0.869; reflections 22 %.
- 09-03 12:47 — MDO v1 posthoc audit — `6cb9a1af` -> rebased `e9f9af16` (ff `22e2156b..e9f9af16`) — 137/137 byte-exact;
  source hash from blobs at `4898d0fd`, `c553124b`, `e642f38c`, `ba6875f6`; six disclosures.
- 09-03 13:02 — PIC phase 4 — `24ab82f4`, `a707fc1a`, `5564480a` (ff `c32dd780`) — I_d 3.44 mA, 46 % utilisation, peak
  n_e 1.64e18 = 4.1 n_max; seed-b PID 49716.
- 09-03 13:39 — surrogate v1 REJECTED — `aa9349a9` / `b602d147` / `b400d924` / `bfe123d4` (ff `c32dd780..bfe123d4`;
  dataset blob `858de21a` at `ab7c2897`) — pooled RMSE 0.0562 vs 0.05; ridge 0.0546; coverage 0.80.
- 09-03 13:58 — paper: screening v1 admitted — `3003325d` (ff `bfe123d4..3003325d`) — CLM-045..052, 271 macros; 139 paper
  tests; PDF 33 pages `67a531f9` (`67a531f9562f785f2eef7a5c6f053c3f2c4cb2918c4e80c240735939670fd720`).
- 09-03 14:26 — surrogate v2 rejected — `21118507` / `503bf87f` / `a2b503be` / `783a82c6` (ff `3003325d..783a82c6`) —
  pooled RMSE 0.0337 (pass), cells 0.0904 (fail), ridge 0.0334.
- 09-03 16:59 — MDO L0 v2 ACCEPTED — `19c91a90` / `99914dc2` / `a003f766` / `0ea33a7e` (ff `783a82c6..0ea33a7e`) — 1440
  evals, 12/12 gates, ~83 min; HV 9.269e-4 / 2.159e-3 / 2.151e-3 vs dense 1.9073e-3; robust front 49/50/94; CL-2 Jaccard 0.
- 09-03 17:20 — PIC seed-b comparison — `41ccb1ef`, `96220ffc` — <= 1.1 % on every plateau quantity; W x0.7 launched
  (PID 9856).
- 09-03 17:49 — literature review: reduced models / cusp loss / topology — `66879e00` (ff `96220ffc..66879e00`) — 72 refs.
- 09-03 17:53 — literature review: PIC-MCC — `bf43a7fa` -> rebased `ccb22d5d` (ff `66879e00..ccb22d5d`) — 116 refs.
- 09-03 17:54 — literature review: surrogate / MDO / validation — `af98b3dd` -> rebased `b6bb6215` (ff
  `ccb22d5d..b6bb6215`) — 157 refs (345 total).
- 09-03 18:05 — paper: MDO v2 admitted — `a3793c27` (rebased over `41ccb1ef..b6bb6215`; ff `b6bb6215..a3793c27`) —
  CLM-053..060, 611 macros; 165 paper tests; PDF 41 pages `86796210`
  (`867962101b0eff10f8023c44b96f36fa8dea5c1633678a1d197bf8e321348431`).
- 09-03 18:37 — literature synthesis — `11a10873` -> rebased `8674cc5a` (ff `a3793c27..8674cc5a`) — 60 recommendations
  53/6/1; canvas 35 ladder rows.
- 09-03 19:03 — PIC v1.4 — `112bb250` (ff `8674cc5a..112bb250`) — recycling, peak-node gate, CUDA-graph step; 111 tests;
  ms/step 7.8 -> 1.5 (9 k).
- 09-03 19:06 — plasma v2 sheath closure — `e75151ce` -> rebased `fb5408bf` (branch `ea798971`; ff `112bb250..fb5408bf`)
  — rank 21 -> 28 -> 31; 53 tests; SCL 73/80, no-emission 0/96.
- 09-03 19:57 — cusp topology v3 REJECTED -> v3.1 ACCEPTED — v3 `bce595dc`, `69159934`, `8cbcdbe6`, audit `9fa6359a`;
  v3.1 `ca811d11`, `988220f3`, `1600cfd3`, `cec47f12`, dashboard `9abbd537` (ff `fb5408bf..9abbd537`) — 281/281 stable;
  wall cusps 0:6/1:140/2:36/3:56/4:25/5:6/6:6/7:6; P2 6.028/12.000/17.972 mm.
- 09-03 20:08 — TWT/PPM literature review — `beb4772c` (ff `9abbd537..beb4772c`) — 51 refs; wall cusp / axis peak
  0.45-0.61; Koch rho max 1.03 of 96.
- 09-03 21:32 — L1a sweep v3 ACCEPTED — `b04d5935` / `1923ef76` / `2cfe8223` / `44d0c63c` (ff `beb4772c -> 44d0c63c`) —
  224/224 in 29 min, 11/11 gates; 15/128 HEMP-like; x* 2.34 (r_w/L 0.745).
- 09-03 21:41 — paper: topology v3.1 admitted — `726c8a69` -> rebased `13d8ac6a` (ff `44d0c63c..13d8ac6a`) — CLM-061..068,
  430 macros; 197 paper tests; PDF 49 pages `34e11c8e`.
- 09-03 22:54 — PIC v2.0 plume + frame recorder / video renderer — `542496fb` (W x0.7 record), `9c7d944a` (W x0.7
  finalised), `e3e9167d` / `f7169279` (v2.0), `6bd5e5b0` / `f3732d9a` (dashboards), `ff8e0baa` (ceiling fix), `1fb8561d`
  (frames; ff to `1fb8561d`) — 172,800 cells, 4.2 ms/step at 0.55 M; 146 tests; W x0.7 vs base I_d +5.7 %; plume launch 3
  PID 28860.
- 09-03 23:08 — screening v2 ACCEPTED — `a7a884bf` / `cef1ee59` / `26029b72` / `bb756418` / `eef7ac82`, merge `066234d9`
  (ff `1fb8561d..066234d9`) — 97 designs, 377 cells, 1105 cases, 104,832 orbits, 86 min; 181/181 interior cells at 1.0;
  manifest `876dc7e1` published post hoc (EMFILE, 16,957 files > 8192).
- 09-03 23:45 — paper: sweep v3 admitted + reflections re-scoped — `88386161`, `ba852122` — CLM-069..076; 228 paper tests;
  PDF 56 pages `b2440f6e`.
- 09-04 00:20 — plume attempt 3 no ignition -> field-line diagnosis + cathode on the channel tube — `2b3372a0` (attempt-3
  record; `978000c4` on the pic branch), `eb8585c3` / `c6219bf3` (fix), `f5582255` (merge-back) — 38 frames, first
  video; 0/24 -> 24/24 connected; 153 tests; attempt 4 PID 53756.
- 09-04 01:35 — plume attempts 4-5: neutral crash, ROOT CAUSE stale MCC density in the CUDA-graph step — `ad5be433` /
  `0f583df9` (relaxation guard), `3fe66bde` / `4dc40390` (device-array fix) — 155 tests; attempt 6 PID 53824 igniting.
- 09-04 02:30 — paper: screening v2 admitted — `8e13364c` (rebased onto `4dc40390`; ff `4dc40390 -> 8e13364c`) —
  CLM-077..085, 380 macros; 257 paper tests; PDF 65 pages `ba7441c9`
  (`ba7441c972a345e9390fd548ae58cd41fa43c15a984a653d38f9a2dd56f534b6`).

## 2026-09-04 04:30 AEST — PIC plume attempt 6 stopped by a shot-noise gate; v2.0.1; attempt 7

- Task summary:
  - Attempt 6 (PID 53824, model v2.0, frames ON) ignited cleanly: two-stage ignition gates passed at 0.75 µs
    (S/S_ref 1.42, N_e ratio 2.00) and 1.5 µs (1.70, 3.60); cathode connectivity 24/24; S_ref 3.18e16 s⁻¹.
    Ran to 2.466 µs / 1,644,000 steps / 82 frames / 8430 s wall, then the plume-boundary gate stopped it
    66 ns after arming (2.4 µs): "net charge at far-field nodes 0.259 of peak (limit 0.25)".
  - Root cause (numerical): the gate read max |n_i − n_e| over 481 far-field nodes from a single-step deposit.
    The axis corner node (0, 720) has shape volume π Δr² Δz / 6 = 6.545e-14 m³, so one macro-ion (W = 6e4) reads
    9.17e17 m⁻³ = 0.395 of the 2.32e18 peak; the final checkpoint held 0.66 macro-ions / 0 electrons there
    (= 0.259). Interval-averaged far-field max|net|/peak 0.030–0.033, volume-weighted 1e-4; the far plane is an
    ordinary Dirichlet sheath (φ 44 → 0 V over ~0.5 mm ≈ 3.5 λ_D). Not a CUDA-graph staleness case.
  - Health at stop: ω_pe Δt 0.134 (limit 0.2), 3.2–3.6 cells/λ_D (4.5), energy residual −3.4 % of electrode work,
    n_g 2.77e19 at its fixed point (2.5–2.8e19), neutral ledger closed to 0.05 atoms, I_d 6.3–6.9 mA still rising
    (+8.6 % drift), N_e +20 %, 0.80 transits — no plateau yet.
- Changes:
  - `45edd30e` fix(pic2d): `PlumeBoundaryGateConfig.min_macro_particles_per_node = 32` (mirrors the peak-Debye
    gate floor); raw statistic, node and macro count recorded (`charge_fraction_of_peak_raw`,
    `far_field_raw_max_node`, `far_field_resolved_nodes`; log column `q_far=…(raw …/…n)`); spec
    `pic2d-model-v2.0.json` + `protocol.json` updated; regression test (one corner ion → unresolved, no stop;
    floor 1 with two ions reproduces the pre-fix stop). tests/pic2d 156 passed (CUDA, 165 s).
  - `9cf7ca39` attempt-6 record `results-attempt6-gate-shot-noise/` (results-only + .gitignore); video 5 MP4s +
    HTML player (untracked). `895ea58d` attempt-7 launch-log entry. origin/feat/sota-foundation = `895ea58d`.
  - Attempt 7: PID 52176, 04:19:25 AEST, fresh start (gate field is in the config identity), frames ON;
    verdicts ≈ 05:03 / 05:45, gate arms ≈ 06:45, budget ≈ 08:19. First 213 series records replay attempt 6
    bitwise (N_e, N_i, φ_max, K_e, I_d, n_g).
- Validation: gate trigger reproduced from the final checkpoint; time series shows 3 % of 1.5–2.4 µs samples
  above 0.25 with lag-1 autocorrelation 0.84 and ~10-sample ramps (one ion crossing the last 50 µm cell).
- Follow-up notes / risks:
  - Ionisation-rate video panel is shot-noise by construction (72.5 % of resolved nodes zero events per 30 ns
    frame; one axis-node event sets the colour top); the dashboard event-count mask `6bd5e5b0` is not applied by
    the video renderer. Renderer fix (rolling window + event mask + fixed percentile scale) in progress.
  - `step_graph` in the launch-time provenance line is always `false` (captured lazily; summary.json shows
    `true`) — cosmetic, fix when the provenance line is next touched.
  - If attempt 7 reaches the plateau, thrust / exit-plane diagnostics become the first plume-domain results;
    the PIC design mini-sweep (3–5 designs) depends on it.

## 2026-09-04 05:05 AEST — PIC video renderer v0.2: ionisation-rate panel rebuilt

- Task summary: the user-flagged "sketchy" ionisation-rate panel. Frames (`frame-NNNNNN.npz`, 30 ns, float32)
  carry `ionization_rate_per_m3_s` + `sample_count_e` but no raw event counts; events derived as the dashboard
  does (`rate × V_node × Δt_frame / W`, W = 6e4, dt = 1.5 ps, 20,000 steps/frame): domain total integer to 2e-5
  per frame (9038, 12266, 14396, …), per-node values fractional (bilinear deposit). Frame maps integrate to S
  (6.80e16 s⁻¹ last-10-frame mean, = series).
- Changes (`modern/visualization/render_pic2d_video.py` v0.2, `modern/tests/pic2d/test_pic2d_video_renderer.py`):
  - `ionisation_events` / `causal_window_sum` / `windowed_ionisation`: trailing K-frame causal window (partial over
    the first K−1 frames); windowed rate = time-weighted mean of frame rates; integrates to window-mean S (4e-16).
  - `choose_window`: smallest K whose median electron-resolved (≥ 20 e-samples) node with ≥ 1 event weight holds
    ≥ 10 events → attempt 6: K = 11 frames = 330 ns (median then 34.6; at K = 1 it is 2.5 with 73 % empty nodes).
  - Mask: dashboard semantics (`6bd5e5b0`) on windowed event weight, threshold 20 (`MIN_SAMPLES_DEFAULT`);
    unresolved → grey. Attempt 6: 5.6 % of plasma nodes resolved, carrying 67 % of S (77 % last frame) — printed
    per frame in the legend.
  - Scale: fixed log10 at the 0.5–99.5 percentile of resolved windowed values over the run: 1.62e23–2.88e24
    m⁻³s⁻¹ (1.25 decades; old per-run max 2.1e25 was an axis node). Player payload schema 0.2.0 (per-map `mask`,
    ionisation `window`); `validate_player_payload` enforces causality, mask and percentile declarations.
  - CLI: `--suffix=-v2`, `--iz-window`, `--iz-min-events`, `--iz-percentiles`, `--iz-target-median-events`,
    `--cusps` (explicit, because `cusp_planes()` silently loses the overlay under protocol hash drift).
  - Other four panels unchanged (already run-wide fixed scales); their v2 MP4s were byte-identical before the
    cusp overlay was added.
  - Tests 4 → 13 (Poisson convergence within 5σ and RMS ~K^-1/2, time-weighted mean identity, integral = S,
    single-event axis node masked and no longer the scale top, `choose_window` minimality by brute force, robust
    fall-backs, legend/payload declarations, suffix/CLI on a tiny real run). tests/pic2d 165, tests/visualization
    148 passed; ruff clean on both touched files (13 pre-existing findings fixed there; ~230 elsewhere untouched).
  - Commits `bbf74ea0` (code + tests), `ab322245` (README paragraph + spec `pic2d-model-v2.0.json`:
    `colour_scale` scoped to densities, `ionisation_panel_v0_2`, test inventory). ff-push `895ea58d..ab322245`.
- Re-render: `…\uni-project-pic2d\modern\experiments\pic2d_cft_plume_v1\results-attempt6-gate-shot-noise\video\
  pic2d-results-attempt6-gate-shot-noise-{n_e_per_m3,n_i_per_m3,phi_v,t_e_ev,ionization_rate_per_m3_s}-v2.mp4` +
  `…-timeseries-v2.html` (untracked, next to the originals); ionisation MP4 4.48 → 0.72 MB. Base-plateau
  re-render skipped: no `frames/` for any steady-state run (recorder predates them; only plume attempts 3–6).
- Physics visible: ionisation confined to the bore (r ≲ 1.8 mm), organised by the three P2 cusp planes
  (6.03/12.0/17.97 mm) as "flames" from a near-axis band (r 0.3–1.2 mm) to the wall, brightest 2–3e24 m⁻³s⁻¹
  just at/downstream of each cusp; between cusps a near-axis band 5e23–1e24, strongest between cusps 2 and 3;
  decay through the cone, faint near-axis strip z 21–24 mm, small resolved patch just outside the exit
  (z 25–26 mm, r 1.5–3 mm); plume proper and axis row (V ≈ 1e-13 m³) grey = resolution limit. Ignition
  (t < 0.3 µs) starts as three separate blobs at the cusps, exit-side first; stationary pattern by 0.6 µs while
  S grows 1.8e16 → 7.0e16 s⁻¹.
- Follow-up notes / risks:
  - Frame recorder should store raw macro-event counts (and electron-sample counts per species) so downstream
    tools need not derive them; consider recording frames on the next steady-state run for a channel-only
    comparison.
  - `cusp_planes()` should raise (or warn loudly) on protocol hash drift instead of silently dropping the overlay.
  - The plume needs a longer window or a coarser bin to resolve ionisation there; at 330 ns it stays grey.

## 2026-09-04 06:58 AEST — L1b/P2 material-aware HEMP confirmation v1 (rejected) → v1.1 CONFIRMED

- Task summary: material-aware (adaptive P2 FEM; soft-iron poles μr 4000, iron return yoke, recoil-remanence
  magnets, linear) confirmation of the 15 HEMP-like designs from L1a sweep v3, CPU-only while the GPU held the PIC
  plume run. Experiment `modern/experiments/l1b_hemp_confirmation_v1/`.
- Commit chain on feat/sota-foundation (rebased over `ab322245`, ff-push `ab322245..560909f7`; pre-rebase chain on
  `origin/exp/l1b-hemp-confirmation-v1`): `6e9f056c` v1 code+tests+shakedown · `fb143eb2` v1 prereg · `2d8d6705` v1
  result = **development_rejection** · `b6125fe7` v1.1 code + v1 posthoc · `c8692ff2` v1.1 prereg · `54cd3e82` v1.1
  result **CONFIRMED** (results-only, 134 files, 7.5 MB) · `560909f7` dashboard
  `modern/visualization/l1b-hemp-confirmation-v1.html` (524 kB) + launch log.
- v1 rejection: designs 028 and 048 failed the level-0 mesh-angle gate (10°, inherited from the fem_reference
  qualification) before any solve — body-fitted mesher slivers (028: 0.254 mm exit taper → three 5.3° triangles;
  048: injector end 0.045 mm from the first magnet edge → 13,816 anisotropic 9.3° triangles), unchanged at 3/4/5
  feature elements. Committed as recorded with `POSTHOC_REJECTION.md`. v1.1: disclosed 5° gate, per-level sliver
  statistics, whole-set mesh preflight (v1 shakedown had covered 3/15 designs).
- v1.1 gates: (a) 30/30 P2 solves converged, relative true residual ≤ 2.0e-10, levels 24k–117k / 50k–466k DOFs,
  11/11 binding integrity gates, determinism replay bit-identical; (b) cusp count unchanged 15/15; (c) 37/37 cusps
  matched bijectively, max shift 0.362 mm = 0.80 of tol max(r_w/8, L1a dz) = 0.45–0.53 mm, median 0.267 mm;
  (d) HEMP-like preserved 14/15 (028: ρ 1.515 → 1.464); ρ P2/L1a 0.94–1.45 (median 1.06); wall |B| at cusps
  1.05–1.53× L1a (median 1.23), axis peak 0.98–1.35×; channel axis nulls move ≤ 1.07 mm, separatrix lean
  0.46 → 1.14 mm; level 0→1 cusp shift ≤ 1.4 µm; 2× sampling stable 15/15.
- Pre-shakedown fixes: L1a reference lacked per-cell wall maxima; pooled axis-null matching dominated by end nulls
  (split into channel/outside populations + separatrix lean); mesh sizing frozen at 3 feature elements, 600k cap.
- Runtime: design stage 3079 s + assessment 305 s, one CPU worker, BLAS 1 thread, 28–360 s/design (median 137 s);
  peak RSS 240 MB (6.8 % of the 3.51 GB budget). Tests: 62 experiment+dashboard, visualization 148; ruff clean
  (E4/E7/E9/F) on touched files. Main checkout `git pull --ff-only` → `560909f7`.
- Follow-up notes / risks:
  - Paper admission launched (gate kind numerical-screening or closest existing; v1 rejection disclosed alongside;
    sweep-v3 "awaits material-aware confirmation" wording to be amended).
  - Not claimed: saturation / B-H nonlinearity — a nonlinear-iron check is the natural next fidelity step if the
    wall-field increase (5–53 %) matters for the PIC design sweep.
  - Axis-null positions and any ρ ≥ 1.5 threshold statement are NOT robust to iron (028 crosses) — the HEMP-like
    catalogue should carry ρ under both fields.

## 2026-09-04 09:06 AEST — paper: L1b/P2 HEMP confirmation admitted; PIC attempt 7 ended at budget

- Task summary: admission of the L1b/P2 material-aware HEMP confirmation v1.1 to the paper, screening-v2 pattern.
  Single commit `1a7eaea9`, ff-push `560909f7 → 1a7eaea9`; feature branch `paper/l1b-hemp-confirmation-claim`
  also pushed; main checkout `git pull --ff-only` → `1a7eaea9`.
- Changes:
  - Gate `GATE-L1B-HEMP-CONFIRMATION-V1-1`, kind `numerical-screening`, recorded outcome
    `accepted-material-aware-confirmation` (a sixth outcome value; every earlier outcome text says "on L1a
    linear-vacuum fields"); kind description amended to name it; checker requires the justification and refuses
    any L1a screening outcome for this gate.
  - Manifest `paper-material-aware-confirmation-manifest` 1.0 (`L1B-HEMP-CONFIRMATION-V1-1-20260904-15-V1`): all
    134 bundle files at `54cd3e82` + 4 frozen at `c8692ff2` (138 sources); references at `2cfe8223` / `cec47f12`;
    lineage = whole v1 rejection bundle (104 files at `2d8d6705`, 4 at `fb143eb2`, `POSTHOC_REJECTION.md` at
    `b6125fe7`; 109 entries); dashboard at `560909f7`; 184 metrics.
  - Claims CLM-086 (abstract) … CLM-093 (Discussion); CLM-089 gates b/c → CONFIRMED, positions robust within one
    bore element, "two field models, not a plasma"; CLM-090 wall-field rise +5…+53 %, axis nulls NOT robust;
    CLM-091 three verified disclosures (v1 rejection; shakedown timing projection 100.3 min > 90 min budget, stage
    took 51.3 min; sealed shakedown disclosure still says "three REAL designs" though v1.1 ran five — not quoted).
    Amended CLM-028/075/076 to cite Section 16.
  - 320 `\Hmc…` macros (+4 table macros); Section 16 pp. 54–60 (Tables 36–39); Discussion p. 65. Tests 19
    admission + 9 evidence new, 4 updated → 285/285 (976 s). check_paper green (~100 s warm). PDF 73 pages,
    784,052 B, two builds byte-identical, SHA-256
    `105b522562509809e750bdf5a0c5fbcde7d87f78714d8ff113e78194d8b0309b`. ruff clean on touched files.
  - Artifact corrections vs the brief: tolerance range 0.451–0.523 mm (record max 0.5229); design 048 recorded
    level-0 minimum angle 5.6° (13,816 of 46,582 elements < 10°) — 9.3° described the sliver population.
  - Rebase binding: bundle lock names pre-rebase prereg `ead9b525` (and predecessor block `b9449ee5` / `978c71be`)
    on `origin/exp/l1b-hemp-confirmation-v1`; bound as strings; rebased `c8692ff2` bound by recomputing the sealed
    experiment-code / dependency / field-pipeline hashes from its blobs (equal; experiment tree verified equal).
- PIC attempt 7 (PID 52176, 04:19 start, 4 h budget): at 09:06 the process is gone, `run_state.json` still
  `finished=false`, checkpoint 2,520,000 steps / 3.78 µs / 126 frames / 14,443 s wall (= budget). Both ignition
  gates passed; ran 1.3 µs past attempt 6's stop, so the v2.0.1 floor held. nvidia-smi: 5315 MiB used at 2 % with
  no PIC process. Finalization/diagnosis agent launched; stale watcher loop (PID 42272) stopped.
- Follow-up notes / risks:
  - If the budget stop failed to write the terminal state, the finalizer needs a regression test (the EMFILE fix
    added `validate_bundle(manifest_override)`; check whether this is the same class).
  - Identify what holds 5.3 GB of GPU memory before launching attempt 8.
  - Paper `sec:mdo-l0-v2` duplicate-label warning is pre-existing and still open.

## 2026-09-04 09:56 AEST — PIC attempt 7 finalized (budget stop + finalizer crash); first plume physics; attempt 8 resume

- How attempt 7 ended: `run.log` last line step 2,515,200 `wall=4.00 h`; checkpoint at 2,520,000 (08:25:18–20)
  recorded `wall_seconds_total 14,443 > 14,400` → loop broke with `wall_clock_budget_reached`.
  `write_final_artifacts` wrote maps.npz (window 2.0–2.4 M steps = 3.0–3.6 µs), series.npz, checkpoint-final, then
  died at `write_canonical_json(summary.json)`: `ValueError: Out of range float values are not JSON compliant: nan`
  → `OrbitValidationError`. Cause: `_gpu_utilisation()` returned `float('nan')` on `nvidia-smi` 5 s timeouts (17 of
  238 per-minute calls; median call 2.3 s; 557 s = 3.9 % of budget). run_state.json was written before the summary
  → stayed finished=false. GPU 5.3 GB "held": desktop apps only. The PIC runner does not use experiment_runtime
  bundles, so `validate_bundle(manifest_override)` did not apply.
- Changes (ff-push `1a7eaea9..8556a401`; main checkout pulled ff):
  - `3b8b577a` fix: `_gpu_utilisation → float | None`; summary sanitises samples; any further non-canonical summary
    → run_state `finalization_error` with finished=false (no fabricated stop); `finalize --recover-runner-stop
    --stop-reason …` rebuilds the summary from the runner's stop artifacts, accepting only an evidenced stop reason
    (wall > budget, or plateau rule holding on the series), proving maps.npz round-trips byte-exactly, hash-checking
    maps + checkpoint-final. 4 regression tests; tests/pic2d 160 → 169.
  - `24ea2f65` results-only record `results-attempt7-wall-budget-no-plateau/` + README launch log (finalized via
    recovery at 09:23: `wall_clock_budget_reached`, maps/checkpoint byte-identical, `run_state.finalization_recovery`).
  - `e8b3fb7b` sessions record the wall budget in force. `8556a401` attempt-8 launch-log entry.
- Physics (3.78 µs = 1.22 transits, 126 frames): NO plateau (rule ≥ 3 transits, trailing-20 % drifts < 5 %; I_d
  −13.1 %, N_e +21.8 %, n_g −4.5 %; S triad soft-fail +15.5 %); energy residual +1.44 %. Trailing 20 % (3.02–3.78 µs,
  mean ± block SE): I_d 5.99 ± 0.06 mA (7.8 % per-record scatter = shot noise of 190 macro-e/record); S 8.57 ± 0.11e16
  s⁻¹; utilisation gross 0.94 / net 0.52; N_e 2.20 ± 0.04 M, N_i 2.23 M (+0.6 M/µs; 21 % in plume box); n_g 2.54 ±
  0.01e19; peak n_e 2.95e18 at 3.64 cells/λ_D; ω_pe Δt 0.151 (max 0.182). Gate v2.0.1 after arming: resolved 0.000 in
  all 4601 records (0 resolved nodes), raw > 0.25 in 0.8 %; window-averaged far-field max|n_i−n_e|/peak 0.035 →
  gate INERT on single-step deposit. Attempt-6 replay bitwise in dynamics (8219 records; host ledgers 1 ULP).
- Thrust / exit plane (DEVELOPMENT, beam drifting +20 %/window, not quotable): T_flux 19.4 ± 0.4 µN + cold gas
  1.56 → T_total 20.9 ± 0.4 µN; momentum balance 19.6 ± 0.4 µN (closure −7.9 %); I_beam 0.96 ± 0.02 mA (58,555
  crossings; 0.874 mA through z = 36 mm, 0.064 side); IEDF mean 184 eV, peak 133 eV; half-angles 50/90/95 % =
  8°/29°/60°; Isp 112 s; anode efficiency 0.6 %. Channel exit (z = 24 mm): net ion current 2.24 mA, mean ion
  energy 28 eV, momentum flux 29 µN; 90→10 % acceleration region z = 5.0→35.8 mm. Plume: axis n_i 6.4e17 at exit →
  1.0e18 at z = 27 mm (focus past the axis null), 50 % at 32.8 mm, 14.6 % at far plane (10 %/1 % contours do not fit
  the 12 mm box); r(10 %) 1.3–1.9 mm, r(1 %) 2.9–3.9 mm.
- Video (renderer v0.2, `--cusps`, K = 10): `…\results-attempt7-wall-budget-no-plateau\video\
  pic2d-results-attempt7-wall-budget-no-plateau-{n_e_per_m3,n_i_per_m3,phi_v,t_e_ev,ionization_rate_per_m3_s}.mp4`
  + `…-timeseries.html` (untracked).
- Attempt 8: RESUME from the 3.78 µs checkpoint; two independent scratch resumes bitwise identical (checkpoint hash
  `7b95f12a…`, 142/150 series fields; 8 peak-node diagnostics differ at 1 ULP). PID 51256, 09:51:07 AEST, frames ON
  (126 kept), `--wall-budget-seconds 50400`; 7.0–7.15 ms/step, GPU 100 %; ≥ 3 transits (9.3 µs) after +3.68 M steps
  ≈ 17:30 AEST; budget end ≈ 20:00 (~3.3–3.5 transits). Watcher loop restarted (also fires on process exit).
- Follow-up notes / risks:
  - Gate v2.0.2 (interval-averaged statistic from window/frame accumulators) + nvidia-smi cadence off the step
    loop: launched. Attempts 7–8 run v2.0.1 with the gate inert — must be disclosed in any admission.
  - Plume box is too short axially for the 10 %/1 % contours and beam divergence → extend the domain in the next
    model revision (cost: more nodes; check host/GPU memory).
  - Thrust numbers are development-only until the plateau rule holds; the paper must not quote them before then.
  - Concurrent host factorisations oversubscribe BLAS threads — run preflights sequentially.
  - Resume hygiene: while attempt 8 (PID 51256) steps, `run_state.json` still carries attempt 7's `finished: true`,
    `stop_reason: wall_clock_budget_reached` and `finalization_recovery` (checkpoint_step advancing 2,520,000 →
    2,560,000 proves the run is live). The resume path must reset `finished`/`stop_reason` and demote the recovery
    block to history; until then, watchers must key on the PID, not on `finished`.

## 2026-09-04 10:41 AEST — PIC model v2.0.2: live plume-boundary gate, background GPU sampler

- Task summary: replace the inert v2.0.1 gate and take nvidia-smi off the step loop. One commit `0251ff10`
  (`fix(pic2d)`), rebased onto `8556a401`, ff-push `8556a401..0251ff10`; main checkout pulled ff. 11 files:
  `pic2d/simulation.py`, `pic2d/warp_backend.py`, `experiments/pic2d_cft_steady_state_v1/run.py`, new
  `…/gpu_sampler.py`, `experiments/pic2d_cft_plume_v1/{protocol.json,README.md}`, `spec/pic2d/pic2d-model-v2.0.json`,
  tests `test_pic2d_v20_plume.py`, `test_pic2d_warp_parity.py`, `test_pic2d_steady_state_runner.py`, new
  `test_pic2d_gpu_sampler.py`.
- Gate: `far_field_window_sums()` (both backends) returns the far-field rows of the window sums Σn_e / Σn_i (the
  `d_n_e/d_n_i` device sums behind maps.npz and frames), accumulated step count and a reset generation; called from
  `_plume_record` at the series-record host sync only. `FarFieldChargeWindow` bridges the runner's 400,000-step
  resets (carry keyed on `diagnostic_generation`) and keeps a ring of totals → trailing window ≥ 400,000 steps
  (0.6 µs = 20 frames), an exact difference of two totals; denominator = record-mean of the instantaneous peak n_e.
  Enforced only when armed (2.4 µs) and the window is complete (first 0.6 µs after a resume recorded, not gated —
  disclosed). Floor ≥ 64,000 accumulated macro-particle-steps per node (32 × 2000). Calibration on attempt-6/7
  maps: 77 / 121 of 481 far-field nodes resolved, gate 0.0249 / 0.0339 (node (1,720)) vs 0.000 before; a one-frame
  window rejected (~1e-5 false-trip per window at the corner node × ~300 windows). Threshold 0.25 unchanged. Log
  column `q_far=<gate>(w<steps>/<resolved>n raw … dep …)`; series/summary/status carry raw window statistic, the
  v2.0.1 single-deposit witness and the window length.
- Sampler: `GpuUtilisationSampler` daemon thread, default 300 s (`--gpu-sample-interval-seconds`), shared last
  value, `float | None`; `summary.gpu_utilisation_sampler` records cadence/calls/failures. `step_graph` provenance
  → "lazy" before first capture / True / False.
- Tests: tests/pic2d 175 passed (CUDA, 152 s); ruff E4/E7/E9/F clean on touched files (6 pre-existing findings
  fixed). New: `…_fails_closed_on_a_sustained_far_field_charge_pile_up`,
  `…_ignores_single_deposit_shot_noise_on_the_axis_corner_node` (sitting + crossing ion),
  `…_statistic_is_live_on_a_uniform_quasi_neutral_plume`, `…_window_bridges_the_runner_accumulator_resets_and_
  restarts_on_load_state`, parity of the window statistic cpu/warp-cpu/cuda + "lazy", runner pins, 4 sampler tests
  incl. `test_a_hung_sampler_never_blocks_the_step_loop`. Parity maps now `allclose(rtol=1e-12)` (device atomics
  once accumulation is on).
- Attempt 8 (PID 51256) untouched, alive, keeps v2.0.1 by config identity (gate fields in `config_sha256`); v2.0.2
  applies to fresh starts — stated in spec, protocol, README.
- Follow-up notes / risks:
  - Attempts 7–8 ran with an inert boundary gate; the window-averaged statistic (0.035) shows the boundary was
    fine, but any admission must disclose it.
  - v2.1 prep launched: axial plume-box extension (decay fit from attempt-7 maps, cost table: nodes, particles,
    GPU/host memory, ms/step, factorisation) + resume state hygiene; no launch until attempt 8 ends (~20:00).

## 2026-09-04 11:24 AEST — PIC model v2.1 prepared (48 × 12 mm plume box), resume hygiene, exit-plane mask fix

- Commits (ff `0251ff10..ba57537f`; main checkout at `ba57537f`): `ce8628f4` fix(pic2d): resume/finalize demote
  the previous terminal state to `run_state.history`, write finished=false before the first step, `status()` reads
  run_state.json (was keyed on a stale summary.json); `1043f71d` feat(pic2d): v2.1 domain extension as configuration
  — `p2_plume_field_map` v2 extension (`extension_path`, hash-bound map, supported-box gate, `field_source`
  provenance; v1 provenance bit-for-bit), `spec/pic2d/p2-field-plume-extension-v2.json`, exit-plane column by index,
  protocol key `field_plume_extension` in `load_inputs`, `experiments/pic2d_cft_plume_v2_1/` (protocol, run.py,
  README; fresh identity; NOT launched), v2.0.x identities pinned (`1937f379…`, `4c969bff…`, v1 field `d30d2d24…`);
  `ba57537f` docs: `spec/pic2d/pic2d-model-v2.1.json` + README.
- Mesh bug found and fixed: exit-plane column selected by floating-point z (0.044/880 and 0.06/1200) misclassified
  node 480 and lost 180 plume cells; v2.0 masks unchanged (pinned), so no published result changed.
- Decay fit (attempt-7 maps, 3.0–3.6 µs): axis n_i 6.40e17 at exit → 2.57e18 at z = 27.45 mm → 47 % at 33 mm →
  14.6 % at 36 mm; exponential over six windows L = 2.57–3.07 mm (rms 0.04–0.14 in ln n) → z10 = 37.1–37.8 mm,
  z1 = 43.1–44.9 mm; conical power law about a scanned virtual origin z10 = 37.6–38.6, z1 = 45.6–57.9 mm (p = 2.35
  about z0 = 27 mm); all lower bounds (0 V plane inside the acceleration region). Radial: 10 %/1 % contours 0.8–1.6 /
  2.2–3.2 mm; far-plane current 50/90/95/99 % inside 1.4/4.0/5.6/9.3 mm (cones 6.7/18.4/25/37.8°); the 60° "95 %
  half-angle" is a side-wall population (6.8 % of crossings > 45° within 7 mm of the front face) → r_far stays 12 mm.
  Far-plane charge fraction 0.22/0.12/0.08/0.06/0.038/0.035 at 30–36 mm → ~0.005–0.01 expected at 48 mm.
- Proposal (z_far, r_far) = (48, 12) mm, L_plume = 24 mm = 1.0 L_channel, uniform 50 µm, 240 × 960, 135,540 plasma
  cells, 1.78 GB inverse blocks, ~12 min factorisation, 8.2 ms/step (+16 %), 3.81 µs/transit, 17.4 h (+~2 h particle
  trend) to 3 transits, ~8.8 GB GPU, host peak ~1.9 GB. Field: level-1 authority FEM box ends at 36.25 mm; using the
  qualification chain's `divergent-exit-stack.domain-padding-1.5` checkpoint (r ≤ 48.75, z ≤ 60.75 mm): channel
  agreement 0.74 mT max (gate 20 mT); v2.0 far plume carried a 15 % level-1 truncation at 36 mm (2.4 of 16.5 mT;
  padding-1.0 vs 1.5 agree to 0.38 mT); pole-face corner nodes differ up to 170 mT (level-0 vs level-1 mesh; recorded
  not gated); axis |B| 14/4.3/2.6/0.86 mT at 36/44/48/60 mm. Non-uniform spacing unsupported (scalar dr/dz in Grid2D,
  mesh conductances, cell_index, all Warp kernels); graded / channel-only-fine / two-domain costed (days–weeks).
- Cost table (z×r → ms/step → 3 transits → GPU): 36×12 7.1 → 12.2 h → 8.0 GB; 44×12 7.8 → 15.5 → 8.5; 48×12 8.2 →
  17.4 → 8.8; 48×16 9.9 → 20.9 → 9.4; 60×12 9.6 → 24.2 → 9.8; 60×24 15.9 → 39.8 → 12.7.
- Tests: `test_pic2d_v21_domain_extension.py` (17) + `test_resume_resets_the_terminal_state_and_keeps_the_previous_
  stop_in_history`; tests/pic2d 193 passed (CUDA, 204 s); ruff clean via `uvx ruff`. Attempt 8 (PID 51256) alive,
  untouched.
- Follow-up notes / risks:
  - Launch v2.1 (fresh identity) after attempt 8 ends (~20:00); ~17–20 h to 3 transits — schedule against the
    mini-sweep's GPU needs.
  - Pole-face corner-node field differences (170 mT) between mesh levels are recorded, not gated — check they do not
    touch the cathode flux tube or cusp planes before the v2.1 launch.
  - Mini-sweep preparation launched (designs, fields, closure targets, cost; draft dir; whole-set preflight).

## 2026-09-04 12:00 AEST — PIC attempt 8: grid-heating triad stop diagnosed as finite-grid-heating runaway

- Attempt 8 (resume, v2.0.1) ended 11:38 AEST cleanly (finalizer 3b8b577a worked; two `null` GPU samples):
  step 3,320,000 = 4.980 µs = 1.61 transits, 166 frames, 20,611 s cumulative wall (6,168 s this session, 7.7
  ms/step), `grid_heating_triad_gate_stopped_run`. Triad at stop (trailing 20 % = 3.98–4.98 µs): S drift +0.253
  (hard 0.25 → tripped); T_e,dense +0.155 (soft); ω_pe Δt +0.048; cumulative energy residual/electrode work
  +0.0857 (limit 0.10; +0.005 per 60 ns → ≈ 5.15 µs).
- Verdict: numerical (finite-grid heating). Residual per 0.4 µs segment / electrode work: −0.5 % (2.0–2.4 µs),
  +2.4, +5.8, +11.3, +15.3, +23.5, +37.0, +54.8 % (4.8–5.0 µs); residual power −9 → +719 mW; peak Δ/λ_D 3.23 →
  3.75. Attempt 8 alone +0.558 µJ on 1.844 µJ = 30 %. Last 0.4 µs: electrode +1373 mW, accounted sources −615 mW,
  residual +646 mW = 47 % of discharge power. Sign change at 2.0–2.4 µs when Δ/λ_D crossed ≈ 3.2 (CIC threshold π).
  Accepted channel-only plateau: Δ/λ_D 3.17 (1.64e18, T_e 7.4 eV, node (14, 286)), residual +0.4 % (1034 mW
  electrode, +4 mW residual) → the pair brackets the onset; the declared 4.5 gate is not protective. Corroboration
  3.78 → 4.98 µs: T_e,dense 9.3 → 10.4 eV, K_e/N_e 13.3 → 14.3 eV while I_d 5.58 → 4.40 mA; S/N_e constant
  3.77e10 s⁻¹; S/(N_e·n_g) +20 %; N_e 2.51 → 3.13 M; S 9.3 → 11.8e16; n_g 2.54 → 2.14e19 (gross utilisation
  1.02 → 1.31, recycling 73 % of feed); peak 3.08 → 3.54e18 at r 0.7 mm, z 14.4 mm; λ_D 12.9–13.4 µm. No plateau
  criterion met (I_d −29.8 %, N_e +21.9 %, n_g −22.2 %). Usability: heating from ≈ 2.4 µs, > 10 % of power from
  ≈ 3.2 µs; nothing after usable for thrust; attempt-7 development window (6–15 % residual) non-quotable.
- 6 mA vs 3.44 mA: cathode closure. Base injected fixed 3.00 mA at the exit plane (1.84 mA escaped, net 1.16);
  v2.0 emits I_d itself on the flux tube, uncapped; pre-onset (2.0–2.4 µs, residual ≈ 0) I_d already 6.0 mA,
  S 6.6e16 (1.7× base), N_e 1.46 M vs 1.0 M; not neutrals (n_g 2.8 vs 2.97e19), not anode fall (+25 vs +40 V).
- Record: `results-attempt8-grid-heating-triad-stop/` (13 files) + README + .gitignore → `ac248e05`, rebased onto
  `ba57537f`, ff-push `ba57537f..ac248e05`; main checkout at `ac248e05`. Video (renderer v0.2, K = 10, 6.5 % nodes
  resolved / 73 % of S): `…\results-attempt8-grid-heating-triad-stop\video\pic2d-results-attempt8-grid-heating-
  triad-stop-{n_e_per_m3,n_i_per_m3,phi_v,t_e_ev,ionization_rate_per_m3_s}.mp4` + `…-timeseries.html` (untracked).
  Development numbers (NOT usable): T_total 24.8 ± 0.3 µN, momentum closure +0.24 (balance no longer closes),
  I_beam 1.29 mA, IEDF 143/101 eV, half-angles 8/23/55°, Isp 139 s, axis n_i 26 % of exit at far plane.
- Resolution decision — nothing launched. From the max-record peak (3.69e18, 11.1 eV → λ_D 12.9 µm): CIC threshold
  with 20 % margin Δ ≤ 32.4 µm; ω_pe Δt ≤ 0.16 → Δt ≤ 1.48 ps. Cost to 3 transits: v2.1 48×12 at 50 µm 9.7 ms/step
  20.5 h 9.5 GB (heats); 40 µm 15.2 ms 32.2 h 12.5 GB (= π, no margin); 33.3 µm 22.4 ms 47.5 h (50.9 at 1.4 ps)
  16.6 GB; 25 µm 42.7 ms 90 h 28.8 GB; channel-only 3×24 at 33.3 µm 9.8 ms 13–14 h (≈ 6–7 at base load) 9.8 GB.
  (a) smaller Δt does not fix Δ/λ_D; (c) lower W statistics only; (d) implicit/energy-conserving scheme not
  available; (e) 50 µm resolved only if peak n_e ≤ 1.4e18 → current-limited cathode ~3 mA or lower ṁ.
- Adopted: PIC v2.0.3 gates (hard Δ/λ_D ≤ π on the interval-averaged peak, soft 2.5; windowed residual-power gate
  ≥ 5 % of electrode work) + preregistered channel-only 33 µm / 1.4 ps refinement of the accepted 3.44 mA plateau
  (acceptance: plateau + residual < 2 % + convergence vs 50 µm within tolerance, else the 50 µm plateau is
  re-classified "resolution-limited") — agent launched, also listing paper PIC claims needing amendment.
- Follow-up notes / risks:
  - Paper: any PIC-2D plateau claim, the Debye-gate 4.5 statement and any thrust/plume number need amendment or
    disclosure once the impact check reports.
  - Mini-sweep prep must assume ≤ 33 µm (channel-only ~13 h/design) or a lower operating point — feed this to the
    prep agent's cost table.
  - v2.1 plume run deferred: 47.5 h at 33 µm, or a lower operating point at 50 µm (peak n_e ≤ 1.4e18).

## 2026-09-04 12:48 AEST — mini-sweep v1 DRAFT landed; worktree cleanup; cloud 8×H100 kick-off

- Mini-sweep v1 DRAFT (`modern/experiments/pic2d_design_mini_sweep_v1/`, ff `ac248e05..6440518d`: `8704bf7c` feat,
  `805cd09e` data (LFS fields), `b4ddefff` docs, `6440518d` tests; tests/pic2d 206). Designs: divergent-exit-stack
  (reference, 3 cusps/4 cells, ρ 0.60), l1a-gs-v2-047 (low ρ 0.35 → 0.38 under iron; anode-side null moves −1.40 →
  −0.11 mm → anode-edge boundary cusp under the v3.1 0.25 mm rule, disclosed, substitute 061 named), l1a-gs-v3-009
  (mid ρ 0.90 → 0.92), l1a-gs-v3-056 (HEMP-like 1.99 → 2.36), optional l1a-gs-v3-106 (4 cusps, 2.56 → 2.93). Fields:
  L1b kept only bore samples of padding-0.5 domains → four new padded level-0 material-aware P2 solves (CPU
  sequential, 107–159 s each, RSS 503 MB, 278k–379k DOFs, residual ≤ 1.99e-10, min angles 22.7–32.4°), hash-bound;
  gates: cusp positions within 0.08–0.30 mm of L1a, |ΔB| vs L1b level-1 1.51 / 0.95 mT (gate 20 mT), ρ_iron within
  0.6 % of L1b. Closure targets → plasma-network v2 (per-cusp transit-loss p_k → `declared_cusp_probabilities` /
  `anode_cusp_probability`; sheath drop → `CuspSheathSpec.area_ratio`; leak-width FWHM → `leak_width_prefactor`; cell
  potentials → `PotentialClosure`; per-cell ionisation share, ion wall-loss fraction, I_d, S, utilisation). Preflight
  5/5 green (056 in the 24 mm box needs dt 1.3 ps: 0.821 T pole faces → ω_ce·dt 0.217; 047 cathode annulus r_outer
  0.9× → 24/24). Cost (3 transits, serial, four primary designs): channel-only 50 µm 12.7 h; 33 µm/1.4 ps 16.9 h;
  12 mm box 48.7 h; 24 mm box 69.0 h. Prereg blocked on the grid decision from the 33 µm refinement.
- Worktree cleanup (user request): 49 registered worktrees beside the repo; 41 clean+merged removed, 27 merged
  branches deleted (`exp/cft-orbit-wall-loss-v3` kept: on origin), `.worktrees/` gitignored (`25e86dca`); live
  worktrees `uni-project-cloud`, `-pic2d`, `-ss3-dev` left in place until their agents finish, then to be moved with
  `git worktree move` under `.worktrees/`. Stale dashboard preview servers (ports 8765/8766/8790) held two empty
  directories; killed; `uni-project-vizfix\modern` still held by an idle shell (delete later).
- Cloud: Lambda 8×H100 SXM (us-west-3, filesystem `h100-files`). SSH key generated `~/.ssh/lambda_h100`, public key
  registered in Lambda as `cft-key`; the instance first launched with `grabby` (private key not on this machine) →
  relaunch required. Kit (bootstrap, per-GPU concurrency benchmark, 8-GPU scheduler, PLAN.md) being written in
  `modern/tools/cloud/`.
- Host lag diagnosis: not this project — WSL runs `spacepdhcg gtoc12 cluster-fleet` (4 × 99 % CPU, budget to
  ~14:10), 2 × `pytest weldsim/tests`, and a `device_scvx_integration_test` on the GPU (~6 GB); vmmemWSL 8 GB.
- Follow-up notes / risks:
  - Move the three live worktrees under `.worktrees/` once their agents report; then `git worktree prune`.
  - Mini-sweep prereg: decide grid (50 vs 33 µm) from the refinement's convergence verdict; GPU model in the prereg
    must be the H100 if it runs on Lambda.
  - Delete `uni-project-vizfix` after the holding shell exits.

## 2026-09-04 13:19 AEST — PIC v2.0.3 gates merged; ss-v4 33 µm refinement preregistered and launched

- Commits (ff `25e86dca..a366e556`): `ceb9b172` v2.0.3 gates + tests + spec/protocol entries; `392129e5`
  preregister `pic2d_cft_steady_state_v4`; `a366e556` launch log. Main checkout ff → `a366e556`.
- v2.0.3 gates (`simulation.py`, runner): peak-Debye gate window mode — `PeakDebyeGateConfig(window_steps,
  soft_cells_per_debye, window_snapshot_steps)`, `PeakDebyeWindow`, `window_peak_debye`; statistic = interval-
  averaged peak from the electron window sums of the whole node map (Σn_e, Σw, Σw·v, Σw·v²; same accumulators as
  maps.npz/frames), carry across the 400k resets, ring of cumulative snapshots every 40,000 steps (~40 MB); peak =
  densest node with mean occupancy ≥ 32 macro-electrons; T_e = window moment temperature; hard π fail-closed once
  the window is complete; soft 2.5 recorded + plateau precondition (`plateau.peak_debye_soft_ok`), never a stop;
  single-step sample kept as witness. Window keys emitted only in window mode → v1.4 / v2.0.0–2.0.2 identities
  unchanged (`1937f379…`/`4c969bff…` reproduced); v2.0.3 identities `f7a4bedd…`/`e1377abd…` pinned. Windowed
  residual-power gate (`evaluate_triad`/`windowed_energy_residual`): trailing-400k ledger residual / electrode work,
  one-sided (positive = heating) ≥ 5 % → stop from the first complete window; cumulative ratio witness only (10 %
  soft). Calibration on accepted plateaus: base −12.7 → −0.2 % (max +0.37 %), seed-b −1.5 %, W×0.7 −4.2 %; on
  attempt 8 it would have fired ≈ 3.1 µs (actual 4.98). Tests `test_pic2d_v203_gates.py` (9): identity contract;
  window statistic == window maps' peak; sustained over-dense state trips only when the window completes; 1.6π
  single-record spike does not trip (0.95π averaged); continuity across resets / `load_state`; attempt-8-like ramp
  windowed 3.36 µs vs cumulative 6.3 µs; never fires on the real accepted base series; runner integration.
  tests/pic2d 220 passed (CUDA); ruff clean.
- Campaign ss-v4 (v3 dir exists, never launched, untouched): grid 90 × 720 over 3 × 24 mm → Δ = 33.33 µm (bore 60,
  cone start 540, exit 90 cells; 45,810 plasma cells, 65,611 nodes); Δt 1.4 ps (ω_pe Δt 0.101 at the v2 peak,
  Courant 0.50); W 26,666.7 (= 6e4/2.25, same particles/cell, ≈ 4.5 M at plateau); operating point / v1.3 closure /
  injection / seed bit-for-bit v2 attempt-2; frames ON (28 ns); plateau rule + v2.0.3 preconditions; budget 86,400 s.
  Acceptance: (a) plateau; (b) windowed residual < +2 %; (c) |Δ| ≤ 10 % I_d/S/utilisation/n_g/I_beam, ≤ 20 % peak
  n_e / T_e,peak (2× convergence-pair bands: seed-b I_d −0.1/−0.9 %, S −0.8 %, n_g +0.7 %, peak n_e −8.2 %; W×0.7
  I_d +5.7 %, S −4.6 %, n_g +4.0 %, peak n_e −11.9 %, T_e,peak −9.3 %); (d) verdicts converged / resolution_limited /
  refinement_heating / no_plateau. Expected Δ/λ_D at the v2 peak 2.11. Preflight (real P2, CUDA, production step):
  field 2.7 s, max |B| 0.291 T; factorisation 95–107 s; 2.54 ms/step at seed load (1.2 M), 4.36 ms/step at 4.5 M
  (0.565 ms/M) → 3 transits (5,142,858 steps) ≈ 6.2 h, v2 verdict time 7.68 µs ≈ 6.6 h; device +0.74/+1.38 GB, host
  0.73 GB. Shakedown 100,000 steps (6 min): 25 checkpoints, 2 window resets, 50 frames, finalize + assess, gate
  enforced 301/500 records, verdict no_plateau as expected. Launch discipline: clean worktree, `--expect-commit`,
  protocol blob == HEAD, preflight/shakedown present, O_EXCL `results/execution-lock.json`.
- Launch 1: worktree `uni-project-pic2d-ss3` (detached at 392129e5; beside the repo — move under `.worktrees/`
  after the run), PID 18068, 13:11:55 AEST; 2.50 ms/step over 61,200 steps, GPU 99 %; ignition follows v2 (I_d
  1.25 → 1.5 mA, n_g 5.5 → 4.5e19); windowed residual −8.3 %. Verdict ≈ 18:45–19:30 AEST; budget end ≈ 13:15 5 Sep.
  Watcher keyed on PID 18068.
- Paper impact: no PIC-2D result admitted (no PIC claim in claims.json; GATE-L3 planned/closed; no plateau /
  Debye-4.5 / thrust number in sections or macros) → nothing to retract. `\CtvPTwoPicPlanesMm` (6.00/12.00/17.95 mm)
  is a field-map descriptor (B_z sign changes of the sampled P2 map), unaffected; optional wording "field-map
  descriptor on the kinetic workstream's grid". For a future PIC admission: GATE-L3 evidence must add windowed
  residual power + Δ/λ_D ≤ π convergence check; the 50 µm plateau admission must disclose the uncalibrated 4.5
  gate (3.17 on the threshold, residual +0.4 %) and cite ss-v4 as its convergence test; attempt-7/8 thrust numbers
  never quotable.
- Housekeeping: idle `uni-project-pic2d` moved to `uni-project/.worktrees/pic2d` (`git worktree move`); `-cloud`
  (agent active) and `-pic2d-ss3` (live run) remain beside the repo for now.
- Follow-up notes / risks:
  - `preflight.json` / `shakedown.json` record `git_head ac248e05` (pre-commit working tree; disclosed
    non-evidentiary).
  - A host `pytest tests/tools` process (PID 22788) belongs to the cloud-kit agent.

## 2026-09-04 13:23 AEST — Lambda cloud kit landed (`8fe5d00c`)

- `modern/tools/cloud/`: `bootstrap_lambda.sh` (idempotent; `EXPECTED_GPUS` default 8, `TEST_GPU` 0; fails closed
  on GPU count mismatch, driver < 525, or an LFS pointer where the P2 level-1 arrays should be; https token via a
  call-time credential helper or ssh deploy key — no secret on disk; uv-managed Python 3.12 venv; Warp smoke;
  `pytest tests/pic2d -x -q` on GPU 0 timed; `$WORK/provision.log`), `requirements-pic.txt` (no lock in repo → pins
  mirror the local anchor env: `warp-lang==1.14.0` PyPI wheel = CUDA 12.9 build, NVRTC embedded, links only
  `libcuda.so.1` → any driver ≥ 525, no toolkit; `numpy==2.5.2`, `pytest==9.1.1`, `pillow==12.3.0`, `pyyaml==6.0.3`;
  resolves with `uv pip compile` for linux/py3.12), `bench_gpu_concurrency.py` + `bench.sh` (N = 1/2/4 per GPU,
  runner construction path, file barrier, timing via the v4 preflight's `_time_steps`, nvidia-smi + Warp mempool
  memory, `factorise` mode with BLAS threads = floor(CPUs/N); JSON + markdown; 5090 anchors in a registry),
  `schedule.py` + `jobs.yaml` (slots, tmux/setsid wrapper, `state.json` pid/GPU index/name/UUID/start/end/exit,
  Warp `cuda:0` UUID cross-checked against nvidia-smi under `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `merge-base
  --is-ancestor` + byte-identical protocol checks, per-job detached worktrees, `status` with steps/t/transits/
  ms-step/ETA/budget/stop_reason), `PLAN.md`; tests `modern/tests/tools/` 28 pass; bash -n / py_compile / ruff clean.
- PLAN job map (5090 rate; cost = box makespan × $24/h for 8×): (i) `ss33-seed-b` 6.2–6.6 h, `ss33-w-0.7` 7.7–8.2 h
  (v4 measured 4.36 ms/step at 4.5 M → 6–7 h, not 13–14; v4 `launch` has no `--case`, replications need their own
  prereg); (ii) `ss25-base` 15–16 h (v4-calibrated) to 35–37 h (README model) — needs a 25 µm sibling protocol;
  (iii) mini-sweep ×4 = 16.9 h serial (draft, not preregistered, `--allow-launch` guard); (iv) `plume-v2.1-33um`
  51–52 h, `plume-v2.1-50um-3mA` 17–20 h (protocol commits needed). Totals ~110–130 GPU-h; full makespan 52 h =
  $1,250 at ~30 % utilisation (set by the 33 µm plume run); without it 20–37 h.
- Assumptions to verify on first login: driver ≥ 525, MIG off, glibc ≥ 2.28, passwordless sudo, persistent FS path,
  tmux/git-lfs, Warp `cuda:0` UUID == nvidia-smi's; the runner's `gpu_utilisation` samples are box-wide first-GPU
  readings — use `state.json`/`status` for the pinned GPU.
- Context change: the user launched a SINGLE H100 SXM (us-southeast-1, key `cft-key`) first, to benchmark before
  scaling; `EXPECTED_GPUS=1` for that box. Cloud worktree removed by the agent; main checkout ff → `8fe5d00c`.

## 2026-09-04 14:35 AEST — H100 provisioned and benchmarked; cross-platform test pins found

- Box: `ubuntu@68.209.75.2` (gpu_1x_h100_sxm5, us-southeast-1), H100 80GB HBM3, driver 580.105.08 (CUDA 13.0),
  26 vCPU, 221 GiB RAM, 2.7 TB disk, `/lambda/nfs/h100-files` mounted. Key `~/.ssh/lambda_h100` (Lambda `cft-key`).
  Deploy key `~/.ssh/open_cft_deploy` on the box = GitHub deploy key `lambda-deploy` (write) on
  terrorproforma/open-cft; `~/.ssh/config` pins it for github.com. Bootstrap (`EXPECTED_GPUS=1`, WORK=
  `/lambda/nfs/h100-files/cft`) → clone at `8fe5d00c` (later pulled to `af9e79d1`), uv Python 3.12.14, warp-lang
  1.14.0 (CUDA 12.9 build, toolkit 12.8 on image), numpy 2.5.2 (scipy-openblas 0.3.34 Haswell DYNAMIC_ARCH),
  `provision.log`. Bootstrap's `pytest -x` stopped on `test_discrete_gauss_law_with_volume_and_surface_charge`:
  one node with source 4.3e-18 (700× below the peak) at 1.07e-9 relative, |residual| 4.6e-27, max|err|/max|s| = 1e-11
  → scale-aware bound (`af9e79d1`, pushed; box pulled). Full suite on the H100: 216 passed, 2 failed, 2 skipped
  (node missing for JS syntax checks): (1) `test_generation_is_byte_deterministic_and_checked_html_is_current` —
  within-platform determinism holds, checked-in Windows HTML differs at last digits in numeric payload arrays;
  (2) `test_v2_field_extension_serves_the_48_mm_box…` — v1 sampled field-map hash `d30d2d24…` (Windows) is
  `1f124047…` on Linux. Fix + binding audit agent launched (file-byte bindings + tolerance gate on derived maps).
- Benchmarks (`modern/tools/cloud/bench.sh`, 400 warm-up + 2000 timed steps, production load; reports in
  `/lambda/nfs/h100-files/cft/bench*`): per-process H100 vs 5090 — channel-50 1.71 ms (1.16×), channel-33 3.31 ms
  (1.32×), plume-v2.0-50 7.20 ms (0.98×); seed load 1.4–2.1×. Without MPS N = 2/4 aggregate 0.93–0.96× (time-
  slicing). With CUDA MPS: channel-33 N=1/2/4/8 → 3.37 / 5.45 / 8.71 / 17.1 ms per process, aggregate 1.00 / 1.24 /
  1.54 / 1.58×; plume-v2.0 7.23 / 11.66 / 18.96 / 37.6 ms, aggregate 1.00 / 1.24 / 1.52 / 1.54× → `slots_per_gpu = 4`
  under MPS ≈ 2 5090-equivalents per H100, each job at ~0.5× 5090 speed; no single run gets faster. GPU memory per
  process 1.1 / 1.8 / 3.3 GB (channel-50 / channel-33 / plume-v2.0). Host factorisation on the box: 0.4 s
  (channel-50), 1.3 s (channel-33), 3.3 s (plume-v2.0), 5.8 s (plume-v2.1) at 16 threads; 8 concurrent ≤ 16 s —
  vs 5–12 min on the Windows PC (local slowness is a Windows/BLAS issue).
- Implication: the cloud buys parallel slots, not wall-clock per run. Recommended use: 1–2 H100 with MPS 4 slots
  for the mini-sweep (4 designs in parallel ≈ 12–14 h) and the 33 µm replications / 25 µm ladder; the 33 µm plume
  run (≈ 48 h) is no faster on an H100 than on the local 5090. 8× H100 at $24/h not justified by throughput.
- Local ss-v4 refinement: 2.47 µs at 1.41 h, 3.3 ms/step (particles growing), I_d ≈ 3 mA, peak 9e17, 1.7 cells/λ_D,
  windowed residual −10.6 %, ω_pe Δt 0.09 — healthy.
- Follow-up notes / risks:
  - Box idles at $3.29/h until the binding audit lands and the mini-sweep is preregistered for the H100 (GPU model
    in the prereg; MPS does not change a process's own kernel order → same-seed determinism should hold; verify
    with a scratch replay before the prereg).
  - Under MPS the runner's `gpu_utilisation` sampler reads the whole GPU.
  - `h100-files` is in the same region after all (mounted) — the earlier region concern was wrong.

## 2026-09-04 16:03 AEST — cross-platform bindings (79c2a3f8); mini-sweep v1 preregistered and running on the H100

- Cross-platform fix (ff `af9e79d1..79c2a3f8`): `0ac8d9b8` `fields.py` `MagneticFieldMap.source_sha256` = content
  hash of grid + declared provenance with CPU-derived blocks removed (`channel_cross_check`, `withheld_midcell_error`,
  `node_reference_b_max_abs_error_t`, `certificate`, `certified_max_b_t`); `compare_field_arrays()` with
  `FIELD_REPLAY_RTOL = 1e-12`, `FIELD_REPLAY_ATOL_OVER_MAX_B = 1e-12`; `artifacts.save_checkpoint(field=…)` binds
  `field_source_sha256` + writes `<name>.field.npz` anchor; `load_checkpoint` fail-closed on source identity, then
  `bitwise` (content hash equal) or `numerical` (allclose to anchor) mode, recorded in `session.field_identity`;
  `runtime_identity()` records `platform_fingerprint` (os, libc, numpy, compiler, SIMD baseline/dispatch, BLAS
  name/version/runtime core via `scipy_openblas_get_corename64_`, cpu model) + `gpu` (name/arch/UUID/driver/toolkit
  from the initialised Warp runtime). `79c2a3f8`: dashboard anchor sidecar `pic2d-cft-steady-state.anchor-platform
  .json` (HTML sha + fingerprint) → byte-exact on the anchor, else structural compare (numeric leaves within one unit
  of the last recorded significant digit, 1e-9 floor); v2.1 coarse field pinned by `V20_COARSE_FIELD_SOURCE_SHA256
  = 577943ec…` + anchor `tests/pic2d/anchors/p2-plume-field-v1-coarse-60x72.npz` (rel ≤ 1e-12); Linux hash
  `1f124047…` recorded. Audit: no launcher verifies a CPU-derived hash (`config_sha256` from protocol scalars
  reproduced on Linux; P2 bundle / cross-section / code hashes are file-byte); only cross-box resume was blocked.
  Local anchor fingerprint `f554af1c…` (Windows 11, numpy 2.5.2 msvc 19.44, X86_V3, OpenBLAS 0.3.34 Haswell). tests/
  pic2d 228 (5090). Operator notes in PLAN.md §6 (checkpoints before 0ac8d9b8 have no anchor → not resumable on the
  H100; regenerating a dashboard on the box makes it the anchor).
- Mini-sweep v1 (ff `79c2a3f8..1506f219`): `412d240f` composer `channel-33um` on the ss-v4 template + compose /
  launch / shakedown / mps-replay / assess stages (`--grid`); `8b21b868` budget block fix (`n_max_per_m3`,
  `n_eq_projected_per_m3` missing → KeyError at finalization, found by the box shakedown); `c51b6ea3` / `7717062b`
  replay criterion (physics bitwise, float-atomic diagnostics ≤ 1e-6); `c6a58172` test helper; **`291a9227` PREREG**
  (`protocol.json` v1.0.0, 6 sealed run protocols, `preflight-channel-33um.json`, `shakedown-channel-33um.json`,
  `mps-replay.json`, README §7–8); `9c426f90` jobs.yaml (single H100, `slots_per_gpu 4`, MPS env, 4 jobs enabled at
  291a9227); `a20ec2fa` launch log; `1506f219` tests/tools pin. Designs (Δt 1.4 ps, v1.3 closure + v2.0.3 gates,
  frames ON, seed 20260903): divergent-exit-stack ρ 0.60 90 × 720 W 26,666.7 budget 18.8 h PID 19764;
  l1a-gs-v3-056 ρ 2.36 115 × 512 W 26,566.8 budget 25.3 h PID 19913; l1a-gs-v2-047 ρ 0.38 66 × 778 W 26,655.3
  budget 13.3 h PID 20079; l1a-gs-v3-009 ρ 0.92 80 × 684 W 26,799.2 budget 23.0 h PID 20189. Launched 05:52–05:56
  UTC (15:52–15:56 AEST); early 3.7–5.7 ms/step at seed load; GPU 100 %, 5.0 GB; all ignited (I_d 1.3–2.2 mA at
  0.1–0.25 µs). 3-transit ETAs (MPS-4 upper bounds): 047 ~01:00, ref ~04:30, 009 ~07:00, 056 ~08:40 AEST 5 Sep.
  Preflight 5/5 (ω_ce·Δt 0.026–0.079 → 1.4 ps admissible for all; 056's 1.3 ps was a plume-box property); shakedown
  056 100,000 steps 283 s incl. finalize + assess (Kornfeld p 0.136/0.096/0.071, non-evidentiary); MPS replay:
  physics bitwise, float-atomic diagnostics ≤ 2.2e-13 (same solo-vs-solo). Draft changes: ss-v4 template (v1.3, no
  recycling) for comparability with the only 33 µm reference; grid target 24 mm/720; W parity 6e4·dr·dz/(50 µm)²;
  cap 12 M; 047 kept (anode-edge loss reported as `anode_edge_electron_wall_current_a`), 061 rejected (taper
  confounder); 056 seed replicate + 106 sealed, not launched; ss-v4 verdict a predeclared per-design caveat
  (`assess` reads v4's assessment.json when present).
- Follow-up notes / risks:
  - Xid 31 under MPS when a PIC process was killed mid-step → sibling clients torn down; add a SIGTERM handler to
    the shared runner; never kill sweep processes; check `dmesg | grep Xid` before relaunch.
  - Slots free up in order 047 → ref → 009 → 056; queue the 056 seed replicate / 106 / 33 µm replications there.
  - Interim visualisation agent running (comparison panel + per-design videos from frames; media under
    `.worktrees/interim-sweep-media/`).

## 2026-09-04 16:40 AEST — interim visualisation of the four H100 mini-sweep runs (`5da74ee6`)

- `modern/visualization/interim_sweep_panel.py` (`manifest` / `status` / `render`) + `tests/visualization/
  test_interim_sweep_panel.py` (6: truncated mid-write frame skipped, gap ends staging, results dir byte-identical
  afterwards, partial status line tolerated, players validate with the design's cusps, byte-deterministic PNGs);
  ff-push `1506f219..5da74ee6`; box and main checkout at `5da74ee6`. Stages symlinked frame mirrors + a synthesised
  `summary.json` (grid/W/dt from `protocol.json`) so the untouched v0.2 renderer works on running jobs. Box rerun:
  `bash /lambda/nfs/h100-files/cft/interim/rerender.sh` (`--no-videos` for PNGs only).
- Media (gitignored): `.worktrees/interim-sweep-media/interim-sweep-panel.png` (4 designs × n_i / φ / windowed
  ionisation at the latest frame, each design's own cusp planes; shared scales n_i 4.6e14–4.6e17, φ −28.8…318.9 V,
  ionisation 5.0e22–8.2e23 with K = 20–28 frames = 560–784 ns), `interim-sweep-timeseries.png` (I_d, S, N_e, peak
  Δ/λ_D vs t), 8 MP4s (n_i + ionisation per design, INTERIM banner), `interim-sweep-report.json`.
- Status at 06:33 UTC (~40 min in): 047 t 0.845 µs 0.326 transits 4.00 ms/step I_d 1.28 mA S 1.06e16 N_e 0.39 M
  n_g 4.29e19 Δ/λ_D 1.22|0.81 res_w −8.2 % T_e 7.4 eV; ref 0.799 µs 0.333 4.88 ms 1.33 mA 1.85e16 0.74 M 4.41e19
  1.31|0.78 −14.4 % 6.9 eV; 009 0.669 µs 0.294 4.88 ms 2.39 mA 2.92e16 0.96 M 3.31e19 1.04|0.62 −9.9 % 9.4 eV;
  056 0.579 µs 0.339 6.27 ms 2.90 mA 3.64e16 1.76 M 3.97e19 1.29|0.72 −15.6 % 7.0 eV. No hard failures; triad
  unenforced (windows incomplete); ms/step rising with N_e → ETAs later than the 06:07 estimate.
- Physics (ignition phase, not plateau): 047 four broad flat cells, axis-hugging n_i, small near-axis flames at each
  cusp, anode cell unresolved; ref three U-cells, flames at 6.03/12.0/17.97 mm brightest at 12.0/17.97 (as plume
  attempt 6); 009 thin curved separatrix sheets, ionisation concentrated in the exit partial cell (17–22.8 mm);
  056 sharpest sheets, axis-attached weak flames at 4.31/8.53 mm, exit cell (12.74–17.05 mm) ionises across the
  bore and dominates S. Trend: cusp-anchored axial flames (low ρ) → exit-cell body (high ρ); I_d and S rise
  monotonically with ρ (partly bore-volume). φ staircase anode→exit in all, plasma ~19 V above the 300 V anode.
- Follow-up notes / risks:
  - Refresh the panel every few hours (`rerender.sh`); final dashboard after the 3-transit records.
  - `_MASKS_CACHE` keyed on `id(grid)` in the renderer — fix to value-keyed when next touched.

## 2026-09-04 17:44 AEST — external validation v0 (code-to-code vs Brandt 2016) prepared as a DRAFT

- Commits (ff `5da74ee6..7fa9e6c6`; branch `exp/external-validation-v0` also pushed; main checkout ff): `645c7de4`
  package `modern/experiments/pic2d_external_validation_v0/` (reference, geometry, fields, comparison, protocol,
  preflight, run); `e7c5f017` data (LFS P2 checkpoint + `binding.json`, `comparison-spec.json`, 3 draft protocols,
  `preflight-channel-20um.json`, `protocol.json`, `.gitignore` negation); `4ddfc319` README; `7fa9e6c6` 18 tests.
  ruff clean; CPU solves 171 s + 54 s, RSS ≤ 150 MB.
- Reference: Brandt, Schneider, Duras, Kahnfeld, Hey, Kersten, Jansen, Braxmaier (2016), Trans. JSASS Aerospace
  Tech. Japan 14(ists30) Pb_235–242, doi:10.2322/tastj.14.Pb_235 (Crossref-verified, full text read); magnet stack
  from the Kiel 2017 thesis (urn:nbn:de:gbv:8-diss-224024), whose second run (4.7 mA / 60°) vs the paper (4.3 mA /
  50°) gives the reference's own reproducibility. Rejected alternatives (with DOIs in the README): Matyash 2010
  (DM3a), Matthias 2019 (own lineage), Kahnfeld 2018 (breathing), Lewerentz 2022 (MS4), Keller 2015 (experiment →
  v2's target).
- Setup (README §2, page/figure-sourced): channel 14 × 1.5 mm; Al₂O₃ 1.5–2.5 mm ε_r 9; grounded body 2.5–5.12 mm;
  anode 400 V; 0.27 sccm = 1.1e17/s; static DSMC neutrals mean 2e20 (6e20 → 1e20), 500 K; rim electron source
  17.55 mA of which ~1.8–1.95 mA reach the channel; 3 SmCo rings 5 mm × r 2.5–15 mm + 5 iron rings 0.5 mm × r
  2.5–8 mm, anode at the mid-plane of magnet 1; anchors 0.6 T at (0, 11 mm), ~0.7 T max, 0.05 T at (0, 17 mm), exit
  null ~16 mm; their grid 1024 × 256 at 20 µm, Δt 3.17 ps, 2.4e7 steps to 76 µs, 1:2618 in a factor-4/8 self-similar
  scaled system. Reported D: I_a 4.3 mA, net ionisation 24 %, I_beam 2.5 mA, ~5 V anode-cell potential, cusp drops
  ~10 / ~5 V, n_i ~1e19 (figure), wall ion 160 eV / 640 A/m² (figures), plume peak 50° (figure), ~200 eV near the
  exit cusp (figure).
- Geometry mapping (v1.1, approximations A1–A9): axial offset 2.5 mm exact; interior rings only; ring radius forced
  to 15 mm; linear µr 4000 iron; dielectric 0.9 mm (clearance rule, µr 1); µr-1 yoke placeholder; remanence 1.05 T
  post-scaled to the 0.6 T anchor; channel-only PIC box (plume 1024 × 256 costed only). Field (415,859 DOFs, min
  angle 31.7°, residual 1.99e-10): G1 scale 1.0814 → 1.135 T; G2 nulls 2.634/8.366 mm vs rings 2.75/8.25; G3 exit
  null 15.85 mm; G4 0.053 T at 17 mm; G5 maxima 0.698 T at 5.55 mm, 0.601 T at 11.15 mm; G7 pass. Genealogy: G5
  ("maximum at 11 mm") and G6 ("wall cusp field 0.10–0.35 T") misread the anchors → revised before composition, originals
  kept in `binding.json`. No-ring bracket: nulls move 0.12 mm, wall cusp field 0.49 → 0.40 T.
- Protocol (ss-v4 template): primary `channel-20um` 75 × 700, Δt 0.7 ps, static Xe 2e20 / 500 K, 400 V, 1.8 mA / 1 eV
  exit-plane injection, seed 5e16, v2.0.3 gates, frames 28 ns, W 82,466.8 (parity would be 103 M particles → 12 M cap,
  disclosed), transit 1.4 µs, 6.0 M steps, 18.3 ms/step at MPS-4 → 30.6 h (11.9 h solo), budget 46 h, 17.4 GB. Drafts
  `bohm-0.4` (v1.4 hook, α = the reference's D⊥) and `15um` (69.5 h). 33 µm inadmissible a priori (4.48 cells/λ_D at
  1e19 / 10 eV > π); 20 µm at 2.69 (between soft 2.5 and hard π); 15 µm meets the soft margin.
- Comparison spec (ASME V&V 20, k = 2): 12 quantities, 10 comparable channel-only; u_D from stated precision +
  digitisation + paper-vs-thesis spread; u_num from the seed-b / W×0.7 bands (5.7 % currents, 4.6 % S, 11.9 % peak
  n_e, 9.3 % T_e; 2 V potential steps); u_input (B ±8 %, neutral profile, source ±0.15 mA) declared, not propagated →
  every row conditional; tolerances 20 % / ±5 V / 0.3 dex; potential-step rows non-discriminating (2u_val 6.4 V > 5 V).
- Preflight 3/3 options PASS (14 s): hashes, regate, grid, node map (0.772 T max, ω_ce Δt 0.095, ω_pe Δt 0.125,
  Courant 0.42), masks, `runner.build_config` static neutrals, comparison spec, cost.
- Inconclusiveness declared: hard peak-Debye stop (envelope 1.36e19 × T_e/10 eV on 20 µm — most likely, since our
  closure confines better); no plateau in 46 h; plateau between soft/hard; residual ≥ 2 %; no ignition (no
  adjustment allowed); a field gate failing; discrepancies all in the closure-predicted direction (→ bohm-0.4
  discriminates); rows where the reference spread exceeds the tolerance; plume-only rows not compared.
- Follow-up notes / risks:
  - Launch needs a box preflight + shakedown + prereg commit; `run.py launch` refuses until then. Slot plan: after
    047 frees (~01:00) or solo after the sweep (~09:00 → done ~21:00 5 Sep).
  - Root `*.npz` ignore dropped an LFS sidecar once — verify `git lfs ls-files` after data commits.

## 2026-09-04 19:44 AEST — ss-v4 verdict `resolution_limited`; v5 25 µm ladder point launched

- ss-v4 launch 1 (PID 18068) finished 18:17 AEST: `plateau_reached_after_min_transit_times` at step 5,200,000 =
  7.28 µs = 3.03 ion transits (runner uses `budget_v1_3.ion_transit_time_s = 2.4 µs`, the measured v2 residence
  time, evaluated at 40,000-step checkpoints; 3 transits = 5,142,858 steps), wall 18,013 s, 4.05 ms/step at the
  end, no finalization_error.
- Assessment (`results/assessment.json`, reference consistency 7/7): (a) plateau ✓ (drifts I_d +3.0 %, N_e +4.9 %,
  n_g −0.5 %; triad soft ok); (b) windowed residual −7.67 % < +2 % ✓ (cumulative −9.1 %; base +0.37 %); (c)
  convergence vs the 50 µm base (24ab82f4) — I_d 3.801 vs 3.444 mA **+10.35 %** (tol 10, NO), I_beam 2.459 vs 2.291
  +7.34 %, S 3.595 vs 3.930e16 −8.52 %, utilisation 0.4204 vs 0.4596 −8.52 %, n_g 3.188 vs 2.973e19 +7.25 %, peak n_e
  1.287 vs 1.637e18 **−21.4 %** (tol 20, NO), T_e,peak 5.577 vs 7.387 eV **−24.5 %** (tol 20, NO); bands seed-b
  (−0.08 / +0.68 / −0.80 / −0.80 / +0.73 / −8.19 / −1.1 %) and W×0.7 (+5.68 / +3.55 / −4.64 / −4.64 / +3.95 / −11.89 /
  −9.3 %); (d) verdict **resolution_limited** — the 50 µm plateau carries the label; v4 supersedes I_d, peak n_e,
  T_e,peak. Δ/λ_D window 2.15 (trailing mean 2.10, 0 records > 2.5, single-step max 2.46) → `peak_debye_soft_ok`;
  same peak location (r 0.67, z 14.3 mm). Shifts point in the W×0.7 direction at ≈ 2× its size → grid and W effects
  entangled; the protocol's W-only follow-up stays open.
- Commits (ff `7fa9e6c6..427c7918`): `0d228ad2` results-only record + README verdict (PIC results are plain git;
  56 MB status.jsonl triggers GitHub's size warning); `abac6d9e` `modern/visualization/pic2d-cft-steady-state-v4.html`
  (2.9 MB, offline, node-checked, anchor sidecar regenerated locally; verdict pill, acceptance (a)–(d), convergence
  table with bands, time series incl. windowed residual recomputed for all four runs to 1e-12, Δ/λ_D window +
  witness, ω_pe Δt, peak profiles; 8 tests); `0e09e749` renderer v0.2.1 (HTML player refused a preregistered run:
  hard-coded development status) + video `…\.worktrees\pic2d-ss3\…\results\video\pic2d-pic2d_cft_steady_state_v4-
  {n_e_per_m3,n_i_per_m3,phi_v,t_e_ev,ionization_rate_per_m3_s}.mp4` + `-timeseries.html` (untracked); `69ff435d`
  PREREG `pic2d_cft_steady_state_v5`; `427c7918` launch log.
- v5 (25 µm / 1.0 ps ladder point): grid 120 × 960 (bore 80, cone 720, exit 120), W 15,000 (parity), Δt 1.0 ps
  (ω_pe Δt 0.064, Courant 0.47), v2.0.3 gates verbatim, primary reference = v4 33 µm plateau (same 10/20 %
  tolerances), 50 µm base reported not judged; verdicts converged / resolution_limited (→ W-only / operating-point
  follow-up) / refinement_heating / no_plateau; expected Δ/λ_D 1.62; 5 tests; tests/pic2d 267, tests/visualization
  160. Preflight under GPU contention (the hybrid-l2 agent had 11 `warp-cuda:0` processes at 100 % since 18:13):
  17.4 ms/step at 8 M particles contended, solo estimate 7.3–7.8 ms → ~15 h; budget 48 h; shakedown 100k steps
  through finalize + assess OK. Launch 1: PID 43572, 19:29:53 AEST, `.worktrees/pic2d-ss5`, lock at 69ff435d; at
  61,000 steps 3.9 ms/step (contention easing). Verdict ≈ 10:15–11:15 AEST 5 Sep solo (worst case 06:30 6 Sep);
  budget end ≈ 19:40 6 Sep.
- Coordinator action: interrupted the hybrid-l2 agent (CPU only; stop CUDA processes at the next checkpoint; record
  the contention). `uni-project-pic2d-ss3` moved to `.worktrees/pic2d-ss3`; projects folder now clean except the
  empty `uni-project-vizfix` held by an idle shell.
- Paper (report only): a first PIC admission = `numerical-campaign` gate (e.g. GATE-PIC2D-STEADY-STATE-V4 under
  GATE-L3) binding the v4 bundle + v2 base + pair; claims: one-shot preregistered 33 µm plateau at 3.03 transits;
  plateau values with bands; the recorded resolution-limited statement (33 µm convergence untested until v5); the
  uncalibrated 4.5 gate disclosure + v2.0.3 replacement; the attempt-8 heating diagnosis. Disallowed: thrust / Isp /
  efficiency / plume numbers, experimental validation, any "converged" wording before v5.
- Follow-up notes / risks:
  - W-only refinement (same grid, W×0.5) to disentangle grid from particle-weight effects — queue on a free slot.
  - Ask the mini-sweep `assess` to cite `resolution_limited` for the reference (it reads v4's assessment.json).

## 2026-09-04 21:00 AEST — hybrid L2 v2 PARKED at the user's request (branch only)

- The revival ran a CPU-heavy queue (and, per the v5 preflight, eleven concurrent processes 18:13–19:4x) on the
  user's PC while the preregistered v5 PIC run executed; the user cancelled it. Workers (4 launch cases + a
  re-spawning queue runner) killed by the coordinator and the agent; 0 remain; PIC v5 (PID 43572) untouched, now
  100 % GPU alone, 9.35 ms/step at 0.6 µs (particles growing).
- Record: branch `feat/hybrid-l2-v2` @ `277fc911` pushed (NOT merged): `cft_revival.hybrid` L2 v2 (cells, rates,
  PB solver, ions, l2, checkpoint v2, gates), tests, `docs/hybrid-l2-v2.md`, protocol/preflight/shakedown, runner
  (STOP file + resume), base-case result + summaries of seed-b / spatial-coarse / spatial-fine / temporal-coarse,
  dashboard `modern/visualization/hybrid-l2-v2.html` (verdict `not_evaluable`), README: development model, NOT
  admitted, PARKED. Comparison FAIL 24/28 outside tolerance — I_d 7.52 vs 3.44 mA (+98 % vs v4's 3.80), anode ion
  fraction 0.155 vs 0.014, peak n_e ×9, n_g −43 %, cusp-2 near-wall drop wrong sign; ladder at kill: spatial 3/3,
  temporal 2/3, statistical 0/2, closure 0/4; PIC/L2 wall-clock 1.66 (no speed advantage). Diagnosis (untested):
  the cusp conductance G_k extracted from the PIC plateau is linear and lacks sheath-limited saturation, so the
  anode-side cell at 358 V (PIC 309 V) draws ~2× the electron current through the cusps and the surplus ions exit
  via the anode; over-wide leak half-widths w_k the second suspect.
- Decision: L2 parked; closure route = PIC design sweep → plasma-network v2 calibration → MDO v3. Worktree removed.
- Follow-up notes / risks:
  - Heavy compute must go to the Lambda box; the local PC hosts at most one preregistered GPU run.
  - The roadmap row `hybrid` should read PARKED (failed comparison recorded on a branch), not RUNNING.

## 2026-09-04 22:25 AEST — PIC focus: acceleration literature review + code performance audit; perf work launched

- User direction (21:23): 0-D development dropped; focus on the full PIC; all PIC runs on the Lambda H100. Before
  the plume run: review the PIC code and the literature for dramatic speed-ups.
- Literature review `modern/docs/literature/pic-acceleration-methods.md` (`4e953d68`, 147 verified refs, indexed
  in LITERATURE_SYNTHESIS.md §0): ranked table — (1) Warp GMG Poisson 1.3× channel / 2.2× plume, no physics risk;
  (2) kernel fusion + cell sort 1.2–1.4×; (3) explicit energy-conserving (Lewis 1970; Barnes–Chacón 2021; Powis–
  Kaganovich 2024) gather → 33→50 µm legal, ~2× / ~2.5×, per-cusp sheath drops (1–3 cells) + momentum conservation
  at risk; (4) coarse-to-fine restart ≤ 1.5×; (5) Barnes/ECsim semi-implicit 2×/2× → 3–3.5× (4×/4× ~8–10×; Marks &
  Gorodetsky 2025 WarpX Hall-thruster precedent), T_e and cusp sheaths unresolved by construction; (6) particle
  merging 1.2–1.4× plume; (7) mixed precision 1.1–1.3×; (8) binomial filtering; (9) permittivity scaling γ = 4 8×
  raw but distorts sheath / wall-ion energy / peak n_e / n_g (screening only); multi-GPU, mass scaling, sparse
  grids, fully implicit, δf not recommended. Stack 1+2+7 ≈ 2× channel / 3–3.5× plume; + EC at 50 µm ≈ 2.5–3× /
  ~7×. No published energy-conserving PIC of a cusped-field device → any scheme change validated against our own
  explicit 33/25 µm ladder.
- Performance audit `modern/docs/pic2d-performance-audit.md` (`c2d3b88d`; 4 H100 probes, 4.9 GPU-min, Warp event
  timing, graph off): channel-33 production solo 3.31 ms/step — block_backward/forward 91+91 launches 0.97 ms
  (29 %); born-ledger diagnostics (`momentum_sum_kernel` ×2 4096 threads striding 2 M flags + `energy_sum_kernel`
  + three 1-thread `deferred_add_kernel`) 1.1–1.2 ms (~35 %); `deposit_moment_kernel` window moments 0.67 ms
  (20 %, 20 float64 same-node atomics per electron); push 0.31; deposits 0.15; MCC 0.09; D2H 0.03. Plume-50 same
  ordering (Poisson 36 %, ledger 28 %, moments 18 %). Solo fit 0.27 ms + 4.1 µs × sweep launches + 0.30 ms/GB
  inverse blocks + 0.97 ms/M electrons (anchors +2…+7 %). Recommendations: (1) fold born tallies into `mcc_kernel`
  + two-stage reduce → ×1.5 (0.5–1 d, physics bitwise); (2) window moments on a second stream / K = 5 sampling →
  cumulative ×1.9; (3) grid-dependent Poisson: partitioned Thomas (channel) / Warp GMG (plume: 2.6 → ≈ 1.0 ms;
  33 µm plume box ≈ 17 → 4–5 ms/step, 47 h → ~12 h). Sorting breaks bitwise replay (MCC RNG keyed on particle
  index); mixed precision ≤ 5 %; multi-GPU argued against.
- Launched: perf-1 agent (v2.0.5: born-ledger fold + two-stage reduce; window moments second stream / K sampling;
  Class A/A′ verification on the H100 as MPS client) and perf-2 agent (Warp GMG Poisson, new modules, Class C).
  Also since 427c7918: `79e6a670` model v2.0.4 (ω_pe Δt gate reads the resolved-node single-step peak),
  `206892e3`/`4d32e89c` v5.1 H100 shakedown (7.78 ms/step) + amendment, `42e30aaa`/`183e32a8` ext-val shakedown
  criterion + launch-box timing (5.58 ms/step at seed).
- Decision: the plume run waits for perf-1/perf-2 (target ≈ 12 h instead of 48 h) — no second H100 needed yet.

## 2026-09-04 22:55 AEST — ext-val v0 preregistered + launched on the H100; 047 plateau; 056 gate stop (suspect)

- External validation v0 `channel-20um`: box preflight 3/3 (field `field_source_sha256 0562cb3f…`, 52,500 plasma
  cells, node map 76 × 701, max |B| 0.772 T, Δt 0.7 ps: ω_ce Δt 0.095, ω_pe Δt 0.125, Courant 0.42, 2.69 cells/λ_D,
  factorisation 1.2 s); GPU timing 5.12 ms/step seed, 12.54 ms/step at the 12 M-particle cap with 4 other MPS
  clients → 20.9 h to 3 transits (cost model said 30.6); shakedown 100k steps through run → assess → compare
  (10 rows) → re-finalize PASS. Defect found: the runtime ω_pe Δt gate read the max over every node of the
  single-step deposit (one macro-electron of W 82,467 on a 20 µm axis node = 1.3e19 → ω_pe Δt 0.14; two → 0.20)
  → model **v2.0.4** `79e6a670` (peak over nodes with ≥ 32 macro-electrons; raw recorded; CPU/Warp parity; 5
  tests; pic2d 254 + ext-val 21 pass). Stages `05e8d68b` (preflight --gpu-timing, shakedown, launch --expect-commit
  --require-mps, status, finalize), `42e30aaa`, `183e32a8`; **PREREG `3dc12cf6`** (sealed protocol 3ec0d405…;
  execution block H100 / one of four MPS slots; budget 46.0 h = 1.5 × max(model, measured); comparison-spec
  byte-identical 543c81fc…); launch config `7697ce9f`; launch log `6b1cf0fb`. **Launch: PID 31588, 22:26:44 AEST**
  into the slot 056 had freed (4th client, never 5th); 4.1–4.6 ms/step at seed load, ~1 GB (→ ~17 GB at 12 M);
  ignition N_e 48k → 188k, S 2.8 → 8.8e16 by 0.23 µs, residual −11…−21 %. Verdict ≈ 14:00–21:00 AEST 5 Sep
  (06:00 if solo); budget end ≈ 20:30 AEST 6 Sep. Inconclusive-most-likely: hard peak-Debye stop (1.36e19 ×
  T_e/10 eV on 20 µm); plateau between soft/hard; potential-step and I_a rows non-discriminating.
- Sweep: **047** exited 22:49:44 AEST on the plateau rule — step 5,560,000 = 7.784 µs = 3.003 transits, 278
  frames, 24,888 s (6.91 h), exit 0. **056** (HEMP-like) had stopped at 20:52 AEST on the triad
  `omega_pe_dt_drift 0.283 > 0.25` at 2.07 transits — its worktree runs pre-v2.0.4 code whose triad member reads
  the RAW single-node ω_pe Δt → suspected shot-noise artefact; diagnosis + record + possible v2.0.4 relaunch agent
  launched. Steady-state v5.1 launch 2 (`ss25-base`, PID 32709) took 047's slot at 22:51 (other agent).
- Main checkout ff → `6b1cf0fb`.
- Follow-up notes / risks:
  - When the sweep is assessed, the triad readings of ref/009 must be re-derived with the v2.0.4 statistic too.
  - ext-val will slow to ~12.5 ms/step as particles grow; speeds up as sweep slots free.

## 2026-09-04 22:58 AEST — v5 moved off the local GPU: launch 1 WITHDRAWN, launch 2 on the H100

- Launch 1 (RTX 5090, PID 43572) stopped 21:26:48 AEST by `Stop-Process` 12 ms after `checkpoint_step` became
  800,000 (t = 0.800 µs; 1,297,563 e⁻ / 1,324,061 Xe⁺; arrays `5e978213…`, field anchor `c14d313b…`); 6,600 s,
  40 frames, no terminal state (runner has no clean-stop channel; finalizer recovery not invoked). Readings
  healthy (I_d 1.1–1.5 mA, window Δ/λ_D 0.70, residual −14.2 %, 8.8–9.5 ms/step solo). Record `a0235676`:
  README "WITHDRAWN by the user … not a result, not a failure"; results tracked under
  `results-launch1-withdrawn/` (107 MB arrays + frames untracked). Local GPU: no process of ours.
- Amendment v5.1 `a529b457`: `amendments[0]` + `launch_platform` (H100, MPS, scheduler job, withdrawn record),
  `wall_budget_seconds` 172,800 → 117,000 s (1.5 × the box-measured 21.6 h); config identity `efb9bb09…` unchanged.
  H100 preflight (4th MPS client): factorisation 3.1 s (365 s on Windows), 6.99 ms/step seed, 10.82 ms/step at
  8.0 M particles (0.668 ms/M), +2.66 GB, expected Δ/λ_D 1.62. v2.0.4 landed between shakedown and launch →
  `amendments[1]` `4d32e89c`; shakedown re-run at the launch code `351257f2` (100k steps through finalize +
  assess, 7.92 ms/step, peak-Debye window enforced 301/500, residual −11.7 %, `no_plateau`, reference consistency
  7/7 + 7/7; counts bitwise vs the first H100 shakedown).
- Launch 2: PID 32709, 22:51:44 AEST, `schedule.py launch --only ss25-base` (jobs.yaml `31003e3b`, 16/16
  scheduler tests), worktree `jobs/ss25-base/tree` at `351257f2`, lock + Warp UUID check; took the slot 047
  freed (box stays at four PIC clients: ref, 009, ext-val, v5). 6.6 ms/step at seed load, 1,930 MiB. Verdict
  ≈ 14:00–17:00 AEST 5 Sep once the sweep runs end (20:30 at the contended rate); budget end ≈ 07:20 6 Sep.
  Launch log `ce1d96cb`; main checkout ff → `ce1d96cb`.
- Follow-up notes / risks:
  - Add a STOP-file / clean-stop handler to the shared runner so withdrawals write a terminal state.

## 2026-09-04 23:35 AEST — PIC model v2.0.5 (physics-bitwise performance) merged

- Commits `f80c6441` (code + tests: `warp_backend.py`, `simulation.py`, `frames.py`, runner `build_config`,
  `tests/pic2d/test_pic2d_v205_performance.py` 15 CPU + 2 CUDA) and `8aca6c3a` (spec `performance_v2_0_5`, audit
  doc §1/§11); ff-push `8f68e865..8aca6c3a`; main checkout ff → `8aca6c3a`. Box verification ran on `a156fd84`
  (pre-rebase, identical contents).
- Changes: (1) `mcc_kernel` tile-reduces `ke_born` / `pz_born`; `spawn_kernel` deposits the born ion into `acc_i`
  (int64) and the ionisation map; removed `energy_sum`, 2× `momentum_sum`, 3× `deferred_add`, `deposit_unit`, the
  born `deposit_fixed` and the `born_*` arrays (−5 launches, −5 flag passes). (2) Window diagnostics on a forked
  `side_stream` with `wp.Event` fork/join inside the captured step (graph-capturable in Warp 1.14); `PIC2DConfig.
  moment_sample_interval` K (phase on `diag_steps % K`; `moment_samples` travels with the sums; frames / window
  ring / gate updated; runner reads `numerics.performance.moment_sample_interval`, must divide `device_sync_steps`);
  `d_w` kept (denominator under sampling). Identity: K = 1 → `to_dict` unchanged, all v2.0.x pins hold (v4 pin
  asserted); K ≠ 1 → enters `config_sha256`, gate record gains `window_moment_samples`. Recommended production K = 5.
- Verification (H100 extra MPS client, 28.3 client-minutes, nothing signalled, no Xid): graph vs direct bitwise
  (K = 1, 5); old vs new channel-33 v4 protocol 40,000 steps: particles / φ / surface / counts / I_d / S / n_g /
  per-step maps bitwise, `ke_born_ions_j` 3.9e-16, `pz_born` 8.7e-16 rel (declared ≤ 1e-12), float-atomic
  diagnostics ≤ 3.1e-15; K = 1 vs 5: physics bitwise, gated `cells_per_debye` rel diff median 1.7e-5 / max 1.6e-3,
  same peak node 200/200, `t_e_ev` resolved nodes median 1.9e-5 / max 7.3e-5; resume of the live sweep-reference
  checkpoint (step 4.6 M) 6,000 steps old vs new: `checkpoint-final.npz` byte-identical. Box CUDA modules 79
  passed; CPU suite 270 passed / 12 skipped; local (CUDA hidden) 273 / 9. Replay lengths scaled 100k → 40k and
  400k → 6k for the GPU-minute cap.
- Timing (identical contention, 7–8 clients): channel-33 13.48 → 11.12 ms (×1.21); plume-50 26.00 → 23.81 ms
  (×1.09); channel-33 resume at plateau load 11.11 → 9.03 ms (×1.23). Per-kernel: the 2.3 ms of removed kernels
  gone; `deposit_moment` 0.69 → 0.15 ms at K = 5; `mcc` +0.10 ms. Solo not measurable (4 preregistered runs own
  the GPU); audit-model estimate channel-33 3.31 → ≈ 2.2 → ≈ 1.7 ms (×1.9), plume-50 7.2 → ≈ 4.1–5.2 ms. A 1-minute
  solo probe when the GPU frees settles it.
- Also landed (sweep agent): `ccee5c60` 056 launch-1 STOPPED record; `ee35bc84` mini-sweep AMENDMENT 1 (056 launch
  2 under v2.0.4); `8f68e865` jobs.yaml `sweep-056-launch2` enabled at ee35bc84 (report pending).
- Follow-up notes / risks:
  - Solo timing probe when the GPU frees; set K = 5 in the next fresh protocols (identity change disclosed).
  - Benign "Invalid CUDA_VISIBLE_DEVICES -1" lines in the MPS server log from CPU-only test passes.

## 2026-09-05 00:10 AEST — `poisson_gmg_v1`: Warp-native geometric multigrid field solve merged

- Commits (ff `9ca63421..e1a24aec`; main checkout ff → `e1a24aec`): `9c2e4222` `pic2d/poisson_mg.py` (hierarchy +
  numpy reference), `pic2d/warp_poisson_mg.py` (kernels + `WarpPoissonMG`), `PoissonConfig2D` `device-mg` + `mg_*`,
  `Poisson2D` dispatch, one-branch hook in `WarpBackend.__init__`, 22 tests; `7cd03b65` default 14 cycles + runner
  hook `numerics.poisson = {"method": "device-mg", …}` (recorded protocols still build block-Thomas — identities
  unchanged, test-pinned); `e1a24aec` spec `poisson_gmg_v1` + audit §12. Rebase conflicts only in docs (v2.0.5
  landed concurrently); CUDA graph test still bitwise with v2.0.5's side-stream fork (26/26 on the H100). Local 292
  passed / 12 CUDA-skipped; ruff clean on new files.
- Design: node-based on the existing masked FV operator (conductances, r-weighting, Dirichlet anode/exit/far-field/
  body, Neumann dielectric); vertex-centred coarsening by 2; operator-dependent (Alcouffe–Dendy) interpolation;
  Galerkin 9-point coarse operators (symmetry 1e-11); dense coarsest inverse ≤ 1024 unknowns; V(2,2) damped Jacobi
  ω 0.8; fused residual+restrict; fixed 14 cycles, warm start; launch sequence 12 + 14·((L−1)·6+1) captured in the
  step graph; true residual in-graph each step with interval max; `verify()` at host syncs fail-closed (fired once
  in development at 12 cycles on 360 × 1440). Bug fixed: concave cone-corner stencil mass lumped into a solid parent
  → 0.45/cycle slow mode; now to the surviving parent; constants preserved on Neumann rows.
- Convergence (zero start, per-cycle factor; residual after 14 cycles): channel-50 0.127 / 2.7e-13; channel-33 on
  the v4 plateau maps 0.127–0.130 / 2.8e-13; channel-25 0.127 / 2.7e-13; plume-50 on attempt-7/8 maps 0.127–0.133 /
  3.0e-13; plume-v2.1-50 (240 × 960) 0.14–0.17 / 1.4e-11; plume-v2.1-33 (360 × 1440) 0.14–0.18 / 8.4e-12.
- Timing (H100, contended, extra MPS client): block-Thomas vs GMG ms/solve — channel-33 3.97 / 12.1; plume-50 10.3 /
  12.0; plume-v2.1-50 26.0 / 25.4; plume-v2.1-33 22–41 / 19.0 (frees 6.0 GB + 18–24 s factorisation). Tiny dependent
  kernels cost 38–81 µs under MPS vs 3–4 µs solo → GMG penalised; solo cost-model estimate channel-33 1.1 vs 0.97 ms
  (not faster), plume-50 1.5 vs 2.6, plume-33 1.8 vs 6.5 — model values until a solo probe.
- Class C: one-step φ parity 3.8e-10 V (channel-33), 8.5e-10 V (plume-50); 5e-9 / 4.6e-8 V on the v2.1 boxes.
  Same-seed plume-v2.0-50 replay 60,000 steps per solver (identical to step 6,400, chaotic divergence after):
  trailing-half BT → GMG I_d +1.0 % (sd 14.5 %), S +1.1 %, N_e −0.001 %, n_g −0.11 %, peak n_e −0.95 %, window
  Δ/λ_D −4.2 % (sd 11.6 %), thrust −0.11 %, ionisations +0.21 % — inside the ±5 % band; no gate fired; interval-worst
  contract ratio 8.2e-5. GPU 38.7 client-minutes; a `timeout` wrapper on early replay sessions was removed before it
  could SIGTERM a PIC process.
- Remaining before a protocol may name `device-mg`: solo probe; the preregistered v4 33 µm plateau replay campaign;
  optional launch fusion (~30 % fewer launches).
- Also: `9ca63421` sweep 056 launch 2 (amendment 1) launched on the H100 (details in the sweep agent's report).

## 2026-09-05 00:22 AEST — sweep: 056 stop = shot-noise artefact (confirmed), 047/009 plateaus, 056 launch 2

- 056 diagnosis (read-only from series/summary): tripped member `trailing_time_drift(peak_omega_pe_dt)` on the RAW
  single-node statistic +0.283 (hard 0.25); the v2.0.4 resolved-node reading +0.0165 (soft 0.05), 400k-window +0.034;
  S / T_e,dense / I_d drifts +2.6 / +3.6 / +1.8 %; residual −7.58 % windowed (cooling, flat per 0.4 µs segment);
  Δ/λ_D window 1.54. Raw argmax sat on an axis node holding ≤ 4 macro-electrons in 96.4 % of 2,521 trailing records
  while the resolved peak held 157–305; checkpoints 1.905 → 2.069 transits raw 0.050 → 0.283 while resolved
  0.019 → 0.017. No heating signature (T_e,dense 9.2 → 5.3 eV while I_d rose 2.75 → 5.44 mA). Launch 2's first record
  confirms the proxy (resolved 0.0283 vs reconstruction 0.0276).
- Records: `b424ea37` 047 plateau (3.003 transits, 6.91 h, I_d 1.925 mA, S 1.45e16, n_g 3.76e19, util 0.316,
  residual −7.1 %); `ccee5c60` 056 launch-1 archived `…-launch1-triad-gate-stop/` + `triad-stop-diagnosis.json` +
  raw-statistic disclosure. Amendment 1 `ee35bc84` (056 protocol 35760e9b → 8b876b31, others byte-identical; tests
  20/20); jobs.yaml `8f68e865`; **056 launch 2 PID 38282 at 00:00:38 AEST** (identity `3d247f1ea3f6` = launch 1),
  6.1 ms/step, step 4400 bitwise vs launch 1; ETA 3 transits ~06:15 AEST; budget end ~01:20 6 Sep; launch log
  `9ca63421`. 009 FINISHED 00:00:46 AEST on the plateau rule (4,920,000 steps, 6.888 µs, 3.02 transits; record
  pending). Reference past 3 transits (3.04), no plateau yet, 10.4 h budget left.
- Ext-val v0 launch 1 stopped itself at 23:56 AEST on the windowed **residual-POWER** gate (+7.4 %) — a heating
  signature, not a drift member (diagnosis agent running; the earlier "triad" attribution came from the scheduler's
  stop-reason string).
- Panels: `interim-sweep-panel-2333.png`, `-0020.png` (+ timeseries) under `.worktrees/interim-sweep-media/`.
- Follow-up notes / risks:
  - Sweep-wide `assess` + 009 record + dashboard after the reference lands (needs v4's `assessment.json` →
    `resolution_limited` citation); ref/009 triad readings to be re-derived with the v2.0.4 statistic.

## 2026-09-05 00:52 AEST — ext-val v0 launch 1: genuine heating; energy-ledger W omission found

- Gate: the v2.0.3 windowed residual-POWER member fired (+0.0743 ≥ 0.05; 2.12e-8 J over 2.85e-7 J) at step 1,040,000
  = 0.728 µs = 0.52 transits — enforced from the first complete 400k window by design; the drift members (S +0.58,
  resolved ω_pe Δt +0.32, T_e,dense +0.19) were not enforced (arming 1.0 transit). Verdict **(a) genuine finite-grid
  heating**, and the gate fired ~0.4 µs LATE.
- Ledger finding: `inelastic_loss_j` = macro-event count × threshold WITHOUT W (mcc.py tally, warp_backend flush).
  Particle-side identity closes to 4.5e-14 J/record; recorded residual = H − L_inel with H = field work + ΔU −
  electrode work (true numerical energy creation); recorded/unscaled = 1/W on ss-v4, 047, 056-L1, attempts 7/8.
  True H / electrode work per 28-ns frame: −0.7 % (0.03 µs) → +5.2 (0.20) → +11.8 (0.31) → +25.3 (0.45) → +50.6
  (0.59) → +111.9 % at the stop (1.22 W numerical vs 1.09 W electrode); cumulative +32 %; corrected trailing reading
  crossed 5 % at step 480k while the recorded statistic read −22 %. Corrected end states elsewhere: v2 base 50 µm
  ≈ +13 %, ss-v4 ≈ +1.9 %, 047 ≈ +2.6 %, 056-L1 ≈ +0.7 %, attempt 8 ≈ +80 %.
- Debye: resolved peak 1.19 → 2.85 cells/λ_D (0 nodes > π, 1,508 > 2.5 = 8.7 % of electrons, near-axis column r
  0.12–0.30 mm, z 3.2–7.3 mm); the axis itself unresolved by construction (0.76 macro-electrons per axis node at 1e19,
  W 82,467) and at n_i 1.07–1.34e19 → 2.9–3.3 cells/λ_D (past π), invisible to the window gate (2.26). Cells hold
  2.6–58 macro-electrons at 8.6× parity weight → stochastic heating precedes the CIC threshold. I_d rose +10 %.
- Heading vs Brandt (transient, not quotable): I_a 2.61 vs 4.3 mA; I_beam 0.81 vs 2.5; n_i 9.3e18 vs ~1e19 (the one
  row inside u_val); anode-cell potential +16.4 vs ~5 V; cusp drops −1.3 / +41 vs ~10 / ~5 V; wall ion energy 381 vs
  160 eV; ion production 43 mA-equivalent vs 2.7 mA losses (inventory doubling 0.24 µs; S = 2.5× feed under static
  2e20) — an avalanche under our closure (no Bohm transport / SEE).
- Record `036bd679` (`results/channel-20um-launch1-triad-gate-stop/` + assessment/comparison + `triad-stop-
  diagnosis.json`; README §10). No amendment, no launch 2; 15 µm sibling NOT recommended as sealed (0.42× particles
  per cell; avalanche exceeds its envelope too). Route: (i) model v2.0.6 ledger W fix + gate recalibration on the
  corrected statistic; (ii) peak-Debye floor in accumulated particle-steps; (iii) the sealed `bohm-0.4` variant.
  v2.0.6 agent launched (also post-hoc ledger recomputation sidecars for every record). Sweep reference FINISHED
  (3.06 transits); ss25-base + 056-L2 running.
- Follow-up notes / risks:
  - Every "(b) residual < +2 %" acceptance must be re-read on the corrected statistic before it is quoted; the paper
    must say the 50 µm base plateau was heating.
  - The v4_fast qualification (in preregistration) runs v2.0.5; its residual acceptance to be recomputed post hoc.

## 2026-09-05 01:34 AEST — AI surrogate plan from the Reality-Simulator/ai architecture (`f92a7237`, doc only)

- `modern/docs/ai-surrogate-from-pic-plan.md` (677 lines); main checkout ff → `f92a7237`. Welding architecture:
  simulator oracle → full-physics surrogate → real-time student; time-stepping world model `S(G, M, B, X_0,
  U_{0:T}, θ) → (fields, quality scalars, uncertainty)`; U-Net / FNO / TFNO / U-FNO behind one contract; three
  inference modes (objective-only, sparse-query, full-field); heteroscedastic heads + deep ensemble + base-plus-
  residual multi-fidelity; registry-generated channel layout failing closed; train-only normalisation, hash-based
  whole-scenario splits; PINO losses (energy, mass, constitutive, boundary, simplex, Jacobian matching); LHS
  campaign generator; CEM + projected-Adam planner with CVaR; every optimum verified by the simulator. Status there:
  software only, no training on real output yet.
- PIC mapping: geometry v1.1 → material-aware P2 field `(B_r, B_z)` + masks on a canonical `(r/r_w, z/L_ch)` grid +
  topology scalars; static operating point → MLP context; closure/model tokens; grid rung 50/33/25 µm as the
  fidelity axis; seed/W replicates as the band; scalars (I_d, S, utilisation, n_g, I_beam, wall currents, peak n_e,
  T_e, per-cusp p_k / L_k / leak width / sheath drop, cell shares) from summary + `closure.extract_targets`; fields
  from `maps.npz` with `sample_count_e` as per-node label variance; feasibility classifier (ignites / plateaus /
  heats / avalanches); ledgers as label-qualification gates, not targets; real-time student dropped.
- Data reality: 4 distinct designs with qualified/imminent 33 µm plateaus, one operating point, one closure, one
  50 µm replicate pair, zero plume plateaus; 4–11 GPU-h per channel-33 plateau, 17–52 per plume run, ~USD 3/GPU-h.
  Targets: ~30 plateaus (120–250 GPU-h) for a gated trend model on the 4-cusp family; 70–100 plateaus (400–800
  GPU-h, USD 1.2–2.4k; 11–22 days on one H100 MPS-4 or 1.5–3 days on eight) for a 7-D design surrogate; plume heads
  +400–1200 GPU-h after a qualified plume model.
- Architecture: (a) GP on physical features with known per-row heteroskedastic noise in log space (BoTorch
  `SingleTaskGP` `train_Yvar`; `TwoFidelityAR1` / `MultiTaskGP` over grid rungs), ridge/tree baselines, feasibility
  classifier; (b) at ≥ 50–100 map sets: `PODFieldSurrogate` → 2-D FNO ensemble (adapted from the welding `fno.py`)
  with shot-noise-weighted NLL + discrete Gauss-law + integral-consistency terms. Ingestion via experiment_runtime;
  gates: preregistered held-out designs, RMSE ≤ 1.5× replicate floor, coverage [0.85, 0.97], reliability ceiling, OOD
  refusal; plugs in as the F2 source of `cft_revival.optimization`, F3 = preregistered PIC batches; Pareto from
  F3-verified points only.
- Phases: 0 now (ledger fix, occupancy floors, plume qualification, fast-solver probes, v5, sweep assess/targets,
  frozen record contract); 1 dataset schema + ingestion + noise-floor baseline (CPU, ~1 week); 2 campaign generator
  + preregistered H100 batches (300–800 GPU-h); 3 field operator (5–10 GPU-h); 4 UQ-aware optimisation + PIC
  confirmation (100–300 GPU-h; plume +400–1200). First useful loop ~400–1100 GPU-h, USD 1.2–3.3k.
- Follow-up notes / risks: GFM tables break on `|B|` in backticks → escape `\|B\|`; `git worktree remove` on Windows
  may need `Remove-Item -Recurse -Force` + `git worktree prune`.

## 2026-09-05 01:51 AEST — PIC model v2.0.6 merged: ledger W fix, corrected sidecars, gate recalibration, Debye floor

- Commits (ff `f92a7237..8c70cff0`; main checkout ff → `8c70cff0`): `4b53012d` code + tests (ledger fix CPU + Warp
  flush; `cft_revival.pic2d.ledger_recompute`; accumulated-particle-step Debye floor; 27 tests); `3ec2af92`
  `02013df0` `ceb6bd57` `37665d70` `c95919a3` `b498d2b7` results-only `ledger-corrected.json` sidecars (+hash) per
  experiment with README notes; `8c70cff0` spec entries `energy_ledger_correction_v2_0_6` / `gate_recalibration_
  v2_0_6` / `peak_debye_gate_accumulated_floor_v2_0_6`, version 0.6.0, audit §13.
- Ledger: `cumulative["inelastic_loss_j"] = W·(n_exc E_exc + n_ion E_ion)e` in both backends; unscaled kept as
  `inelastic_loss_per_weight_j`; series gains `interval_inelastic_loss_j`. Particle-side identity closes to ≤ 1.5e-15
  J/record (cpu), 4.4e-15 (warp-cpu), cuda passes; recorded residual = H to round-off; field-free collisional box:
  corrected −0.01…−0.03 % of L_inel vs old −100 %. Physics bitwise; `config_sha256` unchanged (bug fix).
- Recorded → corrected trailing-400k residual at the last record (5 % gate firing rec → corr; (b) < 2 %): v2 base
  +0.4 → **+13.0 %** (never → 2.70 µs; pass → FAIL); seed-b −1.4 → +11.1 % (→ 2.76 µs; FAIL); W×0.7 −4.1 → +7.2 %
  (→ 4.50 µs; FAIL); **ss-v4 −7.7 → +2.46 %** (never; **pass → FAIL**, still rising); ss-v5 L1 (0.8 µs) −14.2 →
  +0.3 % (pass); sweep 047 −7.1 → +0.9 % (pass); 056 L1 −7.6 → +0.6 % (pass); plume 3/4/6/7/8 +8.5→17.0 / +12.8→24.1 /
  −0.6→11.0 / +12.1→28.1 / +41.7→67.3 % (all → 0.66 µs); ext-val L1 +7.4 → +61.7 % (0.73 → 0.34 µs). Cross-check
  Σ(H − recorded) vs (W−1)·L: 7e-5…9e-5 relative on every ignited run. 009 / reference sidecars to be produced when
  their records land.
- Gate recalibration: keep 5 % hard one-sided from the first complete window (2× margin over the 33 µm maxima);
  keep (b) < 2 % (not loosened). Running campaigns (v5 L2, 056 L2, reference) run pre-v2.0.6 code → evaluate (b) on
  the sidecar. Peak-Debye floor `min_accumulated_macro_particle_steps_at_peak` 64,000 (32 × 2000): v4 map 2.154 →
  2.154 same node (resolved nodes 19,650 → 42,130); attempt 8 3.608 trips; ext-val end state 2.18 in the 240k window
  (axis column now gate-able). Enters `config_sha256` only when declared.
- Verification: local tests/pic2d 319 passed / 12 CUDA-skipped; H100 4th MPS client 86 passed in 70 s (1.2
  GPU-minutes), no Xid. Follow-up agent launched to re-read every recorded acceptance (b) on the corrected ledger
  (v4 → (b) FAIL; dashboards; downstream contracts; paper-impact list).
- Follow-up notes / risks:
  - The 33 µm reference plateau is heating at +2.5 % → not a clean reference; v5 25 µm is the pending test; a
    plume run at 33 µm on the reference design would likely fail (b) as well — decide the plume grid after v5.

## 2026-09-05 01:57 AEST — PIC physics completeness audit (`0901138a`); R1/R2/R3 implementation launched

- `modern/docs/pic2d-physics-completeness-audit.md` (847 lines, 151 Crossref-verified DOIs, 91 new; indexed in
  LITERATURE_SYNTHESIS.md). Code at 036bd679 vs SOTA HEMPT PIC (Brandt 2016: e-n + Coulomb + CEX, static DSMC
  neutrals, Bohm D⊥ = 0.4 kT_e/eB, ε_r dielectric with grounded backing, crude SEE). Gap table: (c) anomalous
  cross-field transport — Bohm hook OFF, `bohm-0.4` sealed never run; structural 2-D limit; likely cause of the
  ext-val avalanche; rank 1, effort S–M. (a) SEE from dielectric — yield scaffold only, emission refused;
  flux-averaged δ ≈ 0.28/0.44 at 7 eV → cusp sheath drops −10…−45 %, wall e-power ×1.5–2, T_e ↓, I_d ↑; rank 2, M–L.
  (e1/e4) 4 excitation levels; Xe⁺–Xe CEX + MEX absent — λ_CEX ≈ 60 mm vs 24 mm channel → 15–30 % of channel-born
  ions exchange; IEDF low-energy tail; thrust to fast neutrals the ledger lacks; rank 3, S / M. (d) Coulomb absent —
  ν_ee/ν_en 0.15–0.4 at 1e18, 1.4–3.4 at Brandt's 1e19; S +5…20 %; rank 4, L. (f/e2) spatial neutrals; metastables +
  stepwise (≈ 0.2× ground, ×3 uncertain) → S +10…25 %; rank 5, L. (e3, g, h, b, i) Xe²⁺, cathode gas, ion-SEE/
  sputter, ε_r + backing, self-field (β ≤ 4e-3) ≤ few % each. Roadmap R1 α-series {0, 1/64, 1/16, 0.345} + ext-val
  bohm-0.4 → I_d up monotone (+20…60 % at 1/16), peak n_e/S/T_e down; R2 SEE (BN, Al2O3) → sheath drops −10…45 %,
  T_e −10…25 %, I_d +10…30 %; R3 excitation split + CEX/MEX → I_d/S ≈ unchanged, IEDF +15–30 % low-energy ions;
  R4 Coulomb → S +5…20 %; R5 spatial neutrals + metastables → ionisation upstream, S ±10 % / +10…25 %; ≈ 15 H100
  runs, 75–90 GPU-h under MPS-4. 2-D limitation: no azimuthal coordinate → no ECDI/spokes; every I_d/S/utilisation/
  cusp-loss number is conditional on a declared α closure; closing it needs r–θ / z–θ LANDMARK companions or a 30°
  3-D wedge (~3 days/run on one H100; full 3-D ~23 days). Ten guessed DOIs caught and corrected (§8.1).
- Launched: R1 agent (verify/implement the Bohm α closure; α-series campaign prereg + launch on the H100; ext-val
  bohm-0.4 launch 2 via amendment), R2 agent (SEE for BN/Al2O3 + dielectric surface charge + ledger + Hobbs–Wesson
  sheath test; no launch), R3 agent (multi-level excitation from LXCat + Xe⁺–Xe CEX/MEX + fast-neutral ledger; no
  launch). User order fixed: finish 2-D physics → 3-D design + verify → AI run.

## 2026-09-05 02:14 AEST — surrogate plan revised for Heaviside / RF Studio and the two Reality-Simulator cores (`4e2f7467`)

- Reality-Simulator/ai: one three-level system; the competition is inside the large surrogate's spatial core —
  Core A grid neural operator (FNO/TFNO/PINO/GINO lineage; implemented: `surrogate/{fno,tfno,ufno,unet,…}`,
  `configs/surrogate/model_ladder.yaml`; "U-FNO is the current leading hypothesis") vs Core B latent-token
  transformer (UPT / CoDA-NO / Poseidon lineage; ablation pieces only — `CodomainAttentionFusion3d`, attention in
  `material_encoder.py`; no latent-token backbone exists; "do not begin with a colossal transformer"). Decision
  criteria (never exercised: no model trained on real output): identical data/splits/compute; field error by
  variable/region/horizon, interface/spectral error, conservation, rollout drift, geometry/parameter/resolution
  holdouts, calibration, latency, memory; every Core-B mechanism must beat a controlled baseline.
- Heaviside / RF Studio (Arena Physica; company posts 2026-03-31 and 2026-09-01; no paper/weights/code): Heaviside-0
  transformer forward model 2.5-D stacks → S-parameters; Marconi-0 conditional-diffusion inverse model, proposer +
  verifier with the solver as final authority; Heaviside-1 350 M params, 3-D geometry + materials + excitation →
  complex E/H at arbitrary probes; CLAIMS 250 k designs / 500 B field samples, 10⁵× speed, ~1 dB S-params; EMVal
  ~19 % relative L2 in-distribution, ~33 % on unseen families; field supervision ≈ S-only in-distribution but far
  better OOD (0.99 → 0.53 dB); OOD = held-out design templates. Checkable: EMVal v0 public split (500 boards, 101
  frequencies, 10 k-probe clouds, ~41 GB). Not checkable: corpus, solver, speed-ups, private splits. No per-
  prediction uncertainty — the one part not imported.
- Three changes to our recommendation: (1) fields-first supervision (plateau maps primary; POD-GP on maps from
  batch 1; scalar GP remains the noise-floor instrument; our own field-vs-scalar ablation is a Phase-3 deliverable);
  (2) tokenised transformer-core operator as the level-(b) target (patch tokens over the canonical grid + operating-
  point/closure/rung/code tokens + arbitrary-point query decoder, 1–10 M params, masked-patch pretraining on cheap
  fidelities; testable only at ≥ 100–300 map sets; ladder POD-GP → FNO → transformer decided on held-out families at
  fixed compute; U-FNO dropped; azimuthal Fourier modes as tokens make a 2-D record the m = 0 slice of a 3-D record
  = the 2-D → 3-D transfer path); (3) held-out design FAMILY + frozen challenge split (EMVal discipline).
- Phase order (user-fixed): 0 finish 2-D physics (R1–R6, fast solver, ladder verdicts, ext-val like-for-like, record
  contract freeze; 170–290 GPU-h); 1 3-D PIC design + verification (FFT-in-θ Poisson on the axisymmetric masks, m = 0
  = today's solver; manufactured solutions; axisymmetric-limit replay within the 2-D band; LANDMARK z–θ (Charoy 2019)
  and r–θ (Villafana 2021) in the slab limit; sector convergence 90/180/360°; full annulus N_θ = 64 INADMISSIBLE on the
  π gate (r dθ/λ_D = 4.4 at the peak), 128 admissible, 256 comfortable but 74 GB; a 90° sector × 64 cells ≈ 100–300
  GPU-h per anchor = 10–80× a 2-D label; 600–1,200 GPU-h); 2 data plane + noise floor (0 GPU-h); 3 AI campaign —
  label source by a preregistered three-anchor test: A 2-D + α calibrated on 3–5 3-D anchors (700–2,300 GPU-h), C
  multi-fidelity (1,200–4,400), B all-3-D 7–30 k GPU-h unaffordable; 4 optimisation (300–1,200). Total option A
  ~1,800–5,000 GPU-h, ~USD 5–15 k, 3–5 months. Main checkout ff → `4e2f7467`.

## 2026-09-05 02:35 AEST — `pic2d_cft_steady_state_v4_fast` (GMG + K = 5 qualification) preregistered and launched

- Prereg `b09f2b71` (experiment dir + tests), jobs.yaml `6807f041`, launch log `02aef43d`; main checkout ff →
  `02aef43d`. Protocol = v4 bit-for-bit except `numerics.poisson = {device-mg, 14 cycles, 2+2 sweeps, ω 0.8, coarsest
  1024}` and `moment_sample_interval = 5` (identity `a6275830` vs v4 `f10772b2`; test pins exactly two differing
  keys); accumulated peak-Debye floor deliberately NOT declared (would be a third difference; v4's window reads 2.154
  under both floors). Budget 102,100 s = 1.5 × 18.9 h contended estimate.
- Acceptance: (a) v4 plateau rule verbatim; (b) REVISED for v2.0.6 — the replay must reproduce v4's corrected +2.46 %
  residual within ±1 pp two-sided (seed-pair spread ≈ 0.4 pp); "< +2 %" reported not judged (v4 itself fails it);
  (c) seed-b band: I_d/I_beam/S/utilisation/n_g ≤ 2 %, peak n_e ≤ 10 %, T_e,peak ≤ 3 %; (d) multigrid contract never
  missed (runner terminal state + provenance; a `PIC2DConvergenceError` crash — not caught by the shared runner —
  classified via `assess --runner-crash-log` → `not_qualified`); (e) qualified / not_qualified / heating / no_plateau.
- Preflight (3rd MPS client): fast 11.11 ms/step seed, 13.23 at 4.5 M particles vs v4 3.15 / 4.56 → 2.9× SLOWER
  contended (latency-bound multigrid; as predicted for channel grids); solver build 0.6 s, 23 MB, 278 launches/solve,
  contract ratio 7.5e-9. Shakedown 100k steps through finalize + assess, four runs physics-bitwise, final at v2.0.6:
  12.22 ms/step, corrected seed-window residual +0.06 %, reference consistency 8/8, `no_plateau` as expected. No solo
  moment → solo probe not run (model 2.2–2.6 ms/step).
- Launch: PID 44430, 02:28:46 AEST, tmux `pic-ss33-fast`, lock b09f2b71 / protocol d353baea / config a6275830, three
  PIC clients total; 12.1–12.3 ms/step at seed load. Verdict ≈ 20:05–21:35 AEST 5 Sep contended (≈ 12:30 if the GPU
  frees); budget end ≈ 06:50 6 Sep.
- Unlock if `qualified`: 33 µm plume run on GMG ≈ 6–7 ms/step solo → 13–15 h (vs ≈ 45 h + 6 GB block-Thomas), but
  63–85 h as a 3–4-slot MPS client (37.8 ms/step measured) → schedule it SOLO.
- Follow-up notes / risks:
  - Shared runner should catch `PIC2DConvergenceError` and write a terminal state (GMG contract miss).
  - GPU plan: after the running jobs end, keep the box to ≤ 3 clients when latency-bound jobs (GMG) are on it.

## 2026-09-05 02:44 AEST — corrected-ledger re-read propagated through every recorded acceptance (`219f7ff3`)

- 9 commits ff `02aef43d..219f7ff3`; main checkout ff → `219f7ff3`. `6bd12470` v4 `results/assessment-corrected-
  ledger.json` (+ sha; `assess_corrected_ledger.py` bound to sidecar + assessment + summary + protocol, 7/7 binding
  checks), README launch-log + "Corrected-ledger re-read"; `5d48f263` v2 README ("the 50 µm plateaus were heating at
  +7…+13 %"); `ede4450d` plume README; `9cecb9bd` ext-val README §11.1; `5c4aee06` mini-sweep README; `fb0c278d`
  mini-sweep `assess_run` reads v4's re-read (both readings, `convergence_statement_corrected_ledger`) and each run's
  own `ledger-corrected.json` for (b) (fail-closed on a foreign sidecar) + tests; `a3471a8d` dashboards
  `pic2d-cft-steady-state-v4.html` (two verdict pills, 3-column acceptance table, ledger panel; anchor sidecar
  regenerated) and `pic2d-cft-steady-state.html` (heating panel, recorded-vs-corrected plot) + tests; `8b3f4030`
  re-read script tests; `219f7ff3` v4-fast README documentary note (protocol untouched; it already judges (b) against
  +2.46 %). tests/pic2d 329 passed / 12 CUDA-skipped; ruff clean on touched files.
- v4 verdict statement as committed: "plateau reached; convergence vs 50 µm as recorded (resolution_limited for
  50 µm); residual precondition (b) FAILED on the corrected ledger → the 33 µm plateau is itself heating at +2.5 % of
  electrode work and is NOT a clean reference; 25 µm (v5) pending". Predeclared tree on the corrected (b) →
  `refinement_heating`, recorded beside the standing verdict. Sweep 047 (+0.9 %) and 056-L1 (+0.6 %) pass; v2 base /
  seed-b / W×0.7 heating +13.0 / +11.1 / +7.2 % (5 % gate at 2.70 / 2.76 / 4.50 µs); plume attempts +11…+67 % (gate
  0.66 µs); ext-val +61.7 % (gate 0.34 µs vs recorded stop 0.73).
- Paper (report only): may claim the 33 µm plateau values with bands as the best-resolved result WITH the disclosure
  that (b) fails on the corrected ledger; the 50 µm statement strengthened to "heating at +7–13 %"; the pre-v2.0.6
  bias disclosure; v5 as the pending test; the attempt-8 diagnosis. Disallowed: "converged" / "energy-conserving" for
  either grid; thrust / plume / Isp; any pre-v2.0.6 residual without the corrected value beside it.

## 2026-09-05 03:24 AEST — R3 `xe_collision_set_v2` (model v2.3.0) merged; R1 commits landed (report pending)

- R3 (ff `de16f8ce..4219b654`; main checkout ff → `4219b654`): `8d89f7a1` spec data `xenon-cross-sections-v2.json`
  + `xenon-ion-neutral-cross-sections-v1.json` + builders + source extract; `bc70479a` code (`mcc.py` generalised,
  new `cross_sections_xe.py`, `ion_mcc.py`, `warp_ion_mcc.py`; hooks in `warp_backend.py`, `simulation.py`,
  `neutrals.py`, `artifacts.py`, shared runner); `72247ee3` 27 tests; `e898b339` `pic2d-model-v2.3.json` entry +
  `docs/pic2d-xe-collision-set-v2.md` + shakedown dir; `4219b654` box shakedown record (non-evidentiary).
- Processes: elastic + ionisation byte-identical to v1 (LXCat Biagi-v7.1, export 2023-05-21, re-downloaded +
  sha256-verified; payload `9b39858a…`); excitation split into 8.315 / 9.447 / 9.917 / 11.7 eV (Biagi-v7.1's four
  levels; Σ levels = lumped v1 table to 0.24 % above 10 eV → plateau changes attributable to the per-event loss
  8.32 → 9.4–10.1 eV); CEX (87.3 − 13.6 log₁₀E) Å² (Miller 2002, doi:10.1063/1.1426246); MEX 3.39e-19 E^-½ m²
  isotropic CM (Phelps via Piscitelli 2003 doi:10.1103/PhysRevE.68.046408; bytes from `BLAST-WarpX/warpx-data@
  c42f106f`, sha256-pinned; payload `6f259ba9…`). e2/e3 (metastables, Xe²⁺) out of scope, stated.
- Fast-neutral contract: fate decided at the CEX event by a straight-line march through the cell mask (identical
  arithmetic both backends): thermal (< 4 v_th → stays), exit through aperture → inventory sink F in
  `V dn/dt = Q + R − S − F − cn` (`fast_neutral_exit`), `pz_fast_neutral_exit` → `thrust_total_n`,
  `ke_fast_neutral_exit_j`; wall/cone/anode → thermalises (`pz_fast_neutral_wall` → `force_on_thruster_n`);
  plume-born → leaves. Energy ledger gains `ion_neutral_loss_j`; momentum gains `pz_ion_collisions`. No double count.
- Identity: `MCCConfig.collision_set` enters `config_sha256` (hashes recomputed from files; fail-closed); legacy set
  replays v2.0.6 bitwise on numpy (`4e512742…`) and warp-cpu (`6b775d91…`). Tests: local pic2d 345 passed / 13
  CUDA-skipped; box CUDA 29 + 30 passed incl. graph-vs-direct bitwise with the ion MCC; ruff clean on touched files.
- Shakedown (100k steps, 0.058 transits, 4.52 ms/step contended): finalize + assess ran (`no_plateau`), residual
  +0.09 %, early CEX 9.4e14/s, MEX 4.9e14/s vs S 1.6e16/s (CEX/S 5.8 %); fast neutrals 324 exit / 3103 wall / 674
  thermal; level shares 22/20/40/18 %; IEDF still the seed transient. GPU 11.3 minutes.
- R1 commits already on origin (`f1255832` model v2.1.0 anomalous-transport closure `bohm_perpendicular_rot…`,
  `2dcaebbc`/`057841cf` α-series PREREGISTERED {1/64, 1/16, 0.345}, `b14a9350` jobs, `c1508c06`/`a1065ce4` ext-val
  AMENDMENT 1 (`channel-20um-bohm-0.4` = launch 2) PREREGISTERED, `7717fdec` LAUNCH 1 α = 1/16 PID 46438 03:08 AEST,
  `de16f8ce` spec `anomalous_transport_v2_1_0`) — agent report pending.

## 2026-09-05 03:26 AEST — R1 report: anomalous transport v2.1.0, α-series preregistered + launched, ext-val bohm-0.4 amended

- Audit of the v1.4 Bohm hook vs Brandt 2016 (p. Pb_237): rate ν_an = α ω_ce from the gathered local |B| (right),
  per-electron exact Poisson 1 − exp(−ν Δt) (right), speed-preserving (right, no ledger energy term), counted in
  `cumulative["anomalous"]` + `pz_collisions` (right), outside the MCC null budget (acceptable: O(1e-7) coupling;
  keeps α = 0 bitwise), **event model WRONG**: isotropic redirect randomising v∥ (pitch-angle scattering into the
  loss cones) where Brandt rotates only v⊥ about B. v2.1.0 (`f1255832`): `AnomalousCollisionConfig(alpha, model)`
  with `bohm_perpendicular_rotation` (Rodrigues rotation by a uniform angle; |v|, v∥ unchanged to round-off; guiding
  centre shift 2 r_L sin(φ/2)) in CPU + Warp (mode 1); isotropic kept as the v1.4 default so recorded identities
  resolve. Diffusion test (24–60 k electrons, uniform 0.05 T, MSD fit): D⊥ = (kT_e/eB)·α/(1+α²) within 5 % for α =
  1/16 and 0.345 under both models (e.g. α 0.345: 31.9 measured vs exact 30.8 vs naive 34.5 m²/s); <|ΔX|²> = 2 r_L²;
  KE unchanged 1e-13; Warp = CPU on recovered angles 1e-9; graph vs direct bitwise with the hook on. Identity: α = 0
  reproduces ss-v4 `f10772b2…` byte for byte (pinned); {model, α}, K = 5 and the v2.0.6 Debye floor enter
  `config_sha256`.
- α-series `modern/experiments/pic2d_anomalous_transport_v1/` PREREG `057841cf` (draft `2dcaebbc`, jobs `b14a9350`):
  ss-v4 template + rotation closure α ∈ {1/64, 1/16, 0.345} + accumulated-floor Debye gate 64,000 + K = 5; α = 0 =
  the recorded ss-v4 plateau with its (b) FAIL stated. Acceptance: v4 plateau rule; corrected residual < +2 %; shift
  table vs the 50 µm particle band (I_d, I_beam, S, utilisation, n_g, peak n_e, T_e,peak); per-cusp report at
  6.028/12.000/17.972 mm; audit hypotheses as predeclared signs; verdict trend_confirmed / not / inconclusive by
  monotonicity of I_d and peak n_e. Identities 28ca0391 / 90cf53f1 / 8ea88273. Box preflight (4th client) 3.7 ms/step
  seed, 4.77 at 4.5 M → 6.8 h to 3 transits, budgets 37,200 s; shakedown α = 1/16 100k steps (110.8 M anomalous
  events; accumulated-floor gate enforced 301/500 with 37,147 resolved nodes where the occupancy floor resolved 0).
  **LAUNCH 1 `alpha-1over16` PID 46438 03:08:39 AEST**, lock 057841cf, 4.7–4.9 ms/step; ETA 3 transits ≈ 10:10 AEST,
  budget end ≈ 13:30 (`7717fdec`). Transient 5th client (R3 shakedown) slowed all runs ~40 % while it lasted.
- Ext-val `bohm-0.4` AMENDMENT 1 `a1065ce4` (stages `c1508c06`): sealed `1aaa080d41cd` — rotation model α = 0.4
  (Brandt's coefficient; exact factor 0.345 disclosed), v2.0.6 floor, K = 5, corrected ledger; grid/Δt/W 82,466.8
  (12 M cap)/seed/operating point/thresholds/arming/comparison spec unchanged. Preflight 2.71 / 7.22 ms/step at 12 M
  → 12.0 h; budget stays 46.0 h; shakedown passed (10 comparison rows). Queued as `ext-val-v0-channel-20um-bohm-0.4`
  for the next free slot (after sweep-056-launch2 ≈ 07:40 AEST), then `at-alpha-1over64`, then `at-alpha-0.345` via
  the box slot-waiter (tmux `r1-queue`, `launch --only`, retries only on "no free slot", never `--force`).
- Expected: I_d up monotonically (+20…60 % at 1/16), S / utilisation / peak n_e / T_e,peak / I_beam down, n_g up,
  cusp wall e⁻ current up, sheath drops down; a case passing (b) while α = 0 fails = evidence the leak bounds n_e.
  Ext-val: plateau with I_a → 4.3 mA, n_i → 1e19, or another heating/envelope stop → remaining gap on SEE / neutrals /
  W parity. Tests: local pic2d 358 passed / 13 CUDA-skipped; box CUDA 28 passed; ruff zero new findings.
- Follow-ups: launch-log commits when the queue fires (flip `enabled`, README entries ext-val §12 / α-series §7);
  results commits + `assess --case` / `--series` as cases finish; `run.py compare` for ext-val.

## 2026-09-05 03:57 AEST — R2 `see_dielectric_v1` (model v2.2.0) merged; R2/R3 campaigns + R4/R5 builds launched

- R2 (ff `4219b654..8e02db57`; main checkout ff → `8e02db57`): `385f1db2` code + tests (`pic2d/see.py`, `warp_see.py`,
  wall-kernel hooks, ledger, diagnostics, runner `numerics.see`, `test_pic2d_v22_see.py` 13 tests); `f6ab049f` spec
  `pic2d-model-v2.2.json` + audit §10; `8e02db57` rebase follow-up over R1 + R3 (SEE seed column 4 / stream 5; stats
  slots after the ion-MCC block; both ledger terms in the sources).
- Model: Vaughan 1989/1993 δ(E, θ) + Sydorenko 2006 three-component split (elastic r_e 0.03 keeps impact speed;
  inelastic r_i 0.07 uniform in (0, E); true secondaries half-Maxwellian T_see 2 eV); cosine law about the inward
  face normal; integer yield floor + Bernoulli at the impacting macro-weight; emitted at the impact segment's last
  plasma cell; ion-induced yield default 0; NO Hobbs–Wesson cap (the virtual cathode is the PIC's) — effective
  yield and wall potential recorded per interval and per wall cell. BN: Vaughan fit of Villemant 2019 (EPL 127
  23001; PICLas tabulation) δ_max 2.016 at 299 eV, k 0.563; checked vs Dunaevsky 2003 (PoP 10 2574): crossover 35.7
  vs 35 eV, δ(10 eV) 0.51 vs 0.54; flux-averaged 0.48/0.58/0.69/0.98 at 5/7/10/20 eV (supersedes the audit's 0.28 at
  7 eV); T_cr 20.3 eV. Al2O3: declared bracket (δ_max 6.4 at 650 eV, threshold 12.5 eV, Sydorenko low-energy bump);
  Tondu 2011 not digitised (flagged).
- Wall before/now: electrons were absorbed with charge deposited into the accumulated surface charge on the
  plasma-side wall nodes (Poisson RHS on the floating Neumann dielectric — present since v1.0); now surface charge
  changes by absorbed − emitted; Dirichlet surfaces and the grounded front-face conductor non-emitting. Ledger:
  `ke_see_emitted_j` injected, `pz_see_emitted`; series `see` block (effective yield, emission current, backscatter
  fraction, mean emitted energy, wall potential stats, HW flag); maps/frames gain `wall_see_flux_per_m2_s`,
  `wall_see_effective_yield`, `wall_see_mean_energy_ev`.
- Tests: suite 395 passed / 15 CUDA-skipped; CUDA modules green on the H100; SEE-off pinned to v2.0.6 (identity,
  keys, 300-step tallies; whole-state bitwise offline); graph vs direct bitwise with SEE on; Hobbs–Wesson slab
  (2e16 / 4 eV afterglow, floating dielectric, constant yield 0/0.5/0.9/1.5/3.0): drops 11.09/9.71/7.63/3.69/2.86 V
  at T_e 3.08 eV, fall matches T_e ln(1/(1 − δ_eff)) within 0.03/0.07 T_e; at δ ≥ 1 effective yield saturates 0.89/
  0.95 and the drop is 1.20/0.93 T_e vs the SCL 1.02 T_e.
- Shakedown (ss-v4 protocol, SEE(BN), 100k steps, 5.17 ms/step contended; ~14.2 GPU-minutes incl. tests): vs the
  SEE-off shakedown window I_d ×2.06, S ×1.14, peak n_e −14 %, T_e,peak +6.6 %, I_beam ×1.37; effective yield 0.915,
  emission 19.7 vs impact 21.6 mA, backscattered 0.100, mean emitted 6.3 eV; per cusp effective yields 0.966/0.961/
  0.998 (at the Hobbs–Wesson limit); near-wall drops −5.5/−3.5/−2.1 V (inverted/SCL signature). Record at
  `$WORK/r2/shakedown-see-bn.json`. Open: gap (b) permittivity/backing; Al2O3 digitisation; R2 runs must read
  per-cell effective yields before any sheath-drop claim.
- Launched: R2/R3 campaign prereg agent (`pic2d_physics_effects_v1`: see-bn, xe-set-v2, combined; queued after the
  R1 queue), R4 Coulomb agent (`coulomb_v1`: Takizuka–Abe/Nanbu per cell with a pairing permutation that never
  reorders particles), R5 spatial-neutrals + metastables agent (`neutrals_spatial_v1`, `metastables_v1`).
- GPU backlog: 4 running (v5, 056 L2, v4_fast, α = 1/16) + queued (bohm-0.4, α 1/64, α 0.345, see-bn, xe-set-v2,
  combined) ≈ 2 days at MPS-4 → a second H100 would halve it.

## 2026-09-05 05:29 AEST — physics-effects campaign preregistered (79a7c87a); α = 1/16 stopped at arming

- `pic2d_physics_effects_v1` (draft `b27da394` + `slot_queue.sh`; **PREREG `79a7c87a`**; jobs `4a0a43ca`; main
  checkout ff → `4a0a43ca`): three sealed cases on the ss-v4 template (α = 0, v2.0.6 floor 64,000, K = 5) —
  `see-bn` (`config d45b0f859bf6`, budget 49,200 s), `xe-set-v2` (`7cfaa7847fb5`, 43,200 s), `see-bn+xe-set-v2`
  (`815762d7faab`, 50,400 s); reference = ss-v4 `f10772b25b03` with (b) FAIL stated; reference block extended (IEDF
  fraction < 30 eV 0.0671, anode ion current 59.2 µA, wall e⁻ power 64.8 mW, flux-weighted wall-ion energy 60.5 eV).
  Acceptance: plateau rule; corrected residual < 2 % → plateau_clean / plateau_heating / no_plateau; shift table with
  the 50 µm band; IEDF low-energy fraction absolute band 0.03; per-cusp report (SEE: effective yield, current,
  emitted energy, SCL flag η ≥ 0.983 OR near-wall drop < 0; set: CEX/MEX/fast-neutral rates, IEDF descriptors,
  fast-neutral exit momentum, level shares); hypotheses per the audit; `assess --campaign` additive / interacting /
  not_evaluable. Preflight (5th client): see-bn 6.33 ms/step at 4.5 M → 9.05 h; xe-set-v2 5.60 → 8.0 h; combined 6.52
  → 9.31 h; factorisation 150–160 s (shared CPUs). Shakedowns 100k steps each through finalize + assess: every effect
  path live (974,643 SEE impacts → 844,591 emitted; CEX 4,101 / MEX 3,376 / 4 levels; consistency 11/11); see-bn
  shows the virtual-cathode reading again (η 0.94/0.94/0.96, negative near-wall drops) — transient diagnostics only.
  Queue: tmux `pe-queue` chained after `$WORK/r1/queue.log` "queue done"; expected launches ≈ 14–16 / 18 / 21–22 AEST
  5 Sep; GPU backlog ≈ 40 slot-hours queued + 31–38 running ≈ 18–20 box-hours; chain ends ≈ 07:00 AEST 6 Sep.
- **`at-alpha-1over16` (PID 46438) STOPPED** on the triad gate at exactly 1.00 transit (2.41 µs) — the instant the
  drift members armed; ext-val `bohm-0.4` launch 2 running since 05:07 AEST under the same arming rule. Diagnosis /
  arming-amendment / relaunch agent launched (pauses both waiters first).
- Follow-up notes / risks:
  - `tests/tools/test_cloud_schedule.py::test_shipped_jobs_yaml…` red on the box since the R1 jobs commits (enabled-
    list / "disabled ⇒ commit None" assumptions) — rewrite for the queue era (assigned).
  - Box `$WORK/pe/tree` removable after the PE launches.

## 2026-09-05 05:52 AEST — R4 `coulomb_v1` (model v2.4.0) merged

- Commits (ff `4a0a43ca..82255081`; main checkout ff → `82255081`): `f5eb08ad` `pic2d/coulomb.py` (config, NRL ln Λ,
  Nanbu angle, exact pair kinematics, numpy operator, reference rates), `pic2d/warp_coulomb.py`, hooks in
  `simulation.py` / `warp_backend.py` / `frames.py` / runner, `test_pic2d_v24_coulomb.py` (17); `be279bd5` spec
  `pic2d-model-v2.4.json`, `docs/pic2d-coulomb-v1.md`, audit §11; `31c765b8` NRL Spitzer peak frequency recorded
  beside the pair-mean rate; `82255081` box shakedown record.
- Method: every `cycle_steps` = 10 steps after push/absorb, before ion-MCC / anomalous / MCC; per cell: e–e Takizuka–
  Abe random-permutation pairing (odd cells → triplet at Δt_c/2); e–i every electron once vs ion (l + shift) mod N_i
  at field density n_i; i–i implemented, off by default (ν_ii/ν_ee ~ 1e-3). Nanbu 1997 cumulative angle with exact
  small-s mean, Pérez-2012 fit to 3, 3e^{−s} to 6, isotropic beyond; exact CoM rotation → pair momentum and classical
  energy to round-off; NRL ln Λ from the cell's n, T (floors 0.01 eV / 2.0); k = 10 keeps peak-cell s ~ 4e-5 at 1e18.
  GPU: per-cycle cell-sorted slot permutation (cell kernel → atomics → scan → scatter → deterministic rank), prepare
  kernel (moments, Fisher–Yates, e–i shift), like/unlike pair kernels; particles never moved; seed-table column 5
  (stream 6); fixed shapes inside the CUDA graph. Ledger: `pz_coulomb` (≈ 0), `ke_coulomb_j` (relativistic O(v²/c²)
  remainder, ~1e-9 K_e), pair counts / Σs / Σln Λ / large-s counts; series `coulomb` block; maps `coulomb_nu_ee_per_s`,
  `coulomb_nu_ei_per_s` + per-cusp column reading.
- Tests (local 410 passed / 17 CUDA-skipped; box 21 + 9 + 3 CUDA): Trubnikov isotropisation ratio 0.963 / 1.048 at
  N = 200k (three backends within 10 %); Spitzer/Lorentz drift decay 0.959 (initial slope = Braginskii within 10 %);
  two-temperature e–i: Landau integral = Spitzer ν_ε within 8 % over m_i/m_e 10/100/1000, realised/expected 1.003;
  e–i alone reads 0.74 of Spitzer (non-Maxwellian, documented); Coulomb-off = v2.2.0 pin `9690a3bf…` bitwise (cpu +
  warp-cpu); graph vs direct bitwise with e-e + e-i + i-i on; k 1 vs 2 invariant; `pz_coulomb` < 1e-9.
- H100 (≈ 15 GPU-min, 6th MPS client): cycle 4.49 ms (sort 1.14, pairs 3.34); +0.48 ms/step amortised = +7.3 % of
  the contended step at 4.5 M particles (audit a-priori +2–3 %). Shakedown (R3 protocol + coulomb; R3 record = off
  twin): 100k steps through finalize + assess (`no_plateau`), residual +0.14 % vs +0.09 %; Spitzer ν_e at the peak
  cell 2.8e5 /s (n_e 1.7e17, T_e 8.2 eV, ln Λ 13.2) vs ν_en 1.17e7 → 0.024 in the seed transient (~0.3 scaled to the
  plateau); cusp columns 5.2e4/4.2e4/3.4e4 /s; S −0.5 %, I_d −0.4 %, T_e,peak +3.2 %, n_e,peak −2.4 % (shot noise).
- Follow-up: R4 campaign case (`coulomb` on the reference at 33 µm) to be added to the physics-effects queue after
  the R1 arming amendment lands; identity policy: Coulomb parameters enter `config_sha256`.

## 2026-09-05 06:25 AEST — R5 `neutrals_spatial_v1` + `metastables_v1` (model v2.5.0) merged — 2-D physics set complete in code

- Commits (ff `82255081..55092f4c`; main checkout ff → `55092f4c`): `8a35a44b` feat (`pic2d/neutrals_spatial.py`,
  `warp_neutrals.py`; hooks in `mcc.py`, `ion_mcc.py`, `simulation.py`, `warp_backend.py`, `warp_ion_mcc.py`,
  `artifacts.py`, `frames.py`, runner); `72bf72dc` 18 tests + seed-layout pins; `c416f781` spec
  `pic2d-model-v2.5.json`, audit e2/f + §12, shakedown dir; `55092f4c` box shakedown record.
- Model A (test-particle DSMC-lite): Kn ≈ 10–100 → free-molecular flight + diffuse wall reflection at T_w; ~8.7e13
  atoms → ~4 M macro-neutrals at W_n 2.2e7 (~60/cell), 0.5 GB, ~9 kernels per sub-step every 200 steps; 4.9–5.1 vs
  4.52 ms/step (5th client). Route B rejected (cannot carry CEX fast neutrals, wall-temperature mixture, recycling at
  the impact point). Contracts: nearest-cell density deposit each sub-step into a device-resident per-cell array
  (never a kernel scalar); ion MCC samples the cell's drift + thermal; born ions take the local gas velocity; per-cell
  integer sinks (W/2²⁰) with debt carry — `sink_consistency_atoms = 0` exactly; recycling at the ion's last plasma
  cell; CEX fast neutrals become particles; atom ledger (neutral time) closes to 1.4 atoms on 8.6e13; time
  acceleration F (declared, default 1) with real-time rates = ledger/F. Identity: `neutrals_spatial` block enters
  `config_sha256` only when declared; `inventory-0d` replays 8e02db57 bitwise (cpu + warp-cpu; identities
  `931a6a04`/`c269ab72`); CUDA graph vs direct bitwise with the model on; checkpoints resume bitwise.
- Metastables: Xe 6s[3/2]₂ pool; branching (0.45, 0.35, 0.50, 0.35) from BSR level shares + Aymar–Coulombe 1978 6p
  cascade (net ≈ 0.43, ×3 uncertain); stepwise ionisation BEB (Kim & Rudd 1994; B 3.815 eV; peak 8.4e-20 m² at 16.6 eV;
  `stepwise_scale`); superelastic by detailed balance; wall de-excitation; optional radiative rate; energy ledger
  +(E_iz − E_m) / −E_m per event; identity closes. Correction to the audit: trapped resonance levels (~0.3 µs Holstein)
  not pooled. DOIs 10.1103/PhysRevA.20.855, 10.1103/PhysRevA.15.517, 10.1088/0953-4075/32/17/309, 10.1016/0092-640X(78)
  90007-4.
- Tests: Clausing K 0.3548 vs 0.3564 (L/D 2); Knudsen gradient within the end correction; stepwise/superelastic rates
  within 4σ; depletion exact; ledger residuals ≤ 1e-9. Local pic2d 435 passed / 18 skipped; box 14 + 37 CUDA/graph/
  parity/identity regressions. Ruff no new findings.
- Shakedown (100k steps, 0.14 µs, finalize + assess, residual +0.05 %; 13.6 GPU-min): channel-mean n_g 2.49e20 =
  4.5× the 0-D fixed point at the same feed (Knudsen closed-end profile anode 5.45e20 → 6 mm 4.1e20 → 12 mm 2.5e20 →
  18 mm 1.1e20 → exit 7.0e19); S 9.1–9.3e16/s (utilisation 1.06–1.08, seed transient); metastable fraction 0.27–0.33 %
  (up to 11 % locally), stepwise 3.1–3.4 % of ionisation; ionisation centroid 13.2 mm vs 0-D 13.4 (not like-for-like).
  Consequence: the spatial plateau at the same feed is a DIFFERENT operating point from every recorded plateau; the
  density ceiling must exceed the Knudsen anode density (fail-closed; clamp 3e-4 vs 1e-3 limit).
- Status: R1–R5 all in code (v2.1.0–v2.5.0), each off-switch bitwise vs its predecessor. Campaign cases still to
  queue: `coulomb`, `neutrals-spatial` (+ F qualification), and a FULL-PHYSICS combined case — after the arming
  amendment lands. Empty `.worktrees/r5-neutrals` removed; `.worktrees/r1-arming` is the arming agent's live tree.

## 2026-09-05 06:40 AEST — canvas: fold 219f7ff3..55092f4c (27 commits) into open-cft-roadmap-status

- Read-only git: `git fetch origin` 06:27 → origin/feat/sota-foundation 785a1594 (SIX commits later than the brief's
  55092f4c: e47ae78a model v2.1.1 arming latch, 0916a4f8 α = 1/16 launch-1 record — EXTINCTION, not relaunched,
  33be2a89 α-series AMENDMENT 1, 73d495c8 tests, 9daa1643 jobs.yaml, 785a1594 ext-val bohm-0.4 launch-2 log — NOT
  folded, named as unfolded); local HEAD 55092f4c = the fold; `rev-list --count` main..55092f4c 294, main..origin 300,
  219f7ff3..55092f4c 27; `git log --stat 219f7ff3..55092f4c` read for every path / line count quoted.
- Read-only ssh to the box (`~/.ssh/lambda_h100`, ubuntu@68.209.75.2; box repo at 785a1594): `schedule.py status`
  06:33 AEST — 4 clients: ss25-base 32709 1.65/3 transits 7.89 ms/step ETA 7.1 h; sweep-056-launch2 38282 2.85/3
  7.85 ms/step ETA 0.4 h; ss33-fast 44430 0.61/3 14.00 ms/step ETA 16.0 h; ext-val-v0-channel-20um-bohm-0.4 49403
  (launched 19:07:11Z = 05:07:11 AEST) 0.81/3 2.98 ms/step, budget left 44.6 h; at-alpha-1over16 finished at
  1,720,000 steps = 1.00 transit (triad gate). `$WORK/r1/queue.log`: PAUSED 19:35:25Z by the arming-rule agent →
  RESTART 20:26:37Z after AMENDMENT 1, waiting for a slot for at-alpha-1over64; pe-queue chained, restarted 20:26:40Z.
  Nothing launched, signalled, edited or pulled.
- Canvas `C:/Users/Angus/.cursor/projects/c-Users-Angus-Desktop-projects-uni-project/canvases/open-cft-roadmap-status.canvas.tsx`:
  pic-physics 2 → 4 — rung 2 ok (R1 f1255832, R2 385f1db2 / 8e02db57, R3 8d89f7a1 / bc70479a, R4 f5eb08ad,
  R5 8a35a44b + campaign code 2dcaebbc / b27da394 / c1508c06), rung 3 ok (tests/pic2d 435 locally; box CUDA green),
  rung 4 CAVEAT (α-series 057841cf / b14a9350 / 7717fdec — 1/16 stopped at exactly 1.00 transit at arming, amendment
  in progress at the fold; physics-effects 79a7c87a / 4a0a43ca queued; bohm-0.4 launch 2 running; shakedowns 4219b654 /
  82255081 / 55092f4c non-evidentiary), rung 5 `no` (records pending), stop RUNNING "R1–R5 in code · campaigns queued ·
  α arming amendment in progress"; validation-v0-v2 rung 4 caveat + rung 5 text (AMENDMENT 1 a1065ce4; launch 2 PID
  49403), stop FAILED GATES → RUNNING "launch 2 bohm-0.4 running · verdict ≈ 12 h solo-equivalent"; mini-sweep
  (056 L2 ETA ≈ 07:00) and ss-v5 (ETA ≈ 13:40) from the scheduler; pic-performance unchanged (live reading in the GPU
  row); pic-v14 stop lead. RIGHT_NOW rewritten to 8 rows (Direction · R1–R5 in code · Campaigns · α arming stop ·
  Ledger · Ladder v5 / v4_fast · Ext-val bohm-0.4 · Paper); key findings #12 (Knudsen 4.5×) / #13 (Bohm event model +
  arming stop); Actions (R1–R5 code DONE; campaigns RUNNING; coulomb / neutrals-spatial / full-physics + 2nd-H100
  queued; ext-val + H100 queue updated); Evidence: 24 new commit rows + the 3 re-headed + 6 unfolded + main 294/300;
  one sources row; 3 interpretation rules (event model; arm relative to the discharge + ignition gate; 0-D inventory
  pinned to the exit density); GPU / Cloud rows; Status date; roadmap steps 9 / 7 + the Stat label; Phases P5 / P7;
  mergeTruth / HEAD_NOW / RECENT_MARKERS / header chips / JSDoc changelog.
- Validation: Node recount (temp script, deleted) → 44 rows, ids unique, 8 cells each; 7:17(14) 6:1(1) 5:9(7) 4:12(6)
  3:0 2:0 1:5(2); RUNNING 6; FAILED GATES 7; NULL RESULT 2; merged 36/2/2/4; chips 0/44 · 17 · 36/44 · 0 on main;
  file CRLF-only (6666 lines); Canvas TypeScript check "no errors" after every edit.
- Follow-ups: fold the six arming-amendment commits at the next brief (pic-physics rung-4 text → the extinction
  reading becomes the folded one; validation row text); the bohm-0.4 run's 1.0-transit arming at 1.40 µs ≈ 06:52 AEST
  (stop or not — read first); the 056 L2 record (~07:00) + 009 / reference records + sweep-wide assess; the v5 verdict
  (~13:40–17:00); the v4_fast verdict (scheduler ETA ≈ 22:30 vs the row's ≈ 20:00); the 1/64 launch when a slot frees;
  the coulomb / neutrals-spatial / full-physics preregistrations; the second-H100 decision.

## 2026-09-05 07:17 AEST — α = 1/16: EXTINCTION (not heating); model v2.1.1 arming latch; α-series amendment 1

- Waiters paused 05:35 AEST (no PIC children); four PIC clients untouched; no Xid.
- Diagnosis: stop at step 1,720,000 = 2.408 µs = 1.0033 transits (first checkpoint after arming): `ionisation_rate_
  drift −0.618`, `t_e_dense_drift +0.366` (ω_pe Δt member undefined — no node ≥ 32 macro-electrons after 1.1 µs).
  N_e 6.06e5 (seed) → 4.62e5 (0.1 µs) → 1.89e5 (1.0) → 5.78e4 (2.0) → 3.74e4 (2.408), e-fold 0.88 µs ≈ r_w²/4D⊥ at
  D⊥ = kT_e/16eB; I_d 3.10 → 0.93 → 0.14 → 0.06 mA while the injected 3.0 mA returned through the exit plane (0.32 →
  2.21 → 2.99 → 2.87 mA); S 2.1e16 → 6.2e15 → 2.9e14 → 0; n_g back at 5.5e19 (α = 0 at the same times: N_e 5.5e5 →
  1.29e6, S 1.5e16 → 3.2e16). Not heating (windowed corrected residual +1.15 %, cumulative −0.17 %; accumulated-floor
  peak 0.48 cells/λ_D); not re-equilibration (monotonic decay from the seed); T_e,dense member = shot noise of an
  undefined statistic; the S member = the real decay. Hook rate 1.66e9 /s per electron = ω_ce/16 at ⟨|B|⟩ 0.15 T.
  **Verdict: extinction under the closure** — no self-sustained discharge at α = 1/16 in this model (v1.3 closure,
  3 mA / 2 eV exit injection, no SEE, n_g0 5.5e19). Hypothesis "I_d +20…60 %" contradicted in the strongest form;
  `assess --case` → `no_plateau`; not relaunched (bitwise replay).
- Model v2.1.1 `e47ae78a`: `stopping_rule.grid_heating_triad.drift_members_arming` {min_transit_times 2.0,
  settle_quantity discharge_current, settle_drift_max 0.05, settle_check_cadence_steps 40000}; residual-power and
  peak-Debye gates untouched; absent block = v1.4 rule. Calibration: ss-v4 latch closes at 2.66 transits (drift
  +0.049); 047/009/056-L2 inside 0.05 from 2 transits; attempt 8 still stopped by the residual member. Spec 1.1.0
  `triad_drift_arming_v2_1_1`; 8 tests incl. a synthetic 0.41-drift re-equilibration.
- Records / amendment: `0916a4f8` launch-1 record (+ `triad-stop-diagnosis.json`); **`33be2a89` AMENDMENT 1** —
  latch + `ignition_gate` (N_e ≥ 0.6 & S ≥ 0.3 of the 0.05–0.2 µs reference at 1.0 µs; ≥ 0.6 / ≥ 0.4 at 2.0 µs;
  accepted runs ≥ 1.31/0.96; the extinguished run 0.45/0.37 → would have stopped at 1.008 µs); three cases re-sealed
  (`cb8fb8da`/`7bfd763b`/`7c6f288e`), identities unchanged, pre-amendment seals kept as genealogy; `73d495c8` tests;
  `9daa1643` jobs.yaml (1/64 & 0.345 at 33be2a89; 1/16 finished) + tools test rewritten (16 passed locally + box);
  `785a1594` ext-val README §13. Origin moved twice (R4, R5); a first re-apply corrupted embedded SHAs via
  short-prefix matching — caught, redone. Main checkout ff → `785a1594`.
- Queue: `r1-queue` restarted 06:26 AEST: `at-alpha-1over64` → `at-alpha-0.345` at 33be2a89; `pe-queue` chained.
  056 L2 at 3.10 transits still running → 1/64 launches within ~10 min of its stop; 0.345 next (ext-val bohm-0.4
  ≈ 3 h, ss25 ≈ 6.7 h left).
- Ext-val bohm-0.4 (α 0.345, n_g 2e20) passed the 1.0-transit arming WITHOUT a stop: at 1.23 transits N_e ratio
  0.91 → 1.12, S ratio 1.5, drifts I_d +0.03 / S +0.02, residual +0.2 % — a sustained marginal discharge at 4× our
  gas density. Left running (a launch 3 under amendment 2 only if it stops on the drift members).
- Interpretation: a Bohm-leaky discharge needs the denser gas; R5's Knudsen profile (4.5× the 0-D density) is the
  operating point where the full-physics model must be tested — the α-series at the dilute 0-D point may extinguish
  at every α > 1/64.

## 2026-09-05 10:04 AEST — `pic2d_full_physics_v1` PREREGISTERED (b45f6728) and queued; α = 1/64 also extinguished

- Commits (ff to `04a3ae4f`; main checkout ff): `1d223a6c` draft, `3e19d9b2` assess fix, `2ffbe0ea` SHARED FIX
  `artifacts.save_checkpoint` shadowed `name` → every spatial-neutrals checkpoint was written as `thermal_speed.*`
  (no resume/finalize possible; regression test), `b348f0ed` seals with budgets, `9ab1182f` `--reuse-run`,
  **`b45f6728` PREREG**, `04a3ae4f` jobs.yaml + tools test. Cases (ss-v4 template + v2.0.6 floor + K = 5 + v2.1.1
  arming latch + ignition gate): `full-physics-alpha0.345` (`98cc5cbc`, 8.68/11.08 ms/step at 4.5/7.7 M, budget
  85,800 s), `full-physics-alpha0` (`7587b0f3`, 84,000 s), `neutrals-spatial` F = 1 (`66cb501c`, 65,400 s; carries
  xe_collision_set_v2 because the runner refuses metastables on the legacy set — pe `xe-set-v2` is the secondary
  reference), `full-physics-alpha1over16` (`198fb4c6`, 86,400 s), `coulomb` (`49b30f51`, 55,800 s),
  `neutrals-spatial-F10` (`e7a2d9b1`, 65,400 s). Neutral memory ~0.9 GB; MCC ceiling 1.5e21 = 2.75× the Knudsen
  anode density (fail-closed). Acceptance: (a) plateau with the latch; (b) corrected residual < 2 %; (c) shift table
  + IEDF 0.03 / centroid 1 mm absolute bands; per-cusp report; (d) plateau_clean / heating / no_plateau /
  extinguished; (e) sustain table + α-trend (key hypothesis `sustains`); (f) F qualification (F10 inside the band of
  F1); (g) additivity with R5 as the operating-point change.
- Shakedowns (6 preflight + 6 shakedown through finalize + assess + campaign): **α cases at 0.14 µs do NOT decay** —
  α 0.345 N_e flat (0.99), I_d 3.7 mA, S/S_ref 1.18; α 1/16 rising (1.09) (the dilute-gas 1/16 had lost 24 % by
  0.1 µs). R5 profile inner third 5.51e20 → 4.18e20 → 2.79e20 → 1.35e20 → 4.04e19, channel mean 2.52e20; metastables
  at F = 1 are quantised (W_meta 4.4e5; fraction ≤ 6e-6 at 0.14 µs vs 3.5e-4 / stepwise 0.64 % at F10 — the
  predeclared F distortion, documented). SEE η 0.87–0.91; CEX/S 0.06 → 0.26 with α; Spitzer ν_e/ν_en 0.031 (legacy
  gas) / 0.008–0.012 (dense gas); gross utilisation 1.13 for R5 alone. Three defects caught (per-cusp Coulomb
  KeyError; the checkpoint name shadowing; axis-cell profile shot noise → inner-third statistic).
- Queue `fp-queue` chained after pe-queue: 0.345 → α0 → neutrals-spatial → 1/16 → coulomb → F10; first launch ≈
  20:00 AEST – 01:30 6 Sep; campaign end ≈ 09:00–11:00 AEST 7 Sep. Backlog ≈ 116–121 slot-hours (worst case 175) →
  29–30 box-hours; a second H100 would finish the fp cases ~20 h earlier.
- Observed on the box: **`at-alpha-1over64` EXTINGUISHED** (`no_ignition` at 2 µs, S/S_ref 0.20 — the new ignition
  gate fired); **ext-val bohm-0.4 launch 2 stopped at 1.26 transits** on `t_e_dense_drift −0.328` (N_e 5.6e4, I_d
  1.71 mA); `at-alpha-0.345` running since 09:42 AEST. Record/diagnosis agent launched (also sweep 009/reference/056-L2
  records + sweep assess + dashboard).
- Lessons: shakedowns must run finalize ON THE RESULTS (the R5 shakedown never did → the checkpoint-name bug hid);
  Windows `ssh.exe` strips inner double quotes (helper scripts).

## 2026-09-05 11:01 AEST — α-series FINAL (all extinguish); ext-val L2 artefact stop; sweep 009 + reference recorded; sweep dashboard

- Commits (ff `04a3ae4f..a9293369`; main checkout ff → `a9293369`): `8567a147` α 1/64 record; `cd9bb41c` ext-val L2
  record; `00d2822d` sweep 009; `3a6b9b26` sweep reference; `2d273d93` 047 assess/targets + README; `efb0aa0f`
  dashboard `modern/visualization/pic2d-design-mini-sweep-v1.html` (+ anchor sidecar, generator, 9 tests);
  `a9293369` α 0.345 record + final series assessment.
- α-series: 1/64 (launch 2, PID 54512) `no_ignition` at the 2.0 µs check (S/S_ref 0.20, N_e/N_ref 0.65; passed the
  1.0 µs check 0.95/0.48), 0.84 transits, 2.18 h; N_e 6.06e5 → 4.88e5 → 3.28e5 (e-fold 2.4 µs); I_d 3.10 → 1.45 →
  0.99 mA (α = 0: 1.05 → 1.40 → 3.07); S 2.9e16 → 4.0e15; 2.5 of 3.0 mA returning through the exit; residual
  +0.005 %, 0.77 cells/λ_D. 0.345 (launch 3, PID 58055) `no_ignition` at 1.0 µs (N_e 0.18, S 0.11), e-fold 0.47 µs.
  Series verdict `inconclusive` by rule; in words: no self-sustained discharge for any α ≥ 1/64 at the 0-D density
  under the v1.3 closure without SEE; decay time ≈ D⊥^−0.55; follow-up = full-physics (Knudsen gas). Ignition gate
  saved ≈ 18 GPU-h.
- Ext-val bohm-0.4 L2 (PID 49403): stop = ARTEFACT (c) on a re-equilibrated marginal discharge (b): the
  `t_e_dense_drift −0.328` member read an undefined statistic (densest node 0.24–1.5 macro-electrons, median 0.62;
  T_e,dense 0 in 73 % of trailing records; ±0.22 random walk for 13 checkpoints after the sealed 1.0-transit arming);
  the ω_pe Δt member was correctly `None` under its floor — the T_e,dense member has no floor (code follow-up).
  Discharge: N_e 5.75e4 → 3.87e4 (0.3 µs) → 5.55e4 at the stop (+1.5 %/window), S/S_ref 1.50 flat, I_d drift +0.7 %,
  residual +0.25 %, Δ/λ_D 0.53; injected 1.80 mA → anode 1.75 mA; ionisation 2.45 mA-eq → walls 2.28/2.28 mA. vs
  Brandt (10 rows, not quotable): I_a 1.71 vs 4.3 mA, I_beam 0.10 vs 2.5, resolved n_i 4.7e17 vs ~1e19 (−1.32 dex),
  near-anode potential +9.7 vs ~5 V (inside u_val), cusp drops 17/93 vs ~10/~5 V, wall ion energy 329 vs 160 eV.
  INCONCLUSIVE; no launch 3 now — the right launch 3 is a Brandt-geometry full-physics case at particle parity
  (W ~ 1e4) with the v2.1.1 latch + a seed-calibrated ignition gate.
- Sweep table (33 µm, 3 transits): 047 ρ 0.38 — I_d 1.925 mA, I_beam 0.655, S 1.46e16, util 0.316, n_g 3.76e19,
  peak n_e 7.8e17, ion wall loss/S 0.70, centroid 0.65 L, residual +0.91 %, `closure_quotable`; reference ρ 0.60 —
  3.805 / 2.465 / 3.60e16 / 0.421 / 3.18e19 / 1.28e18 / 0.53 / 0.61 L / **+2.47 % (b FAIL)** `plateau_with_heating`
  (reproduces ss-v4 under MPS incl. the heating); 009 ρ 0.92 — 4.408 / 1.818 / 3.36e16 / 0.491 / 2.80e19 / 8.0e17 /
  0.64 / 0.70 L / +0.31 % `closure_quotable`; 056 ρ 2.36 L1 interim (5.41 mA, 0.311, stopped by artefact) — L2 at
  10:55 AEST 4.16 transits, I_d ≈ 7.8 mA, N_e 3.65 M, still densifying, no plateau (budget ~01:20 6 Sep). Sweep-wide
  reading provisional. Closure targets recorded data only. Interim panel `-1000` in `.worktrees/interim-sweep-media/`.
- Follow-ups: T_e,dense drift member needs the occupancy floor (model v2.1.2); 056 L2 record when it lands; sweep
  dashboard re-render then.

## 2026-09-05 12:21 AEST — model v2.1.2 (T_e,dense floor + stop-statistic audit) merged; canvas at a9293369

- `cdb452b8` (ff from a9293369; main checkout ff): `simulation.peak_node_debye` `t_e_dense_ev` = density-weighted T_e
  over the RESOLVED dense set (≥ 32 macro-electrons under v2.0.3+ gates, ≥ 16 without; AND ≥ dense_fraction × resolved
  peak); undefined → 0.0 with `t_e_dense_resolved` False / `…_resolved_node_count` 0; unfloored witness kept
  (`t_e_dense_raw_ev`, `dense_node_count_raw`). Runner `evaluate_triad`: both single-step members (`t_e_dense_drift`,
  `omega_pe_dt_drift`) are drifts only when every trailing-window record is resolved, else `None` (recorded, never
  enforced); legacy series read via the exact proxy `macro_particles_at_peak ≥ floor`. Diagnostic only — identity
  untouched (ss-v4 `f10772b2` pinned). Spec `triad_te_dense_floor_v2_1_2` with the member audit: INTEGRAL (residual
  power, cumulative residual, S/I_d/N_e/n_g drifts, ignition ratios, ceilings), FLOOR-PROTECTED (T_e,dense v2.1.2;
  ω_pe Δt v2.0.4 + all-resolved rule; peak-Debye window v2.0.3/v2.0.6; peak-Debye single-step v1.4; plume boundary
  v2.0.2); Courant per particle. Re-read: ext-val L2 `cd9bb41c` → `t_e_dense_drift` None at the stop and at all 14
  checkpoints after arming (unfloored witness −0.3285 reproduces the recorded stop); ss-v4 / 047 / 009 / reference /
  056-L1 / 1/64 / attempt 8 bitwise-identical drifts; attempt 8 still stopped by the residual member; α 1/16 stands on
  S −0.618. fp README §4 documentary note (protocol untouched). Tests: 13 new; tests/pic2d 518 passed locally; 97 on
  the H100 (5th MPS client, 80 s). Disclosure: the local suite ran without `CUDA_VISIBLE_DEVICES=-1` → ~9 min on the
  local 5090 (no project run on it).
- Canvas folded to a9293369 (cdb452b8 unfolded): validation → FAILED GATES (artefact stop, INCONCLUSIVE); mini-sweep
  5 → 6 (caveats; 047/009 accepted per design, reference fails (b), provisional); pic-physics rung 4/5 texts; viz +
  sweep dashboard; key finding #14; 16 commit rows; 2 interpretation rules; GPU/Cloud rows (11:54 scheduler read: 4
  clients, pe/fp queues, ~30 box-hours). Counts N 44; 7: 17 (14) · 6: 2 (2) · 5: 8 (6) · 4: 12 (6) · 1: 5 (2); RUNNING
  5; FAILED GATES 8; chips 0/44 · 17 · 36/44 · 0.

## Open follow-ups

Every `Follow-up notes / risks` bullet of the archived devlog, verbatim (2 literal duplicates dropped),
grouped under the entry it came from. Items marked done in later entries are NOT removed here; read
the chronology above to see what has since landed.

### 2026-09-03 23:15 - Orbit wall-loss geometry screening v2 (catalogue...

  - Surrogate v3 must model the partial cells; interior cells are constants (1.0) under this
    launch design. The 0.02 readiness floor is missed by the 77 topped-up cells at p ~ 0.5.
  - Paper admission of v2 must cite `POSTHOC_FINALIZATION.md`. Detached run worktree
    `uni-project-orbit-geo2-run` and the Git-common lock stay. Main tree needs `git pull --ff-only`.

### 2026-09-03 21:35 - Paper admission: cusp topology search v3.1 (GATE...

  - The brief's "every vector null sits on the axis" is NOT what the sealed v1 dataset
    records (20 of 200 in-channel X roots are off-axis bilinear roots at 0.16-0.54 r_w,
    excluded by the frozen rule, none at the wall); the paper states the split.
  - The catalogue has no admitted consumer; screening v2 launched from catalogue cells is
    named future work in Section 11. Canvas not updated from this subagent.
  - Feature branch `origin/paper/topology-v31-claim` stays at pre-rebase `726c8a69`.
    Main tree `uni-project` is behind; needs `git pull --ff-only`.

### 2026-09-03 20:35 - literature review: TWT/PPM focusing physics inhe...

  - Koch's rho is unreachable in the current catalogue (max 1.03 of 96 designs); any
    HEMP-like study needs r_w/L >= 0.6 (and then iron matters) or tapered stacks.
  - Screening v2: launch by v3.1 cell position and stratify by direction toward the nearer
    magnet centre; record |B| and field rotation at turning points.
  - Main tree `uni-project` needs `git pull --ff-only`.

### 2026-09-03 20:05 - Cusp topology search v3 (recorded rejection) and...

  - Paper Section 8 nulls remain true under their frozen definition; a paper admission
    of v3.1 (numerical-screening gate, Discussion rewording "non-standard definition")
    is the next step and was not done here.
  - The catalogue (`cusp_topology_search_v3_1/results/artifacts/cusp-cell-catalogue.json`,
    loader `catalogue.load_catalogue`) is ready for the screening's launch design and
    the MDO closures under its label; no consumer has ingested it yet.
  - Detached run worktrees `uni-project-topo-v3-run`, `uni-project-topo-v31-run` and the
    Git-common locks stay by design. Main tree `uni-project` needs `git pull --ff-only`.

### 2026-09-03 19:05 - pic2d model v1.4 (recycling, peak gate, triad, C...

  - W x 0.7 (PID 9856) still running at 19:00 (step 2.50 M, 1.56 transits, 2.0 ms/step,
    5880 s budget left; ETA ~20:35 AEST). Per instruction, returned without waiting; the
    W x 0.7 comparison and the v3 launch (`python -m experiments.pic2d_cft_steady_state_v3.run run`,
    detached, 3.5 h) are the next actions once `results-w-0.7/run_state.json` says finished.
  - 60 x 480 grid resolves the peak at 3-4 cells/lambda_D (declared); SEE BN parameters are
    provisional; ms/step at plateau counts not yet measured.

### 2026-09-03 19:40 - plasma v2: sheath-closed four-cell power balance...

  - Development model only; corrections remain PROPOSED_NOT_ACCEPTED; potentials declared;
    PIC shows a staircase; CL-4 needs a cusp sheath-edge density; mode B admissible band
    is a hairline. Main tree `uni-project` needs `git pull --ff-only`.

### 2026-09-03 18:35 - Literature synthesis and revised roadmap; canvas...

  - The canvas is being edited concurrently by another agent (an action row appeared at 18:27
    while editing); re-read anchors before each edit.
  - The three launched streams had zero commits at 18:10; their rows earn nothing beyond
    Specified until a commit exists.
  - Main tree `uni-project` is behind origin (`8674cc5a`); needs `git pull --ff-only`.

### 2026-09-03 18:05 - MDO L0 campaign v2 admitted to the paper claim m...

  - The v2 checker now binds the v1 results tree, the v1 audit blob and the
    screening dataset blob; any change there requires re-admission.
  - Main tree `feat/sota-foundation` is behind; needs `git pull --ff-only`.

### 2026-09-03 18:30 - Literature review: surrogate, MDO and external-v...

  - The brief's "Ma 2024 AST" is Yeo, Gadisa, Ogawa, Bang 2024 (FEEP); no TU Berlin HEMP
    dataset exists; Liu 2015 "Plume control" and Courtney IEPC-2007-39 omitted (404).
  - HEMP conference data carry no numerical uncertainties; any V&V 20 comparison must
    declare an assumed u_D.
  - Main tree `uni-project` is behind; needs `git pull --ff-only`.

### 2026-09-03 18:10 - PIC-MCC blockers literature review (docs/literat...

  - Recommendations that change the fixed point immediately: wall-ion recycling in the 0-D
    inventory (n_g* 2.97e19 -> ~4.5e19 before S responds); peak-node Debye gate; CUDA-graph
    capture of the whole step. The proposal's 3-transit/5-transit plateau is a property of
    the neutral closure; a >= 50 us development case with physical neutral transport is
    recommended before freezing the campaign.
  - Main tree `uni-project` is at `96220ffc`; needs `git pull --ff-only`. Worktree
    `uni-project-lit-pic` can be pruned (fully pushed).

### 2026-09-03 18:40 - Literature review of the reduced-model, cusp-los...

  - The supplied "downstream paper" DOI 10.1016/j.ast.2024.109516 is a FEEP paper
    (Yeo, Gadisa, Ogawa, Bang 2024); the brief's "Ma et al. 2024 AST" was not
    found. Recorded in the review, Section 0 and entry 73.
  - Full texts of Kahnfeld 2019 (Rev. Mod. Plasma Phys.), Matyash 2010 (IEEE TPS)
    and Kalentev 2014 (CPP) were not retrievable (timeouts); no figure numbers are
    cited from them.
  - Recommendations are proposals for new preregistrations / a ledger revision;
    nothing recorded is changed.

### 2026-09-03 17:00 - MDO L0 campaign v2 (screened catalogue x operati...

  - Every efficacy statement is reported-not-binding with seed counts; three seeds carry
    no significance. A design "wins" under CL-1 only; the CL-2 front is disjoint.
  - BO seed 101 never found design 49 (converged on 50): the categorical kernel treats
    the 96 designs as exchangeable, so finding the best design is exploration-limited at
    160 evaluations; a v3 could give the GP the screening P(wall) as design descriptors.
  - Paper admission of v2 (GATE-MDO-L0-V2) not done here; the paper's v1 Limitations
    ("geometry excluded because no map survives the audit") now has a recorded successor.
  - Main tree `uni-project` at `783a82c6`; needs `git pull --ff-only`. Run worktree and
    Git-common lock `mdo-l0-campaign-v2.execution.lock` kept by design.

### 2026-09-03 15:35 - Wall-loss geometry surrogate v2 (derived feature...

  - The 2x-baseline gate is structurally unmeetable once baselines share informative
    features; a v3 protocol should gate against the binomial floor.
  - Cell labels (128 launches) have floor 0.035 against a 0.05 gate; more launches per
    design, not more designs, is the next unit of evidence.

### 2026-09-03 14:55 - Orbit wall-loss geometry screening v1 admitted t...

  - The brief's geometry trend (long channel / large radius lose least) is
    not a population trend (rho -0.05 / -0.12); the paper reports the extremes
    and rank correlations as observations only.
  - Refined re-solve / cross-resolution diagnostic exists for the 4
    representatives only (protocol says every design; `designs.py` gates it on
    `include_refined`); disclosed in the paper; experiment README overstates.
  - The concurrent geometry surrogate v1 was REJECTED (`rejected_surrogate`);
    nothing of it is admitted, and the paper still says no surrogate or
    optimisation consuming the dataset is admitted.

### 2026-09-03 14:45 - wall-loss geometry surrogate v1 (preregistered,...

  - The surrogate is NOT an MDO v2 input; the recorded predictor.json is for
    audit. A v2 should use realised-geometry / cusp-relative inputs, more
    designs or launches, and a larger method-selection role.
  - Main tree `uni-project` still at `22e2156b`; needs `git pull --ff-only`.
  - Run worktree kept; Git-common lock stays by design.

### 2026-09-03 14:05 - pic2d phase 4: v1.3 plateau finalized, convergen...

  - `seed-b` running detached (PID 49716, launched 02:32:41Z, 3.5 h budget,
    `results-seed-b/`); at 1.27 us it tracks the base run within 1 %. ETA to 5.12 M
    steps ~3.4 h at 2.5 ms/step - inside the budget only if the average stays <= 2.5.
    `w-half` (W = 4.2e4, not W/2: budget) is to be launched only after seed-b ends.
  - The N_e drift passed by 0.02 % while omega_pe dt was still rising: the campaign
    proposal tightens the criterion (3 %, two consecutive checkpoints, >= 5 transits,
    peak density tracked).
  - ROADMAP_AUDIT.md (dated audit) still says "steady-state v2 result NOT DONE"; it
    predates the plateau and was not edited.

### 2026-09-03 13:55 - orbit wall-loss geometry screening v1 (first wal...

  - Main tree `uni-project` still at `8babb31e`; needs `git pull --ff-only`.
  - The Git-common lock `orbit-wall-loss-geometry-screening-v1.execution.lock`
    stays by design; the run worktree is kept (validate_bundle root identity).
  - The dataset is screening input only: surrogate/MDO consumers must carry the
    label; the MDO's cusp-probability closure can now be made design-dependent.

### 2026-09-03 13:10 - MDO L0 campaign v1 independent post-hoc audit

  - Disclosures for a v2: bind `experiment_runtime`/`models`/`kernels`/`run.py`
    into the hash scope (active_learning/surrogates are never imported);
    freeze unrounded scenario probabilities (Jeffreys rule gives S 8.06e-8 vs
    frozen 6.86e-8); fix the "sequential greedy batch" label (joint q was
    used) and pymoo's `generations_completed` 7; NSGA-III re-evaluated
    2/3/5 duplicates; binding gates are integrity gates only.
  - The 12 foreign python workers (started 10:56) still saturate the CPU.

### 2026-09-03 13:40 - Four-cell power-balance closure analysis admitte...

  - The brief said the legacy solver "accepted exit-flag-4 floors"; the audit
    and `Performance_est.m:91-128` say flags 1-3 accepted by status and flag
    4 rejected. The claim uses the audit's wording.
  - The doc's "line 137" is the anode term; the +IE cusp terms are on 136.
    The generator binds both and requires the documented line in the span.
  - Any change to `cft_revival/plasma` now fails the paper checker until the
    analysis is re-admitted at the new revision (by design).
  - check_paper now takes ~90 s (35 s recomputation, cached per process).

### 2026-09-03 12:05 - MDO L0 campaign v1 admitted to the paper claim m...

  - The concurrent `global-plasma-closure-analysis.md` reproduced the 13/80
    probe and proposes a ledger correction (`PROPOSED_NOT_ACCEPTED`); the
    paper's statements are scoped to the recorded probe and stay true.
  - Robust constraint enforces the worst *sampled* case (max S 0.704); stated
    in the section, Limitations and notation.
  - Main tree `feat/sota-foundation` behind; needs `git pull --ff-only`.

### 2026-09-03 11:35 - Four-cell discharge closure analysis for p != 0

  - MDO v1 disclosure remains historically correct; the solver is now usable
    only for anode-cusp-only loss. A v2 with interior cusp probabilities needs
    the proposal accepted together with a potential closure (model decision).
  - Pre-existing, not mine: `experiments/cft_topology_characterization_v1/
    tests/test_characterization.py::test_accepted_dependencies_match_coupling_v3_commit`
    fails on the base commit (spec dir diverged from `f80a360f`).
  - Main tree is behind 1 (`git pull --ff-only` needed; not done from here).

### 2026-09-03 10:25 - MDO L0 campaign v1: first preregistered optimise...

  - Corrected four-cell solver (`cft_revival.plasma`) closes only for
    p = 0 (probe 13/80); nonzero cusp probabilities have no exact solution
    at 1e-8 - a model-level finding worth its own ticket.
  - The enforced worst case is the worst sampled case (max S 0.704); at
    S = 1 110/114 robust-Pareto designs violate the margin.
  - No geometry -> L0 map exists; geometry radii excluded. Any future MDO
    with geometry needs a validated geometry -> cusp/closure model.
  - `campaign-v1.json#benchmark.results` remains null by validator design.
  - Main tree `feat/sota-foundation` is behind 4; needs `git pull --ff-only`.

### 2026-09-03 07:10 — PIC-2D phase 2: model v1.1 and snapshot v2

  - Next operating point must come from the kinetic loss fraction
    (0.10–0.35 after one transit time), not the 0-D Bohm bound; neutral
    depletion and ion–neutral CEX still absent; `protocol.json` of v2 is
    hash-bound and must not be edited post hoc.
  - A PowerShell `Set-Content -NoNewline` rewrite destroyed the untracked v2
    runner mid-session (recovered from indentation); WIP is now committed
    before any shell-side file rewrite.

### 2026-09-03 04:55 — L1a sweep v2 + topology nulls admitted to the paper

  - `four_cell_topology_search_v2/tests/test_search_v2.py::
    test_result_lifecycle_before_or_after_single_run` (experiment-local, not
    collected) fails on LF checkouts via `validate_results`; needs an
    audited-rule branch like the sweep's `protocol.py` if it is to be green.
  - Main tree still at `7a30fc2e`; needs `git pull --ff-only`.
  - Discussion now cites the nulls as evidence (CLM-028) with the labelled
    interpretation framing kept; GATE-L1 remains closed.

### 2026-09-03 03:15 — PIC-2D axisymmetric PIC-MCC (feat/pic-2d-axisymm...

  - Operating point over-drives the v1 model (no anomalous transport, static
    neutrals): density runs past the resolvable Debye/ω_pe envelope in
    <100 ns. Next: milder point or finer grid/Δt budget before any
    preregistered campaign; ion–neutral collisions, SEE, electrode-work ledger.
  - Review before merge into feat/sota-foundation.

### 2026-09-03 03:20

  - Pre-existing: `\EvidenceRevision` renders with a space inside the hash
    (newline in the `\newcommand` body); left untouched, now visible next to
    the new macro on the title date line.
  - Topology results remain unadmitted; the Discussion phrases the four-cell
    topology as an open question and cites nothing.

### 2026-09-03 02:50

  - The L1a v2 dashboard failure needs a user decision (EOL-normalised
    verifier with disclosure vs. leaving it red); frozen `results/` must not
    be edited.

### 2026-09-03 02:40

  - Manuscript integration of the wall-loss subsection needs a claims.json
    record and a gate/manifest entry; the evidence file records the boundary.
  - Domain exit is 23 mm in the artifacts (task text said 24 mm); the
    dashboard and paper use the artifact value.

### 2026-09-03 01:45

  - Readers (`load_artifact`, fields `_validate_file_sidecar`) still use
    universal newlines; the strict byte contract lives in the
    experiment_runtime manifest. Left as is (no reader change requested).
  - The nine recorded CRLF hashes remain in the immutable v4 bundle by
    design; reviewers use `audit_sidecar_eol.py`, not `validate_bundle`.
  - Unexpected: the version bump broke 7 frozen-v4 tests until they were
    made lifecycle-aware; any future orbit_mc change relies on that branch.

### 2026-09-02 23:30

  - When attempt 6 ends: render video, plume dashboard, compare exit
    potential structure vs v1.x (Dirichlet exit forced the drop inside the
    channel), thrust closure, divergence, IEDF; then decide the >= 50 us case.
  - If the plume run does not ignite by ~1 us (S flat, N_e decaying), stop
    and diagnose the cathode coupling (electrons must penetrate the exit
    field from an off-axis plume position) before spending the 4 h budget.
  - Screening v2 launches at catalogue cell CENTRES may repeat the
    launch-position bias (few reflections); if so, v2.1 must stratify
    launch z within each cell. Check when v2 reports.
  - Screening v2 allocation rule (frozen): 128/cell, top-up to 512 where
    Wilson width > 0.10; scrambled Sobol; 10% 2N control.
  - Order after topo v3: screening v2 -> surrogate v3 -> MDO v3; PIC v1.4
    -> steady-state v3 -> >= 50 us case -> breathing decision -> campaign.
  - Paper Section 8 nulls stay true under their definition, but the
    Discussion must say the definition was non-standard once v3 lands.
  - Plasma network v2 (sheath rows R28-R31, CL-3 potentials) is the next
    model change - after the literature synthesis.
  - When W x0.7 ends (~20:35 AEST): common-window comparison, convergence
    statement (seed + particle-weight), dashboard, then re-audit.
  - MDO v2 needs a posthoc audit before its "numerically accepted" rung is
    uncaveated; screening v2 (4x launches) is the recorded route to a
    surrogate v3; PIC campaign v1 proposal awaits the convergence pair.
  - Fix orbit_wall_loss_geometry_screening_v1 README overstatement
    (refined re-solve for representatives only) - docs, not evidence.
  - When seed-b ends: resume pic2d agent to launch w-0.7, then convergence
    statement + dashboard refresh; ROADMAP_AUDIT.md is dated and still
    says steady-state v2 NOT DONE (re-audit at the end).
  - Paper Discussion must scope the "mirror picture unsupported" claim to
    v4's field; the screening shows reflections elsewhere.
  - Merging feat/sota-foundation -> main is the "approved" stage; needs the
    user's explicit go (human gate), not an agent decision.
  - `modern/experiments/cft_topology_characterization_v1/tests/...
    test_accepted_dependencies_match_coupling_v3_commit` fails on base and
    is not collected by the top-level suite (experiment-local tests dir).
  - When the steady-state v2 run ends: finalize, commit small results,
    regenerate pic2d dashboard, canvas refresh, and if plateau reached
    consider the preregistered PIC campaign design.
  - MDO cannot use the mirror-formula cusp probabilities (falsified by v4);
    cusp/wall-loss probabilities are uncertain inputs with declared ranges.
  - With static neutrals the model has only avalanche or no-ignition; a
    physical steady state needs neutral depletion (v1.3).
  - PIC steady-state run will occupy the RTX 5090 for hours; check
    `pic2d_cft_steady_state_v1/results/status.jsonl` before launching other
    GPU work.
  - four_cell_topology_search_v2 `validate_results` refuses the bundle on LF
    checkouts (code, not evidence) - disclosed in its audit; fix if enforced.
  - PIC v1 over-densifies because static neutrals at 5e20 m^-3 give no
    saturation channel before ions reach boundaries (~1 us); operating
    point must be budgeted from a 0-D equilibrium estimate.
  - Next paper admissions: topology null result + L1a sweep v2 (so the
    Discussion can cite them as evidence rather than open question).
  - The four-cell wall-cusp topology targeted by the original design
    parameterisation is undemonstrated in the explored space; the paper must
    say so rather than assume it.
  - PIC-MCC v1 is development/screening, not preregistered; claim boundary
    must be explicit in every artifact.
  - mu-variation (median 0.12, max 0.69) is expected non-adiabatic cusp
    physics; it is a diagnostic and must never be a per-orbit gate.
  - v4 phase 2 must rebase onto >= 3ab50ef5 (includes fab0eccc), re-smudge
    its worktree, adopt the 1.6.0 witness contract, then rerun the shakedown
    before `prepare` binds the orbit_mc source hash.
  - v4 phase 2 (rebase onto v1.6, prepare, one detached execution) waits on
    v1.6 landing and phase 1 shakedown passing.
  - `l1a_plasma_coupling` tests fail (3F/3E/2P): its adapter predates the
    axisymmetric serialization v1.2 migration (`dbcab646`). Needs an adapter
    update or explicit retirement.
  - `l1a_geometry_sweep/results/` and `l1a_plasma_coupling/results/` are
    gitignored but referenced by committed tests; fresh clones will fail
    those tests unless results are force-added.
  - Four same-basename `test_dashboard.py` modules without `__init__.py`
    clash in a single full-suite pytest run.

### 2026-09-02 05:00

  - Installed PyTorch 2.13.0+cu130, BoTorch 0.18.1, GPyTorch 1.15.2, and pymoo 0.6.2.
  - Verified real synchronized CUDA float64 execution on RTX 5090 sm_120, a CUDA float64 GP posterior, constrained mixed-direction qLogNEHVI/qLogNParEGO optimization, and deterministic NSGA-III/MOEA-D smokes.
  - Global package snapshots remained exactly equal; `pip check` passed.
  - Final audit stopped 12 late stale `torch==2.9.1+cu128` launcher/child
    processes and removed their 1.01 GiB partial download tree, then reran
    the complete GPU/optimizer verification successfully.
  - Recorded commands, inventory, evidence, footprint, and the optional missing-`cl` fallback warning in `.venv-sota/provision-report.txt`.

### 2026-09-03 21:40 - L1a geometry sweep v3: HEMP-like wall-radius-to-...

  - L1b/P2 confirmation of rho for r_w/L > 0.5 is queued (15 HEMP-like
    designs are the list); the paper's design-space paragraph should cite this campaign;
    screening v2 can stratify launches by regime through the v3 catalogue; memory files
    need an archive-first rollover.

### 2026-09-04 02:30 - paper: admit wall-loss geometry screening v2 (ca...

  - the scratchpad rollover is overdue (deferred for concurrent
    streams); the checker adds ~23 s to `check_paper.py`; a surrogate v3 / MDO v3 on the v2
    labels would contradict CLM-085 unless the closure changes.

### 2026-09-05 03:00 - canvas: fold ce1d96cb..219f7ff3 (34 commits) into open-cft-roadmap-status

  - Read-only git: `git fetch origin` 02:47 -> origin/feat/sota-foundation c1508c06 (three commits later than the
    brief: f1255832 model v2.1.0, 2dcaebbc anomalous transport v1 DRAFT, c1508c06 ext-val amendment 1 - NOT folded,
    named as unfolded); local HEAD 219f7ff3 = the fold; main 267 behind at the fold / 270 at c1508c06;
    `merge-base --is-ancestor` b09f2b71 -> 02aef43d and ee35bc84 -> 9ca63421 both true.
  - Canvas `C:/Users/Angus/.cursor/projects/c-Users-Angus-Desktop-projects-uni-project/canvases/open-cft-roadmap-status.canvas.tsx`:
    user direction 01:12 / 01:49 (PIC accuracy first; 2-D physics -> 3-D PIC -> AI) folded as three new ladder rows
    (pic-physics RUNNING at rung 2 caveat; pic-3d NOT STARTED at rung 1 caveat; ai-surrogate NOT STARTED at rung 1;
    N 41 -> 44) and roadmap steps 9 / 10 / 11, the Next-in-order array re-sequenced 1, 2, 4b, 6, 9, 7, 10, 11, 8, 3,
    4, 5, G, G2, L. Performance: v2.0.5 f80c6441 / 8aca6c3a + poisson_gmg_v1 9c2e4222 / 7cd03b65 / e1a24aec -> rungs
    2 / 3 of pic-performance; v4_fast qualification b09f2b71 / 6807f041 / 02aef43d (PID 44430) -> rung 4 caveat,
    RUNNING. Model v2.0.6 4b53012d / 8c70cff0 (energy-ledger W fix) + six sidecar commits + the nine-commit re-read
    chain -> ss-v4 rung 6 caveat "(b) FAILS on the corrected ledger (+2.46 %)", ss-v2 "HEATING +7-13 %", pic-v14
    rungs 2 / 3, key finding #10. Ext-val launch 1 stop 036bd679 / b498d2b7 / 9cecb9bd -> validation row rung 4
    caveat, rung 5 ok, stop FAILED GATES (inconclusive; bohm-0.4 launch 2 pending). Sweep b424ea37 / ccee5c60 /
    ee35bc84 / 8f68e865 / 9ca63421 (+ c95919a3 / 5c4aee06 / fb0c278d) -> mini-sweep rung 5 caveat text, stop "056
    L2 running · assess pending". Audit 0901138a and plan f92a7237 -> 4e2f7467 -> key finding #11. RIGHT_NOW rewritten
    to 8 short rows; Actions (+9 new, 8 updated); 35 commit rows + the main row; one sources row; 3 interpretation
    rules; GPU / Cloud rows; mergeTruth / HEAD_NOW / RECENT_MARKERS / header chips / JSDoc changelog.
  - Validation: Node recount with the canvas's own caveat rule -> 44 rows; 7:17(14) 6:1(1) 5:9(7) 4:11(5) 3:0 2:1(1)
    1:5(2); RUNNING 5; FAILED GATES 8; NULL RESULT 2; merged 36/2/2/4; chips 0/44 · 17 · 36/44 · 0 on main; ids
    unique, every row 8 cells; file stays CRLF-only; Canvas TypeScript check "no errors" after every edit.
  - Follow-ups: fold f1255832 / 2dcaebbc / c1508c06 at the next brief (pic-physics rung 2 -> ok; validation stop
    text); the 009 / reference / 056-L2 records and the sweep-wide assess; the v4_fast verdict (~20:05-21:35 AEST)
    and the v5 verdict (~14:00-17:00); the pre-existing viz-row gap (caveat at rung 7 after a `no` at 5) is untouched.

## Archive

- Full entries 2026-09-02 05:00 -> 2026-09-04 02:30 (task summaries, changes, validation, per-entry follow-ups, the
  2026-09-02 23:30 running log): `.cursor/memory/archive/DEVLOG-2026-09-04-0220.md` (169,180 bytes,
  2437 lines, SHA-256 `c7183ce743308b28e249d48a4f68df0ec1e54afd91a14ff6de1c091cf917f7a8`), byte-identical copy of the live file at rollover.
- Companion scratchpad archive: `.cursor/memory/archive/AGENT_SCRATCHPAD-2026-09-04-0220.md`.
