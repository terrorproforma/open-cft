# PIC-2D Workstream Learning Scratchpad

File policy: `COMMITTED` workstream evidence (`modern/docs/workstreams/pic2d-*`).

## Preflight guardrails

- [user] Work only in the `feat/pic-2d-axisymmetric` worktree; never touch the
  main tree; keep `cft_revival.pic` untouched; write tracked text as UTF-8 LF.
- [user] Respect codebase norms: typed models, fail-closed validation, canonical
  JSON + SHA-256 provenance, deterministic seeds, CPU reference vs GPU parity,
  honest claim boundaries. Match the orbit_mc B convention exactly and test it.
- [self] Every physics simplification must be written down where the result is
  reported (protocol, summary, dashboard, docs), not only in code comments.
- [tool] The Write tool emits CRLF on Windows; run the LF normaliser over every
  new tracked file before committing and check `git ls-files --eol`.
- [tool] PowerShell has no heredocs and mangles quotes inside `python -c`; put
  multi-line Python into a temp file and run it.

## 2026-09-03 session

### Learnings

- [self] Bilinear deposition with geometric node volumes gives a 4/3 density
  bias on the axis; shape-function volumes (∫ S_n 2πr) remove it, and the
  finite-volume Gauss law then needs the ratio V_geom/V_shape (3/4 on the
  axis). A uniform-density test over the axis nodes discriminates the two.
- [self] Jacobi-PCG needs ~N iterations per solve on these grids (470 at
  31×241, 1100 at 61×481) and warm starts save only 1–2 orders of the required
  10 because particle noise re-excites the residual every step. An exact
  block-Thomas factorisation over axial columns solves in 1–20 ms and is
  deterministic; it replaced the GPU PCG as the default field solve.
- [self] Restarting CG after every chunk destroys its Krylov history: the
  residual stalled at 5e-3 relative for 550 iterations. Check the recurrence
  residual between chunks and only recompute the true residual at the end.
- [tool] Warp `@wp.func` arguments are passed by value: wrapping
  `wp.randf(state)` in a helper returns the same number every call. Draw
  randoms inline in the kernel.
- [tool] Per-thread serial reduction loops (256 dependent loads) made each dot
  product ~180 µs on the RTX 5090; a strided 4096-thread stage plus grouped
  stages is ~8 µs. Device→host reads cost ~0.5 ms each because they wait for
  the queued kernels; consolidate per-step statistics into one array.
- [tool] `wp.utils.array_sum` does not support int32; track alive counts on the
  host and assert against the compaction scan.
- [self] A stair-step cone means a particle can legitimately land two cells
  above the *new* column's staircase after a diagonal move. The right
  fail-closed criterion is "no plasma node in the landing cell", not a column
  top comparison. For a straight bore whose wall is the box edge, one cell
  beyond the box is a wall hit, not an error.
- [self] The direct solver's true-residual check must use the full equation
  `Q − Aφ` with Dirichlet values inside φ; subtracting the boundary offset
  twice made a correct solution look unconverged.
- [self] At 300 V, 0.1 A injection and n_g = 5e20 m⁻³ the v1 model runs away
  in density: peak n_e reached ~1.3e18 m⁻³ within 55 ns and the ω_pe Δt gate
  stopped the coarse run fail-closed. That is the correct outcome of the gate
  and a finding about the operating point, not a reason to loosen the gate.
- [self] orbit_mc counts a roundoff-sized final deadline fraction as an extra
  step; compare elapsed time and position, not step counts.
- [self] The E×B drift test at r = 1.5 mm disagreed by 8 % because of finite
  gyroradius/curvature effects; at r = 1 m it agrees to 0.4 % (Boris phase
  error over 20 gyrations). Test the scheme where the analytic result holds.

### What worked

- Scratch scripts under %TEMP% before formal tests caught the axis volume,
  CG restart and Warp RNG issues quickly.
- Sharing the host direct solve between backends gives bit-identical φ and
  turns the CPU/GPU comparison into a pure particle-kernel parity test.

### Open risks

- The snapshot runs end at the stability gate (~55 ns), long before an ion
  transit time; no plateau or converged discharge current exists yet.
- The cone stair-step and bilinear B interpolation are first-order geometry
  approximations near the exit; effects are unquantified.
- Ledger residual includes untracked electrode work; energy closure is not
  demonstrated for the open system.

## 2026-09-03 phase 2 (v1.1 step, snapshot v2)

### Learnings

- [self] The runtime `ω_pe Δt` gate reads the instantaneous peak *node*
  density. Axis nodes hold a handful of macro-particles, so the gate sees
  2–3× shot noise above the window-averaged peak (v1: 3.2e18 node vs 1.5e18
  window; v2 attempt 1: 1.4e18 vs 4.7e17). Budget Δt against the node peak,
  not the physical peak, or the gate fires on noise. Halving Δt (1.5 ps) put
  the trip density 14× above the design ceiling.
- [self] With static neutrals there is no saturation channel until ions reach
  the boundaries (~1 µs): the 0-D particle balance at the *observed* T_e tells
  you whether the avalanche can saturate at all (v1: source/loss = 12 at
  n_g = 5e20, 2.4 at 1e20, 1.2 at 5e19). Pick n_g from that ratio, then the
  injection current from the power balance.
- [tool] Warp compiles one block width per module: a `wp.tile_*` kernel with
  `block_dim=64` and another launched at the default 256 in the same module
  fails at compile time (“last dimension 64 does not match block width 256”).
  Use one block constant for every tiled kernel in the module.
- [tool] `wp.tile_broadcast`/`wp.untile` assume block_dim > 1 and break on the
  Warp CPU device; `wp.tile_extract(wp.tile_sum(wp.tile(x)), 0)` works on both.
- [tool] Millions of same-address float64 `wp.atomic_add` calls serialise; one
  block reduction plus one atomic per 256 threads made the push/MCC kernels
  cheap and the tallies stay exact integers. On this WDDM host a kernel launch
  costs ~0.5 ms under GPU sharing, so fusing launches beats vectorising.
- [tool] `Set-Content -NoNewline` on a line array joins lines with *nothing*
  and destroyed a 450-line untracked file; indentation survived so a regex
  splitter on runs of ≥4 spaces recovered it, but the fix is to commit WIP
  before any shell-side rewrite and to never edit tracked text from
  PowerShell one-liners.
- [self] CPU/GPU parity of tallies must be tested without injection: the numpy
  and Philox streams sample different injected particles, so only absorbed
  counts of the common seed population are comparable.
- [self] Ion subcycling k vs 2k: positions and φ agree to 1e-6/1e-3 but the
  ion kinetic energy of a cold population accelerating from rest differs by
  the half-step stagger (~2 %); test the observable that is supposed to agree.
- [self] A 0-D balance with unmagnetised Bohm loss to every surface is a
  *lower bound* on n_eq, and in a cusped field it is a loose one: after one
  transit time the kinetic ion loss was only 10–35 % of the ionisation rate,
  so the density ran 3.7–5.9× past the ceiling that bound was used to set. The
  next operating point must be set from the measured kinetic loss fraction
  (or from a lower injection current), not from the 0-D bound.
- [self] Grid heating shows up as three consistent signals: hotter electrons on
  the coarse grid (1.5–1.7×), a higher ionisation rate at equal density
  (3.5×), and a positive ledger residual (+41 % of the electrode work vs −13 to
  −18 % fine). Report all three together; one alone is ambiguous.
- [tool] `Get-Item ... | Select-Object Length` prints nothing for a single
  file in this shell; use `(Get-Item f).Length` or `Test-Path` when scripting
  checks.

### What worked

- Diagnosing from the artifacts (checkpoint peak node, series, ledger) before
  touching the code gave the operating point and the Δt budget in one pass.
- Direct device block-Thomas beat warm-started PCG on these grids; keeping PCG
  as a cross-check rather than the default avoided a tolerance argument.
- Fail-closed gates stopped the first v2 attempt in 3–5 minutes, cheaply.

### Open risks

- No ion–neutral collisions or neutral depletion: at 1e20 m⁻³ depletion is
  ~1 % over 1.6 µs (locally a few %), acceptable for a screening snapshot only.
- The runtime gate is a device-side running maximum of the per-step peak node
  density, enforced at the host sync: a violation is always caught, but the
  run advances up to `device_sync_steps` (200) steps past it before stopping,
  and those steps are in the checkpoint.
- The ledger residual after electrode work is the scheme's grid heating and is
  reported per case; it is not bounded analytically.

## 2026-09-03 phase 3 (v1.2 sizing, steady-state runner)

### Learnings

- [self] Size the operating point from two measured numbers, not a bound:
  ν_iz = S/N_e (ionisations per plasma electron per second) and τ_i,eff = N_i/L
  (ion inventory per unit loss rate). Their product decides everything: ν_iz τ
  > 1 is an avalanche with no static-neutral equilibrium (v2: 2.9), ν_iz τ < 1
  is a beam-sustained discharge with N_eq = a τ / (1 − ν_iz τ). A flat loss
  fraction f = L/S (0.30–0.35 for 1.5 µs) is the signature of the avalanche:
  both terms scale with the inventory, so f cannot approach 1 by itself.
- [self] τ_i,eff = N_i/L is only meaningful after the first ions have reached
  the boundaries: it rose 1.06 → 2.40 µs over the first microsecond and then
  saturated. Read the plateau of τ, not its early value.
- [self] The v2 "one ion transit time = 1 µs" was a free-fall estimate; the
  kinetic residence time is 2.4×. Plateau rules in transit-time units must use
  the measured value (protocol `budget_v1_2.ion_transit_time_s`).
- [self] Near threshold N_eq ∝ 1/(1 − ν_iz τ): a 1.5× uncertainty in the rate
  coefficient (hotter electrons at lower collisionality) moves n_eq from 0.23
  to 0.37 n_max at ν_iz τ = 0.44 but to > n_max at ν_iz τ = 0.73. Choose the
  margin from the sensitivity, then let the fail-closed gate cover the rest.
- [self] Resume correctness is more than the dynamical state: `Simulation`
  keeps `_last_cumulative`, `_last_energy`, `_last_electrode` and the interval
  step base outside the checkpoint. `load_state` must re-base them or the first
  resumed record reports currents over the wrong interval (caught by the
  bitwise resume test comparing interval currents, not just final arrays).
- [self] Atomic checkpoint replacement on Windows: write into `checkpoint-tmp`,
  rename live → `checkpoint-old`, rename tmp → live, delete old; the resume
  path accepts either `checkpoint` or `checkpoint-old` so a crash inside the
  swap still resumes.
- [tool] `Set-Content -NoNewline` on a pipeline of lines concatenates them into
  one line — it destroyed an untracked `run.py` in phase 2. Guardrail: never
  rewrite a tracked (or not-yet-committed) source file from a PowerShell
  pipeline; edit with the file tools, or write a temp `.py` and run it; commit
  WIP before any shell-side rewrite so `git restore` exists.
- [tool] `Start-Process python -ArgumentList ... -WindowStyle Hidden
  -RedirectStandardOutput/-RedirectStandardError -WorkingDirectory $PWD
  -PassThru` detaches cleanly; pass `-u` so the log is unbuffered, set
  `$env:PYTHONPATH` before (inherited), and record `$p.Id`. The process also
  writes its own `run.pid`.
- [tool] `nvidia-smi` shows 100 % utilisation on this WDDM host from display
  clients alone (Chrome/Electron apps, 107 W); "GPU otherwise idle" must be
  checked with `--query-compute-apps` / `pmon` (type C), not the utilisation
  figure.
- [tool] Append-only `.jsonl` logs should not use `allow_nan=False`: a NaN in
  one diagnostic must not end a 12 h run. Canonical artifacts keep the strict
  writer.

### What worked

- Extracting S(t), L(t), f(t) and τ(t) from `series.npz` took one script and
  gave a mechanism (avalanche) plus a quantitative design rule in < 30 min.
- Testing the runner with a tiny CPU protocol (12×96, 20-step syncs, 40-step
  chunks) made the cadence, resume and stray-record cases run in 2 s.

### Open risks

- The projected n_eq assumes τ_i,eff and the beam-driven term scale as stated;
  if the electrons run hotter at n_g = 1.5e19 the run may still approach or
  exceed n_max (the gate protects the numerics, not the budget).
- A resume restarts the map window and the interval ledger; a run with many
  resumes has a piecewise ledger (each session's first record has zero
  residual/electrode work) — reported in `summary.json.ledger.note`.
- The plateau rule is evaluated on I_d (noisy, ~15 % per 200 steps) and N_e;
  with ~8000 samples in the trailing window the drift noise is ~0.6 %, so a
  false plateau from noise is unlikely, but a slow drift below 5 %/2.4 µs
  would be accepted as "plateau" — the summary reports the drifts.

## 2026-09-03 phase 3b — neutral inventory, finalize

### Learned

- A "quasi-steady" 0-D neutral model with a fast artificial relaxation is NOT a
  conservation law unless the artificial term is booked explicitly. Writing the ODE as
  V dn/dt = Q − S − c n − (V/τ_g)(n − n*) and integrating the relaxation term into its
  own ledger makes the atom balance close to round-off while keeping the fixed point
  independent of τ_g. The test then checks closure AND that the artificial ledger
  vanishes at the fixed point — that is what makes "only the fixed point is
  physical" a checkable statement instead of a disclaimer.
- Solve the per-interval linear ODE exactly (expm1 for 1 − e^{−rΔt}); an explicit
  Euler step with τ_g = 100 intervals would have been fine numerically, but the exact
  solution makes the ledgers exact integrals and the test tolerance 1e-9 instead of
  "small".
- Null-collision MCC with a time-varying density: keep ν_max at the ceiling and scale
  the real frequencies — the candidate fraction stays constant, only the null share
  changes. Cheap, exact, and the fail-closed bound is a one-line check (scale ≤ 1).
  The Warp kernel already took the density as a per-launch scalar, so nothing in the
  kernel changed.
- Adding an optional key to `PIC2DConfig.to_dict()` changes every config identity and
  would have invalidated the checkpoint of the run in flight. Emit the key only when
  the feature is on.
- The device-side window accumulators die with the process; a `finalize` command can
  only produce instantaneous checkpoint maps. Label them (`maps_kind`) rather than
  pretending they are window averages.
- The v2-derived linear coefficients (beam ionisation ∝ n_g) overpredict at low n_g:
  the beam mirrors back before colliding, so the ionisation per injected electron
  collapses faster than linearly. Treat linear projections as bounds and bracket.

### Guardrails

- Never terminate a GPU run between a sync and its checkpoint without checking
  `run_state.json.checkpoint_step` — the runner's resume drops stray series records
  past the checkpoint, so the state is consistent, but the finalize step then
  reflects the checkpoint step, not the last status line.
- Keep `series_interval_steps == device_sync_steps` when the inventory is on (the
  update uses the interval ionisation count); validated in `build_config`.

### Learned (launch, attempt 1 → 2)

- Ignition in this channel is a threshold in the *seed*, not only in n_g: the same
  3 mA beam returns 91–96 % at 5e19 with a 1e16 seed and 60 % with a 5e16 seed at
  5.5e19. A linear (per-electron) avalanche argument cannot see this; the seed builds
  the plasma potential that traps and scatters the beam. When a linear projection
  says "marginal", look at what the beam is doing (returned fraction, ionisations
  per injected electron) before turning the density knob.
- Compare against the closest run that worked *at the same time offset* (v2 fine at
  0–100 ns) — the beam was fully absorbed from the first 100 ns there. That single
  comparison produced the diagnosis in one script.
- Config identity is the whole config, including backend-dependent choices (Poisson
  method). Any command that reloads a checkpoint must rebuild the config the way the
  run did; test it on a GPU-produced checkpoint, not only on the CPU test protocol.
- `Get-Process`/`Get-Item` output can vanish when piped together with other objects in
  one PowerShell statement list; print scalar properties explicitly (`(Get-Process
  -Id $pid).Id`) when checking liveness.
- The generic `results/` ignore rule only matches directories named `results`; an
  archived `results-attempt1-…` directory is untracked-visible unless ignored
  explicitly. Negating a directory (`!…/results/`) un-ignores *everything* inside;
  re-ignore what must stay out (checkpoint arrays, series.jsonl, logs, pids) and
  check `git status` before committing.

## Phase 4 — plateau finalized, convergence pair, dashboard

### Learned (what the plateau is, physically)

- What made ignition work was the *seed density*, not the neutral density: the same
  3 mA beam at 5e19 with a 1e16 seed returned 91–96 % and died; at 5.5e19 with a
  5e16 seed the beam was absorbed from the first 100 ns. The seed builds the potential
  hump (φ_max 337 V > the 300 V anode) that traps the beam electrons long enough to
  collide. Once built, the structure is self-sustaining: 2.1 ionisations per injected
  electron at the plateau.
- The plateau is a *depletion* balance, not a loss balance: the static-background runs
  (snapshot v2) had no saturation channel and grew past the budget; with the inventory
  the ionisation eats 46 % of the feed, n_g falls to 0.54 n_g0, and S saturates where
  Q_in = S + c n_g. The window mean sits on the analytic fixed point to 0.03 % — a
  checkable statement, not a disclaimer.
- Physical reading of the numbers: utilisation S/Q_in = 0.46 (a real CFT runs at
  0.8–0.95 — the channel is too short/too low current for full utilisation at 0.019 mg/s);
  beam fraction I_beam/I_d = 0.67 (0.5–0.7 is the range reported for HEMP-type
  thrusters); wall ion current ≈ 1.6 × the beam current, concentrated at the cusp
  planes (12.2 mm peak) with 126 eV mean impact energy near the last cusp — the
  cusp-erosion signature; the density peak sits in the magnetic bottle between the last
  two cusps, not at a cusp.
- The 0-D projection worked for the *mean* (0.93 × n_eq) and failed for the *peak*
  (4.1 × n_max): a bottle geometry concentrates the plasma in ~1/4 of the volume.
  Budget the peak, not the mean, next time (or budget with a fill-factor ≤ 0.3).
- A drift criterion on the electron count is weak when the plasma is still slowly
  densifying: +4.98 % passed at the first eligible checkpoint while ω_pe Δt was still
  rising monotonically. For the campaign, require two consecutive passing windows or add
  ω_pe Δt / peak n_e to the tracked drifts.

### Learned (tooling)

- A finished run must stay hash-bound to its protocol file: put later additions
  (variants) in a sibling file instead of editing the protocol; the dashboard then
  verifies every embedded case against the frozen hash and fails closed on drift.
- Runner variants as a `--case` merge (seed, W, wall budget) keep one code path and give
  each case its own results dir and config identity; test that identities differ.
- Throughput on this WDDM GPU: ~1.0 ms/step at ≤ 0.3 M particles, 2.5 ms/step from
  ~0.35 M, 2.0 ms/step average over a 2 M-particle run. Budget with 2.5 ms/step for
  runs that spend most of their time above 0.35 M.
- Headless screenshots catch what tests cannot: a seed-transient spike hiding the whole
  currents plot, log-axis labels in log10 units, and a trace hidden under a coincident
  one. Review the screenshot before committing the HTML.
- `Get-Content` in Windows PowerShell 5 reads UTF-8 markdown as ANSI (mojibake); append
  to UTF-8 docs from Python and keep LF.
- The maps' radial grid spans the exit radius (3 mm, Δr = 50 µm), not the bore (2 mm):
  compute node coordinates from the config's `dr_m`, never from "bore / cells".


## 2026-09-03 (Phase 4, part 2): what two seeds taught

### Learned (physics / statistics)

- Seed-to-seed spread of the v1.3 plateau quantities at the same simulated time is
  0.4–1.1 % (I_d +0.40 %, I_beam +0.64 %, S +1.05 %, n_g −0.83 %, N_e +0.57 %,
  ⟨T_e⟩ +0.53 %). The differences are internally consistent: a 0.6 % denser, 0.5 % hotter
  plasma ionises 1.1 % more, and the quasi-steady inventory answers with Δn_g = −ΔS/c
  (−0.87 % predicted, −0.83 % observed). Use S/n_g coupling as a sanity check between runs.
- Pure shot noise (1/√N ≈ 0.1 % for 1 M particles; 0.15–0.2 % for a current averaged over
  a µs) is a lower bound that the observed spread exceeds by 3–7×. Run-to-run variance is
  set by correlated plasma fluctuations and by the phase of the slow densification, not by
  counting statistics — so a "1 M particles is enough" argument based on √N alone is not
  valid; the spread must be measured with seeds.
- Batch-means SEs over 30 ns blocks are right for currents and counts (|z| ≤ 1.6 between
  seeds) but far too optimistic for smooth integrated quantities (S, n_g, T_e give
  |z| ≈ 5–6 for < 1 % differences). Quote z with the block length and treat the smooth
  quantities' spread as a trajectory offset, not a fluctuation.
- Time-offset comparison is a trap: base plateau (7.2–7.68 µs) vs seed-b (4.85–6.06 µs)
  shows N_e −7 %, wall currents −10 %, φ_mean +5 % — all densification between 6 and
  7.7 µs, not seed variance. I_d and I_beam saturate first; the density and wall loss keep
  growing. Always compare at the same simulated time; label any offset comparison as such.
- The exit-plane axis cell is shot-noise dominated (j_z on axis differs by 34 % while the
  total beam current agrees to 0.6 %). Compare fluxes as totals and as L2 profile
  differences, not point values at r → 0.

### Learned (tooling)

- A graceful budget stop IS a finalization (window-average maps, summary, checkpoint-final).
  Re-finalizing from the checkpoint would downgrade the maps to instantaneous ones; make
  the tool refuse that by default (`--allow-refinalize`) instead of relying on memory.
- Name variants by what they are (`w-0.7`, not `w-half`) before the first byte is written:
  results dirs, config ids and .gitignore lines inherit the name and are awkward to rename
  after a run exists.
- Projected ETAs at launch overestimate throughput: ms/step scales with the particle count
  that the case will reach, not the count at minute 3. Project with the expected plateau
  count (base: 2.0 ms/step at 1.0 M; w-0.7 will carry 1.4 M).
- Embedding the comparison into the dashboard payload (computed from the hash-verified
  series) beats a side JSON: it is regenerated deterministically, validated (one comparison
  per finished variant) and tested, and the w-0.7 case will get the same treatment for free.
