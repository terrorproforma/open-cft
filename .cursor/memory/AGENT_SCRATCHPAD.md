# Agent Scratchpad

## File Policy

- Current policy: `COMMITTED`
- Rationale: User requires the learning-scratchpad loop; current task forbids edits to tracked files, so this new local note is not staged or committed.

## Retained Lessons

### Process & preregistration

- [self] Before any one-shot preregistered campaign, run a labelled NON-EVIDENTIARY
  shakedown of the full production path on the REAL input data (real P2 field,
  real launch manifest, full validator). Synthetic preflights passed three
  times while the real field exposed a fresh defect each time (orbit v1/v2/v3).
- [self] "One immutable attempt, no rerun" is only cheap if the code has already
  seen real data. Otherwise each latent bug costs a whole prereg cycle.
- [self] Confirmed twice in one night: the v4 shakedown's first run caught a
  `zip(..., strict=True)` bug in the assessment stage that no synthetic
  preflight or unit test reached. Shakedowns must run the assessment and
  export stages too, not just integration.
- [self] (L1b HEMP v1, 2026-09-04) A shakedown on 3 of 15 inputs is not a preflight. v1 died
  at design 028/048 on the level-0 mesh-angle gate (geometry slivers) before any solve, costing
  a full prereg cycle. Rule: every cheap, deterministic gate (mesh quality, geometry validity,
  input schema) must run over the WHOLE input set before the prereg commit;   inherited gates
  (here the fem_reference 10 deg angle) need re-justifying for the new geometry family.
- [self] (paper L1b admission, 2026-09-04 09:00) When a study's object contradicts every existing
  gate OUTCOME text (all earlier screening outcomes say "on L1a linear-vacuum fields"), add a new
  outcome VALUE under the existing kind, not a new kind, and make the checker require the kind
  description to name it (and refuse the old outcomes for that gate).
- [self] After a rebased campaign chain, the bundle's lock names the PRE-rebase prereg SHA
  (unreachable from HEAD). Never resolve it in a generator: bind it as a string, and bind the
  rebased commit by recomputing the sealed experiment-code/dependency hashes from its blobs.
- [self] Preregistration commits must be experiment-path isolated; commit
  tests separately (or inside the experiment dir) so `_bind_preregistration`
  accepts HEAD.
- [self] A saturated label (all interior cells at 1.0) is a finding that
  ends a chain, not a dataset to fit: stop the surrogate/MDO iteration and
  move the closure source to the next fidelity (PIC).
- [self] A null under a non-standard definition is a statement about the
  definition. Before freezing a definition in a preregistration, check it
  against the field's literature (topology v2 looked for wall-side nulls
  that PPM stacks never have; v3.1 with the textbook definition found N-1
  cusps per N stages).
- [self] A preregistered experiment's tests that bind the LIVE worktree to
  the frozen code contract are right only until the one execution. After the
  terminal bundle exists, switch them to the RECORDED contract (bundle +
  preregistration-commit blobs) and assert the gate stays closed; otherwise
  the first legitimate package change after the campaign turns the
  experiment's own tests red for reasons unrelated to its evidence.
- [tool] `validate_bundle` binds `root_identity` (volume/file id/final path)
  to the producing directory; it can never pass on another checkout. The
  reviewer-reproducible check is the manifest inventory (byte hashes +
  sidecar pairs), which is what `audit_sidecar_eol.py` does.
- [self] A finding never executed under a protocol needs its own paper gate
  kind (`analytic-consistency`: a derivation verified numerically, pinned by
  tests, recomputed by the checker); do not squeeze it into
  `numerical-campaign`/`numerical-screening`, whose bundle rules would then
  mean nothing. Define "accepted" as "derivation + verification admitted as
  recorded" and list what it does NOT accept.
- [self] Verify a brief's numbers against the files before writing claims:
  the legacy defect is "lsqnonlin flags 1-3 accepted by status, flag 4
  rejected", not "flag 4 accepted"; the `+IE` cusp terms are on line 136 of
  `Power_B_EQs.m` (the doc says 137 = the anode term).
- [tool] orbit_mc v1.7 validates every launch id against
  `<campaign_id>:E[0-9]+:P[0-9]+:X[0-9]+:D[+-]1:G[0-9]+`; custom launch sets must encode
  their coordinates into that grammar (X = cell index, G = Sobol index here).

### Physics findings

- [self] Do not extrapolate axis-fitted field harmonics to the wall: I_1(k kappa r)
  amplifies aliasing noise 50-80x for k = 5 at x_w ~ 1. Fit harmonics on the wall
  profile; the fundamental alone reproduces the wall B_r of a PPM stack to 10 %.
- [self] Before interpreting reflection statistics, locate the launch cells relative to
  the field extrema: v4 launched 0.5 mm from the magnet centres (|B| maxima, no mirror
  possible) and saw 0 reflections; the L1a screening launched some cells near the nulls
  and saw 32-88/128. Same physics, different launch positions.
- [self] PIC operating points must be sized from the MEASURED kinetic loss
  time (tau_i,eff 2.4 us here), not a Bohm bound; check nu_iz*tau < 1 or
  the run is an avalanche with no equilibrium.
- [self] Static-neutral PIC has no physical steady state in this channel:
  either avalanche (n_g >= 3.4e19) or no ignition. A neutral inventory
  model is required for a real plateau.
- [self] In a collisionless electrostatic plume there is no cross-field
  path: an off-axis cathode must sit on the channel-connected flux tube or
  nothing couples. Trace field lines before placing sources.
- [self] Test-particle reflection statistics depend on WHERE you launch
  relative to the field maxima; a launch design must stratify position
  within each cell or the "reflection" estimand is a launch artefact (v4).
- [self] Check the design space against the device's own design criterion
  before optimising in it (Koch's rho was never reachable in the legacy
  parameterisation - a whole MDO study on the wrong region).
- [self] In a multi-cell global model the sheath rows cannot fix the cell
  potentials (density cancels in the ambipolar balance); the interior
  potential structure must come from a kinetic model or be declared. The
  PIC shows a staircase, not a flat interior.
- [self] A design ranking produced under a declared closure is a property
  of the closure: report at least two closures and their Pareto overlap
  before any "design X is best" sentence (MDO v2: CL-1 vs CL-2 Jaccard 0).
- [self] A single-design result is field-specific until a sweep says
  otherwise: v4 (P2 divergent-exit) had zero reflections; 96 L1a designs
  all reflect. Generalise only from the sweep.
- [self] The collisionless test-particle wall-hit probability (v4) and the
  Kornfeld per-cusp-transit loss probability are different quantities; the
  legacy chain conflated them. Never feed one into a model expecting the
  other without a declared closure.
- [self] MDO v1 probe (2026-09-03 08:20): the corrected Kornfeld 4-cell solver
  (`cft_revival.plasma`) closes ONLY for p = (0,0,0,0); every nonzero cusp
  probability tried (Ua 150-1000 V, Ia 0.1-3 A) hit `iteration_limit` with
  residual floors 5e-4..0.19, at ~3 s per multistart solve (pure Python).
  It is unusable as an optimiser evaluation chain; cusp probabilities must
  enter the L0 model through an explicitly declared closure instead.
- [self] Root cause of the above (2026-09-03 11:30, `266d8a99`): on the
  R00-R26 manifold the global row is exactly
  `2*(je3(1-p4)+I4)*(phi4-Ua) + EI*(p1 je0 + p2 je1 + p3 je2)` -> no
  admissible root for any interior p > 0 (source double-books cusp
  recombination energy; anode electron term has an ion's sign). Solutions
  exist only for p1=p2=p3=0 (any p4) at phi4 = Ua. Correction is
  PROPOSED_NOT_ACCEPTED (it frees all four potentials). Before blaming a
  solver floor, substitute the consistent rows into the suspect row.

### Numerics & GPU

- [tool] Long GPU runs: launch detached with checkpoints + status.jsonl and
  return; never keep a subagent waiting on a multi-hour job.
- [tool] CUDA-graph capture freezes every kernel SCALAR argument at capture
  time; any per-step host input (neutral density, rates, counts) must live
  in a device array. The v1.4 graph baked in the MCC n_g and silently ran
  stale for two plume attempts. Test: graph vs direct with a CHANGING input.
- [self] Calibrate a fail-closed ignition/plateau gate on a run that
  succeeded as well as on the failures; a gate tuned on failures alone
  rejected the one run that had actually ignited.
- [tool] orbit_mc's `wilson_interval(0, n).lower` is a positive round-off
  for 734 of the first 4000 n; validators requiring lower <= p reject
  zero-count cases at those n. Size cases accordingly or fix in v1.8.
- [self] When two surrogates fail for label noise, skip the surrogate and
  optimise over the measured catalogue directly; a discrete design set with
  measured outputs is a valid design space.
- [self] Before fitting a surrogate, check whether the input->target map
  has step discontinuities (discrete design selectors). A stationary GP on
  raw design parameters fails there; use derived physical features and/or
  per-category models, and always include a tree baseline.
- [self] A residual floor that scales linearly with a parameter under
  continuation, with a well-conditioned Jacobian and |J^T r| ~ floor on a
  bound face, is a MODEL inconsistency, not a solver failure. Derive the
  closed-form reduced row before spending more solver effort.
- [tool] BoTorch GP fits on CUDA are 20-40x SLOWER than CPU while another
  process saturates the GPU; measure before assuming CUDA helps.
- [self] A numerical gate that most healthy orbits fail is a code defect
  until proven otherwise (chord-interpolated event velocity vs 1e-10 energy
  gate). Fix the observable; do not loosen the gate.
- [self] Warp CUDA is only worth it for batched kernels; a per-particle,
  host-driven event loop is 18x slower on GPU than numpy.
- [self] PIC-2D: Jacobi-PCG is the wrong default for masked cylindrical
  Poisson at these sizes; an exact block-Thomas column factorisation is faster,
  deterministic and shared by CPU and GPU backends. Never restart CG per chunk.
- [tool] Warp: RNG state is by-value inside `@wp.func`; strided reductions,
  one host read per step; `array_sum` has no int32.
- [self] `sorted()` is not a projection onto the ordered cone; it permutes
  variable identities and stalled the LM (3/16 zero-cusp cases at 1000 V).
  Use pool-adjacent-violators (`project_nondecreasing`).
- [self] Least-squares stall floors move with start count / iteration budget
  (5%-20%); when a paper checker recomputes them, declare the reduced
  protocol, a tolerance (25%) and a recording precision (3 sig. digits), and
  cache the recomputation per process (~35 s once, not per test).
- [tool] (owlgs v2, 2026-09-03 20:40) orbit_mc v1.7 `_validate_probability` (artifact
  sealing) and `coupling_v42_handoff` require `lower <= p <= upper` verbatim, but
  `wilson_interval(0, n).lower` is a POSITIVE round-off (~1e-17) for 734 of the first
  4000 n (3, 6, 7, 12, 14, 24, ..., 384, 768, 1536, ...) and `wilson_interval(n, n).upper`
  is 1 - ulp for 1238 of them (512, 640, 1152, ...). Any case whose zero-count category
  (timeouts are ALWAYS zero) lands on such an n raises at sealing and kills the one-shot
  run; v1 survived only because 512 is k=0-safe. Frozen package -> choose case sizes from
  the safe set (128, 64, 16 are safe for both ends); never let a case size be a free
  function of the design (3 cells x 128 = 384 would have died on 30/96 designs).
- [self] (PIC plume attempt 6, 2026-09-04 04:20) A max-over-nodes statistic taken from a
  SINGLE-step particle deposit is a shot-noise extreme value, and the smallest-volume node
  decides: the axis corner node of the far plane has V = pi dr^2 dz / 6 = 6.5e-14 m^3, so one
  macro-ion (W = 6e4) reads 0.39 of the peak density and tripped the plume-boundary gate
  (0.259 > 0.25) while the interval-averaged far-field charge fraction was 0.03. Every
  density-based gate needs the peak-Debye gate's sample-size floor (>= 32 macro-particles
  per node) and should record the raw statistic alongside. Same-seed PIC reruns replay
  bitwise -> a free regression check for diagnostic-only fixes (attempt 7 matched attempt 6
  over 213 records).
- [self] (PIC attempt 7, 2026-09-04 09:50) A sample-size floor on a SINGLE-step deposit can make a
  gate inert: with the 32-macro-particle floor, zero far-field nodes were ever "resolved" in 4601
  records, so the gate could never fire. After adding a floor, check in production that the
  resolved-node count is > 0 — "the false trigger stopped" is not the same as "the gate works".
  Density gates must read interval-averaged accumulators (the maps/frame windows), not one step.
- [self] (PIC attempt 8, 2026-09-04 12:00) FINITE-GRID HEATING: the declared peak-Debye gate
  Delta/lambda_D <= 4.5 was never calibrated against measured heating and is NOT protective. The
  energy-ledger residual per 0.4 us segment went -0.5% -> +2.4 -> +5.8 -> ... -> +54.8% of electrode
  work once peak Delta/lambda_D crossed ~3.2 (the CIC threshold pi); the accepted channel-only
  plateau sits at 3.17 with residual +0.4% -> the two runs bracket the onset. Symptoms of heating vs
  a denser plateau: T_e and K_e/N_e rise WHILE I_d falls; S/N_e constant; n_g fixed point slides
  down with gross utilisation > 1 sustained by recycling. Gate on the trailing-window residual
  POWER (>= 5% of electrode work) — the CUMULATIVE residual ratio lags a runaway by ~1 us and let
  attempt 7 look healthy (1.4%) while the last segment was already 15%. Hard gate Delta/lambda_D <= pi.
- [self] (v2.0.3, 2026-09-04 13:15) Calibrate every new gate on the ACCEPTED runs before enforcing
  it: the accepted plateaus run slightly cooling (windowed residual -12.7% -> -0.2%, seed-b -1.5%,
  W x0.7 -4.2%), so a two-sided 5% residual-power bound would have killed all three before 4 us.
  The gate is one-sided (positive = heating). Also: a preflight timing at production load beats a
  cost-table extrapolation — the 33 um channel-only run measured 2.5 ms/step (6.2 h to 3 transits)
  where the attempt-8 table extrapolated 9.8 ms / 13-14 h.
- [tool] (H100 benchmark, 2026-09-04 14:30) For THIS PIC code an H100 80GB SXM is ~1x a 5090 per
  process at production load (channel-50 1.16x, channel-33 1.32x, plume-v2.0 0.98x; at seed load
  1.4-2.1x) — the step is latency-bound small kernels on 45k-135k cells, so the 5090's clocks cancel
  HBM3. Without MPS, N processes per GPU just time-slice (aggregate 0.93-0.96x). WITH CUDA MPS
  (`nvidia-cuda-mps-control -d`, CUDA_MPS_PIPE_DIRECTORY) 4 processes give 1.54x aggregate (N=8:
  1.58x, saturated) at 0.5x per-process speed -> one H100 ~ 2 5090-equivalents of throughput, but NO
  single run gets faster than on the 5090. Host factorisation on the Linux box is 0.4-6 s (vs 5-12
  min on the Windows PC) — the local slowness is a Windows/BLAS issue, not the problem size.
  Cloud value = parallel slots, not wall-clock per run. Set `slots_per_gpu = 4` with MPS.
- [tool] (H100 MPS, 2026-09-04 16:00) NEVER kill a PIC process under CUDA MPS mid-step: it raised
  Xid 31 in the MPS server, the next client to connect was torn down and its sibling got a sticky
  `unspecified launch failure` at Warp init. Check `server.log` / `dmesg | grep Xid` before
  relaunching. Follow-up: SIGTERM handler in the shared runner. Under MPS the PHYSICS state replays
  bitwise (particles, deposits, phi, currents); only float-atomic DIAGNOSTIC accumulators (T_e maps,
  peak Debye stats, ledger sums) differ at <= 2e-13 — same pattern solo-vs-solo, so MPS-neutral.
- [self] A composer that never reached finalization hid a `KeyError` in the budget block
  (`n_max_per_m3` / `n_eq_projected_per_m3`) — shakedowns must run to finalize + assess on the
  target machine, not stop at the step loop.
- [self] (v2.1.2, 2026-09-05 12:20) Stop-statistic audit complete: INTEGRAL members (residual power, S/I_d/N_e/
  n_g drifts, ignition ratios, ceilings) are immune; SINGLE-STEP members (T_e,dense, omega_pe dt) are floor-
  protected AND read None unless EVERY trailing-window record is resolved; window members carry accumulated
  floors. No raw single-node statistic feeds a stop any more. Re-read of records: the ext-val L2 stop
  disappears; all accepted verdicts unchanged. Local pytest of tests/pic2d WITHOUT `CUDA_VISIBLE_DEVICES=-1`
  uses the local 5090 (~9 min) — always set it for local runs (user directive: no local GPU).
- [self] (records, 2026-09-05 11:00) alpha-series FINAL: 1/64, 1/16 and 0.345 ALL extinguish at the 0-D gas
  (decay times 2.4 / 0.88 / 0.47 us ~ D_perp^-0.55); series verdict `inconclusive` by rule, in words "no
  self-sustained discharge for any alpha >= 1/64 under the v1.3 closure without SEE at n_g0 5.5e19". The
  two-checkpoint ignition gate was needed (1/64 passed the 1 us check, failed at 2 us). Ext-val bohm-0.4
  L2: a marginal SUSTAINED low-density discharge (I_a 1.71 vs 4.3 mA, n_i 5e17 vs 1e19) stopped by an
  ARTEFACT — the T_e,dense drift member has NO occupancy floor (densest node held 0.2-1.5 macro-electrons;
  T_e,dense = 0 in 73% of records) -> add the v2.0.4 floor to that member (follow-up). No launch 3 until a
  Brandt-geometry full-physics case at particle parity (W ~1e4) exists.
- [tool] Windows `ssh.exe` strips inner double quotes: `grep -E "a|b"` becomes a shell pipe on the remote —
  helper scripts only. PowerShell `if (git ...)` tests OUTPUT, not the exit code — use `$LASTEXITCODE`.
- [self] (full-physics prereg, 2026-09-05 10:00) A shakedown must run FINALIZE on the shakedown's own
  results and RESUME from its checkpoint — the R5 shakedown never did, so `save_checkpoint` shadowing
  `name` (every spatial-neutrals checkpoint written as `thermal_speed.*`) hid until the campaign prereg.
  Metastables at F = 1 are quantised at W_meta 4.4e5 (fraction <= 6e-6 at 0.14 us) — a declared F distortion.
  Set density ceilings from the KNUDSEN anode density with a >= 2.5x margin (1.5e21 here), fail-closed.
- [self] (alpha = 1/16, 2026-09-05 07:15) PHYSICS: with Bohm transport alpha = 1/16 the reference discharge
  EXTINGUISHES in our model (v1.3 closure, 3 mA / 2 eV exit injection, no SEE, 0-D gas n_g 5.5e19): the seed
  decays with e-fold 0.88 us ~ r_w^2 / 4 D_perp, I_d 3.1 -> 0.06 mA, the injected beam returns through the
  exit plane; no heating (residual +1.15%). Meanwhile the Brandt-point ext-val at alpha = 0.345 and
  n_g 2e20 (4x denser) SUSTAINS marginally. => a Bohm-leaky discharge needs the denser (Knudsen) gas
  that R5 provides — the operating point, not the closure alone, decides. A drift-member stop must be
  classified heating / re-equilibration / EXTINCTION / artefact before any relaunch (extinction replays
  bitwise into extinction — do not relaunch). Model v2.1.1: drift members arm only after >= 2 transits
  AND a settled-once latch on the I_d drift (< 0.05); calibrated: v4 latch closes at 2.66 transits,
  attempt 8 still stopped by the residual member. Ignition gate for the alpha-series (N_e >= 0.6,
  S >= 0.3 of the reference at 1 us) would have stopped the extinguishing run at 1.0 us.
- [tool] When re-applying a commit chain whose files cite their own SHAs, map full 40-char SHAs first —
  short-prefix replacement corrupts embedded full SHAs (a first attempt was caught and redone).
- [self] (R5 spatial neutrals, 2026-09-05 06:20) The 0-D neutral inventory EQUATED THE WHOLE CHANNEL TO
  THE EXIT DENSITY: the free-molecular closed-end (Knudsen) profile at the same feed is 5.45e20 at the
  anode -> 7.0e19 at the exit, channel mean 2.49e20 = 4.5x the 0-D fixed point. Every plateau so far
  sits at a different (too dilute) operating point than the spatial model gives — the "flames" will
  move. Physical neutral relaxation is 0.2-2 ms >> our 5-10 us runs -> a declared time-acceleration
  factor F for neutral time (default 1 = physical) is a new numerical parameter to qualify before any
  R5 campaign claims a neutral steady state. Trapped Xe resonance levels have ~0.3 us Holstein
  lifetimes << collision times -> not pooled; only 6s[3/2]_2 metastables are.
- [tool] Kn ~ 10-100 in the channel (lambda_nn 3-30 cm vs 4 mm bore): test-particle DSMC-lite neutrals
  (~4 M macro-neutrals at W_n 2.2e7, ~60/cell) cost ~10% of the step; per-cell INTEGER sinks (units
  W/2^20) with a per-cell debt carry make the atom ledger close exactly (1.4 atoms on 8.6e13).
- [self] (R4 Coulomb, 2026-09-05 05:50) The pair-mean "collision frequency" <s>/dt_c of a Coulomb operator
  is a 1/g^3-weighted mean — log-divergent, sample-size dependent, ~13x the Spitzer rate; record the NRL
  Spitzer form from n_e, T_e as the comparable number. Relaxation tests vs Spitzer must compare with the
  Landau integral of the ACTUAL distribution or keep like-particle collisions on (e-i alone leaves the
  Maxwellian and reads 0.74 of Spitzer). GPU pairing via a cell-sorted SLOT PERMUTATION (particles never
  moved) keeps the index-keyed RNG contract: Coulomb-off bitwise, graph = direct bitwise with it on.
- [tool] A Warp CPU array's `.numpy()` is a VIEW — copy before `zero_()`. `wp.utils.array_scan` and `zero_()`
  capture fine inside the step graph. On the box `python -m experiments…` needs
  `PYTHONPATH=<tree>/modern/src:<tree>/modern` (package not installed in .venv-pic).
- [self] (alpha-series, 2026-09-05 05:30) Drift-member gates ARMED AT A FIXED TRANSIT COUNT kill runs
  whose physics is still re-equilibrating: alpha = 1/16 stopped at exactly 1.00 transit, the instant the
  drift members armed, because a stronger cross-field leak moves I_d/S to a new state over > 1 transit.
  Arming must be relative to the discharge ("settled once" latch on the I_d drift, >= 2 transits), with
  the residual-POWER and accumulated-floor Debye gates as the physics protections from their windows.
  Every gate calibrated on alpha = 0 plateaus is suspect when the physics changes — re-calibrate on
  the new closure's own shakedown before sealing a campaign.
- [tool] The Write tool once leaked `NNN|` line-number prefixes into a file (caught by a syntax check)
  — always syntax-check / import-check written files before committing. Host factorisation on the
  box is 150 s when 4 PIC jobs share the CPUs vs 1.7 s solo.
- [self] (R1, 2026-09-05 03:25) Always check the literature EVENT MODEL, not just the coefficient, before
  sealing a comparison run: the v1.4 Bohm hook had the right rate (nu_an = alpha omega_ce, exact Poisson,
  KE-preserving) but the WRONG event — an isotropic redirect that also randomises v_parallel (pitch-angle
  scattering into the cusp loss cones) — while Brandt 2016 rotates only the perpendicular velocity
  about B. The sealed ext-val bohm-0.4 would have run the wrong model. v2.1.0 adds
  `bohm_perpendicular_rotation` (Rodrigues rotation by a uniform angle; v_parallel unchanged).
  Green-Kubo: D_perp = (kT_e/eB) alpha/(1+alpha^2), so Brandt's nu = 0.4 omega_ce is D = 0.345 kT_e/eB.
  Separate exact-Poisson process outside the MCC null budget is what keeps alpha = 0 bitwise.
- [tool] `launch --only` in schedule.py ignores `enabled`; a plain `launch` must never see a waiting job
  enabled with no free slot. `git pull` on the box aborts when preflight/shakedown records produced there
  were committed elsewhere — move the box copies aside first. Ruff counts from coloured output need
  `--statistics`; a `^path` regex silently gives 0.
- [tool] (R3, 2026-09-05 03:20) Cross-section provenance: LXCat exports carry ELECTRON sets only; Phelps'
  JILA data tree is gone — the BLAST-WarpX `warpx-data` mirror is the only scriptable Phelps Xe+/Xe
  source (pin its commit + sha256; state the lab-frame energy convention E = 1/2 M |v_i - v_n|^2).
  The audit's four Xe excitation levels ARE Biagi-v7.1's (8.315/9.447/9.917/11.7 eV); Biagi-8.9 has 33
  levels ~2x lower in total, Hayashi 6 partial, Morgan lumped — record cross-checks, pick one.
- [tool] Warp: variables mutated inside dynamic loops must be declared dynamic (`int(0)`, `F64(x)`).
  The stair-step cone mask makes a "free" straight-line march hit one cell before the true wall —
  geometry tests must use the stair step, not the analytic cone.
- [self] (ledger re-read, 2026-09-05 02:44) A post-hoc re-read of a predeclared acceptance must (1) carry
  BOTH readings (recorded + corrected) with the recorded verdict standing, (2) bind every input by byte
  hash (sidecar, assessment, summary, protocol), and (3) apply the predeclared decision tree
  mechanically to the corrected value and record that verdict BESIDE the recorded one (v4:
  `resolution_limited` recorded; `refinement_heating` on the corrected (b)) — never pick one. Origin
  moves during long tasks: re-check a target's current form before writing coordination notes.
- [self] (v2.0.6, 2026-09-05 01:50) EXACT recomputation beat the estimate: ss-v4's corrected residual is
  +2.46% (estimate said +1.9%) -> its acceptance (b) < 2% FAILS; every 50 um plateau was heating at
  +7-13%; plume attempts +11-67%; ext-val +62% (gate would have fired at 0.34 us not 0.73). Never
  quote a "marginal pass" from an estimate — recompute exactly first. A ledger term without W was
  invisible to every "residual ~ 0" calibration because the bias looked like cooling: audit the UNIT
  consistency of every tally term whenever a gate is calibrated. `maps.npz sample_count_e` is the SUM
  of weights, not mean occupancy. Gate decisions: 5% hard one-sided from the first complete window
  KEPT; (b) < 2% KEPT (not loosened to rescue v4). Peak-Debye gate: accumulated-particle-step floor
  64,000 gates the densest node by accumulated weight (axis columns now visible).
- [self] (ext-val diagnosis, 2026-09-05 00:50) LEDGER BUG: `inelastic_loss_j` counted macro-EVENTS x
  threshold without the macro-weight W (mcc.py tally + warp flush) -> every recorded energy residual
  in the project is biased NEGATIVE by the inelastic power (7-14% of electrode work at plateau).
  Corrected end states: v2 base 50 um ~ +13% (it WAS heating), ss-v4 33 um ~ +1.9%, 047 ~ +2.6%,
  056-L1 ~ +0.7%, attempt 8 ~ +80%. A residual that "closes to -7%" while ionisation is 10% of the
  power is a sign the sink is missing, not that the scheme cools. Verify a ledger by closing the
  PARTICLE-side identity (dKE = field work + injected - absorbed + born - W x sum n E) to round-off.
- [self] Occupancy floors make density gates BLIND to axis-peaked columns: at 1e19 / W 8.2e4 an axis
  node holds 0.76 macro-electrons per step, so the >= 32 floor never resolves it while it sits past
  pi cells/lambda_D. Express floors in ACCUMULATED particle-steps over the window (as the v2.0.2
  boundary gate does). 8.6x parity weight heats stochastically (1/N_c) before the CIC threshold —
  W parity is a resolution requirement, not a cost knob. Ext-val at Brandt's 20 um / W 82k: our
  closure (no Bohm transport) produced an ionisation avalanche (S = 2.5x feed under static 2e20) ->
  inconclusive by predeclared rule; route = fix ledger, particle-step floor, then the bohm-0.4 variant.
- [self] (GMG Poisson, 2026-09-05 00:10) Operator-dependent (Alcouffe-Dendy) interpolation +
  Galerkin coarsening on the masked axisymmetric FV operator gives a uniform 0.127/cycle contraction
  on every production mask — BUT only after fixing concave stair-step CORNERS of the cone: a coarse
  9-point stencil had mass toward a solid parent; lumping it into a non-existent node left a
  0.45/cycle slow mode. Mass must go to the surviving parent; constants must be preserved exactly on
  every Neumann row (pin it with a test). Fixed 14 V(2,2) cycles + in-graph true residual with a
  fail-closed `verify()` at host syncs (no per-step fallback: that would need a host sync).
- [tool] Under MPS, ~300-450 tiny DEPENDENT kernels cost 38-81 us each (vs 3-4 us solo), so a
  latency-bound multigrid looks slower than block-Thomas while contended; judge only from a solo
  probe. The runner honours the wall budget only at checkpoint boundaries — never wrap a PIC process
  in `timeout` (it would SIGTERM mid-step -> Xid 31 under MPS).
- [self] (PIC v2.0.5, 2026-09-04 23:30) Physics-bitwise performance changes ARE possible in this code
  and verifiable: born-ledger tallies folded into `mcc_kernel`, born deposits fused into `spawn_kernel`
  (int64 -> bitwise), window diagnostics forked onto a side stream inside the CUDA graph (Warp 1.14
  captures wp.Event fork/join), `moment_sample_interval` K (K != 1 enters config_sha256). Old-vs-new
  40k-step replay + 6k-step resume of a live checkpoint: particles/phi/counts/I_d/S bitwise, ledger
  scalars <= 8.7e-16 rel. Under MPS contention the gain shows small (x1.21 channel / x1.09 plume)
  because the sweep-launch chain dominates and saturated SMs leave nothing to overlap — measure
  speed-ups SOLO or they are underestimated. K=5 moves the gated Delta/lambda_D by 1.7e-5 median.
- [self] (v5 move, 2026-09-04 22:55) The shared PIC runner has NO clean-stop channel (no STOP file,
  no flag; Windows has no SIGTERM) and the finalizer's recovery accepts only evidenced reasons —
  a user withdrawal is recorded as "WITHDRAWN, not a result, not a failure" with `results/` renamed
  to `results-launch1-withdrawn/` so the next execution keeps `results/`. Stop at a checkpoint
  boundary by polling run_state (12 ms after `checkpoint_step` advanced). Add a STOP-file handler
  to the runner (follow-up). Windows host factorisation 365 s vs 3.1 s on the Linux box.
- [tool] `AwaitShell` sleeps without a shell id do not track wall time reliably — read the box clock.
- [self] (v2.0.4, 2026-09-04 22:50) THIRD shot-noise gate in one day: the runtime omega_pe dt gate took
  the max over EVERY node of the single-step deposit; at 20 um / W 8.2e4 one macro-electron on an
  axis node reads n = 1.3e19 (omega_pe dt 0.14), two read 0.20 -> it killed the ext-val preflight
  before the first timed step and would have stopped the production run. v2.0.4 reads the peak over
  nodes holding >= 32 macro-electrons (raw kept as witness). RULE: every single-step density-derived
  gate statistic needs the same occupancy floor; audit them all at once, not one per incident. The
  sweep's triad `omega_pe_dt_drift` still reads the RAW statistic (worktrees locked pre-v2.0.4) —
  056's 20:52 stop may be an artefact; diagnose before assessing.
- [self] (PIC perf audit, 2026-09-04 22:20; `modern/docs/pic2d-performance-audit.md` c2d3b88d) The
  step is LATENCY-dominated and the biggest single cost was two DIAGNOSTICS: the born-particle
  ledger (ke_born, pz_born) computed with 16-block strided sums + single-thread 4096-element
  `deferred_add` = ~35% of the 3.31 ms channel-33 step (the same kernel with 262k threads is 14x
  faster). Block-Thomas Poisson 29-36% (2x91 dependent launches; O(N^1.5), 2 GB reads/step on
  plume-50, ~12 GB on the 33 um plume box); per-step window-moment deposition ~20% (20 float64
  same-node atomics per electron). Physics kernels only 45-55%. Profile before optimising physics.
  Solo-fit: 0.27 ms + 4.1 us x sweep launches + 0.30 ms/GB inverse blocks + 0.97 ms per M electrons.
  Under MPS, tiny kernels are inflated x7-10, big ones x1.1-1.4 -> never rank by contended shares.
- [tool] Warp timing: `wp.timing_begin/end` with TIMING_GRAPH gives graph GPU time; per-kernel needs
  the graph off; `wp.utils.array_scan` (CUB) is not graph-captured; `nsys` is absent on Lambda Stack.
- [self] (PIC acceleration review, 2026-09-04 22:10; `modern/docs/literature/pic-acceleration-methods.md`,
  147 refs) Physics-neutral speed-ups for our explicit 2D axisymmetric PIC-MCC: Warp multigrid
  Poisson (1.3x channel / 2.2x plume; frees ~6 GB and the host factorisation), kernel fusion 480->~50
  launches + periodic cell sort (1.2-1.4x), mixed precision (1.1-1.3x) => ~2x channel, ~3-3.5x plume.
  Explicit energy-conserving (Lewis) gather removes the hard pi Debye gate and allows 33->50 um
  (~2x / ~2.5x) but puts per-cusp sheath drops (1-3 cells) and momentum conservation at risk;
  Barnes/ECsim semi-implicit coarsening 3.5-10x (WarpX Hall-thruster precedent) leaves T_e and cusp
  sheaths unresolved by construction. Permittivity scaling gamma=4 is 8x raw but distorts exactly
  the sheath / wall-ion-energy / peak n_e quantities we claim -> screening only. The literature has
  NO energy-conserving PIC of a cusped-field device: every scheme change must be validated against
  our own explicit 33/25 um ladder.
- [self] (ss-v4 verdict, 2026-09-04 19:40) The 50 um PIC plateau is RESOLUTION-LIMITED: at 33 um
  I_d +10.4%, peak n_e -21%, T_e,peak -24.5% (S, utilisation, n_g, I_beam within 10%). Lower T_e and
  peak n_e at finer grid = less finite-grid heating; the shifts point in the W x0.7 direction at ~2x
  its size, so grid and particle-weight effects are entangled — a W-only follow-up is still owed.
  "Converged" may not be said of either grid until the 25 um point (v5) reports.
- [self] (frozen-contract fix, 2026-09-05 15:30) Post-execution verifiers must recompute sealed digests from
  the RECORD'S OWN INVENTORY at the lock commit (`cft_revival.provenance`: batch `git cat-file`, fail-closed
  on unreachable commit / missing blob / CR) and RECORD live-tree drift (added/removed/changed) — never
  assert live == sealed after execution; the sealed `*_current` verifier stays for the pre-execution
  lifecycle (`prepare` / `execute`), and editing it would itself change `experiment_code_sha256`. Audit
  documents pinned by the paper are immutable: compare doc tables clause-wise, not verbatim. A gate that
  can be `not_evaluated` (CUDA hidden) must fail CLOSED in tests — bind to the recorded gate file instead.
  Same time bomb remains in l1b_hemp_confirmation (v1/v1.1) and MDO v1's hash-scoped live assertions —
  apply `frozen_contract.py` before the next shared-package commit. `rg` from the root skips .worktrees/
  (`--no-ignore`).
- [user] (2026-09-05 13:12) The roadmap canvas is now PERMANENT in the repo: `modern/visualization/
  roadmap-status.html` (self-contained React build) generated from `modern/visualization/roadmap-status/
  roadmap-status.canvas.tsx` by `build.mjs --sync` (copies the live canvas). Every canvas fold must now
  ALSO run the build, update the anchor sidecar and commit the HTML + canvas copy. `.cursor/memory/` is
  COMMITTED from now on (16fea450) — edits to the scratchpad/devlog show as modified files; commit them at
  milestones. main == feat/sota-foundation (fast-forward 317 commits at 16fea450; both 3aa7b0fb); keep
  main fast-forwarded after each green milestone (`git push origin feat/sota-foundation:main`).
- [tool] Full CPU-only suite: `modern/tests` ~32 min single-process (pytest-xdist not installed), `paper/tests`
  285, `check_paper` ~190 s. Always `CUDA_VISIBLE_DEVICES=-1` locally.
- [user] (2026-09-05 01:49) FIXED ORDER: (1) finish the 2-D PIC physics, (2) design the 3-D PIC and
  VERIFY it works, (3) then the AI run. The surrogate architecture reference is Arena Physica's
  Heaviside-1 / RF Studio (transformer-core physics operator; "much closer to what we are doing") —
  and Reality-Simulator/ai contains TWO competing models (read its docs, not just the code).
- [user] (2026-09-05 01:37) ROADMAP AFTER 2D (superseded by the 01:49 order above): move the PIC to 3-D (azimuthal physics: E x B drift
  instabilities / anomalous transport that 2-D axisymmetric cannot carry). Plan 2-D -> 3-D transfer
  (code architecture, cost, surrogate transfer learning) as a later phase; the physics audit must
  state exactly which claims need 3-D.
- [user] (2026-09-05 01:12) PRIORITY: make the PIC "the best (SOTA) PIC plasma simulator" — as fast AND
  accurate as possible, ACCURACY FIRST: "every physical interaction of consequence" must be in it
  (SEE from dielectric walls, dielectric charging, anomalous transport / 2D limitation, Coulomb
  collisions, full Xe collision set incl. metastables/CEX/Xe2+, spatial neutrals, cathode model...).
  LATER: an AI surrogate trained on PIC outputs, following the architecture in
  C:\Users\Angus\Desktop\projects\Reality-Simulator\ai (built for welding) — study/plan now, build later.
- [user] (2026-09-04 21:23) DIRECTION CHANGE: no more 0-D model development ("ancient history") —
  the plasma-network v2 calibration / MDO v3 closure route is DROPPED as a development target
  (the mini-sweep's closure targets remain recorded data only). Focus = the full PIC-MCC: convergence
  ladder, design sweep, code-to-code validation, plume/thrust. ALL PIC runs execute on the Lambda
  H100, never locally (the local v5 launch was withdrawn and relaunched on the H100).
- [user] (2026-09-04 20:55, angry) Heavy compute belongs on the Lambda H100, NOT the local PC — the
  user's machine must stay usable. Any campaign / ladder / replication work must be scheduled on
  the cloud box (or explicitly approved for local). A sub-agent's queue runner RE-LAUNCHED killed
  cases; when stopping an agent, kill its queue/parent first, then the workers, then verify with a
  process filter. Hybrid L2 v2 is PARKED by the user (branch `feat/hybrid-l2-v2` @ 277fc911, not
  merged): comparison FAIL 24/28 (I_d 7.52 vs 3.44 mA; anode ion fraction 0.155 vs 0.014), PIC/L2
  wall-clock 1.66 (no speed advantage); diagnosis = linear cusp conductance G_k without sheath-
  limited saturation. Do not resume it without the user asking.
- [self] Sub-agents given "CPU primary, short CUDA tests OK" still spun up 11 CUDA processes at 100%
  on the local GPU while a preregistered run was executing. State GPU ownership EXPLICITLY in every
  brief ("the local GPU is owned by PID X; you may not start CUDA processes") and check nvidia-smi
  compute-apps before launching anything preregistered; record contention in the launch log.
- [self] (ext. validation v0, 2026-09-04 17:40) A predeclared field gate can fail on a MISREAD
  anchor ("e.g. at Z = 11 mm" was an example point, "0.2 T near the cusps" was the low-field region):
  in a DRAFT, revise the gate with a recorded genealogy (original rule + outcome kept in
  binding.json) rather than silently. Published PIC numerics of a self-similar SCALED system are
  original-system values: Brandt's omega_pe dt = 0.2 in their frame is 0.56 in ours, so "same
  dt/dx" is not "same resolution" — recompute admissibility in our frame.
- [tool] The root `*.npz` gitignore silently dropped an LFS sidecar from a data commit — run
  `git lfs ls-files` after every fields/data commit and add the `!path/**/*.npz` negation.
- [self] (interim viz, 2026-09-04 16:40) A running PIC job has no summary.json; the v0.2 renderer
  only needs grid/W/dt, so a synthesised summary + a symlinked frame mirror gives read-only
  rendering of LIVE runs (`modern/visualization/interim_sweep_panel.py`, rerender.sh on the box).
  `run_state.frames_written` lags the frame files by one (frame written before its checkpoint).
  Status-series strips need robust 1-99 pct axis ranges — the seed transient (I_d 157 mA at t=0;
  single-step Delta/lambda_D ~1e8 while lambda_D is undefined) flattens every axis otherwise.
- [tool] `render_pic2d_video._MASKS_CACHE` is keyed on `id(grid)`: a freed grid's id can be reused
  in one process -> stale masks. Write the cache entry explicitly or key on grid values.
- [tool] Python 3.12 `@dataclass` inside a module loaded via `spec_from_file_location` fails
  (`'NoneType' object has no attribute '__dict__'`) unless `sys.modules[spec.name] = module` is set
  before `exec_module`.
- [tool] PowerShell `Set-Content -Encoding utf8` writes a BOM (it leaked into a commit subject);
  use `[IO.File]::WriteAllText(path, text, [Text.UTF8Encoding]::new($false))` for git message files.
- [tool] Linux/OpenBLAS (Haswell DYNAMIC_ARCH) vs Windows: CPU-derived float arrays differ at ULP
  level (P2->PIC sampled field map hash d30d2d24 -> 1f124047; dashboard payload reductions; a
  Gauss-law residual on a near-zero node at 1.07e-9 relative). Byte pins of derived arrays are
  anchor-platform-only; cross-platform checks need scale-aware tolerances; bindings must hash
  stored file bytes, not derived maps.
- [tool] ssh from PowerShell: nested quotes with `$(...)`/`\"` inside a double-quoted remote command
  break (and `2>/dev/null` gets parsed by PowerShell as a local redirect to C:\dev\null). Write a
  .sh helper, scp it, `sed -i 's/\r$//'` (Windows CRLF), run it. tmux commands with env vars also
  need a script, not inline `export ... ;`.
- [self] Resolution cost reality (RTX 5090, this code): plume box 48x12 mm at 33 um = 47.5 h to
  3 transits; channel-only 3x24 mm at 33 um / 1.4 ps = 13-14 h. Smaller Delta t does not fix
  Delta/lambda_D; lower W is statistics only; no implicit/energy-conserving scheme available.
  For a plume run at 50 um the peak n_e must stay <= 1.4e18 (current-limited cathode ~3 mA or
  lower mdot). Any "development" thrust number with > 5% residual power is non-quotable.
- [self] (v2.0.2, 2026-09-04 10:40) Gate design that worked: read the SAME window accumulators the
  maps/frames use, only at the existing series-record host sync (no per-step sync, nothing inside
  the CUDA graph); bridge the runner's 400k-step accumulator resets with a carry keyed on the
  diagnostic generation; express the floor in accumulated particle-STEPS (64,000 = 32 crossings x
  ~2000 steps a 15 km/s ion spends on a 50 um node) so fast beam ions count as independent
  crossings; and calibrate on real maps until the gate is LIVE (77-121/481 nodes resolved,
  reading 0.025-0.034 vs threshold 0.25). Accumulation turns exact map parity into
  allclose(rtol 1e-12) across cpu/warp-cpu/cuda (device atomics).
- [self] (v2.1 prep, 2026-09-04 11:20) Classify grid planes by INDEX, never by a floating-point z
  comparison: `z == 0.044` with dz = 50 um misclassified node 480 (0.044/880) and silently lost 180
  plume cells from the exit-plane diagnostics (v2.0 masks pinned unchanged, so no result changed,
  but the same code at 0.06/1200 would have failed too).
- [self] A far-field Dirichlet (chamber-reference) plane must sit past the 10% axis-density point
  and OUTSIDE the acceleration region (v2.0's 36 mm plane had 10% of the axis potential drop still
  ahead of it; 14.6% of peak n_i at the wall). Fit the decay from the previous run (exponential
  and conical power law bracket z1) and check the FEM box actually covers the PIC box: the v2.0
  far plume carried a 15% level-1 field truncation at 36 mm. The qualification chain's
  `domain-padding-1.5` checkpoint (r <= 48.75, z <= 60.75 mm) is a ready far-field source.
- [self] Runner `status()` keyed `finished` on a stale summary.json; run_state is the authority,
  and a resume must demote the previous terminal state to `history` and write finished=false
  BEFORE its first step.
- [tool] Host telemetry must be NaN-safe and off the step loop: `nvidia-smi` timed out (5 s) on
  17/238 samples under desktop contention, put `float('nan')` into summary.json, the canonical
  JSON writer refused it, and the finalizer died AFTER the run had stopped correctly on budget,
  leaving run_state at finished=false. Sampling cost 3.9% of the wall budget. A finalizer must
  leave either a terminal state or a recorded `finalization_error` — never silence.
- [self] Windows GPU memory "held by nobody" (5.3 GB, 2% util, no compute apps) is desktop apps
  (Chrome/Cursor/Discord/Edge) — check `nvidia-smi --query-compute-apps` before suspecting a zombie.
- [self] Concurrent host factorisations oversubscribe BLAS threads (one took 20 min without
  finishing when two ran at once) — run PIC preflight/resume checks sequentially.
- [self] Per-frame PIC ionisation-rate maps are shot-noise by construction (30 ns frames:
  72% of resolved nodes hold zero events, one axis-node event sets the colour top). Render
  with a rolling window + event-count mask + fixed percentile colour scale, and label the
  window/mask on the panel; the integral (S) is fine even when the map looks "sketchy".
- [self] (renderer v0.2, 2026-09-04 05:05) Per-node PIC "event counts" recovered from a bilinear
  deposit (rate x V_node x dt / W) are FRACTIONAL (18% of nodes); integrality holds only for the
  domain sum (integer to 2e-5). Treat them as effective counts (same semantics as `sample_count_e`),
  never round them. Window choice rule that worked: smallest K where the median resolved ionising
  node holds >= 10 events (K = 11 frames = 330 ns here); mask >= 20 windowed events; fixed scale at
  the 0.5-99.5 percentile of resolved windowed values over the run.
- [tool] `cusp_planes()` in the PIC dashboard/video path swallows the protocol-hash-drift error, so a
  re-render of an older run silently loses the cusp overlay -> pass `--cusps 0.006028 0.012 0.017972`
  explicitly. argparse needs `--suffix=-v2` (equals sign) for a value starting with `-`.
- [tool] ffmpeg single-frame extraction on ffmpeg 8: `-vf "select=eq(n\,N)" -fps_mode vfr -frames:v 1
  -update 1 out.png` (without `-update 1` the image2 muxer refuses a non-pattern filename). Also:
  this PowerShell session sometimes returns empty stdout for Get-ChildItem/Test-Path after ffmpeg
  ran; `cmd /c dir` still works.

### Repo/tooling

- [tool] PowerShell `>` redirect of Python stdout produces UTF-16 with BOM; read it back
  as utf-16 or write the file from Python.
- [tool] Every text write that feeds a byte hash needs `newline="\n"` on
  Windows (`write_text(..., encoding=...)` alone emits CRLF). The v4 bundle
  recorded CRLF sidecar hashes that Git normalised to LF.
- [tool] Windows CRT caps open descriptors at 8192; bundles with >8k files
  need the pin cap (now 4096 in experiment_runtime) or chunked inventories.
- [self] When a run fails, check whether an uncommitted fix already exists in the
  main tree (mtimes, `git status`) before writing a new one.
- [self] Committing from many worktrees leaves the main tree diverged and dirty
  with duplicates; reconcile (fetch/rebase/push) after every accepted stream.
- [tool] Reconciling a dirty tree: classify every path by blob hash against
  origin BEFORE stashing; use `git restore --source=stash@{0}^3` selectively
  instead of `git stash pop` (pop aborts wholesale on "already exists").
- [tool] Large fem_reference JSON are Git LFS pointers on origin; compare
  local sha256 against the pointer oid, not the pointer text.
- [tool] To compare a working-tree file with origin, use `git hash-object` vs
  `git rev-parse origin/branch:path`, or `git diff origin/branch -- path`.
  Piping `git show` through PowerShell `Set-Content`/`Out-File` corrupts
  line endings and produces fake whole-file diffs.
- [tool] System `core.autocrlf=true` silently rewrote 760 hash-bound files to
  CRLF on checkout. Root `.gitattributes` (`* text=auto eol=lf`) now pins LF
  (commit fab0eccc). Any worktree created BEFORE fab0eccc must be re-smudged
  (delete + `git checkout --` the `w/crlf` files) before computing code or
  artifact hashes. `git checkout -- path` alone does NOT re-smudge; the file
  must be removed first (stat cache).
- [tool] The Write tool emits CRLF on Windows; after writing a tracked text
  file, re-checkout it so the working copy is LF.
- [self] Tests that compare a fixed fixture timestamp to the wall clock are
  time bombs (tests/coupling failed 24 h after the fixture was written).
  Fixed in `4661a7be`. Guardrail: any coupling builder/verifier call in tests
  must pass `reference_time_utc=NOW` (or bind it once via `functools.partial`
  on the module alias); prove it by monkeypatching `datetime.now` to raise.
- [tool] Anything whose bytes are hashed must be written as bytes or with
  `newline="\n"`. `Path.write_text` without it emits CRLF on Windows; Git
  (`eol=lf`) stores LF, so recorded byte hashes become unreproducible from a
  checkout while every content hash stays valid. Readers using universal
  newlines hide this; only byte-exact manifests see it. Guard with a
  fail-closed AST lint, not a one-off grep (orbit_mc v1.7).
- [tool] PowerShell: `git commit -F -` with a here-string does not read
  stdin; write the message to `$env:TEMP` and `-F <file>`. .NET
  `[IO.File]::ReadAllBytes` ignores the PowerShell `cd`; use Python for
  byte checks. `Out-File -Encoding utf8` adds a BOM.
- [tool] Same-basename test modules (`test_specs.py`, `test_audit_hardening.py`)
  clash when several `tests/<pkg>` dirs run in one pytest invocation; run
  the directories separately.
- [tool] `.gitignore` negations for tracked result files must follow the
  generic `results/` and `*.npz` rules.
- [tool] `sci`-formatted macros carry `$...$`; inside a caption's math they
  give "Missing $ inserted" two files away. `\allowbreak` in `\texttt` gives
  breaks but no stretch; use `\hspace{0pt plus 1.5pt}` after operators.
- [tool] Per-file `git rev-parse`/`show` is ~170 ms here; `git ls-tree -r -z`
  per commit + one `git cat-file --batch` binds 14 files in ~1 s.
- [tool] canonical JSON integers are signed 64-bit; uint64 seeds (sha256[:8]) must be
  emitted as decimal strings (`CanonicalizationError: integers must fit signed 64-bit`).

### Canvas/reporting

- [self] Pooled equal-weight probabilities over strata are design averages;
  report per-stratum structure (here bimodal by cell) before any pooled
  number.
- [user] Every progress bar must carry its basis and a "why" (building /
  failed gates / blocked by physics / needs data / accepted). Stale
  pre-audit bars next to audited phases read as dishonest.
- [user] The canvas must answer "where are we" on the first screen in
  <10 s: one-line header + chips, stage strip, <= 8-row "right now" table,
  <= 6 key findings, numbered "next". Everything else collapsed into a
  Details tab; archive old milestones. Never append run-on status prose to
  the header again.
- [user] Prefer STAGE LADDERS over percentages for status: Specified ->
  Code written -> Tests pass -> Runs on real inputs -> Preregistered run
  recorded -> Numerically accepted -> In the paper -> Externally validated,
  plus a merged flag and stop-annotations (RUNNING / FAILED GATES / NULL /
  BLOCKED BY PHYSICS / NEEDS DATA). Percentages only inside the audit card.
- [self] Low bars are rarely "testing remains": surrogates failed their
  gates for lack of physically-varying data; hybrid is blocked by the
  topology null; external validation needs a self-consistent prediction.
  Unblock by producing the missing DATA (geometry wall-loss dataset).
- [user] Keep the "Open Cft Roadmap Status" canvas
  (`.cursor/projects/.../canvases/open-cft-roadmap-status.canvas.tsx`) up to
  date at every milestone (each agent completion, each push, each campaign
  state change). Read the canvas skill before editing it.
- [self] (canvas 2026-09-05 06:40) The scheduler on the box CAN be read for a brief without touching
  anything: `ssh -i ~/.ssh/lambda_h100 -o BatchMode=yes ubuntu@68.209.75.2 'cd
  /lambda/nfs/h100-files/cft/uni-project && python3 modern/tools/cloud/schedule.py status'` plus the
  two slot-waiter logs (`$WORK/r1/queue.log`, `$WORK/pe-queue/queue.log`), `tmux ls` and
  `nvidia-smi --query-compute-apps` — all read-only; ~5 s. Quote the scheduler's ETA with its basis
  (steps-to-target x CURRENT ms/step): at the seed load it under-states a run that grows toward a
  particle cap (bohm-0.4: 3.6 h scheduler vs 12 h by the 12 M-load preflight).
- [self] (canvas 2026-09-05 06:40) When the brief's interpretation of an event and an UNFOLDED origin
  commit disagree (1/16 stop: "re-equilibration" vs the record's "extinction"), the canvas folds the
  brief's reading as the state AT THE FOLD and quotes the later record beside it, marked unfolded —
  never silently adopts the newer reading, never hides it. Same for a row the user says "unchanged"
  when the live reading differs (v4_fast ETA): leave the row, put the live number where live numbers
  live (GPU row / Right-now) with the row's figure named as the brief's.
- [self] (canvas 2026-09-05 06:40) The 2.2 MB canvas is edited with StrReplace on unique anchors
  (the first ~200 chars of a `name:` / `lineage:` / `stop:` string; the `id:` line for a row); read
  long lines through PowerShell `Substring` heads rather than the Read tool (a 42 k-char line blows
  the 100 k-char read cap even with a 26-line window). Prepend new text and keep the old after "The
  HH:MM reading follows. —" — the file's archive convention. Recount in Node from a temp script
  OUTSIDE the repo (`%TEMP%`), delete it after.

### User preferences

- [user] For any blocker, search the literature for SOTA approaches and
  documented pitfalls before iterating further; cite verified DOIs only.
  Reviews live under `modern/docs/literature/`.
- [user] Inspect actual environment, process, package, disk, and network state before retrying failed provisioning.
- [user] Keep ML provisioning isolated in `.venv-sota`; never install globally.
- [user] Do not claim GPU support unless a real CUDA tensor operation succeeds.
- [user] (2026-09-04 12:40, angry) NEVER create git worktrees beside the repo in
  `C:\Users\Angus\Desktop\projects\`. All agent worktrees go under
  `C:\Users\Angus\Desktop\projects\uni-project\.worktrees\<name>` (gitignored since `25e86dca`),
  and every agent removes its worktree (`git worktree remove` + `git worktree prune`) when done.
  49 stale `uni-project-*` folders had accumulated; 41 removed + 27 merged branches deleted.
  Large untracked run outputs (frames, videos) stay inside the experiment's `results-*/` (ignored)
  in the persistent runner worktree; never loose in the projects folder.
- [user] Stale `python -m http.server` dashboard previews hold their cwd and block worktree
  deletion on Windows — always stop preview servers before removing a worktree.

## Session summary 2026-09-02/03/04 (condensed)

One bullet per session entry, chronological (AEST). Full entries (task summary, mistakes and fixes, what
worked, guardrails) live in the archived file listed under Archive.

- 09-02 05:00 `.venv-sota` provisioning: three repeated PyTorch commands had left six orphaned launcher/child
  processes (12 stopped at the final audit); healthy Python 3.12 venv kept; torch 2.13.0+cu130, BoTorch 0.18.1,
  GPyTorch 1.15.2, pymoo 0.6.2; CUDA float64 on RTX 5090 sm_120, GP posterior, qLogNEHVI/qLogNParEGO and
  NSGA-III/MOEA-D smokes all pass. No commit.
- 09-02 23:30 roadmap stall diagnosed: orbit wall-loss v1/v2/v3 each consumed a prereg cycle and died on code
  (v3: `_first_cylinder_crossing` fraction 0.0 -> `step_dt = 0` -> STEP_LIMIT witness rejected; integrator v1.4,
  commit `25dbeaaf`); uncommitted v1.5 fix (102/102 tests) had never seen the real P2 field; main tree
  `ahead 1, behind 13`, 84 dirty entries, 33 worktrees. Delegated v1.5 real-field shakedown + main-tree reconcile.
- 09-03 00:06-00:45 orbit_mc v1.5 merged `7cf65053` (real-field shakedown 512/512 validators on four cases);
  `.gitattributes` eol=lf `fab0eccc` (760 files re-smudged); l1a_plasma_coupling `40dcaa4c`; orbit_mc v1.6
  `3ab50ef5` (Boris sub-push event velocity, energy error 0.0, 1e-10 gate 512/512, 120 tests); Warp CUDA 18x
  slower for the per-particle path -> CPU.
- 09-03 00:25 tests/coupling wall-clock time bomb: 9 tests expired at 2026-09-02T12:00Z; fixed at the module
  alias with `partial(..., reference_time_utc=NOW)` (`4661a7be`); 143 passed; determinism proved by poisoning
  `datetime.now`.
- 09-03 01:20 wall-loss v4 ACCEPTED (`23d37bee` tests / `757e365f` prereg / `6922a3cf` result): one execution
  667 s, 15/15 binding gates, 289 validators / 0 failures, 4608 orbits, wall-hit 0.641-0.645, escape 0.355-0.359,
  0 reflections, convergence changes <= 2/512; shakedown had caught a `zip(..., strict=True)` bug in assessment.
- 09-03 01:45 orbit_mc v1.7 LF sidecars `cc4bd5e1` + v4 posthoc audit `258f69b2`: exactly 9 sidecars differ by EOL
  only, 378 entries byte-exact, evidence ACCEPTED; 7 frozen-v4 tests made lifecycle-aware; `results/` tree
  `447a5cf7` unchanged.
- 09-03 02:06 plasma topology dashboard `16670281` (72 hashed sources, 17 tests): strict wall-cusp topology found
  nowhere (characterization v1 0/0 over 56 designs, four-cell v2 0/128).
- 09-03 02:40 v4 dashboard + paper evidence `bc2f8e47`, `ea867bf1`, `5b85d2ad` (681,963 B HTML, 130 macros,
  13 + 8 tests; branch stays at pre-rebase `fb4117e4`); five stale http.servers on port 8765 from other agents.
- 09-03 02:50 stale design gallery fixed `8466c37a`: `config_sha256` pin was the CRLF-smudged `a4703ac1`,
  committed LF blob `2d727b1a`; tests/visualization 93/93 (was 65 passed / 2 failed / 13 errors).
- 09-03 03:20 paper: wall-loss v4 admitted `6f3e6dd5` (pushed as `0fabda2c`, rebased): new gate kind
  `numerical-campaign`, GATE-WALL-LOSS-V4, CLM-012..017, 14 adversarial tests (33 paper tests), 11-page PDF
  `bdfdba4c`.
- 09-03 03:15 PIC-2D phase 1 `dd5f2ff1` (after `53ac3b02`, `f44a7399`, `d58fdca1`): `cft_revival.pic2d`, 58 tests,
  Poisson order 1.999; snapshot v1 stopped fail-closed by the omega_pe dt gate at 49-60 ns.
- 09-03 03:46 test health `9e68df21..7a30fc2e`: modern/tests 1677 passed / 0 failed / 5 skips in one invocation;
  pic2d phase 1 merged `df4b2d77..62de2ca3`.
- 09-03 04:55 paper: L1a sweep v2 + topology nulls admitted `f171e9ec` (four-cell v2 EOL audit `605be5ce`: CRLF
  digest `ec2e9a73` of LF blob `5c195119`); gate kind `numerical-screening`, CLM-018..028, 25 tests (58 paper
  tests), 17-page PDF `6b4c6978`.
- 09-03 07:10 PIC-2D phase 2 (`3a42bcd7`, `a0fc4a20`, `1cdaae80`): 40.7 -> 5.46 ms/step at 5.4 M; snapshot v2 no
  plateau, density 3.7-5.9x the a-priori ceiling; a `Set-Content -NoNewline` one-liner destroyed an untracked runner.
- 09-03 07:20-08:03 PIC phase 3 `44b7c8dc` (tau_i,eff 2.4 us, nu_iz*tau 2.9 -> avalanche; v1.2 at n_g 1.5e19 no
  ignition); roadmap audit `cc7706b2`: 63 % (58-68) vs canvas 70 %, suite 1702/0/5.
- 09-03 09:18 PIC v1.3 neutral inventory `520e6b41..8babb31e` (388 tests); attempt 2 (n_g0 5.5e19, seed 5e16,
  0.0186 mg/s) igniting, PID 40636.
- 09-03 10:25 MDO L0 v1 ACCEPTED (`fdc6b37d` / `4898d0fd` / `c553124b` / `e642f38c`): 864 evaluations, 8/8 gates,
  28 min; qLogNEHVI HV 0.00386 beat LHS 3/3 and NSGA-III 3/3 (1.02x the 8192-point dense reference); Kornfeld
  solver probe closes 13/80, all at p = 0; five shakedowns before the freeze.
- 09-03 11:35 four-cell closure analysis `266d8a99`: on the R00-R26 manifold R27 is exactly
  `2*(je3(1-p4)+I4)*(phi4-Ua) + EI*(p1 je0 + p2 je1 + p3 je2)` -> no root for interior p > 0; `sorted()` ->
  PAV projection (zero-cusp grid 16/16); correction PROPOSED_NOT_ACCEPTED; 24 tests.
- 09-03 12:05 paper: MDO v1 admitted `ba6875f6` (pushed `9f351776`): GATE-MDO-L0-V1, 334 `Mdo` macros,
  CLM-029..035, 25 tests (83 paper tests), 22-page PDF `e7900c10`.
- 09-03 12:18 PIC steady-state v2 reached a plateau: 3.2 transits, 7.68 us, 5.12 M steps, 2.8 h; I_d 3.5-4.0 mA,
  n_g 2.95e19; first PIC plateau in the project (development, single seed).
- 09-03 13:10 MDO v1 posthoc audit `6cb9a1af` -> `e9f9af16`: ACCEPTED WITH DISCLOSURES; 137/137 byte-exact,
  replays bit-exact (BO seed 101 took 3708 s under contention), six disclosures F9/F10/F22/F26/F27/F28.
- 09-03 13:40 paper: closure analysis admitted `d09ffee2`: new gate kind `analytic-consistency`, 163 `Fcc` macros,
  CLM-036..044, 28 tests (111 paper tests), 27-page PDF `6ac978b2`.
- 09-03 13:55 geometry screening v1 RECORDED (`484335c2` / `c86bfca3` / `ce7cb895` / `5f4a6426` / `ab7c2897`,
  merge `22e2156b`): 96 L1a designs, 196 cases, 100 352 orbits, 6664 validators / 0 failures, 95 min (70
  projected); P(wall) 0.375-0.869; reflections in EVERY design (22 %); shakedown caught a numpy-bool
  canonicalization error.
- 09-03 14:05 PIC phase 4 (`24ab82f4`, `a707fc1a`, `5564480a`; ff `c32dd780`): plateau drifts I_d +0.084 %,
  N_e +4.98 %; I_d 3.44 mA, 46 % utilisation (gross), peak n_e 1.64e18 = 4.1 n_max at 3 lambda_D; seed-b PID 49716.
- 09-03 14:45 surrogate v1 REJECTED (`aa9349a9` / `b602d147` / `b400d924` / `bfe123d4`): pooled RMSE 0.0562 (gate
  0.05), ridge 0.0546, coverage 0.80; step discontinuities of the design -> geometry map carry the signal.
- 09-03 14:55 paper: screening v1 admitted `3003325d`: GATE-WALL-LOSS-GEOMETRY-SCREENING-V1, 271 `Wlg` macros,
  CLM-045..052, 28 tests (139 paper tests), 33-page PDF `67a531f9`; Spearman rho L -0.05 / r_w -0.12 (no design rule).
- 09-03 15:35 surrogate v2 rejected (`21118507` / `503bf87f` / `a2b503be` / `783a82c6`): derived features cut pooled
  RMSE 0.0562 -> 0.0337 (pass), cells 0.0904 (fail), ridge 0.0334 -> 2x-baseline gate unmeetable; binomial floor
  0.035/cell is the limit.
- 09-03 17:00 MDO L0 v2 ACCEPTED (`19c91a90` / `99914dc2` / `a003f766` / `0ea33a7e`): 1440 evaluations, 12/12
  gates, ~83 min; qLogNEHVI 1.13x dense (seeds 202/303), seed 101 stuck on design 50 (0.49x); robust front designs
  49/50/94; CL-1 vs CL-2 Jaccard 0.
- 09-03 17:20 PIC seed-b comparison (`41ccb1ef`, `96220ffc`): <= 1.1 % on every plateau quantity; W x0.7 launched
  (PID 9856).
- 09-03 18:10-18:40 literature reviews: reduced models `66879e00` (72 refs), PIC-MCC `ccb22d5d` (116 refs; pushed
  `bf43a7fa`), surrogate/MDO/validation `b6bb6215` (157 refs; pushed `af98b3dd`); the brief's "Ma 2024 AST" DOI is
  a FEEP paper (Yeo 2024); no TU Berlin HEMP dataset exists.
- 09-03 18:05 paper: MDO v2 admitted `a3793c27`: GATE-MDO-L0-V2, 611 `Mdb` macros, CLM-053..060, 26 tests (165 paper
  tests), 41-page PDF `86796210`; 91 infeasible = 88 BO + 3 NSGA-III.
- 09-03 18:35 literature synthesis `8674cc5a` (pushed `11a10873`): 60 atomic recommendations, 53 ADOPT / 6 DEFER /
  1 REJECT; canvas `Literature roadmap` tab, 35 ladder rows, stage counts recomputed in Node.
- 09-03 19:05 PIC v1.4 `112bb250`: wall-ion recycling, peak-node Debye gate, whole-step CUDA graph (bitwise vs
  direct); 111 tests; expected fixed point n_g* ~4.1e19, net utilisation ~25 %.
- 09-03 19:40 plasma v2 `fb5408bf` (commit `e75151ce`; branch stays `ea798971`): rows R28-R37, rank 21 -> 28 -> 31,
  53 tests; SCL sheath closes 73/80, no-emission 0/96; Kornfeld DM9.2 reproduced with phi_1 = 14.07 V.
- 09-03 20:05 cusp topology v3 REJECTED (`bce595dc`, `69159934`, `8cbcdbe6`, audit `9fa6359a`: 26/206 axis clusters
  dropped by an `r_m == 0.0` filter) -> v3.1 ACCEPTED (`ca811d11`, `988220f3`, `1600cfd3`, `cec47f12`, dashboard
  `9abbd537`): 281/281 stable; N stages -> N-1 wall cusps (83/96); P2 cusps 6.028 / 12.000 / 17.972 mm.
- 09-03 20:35 TWT/PPM review `beb4772c` (51 refs): every recorded field is single-harmonic PPM at the wall; wall
  cusp / axis peak 0.45-0.61 = I_1(pi r_w/L); Koch rho unreachable (max 1.03 of 96); v4 zero reflections = launch
  0.5 mm from the magnet centres.
- 09-03 21:35 paper: topology v3.1 admitted `13d8ac6a` (pushed `726c8a69`): GATE-CUSP-TOPOLOGY-V3-1, 430 `Ctv`
  macros, CLM-061..068, 32 tests (197 paper tests), 49-page PDF `34e11c8e`.
- 09-03 21:40 L1a sweep v3 ACCEPTED (`b04d5935` / `1923ef76` / `2cfe8223` / `44d0c63c`): 128 Sobol designs on
  r_w/L 0.215-1.235 + 96 v2 held-out, 11/11 gates, 29 min; 15/128 HEMP-like (rho >= 1.5); realised threshold
  r_w/L 0.745 vs I_1 estimate 0.617; v2 region 0/102, max rho 0.993.
- 09-03 21:55 PIC v2.0 plume (`f7169279`, dashboards `f3732d9a`, W x0.7 record `542496fb`): 241x721 nodes,
  L_plume = 0.5 L_channel (bounded by the FEM domain z <= 36.25 mm); 4.2 ms/step at 0.55 M; 138 tests.
- 09-03 22:55 frame recorder + video renderer `1fb8561d` (after `ff8e0baa`, `6bd5e5b0`, `e3e9167d`, `9c7d944a`):
  exact interval averages from cumulative-sum differences, 146 tests; W x0.7 vs base I_d +5.7 %, peak n_e -12 %;
  plume launch 3 PID 28860.
- 09-03 23:15 screening v2 ACCEPTED (`a7a884bf` / `cef1ee59` / `26029b72` / `bb756418` / `eef7ac82`, merge
  `066234d9`): 97 designs, 377 cells, 1105 cases, 104,832 orbits, 86 min; all 181 interior cells at P(wall) = 1;
  exit-side median 0.50; EMFILE at publication (16,957 files > 8192) -> post hoc manifest + 4096 pin cap, disclosed.
- 09-03 23:55 paper: sweep v3 admitted + reflections re-scoped (`88386161`, `ba852122`): GATE-L1A-SWEEP-V3,
  CLM-069..076, CLM-016/017/044/052 amended; 228 paper tests; 56-page PDF `b2440f6e`.
- 09-04 00:20 plume attempt 3 no ignition (38 frames -> first real video): field-line tracer shows the cathode
  annulus on pole-face lines (0/24 connected); cathode moved to the channel tube (24/24) + connectivity gate +
  two-stage ignition gate `eb8585c3` / `c6219bf3`; records `2b3372a0` / `978000c4`; merge-back `f5582255`;
  attempt 4 PID 53756.
- 09-04 01:35 plume attempts 4-5 ignited then n_g crashed; relaxation guard `ad5be433` / `0f583df9` (kept as a
  guard); ROOT CAUSE: MCC neutral density baked into the CUDA-graph step as a kernel scalar -> device array
  `3fe66bde` / `4dc40390`, 155 tests; attempt 6 PID 53824 igniting cleanly.
- 09-04 02:30 paper: screening v2 admitted `8e13364c`: GATE-WALL-LOSS-GEOMETRY-SCREENING-V2, 380 `Wlh` macros,
  CLM-077..085, 257 paper tests, 65-page PDF `ba7441c9`; check_paper 107 s warm / 350 s cold.
- 09-04 02:20 archive-first rollover of both memory files (this file and DEVLOG.md; untracked, no git write):
  byte-identical copies under `.cursor/memory/archive/`, live files rebuilt by a %TEMP% script from the archived
  snapshot and verified (166/166 hex tokens, 67/67 lessons, 115/115 follow-ups; UTF-8, LF, 0 CR). [tool] A
  token check `\b[0-9a-f]{7,}\b` treats a 64-hex hash as ONE token, so a source that quotes both `e7900c10...`
  and the full digest needs both forms re-emitted. [tool] `python-markdown` is not installed here;
  `pip install --target %TEMP%\mdcheck markdown` + `PYTHONPATH` gives a parser without touching the global env.
- 09-04 02:40 canvas at origin head `8e13364c` after two folds (02:20: plume attempts 3-6 + CUDA-graph fix;
  02:40: screening-v2 paper admission). Ladder recount: stage 7 = 16 (13 caveat), RUNNING 1 (pic-v14), chips
  `0/38 externally validated · 16 in the paper · 29/38 merged · 0 on main`. [self] Fold canvas milestones one
  commit at a time and recount `ladderRows` in Node each time; the chips are derived, so a stale row shows up
  as a chip mismatch. L1b/P2 material-aware confirmation of the 15 HEMP-like v3 designs launched (CPU only;
  GPU held by plume attempt 6 PID 53824 until ~05:30).
- 09-04 04:20 plume attempt 6 IGNITED (gates 0.75 us: S 1.42 / N_e 2.00; 1.5 us: 1.70 / 3.60; 24/24
  connected), 2.466 us / 1.644M steps / 82 frames / 8430 s, stopped by the plume-boundary gate 66 ns
  after arming on a single macro-ion at the axis corner node (numerical, see lesson). v2.0.1 gate
  sample-size floor `45edd30e`, 156/156 pic2d tests; record `9cf7ca39`; attempt 7 PID 52176 launched
  04:19 (verdicts ~05:03 / ~05:45, gate arms ~06:45, budget ~08:19) `895ea58d`; replays attempt 6
  bitwise. Physics at stop: I_d 6.3-6.9 mA still rising, n_g 2.77e19 at fixed point, energy residual
  -3.4%, far plane an ordinary Dirichlet sheath (phi 44 -> 0 V over ~3.5 lambda_D). Video renderer
  ionisation-panel fix + canvas fold launched.
- 09-04 05:05 PIC video renderer v0.2 `bbf74ea0` / `ab322245`: ionisation panel = causal 11-frame (330 ns) window
  + >= 20-event mask (grey = unresolved) + fixed 0.5-99.5 pct log scale (1.62e23-2.88e24 m^-3 s^-1); 13 renderer
  tests, pic2d 165, visualization 148 green. Attempt-6 re-render `*-v2.mp4` (4.48 -> 0.72 MB): ionisation confined
  to the bore, organised as three "flames" at the P2 cusp planes 6.03/12.0/17.97 mm from a near-axis band
  (r 0.3-1.2 mm), brightest 2-3e24 just downstream of each cusp; resolved nodes = 5.6% of plasma nodes carrying
  67-77% of S; plume unresolved at this cadence. No frames exist for the steady-state runs (recorder newer).
  Canvas fold of `ab322245` launched.
- 09-04 06:58 L1b/P2 HEMP confirmation v1.1 CONFIRMED `54cd3e82` (dashboard `560909f7`, ff `ab322245..560909f7`):
  15/15 cusp counts unchanged under linear iron (mu_r 4000 poles + yoke, recoil magnets), 37/37 cusps matched,
  max shift 0.362 mm = 0.80 tol, HEMP-like 14/15 (028 rho 1.515 -> 1.464), wall |B| at cusps 1.05-1.53x, axis
  nulls move <= 1.07 mm. v1 `2d8d6705` was a development_rejection (2/15 level-0 meshes < 10 deg angle gate,
  slivers at 028 exit taper / 048 injector-magnet gap) -> v1.1 `c8692ff2` re-prereg with disclosed 5 deg gate +
  whole-set mesh preflight. 56 min CPU, RSS 240 MB. Main checkout pulled ff to `560909f7`. Paper admission +
  canvas fold launched. Attempt 7 alive at 2.58 us (past attempt 6's 2.466 us stop; both ignition gates passed).
- 09-04 09:06 paper: L1b confirmation admitted `1a7eaea9` (ff `560909f7..1a7eaea9`): GATE-L1B-HEMP-CONFIRMATION-V1-1
  (numerical-screening, new 6th outcome `accepted-material-aware-confirmation`), manifest
  `paper-material-aware-confirmation-manifest` 138 sources + 109 lineage (whole v1 rejection bundle), CLM-086..093,
  CLM-028/075/076 amended, 320 `Hmc` macros, Section 16 pp 54-60, 285/285 paper tests, 73-page PDF `105b5225`.
  Corrections vs brief: tol range 0.451-0.523 mm; 048 min angle 5.6 deg. Main checkout ff to `1a7eaea9`.
  Attempt 7 PID gone at 09:06 with run_state finished=false at 3.78 us / 2.52M steps / 126 frames / 14443 s
  (= 4 h budget) -> finalization/diagnosis agent launched; 5.3 GB GPU memory held by an unidentified process.
- 09-04 09:56 attempt 7 = clean budget stop (wall_clock_budget_reached at 3.78 us / 1.22 transits) + finalizer NaN
  crash (nvidia-smi timeout -> NaN -> canonical JSON refused). Fix + fail-closed `finalize --recover-runner-stop`
  `3b8b577a` (169 pic2d tests); record `24ea2f65`; sessions record wall budget `e8b3fb7b`; attempt-8 log `8556a401`.
  NO plateau (I_d -13%, N_e +22% trailing drifts). First plume/thrust DEVELOPMENT numbers: I_d 5.99 +- 0.06 mA,
  S 8.57e16/s, I_beam 0.96 mA, T_total 20.9 +- 0.4 uN (drifting +20%/window, not quotable), IEDF mean 184 eV,
  half-angles 8/29/60 deg (50/90/95%), Isp 112 s, eta_a 0.6%; plume 10%/1% contours do not fit the 12 mm box.
  Gate v2.0.1 INERT (0 resolved nodes ever) -> v2.0.2 interval-averaged gate + nvidia-smi cadence launched.
  Attempt 8 = RESUME from 3.78 us, PID 51256, 09:51, budget 50,400 s, plateau verdict >= ~17:30, end ~20:00;
  two scratch resumes bitwise identical. Main checkout ff to `8556a401`. Canvas fold launched.
- 09-04 10:41 PIC v2.0.2 `0251ff10` (ff `8556a401..0251ff10`): plume-boundary gate on a trailing 400k-step window
  of the far-field accumulators (`far_field_window_sums()`, `FarFieldChargeWindow`), floor 64,000 particle-steps
  per node, live at 0.025/0.034 on attempt-6/7 maps; `GpuUtilisationSampler` daemon (300 s, NaN-safe);
  `step_graph` "lazy"; 175 pic2d tests. Attempt 8 keeps v2.0.1 by config identity. Main checkout ff to `0251ff10`.
  Launched: v2.1 prep (axial plume-box extension with cost table + resume state hygiene; no launch) and canvas fold.
  Watcher on attempt 8 now keys on PID 51256 (run_state `finished` is stale during a resume).
- 09-04 11:24 PIC v2.1 PREPARED (not launched) `ce8628f4` resume hygiene / `1043f71d` domain extension as config +
  exit-plane index fix / `ba57537f` spec v2.1 (ff `0251ff10..ba57537f`; 193 pic2d tests). Attempt-7 decay fit: axis
  n_i peak 2.57e18 at z 27.45 mm, L_e 2.6-3.1 mm, z10 37-39 mm, z1 43-58 mm (lower bounds). Proposal (z_far, r_far)
  = (48, 12) mm, uniform 50 um, 240 x 960, 8.2 ms/step (+16%), ~17-20 h to 3 transits, 8.8 GB GPU, far field from the
  `domain-padding-1.5` P2 checkpoint (channel agreement 0.74 mT). Non-uniform spacing unsupported by kernels.
  `experiments/pic2d_cft_plume_v2_1/` ready. Launched: PIC design mini-sweep PREP (4 designs across rho, fields,
  closure targets -> plasma-network v2, cost; draft dir with whole-set preflight; no prereg) + canvas fold.
- 09-04 11:38 attempt 8 ENDED cleanly (finalizer OK): `grid_heating_triad_gate_stopped_run` at 3.32M steps /
  4.98 us / 1.6 transits / 166 frames / 20,610 s. Discharge kept densifying past the 50 um grid's Debye floor
  (attempt-7 end: 3.64 cells/lambda_D vs 4.5) before any plateau. Diagnosis (physical vs grid heating) + record +
  video + resolution decision (Delta / Delta t / operating point) + v2.1 launch-or-report agent running; canvas
  status update running. GPU free.
- 09-04 12:00 attempt 8 diagnosed `ac248e05` (record + README; video rendered): S-drift triad member tripped
  (+0.253); VERDICT numerical finite-grid heating (residual power 47% of discharge power in the last 0.4 us; onset
  2.0-2.4 us at Delta/lambda_D ~ 3.2). Debye gate 4.5 not protective; base plateau at 3.17 / +0.4% is on the
  threshold. ~6 mA vs 3.44 mA is physical (uncapped flux-tube cathode). Nothing after ~3.2 us usable for thrust;
  attempt-7 dev thrust stays non-quotable. v2.1 launch NOT taken (33 um = 47.5 h). Adopted: v2.0.3 gates (hard
  pi on averaged peak + windowed residual-power gate) + preregistered channel-only 33 um / 1.4 ps refinement of
  the accepted plateau (13-14 h) -> agent running (also paper PIC-claim impact check). Canvas fold running.
  Main checkout at `ac248e05`.
- 09-04 12:46 PIC design mini-sweep v1 DRAFT on origin `8704bf7c`/`805cd09e`/`b4ddefff`/`6440518d` (206 pic2d
  tests): designs divergent-exit-stack (ref, rho 0.60), l1a-gs-v2-047 (0.38 under iron; anode-edge boundary cusp
  disclosed, substitute 061), l1a-gs-v3-009 (0.92), l1a-gs-v3-056 (2.36), optional 106 (2.93); four new padded
  level-0 material-aware P2 fields (LFS, gates passed, |dB| <= 1.5 mT vs L1b); closure targets mapped to
  plasma-network v2; whole-set preflight 5/5 green (056 needs dt 1.3 ps in the 24 mm box; 047 cathode r_outer
  0.9x). Cost channel-only 4 designs: 12.7 h @50 um, 16.9 h @33 um/1.4 ps. Prereg blocked on the grid decision.
- 09-04 12:48 Cloud: user provisioning Lambda 8x H100 SXM (us-west-3, fs `h100-files`); my key `~/.ssh/lambda_h100`
  registered as `cft-key` (instance must be relaunched with it; `grabby` private key is not on this machine).
  Bootstrap/benchmark/scheduler kit agent running (`uni-project-cloud`). Housekeeping: 41 stale worktrees + 27
  merged branches removed; `.worktrees/` gitignored `25e86dca`; live worktrees `-cloud`, `-pic2d`, `-ss3-dev` to be
  moved under `.worktrees/` when their agents finish; `uni-project-vizfix` empty dir held by an idle shell.
  Lag on the PC is from OTHER projects in WSL (gtoc12 cluster-fleet 4x99% CPU until ~14:10, 2x weldsim pytest,
  a spacepdhcg CUDA test) — not ours; user to decide on killing them.
- 09-04 13:19 PIC v2.0.3 gates `ceb9b172` (window-mode peak-Debye hard pi / soft 2.5 on the interval-averaged
  peak; one-sided windowed residual-power gate >= 5%; 220 pic2d tests) + `pic2d_cft_steady_state_v4` prereg
  `392129e5` (90 x 720, 33.33 um, dt 1.4 ps, W 26,666.7, budget 24 h, acceptance a-d, verdicts converged /
  resolution_limited / refinement_heating / no_plateau) + launch log `a366e556`. LAUNCH 1 PID 18068 13:11:55 in
  worktree `uni-project-pic2d-ss3` (beside the repo — move under .worktrees/ after the run), 2.50 ms/step, GPU 99%,
  verdict ~18:45-19:30, budget end 13:15 5 Sep. Paper: no PIC claim admitted -> nothing to retract;
  `\CtvPTwoPicPlanesMm` is a field-map descriptor (unaffected). Main checkout ff to `a366e556`; idle `pic2d`
  worktree moved to `.worktrees/pic2d`. Single H100 (us-southeast-1, cft-key) booting; IP pending.
- 09-04 13:23 Cloud kit `8fe5d00c` in `modern/tools/cloud/`: `bootstrap_lambda.sh` (EXPECTED_GPUS env, default 8 ->
  set 1 for the single box; fails closed on driver < 525 / LFS pointers; uv Python 3.12 venv; Warp smoke; pytest
  pic2d on GPU 0), `requirements-pic.txt` (repo has no lock -> pins mirror the local anchor env: warp-lang 1.14.0
  PyPI CUDA-12.9 wheel, driver >= 525, no toolkit; numpy 2.5.2), `bench_gpu_concurrency.py`/`bench.sh` (N = 1/2/4
  per GPU via the v4 preflight `_time_steps`), `schedule.py`/`jobs.yaml` (tmux wrapper, Warp UUID vs nvidia-smi
  cross-check, prereg ancestor + protocol byte checks, per-job detached worktrees), `PLAN.md` (queue ~110-130
  GPU-h; makespan set by the 33 um plume run 51-52 h; without it 20-37 h). 28 tests. [tool] PowerShell drops an
  empty-string env var, so `CUDA_VISIBLE_DEVICES=""` does NOT hide the GPU — use `-1`. Main checkout ff to `8fe5d00c`.
- 09-04 14:35 H100 box live: ubuntu@68.209.75.2 (us-southeast-1, key ~/.ssh/lambda_h100, fs /lambda/nfs/h100-files,
  WORK=/lambda/nfs/h100-files/cft, repo clone via deploy key ~/.ssh/open_cft_deploy = GitHub deploy key
  "lambda-deploy" with write access, venv .venv-pic). Bootstrap OK (driver 580.105, CUDA 13.0, Warp 1.14.0 cu12.9,
  Python 3.12.14). pic2d suite on H100: 216 pass / 2 cross-platform pins fail (dashboard byte pin, v1 field-map
  hash) -> fix + binding audit agent running; Gauss-law scale-aware bound fixed `af9e79d1`. Benchmarks in
  /lambda/nfs/h100-files/cft/bench and bench-mps (see lesson: ~1x 5090 per process; MPS 4 slots 1.54x aggregate).
  Local ss-v4 refinement at 2.47 us / 1.41 h, 3.3 ms/step, I_d ~3 mA, peak 9e17, 1.7 cells/lD, res_w -10.6%.
- 09-04 14:52 cross-platform fix `0ac8d9b8`/`79c2a3f8`: `MagneticFieldMap.source_sha256` (grid + declared provenance
  minus CPU-derived blocks) binds checkpoints; anchor `<name>.field.npz` + allclose 1e-12 replay gate
  (`bitwise`/`numerical` mode recorded per session); `platform_fingerprint()` (OS, libc, numpy, compiler, SIMD,
  BLAS core) + `gpu_identity()` in every record; dashboard byte pin only on the anchor fingerprint else structural
  1-ulp-of-recorded-digit compare; coarse-field pin via source hash + anchor npz. Audit: no launcher verifies a
  derived hash -> nothing refuses on Linux; only cross-box RESUME was blocked. 228 pic2d tests.
- 09-05 00:22 sweep: 056 launch-1 stop CONFIRMED shot-noise artefact (raw omega_pe dt drift +0.283 vs resolved +0.0165;
  argmax on <= 4-macro-electron axis nodes 96% of records; cooling, no heating) -> records `b424ea37` (047 plateau
  I_d 1.925 mA) / `ccee5c60` (056 L1 + diagnosis); amendment 1 `ee35bc84`; 056 launch 2 PID 38282 00:00 AEST, ETA
  ~06:15. 009 plateau 3.02 transits (record pending); reference past 3 transits, no plateau yet. Ext-val v0 launch 1
  stopped 23:56 on the windowed residual-POWER gate +7.4% (heating signature, diagnosis running). GMG `poisson_gmg_v1`
  merged `9c2e4222`/`7cd03b65`/`e1a24aec`; v4_fast qualification campaign agent running; main at `e1a24aec`.
- 09-04 16:03 PIC design mini-sweep v1 PREREGISTERED `291a9227` + LAUNCHED on the H100 under MPS-4 (jobs.yaml
  `9c426f90`, launch log `a20ec2fa`, head `1506f219`): 4 designs at dr = dz = 33.3 um / 1.4 ps, ss-v4 template
  (v1.3 closure + v2.0.3 gates), W parity, seed 20260903, frames ON. PIDs 19764 ref (90x720, 18.8 h budget), 19913
  056 (115x512, 25.3 h), 20079 047 (66x778, 13.3 h), 20189 009 (80x684, 23.0 h). 3-transit ETAs ~01:00 / 04:30 /
  07:00 / 08:40 AEST 5 Sep. Preflight 5/5, shakedown (056, 100k steps) incl. finalize+assess, MPS replay physics
  bitwise. Draft changes: v4 template not v1.4, 24 mm/720 grid target, W parity, 047 kept (anode-edge disclosed),
  ss-v4 verdict = predeclared caveat. Interim visualisation agent launched (comparison panel + videos).
- 09-04 16:10 canvas `open-cft-roadmap-status.canvas.tsx` folded a366e556..1506f219 (13 ff commits; git read-only,
  fetch 16:07 = brief's head = local HEAD; main 207 behind). pic-design-mini-sweep 3 -> 5 (caveat at 5, RUNNING);
  counts 7:17(14) 6:0 5:8(6) 4:9(3) 3:1 2:0 1:4(1), RUNNING 3, merged 32/2/1/4, chips 0/39 · 17 · 32/39 · 0 -
  recounted in Node with the canvas's own `cells.slice(0, stage)` caveat rule (a whole-row `some(caveat)` over-counts
  stage 4 as (4)). [lesson] RECENT_MARKERS collide silently: bare `16:00` matched the 09-03 seed-b "ETA ~16:00" and
  `20260903` matched manifest ids - use `16:00:44` / `seed 20260903`; check every new marker against lines that
  carry no fresh SHA before saving. TS check clean on every edit.
- 09-04 16:40 interim viz of the 4 H100 runs `5da74ee6` (`interim_sweep_panel.py` + 6 tests; media in
  `.worktrees/interim-sweep-media/`, box `$WORK/interim/`, `rerender.sh`). At 0.29-0.34 transits all healthy
  (peak Delta/lambda_D 1.0-1.3 vs pi; residual_w -8..-16% cooling; T_e 7-9 eV). I_d 1.28/1.33/2.39/2.90 mA and
  S 1.06/1.85/2.92/3.64e16/s for rho 0.38/0.60/0.92/2.36. Ionisation: cusp-anchored near-axis flames at low rho
  -> exit-cell-dominated body at high rho; n_i broad arches (047) -> sharp separatrix sheets (009, 056). ETAs
  drifting later (ms/step rising with N_e: 056 5.8 -> 6.3).
- 09-04 17:44 external validation v0 DRAFT `645c7de4`/`e7c5f017`/`4ddfc319`/`7fa9e6c6` (ff `5da74ee6..7fa9e6c6`):
  reference Brandt et al. 2016 doi:10.2322/tastj.14.Pb_235 (+ Kiel 2017 thesis for the magnet stack; paper-vs-thesis
  spread 4.3 vs 4.7 mA used in u_D). Geometry mapped (A1-A9 approximations), material-aware P2 field passes anchor
  gates after a one-parameter remanence scale 1.081 (nulls 2.63/8.37 mm; exit null 15.85; 0.053 T at 17 mm; 0.70 T
  axis max); G5/G6 misread -> revised with genealogy. Protocol channel-20um on the ss-v4 template (75x700, dt 0.7 ps,
  static Xe 2e20, 400 V, 1.8 mA source, W 82,467 at the 12 M cap; 18.3 ms/step at MPS-4 -> 30.6 h, 11.9 h solo,
  budget 46 h, 17.4 GB); 33 um inadmissible a priori (4.5 cells/lambda_D at 1e19); 15um + bohm-0.4 variants sealed.
  V&V20 spec 12 rows (10 comparable), tolerances 20% / +-5 V / 0.3 dex; potential-step rows non-discriminating.
  Preflight 3/3. NOT preregistered / launched: plan = launch on the H100 when 047's slot frees (~01:00) or solo after
  the sweep. Hybrid L2 v2 agent running.
- 09-04 19:44 ss-v4 33 um refinement FINISHED 18:17 (plateau at 5.2M steps / 7.28 us / 3.03 transits of the 2.4 us
  v2 ion transit; 5.0 h). Verdict `resolution_limited`: I_d 3.80 mA (+10.35% vs 3.44), I_beam 2.46 (+7.3%), S 3.60e16
  (-8.5%), util 0.42 (-8.5%), n_g 3.19e19 (+7.3%), peak n_e 1.29e18 (-21.4%), T_e,peak 5.58 eV (-24.5%); residual_w
  -7.7%; Delta/lambda_D window 2.15 (soft ok). Record `0d228ad2`, dashboard `pic2d-cft-steady-state-v4.html`
  `abac6d9e`, renderer v0.2.1 `0e09e749`; v5 25 um / 1.0 ps ladder point PREREG `69ff435d` (120x960, W 15,000,
  expected Delta/lambda_D 1.62, budget 48 h) LAUNCHED PID 43572 19:29 in `.worktrees/pic2d-ss5` (`427c7918`);
  verdict ~10:15-11:15 5 Sep solo. Preflight was contended by the hybrid agent's 11 CUDA processes -> interrupted
  it (CPU only). ss3 worktree moved under `.worktrees/`. Paper: first PIC admission would be a numerical-campaign
  gate (v4 plateau values + bands, resolution-limited statement, gate history, heating diagnosis); no thrust/plume.
- 09-04 19:50 canvas `open-cft-roadmap-status.canvas.tsx` folded 1506f219..427c7918 (10 ff commits; git read-only,
  fetch 19:48 = brief's head = local HEAD; main 217 behind; nothing unfolded). NEW ROW `ss-v5-ladder` (N 40; rungs
  1-4 ok, caveat at 5, RUNNING). Rung moves: ss-v4-refinement 5 -> 6 (rung 5 ok on `0d228ad2`, rung 6 CAVEAT
  "accepted plateau; resolution_limited for the 50 um base; 33 um convergence untested until v5", stop NEEDS DATA);
  validation-v0-v2 1 -> 3 (draft, merged none -> feature, NOT STARTED). Text-only: ss-v2 rung 6 / stop lead
  RESOLUTION-LIMITED with state kept `no` (a coarse plateau that failed its refinement pair is not "accepted" - same
  reasoning as the 12:05 rung-4 caveat placement); hybrid BLOCKED BY PHYSICS -> RUNNING (v2 revival is a LOCAL branch
  `feat/hybrid-l2-v2` at 386c9070, on no remote -> no rung; noted that a prereg not on origin is not a prereg and the
  closures inherit the 50 um resolution-limited caveat); pic-v14 RUNNING re-pointed at v5. Counts 7:17(14) 6:1(1)
  5:8(6) 4:9(3) 3:2 2:0 1:3(1), RUNNING 4 (pic-v14, mini-sweep, ss-v5, hybrid), merged 34/2/1/3, chips 0/40 · 17 ·
  34/40 · 0 - recounted in Node (CRLF file: normalise `\r\n` before regex-splitting rows, else 0 rows parse; two rows
  - n-cell, paper - format `stop:` multi-line and need a separate check). Commit-body cross-check: `0d228ad2` says
  n_g +7.2 % where the brief says +7.3 % -> canvas quotes the commit and notes the brief. RIGHT_NOW at the 8-row cap:
  Plasma network v2 retired to make room for the Validation v0 row. TS check clean on every edit.
- 09-05 03:00 canvas `open-cft-roadmap-status.canvas.tsx` folded ce1d96cb..219f7ff3 (34 ff commits; git read-only,
  fetch 02:47: origin at c1508c06 = THREE commits later than the brief -> f1255832 / 2dcaebbc / c1508c06 noted as
  unfolded on HEAD_NOW / mergeTruth / pic-physics rung 2 / validation row; local HEAD 219f7ff3 = the fold; main 267
  behind at the fold, 270 at c1508c06). User direction 01:12 / 01:49 (accuracy first: 2-D physics -> 3-D PIC -> AI)
  -> THREE NEW ROWS (N 44): `pic-physics` (rung 1 = audit 0901138a, rung 2 caveat R1/R2/R3, RUNNING), `pic-3d`
  (rung 1 caveat = plan-level design 4e2f7467, NOT STARTED, merged none), `ai-surrogate` (rung 1 = f92a7237 ->
  4e2f7467, NOT STARTED); roadmap steps 9/10/11 added and the array re-sequenced 1,2,4b,6,9,7,10,11,8,3,4,5,G,G2,L
  with a Node script (items regex-split on the file's CRLF eol, content-equality check before writing). Rung moves:
  pic-performance 2 -> 4 (rungs 2/3 ok = v2.0.5 f80c6441 + GMG 9c2e4222; rung 4 caveat = v4_fast RUNNING PID 44430,
  prereg b09f2b71); validation-v0-v2 stays 5 but rung 4 -> caveat (gate stop 036bd679) and rung 5 -> ok (recorded),
  stop RUNNING -> FAILED GATES "inconclusive (heating at the published grid) · bohm-0.4 launch 2 pending";
  ss-v4-refinement rung 6 caveat rewritten "(b) FAILS on the corrected ledger (+2.46 %); heating; NOT a clean
  reference" (state kept cav - the record's own gates passed; a post-hoc re-read lives in the caveat, never moves a
  rung); ss-v2 name/rung 4/rung 6 text HEATING +7-13 %. Model v2.0.6 = the ledger W bug: every recorded residual
  7-14 % too low; RIGHT_NOW rewritten to 8 short rows (splice via Node between the JSDoc marker and `/** <= 9
  findings`); key findings #10 (ledger) / #11 (physics audit + order); 3 interpretation rules; 35 commit rows; GPU /
  Cloud rows (H100: 32709 / 38282 / 44430; local none). Counts 7:17(14) 6:1(1) 5:9(7) 4:11(5) 3:0 2:1(1) 1:5(2),
  RUNNING 5 (pic-v14, mini-sweep, ss-v5, pic-performance, pic-physics), FAILED GATES 8, merged 36/2/2/4, chips 0/44 ·
  17 · 36/44 · 0 - recounted in Node. Lesson: the canvas's `stageCounts` counts caveats only BELOW the highest rung
  (`cells.slice(0, stageIndex)`) - a recount that counts any caveat cell over-reports "(n)" (viz has a caveat at 7
  after a gap at 5); mirror the canvas rule. RECENT_MARKERS: bare "energy ledger" / "Phase 1" collide with archived
  phase / experiment text - use hyphenated or longer tokens; bare "02:12" / "02:33" / "02:40" collide -> use seconds.
  TS check clean on every edit.
- 09-05 06:40 canvas `open-cft-roadmap-status.canvas.tsx` folded 219f7ff3..55092f4c (27 ff commits; git read-only,
  fetch 06:27: origin at 785a1594 = SIX commits later than the brief -> e47ae78a / 0916a4f8 / 33be2a89 / 73d495c8 /
  9daa1643 / 785a1594 noted as unfolded on HEAD_NOW / mergeTruth / pic-physics rung 4 / validation row / Right-now;
  local HEAD 55092f4c = the fold; main 294 behind at the fold, 300 at 785a1594). Scheduler read READ-ONLY over ssh
  (`ssh -i ~/.ssh/lambda_h100 ubuntu@68.209.75.2 'cd /lambda/nfs/h100-files/cft/uni-project && python3
  modern/tools/cloud/schedule.py status'` + `tail $WORK/r1/queue.log`, `$WORK/pe-queue/queue.log`, `tmux ls`,
  nvidia-smi): 4 clients (32709 v5 1.65/3 ETA 7.1 h; 38282 056 L2 2.85/3 ETA 0.4 h; 44430 v4_fast 0.61/3 at
  14.0 ms/step ETA 16.0 h; 49403 bohm-0.4 0.81/3 since 05:07:11), at-alpha-1over16 finished at 1.00 transit,
  r1-queue PAUSED 05:35 -> RESTARTED 06:26 under the amendment (waiting for a slot for 1/64), pe-queue chained.
  Row moves: pic-physics 2 -> 4 (rungs 2/3 ok = R1-R5 code + 435 tests; rung 4 caveat = campaigns launching +
  four NON-EVIDENTIARY shakedowns + the 1/16 arming stop; rung 5 `no` = records pending) - the user's brief placed
  campaigns at rung 4, so the mini-sweep convention (prereg = rung-5 caveat) was NOT used here; validation-v0-v2
  stop FAILED GATES -> RUNNING (rung 4 caveat text = launch 2 running, rung 5 stays ok on launch 1's record with
  the amendment named); ETAs on mini-sweep / v5 from the scheduler; pic-performance left as instructed (the live
  0.61/3 / 14.0 ms/step / ETA 22:30 reading put in the GPU row + Right-now, with the row's ~20:00 window named as
  the brief's). Counts 7:17(14) 6:1(1) 5:9(7) 4:12(6) 3:0 2:0 1:5(2), RUNNING 6, FAILED GATES 7, merged 36/2/2/4,
  chips 0/44 · 17 · 36/44 · 0 - recounted in Node (temp script under %TEMP%, deleted). Honesty call: the brief read
  the 1/16 stop as "re-equilibration"; the UNFOLDED record 0916a4f8 reads an EXTINCTION (N_e e-fold 0.88 us,
  hypothesis contradicted, not relaunched) -> the canvas carries the brief's reading AS THE FOLD STATE and quotes
  the unfolded reading beside it (rung-4 cell, Right-now, key finding #13, the arming interpretation rule) rather
  than choosing one. TS check clean on every edit; file stays CRLF-only (6666 lines).

## Archive

- Full session entries 2026-09-02 05:00 -> 2026-09-04 02:30 (task summaries, mistakes and fixes, what worked,
  guardrails, loose lessons appended between entries): `.cursor/memory/archive/AGENT_SCRATCHPAD-2026-09-04-0220.md`
  (128,154 bytes, 2141 lines, SHA-256 `2df5d7fef70d7ad12547fca91c8d4ebf4f844f25400fbc9dd01912c2c7cc3fa3`), byte-identical copy of the live file at
  rollover 2026-09-04 02:20 AEST.
- Engineering timeline for the same window: `.cursor/memory/archive/DEVLOG-2026-09-04-0220.md`.
