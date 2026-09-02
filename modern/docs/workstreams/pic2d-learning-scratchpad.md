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
