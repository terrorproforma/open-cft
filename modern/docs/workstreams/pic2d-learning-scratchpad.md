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
