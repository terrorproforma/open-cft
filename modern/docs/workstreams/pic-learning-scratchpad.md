# PIC Workstream Learning Scratchpad

File policy: `COMMITTED` workstream evidence. This location is used because the
task permits only new `modern/docs/workstreams/pic-*` documentation paths.

## Preflight guardrails

- [user] Own only new PIC paths; never edit shared, field, hybrid, FYP, or Git
  state and never install, commit, push, or run throughput benchmarks.
- [user] Prefer individually verified kernels and an explicit integration gate
  over fabricated CFT outputs.
- [self] Treat optional Warp execution as separate from WarpX/PICMI
  availability; one does not imply the other.
- [self] Require a recomputed true residual before publishing a Poisson solve.
- [self] Synthetic cross sections verify numerics only and must be labeled so
  they cannot be mistaken for xenon evidence.

## 2026-09-01 session

### Task summary

- Built a dependency-free periodic 1D3V electrostatic CPU reference and
  optional genuine Warp CIC/gather/push kernels.
- Added deterministic elastic MCC, diagnostics, stability gates, checkpoint
  hashes, provenance, and future WarpX/LXCat adapter contracts.

### Learnings

- [tool] Warp 1.14.0 was installed with CPU and RTX 5090 `cuda:0`; `pywarpx`,
  `amrex`, and PICMI were absent. No installation was needed or attempted.
- [self] A normalized test choice `q=sqrt(epsilon_0)`, `m=1`, and physical
  density `n=1` gives `omega_p=1 rad/s`, making the plasma quarter-period gate
  transparent and dimensionally traceable.
- [self] Comparing energy-envelope width over equal physical duration is a
  more robust numerical-heating trend test than comparing endpoint energy,
  whose phase can reverse the apparent ordering.
- [self] The identical CIC shape gives both a charge-conservation test and a
  strong deposition/gather adjoint test.
- [self] Periodic Poisson must record removed mean charge: neutralizing the
  nullspace silently would hide an important physical boundary assumption.
- [self] Warp atomic deposition order can differ for larger populations, so
  parity should use scaled tolerances and charge invariants rather than demand
  universal bitwise identity.

### What worked

- Small manufactured and statistical tests exercised scientific invariants
  without running a throughput workload.
- Compiling the Warp kernels on both available devices before final validation
  exposed backend syntax risk early.

### Guardrails for next session

- Do not label an axisymmetric interface as axisymmetric implementation.
- Do not ingest LXCat values unless exact source bytes, source identity, units,
  process, and SHA-256 are retained.
- Add wall/sheath or ionization physics only with independent conservation and
  limiting-case tests.
- Keep CPU integrated verification distinct from per-kernel Warp parity until
  the Warp Poisson and whole-step ordering are independently tested.

### Open risks

- Periodic 1D geometry cannot represent a CFT channel, plume, material wall,
  axis, cathode, or open boundary.
- Infinite-mass isotropic elastic scattering is a verification operator, not a
  complete electron-xenon collision model.
- No external physical dataset or WarpX result has been validated.

## 2026-09-01 audit-correction session

### Corrective learnings

- [user] A recomputed residual is insufficient if its norm and tolerance can
  both overflow. Convergence publication must require independently finite,
  overflow-safe norms and a finite tolerance.
- [self] Normalizing the Poisson right-hand side before CG prevents avoidable
  inner-product overflow; physical potential and true residual must still be
  reconstructed and independently range-checked.
- [user] A centred nodal gradient erases the discrete Nyquist field. Store
  operator-consistent `E` on faces and use those faces for field energy.
- [self] Symmetric face-to-node reconstruction followed by nodal CIC preserves
  the tested periodic one-particle zero-self-force property. This smoothing
  intentionally does not transfer Nyquist force to particles.
- [user] Transactionality includes RNG and counters, not only particle arrays.
  MCC must clone RNG state and propose every event before any commit.
- [self] Stepper transactionality needs two field solves: the second field
  supports a time-centred `K_half + mean(UE_n, UE_n+1)` energy diagnostic and
  is validated before publishing particles.
- [user] Hash validity is not state validity. Rehashed checkpoints require a
  closed schema, typed reconstruction, identity checks, and stagger/runtime
  consistency.
- [self] A named transverse area makes charge-volume and joule dimensions
  explicit; a silent unit-area convention invites mixed J/J-per-m2 results.
- [tool] Default and importlib full suites collected the renamed PIC tests
  without basename collisions. Concurrent non-PIC work remained red in
  coupling plus magnetics/plasma and axisymmetric visualization.

### Retained guardrails

- Reject nonrepresentable source, iterate, diagnostic, or derived state with a
  typed PIC error; never use infinities as convergence sentinels.
- Validate mutable state at every stage boundary and publish only after all
  post-stage checks pass.
- Keep claims at reduced-kernel correctness and small integration smoke scope.

## 2026-09-02 final acceptance session

- [user] A zero-density shortcut is not permission to skip mutable-state,
  dimension, species-weight, or metric validation.
- [self] Finite input factors do not imply a representable quotient. Extreme
  transverse area can silently round nonzero volumetric charge density to zero.
- [self] Mantissa/exponent product-ratio evaluation permits one final binary64
  rounding and cleanly separates accepted subnormal densities from typed
  underflow rejection.
- [self] CPU and Warp publication must independently integrate deposited
  density back through explicit `area*dx` and compare it to robust represented
  charge; host preflight alone does not prove device preservation.
- [user] Keep measured fixture behavior separate from the acceptance contract:
  the current energy-envelope ratio is about `1.93`, while the test guarantees
  only `>1.5`; neither establishes a general convergence order.
