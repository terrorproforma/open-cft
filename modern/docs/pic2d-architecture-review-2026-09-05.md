# PIC2D architecture and accuracy-preserving performance review

Review baseline: `07b8b72667e00e2dba2db35a144c8db2ba40dfa9`, shared by `main` and
`feat/sota-foundation` when inspected on 2026-09-05. This review and its ordering
probe do not change the production solver, protocols, or recorded results.

## Decision

Keep the Python orchestration / Warp numerical architecture. The timestep is
already device-resident, structure-of-arrays, graph-captured, and substantially
fused. Rewriting it wholesale in C++ would not remove its main costs: dependent
Poisson launches, particle passes, atomics, and collision ordering.

Optimise the current full-physics configuration, separately on channel and plume
grids. First establish uncontended baselines; then remove unnecessary transfers,
replace quadratic Coulomb ordering, and reduce the Poisson critical path.
Treat faster numerical methods as new configurations requiring physical
observable convergence. There is no evidence here for a guaranteed further
2x, 5x, or 10x end-to-end speedup at unchanged accuracy.

The governing metric should be **wall time to an accepted physical result at a
specified uncertainty**, accompanied by cost, memory, and physical time reached.
Steps per second alone rewards under-resolved or prematurely stopped runs.

## What is actually present

The root README and general ARCHITECTURE document lag the production PIC2D
implementation. The current execution path is:

1. A protocol builds `PIC2DConfig`, immutable field/cross-section inputs and a
   `Simulation`. The shared experiment runner controls recording and acceptance.
2. `WarpBackend` retains electron/ion arrays, mesh fields, ledgers, RNG seeds,
   live counts and optional neutral state on the device. Particle components are
   separate float64 arrays. Charge and selected wall/neutral tallies use integer
   accumulation with prescribed quantisation.
3. Each step deposits charge, solves the masked cylindrical finite-volume
   Poisson problem, computes E and pushes particles in prescribed B. Ion motion
   and charge redeposition are subcycled.
4. Optional SEE, anomalous transport, Coulomb collisions, ion-neutral collisions,
   electron MCC and births execute at their defined positions in the step.
   Injection and neutral substeps update device controls and arrays.
5. Diagnostic moments fork onto a side stream and join before the particle push
   can overwrite their inputs. Sync intervals flush ledgers, compact particles,
   update controls and evaluate diagnostics. Checkpoints bind configuration and
   input identities.

| Component | Current implementation | Assessment |
|---|---|---|
| Particle layout | float64 structure-of-arrays | Good foundation; already implemented |
| Whole-step dispatch | Cached CUDA graphs with device counts/seeds/controls | Keep; Python is not issuing every individual GPU kernel in production |
| Deposition | Fixed-point CIC; cached ion charge plus birth increments | Preserve integer quantisation and birth/subcycle bookkeeping |
| Particle motion | Relativistic Boris; prescribed magnetic map | Keep as numerical reference |
| Direct Poisson | Precomputed dense inverse blocks; serial forward/backward radial sweeps | Effective on channel grids; large memory and launch chains on plume grids |
| Multigrid | Operator-dependent interpolation, Galerkin coarse operators, warm-started fixed 14 V(2,2) cycles | Already built; not a universal speed win |
| Birth diagnostics | Ledger reductions folded into MCC; birth deposits fused into spawn | Already fixed; do not count these savings again |
| Electron moments | Side stream, configurable sample interval K; default 1, newer protocols use 5 | Already implemented; sampled gate statistics require qualification |
| Coulomb | Deterministic cell permutation; cell moments/shuffle; binary collisions | Quadratic within-cell ranking is a new scaling target |
| Spatial gas | Device test-particle neutrals, fast CEX neutrals, metastables, exact atom accounting | A separate slow physical timescale; changes what “steady” means |

Primary code: [warp_backend.py](../src/cft_revival/pic2d/warp_backend.py),
[warp_poisson_mg.py](../src/cft_revival/pic2d/warp_poisson_mg.py),
[warp_coulomb.py](../src/cft_revival/pic2d/warp_coulomb.py),
[warp_neutrals.py](../src/cft_revival/pic2d/warp_neutrals.py),
[simulation.py](../src/cft_revival/pic2d/simulation.py).

## Measurements: completed savings versus remaining work

These are repository-recorded H100 measurements, not new GPU measurements from
this review. The historical audit contains both measurements and fitted solo
estimates; they must not be mixed.

| Recorded comparison | Result | Interpretation |
|---|---|---|
| v2.0.5 changes, channel-33, same contended A/B | 13.48 → 11.12 ms/step | 1.21x measured; already implemented |
| v2.0.5 changes, plume-v2.0-50, same contended A/B | 26.00 → 23.82 ms/step with K=5 | 1.09x measured; already implemented |
| Channel-33 fast preflight, MG/K=5 versus BT/K=1, synthetic plateau load, third MPS client | 13.23 versus 4.56 ms/step | MG configuration was 2.90x slower under this load |
| Plume-v2.1-33, earlier contended production-step probe | BT 40.7 versus MG 37.8 ms/step | 1.08x measured, with variable contention; not a clean solo forecast |
| Plume-v2.1-33 solver storage | BT inverse blocks about 6.0 GB; MG device arrays about 149 MB | Strong memory benefit independent of the timing uncertainty |

Sources: [performance audit sections 11–13](pic2d-performance-audit.md) and
[fast replay preflight/launch record](../experiments/pic2d_cft_steady_state_v4_fast/README.md).
The latter documents a launch, but no final fast-replay assessment is tracked at
the reviewed main commit. A branch listing also found no separate ss33-fast
result branch. This is not a claim about the live GPU job's current state.

The older audit's 3.31 ms channel and 7.20 ms plume solo anchors precede later
physics additions. Its projected 1.7–1.8 ms channel figure is a model, not a
measurement of the current full-physics code. Current full-physics preflights
exist, but a new component profile is needed: SEE, Coulomb, spatial neutrals and
the higher MCC ceiling alter both workload and particle distribution.

## Prioritised implementation programme

### 1. Establish a reproducible current baseline

Extend the existing benchmark rather than create a separate simulation path.
Retain protocol/commit/input identities. Benchmark:

- channel-33 and channel-25; plume-v2.0-50 and plume-v2.1-33;
- original closure and the enabled full-physics configuration;
- seed, representative dense state and an actual checkpoint state where available;
- dedicated GPU first; then explicit 2/4-client throughput experiments.

Report medians and p95s over repeated warmed windows, live/capacity counts,
cell-occupancy quantiles/max, solver iterations/launches, memory, transfer bytes,
and numerical acceptance. Sample the full ion/Coulomb/neutral/diagnostic cadence,
including synchronisation and capacity-growth events in a separate long window.

There are two useful timings. GPU-event timing identifies compute/launch cost;
production wall timing includes synchronisation, recording, export and I/O.
`_time_steps()` currently times `Simulation.run()`, which returns a full
`export_state()`. This is a legitimate end-to-end number but is not a pure
steady-step number. Keep it and add a separately labelled no-export measurement.
Never quietly reinterpret old anchors after changing the benchmark.

Do not launch benchmark clients beside a run and call the result a solo speedup.
Do not terminate or change an existing campaign to create a benchmark slot.

### 2. Remove unnecessary transfers without changing the equations

Concrete sites:

- `WarpBackend._compact`: `self.offsets.numpy()[used - 1]` downloads the entire
  scan array to read one count. `WarpNeutralModel.compact` repeats this pattern.
  Copy one device element into a reusable host scalar buffer instead. Keep the
  count-versus-ledger check and stable compaction order. Initially retain both
  scans to isolate the change; a later exclusive-scan-plus-last-flag count can
  remove the second scan with an independent count witness.
- `_download` and neutral `download`: `.numpy()[:n]` downloads capacity first.
  Copy only the live prefix into host storage.
- `far_field_window_sums`: `.numpy()[far]` downloads the whole node array before
  slicing. Gather/copy the boundary slab on device where worthwhile.
- Add a device-resident advance API so ordinary runner chunks need not export
  all particles. Keep explicit snapshots for requested checkpoints and callers
  that need a host state; never change the existing return contract silently.

At an illustrative capacity of 4 million int32 entries, a count read is 16 MB
instead of 4 bytes. This is a transfer-size fact, not a claimed step speedup:
the operation runs at sync intervals, so its end-to-end share must be measured.

Acceptance: exact particle order/state, integer ledger and checkpoint parity;
unchanged overflow/count failures; tests with holes, empty species, capacity
growth and resume. CUDA transfer measurements remain required.

### 3. Replace the Coulomb rank stage with exact-order radix sorting

`coulomb_rank_kernel` computes each particle's rank by scanning every entry in
its cell. With occupancy k_c, its work is sum_c(k_c²). A concentrated cusp can
therefore dominate even when total particle count is unchanged. The subsequent
cell preparation also serially sums and shuffles each segment; profile that
separately after removing rank cost.

Use a permutation sorted by the unique key `(cell_id, original_slot)`. A positive
int64 key `cell_id * 2**32 + original_slot` works within the current int32 index
bounds; inactive entries use a sentinel after the live keys. Warp 1.14 supplies
`wp.utils.radix_sort_pairs` with int64 keys/int32 values. Its arrays require
2*capacity entries: 24 bytes per capacity slot for these key/value arrays, plus
other stage state and sort workspace. Warm/reserve the workspace on the actual
stream before capture and check capture/reallocation behaviour on CUDA.

This sorts only a permutation. It does not rearrange particles or change their
RNG slot keys. Restore identical starts and cell-of-sorted arrays, then keep the
existing moments, Fisher–Yates shuffle, pair construction and collision kernels.
Identical ordered input permits identical pairing; sorting physical particle
arrays would instead require a persistent-ID RNG migration and is a different
project.

The accompanying [ordering probe](../tools/coulomb_ordering_probe.py) compares the
existing Warp rank kernel against radix ordering on CPU. All six cases matched
exactly, including holes, empty populations and a dense hotspot. The hotspot has
6,144 live particles and 4,685,544 rank comparisons. This establishes the
ordering replacement on those fixtures, not full collision replay or GPU speed.

Acceptance before production: identical permutations, RNG/pair identities and
particle states over full Coulomb cycles; graph/direct and resume parity;
relaxation/conservation tests; performance across occupancy distributions,
including memory and sort setup. Retain the old path as a comparison until
the benefit is measured. Small, dilute cases may not benefit from radix overhead.

### 4. Reduce Poisson serial depth; select by grid and deployment

Keep block-Thomas as the channel baseline. Retain MG as the scalable plume
candidate, subject to its qualification. At default settings the current MG
sequence costs 278 / 362 / 446 launches for four/five/six levels. Its coarse
dense solve uses one thread per output row with a serial dot product, leaving
another small parallelisation opportunity.

Two stages of work:

1. Fuse prolongation with the first post-smoothing operation; investigate
   temporally blocked smoothing and tiled coarse solves. Preserve the exact
   masked/Galerkin operator and independently check true residuals. An ordinary
   CUDA block barrier is not a grid-wide barrier: two Jacobi sweeps cannot simply
   be put into one in-place kernel. Use valid halo/redundant evaluation or a
   supported global synchronisation design. Likewise, two-colour red-black
   ordering is not independent for a nine-point stencil with diagonal links;
   choose colouring appropriate to each level.
2. Prototype a bounded, device-controlled convergence loop using Warp's existing
   conditional graph capability. Start from the previous potential, execute a
   small cycle batch, evaluate the independent true residual, and continue only
   while needed, with a hard maximum and fail-closed status. Include check and
   conditional-node overhead in timing. Fixed 14 cycles remain the reference.

The second stage preserves the residual target but changes floating-point
solutions and hence long-run stochastic trajectories. Require a new solver
identity, field/E-force comparisons, manufactured convergence, Gauss law,
surface-charge tests and statistical replay. Do not lower residual tolerances
or remove the interval-worst failure witness merely to win a benchmark.

Alternative for channel grids: a partitioned direct/SPIKE or cyclic-reduction
spike can shorten the block-Thomas dependency chain, but dense block work and
setup may erase the gain. It is a measured competitor, not an automatic upgrade.

### 5. Particle locality and event pipelines after the profile

Charge/field gather and moment deposition are candidates for cell/tile ordering
and local accumulation. Prefer indirection over the existing particle slots
first. For bitwise fixed-point deposition, quantise each particle contribution
before integer tile summation; quantising one combined float sum changes the
method. Bound local/global int64 overflow.

MCC still scans capacity-sized birth flags even when few events occur. A compact
event list may help, but must preserve unique child allocation, birth order,
product RNG streams, overflow checks and incremental ion charge. Use the current
full-physics candidate rate, not the much lower historical 0-D-gas rate, for the
cost argument.

Fusion must preserve the actual stage order, including SEE, anomalous rotation,
Coulomb and ion MCC. A blanket push+MCC fusion skips the intervening operators
in the present full-physics sequence. Profile register pressure and occupancy
before growing the already large particle kernels further.

The WarpX GPU-porting paper supports particle locality as an architectural
candidate, not an Open-CFT speedup estimate:
[Myers et al., 2021](https://arxiv.org/html/2101.12149v2).

## What preserving accuracy requires

Separate three claims:

1. **Same discrete calculation:** scalar transfers, exact permutation sorting,
   stable compaction and compatible integer reductions should preserve particle
   state bitwise. Require replay and checkpoint tests.
2. **Same resolved physics within uncertainty:** changed solver/reduction order,
   moment cadence and precision may need statistical comparisons and convergence,
   not long-trajectory bitwise equality.
3. **Predictive thruster accuracy:** neither of those proves agreement with the
   device. Closures, boundary conditions, dimensionality and data need validation.

The v2.0.6 ledger correction added the missing macro weight to inelastic losses.
The recorded ss-v4 window then reads **+2.46%** heating and fails its own **<+2%**
acceptance, despite the older PASS. The 5% emergency stop is not an accuracy
target. Optimised runs must be assessed with the corrected ledger, resolved
Debye/plasma-frequency gates, documented dense-temperature floor and the declared
physical acceptance. Passing a unit suite or staying just below a stop is not
convergence. See [performance audit section 13](pic2d-performance-audit.md).

K=5 diagnostic sampling needs an additional current-model check: Coulomb runs
every 10 steps and neutrals every 200 in the full-physics protocol. Sampling
every 5 steps observes only selected subcycle phases. This is a potential aliasing
bias, not a demonstrated failure. Compare K=1 and K=5, sample offsets, gate
crossing decisions and occupancy-dependent temperatures on the full model.

Keep float64 particles/fields and the current integer deposition baseline.
Any lower-precision preconditioner or temporary storage needs final float64 true
residual/force checks. Global FP32 or fast-math, fewer particles, increased dt,
permittivity/mass scaling, smoothing, particle merging and neutral acceleration
are numerical/model changes with separate convergence studies.

The code remains axisymmetric in position: carrying v_theta does not resolve
azimuthal fields or instabilities. Bohm transport is a closure. SEE, Coulomb,
ion-neutral collisions and spatial neutrals now exist, but their availability
does not make the full CFT externally validated. The top of the older physics
audit must be read together with its later implementation updates and the
[full-physics campaign](../experiments/pic2d_full_physics_v1/README.md).

For statistical acceptance, preregister tolerances for discharge/beam current,
ionisation, density, electron temperature/EEDF tail, per-cusp wall loss and sheath
drop, IEDF/divergence and thrust. Use independent seeds and autocorrelation-aware
uncertainties; record-to-record scatter is not the uncertainty of a mean.
Keep grid, timestep and particle-number convergence distinct from the code A/B.

## The longer-term architectural limit

The full-physics campaign explicitly distinguishes a plasma plateau after
7.2 microseconds from neutral equilibration on roughly 0.2–2 milliseconds.
At dt=1.4 ps, 7.2 microseconds needs about 5.14 million steps. Every 1 ms/step
therefore costs 1.43 hours for that interval. At a hypothetical 2 ms/step,
0.2–2 ms of physical evolution costs about 79–794 hours. These are arithmetic
illustrations, not measured timings or guaranteed equilibration times.

For steady-state design objectives, a separately validated plasma/neutral
multirate or fixed-point method could address this timescale separation. It must
be compared with F=1 physical-time windows, show convergence in coupling interval
and initial condition, and distinguish a steady state from breathing dynamics.
For transient/breathing claims the physical relative timescales must remain.
Energy-conserving/implicit PIC is a later research path; conservation alone does
not guarantee resolved cusp sheaths or accurate transport on a coarse grid.

Independent design/seed runs should be distributed across GPUs before attempting
domain decomposition of one small 2D problem. Benchmark MPS by aggregate accepted
throughput and longest-job completion time: historical MG contention was severe.
No live campaign or paid GPU job was launched or modified during this review.

## Verification performed and next reviewable changes

Environment: Python 3.12, NumPy 2.5.2, Warp 1.14.0, pytest 9.1.1, CPU only.
The focused existing suite passed **69 tests** with 10 skips. Three skips were
real-map fixtures absent from the initial sparse checkout; after retrieving
those tracked maps, all **3 production-map tests passed**. Total unique executed
tests: **72 passed**; **7 CUDA-only cases unavailable**. Tested areas include
particle kernels, Poisson/Gauss law, MG convergence, performance-parity fixtures,
the corrected ledger and Coulomb conservation/relaxation. This was not the full
repository suite. The six-case ordering probe also passed.

Reproduction from `modern/`, with the packages above installed:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m pytest \
  tests/pic2d/test_pic2d_kernels.py \
  tests/pic2d/test_pic2d_mesh_poisson.py \
  tests/pic2d/test_pic2d_poisson_mg.py \
  tests/pic2d/test_pic2d_v205_performance.py \
  tests/pic2d/test_pic2d_v206_ledger.py \
  tests/pic2d/test_pic2d_v24_coulomb.py -q
PYTHONPATH=src python -m tools.coulomb_ordering_probe
```

The next implementation changes should be small enough to measure independently:

1. Benchmark current physics with separate step/export/I/O timings, plus scalar
   count downloads and live-prefix exports.
2. Replace Coulomb rank with the exact permutation sorter; qualify CUDA graphs,
   full collision replay, occupancy scaling and memory.
3. Reduce MG launches, then separately test residual-controlled cycles; retain
   BT as the channel comparator and perform corrected statistical acceptance.
4. Only after those measurements, select particle tiling/event-list work or
   pursue the larger multirate/implicit research programme.
