# `cft_revival.pic2d` performance audit (H100, 2026-09-04)

Read-only audit of the production PIC-MCC step (`modern/src/cft_revival/pic2d/warp_backend.py`,
`simulation.py`, `poisson.py`, `mcc.py`, the shared runner `experiments/pic2d_cft_steady_state_v1/run.py`
that v4 / v5 / the mini-sweep / the external validation reuse) at commit `a0235676` (the step code is
byte-identical at `a529b457`, the box head during the probes). Nothing in the production step was changed;
this document is the only deliverable. All numbers are from the Lambda H100 box
(`ubuntu@68.209.75.2`, H100 80GB HBM3, driver 580.105.08, Warp 1.14.0 cu12.9) and from the recorded solo
benchmarks in `/lambda/nfs/h100-files/cft/bench*` (`tools/cloud/bench_gpu_concurrency.py`, same step code).

§12 (added later the same day) is a second exception to "nothing changed" beside §11: it records the built and measured
`poisson_gmg_v1` multigrid field solve (§8 item 3), selectable per protocol, not used by any preregistered run.

Companion document: `modern/docs/literature/pic-acceleration-methods.md` (cited review of acceleration
methods, written against the earlier 5090-based cost anatomy "block-Thomas sweeps + inverse-block reads
+ 0.733 ms per M particles"). This audit is the measured H100 cost anatomy that review's options should
be weighed against; §6.1 reconciles the two cost models.

## 1. Summary

* The production step is **GPU-bound and latency-dominated, not bandwidth- or compute-bound**: the
  CUDA-graph replay keeps the GPU busy (host issue + gaps ≈ 1 % of the step), but roughly half of the
  GPU time is spent in dependent chains of tiny kernels and in reductions that use a handful of thread
  blocks. Solo H100 step times: channel-33 µm **3.31 ms** (2.15 M electrons + 2.26 M ions),
  plume-v2.0-50 µm **7.20 ms** (4.25 M + 4.35 M). Two-load fits give
  `step = fixed + slope × N_e` with **fixed 1.24 ms + 0.97 ms per M electrons** (channel-33) and
  **2.86 ms + 1.02 ms per M electrons** (plume-50); the per-electron cost is the same on every grid.
* **Dominant bottleneck (by a small margin): the born-particle ledger diagnostics, not the Poisson
  solve.** Every step runs three 4096-thread strided sums over the whole electron array
  (`energy_sum_kernel`, 2 × `momentum_sum_kernel`, each with 16 thread blocks at ~20 GB/s) followed by
  three **single-thread** `deferred_add_kernel` launches that add 4096 partials serially
  (360 µs each contended, 70–115 µs solo). Together ≈ **1.1–1.2 ms of the 3.31 ms channel-33 step
  (≈ 35 %)** and ≈ 2.0 ms of the 7.2 ms plume step (28 %). They compute two ledger scalars
  (`ke_born_ions_j`, `pz_born`) that the MCC kernel already has in registers.
* **Second: the exact block-Thomas Poisson solve** — 2 (n_r + 1) dependent dense row-block matvecs per
  step (182 launches / 0.75 GB of inverse-block reads on channel-33; 482 / 2.0 GB on plume-50). It fits
  `4.1 µs per launch + 0.30 ms per GB` across three grids, i.e. it sits at the in-graph launch-latency
  floor: **0.97 ms (29 %)** on channel-33, **2.6 ms (36 %)** on plume-50. It does not dominate the
  channel grids; it will dominate the 33 µm plume box (722 launches, 12 GB per solve → ≈ 6.5 ms).
* **Third: the window/frame diagnostic moment deposition** (`deposit_moment_kernel`: 20 float64
  same-node atomics per electron, every step because the runner accumulates from step 0):
  **0.65 ms (20 %)** on channel-33, 1.3 ms (18 %) on plume-50 (accumulate on/off graph difference
  measured directly).
* Everything the physics actually needs — charge deposition, Poisson solve, field, Boris push, wall
  handling, MCC + spawn — is ≈ **45–55 % of the step** (channel-33 / plume-50); the rest is diagnostics
  and reduction plumbing.
* Ions are already subcycled (`ion_subcycle = 8`); ion push + redeposit cost ≈ 3–4 %. Host-side work
  (sync every 200 steps, series record, checkpoint every 40 000 steps, frame every 20 000) is 2–3 %.
* Top three recommendations (details and verification in §8–9):
  1. Fold the born-ion kinetic energy and born momentum tallies into the MCC kernel's existing tile
     reductions and delete the strided sums + single-thread adds (also use the two-stage reduction for
     every remaining `final_sum`/`deferred_add`). **−1.1 ms channel-33 (×1.5), −2.0 ms plume-50
     (×1.4)**; 0.5–1 day; physics bitwise, two ledger scalars change at round-off.
  2. Take the window moment deposition off the per-step critical path: sample every K = 5 steps (the
     window statistics are 400 000-step averages; `diag_steps` already counts accumulated steps) and/or
     capture it on a forked stream inside the step graph so it overlaps the latency-bound Poisson
     chain. **−0.5 … −0.65 ms channel-33, −1.0 … −1.3 ms plume-50**; 1–3 days; physics bitwise,
     window maps/gate statistics change at the sampling-noise level (K) or not at all (overlap).
  3. Poisson: keep the exact block-Thomas on channel grids (it is within ≈ 1.5× of the latency floor;
     an exact partitioned/SPIKE variant halves the chain for ≈ −0.35 ms) and build a Warp-native
     geometric multigrid with a fixed, graph-capturable V-cycle count for the plume grids, where the
     dense inverse blocks' bandwidth term takes over (plume-50 2.6 → ≈ 1.0 ms; plume-33 ≈ 6.5 → ≈ 1.2 ms).
     1–3 weeks; numerics change within the 1e-10 residual contract, so only statistical parity with the
     accepted plateau is possible.
  Combined: **channel-33 3.31 → ≈ 1.7 ms after (1)+(2) (≈ ×1.9), ≈ 1.2 ms with (3) (≈ ×2.7);
  plume-33 (v2.1 box) ≈ 17 ms (model) → ≈ 10 ms → ≈ 4 ms (≈ ×4).**
* Probe cost on the box: four short MPS-client processes, 291 s of wall time in total (≈ 4.9 GPU-minutes
  as an extra client; no process was signalled; `dmesg` shows no Xid).
* **Status (2026-09-04, later the same day): recommendations (1) and (2) are implemented as model v2.0.5**
  (`f80c6441`; the box verified `a156fd84`, the same file contents before the rebase; spec `pic2d-model-v2.0.json#performance_v2_0_5`); the measured outcome — contended A/B on the
  H100 under identical MPS load, per-kernel tables before/after, the physics-bitwise replays and the
  fitted-solo estimate — is in §11. Recommendation (3) is untouched.

## 2. Method and honesty notes

* **Code reading** (§3–5) covers `warp_backend.py` (`WarpBackend._launch_step`, `_sync`,
  `WarpBlockThomas`, the kernels), `simulation.py` (`Simulation.run`, `_record`, window classes),
  `poisson.py`, `mcc.py`, `mesh.py`, `frames.py`, and the runner loop in `run.py`.
* **Measurements** come from a read-only probe (`/lambda/nfs/h100-files/cft/perf-audit/profile_pic2d_step.py`,
  not part of the repository) that builds the simulation through the production path
  (`build_config → load_inputs → Simulation`, the same `CONFIGS`/`load_bench_protocol` as
  `tools/cloud/bench_gpu_concurrency.py`, production particle load), warms up 400 steps and then runs:
  S1 2000 graph-mode steps (wall, per-graph CUDA-event time via `wp.TIMING_GRAPH`, every kernel /
  memcpy / memset issued outside the graph, host wall of `_sync`, `_record`, `export_state`);
  S2 1000–2000 steps with the step graph and the Poisson graph disabled so every kernel is timed by
  CUDA events (`wp.timing_begin/timing_end`, `TIMING_KERNEL|KERNEL_BUILTIN|MEMCPY|MEMSET`);
  S3 the bound Poisson graph alone × 200 and 200 graph steps with `accumulate=False`;
  S4 one production checkpoint save and one frame-like capture; S5 the production reduction kernels
  launched with different shapes. `nsys` is not installed on the image; Warp's event timing was used.
* **Contention.** The probes ran as the 5th–6th CUDA-MPS client next to the three live mini-sweep
  runs, the v5 shakedown and (briefly) the external-validation shakedown, all at 100 % GPU utilisation.
  Every absolute kernel time below is therefore **contended**: the graph step took 2.5× its solo time
  (8.21 vs 3.31 ms channel-33; 17.8 vs 7.2 ms plume-50). The inflation is very uneven: 1-thread and
  node-sized kernels wait ≈ 30 µs for a scheduling slot (× 7–10), the dense-block sweeps take 37–39 µs
  per launch (× 9), while the large particle kernels run at ≈ 1.1–1.4× their solo time (the other
  clients are themselves latency-bound and leave the bandwidth free — the same effect that gives MPS
  its 1.54× aggregate). **Solo estimates** are therefore derived, not measured: the fixed part from
  the three-grid launch model fitted to the solo anchors, the particle part by distributing the solo
  per-electron slope over the kernels in proportion to their contended per-electron slopes (two loads
  on channel-33). The model reproduces the solo anchors to +7 % (channel-33) and +2 % (plume-50);
  treat individual solo entries as ±25 %.
* **Kernel-level events add overhead** and S2 runs without graphs (host issue ≈ 55 µs per launch on the
  loaded host: 12 ms/step channel-33, 27 ms/step plume — which is why the step graph exists). The
  per-kernel GPU times are still bracketed per kernel and independent of the host gaps.
* `wp.utils.array_scan` (CUB) is not captured by Warp's timing; its cost (≈ 3 launches, tens of µs
  solo) is inside the graph total but absent from the per-kernel tables.
* The recorded solo anchors (bench `_time_steps`) include one `export_state` (full particle download,
  0.25 s on channel-33) at the end of the 2000 timed steps: 0.13 ms/step of the 3.31 ms is that
  download, so the steady production step is ≈ 3.2 ms. In production `export_state` runs once per
  20 000-step chunk (0.01 ms/step).

Probe records (JSON + log, kept on the box): `perf-audit/channel-33um-production.json` (11:50 UTC,
3–4 other clients), `channel-33um-seed.json` (11:55, 4–5), `plume-v2.0-50um.json` (11:53, 3–5),
`channel-25um-120350.json` (12:03, 3–4).

## 3. The production step, end to end

Configuration common to v4 / v5 / mini-sweep / ext-val: `poisson.method = "device-direct"`,
`fixed_point_deposition = True`, `device_sync_steps = series_interval_steps = 200`, `ion_subcycle = 8`,
`step_graph = True`, frames every 20 000 steps, window 400 000, checkpoint 40 000, MCC on
(`P_candidate = 1 − exp(−ν_max Δt) = 6.0e-5` per electron per step on channel-33, 1.3e-4 on plume),
emission on (`max_inject_per_step` 1 / 2 / 3 on channel-33 / 25 / plume), `anomalous = None`.

**dtypes.** Every particle and node array is `float64`; the charge/wall accumulators are `int64`
fixed point (`2^-40` quantum, order-independent, bitwise across backends); flags, slots, offsets and
stencil codes are `int32`; the inverse blocks are `float64`. Nothing in the step is `float32`
(frames are converted on the host). The diagnostic window sums are `float64` atomics.

**The runner always accumulates.** `run.py` calls `sim.run(chunk, accumulate_from_step=window_start)`
with `window_start ≤ step`, so `accumulate` is `True` on every production step; the "accumulate off"
graph variant never runs in production.

### 3.1 Kernel sequence of one step (graph-captured, `_launch_step(fixed_shape=True)`)

Launch dimensions in graph mode are the array **capacities** (≈ 2× the live count after `_upload`,
≥ 1.5× after growth); every particle kernel exits early for `p ≥ slots[slot]`. `N_e`, `N_i` = live
counts, `cap` = capacity, `nodes = (n_r+1)(n_z+1)`, `m = n_z+1`.

| # | kernel | dim | class | reads / writes per element | atomics | launches/step |
|---|---|---|---|---|---|---|
| 1 | `acc_e.zero_()` | nodes | memset | 8 B | – | 1 |
| 2 | `deposit_fixed_kernel` (electrons) | cap_e | per-particle | r, z, alive (20 B) | 4 × int64 add per particle | 1 |
| 3 | `int_to_charge_kernel` (q_e) | nodes | per-node | 8 → 8 B | – | 1 |
| 4–5 | `acc_i.zero_()` + `deposit_fixed_kernel` (ions) | cap_i | per-particle | as 2 | 4 × int64 | 1/8 (`redo_ions` after an ion push) |
| 6 | `int_to_charge_kernel` (q_i) | nodes | per-node | | – | 1 |
| 7 | `source_kernel` | nodes | per-node | q_e, q_i, ratio, surface, offset, unknown → rhs | – | 1 |
| 8 | `block_forward_kernel` × (n_r+1) | m × 256 threads | dense row-block matvec | G_{i−1} row block m² × 8 B + y, b, c | – (tile_sum) | n_r+1 |
| 9 | `block_backward_kernel` × (n_r+1) | m × 256 | dense row-block matvec | G_i row block m² × 8 B | – | n_r+1 |
| 10 | `apply_dirichlet_kernel` | nodes | per-node | x, boundary, unknown → phi | – | 1 |
| 11 | `efield_kernel` | nodes | per-node | phi (5-point, coded stencils) → e_r, e_z | – | 1 |
| 12 | `peak_density_kernel` | nodes | per-node → scalar | q_e, inverse_volume | 1 × float64 `atomic_max` on ONE address per node | 1 |
| 13 | `axpy_kernel` (d_phi += phi) | nodes | per-node accumulator | | – | 1 (accumulate) |
| 14–15 | `abs_axpy_kernel` (d_n_e, d_n_i) | nodes | per-node accumulator | | – | 2 (accumulate) |
| 16 | `deposit_moment_kernel` (electrons) | cap_e | per-particle | r, z, vr, vt, vz, alive (44 B) | **20 × float64 add** per particle (5 arrays × 4 nodes) | 1 (accumulate) |
| 17 | `push_kernel` (electrons) | pad(cap_e, 256) | per-particle | 44 B in, 16 gathers (E_r, E_z, B_r, B_z × 4 nodes), 40 B out, plasma_cell | boundary hits only (3–8 per absorbed particle: stats, wall int64 ×4, wall columns); 3 tile_atomic_add + 1 atomic_max per block | 1 |
| 18 | `push_kernel` (ions, Δt × 8) | pad(cap_i, 256) | per-particle | as 17 | as 17 | 1/8 |
| 19 | `wall_int_to_charge_kernel` | nodes | per-node | acc_wall → surface (+ reset) | – | 1 |
| 20 | `mcc_kernel` (electrons) | pad(cap_e, 256) | per-particle | alive (4 B), seed table, `randf` per particle; velocities + σ-table only for the 6e-5 candidates; writes `ionize[p]=0` for all p < cap_e | 7 tile_atomic_add per block | 1 |
| 21 | `array_scan` (CUB, exclusive) on `ionize[:cap_e]` | cap_e | scan | 4 B → 4 B | – | 1 (2–3 CUB launches) |
| 22 | `spawn_kernel` | cap_e | per-particle | writes `born_flag[p]=0` for all p; copies 8 values per birth | 1 float64 on overflow | 1 |
| 23 | `energy_sum_kernel` (born ions) | **4096 threads** | strided reduction over `slots[0]` | born_flag 4 B per electron (KE only for births) | – | 1 |
| 24 | `deferred_add_kernel` | **1 thread** | serial sum of 4096 partials | 32 KB | – | 1 |
| 25–28 | `momentum_sum_kernel` + `deferred_add_kernel` (born ions, secondary electrons) | 4096 / 1 | as 23–24 | born_flag 4 B per electron each | – | 2 + 2 |
| 29 | `deposit_fixed_kernel` (born ions → acc_i) | cap_e | per-particle | born_flag 4 B | 4 × int64 per birth | 1 |
| 30 | `deposit_unit_kernel` (born → d_ion) | cap_e | per-particle | born_flag 4 B | 4 × float64 per birth | 1 (accumulate) |
| 31 | `spawn_commit_kernel` | 1 | scalar | | – | 1 |
| 32 | `inject_control_kernel` | 1 | scalar | | – | 1 |
| 33 | `inject_kernel` | max_inject_per_step (1–3) | scalar-ish | 7 `randf` | 2 float64 | 1 |
| 34 | `add_injected_slots_kernel` | 1 | scalar | | – | 1 |
| 35 | `carry_kernel` | 1 | scalar | | – | 1 |
| 36 | `tick_kernel` | 1 | scalar | | – | 1 |

Launch count per step: channel-33 **≈ 215** (182 sweeps + 31 kernels + 2 memsets + CUB scan);
channel-25 ≈ 275; plume-50 **≈ 515** (482 sweeps). The "482 launches" of the plume cost model are
the sweeps alone.

Per-electron work per step: 6 full passes over the particle arrays (deposit, moments, push, MCC,
scan, spawn) plus **5 passes over a 4-byte flag array** (energy_sum, 2 × momentum_sum, born deposit,
deposit_unit) and one flag-array write in `spawn` (`born_flag = 0`) and one in `mcc` (`ionize = 0`,
over the capacity). Atomics per electron per step: 4 int64 (charge) + 20 float64 (moments) = 24;
per ion: 4 int64 every 8th step.

### 3.2 What the CUDA graph captures and what stays outside

* **Captured (one graph per `(ion_step, redo_ions, accumulate, array pointers, capacities)`):**
  everything in §3.1 including the Poisson sweeps (raw launches inside the step capture) and the CUB
  scan. Scalars are frozen at capture (v1.4 lesson): the neutral density, injection rate/carry, seeds
  and slot counts live in device arrays. Warm-up captures 3 variants (ion step on/off × ion redeposit).
  Graph re-capture only after a particle-array reallocation (`_grow`, 1.5× steps).
* **Outside the graph, once per 200-step sync interval** (`step()` on the last step +
  `_sync()`): `queue_residual_check` (matvec, residual, 2 × dot_stride, 2 × reduce_stage,
  2 × final_sum = 8 launches), D2H of `stats` (41 doubles), `slots` (2 ints), `scalars` (verify);
  compaction per species: `array_scan` ×2, **D2H of the whole `offsets` array** (`self.offsets.numpy()`
  copies cap × 4 B = 18 MB channel-33 / 35 MB plume to read one element), `compact_kernel`, 6 D2D copies,
  2 memsets; H2D of the seed table (600 ints), slots, `inject_ctrl`, and `neutral_density_ctrl`
  (`set_neutral_scale` from `_record`). `_sync` alone: 5 D2H + 3–4 H2D transfers per interval.
* **Outside the graph, once per series record (= every 200 steps):** `series_sample` (energy_sum ×2,
  momentum_sum ×2, sum, 5 × final_sum; D2H of `sample_out`, `phi`, `surface`), `peak_node_sample`
  (one extra `deposit_moment_kernel` over all electrons + 5 node-array D2H), `peak_window_sums`
  (6 node-array D2H), on plume grids `charge_maps` (2) and `far_field_window_sums` (2 partial), then
  host numpy: `field_energy_j`, `induced_electrode_charge_c`, `peak_node_debye`, the window ring
  updates, `boundary_forces_n` / `apply_operator` (plume), record serialisation, `status_from_record`,
  two JSONL appends. 15–19 node arrays per record (0.5 MB each on channel-33, 1.4 MB on plume).
* **Every 20 000 steps:** frame capture (`diagnostic_sums` = 19 device arrays D2H, `interval_maps`,
  `savez_compressed`: 2 + 62 ms channel-33, 4 + 45 ms plume, measured) and `export_state` at the end of
  each `sim.run` chunk (full particle D2H: 0.25 s channel-33, 0.50 s plume).
* **Every 40 000 steps:** `save_checkpoint_atomic` (0.79 s / 173 MB channel-33, 1.41 s / 345 MB plume,
  1.32 s / 309 MB channel-25, measured on the NFS results directory), plus plateau / triad / ignition
  evaluation over the series arrays (negligible).

Host syncs per step: **0** inside the interval. Per interval: the D2H reads above (each a stream
sync). Measured host wall per interval (contended): `_sync` 5.3 ms channel-33 / 40 ms plume,
`_record` 13 ms / 38 ms → 0.09 / 0.39 ms per step contended, ≈ 0.05–0.15 ms solo (1.5–3 % of the
step). The graph issue itself costs 5–6 µs per step; the wall − GPU gap is 0.08 ms/step (1 %).

## 4. The Poisson solve (`WarpBlockThomas`)

Operator: masked cylindrical finite-volume Laplacian on the node mesh (`mesh.py`: edge conductances
`ε₀ A/ℓ`, homogeneous Neumann into solids, Dirichlet anode / exit / far field, identity rows for
non-plasma nodes), SPD on the unknowns. The device-direct method blocks the unknowns by **radial row**
`i` (one block = the whole axial row, `m = n_z + 1` nodes; coupling to row `i+1` is the diagonal
`−cond_r`). On the host (`__init__`, numpy) it forms the Schur complements
`S_i = D_i − C_{i−1} S_{i−1}^{-1} C_{i−1}` and stores **every `G_i = S_i^{-1}` as a dense `m × m`
float64 block** (`np.linalg.inv`, O(n_r m³) ≈ 10¹¹ flop: 1.3 s on the box for channel-33, 3.3 s
plume-50, 5.8 s v2.1 at 16 threads — recorded 2026-09-04; 5–12 min on the Windows PC). A solve is then

```
forward   y_i = b_i − C_{i−1} (G_{i−1} y_{i−1})      i = 0 … n_r      (one launch each)
backward  x_i = G_i (y_i − C_i x_{i+1})                i = n_r … 0      (one launch each)
```

= **2 (n_r + 1) sequential dense matvecs**, each reading one `m × m` block (`block_forward_kernel` /
`block_backward_kernel`: one 256-lane block per output row, coalesced strided reads, deterministic
tile reduction, no atomics), captured in the step graph; no host sync; the true residual is checked
against `relative_tolerance = 1e-10 · |rhs|` at every sync (`verify`).

| grid | cells | nodes | unknowns | m | blocks | G (GB) | read per solve (GB) | sweep launches | solo Poisson (model) |
|---|---|---|---|---|---|---|---|---|---|
| channel-50 | 60 × 480 | 29 341 | 20 779 | 481 | 61 | 0.11 | 0.22 | 122 | 0.57 ms (33 % of 1.71) |
| channel-33 (v4, sweep) | 90 × 720 | 65 611 | 46 469 | 721 | 91 | 0.38 | 0.75 | 182 | **0.97 ms (29 % of 3.31)** |
| channel-25 (v5) | 120 × 960 | 116 281 | 82 359 | 961 | 121 | 0.89 | 1.78 | 242 | 1.5 ms |
| plume-v2.0-50 | 240 × 720 | 173 761 | 78 228 | 721 | 241 | 1.00 | 2.00 | 482 | **2.6 ms (36 % of 7.20)** |
| plume-v2.1-50 (48 × 12 mm) | 240 × 960 | 232 561 | ≈ 136 k | 961 | 241 | 1.78 | 3.56 | 482 | 3.0 ms |
| plume-v2.1-33 | 360 × 1440 | 520 201 | ≈ 300 k | 1441 | 361 | 6.0 | 12.0 | 722 | **≈ 6.5 ms** (≈ 40 % of ≈ 17) |

**Complexity.** Time per solve O(n_r m²) = O(N_nodes · m), memory O(n_r m²): both O(N^1.5) at fixed
aspect ratio — yes, it is the O(N^1.5) method the question suspected. The dense blocks are also
wasteful on the L-shaped plume domain: 55 % of the plume nodes are solids (identity rows), yet every
`G_i` is stored and read as a full `m × m` block (rows `i` above the channel wall have only the 241
plume-box unknowns of their 721 nodes).

**Measured cost.** Contended: 37–39 µs per launch on every grid (m = 721 and 961 alike — the launch is
latency-bound, not bandwidth-bound), 4.16 ms per solve on channel-33 (S3, graph alone), 10.6 ms on
plume-50, 5.9 ms on channel-25. Solo (three-grid fit of the anchors' fixed part):
`4.1 µs × launches + 0.30 ms/GB × bytes + 0.27 ms` reproduces channel-50 / 33 / plume-50 to
< 2 %; the 0.27 ms residual is the particle-independent `deferred_add` chain + node kernels. The
H100's dependent in-graph kernel-to-kernel latency is ≈ 2.5–3 µs and 4.16 MB at 3 TB/s is 1.4 µs,
so **the sweep is at ≈ 1.5× the hardware floor for this algorithm**; on channel grids nothing short
of a shorter dependency chain helps, on plume grids the bytes term (0.6 ms of 2.6; 3.5 ms of 6.5 at
33 µm) becomes the lever.

**Does it dominate?** No — 29–36 % on the current grids; the born-ledger and window diagnostics
together are larger (§6.3). It becomes the largest single item once those are removed, and it is the
scaling wall for the 33 µm plume box (12 GB of inverse-block reads per step, 6 GB of device memory).

## 5. Deposition, MCC, diagnostics — what the code does

* **Charge deposition** (`deposit_fixed_kernel`): bilinear (CIC) weights on the node mesh, rounded to
  `2^-40` fixed point, `int64` atomics (4 per particle), converted to coulombs by `int_to_charge_kernel`.
  Order-independent → bitwise identical across CPU / warp-cpu / cuda / MPS. Ions are frozen between
  subcycle pushes; births are added incrementally with a second `deposit_fixed` over `born_flag`.
* **Wall charge** (inside `push_kernel`): renormalised bilinear surface deposit onto plasma nodes at the
  crossing point, `int64` fixed point, converted by `wall_int_to_charge_kernel` (also resets).
* **Push** (`push_kernel`): gather of E_r, E_z, B_r, B_z at the 4 cell nodes (bilinear), relativistic
  Boris in orbit_mc operation order (all `float64`, 6 `sqrt`), position advance with the
  (x, y) → (r, θ) rotation, boundary classification (anode / exit / wall / Courant violation), tile-reduced
  work and momentum tallies, `atomic_max` of the block speed. Unsorted particles: the 16 gathers are
  random accesses into 0.5–1.4 MB node arrays (all four fit in the 50 MB L2), so the gather is
  L2-resident, not DRAM-bound.
* **MCC** (`mcc_kernel`): null-collision method; `nu_max` from the LXCat Biagi table at the configured
  n_g, `P = 1 − exp(−ν_max Δt) = 6e-5` (channel) / 1.3e-4 (plume); counter-based Warp RNG
  `rand_init(seed_table[3·counter], p)` **keyed on the particle index p**; every electron draws one
  `randf`; only candidates read velocities, look up σ (piecewise-linear, 40 001 × 3 table, 0.96 MB) and
  branch elastic / excitation / ionisation / null; the neutral density is read from a device array
  (v1.3 inventory scale, graph-safe). Births: `ionize` flag → CUB exclusive scan → `spawn_kernel`
  writes the secondary electron and the Maxwellian ion into the free slots; `spawn_commit` advances
  the slot counts. MCC + scan + spawn ≈ 4 % of the step: it is one full pass over the electrons plus
  two capacity-sized flag writes, not the collision physics.
* **Ledger / stats**: 41 float64 slots, block-tile reductions inside push / mcc / bohm / inject, read
  once per sync. Exception: the **born-ion kinetic energy and the born momenta**, which use the
  separate strided-sum + single-thread-add kernels (§3.1 rows 23–28) every step.
* **Window accumulators** (`accumulate=True`): `d_phi += phi` (node axpy), `d_n_e/d_n_i += |q|/V`
  (node axpy), **`deposit_moment_kernel`** (weight, v_r, v_θ, v_z, v² sums: 20 float64 atomics per
  electron; note `d_w` is redundant with `d_n_e` — same particles, same weights, only the fixed-point
  rounding differs), `deposit_unit` for the ionisation-rate map, wall / exit / side / θ / IEDF profiles
  inside push. Frames and gates read cumulative-sum differences of these arrays (exact interval
  averages, "nothing added to the step kernels" — true for the frames, but the per-step deposition
  that feeds them is 18–20 % of the step).
* **Per-record host work** (§3.2): the peak-Debye single-step witness re-deposits all electron moments
  and downloads 5 maps; the window gate downloads 6 maps to find one node's `argmax`; the plume gate
  downloads 2 charge maps and 2 far-field slices; compaction downloads the full `offsets` array twice.

## 6. Measurements

### 6.1 Solo anchors and two-load fits (H100, graph mode, bench `af9e79d1`)

| config | seed load | production load | fit: fixed | fit: per M electrons | fixed share at production |
|---|---|---|---|---|---|
| channel-50 | 1.075 ms @ 0.26 M e | 1.708 ms @ 0.95 M e | 0.83 ms | 0.92 ms | 49 % |
| channel-33 | 1.816 ms @ 0.60 M e | 3.312 ms @ 2.15 M e | 1.24 ms | 0.97 ms | 37 % |
| plume-v2.0-50 | 3.129 ms @ 0.27 M e | 7.196 ms @ 4.25 M e | 2.86 ms | 1.02 ms | 40 % |

The per-electron cost (0.92–1.02 ms per M electrons per step ≈ **1 ns per electron-step**) is
independent of the grid; a well-tuned electrostatic PIC push + deposit on an H100 is 0.05–0.1 ns.
The fixed part follows `0.27 ms + 4.1 µs × sweep launches + 0.30 ms per GB of inverse blocks`.
(The earlier plume cost model "3.82 ms fixed + 0.73 ms per M particles" was per M *total* particles on
the RTX 5090; on the H100 the same split reads 2.86 ms + 0.51 ms per M total = 1.02 per M electrons.)

### 6.2 Per-kernel profile, channel-33 µm at production load (contended; 2.02 M e, 2.26 M i)

S1 graph step 8.21 ms GPU (wall 8.44); S2 kernel sum 11.62 ms/step over 2000 direct-launch steps
(425 525 timed activities). Solo estimate column as described in §2; shares are of the 3.31 ms solo step.

| kernel | calls/step | contended ms/step | contended µs/launch | solo est. ms/step | solo share | bytes/step (model) | bound |
|---|---|---|---|---|---|---|---|
| `block_backward_kernel` | 91 | 3.49 | 38 | 0.49 | 15 % | 380 MB | latency: 91-deep dependent chain + 4.2 MB dense block per launch |
| `block_forward_kernel` | 91 | 3.45 | 38 | 0.48 | 15 % | 380 MB | as above |
| `deferred_add_kernel` (born KE, born p_z ×2) | 3 | 1.08 | 360 | 0.20–0.35 | 6–10 % | 0.1 MB | **latency: 1 thread, 4096 serial dependent loads** |
| `momentum_sum_kernel` (born ions, secondaries) | 2 | 0.83 | 410 | 0.56 | 17 % | 16 MB | **occupancy/latency: 4096 threads (16 blocks) stride 2 M flags at ≈ 20 GB/s** |
| `deposit_moment_kernel` (window moments) | 1 | 0.69 | 690 | 0.67 | 20 % | 730 MB (incl. 20 RMW atomics/e) | **atomics: 20 float64 same-node adds per electron** |
| `push_kernel` (e every step, i every 8th) | 1.12 | 0.38 | 337 | 0.31 | 9 % | 500 MB | bandwidth + L2 gather (16 random node reads per particle) |
| `energy_sum_kernel` (born ions) | 1 | 0.37 | 362 | 0.33 | 10 % | 8 MB | occupancy/latency (as momentum_sum) |
| `deposit_fixed_kernel` (e, born, i/8) | 2.12 | 0.26 | 121 | 0.15 | 5 % | 200 MB | atomics: 4 int64 adds per particle |
| `memcpy DtoH` (sync + record) | 0.11 | 0.19 | 1733 | 0.03 | 1 % | 0.2 MB/step (18 MB `offsets` ×2 + 15 maps per 200 steps) | pageable D2H, stream sync |
| `mcc_kernel` | 1 | 0.14 | 143 | 0.09 | 3 % | 26 MB | bandwidth (one RNG draw per electron; flag write over capacity) |
| `deposit_unit_kernel` (born → d_ion) | 1 | 0.08 | 77 | 0.02 | 1 % | 8 MB | flag pass |
| `spawn_kernel` | 1 | 0.07 | 66 | 0.02 | 1 % | 26 MB | flag pass + `born_flag = 0` over capacity |
| 10 node kernels (`int_to_charge` ×2, `abs_axpy` ×2, `axpy`, `source`, `apply_dirichlet`, `efield`, `wall_int_to_charge`, `peak_density`) | 10 | 0.35 | 29–42 | 0.04 | 1 % | 25 MB | latency (0.5 MB arrays); `peak_density` = 65 k same-address `atomic_max` |
| 6 scalar kernels (`spawn_commit`, `inject_control`, `inject`, `add_injected_slots`, `carry`, `tick`) | 6 | 0.19 | 27–39 | 0.02 | 1 % | – | latency (1 thread) |
| memsets | 1.2 | 0.03 | 28 | 0.005 | – | 1 MB | latency |
| host `_sync` + `_record` (per 200 steps, wall) | 0.01 | 0.09 (wall) | 5.3 + 13.2 ms per interval | 0.05–0.08 | 2 % | – | D2H + numpy |
| **sum** | ≈ 215 | 11.6 (GPU) | | **3.5** | 106 % of 3.31 | | |

### 6.3 Per-kernel profile, plume-v2.0-50 µm at production load (contended; 4.15 M e, 4.34 M i)

S1 graph step 17.77 ms GPU (wall 18.41); S2 kernel sum 25.95 ms/step over 1000 steps. Shares of the
7.20 ms solo step.

| kernel | calls/step | contended ms/step | µs/launch | solo est. ms/step | solo share | bytes/step | bound |
|---|---|---|---|---|---|---|---|
| `block_backward_kernel` | 241 | 9.35 | 39 | 1.32 | 18 % | 1.0 GB | latency chain + dense blocks |
| `block_forward_kernel` | 241 | 8.83 | 37 | 1.25 | 17 % | 1.0 GB | as above |
| `momentum_sum_kernel` ×2 | 2 | 1.54 | 766 | 1.10 | 15 % | 33 MB | 16-block strided sum |
| `deposit_moment_kernel` | 1 | 1.42 | 1416 | 1.28 | 18 % | 1.5 GB | 20 float64 atomics / e |
| `deferred_add_kernel` ×3 | 3 | 1.10 | 366 | 0.20–0.35 | 3–5 % | 0.1 MB | 1-thread serial sum |
| `energy_sum_kernel` | 1 | 0.70 | 695 | 0.66 | 9 % | 17 MB | 16-block strided sum |
| `push_kernel` | 1.12 | 0.70 | 621 | 0.62 | 9 % | 1.0 GB | bandwidth + L2 gather |
| `memcpy DtoH` | 0.15 | 0.59 | 4003 | 0.08 | 1 % | (35 MB `offsets` ×2 + 19 maps per 200 steps) | pageable D2H |
| `deposit_fixed_kernel` ×2.12 | 2.12 | 0.52 | 246 | 0.30 | 4 % | 410 MB | int64 atomics |
| `mcc_kernel` | 1 | 0.25 | 253 | 0.17 | 2 % | 51 MB | bandwidth |
| `deposit_unit` + `spawn` | 2 | 0.20 | 121 / 77 | 0.09 | 1 % | 68 MB | flag passes |
| 10 node kernels | 10 | 0.44 | 34–45 | 0.04 | 1 % | 60 MB | latency |
| 6 scalar kernels + memsets | 7.2 | 0.31 | 37–56 | 0.03 | – | – | latency |
| host `_sync` + `_record` | 0.01 | 0.39 (wall) | 40 + 38 ms per interval | 0.15 | 2 % | – | D2H + numpy (two 4 M compactions) |
| **sum** | ≈ 515 | 25.9 (GPU) | | **7.3** | 102 % | | |

Channel-25 µm (v5 grid, 3.6 M e / 4.0 M i, contended): graph step 12.15 ms; sweeps 9.44 ms (242
launches at 38–40 µs), momentum_sum 1.37, deposit_moment 1.23, deferred_add 1.09, push 0.64,
energy_sum 0.61, deposit_fixed 0.39, mcc 0.23 — the same ordering. Model solo ≈ 5.8 ms at this load
(the v5.1 preflight measured 10.8 ms/step at plateau load under MPS-4; the local 5090 ran 9.4 ms/step
at 0.6 µs).

### 6.4 Grouped solo budget (the ranked answer)

| phase | channel-33 (3.31 ms) | plume-50 (7.20 ms) | what it is |
|---|---|---|---|
| born-ledger reductions (3 strided sums + 3 single-thread adds) | **1.1–1.2 ms (33–36 %)** | **2.0 ms (28 %)** | diagnostics: `ke_born_ions_j`, `pz_born` |
| Poisson block-Thomas (2(n_r+1) sweeps) | **0.97 ms (29 %)** | **2.6 ms (36 %)** | physics |
| window moment deposition (`deposit_moment` + 3 node axpys + `deposit_unit`) | **0.65–0.7 ms (20 %)** | **1.3 ms (18 %)** | diagnostics (frames, maps, window gates) |
| push (e + i/8, incl. wall handling) | 0.31 ms (9 %) | 0.62 ms (9 %) | physics |
| charge deposition (e, born, i/8) + conversions | 0.17 ms (5 %) | 0.33 ms (5 %) | physics |
| MCC + scan + spawn + commit | 0.12 ms (4 %) | 0.23 ms (3 %) | physics |
| field (`source`, `apply_dirichlet`, `efield`), gate scalar (`peak_density`), injection, tick | 0.05 ms (1.5 %) | 0.05 ms (1 %) | physics + 1 gate scalar |
| host per interval (sync, compaction, record D2H + numpy) | 0.05–0.08 ms (2 %) | 0.15 ms (2 %) | bookkeeping + diagnostics |
| frames (per 20 k steps), checkpoints (per 40 k), `export_state` (per chunk) | < 0.03 ms (< 1 %) | < 0.05 ms (< 1 %) | I/O |

Direct measurements behind the two diagnostic rows: S3 graph step with `accumulate` on / off:
8.14 / 7.40 ms channel-33 (−0.75 ms contended), 17.85 / 16.28 ms plume (−1.57), 12.15 / 10.75
channel-25 (−1.41) — the moment deposition alone. S5 (channel-25 arrays, 3.6 M e, contended): the
production `energy_sum_kernel` over `born_flag` takes 611 µs with the production 4096-thread launch,
118 µs with 32 768 threads, **43 µs with 262 144 threads** (14×); over all alive electrons with the KE
computed (the `series_sample` use) 1472 → 261 → 98 µs; `deferred_add_kernel` (1 thread) 337 µs vs the
existing two-stage `reduce_stage` + `final_sum` pair 29 + 38 µs (both at the contended latency floor,
≈ 6–8 µs solo). Nothing new was compiled for S5 — only launch shapes changed.

### 6.5 Findings

1. **~1/3 of the production step is two ledger diagnostics.** Rows 23–28 of §3.1 exist to accumulate the
   kinetic energy of born ions and the axial momentum of born ions and secondaries. Their cost is
   entirely the reduction shape (16 blocks striding 8–17 MB, then one thread adding 4096 numbers, three
   times) — not the arithmetic. The MCC kernel already holds the ion and secondary velocities of every
   ionisation and already tile-reduces seven tallies into `stats`.
2. **The window moment deposition runs every step and costs ~1/5 of the step.** 20 float64 same-node
   atomics per electron; `d_w` duplicates `d_n_e`; the runner never runs the `accumulate=False` variant.
3. **The Poisson sweep is at ≈ 1.5× its launch-latency floor on channel grids** (4.1 µs per dependent
   launch vs ≈ 2.5–3 µs hardware; bytes 0.22 of 0.97 ms). On the plume grids the dense-block bytes
   term grows to 0.6 ms (50 µm) and 3.5 ms (33 µm) and the chain to 482 / 722 launches; the method is
   O(N^1.5) in time and memory and stores identity rows for the 55 % solid nodes of the L-shape.
4. **The physics kernels are cheap.** Push + charge deposition + MCC + field ≈ 0.65 ms (20 %) on
   channel-33 at 2 M electrons. Fusion / sorting / mixed precision act on this 20 %.
5. **The graph works.** Host issue and gaps ≈ 1 %; without the graph the host would need 12–27 ms per
   step just to issue launches (S2 `_launch_step` wall). Diagnostics-cadence or sync-cadence changes buy
   at most 2–3 %.
6. **Ions are already subcycled** (`ion_subcycle = 8`, Δt_i = 11–12 ps, Courant 0.17 µm per ion step);
   ion push + redeposit is ≈ 3–4 % of the step. Candidate (g) is implemented; nothing is left there.
7. **MPS implication.** Because ≈ 50 % of the step is latency-bound, four processes overlap to 1.54×
   aggregate. Removing the diagnostics reductions makes the remaining step *more* latency-dominated
   (Poisson ≈ 50 %), so the per-GPU aggregate under MPS should improve beyond the per-run speed-up.
8. Small items found in passing: `_compact` copies the whole `offsets` array to the host to read one
   element (2 × 18–35 MB per sync, ≈ 0.5 % of the step); `peak_node_sample` re-deposits all electron
   moments and downloads 5 maps per record for a witness statistic; `peak_window_sums` downloads 6 maps
   to locate one `argmax`; `spawn_kernel` and `mcc_kernel` each write a capacity-sized flag array every
   step.

## 7. Candidates evaluated against the profile

Speed-ups are estimated from §6.4 for **channel-33 (v4 grid, 3.31 ms solo)** and for the **33 µm
v2.1 plume box (360 × 1440; model ≈ 17 ms solo at ≈ 10 M electrons at W parity; README table 22.4)**;
plume-50 in brackets. Effort = implementation + tests + the verification protocol of §9. Risk classes:
**A** physics bitwise (diagnostic-only, round-off in float-atomic sums); **A′** physics bitwise,
diagnostic statistics change at a declared noise level; **C** numerics change within a declared
tolerance, statistical parity only; **D** model/statistics change (needs a physics band).

| # | candidate | profile evidence | channel-33 saving | plume-33 saving (plume-50) | effort | risk | rank |
|---|---|---|---|---|---|---|---|
| k | **Fold born KE / born p_z into `mcc_kernel` tile tallies; drop `energy_sum`/`momentum_sum`/`deferred_add` from the step; two-stage reduce for the remaining per-record sums** | 1.1–1.2 ms (35 %) / 2.0 ms | **−1.1 ms → 2.2 ms (×1.5)** | **−4 ms → ≈ 12 ms (×1.35)** (−2.0 ms) | 0.5–1 d | A | **1** |
| f1 | **Window moments every K = 5 steps** (runner passes `accumulate` on K-multiples; `diag_steps` already counts accumulated steps; frame validator uses cadence/K) | 0.65 / 1.3 ms | −0.5 ms → 1.7 ms (×1.9 cum.) | −2.4 ms → ≈ 10 ms (×1.65 cum.) | 1–2 d | A′ | **2a** |
| f2 | **Fork the diagnostic branch onto a second stream inside the step graph** (moments + `abs_axpy`s overlap the Poisson chain, which leaves the SMs ≈ 60 % idle) | Poisson 0.97 ms latency-bound; moments 0.65 ms independent of φ | hides 0.4–0.65 ms | hides ≈ 1–3 ms | 2–3 d | A | **2b** |
| f3 | Drop `d_w` (= `d_n_e` up to 2^-40 rounding); fuse moments into the charge deposit pass | 4 of 20 atomics; one 44 B pass | −0.15 ms | −0.6 ms | 1–2 d | A (maps identical to 1e-12) | 4 |
| a1 | **Warp-native geometric multigrid, fixed V-cycle count, graph-captured, residual verified at sync** (replaces the dense blocks) | Poisson bytes 0.22 / 3.5 ms; chain 182 / 722 launches | −0.1 … −0.3 ms (≈ 300 launches ≈ 182; only the bytes go) | **−5 ms** (−1.6 ms) | 2–3 wk | C | **3 (plume)** |
| a2 | **Exact partitioned block-Thomas (SPIKE-type, P = 2–4 partitions solved concurrently + an interface system)** — halves the dependent chain, same arithmetic class | 4.1 µs × launches | −0.35 … −0.45 ms | −1.5 ms (−1.0) | 1–2 wk | C (exact to round-off) | 3 (channel) |
| a3 | Separable FFT/DST solver + capacitance-matrix or CG correction for the masked rows (cuFFT via cupy) | would remove both terms | −0.8 ms | −6 ms | 3–5 wk, new dependency | C | 6 |
| a4 | Compress `G_i` to unknown × unknown sub-blocks on the L-shape | 55 % identity rows on plume | 0 | −1.5 ms (−0.3) | 3–4 d | A (same numbers) | 5 (plume only) |
| a5 | Jacobi-PCG (`WarpPoisson`, exists) | ≈ 470 iterations at 31 × 241; ≥ 1000 here, host loop | slower | slower | – | – | reject |
| d1 | Fuse MCC into the push (same particle index → same RNG draw), deposit-at-new-position into the push tail, born deposits into `spawn` | push+mcc+deposits ≈ 0.6 ms are ≈ 6 passes | −0.1 … −0.15 ms | −0.5 ms | 2–3 d | A (bitwise if operation order kept) | 4 |
| b | Sort particles by cell every K steps + tile/shared-memory deposition | gathers are already L2-resident; atomics are the cost (0.15 + 0.65 ms) | −0.3 … −0.5 ms only with tile deposition; **reorders particle indices → MCC RNG keyed on `p` → not bitwise** unless the RNG is re-keyed on a persistent id | −1.5 ms | 1–2 wk | D (or C after RNG re-key) | 7 |
| c | Mixed precision (float32 x, v; float64 accumulators) | bandwidth-bound share ≈ 10 % (push); atomics/latency dominate; fixed-point 2^-40 and the 1e-10 residual contract are float64-defined | ≤ −0.15 ms (5 %) | ≤ −0.7 ms | 2–3 wk + conservation study | D | 9 (not recommended) |
| e | Fewer host syncs / larger record cadence | host ≈ 2 %; sync = series = neutral-inventory cadence (closure discretisation) | ≤ −0.05 ms | ≤ −0.2 ms | 1 d | C (inventory cadence) | 8 |
| e′ | `offsets[used−1:used].numpy()` instead of the whole array; skip `peak_node_sample`'s 5-map download when the window gate is active; device-side `argmax` for the window gate | 0.5 % + record D2H | −0.03 ms | −0.15 ms | 0.5 d | A | 5 |
| g | Ion subcycling | already k = 8; ions ≈ 3–4 % | 0 | 0 | – | – | done |
| h | Fewer particles (larger W) | 0.97 ms per M e; W × 0.7 moved I_d +5.7 %, peak n_e −12 % (entangled with Δ) | linear in N_e | linear | – | D | not a code item — after (1)+(2)+(d1) the per-electron cost is ≈ 0.2 ms/M, which is what makes a *lower* W (the owed W-only refinement) affordable |
| i | Multi-GPU per run | step 1–7 ms, half latency-bound; the r-chain of the direct solve would cross GPUs; 2–10 M particles under-fill one H100 | < 1× | < 1× | – | – | against: use GPUs for independent runs (ladder, sweep, replicates) under MPS |

**Cumulative estimate** (model, ±25 %): channel-33: 3.31 → 2.2 (k) → 1.7 (f1 or f2) → 1.55 (d1, f3,
e′) → **≈ 1.2 ms with a2 (×2.7)**; 5.2 M steps: 4.8 h → ≈ 1.8 h solo. Plume-33 (v2.1 box): ≈ 17
→ 12 (k) → 10 (f1/f2) → **≈ 4–5 ms with a1 (×3.5–4)**; 3 transits: ≈ 47 h → ≈ 12 h. Channel-25 (v5,
≈ 4 M e): ≈ 5.8 → 3.8 (k) → 2.9 (f) → ≈ 2.2 ms (a2, d1). Under MPS-4 the aggregate per GPU should rise
from 1.54× to ≈ 2× because the remaining step is more latency-bound.

## 8. Ranked plan

1. **Born-ledger fold (k) + reduction shapes (S5) — first, alone, and shipped before anything else.**
   In `mcc_kernel`: two more per-thread scalars (`ke_born = kinetic_energy(ion v, m_i W)`,
   `pz_born = m_i W v_z,ion + m_e W v_z,sec`) reduced with the existing `wp.tile_atomic_add` pattern into
   `STATS_KE_BORN` / `STATS_PZ_BORN`; remove rows 23–28 from `_launch_step`; fuse the born charge and
   `d_ion` deposits into `spawn_kernel` (fixed-point int64 → bitwise `q_i`); for the per-record sums in
   `series_sample` launch `energy_sum`/`momentum_sum` with ≥ 262 144 threads and reduce with
   `reduce_stage` + `final_sum` (the pattern `WarpPoisson._reduce` already uses). Expected: channel-33
   3.31 → ≈ 2.2 ms, plume-50 7.2 → ≈ 5.2 ms, channel-25 ≈ 5.8 → ≈ 3.8 ms. Only two ledger scalars
   change (summation order → ≤ 1e-12 relative), the same class as the recorded MPS float-atomic
   tolerance. Effort 0.5–1 day including §9-A.
2. **Window diagnostics off the critical path.** Preferred order: (f2) fork/join the diagnostic branch
   in the step graph (Warp: capture on a second `wp.Stream` with `wp.record_event`/`wait_event`, the
   moments read x^n, v^n before the push, so they can start right after the deposits and overlap the
   sweeps); if the overlap does not hide ≥ 70 % of it, add (f1) K = 5 sampling with the frame validator
   and window classes taking `accumulated_steps` rather than `end − start`. Drop `d_w` (f3). Expected
   cumulative: channel-33 ≈ 1.7 ms, plume-50 ≈ 4.2 ms. Effort 2–4 days including §9-A/A′.
3. **Poisson, grid-dependent.** Channel grids: (a2) exact partitioned block-Thomas — P concurrent
   partitions of n_r/P rows each (one launch per row step, all partitions in the same launch), then a
   dense interface system of (P − 1) m unknowns whose inverse is precomputed on the host (P = 4:
   2163 × 2163 = 37 MB), then the back-substitution; chain 182 → ≈ 95 launches. Plume grids: (a1)
   geometric multigrid in Warp — red-black or damped-Jacobi smoothing on the masked FV operator,
   Galerkin coarsening of the conductances, a fixed schedule (e.g. 10 V-cycles, 5 levels, ≈ 300
   launches ≈ 1 ms) captured in the step graph, the same residual contract verified at every sync
   (`verify` raises if 1e-10 is not met — exactly how the direct solve is guarded today), host
   factorisation and 1–6 GB of inverse blocks gone. Do not adopt cuSPARSE/AmgX via cupy: their host
   convergence loops cannot be graph-captured (the reason `WarpPoisson` lost to the direct solve) and
   they add a second CUDA runtime to the MPS clients. Effort 1–2 weeks (a2), 2–3 weeks (a1); §9-C.
4. **Fusions (d1, f3) and the small D2H fixes (e′).** After 1–3 they are worth ≈ 0.2 ms on channel-33.
5. **Not now:** sorting (b) until a persistent-id RNG exists (otherwise every run becomes a
   different random realisation and the same-seed bitwise replay — the project's free regression check
   — is lost); mixed precision (c); multi-GPU (i); larger W (h).

Effort/risk summary for the top three: (1) 0.5–1 d, risk A; (2) 2–4 d, risk A/A′; (3) 1–3 wk, risk C.

## 9. Verification protocol before any change is used in a preregistered run

Common rules: the production step code is hash-bound by every preregistration
(`_bind_preregistration`, `code_identity`); running preregistered runs (mini-sweep at `291a9227`,
v5.1 at `efb9bb09`) keep the code they were sealed with — none of this applies retroactively. A new
preregistration names the code commit; its README states which class of change happened since the
reference plateau it compares against and cites the replay record below. Same-seed replays run on
the H100 as an MPS client (the recorded MPS-replay criterion: physics bitwise, float-atomic diagnostics
≤ 1e-6 — solo-vs-solo shows the same 2e-13 pattern).

**Class A (diagnostic-only; items k, S5, f2, f3, d1, e′, a4).**
1. Unit tests (tests/pic2d): graph-vs-direct bitwise test extended to the changed kernels (the v1.4
   pattern); CPU reference (`CPUBackend`) untouched → the existing warp-parity tests stay green; new
   test that the folded `ke_born_ions_j` / `pz_born` equal the strided-sum values to 1e-12 relative on
   a 2000-step tiny GPU case; for f2 a test that the two-stream graph replays the single-stream graph
   bitwise on particles, `phi`, `surface`, `stats` (float-atomic slots to 1e-12).
2. Same-seed 100 000-step replay of the v4 protocol from the seed on the H100 (≈ 5 min): `series.jsonl`
   equal on every key except the declared ledger keys and the quantities derived from them
   (`ke_born_ions_j`, `pz_born` → `interval_sources_j`, `interval_residual_j`, `born_momentum_rate_n`,
   `interval_ledger_residual_kg_m_s`; ≤ 1e-9 relative); `checkpoint-final` particle arrays, `phi`,
   `surface_charge` bitwise; window maps (`maps.npz`) allclose 1e-12.
3. Accepted-plateau replay: resume the v4 final checkpoint (`0d228ad2` results) for 400 000 steps with
   old and new code; series bitwise except the declared keys; energy-ledger residual identical.
4. Record the replay JSON under the change's docs; the next preregistration cites it.

**Class A′ (f1, K-step moment sampling): physics bitwise, statistics change.**
1.–3. as A (physics keys bitwise), plus: the window statistics (peak Δ/λ_D window, T_e maps, far-field
   window charge fraction, frame maps) compared K = 1 vs K = 5 over the 400 000-step resume of the v4
   plateau; require |Δ| of the gate statistics < 1 % and the map differences within the frame
   shot-noise band (§ renderer v0.2 event-count masks). Declare K and the resulting sample count per
   window in the protocol; re-evaluate the accepted plateaus' gate values under K (they must not flip:
   v4 window 2.15 vs soft 2.5 / hard π; base 3.17 vs π).

**Class C (a1 multigrid, a2 SPIKE, a3 FFT, e): numerics within tolerance, statistical parity.**
1. Solver unit tests: manufactured-solution order (existing `Poisson order 1.999` test), Gauss law
   with surface charge (existing), residual contract 1e-10 on all four production masks (channel-50/33/25,
   plume L-shape), SPD/consistency vs the host `Poisson2D` to 1e-12 on random right-hand sides.
2. One-step parity: `phi` from the new solver vs the block-Thomas on the same deposit, max |Δφ| ≤
   1e-9 V on every production grid (both meet the same residual bound; the difference is round-off).
3. Trajectory divergence record: 20 000-step same-seed replay; report the growth of the particle-state
   difference (expected chaotic separation after O(10³) steps) — documents why bitwise parity is
   impossible, not a gate.
4. **Statistical replay of the accepted v4 33 µm plateau** with the new solver: full protocol, same
   seed, to 3 transits; acceptance = every plateau quantity (I_d, I_beam, S, utilisation, n_g, peak n_e,
   T_e,peak, windowed residual, Δ/λ_D window) inside the recorded seed-b band (≤ 1.1 % on currents;
   −8 % on peak n_e; ≤ 1.1 % T_e) — the same table the v4 assessment used; ≈ 2 h with (1)+(2) applied.
   Replay the 50 µm base as well (cheap) for the cross-grid statement.
5. Only after 4 passes may a preregistration use the solver; its protocol records `poisson.method`
   and the replay record; results across solver versions are compared only with the replay caveat.

**Class D (b sorting without RNG re-key, c mixed precision, h larger W):** requires a physics band
study (seed pair + W pair) at the new numerics before any prereg, i.e. the same cost as a new
convergence-ladder point; not recommended in this cycle.

## 10. Appendix

### 10.1 Probe design (for reproduction)

`profile_pic2d_step.py --config {channel-33um|plume-v2.0-50um|channel-25um} --load production --steps 2000
--kernel-steps {2000|1000} --micro-steps 200 [--s5]`, run from `modern/` with `PYTHONPATH=src:.`,
`CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps`, 4 BLAS threads, in tmux (`run_probe.sh`). It monkey-patches
timers onto `WarpBackend._sync`, `_step_graph_launch`, `_launch_step`, `export_state` and
`Simulation._record` (perf_counter with a device sync before `_sync` / after `_record`), toggles
`backend.step_graph` and `backend.device_direct.graph` for S2, and launches the production reduction
kernels with alternative `dim`/`threads` arguments for S5. It writes to a scratch directory it removes.
Runtime per probe 59–102 s including construction (2–5 s factorisation, 3–6 s field/seed), warm-up
(3–8 s, 3 graph captures) and S4.

### 10.2 Cost model used for the extrapolations

`step ≈ 0.27 + 0.0041 · L + 0.296 · B + s · N_e` ms with L = 2 (n_r + 1) sweep launches, B = GB of
inverse blocks read per solve, N_e in millions, and s the per-electron slope split as: born-ledger sums
0.40 (+ 0.27 ms fixed in the 0.27 term), window moments 0.30, push 0.145, charge deposits + flag passes
0.07, MCC + scan + spawn 0.05 (Σ 0.97). "After" values set the born term to 0, moments to 0.06 (K = 5)
or hidden (overlap), and replace the Poisson term by 0.6 ms (SPIKE, channel) or ≈ 1.0–1.2 ms (MG).

### 10.3 GPU use on the box

Four probe processes, 60 + 102 + 60 + 69 = 291 s wall as an extra MPS client (≈ 4.9 GPU-minutes of
client time; the actual SM share was smaller), 1.8–3.3 GB device memory each, released on exit;
`dmesg | grep -i xid` empty before and after; MPS `server.log` shows clean client connect/exit only.
No process on the box was signalled. Scratch checkpoints/frames were deleted; the JSON/log records
stay in `/lambda/nfs/h100-files/cft/perf-audit/` (≈ 80 kB).

## 11. Measured outcome of recommendations (1) and (2): model v2.0.5

Implemented 2026-09-04 in `modern/src/cft_revival/pic2d/warp_backend.py` / `simulation.py` /
`frames.py` and the shared runner (`f80c6441`, verified on the box as `a156fd84` before the rebase; spec entry `pic2d-model-v2.0.json#performance_v2_0_5`,
tests `tests/pic2d/test_pic2d_v205_performance.py`). What changed, in the terms of §3.1:

* rows 23–28 (`energy_sum` / 2 × `momentum_sum` / 3 × `deferred_add`) are gone: `mcc_kernel` tile-reduces
  `ke_born` and `pz_born` next to its seven tallies (the values it already holds in registers);
* rows 29–30 (born `deposit_fixed`, `deposit_unit`) are fused into `spawn_kernel` (int64 fixed point →
  `acc_i` bitwise the old separate deposit); `deposit_unit_kernel`, `deferred_add_kernel` and the
  `born_r/z/flag` arrays are deleted. Per step: −5 launches, −5 capacity-sized flag passes;
* rows 14–16 (`abs_axpy` × 2, `deposit_moment`) run on a second stream inside the captured step, forked
  after row 6 and joined before row 17 (CUDA graph fork/join through `wp.Event`; graph-capturable in
  Warp 1.14 — no fallback was needed); row 16 is additionally sampled every K accumulated steps
  (`PIC2DConfig.moment_sample_interval`, default 1; K = 5 recommended; `moment_samples` travels with the
  window sums; K ≠ 1 enters `config_sha256`). Row 13 (`d_phi += phi`) stays after the solve.

### 11.1 Measured, contended (the only measurement possible while four preregistered runs own the GPU)

Identical-contention A/B: old (`ce1d96cb`, step code = `a529b457` = this audit's), new K = 1 and new K = 5
launched together as three extra MPS clients next to the four production runs (7 clients), production
load through `tools/cloud/bench_gpu_concurrency` `CONFIGS`, the v4 `_time_steps` over 4000 (channel) /
2000 (plume) graph steps:

| load | old ms/step | new K = 1 | new K = 5 | contended speed-up |
|---|---|---|---|---|
| channel-33 (2.02 M e, 2.26 M i) | 13.48 | 11.12 | 11.12 | **×1.21** |
| plume-v2.0-50 (4.15 M e, 4.35 M i) | 26.00 | 23.81 | 23.82 | **×1.09** |
| channel-33, sequential probes, 5 clients | 8.15 | 6.40 | 6.71 | ×1.21–1.27 (background varies) |
| channel-33 replay from the seed (0.55 M e, 7 clients, 40 000 steps) | 8.84 | 7.21 | 7.21 | ×1.23 |
| channel-33 resume at the plateau load (1.94 M e, 8 clients, 6 000 steps) | 11.11 | 9.03 | – | ×1.23 |

Per-kernel table, channel-33 production load, direct launches (graph off), CUDA-event timing over 300
steps, 5 clients (compare §6.2; contended numbers, same inflation pattern):

| kernel (calls/step) | old ms/step | new K = 1 | new K = 5 |
|---|---|---|---|
| `block_backward` + `block_forward` (182) | 7.17 | 7.10 | 8.08 (background) |
| `deferred_add_kernel` (3) | **1.04** | – | – |
| `momentum_sum_kernel` (2) | **0.79** | – | – |
| `energy_sum_kernel` (1) | **0.38** | – | – |
| `deposit_unit_kernel` (1) + born `deposit_fixed` (1) | 0.09 + ≈ 0.09 | – | – |
| `deposit_moment_kernel` (1 → 0.21) | 0.69 | 0.69 | **0.15** |
| `push_kernel` (1.12) | 0.38 | 0.37 | 0.39 |
| `mcc_kernel` (1) | 0.14 | 0.24 (+ KE of the born ion, 2 more tile sums) | 0.23 |
| `deposit_fixed_kernel` (2.13 → 1.13) | 0.28 | 0.20 | 0.21 |
| `spawn_kernel` (1) | 0.07 | 0.06 (+ born deposits) | 0.06 |
| memcpy DtoH (0.16) | 0.57 | 0.42 | 0.52 |
| kernel sum | 12.20 | 9.78 | 10.45 |

The 2.3 ms of removed kernels show up as −1.75 … −2.4 ms in the contended graph step; K = 5 removes a
further 0.55 ms of direct-launch kernel time but is invisible in the contended graph step (noise ±5 %):
under 7-client MPS the step is set by the 182 / 482 dependent sweep launches waiting for scheduling slots,
and the saturated SMs leave nothing for the fork to overlap into. Both effects are solo effects.

### 11.2 Fitted solo (model, ±25 %; a one-minute solo probe when the GPU is free settles it)

§10.2 cost model with the removed kernels' solo attribution (born-ledger sums 0.40 ms per M e + the
0.20–0.35 ms `deferred_add` chain; flag passes 0.03) and the additions (`mcc` +0.03–0.05, `spawn` +0.01):

| load | audit solo anchor | (1) alone | (1)+(2) fork, ≥ 70 % overlap | (1)+(2) K = 5 |
|---|---|---|---|---|
| channel-33 | 3.31 ms | ≈ 2.2–2.3 (×1.45–1.5) | ≈ 1.7 (×1.9) | ≈ 1.7–1.8 (×1.85) |
| plume-v2.0-50 | 7.20 ms | ≈ 5.0–5.2 (×1.4) | ≈ 4.1–4.3 (×1.7) | ≈ 4.2 (×1.7) |

### 11.3 Verification (Class A / A′, executed on the H100 as an extra MPS client; budget-scaled)

Budget: ≤ 30 GPU-minutes, every probe ≤ 5–6 min, four preregistered runs untouched. The §9 lengths were
scaled to the budget: 40 000-step same-seed replays from the seed (not 100 000) and a 6 000-step resume of
an anchored production checkpoint (not 400 000); every record of every run was compared.

1. Graph vs direct (K = 1 and 5, 200 steps, `test_forked_step_graph_replays_the_direct_launches`): particles,
   φ, surface charge, counts bitwise; born/momentum ledger and window moment sums to 1e-12. Passed on cuda:0.
2. Old vs new, channel-33 v4 protocol from the seed, 40 000 steps (200 records, 2 frames, 1 checkpoint):
   `checkpoint-final` particle arrays / φ / surface charge bitwise; every series physics key (counts, I_d,
   I_beam, S, n_g, φ statistics, surface charge, kinetic energies) bitwise on 200/200 records; final counts
   545 869 e / 618 323 i in both; `maps.npz` n_e / n_i / φ / ionisation / fluxes bitwise (18/19 keys — the
   19th is the new `moment_samples` array). Ledger scalars: `ke_born_ions_j` 3.9e-16, `pz_born` 8.7e-16
   relative (declared ≤ 1e-12); derived `interval_sources_j` 2.3e-14, `interval_residual_j` 1.5e-13;
   float-atomic diagnostics ≤ 3.1e-15 (the recorded MPS-replay pattern).
3. K = 1 vs K = 5 on the same 40 000 steps: physics and per-step maps bitwise; `moment_samples` 40 000 vs
   8 000 (sample-count ratio 0.2000); peak-Debye window statistic over 200 records: `cells_per_debye`
   relative difference median 1.7e-5, max 1.6e-3; `t_e_peak` median 3.4e-5, max 3.1e-3; same peak node
   200/200; `n_e_peak` identical; `t_e_ev` map on nodes with mean occupancy ≥ 32 in both (185 nodes) median
   1.9e-5, p95 4.8e-5, max 7.3e-5; occupancy ≥ 4 (35 467 nodes) median 6.0e-5, p95 5.5e-4. An electron
   sits on a 33 µm node for ≈ 24 steps, so every 5th step keeps almost all the independent information.
4. Anchored resume: `checkpoint-latest` of the running mini-sweep reference design (channel-33 production run
   at `291a9227`, step 4 600 000 = 6.44 µs, 1.94 M e + 1.96 M i, I_d 3.81 mA; post-`0ac8d9b8` field
   anchor; read-only copy), 6 000 steps with old and new code (`require_same_code=False`, field replay
   bitwise): `checkpoint-final.npz` byte-identical (sha256 equal, cumulative ledger included); series physics
   bitwise; only `peak_node` float-atomic statistics differ (≤ 1.2e-15).
5. cpu / warp-cpu / cuda parity: the new tests (K = 3 vs 1 bitwise on cpu and warp-cpu; sampled sums exactly
   the sampled per-step moments; born tallies vs particle sums 1e-12; fused deposit bitwise) plus the
   unchanged parity suite. Box: 8 CUDA modules on cuda:0 79 passed / 71 s; whole `tests/pic2d` CPU-only 270
   passed, 12 skipped (9 CUDA-only = run in the CUDA pass, 3 node.js-absent). Local (Windows, CUDA hidden):
   273 passed, 9 skipped.

GPU use: 1698 s of client wall time (28.3 client-minutes; SM share ≈ 1/7–1/8 per probe): timing 94 s,
concurrent A/B 185 + 263 s, three concurrent 40k replays 949 s, two concurrent resumes 136 s, CUDA tests
71 s. No process signalled; `dmesg` shows no new Xid; MPS `server.log` clean apart from benign
"Invalid CUDA_VISIBLE_DEVICES -1" rejections from the CPU-only test pass. Probe script, JSON records and
logs: `/lambda/nfs/h100-files/cft/perf1/` (not in the repository).

## 12. poisson_gmg_v1 — the geometric multigrid field solve, built and measured (2026-09-04, follow-up to §8 item 3)

Item (a1) of §7/§8 exists: `poisson.method = "device-mg"` (`cft_revival.pic2d.poisson_mg` = hierarchy + numpy
reference, `cft_revival.pic2d.warp_poisson_mg` = Warp kernels + `WarpPoissonMG`; hook = one branch in
`WarpBackend.__init__` behind the `WarpBlockThomas` interface, `Poisson2D` dispatch for the CPU backend,
`numerics.poisson` object in the shared runner). Spec entry `poisson_gmg_v1` in `spec/pic2d/pic2d-model-v2.0.json`.
Everything below was measured on the box **as an extra MPS client beside four production runs** (mini-sweep ref/009,
ext-val channel-20um, ss-v5 25 µm; GPU at 100 %), i.e. **contended**, and the solve is latency-bound, so the
absolute ms are a share of the GPU, not the solo cost; the audit's solo cost model (§10.2) is used for the solo
statement and is flagged as such. 38.5 GPU-minutes of client time in total (tests 0.4 min, six timing probes 6.3 min,
six replay sessions 31.9 min); scratch in `/lambda/nfs/h100-files/cft/perf2/` (5.5 MB of JSON/JSONL/logs kept,
checkpoints deleted, worktree removed); no process signalled, `dmesg | grep -i xid` empty, MPS `server.log` clean.

### 12.1 Design (what was built)

* **Unknowns/operator**: the node mesh of `mesh.py`; unknowns = plasma nodes minus Dirichlet (anode, exit/far field,
  grounded body conductor); operator `A_uu` = conductance graph Laplacian with the `2πr` finite-volume weighting,
  homogeneous Neumann into the dielectric solids, Dirichlet couplings moved to the right-hand side exactly as the
  block-Thomas paths do (`source_kernel`, `apply_dirichlet_kernel` are shared). No permittivity map exists in the
  model (the dielectric is a perfect insulator with surface charge), so none is coarsened.
* **Coarsening**: vertex-centred by two in r and z; an axis with an odd number of cells keeps its last node
  (90 × 720 → 46 × 361 → 24 × 181 → 13 × 91). The coarse unknown set is the image of the fine one, so the bore,
  the stair-stepped cone, the exit lip, the dielectric flange and the electrodes are represented on every level by
  the *operator*, not by a re-classified mask.
* **Transfers**: operator-dependent (Alcouffe–Dendy "black-box") interpolation from the level's own stencil —
  a fine node between two coarse nodes takes the collapsed-stencil weights (conductance-proportional on level 0),
  a fine cell-centre node the fine equation through the four corners; a coupling to a Dirichlet node is absent
  from `A_uu` while its conductance stays in the diagonal (weights decay towards electrodes), a solid neighbour has
  no conductance (weights reflect across the walls). **One correction was necessary**: at a concave corner of the
  stair-stepped cone a parent can be a solid node while the 9-point coarse stencil has non-zero diagonal entries
  towards its side; lumping them into a node that does not exist lost that mass and left a **0.45/cycle slow mode
  on the cone of channel-33 and plume-50** (channel-50/25 were unaffected because their cone corner stays on a
  coarse node down to the coarsest level). The mass now goes to the surviving parent, constants are exactly
  preserved on every pure-Neumann row of every level (pinned by a test), and the contraction is uniform.
  Restriction = transpose.
* **Coarse operators**: Galerkin `P^T A P` (a symmetric 9-point stencil; the assembled symmetry is checked to
  1e-11 as the self-test of the construction), positive diagonals asserted; recursion stops at ≤ 1024 unknowns;
  the coarsest operator is inverted densely once on the host (307–752 unknowns on the production grids:
  0.75–4.5 MB, ms to build).
* **Cycle**: V(2,2), damped Jacobi ω = 0.8 (ω = 0.9 develops a 0.39 mode on the 9-point coarse operators; 0.7 is
  slower at 0.175; V(1,1) 0.35; V(3,3) 0.073 but more launches), fused residual-and-restrict kernel, **fixed
  count of 14 cycles**, warm start from the previous potential; one fixed kernel sequence
  `12 + 14 × ((levels − 1) × 6 + 1)` launches (278 channel-33, 362 plume-50, 446 plume-33) captured inside the step
  graph. Convergence is *verified*, not iterated: the true residual with the mesh conductances (`matvec_kernel`,
  independent of the multigrid's arrays) is computed inside the graph every step and a running maximum of the
  contract ratio `|r|² / max(abs², rel² |rhs|²)` is kept over the sync interval; `verify()` (the existing hook at
  every host sync) raises `PIC2DConvergenceError` if the last residual or the interval maximum misses the contract.
  **Decision: fail-closed stop, no per-step fall-back** — a fall-back would need a host synchronisation per step
  and the block-Thomas graph is not resident when the multigrid is selected; a missed contract is a configuration
  error (too few cycles for the source) fixed by resuming from the last checkpoint with a larger `mg_cycles`,
  exactly how a gate stop is handled. The fail-closed path fired once during this work (12 cycles on the 33 µm
  plume box, below) and did what it should.
* **Identity**: `PoissonConfig2D.to_dict()` carries a `multigrid` block only for `device-mg`; every recorded
  `config_sha256` is unchanged, and a protocol naming the multigrid is a different identity (a checkpoint never
  crosses solvers silently; pinned by tests on the plume-v1 protocol).

### 12.2 Convergence (host reference, numpy; zero start; relative true residual after k V(2,2) cycles)

| grid | unknowns | levels (coarsest unknowns) | source | factor/cycle | after 12 | after 14 |
|---|---|---|---|---|---|---|
| channel-50 60 × 480 | 20 779 | 4 (366) | random | 0.127 | 1.6e-11 | 2.7e-13 |
| channel-33 90 × 720 | 46 469 | 4 (752) | **v4 plateau maps** (`0d228ad2`) | 0.127–0.130 | 1.7e-11 | 2.8e-13 |
| channel-25 120 × 960 | 82 359 | 5 (366) | random | 0.127 | 1.6e-11 | 2.7e-13 |
| plume-v2.0-50 240 × 720 | 78 228 | 5 (307) | **attempt-7 / attempt-8 maps** | 0.127–0.133 | 1.7e-11 | 3.0e-13 |
| plume-v2.1-50 240 × 960 | 135 828 | 5 (532) | smooth synthetic (sheath band at the wall) | 0.14–0.17 | 9.5e-11 | 1.4e-11 |
| plume-v2.1-33 360 × 1440 | 305 442 | 6 (313) | smooth synthetic | 0.14–0.18 | **2.8e-10 (fails)** | 8.4e-12 |
| plume-v2.1-33 | | | random | 0.127 | 1.6e-11 | 2.7e-13 |

The residual cone mode (0.14–0.18 with a wall-heavy smooth source on the two largest grids) is what set the default
at 14 cycles (≥ 12× margin on every grid from a zero start; ≥ 300× on the production grids with the real charge).
Warm start from the window-averaged `phi_v` of the maps (a pessimistic warm start) begins one decade lower.
Manufactured solution: order 1.9+ (the `test_pic2d_mesh_poisson` solution); Dirichlet nodes exact; Gauss law with
surface charge at the scale-aware 1e-9 bound; block-Thomas parity ≤ 1e-8 V from a zero start and ≤ 1e-9 V warm-started
(channel/plume test grids). Tests: `tests/pic2d/test_pic2d_poisson_mg.py` — 19 CPU (numpy + warp-cpu) + 3 CUDA
(graph replay bitwise over 100 steps with ionisation/injection, cpu/cuda parity), 25 passed on the box.

### 12.3 Timing and memory on the H100 (contended: extra MPS client beside four production runs)

Solve alone (both solvers bound to the same device charge arrays, 100–200 graph launches, warm start; the real
window-averaged charge where maps exist):

| grid | BT ms/solve (launches) | GMG ms/solve (launches, cycles) | BT inverse blocks / device | GMG device (+host) | GMG build vs BT factorisation | residual/tol BT / GMG warm / GMG zero |
|---|---|---|---|---|---|---|
| channel-33 | 3.97 (184) | 12.1 (240, 12) | 0.38 GB / +354 MiB | +2 MiB (23 MB arrays, 4.5 MB host) | 0.18 s vs 1.05 s | 8.4e-4 / 3.9e-4 / 0.17 |
| plume-v2.0-50 | 10.3 (484) | 12.0 (312, 12) | 1.00 GB / +962 MiB | +34 MiB (50 MB) | 0.31 s vs 2.6 s | 9.9e-4 / 4.9e-4 / 0.17 |
| plume-v2.1-50 (240 × 960) | 26.0 (484) | 25.4 (312, 12) | 1.78 GB / +1.73 GiB | +66 MiB (68 MB) | 0.48 s vs 5.8 s | 2.0e-2 / 9.9e-3 / 0.94 |
| plume-v2.1-33 (360 × 1440) | 22–41 (724; three probes) | 19.0 (446, 14) | **6.00 GB / +5.79 GiB** | +130 MiB (149 MB) | 1.2 s vs 18–24 s | 2.6e-2 / 1.2e-2 / 0.084 |

Production step (bench protocol at production load, `_time_steps`, 1000 steps after 200 warm-up, step graph):

| grid | electrons | BT ms/step (GPU MiB) | GMG ms/step (GPU MiB) | ratio |
|---|---|---|---|---|
| channel-33 | 2.26 M | 8.45 (1866) | 19.2 (1514) | **2.3× slower** |
| plume-v2.0-50 | 4.35 M | 18.4 (3308) | 20.2 (2380) | 1.1× slower |
| plume-v2.1-50 | 4.35 M | 26.6 (4076) | 21.7 (2412) | 1.23× faster |
| plume-v2.1-33 | 9.80 M | 40.7 (10 382) | 37.8 (4750) | 1.08× faster |

Reading these honestly: under MPS contention every one of the ~300–450 small dependent kernels waits for an SM
share (38–81 µs per launch measured, against ≈ 21 µs for the fat block-Thomas row kernels and the audit's 2.5–4 µs
solo), so the contended numbers penalise the latency-bound multigrid far more than the bandwidth-bound direct solve;
the contention itself varied by 2× between probes minutes apart (block-Thomas 22 → 41 ms on the same grid).
**Solo estimate (audit cost model §10.2, `4.1 µs × launches + 0.30 ms/GB`)**: channel-33 GMG ≈ 1.1 ms vs BT 0.97 ms
(as predicted in §7: not faster on channel grids — 278 launches against 184 and no bytes to save); plume-50
≈ 1.5 ms vs 2.6 ms (≈ 1.7×; the audit target 2.6 → ≈ 1.0 needs ≈ 30 % fewer launches, see 12.5); plume-v2.1-33
≈ 1.8 ms vs ≈ 6.5 ms (≈ 3.5×), i.e. the 17 → ≈ 12 ms/step of the §7 table for that box. **These solo figures are
model values until a solo probe runs** (the box has had four production clients since the multigrid existed; a
solo probe is a 3-minute job when a slot is free). What is measured beyond doubt: the multigrid removes the
inverse blocks (0.38–6.0 GB device, the same on the host) and the host factorisation (1–24 s per launch/resume,
5–12 min on the Windows PC), and its device footprint is 23–149 MB; on the two v2.1 boxes it is already faster
than the direct solve even under contention.

### 12.4 Class C verification (§9-C items 1–3, and a shortened item 4 proxy)

1. Solver unit tests: see 12.2 (manufactured order, Gauss law, residual contract on the production masks with the real
   maps, host/device consistency).
2. **One-step φ parity on real ρ** (warm start from the block-Thomas potential, the production situation):
   max |Δφ| **3.8e-10 V** channel-33 (v4 maps), **8.5e-10 V** plume-50 (attempt-7 maps) — both ≤ 1e-9 V at the
   300 V scale; 5.0e-9 V on 240 × 960 and 4.6e-8 V on 360 × 1440 (the contract's 1e-10 |rhs| bounds the potential
   error more loosely as the unknown count grows — the block-Thomas itself sits at 2–3 % of the tolerance there —
   so the 1e-9 V figure is a channel/plume-50 statement, not a grid-independent one). Zero-start solves: 4.8e-9 V on
   both production grids (contract-level).
3. **Trajectory divergence** (same-seed 60 000-step plume-v2.0-50 replay, below): every series record identical
   until step 6 400 (|ΔW_field|/W 4e-14 … 5e-13 up to step 2 000, 3e-7 at 5 000); the first divergent record is at
   step 6 600 (N_e 246 720 vs 246 723, I_d 3 %), |ΔW_field|/W = 1.4e-2 by step 10 000 and 4e-3 … 9e-3 thereafter —
   the expected chaotic separation after O(5 × 10³) steps from round-off-different fields; bitwise parity is
   impossible by construction, statistical parity is the criterion.
4. **Same-seed replay, plume-v2.0-50 production protocol** (`experiments/pic2d_cft_plume_v1/protocol.json`, seed
   20260903, v2.0.x closure and gates, frames off, checkpoint cadence 20 000 so every ≤ 6-minute MPS session ends on
   a checkpoint the next resumes from; **60 000 steps = 0.09 µs per solver, three sessions each** — the requested
   100 000 did not fit the 45-GPU-minute budget at the contended 13.2 (BT) / 17.5 (GMG) ms/step). Pass criteria
   declared before running: trailing-half means of I_d, S, N_e, n_g and peak n_e within ± 5 % (the particle band);
   integer tallies of the two runs within the same band; no gate firing in one run only; the multigrid's verified
   contract ratio < 1 at every sync. Result (trailing 150 records, steps 30 200–60 000; sd = record-to-record scatter
   of the block-Thomas run):

   | quantity | block-Thomas | multigrid | Δ rel | BT sd/mean |
   |---|---|---|---|---|
   | I_d (A) | 2.376e-3 | 2.400e-3 | +1.0 % | 14.5 % |
   | S (1/s) | 2.785e16 | 2.816e16 | +1.1 % | 9.2 % |
   | N_e | 257 320 | 257 320 | −0.001 % | 1.1 % |
   | N_i | 284 000 | 284 040 | +0.015 % | 0.24 % |
   | n_g (m⁻³) | 5.4433e19 | 5.4372e19 | −0.11 % | 2.4 % |
   | peak n_e, window-gate statistic (m⁻³) | 1.5636e17 | 1.5487e17 | −0.95 % | 14.0 % |
   | window gate Δ/λ_D | 0.665 | 0.637 | −4.2 % | 11.6 % |
   | wall ion current (A) | 3.918e-3 | 3.920e-3 | +0.06 % | 15.6 % |
   | field energy (J) | 6.691e-9 | 6.685e-9 | −0.08 % | 2.7 % |
   | electron kinetic energy (J) | 3.539e-8 | 3.552e-8 | +0.36 % | 5.4 % |
   | thrust total (N) | 3.302e-6 | 3.298e-6 | −0.11 % | 2.7 % |
   | cumulative ionisations / wall ions | 25 153 / 25 844 | 25 205 / 25 852 | +0.21 % / +0.03 % | — |

   Every declared quantity is inside the ± 5 % band and inside one standard deviation of its own record-to-record
   scatter; both runs stopped on `wall_clock_budget_reached` at 60 000 with no gate armed or fired (ignition gates
   arm at 0.75 µs). **The multigrid's interval-worst contract ratio over the 300 sync intervals was
   8.2e-5** (median 6.6e-5): a warm-started production solve ends ≈ 1.2 × 10⁴ below the contract at 14 cycles —
   the production margin the fixed count was asked to demonstrate; 11 cycles would still keep > 100× (never fewer:
   the first solve of a run starts from the vacuum potential of the bind warm-up, and a cold zero start needs ≥ 12).
   Records: `/lambda/nfs/h100-files/cft/perf2/replay-device-{direct,mg}/{series,status}.jsonl, summary.json`,
   `replay-device-mg/mg-worst-ratio.jsonl`. **Class C item 4 proper — the full-protocol replay of the accepted v4
   33 µm plateau under the multigrid to 3 transits within the seed-b band — is a preregistered campaign and has not
   run**; nothing here admits the multigrid to a preregistration.

### 12.5 What remains before a plume run may use the multigrid

1. A **solo** timing probe (3 min when the box has a free slot) to replace the model values of 12.3 with measured
   ms/solve and ms/step for channel-33, plume-50 and the two v2.1 boxes.
2. The **v4 33 µm plateau replay campaign** (§9-C item 4): preregistered, same seed, to 3 transits, acceptance =
   every plateau quantity inside the recorded seed-b band; ≈ 2 h with the born-ledger and window-diagnostic items of
   §8 applied, ≈ 5 h without. Only after it passes may a protocol name `numerics.poisson = {"method": "device-mg"}`.
3. Launch count (the only lever left for the solve): fuse prolongation with the first post-smoothing sweep and the
   two pre-smoothing sweeps into one kernel (each removes `levels − 1` launches per cycle: 362 → ≈ 250 on plume-50,
   the audit's 1.0 ms target); measure the cone mode with a wall-heavy real charge (a v2.1 plume run) and, if it
   persists, trade the 14 cycles for 12 with a red-black level-0 smoother.
4. The v2.1 33 µm plume box (`pic2d_cft_plume_v2_1`, 360 × 1440) is the run this solver was built for: its 6 GB of
   inverse blocks and 20-second factorisations are gone and the step is already faster under contention; its
   cost table should be re-issued after item 1.

## 13. Model v2.0.6 — energy-ledger correction and the accumulated-particle-step Debye floor (2026-09-05; not a performance change)

Recorded here because §§3, 5 and 11 describe the ledger's cost and its bitwise contract, and because the number the
residual-power gate reads has changed. Spec entries `pic2d-model-v2.0.json#gates_v2_0.energy_ledger_correction_v2_0_6`,
`gate_recalibration_v2_0_6`, `peak_debye_gate_accumulated_floor_v2_0_6`.

* **What was wrong.** Both backends added the MCC tally's per-macro-event threshold energy `(n_exc E_exc + n_ion E_ion) e`
  to `cumulative["inelastic_loss_j"]` without the macro weight `W` (every other ledger term carries `W`). The recorded
  interval residual was therefore `H − L_inel` with `H = field work + dU − electrode work` the true numerical energy
  creation and `L_inel` the W-scaled inelastic power: every residual recorded up to v2.0.5 read too negative by the
  inelastic power (7–14 % of the electrode work at the accepted plateaus). Found by the external-validation launch-1
  diagnosis (036bd679); verified by the particle-side identity closing to round-off in both backends
  (`tests/pic2d/test_pic2d_v206_ledger.py`).
* **Cost.** One multiply and one dictionary add in the host flush (`WarpBackend.flush`, after the graph) and in the CPU
  step; no kernel, no launch, no device array changed. The step graph and the physics state are bitwise those of v2.0.5;
  the §11 timings stand. The series record gains one scalar (`interval_inelastic_loss_j`).
* **Post hoc.** `python -m cft_revival.pic2d.ledger_recompute <results-dir>` rebuilds the corrected windowed and
  cumulative residual from the recorded `series.npz` (`H` per record; sidecar `ledger-corrected.json`; recorded files
  untouched). Corrected end-state windows: v2 base / seed-b / W×0.7 (50 µm) +13.0 / +11.1 / +7.2 % — the 50 µm
  plateaus were heating; ss-v4 (33 µm) +2.46 % (acceptance (b) < +2 % recorded PASS → corrected FAIL); 047 +0.9 %;
  056 L1 +0.6 %; v5 L1 (25 µm, 0.8 µs) +0.3 %; attempt 8 +67 %; ext-val L1 +62 % (5 % crossed at 0.34 µs, recorded 0.73).
* **Gate.** The 5 % one-sided windowed residual-power stop is kept on the corrected statistic (2× margin over the accepted
  33 µm maxima; catches attempt 8 at 0.66 µs and the ext-val avalanche at 0.34 µs). Running preregistered campaigns
  (ss-v5, sweep 056 L2, reference) execute pre-v2.0.6 code: assess their (b) on the sidecar.
* **Peak-Debye floor.** `PeakDebyeGateConfig.min_accumulated_macro_particle_steps_at_peak` (64 000 = 32 samples ×
  2000 steps, the v2.0.2 plume-gate figure) gates the densest node with that much accumulated electron weight over the
  window (the ext-val axis column at 0.76 macro-electrons per step had ~300 000 macro-electron-steps and was invisible to
  the ≥ 32 mean-occupancy floor); the v2.0.3 reading stays as the witness. Enters `config_sha256` only when declared; the
  v4 plateau reads 2.154 under both floors, attempt 8's final window 3.61 (trips). Host-side only (the same window sums
  as v2.0.3; one extra comparison per node at the record sync): no step cost.
