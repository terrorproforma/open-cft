# `cft_revival.pic2d` performance audit (H100, 2026-09-04)

Read-only audit of the production PIC-MCC step (`modern/src/cft_revival/pic2d/warp_backend.py`,
`simulation.py`, `poisson.py`, `mcc.py`, the shared runner `experiments/pic2d_cft_steady_state_v1/run.py`
that v4 / v5 / the mini-sweep / the external validation reuse) at commit `a0235676` (the step code is
byte-identical at `a529b457`, the box head during the probes). Nothing in the production step was changed;
this document is the only deliverable. All numbers are from the Lambda H100 box
(`ubuntu@68.209.75.2`, H100 80GB HBM3, driver 580.105.08, Warp 1.14.0 cu12.9) and from the recorded solo
benchmarks in `/lambda/nfs/h100-files/cft/bench*` (`tools/cloud/bench_gpu_concurrency.py`, same step code).

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
