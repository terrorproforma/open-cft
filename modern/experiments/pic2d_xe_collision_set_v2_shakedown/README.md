# `pic2d_xe_collision_set_v2_shakedown` - model v2.3.0 shakedown on the ss-v4 33 µm protocol

**NON-EVIDENTIARY. Not preregistered. No result.** The coordinator schedules the R3 comparison runs
(`modern/spec/pic2d/pic2d-model-v2.3.json` → `xe_collision_set_v2.predeclared_expectations`).

`protocol.json` = the preregistered `pic2d_cft_steady_state_v4` protocol (90 × 720, 33.3 µm, 1.4 ps, W 26 666.7,
seed 20260903, v1.3 closure, v2.0.3 gates) with `operating_point.collision_set = {name: xe_collision_set_v2,
ion_neutral: true}` and its identity / status fields changed. `run.py shakedown` calls the v4 shakedown
(shrunk cadences, 100 000 steps, finalize + assess) and adds the collision readings; `run.py compare-iedf` puts
the exit-plane IEDF shape next to a recorded run's.

## Record (2026-09-04 17:04–17:14 UTC, Lambda H100 as the 5th CUDA-MPS client, code 6defd5ed = the R3 tree
before its rebase onto 057841cf; `shakedown.json`, `iedf-comparison.json`)

* 100 000 steps (0.140 µs = 0.058 ion transits), `target_steps_reached`, 4.52 ms/step contended, 588 s;
  finalize + assess ran (verdict `no_plateau`, as any 0.14 µs run must); the peak-Debye window was enforced in
  301/500 records (max 0.59 cells/λ_D), the residual window complete in 280 records at +0.09 % of the electrode
  work on the W-corrected ledger with the new `ion_neutral_loss_j` sink booked; 0 ceiling violations, 0 unresolved
  fast-neutral marches; CUDA graph active.
* Early collision readings (trailing half, 70–140 ns; the discharge is in its seed transient, ions still slow):
  CEX 9.4e14 /s, MEX 4.9e14 /s against S = 1.6e16 /s (CEX/S 5.8 %); the null-collision operator accepted
  7 477 of 111 022 candidates (6.7 %); cumulative 4 101 CEX / 3 376 MEX events over 83 452 ionisations;
  0.31 CEX events per exit-plane ion. Implied ⟨σ v_rel⟩ = 1.3e-15 m³/s → v_rel ≈ 1.6 km/s (ions of ~2 eV), i.e.
  the seed population; the 15–30 % beam-exchange expectation is a plateau statement and cannot be read here.
* Fast-neutral fates so far: 324 exit through the aperture (inventory sink 6.8e13 /s = 0.4 % of S), 3 103 hit
  a wall, 674 below the thermal threshold - the slow, near-wall ions of the transient; `pz_fast_neutral_exit`
  7.5e-15 kg m/s cumulative (fast-neutral thrust term live), `ke_fast_neutral_exit_j` 2.3e-11 J.
* Excitation levels: 16 366 / 14 740 / 29 944 / 13 449 (22 / 20 / 40 / 18 %) of 74 499 excitations; the
  inelastic ledger sums per level × W (7.44e-9 J cumulative).
* Exit-plane IEDF at 0.14 µs (1 789 macro-ions in the last 40 000-step window): mean 18 eV, 92 % below 30 eV -
  the seed-transient population; the recorded ss-v4 plateau IEDF (322 364 macro-ions; mean 144 eV, 7.4 % below
  30 eV, 21 % below 75 eV) is listed alongside as the shape reference the R3 runs will be compared with, not as a
  before/after at equal time.
* GPU use of the whole R3 box session: 1.5 min of CUDA tests (29 collision-set tests + 30 ledger / parity /
  hook regressions) + 9.8 min shakedown = 11.3 GPU-minutes as an extra MPS client.
