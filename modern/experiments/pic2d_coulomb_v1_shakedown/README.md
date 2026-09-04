# `pic2d_coulomb_v1_shakedown` - model v2.4.0 shakedown on the ss-v4 33 um protocol

**NON-EVIDENTIARY. Not preregistered. No result.** The coordinator schedules the R4 comparison runs
(`modern/spec/pic2d/pic2d-model-v2.4.json` -> `coulomb_v1.predeclared_expectations`).

`protocol.json` = the R3 shakedown protocol (`pic2d_xe_collision_set_v2_shakedown/protocol.json`: the
preregistered `pic2d_cft_steady_state_v4` protocol - 90 x 720, 33.3 um, 1.4 ps, W 26 666.7, seed 20260903,
v1.3 closure, v2.0.3 gates - with `operating_point.collision_set = xe_collision_set_v2`) plus
`numerics.coulomb = {enabled, e-e, e-i, i-i off, cycle_steps 10, ln Lambda floor 2, T floor 0.01 eV}` and its
identity / status fields changed. The R3 shakedown record is therefore the **Coulomb-OFF twin** at the same
100 000 steps (its code path replays bitwise without the block).

* `run.py cost` times the Coulomb stage on the protocol's grid with a synthetic 4.5 M-particle plateau load
  (2.25 M e- + 2.25 M Xe+ uniform in the channel; stage alone and inside the captured step) -> `cost.json`.
* `run.py shakedown` calls the v4 shakedown (shrunk cadences, 100 000 steps, finalize + assess) and adds the
  Coulomb readings (trailing-half mean nu_ee / nu_ei / nu_en / nu_ee over nu_en; nu_ee / nu_ei at the window's
  peak-density cell and per cusp column from `maps.npz`) and the direction of S / I_d / T_e,peak against the
  off twin -> `shakedown.json`.

## Record (2026-09-04 19:21-19:40 UTC, Lambda H100 as the 6th CUDA-MPS client beside 4 preregistered runs and
another agent's physics-effects shakedown; code e0d0a28a, readings refreshed at 4778f460 = the R4 tree before its
rebase onto 4a0a43ca (same blobs: f5eb08ad / be279bd5 / 31c765b8 after the rebase); `shakedown.json`, `cost.json`)

### CUDA tests (90 s wall, ~1.5 GPU-minutes)

`test_pic2d_v24_coulomb.py` 21 passed on the box (all 17 tests incl. the cuda harness variants: Trubnikov relaxation
and Lorentz drift on the device, CUDA graph vs direct launches bitwise with e-e + e-i + i-i on and the same-seed
replay, cuda stage vs cpu stage counts within 2 %); the graph / parity regressions of the earlier modules 9 + 3 passed
(`test_pic2d_warp_parity.py`, the SEE graph test, the collision-set CUDA tests).

### Cost (`run.py cost`, 4.5 M synthetic plateau load = 2.25 M e- + 2.25 M Xe+ uniform in the channel, the protocol's
90 x 720 grid, capacities 4.5 M per species, 15 repeats, CONTENDED: 5 other MPS clients at 100 % GPU)

| quantity | ms |
|---|---|
| Coulomb cycle, stage alone (direct launches, e-e + e-i) | 4.49 median (4.30 min) |
| of which cell sort of both species (cell / scan / scatter / rank) | 1.14 |
| of which per-cell prepare + pair kernels | 3.34 |
| step WITH the Coulomb cycle (captured graph; ion step + redeposit variant) | 11.45 |
| ordinary step (same variant, same contention) | 6.62 |
| overhead per Coulomb cycle inside the graph | 4.82 |
| amortised over `cycle_steps` = 10 | **0.48 ms/step = +7.3 %** of the contended step |

Under MPS contention tiny kernels inflate 7-10x and large ones 1.1-1.4x (perf-audit lesson), so both the step and the
stage are inflated; the ratio is the honest figure until a solo probe. Every step (k = 1) would cost +73 %; the
audit's a-priori figures were +15-30 % every step / +2-3 % amortised over 10-20 steps. The shakedown's own step
rate: 4.79 ms/step against the off twin's 4.52 ms/step at the same seed load (+6 %).

### Shakedown (100 000 steps, `target_steps_reached`, 633 s, 4.79 ms/step contended, 50 frames)

* finalize + assess ran (verdict `no_plateau`, as any 0.14 us run must; (b) passes at +0.12 %); the peak-Debye window
  was enforced in 301/500 records (max 0.57 cells/lambda_D, 1239 resolved nodes at the end), the residual window
  complete in 280 records at +0.14 % of the electrode work (off twin +0.09 %); CUDA graph active; no Xid (the two
  Xid-31 lines in dmesg are the 2026-09-04 04:15 events); GPU client removed cleanly.
* Conservation through the run: `pz_coulomb` 5.0e-29 kg m/s cumulative over 3.0e9 e-e + 5.5e9 e-i pair collisions
  (10 000 cycles); `ke_coulomb_j` 3.9e-16 J cumulative against K_e ~ 2e-8 J (2e-8 relative): the elastic operator
  leaves the energy ledger untouched at the level the residual gate reads.
* Deflection statistics (trailing half): mean s per pair 6.4e-5 (e-e) / 4.5e-5 (e-i); pairs with s > 1: 3.2e-6 /
  2.2e-6 of all pairs; mean ln Lambda 13.08 / 13.04 - the k = 10 cycle is comfortably inside the small-angle regime
  at this (transient, ~1e17 m^-3) density and stays so at the 1e18 plateau (s scales with n).
* Frequencies (trailing half, 70-140 ns; the discharge is in its seed transient: peak n_e 1.7e17 window mean, 10x
  below the ss-v4 plateau 1.3e18; T_e,peak 8.2 eV): the NRL / Spitzer electron collision rate
  `2.91e-6 n lnL T^-3/2` at the window's peak cell 2.8e5 /s (cusp columns 6.03 / 12.0 / 17.97 mm, electron-weighted:
  5.2e4 / 4.2e4 / 3.4e4 /s; electron-weighted mean over the plasma 1.8e5 /s) against nu_en (MCC elastic events per
  electron) 1.17e7 /s: **nu_e / nu_en 0.024 at the peak cell of the transient**; scaled to the plateau peak (n x8,
  T 5.6 eV) that is ~4e6 /s and a ratio ~0.3 - consistent with the audit's 0.15-0.4 estimate, to be READ from the
  R4 runs, not assumed. The operator's own pair-mean deflection rates `<s>/dt_c` are larger (peak cell nu_ee 3.7e6,
  nu_ei 7.8e6; plasma means 5.0e6 / 3.2e6; the series' `nu_ee_over_nu_en` 0.43): a 1/g^3-weighted mean is dominated
  by the slowest pairs and grows logarithmically with the sample (the population mean is infinite) - it is recorded as
  the operator's realised deflection statistic, not as the collision frequency of the audit (the two definitions are
  stated in `Simulation._coulomb_record`).
* Direction against the off twin at equal steps (trailing-half means; SEED-TRANSIENT readings, a direction indicator
  only): S 1.609e16 vs 1.616e16 /s (-0.5 %), I_d 1.321 vs 1.327 mA (-0.4 %), I_beam 0.315 vs 0.316 mA (-0.3 %),
  window T_e,peak 8.16 vs 7.91 eV (+3.2 %), window peak n_e 1.32e17 vs 1.35e17 (-2.4 %). At 0.14 us and 1e17 m^-3 the
  Coulomb rates are an order of magnitude below their plateau values and the two runs differ by a fresh random stream
  as well, so these are inside the shot noise of a 70-ns average; the audit's expectations (S +5-20 %, T_e,peak -5 %)
  are plateau statements for the R4 runs.
* GPU use of the whole R4 box session: 1.5 min of CUDA tests + 2.6 min cost probe + 10.6 min shakedown =
  **~15 GPU-minutes** as an extra MPS client; nothing signalled, no `timeout` wrapper, the box worktree removed after.
