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

## Record

(filled from the Lambda H100 session; see the commit that adds `shakedown.json` / `cost.json`)
