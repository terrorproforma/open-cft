# PIC-2D full physics v1 — Coulomb (R4), the spatial Knudsen gas with metastables (R5) and every R1–R5 effect together (models v2.1.0–v2.5.0)

**Status: DRAFT (composition, tests, README); the preregistration commit carries the six preflight and six shakedown records and the
measured budgets — see §5 and §7.**

Six one-shot executions of the reference design at 33 µm on the ss-v4 template (90 × 720, Δt 1.4 ps, W 26 666.7, seed 20260903, frames ON,
the v2.0.6 gates with the accumulated-floor Debye gate 64 000, K = 5, the **model v2.1.1 drift-member arming latch** and the **ignition gate** of
the α-series amendment 1, the corrected ledger), every shift read against the RECORDED ss-v4 plateau (`pic2d_cft_steady_state_v4/results`,
`0d228ad2`, which fails its own acceptance (b) at +2.46 % on the corrected ledger — stated in every sealed protocol):

| case | α | SEE (BN) | Xe set v2 | Coulomb | spatial gas + metastables | F | question |
|---|---|---|---|---|---|---|---|
| `coulomb` | 0 | off | off (legacy) | **on** (e–e + e–i, cycle 10, i–i off) | off (0-D inventory) | – | R4 alone vs ss-v4 |
| `neutrals-spatial` | 0 | off | on (required by the metastables' level-resolved branching) | off | **on** | **1** (physical) | R5 = the operating-point change (channel mean n_g 2.5e20 vs the 0-D 5.5e19 → 3.2e19) |
| `neutrals-spatial-F10` | 0 | off | on | off | on | **10** | the time-acceleration QUALIFICATION twin: plateau scalars inside the particle band of F = 1 → F qualified, else disqualified |
| `full-physics-alpha0` | 0 | on | on | on | on | 1 | the full model without transport: the additivity statement's combined case, the α-trend reference |
| `full-physics-alpha1over16` | 1/16 | on | on | on | on | 1 | **does the Knudsen gas sustain the classical-Bohm-leaky discharge** that extinguished at the dilute 0-D gas (0916a4f8)? |
| `full-physics-alpha0.345` | 0.345 | on | on | on | on | 1 | the same at Brandt's coefficient (the strongest leak; sustained marginally at Brandt's static 2e20 in ext-val bohm-0.4) — launched FIRST |

Launch order (the sustain question decides the value of the rest): `full-physics-alpha0.345` → `full-physics-alpha0` → `neutrals-spatial` →
`full-physics-alpha1over16` → `coulomb` → `neutrals-spatial-F10`, one H100 MPS slot each, chained AFTER the physics-effects queue
(`pe-queue`: see-bn → xe-set-v2 → see-bn+xe-set-v2) by the box slot-waiter `fp-queue` (`tools/cloud/slot_queue.sh fp-queue --after $WORK/pe-queue/queue.log …`).

## 1. Why

The α-series' first launch (`pic2d_anomalous_transport_v1`, α = 1/16, `0916a4f8`) did not re-equilibrate — it **extinguished**: N_e decayed from
the seed with an e-fold of 0.88 µs ≈ r_w²/4D⊥, I_d 3.1 → 0.06 mA, the injected 3 mA returned through the exit plane, S → 0 (residual +1.15 %,
0.48 cells/λ_D: not heating). At the same time the external-validation launch 2 (`bohm-0.4`, α = 0.345) at Brandt's static gas 2e20 — 4× our
0-D density — sustained a marginal discharge past 1.2 transits. And the R5 shakedown (`55092f4c`) showed that the 0-D inventory had equated the
whole channel to the EXIT density: the free-molecular closed-end (Knudsen) profile at the same feed is 5.45e20 at the anode → 7.0e19 at the exit,
channel mean 2.49e20 = 4.5× the 0-D fixed point. Every plateau recorded so far sits at a different (too dilute) operating point than the spatial
model gives. Hence the central hypothesis of this campaign: **a Bohm-leaky discharge needs the denser gas — the operating point, not the closure
alone, decides.** The Coulomb case (R4) and the R5 case isolate the two remaining single effects of the completeness audit; the three full-physics
cases put every R1–R5 effect together and ask, first of all, whether the discharge exists.

## 2. The models under test (all in code, each off-switch bitwise vs its predecessor)

* **Coulomb `coulomb_v1`** (v2.4.0, `f5eb08ad`..`82255081`): per cell every 10 steps, e–e Takizuka–Abe random-permutation pairing, e–i every
  electron once against a cyclically shifted ion at the field ion density; Nanbu 1997 cumulative angle, exact CoM rotation, NRL ln Λ from the
  cell moments (floors 0.01 eV / 2.0). i–i off. The series carries the operator's pair-mean deflection rate (a 1/g³-weighted mean, ~13× the
  Spitzer rate) AND the NRL Spitzer ν_e at the peak node with its ratio to ν_en — the audit's gap-(d) number. Box cost +0.48 ms/step amortised.
* **Spatial neutrals `neutrals_spatial_v1` + `metastables_v1`** (v2.5.0, `8a35a44b`..`55092f4c`): test-particle free-molecular gas (Kn 10–100),
  cosine-law feed at 300 K from the anode face, diffuse wall reflection at T_w 500 K, wall-ion recycling at the impact cell, CEX fast neutrals as
  particles, nearest-cell density published every 200-step sub-step to a device-resident per-cell array read by both MCCs; per-cell integer sinks
  with a debt carry (the atom ledger closes exactly); Knudsen initial profile; ~4 M macro-neutrals at W_n 2.2e7 (~0.5 GB). Metastables: the Xe
  6s[3/2]₂ pool at 2 % of W_n with the Biagi-v7.1 level branching (0.45/0.35/0.50/0.35), BEB stepwise ionisation, superelastic return, wall
  de-excitation. **`metastables_v1` requires `xe_collision_set_v2`** (the branching is level-resolved; the runner refuses it on the lumped legacy
  set), so the R5 cases carry the collision set v2 — the R5-ALONE shift is read against the physics-effects `xe-set-v2` record as the secondary
  reference once it exists (until then the set-v2 contribution is predeclared inside the band on every plateau scalar).
* **Time acceleration F** (declared numerical parameter of the spatial model; 1 = physical): neutral time = F × plasma time. WHY it exists: the
  physical neutral relaxation (tube residence V/c ≈ 0.22 ms; re-equilibration of the Knudsen profile to the depleted steady state 0.2–2 ms) is
  30–300× longer than 3 ion transits (7.2 µs), so at F = 1 the gas over the run is the initial Knudsen profile minus ~0.3 % depletion — a
  QUASI-FROZEN gas with the right profile, not a neutral steady state (that needs F ~ 100–300). WHAT F distorts: the neutral response time
  (depletion, recycling refill, CEX flight, the metastable pool's filling — lifetime ~ the wall transit, 5–10 µs of neutral time — and the thermal
  transit all run F× faster in plasma time; the gas then answers transit-scale fluctuations of S, which a physical gas cannot). F must NOT change
  the plasma plateau while the gas is quasi-static: the `neutrals-spatial` / `neutrals-spatial-F10` pair is the predeclared qualification.
* **MCC ceiling** for the spatial cases: `operating_point.neutral_density_per_m3` = **1.5e21** = 2.75× the Knudsen anode density (fail-closed: the
  published density is clamped there and the run ends when > 1e-3 of an interval's plasma cell-substeps are clamped). The R5 shakedown's 1e21 gave
  a 3.5e-4 violation fraction (a 3× margin only): the smallest axis cells hold ~2.9 macro-neutrals, so ~7 % of the dense axis cells clamp at 1.8×
  and ~1 % at 2.75× (→ ~5e-5, a 20× margin) while a real inventory pile-up above 2.75× the anode density still ends the run. The ceiling sets only
  the null-collision candidate rate (~0.3 % of the electrons per step) and the metastable ceiling (0.05 × n_g0).
* **SEE (BN)** and **collision set v2** exactly as sealed by the physics-effects campaign (`79a7c87a`; blocks byte-identical, test-pinned).
* **Bohm perpendicular rotation** (v2.1.0) at α = 1/16 and 0.345 exactly as sealed by the α-series (`33be2a89`).
* **Arming latch (v2.1.1) + ignition gate**: numerically identical to the α-series amendment 1 (min_transit_times 2.0, I_d drift < 0.05 at a
  40 000-step checkpoint; reference 0.05–0.2 µs, N_e ≥ 0.6 & S ≥ 0.3 at 1.0 µs, ≥ 0.6 / ≥ 0.4 at 2.0 µs). Both are stopping-rule keys outside
  `config_sha256`. Every case here changes the operating point or the closure, so the v1.4 fixed-transit arming (calibrated on α = 0 plateaus at the
  0-D gas) is superseded; the residual-power and peak-Debye gates stay the physics protections. An **extinction** (stop reason `no_ignition`) is a
  valid recorded outcome, never a reason to adjust the seed, the injection or α.

## 3. Design (frozen in `protocols/<case>.json`; the campaign protocol is `protocol.json`)

| block | value | identity |
|---|---|---|
| template | ss-v4 protocol byte-for-byte except the rows below (90 × 720, Δt 1.4 ps, W 26 666.7, seed 20260903, 3 mA / 2 eV exit injection, feed 8.551e16 /s, frames ON) | v4 `f10772b2` |
| `numerics.coulomb` | `{enabled, electron_electron, electron_ion, ion_ion false, cycle_steps 10, coulomb_log_floor 2.0, min_temperature_ev 0.01}` | `CoulombConfig.to_dict` |
| `operating_point.neutrals` | `neutrals_spatial_v1`: feed = v4's, W_n 2.2e7, substep 200, F 1 / 10, T_w 500 K, accommodation 1, recycling on (recomb 1), Knudsen initial profile, ceiling-violation tolerance 1e-3; `metastables_v1` as in the R5 shakedown | `SpatialNeutralConfig.to_dict` (F included) |
| `operating_point.neutral_density_per_m3` (spatial cases) | 1.5e21 (MCC ceiling only) | `MCCConfig` |
| `numerics.see` / `operating_point.collision_set` | the physics-effects blocks | `SEEConfig` / `MCCConfig.collision_set` |
| `numerics.anomalous_collisions` | `{model bohm_perpendicular_rotation, alpha 0.0625 / 0.345}` | in `config_sha256` |
| gates | v2.0.6 floor 64 000, K = 5, v2.0.3 thresholds unchanged; arming latch + ignition gate (stopping rule) | floor / K in identity; rules outside |
| budgets | 1.5 × the launch-box measured plateau-load ms/step × 5 142 858 steps (the larger of the 4.5 M and 7.7 M-particle timings for the spatial cases) | §5 |

Identities (warp-cuda `config_sha256` prefixes at the draft): coulomb `49b30f51`, neutrals-spatial `66cb501c`, neutrals-spatial-F10 `e7a2d9b1`,
full-physics-alpha0 `7587b0f3`, full-physics-alpha1over16 `198fb4c6`, full-physics-alpha0.345 `98cc5cbc`; six distinct, none equal to v4's.

## 4. Predeclared acceptance and hypotheses (`stopping_rule.acceptance`; `run.py assess`)

* **(a) plateau**: the v4 rule (≥ 3 transits, trailing-20 % drifts of I_d / N_e / n_g < 5 %, triad soft bounds, Debye soft margin 2.5); the triad's
  drift members arm by the v2.1.1 latch.
* **(b) corrected residual** < +2 % (one-sided; the reference reads +2.46 % = FAIL).
* **(c) shift table vs ss-v4** with the 50 µm particle band (I_d 5.7 %, I_beam 5.7 %, S 4.6 %, utilisation 4.6 %, n_g 4.0 %, peak n_e 11.9 %, T_e,peak 9.3 %),
  absolute bands for the IEDF low-energy fraction (0.03) and the ionisation centroid (1 mm); plus anode ion current, wall electron power, wall-ion
  energy, the anode/exit density ratio, the depletion fraction, the metastable fraction, the stepwise share and the Spitzer ν_e/ν_en at the peak
  (reported). For the spatial cases the channel-mean n_g shift (+680 % by construction) is REPORTED, never judged. Per-cusp report (6.028 / 12.000 /
  17.972 mm ± 1 mm): wall currents, sheath and near-wall drops, near-wall T_e, wall-ion energy; SEE effective yield / current / emitted energy / SCL
  flag; CEX/S; Coulomb pair-mean and Spitzer rates in the cusp columns and at the peak cell; the local neutral density and metastable fraction at the
  cusp plane; the ionisation centroid, quartiles and the fraction upstream of 12 mm.
* **(d) verdicts**: `plateau_clean` / `plateau_heating` / `no_plateau` (stop class: peak_debye_gate / residual_power / triad_drift / budget / other) /
  **`extinguished`** (stop reason `no_ignition`, or a late decay the latch never armed on: trailing N_e < 0.25 × its maximum AND trailing I_d < 0.25 ×
  its running maximum). Hypothesis verdict `confirmed` / `not_confirmed` / `inconclusive` by the physics-effects rule; for the α cases the KEY is
  **sustains** (ignition gate passed at 1 and 2 µs AND (a)), an extinction = `not_confirmed` (the hypothesis contradicted in the strongest form),
  and the α-trend sign rows are judged against `full-physics-alpha0` when its record exists (secondary).
* **(e) sustain table** (`assess --campaign`): the three full-physics α points beside the dilute-gas α-series outcomes (1/16: extinguished, `0916a4f8`;
  1/64 and 0.345 as recorded) and the ext-val bohm-0.4 record; "the Knudsen gas sustains the Bohm-leaky discharge at α = X: YES / NO / UNDECIDED".
  α-trend: monotone I_d and peak n_e across the reached points (n_g excluded — the frozen gas cannot move it).
* **(f) F qualification**: both F members at (a) AND every plateau scalar of F = 10 inside the particle band of F = 1 → `F_qualified` (F may be used in
  later runs; a further qualification at F ~ 100 is owed), else `F_disqualified` (only F = 1 runs quotable); the metastable fraction, the stepwise share
  and the depletion fraction are expected to differ and are reported, not judged.
* **(g) additivity**: interaction = shift(full-physics-alpha0) − [shift(see-bn+xe-set-v2) + shift(coulomb) + (shift(neutrals-spatial) − shift(xe-set-v2))]
  per banded quantity → additive / interacting / not_evaluable (needs the physics-effects records). **R5 as the operating-point change**: whether
  |shift_R5| exceeds the sum of the other parts is reported per quantity.

Hypotheses (signs; `protocol.HYPOTHESES_*`): **coulomb** S +5…20 % (key), utilisation / I_beam / peak n_e up, I_d unchanged, T_e,peak down, n_g down,
Spitzer ν_e/ν_en 0.15–0.4 at the peak. **neutrals-spatial (+ set v2)**: S ×2–4 (key), I_d +30…100 % (key), utilisation up (MAY EXCEED 1 at F = 1: the
frozen gas does not deplete — the recorded signature that no neutral steady state was reached), I_beam / peak n_e up, T_e,peak −10…30 %, ionisation
centroid −1…−4 mm (upstream), IEDF low-energy fraction +0.15…0.40, anode ion current up, anode/exit ratio 5–10, metastable fraction 0.2–0.5 %, stepwise
3–6 % of S. A peak-Debye stop at the denser gas is a recorded resolution outcome. **full-physics-alpha0**: I_d +50…150 %, S ×2–4, T_e,peak −20…40 %
(keys), peak n_e up (the gas dominates SEE), wall e⁻ power up, wall-ion energy down, cusp sheath drops −10…45 % beside the SEE SCL flags.
**full-physics α cases**: sustains (key); vs α = 0: I_d up, S / utilisation / peak n_e / T_e,peak / I_beam down, cusp wall e⁻ current up, sheath drops
down, n_g unchanged (frozen gas).

## 5. Launch-box preflight and shakedown (non-evidentiary)

Filled at the preregistration commit from `preflight-<case>.json` / `shakedown-<case>.json` (Lambda H100 as an extra CUDA-MPS client beside the
four production runs; the box tree `$WORK/fp/tree` at the draft commit).

## 6. Commands (from `modern/`, `PYTHONPATH=src:.`)

```
python -m experiments.pic2d_full_physics_v1.run compose [--budget-from-preflight]
python -m experiments.pic2d_full_physics_v1.run preflight --case <case> --gpu-timing
python -m experiments.pic2d_full_physics_v1.run shakedown --case <case>
python -m experiments.pic2d_full_physics_v1.run launch --case <case> --expect-commit <prereg sha> --require-mps
python -m experiments.pic2d_full_physics_v1.run assess --case <case>
python -m experiments.pic2d_full_physics_v1.run assess --campaign
python -m pytest tests/pic2d/test_pic2d_full_physics_v1.py
```

## 7. Preregistration and launch log

* DRAFT: composition + tests + README (this commit); six sealed protocols with the a-priori 15.0 h budgets.
* PREREG: the experiment-dir-only commit carrying the box records and the measured budgets (§5).
* jobs: `tools/cloud/jobs.yaml` entries `fp-*` (enabled false; `--expect-commit` = the prereg SHA), `fp-queue` chained after `pe-queue`.

## 8. Claim boundary

Preregistered full-physics campaign on the reference design at 33.3 µm / 1.4 ps / W 2.667e4 at the ss-v4 feed and injection, against the recorded
ss-v4 plateau: the outcomes are (i) whether the full-physics discharge EXISTS at each α at this operating point, (ii) the SIGN of each effect on the
plateau quantities, the exit IEDF, the ionisation centroid and the per-cusp readings with the magnitudes recorded, (iii) whether F = 10 moves the
plateau, (iv) whether the effects add. Every discharge quantity of the 2-D axisymmetric model is conditional on the declared α; the F = 1 gas is
quasi-frozen at the Knudsen profile (no neutral steady state is claimed; a gross utilisation > 1 is the signature of that); not validated against
experiment; not a thruster performance prediction.
