# PIC-2D physics effects v1 — the preregistered SEE(BN) / xenon-collision-set-v2 campaign (roadmap R2 + R3, models v2.2.0 + v2.3.0)

**Status: DRAFT (this commit): package + tests; no preflight / shakedown records, nothing launched.** See §7 for the preregistration
and launch log once the records exist.

Three one-shot executions of the reference design at 33 µm, each isolating one physics change of the completeness audit against the
RECORDED ss-v4 plateau (`pic2d_cft_steady_state_v4/results`, `0d228ad2`), which is not re-run:

| case | SEE from the BN wall (model v2.2.0) | xenon collision set v2 (model v2.3.0) | question |
|---|---|---|---|
| `see-bn` | on | off (legacy lumped set, collisionless ions) | R2 alone: sheath drops, T_e, I_d, the cusp SCL state |
| `xe-set-v2` | off (absorbing wall) | on (4 excitation levels + Xe⁺/Xe CEX + MEX, fast-neutral contract) | R3 alone: I_d/S ≈ unchanged, the exit IEDF's slow population, thrust redistributed |
| `see-bn+xe-set-v2` | on | on | do the two effects add? (combined vs sum of parts) |

The anomalous-transport closure is OFF in every case (α = 0): the R1 α-series (`pic2d_anomalous_transport_v1`) carries that closure, so
here each effect is read against the same recorded reference and the physics identity of a case differs from v4's by the effect block(s),
K = 5 and the declared Debye floor only.

## 1. Why

The physics completeness audit (`modern/docs/pic2d-physics-completeness-audit.md`, `0901138a`) ranks secondary electron emission from
the dielectric wall (gap (a), R2) second and the collision-set gaps (e1 four excitation levels, e4 Xe⁺–Xe charge exchange / momentum
transfer, R3) third after the transport closure. Both are now code (v2.2.0 `see_dielectric_v1`, `385f1db2`/`8e02db57`; v2.3.0
`xe_collision_set_v2`, `bc70479a`..`4219b654`) with box shakedowns that were explicitly NOT results. This campaign is the predeclared
measurement of their signs on the accepted 33 µm plateau.

The R2 shakedown (`$WORK/r2/shakedown-see-bn.json`, code `4ca89e72`, 100k steps, non-evidentiary) is the reason the per-cusp SCL reading is
a primary quantity here: with the Villemant BN fit the per-cusp effective yields read 0.96–1.00 and the near-wall potential drop went
NEGATIVE at all three cusps (a virtual cathode) — the cusps sat at the Hobbs–Wesson limit — while I_d read 2.06× the SEE-off shakedown's
transient value. The R3 shakedown (`pic2d_xe_collision_set_v2_shakedown`, code `6defd5ed`) read CEX 9.4e14 /s and MEX 4.9e14 /s against
S 1.6e16 /s (CEX/S 5.8 %), 0.31 CEX events per exit ion, level shares 22/20/40/18 %, residual +0.09 % with the new sink booked. Neither
shakedown used this campaign's protocol (no K = 5, no Debye floor, pre-rebase seed-stream layout, different code trees), so both are
re-run here under the sealed protocols (§5).

## 2. What the effects do (the models under test)

**SEE (BN)** — every electron impact on the dielectric (channel wall + cone stair; boundary code 3) emits ⌊δ⌋ + Bernoulli(δ − ⌊δ⌋)
macro-electrons of the impacting weight from the Vaughan yield δ(E, θ) with the BN constants δ_max 2.016 at 299 eV, threshold 0, k 0.563
(the PICLas tabulation of Villemant et al. 2019; first crossover 35.7 eV vs Dunaevsky 2003's 35 eV; flux-averaged yield 0.48 / 0.58 /
0.69 at T_e 5 / 7 / 10 eV, critical temperature 20.3 eV), split Sydorenko-style (elastic 3 %, inelastic 7 %, the rest true secondaries as
a flux half-Maxwellian at T_see 2 eV with the cosine law). No Hobbs–Wesson cap: the space-charge-limited sheath emerges, and the wall-
defined effective yield + wall potential per cusp are recorded so the regime is diagnosable. Surface charge = absorbed − emitted;
`ke_see_emitted_j` is an injected ledger term (the particle-side identity closes to round-off with SEE on).

**Collision set v2** — the electron elastic and ionisation tables are byte-identical to the legacy set; the lumped 8.32 eV excitation is
split into the Biagi-v7.1 levels 8.315 / 9.447 / 9.917 / 11.7 eV (their sum equals the lumped table to 0.24 % above 10 eV, so any
plateau change is attributable to the energy removed per event, 8.32 → 9.4–10.1 eV). Ions gain a null-collision operator per sub-step
against the inventory density with a Maxwellian atom: resonant charge exchange (Miller 2002, (87.3 − 13.6 log₁₀E) Å²) and momentum
transfer (Phelps isotropic, 3.39e-19 E^−½ m²). A CEX event turns the ion into a fast neutral whose fate is decided by a straight-line
march through the cell mask (exit → inventory sink F + `pz_fast_neutral_exit`; wall → thermalises; thermal → stays); the energy handed to
the neutrals is the sink `ion_neutral_loss_j`.

## 3. Design (frozen in `protocols/<case>.json`; the campaign protocol is `protocol.json`)

| block | value | source |
|---|---|---|
| template | ss-v4 protocol: divergent-exit-stack, 90 × 720 cells (33.33 µm), Δt 1.4 ps, W 26 666.7, v1.3 closure (0-D inventory, no recycling), seed 20260903, frames ON (20 000-step cadence) | `pic2d_cft_steady_state_v4/protocol.json`, byte-for-byte except the rows below |
| SEE block (`numerics.see`) | `enabled`, `material BN`, `vaughan_components`, T_see 2 eV, ion-induced yield 0, cap 8 per impact, `space_charge_limit_yield` 0.983, no overrides | spec v2.2 `see_dielectric_v1`; `SEEConfig.to_dict` in `config_sha256` |
| collision set (`operating_point.collision_set`) | `{name: xe_collision_set_v2, ion_neutral: true}` (default table grid 0.05 eV → 2000 eV, fast-neutral threshold 4 v_th); the electron / ion-neutral payload hashes are recomputed from the spec files, never read from the protocol | spec v2.3 `xe_collision_set_v2`; `MCCConfig.to_dict()['collision_set']` in `config_sha256` |
| transport | NONE (α = 0) | the R1 α-series carries it |
| reference | the recorded ss-v4 plateau: I_d 3.801 mA, I_beam 2.459 mA, S 3.595e16 /s, utilisation 0.420, n_g 3.188e19, peak n_e 1.287e18 at node (20, 429), T_e,peak 5.58 eV, Δ/λ_D 2.15; added here from the same artifacts: IEDF low-energy fraction (< 30 eV) 0.0671, anode ion current 59.2 µA, wall electron power 64.8 mW, flux-weighted wall-ion impact energy 60.5 eV | `reference_run` (= v5's pinned block + the recomputed extras; `assess` refuses if the artifacts disagree) |
| reference caveat | its acceptance (b) FAILS on the corrected ledger: +2.46 % of the electrode work in the trailing 400k window (recorded −7.67 % before the v2.0.6 W fix) | `results/ledger-corrected.json` (`02013df0`) |
| gates | v2.0.3 window-mode peak-Debye gate (hard π, soft 2.5) with the v2.0.6 accumulated-particle-step floor 64 000; windowed residual-power gate 5 % on the v2.0.6 W-corrected ledger; triad drift members; runtime ω_pe Δt (v2.0.4 resolved statistic); + the ion-MCC ceiling (v2.3.0) and the SEE birth-reservation overflow (v2.2.0) fail closed | thresholds byte-for-byte v4's |
| diagnostics | K = 5 electron-moment sampling (v2.0.5; physics bitwise, enters `config_sha256`) | `8aca6c3a` |
| budget | 1.5 × the launch-box plateau-load preflight rate × 5 142 858 steps (3 transits), per case (`preflight-<case>.json` → `compose --budget-from-preflight`) | §5 |
| identities (warp-cuda) | reference `f10772b25b03…` (= v4, test-pinned); `see-bn` `d45b0f859bf6`; `xe-set-v2` `7cfaa7847fb5`; `see-bn+xe-set-v2` `815762d7faab` | `config_sha256` |

## 4. Predeclared acceptance and hypotheses (`stopping_rule.acceptance`; `run.py assess`)

Per case: (a) the v4 plateau rule (≥ 3 transits, trailing-20 % drifts of I_d, N_e, n_g < 5 %, triad soft bounds, peak-Debye soft margin
2.5); (b) corrected windowed residual < +2 % (one-sided). Plateau status `plateau_clean` (a ∧ b), `plateau_heating` (a ∧ ¬b — like the
reference), `no_plateau`.

Shift table against the reference for I_d, I_beam, S, utilisation, n_g, peak n_e, T_e,peak (relative, with the 50 µm pair's particle band
5.7 / 5.7 / 4.6 / 4.6 / 4.0 / 11.9 / 9.3 %), the exit-plane IEDF low-energy fraction (ABSOLUTE, declared band 0.03 — no replicate exists
for it), and anode ion current / wall electron power / wall-ion impact energy (reported with their signs, no band). A shift with a "+"/"−"
hypothesis is CONFIRMING when it has the declared sign and exceeds the band, CONTRADICTING when it has the opposite sign and exceeds the
band, INSIDE THE BAND otherwise; a "0" hypothesis is CONFIRMING inside the band and CONTRADICTING beyond it either way.

Per-cusp report (planes 6.028 / 12.000 / 17.972 mm, ±1 mm): electron and ion wall current, axis-to-wall drop, near-wall drop
φ[wall−3] − φ[wall] (negative = virtual cathode), near-wall T_e, wall-ion impact energy; with SEE: effective yield, SEE current, mean
emitted energy and the SCL flag (effective yield ≥ 0.983 OR near-wall drop < 0). Effect diagnostics: SEE window emission current /
effective yield / wall-potential statistics; CEX / MEX / fast-neutral rates, CEX/S, IEDF descriptors, fast-neutral exit momentum rate
(trailing window from `series.jsonl`; run average as witness), excitation level shares.

Hypotheses (the SIGN is the hypothesis, the magnitude is the measurement; audit §4.a / §4.e, spec v2.2 / v2.3):

| quantity | `see-bn` | `xe-set-v2` | `see-bn+xe-set-v2` |
|---|---|---|---|
| I_d | **+** (+10…30 %) | **0** (< 5 %) | **+** |
| T_e,peak | **−** (−10…25 %) | − (−3…5 %, inside band) | **−** |
| peak n_e | − (−5…15 %) | — | − |
| S, utilisation | — | − (−3…5 %, inside band) | − |
| IEDF low-energy fraction (< 30 eV) | — | **+** (+0.15…0.30 abs) | **+** |
| anode ion current | — | + (reported) | + (reported) |
| cusp sheath drops | − (−10…45 %; reported per cusp) | — | − |
| wall electron power | + (×1.5–2; reported) | — | + |
| wall-ion impact energy | − (reported) | — | − |
| fast-neutral exit rate | — | > 0 (reference 0; reported) | > 0 |

Bold = the case's KEY quantities. Verdict per case: `confirmed` = (a) ∧ every key quantity CONFIRMING ∧ no banded hypothesis
CONTRADICTING; `not_confirmed` = (a) ∧ any banded hypothesis CONTRADICTING; `inconclusive` = ¬(a), or (a) with no contradiction but a key
quantity inside the band. Combined vs sum of parts (`assess --campaign`, needs all three at (a)): for every banded quantity
interaction = shift(combined) − [shift(see-bn) + shift(xe-set-v2)]; `additive` when |interaction| ≤ band, `super_additive` /
`sub_additive` otherwise (sign relative to the sum of parts); the statement is `additive` / `interacting` / `not_evaluable`. No
interaction sign is predeclared.

## 5. Launch-box preflight and shakedown (non-evidentiary)

Pending at the draft commit: `preflight --case <case> --gpu-timing` (3 cases, as an extra CUDA-MPS client under the current load) and
`shakedown --case <case>` (100 000 steps of EVERY case through run → finalize → assess case + campaign) on the Lambda H100; the budgets are
derived from the preflights before the preregistration commit.

## 6. Commands (from `modern/`, `PYTHONPATH=src:.`)

```
python -m experiments.pic2d_physics_effects_v1.run compose [--budget-from-preflight]
python -m experiments.pic2d_physics_effects_v1.run preflight --case see-bn --gpu-timing
python -m experiments.pic2d_physics_effects_v1.run shakedown --case see-bn
python -m experiments.pic2d_physics_effects_v1.run launch --case see-bn --expect-commit <prereg sha> --require-mps
python -m experiments.pic2d_physics_effects_v1.run status
python -m experiments.pic2d_physics_effects_v1.run assess --case see-bn
python -m experiments.pic2d_physics_effects_v1.run assess --campaign
```

Launch order (one H100 MPS slot each, AFTER the R1 queue `ext-val-v0-channel-20um-bohm-0.4` → `at-alpha-1over64` → `at-alpha-0.345`):
`see-bn` → `xe-set-v2` → `see-bn+xe-set-v2` (`modern/tools/cloud/jobs.yaml` jobs `pe-see-bn`, `pe-xe-set-v2`, `pe-see-bn+xe-set-v2`;
chained slot-waiter `modern/tools/cloud/slot_queue.sh pe-queue --after $WORK/r1/queue.log …`, which never calls the scheduler while the
R1 queue is alive).

## 7. Preregistration and launch log

- Draft (this commit): package + 8 tests; no records, no launch.

## 8. Claim boundary

Preregistered physics-effect study of a development model on one design, one operating point, one seed per case, α = 0, against a
reference plateau that itself heats at +2.46 % on the corrected ledger. The outcome is the sign of each effect (and whether the two add);
the magnitudes are recorded; every discharge quantity of the 2D axisymmetric model is conditional on the declared transport closure; not
validated against experiment; not a thruster performance prediction.
