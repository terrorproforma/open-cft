# PIC-2D physics effects v1 — the preregistered SEE(BN) / xenon-collision-set-v2 campaign (roadmap R2 + R3, models v2.2.0 + v2.3.0)

**Status: PREREGISTERED (the commit carrying this README, the sealed `protocols/*.json` with the measured budgets, the three preflight and
the three shakedown records); nothing launched at that commit — the launch log is §7.**

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

## 5. Launch-box preflight and shakedown (non-evidentiary; H100, 18:25–19:22 UTC 2026-09-04, code `b27da394`, as the 5th CUDA-MPS client beside ss25-base, sweep-056-launch2, ss33-fast and at-alpha-1over16)

`preflight-<case>.json` (`preflight --case … --gpu-timing`, 2000 timed steps after 200 warm-up, block-Thomas + CUDA-graph step; the rates
are 5-client rates and therefore upper bounds for a 4-client slot):

| case | factorisation | seed load (645 k e⁻) | plateau load (2.26 M e⁻ + 2.26 M i) | events over the 2200 plateau-load steps | device pool | 3 transits at the plateau load | budget (× 1.5, 10-min ceiling) |
|---|---|---|---|---|---|---|---|
| `see-bn` | 157 s | 4.81 ms/step | 6.33 ms/step | 375 k SEE impacts → 228 k emitted (0.61) | 1.38 GB | 9.05 h | **49 200 s (13.7 h)** |
| `xe-set-v2` | 149 s | 4.54 ms/step | 5.60 ms/step | CEX 85, MEX 759, level-1 excitations 1309 | 1.38 GB | 8.00 h | **43 200 s (12.0 h)** |
| `see-bn+xe-set-v2` | 162 s | 4.89 ms/step | 6.52 ms/step | 376 k → 230 k SEE; CEX 90, MEX 753 | 1.45 GB | 9.31 h | **50 400 s (14.0 h)** |

Field `abf26c5c4fa6` (max |B| 0.291 T), 45 810 plasma cells. SEE costs ≈ +13 % per step over the collision set alone at this load (6.33 vs
5.60 ms/step); the collision set itself is within the contention noise of the α-series' 4.77 ms/step measured with one client fewer. The
host factorisation read 150–160 s here against the α-series' 1.7 s: the box's CPUs were shared by four PIC processes plus this one (the
BLAS threads oversubscribe) — a one-off cost at launch, not a rate. The budget is the declared 1.5 × measured rule; a wall-budget stop is
resumable (new session, same identity, disclosed).

`shakedown-<case>.json` (`shakedown --case …`; 100 000 steps of EVERY case with shrunk cadences — series / sync 200, checkpoint 4000,
window 40 000, frames 2000; every gate, the grid, Δt, W, the effect blocks, field and seed the real ones; the shakedown identity differs
from the sealed one only through the shrunk cadences, which enter the gate config):

| case | ms/step | run / re-finalize | final e⁻ / Xe⁺ | effect events (cumulative) | peak-Debye window enforced | residual window (W-corrected) | assess |
|---|---|---|---|---|---|---|---|
| `see-bn` | 4.97 | 664 s / 162 s | 569 253 / 612 227 | 974 643 SEE impacts → 844 591 emitted (cumulative effective yield 0.86; backscattered 10 %; 0 clamped) | 301/500 records, max 0.52 cells/λ_D, 37 766 resolved nodes | 280 complete, last +0.08 % | `no_plateau` / `inconclusive`; reference consistency 11/11 |
| `xe-set-v2` | 4.10 | 568 s / 151 s | 554 859 / 600 955 | CEX 4 101, MEX 3 376, candidates 111 022 (null 103 545), fast neutrals 324 exit / 3 103 wall / 674 thermal, 0 unresolved, 0 ceiling violations, levels 16 366 / 14 740 / 29 944 / 13 449 | 301/500, max 0.59, 38 312 nodes | 280 complete, last +0.09 % | `no_plateau` / `inconclusive`; 11/11 |
| `see-bn+xe-set-v2` | 4.76 | 627 s / 166 s | 568 625 / 611 549 | 981 297 → 851 593 SEE; CEX 4 242, MEX 3 456, fast neutrals 464 / 3 132 / 646 | 301/500, max 0.52, 37 760 nodes | 280 complete, last +0.08 % | `no_plateau` / `inconclusive`; 11/11 |

Every effect path is live (`gate_not_inert_check`: SEE events, CEX events and all four excitation levels non-zero where declared; the
accumulated-floor peak-Debye window enforced; the residual window completed), `assess --case` formed the shift table, the per-cusp report
with the SEE / collision columns and the IEDF descriptors against the reference (11/11 pinned quantities recomputed from the v4
artifacts agree), `assess --campaign` returned `not_evaluable` (no case at (a)), and the re-finalize from the checkpoint ran. The
`xe-set-v2` shakedown reproduces the R3 shakedown (`6defd5ed`, `pic2d_xe_collision_set_v2_shakedown/shakedown.json`) event for event
(CEX 4 101, MEX 3 376, fast neutrals 324/3103/674, ionisations 83 452, exit ions 13 309, final ions 600 955): the rebase onto v2.2.0 and
the K = 5 / Debye-floor declarations are physics-neutral for this case. The `see-bn` shakedown is NOT bitwise vs the R2 shakedown
(`4ca89e72`: 569 456 / 612 467 final particles) because the rebase moved the SEE seed-table column (4) — the same physics, another stream.

**0.14 µs readings are the seed transient, not physics, and are not quoted as results.** What they show about the DIAGNOSTICS: with SEE the
per-cusp effective yields read 0.94 / 0.94 / 0.96 (window 60k–100k) with the near-wall drop NEGATIVE at 12.0 and 17.97 mm (`see-bn`: 2 of
3 cusps flagged space-charge-limited; combined: 3 of 3) — the R2 shakedown's virtual-cathode reading recurs, so the SCL flag will be a
primary column of the plateau report; wall potential mean 190 V (min −18 V, max 304 V), plasma-minus-wall 12.4 V, emitted mean energy
6.4 eV, SEE current 19.6 mA against 21.4 mA impacting. With the collision set: CEX/S 5.8 %, fast-neutral exit momentum rate 0.067 µN vs
exit-ion 1.83 µN over the last 20 000 steps (`series.jsonl` differences work), level shares 22 / 20 / 40 / 18 %, the exit IEDF still the
seed population (mean 18 eV, 91 % below 30 eV — the "+0.84 confirming" row of the shakedown assessment is that transient, the plateau
reference reads 0.067).

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

- Draft `b27da394`: package + 8 tests + `tools/cloud/slot_queue.sh`; no records, no launch.
- **PREREGISTERED** at the commit carrying this README, the sealed `protocols/*.json` with the measured budgets (sha256 `545b2e92…`
  `see-bn`, `83bd66b9…` `xe-set-v2`, `de0f2fda…` `see-bn+xe-set-v2`; campaign `protocol.json` `1a1d8d94…`), the three preflight records and
  the three shakedown records (§5). Launch order `see-bn` → `xe-set-v2` → `see-bn+xe-set-v2`, one H100 MPS slot each via
  `tools/cloud/schedule.py` jobs `pe-see-bn` / `pe-xe-set-v2` / `pe-see-bn+xe-set-v2` (jobs.yaml commit after this one), strictly AFTER
  the R1 queue (`ext-val-v0-channel-20um-bohm-0.4` → `at-alpha-1over64` → `at-alpha-0.345`) through the chained slot-waiter
  `slot_queue.sh pe-queue --after $WORK/r1/queue.log` (tmux `pe-queue`), which never calls the scheduler while the R1 queue is alive.

## 8. Claim boundary

Preregistered physics-effect study of a development model on one design, one operating point, one seed per case, α = 0, against a
reference plateau that itself heats at +2.46 % on the corrected ledger. The outcome is the sign of each effect (and whether the two add);
the magnitudes are recorded; every discharge quantity of the 2D axisymmetric model is conditional on the declared transport closure; not
validated against experiment; not a thruster performance prediction.
