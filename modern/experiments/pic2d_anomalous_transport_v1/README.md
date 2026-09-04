# PIC-2D anomalous cross-field transport v1 — the preregistered Bohm α-series (roadmap R1, model v2.1.0)

**Status: PREREGISTERED (see §7 for the commit); nothing launched at the draft commit.** Three one-shot executions of the
reference design at 33 µm with the Bohm-type anomalous transport closure at α ∈ {1/64, 1/16, 0.345}; the α = 0 point of the series
is the RECORDED ss-v4 plateau (`pic2d_cft_steady_state_v4/results`, `0d228ad2`), which is not re-run.

## 1. Why

The physics completeness audit (`modern/docs/pic2d-physics-completeness-audit.md`, `0901138a`, §4.c) ranks anomalous cross-field
electron transport as the structural gap of the 2D axisymmetric model: the E × B drift instabilities that carry it in the real
device cannot exist in (r, z), every HEMPT PIC we could read imposes a Bohm-type closure (Brandt et al. 2016: D⊥ = 0.4 kT_e/eB
"derived from a 3D simulation of a similar thruster model"), and our closure without it produced an ionisation avalanche at Brandt's
operating point (ext-val v0 launch 1). Roadmap R1 is the predeclared α-series on the accepted 33 µm plateau, then the sealed ext-val
`bohm-0.4` run.

## 2. What the hook did, what it does now (the audit of R1, model v2.1.0)

The v1.4 hook (`cft_revival.pic2d.sensitivity`, `warp_backend.bohm_kernel`) already applied the right event statistics: an exact
Poisson probability `1 − exp(−α ω_ce Δt)` per electron per step at the electron's local |B| (bilinear gather, so the probability
vanishes at the nulls and peaks at the cusps), a speed-preserving event (kinetic energy conserved to round-off — no ledger energy
term is owed, the event is elastic), the count tallied in `cumulative["anomalous"]` and the axial momentum handed to the
"turbulent field" in `pz_collisions`, drawn from its own seed stream after the push and before the MCC. It is **not** part of the MCC
null-collision budget: it is a separate exact-Poisson process, statistically equivalent to a joint budget to
O(ν_an Δt × ν_mcc Δt) ≈ 1e-7 (ν_an Δt ≤ 0.026 at 0.29 T / 1.4 ps / α 0.345; ν_mcc Δt ~ 1e-5), which is what keeps the α = 0 path
bitwise identical to the model without the hook.

What it lacked was the reference's **event model**: the v1.4 event redirected the velocity isotropically, which also randomises the
parallel speed — a pitch-angle scattering that feeds the cusp loss cones directly. Brandt et al. 2016 (p. Pb_237) rotate "only the
component of the velocity vector perpendicular to the local magnetic field direction … to ensure that the speed of the electrons
along the magnetic field lines does not change". Model v2.1.0 adds that event (`bohm_perpendicular_rotation`: a Rodrigues rotation
about the local B by a uniform random angle; |v| and v∥ unchanged to round-off, gyro-centre shifted by 2 r_L sin(φ/2)) in both
backends, selected by `numerics.anomalous_collisions.model`; the isotropic model stays as the recorded v1.4 default, so every recorded
identity is unchanged.

Both events give the same cross-field diffusion coefficient for a Maxwellian, D⊥ = (kT_e/eB) · α/(1 + α²) (Green-Kubo: the velocity
autocorrelation of a gyrating electron whose gyro-phase is reset at Poisson rate ν is ⟨v_x²⟩ e^{−νt} cos ω_ce t). The diffusion test
(`tests/pic2d/test_pic2d_v210_anomalous_transport.py`: 24 000–60 000 test electrons gyrating in a uniform 0.05 T with no E, the hook
applied per step, MSD fitted over 20–30 collision times) reproduces it within 5 % for α = 1/16 and 0.345 under BOTH models, and shows
the 0.345 case sits 10.6 % BELOW the naive α kT_e/eB — the audit's exact factor for Brandt's ν = 0.4 ω_ce.

## 3. Design (frozen in `protocols/<case>.json`; the campaign protocol is `protocol.json`)

| block | value | source |
|---|---|---|
| template | ss-v4 protocol: divergent-exit-stack, 90 × 720 cells (33.33 µm), Δt 1.4 ps, W 26 666.7, v1.3 closure (0-D inventory, no recycling), seed 20260903, frames ON (20 000-step cadence) | `pic2d_cft_steady_state_v4/protocol.json`, byte-for-byte except the rows below |
| closure | `bohm_perpendicular_rotation`, ν_an = α ω_ce; α = 1/64 (`alpha-1over64`), 1/16 (`alpha-1over16`), 0.345 (`alpha-0.345`) | audit §4.c; D⊥/(kT_e/eB) = 0.0156 / 0.0623 / 0.308 |
| α = 0 | the recorded ss-v4 plateau: I_d 3.801 mA, I_beam 2.459 mA, S 3.595e16 /s, utilisation 0.420, n_g 3.188e19, peak n_e 1.287e18 at node (20, 429), T_e,peak 5.58 eV, Δ/λ_D 2.15 | `reference_run` (= v5's pinned block; `assess` refuses if the artifacts disagree) |
| α = 0 caveat | its acceptance (b) FAILS on the corrected ledger: +2.46 % of the electrode work in the trailing 400k window (recorded −7.67 % before the v2.0.6 W fix) | `results/ledger-corrected.json` (`02013df0`) |
| gates | v2.0.3 window-mode peak-Debye gate (hard π, soft 2.5) with the v2.0.6 accumulated-particle-step floor 64 000; windowed residual-power gate 5 % on the v2.0.6 W-corrected ledger; triad drift members; runtime ω_pe Δt (v2.0.4 resolved statistic) | thresholds byte-for-byte v4's; the floor and the ledger fix are v2.0.6 |
| diagnostics | K = 5 electron-moment sampling (v2.0.5; physics bitwise, enters `config_sha256`) | `8aca6c3a` |
| budget | 1.5 × the launch-box plateau-load preflight rate × 5 142 858 steps (3 transits), per case (`preflight-<case>.json` → `compose --budget-from-preflight`) | §5 |
| identities (warp-cuda) | α = 0: `f10772b25b03…` (= v4, test-pinned); 1/64 `28ca0391fdb0`; 1/16 `90cf53f1aef1`; 0.345 `8ea882736213` | `config_sha256` |

## 4. Predeclared acceptance and hypotheses (`stopping_rule.acceptance`; `run.py assess`)

Per case: (a) the v4 plateau rule (≥ 3 transits, trailing-20 % drifts of I_d, N_e, n_g < 5 %, triad soft bounds, peak-Debye soft
margin 2.5); (b) corrected windowed residual < +2 % (one-sided). Verdict `plateau_clean` (a ∧ b), `plateau_heating` (a ∧ ¬b — like the
α = 0 reference), `no_plateau`.

Shift table against α = 0 for I_d, I_beam, S, utilisation, n_g, peak n_e, T_e,peak with the recorded particle band (the 50 µm pair's
seed-b / W×0.7 bands: 5.7 / 5.7 / 4.6 / 4.6 / 4.0 / 11.9 / 9.3 %); a shift is CONFIRMING when it has the declared sign and exceeds the
band, CONTRADICTING when it has the opposite sign and exceeds the band, INSIDE THE BAND otherwise. Per-cusp report (planes 6.028 /
12.000 / 17.972 mm, ±1 mm): electron and ion wall current, axis-to-wall potential drop, near-wall T_e, beside the v4 values.

Hypotheses (audit §4.c / R1 — the SIGN is the hypothesis, the magnitude is the measurement):

| quantity | sign | expected at α = 1/16 |
|---|---|---|
| I_d | + | +20 to +60 % |
| S, utilisation | − | −10 to −40 % |
| n_g | + | +5 to +25 % |
| peak n_e | − | −15 to −40 % |
| T_e,peak | − | −5 to −25 % |
| I_beam | − (weak) | −10 to −40 % |
| per-cusp electron wall current | + | up |
| cusp sheath drop | − | down |

Series verdict (`assess --series`): `trend_confirmed` = ≥ 3 of the 4 points reached (a), I_d and peak n_e monotone in the declared
direction over α (reversals inside the band are ties), no monotone quantity contradicting at a reached point; `trend_not_confirmed`
= ≥ 3 points reached and a reversal or a contradiction (a finding about the closure's sign); `inconclusive` = < 3 points reached, or
every I_d / peak n_e shift inside the band. No α is "chosen": α stays a declared closure parameter (audit §6) until an r–θ / z–θ
companion campaign supplies the mobility.

## 5. Launch-box preflight and shakedown (non-evidentiary)

Filled in at the preregistration commit (§7): `preflight-<case>.json` (H100, CUDA MPS client; field, mesh, factorisation, ms/step at
the seed and at a synthetic 4.5 M-particle load, the anomalous event counts, the budget derivation) and `shakedown-alpha-1over16.json`
(100 000 steps of the real case with shrunk cadences through run → assess (case + series) → re-finalize; the peak-Debye window
enforced with the accumulated floor, the residual window complete, anomalous events non-zero).

## 6. Commands (from `modern/`, `PYTHONPATH=src:.`)

```
python -m experiments.pic2d_anomalous_transport_v1.run compose [--budget-from-preflight]
python -m experiments.pic2d_anomalous_transport_v1.run preflight --case alpha-1over16 --gpu-timing
python -m experiments.pic2d_anomalous_transport_v1.run shakedown --case alpha-1over16
python -m experiments.pic2d_anomalous_transport_v1.run launch --case alpha-1over16 --expect-commit <prereg sha> --require-mps
python -m experiments.pic2d_anomalous_transport_v1.run status
python -m experiments.pic2d_anomalous_transport_v1.run assess --case alpha-1over16
python -m experiments.pic2d_anomalous_transport_v1.run assess --series
```

Launch order (one H100 MPS slot each, as the scheduler frees them): `alpha-1over16` → `alpha-1over64` → `alpha-0.345`
(`modern/tools/cloud/jobs.yaml` jobs `at-alpha-1over16`, `at-alpha-1over64`, `at-alpha-0.345`).

## 7. Preregistration and launch log

- Draft: code + package + tests (this README's §1–4, §6), no preflight / shakedown records, no launch.

## 8. Claim boundary

Preregistered closure-sensitivity study of a development model on one design, one operating point, one seed per α, against a
reference plateau that itself heats at +2.46 % on the corrected ledger. The outcome is the sign and monotonicity of the closure's
effect; the magnitudes are recorded; every discharge quantity of the 2D axisymmetric model is conditional on the declared α; not
validated against experiment; not a thruster performance prediction.
