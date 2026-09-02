# pic2d steady-state run v2 — model v1.3 (quasi-steady neutral inventory)

**Status: development / screening. Not preregistered, not validated, not a performance
prediction.** One detached, checkpointed, resumable GPU run of the divergent-exit CFT
channel with the v1.3 quasi-steady 0-D neutral inventory
(`modern/spec/pic2d/pic2d-model-v1.3.json`).

## What changed vs. `pic2d_cft_steady_state_v1`

* The neutral density is a state `n_g(t)`, uniform in space, driven by a prescribed
  feed `Q_in = 8.55e16 atoms/s` (0.0186 mg/s, the effusion at `n_g0 = 5.5e19 m⁻³`;
  attempt 1 used 7.77e16 / 5e19), the
  ionisation sink measured from the MCC tallies, and thermal effusion through the exit
  plane (`c n_g`, `c = v̄ A_exit / 4 = 1.55e-3 m³/s` at 300 K).
* The transient toward the fixed point `n_g* = (Q_in − S)/c` is **artificial**
  (`τ_g = 30 ns`; the physical effusion time is 221 µs). Only the fixed point is
  physical. The artificial term has its own atom ledger so the balance
  `fed − ionised − effused − artificial = V Δn_g` closes to round-off.
* The MCC scales the real-collision frequency by `n_g / n_g0`; the null-collision
  ceiling stays at `n_g0`. `n_g > n_g0` or exhaustion fails closed.
* The plateau rule also requires `n_g` drift < 5 % over the trailing 20 %.
* Expected fixed point: `n_g ≈ 2.3–3.4e19`, `n_e ≈ 1.1–3.2e17` (0.28–0.8 of the
  resolvability ceiling `n_max = 4e17`; 2.3e17 = 0.58 at the ν_iz τ = 1 point for
  attempt 2); `W = 6e4` keeps 1.3–3.7 M macro-particles.

## Attempts

| attempt | n_g0 / Q_in | seed | outcome |
| --- | --- | --- | --- |
| 1 (`results-attempt1-ng0-5e19-seed1e16/`) | 5e19 / 7.77e16 s⁻¹ (0.0170 mg/s) | 1e16 m⁻³, 5 eV | **no ignition**: stopped at 1.14 µs; S fell 2.9e15 → 1.5e15 s⁻¹ as the seed cooled (7.9 → 5.0 eV), 91–96 % of the beam returned to the exit plane, I_d 0.2 mA, 0.12 ionisations per injected electron; n_g stayed at 4.9e19 (98 % of n_g0) |
| 2 (`results/`) | 5.5e19 / 8.55e16 s⁻¹ (0.0186 mg/s) | 5e16 m⁻³, 5 eV (the v1/v2 snapshot seed) | launched; at 0.13 µs: S 1.6e16 s⁻¹ and T_e rising, 60 % beam return, I_d 1.2–1.6 mA, n_g tracking its fixed point (4.46e19) |

The seed is an initial condition (not replenished); the diagnosis and both adjustments
are recorded in `protocol.json` (`attempts`, `seed_justification`, `feed_justification`).

## Running (from `modern/`)

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
python -m experiments.pic2d_cft_steady_state_v2.run run        # start or resume
python -m experiments.pic2d_cft_steady_state_v2.run status     # last status line + ETA
python -m experiments.pic2d_cft_steady_state_v2.run finalize   # artifacts from the checkpoint, no stepping
Get-Content experiments\pic2d_cft_steady_state_v2\results\status.jsonl -Tail 1
```

Detached launch and file layout are identical to v1 (see its README); `status.jsonl`
lines additionally carry `n_g_per_m3`, `n_g_fixed_point_per_m3`, `ionization_rate_per_s`,
`effusion_rate_per_s` and `neutral_ledger_residual_atoms`, and the summary has a
`neutral_inventory` block (final density, fixed point, ledgers, closure, utilisation).
