# pic2d steady-state run v2 — model v1.3 (quasi-steady neutral inventory)

**Status: development / screening. Not preregistered, not validated, not a performance
prediction.** One detached, checkpointed, resumable GPU run of the divergent-exit CFT
channel with the v1.3 quasi-steady 0-D neutral inventory
(`modern/spec/pic2d/pic2d-model-v1.3.json`).

## What changed vs. `pic2d_cft_steady_state_v1`

* The neutral density is a state `n_g(t)`, uniform in space, driven by a prescribed
  feed `Q_in = 7.77e16 atoms/s` (0.017 mg/s, the effusion at `n_g0 = 5e19 m⁻³`), the
  ionisation sink measured from the MCC tallies, and thermal effusion through the exit
  plane (`c n_g`, `c = v̄ A_exit / 4 = 1.55e-3 m³/s` at 300 K).
* The transient toward the fixed point `n_g* = (Q_in − S)/c` is **artificial**
  (`τ_g = 30 ns`; the physical effusion time is 221 µs). Only the fixed point is
  physical. The artificial term has its own atom ledger so the balance
  `fed − ionised − effused − artificial = V Δn_g` closes to round-off.
* The MCC scales the real-collision frequency by `n_g / n_g0`; the null-collision
  ceiling stays at `n_g0`. `n_g > n_g0` or exhaustion fails closed.
* The plateau rule also requires `n_g` drift < 5 % over the trailing 20 %.
* Expected fixed point: `n_g ≈ 2.3–3.4e19`, `n_e ≈ 1.1–2.9e17` (0.28–0.73 of the
  resolvability ceiling `n_max = 4e17`); `W = 6e4` keeps 1.3–3.2 M macro-particles.

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
