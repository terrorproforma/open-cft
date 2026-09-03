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
| 2 (`results/`) | 5.5e19 / 8.55e16 s⁻¹ (0.0186 mg/s) | 5e16 m⁻³, 5 eV (the v1/v2 snapshot seed) | **plateau** declared at step 5 120 000 (7.68 µs = 3.2 ion transit times, 10 141 s wall, 2.0 ms/step at ~1.0 M + 1.0 M macro-particles); trailing-20 % drifts I_d +0.08 %, N_e +4.98 % (marginal), n_g −0.18 % |

The seed is an initial condition (not replenished); the diagnosis and both adjustments
are recorded in `protocol.json` (`attempts`, `seed_justification`, `feed_justification`).

## Plateau result (attempt 2, window = last 320 000 steps, 7.20–7.68 µs)

| quantity | value |
| --- | --- |
| I_d (anode e⁻ − anode Xe⁺) | 3.44 ± 0.33 mA (interval scatter) |
| I_beam,i (exit plane) | 2.29 mA = 0.67 I_d |
| wall Xe⁺ / wall e⁻ / returned e⁻ / injected e⁻ | 3.72 / 3.73 / 1.84 / 3.00 mA |
| S (ionisation rate) | 3.93e16 s⁻¹ (6.3 mA equivalent; 2.1 ionisations per injected electron) |
| utilisation S / Q_in | 46 % |
| n_g window mean / analytic fixed point (Q_in − S)/c | 2.969e19 / 2.970e19 m⁻³ (0.03 %); n_g / n_g0 = 0.54 |
| n_e mean / peak (window maps) | 2.13e17 (0.93 × the projected 0-D n_eq) / **1.64e18 = 4.1 × n_max** at z = 14.3 mm, r = 0.70 mm |
| ⟨T_e⟩ (density-weighted) · T_e max | 8.2 eV · 59 eV |
| φ range | −10.7 … 337 V (anode 300 V) |
| energy-ledger residual (with electrode work) | −2.9e-7 J = −4.4 % of the 6.5e-6 J electrode work; interval RMS 1.5e-11 J |
| neutral-ledger closure | 0.14 atoms = 7e-15 of the inventory |
| resolvability at the peak node | λ_D = 16.6 µm → Δz/λ_D = Δr/λ_D = 3.0 (**under-resolved**); ω_pe Δt 0.108 at the peak, max observed 0.118 (gate 0.2) |
| cusp planes (B_z = 0 on axis) | z = 6.0, 12.0, 17.95 mm; the density peak sits in the magnetic bottle between the last two cusps, the wall ion flux peaks at the cusps |

The plateau is a **single-seed development result**: the peak-density region exceeds
the a-priori resolvability budget by 4× (the mean density is inside it), the electron
count was still drifting +5 % per 1.5 µs when the criterion passed, and no convergence
pair exists yet. Dashboard: `modern/visualization/pic2d-cft-steady-state.html`.

## Convergence pair (`variants.json`, run sequentially with `--case`)

`protocol.json` is frozen (the finished run is hash-bound to it); the variants live in
`variants.json` and write to `results-<case>/`:

| case | override | purpose | status |
| --- | --- | --- | --- |
| `seed-b` | RNG seed 20260904, 3.5 h wall budget | statistical variance of the plateau quantities | launched 2026-09-03T02:32Z (PID 49716) |
| `w-half` | W = 4.2e4 (0.7 W, **not** W/2 — see the note in `variants.json`: W/2 projects to 4.3 h > the 3.5 h budget), 3.5 h wall budget | particle-resolution sensitivity | pending (after `seed-b`) |

```powershell
python -m experiments.pic2d_cft_steady_state_v2.run --case seed-b run
python -m experiments.pic2d_cft_steady_state_v2.run --case seed-b status
python -m experiments.pic2d_cft_steady_state_v2.run --case w-half run      # only after seed-b has ended
```

Regenerate the dashboard after a variant finishes; finished variants are embedded
automatically and the convergence table fills in.

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
