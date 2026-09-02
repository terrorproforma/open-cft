# PIC-2D CFT snapshot v2 (model v1.1; development / screening)

Status: `development_screening_not_preregistered`. Second development snapshot
of `cft_revival.pic2d` on the qualified P2 `divergent-exit-stack` field, at the
v1.1 operating point chosen from the v1 diagnosis
(`../pic2d_cft_snapshot_v1/results/diagnosis.json`). Not preregistered, no
experimental comparison, no performance claim. Every simplification is listed
in `protocol.json` and repeated in each `summary.json` and in the dashboard.

From `modern/`:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
python -m experiments.pic2d_cft_snapshot_v2.run budget
python -m experiments.pic2d_cft_snapshot_v2.run case coarse-w2.4e5 --max-wall-seconds 7200
python -m experiments.pic2d_cft_snapshot_v2.run case coarse-w1.2e5 --max-wall-seconds 7200
python -m experiments.pic2d_cft_snapshot_v2.run case fine-w6e4 --max-wall-seconds 7200
python -m experiments.pic2d_cft_snapshot_v2.run case fine-w3e4 --max-wall-seconds 7200
python -m experiments.pic2d_cft_snapshot_v2.run summarize
python visualization/generate_pic2d_cft_snapshot.py
```

Operating point (v1.1): anode 300 V, exit plane 0 V, uniform static Xe
background 1e20 m⁻³ at 300 K, 3 mA of 2 eV electrons injected at the exit
plane, 1e16 m⁻³ / 5 eV seed plasma, Δt = 1.5 ps, ion subcycle k = 8, device
block-Thomas Poisson solve, host sync every 200 steps. Grids 30×240 (100 µm)
and 60×480 (50 µm), two macro weights each. A-priori budget: n_max = 4e17 m⁻³,
T_e = 8 eV → λ_D,min = 33 µm, ω_pe Δt = 0.054, Ω_ce Δt = 0.077.

Stopping rule: at least `min_steps` (666 667 = 1 µs ≈ one ion transit time),
then stop at the first 180 ns window boundary where both the discharge current
and the plasma electron count drift < 5 % over the trailing 20 % of the run
(linear fit), else at `target_steps` (1.6 µs) or the wall-clock budget; the
fail-closed runtime gates (`ω_pe Δt ≤ 0.2` on the peak node density, one-cell
Courant crossing, Poisson residual contract) can end a case earlier. The reason
is recorded in `summary.json`.

## Result (2026-09-03, RTX 5090, four cases sharing the GPU)

| case | steps | t (µs) | stop | window I_d (mA) | I_beam,i (mA) | peak / mean n_e (m⁻³) | ⟨T_e⟩_n (eV) | plateau (drift I_d, N_e) |
|---|---|---|---|---|---|---|---|---|
| coarse-w2.4e5 | 621 200 | 0.93 | ω_pe Δt gate | 4.36 | 1.35 | 2.36e18 / 5.71e17 | 11.1 | no (14 %, 69 %) |
| coarse-w1.2e5 | 757 000 | 1.14 | ω_pe Δt gate | 4.70 | 1.68 | 2.32e18 / 5.02e17 | 9.5 | no (22 %, 64 %) |
| fine-w6e4 | 1 022 000 | 1.53 | wall budget | 5.19 | 2.20 | 2.18e18 / 4.62e17 | 6.4 | no (12 %, 24 %) |
| fine-w3e4 | 802 000 | 1.20 | wall budget | 4.43 | 1.57 | 1.46e18 / 3.21e17 | 6.4 | no (9 %, 19 %) |

No case reached a plateau: at this operating point the ion loss is 10 % (coarse)
to 35 % (fine) of the ionisation rate after one ion transit time and the
density is still growing roughly linearly; the window-averaged peak density
exceeds the a-priori design ceiling by 3.7–5.9× and the cells are 3–6 λ_D at the
end. The coarse grid runs ~1.5–1.7× hotter than the fine grid and ionises
3.5× faster (grid heating; its ledger residual is +41 % of the electrode work,
the fine grid's is −13 to −18 %). Between-grid spreads are 43–57 % in density
and T_e, 50 % in the ion currents, 18 % in the discharge current: not
converged. The first v2 attempt at Δt = 3 ps was stopped by the gate at
0.27 µs on axis-node shot noise; see `protocol.json` (`dt_justification`).
