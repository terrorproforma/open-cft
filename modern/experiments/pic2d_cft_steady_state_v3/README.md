# pic2d steady-state run v3 — model v1.4 (wall-ion recycling, peak-node Debye gate, CUDA-graph step)

**Status: development / screening. Not preregistered, not validated, not a performance
prediction.** One detached, checkpointed, resumable GPU run of the divergent-exit CFT
channel at the v2 operating point with model v1.4
(`modern/spec/pic2d/pic2d-model-v1.4.json`), built from the PIC literature review
`modern/docs/literature/pic-mcc-blockers.md`.

## What changed vs. `pic2d_cft_steady_state_v2` (model v1.3)

* **Wall-ion recycling** (review blocker 3): every Xe⁺ absorbed at the dielectric wall or the
  anode returns to the 0-D neutral inventory as a thermal neutral (recombination coefficient
  1.0, wall temperature 400 K). Balance
  `fed + recycled − ionised − effused − artificial = V Δn_g`; fifth ledger `recycled`.
  **Gross** (`S/Q_in`) and **net** (`(S − R)/Q_in`, beam ions per fed atom) utilisation are
  both reported. In v2, 59.9 % of the ionisation was absorbed at the walls and lost from the
  atom balance.
* **Peak-node Debye gate, fail-closed** (blocker 1): at every series record the densest node
  holding ≥ 32 macro-electrons is located and `max(Δr, Δz)/λ_D` there must stay ≤ 4.5.
  Expected ≈ 3.7 at the recycled fixed point on this grid (v2 window-average peak: 3.17). The
  campaign gate is 2.0 (Brandt et al. 2016) on the 30 µm grid; this development grid is
  declared under-resolved at the peak.
* **Grid-heating triad** recorded and gated: energy-ledger residual / electrode work (≤ 10 %),
  dense-cell `T_e` and `S` drifts, peak `ω_pe Δt` drift (soft 5 % = plateau precondition,
  hard 25 % = fail-closed stop after one transit).
* **Artificial neutral relaxation is now an explicit option** (default OFF in the model); this
  development run keeps it **ON** (30 ns) and records it
  (`summary.provenance.v1_4_options.neutral_relaxation`).
* **CUDA-graph replay of the whole step** (blocker 2): bitwise-identical to the direct
  launches; this run's `ms/step` is the 1–2 M particle measurement of the graph speed-up.
* Sensitivity hooks exist but are OFF (Bohm scattering `α`, SEE scaffold).

## Expected fixed point (a-priori budget, `protocol.json` → `budget_v1_4`)

| quantity | v1.3 plateau (v2) | v1.4 expectation |
| --- | --- | --- |
| n_g* | 2.97e19 m⁻³ | ≈ 4.1e19 (3.8–4.5e19); frozen-S bound 4.49e19 |
| S | 3.93e16 s⁻¹ (gross 46 %) | ≈ 5.4e16 (gross 63 %) |
| recycled R | — (2.36e16 lost) | ≈ 3.2e16 |
| net utilisation | ≈ 36 % (implied) | ≈ 25 % |
| n_e mean / peak node | 2.13e17 / 1.64e18 | ≈ 2.9e17 / 2.2e18 |
| cells per λ_D at the peak | 3.17 | ≈ 3.7 (gate 4.5) |
| macro-particles | 2.0 M | ≈ 3.3 M |

Brandt et al. 2016 (different device, static neutrals, Bohm + SEE) report net ionisation
24 % and beam 2.5 mA of 4.3 mA anode current: **model-to-model context labelled by closure,
never validation.**

## Commands (from `modern/`)

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
python -m experiments.pic2d_cft_steady_state_v3.run run       # start / resume (detached launch: see v2 README)
python -m experiments.pic2d_cft_steady_state_v3.run status
python -m experiments.pic2d_cft_steady_state_v3.run finalize  # only for an externally stopped run
```

Artifacts as in v2 (`status.jsonl`, `series.jsonl`, `checkpoint/`, `summary.json`,
`maps.npz`, `series.npz`, `run_state.json`); status lines additionally carry `peak_node`
(n_e, T_e, cells/λ_D, particles at the peak), `recycled_rate_per_s`,
`gross_utilisation` / `net_utilisation` and `grid_heating_triad`.

## Claim boundary

Development/screening run; single seed; 33 × 50 µm grid resolving the peak at 3–4 cells
per λ_D (declared, gated at 4.5); artificial neutral transient (only the fixed point is
physical; no breathing possible under this closure); no ion–neutral collisions, no SEE
emission, no anomalous transport, Dirichlet exit, prescribed B, electrostatic axisymmetric.
