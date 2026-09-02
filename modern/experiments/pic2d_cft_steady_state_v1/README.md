# pic2d CFT steady-state run v1 (model v1.2) — development / screening

One dedicated 2-D axisymmetric PIC-MCC run of the divergent-exit CFT channel at an
operating point sized from the **measured** v2 kinetics, executed as a detached,
checkpointed, resumable background job until the discharge plateaus.

**Status: development/screening. Not preregistered, not validated against
experiment, not a thruster performance prediction.**

**Outcome (no-ignition reference):** at n_g = 1.5e19 m⁻³ the 3 mA beam mirrors back to
the exit plane (≈ 2.9 of 3 mA returned) before it collides; the seed plasma decays and
the discharge current settles at a trivial beam-driven floor. This run is kept as the
static-neutral **no-ignition reference** for model v1.3 (`pic2d_cft_steady_state_v2`,
quasi-steady neutral inventory). It was closed with the runner's `finalize` command
(artifacts from the last checkpoint; the maps are instantaneous checkpoint maps).

`run.py` here is the shared runner: `pic2d_cft_steady_state_v2/run.py` calls it with
its own protocol. Commands: `run` (start/resume), `status`, `finalize`.

## Why the operating point moved (v1.1 → v1.2)

The v2 fine cases (n_g = 1e20 m⁻³, 3 mA) never plateaued because the discharge was
super-critical: the per-electron ionisation frequency ν_iz = S/N_e = 1.2e6 s⁻¹ times
the kinetic ion residence time τ_i,eff = N_i/L = 2.4 µs gives ν_iz τ = 2.9 > 1, so
the plasma inventory grows exponentially (observed e-folding 1.1 µs) with a flat loss
fraction f = L/S ≈ 0.30–0.35. With static neutrals the only equilibrium is
sub-critical (ν_iz τ < 1, needs n_g < 3.4e19) and beam-sustained:

    N_eq = a τ / (1 − ν_iz τ),   a = beam-driven ionisation ≈ 5e16 s⁻¹ at (3 mA, 1e20) ∝ n_g I_inj

At n_g = 1.5e19 and 3 mA: ν_iz τ = 0.44, n_eq ≈ 9.3e16 m⁻³ = 0.23 × the fine-grid
ceiling n_max = 4e17, approach time constant τ/(1 − ν_iz τ) = 4.3 µs. Full table in
`protocol.json` (`kinetic_sizing_from_v2`, `budget_v1_2`) and
`modern/spec/pic2d/pic2d-model-v1.2.json`.

## Running

From `modern/` (PowerShell):

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
$res = "$PWD\experiments\pic2d_cft_steady_state_v1\results"
Start-Process python -ArgumentList "-u","-m","experiments.pic2d_cft_steady_state_v1.run","run" `
    -WorkingDirectory $PWD -WindowStyle Hidden `
    -RedirectStandardOutput "$res\run.log" -RedirectStandardError "$res\run.err"
```

The same command **resumes** from `results/checkpoint/checkpoint-latest.json` if it
exists (bitwise dynamical state; the series history is reloaded from
`series.jsonl`). Add `--ignore-code-identity` after `run` only if the package source
changed since the checkpoint and you accept that.

Progress:

```powershell
Get-Content results\status.jsonl -Tail 1
python -m experiments.pic2d_cft_steady_state_v1.run status   # last line + ETA to 3/5/10 transit times
Get-Content results\run.pid                                    # PID of the running process
python -m experiments.pic2d_cft_steady_state_v1.run finalize   # close a stopped run from its checkpoint (no stepping)
```

## Files under `results/`

| file | content |
| --- | --- |
| `status.jsonl` | one line per 200-step sync: step, t, N_e, N_i, I_d, I_beam,i, peak-node/mean n_e, ⟨T_e⟩, max ω_pe Δt, cumulative wall s, ms/step, plateau drifts |
| `series.jsonl` | full series record per sync (source of `series.npz`) |
| `checkpoint/checkpoint-latest.*` | resumable state, rewritten atomically after every 40 000-step chunk |
| `run_state.json` | cumulative wall time, sessions, last checkpoint step |
| `run.pid`, `run.log`, `run.err` | process id and logs |
| `summary.json`, `series.npz`, `maps.npz`, `checkpoint-final.*` | written on any stop (or by `finalize`; then `maps_kind = instantaneous_checkpoint`) |

## Stopping rule

Plateau: relative drift (linear fit) of both I_d and N_e < 5 % over the trailing
20 % of elapsed simulated time, only after ≥ 3 ion transit times (3 × 2.4 µs =
4.8 M steps). Other stops: 12 h cumulative wall budget, the fail-closed runtime
stability gate, or `--max-steps`. Every stop writes the artifacts and exits 0.
