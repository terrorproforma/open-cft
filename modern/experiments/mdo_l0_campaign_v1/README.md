# MDO L0 campaign v1

First preregistered multi-objective optimisation campaign on the corrected
physics. It optimises the L0 conservation model (`cft_revival.physics`) under
declared input uncertainty and compares three optimisers at an identical
evaluation budget. **It makes no thruster-performance claim** (see
`protocol.json#claim_boundary`).

## What is optimised

| Item | Declaration |
|---|---|
| Design variables | `discharge_voltage_v` [150, 500] V, `anode_current_a` [0.1, 2.5] A, `propellant_mass_flow_kg_per_s` [2e-7, 2e-6] kg/s (legacy Ua, Ia, mdot clipped to the accepted L0 sweep domain) |
| Excluded | the five legacy radii: no geometry-to-L0 map survives the audit (mirror formula falsified by wall-loss v4) |
| Uncertain inputs | four per-cell cusp probabilities `p_k ~ U[0, 0.45]` (wide prior calibrated so the implied CL-1 survival matches the v4 pooled electron survival 0.357), ionised fraction `U[0.65, 0.98]`, Xe2+ share `U[0, 0.15]`, divergence factor `U[0.75, 0.98]` |
| Closure CL-1 | `S(p) = prod(1 - p_k)`; exhaust ionised fraction `eta_ion * S`; anode current unaffected. Declared, not derived |
| Objectives | `axial_thrust_n` max, `specific_impulse_s` max, `thruster_electrical_to_beam_efficiency` max, `anode_input_power_w` min — all L0 ledgered outputs |
| Constraint | `anode_current_a - max_theta beam_current_a >= 0` (worst case over the frozen 64-point QMC sample; fail-closed, no objectives when violated) |
| Robust objectives | CVaR (worst 16 of 64) per objective |
| Optimisers | BoTorch constrained qLogNEHVI (independent GPs, cpu float64), pymoo NSGA-III, two-stage Latin-hypercube random baseline (stdlib RNG) |
| Budget | 96 evaluations per run, 16 shared initial points, seeds 101/202/303 |

## Lifecycle

```powershell
cd modern
$env:PYTHONPATH = "$PWD\src;$PWD"; $env:PYTHONDONTWRITEBYTECODE = '1'
$py = "C:\Users\Angus\Desktop\projects\uni-project\.venv-sota\Scripts\python.exe"
& $py -m experiments.mdo_l0_campaign_v1.run shakedown   # BEFORE prepare; writes shakedown.json
& $py -m experiments.mdo_l0_campaign_v1.run prepare     # refuses without a passing, hash-matched shakedown
# commit "preregister MDO L0 campaign v1", push, then from a clean detached worktree:
& $py -m experiments.mdo_l0_campaign_v1.run execute
& $py -m experiments.mdo_l0_campaign_v1.run validate
```

The shakedown runs the complete production path (prebundle, development with
the dense reference, assessment with all three optimisers, gates, export)
with a 24-evaluation budget and seeds `>= 900000`; `prepare` binds its file
hash, the protocol hash and the code hash (optimization, active_learning,
surrogates, physics packages, `campaign-v1.json`, and this experiment's
modules) into `authorities.json`; `execute` re-verifies all of them and the
Git-common lock before the single immutable attempt.

## Gates

Binding (terminal state): bit-exact replay of every evaluation, L0 domain
integrity, monotone hypervolume curves, exact budgets, shared initial
designs, frozen sample hash, Pareto replay, code contract. Reported without
tuning: BO beats random (>= 2 of 3 seeds), BO beats NSGA-III, design-set
invariance to the cusp prior on the common feasible set, robust-vs-nominal
front differences.

## Findings recorded before preregistration

* The corrected four-cell discharge solver (`cft_revival.plasma`) does not
  close for any nonzero cusp probability tried (13/80 probe cases converged,
  all at `p = 0`; ~3 s per solve). It cannot serve as the evaluation chain.
* First in-repo BoTorch execution exposed that `botorch_adapter.load_api`
  looked up `optimize_acqf_list` in `botorch.optim`, where 0.18.1 does not
  export it (`botorch.optim.optimize` does); fixed, and `build_qlognehvi`
  gained a `sampler` argument because the sampler cannot be replaced after
  construction.
* GP fits and acquisition on `cuda:0` were 20-40x slower than cpu while the
  RTX 5090 was occupied by the PIC steady-state run; the protocol declares
  cpu and records a CUDA float64 probe only.
