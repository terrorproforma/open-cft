# DEVLOG — mdo_l0_campaign_v1

## 2026-09-03 08:05-09:40 AEST — design, shakedown, preregistration

- Worktree `C:\Users\Angus\Desktop\projects\uni-project-mdo-v1`, branch
  `exp/mdo-l0-campaign-v1` from `origin/feat/sota-foundation` (`cc7706b2`),
  `git ls-files --eol` shows no `w/crlf`.
- Probe of the corrected four-cell solver (`.venv-sota`, `solve_global_discharge_multistart_cpu`,
  3 starts, tol 1e-8) over Ua {150,300,500,1000} V, Ia {0.1,0.5,1,3} A and
  five cusp vectors: 13/80 converged, all with `p = (0,0,0,0)`; nonzero `p`
  ended at `iteration_limit` with residual floors 4.7e-4..0.196; ~3 s per
  solve. Decision: cusp probabilities enter the L0 model through the declared
  closure CL-1 (`S = prod(1 - p_k)` on the ionised fraction).
- Geometry radii excluded from the design vector (no geometry-to-L0 map after
  the v4 falsification of the mirror step); recorded as a structural finding.
- `model.py`: 3 design variables, 7 uniform uncertain inputs, frozen 64-row
  Halton sample (`6e574ff1...`), CVaR(0.25) robust objectives, worst-case
  beam-current margin constraint, nominal re-evaluation, exact numpy
  hypervolume (cross-checked against pymoo to 1e-12 on random 2/3/4-D sets).
- `optimizers.py`: shared LHS initial design per seed (stdlib `random.Random`,
  no scipy: the repository's standard test interpreter has no scipy, and the
  baseline must replay anywhere); BoTorch
  qLogNEHVI through `botorch_adapter.build_qlognehvi` (objective GPs on
  feasible points, constraint GP on all points); pymoo NSGA-III (energy
  reference directions, `eliminate_duplicates=False` for exact budgets);
  two-stage LHS baseline. pymoo placeholders for infeasible individuals are
  never recorded.
- Adapter defects found on first execution: `optimize_acqf_list` resolved
  from the wrong module in BoTorch 0.18.1; sampler replacement after
  construction breaks qLogNEHVI's cached base samples. Both fixed in
  `cft_revival/optimization/botorch_adapter.py`.
- Device timing under the concurrent PIC GPU run: cpu fit 0.3 s / acq 19-29 s
  per iteration (joint q=4, maxiter 100, 32 MC); cuda:0 fit 13.6 s / acq
  224-290 s. Protocol declares cpu; a CUDA float64 probe is recorded.
- Budget sized from the shakedown: 96 evaluations per run (16 + 20x4), 3
  strategies x 3 seeds = 864 evaluations; expected <= ~40 min.
- First shakedown (budget 24, seeds 900101/900202): `accepted_result`,
  bundle validated, all binding gates true, 162 s; BO beat Sobol and NSGA-III
  on both shakedown seeds (non-evidentiary). The design-set invariance test
  was confounded by the moving feasible set and was restricted to the common
  feasible set before the freeze; the shakedown was rerun on the final code.
- Pre-freeze defects caught by the shakedown/tests and fixed (each followed by
  a fresh shakedown): `all_runs_present` compared plan run ids with the
  `kind:` prefix against record keys without it; `plan_record` emitted tuples
  whose typed-canonical tags made the shakedown record un-rehashable
  (`'__cft_type__' is reserved` in `prepare`); the exact-arithmetic front
  invariance check tripped on a one-ulp anode-power tie (L0 recomputes
  `Ua * I_beam / beam_fraction`), now compared with roundoff-aware dominance.
- Final shakedown (seeds 900101/900202, 24 evaluations per run): terminal
  `accepted_result`, bundle validated, 8/8 binding gates, 142 s; non-evidentiary
  outcomes: BO 0.00370/0.00354 vs NSGA-III 0.00201/0.00070 vs LHS
  0.00213/0.00169 hypervolume. Tests: 48 passed / 3 skipped (venv),
  104 passed / 23 skipped (system interpreter, ML runtime absent).
- `prepare`: protocol `09755b85...`, source `da21671f...`, shakedown file
  `8b5a8293...`, evidentiary design `648ad3c5...`, 9 runs. Tests/support commit
  `a1a53300`; preregistration commit follows with the exact subject
  `preregister MDO L0 campaign v1`.

## 2026-09-03 09:33-10:20 AEST — preregistration, execution, record

- `feat/sota-foundation` had moved (`cc7706b2..8babb31e`, pic2d commits);
  both commits were rebased onto `8babb31e` BEFORE the preregistration was
  pushed: tests `fdc6b37d`, preregistration `4898d0fd` (source hash still
  `da21671f...`). Pushed `origin/exp/mdo-l0-campaign-v1`.
- Detached at `4898d0fd`, clean worktree, `run execute` (PID 30936, venv
  interpreter): prebundle re-verified protocol/source/package/shakedown
  authorities; development: cpu + cuda:0 float64 probes, dense reference
  8192 designs in 28.6 s (6576 feasible; robust HV 0.003798, nominal HV
  0.04677; separability spread ~1e-15; replay 32/32); assessment 1600 s.
- Terminal `accepted_result`, manifest `2a326f3c...`, 143 entries, 8/8
  binding gates. Hypervolume: qLogNEHVI 0.003863/0.003877/0.003860,
  NSGA-III 0.002926/0.003505/0.003271, LHS 0.002844/0.003213/0.002804;
  BO beats random 3/3, NSGA-III 3/3; BO seed std 9.2e-6. Pareto sizes BO
  28/30/27, NSGA-III 31/24/32, LHS 19/18/24; infeasible BO 18/14/11,
  NSGA-III 7/17/8, LHS 18/18/19. BO acquisition 510-530 s per seed (GP fits
  7 s, L0 evaluations 0.4 s); NSGA-III/LHS ~1 s.
- Design-set invariance on the common feasible set: identical (exactly, and
  up to ties) for cusp priors U[0, 0], U[0, 0.2], U[0, 0.45], U[0, 0.7];
  unrestricted Jaccard 0.00/0.01/1.00/0.41 because the feasible set moves
  (397/480/644/687 feasible of 734 unique designs).
- Robust vs nominal pooled fronts: 114 vs 62 designs, 24 shared, Jaccard
  0.158; the nominal front reaches efficiency 0.89 and Isp 725 s where the
  robust CVaR front reaches 0.255 and 413 s at the same anode powers.
- Scenarios on the robust-Pareto designs: no_wall_loss S=1 -> 110/114
  infeasible (sampled max S 0.704 bounds the enforced constraint);
  v4_per_cell_jeffreys S=6.9e-8 -> thrust 0.
- Result commit `c553124b` (results force-added past `.gitignore:48
  Results/`; test manifest key fix; instance index pointer). Dashboard
  `visualization/mdo-l0-campaign-v1.html` (340,268 bytes, 137 files
  verified, 0 JS errors headless, screenshots under %TEMP%\mdo_scratch).
- Not observed anywhere in this campaign: any thruster-performance claim.
