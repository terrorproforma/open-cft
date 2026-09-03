# MDO L0 campaign v1 posthoc audit

## Verdict

Evidence **ACCEPTED WITH DISCLOSURES** as
`l0_model_robust_multiobjective_optimisation_under_declared_input_uncertainty_not_thruster_performance`.

The bundle is byte-exact on this LF checkout (137/137 files, 0 EOL-only cases,
0 mismatches), the preregistration hash chain closes from Git blobs alone, every
recorded evaluation replays bit-exactly through the package and to 3e-13
relative through an independent re-implementation of the evaluation chain, the
optimiser trajectories replay bit-exactly with the pinned runtime, and every
statistic in the results (hypervolumes, Pareto sets, paired comparisons, the
Jaccard index, the prior-sensitivity and scenario tables, the Wilson interval)
recomputes with an independent implementation. No defect in the evidence was
found. Six disclosures are recorded below (F9, F10, F22, F26, F27, F28); none
changes a number, a gate or the classification, and none requires a change to
the frozen files or to `results/`. What a v2 must change is listed in section 6.

## Immutable bindings

- preregistration commit: `4898d0fd3decddc5f308072e724d1936660c00e9`
  (`preregister MDO L0 campaign v1`, authored 2026-09-03T09:34:16+10:00,
  committed 09:34:18 after the rebase onto `8babb31e`; 11 files, all under
  `modern/experiments/mdo_l0_campaign_v1/`, no `results/`);
- result commit: `c553124b7393890d8ee9c6fc022e536c8a1fd35e`
  (`record MDO L0 campaign v1 result`; parent `4898d0fd`; 138 files under
  `results/` + 2 files outside it, see F9);
- dashboard commit `e642f38cd613e3d687c32777080d8aefae93c7b3`; paper gate
  `GATE-MDO-L0-V1` (`numerical-campaign`, `opens_level: null`) at
  `ba6875f604746e8fbeaf2aee2bdf06b8f06bdc04`;
- `results/` tree `89e6e69f861fa201f8ad91ca9635577eba44a683` (identical at
  `c553124b` and at this overlay's `HEAD`; no later commit touches it);
- `results/manifest.json` SHA-256
  `2a326f3cf2e286a0ba8f9c91871d78b935a11e3e2c56bd9164acdc6166485381`
  (143 entries: 137 files + 6 required directories; `lock_byte_sha256`
  `194fabb7e750fce552fa99930a46cdd3805b16206cd66f82062b54d9cd7599fa`,
  `terminal_byte_sha256`
  `2e96cbfe1c4f3432f4b25cd31a726e7e1c63f1e32bb02cd4b72a0e978d80b680`,
  `transition_log_sha256`
  `ae74b3d651e8988bc28b08f67faeae4722a835805ca310eb043d8f44413efcec`);
- frozen inputs at `4898d0fd` (blobs identical at `HEAD`): `protocol.json`
  `06eef451c8c5f3a3b161f893fc9116787caf2c4c`, `authorities.json`
  `6d2ba9a87327ba66e626e7ae98031ed9c5392953`, `shakedown.json`
  `bec04e5b7df0c2210942bc58de68a542e43751a1`; frozen code `model.py`
  `b300e0a68e8d06edbc62f857f37e8f232f8c6253`, `optimizers.py`
  `431116046125da84b705937f7df83892a9abb122`, `experiment.py`
  `bf5037e553ed4d8af5d8dc9d05b1bad94854ce66`, `run.py`
  `8c2d241dad2375fa0d82212c08ebb4bb4ab05376`;
- protocol semantic SHA-256
  `09755b85393d3b3248941ce52f8c21edb832ce30c6f31e5c6919079c41d496ba`;
  code-contract source SHA-256 (37 files, LF)
  `da21671f9661f183f3f980044e000a1fdfd0d3495782ed3b0e1fbf5763a9682e`;
  shakedown file SHA-256
  `8b5a829302e7aa800d2c60ca1146d86195a71594482c1699a2698e79d76d5c1e`;
  frozen QMC sample SHA-256
  `6e574ff122894e0facf951cdf89069c1b4625d6082a33b7026ff4d8a776db33e`;
- pinned runtime: Python 3.12.10, torch 2.13.0+cu130, botorch 0.18.1,
  gpytorch 1.15.2, pymoo 0.6.2, numpy 2.5.2, scipy 1.18.1; GP fits and
  acquisition on `cpu` float64 (declared; a CUDA float64 probe on the RTX 5090
  is recorded in `artifacts/device-probes.json` but no CUDA work was done).

## 1. Findings

Every row is produced by `audit_replay.py` (read-only) and re-derived by
`tests/experiments/mdo_l0_campaign_v1/test_posthoc_audit.py`. "PASS" means the
recorded value was reproduced; "DISCLOSURE" means a fact the reader must know
that does not change any recorded number; "FAIL" does not occur. Rows F3 and
F24 depend on the machine (Git-common lock file; optional ML stages) and are
compared by id only in the test.

| id | check | observed | verdict |
| --- | --- | --- | --- |
| F1 | manifest entries byte-exact on this LF checkout | 137 byte-exact, 0 EOL-only, 0 mismatch of 137 files (+6 dirs); 68 sidecar pairs consistent=True; CR bytes in 0 files | PASS |
| F2 | lock / terminal / transition-log hashes and ordering | lock=True, terminal=True, transitions 1..9 contiguous=True, 13 access records before their operations=True, counters precede access=True, run spacing >= wall clocks=True | PASS |
| F3 | Git-common execution lock | content == prereg commit: True, created 0.011 s before the runtime lock | PASS |
| F4 | preregistration commit isolation / push / no results | isolated=True, no results=True, on origin/exp branch=True, result parent is prereg=True | PASS |
| F5 | protocol / authorities / shakedown bindings | protocol semantic == authorities == shakedown record: True; shakedown file sha == authorities: True; sealed copies equal frozen files: True | PASS |
| F6 | shakedown non-evidentiary and disjoint | evidentiary=False, outcomes_enter_estimand=False, seeds [900101, 900202] vs [101, 202, 303], initial-design overlap recomputed=0, temp result root=True | PASS |
| F7 | code hash from Git blobs at prereg/result/dashboard/paper commits | 4898d0fd=True, c553124b=True, e642f38c=True, ba6875f6=True; working tree=True; shakedown head package files 34/34 | PASS |
| F8 | no hashed-source or frozen-file change since prereg; results tree immutable | hashed sources untouched=True, frozen files untouched=True, results tree 89e6e69f861f unchanged=True, non-scoped deps unchanged=True | PASS |
| F9 | files outside results/ in the result commit | modern/spec/optimization/mdo-l0-campaign-v1.json, modern/tests/experiments/mdo_l0_campaign_v1/test_mdo_v1_results.py | DISCLOSURE |
| F10 | hash scope vs modules actually imported | never imported but hash-bound: ['cft_revival.active_learning', 'cft_revival.surrogates']; imported but not hash-bound: ['cft_revival.experiment_runtime', 'cft_revival.kernels', 'cft_revival.models'] | DISCLOSURE |
| F11 | frozen QMC sample and nominal point (independent radical inverse) | sample bit-exact=True, nominal bit-exact=True, survival 0.1556..0.7042 (mean 0.3598) | PASS |
| F12 | shared 16-point initial design per seed (stdlib LHS) | 9/9 runs bit-exact | PASS |
| F13 | LHS baseline: all 96 designs per seed | 3/3 seeds bit-exact | PASS |
| F14 | 864 records through an independent CL-1 + L0 + CVaR implementation | worst relative difference 3.1e-13 (tolerance 1e-12); status counts {'success': 734, 'infeasible': 130}; fail-closed inconsistencies 0; out of bounds 0; non-finite 0 | PASS |
| F15 | 864 records through the package (model.evaluate_design) | 864 replayed, 0 mismatches, design ids recompute=True | PASS |
| F16 | Pareto sets (pairwise dominance) and final hypervolumes (WFG) | Pareto indices equal 9/9=True; HV worst relative 4.4e-16 (2/9 bit-exact); curve spot checks worst 6.2e-16 | PASS |
| F17 | BO beats random / NSGA-III (paired by seed, independent HV) | 3/3 and 3/3; one-sided sign-test p = 0.125; the predeclared >=2/3 rule passes with probability 0.5 under a no-difference null | PASS |
| F18 | pooled robust vs nominal fronts | robust 114 vs nominal 62, shared 24, Jaccard 0.157895 (recorded equal=True); HV rel diff 1.8e-15 / 1.7e-16 | PASS |
| F19 | dense 8192-point reference (independent fronts/HV from the sealed columns) | feasible 6576, infeasible 1616, all within bounds=True; robust front 291 (recorded 291), HV rel diff 1.0e-15; nominal front 166 (recorded 166), HV rel diff 7.4e-16; 256 designs spot-replayed independently (worst 1.3e-15); BO mean attains 1.0181 of the reference | PASS |
| F20 | prior-sensitivity table (independent re-evaluation of 758 designs x 4 priors) | a=0.0: feasible 397, front 74, common-set identical True, Jaccard 0.000; a=0.2: feasible 480, front 77, common-set identical True, Jaccard 0.011; a=0.45: feasible 644, front 114, common-set identical True, Jaccard 1.000; a=0.7: feasible 687, front 103, common-set identical True, Jaccard 0.409; all match recorded=True | PASS |
| F21 | scenario table (114 robust-Pareto designs x 5 scenarios) | Jeffreys S = 6.858e-08, thrust max 2.703e-09 N; no_wall_loss 4 evaluated / 110 infeasible; all match recorded=True | PASS |
| F22 | Jeffreys rule vs frozen scenario numbers | rule rounded to 4 dp [0.5711, 0.9996, 0.9996, 0.0004] vs frozen [0.5712, 0.9996, 0.9996, 0.0004]; S(frozen) 6.8581e-08 vs S(rule) 8.0618e-08 | DISCLOSURE |
| F23 | Wilson interval / prior calibration | Wilson(330/512) equals protocol authority=True; implied survival 0.3608 vs v4 0.3572 (gap 0.0035 < 0.005: True) | PASS |
| F24 | package replays (pinned runtime `.venv-sota`, cpu; recorded from `audit_replay.py --nsga3 --dense --bo 101`) | dense: bit-exact=True in 52 s (8192 designs, every column, fronts, separability report); NSGA-III: bit-exact 3/3 seeds in 1.4-2.2 s each (96 records, curves, Pareto indices); qLogNEHVI seed 101: bit-exact=True (96 records, 96-point curve, final HV 0.0038634857735177987, Pareto indices, all 20 acquisition values, max abs diff 0.0), wall 633.8 s vs 521.7 s recorded; a second, earlier run under heavy CPU contention was also bit-exact in 3708.2 s (section 2) | PASS |
| F25 | claim boundary consistency | classification identifier in 6/8 documents (boundary sentence only in ['README.md', 'spec/optimization/mdo-l0-campaign-v1.json']); campaign-result claim boundary == protocol: True; geometry exclusion in protocol/sealed protocol/paper: True; CLM-030 non-claims cover performance+geometry: True; gate numerical-campaign opens_level=None; campaign-v1 benchmark null=True | PASS |
| F26 | binding gates are recording-integrity gates | 8/8 passed; none is an outcome gate (see semantics); acceptance = pipeline integrity, efficacy statements are reported-not-binding | DISCLOSURE |
| F27 | NSGA-III duplicate evaluations (eliminate_duplicates=false) | nsga3:101: 2, nsga3:202: 3, nsga3:303: 5 | DISCLOSURE |
| F28 | descriptive labels in run artifacts | qLogNEHVI optimizer.acquisition says 'sequential greedy batch' while protocol/code use joint q (optimize_acqf sequential=False); pymoo reports generations_completed {7} for 6 declared generations (provenance generations 0..5, 96 evaluations); iterations {20}, device {'cpu'} | DISCLOSURE |

## 2. Replay: what is bit-exact and what is within tolerance

All replays were run on this LF checkout (`audit/mdo-l0-v1-posthoc` from
`ba6875f6`) on the producing host (`DESKTOP-31AD96J`, 24 CPUs, Windows 11
10.0.26200), with the RTX 5090 occupied by a PIC run and twelve unrelated
worker processes saturating the CPU. Nothing was run on CUDA.

**Bit-exact (byte-for-byte or float-for-float equality with the sealed artifacts)**

- the frozen 64-row QMC sample and the nominal point, regenerated by an
  independent radical-inverse implementation (F11), and their SHA-256 through
  the package (`sample_hash`);
- (a) the shared 16-point initial design of every seed, regenerated from
  `random.Random(seed)` with an independent LHS implementation, for all
  9 runs (F12);
- (b) all 96 LHS designs of each seed (F13), and all 96 NSGA-III records of
  each seed re-run through pymoo 0.6.2 (`opt.run_nsga3`, energy reference
  directions seed 1, `eliminate_duplicates=False`): design values, margins,
  status, robust statistics, nominal objectives, sample-result hashes,
  batch/provenance labels, the 96-point hypervolume curve and the Pareto
  indices, 3/3 seeds, 0.9-1.5 s each (F24);
- (c) qLogNEHVI seed 101 re-run end-to-end through BoTorch 0.18.1 /
  torch 2.13.0+cu130 on cpu float64 with the protocol's settings
  (`torch.manual_seed(101)`, SobolQMCNormalSampler seed `101000 + it`,
  `optimize_acqf` joint q=4, 4 restarts, 128 raw samples, L-BFGS-B
  maxiter 100, batch_limit 5): all 96 records (including the 80 GP-chosen
  candidates), the 96-point hypervolume curve, the Pareto indices and all
  20 recorded acquisition values equal the sealed run (max abs difference
  0.0; torch 24 threads; F24). The replay was run twice with identical
  results. The recorded run (`audit_replay.py --nsga3 --bo 101`) took
  633.8 s against 521.7 s for the campaign (acquisition 18.9-45.0 s per
  iteration against 16.6-42.2 s; GP fits 0.3-0.9 s against 0.3-0.7 s) while
  twelve foreign worker processes and the PIC run were still active. An
  earlier run of the same `opt.run_qlognehvi` call took 3708.2 s because its
  first acquisition alone took 2706.9 s (19.7 s recorded) while those
  processes and this audit's other stages saturated the 24 CPUs; its
  iterations 2-20 took 22.6-310.3 s. Wall time is therefore the only quantity
  that did not reproduce, and it is not a recorded estimand. The arithmetic
  was unaffected (BoTorch's fused qLogEHVI C++ extension is absent in both
  the campaign and the replays; the pure-Python fallback was used
  throughout). Seeds 202 and 303 were not re-run through BoTorch (the
  protocol's `seed_policy` records the candidate values so the L0 replay
  never depends on acquisition reproducibility; F15 covers all 288 of their
  records bit-exactly);
- (d) the dense 8192-point reference through the package
  (`experiment.dense_reference(8192, 20260903)`): every column of the sealed
  compact record (values, design ids, status, both margins, robust and
  nominal objectives, sample-result hashes), the robust/nominal fronts
  (291/166 designs, HV 0.003797983245976796 / 0.04676913466378779) and the
  separability report (F24);
- all 864 evaluation records through `model.evaluate_design` on this checkout
  (F15), the pooled and per-strategy fronts (`experiment.pooled_fronts`) and
  the full sensitivity artifact (`experiment.cusp_sensitivity`, 13.9 s);
- the Wilson interval of the v4 coupling export, 330/512 ->
  [0.6021349532568827, 0.6847749053232215] (F23);
- the paired counts (3/3 and 3/3), the front sizes 114/62, the 24 shared
  designs and the Jaccard index 24/152 = 0.15789473684210525 (F17, F18), the
  prior table's feasible/front/common-set counts and Jaccard values (F20), the
  scenario table's evaluated/infeasible counts (F21).

**Within tolerance (independent implementation, different operation order)**

The audit's own evaluation chain (plain IEEE-754 arithmetic: CL-1 survival,
charge fractions, beam current, `v = sqrt(2 z e U / m)`, thrust, Isp,
`U I_beam / (U I_a + 15 W)`, `U I_a`; CVaR = mean of the 16 worst of 64) is
compared with the sealed numbers at the relative tolerance 1e-12 declared by
this audit (the protocol declares bit-exactness only for the package replay,
which is met above):

- 734 successful records: robust objectives 4.4e-16, robust statistics
  5.8e-16, nominal objectives 5.3e-16; both margins on all 864 records
  3.1e-13 (the margin is a difference of nearly equal currents, so the
  relative figure is inflated near zero; every sign agrees, F14);
- final hypervolume of every run by the WFG exclusive-hypervolume recursion
  versus the package's slicing algorithm: worst 4.4e-16, 2 of 9 exactly equal
  (F16); every eighth point of every curve: 6.2e-16; pooled robust/nominal
  hypervolumes 1.8e-15 / 1.7e-16 (F18); dense robust/nominal 1.0e-15 /
  7.4e-16 (F19); the four prior hypervolumes <= 1.3e-15 and the five scenario
  hypervolumes <= 1.0e-15 (F20, F21);
- 256 dense designs spot-replayed independently: 1.3e-15 (F19).

## 3. Preregistration and code integrity (from Git alone)

- The code-contract hash recomputed from the Git blobs of the 37 hashed files
  at `4898d0fd`, `c553124b`, `e642f38c` and `ba6875f6` equals the recorded
  `da21671f...` with identical per-file entries; the working tree gives the
  same value; no commit after `4898d0fd` touches a hashed source, a frozen
  file or `results/` (F7, F8). `cft_revival.experiment_runtime`,
  `cft_revival.models` and `cft_revival.kernels` (imported but outside the
  hash scope, F10) are also unchanged between `4898d0fd` and `HEAD`, so this
  checkout runs exactly the executed code.
- The shakedown record (`generated_at 2026-09-02T23:33:01Z`) was produced at
  `a1a53300` with only the nine then-untracked experiment files dirty; the 34
  package files of the hash scope are identical at `a1a53300`, and `a1a53300`
  is the same tests commit as `fdc6b37d` (= `4898d0fd^`) before the rebase
  onto `8babb31e` (same subject and author date; no difference under
  `modern/src`, `modern/spec/optimization` or the experiment directory). The
  shakedown's protocol semantic hash and source hash equal the frozen
  authorities, so no protocol or code edit happened between the recorded
  shakedown and the freeze; the earlier shakedowns the experiment DEVLOG
  mentions (the first one and one after each fixed pre-freeze defect) are not
  recorded anywhere and are not evidence; only the last one is bound.
- `_bind_preregistration` conditions: HEAD subject, push to
  `origin/exp/mdo-l0-campaign-v1`, experiment-path isolation and absence of
  results are verifiable from history (F4). Detached HEAD and a clean worktree
  are enforced by `run.py` at execution time and cannot be re-verified from
  history; `run.py` is not in the hash scope (F10) but its blob at `4898d0fd`
  is the one in this checkout, and the corroborating facts are the Git-common
  lock (content `4898d0fd`, created 11 ms before the runtime lock, F3), the
  runtime lock's `commit`, the 37 s between the rebased commit (09:34:18) and
  the lock (09:34:54.98), and the code-contract gate re-verified inside the
  run. `clean_worktree_attested: true` is a runtime constant, as the shakedown
  record itself discloses.
- Access records: 13, each `recorded_before_operation: true`, each preceded by
  its `before-<operation>` counter and each phase access preceding the
  `<phase>-started` transition; the gap between consecutive solver accesses
  is never smaller than the recorded wall clock of the run it announces
  (qLogNEHVI 521.75/536.63/517.94 s vs 521.71/536.60/517.90 s recorded).
  Lock to terminal 1635.5 s; development 34.7 s; assessment 1600.5 s
  (BO acquisition 514.2 + 529.0 + 509.7 s, GP fits 6.8 + 6.8 + 7.4 s).

## 4. Disclosures

1. **F9 - two files outside `results/` in the result commit.**
   `modern/spec/optimization/mdo-l0-campaign-v1.json` changed `"results": null`
   to the recorded bundle pointer (manifest SHA-256, terminal state,
   preregistration commit, counts), which the protocol itself announces
   (`prior_model_disclosure.campaign_spec_v1_4`); and
   `modern/tests/experiments/mdo_l0_campaign_v1/test_mdo_v1_results.py` changed
   the manifest key it reads from `entry["sha256"]` to `entry["byte_sha256"]`
   plus a length check. The second is a bug fix to a test that would have
   failed against any real bundle; it touches no hashed source and no frozen
   file, and `campaign-v1.json#benchmark.results` stays null. Neither file
   feeds the evidence, but "no code change outside results/" is not literally
   true and is recorded here.
2. **F10 - hash scope partly inert, partly incomplete.**
   `cft_revival.active_learning` (8 files) and `cft_revival.surrogates`
   (10 files) are hash-bound but never imported by the campaign;
   `cft_revival.experiment_runtime` (which defines the canonical bytes behind
   `sample_sha256`, `sample_result_sha256`, `design_sha256` and every artifact
   hash), `cft_revival.models` and `cft_revival.kernels` are imported but not
   hash-bound; `run.py` (lock and preregistration binding) is not hash-bound.
   The commit pin closes the gap for this campaign (section 3), not the
   contract.
3. **F22 - the Jeffreys scenario numbers are not the stated rule.** The
   protocol declares `v4_per_cell_jeffreys` as `(successes + 0.5) /
   (trials + 1)` per cell and freezes `[0.5712, 0.9996, 0.9996, 0.0004]`. The
   rule gives 0.571118 (rounds to 0.5711, not 0.5712), 0.999566, 0.999566 and
   0.000434. Because S is dominated by `(1 - p_2)(1 - p_3)`, the frozen
   numbers give S = 6.8581e-08 while the unrounded rule gives 8.0618e-08 (17 %
   apart); both give a beam that is ~1e-7 of the produced ions and thrust
   below 3 nN, so the recorded reading ("the v4 wall-hit estimand cannot be
   the Kornfeld cusp probability of a sustained discharge") is unchanged. The
   frozen numbers are what was evaluated and what replays; the paper quotes
   the frozen S.
4. **F26 - acceptance is an integrity verdict, not an efficacy verdict.** All
   eight binding gates test that the recording is internally consistent
   (replay, fail-closed rule, budgets, shared initial designs, frozen sample
   hash, Pareto replay, code contract already enforced at prebundle) or are
   implied by construction (a hypervolume of a growing set is monotone for a
   correct implementation). `accepted_result` therefore means "the declared
   pipeline ran once and recorded consistently"; every statement about the
   optimisers is in `reported_not_binding`, exactly as the protocol's
   `terminal_rule` says. The reported paired rule (>= 2 of 3 seeds) passes
   with probability 0.5 under a no-difference null; the observed 3/3 has a
   one-sided sign-test p of 0.125. The bundle and the paper claim counts, not
   significance, which is correct; a reader should not infer more.
5. **F27 - NSGA-III re-evaluated 2, 3 and 5 duplicate designs** (seeds
   101/202/303; `eliminate_duplicates: false` is declared), so its unique-design
   budget was 94/93/91 of 96 while qLogNEHVI and LHS used 96 unique designs
   each. The duplicates are exact re-evaluations (identical objective vectors)
   and cannot change a front; the pooled unique count is 864 - 96 (shared
   initial designs counted three times) - 10 = 758.
6. **F28 - two descriptive labels in the run artifacts are wrong or
   misleading**, the numbers are not: `artifacts/runs/qlognehvi-*.json#
   optimizer.acquisition` is the hard-coded string "..., sequential greedy
   batch" while the protocol declares and the code executed joint-q
   `optimize_acqf(sequential=False)` (the replay in F24 confirms the joint
   path reproduces the candidates); `artifacts/runs/nsga3-*.json#
   optimizer.generations_completed` is pymoo's post-incremented counter (7)
   for the 6 declared generations (provenance labels `generation=0..5`,
   `pymoo_reported_evaluations` 96).

Also observed and found consistent (not disclosures): the experiment README
and the spec index carry the boundary sentence "no thruster-performance
claim" rather than the classification identifier (F25); `label_access_count`
is 0 (the campaign has no label operations); the sealed
`artifacts/shakedown.json` is a `write_blob` copy with `semantic_sha256: null`
in its sidecar (byte hash bound; bytes equal the frozen file); `results/` is
masked by the case-insensitive `.gitignore:48 Results/` rule and was
force-added (138 tracked files, as for the v3/v4 wall-loss bundles); the
experiment DEVLOG's phrase "of 734 unique designs" is a slip for the 758
unique designs (734 is the count of successful evaluations), a documentation
file that is neither frozen nor evidence; the summary figure "thrust
<= 2.70e-9 N" for the Jeffreys scenario is the 3-significant-digit rounding of
the recorded maximum 2.7027199906330584e-09 N (which is above 2.70e-9 N by
0.1 %), and "Jeffreys S = 6.86e-8" rounds 6.85805567999849e-08.

## 5. Physics domain, closure and claim boundary

- Every one of the 864 campaign designs and the 8192 dense designs lies inside
  the declared domain `[150, 500] V x [0.1, 2.5] A x [2e-7, 2e-6] kg/s`; no
  NaN or inf anywhere; 130 infeasible evaluations, each with a negative robust
  margin recomputed independently, failure code
  `beam_current_exceeds_anode_current`, no robust objectives and no
  sample-result hash; 734 successful evaluations, each with a non-negative
  margin and finite objectives; nominal objectives present iff the nominal
  margin is non-negative (F14).
- Closure CL-1 is applied exactly as written: `S = prod(1 - p_k)`,
  `eta_m = eta_ion * S`, neutral `1 - eta_m`, Xe2+ `eta_m * zeta`, Xe+ the
  remainder, beam current `mdot (e / m_Xe)(f_+ + 2 f_2+)`, anode current
  untouched; the independent chain that codes only these formulas reproduces
  every sealed objective to <= 5.8e-16 relative (F14), the survival range of
  the frozen sample is 0.1556..0.7042 (mean 0.3598; the "worst sampled case"
  0.704 the README warns about), and the `no_wall_loss` scenario (S = 1,
  outside the sampled support) makes 110 of the 114 robust-Pareto designs
  infeasible exactly as recorded (F21).
- The classification identifier is present in `protocol.json`, the sealed
  `artifacts/protocol.json` and `artifacts/campaign-result.json`, the
  dashboard, `paper/evidence/claims.json` (CLM-029, 030, 032, 033, 034),
  `result-gates.json` and the paper manifest; `campaign-result.json` carries
  the protocol's claim-boundary statement verbatim and the closure id; the
  protocol's four `forbidden_readings`, CLM-030's six `non_claims` ("makes no
  thruster-performance ... claim", "says nothing about geometry") and CLM-034's
  "geometry variables are excluded because no geometry-to-model map survives
  the audit" agree with `protocol.json#claim_boundary.why_geometry_variables_are_excluded`
  and the five `excluded_legacy_variables`; the gate is `numerical-campaign`
  with `opens_level: null`; `campaign-v1.json#benchmark.results` is null (F25).

## 6. What a v2 must change (no change to v1 evidence)

1. Bind `cft_revival.experiment_runtime`, `cft_revival.models`,
   `cft_revival.kernels` and `run.py` into `code_contract.source_hash_scope`
   and drop the never-imported packages, or state that the commit pin is the
   binding (F10).
2. Freeze scenario probabilities as the unrounded rule values (or state the
   rounding) so the stated rule and the frozen numbers coincide (F22).
3. Emit the acquisition description from the actual `optimize_acqf` arguments
   and record pymoo's generation count as `declared_generations` plus
   `pymoo_n_gen` (F28).
4. Either set `eliminate_duplicates=True` for NSGA-III or report the unique
   design count next to the evaluation count (F27).
5. Add an outcome-level binding gate if acceptance is meant to carry any
   efficacy meaning, or keep the present wording and add the seed count to
   every "BO beats" sentence (F26); three seeds cannot support a significance
   claim and the paper makes none.
6. Keep the result commit strictly to `results/` (move pointer and test
   updates to a following commit) so "no change outside results/" is literally
   true (F9).

## 7. Reviewer reproduction

From `modern/` (PowerShell shown; the script is read-only on `results/`):

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"; $env:PYTHONDONTWRITEBYTECODE='1'
# must print nothing: the bundle is LF on disk
git ls-files --eol -- experiments/mdo_l0_campaign_v1/results | Select-String 'w/crlf'
# the table in section 1 (system interpreter, no ML runtime needed, ~1 min)
python -m experiments.mdo_l0_campaign_v1.audit_replay --table
# full JSON report; exit code 0 iff passed (disclosures never fail it)
python -m experiments.mdo_l0_campaign_v1.audit_replay --json $env:TEMP\mdo-audit.json
python -m pytest tests/experiments/mdo_l0_campaign_v1/test_posthoc_audit.py -q
# pinned runtime: NSGA-III (seconds), dense reference (~1 min) and one BO seed (~9 min idle cpu)
$py = "C:\Users\Angus\Desktop\projects\uni-project\.venv-sota\Scripts\python.exe"
& $py -m experiments.mdo_l0_campaign_v1.audit_replay --table --nsga3 --dense --bo 101
```

The report must show `bundle.counts = {byte_exact: 137, eol_only: 0,
mismatch: 0}`, `independent.status_counts = {success: 734, infeasible: 130}`,
`package.records_864.mismatches = []`, `failures = []`, `disclosures = [F9,
F10, F22, F26, F27, F28]` and `passed: true`. With the pinned runtime the
`--nsga3`, `--dense` and `--bo 101` stages must each report `bit_exact: true`.
The test module additionally re-derives every figure above from the bundle,
proves the script left `results/` byte-identical, checks the §1 table against
a live recomputation, and asserts the results tree hash and the frozen blobs
unchanged.

## 8. Interpretation (interpretation, not evidence)

*This section interprets the accepted numbers and carries no evidentiary
weight.* The campaign shows that, on the corrected L0 model under CL-1 and the
declared priors, constrained qLogNEHVI reaches the 8192-point dense-reference
robust hypervolume with 96 evaluations while NSGA-III and a Latin-hypercube
baseline reach 0.75-0.92 of it; that is a statement about optimisers on a
cheap, separable, three-variable model with one active constraint, and the
separability report (ratio spread ~1e-15) shows why the robust front is the
nominal front's feasible subset ordering rather than a new trade-off. The
robust/nominal split (114 vs 62 designs, 24 shared) is driven entirely by the
anode-current margin under the worst sampled cusp losses, so it measures the
prior's width, not any physics. Nothing here bears on a thruster.

## 9. Overlay contents

This audit adds only `POSTHOC_AUDIT.md`, `audit_replay.py` and
`tests/experiments/mdo_l0_campaign_v1/test_posthoc_audit.py`. No file under
`results/`, no frozen preregistration file, no hashed source and nothing under
`FYP/` is modified by it; the results tree hash and the frozen blobs are
asserted unchanged by the test.
