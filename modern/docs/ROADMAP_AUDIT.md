# Final roadmap audit against committed evidence

Audited tree: `origin/feat/sota-foundation` at `44b7c8dcd1b5c1392730fe8592cf777dee19a5a3`
(`feat(pic2d): resumable steady-state runner with plateau criterion`, 2026-09-03 07:19 +1000),
checked out fresh into a dedicated worktree (`docs/roadmap-audit`). Audit date 2026-09-03.
The audit commit itself sits on top of `67b04f87` because two pic2d commits landed while it
was written; they are assessed in section 7 and change no verdict.
Every number below was read from a committed file, a Git object, or a test run executed for
this audit; nothing is taken from memory or from the live tracker without re-verification.
Paths are relative to the repository root unless stated. Where the tracker canvas
(`open-cft-roadmap-status.canvas.tsx`) disagrees with the committed evidence, the
disagreement is stated explicitly (section 6).

Verdict vocabulary used throughout:

- **ACCEPTED evidence** — preregistered or independently audited, hash-bound, replayed, and
  (where applicable) admitted to the paper claim matrix at its recorded outcome.
- **RECORDED-DEVELOPMENT** — executed and committed, but not preregistered or not
  independently accepted; bounds a problem, creates no claim.
- **SCREENING-ONLY** — a run whose own artifacts label it screening / `NOT_EVALUATED` /
  `SCREENING_NOT_ACCEPTED`.
- **NOT DONE** — no committed artifact exists.
- **OUT OF SCOPE** — deliberately excluded by the recorded design decisions.

Test runs performed for this audit (fresh worktree, system Python 3.12.10, pytest via
`modern/pyproject.toml` `pythonpath = ["src", "."]`, `--import-mode=importlib`):

| Suite | Command (from) | Result |
|---|---|---|
| `modern/tests` | `python -m pytest tests -q -p no:cacheprovider` (from `modern/`) | **1702 passed, 5 skipped, 0 failed**, 7 warnings, 930.03 s (15 min 30 s); 1707 collected |
| `paper/tests` | `python -m pytest paper/tests -q -p no:cacheprovider` (from repo root) | **58 passed, 298 subtests passed**, 73.49 s |

The five skips are the documented ones (`modern/docs/workstreams/test-health-devlog.md:128-131`):
Windows directory-symlink privilege (`tests/experiment_runtime/test_canonical_and_filesystem.py:268`),
orbit v4 audit overlay commit (`tests/experiments/cft_orbit_wall_loss_v4/test_posthoc_audit.py:385`),
field-surrogate v1 pre-execution and v2 no-terminal-failure (`tests/experiments/l1a_field_surrogate_v{1,2}/test_results.py`),
optional pybind11 extension (`tests/test_kernels.py:71`). The count has risen from the 1677
recorded at `7a30fc2e` (`test-health-devlog.md:128`) because the pic2d phase-2/3 tests
(`tests/pic2d/test_pic2d_v11_step.py`, `test_pic2d_steady_state_runner.py`) landed since.

---

## 1. Executive verdict

The rebuild has produced a large, well-instrumented numerical foundation and a small body of
accepted evidence — four accepted numerical results and two accepted nulls — none of which is
a thruster-performance result. **Genuinely done (ACCEPTED evidence):** the legacy MATLAB audit (`modern/docs/AUDIT.md`, 9 confirmed defects,
all either fixed in the corrected formulations or quarantined and superseded, none silently
reproduced); the SI-explicit L0 conservation model with 8,192-point CPU/CUDA parity
(`modern/docs/FIRST_RESULTS.md`); the preregistered 96-design L1a field-only sweep v2
(`f30cb42e`, 96/96, seven gates, admitted as `GATE-L1A-SWEEP-V2`); the audited P2 adaptive
FEM qualification of one of three designs (`a1158bad`, divergent-exit-stack
`NUMERICAL_P2_QUALIFIED`); the preregistered full-orbit wall-loss campaign v4 (`6922a3cf`,
audited `258f69b2`, 15/15 binding gates, 4608 orbits, admitted as `GATE-WALL-LOSS-V4`); and
two preregistered/recorded topology nulls (four-cell v2 `7120e8ed` 0/128 stable;
characterization v1 `3ce6c546` 0 cusps / 0 cells) admitted to the manuscript as nulls. The
reproducibility infrastructure is real: the whole `modern/tests` tree is green in one
invocation on a fresh checkout (1702 / 0 / 5) and the paper checker passes (58 / 298
subtests). **Development-only (RECORDED-DEVELOPMENT / SCREENING-ONLY):** the verified 1-D
and 2-D PIC-MCC solvers, whose three executions reached no plateau (snapshot v1 fail-closed at
49-60 ns; snapshot v2 at 0.93-1.53 ion transits with I_d drift 9-22 %; steady-state v1 still
running uncommitted and, per the working tree, not igniting); the L1b material-field solver
(all 54 publication gates `NOT_EVALUATED`); the nine L0 surrogate campaigns and ten L1a
field-surrogate campaigns, all of which ended in negative or invalidated outcomes; the
coupling v4.2 held-out promotion, attempted seven times and never achieved; and the corrected
global-plasma network, which has no physical topology to consume. **Not done:** any MDO or
Bayesian-optimisation run on the new physics (the optimisation, active-learning and surrogate
packages have 210 collected tests and a provisioned CUDA BoTorch/pymoo environment, but
`modern/spec/optimization/campaign-v1.json` still records `"results": null` and no
experiment directory contains an optimisation campaign); consumer integration of the v4
wall-loss export (`export_only_pending_consumer_integration`); a converged, preregistered PIC
result; the material-aware field solver the ISTS study assumed; any external or experimental
validation (0 %); the fresh-clone release audit; and a submission-ready paper (the 17-page
manuscript is evidence-gated and reproducible but L1-L3 gates are closed by construction,
PIC/MDO/experiment are absent, and all human authorship gates are `human-approval-required`).
Applying the canvas's own five-component method (implemented software 30 / independent
acceptance 25 / integrated delivery 15 / completed simulation evidence 15 / external
validation 15; phase weights 6/10/14/14/14/12/12/12/6) to the committed evidence gives
**63 % (planning range 58-68 %)**, not the canvas's 70 %. The difference comes from four
phases the canvas scored on completed activity rather than on delivered outcome: P0 (90 → 75;
the equation ledger has no domain sign-off and six physics questions from the audit remain
unaddressed), P4 (94 → 76; the phase is titled "coupling and held-out promotion" and neither
promotion nor consumer integration exists), P5 (64 → 50; three PIC runs, no plateau, the
only steady-state result is uncommitted and shows no ignition), and P6 (44 → 40; no MDO run
exists on the new physics). Against the user's original five asks the honest summary is:
rebuild — partial (Python complete for the corrected models, C++ one kernel, GPU where it
paid off, legacy end-to-end chain deliberately not reproduced); legacy audit — done; SOTA
plasma simulation — built and verified, no converged result; SOTA surrogate/MDO — built,
never run on the physics; run/validate/visualise — numerical campaigns run and visualised,
validation numerical-only; paper — updated and reproducible, not submittable.

### 1.1 Re-derived phase scores

Component scores are out of 30 / 25 / 15 / 15 / 15 (sum 100). "Canvas" is the value in the
tracker at the start of this audit; "Audit" is the value re-derived here.

| Phase | Weight | Software | Acceptance | Delivery | Sim. evidence | External | **Audit** | Canvas | Why the audit differs |
|---|---|---|---|---|---|---|---|---|---|
| P0 Preservation, audit, provenance | 6 | 30 | 15 | 15 | 12 | 3 | **75** | 90 | No domain sign-off on `modern/spec/plasma/equation-ledger.json`; AUDIT.md items S3/S7/S8/S9 not physically resolved (section 2.1); ISTS paper reconciled but Kornfeld source equations still absent (`modern/LEARNING_SCRATCHPAD.md:52-62`) |
| P1 Infrastructure and runtime | 10 | 30 | 24 | 15 | 15 | 12 | **96** | 99 | Full suite green on a fresh worktree (this audit) counts as the external check; fresh-clone release audit, `Results/` ignore rule and `validate_results` LF refusal still open (section 5) |
| P2 L0, geometry, L1a | 14 | 30 | 25 | 15 | 15 | 0 | **85** | 85 | Agrees |
| P3 L1b + P2 FEM | 14 | 22 | 15 | 15 | 13 | 0 | **65** | 70 | 1 of 3 designs qualified; L1b 54/54 gates `NOT_EVALUATED`; no nonlinear B-H or measured parity |
| P4 Topology, coupling, held-out promotion | 14 | 30 | 17 | 15 | 14 | 0 | **76** | 94 | v4 accepted and the nulls are complete, but held-out promotion 0/7 attempts and the coupling consumer is not integrated |
| P5 Hybrid and PIC-MCC | 12 | 25 | 6 | 14 | 5 | 0 | **50** | 64 | Software verification only; no plateau in any committed run; steady-state v1 results uncommitted |
| P6 Surrogates, BO, MDO, UQ | 12 | 24 | 3 | 12 | 1 | 0 | **40** | 44 | No accepted learned model; no optimisation run on the new physics; pymoo never imported in-repo; BoTorch runtime verified only in an untracked venv |
| P7 Integrated campaigns, external validation | 12 | 12 | 3 | 5 | 0 | 0 | **20** | 20 | Agrees |
| P8 Visualisation, paper, release, audit | 6 | 28 | 16 | 15 | 11 | 0 | **70** | 74 | Dashboards browser-checked manually, not in tests; no PDF committed; release checklist and human gates open; this audit closes the "completion audit" item |
| **Weighted total** | 100 | | | | | | **63.1 %** | 69.96 % | |

Weighted total: 6·75 + 10·96 + 14·85 + 14·65 + 14·76 + 12·50 + 12·40 + 12·20 + 6·70 = 6314 → 63.1 %.
Canvas at audit start: 6·90 + 10·99 + 14·85 + 14·70 + 14·94 + 12·64 + 12·44 + 12·20 + 6·74 = 6996 → 69.96 %.

---

## 2. Requirement-by-requirement audit

### 2.1 Legacy defect audit (`modern/docs/AUDIT.md`)

Preservation: exactly one commit in the whole history touches `FYP/` (`7ca3dc2d`,
2026-09-01); the `FYP` tree object is identical at `7ca3dc2d` and HEAD
(`9c8172ae8b976508f714f13458a8f2b82b06b2a7`); 16 tracked files, matching `AUDIT.md:5`.
Verdict: **ACCEPTED evidence** (preservation).

Disposition of every defect listed in `AUDIT.md`. "Fixed" means the corrected formulation in
`modern/` closes the defect; "superseded" means the rebuild does not reproduce the code path.

| # | AUDIT.md item (legacy location) | Disposition in rebuild | Evidence |
|---|---|---|---|
| C1 | `Ua`/`Ia` locals read as empty globals (`Performance_est.m:10,31-32`; `boundaries.m:14`; `Power_B_EQs.m:77`) | **FIXED** in `cft_revival.plasma` (explicit typed inputs); legacy solver **SUPERSEDED/quarantined** | `modern/src/cft_revival/plasma/models.py:67-68,80-117`; `modern/src/cft_revival/backends.py:82-92` (`UnimplementedPlasmaBackend`); `modern/docs/MIGRATION.md:19-22` |
| C2 | 11 Boolean comparisons as least-squares residuals; 33+11 rows for 30 unknowns (`Power_B_EQs.m:13,140-168`) | **FIXED** — inequalities are constraint margins, not residual rows; 28-row, 25-state overdetermined LM system | `modern/src/cft_revival/plasma/residuals.py:449-483`; `modern/spec/plasma/equation-ledger.json:23-50`; `modern/docs/workstreams/global-plasma-formulation.md:29-31,105-108` |
| C3 | Hard-coded `x(9)-1000>=0` instead of variable `Ua` (`Power_B_EQs.m:164`) | **FIXED** — margin `phi[3] - anode_voltage_v` | `residuals.py:461`; `global-plasma-formulation.md:107` |
| C4 | Executable `+CE` excitation sign contradicts description (`Power_B_EQs.m:111-126`) | **FIXED** — excitation subtracted as a loss, closure proven on the published DM9.2 table | `equation-ledger.json:46-49` (R23-R26); `residuals.py:199,211,239`; `global-plasma-formulation.md:61-83` |
| C5 | 12-way FEMM parallelism knowingly unsafe (`params.m:7`; `FEMMrun.m:6-10,162-164`) | **FIXED by guard** (`serialize_femm` must be true); FEMM automation itself **OUT OF SCOPE** (never driven) | `modern/src/cft_revival/models.py:217-218`; `modern/config/default.json:6`; `backends.py:57-63`; `ARCHITECTURE.md:98-111` |
| C6 | Cusp extraction: stale wildcard files, window means called extrema, NaN propagation, cusp-4 swap producing complex angle (`cusp_prob.m:7-8,60-190`) | **FIXED (fail-closed)** for numerics; hard-coded windows and swap **preserved as labelled compatibility behaviour** | `modern/src/cft_revival/kernels.py:47-104`; `backends.py:66-79`; `modern/tests/test_kernels.py:53-64,83-93`; `MIGRATION.md:47-48` |
| C7 | Exit-flag-only acceptance, residual discarded, `TolFun=1e-50` (`HEMP_solver.m:61,64`; `Performance_est.m:91-128`) | **FIXED** — publish only when residual tolerance and feasibility are met; gradient/step stops labelled as non-convergence | `modern/src/cft_revival/plasma/solver.py:36-38,359,393-394,447-462,595-600`; `modern/tests/plasma/test_solver.py:78,96,158` |
| C8 | Non-finite thrust/Isp reported as success (`Performance_est.m:153-156,249-337`) | **FIXED** | `kernels.py:137-139,153-154`; `modern/spec/physics/equation-ledger.json:5` |
| C9 | `findLocalMin` not robust, no caller (`findLocalMin.m:20-22`) | **N/A** (not migrated) | `MIGRATION.md:39-40` |
| S1 | Cathode Te = 0 "probably incorrect" (`Performance_est.m:28`) | **PARTIALLY** — explicit validated input, default still 0.0 | `plasma/models.py:71,150` |
| S2 | Mass utilisation up to 1.2 (`Performance_est.m:41-42`) | **FIXED in L0** (η_m = f1+f2 ≤ 1 by construction); compatibility kernel bounds only total efficiency | `equation-ledger.json:64-75` (PHY-L0-003); `kernels.py:138` |
| S3 | Grid efficiency `1-phi1/phi2` unverified (`Performance_est.m:135`) | **NOT ADDRESSED** (translated verbatim in the compatibility kernel; L0 avoids the quantity) | `kernels.py:135`; `physics-foundation.md:89-91` |
| S4 | `abs(x(27))` non-differentiable residual (`Power_B_EQs.m:93`) | **FIXED** — signed `ji4` with bound | `residuals.py:98,107`; `equation-ledger.json:34` (R11) |
| S5 | Mixed physical scales, no normalisation | **FIXED** — currents ÷ `Ia`, energies ÷ `Ua·Ia` | `global-plasma-formulation.md:96-97`; `residuals.py:256` |
| S6 | FEMMrun varies radial geometry only (`FEMMrun.m:23-80`) | **SUPERSEDED** by geometry v1.1 + L1a/L1b/P2 solvers; FEMM geometry generation not translated | `ARCHITECTURE.md:162-209`; `MIGRATION.md:13-14` |
| S7 | Hard-coded axial cusp windows (`cusp_prob.m:11-27`) | **NOT ADDRESSED physically**; preserved as compatibility; coupling v4 replaces windows with detected wall-cusp maxima | `kernels.py:92-100`; `coupling-formulation.md:3-49` |
| S8 | Cusp-4 "anode condition" swap unsourced (`cusp_prob.m:174-181`) | **NOT ADDRESSED physically**; preserved + fail-closed | `kernels.py:102-104`; `test_kernels.py:90-93` |
| S9 | FEMM asymptotic boundary coefficient `0.2` vs mm units (`FEMMrun.m:89-97`) | **NOT ADDRESSED (deferred)** | `ARCHITECTURE.md:185-187` |
| S10 | `plasmaCalc.m` exploratory, disconnected | **N/A** | `MIGRATION.md:39-40` |
| O1-O5 | Objective sign storage; failure codes as labels; radii ordering; gen-35/ind-39 rejection; plotter assumptions | O1 fixed (physical sign), O3 preserved intentionally, O4 superseded (no equivalent), O2/O5 planned | `MIGRATION.md:11-12,36-37,45-46,50-51`; `AUDIT.md:170-171` |
| R1-R3 | Path/CWD dependence, no run identity, `csvwrite` without headers, RNG without provenance | **SUPERSEDED** by hash-pinned manifests, canonical JSON and sidecars everywhere | `MIGRATION.md:56-57,81-82`; `modern/src/cft_revival/experiment_runtime/` |
| D1 | External deps (`Range`, `Surrogate`, FEMM `mi_*`, Optimization Toolbox) and archived runs absent | **NOT RECOVERABLE** — stated blocker | `AUDIT.md:217-231`; `MIGRATION.md:103-106` |

Verdict for the requirement "audit of legacy code errors": **ACCEPTED evidence** for the audit
itself; **6 physics questions remain open** (S1 default, S3, S7, S8, S9, and the Kornfeld
residual sources in `LEARNING_SCRATCHPAD.md:52-62`). The legacy end-to-end chain
(FEMM → cusp windows → 30-unknown solve → performance) is deliberately not reproduced; the
manuscript states this (`paper/manuscript.tex:246-263`, Legacy audit section).

### 2.2 L0 global plasma model and the L0 surrogate campaigns

| Item | Evidence | Verdict |
|---|---|---|
| L0 conservation model (`cft_revival.physics`): 13 ledgered equations, SI-explicit, external closures | `modern/spec/physics/equation-ledger.json:38-216`; `modern/docs/workstreams/physics-foundation.md:52-94`; 86 tests collected in `tests/physics` | **ACCEPTED evidence** (implementation consistency only; `physics-foundation.md:96-117` forbids predictive claims) |
| 8,192-point RTX 5090 sweep, Python/Warp parity on all 26 fields, conservation residuals ≤ 4.43e-16 | `modern/docs/FIRST_RESULTS.md:20-22,78-93`; `modern/config/l0-deterministic-sweep.json`; paper CLM-005…CLM-008 (`paper/evidence/claims.json`) | **ACCEPTED evidence**; timing (0.634 s) explicitly not a benchmark (`FIRST_RESULTS.md:46-52`) |
| Corrected Kornfeld 4-cell global discharge (`cft_revival.plasma`): 25 states, 28 rows; DM9.2 table closure max residual 1.49e-3 at R27, rank 22/25 | `global-plasma-formulation.md:14-31,120-127`; `global-plasma-devlog.md:83-101`; 36 tests | **RECORDED-DEVELOPMENT** — numerically verified foundation, not a predictive model (`ARCHITECTURE.md:63-65`) |
| `l0_surrogate` v1: 96-row GP, RMSE gates pass, coverage gate fails | `modern/experiments/l0_surrogate/artifacts/benchmark.json:12-25,441`; commit `9953c41e` | failed development evidence |
| v2: crashed in `_save_models` before assessment | `l0_surrogate_v2/results/failure-manifest.json:9-20`; `aa71f065` | execution failure |
| v3: failed predeclared gates + provenance failure (wrong SHA recorded) | `l0_surrogate_v3/results/run-manifest.json:725`; `provenance-failure.json:5-7`; `4af90a34` | invalid, not accepted |
| v4: failed coverage gates, identity valid | `l0_surrogate_v4/results/run-manifest.json:745-747`; `2c310750` | valid failed |
| v5: 0/3 replicates passed | `l0_surrogate_v5/results/run-manifest.json:700`; `8f2d283b` | valid failed |
| v6: `status: accepted` at 96 rows (thrust NRMSE 2.185 %, coverage 0.913) — later found by v7 to use a pooled-row conformal rank that is not group-valid | `l0_surrogate_v6/results/run-manifest.json:443`; `l0_surrogate_v7/README.md:3-4`; `020aa584` | empirical software-emulation only; **never promoted** as a learned surrogate |
| v7: exact group conformal; worst thrust 17.30 % > 15 % gate; row coverage 0.964 > 0.95 upper gate | `l0_surrogate_v7/RESULTS_REPORT.md:57-62`; `c482fb57` | valid failed prospective validation |
| v8: no candidate passed development selection (best OOD worst 24.82 %) | `l0_surrogate_v8/RESULTS_REPORT.md:19-30`; `41e0c1cd` | failed development |
| v9: run manifest `accepted`, overturned by post-hoc audit: the analytic leading term **is** the L0 target for both outputs, so the GP learned only roundoff (errors ~1e-16); preflight touched a frozen calibration index (barrier invalid); GP 481× slower than the algebra | `l0_surrogate_v9/POSTHOC_AUDIT.md:5-53`; `posthoc-audit-status.json:4,23-48` (`NUMERICAL_PASS_SCIENTIFIC_SURROGATE_FAIL`); `71712057`, audit `bde85635` | **invalidated (tautology)**; sole permitted claim "Algebraic implementation plus negligible GP matched fresh same-domain evaluations" |
| Any L0 surrogate accepted as a learned model? | `ARCHITECTURE.md:66-68`, `MIGRATION.md:30-32`: "held-out surrogate quality gate failed … do not authorize replacement-model accuracy claims" | **NOT DONE** (by evidence, not by omission) |

There is no `l0_surrogate_v10`; `l0_surrogate_v9/POSTHOC_AUDIT.md:77-79` explicitly says none
should exist. The "v10 local interpolator failure" is the L1a field-surrogate v10 (section 2.3).

### 2.3 L1a axisymmetric magnetics, geometry, sweeps, field surrogates

| Item | Evidence | Verdict |
|---|---|---|
| `cft_revival.fields` — L1a ψ = r·A_φ constant-permeability equivalent-current FDM; Python PCG reference + Warp CPU/CUDA; artifact schema `cft-axisymmetric-field-map/1.2.0` | `modern/src/cft_revival/fields/artifacts.py:29-32`; parity ≤ 2.34e-14 on three designs (`axisymmetric-workstream-report.md:84-90`); `tests/fields/test_warp_and_artifacts.py:34-50` (< 2e-12 gate); 62 tests | **ACCEPTED evidence** (linear-vacuum FDM, "not FEM", `axisymmetric-formulation.md:7`) |
| `cft_revival.geometry` v1.1 — parametric PPM-stack geometry, viewer, magnetics handoff; three hypothetical artifacts | `geometry/model.py:18`; `modern/examples/geometry/artifacts/manifest.json` (`claim_limit: "Hypothetical geometry artifacts only…"`); 41 tests | **ACCEPTED evidence** (contracts); not hardware-validated |
| `cft_revival.magnetics` — constitutive/source/handoff contracts, no solve | `magnetics/serialization.py:33`; `magnetics-foundation.md:3-31`; 52 tests | contracts only; no downstream accepted evidence |
| `modern/examples/axisymmetric/` — three v1.2 artifacts, |B|max 0.0450/0.0384/0.0364 T, residuals ≤ 9.67e-11 | `results/manifest-l1a-v1.json:7-86`; `report.md:100-138` | **ACCEPTED evidence**; timings diagnostic only |
| L1a geometry sweep v1 — 96 designs, 25 nondominated, not preregistered; results force-tracked at `b2d5d79b` | `modern/experiments/l1a_geometry_sweep/results/report.md:5-8`; `97aaa192` | **RECORDED-DEVELOPMENT** (post-hoc) |
| L1a geometry sweep v2 — preregistered `092f5fae`, result `f30cb42e`, 96/96, 0 failed, 25 nondominated; seven gates all pass (boundary 0.01165 ≤ 0.05; residual 9.998e-11 ≤ 1e-10; CPU/CUDA parity ψ 1.63e-15; flux identity 4.55e-13; source representation 0.01748; topology confidence 0.9118; manufacturability 5.0e-5) | `l1a_geometry_sweep_v2/results/REPORT.md:6-20`; `protocol.json:91-102`; `POSTHOC_AUDIT.md` (only defect: CRLF digest of `protocol.json.sha256`); admitted as `GATE-L1A-SWEEP-V2` (`paper/evidence/result-gates.json:173-232`) | **ACCEPTED evidence** (field-only screening; no thrust/plasma/thermal claim, `protocol.json:124-128`) |
| Field surrogate v1 — case-preparation failure at index 69 (`GeometryValidationError`) | `l1a_field_surrogate_v1/results/failure-manifest.json:4-8`; `9e683936` | failed (code) |
| Field surrogate v2 — 112 coarse + 80 fine labels; best AR1 worst NRMSE 35.99 % and POD worst L2 41.93 % vs 5 % gates | `l1a_field_surrogate_v2/RESULTS_REPORT.md:5-18`; `a4866c56` | valid development rejection |
| Field surrogate v3-v5 (remote branches only) — `NameError: math`, `FileNotFoundError .working`, `detached worktree is not clean` | `origin/exp/l1a-field-surrogate-v3:…/RESULTS_REPORT.md:13-16`; v4 `:15-25`; v5 `results/terminal.json` | infrastructure failures |
| Field surrogate v6-v9 — `development_rejection`; best field worst-L2 0.2650 / 0.0971 / **0.0647** (v8, closest to the 5 % gate) / 0.1144 | `origin/exp/l1a-field-surrogate-v{6,7,8,9}:…/results/artifacts/frozen-method-selection.json` | valid development rejections |
| Field surrogate v10 — 24 localised-interpolator candidates, all `passed: False`; best worst-L2 0.3420 at budget 270 (0.3225 at budget 162) vs coarse baseline 0.3053, i.e. every interpolator worse than the uncorrected coarse solve; gate `latency_median_speedup_min 3.0` never reached (run ended at development rejection) | `origin/exp/l1a-field-surrogate-v10:modern/experiments/l1a_field_surrogate_v10/results/artifacts/frozen-method-selection.json`; `…/preregistered-protocol.json`; result commit `dee13dae` (2026-09-02 20:33 +1000) | valid development rejection; **sequence stopped** |
| Stop-decision rationale for the field-surrogate line | `modern/docs/workstreams/surrogates-devlog.md` and `surrogates-learning-ledger.md` cover the L0 GP package only; no v3-v10 text; no in-repo record of the "0.209 s fine vs 0.105 s coarse" timing quoted by the tracker | **NOT RECORDED in the repository** (documentation debt; section 5) |

### 2.4 P2 adaptive FEM qualification and L1b material fields

| Item | Evidence | Verdict |
|---|---|---|
| `cft_revival.fem_reference` — body-fitted P2 FEM reference, schema `…/1.3.0`; 37 tests + 10 dashboard tests | `fem_reference/artifacts.py:32-35`; `fem-reference-devlog.md:320` | software accepted |
| Third-level campaign, three designs, LFS-tracked at `a1158bad`: historical-envelope-baseline `SCREENING_ONLY` (orders 1.10/1.34/**−0.20**), compact-high-gradient-stack `SCREENING_ONLY` (orders incl. **−0.63**, **−5.28**), divergent-exit-stack **`NUMERICAL_P2_QUALIFIED`** (orders 1.3483 / 1.3534 / 1.3776 / 1.3955; 1,198,545 DOFs at L2) | `modern/examples/fem_reference/artifacts/third-level/<design>/manifest.json`; `fem-reference-devlog.md:326-358` | **ACCEPTED evidence** for divergent-exit-stack; **SCREENING-ONLY** for the other two |
| Independent replay: exact mesh/solution hashes and zero ψ replay error for all three | `fem-reference-devlog.md:360-362`; `examples/fem_reference/replay_artifacts.py` | accepted |
| RAM gate history: 400,000-DOF cap → 1,500,000 cap with ≥ 8 GiB free preflight (host had 12 MB, then 319 MB free) → calibrated 1.75×+256 MiB model → `ResourceBlockedError(NOT_EVALUATED)` guard → third level ran with 31-36 GB free (peak working sets 334 MB / 108 MB / 392 MB) | `fem-reference-devlog.md:148-162,187-204,243-252,262-296,300-329`; manifests `resource_policy_revision.preflight` | recorded |
| Downstream consumers of the qualified field: wall-loss v4 (cross-map convergence ≤ 0.0039 across primary/refined/enlarged) and pic2d (hash-verified load) | `cft_orbit_wall_loss_v4/results/artifacts/probability-convergence.json`; `modern/spec/pic2d/p2-field-authority-v1.json:34-39` | accepted consumer-side confirmation of mesh sufficiency for one estimand |
| Dashboard `fem-reference-p2-qualification.html` (3.24 MB) committed at `d47db654`, re-pinned to `a1158bad` | `examples/fem_reference/visualization/DEVLOG.md:51-57` | accepted (rendering) |
| `cft_revival.material_fields` L1b v1.4 — linear recoil PM / linear iron FV solver with Warp PCG; executed campaign memory-limited (60×120 base instead of the preregistered 467×1814); **all 54 gates `NOT_EVALUATED`** (18 × 3 designs, verified by parsing `acceptance.gates`); `acceptance.status: SCREENING_NOT_ACCEPTED`; `STRUCTURED_GRID_L1B_INSUFFICIENT` | `modern/examples/material_fields/artifacts/*.material-field.json`; `material-fields-devlog.md:200-248`; 40 tests | **SCREENING-ONLY** |
| 7a9aff4d re-bind — CRLF-era source-code digests re-bound by the workstream's replay-guarded `refresh_artifact_metadata.py`; only hash leaves changed; isolated commit flagged for review | `examples/material_fields/POSTHOC_AUDIT.md:1-131`; commit `7a9aff4d` (20 files) | audit accepted; review pending |
| Material-aware production field solver (nonlinear B-H, FEMM parity, measured field maps) | `ARCHITECTURE.md:189-209` verification ladder steps 5-6 | **NOT DONE** |

### 2.5 Plasma topology characterisation, four-cell search, plasma network

| Item | Evidence | Verdict |
|---|---|---|
| Four-cell proxy search v1 — 128 designs on deprecated coupling-v2 same-z proxy; 2 "compatible", 0 states; not preregistered | `modern/experiments/four_cell_topology_search/results/report.md:3-13,66-80`; `4afcecfb` | superseded / invalid proxy evidence |
| Four-cell topology search v2 — preregistered `d6317910`, bound to coupling v3; 128/128 evaluated, 128 three-map field-accepted, **0 stable**, `TOPOLOGY_COUNT 128`, `TOPOLOGY_UNSTABLE 128`; interior cusps per map 10-18 (primary 11-17, modal 14) vs target 4; result `7120e8ed`; EOL audit `605be5ce` | `four_cell_topology_search_v2/results/manifest.json` `summary`; `results/dataset.json` `cases[*].topology.count_by_role`; `preregistered-protocol.json` `topology`; `POSTHOC_AUDIT.md`; admitted as `GATE-FOUR-CELL-V2` (`result-gates.json:234-303`), CLM-022…024, CLM-028 | **ACCEPTED evidence (preregistered null)** — disclosed: GPU replay 2 of 4 diagnostic passes (9.42e-6 / 6.50e-6 vs 5e-6 limit; fields reproduced to ≤ 4.4e-15 T); `experiment.py:1844-1849` `validate_results` refuses the bundle on LF checkouts (code defect, left as-is) |
| Topology characterization v1 — preregistered `af88470b`; 56 designs × 3 maps; **0 stable eligible cusps / 0 cells**; primary-map nulls 1276 (X 520 / O 532 / degenerate 224), all excluded; GPU replay 3/3 | `cft_topology_characterization_v1/results/report.md:5-49`; `results/dataset.json`; `3ce6c546`; admitted as `GATE-TOPOLOGY-CHAR-V1` (`result-gates.json:305-370`) | **ACCEPTED evidence (recorded characterisation)** |
| `l1a_plasma_coupling` — L1a → coupling-v2 proxy → four-cell screening; 0 accepted / 3 failed / 0 plasma solves; results gitignored (untracked) | `modern/experiments/l1a_plasma_coupling/DEVLOG.md:13,80-81`; `40dcaa4c` | **RECORDED-DEVELOPMENT**, results not in repo; still on the deprecated v2 proxy (`experiment.py:20,61-67`, `DeprecationWarning` in the test run) |
| `cft_revival.plasma_network` — topology-general N-cell network, 6N+1 states / 7N residuals; manufactured N = 1…6 residuals 1.36e-16…1.45e-16 | `plasma-network-formulation.md:19-35`; `plasma-network-devlog.md:25-29`; 64 tests | software verified; **no physical topology input** (four-cell v2 `coupled_count 0`) |
| Plasma topology results dashboard (3.67 MB, 72 hashed sources) | `modern/visualization/plasma-topology-results.html`; `16670281`; 17 tests | accepted (rendering) |

### 2.6 Coupling v2 / v3 / v4 and held-out promotion

| Item | Evidence | Verdict |
|---|---|---|
| Coupling v2 (`cft-field-plasma-coupling/2.0.0`) — same-z axis/wall comparison | `coupling/records.py:30`; `coupling-learning-report.md:118-121` ("prohibited physical comparison") | **rejected as physics**; survives only as `screening_proxy` with `DeprecationWarning` |
| Coupling v3 (`…/3.0.0`) — connected constant-ψ flux-surface mirror, three-map stability | `coupling/v3_records.py:52`; `coupling-devlog.md:130-155` | software-validated; used by the two topology nulls; now "historical" (`coupling-formulation.md:93-96`) |
| Coupling v4 / 4.2 (`…/4.2.0`) — HEMP wall-cusp contract (|B_r| maxima, cells between stable cusp planes), field artifact v1.2 canonical-byte authority; 143 tests | `coupling/v4_records.py:56`; `coupling-formulation.md:3-49`; `coupling-devlog.md:291-309` | software-validated |
| Held-out promotion (`cft-hemp-wall-cusp-v4` 4.0.0, 56-case development manifest, 24 held-out cases): v1 `TypeError` (`BoundaryNullDiagnostic` not iterable) after 1/24; v2 `TypeError` (datetime not serialisable) before any access; v3 canonical-payload SHA mismatch (−0.0 normalisation); v4 `EvidenceVerificationError`; v5 `CanonicalizationError`; v6 `assessment_rejection` (0/720 resolved orbits); v7 `assessment_rejection` (644 numerically converged, 0/720 physically adiabatic) | `cft_wall_cusp_validation_v1/results/failure.json:20-38`; `_v2/results/failure.json:4-16`; `origin/exp/cft-wall-cusp-validation-v{3..7}:…/results/terminal.json`; `876c3a9d`, `aa4d92ca` | **NOT DONE** — 0 of 7 attempts promoted; `criterion_numerically_promoted: false` everywhere |
| Wall-loss v4 coupling export — `electron_dielectric_wall_loss_probability` 0.64453125, 95 % CI [0.60213, 0.68477], `integration_status: export_only_pending_consumer_integration` | `cft_orbit_wall_loss_v4/results/artifacts/coupling-export-only.json` | export emitted; **consumer integration NOT DONE** — no file in `coupling/` or `plasma_network/` references the export (`orbit-mc-integration.md:41-43`: "No public coupling-v4.2/plasma-network consumer currently accepts this object") |

### 2.7 Full-orbit wall-loss campaigns and `orbit_mc`

| Item | Evidence | Verdict |
|---|---|---|
| `cft_revival.orbit_mc` `__version__ 1.7.0`; result/checkpoint schema 1.6.0; handoff `cft-revival-orbit-mc-coupling-v4.2/1.3.0`; 147 tests | `orbit_mc/__init__.py:3`; `orbit_mc/artifacts.py:31-33` | software accepted |
| v1.4 `25dbeaaf` (mandatory batch-manifest hash); v1.5 `7cf65053` (tolerance-close events, zero-progress guard, v3 root cause); v1.6 `3ab50ef5` (Boris-pushed event velocity, energy error 6.1e-4 → 0.0); v1.7 `cc4bd5e1` (LF sidecars + AST newline lint) | `orbit-mc-devlog.md:172-430` | accepted fixes, each with shakedown |
| Wall-loss v1 (`9940653d`) `prebundle_failure`: "launch manifest differs from preregistered authority" | `origin/exp/cft-orbit-wall-loss-v1:…/results/terminal.json`; `cft_orbit_wall_loss_v4/protocol.json#prior_campaign_disclosure` | failed on code; one-shot preregistration consumed |
| Wall-loss v2 (`419efefc`) `runtime_failure`: "ordered launch/result/campaign identities are inconsistent" | `origin/exp/cft-orbit-wall-loss-v2:…/results/terminal.json` | failed on code |
| Wall-loss v3 (`09256fb1`) `runtime_failure`: "physical event witness requires a positive step" (zero-step wall stall); second latent bug `zip(ordered, ordered[1:], strict=True)` in `_convergence` found by the v4 shakedown | `origin/exp/cft-orbit-wall-loss-v3:…/results/terminal.json`; `orbit-mc-devlog.md:221-232`; `cft_orbit_wall_loss_v4/DEVLOG.md:77-83` | failed on code; motivated the shakedown rule |
| Shakedown rule — non-evidentiary disjoint 64-launch design through the full prebundle/development/assessment/export path; `prepare` refuses without a passing, hash-matched `shakedown.json` | `cft_orbit_wall_loss_v4/shakedown.json` (`evidentiary: false`, `outcomes_enter_estimand: false`, `passed: true`, orbit_mc 1.6.0); `README.md:56-66` | accepted process control |
| **Wall-loss v4** — preregistered `757e365f`, recorded `6922a3cf`, audited `258f69b2`; `state: accepted_result`; 15/15 binding gates `passed: true`; 4608 orbits (9 cases × 512); 289 validators / 0 failures; exact authority replay 9; pooled 2962 wall_hit / 1646 domain_escape / 0 reflected; per-case wall-hit 0.6426 [0.6001, 0.6829] (primary-N, refined-N), 0.6445 [0.6021, 0.6848] (primary/refined 2N, 4N), 0.6406 [0.5982, 0.6810] (enlarged N/2N/4N); timestep changes ≤ 0.001953, cross-map ≤ 0.003906, all Wilson overlaps true; max relative energy error 0.0; wall endpoint 4.34e-19 m | `cft_orbit_wall_loss_v4/results/artifacts/{gates,campaign-result,probability-convergence}.json`; `results/terminal.json`; `POSTHOC_AUDIT.md:5-25,196-218`; admitted as `GATE-WALL-LOSS-V4` (`result-gates.json:113-171`), CLM-012…CLM-017 | **ACCEPTED evidence** — classification `collisionless_prescribed_field_test_particle_wall_loss_not_pic` |
| Per-stratum structure (summed over 9 cases from `results/artifacts/summaries/*.json`): cells 2 and 3 → 16/16 wall_hit in every stratum; cell 4 (beyond the 18 mm wall end) → 0/16; cell 1 direction −1 → 576/576 wall_hit, direction +1 → 82 wall_hit / 494 escape | `POSTHOC_AUDIT.md:211-218` | recorded diagnostic, not a gate |
| Post-hoc audit: 378 files byte-exact, 9 `orbits/*.json.sha256` EOL-only (CRLF written by orbit_mc 1.6.0), duplicate `execute` refused by the Git-common lock, `results/` force-added past the `Results/` ignore rule | `POSTHOC_AUDIT.md:47-90,147-172`; 12 audit tests | evidence ACCEPTED, recording-layer defect only |
| GPU orbit path — Warp per-push kernel 1.6 s/orbit on the RTX 5090 vs 90 ms numpy (~18× slower), velocity parity 1e-14 | `orbit-mc-devlog.md:285-287`; `orbit-mc-learning.md:171-174` | ruled out as throughput path; kept as parity gate (the tracker's "1.6-1.8 s" upper figure is not in any committed file) |
| Wall-loss v4 results dashboard (683 KB, 387/387 manifest files verified) | `modern/visualization/wall-loss-v4-results.html`; `bc2f8e47`; 13 tests | accepted (rendering) |

### 2.8 PIC-MCC

| Item | Evidence | Verdict |
|---|---|---|
| 1-D foundation `cft_revival.pic` — periodic 1D3V electrostatic; Poisson (mean-zero CG, true residual), CIC with adjoint identity, leapfrog/Boris, transactional synthetic MCC, Warp CIC/gather/push parity on CPU and `cuda:0`, Courant/ω_p·dt/MCC gates; spec status `reduced-kernel-correctness-verified-not-predictive-cft`; cross sections synthetic | `modern/spec/pic/pic-foundation-v1.json:4,6,91,111`; `pic-foundation.md:36-111,144-146`; 63 tests | software verified; **RECORDED-DEVELOPMENT** |
| 2-D `cft_revival.pic2d` `__version__ 0.1.0` (specs record model versions 0.2.0 / 0.2.1) — cylindrical Poisson (block-Thomas direct + GPU PCG), bilinear fixed-point deposit (bit-identical CPU/GPU), Boris in the hash-verified P2 divergent-exit field, kinetic e⁻ + Xe⁺, static neutrals, null-collision MCC from LXCat Biagi-v7.1 (payload hash `4d37732c…`, upstream commit pinned), Warp CUDA, bitwise checkpoint/resume; Poisson orders 1.9988/1.9997/1.9999, direct == dense 2e-11 V, Gauss 1e-9, Boris ≤ 8 ulp of orbit_mc, P2-orbit error 8.0e-3 → 2.0e-3 → 2.8e-4 gyroradii, MCC within 4σ | `pic2d/__init__.py:27`; `pic2d-model-v1.1.json:4`; `pic2d-devlog.md:47-68`; `spec/pic2d/xenon-cross-sections-v1.json`; `spec/pic2d/p2-field-authority-v1.json:34-39`; 73 tests | software verified (merged `df4b2d77…62de2ca3`, `3a42bcd7…1cdaae80`, `44b7c8dc`) |
| Model v1.1 — device block-Thomas under true-residual contract, exact tile reductions, ion subcycling k = 8, electrode-work ledger; step 40.7 → 5.46 ms at 5.4 M particles, 2.04 ms at 1.53 M; ≤ 1.5 ms target **not met** (~1.2 ms WDDM launch floor) | `spec/pic2d/pic2d-model-v1.1.json:51-57`; `pic2d-devlog.md:147-162,191-196` | software; target missed and recorded |
| Snapshot v1 — four cases, all `runtime_stability_gate_stopped_run` at 49.0-60.2 ns (ω_pe·Δt ≤ 0.2 tripped on axis nodes at n_e ≈ 3.3e18); loss/source 2.6 %; no plateau | `pic2d_cft_snapshot_v1/results/manifest.json` (`status: development_screening_not_preregistered`); `results/diagnosis.json` | **RECORDED-DEVELOPMENT** (fail-closed) |
| Snapshot v2 — four cases to 0.932 / 1.136 / 1.533 / 1.203 τ_i; `plateau.reached false` in all; I_d drift 0.094-0.222, N_e drift 0.186-0.691; between-case spreads φ_max 0.108, I_d 0.176, peak n_e 0.432, mean n_e 0.540, ⟨T_e⟩ 0.574; coarse pair ledger residual +0.41 vs fine −0.13…−0.18 (grid heating) | `pic2d_cft_snapshot_v2/results/manifest.json`; `README.md:47-55`; `pic2d-devlog.md:181-215` | **RECORDED-DEVELOPMENT** — no plateau, not converged |
| Steady-state v1 — protocol committed (`status: development_screening_not_preregistered`; model v1.2: n_g 1.5e19 m⁻³, 3 mA at 2 eV, 300 V, Δt 1.5 ps, fine grid, W 3e4; plateau rule < 5 % drift over trailing 20 % after ≥ 3 × 2.4 µs; 12 h budget); runner + 8 tests committed; **no results committed** | `pic2d_cft_steady_state_v1/{protocol.json,run.py,README.md}`; `pic2d-devlog.md:236-319` | protocol + runner committed; **result NOT DONE** at HEAD |
| Working-tree observation (uncommitted, pic2d worktree, 2026-09-03 07:27 local): `run_state.json` `finished: false`, checkpoint step 920,000 (t = 1.38 µs, 0.575 transits); `status.jsonl` at step 938,000: electrons 44,457 (down from 106,286 at step 200), I_d 0.0-0.08 mA vs 3 mA injected, ⟨T_e⟩ 2.6 eV, `plateau.reached false` — the discharge is decaying, not igniting | `C:\…\uni-project-pic2d\modern\experiments\pic2d_cft_steady_state_v1\results\{run_state.json,status.jsonl,run.log}` (untracked) | **not evidence**; the "no-ignition reference" interpretation exists only in untracked notes (`.cursor/memory/DEVLOG.md`) |
| v1.3 quasi-steady neutral inventory — at the audit snapshot (`44b7c8dc`) `pic2d/neutrals.py` was untracked in the pic2d worktree; it landed during the audit as `520e6b41` (spec `pic2d-model-v1.3.json`, 9 conservation tests: one scalar n_g(t) driven by prescribed feed, measured MCC ionisation sink, thermal effusion, artificial τ_g = 30 ns relaxation; only the fixed point is physical) | `520e6b41`; `modern/spec/pic2d/pic2d-model-v1.3.json`; `modern/tests/pic2d/test_pic2d_neutral_inventory.py`; see section 7 | software committed after the snapshot; **no run** |
| Steady-state v2 (model v1.3; n_g0 5e19, Q_in 7.77e16 atoms/s = 0.0170 mg/s, 300 V, 3 mA, 60×480, W 6e4) — protocol, README and thin runner committed as `67b04f87`; `status: development / screening`; **no results committed** | `modern/experiments/pic2d_cft_steady_state_v2/{protocol.json,README.md,run.py}` at `67b04f87` | protocol only; **result NOT DONE** |
| Any preregistered or accepted PIC result | `pic2d-formulation.md:9-11` ("no run under this package is preregistered"); `result-gates.json:87-111` GATE-L3 closed | **NOT DONE** |
| pic2d snapshot dashboard (5.66 MB, v2 cases + v1 history + claim-boundary panel) | `modern/visualization/pic2d-cft-snapshot.html`; 9 tests | accepted (rendering of development evidence) |

### 2.9 Optimisation / MDO / surrogates / active learning / UQ

| Item | Evidence | Verdict |
|---|---|---|
| `cft_revival.optimization` — immutable domain/identity, constrained mixed-direction Pareto, hash-chained async ask/tell campaign with F3 quotas, shifted-Halton sampling, guardrails, lazy BoTorch adapter; campaign spec v1.4 | `optimization-workstream-report.md:3-25,72-73,79`; `spec/optimization/campaign-v1.json:3,163`; 76 tests | software; "MORBO, NSGA-III, MOEA-D, and a validated Sobol benchmark runner are specified, not implemented"; "BoTorch execution remains unverified" (in-repo) |
| `cft_revival.active_learning` — stdlib posterior-adapter layer, acquisition, fidelity selection, CVaR, stopping policy v1.4; "does not train a GP" | `active-learning-foundation.md:12`; 96 tests | software |
| `cft_revival.surrogates` — Matérn-5/2 ARD exact GP, AR1 two-fidelity, POD, OOD detector; one authoritative synthetic-L0 benchmark: NRMSE 0.227 / 0.207 vs 0.05 gate, `model_quality_passed=false` | `surrogates-benchmark-protocol.md:33,53-72`; 38 tests | software; benchmark **failed** |
| `cft_revival.validation` — evidence-authority ceiling, context-of-use ledger, conservation/convergence gates; Yeo 2020 S1 values as published model outputs | `validation-workstream-report.md`; `validation/contracts.py:14` (2.0.0); 59 tests | contracts only |
| `cft_revival.hybrid` — prescribed-field Boris/CIC macroparticles, synthetic collisions, checkpoint v1, optional Warp; six unimplemented gates incl. self-consistent Poisson | `hybrid-formulation.md:3-12,95-140`; `hybrid-roadmap.md`; 70 tests | software; no evidence |
| `.venv-sota` provisioning (untracked, main repo): PASS 2026-09-02; torch 2.13.0+cu130, BoTorch 0.18.1, GPyTorch 1.15.2, pymoo 0.6.2 on the RTX 5090; smokes: CUDA GP posterior, qLogNEHVI/qLogNParEGO, NSGA-III/MOEA-D bitwise-repeatable; repo extra pin `gpytorch<1.15` is unsatisfiable with `botorch>=0.18.1` | `.venv-sota/provision-report.txt`; `modern/pyproject.toml:20-25` | runtime smoke only; **not in the repository**; `pymoo` is never imported under `modern/src` or `modern/tests` |
| Any MDO / BO / Pareto campaign executed on the new physics (L0, L1a, P2, orbit, PIC) | `campaign-v1.json:163` `"results": null`; none of the 23 `modern/experiments/*` directories is an optimisation campaign; `paper/manuscript.tex:473-475` ("no admitted surrogate fit, acquisition execution, hypervolume result, or baseline comparison") | **NOT DONE** |

### 2.10 GPU acceleration ledger

| Where | Status | Evidence |
|---|---|---|
| L0 physics (`physics/warp_backend.py`) and cusp kernel (`warp_backend.py`) | used; float64 CPU/CUDA parity, 8,192-point sweep | `FIRST_RESULTS.md`; `tests/physics/test_warp_backend.py:83-151` |
| L1a fields (`fields/warp_solver.py`) | used; parity ≤ 2.34e-14; sweep v2 CPU/CUDA parity gate 1.63e-15 | `axisymmetric-workstream-report.md:84-90`; `l1a_geometry_sweep_v2/results/REPORT.md` |
| L1b material fields (`material_fields/warp_solver.py`) | used; CPU/CUDA field L2 2.54e-10…1.06e-9 (measured, not gated) | `material-fields-devlog.md:10,23,53` |
| Hybrid (`hybrid/warp_backend.py`) | optional Boris+CIC with `cuda:0` parity tests | `tests/hybrid/test_checkpoint_and_warp.py:416` |
| PIC 1-D (`pic/warp_backend.py`) | CIC + gather/push parity only; no Warp Poisson | `pic-foundation.md:99-100,144-146` |
| PIC 2-D (`pic2d/warp_backend.py`) | primary compute path (all-GPU step, CUDA graphs); the only real GPU workload in the project | `pic2d-devlog.md:33-37,153-162` |
| orbit_mc (`orbit_mc/warp_backend.py`) | **ruled out** as throughput path (1.6 s/orbit vs 90 ms numpy); parity gate only | `orbit-mc-devlog.md:285-287` |
| Four-cell v2 GPU replay | 2 of 4 tolerance replays passed; disclosed | `four_cell_topology_search_v2/POSTHOC_AUDIT.md:121-128` |
| Global plasma, plasma network, FEM reference | no GPU path claimed | `global-plasma-api.md:48`; `fem-reference-formulation.md:12` |
| C++17 native core (`modern/native/`) | one kernel (`cusp_arrival_probability`) + CTest; pybind11 extension not built in either checkout (test skipped) | `native/src/bindings.cpp:8-19`; `tests/test_kernels.py:67-70` |
| FEMM / `mi_*` automation on GPU | **OUT OF SCOPE** by design | `ARCHITECTURE.md:114-133` |

### 2.11 Reproducibility infrastructure

| Item | Evidence | Verdict |
|---|---|---|
| `cft_revival.experiment_runtime` — `O_CREAT|O_EXCL` immutable lock, three-callback lifecycle, atomic data + `.sha256.json` pairs, five terminal states, bijective inventory, `validate_bundle` replay; canonical JSON `cft-typed-canonical-json-v1`; accepted at `231873d2` and `b46e2639`; 133 tests (1 skip) | `experiment-runtime-architecture.md:13-139`; `experiment_runtime/canonical.py:18` | **ACCEPTED evidence**; production use: wall-loss v4 (`accepted_result`, duplicate `execute` refused) |
| LF pin — root `.gitattributes` `* text=auto eol=lf` (`fab0eccc`); AST lint fails closed on text writes without `newline=` in orbit_mc, experiment_runtime, fem_reference, coupling, fields | `.gitattributes:4`; `tests/orbit_mc/test_sidecar_portability.py:42-63,330,368` | accepted |
| `git ls-files --eol` `w/crlf` in the fresh worktree: 3 files + 2 mixed, all under `cft_wall_cusp_validation_v{1,2}/results/` with a local `* -text` attribute (stored CRLF by design; index == worktree) | `modern/experiments/cft_wall_cusp_validation_v{1,2}/results/.gitattributes` | not a defect; the "0 CRLF files" statement holds for smudge divergence only |
| Post-hoc audits (5): wall-loss v4 (evidence ACCEPTED), l1a sweep v2 (ACCEPTED), four-cell v2 (null stands), material_fields (SCREENING_NOT_ACCEPTED unchanged), l0_surrogate v9 (SCIENTIFIC_SURROGATE_FAIL) | `**/POSTHOC_AUDIT.md` | accepted audits |
| Full-suite health in one invocation on a fresh checkout | this audit: 1702 / 0 / 5 in 930 s; paper 58 / 298 subtests | **ACCEPTED evidence** |
| Fresh-clone release audit (optional-dependency closure, dashboards render, paper builds byte-identically, provenance/secret/path/FYP checks as one procedure) | `open-cft-roadmap-status.canvas.tsx` P1/P8 blockers; no script or record in the repo | **NOT DONE** |

### 2.12 Visualisations

All 10 dashboards are self-contained (no `http`, `fetch`, CDN references in any HTML) and each
has an explicit offline test. Browser testing is **manual only**: no headless/Chrome/Playwright
test exists under `modern/tests`; the sessions are recorded in `modern/visualization/DEVLOG.md:124-128,157-158`
and `pic2d-devlog.md:218-219`.

| Dashboard | Bytes | Renders | Tests |
|---|---|---|---|
| `modern/visualization/first-results.html` | 5,994,361 | L0 8,192-point sweep (+ `design-gallery.json`) | 11 + 9 |
| `modern/visualization/geometry-designs.html` | 119,484 | geometry v1.1 bundle | 14 |
| `modern/visualization/axisymmetric-results.html` | 216,376 | L1a v1.2 artifacts | 17 |
| `modern/visualization/pic2d-cft-snapshot.html` | 5,658,702 | pic2d snapshot v2 + v1 history (development) | 9 |
| `modern/visualization/plasma-topology-results.html` | 3,673,115 | characterization v1, four-cell v1/v2, sweep v2, P2 field, coupling failures, orbit v4 | 17 |
| `modern/visualization/wall-loss-v4-results.html` | 683,429 | wall-loss v4 bundle, 387/387 files re-verified | 13 |
| `modern/examples/fem_reference/visualization/fem-reference-p2-qualification.html` | 3,238,342 | P2 qualification (three designs) | 10 |
| `modern/experiments/four_cell_topology_search/visualization/four-cell-topology-search.html` | 2,557,143 | sealed four-cell v1 | 11 |
| `modern/experiments/l1a_geometry_sweep/visualization/l1a-geometry-sweep.html` | 1,559,849 | sweep v1 | 10 |
| `modern/experiments/l1a_geometry_sweep_v2/visualization/l1a-geometry-sweep-v2.html` | 1,436,042 | preregistered sweep v2 | 11 |

Verdict: **ACCEPTED evidence** for rendering and integrity checks; no dashboard creates evidence;
no automated browser test.

### 2.13 Paper

| Item | Evidence | Verdict |
|---|---|---|
| Manuscript structure — 15 sections: Introduction; Literature lineage; Methods architecture; Legacy audit; V&V/UQ protocol; Accepted L0 result; Section 7 wall-loss v4; Section 8 topology screening; Planned L1/L2/L3; Discussion; Limitations; Reproducibility; Conclusion | `paper/manuscript.tex:97-518`; `paper/sections/*.tex` | |
| Gates — GATE-L1/L2/L3 `physics-level` **closed**; GATE-WALL-LOSS-V4 `numerical-campaign` accepted, `opens_level null`; GATE-L1A-SWEEP-V2 / GATE-FOUR-CELL-V2 / GATE-TOPOLOGY-CHAR-V1 `numerical-screening` accepted, `opens_level null` | `paper/evidence/result-gates.json:31-310` | as recorded |
| Claims — CLM-001…CLM-028 all `verified`; four typed manifests bind bundle files by Git blob + SHA-256 (`paper/evidence/manifests/*.json`) | `paper/evidence/claims.json` | as recorded |
| Checker — flattens `\input`, required sections, forbidden wording, claim text equality, gate/artifact hashes, byte-identical regeneration of evidence/TeX | `paper/scripts/check_paper.py:19-34,360-647,951,1260,1980-2002`; 58 tests / 298 subtests pass (this audit) | **ACCEPTED evidence** (binding) |
| Reproducible PDF — `build.py` deterministic (SOURCE_DATE_EPOCH, two builds byte-identical required by `verify_reproducible_build.py`); no PDF is tracked (`*.pdf` and `paper/build/` ignored); recorded 17-page build sha256 `6b4c6978…` at `f171e9ec` in `paper-devlog.md:384-387`; **reproduced byte-identically in this audit** (section 2.13.1) | `paper/scripts/build.py:55-150`; `paper/scripts/verify_reproducible_build.py:21-60`; `.gitignore:82` | **ACCEPTED evidence** (build) |
| Explicitly NOT in the paper: L1-L3 results (closed by construction); PIC / self-consistent plasma (`sections/wall-loss-v4.tex:138-141`); MDO/optimisation outcomes (`manuscript.tex:473-475`); experimental validation (`manuscript.tex:405-406,460-463`); nulls are not existence disproofs (`manuscript.tex:356-358,443-444`) | as cited | honest |
| Submission gates — AUTHOR-IDENTITY satisfied; COAUTHOR / CONTRIBUTION / AFFILIATION / CORRESPONDING-AUTHOR `human-approval-required`; `author-checklist.md` all items unchecked | `paper/evidence/submission-gates.json:6-49`; `paper/author-checklist.md:5-107` | **NOT DONE** (human gates) |

#### 2.13.1 PDF build performed for this audit

`python paper/scripts/build.py` was run once from the fresh worktree at `44b7c8dc`
(MiKTeX pdflatex + bibtex, 20 s). `check_paper.collect_errors` passed, the build produced
`paper/build/manuscript.pdf` — **17 pages, 391,734 bytes, SHA-256
`6b4c6978e56fd5c225a24387f44a84ec080b19f8074733e7d3766b04d34f8701`** — which is
byte-identical to the hash recorded at `f171e9ec` in `modern/docs/workstreams/paper-devlog.md:384-387`
(the tree's paper inputs have not changed since). The output stays gitignored
(`paper/.gitignore:1`). Verdict for "reproducible PDF": **ACCEPTED evidence** (build and
binding only; it accepts no physical result).

### 2.14 Scorecard against the original request

| Original ask | Verdict | Basis |
|---|---|---|
| Rebuild the MATLAB MDO study in Python / C++ / GPU | **Partial (RECORDED-DEVELOPMENT).** Python: L0 conservation model, corrected 4-cell/N-cell discharge network, geometry, L1a/L1b/P2 field solvers, orbit_mc, PIC. C++: one kernel (`cusp_arrival_probability`). GPU: L0, fields, L1b, PIC 1-D/2-D on Warp/CUDA. The legacy end-to-end chain (FEMM → cusp windows → 30-unknown least squares → performance → SAEA) is quarantined by design and not reproduced; FEMM is never driven; no optimiser has been run. | sections 2.1, 2.2, 2.9, 2.10 |
| Audit the legacy code's errors | **ACCEPTED evidence.** 9 confirmed defects + 10 suspicious items catalogued; all confirmed defects fixed or superseded; 6 physics questions still open. | section 2.1 |
| SOTA plasma simulation | **RECORDED-DEVELOPMENT.** Verified 2-D axisymmetric electrostatic PIC-MCC with real Xe cross sections on the qualified P2 field; three runs, no plateau, nothing preregistered. | section 2.8 |
| SOTA surrogate / MDO methods | **NOT DONE (as runs); software exists.** BoTorch/pymoo provisioned outside the repo; no optimisation campaign on any new-physics model; every learned-surrogate campaign (9 L0, 10 L1a) ended negative or invalidated. | sections 2.2, 2.3, 2.9 |
| Everything run, validated, visualised | **Partial.** Runs: 4 accepted numerical results, 2 accepted nulls, ~30 preserved negative/failed campaigns (9 L0 surrogate, 10 L1a field surrogate, 7 wall-cusp held-out, 3 wall-loss, four-cell v1, l1a_plasma_coupling). Validation: numerical/replay only; external physical validation 0 %. Visualisation: 10 offline dashboards, browser-checked manually. | sections 2.4-2.8, 2.11, 2.12 |
| Updated paper | **Partial.** 17-page evidence-gated, reproducible manuscript carrying L0 + wall-loss v4 + topology screening; L1-L3 closed; no PIC/MDO/experimental section; human authorship and submission gates open. | section 2.13 |

---

## 3. Scientific findings so far (interpretation, not claims)

Each item is labelled interpretation; the committed classification of the underlying evidence
is given in brackets.

1. **The magnetic-mirror picture used by the original study is unsupported for this field
   family.** In 4608 collisionless test-particle orbits on the qualified P2 divergent-exit
   field there were 0 reflections in every case (95 % Wilson upper bound 0.0074); the outcome
   is a wall-hit versus axial-escape split decided almost entirely by launch cell (cells 2-3
   → 100 % wall hit; cell 4 → 100 % escape; cell 1 direction-dependent). The pooled 0.643 is an
   equal-weight design average of a bimodal per-cell result, not a physical loss fraction.
   [`collisionless_prescribed_field_test_particle_wall_loss_not_pic`; CLM-016/017;
   `cft_orbit_wall_loss_v4/POSTHOC_AUDIT.md:211-218`.] Interpretation.
2. **The four-cell wall-cusp topology the legacy parameterisation assumed has not been shown
   stable anywhere in the explored design space.** 0/128 preregistered candidates satisfied
   the strict cusp/cell definition (10-18 interior wall cusps per map instead of 4), and 0
   stable eligible cusps / 0 cells were found across 56 characterised designs × 3 maps; axis
   cusps (3-5 per design in sweep v2) do exist. This is a null under frozen definitions, not
   proof that no such design exists. [`GATE-FOUR-CELL-V2` preregistered-null;
   `GATE-TOPOLOGY-CHAR-V1` recorded-characterization; CLM-028.] Interpretation.
3. **With static neutrals the 2-D PIC-MCC discharge has only two regimes: runaway avalanche
   or no ignition.** At n_g = 5e20 and 1e20 m⁻³ the ionisation avalanche outran the kinetic
   ion loss (loss/source 2.6 % at 55.6 ns in v1; ion loss 10-35 % of ionisation after one
   transit in v2, ν_iz·τ ≈ 2.9 > 1) and no plateau formed; at the v1.2 operating point
   (n_g 1.5e19 m⁻³, ν_iz·τ ≈ 0.44) the uncommitted steady-state run shows the electron
   population decaying from 1.06e5 to 4.4e4 macro-particles with I_d ≤ 0.1 mA against 3 mA
   injected, i.e. no ignition. A physical steady state therefore needs a neutral inventory
   (depletion) closure; model v1.3 (`520e6b41`, landed during this audit) supplies one, and
   its first run (steady-state v2, protocol `67b04f87`) has not been executed or committed.
   [development evidence only; `pic2d-devlog.md:130-135,197-204,252-268` and the phase-3b
   section added at `67b04f87`; uncommitted `status.jsonl`.] Interpretation.
4. **Cusp-field electrons are non-adiabatic where it matters.** Per-orbit magnetic-moment
   variation had median 0.14 and maximum 0.63 with 60.5 % of orbits above 0.1, stable across
   time steps and maps; the wall-cusp v6/v7 held-out criterion (0/720 physically adiabatic
   orbits) failed for the same reason. This is why wall-loss authority moved from
   gyro-averaged criteria to full orbits. [`gates.json` `diagnostics_not_gates`;
   `origin/exp/cft-wall-cusp-validation-v7` terminal.] Interpretation.
5. **Learned field surrogates on the present label budget do not beat the coarse solve.** Over
   ten L1a field-surrogate campaigns the best worst-case field L2 error was 6.5 % (v8) against
   a 5 % gate, and every v10 localised interpolator (32-34 %) was worse than the uncorrected
   coarse solve (30.5 %); on L0 the only "passing" GP (v9) was a tautology. [development
   rejections; `origin/exp/l1a-field-surrogate-v{8,10}` bundles; `l0_surrogate_v9/POSTHOC_AUDIT.md`.]
   Interpretation.
6. **The legacy solver's archived results cannot be trusted as a baseline.** The `Ua`/`Ia`
   global/local split (C1) and Boolean residual rows (C2) mean the MATLAB least-squares output
   was not solving the stated model; the ISTS 2017 paper's own inconsistencies (3 vs 4
   objectives, 100 vs 50 generations, surrogate 5 % criterion failed yet Sobol indices
   reported) make it source evidence, not an oracle. [`AUDIT.md:42-70,176-203`; CLM-001/002.]

---

## 4. Remaining work to a defensible "complete"

Ordered by dependency. Effort is a planning estimate in focused working days for one person
with the current tooling; it is not evidence.

| # | Work item | Definition of done | Depends on | Estimate |
|---|---|---|---|---|
| 1 | Commit and close pic2d steady-state v1 as a no-ignition reference | `results/` committed with summary, series, checkpoint hashes; devlog entry; dashboard panel; explicitly `development_screening_not_preregistered` | running job terminates | 1 d |
| 2 | pic2d model v1.3 neutral inventory — **landed at `520e6b41` during this audit** (spec, `neutrals.py`, 9 tests); remaining: ion-neutral CEX either added with a hash-bound cross-section source or explicitly excluded; artificial τ_g relaxation justified or removed in a later model | 1 | 1-2 d remaining |
| 3 | Steady-state v2 (protocol `67b04f87`) to a plateau on the fine grid | plateau rule (< 5 % drift in I_d, N_e and n_g, ≥ 3 τ_i) met with resolvable λ_D and ω_pe·Δt; results committed; then convergence across particle weight and one finer grid | 2 | 5-10 d (GPU-bound; 12 h budgets) |
| 4 | Preregistered pic2d campaign through `experiment_runtime` | lock → single execution → binding gates (conservation, convergence, uncertainty) → replay → `accepted_result`; independent comparison target declared up front | 3 | 5-8 d + run time |
| 5 | Coupling v4.2 consumer integration of the v4 wall-loss export | a consumer in `coupling/` or `plasma_network/` ingests `coupling-export-only.json` as opaque verified evidence under audit; `export_only_pending_consumer_integration` removed by that audit, never by relabelling; l1a_plasma_coupling ported off the v2 proxy | none | 3-5 d |
| 6 | Revised, preregistered topology hypothesis (or criterion) for a field-driven plasma state | protocol frozen before any run; either a design family that yields stable wall-cusp cells under the strict definition or a redefined cell criterion with source; then a plasma-network solve on real topology | 5 | 5-10 d |
| 7 | MDO on the new physics with the provisioned GPU stack | campaign spec v1.4 instance over L1a-sweep-v2 design space (or P2-qualified subset) with frozen F3 reference and cost model; qLogNEHVI/NSGA-III runs recorded through `experiment_runtime`; hypervolume against a Sobol/Halton baseline; `campaign-v1.json` `results` populated; `pymoo`/BoTorch pins made satisfiable | 4 or 6 for a physics objective; can start on L0/L1a objectives now | 8-15 d |
| 8 | Material-aware field qualification | nonlinear B-H, PM recoil, open boundary in P2 FEM; FEMM full-profile parity on the legacy geometry; L1b gates evaluated at the preregistered resolution | hardware with the memory the plan needs | 10-20 d |
| 9 | External validation against published data | licensed dataset with uncertainty (thrust/I_d/Faraday/RPA) for a declared context of use; model-to-experiment comparison recorded through `validation` v2 contracts; opens GATE-L1 at most | 3-8 | 10-20 d + data access |
| 10 | Release checklist | fresh-clone audit script (clean checkout, optional-dep closure, dashboards render, `verify_reproducible_build.py` byte-identical, provenance/secret/path/FYP checks); `validate_results` LF refusal fixed; `Results/` ignore rule replaced; 7a9aff4d reviewed; PIC section only if item 4 lands; human authorship/licensing/submission gates | 1-9 as available | 3-5 d |

---

## 5. Known debts and risks

Debts (from the committed tree and the tracker, verified here):

- `modern/experiments/four_cell_topology_search_v2/experiment.py:1844-1849` — `validate_results`
  refuses the admitted bundle on LF checkouts (protocol digest compared against the live
  working-tree hash). Code defect, disclosed in `POSTHOC_AUDIT.md:112-121`, not fixed.
- Root `.gitignore:47-48` `Results/` matches `results/` case-insensitively on Windows; every
  evidence bundle was force-added (`cft_orbit_wall_loss_v4/POSTHOC_AUDIT.md:167-172`;
  `test-health-devlog.md:47-51`). `l1a_plasma_coupling/results/` remains untracked.
- `modern/experiments/l1a_plasma_coupling/experiment.py:20,61-67` still calls the deprecated
  coupling v2 `build_screening_proxy` (7 `DeprecationWarning`s in the audit test run).
- `7a9aff4d` (material_fields hash-leaf re-bind) is an isolated commit flagged for review and
  not yet reviewed.
- The field-surrogate stop decision (v3-v10) and its rationale exist only on remote branches
  and in the tracker; `surrogates-devlog.md` / `surrogates-learning-ledger.md` do not mention
  them. The tracker's "0.209 s fine vs 0.105 s coarse" latency figures are not in any bundle.
- `pic2d.__version__` is `0.1.0` while the specs record model versions 0.2.0 / 0.2.1; most
  packages expose schema constants rather than `__version__` (section appendix B).
- `modern/pyproject.toml:20-25` optional extra pins `gpytorch<1.15` while `botorch>=0.18.1`
  requires `gpytorch>=1.15.2` (`.venv-sota/provision-report.txt`); the extra is unsatisfiable.
- `pic2d_cft_snapshot_v2/protocol.json` references a non-existent
  `operating-point-v1.1.json` (`pic2d-devlog.md:316-318`); left because its hash is bound.
- `\EvidenceRevision` renders the L0 hash with a stray space (tracker P8; pre-existing).
- Two `-text` mixed-EOL logs under `cft_wall_cusp_validation_v2/results/`.
- `fem-reference-learning.md:140-146` is stale relative to the completed third-level campaign.
- No automated browser test for any dashboard; acceptance rests on logged manual sessions.
- The native pybind11 extension is not built anywhere; its test is permanently skipped.

Risks:

- **Static-neutral PIC cannot produce the deliverable.** Both committed operating points
  avalanche and the uncommitted one does not ignite; without the v1.3 neutral closure there is
  no path to a plateau, and the closure itself introduces a new, unvalidated model term.
- **Windows WDDM launch floor (~1.2 ms/step)** caps pic2d throughput; long plateau runs are
  wall-time-bound (12 h budgets). CUDA-graph capture or a TCC/Linux host is a hard
  prerequisite for convergence studies.
- **No physical topology input exists** for the corrected plasma network; every path to a
  field-driven plasma state runs through a hypothesis that has not been written down.
- **The rebuild has never run an optimiser**, so the central claim of the original study (a
  Pareto front over eight design variables) has no counterpart yet; when it does, the
  objective will be a field-only or test-particle quantity unless items 3-6 land first.
- **Evidence-bearing artifacts depend on hash bindings that were once platform-dependent**;
  the CRLF arc is closed by the LF pin and lint, but any pre-`fab0eccc` worktree must be
  re-smudged before computing hashes (`test-health-devlog.md`).
- **Single author, single machine.** All acceptance to date is internal; no independent domain
  review or external replication has occurred (P7 external 0 %).

---

## 6. Corrections to the tracker made by this audit

| Tracker statement (canvas at audit start) | Committed evidence | Action |
|---|---|---|
| Overall 70 % (65-75 %) | re-derived 63 % (58-68 %) with the same weights and components | phases P0/P1/P3/P4/P5/P6/P8 re-scored (section 1.1) |
| "1677 passed / 0 failed / 5 skipped"; "paper/tests 33 passed / 177 subtests" (Infrastructure row) | 1702 / 0 / 5 at `44b7c8dc`; paper 58 passed / 298 subtests | updated |
| "CUDA Warp … 1.6–1.8 s/orbit" | only 1.6 s/orbit is recorded (`orbit-mc-devlog.md:286`) | corrected |
| "11–17 interior wall cusps per candidate" | 11-17 is the primary-map range; 10-18 across all three maps (`dataset.json`) | qualified |
| v10 "fine 0.209 s vs coarse 0.105 s" | not present in any v10 bundle or doc | flagged as unverifiable |
| "0 CRLF files" | 0 smudge-divergent files; 3 `w/crlf` + 2 mixed stored deliberately under `-text` | qualified |
| "no unmerged branches except PIC phase 3" | `origin/feat/pic-2d-axisymmetric` == `origin/feat/sota-foundation` at `44b7c8dc`; the v1.3 work was uncommitted at the snapshot and landed directly on `feat/sota-foundation` as `520e6b41` / `67b04f87` during the audit (section 7) | updated |
| "browser-tested" dashboards | manual headless sessions logged; no automated browser tests | qualified |
| "Final roadmap audit queued behind PIC phase 3" | performed now, against `44b7c8dc`; PIC phase 3 left as in progress | action marked done |

---

## 7. Head movement during the audit

While this audit was being written against `44b7c8dc`, the pic2d agent pushed two commits to
`origin/feat/sota-foundation`; the audit commit was rebased onto them (never force-pushed).

| Commit | Content | Effect on verdicts |
|---|---|---|
| `520e6b41` `feat(pic2d): v1.3 quasi-steady neutral inventory with conservation tests` (2026-09-03 07:45 +1000) | `modern/spec/pic2d/pic2d-model-v1.3.json`; `pic2d/neutrals.py` (224 lines); MCC real-frequency scaling by n_g/n_g0 on CPU and Warp; n_g and four atom ledgers in checkpoints; 9 tests (`test_pic2d_neutral_inventory.py`: ledger closure, analytic fixed point, MCC scaling, bitwise resume, tamper detection, GPU parity) | Software only. P5 software component would move 25 → 26 at most; no simulation evidence; P5 stays **50** |
| `67b04f87` `feat(pic2d): finalize command and steady-state v2 experiment` (07:47 +1000) | runner made protocol-generic with a `finalize` command; new `modern/experiments/pic2d_cft_steady_state_v2` (protocol, README, thin runner; `development / screening`); +3 runner tests (11); devlog phase-3b section and learning-scratchpad entries; `tests/pic2d tests/pic tests/orbit_mc tests/visualization` reported 388 passed | Protocol only; **no results for steady-state v1 or v2 are committed**; no verdict changes |

The full-suite count (1702 / 0 / 5) in this document is for `44b7c8dc`; the two commits add
12 pic2d tests by their own record and were not re-run as part of this audit.

---

## Appendix A — Test run record

`modern/tests` collected 1707 items; passed by directory (pytest `--collect-only` on the same
tree): root 66, active_learning 96, coupling 143, experiment_runtime 133, experiments 222
(cft_orbit_wall_loss_v4 38, four_cell_topology_search 9, four_cell_topology_search_v2 10,
four_cell_topology_search_visualization 11, l0_surrogate 6 / v2 5 / v3 15 / v4 12 / v5 8 /
v6 7 / v7 10 / v8 10 / v9 13, l1a_field_surrogate_v1 5 / v2 5, l1a_geometry_sweep 6,
l1a_geometry_sweep_v2 23, l1a_geometry_sweep_v2_visualization 11,
l1a_geometry_sweep_visualization 10, l1a_plasma_coupling 8), fem_reference 37,
fem_reference_visualization 10, fields 62, geometry 41, hybrid 70, magnetics 52,
material_fields 40, optimization 76, orbit_mc 147, physics 86, pic 63, pic2d 73, plasma 36,
plasma_network 64, surrogates 38, validation 59, visualization 93. Warp/CUDA tests ran on the
host RTX 5090 concurrently with the pic2d steady-state job; none failed.

Paper build (`python paper/scripts/build.py`, repo root, 2026-09-03): exit 0 in 20 s;
`paper/build/manuscript.pdf` 17 pages, 391,734 bytes, SHA-256
`6b4c6978e56fd5c225a24387f44a84ec080b19f8074733e7d3766b04d34f8701`, equal to the value
recorded at `f171e9ec`; `build-provenance.json` and `tool-versions.json` written alongside
(all gitignored).

## Appendix B — Package inventory at `44b7c8dc`

| Package | Version identity | Source files | Collected tests | Downstream accepted evidence |
|---|---|---|---|---|
| physics | ledger `spec/physics/equation-ledger.json` (13 eq.) | 7 | 86 | L0 8,192-point sweep (`FIRST_RESULTS.md`; CLM-005…008) |
| plasma | `spec/plasma/equation-ledger.json` (R00-R27) | 5 | 36 | none (numerical foundation) |
| plasma_network | — | 8 | 64 | none (0 coupled candidates) |
| geometry | `axisymmetric_cft/1.1.0`, viewer 1.1.0 | 7 | 41 | geometry artifacts; input to sweeps |
| magnetics | handoff 1.0.0 | 6 | 52 | none (contracts) |
| fields | artifact 1.2.0 (legacy 1.1.0) | 7 | 62 | L1a sweep v2 (`GATE-L1A-SWEEP-V2`) |
| fem_reference | result 1.3.0, viewer 1.1.0 | 11 | 37 (+10 viz) | P2 divergent-exit `NUMERICAL_P2_QUALIFIED` (`a1158bad`) |
| material_fields | result 1.4.0 | 9 | 40 | none (`SCREENING_NOT_ACCEPTED`) |
| coupling | v2 2.0.0 / v3 3.0.0 / v4 4.2.0 | 17 | 143 | v3 used by admitted nulls; v4 held-out never promoted |
| orbit_mc | `__version__ 1.7.0`; schemas 1.6.0; handoff 1.3.0 | 8 | 147 | wall-loss v4 (`GATE-WALL-LOSS-V4`) |
| hybrid | checkpoint v1 | 9 | 70 | none |
| pic | `pic-foundation-r2-staggered-transactional` | 8 | 63 | none |
| pic2d | `__version__ 0.1.0` (specs 0.2.0 / 0.2.1) | 11 | 73 | none (development runs) |
| optimization | campaign spec 1.4 | 8 | 76 | none (`results: null`) |
| active_learning | spec v1 | 8 | 96 | none |
| surrogates | exact-gp 2.0.0, pod 1.0.0 | 10 | 38 | none (benchmark failed) |
| validation | contract 2.0.0 | 5 | 59 | none (contracts) |
| experiment_runtime | `cft-typed-canonical-json-v1` | 6 | 133 | carried wall-loss v4 to `accepted_result` |

Git state at audit: 86 commits on `feat/sota-foundation` (first `7ca3dc2d` 2026-09-01,
head `44b7c8dc` 2026-09-03); 125 commits reachable from all 29 remote branches
(`exp/cft-orbit-wall-loss-v1..v4`, `exp/cft-wall-cusp-validation-v3..v7`,
`exp/l1a-field-surrogate-v3..v10`, `feat/orbit-mc-v1.5/1.6/1.7`, `feat/pic-2d-axisymmetric`,
`feat/topology-dashboard`, `feat/wall-loss-v4-dashboard`, `fix/test-health-crlf-era`,
`fix/visualization-stale-gallery`, `paper/topology-and-sweep-claims`,
`paper/wall-loss-v4-claim`, `main`, `feat/sota-foundation`); 13 local worktrees before this
audit's `docs/roadmap-audit`. `origin/feat/pic-2d-axisymmetric` is at the same head as
`feat/sota-foundation`.
