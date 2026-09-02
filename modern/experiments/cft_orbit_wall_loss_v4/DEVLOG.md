# Devlog

## 2026-09-03 — V4 phase 2: orbit_mc v1.6 rebase, LF, shakedown, preregistration

- `git merge --ff-only origin/feat/sota-foundation` → HEAD
  `3ab50ef5c31cfa45f2256ddba18dafa965010c7a` (orbit_mc v1.6,
  `cft_revival.orbit_mc.__version__ == "1.6.0"`). The worktree predated the
  repo-wide LF pin (`fab0eccc`), so 1018 tracked files were CRLF on disk;
  they were deleted in batches and re-smudged with `git checkout --` (count
  now 0) and every untracked v4 file was rewritten LF/UTF-8-no-BOM before any
  hash was computed. `orbit_mc_source_sha256()` now fails closed if any scoped
  file contains a CR byte.
- Contract adoption: `protocol.json#orbit_mc_contract` binds
  `package_version 1.6.0`, result/checkpoint/validation-protocol schema
  `1.6.0` and handoff `coupling-v4.2/1.3.0`; `orbit_mc_contract_report`
  observes `__version__` and the spec file's `schema_version`; `execute`
  re-verifies package version and schema versions against `authorities.json`
  in addition to the source and shakedown hashes.
- Synthetic preflight vector matrix gained `event_velocity_m_per_s`,
  `step_magnetic_midpoint_t`, `step_electric_midpoint_v_per_m` plus a zero
  failure-witness block (11 covered field groups).
- `_result_gate_report`: new check `final_velocity_equals_event_velocity`
  (exact tuple equality, `gates.require_final_velocity_equals_event_velocity`);
  the 1e-10 energy gate stays binding. Magnetic-moment variation moved out of
  the gate namespace into `diagnostics_not_gates.magnetic_moment_variation`
  (min/median/max, counts above 0.1/0.5) and `protocol.json#diagnostics`
  states that μ is diagnostic only.
- Shakedown on v1.6 (99.9 s wall, 9 workers, HEAD `3ab50ef5`, 12 dirty
  entries = the untracked v4 files): `accepted_result`, bundle validated, 289
  validator invocations, 0 failures, 9/9 cases exported. Every case: 38
  `wall_hit` / 22 `domain_escape` / 4 `reflected`, 23–25 tolerance-close
  events, 39–41 interpolated; **max relative energy error 0.0, 0/576 orbits
  with non-zero energy error, 576/576 final-velocity == event-velocity**. All
  15 informational gate checks pass (energy included). μ diagnostic per case:
  min ≈ 0.019, median ≈ 0.12, max ≈ 0.78; 38–41 of 64 orbits above 0.1.
  Integration 7.0–7.4 s (N), 14.9–15.2 s (2N), 30.1–30.3 s (4N) per 64
  orbits; export stage 48.7 s. orbit_mc source hash (LF)
  `007c2d51a44d74f989dae6938d10538454886f2e4970f9a9867aaeac8346aa43`.
- Tests: `tests/experiments/cft_orbit_wall_loss_v4` 24 passed, 1 skipped
  (frozen case files appear after `prepare`); `tests/orbit_mc` 120 passed;
  `tests/experiment_runtime` 132 passed, 1 skipped (symlink privilege).
- Next: commit A (tests), `prepare`, commit B `preregister CFT full-orbit
  wall-loss v4`, push, one detached execution, commit C with results.

## 2026-09-03 — V4 phase 1: scaffold and shakedown proof (orbit_mc v1.5)

- Created worktree `uni-project-cft-orbit-wall-loss-v4` on
  `exp/cft-orbit-wall-loss-v4` from `origin/feat/sota-foundation`
  (`7cf65053c7b5f7efff04c2b98dbfbd7bc10ad610`, orbit_mc v1.5). No commit,
  no push, no `prepare`, no `execute` in this phase.
- Studied v3 end to end (protocol, adapter, experiment, run, results bundle).
  v3 died in `assessment` with `physical event witness requires a positive
  step` after three P2 adapter accesses and one label access
  (`orbit-primary-N`); zero persisted outcomes.
- Ported v3 to v4 with every ID/path/schema string moved to
  `cft-revival.cft-orbit-wall-loss-v4.*/4.0.0`; the P2 adapter is unchanged.
- Introduced `CampaignPlan` so the evidentiary campaign and the shakedown run
  the same prebundle/development/assessment code with different design and
  decision parameters (`binding_gates`).
- Added `orbit_mc_source_sha256()` (all `orbit_mc/*.py` + `spec/orbit_mc/*.json`,
  refusing an orbit_mc imported from outside this worktree) and an
  `orbit_mc_contract` protocol section binding the result/checkpoint/handoff
  schema versions (`1.5.0`/`1.5.0`/`1.3.0`).
- Fresh evidentiary design: radii 1.35/1.60 mm, z 3.5/9.5/15.5/21.5 mm,
  gyrophase offset 7π/48 (= v3 π/16 + π/12). Shakedown design: RNG positions
  (seed namespace `cft-orbit-wall-loss-v4:shakedown`), one gyrophase at
  17π/96, 64 launches per case, batch size 8, partial prefix 4. Disjointness
  against evidentiary-v4 and v1/v2/v3 is proven on launch IDs, seeds,
  positions, (E, pitch, direction, gyrophase) and full phase-space tuples.
- Added `preflight_campaign` (new in v1.5) before every case, a validator
  ledger recording every validator call, and parallel case execution: all
  nine label accesses are recorded in case order, then a spawn process pool
  runs the cases; the main process re-integrates two launches per case and
  compares canonical result records (`cross_process_determinism_sample`);
  the artifact stage (replay + verified reload + handoff) also runs in the
  pool after the gate report.
- Shakedown run 1 (57 s): all nine cases integrated and checkpointed in
  parallel, then the assessment raised `zip() argument 2 is shorter than
  argument 1` inside `_convergence` — a latent v3 bug (`zip(ordered,
  ordered[1:], strict=True)`) that v3 never reached. Fixed to
  `zip(ordered[:-1], ordered[1:], strict=True)`; added a regression test.
  Also made the collector record per-case diagnostics before the convergence
  step so a later failure still leaves them in `shakedown.json`.
- Shakedown run 2 (117.6 s wall, 9 workers): `accepted_result`, bundle
  validated, 289 validator invocations, 0 failures, all nine cases exported
  (sealed artifact + deterministic replay + verified reload; handoff built for
  refined-4N). Every case: 38 `wall_hit` / 22 `domain_escape` / 4 `reflected`,
  23–25 tolerance-close events (36–39 %), no `field_failure`/`step_limit`.
  Integration 8.1–8.5 s (N), 16.1–16.5 s (2N), 30.7–31.3 s (4N) per 64
  orbits under nine concurrent workers; export stage 62.7 s wall. Informational
  gates: only `energy` fails (max 3.41e-3, 362/576 orbits over 1e-10) —
  expected on v1.5. Wall endpoint error 4.3e-19 m; max μ variation 0.78.
- Tests: `tests/experiments/cft_orbit_wall_loss_v4` 21 passed, 1 skipped
  (frozen case files exist only after `prepare`).
- Full-campaign projection (512 launches × 9 cases, nine workers): 4N cases
  dominate at ~0.49 s/orbit under contention → ~4 min integration + ~8.5 min
  for the two replays ≈ 13 min wall; N/2N cases finish earlier.
- Pending for phase 2: rebase onto orbit_mc v1.6 (energy-consistent event
  velocity; new witness fields; schema bump), re-run the shakedown (the
  orbit_mc source hash and contract section make this mandatory), `prepare`,
  commit `preregister CFT full-orbit wall-loss v4`, push, execute once.
