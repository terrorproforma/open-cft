# Devlog — orbit wall-loss geometry screening v1

Classification of everything here: `SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`.

## 2026-09-03 10:10–10:55 AEST — scaffold, shakedown, preregistration

- Worktree `uni-project-orbit-geo`, branch `exp/orbit-wall-loss-geometry-screening-v1`
  from `origin/feat/sota-foundation` (`8babb31e`), LF verified.
- Field provenance decision: only the four sweep-v2 representatives carry stored
  full-field maps, so every design is re-solved with the accepted L1a CPU solver
  (`solve_problem_cpu`, ~2 s per design at 80×144). Probe on design 000: the
  re-solve reproduces the stored CUDA map to 1.5e-21 Wb (ψ) and 5e-16 T (B),
  and the geometry/case hashes equal the sealed raw record; bore interpolation
  rms 0.8 % of |B|max, 2× refined cross-resolution rms 0.6 %.
- Launch rule: four cells at 1/8, 3/8, 5/8, 7/8 of the channel-straight span
  `[injector_length, exit_start]`, radii 0.675/0.800 r_w, v4 energies/pitches/
  directions, 8 gyrophases at 11π/96 (disjoint mod π/4 from v1–v4 and the
  shakedown grid 5π/96). Wall authority `0 ≤ z ≤ exit_start` (v4 rule), domain
  `0 ≤ z ≤ L`, `max_path = 2 L`, `max_time = 2·max_path/v(5 eV)`.
- Shakedown run 1 (10:36): `runtime_failure` in development —
  `CanonicalizationError: unsupported canonical type: bool` (numpy bools in the
  field-evidence checks). Fixed with `_plain()` on every worker payload.
- Shakedown run 2 (10:37): `accepted_result`, 7 cases, 238 validators / 0
  failures, bundle validated, 3/3 designs sealed and handoffs consumed; 107
  reflections in 448 orbits (v4 had zero). Per-orbit 98 ms (N) / 239 ms (2N)
  under concurrent host load → projection 4185 s wall at 12 workers for 196
  cases (100 352 launches, three integrations each) → extension batch included.
- Runs 3–4 re-bound the protocol after wording/`experiment_code_sha256`
  changes; identical outcomes (deterministic).
- Commits: `484335c2` tests (41 passed, 1 skipped), `c86bfca3`
  `preregister orbit wall-loss geometry screening v1` (protocol `dfebcfea`,
  design authorities `43dcd6de`, orbit_mc source `9e3f8712`, field pipeline
  `984f4a66`, shakedown `d988874d`); pushed.
- Execution launched 10:55 from the clean detached worktree
  `uni-project-orbit-geo-run` at `c86bfca3` (`run execute`, 12 workers, CPU
  only; Git-common lock `orbit-wall-loss-geometry-screening-v1.execution.lock`).
- Dashboard generator/template/tests committed as `ce7cb895` while the run
  proceeded (developed against the NON-EVIDENTIARY shakedown bundle with
  `--allow-non-evidentiary`; two headless-render defects fixed: `.axis path`
  needed `fill:none`, and the http.server must be polled for LISTENING before
  the screenshot).
