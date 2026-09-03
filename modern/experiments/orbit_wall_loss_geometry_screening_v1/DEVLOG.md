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

## 2026-09-03 10:55–12:31 AEST — single execution and record

- `run execute` from the clean detached worktree `uni-project-orbit-geo-run` at
  `c86bfca3`: terminal state `accepted_result`, status
  `accepted_screening_dataset`, 96 designs / 196 cases / 100 352 orbits, 6664
  validators passed / 0 failed, 196/196 cases sealed (deterministic write replay
  + verified reload) and their handoffs consumed, 0 exclusions, 0 timeouts,
  structural gates PASS for every design; bundle `validate_bundle` OK
  (2846 manifest entries, 23.8 MB tracked).
- Wall time 95 min (cases 5463 s at 12 workers) against the shakedown projection
  of 4185 s (70 min) and the 90 min budget: the projection under-estimated the
  contention with the concurrent MDO/PIC load (median 2N per-orbit cost 242 ms,
  max 650 ms vs the shakedown's 238 ms mean). Disclosed; no gate depends on time.
- Dataset headline (accepted-2N, 512 launches per design): P(wall) 0.375–0.869,
  median 0.702; all 96 designs converged N→2N (max |ΔP| 0.0059, mean 0.00035);
  reflections in EVERY design (32–282 of 512 at 2N; 11 268 = 22 % of the 2N
  orbits) where v4's P2 design had zero; domain escape 0–0.215 (median 0.069;
  upstream anode plane 1635, exit plane 1127, divergent-section radial 862 at
  2N); per-cell mean P(wall) 0.65 / 0.82 / 0.77 / 0.55 for cells 1–4, 94 of 384
  design-cells saturated at 1.0 and none at 0.0, so the v4 bimodal cell
  structure (cells at 1.0 next to a cell at 0.0) appears in 0 of 96 designs.
  Least wall loss: long channels (`049`, `094`, `050`; L 20–29 mm, r_w 1.8–2.1
  mm, 255–282 reflections); most: `091`, `021`, `043` (L 18–22 mm, small exit
  fractions, 32–74 reflections).
- Field diagnostics: bore interpolation rms ≤ 0.87 % of |B|max for every
  design, 2× cross-resolution rms ≤ 0.66 %; refined-N orbit probability within
  0.0078 of accepted-N on all four representatives.
- μ variation (diagnostic only): per-design medians 0.11–0.47; tolerance-close
  event share 19–49 %.
- Dashboard `modern/visualization/wall-loss-geometry-screening-v1.html`
  (970 340 B, deterministic, offline) regenerated from the sealed bundle; 8
  dashboard tests and 49 experiment tests green in the run worktree; headless
  Chrome screenshots at 1440 px and 430 px under `%TEMP%\orbit_geo_probe\`.
