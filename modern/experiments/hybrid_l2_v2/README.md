# Hybrid L2 v2 - per-cell hybrid on the reference material-aware field vs the PIC base plateau

Model, cell concept, closures, gates and claim boundary: `modern/docs/hybrid-l2-v2.md`.
Protocol: `protocol.json` (frozen at the preregistration commit; `preflight` and `assess` re-derive the
closures and the PIC reference table from the PIC artifacts and refuse on any disagreement).

## Stages

From `modern/` with `$env:PYTHONPATH="$PWD\src;$PWD"` (CPU only; `$env:OPENBLAS_NUM_THREADS="4"` avoids
BLAS oversubscription when several cases run side by side):

```
python -m experiments.hybrid_l2_v2.run preflight              # real field on the three grids, partition check, closures, ms/step
python -m experiments.hybrid_l2_v2.run shakedown              # synthetic full path + real short run, through finalize + assess
python -m experiments.hybrid_l2_v2.run launch --case base --expect-commit <prereg sha>
python -m experiments.hybrid_l2_v2.run launch --case <spatial-coarse|spatial-fine|temporal-coarse|temporal-fine|weight-half|seed-b|closure-g-low|closure-g-high|closure-w-low|closure-w-high> --expect-commit <sha>
python -m experiments.hybrid_l2_v2.run status
python -m experiments.hybrid_l2_v2.run assess                 # GATE-L2 metrics over every finished case -> results/assessment.json
```

Each case writes `results/` (base) or `results-<case>/`: `series.jsonl` (untracked), `status.jsonl`,
`checkpoint-latest.*` (untracked), and at the end `maps.npz`, `series.npz`, `summary.json`,
`l2-targets.json` (the mini-sweep extraction applied to L2's own maps), `checkpoint-final.*`
(the `.npz` particle arrays untracked, the metadata and the field anchor tracked).

## Cases

| case | grid | dt | W | seed | closure scale | role |
|---|---|---|---|---|---|---|
| base | 60 x 480 (50 um) | 1 ns | 3e5 | 20260903 | 1 | headline comparison |
| spatial-coarse / spatial-fine | 30 x 240 / 90 x 720 | 1 ns | 3e5 | 20260903 | 1 | spatial levels |
| temporal-coarse / temporal-fine | 60 x 480 | 2 / 0.5 ns | 3e5 | 20260903 | 1 | temporal levels |
| weight-half / seed-b | 60 x 480 | 1 ns | 1.5e5 / 3e5 | 20260903 / 20260904 | 1 | statistical levels |
| closure-g-low/high, closure-w-low/high | 60 x 480 | 1 ns | 3e5 | 20260903 | G or w x 0.7 / 1.3 | input uncertainty |

## Launch log

* 2026-09-04 18:13-18:15 AEST - launch 1 of all eleven cases from the preregistration commit `386c9070`
  (clean worktree `.worktrees/hybrid-l2-v2`, `--expect-commit 386c9070`, O_EXCL lock per case), detached
  (`Start-Process`, `OPENBLAS_NUM_THREADS=2`), eleven CPU processes side by side on the 24-core host while the
  PIC v4 refinement (GPU) was finishing and another project loaded ~4 cores: PIDs base 26832, spatial-fine 9144,
  temporal-fine 51792, spatial-coarse 2604, temporal-coarse 804, weight-half 28976, seed-b 14364, closure-g-low
  32076, closure-g-high 25564, closure-w-low 12224, closure-w-high 50540. Step cost under this contention:
  150-250 ms/step at 1 us rising to 600-900 ms/step at 3.5 us as the ion count grew (330 k macro-ions in the
  base case at 3.7 us against the ~200 k projected from the PIC plateau density). Process priorities were
  raised for base / spatial-fine / temporal-fine and lowered for the four closure-sensitivity cases at 18:35.
* **Contention record (coordinator directive 19:45 AEST).** The eleven L2 processes were flagged at 19:45 as
  `warp-cuda:0` processes at 100 % on the local RTX 5090 competing with the preregistered PIC run
  `pic2d_cft_steady_state_v5` (PID 43572, launched 19:29 in `.worktrees/pic2d-ss5`, budget set to 48 h because its
  preflight timing was contended). What the host showed at 19:46: `nvidia-smi --query-compute-apps` lists no L2
  PID (the L2 runner is numpy-only - it never calls `wp.init()` and holds no CUDA context; GPU-minutes consumed by
  L2: 0), but the eleven processes at ~1 core each plus `OPENBLAS_NUM_THREADS=2` put the 24-core host at 94-96 %
  load, and the PIC v5 host thread (0.3 cores, GPU-bound) shared that saturated CPU. So the earlier stage of every
  case (18:13-19:46) ran under CPU contention on both sides, and the PIC v5 preflight timing was taken with the L2
  processes loading the host. Action taken at 19:46: all L2 processes set to `Idle` priority (they yield the CPU
  to any normal-priority thread, i.e. to the PIC host thread); the four closure-sensitivity cases stopped
  (`Stop-Process`, last checkpoints 19:39-19:45; series records past the checkpoint are dropped on `--resume`);
  six primary cases (base, spatial-fine, temporal-fine, spatial-coarse, weight-half, seed-b) kept running -
  temporal-coarse had already finished (`max_steps_reached`, 19:37). Host load after: L2 6 cores + others ~4
  of 24 (>= 8 cores free). Wall-time and ms/step figures recorded in every `summary.json` of this campaign are
  therefore contended-host numbers (upper bounds on the true L2 step cost); the cost ratio in `assess` uses them.
  For the record, the GPU contender visible in the host data is the PIC v4 refinement (`pic2d_cft_steady_state_v4`,
  GPU), whose `summary.json` was written at 19:31:26 AEST - it was still running when v5 launched at 19:29:53.
  20:07 AEST: the coordinator's constraint (>= 8 cores free) measured at 7.9-8.3 idle cores with five L2
  processes, so weight-half (old runner, no STOP hook; `Stop-Process`, checkpoint at step 5000) was stopped as
  well and a queue runner (outside the repository, `%TEMP%\l2_queue.ps1`) keeps at most FOUR L2 processes alive,
  resuming weight-half and the four closure cases in that order with `--resume`, `Idle` priority and
  `OMP/OPENBLAS/MKL_NUM_THREADS=1`.
* The runner gained the `STOP` file mechanism (a `STOP` file in a case's results directory ends the run at the
  next series record with `checkpoint-latest` and no finalize; `launch --resume` continues it and truncates the
  series to the checkpoint step) and the `sessions.json` entries record `git_head` and BLAS thread pins. The
  simulation code (`cft_revival.hybrid`) is unchanged since `386c9070`; `summary.json` records `git_head` at
  finalize time, which for sessions finalized after this commit is the runner commit, not `386c9070`.
* **2026-09-04 20:57 AEST - PARKED (user cancelled; coordinator directive).** Status: development model, NOT
  admitted. Record on `feat/hybrid-l2-v2` only (not merged into `feat/sota-foundation`).
  - Comparison **FAIL**: 24 of 28 compared quantities outside the preregistered tolerance (`results/assessment.json`):
    I_d 7.52 mA vs the PIC base 3.44 mA (+118 %; vs the PIC v4 refinement 3.80 mA: +98 %), anode ion fraction
    0.155 vs 0.014, peak n_e 1.50e19 vs 1.64e18 m^-3 (x9), n_g 1.69e19 vs 2.97e19 m^-3 (-43 %), S +51 %,
    gross utilisation 0.69 vs 0.46, cusp-2 near-wall drop -36 V vs +32 V (wrong sign). Within: cell-1 ion wall-loss
    fraction, cell-2 / cell-3 T_e, potential step 1. Interface conservation passed (charge identity 3e-9).
  - Resolution ladder incomplete: spatial 3/3 finished (coarse, base, fine) but temporal 2/3 (temporal-fine killed
    at 7.28 us / 3.03 transits), statistical 0/2 (weight-half killed at ~6 us, seed-b finished but W-half missing),
    closure sensitivity 0/4 (killed at ~5.3 us) - the remaining cases were killed by the coordinator at the user's
    request at 20:5x AEST; GATE-L2 verdict therefore `not_evaluable` on the ladder and `code_comparison_passed: false`.
  - Cost: PIC/L2 wall-clock ratio 1.66 (PIC base 10,141 s on the RTX 5090 vs L2 base 6,116 s on one contended CPU
    core) - no speed advantage as run; the preregistered 10-100x target was not met.
  - GPU contention disclosure: the coordinator observed ~11 CUDA processes at 100 % on the local RTX 5090 from 18:13
    to 19:4x AEST while the preregistered PIC v5 (PID 43572) executed and attributed them to this campaign. The L2
    runner is numpy-only and `nvidia-smi --query-compute-apps` listed none of the eleven L2 PIDs at 19:46 (0
    GPU-minutes by that measure); the eleven processes did saturate the host CPU (94-96 % load) for ~93 minutes, which
    contended the PIC v5 host thread and its preflight timing (its budget was set to 48 h for that reason). If the
    coordinator's attribution stands, the GPU-minutes to disclose are ~11 x 93 min; we could not confirm it from the
    host data. Recorded here as a disclosure either way.
  - Committed here: code, tests, docs, protocol / preflight / shakedown, the base-case result (`results/`: summary,
    maps, series, l2-targets, assessment, final checkpoint metadata + field anchor, sessions, lock; particle arrays,
    `series.jsonl`, `checkpoint-latest.*` and raw logs untracked) and the summaries of the four other finished cases
    (seed-b, spatial-coarse, spatial-fine, temporal-coarse), the dashboard `modern/visualization/hybrid-l2-v2.html` as-is.
  - Diagnosis (best physics guess, untested): the discharge current is set by the cusp conductance closure G_k, which
    was extracted from the PIC plateau as a *linear* conductance (cusp electron current / potential drop) and applied
    at L2's own, higher cell densities and potentials; a Boltzmann-electron cell with a fixed G_k has no
    sheath-limited (saturation) cusp current, so the anode-side cell (cell 0) draws whatever electron current the
    358 V potential (vs 309 V in the PIC) pulls through the cusps, roughly doubling I_d, and the ions born from the
    surplus ionisation leave through the anode (anode ion fraction 0.155 vs 0.014) instead of the walls. Leak
    half-widths w_k (populated flux tubes) are the second suspect: too-wide tubes at cusp 2 flip the near-wall drop.
* The `assess` stage gained the read-only `--pic-v4-results` option after the preregistration commit (the PIC
  33 um refinement `pic2d_cft_steady_state_v4` reached its plateau at 7.28 us / 5.2 M steps / 18,013 s while
  the L2 cases were running); it adds an INFORMATIONAL column and changes nothing in the model, the protocol
  or the gate evaluation.

## Post-hoc audit note (2026-09-05 - merge of the PARKED record into main)

* `origin/feat/hybrid-l2-v2` @ `277fc911` (merge-base `5da74ee6`) was merged into `feat/sota-foundation` / `main`
  with a merge commit that preserves the six branch commits. The verdict (`not_evaluable`, comparison FAIL 24/28),
  `protocol.json`, `preflight.json`, `shakedown.json`, every file under `results*/` and every `.sha256.json` sidecar
  are byte-identical to the branch. `cft_revival.hybrid`, `run.py`, `closure.py` and the 96 hybrid tests are unchanged
  and pass against main's tree (CPU-only). No hybrid test asserts a sealed digest against the live tree, so the
  `frozen_contract` binding used by other experiments was not needed here.
* One post-hoc change, in the visualization only: `modern/visualization/generate_hybrid_l2_v2_dashboard.py` read
  `series.npz` of every finished case, but the record tracks `series.npz` for the base case alone (the four other
  finished cases are recorded by their `summary.json`), so `hybrid-l2-v2.html` could not be regenerated from a clean
  checkout and its four tests errored (`FileNotFoundError: results-seed-b/series.npz`). The generator now embeds
  the base case's series only and reads nothing outside the tracked record; `hybrid-l2-v2.html` was regenerated.
  The embedded payload differs from the branch's dashboard ONLY in `cases.{seed-b, spatial-coarse, spatial-fine,
  temporal-coarse}.series` (600 decimated points each, dropped); verdict, identity hashes, comparison table, GATE-L2
  metrics, level spreads, cost, the base series and both map blocks are identical. The four cases' summaries (stop
  reason, steps, window currents, cells, plateau, energy residual) stay embedded. No recorded value changed.
