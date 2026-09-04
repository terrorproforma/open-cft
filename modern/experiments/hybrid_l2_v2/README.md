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
* The `assess` stage gained the read-only `--pic-v4-results` option after the preregistration commit (the PIC
  33 um refinement `pic2d_cft_steady_state_v4` reached its plateau at 7.28 us / 5.2 M steps / 18,013 s while
  the L2 cases were running); it adds an INFORMATIONAL column and changes nothing in the model, the protocol
  or the gate evaluation.
