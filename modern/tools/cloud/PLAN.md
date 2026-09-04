# Lambda 8x H100 (SXM, 80 GB) - PIC-MCC campaign plan

Box: Ubuntu 22.04 "Lambda Stack" image (NVIDIA driver + CUDA preinstalled), ~200 vCPU, ~1.8 TB RAM,
persistent filesystem under `/home/ubuntu/<fs-name>` (or `/lambda/nfs/<fs-name>`), user `ubuntu`, key
`~/.ssh/lambda_h100`. **$24/h for the whole box**, so the bill is the *makespan* (the longest GPU's
busy time), not the sum of GPU-hours: the plan packs the longest job first and releases the instance
the moment the last result is committed and pushed.

Everything below is at repository state `392129e5` (2026-09-04 ~13:00 AEST: mini-sweep DRAFT
`8704bf7c..6440518d`, v2.0.3 gates `ceb9b172`, the preregistered 33 um refinement steady-state v4
`392129e5`) or later on `feat/sota-foundation`. The three
tools live next to this file: `bootstrap_lambda.sh` (provision), `bench.sh` /
`bench_gpu_concurrency.py` (how many PIC processes per GPU), `schedule.py` + `jobs.yaml` (launch,
provenance, status). Nothing in them edits an experiment protocol or the `cft_revival` package.

## 1. Environment resolution

| item | resolution |
| --- | --- |
| Python | 3.12 (uv-managed; the local anchor is 3.12.10). The repo has **no lock file** (`modern/pyproject.toml` declares ranges only), so the pins live in `modern/tools/cloud/requirements-pic.txt`, mirroring the local main environment that produced every recorded ms/step: `warp-lang==1.14.0`, `numpy==2.5.2`, `pytest==9.1.1` (+ `pyyaml==6.0.3` for the scheduler). The ML stack (`.venv-sota`: torch 2.13.0+cu130, BoTorch 0.18.1) is **not** needed for PIC and is not installed. |
| Warp / CUDA | `warp-lang 1.14.0` PyPI wheel (`manylinux_2_28`) is a **CUDA Toolkit 12.9 build**: it statically embeds NVRTC and links only `libcuda.so.1`, so the box needs **no CUDA toolkit**, only a driver >= 525 (CUDA 12.x minimum). Lambda Stack ships a far newer driver (the local 5090 runs 595.97 = CUDA 13.2 with the same wheel). The `+cu13` GitHub wheel (driver >= 580) is deliberately not used so the build configuration matches the local anchors; `WARP_WHEEL_URL` in the bootstrap overrides it if wanted. Warp JIT-compiles the pic2d module for `sm_90` on first use (cache `~/.cache/warp/1.14.0`); the bootstrap's pytest run warms it before any fan-out. |
| native C++ | `cft_revival._native` (pybind11) is optional; `tests/test_kernels.py` skips without it and pic2d never imports it. `build-essential`/`cmake` are installed anyway (cheap). |
| LFS | the PIC field inputs are LFS objects (`divergent-exit-stack.level-1.json.arrays.npz` 8.8 MB, `domain-padding-1.5` 6.9 MB, the mini-sweep's four material-aware fields under `pic2d_design_mini_sweep_v1/fields/` ~73 MB); the bootstrap pulls those sets first and fails closed if the level-1 file is still a pointer, then (default) the whole ~1 GB. |
| GPU pinning | every process gets `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=<nvidia-smi index>`; the scheduler's wrapper asks Warp inside the job's environment which UUID `cuda:0` is and refuses to start on a mismatch. Reason: the runner records `backend: warp-cuda:0` (always `cuda:0` under a pin) and its `nvidia-smi` utilisation sampler reads the box's *first* GPU, so the runner alone cannot tell you which H100 it ran on; `jobs/<id>/state.json` does. |
| BLAS | the host block-Thomas factorisation is `np.linalg.inv` per radial-row block (OpenBLAS); the scheduler sets `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS = floor(CPUs / total slots)` (cap 16) per job, the benchmark per process - two unpinned factorisations oversubscribed the local box (20 min without finishing). ~200 vCPU / 8 jobs = 16 threads each. |

## 2. Anchors (RTX 5090, Warp 1.14.0 CUDA 12.9 build) the benchmark compares against

| configuration | grid | seed-load ms/step | production-load ms/step | source |
| --- | --- | --- | --- | --- |
| channel-only 50 um (accepted plateau, steady-state v2 base, model v1.3) | 60 x 480 | 2.26 (steps 400-2400, ~0.26 M particles) | **1.98** (session mean, 5.12 M steps, ~1.0 M e-) | `results/status.jsonl`, `summary.json` |
| **channel-only 33 um / 1.4 ps = steady-state v4 (preregistered `392129e5`)** | 90 x 720, W 2.667e4 | **2.54** (2000 steps after 200 warm-up, 1.3 M seed particles) | **4.36** at the 4.5 M synthetic plateau load (0.565 ms per M) -> 6.2 h to 3 transits, 6.6 h to the v2 verdict time | `steady_state_v4/preflight.json` (`_time_steps`, the function the benchmark reuses) |
| plume v2.0 50 um (attempts 7-8) | 240 x 720 | 4.26 (steps 400-2400) | **7.0-7.15** at ~4.4 M particles | plume v1 README launch log |
| plume v2.1 50 um (48 x 12 mm) | 240 x 960 | - | 8.2 (predicted) | v2.1 README cost table |
| channel-only 25 um | 120 x 960 | - | 17.3 (predicted by the plume-README model; ~7.4 by the v4-calibrated one) | plume v1 README resolution decision |
| plume v2.1 33 um / 1.4 ps | 360 x 1440 | - | 22.4 (predicted) | same |

The v4 preflight measured what the plume-v1 README had projected at 9.8 ms/step: the refined channel
step costs 4.36 ms at the plateau load, so the 33 um refinement is a 6-7 h run, not 13-14 h. The
25 um and plume-33 um predictions come from the same over-conservative model and may shrink by a
similar factor; the benchmark measures both grids on day 1 (`channel-25um` is in the registry;
`plume-v2.1-33um` too, off by default because of its ~1 h factorisation).

`bench.sh` runs N = 1, 2, 4 processes on one H100 for `channel-50um`, `channel-33um` (= the v4
protocol) and `plume-v2.0-50um` at the protocol seed load (the launch-bound floor) and at a re-seeded
*production* load (1.75e17 m^-3 = the v4 preflight's plateau load, so the 1.98 / 4.36 / 7.08 anchors
are like-for-like), 2000 timed steps measured by `pic2d_cft_steady_state_v4.run._time_steps` (200
warm-up steps, window accumulation on - the function behind the 5090 anchors) after a 400-step
pre-warm, processes synchronised on a file barrier, nvidia-smi memory per process, Warp mempool
high-water mark, setup (factorisation + upload) seconds. It writes
`gpu-concurrency-*.json/.md` and `host-factorisation-*.json/.md` (CPU-only, N = 1, 4, 8 concurrent
factorisations with pinned BLAS threads) under `$WORK/bench`. Decision rule for `jobs.yaml`:
`slots_per_gpu = 2` only if the N = 2 aggregate speed-up is >= 1.3 and every process still fits the
memory budget; otherwise 1. Note that concurrency never shortens a *single* run - only the
per-process speed-up does - so it does not move the critical path below; it matters for packing
short jobs and for sizing a smaller instance later.

## 3. Job map (GPU-hours at the 5090 rate; H100 factor `s` from the benchmark, assumed 1.0 here)

Costs: steps to 3 transits = 3 x transit / dt; hours = steps x ms/step / 3.6e6; the accepted 50 um
plateau was declared at 3.2 transits (7.68 us), so +7 % is realistic. Memory from the v2.1 cost
table and the v4 preflight. Macro-weight policy: the v4 prereg keeps *particles per cell* (W / 2.25
at 33 um; 4.5 M at the plateau) and its preflight MEASURED 4.36 ms/step there, so (i) is costed on
that measurement; (ii) at W / 4 (8 M particles) is bracketed by the v4-calibrated model (~7.4 ms)
and the plume-README model (17.3 ms); (iii) uses the mini-sweep's own composer numbers (equal W).

| # | job (`jobs.yaml` id) | configuration | ms/step | steps | hours | GPU GB | status of the protocol |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (i) | `ss33-seed-b` | channel-only 90 x 720 = 33 um, dt 1.4 ps, W 2.667e4, seed b (v4 configuration) | 4.36 (measured) | 5.14-5.49 M | **6.2-6.6** | ~4 | v4 is preregistered at `392129e5` for ONE base execution (local 5090); its `launch` has no `--case`: the seed-b / W x 0.7 replications NEED their own prereg (v4 variants or a v4.1 sibling) |
| (i) | `ss33-w-0.7` | same, W x 0.7 (~6.4 M particles) | ~5.4 | 5.14-5.49 M | **7.7-8.2** | ~5 | same replication prereg |
| (ii) | `ss25-base` | channel-only 120 x 960 = 25 um, dt <= 1.0 ps (omega_pe dt 0.07 at the peak), W 1.5e4 -> particles x (50/25)^2 = 8 M | 7.4 (v4-calibrated) to 17.3 (plume-README) | 7.2-7.7 M (3 x 2.4 us / 1.0 ps) | **15-16 to 35-37** | ~13 | NEEDS its own protocol (a 25 um sibling of v4: dt / cells / W are not `--case` keys) + prereg for a 50/33/25 ladder claim |
| (iii) | `sweep-reference`, `sweep-056`, `sweep-047`, `sweep-009` | 4 primary designs x channel-only 33 um / 1.4 ps (`run --design <id> --domain channel --grid 33um`) | 2.9 / 4.8 / 2.0 / 3.4 | 5.14 / 3.65 / 5.55 / 4.88 M | **4.2 + 4.9 + 3.1 + 4.7 = 16.9** (x3 at fixed ppc) | 5-9 each | DRAFT **at HEAD** (`8704bf7c..6440518d`: composer, four P2 fields (LFS), whole-set preflight green, 13 tests) but NOT preregistered - `run` refuses without `--allow-launch`; NEEDS the prereg commit after README s.7's open decisions (v2.0.3 gates, 50 vs 33 um). Optional 5th design 106: +7.6 h; 056 seed replicate: +4.9 h |
| (iv) | `plume-v2.1-33um` | 48 x 12 mm box, 360 x 1440 = 33 um, dt 1.4 ps; ~61 min factorisation, 6.0 GB inverse blocks | 22.4 | 8.16 M (3 x 3.81 us / 1.4 ps) | **51-52** | ~16.6 | NEEDS a v2.1 protocol revision (cells, dt, W 2.67e4, v2.0.3 gates, budget 200 000 s); development run |
| (iv) | `plume-v2.1-50um-3mA` | 48 x 12 mm at 50 um, cathode `max_current_a` 3e-3 so peak n_e <= 1.4e18 | 8.2 | 7.62 M | **17-20** | ~8.8 | NEEDS its own protocol file (separate experiment dir: the frozen-protocol check binds one path to one blob); development run |
| - | `shakedown-ss-v3-graph` | channel v1.4 (graph step), 4000 steps | 2 | 4 k | 0.25 | 6 | protocol frozen since `112bb250`, runs from `ac248e05`; **launchable now**, non-evidentiary |

Totals: **~130 GPU-h** at the 5090 rate for the 9 production jobs with the 25 um upper bound
(~110 GPU-h with its v4-calibrated cost). Per-GPU packing (longest first; the four sweep designs
are short enough to run serially on one GPU):

```
GPU0  plume-v2.1-33um ........................................ 52 h   <- critical path
GPU1  ss25-base ........................ 16-37 h
GPU2  plume-v2.1-50um-3mA ........ 20 h
GPU3  ss33-seed-b 6.5 h -> ss33-w-0.7 8 h -> sweep-056 seed replicate (optional) 5 h  = 20 h
GPU4  sweep-reference 4.2 -> sweep-056 4.9 -> sweep-047 3.1 -> sweep-009 4.7  = 17 h
GPU5  (bench for the first hour; then sweep-106 optional 7.6 h, or a 2nd slot)
GPU6  (shakedown 15 min; then free - 25 um seed replicate if the ladder needs one)
GPU7  free                       (spare for a resume / re-run without waiting)
```

| option | makespan at s = 1.0 | cost at $24/h | utilisation | comment |
| --- | --- | --- | --- | --- |
| A - full queue | **52 h** | **$1,250** | 110-130 / 416 GPU-h = 26-31 % | the single 33 um plume run sets the bill; five GPUs idle most of the time |
| B - without `plume-v2.1-33um` | 20-37 h | $480-890 | 60-80 / 160-296 | the 25 um ladder point is the critical path if it costs the README's 17.3 ms/step; otherwise the 3 mA plume (20 h) is |
| C - (i) + (iii) + `plume-v2.1-50um-3mA` only | 20 h | $480 | 51 / 160 = 32 % | everything preregistered-or-cheap; both long development runs deferred; the 3 mA plume alone sets the makespan |
| D - (i) + (iii) only (nothing that needs a new protocol beyond the replication / sweep preregs) | 17 h | $410 | 31 / 136 | a 2x H100 instance would do the same work |

The low utilisation is the honest picture: this queue has one or two long single-process jobs and a
handful of short ones, so an 8-GPU box is bought for the critical path, not for the width. If option
A or B is chosen, consider releasing this box after the short jobs finish and moving the one long job
to a 1x H100 instance (~$3/h) - the scheduler's `jobs/<id>/tree` worktree and the runner's checkpoint
make the resume a `git clone` + `launch --force` (v4-style: `launch --resume`) away.

With an H100 per-process speed-up `s`, divide makespan and cost by `s` (the step is launch-bound at
~480 sequential block-Thomas launches per plume step and bandwidth-bound in the inverse-block reads;
H100 HBM3 is 1.9x the 5090's bandwidth, launch latency is not better, so expect s ~ 1.0-1.5 - the
benchmark's `per-process x vs 5090` column is the number to plug in). Every job is resumable: a
budget stop is a `run_state.json` with `finished: true` and a checkpoint, and `schedule.py launch
--force` continues it.

### Fill order and what has to land first

1. **T+0 (IP arrives)**: bootstrap (~20-30 min: apt, uv, clone + LFS, venv, Warp smoke, `pytest
   tests/pic2d -x -q` on GPU 0 - 193 tests took 204 s on the 5090). Then `bench.sh` on GPU 0
   (~45-60 min: the plume factorisations dominate) and the `shakedown-ss-v3-graph` job on GPU 1
   (15 min) through the scheduler - the shakedown exercises the whole provenance chain
   (worktree, prereg check, UUID probe, wrapper, `status`) before any real job.
2. **First production launches** = whatever is preregistered/pushed by then, longest first on the
   free GPUs. Expected order of the commits: (a) the replication prereg for the 33 um refinement
   (v4 `392129e5` covers ONE base execution, running locally; seed-b / W x 0.7 need v4 variants or a
   v4.1 sibling with `--case`) -> `ss33-seed-b`, `ss33-w-0.7`; (b) mini-sweep prereg (draft at HEAD, preflight
   green; the open decisions are the v2.0.3 gates and the grid) -> `sweep-reference`, `sweep-056`,
   `sweep-047`, `sweep-009`; (c) v2.1 33 um protocol revision -> `plume-v2.1-33um` (start it as
   early as possible: it is the critical path; its 61 min factorisation runs alone on the CPUs
   first if possible); (d) 25 um protocol -> `ss25-base`; (e) current-limited plume protocol ->
   `plume-v2.1-50um-3mA` after `ss33-seed-b` frees GPU 2 (or as a second slot if the benchmark
   allows). Each landing = edit `jobs.yaml` locally (module, `commit`, `enabled: true`), commit,
   push, `git pull --ff-only` on the box, `schedule.py launch`.
3. **Preregistration discipline on the box**: the scheduler refuses a job whose `commit` is not an
   ancestor of HEAD or whose protocol differs from that commit; it records `preregistered` verbatim
   (development runs stay development runs); it runs each job in a **detached worktree at the
   commit** (`<WORK>/jobs/<id>/tree`), so `_bind_preregistration`-style checks (detached HEAD ==
   prereg commit, clean tree) hold and no results directory is ever shared (a runner *resumes* from
   any checkpoint it finds in its results dir). Lock semantics to respect: (1) the v4 runner's
   `launch` writes an immutable `O_EXCL` `execution-lock.json` into its results dir (commit, protocol
   hash, config hash, host, PID, UTC; same-attempt / different-attempt classification on collision) and
   `launch --resume` continues only under the same commit + protocol - per results dir, so v4-style
   jobs in separate worktrees never collide, and a `--force` relaunch must switch the args to
   `--resume`; (2) the `experiment_runtime` execution lock (immutable `O_EXCL` lock per experiment
   result root) and the `git_common_lock` file some preregistered runners create in the git common
   dir are **per clone**: two executions of the *same* experiment_runtime experiment cannot share one
   clone even from different worktrees. The v1 PIC runner uses neither, so PIC cases (`--case`)
   coexist; the mini-sweep composer also runs on the PIC runner (`run_steady_state` per design,
   results under `results/<design>-channel-33um/`), so its four designs can run concurrently on
   separate GPUs or serially on one - both from their own worktrees.
4. **Release criterion**: `schedule.py status` shows no `running` job and every production job has
   `finished: true` with a terminal `stop_reason` (`plateau_reached_after_min_transit_times`,
   `wall_clock_budget_reached`, or a gate stop); every result is committed from its job worktree
   onto a `results/<id>` branch and pushed, LFS objects included (`git lfs push origin
   results/<id>` if any tracked output is LFS), verified with `git ls-remote origin results/<id>`;
   `$WORK/provision.log`, the bench reports and every `jobs/<id>/state.json` are copied into the
   repo (e.g. `modern/tools/cloud/records/<date>/`) and pushed. Only then terminate the instance;
   the persistent filesystem keeps `$WORK` (checkout, venv, worktrees) for a re-launch.

## 4. Operator runbook (exact commands, in order)

Locally, before anything:

```powershell
git push origin feat/sota-foundation           # the box clones this branch (tools + any prereg)
scp -i ~/.ssh/lambda_h100 modern/tools/cloud/bootstrap_lambda.sh ubuntu@<IP>:~/
ssh -i ~/.ssh/lambda_h100 ubuntu@<IP>
```

On the box (first login; verify the image assumptions in section 5 as you go):

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv    # 8 x H100 80GB, driver >= 525
df -h /home/ubuntu /lambda/nfs 2>/dev/null                                   # find the persistent filesystem
export WORK=/home/ubuntu/<fs-name>/cft && mkdir -p "$WORK"
export REPO_URL=https://github.com/terrorproforma/open-cft.git REPO_REF=feat/sota-foundation
export GIT_TOKEN=<fine-grained token: contents read (write to push results)>   # or: export DEPLOY_KEY=~/.ssh/<key> REPO_URL=git@github.com:terrorproforma/open-cft.git
bash ~/bootstrap_lambda.sh 2>&1 | tee -a "$WORK/bootstrap.out"     # idempotent; writes $WORK/provision.log
tail -n 40 "$WORK/provision.log"
```

Benchmark (GPU 0) and shakedown (auto -> a free GPU) in parallel:

```bash
tmux new -d -s bench "WORK=$WORK BENCH_GPU=0 bash $WORK/uni-project/modern/tools/cloud/bench.sh > $WORK/bench.out 2>&1"
cd "$WORK/uni-project/modern"
PY="$WORK/uni-project/.venv-pic/bin/python"
$PY tools/cloud/schedule.py gpus
$PY tools/cloud/schedule.py plan                          # every enabled job: prereg check + GPU assignment, no launch
$PY tools/cloud/schedule.py launch --only shakedown-ss-v3-graph
$PY tools/cloud/schedule.py status                        # steps, t, transits, ms/step, ETA, stop reason, prereg flag
cat "$WORK/jobs/shakedown-ss-v3-graph/state.json"         # pid, gpu {index,name,uuid}, warp_probe, gpu_uuid_match
tail -f "$WORK/bench.out"                                 # ~45-60 min; then read $WORK/bench/*.md
```

Set `slots_per_gpu` from the benchmark (edit `jobs.yaml` locally, commit, push), then per prereg
landing:

```bash
# locally: fill module/commit, enabled: true in modern/tools/cloud/jobs.yaml; commit; push
git -C "$WORK/uni-project" pull --ff-only origin feat/sota-foundation
$PY tools/cloud/schedule.py plan                          # must print no REFUSED line
$PY tools/cloud/schedule.py launch                        # launches every enabled, not-yet-launched job
$PY tools/cloud/schedule.py status --watch 600            # or: tmux attach -t pic-<id> for a job's console
```

Stop / resume a job (the runner checkpoints every 40 000 steps; a relaunch resumes):

```bash
$PY tools/cloud/schedule.py stop <id>                     # SIGTERM to the runner
$PY tools/cloud/schedule.py launch --only <id> --force    # v1-runner jobs (`run`): resumes from the last checkpoint
# v4-style jobs (`launch --expect-commit ...`): add `--resume` to the job's args in jobs.yaml first -
# the second session must run under the existing execution-lock.json (same commit + protocol)
```

For a preregistered v4-style job the verdict step runs from the job worktree after the stop:

```bash
cd "$WORK/jobs/<id>/tree/modern" && $PY -m experiments.pic2d_cft_steady_state_v4.run assess   # -> assessment.json
```

When a job reaches a terminal `stop_reason`, record it from its worktree (the established
`record ...` / `chore(pic2d)` results-only commits), then push:

```bash
cd "$WORK/jobs/<id>/tree"
git checkout -b results/<id>
git add -f modern/experiments/<experiment>/results-<case>/{summary.json,run_state.json,status.jsonl,maps.npz,series.npz,checkpoint-final.json}*   # + README launch-log paragraph
git -c user.name="<you>" -c user.email="<you>" commit -m "record <experiment> <case> (Lambda H100 GPU <k>, job <id>)"
git push origin results/<id> && git lfs push origin results/<id>
git ls-remote origin results/<id>
```

Release (after the last push):

```bash
$PY tools/cloud/schedule.py status --no-gpu               # no job 'running'
mkdir -p modern/tools/cloud/records/$(date -u +%Y%m%d) && cp "$WORK"/provision.log "$WORK"/bench/*.{json,md} "$WORK"/jobs/*/state.json modern/tools/cloud/records/$(date -u +%Y%m%d)/ 2>/dev/null
# commit + push the records, then terminate the instance in the Lambda console; $WORK survives on the persistent filesystem
```

## 5. Assumptions about the Lambda image to verify on first login

1. `nvidia-smi` reports exactly 8 `NVIDIA H100 80GB HBM3` with MIG disabled and a driver >= 525
   (the bootstrap fails closed on both). Expected: R570-R580 class driver (CUDA 12.8-13.0); the
   local anchor ran 595.97.
2. Ubuntu 22.04 = glibc 2.35 >= the wheel's `manylinux_2_28` floor; `sudo -n` is passwordless for
   `ubuntu`; outbound HTTPS to GitHub, PyPI and astral.sh works.
3. The persistent filesystem is mounted at `/home/ubuntu/<fs-name>` (check `df -h`); the root disk
   is ephemeral. `$WORK` must be on the persistent mount.
4. System Python is 3.10; the venv uses a uv-managed 3.12 (matches the local anchors; the code
   requires >= 3.10).
5. `tmux`, `git-lfs`, `jq` may be missing from the image - the bootstrap installs them
   (`SKIP_APT=1` if apt is unavailable and they are present).
6. ~200 vCPU: 8 jobs x 16 BLAS threads = 128 cores during the factorisations. If `nproc` is much
   smaller, the scheduler's floor(CPUs / slots) still prevents oversubscription.
7. Warp's `cuda:0` UUID under `CUDA_DEVICE_ORDER=PCI_BUS_ID` equals nvidia-smi's index UUID - the
   wrapper checks it; a mismatch means the ordering assumption is wrong and every job is refused
   until `CUDA_VISIBLE_DEVICES` is set by UUID instead (`gpu` in `state.json` records both).
8. GPU persistence mode is optional (`sudo nvidia-smi -pm 1` shaves context creation); ECC on is
   the default and fine.
9. The runner's `gpu_utilisation_percent_samples` in every `summary.json` produced on this box are
   **box-wide first-GPU readings** (the sampler has no `--id`); use `jobs/<id>/state.json` and
   `schedule.py status` (`gpu_live`, per-index) for the pinned GPU. A `--gpu-index` for the sampler
   is a follow-up change to the runner, not made here (package/runner identity untouched).
