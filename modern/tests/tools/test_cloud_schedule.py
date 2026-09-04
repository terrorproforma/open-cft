"""Unit tests for ``tools.cloud.schedule`` (job state, GPU assignment, preregistration-ancestor check).

The git checks run against a throw-away repository built in ``tmp_path``; the launch round trip uses
the ``exec`` launcher with the Warp probe off, so no GPU, tmux or nvidia-smi is needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.cloud import schedule

REPOSITORY = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------------------------------------ fixtures
def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=str(repo), capture_output=True, text=True, check=True,
    )
    return completed.stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


DUMMY_RUNNER = '''"""Dummy experiment runner for the scheduler tests: writes runner-style artifacts and exits."""
import json, os, sys
from pathlib import Path

results = Path(__file__).resolve().parent / "results"
results.mkdir(exist_ok=True)
print("dummy run", sys.argv[1:], "gpu", os.environ.get("CUDA_VISIBLE_DEVICES"), "order", os.environ.get("CUDA_DEVICE_ORDER"),
      "omp", os.environ.get("OMP_NUM_THREADS"), flush=True)
with open(results / "status.jsonl", "w", encoding="utf-8") as f:
    for k in range(1, 6):
        f.write(json.dumps({"step": 200 * k, "time_s": 200 * k * 1.5e-12, "ms_per_step": 2.0 + 0.1 * k, "electrons": 1000 * k,
                            "wall_seconds_total": 0.4 * k}) + "\\n")
(results / "run_state.json").write_text(json.dumps({"checkpoint_step": 1000, "checkpoint_time_s": 1.5e-9,
    "wall_seconds_total": 2.0, "finished": True, "stop_reason": "target_steps_reached", "sessions": []}), encoding="utf-8")
sys.exit(int(os.environ.get("DUMMY_EXIT", "0")))
'''

PROTOCOL = {"numerics": {"dt_s": 1.5e-12}, "budget_v1_4": {"ion_transit_time_s": 2.4e-6},
            "stopping_rule": {"wall_budget_seconds": 3600}}


@pytest.fixture
def repo(tmp_path: Path) -> dict[str, object]:
    """A repository with a dummy experiment: commit A (protocol frozen), commit B (unrelated), branch with D."""

    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _write(root / "modern" / "experiments" / "__init__.py", "")
    _write(root / "modern" / "experiments" / "dummy" / "__init__.py", "")
    _write(root / "modern" / "experiments" / "dummy" / "run.py", DUMMY_RUNNER)
    _write(root / "modern" / "experiments" / "dummy" / "protocol.json", json.dumps(PROTOCOL, indent=1) + "\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "preregister dummy")
    commit_a = _git(root, "rev-parse", "HEAD")
    _write(root / "README.md", "later\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "unrelated")
    commit_b = _git(root, "rev-parse", "HEAD")
    # an unreachable commit on a side branch
    _git(root, "checkout", "-q", "-b", "side", commit_a)
    _write(root / "side.txt", "side\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "side")
    commit_d = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", "main")
    return {"root": root, "a": commit_a, "b": commit_b, "d": commit_d}


def _plan(root: Path, jobs_dir: Path, **overrides) -> schedule.Plan:
    raw = {
        "version": 1,
        "defaults": {"repo": str(root), "python": sys.executable, "cwd": "modern", "jobs_dir": str(jobs_dir),
                     "launcher": "exec", "slots_per_gpu": 1, "gpu_memory_gib": 80, "warp_probe": "off"},
        "gpus": [0, 1, 2, 3],
        "jobs": overrides.pop("jobs", []),
    }
    raw["defaults"].update(overrides)
    return schedule.build_plan(raw, base=root)


# ------------------------------------------------------------------------------------------ specification
def test_build_plan_resolves_defaults_and_rejects_unknown_job_keys(tmp_path: Path) -> None:
    plan = _plan(tmp_path, tmp_path / "jobs", jobs=[{"id": "a", "module": "experiments.dummy.run", "gpu": 2}])
    assert plan.gpus == [0, 1, 2, 3]
    assert plan.defaults.launcher == "exec"
    assert plan.jobs[0].gpu == 2 and plan.jobs[0].checkout == "shared" and plan.jobs[0].preregistered is False
    with pytest.raises(schedule.ScheduleError, match="unknown keys"):
        _plan(tmp_path, tmp_path / "jobs", jobs=[{"id": "a", "module": "m", "gpus": 1}])
    with pytest.raises(schedule.ScheduleError, match="duplicate"):
        _plan(tmp_path, tmp_path / "jobs", jobs=[{"id": "a", "module": "m"}, {"id": "a", "module": "m"}])
    with pytest.raises(schedule.ScheduleError, match="version"):
        schedule.build_plan({"version": 2, "jobs": []}, base=tmp_path)
    with pytest.raises(schedule.ScheduleError, match="launcher"):
        _plan(tmp_path, tmp_path / "jobs", launcher="cron")


# ------------------------------------------------------------------------------------------ GPU assignment
def test_assign_gpus_explicit_then_least_loaded_auto(tmp_path: Path) -> None:
    jobs = [{"id": "x", "module": "m", "gpu": 1}, {"id": "y", "module": "m"}, {"id": "z", "module": "m"},
            {"id": "w", "module": "m", "gpu": 3}]
    plan = _plan(tmp_path, tmp_path / "jobs", jobs=jobs)
    assignment = schedule.assign_gpus(plan, plan.jobs)
    assert assignment == {"x": 1, "w": 3, "y": 0, "z": 2}


def test_assign_gpus_respects_slots_busy_and_memory(tmp_path: Path) -> None:
    jobs = [{"id": f"j{i}", "module": "m"} for i in range(4)]
    plan = _plan(tmp_path, tmp_path / "jobs", jobs=jobs, slots_per_gpu=1)
    # GPUs 0 and 1 carry running jobs -> only two slots left for four jobs
    with pytest.raises(schedule.ScheduleError, match="no free GPU slot"):
        schedule.assign_gpus(plan, plan.jobs, busy={0: ["r0"], 1: ["r1"]})
    plan2 = _plan(tmp_path, tmp_path / "jobs", jobs=jobs, slots_per_gpu=2)
    busy = {0: ["r0"], 1: ["r1"]}
    assignment = schedule.assign_gpus(plan2, plan2.jobs, busy=busy)
    # least-loaded first: the two empty GPUs get a job each, then the half-full ones fill to their 2 slots
    assert assignment["j0"] == 2 and assignment["j1"] == 3 and {assignment["j2"], assignment["j3"]} == {0, 1}
    load = {g: len(busy.get(g, [])) + sum(1 for v in assignment.values() if v == g) for g in plan2.gpus}
    assert load == {0: 2, 1: 2, 2: 1, 3: 1}
    extra = [schedule.JobSpec(id=f"e{i}", module="m") for i in range(3)]      # 6 + 3 > 8 slots
    with pytest.raises(schedule.ScheduleError, match="no free GPU slot"):
        schedule.assign_gpus(plan2, plan2.jobs + extra, busy=busy)
    # explicit GPU without a free slot
    plan3 = _plan(tmp_path, tmp_path / "jobs", jobs=[{"id": "e", "module": "m", "gpu": 0}])
    with pytest.raises(schedule.ScheduleError, match="no free slot"):
        schedule.assign_gpus(plan3, plan3.jobs, busy={0: ["r0"]})
    with pytest.raises(schedule.ScheduleError, match="not in the plan"):
        schedule.assign_gpus(plan3, [schedule.JobSpec(id="q", module="m", gpu=7)])
    # declared memory above 90 % of the GPU
    heavy = [{"id": "h1", "module": "m", "gpu": 0, "gpu_memory_gib": 40}, {"id": "h2", "module": "m", "gpu": 0, "gpu_memory_gib": 40}]
    plan4 = _plan(tmp_path, tmp_path / "jobs", jobs=heavy, slots_per_gpu=2)
    with pytest.raises(schedule.ScheduleError, match="exceeds 90 %"):
        schedule.assign_gpus(plan4, plan4.jobs)


def test_blas_threads_for_never_oversubscribes() -> None:
    plan = schedule.Plan(defaults=schedule.Defaults(repo=Path("."), python="py"), gpus=list(range(8)), jobs=[])
    assert schedule.blas_threads_for(plan, 8, cpu_count=200) == 16      # capped
    assert schedule.blas_threads_for(plan, 16, cpu_count=200) == 12
    assert schedule.blas_threads_for(plan, 8, cpu_count=8) == 1
    assert schedule.blas_threads_for(plan, 64, cpu_count=8) == 1        # never zero


# ------------------------------------------------------------------------------------------ preregistration
def test_prereg_check_accepts_a_frozen_ancestor_commit(repo: dict[str, object]) -> None:
    root = repo["root"]
    record = schedule.prereg_check(root, str(repo["a"]), "modern/experiments/dummy/protocol.json", require_clean=True)
    assert record["ok"] is True and record["problems"] == []
    assert record["is_ancestor_of_head"] is True and record["protocol_frozen"] is True
    assert record["commit"] == repo["a"] and record["head"] == repo["b"]
    # abbreviated refs resolve to the full SHA
    short = schedule.prereg_check(root, str(repo["a"])[:10], None)
    assert short["commit"] == repo["a"]


def test_prereg_check_refuses_unreachable_commits(repo: dict[str, object]) -> None:
    record = schedule.prereg_check(repo["root"], str(repo["d"]), "modern/experiments/dummy/protocol.json")
    assert record["ok"] is False
    assert any("not reachable from HEAD" in p for p in record["problems"])
    with pytest.raises(schedule.ScheduleError, match="not in the repository"):
        schedule.prereg_check(repo["root"], "0" * 40, None)


def test_prereg_check_refuses_a_protocol_changed_after_the_commit(repo: dict[str, object]) -> None:
    root = repo["root"]
    protocol = root / "modern" / "experiments" / "dummy" / "protocol.json"
    # uncommitted edit -> dirty path
    _write(protocol, json.dumps({**PROTOCOL, "numerics": {"dt_s": 1.0e-12}}, indent=1) + "\n")
    dirty = schedule.prereg_check(root, str(repo["a"]), "modern/experiments/dummy/protocol.json")
    assert dirty["ok"] is False and any("modified" in p for p in dirty["problems"])
    # committed edit -> differs between the prereg commit and HEAD
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "edit protocol after prereg")
    changed = schedule.prereg_check(root, str(repo["a"]), "modern/experiments/dummy/protocol.json")
    assert changed["ok"] is False and any("differs between" in p for p in changed["problems"])
    # the new commit itself is fine
    fresh = schedule.prereg_check(root, "HEAD", "modern/experiments/dummy/protocol.json")
    assert fresh["ok"] is True


def test_launch_refuses_without_commit_or_with_unreachable_commit(repo: dict[str, object], tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(schedule, "gpu_inventory", lambda: [{"index": 0, "name": "Fake GPU", "uuid": "GPU-0"}])
    root = repo["root"]
    job = {"id": "nocommit", "module": "experiments.dummy.run", "args": ["run"], "protocol": "modern/experiments/dummy/protocol.json"}
    plan = _plan(root, tmp_path / "jobs", jobs=[job])
    with pytest.raises(schedule.ScheduleError, match="'commit'"):
        schedule.launch_job(plan, plan.jobs[0], 0, threads=2, dry_run=True)
    job["commit"] = str(repo["d"])
    plan = _plan(root, tmp_path / "jobs", jobs=[job])
    with pytest.raises(schedule.ScheduleError, match="refused"):
        schedule.launch_job(plan, plan.jobs[0], 0, threads=2, dry_run=True)
    assert not (tmp_path / "jobs").exists()     # nothing was written for a refused job


# ------------------------------------------------------------------------------------------ state files
def test_state_roundtrip_and_status_words(tmp_path: Path) -> None:
    directory = tmp_path / "jobs" / "j"
    assert schedule.read_state(directory) is None
    assert schedule.job_status_word(None) == "not launched"
    schedule.write_state(directory, {"id": "j", "pid": os.getpid(), "exit_code": None})
    assert schedule.read_state(directory)["pid"] == os.getpid()
    assert schedule.job_status_word(schedule.read_state(directory)) == "running"
    assert schedule.job_status_word({"pid": os.getpid(), "exit_code": 0}) == "finished"
    assert schedule.job_status_word({"pid": 1, "exit_code": 3}) == "failed (3)"
    assert schedule.job_status_word({"pid": None, "exit_code": None, "refused": "uuid"}) == "refused"
    assert (tmp_path / "jobs" / "j" / "state.json").read_bytes().endswith(b"}\n")


def test_occupied_slots_only_counts_running_jobs(tmp_path: Path) -> None:
    plan = _plan(tmp_path, tmp_path / "jobs")
    schedule.write_state(tmp_path / "jobs" / "run", {"id": "run", "pid": os.getpid(), "gpu": {"index": 2}})
    schedule.write_state(tmp_path / "jobs" / "done", {"id": "done", "pid": os.getpid(), "gpu": {"index": 1}, "exit_code": 0})
    busy = schedule.occupied_slots(plan)
    assert busy[2] == ["run"] and busy[1] == []


# ------------------------------------------------------------------------------------------ status / ETA
def test_job_progress_reads_runner_artifacts_and_computes_eta(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "run_state.json").write_text(json.dumps({"checkpoint_step": 400000, "checkpoint_time_s": 6.0e-7,
                                                        "wall_seconds_total": 3000.0, "finished": False}), encoding="utf-8")
    rows = [{"event": "resume", "step": 400000, "time_s": 6.0e-7}]
    rows += [{"step": 400000 + 200 * k, "time_s": (400000 + 200 * k) * 1.5e-12, "ms_per_step": 8.0 + (k % 3),
              "electrons": 1_000_000, "wall_seconds_total": 3000.0 + 2 * k} for k in range(1, 11)]
    (results / "status.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(PROTOCOL), encoding="utf-8")
    state = {"args": ["run", "--wall-budget-seconds", "50400"], "target_transits": 3.0}
    progress = schedule.job_progress(state, results_dir=results, protocol_path=protocol)
    assert progress["steps"] == 402000
    assert progress["ms_per_step"] == 9.0                      # median of 8, 9, 10 ...
    assert progress["transits"] == pytest.approx(402000 * 1.5e-12 / 2.4e-6)
    remaining_steps = (3 * 2.4e-6 - 402000 * 1.5e-12) / 1.5e-12
    assert progress["eta_target_s"] == pytest.approx(remaining_steps * 9.0 / 1e3, rel=1e-6)
    assert progress["wall_budget_seconds"] == 50400.0         # the CLI budget wins over the protocol's 3600
    assert progress["budget_remaining_s"] == pytest.approx(50400.0 - 3020.0)
    assert progress["finished"] is False and progress["stop_reason"] is None
    text = schedule.format_status([{"id": "j", "status": "running", "gpu": 3, "pid": 42, "preregistered": True,
                                    "progress": progress}])
    assert "402,000" in text and "9.00" in text and "yes" in text


def test_job_progress_without_records_uses_the_expected_ms_per_step(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(PROTOCOL), encoding="utf-8")
    progress = schedule.job_progress({"expected_ms_per_step": 9.8, "target_transits": 3}, results_dir=tmp_path / "none",
                                     protocol_path=protocol)
    assert progress["ms_per_step"] == 9.8 and progress["ms_per_step_source"]
    assert progress["eta_target_s"] == pytest.approx((3 * 2.4e-6 / 1.5e-12) * 9.8 / 1e3)
    assert "9.80*" in schedule.format_status([{"id": "j", "status": "not launched", "progress": progress}])


def test_helpers() -> None:
    assert schedule.wall_budget_from_args(["run", "--wall-budget-seconds", "100"]) == 100.0
    assert schedule.wall_budget_from_args(["run", "--wall-budget-seconds=200"]) == 200.0
    assert schedule.wall_budget_from_args(["run"]) is None
    assert schedule.eta_seconds(time_s=None, dt_s=1e-12, transit_s=1e-6, target_transits=3, ms_per_step=1.0) is None
    assert schedule.eta_seconds(time_s=3e-6, dt_s=1e-12, transit_s=1e-6, target_transits=3, ms_per_step=1.0) == 0.0
    assert schedule.uuid_matches("GPU-abc", "gpu-ABC") is True
    assert schedule.uuid_matches("GPU-abc", "GPU-def") is False
    assert schedule.uuid_matches(None, "GPU-def") is None


# ------------------------------------------------------------------------------------------ launch round trip
def test_launch_runs_the_wrapper_and_records_pid_gpu_and_exit_code(repo: dict[str, object], tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(schedule, "gpu_inventory", lambda: [
        {"index": i, "name": "Fake H100", "uuid": f"GPU-{i}", "driver_version": "0", "memory_total_mib": 81920.0, "pci_bus_id": f"0:{i}"}
        for i in range(4)])
    root = repo["root"]
    jobs_dir = tmp_path / "jobs"
    job = {"id": "dummy-a", "module": "experiments.dummy.run", "args": ["run", "--max-steps", "5"], "gpu": 2,
           "commit": str(repo["a"]), "protocol": "modern/experiments/dummy/protocol.json",
           "results": "modern/experiments/dummy/results", "preregistered": True, "env": {"DUMMY_EXIT": "0"}}
    plan = _plan(root, jobs_dir, jobs=[job])
    dry = schedule.launch_job(plan, plan.jobs[0], 2, threads=3, dry_run=True)
    assert dry["cuda_visible_devices"] == "2" and not jobs_dir.exists()
    state = schedule.launch_job(plan, plan.jobs[0], 2, threads=3)
    assert state["launcher"] == "exec" and state["gpu"]["index"] == 2 and state["gpu"]["name"] == "Fake H100"
    assert state["preregistered"] is True and state["prereg_check"]["ok"] is True
    directory = jobs_dir / "dummy-a"
    deadline = time.monotonic() + 60
    final = None
    while time.monotonic() < deadline:
        final = schedule.read_state(directory)
        if final and final.get("exit_code") is not None:
            break
        time.sleep(0.2)
    assert final is not None and final["exit_code"] == 0, (directory / "wrapper.log").read_text(encoding="utf-8")
    assert final["pid"] and final["start_utc"] and final["end_utc"]
    assert final["cuda_visible_devices"] == "2" and final["blas_threads"] == 3
    log = (directory / "run.log").read_text(encoding="utf-8")
    assert "gpu 2 order PCI_BUS_ID omp 3" in log and "--max-steps" in log
    # the job is not relaunched without --force, and status reads the runner artifacts it wrote
    with pytest.raises(schedule.ScheduleError, match="already has a state"):
        schedule.launch_job(plan, plan.jobs[0], 2, threads=3)
    rows = schedule.collect_status(plan, live_gpu=False)
    row = next(r for r in rows if r["id"] == "dummy-a")
    assert row["status"] == "finished" and row["progress"]["steps"] == 1000
    assert row["progress"]["stop_reason"] == "target_steps_reached" and row["progress"]["ms_per_step"] == 2.3
    assert "dummy-a" in schedule.format_status(rows)


def test_launch_records_a_nonzero_exit_code(repo: dict[str, object], tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(schedule, "gpu_inventory", list)     # nvidia-smi absent -> empty inventory
    root = repo["root"]
    job = {"id": "dummy-fail", "module": "experiments.dummy.run", "args": ["run"], "commit": str(repo["a"]),
           "env": {"DUMMY_EXIT": "7"}}
    plan = _plan(root, tmp_path / "jobs", jobs=[job])
    schedule.launch_job(plan, plan.jobs[0], 1, threads=1)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = schedule.read_state(tmp_path / "jobs" / "dummy-fail")
        if state and state.get("exit_code") is not None:
            break
        time.sleep(0.2)
    assert state["exit_code"] == 7 and schedule.job_status_word(state) == "failed (7)"
    assert state["gpu"] == {"index": 1, "name": None, "uuid": None}    # no nvidia-smi: recorded honestly


# ------------------------------------------------------------------------------------------ shipped jobs.yaml
# The QUEUE-ERA contract of tools/cloud/jobs.yaml (2026-09-05). Since the box slot-waiters (tools/cloud/slot_queue.sh: tmux r1-queue ->
# pe-queue) launch preregistered jobs one at a time with `launch --only <id>` - which ignores `enabled` - a WAITING job is listed
# `enabled: false` (a plain `launch` must never abort on "no free GPU slot" while it waits) and still names its prereg / amendment commit
# with a frozen sealed protocol. So neither "disabled => commit None" nor a fixed enabled list holds any more; what holds is:
#   * every job runs from a detached worktree; every PREREGISTERED job names a full SHA that is an ancestor of HEAD, passes it as
#     `--expect-commit`, and points at an existing sealed protocol with a transit time / memory / ms-per-step estimate;
#   * a waiting preregistered job (disabled, no state on the box) passes prereg_check at HEAD (commit reachable, sealed protocol frozen);
#   * a FINISHED launch superseded by an amendment (sweep-056 launch 1, at-alpha-1over16 launch 1) is disabled, still names the commit it
#     ran at, and its sealed protocol is NO LONGER frozen against HEAD (the amendment re-sealed the same path) - pinned so nobody relaunches it;
#   * launch-2 jobs (sweep-056-launch2, ext-val bohm-0.4) and the amended alpha cases name the AMENDMENT commit (after the prereg commit);
#   * non-preregistered slots are disabled: the plume placeholders name no commit, the graph shakedown keeps its recorded commit;
#   * the four-slot MPS accounting refuses a fifth client and accepts the next waiting job once a slot frees.
SWEEP_PREREG_COMMIT = "291a9227669c8927ea5cf7a6de2eed23fe6f73de"
SWEEP_JOBS = ["sweep-reference", "sweep-056", "sweep-047", "sweep-009"]
EXT_VAL_JOB = "ext-val-v0-channel-20um"
EXT_VAL_PREREG_COMMIT = "3dc12cf6d3a299c7c3702a1b2c349d69ffe1ddde"
EXT_VAL_BOHM_JOB = "ext-val-v0-channel-20um-bohm-0.4"
SS25_JOB = "ss25-base"
SWEEP_056_LAUNCH2_JOB = "sweep-056-launch2"
SS33_FAST_JOB = "ss33-fast"
# anomalous transport v1 (R1): prereg 057841cf; launch 1 (alpha-1over16) finished at 1.00 transit on the drift members (extinguished under the
# closure, record 0916a4f8, not relaunched); AMENDMENT 1 (v2.1.1 arming latch + ignition gate) re-sealed all three cases -> the two remaining
# cases wait in the r1-queue at the amendment commit
AT_PREREG_COMMIT = "057841cfcd72d24d09608143319079fc9f750e99"
AT_LAUNCH1_JOB = "at-alpha-1over16"
AT_WAITING_JOBS = ["at-alpha-1over64", "at-alpha-0.345"]
# physics effects v1 (R2 + R3): prereg 79a7c87a; wait in the chained pe-queue behind the r1-queue
PE_PREREG_COMMIT = "79a7c87a2fd1807180958f085f9dc36939488029"
PE_JOBS = ["pe-see-bn", "pe-xe-set-v2", "pe-see-bn+xe-set-v2"]
# full physics v1 (R4 + R5 + R1-R5 combined): prereg b45f6728; wait in the chained fp-queue behind the pe-queue, in the sustain-first order
FP_PREREG_COMMIT = "b45f672802fdb39dac7b427f4df6436694695fb0"
FP_JOBS = ["fp-full-physics-alpha0.345", "fp-full-physics-alpha0", "fp-neutrals-spatial", "fp-full-physics-alpha1over16", "fp-coulomb", "fp-neutrals-spatial-F10"]
PLUME_PLACEHOLDERS = ["plume-v2.1-33um", "plume-v2.1-50um-3mA"]
SUPERSEDED_LAUNCH1_JOBS = ["sweep-056", AT_LAUNCH1_JOB]
# the four CUDA-MPS clients on the box at 19:35 UTC 2026-09-04 (the r1-queue's next launch, at-alpha-1over64, waits for one of them)
LIVE_FOUR = [SS25_JOB, SS33_FAST_JOB, SWEEP_056_LAUNCH2_JOB, EXT_VAL_BOHM_JOB]


def _sealed(job: schedule.JobSpec) -> dict:
    return json.loads((REPOSITORY / job.protocol).read_text(encoding="utf-8"))


def test_shipped_jobs_yaml_is_the_single_h100_mps_configuration_under_the_queue_era_contract() -> None:
    pytest.importorskip("yaml")
    plan = schedule.build_plan(schedule.load_jobs_file(schedule.DEFAULT_JOBS_FILE), source=schedule.DEFAULT_JOBS_FILE)
    assert plan.defaults.repo == REPOSITORY
    # the live box: one H100, four CUDA-MPS slots (bench-mps 2026-09-04), MPS client variables exported to every job
    assert plan.gpus == [0] and plan.defaults.slots_per_gpu == 4
    assert plan.defaults.env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/nvidia-mps" and plan.defaults.env["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/nvidia-log"
    by_id = {j.id: j for j in plan.jobs}
    assert len(by_id) == len(plan.jobs)
    for name in SWEEP_JOBS + [EXT_VAL_JOB, EXT_VAL_BOHM_JOB, SS25_JOB, SWEEP_056_LAUNCH2_JOB, SS33_FAST_JOB, AT_LAUNCH1_JOB, *AT_WAITING_JOBS, *PE_JOBS, *FP_JOBS, *PLUME_PLACEHOLDERS]:
        assert name in by_id, name

    # -- every job: detached worktree; every SCHEDULED preregistered job names a full reachable SHA, passes it as --expect-commit and has
    #    a sealed protocol; a preregistered slot without a commit yet (the v4 replicate placeholders) is disabled
    for job in plan.jobs:
        assert job.checkout == "worktree", job.id
        assert job.gpu_memory_gib and job.expected_ms_per_step and job.gpu_memory_gib <= 0.9 * plan.defaults.gpu_memory_gib, job.id
        if job.preregistered and job.commit:
            assert job.protocol and (REPOSITORY / job.protocol).is_file() and job.results and job.transit_time_s, job.id
            assert len(job.commit) == 40 and schedule.is_ancestor(REPOSITORY, job.commit, "HEAD"), job.id
            assert job.args[:1] == ["launch"] and job.args[job.args.index("--expect-commit") + 1] == job.commit, job.id
        else:
            assert not job.enabled, f"{job.id}: a slot without a preregistered commit must not be planned by a plain `launch`"
    for name in PLUME_PLACEHOLDERS + ["ss33-seed-b", "ss33-w-0.7"]:
        assert by_id[name].commit is None and not by_id[name].enabled, name
    assert by_id["shakedown-ss-v3-graph"].commit and not by_id["shakedown-ss-v3-graph"].preregistered and not by_id["shakedown-ss-v3-graph"].enabled

    # -- waiting preregistered jobs: disabled AND prereg_check ok at HEAD (the slot-waiter's `launch --only` ignores `enabled`)
    for name in AT_WAITING_JOBS + PE_JOBS + FP_JOBS:
        job = by_id[name]
        assert not job.enabled and job.preregistered, name
        check = schedule.prereg_check(REPOSITORY, job.commit, job.protocol)
        assert check["ok"] is True and check["protocol_frozen"] is True, (name, check["problems"])
    # -- finished launch-1 jobs superseded by an amendment: disabled, the commit they ran at, sealed protocol no longer frozen against HEAD
    for name in SUPERSEDED_LAUNCH1_JOBS:
        job = by_id[name]
        assert not job.enabled and job.preregistered, name
        check = schedule.prereg_check(REPOSITORY, job.commit, job.protocol)
        assert check["ok"] is False and any("differs between" in p for p in check["problems"]), (name, check)
    # -- every ENABLED job is preregistered and its commit is reachable (finished or running launches stay enabled; the scheduler plans
    #    only jobs without a state on the box, so they take no slot)
    enabled = [j for j in plan.jobs if j.enabled]
    assert enabled and all(j.preregistered for j in enabled)
    assert {SS25_JOB, SS33_FAST_JOB, SWEEP_056_LAUNCH2_JOB, EXT_VAL_BOHM_JOB} <= {j.id for j in enabled}     # the four live clients
    assert not ({AT_LAUNCH1_JOB, *AT_WAITING_JOBS, *PE_JOBS, *FP_JOBS, "sweep-056"} & {j.id for j in enabled})

    # -- the sweep (prereg 291a9227): launch 1 of 056 superseded by launch 2 at the amendment commit with the same sealed path
    for name in SWEEP_JOBS:
        job = by_id[name]
        assert job.commit == SWEEP_PREREG_COMMIT and job.protocol.startswith("modern/experiments/pic2d_design_mini_sweep_v1/protocols/"), name
        assert "--require-mps" in job.args
    launch2 = by_id[SWEEP_056_LAUNCH2_JOB]
    sweep_056 = by_id["sweep-056"]
    assert launch2.commit != SWEEP_PREREG_COMMIT and schedule.is_ancestor(REPOSITORY, SWEEP_PREREG_COMMIT, launch2.commit)
    assert launch2.protocol == sweep_056.protocol and launch2.results == sweep_056.results and launch2.transit_time_s == sweep_056.transit_time_s
    assert launch2.args == ["launch", "--design", "l1a-gs-v3-056-effcbc8686", "--domain", "channel", "--grid", "33um", "--expect-commit", launch2.commit, "--require-mps"]
    sealed = _sealed(launch2)
    assert sealed["omega_pe_dt_gate_reading"]["statistic"] == "resolved_node_single_step_peak" and sealed["omega_pe_dt_gate_reading"]["min_macro_particles"] == 32

    # -- external validation v0: launch 1 (channel-20um, prereg 3dc12cf6) and the bohm-0.4 launch 2 at its amendment commit
    ext = by_id[EXT_VAL_JOB]
    assert ext.commit == EXT_VAL_PREREG_COMMIT and ext.args == ["launch", "--expect-commit", EXT_VAL_PREREG_COMMIT, "--require-mps"]
    assert ext.protocol == "modern/experiments/pic2d_external_validation_v0/protocols/brandt2016-micro-hempt-v1-channel-20um.json"
    assert ext.results == "modern/experiments/pic2d_external_validation_v0/results/channel-20um" and ext.transit_time_s == 1.4e-6 and ext.gpu_memory_gib >= 17
    bohm = by_id[EXT_VAL_BOHM_JOB]
    assert bohm.commit != EXT_VAL_PREREG_COMMIT and schedule.is_ancestor(REPOSITORY, EXT_VAL_PREREG_COMMIT, bohm.commit)
    assert bohm.args == ["launch", "--variant", "bohm-0.4", "--grid", "20um", "--expect-commit", bohm.commit, "--require-mps"]
    assert bohm.protocol.endswith("brandt2016-micro-hempt-v1-channel-20um-bohm-0.4.json") and bohm.transit_time_s == 1.4e-6
    assert _sealed(bohm)["numerics"]["anomalous_collisions"]["model"] == "bohm_perpendicular_rotation"

    # -- steady-state v5 (25 um ladder point, launch 2) and the v4-fast solver qualification
    ss25 = by_id[SS25_JOB]
    assert ss25.args == ["launch", "--expect-commit", ss25.commit] and ss25.module == "experiments.pic2d_cft_steady_state_v5.run" and "PLACEHOLDER" not in ss25.note
    assert ss25.protocol == "modern/experiments/pic2d_cft_steady_state_v5/protocol.json" and ss25.transit_time_s == pytest.approx(2.4e-6)
    fast = by_id[SS33_FAST_JOB]
    assert fast.args == ["launch", "--expect-commit", fast.commit, "--require-mps"] and fast.module == "experiments.pic2d_cft_steady_state_v4_fast.run"
    protocol = _sealed(fast)
    assert protocol["numerics"]["poisson"]["method"] == "device-mg" and protocol["numerics"]["poisson"]["cycles"] == 14
    assert protocol["numerics"]["performance"]["moment_sample_interval"] == 5 and protocol["stopping_rule"]["wall_budget_seconds"] == 102100

    # -- anomalous transport v1: launch 1 at the prereg commit (finished, superseded); the two remaining cases at AMENDMENT 1
    launch1 = by_id[AT_LAUNCH1_JOB]
    assert launch1.commit == AT_PREREG_COMMIT and launch1.args == ["launch", "--case", "alpha-1over16", "--expect-commit", AT_PREREG_COMMIT, "--require-mps"]
    assert "EXTINGUISHED" in launch1.note and "NOT relaunched" in launch1.note
    amendment = {by_id[name].commit for name in AT_WAITING_JOBS}
    assert len(amendment) == 1
    amendment_commit = amendment.pop()
    assert amendment_commit != AT_PREREG_COMMIT and schedule.is_ancestor(REPOSITORY, AT_PREREG_COMMIT, amendment_commit)
    for name in AT_WAITING_JOBS:
        job = by_id[name]
        case = name.removeprefix("at-")
        assert job.args == ["launch", "--case", case, "--expect-commit", amendment_commit, "--require-mps"] and job.transit_time_s == pytest.approx(2.4e-6)
        assert job.protocol == f"modern/experiments/pic2d_anomalous_transport_v1/protocols/{case}.json" and job.results.endswith(f"/results/{case}")
        sealed = _sealed(job)
        arming = sealed["stopping_rule"]["grid_heating_triad"]["drift_members_arming"]
        assert arming["min_transit_times"] == 2.0 and arming["settle_quantity"] == "discharge_current" and arming["settle_check_cadence_steps"] == 40_000
        assert [c["time_s"] for c in sealed["stopping_rule"]["ignition_gate"]["checks"]] == [1.0e-6, 2.0e-6]
    # the launch-1 sealed path was re-sealed by the amendment: its protocol at HEAD carries the amendment keys the launch never ran under
    assert "drift_members_arming" in _sealed(launch1)["stopping_rule"]["grid_heating_triad"]

    # -- physics effects v1 (chained pe-queue): three cases at the prereg commit, case args, sealed protocols
    for name in PE_JOBS:
        job = by_id[name]
        case = name.removeprefix("pe-")
        assert job.commit == PE_PREREG_COMMIT and job.args == ["launch", "--case", case, "--expect-commit", PE_PREREG_COMMIT, "--require-mps"], name
        assert job.protocol == f"modern/experiments/pic2d_physics_effects_v1/protocols/{case}.json"

    # -- full physics v1 (chained fp-queue after the pe-queue): six cases at the prereg commit in the sustain-first order, case args, sealed protocols with the
    #    v2.1.1 arming latch + ignition gate; the spatial cases carry the MCC ceiling above the Knudsen anode density and no 0-D inventory; the F pair differs in F only
    for name in FP_JOBS:
        job = by_id[name]
        case = name.removeprefix("fp-")
        assert job.commit == FP_PREREG_COMMIT and job.args == ["launch", "--case", case, "--expect-commit", FP_PREREG_COMMIT, "--require-mps"], name
        assert job.protocol == f"modern/experiments/pic2d_full_physics_v1/protocols/{case}.json" and job.results.endswith(f"/results/{case}") and job.transit_time_s == pytest.approx(2.4e-6)
        sealed = _sealed(job)
        assert sealed["stopping_rule"]["grid_heating_triad"]["drift_members_arming"]["min_transit_times"] == 2.0
        assert [c["time_s"] for c in sealed["stopping_rule"]["ignition_gate"]["checks"]] == [1.0e-6, 2.0e-6]
        assert sealed["campaign"]["launch_priority"] == [n.removeprefix("fp-") for n in FP_JOBS]
        op = sealed["operating_point"]
        if "neutrals" in op:
            assert "neutral_inventory" not in op and op["neutral_density_per_m3"] > 5.45e20 and op["neutrals"]["metastables"]["model"] == "metastables_v1"
            assert op["neutrals"]["time_acceleration"] == (10.0 if case.endswith("F10") else 1.0)
        else:
            assert case == "coulomb" and sealed["numerics"]["coulomb"]["cycle_steps"] == 10
    assert _sealed(by_id["fp-full-physics-alpha0.345"])["numerics"]["anomalous_collisions"]["alpha"] == pytest.approx(0.345)
    assert "anomalous_collisions" not in _sealed(by_id["fp-full-physics-alpha0"])["numerics"]

    # -- four-slot accounting: the r1-queue's next launch is refused while the four live clients hold the slots and accepted once one frees;
    #    the chained pe-queue's first job behaves the same; two waiting jobs never share a freed slot
    live = [by_id[name] for name in LIVE_FOUR]
    assert schedule.assign_gpus(plan, live) == {j.id: 0 for j in live}
    nxt = by_id[AT_WAITING_JOBS[0]]
    with pytest.raises(schedule.ScheduleError):
        schedule.assign_gpus(plan, [nxt], busy={0: [j.id for j in live]})
    with pytest.raises(schedule.ScheduleError):
        schedule.assign_gpus(plan, live + [nxt])
    for freed in live:
        remaining = [j.id for j in live if j is not freed]
        assert schedule.assign_gpus(plan, [nxt], busy={0: remaining}) == {nxt.id: 0}
        with pytest.raises(schedule.ScheduleError):
            schedule.assign_gpus(plan, [nxt, by_id[PE_JOBS[0]]], busy={0: remaining})
