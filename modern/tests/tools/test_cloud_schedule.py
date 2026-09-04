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
SWEEP_PREREG_COMMIT = "291a9227669c8927ea5cf7a6de2eed23fe6f73de"
SWEEP_JOBS = ["sweep-reference", "sweep-056", "sweep-047", "sweep-009"]
EXT_VAL_JOB = "ext-val-v0-channel-20um"
EXT_VAL_PREREG_COMMIT = "3dc12cf6d3a299c7c3702a1b2c349d69ffe1ddde"
# steady-state v5 launch 2 (25 um ladder point; launch 1 on the local RTX 5090 withdrawn by the user): the job's commit is the
# v5.1 records commit (protocol amendment + H100 preflight / shakedown), read from jobs.yaml. Slot sequencing on the box
# (2026-09-04): design 056 ended on its triad gate at 10:52 UTC and design 047 reaches 3 transits ~12:50 UTC, so the two
# newcomers (ss25-base, ext-val-v0-channel-20um) enter one after the other; `launch` plans only jobs whose state is "not launched".
SS25_JOB = "ss25-base"
# design 056 launch 1 stopped on its triad gate (raw omega_pe dt statistic) at 10:52 UTC; design 047 finished on the plateau rule at
# 12:49 UTC. Launch 2 of 056 (amendment 1: model v2.0.4 gate reading, protocol.json amendments[0]) is its own job whose commit is
# the AMENDMENT commit (read from jobs.yaml, a full SHA after the prereg commit) and whose sealed protocol carries the
# omega_pe_dt_gate_reading block; it enters the slot the reference or 009 frees.
FINISHED_SWEEP_JOBS = ["sweep-056", "sweep-047"]
SWEEP_056_LAUNCH2_JOB = "sweep-056-launch2"
# steady-state v4-fast (solver qualification: the v4 33 um plateau replayed under device-mg + K = 5): its commit is the preregistration
# commit (read from jobs.yaml), `--require-mps`, the protocol names device-mg / K = 5. It enters a slot freed by the finished runs
# (sweep-reference / 009 / 047 finished on the plateau rule, ext-val v0 on its triad gate by 14:17 UTC 2026-09-04).
SS33_FAST_JOB = "ss33-fast"


def test_shipped_jobs_yaml_is_the_single_h100_mps_configuration_with_the_preregistered_sweep_enabled() -> None:
    pytest.importorskip("yaml")
    plan = schedule.build_plan(schedule.load_jobs_file(schedule.DEFAULT_JOBS_FILE), source=schedule.DEFAULT_JOBS_FILE)
    assert plan.defaults.repo == REPOSITORY
    # the live box: one H100, four CUDA-MPS slots (bench-mps 2026-09-04), MPS client variables exported to every job
    assert plan.gpus == [0] and plan.defaults.slots_per_gpu == 4
    assert plan.defaults.env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/nvidia-mps" and plan.defaults.env["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/nvidia-log"
    enabled = [j for j in plan.jobs if j.enabled]
    # steady-state v5 (section ii of the file), the four sweep designs (056 finished on its triad gate at 10:52 UTC 2026-09-04,
    # 047 on the plateau rule at 12:49 UTC), the 056 launch-2 job (amendment 1) and external validation v0
    assert [j.id for j in enabled] == [SS25_JOB, SS33_FAST_JOB, "sweep-reference", SWEEP_056_LAUNCH2_JOB, "sweep-047", "sweep-009", EXT_VAL_JOB]
    for job in plan.jobs:
        assert job.checkout == "worktree"
        if job.id == "sweep-056":
            # launch 1 of design 056: finished (triad stop) and superseded by the amendment - disabled, still names the prereg commit,
            # and its sealed protocol is NO LONGER frozen against HEAD (amendment 1 re-sealed it) - pinned so nobody relaunches it
            assert not job.enabled and job.commit == SWEEP_PREREG_COMMIT and job.preregistered is True
            check = schedule.prereg_check(REPOSITORY, job.commit, job.protocol)
            assert check["ok"] is False and any("differs between" in p for p in check["problems"])
        elif job.id == SWEEP_056_LAUNCH2_JOB:
            # launch 2 of design 056: the amendment commit (after the prereg commit), the SAME sealed protocol path as sweep-056
            # (re-sealed with the gate-reading block), the canonical results directory, --expect-commit == commit
            sweep_056 = next(j for j in plan.jobs if j.id == "sweep-056")
            assert job.enabled and job.preregistered is True and len(job.commit) == 40 and job.commit != SWEEP_PREREG_COMMIT
            assert schedule.is_ancestor(REPOSITORY, SWEEP_PREREG_COMMIT, job.commit)
            assert job.protocol == sweep_056.protocol and job.results == sweep_056.results and job.transit_time_s == sweep_056.transit_time_s
            assert job.args == ["launch", "--design", "l1a-gs-v3-056-effcbc8686", "--domain", "channel", "--grid", "33um", "--expect-commit", job.commit, "--require-mps"]
            sealed = json.loads((REPOSITORY / job.protocol).read_text(encoding="utf-8"))
            assert sealed["omega_pe_dt_gate_reading"]["statistic"] == "resolved_node_single_step_peak" and sealed["omega_pe_dt_gate_reading"]["min_macro_particles"] == 32
            assert job.gpu_memory_gib and job.expected_ms_per_step
        elif job.id.startswith("sweep-"):
            # every sealed mini-sweep slot (launched or not) names the preregistration commit and its own sealed run protocol
            assert job.commit == SWEEP_PREREG_COMMIT and job.preregistered is True and (REPOSITORY / job.protocol).is_file(), job.id
            assert job.args[:1] == ["launch"] and "--require-mps" in job.args and job.args[job.args.index("--expect-commit") + 1] == SWEEP_PREREG_COMMIT
            assert job.protocol.startswith("modern/experiments/pic2d_design_mini_sweep_v1/protocols/") and job.transit_time_s and job.gpu_memory_gib
        elif job.id == EXT_VAL_JOB:
            assert job.commit == EXT_VAL_PREREG_COMMIT and job.preregistered is True and (REPOSITORY / job.protocol).is_file()
            assert job.args == ["launch", "--expect-commit", EXT_VAL_PREREG_COMMIT, "--require-mps"]
            assert job.protocol == "modern/experiments/pic2d_external_validation_v0/protocols/brandt2016-micro-hempt-v1-channel-20um.json"
            assert job.results == "modern/experiments/pic2d_external_validation_v0/results/channel-20um" and job.transit_time_s == 1.4e-6 and job.gpu_memory_gib >= 17
        elif job.id == SS25_JOB:
            # the v5 launch-2 job: preregistered, `--expect-commit` == `commit` (a full SHA, the v5.1 records commit), the v5 protocol
            assert job.preregistered is True and job.commit and len(job.commit) == 40 and "PLACEHOLDER" not in job.note
            assert job.args == ["launch", "--expect-commit", job.commit] and job.module == "experiments.pic2d_cft_steady_state_v5.run"
            assert job.protocol == "modern/experiments/pic2d_cft_steady_state_v5/protocol.json" and (REPOSITORY / job.protocol).is_file()
            assert job.results == "modern/experiments/pic2d_cft_steady_state_v5/results" and job.transit_time_s == pytest.approx(2.4e-6)
            assert job.gpu_memory_gib and job.expected_ms_per_step
        elif job.id == SS33_FAST_JOB:
            # the solver-qualification replay: preregistered, `--expect-commit` == `commit` (a full SHA, the prereg commit), --require-mps,
            # the v4-fast protocol at HEAD selects the multigrid and K = 5 (the two identity differences vs v4)
            assert job.preregistered is True and job.commit and len(job.commit) == 40 and "PLACEHOLDER" not in job.note
            assert job.args == ["launch", "--expect-commit", job.commit, "--require-mps"] and job.module == "experiments.pic2d_cft_steady_state_v4_fast.run"
            assert job.protocol == "modern/experiments/pic2d_cft_steady_state_v4_fast/protocol.json" and (REPOSITORY / job.protocol).is_file()
            assert job.results == "modern/experiments/pic2d_cft_steady_state_v4_fast/results" and job.transit_time_s == pytest.approx(2.4e-6)
            assert job.gpu_memory_gib and job.expected_ms_per_step
            protocol = json.loads((REPOSITORY / job.protocol).read_text(encoding="utf-8"))
            assert protocol["numerics"]["poisson"]["method"] == "device-mg" and protocol["numerics"]["poisson"]["cycles"] == 14
            assert protocol["numerics"]["performance"]["moment_sample_interval"] == 5 and protocol["stopping_rule"]["wall_budget_seconds"] == 102100
        elif job.id == "shakedown-ss-v3-graph":
            assert not job.enabled and job.preregistered is False and job.commit   # never relabelled; disabled so `launch` takes no slot
        else:
            assert not job.enabled and job.commit is None, f"{job.id}: disabled placeholder slots must not name a commit yet"
    # the enabled jobs fit the memory rule and the prereg check holds for each (commit reachable, sealed protocol frozen)
    assert sum(j.gpu_memory_gib for j in enabled) <= 0.9 * plan.defaults.gpu_memory_gib
    for job in enabled:
        check = schedule.prereg_check(REPOSITORY, job.commit, job.protocol)
        assert check["ok"] is True, (job.id, check["problems"])
    # four MPS slots (2026-09-04 evening): the two sweep runs still executing (reference, 009) + the two newcomers that took the
    # slots 056 and 047 freed (v5, external validation) are the four clients; the 056 launch-2 job is a FIFTH job that fits only
    # once the reference or 009 frees a slot - `launch` plans only "not launched" jobs, so `--only sweep-056-launch2` is refused
    # while four clients run and accepted with three
    running_sweep = [j for j in enabled if j.id.startswith("sweep-") and j.id not in FINISHED_SWEEP_JOBS + [SWEEP_056_LAUNCH2_JOB]]
    newcomers = [j for j in enabled if j.id in (EXT_VAL_JOB, SS25_JOB)]
    launch2 = next(j for j in enabled if j.id == SWEEP_056_LAUNCH2_JOB)
    assert len(running_sweep) == 2 and len(newcomers) == 2
    four = running_sweep + newcomers
    assert len(four) == plan.defaults.slots_per_gpu and schedule.assign_gpus(plan, four) == {j.id: 0 for j in four}
    with pytest.raises(schedule.ScheduleError):        # a fifth concurrent job would exceed the four MPS slots
        schedule.assign_gpus(plan, four + [launch2])
    with pytest.raises(schedule.ScheduleError):
        schedule.assign_gpus(plan, enabled)
    # once the reference or 009 has finished, launch 2 of 056 takes the freed slot (four clients again)
    for finished in running_sweep:
        remaining = [j for j in four if j is not finished] + [launch2]
        assert schedule.assign_gpus(plan, remaining) == {j.id: 0 for j in remaining}
    # and the scheduler's own busy-slot accounting refuses it while four jobs are running
    with pytest.raises(schedule.ScheduleError):
        schedule.assign_gpus(plan, [launch2], busy={0: [j.id for j in four]})
    # 14:17 UTC 2026-09-04: sweep-reference, 009, 047 and ext-val v0 have finished; ss25-base and 056 launch 2 hold two slots -> the
    # solver-qualification job ss33-fast takes a freed slot (three clients), and a fourth newcomer would still fit, a fifth not
    fast = next(j for j in enabled if j.id == SS33_FAST_JOB)
    assert schedule.assign_gpus(plan, [fast], busy={0: [SS25_JOB, SWEEP_056_LAUNCH2_JOB]}) == {fast.id: 0}
    with pytest.raises(schedule.ScheduleError):
        schedule.assign_gpus(plan, [fast], busy={0: [j.id for j in four]})
