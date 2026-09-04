"""Tiny GPU job scheduler for one multi-GPU box (Lambda 8x H100): N GPUs x slots-per-GPU.

Each job in ``jobs.yaml`` is an experiment module + arguments + environment.  ``launch`` pins it to
a GPU (``CUDA_DEVICE_ORDER=PCI_BUS_ID`` + ``CUDA_VISIBLE_DEVICES=<index>``), starts it detached in
``tmux`` (or ``setsid nohup``) through a small wrapper (``_wrap``) that records ``jobs/<id>/state.json``
(pid, GPU model/index/UUID, start, end, exit code) and writes the job's own log; ``status`` tails
``run_state.json`` / ``status.jsonl`` of every job (steps, simulated time, ms/step, stop reason,
ETA to the transit target).

Preregistration discipline (fail-closed, nothing here edits a protocol):

* every job names the commit that froze its protocol (the preregistration commit for a preregistered
  experiment, the protocol-freeze commit for a declared development run); the scheduler refuses to
  launch unless ``git merge-base --is-ancestor <commit> HEAD`` holds in the job's checkout
  (the same check ``_bind_preregistration`` makes against the authorised branch) and the protocol
  file is byte-identical between that commit and the working tree (``git diff --quiet <commit> HEAD --
  <protocol>`` and a clean ``git status`` for that path);
* the job's ``preregistered`` flag is recorded verbatim in ``state.json`` - a development run is
  never relabelled;
* GPU provenance: the runner records ``backend: warp-cuda:0`` (always ``cuda:0`` under
  ``CUDA_VISIBLE_DEVICES``) and its ``nvidia-smi`` utilisation sampler reads the box's FIRST GPU, so
  the scheduler records the pinned GPU's index / name / UUID from ``nvidia-smi --id`` and the
  wrapper cross-checks it against the UUID Warp reports for ``cuda:0`` inside the job's environment
  before the command starts (mismatch -> the job does not start).

Usage (from ``modern/`` on the box; ``schedule.py`` finds the repository from its own location)::

    python tools/cloud/schedule.py gpus
    python tools/cloud/schedule.py plan   [--jobs tools/cloud/jobs.yaml]
    python tools/cloud/schedule.py launch [--only ss33-seed-b ...] [--dry-run]
    python tools/cloud/schedule.py status [--json] [--watch 300]
    python tools/cloud/schedule.py stop ss33-seed-b
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
REPOSITORY = MODERN.parent
DEFAULT_JOBS_FILE = HERE / "jobs.yaml"
STATE_FILE = "state.json"
COMMAND_FILE = "command.json"
RUN_LOG = "run.log"
WRAPPER_LOG = "wrapper.log"
BLAS_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
SCHEMA = "cft-revival.tools.cloud.schedule/0.1.0"


class ScheduleError(RuntimeError):
    """A job cannot be planned or launched as specified (fail closed)."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------------------------------ git helpers
def git(repo: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=False)
    if check and completed.returncode != 0:
        raise ScheduleError(f"git {' '.join(args)} failed in {repo}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def resolve_commit(repo: Path, ref: str) -> str:
    completed = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                               cwd=str(repo), capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ScheduleError(f"commit {ref!r} is not in the repository {repo} (fetch it first)")
    return completed.stdout.strip()


def is_ancestor(repo: Path, commit: str, head: str = "HEAD") -> bool:
    """``git merge-base --is-ancestor commit head``: 0 = yes, 1 = no, anything else = error."""

    completed = subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=str(repo),
                               capture_output=True, text=True, check=False)
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    raise ScheduleError(f"git merge-base --is-ancestor failed: {completed.stderr.strip()}")


def protocol_frozen(repo: Path, commit: str, protocol: str) -> tuple[bool, str]:
    """The protocol file is byte-identical between ``commit``, HEAD and the working tree."""

    diff = subprocess.run(["git", "diff", "--quiet", commit, "HEAD", "--", protocol], cwd=str(repo),
                          capture_output=True, text=True, check=False)
    if diff.returncode == 1:
        return False, f"{protocol} differs between {commit[:12]} and HEAD"
    if diff.returncode != 0:
        raise ScheduleError(f"git diff failed for {protocol}: {diff.stderr.strip()}")
    status = git(repo, "status", "--porcelain", "--", protocol)
    if status:
        return False, f"{protocol} is modified or untracked in the working tree: {status}"
    if not (repo / protocol).is_file():
        return False, f"{protocol} does not exist in the working tree"
    return True, "frozen"


def prereg_check(repo: Path, commit_ref: str, protocol: str | None, *, require_clean: bool = False) -> dict[str, Any]:
    """Fail-closed preregistration / protocol-freeze verification for one job."""

    commit = resolve_commit(repo, commit_ref)
    head = resolve_commit(repo, "HEAD")
    ancestor = is_ancestor(repo, commit, "HEAD")
    record: dict[str, Any] = {"commit_ref": commit_ref, "commit": commit, "head": head, "is_ancestor_of_head": ancestor,
                              "checked_utc": utc_now()}
    problems: list[str] = []
    if not ancestor:
        problems.append(f"{commit_ref} ({commit[:12]}) is not reachable from HEAD {head[:12]}")
    if protocol:
        frozen, detail = protocol_frozen(repo, commit, protocol)
        record["protocol"] = protocol
        record["protocol_frozen"] = frozen
        record["protocol_detail"] = detail
        if not frozen:
            problems.append(detail)
    if require_clean:
        dirty = git(repo, "status", "--porcelain", "--untracked-files=no")
        record["worktree_clean"] = not dirty
        if dirty:
            problems.append("worktree has uncommitted tracked changes:\n" + dirty)
    record["ok"] = not problems
    record["problems"] = problems
    return record


# ------------------------------------------------------------------------------------------ nvidia-smi
def nvidia_smi(args: list[str], timeout_s: float = 10.0) -> list[list[str]]:
    try:
        completed = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True,
                                   text=True, check=True, timeout=timeout_s)
    except Exception:  # noqa: BLE001 - telemetry is optional; the planner works from the yaml
        return []
    return [[cell.strip() for cell in line.split(",")] for line in completed.stdout.splitlines() if line.strip()]


def gpu_inventory() -> list[dict[str, Any]]:
    inventory = []
    for row in nvidia_smi(["--query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id"]):
        if len(row) >= 6:
            inventory.append({"index": int(row[0]), "name": row[1], "uuid": row[2], "driver_version": row[3],
                              "memory_total_mib": float(row[4]), "pci_bus_id": row[5]})
    return inventory


def gpu_live(index: int) -> dict[str, Any]:
    rows = nvidia_smi(["--query-gpu=utilization.gpu,memory.used,memory.total", f"--id={index}"])
    if not rows or len(rows[0]) < 3:
        return {}
    try:
        return {"utilisation_percent": float(rows[0][0]), "memory_used_mib": float(rows[0][1]),
                "memory_total_mib": float(rows[0][2])}
    except ValueError:
        return {}


# ------------------------------------------------------------------------------------------ specification
@dataclass
class JobSpec:
    id: str
    module: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    gpu: int | str = "auto"
    commit: str | None = None
    preregistered: bool = False
    protocol: str | None = None
    results: str | None = None
    checkout: str = "shared"              # shared | worktree
    gpu_memory_gib: float | None = None
    expected_ms_per_step: float | None = None
    target_transits: float = 3.0
    transit_time_s: float | None = None
    enabled: bool = True
    note: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> JobSpec:
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ScheduleError(f"job {raw.get('id')!r}: unknown keys {unknown}")
        if not raw.get("id") or not raw.get("module"):
            raise ScheduleError("every job needs 'id' and 'module'")
        spec = cls(**{k: v for k, v in raw.items()})
        spec.args = [str(a) for a in spec.args]
        spec.env = {str(k): str(v) for k, v in (spec.env or {}).items()}
        if spec.checkout not in ("shared", "worktree"):
            raise ScheduleError(f"job {spec.id}: checkout must be 'shared' or 'worktree'")
        if spec.gpu != "auto":
            spec.gpu = int(spec.gpu)
        return spec


@dataclass
class Defaults:
    repo: Path
    python: str
    cwd: str = "modern"
    jobs_dir: Path = Path("jobs")
    launcher: str = "tmux"                 # tmux | setsid | exec
    slots_per_gpu: int = 1
    gpu_memory_gib: float = 80.0
    env: dict[str, str] = field(default_factory=dict)
    warp_probe: str = "verify"             # verify | record | off
    require_clean: bool = False


@dataclass
class Plan:
    defaults: Defaults
    gpus: list[int]
    jobs: list[JobSpec]
    source: Path | None = None


def load_jobs_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as error:   # pragma: no cover - the cloud venv pins pyyaml
        raise ScheduleError("PyYAML is required to read a .yaml jobs file (uv pip install pyyaml), "
                            "or pass a .json file") from error
    return yaml.safe_load(text)


def build_plan(raw: Mapping[str, Any], *, source: Path | None = None, base: Path | None = None) -> Plan:
    """Validate the jobs document and resolve paths (relative to the file's directory)."""

    if int(raw.get("version", 0)) != 1:
        raise ScheduleError("jobs file must declare version: 1")
    base_dir = base if base is not None else (source.parent if source is not None else Path.cwd())
    d = dict(raw.get("defaults") or {})
    repo = (base_dir / d.get("repo", "../../..")).resolve()
    jobs_dir_raw = Path(d.get("jobs_dir", "../../../../jobs"))
    jobs_dir = jobs_dir_raw if jobs_dir_raw.is_absolute() else (base_dir / jobs_dir_raw).resolve()
    defaults = Defaults(
        repo=repo, python=str(d.get("python", ".venv-pic/bin/python")), cwd=str(d.get("cwd", "modern")),
        jobs_dir=jobs_dir, launcher=str(d.get("launcher", "tmux")), slots_per_gpu=int(d.get("slots_per_gpu", 1)),
        gpu_memory_gib=float(d.get("gpu_memory_gib", 80.0)),
        env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
        warp_probe=str(d.get("warp_probe", "verify")), require_clean=bool(d.get("require_clean", False)),
    )
    if defaults.launcher not in ("tmux", "setsid", "exec"):
        raise ScheduleError("defaults.launcher must be tmux, setsid or exec")
    if defaults.warp_probe not in ("verify", "record", "off"):
        raise ScheduleError("defaults.warp_probe must be verify, record or off")
    gpus_raw = raw.get("gpus", "auto")
    if gpus_raw == "auto":
        gpus = [g["index"] for g in gpu_inventory()]
    else:
        gpus = [int(g) for g in gpus_raw]
    jobs = [JobSpec.from_mapping(j) for j in (raw.get("jobs") or [])]
    ids = [j.id for j in jobs]
    if len(set(ids)) != len(ids):
        raise ScheduleError(f"duplicate job ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    return Plan(defaults=defaults, gpus=gpus, jobs=jobs, source=source)


# ------------------------------------------------------------------------------------------ state files
def job_dir(plan: Plan, job_id: str) -> Path:
    return plan.defaults.jobs_dir / job_id


def read_state(path: Path) -> dict[str, Any] | None:
    state_path = path / STATE_FILE
    if not state_path.is_file():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_state(path: Path, state: Mapping[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    target = path / STATE_FILE
    temporary = path / f".{STATE_FILE}.tmp"
    temporary.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, target)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":   # pragma: no cover - Linux box; kept so `status` works from a Windows checkout
        completed = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                                   check=False)
        return str(pid) in completed.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def job_status_word(state: Mapping[str, Any] | None) -> str:
    if state is None:
        return "not launched"
    if state.get("exit_code") is not None:
        return "finished" if state["exit_code"] == 0 else f"failed ({state['exit_code']})"
    if state.get("refused"):
        return "refused"
    if pid_alive(state.get("pid")) or pid_alive(state.get("wrapper_pid")):
        return "running"
    return "lost (no exit code, pid gone)"


# ------------------------------------------------------------------------------------------ GPU assignment
def occupied_slots(plan: Plan) -> dict[int, list[str]]:
    """GPU index -> ids of jobs whose state says they are still running."""

    busy: dict[int, list[str]] = {g: [] for g in plan.gpus}
    if plan.defaults.jobs_dir.is_dir():
        for path in sorted(plan.defaults.jobs_dir.iterdir()):
            state = read_state(path)
            if state is None or job_status_word(state) != "running":
                continue
            gpu = state.get("gpu", {}).get("index") if isinstance(state.get("gpu"), dict) else state.get("gpu_index")
            if gpu is not None:
                busy.setdefault(int(gpu), []).append(str(state.get("id", path.name)))
    return busy


def assign_gpus(plan: Plan, jobs: list[JobSpec], busy: Mapping[int, list[str]] | None = None,
                memory_estimates: Mapping[str, float] | None = None) -> dict[str, int]:
    """Explicit GPUs first, then ``auto`` jobs onto the least-loaded GPU with a free slot (fail if none)."""

    capacity = plan.defaults.slots_per_gpu
    load: dict[int, int] = {g: len((busy or {}).get(g, [])) for g in plan.gpus}
    memory: dict[int, float] = {g: 0.0 for g in plan.gpus}
    assignment: dict[str, int] = {}
    for job in jobs:
        if job.gpu == "auto":
            continue
        gpu = int(job.gpu)
        if gpu not in load:
            raise ScheduleError(f"job {job.id}: GPU {gpu} is not in the plan's GPU list {plan.gpus}")
        if load[gpu] >= capacity:
            raise ScheduleError(f"job {job.id}: GPU {gpu} has no free slot ({load[gpu]}/{capacity} used)")
        load[gpu] += 1
        assignment[job.id] = gpu
    for job in jobs:
        if job.gpu != "auto":
            continue
        candidates = [g for g in plan.gpus if load[g] < capacity]
        if not candidates:
            raise ScheduleError(f"job {job.id}: no free GPU slot ({len(plan.gpus)} GPUs x {capacity} slots); "
                                "wait for a job to finish or raise slots_per_gpu from the benchmark")
        gpu = min(candidates, key=lambda g: (load[g], memory[g], g))
        load[gpu] += 1
        assignment[job.id] = gpu
    for job in jobs:
        estimate = (memory_estimates or {}).get(job.id, job.gpu_memory_gib)
        if estimate:
            memory[assignment[job.id]] += float(estimate)
    for gpu, total in memory.items():
        if total > 0.9 * plan.defaults.gpu_memory_gib:
            raise ScheduleError(f"GPU {gpu}: declared job memory {total:.1f} GiB exceeds 90 % of "
                                f"{plan.defaults.gpu_memory_gib:.0f} GiB")
    return assignment


def blas_threads_for(plan: Plan, concurrent_jobs: int, cpu_count: int | None = None, cap: int = 16) -> int:
    cores = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    return max(1, min(cap, cores // max(concurrent_jobs, 1)))


# ------------------------------------------------------------------------------------------ launching
def job_checkout(plan: Plan, job: JobSpec, commit: str, *, dry_run: bool = False) -> Path:
    """The repository the job runs from: the shared checkout or a detached worktree at ``commit``."""

    if job.checkout == "shared":
        return plan.defaults.repo
    tree = job_dir(plan, job.id) / "tree"
    if tree.is_dir() and (tree / ".git").exists():
        at = resolve_commit(tree, "HEAD")
        if at != commit:
            raise ScheduleError(f"job {job.id}: worktree {tree} is at {at[:12]}, not {commit[:12]}")
        return tree
    if dry_run:
        return tree
    tree.parent.mkdir(parents=True, exist_ok=True)
    git(plan.defaults.repo, "worktree", "add", "--detach", str(tree), commit)
    return tree


def build_command(plan: Plan, job: JobSpec, checkout: Path, gpu: int, threads: int) -> dict[str, Any]:
    python = Path(plan.defaults.python)
    if not python.is_absolute():
        python = plan.defaults.repo / python
    cwd = checkout / plan.defaults.cwd
    env: dict[str, str] = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu)}
    env.update({name: str(threads) for name in BLAS_THREAD_VARIABLES})
    env.setdefault("PYTHONPATH", f"src{os.pathsep}.")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.update(plan.defaults.env)
    env.update(job.env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)     # the pin is not overridable
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return {"command": [str(python), "-u", "-m", job.module, *job.args], "cwd": str(cwd), "env": env}


def render_shell(command: Mapping[str, Any]) -> str:
    parts = [f"{k}={shlex.quote(v)}" for k, v in command["env"].items()]
    return f"cd {shlex.quote(command['cwd'])} && env {' '.join(parts)} {' '.join(shlex.quote(c) for c in command['command'])}"


def detach(plan: Plan, job: JobSpec, wrapper: list[str], *, log_path: Path) -> dict[str, Any]:
    launcher = plan.defaults.launcher
    if launcher == "tmux":
        session = f"pic-{job.id}"
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(job_dir(plan, job.id)),
                        " ".join(shlex.quote(c) for c in wrapper) + f" >> {shlex.quote(str(log_path))} 2>&1"],
                       check=True)
        return {"launcher": "tmux", "tmux_session": session}
    if launcher == "setsid":
        with open(log_path, "ab") as log:
            proc = subprocess.Popen(["setsid", "nohup", *wrapper], stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, cwd=str(job_dir(plan, job.id)), start_new_session=True)
        return {"launcher": "setsid", "wrapper_pid": proc.pid}
    with open(log_path, "ab") as log:   # exec: plain detached child (tests, or when tmux is unavailable)
        creation = {"creationflags": getattr(subprocess, "DETACHED_PROCESS", 0)} if os.name == "nt" else {"start_new_session": True}
        proc = subprocess.Popen(wrapper, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(job_dir(plan, job.id)), **creation)
    return {"launcher": "exec", "wrapper_pid": proc.pid}


def launch_job(plan: Plan, job: JobSpec, gpu: int, *, threads: int, dry_run: bool = False,
               force: bool = False, log=print) -> dict[str, Any]:
    directory = job_dir(plan, job.id)
    existing = read_state(directory)
    if existing is not None and not force:
        word = job_status_word(existing)
        if word == "running":
            raise ScheduleError(f"job {job.id} is already running (pid {existing.get('pid')}); use stop first")
        if not dry_run:
            raise ScheduleError(f"job {job.id} already has a state ({word}); pass --force to relaunch "
                                "(the runner resumes from its checkpoint)")
    if not job.commit:
        raise ScheduleError(f"job {job.id}: 'commit' (preregistration / protocol-freeze commit) is required")
    check = prereg_check(plan.defaults.repo, job.commit, job.protocol, require_clean=plan.defaults.require_clean)
    if not check["ok"]:
        raise ScheduleError(f"job {job.id} refused:\n  " + "\n  ".join(check["problems"]))
    checkout = job_checkout(plan, job, check["commit"], dry_run=dry_run)
    command = build_command(plan, job, checkout, gpu, threads)
    module_file = Path(command["cwd"]) / (job.module.replace(".", "/") + ".py")
    if not module_file.is_file() and not (dry_run and job.checkout == "worktree"):
        raise ScheduleError(f"job {job.id}: module {job.module} not found at {module_file}")
    inventory = {g["index"]: g for g in gpu_inventory()}
    state: dict[str, Any] = {
        "schema": SCHEMA, "id": job.id, "module": job.module, "args": job.args, "note": job.note,
        "preregistered": bool(job.preregistered), "commit": check["commit"], "commit_ref": job.commit,
        "prereg_check": check, "checkout": str(checkout), "checkout_mode": job.checkout,
        "gpu": inventory.get(gpu, {"index": gpu, "name": None, "uuid": None}),
        "cuda_visible_devices": str(gpu), "cuda_device_order": "PCI_BUS_ID", "blas_threads": threads,
        "results": job.results, "protocol": job.protocol, "expected_ms_per_step": job.expected_ms_per_step,
        "target_transits": job.target_transits, "transit_time_s": job.transit_time_s,
        "command": command["command"], "cwd": command["cwd"], "shell": render_shell(command),
        "planned_utc": utc_now(), "pid": None, "start_utc": None, "end_utc": None, "exit_code": None,
        "warp_probe_mode": plan.defaults.warp_probe,
    }
    if dry_run:
        log(f"[plan] {job.id} -> GPU {gpu} ({state['gpu'].get('name')}) : {state['shell']}")
        return state
    directory.mkdir(parents=True, exist_ok=True)
    (directory / COMMAND_FILE).write_text(json.dumps(command, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    write_state(directory, state)
    wrapper = [command["command"][0], str(HERE / "schedule.py"), "_wrap", str(directory)]
    state.update(detach(plan, job, wrapper, log_path=directory / WRAPPER_LOG))
    state["launched_utc"] = utc_now()
    write_state(directory, state)
    log(f"[launch] {job.id} -> GPU {gpu} ({state['gpu'].get('name')}), {state.get('launcher')} "
        f"{state.get('tmux_session') or state.get('wrapper_pid')}; log {directory / RUN_LOG}")
    return state


# ------------------------------------------------------------------------------------------ wrapper
WARP_PROBE_SOURCE = r"""
import json, sys
import warp as wp
if hasattr(wp, "LOG_WARNING"):
    wp.config.log_level = wp.LOG_WARNING   # no init banner on stdout (config.quiet is deprecated in 1.14)
else:
    wp.config.quiet = True
wp.init()
devices = wp.get_cuda_devices()
out = {"cuda_devices": len(devices)}
if devices:
    d = devices[0]
    out.update({"alias": str(d), "name": d.name, "arch": d.arch, "uuid": getattr(d, "uuid", None),
                "pci_bus_id": getattr(d, "pci_bus_id", None), "total_memory_gib": round(d.total_memory / 2**30, 1)})
print(json.dumps(out))
"""


def warp_probe(python: str, env: Mapping[str, str], cwd: str, timeout_s: float = 120.0) -> dict[str, Any]:
    """Ask Warp, inside the job's environment, which device ``cuda:0`` is."""

    try:
        completed = subprocess.run([python, "-c", WARP_PROBE_SOURCE], env=dict(env), cwd=cwd, capture_output=True,
                                   text=True, timeout=timeout_s, check=False)
        line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
        result = json.loads(line) if line else {"error": completed.stderr.strip()[-2000:]}
        result["returncode"] = completed.returncode
        return result
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}


def uuid_matches(expected: str | None, observed: str | None) -> bool | None:
    if not expected or not observed:
        return None
    return expected.strip().lower() == observed.strip().lower()


def wrap_main(directory: Path) -> int:
    """Run the job's command, recording pid / start / end / exit code and the GPU provenance."""

    state = read_state(directory) or {}
    command = json.loads((directory / COMMAND_FILE).read_text(encoding="utf-8"))
    env = dict(os.environ)
    env.update(command["env"])
    state["wrapper_pid"] = os.getpid()
    mode = state.get("warp_probe_mode", "verify")
    if mode != "off":
        probe = warp_probe(command["command"][0], env, command["cwd"])
        state["warp_probe"] = probe
        match = uuid_matches((state.get("gpu") or {}).get("uuid"), probe.get("uuid"))
        state["gpu_uuid_match"] = match
        if mode == "verify" and (match is False or probe.get("cuda_devices", 0) != 1):
            state["refused"] = (f"Warp sees {probe.get('cuda_devices')} CUDA device(s), uuid {probe.get('uuid')}; "
                                f"nvidia-smi index {state.get('cuda_visible_devices')} is {(state.get('gpu') or {}).get('uuid')}")
            state["end_utc"] = utc_now()
            write_state(directory, state)
            print(f"[wrap] refused: {state['refused']}", flush=True)
            return 3
    with open(directory / RUN_LOG, "ab") as log:
        proc = subprocess.Popen(command["command"], cwd=command["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
        state["pid"] = proc.pid
        state["start_utc"] = utc_now()
        write_state(directory, state)
        print(f"[wrap] started pid {proc.pid}: {' '.join(command['command'])}", flush=True)

        def forward(signum, _frame):   # pragma: no cover - signal path
            with contextlib.suppress(Exception):   # the child may already be gone
                proc.send_signal(signum)

        for name in ("SIGTERM", "SIGINT", "SIGHUP"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), forward)
        exit_code = proc.wait()
    state["exit_code"] = int(exit_code)
    state["end_utc"] = utc_now()
    write_state(directory, state)
    print(f"[wrap] exit {exit_code} at {state['end_utc']}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------ status
def tail_jsonl(path: Path, max_bytes: int = 256 * 1024) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    size = path.stat().st_size
    with open(path, "rb") as handle:
        handle.seek(max(0, size - max_bytes))
        chunk = handle.read()
    lines = chunk.split(b"\n")
    if size > max_bytes:
        lines = lines[1:]     # the first line is probably cut
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def protocol_timing(protocol_path: Path | None) -> dict[str, float | None]:
    """dt and the ion transit time declared by the protocol (budget_* block), if readable."""

    if protocol_path is None or not protocol_path.is_file():
        return {"dt_s": None, "ion_transit_time_s": None, "wall_budget_seconds": None}
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"dt_s": None, "ion_transit_time_s": None, "wall_budget_seconds": None}
    budget_keys = [k for k in protocol if k.startswith("budget")]
    budget = protocol[budget_keys[0]] if budget_keys else {}
    return {
        "dt_s": float(protocol.get("numerics", {}).get("dt_s")) if protocol.get("numerics", {}).get("dt_s") else None,
        "ion_transit_time_s": float(budget["ion_transit_time_s"]) if budget.get("ion_transit_time_s") else None,
        "wall_budget_seconds": (float(protocol["stopping_rule"]["wall_budget_seconds"])
                                if protocol.get("stopping_rule", {}).get("wall_budget_seconds") else None),
    }


def eta_seconds(*, time_s: float | None, dt_s: float | None, transit_s: float | None, target_transits: float,
                ms_per_step: float | None) -> float | None:
    if None in (time_s, dt_s, transit_s, ms_per_step) or not dt_s or not ms_per_step:
        return None
    remaining_steps = math.ceil(max(target_transits * transit_s - time_s, 0.0) / dt_s)
    return remaining_steps * ms_per_step / 1e3


def wall_budget_from_args(args: list[str]) -> float | None:
    for i, arg in enumerate(args):
        if arg == "--wall-budget-seconds" and i + 1 < len(args):
            try:
                return float(args[i + 1])
            except ValueError:
                return None
        if arg.startswith("--wall-budget-seconds="):
            try:
                return float(arg.split("=", 1)[1])
            except ValueError:
                return None
    return None


def job_progress(state: Mapping[str, Any], *, results_dir: Path | None, protocol_path: Path | None,
                 recent: int = 20) -> dict[str, Any]:
    """Progress of one job from its runner artifacts (``run_state.json``, ``status.jsonl``)."""

    timing = protocol_timing(protocol_path)
    transit = state.get("transit_time_s") or timing["ion_transit_time_s"]
    target = float(state.get("target_transits") or 3.0)
    progress: dict[str, Any] = {"steps": None, "time_s": None, "ms_per_step": None, "transits": None,
                                "eta_target_s": None, "stop_reason": None, "finished": None, "wall_seconds_total": None,
                                "electrons": None, "wall_budget_seconds": wall_budget_from_args(list(state.get("args") or []))
                                or timing["wall_budget_seconds"], "target_transits": target}
    if results_dir is None:
        return progress
    run_state_path = results_dir / "run_state.json"
    if run_state_path.is_file():
        try:
            run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
            progress["finished"] = run_state.get("finished")
            progress["stop_reason"] = run_state.get("stop_reason")
            progress["wall_seconds_total"] = run_state.get("wall_seconds_total")
            progress["checkpoint_step"] = run_state.get("checkpoint_step")
            progress["steps"] = run_state.get("checkpoint_step")
            progress["time_s"] = run_state.get("checkpoint_time_s")
        except (OSError, json.JSONDecodeError):
            pass
    rows = [r for r in tail_jsonl(results_dir / "status.jsonl") if r.get("step") is not None]
    if rows:
        last = rows[-1]
        progress["steps"] = last.get("step", progress["steps"])
        progress["time_s"] = last.get("time_s", progress["time_s"])
        progress["electrons"] = last.get("electrons")
        ms = [float(r["ms_per_step"]) for r in rows[-recent:] if r.get("ms_per_step") is not None]
        if ms:
            ms.sort()
            progress["ms_per_step"] = ms[len(ms) // 2]
        if last.get("wall_seconds_total") is not None:
            progress["wall_seconds_total"] = last["wall_seconds_total"]
    if progress["ms_per_step"] is None and state.get("expected_ms_per_step"):
        progress["ms_per_step"] = float(state["expected_ms_per_step"])
        progress["ms_per_step_source"] = "expected (no records yet)"
    if progress["time_s"] is not None and transit:
        progress["transits"] = float(progress["time_s"]) / float(transit)
    progress["eta_target_s"] = eta_seconds(time_s=progress["time_s"] if progress["time_s"] is not None else (0.0 if progress["ms_per_step"] else None),
                                          dt_s=timing["dt_s"], transit_s=transit, target_transits=target,
                                          ms_per_step=progress["ms_per_step"])
    if progress["wall_budget_seconds"] and progress["wall_seconds_total"] is not None:
        progress["budget_remaining_s"] = float(progress["wall_budget_seconds"]) - float(progress["wall_seconds_total"])
    return progress


def collect_status(plan: Plan, *, live_gpu: bool = True) -> list[dict[str, Any]]:
    rows = []
    specs = {j.id: j for j in plan.jobs}
    directories: list[Path] = []
    if plan.defaults.jobs_dir.is_dir():
        directories = sorted(p for p in plan.defaults.jobs_dir.iterdir() if p.is_dir())
    seen = set()
    for directory in directories:
        state = read_state(directory)
        if state is None:
            continue
        seen.add(state.get("id", directory.name))
        rows.append(status_row(plan, state, specs.get(state.get("id", directory.name)), live_gpu=live_gpu))
    for job in plan.jobs:
        if job.id not in seen:
            rows.append({"id": job.id, "status": "not launched" if job.enabled else "disabled", "gpu": job.gpu,
                         "module": job.module, "preregistered": job.preregistered, "progress": {}})
    return rows


def status_row(plan: Plan, state: Mapping[str, Any], spec: JobSpec | None, *, live_gpu: bool) -> dict[str, Any]:
    checkout = Path(state.get("checkout") or plan.defaults.repo)
    results = state.get("results") or (spec.results if spec else None)
    protocol = state.get("protocol") or (spec.protocol if spec else None)
    progress = job_progress(state, results_dir=(checkout / results) if results else None,
                            protocol_path=(checkout / protocol) if protocol else None)
    gpu = state.get("gpu") or {}
    row = {"id": state.get("id"), "status": job_status_word(state), "gpu": gpu.get("index"), "gpu_name": gpu.get("name"),
           "pid": state.get("pid"), "module": state.get("module"), "preregistered": state.get("preregistered"),
           "start_utc": state.get("start_utc"), "end_utc": state.get("end_utc"), "exit_code": state.get("exit_code"),
           "gpu_uuid_match": state.get("gpu_uuid_match"), "progress": progress}
    if live_gpu and gpu.get("index") is not None and row["status"] == "running":
        row["gpu_live"] = gpu_live(int(gpu["index"]))
    return row


def _hours(seconds: float | None) -> str:
    return "-" if seconds is None else f"{seconds / 3600:.1f} h"


def format_status(rows: list[dict[str, Any]]) -> str:
    header = ["id", "status", "gpu", "pid", "steps", "t (us)", "transits", "ms/step", "ETA target", "budget left",
              "stop reason", "prereg"]
    table = [header]
    for r in rows:
        p = r.get("progress") or {}
        table.append([
            str(r.get("id")), str(r.get("status")), str(r.get("gpu")), str(r.get("pid") or "-"),
            f"{p['steps']:,}" if p.get("steps") is not None else "-",
            f"{p['time_s'] * 1e6:.3f}" if p.get("time_s") is not None else "-",
            f"{p['transits']:.2f}/{p.get('target_transits', 3):g}" if p.get("transits") is not None else "-",
            f"{p['ms_per_step']:.2f}" + ("*" if p.get("ms_per_step_source") else "") if p.get("ms_per_step") else "-",
            _hours(p.get("eta_target_s")), _hours(p.get("budget_remaining_s")),
            str(p.get("stop_reason") or ("running" if r.get("status") == "running" else "-")),
            "yes" if r.get("preregistered") else "no",
        ])
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in table]
    lines.insert(1, "  ".join("-" * w for w in widths))
    lines.append("")
    lines.append("* = expected ms/step from jobs.yaml (no series records yet); ETA = steps to target_transits x ms/step")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------ commands
def load_plan(args: argparse.Namespace) -> Plan:
    path = Path(args.jobs).resolve()
    return build_plan(load_jobs_file(path), source=path)


def cmd_gpus(args: argparse.Namespace) -> int:
    inventory = gpu_inventory()
    if args.json:
        print(json.dumps(inventory, indent=1))
        return 0
    for g in inventory:
        live = gpu_live(g["index"])
        print(f"GPU {g['index']}: {g['name']} {g['uuid']} {g['memory_total_mib'] / 1024:.0f} GiB driver {g['driver_version']}"
              + (f" util {live.get('utilisation_percent'):.0f}% used {live.get('memory_used_mib'):.0f} MiB" if live else ""))
    if not inventory:
        print("nvidia-smi reported no GPUs")
    return 0


def selected_jobs(plan: Plan, only: list[str] | None) -> list[JobSpec]:
    jobs = [j for j in plan.jobs if j.enabled]
    if only:
        unknown = sorted(set(only) - {j.id for j in plan.jobs})
        if unknown:
            raise ScheduleError(f"unknown job id(s) {unknown}")
        jobs = [j for j in plan.jobs if j.id in set(only)]
    return jobs


def cmd_plan(args: argparse.Namespace) -> int:
    plan = load_plan(args)
    jobs = selected_jobs(plan, args.only)
    busy = occupied_slots(plan)
    pending = [j for j in jobs if job_status_word(read_state(job_dir(plan, j.id))) in ("not launched",)]
    assignment = assign_gpus(plan, pending, busy)
    threads = blas_threads_for(plan, len(plan.gpus) * plan.defaults.slots_per_gpu)
    print(f"repo {plan.defaults.repo} @ {resolve_commit(plan.defaults.repo, 'HEAD')[:12]}; GPUs {plan.gpus} x "
          f"{plan.defaults.slots_per_gpu} slot(s); jobs dir {plan.defaults.jobs_dir}; {threads} BLAS threads/job")
    problems = 0
    for job in pending:
        try:
            launch_job(plan, job, assignment[job.id], threads=threads, dry_run=True)
        except ScheduleError as error:
            problems += 1
            print(f"[plan] {job.id}: REFUSED - {error}")
    for job in jobs:
        if job not in pending:
            print(f"[plan] {job.id}: {job_status_word(read_state(job_dir(plan, job.id)))} (not re-planned)")
    return 1 if problems else 0


def cmd_launch(args: argparse.Namespace) -> int:
    plan = load_plan(args)
    jobs = selected_jobs(plan, args.only)
    busy = occupied_slots(plan)
    if not args.force:
        jobs = [j for j in jobs if job_status_word(read_state(job_dir(plan, j.id))) == "not launched"]
    assignment = assign_gpus(plan, jobs, busy)
    threads = blas_threads_for(plan, len(plan.gpus) * plan.defaults.slots_per_gpu)
    failures = 0
    for job in jobs:
        try:
            launch_job(plan, job, assignment[job.id], threads=threads, dry_run=args.dry_run, force=args.force)
        except ScheduleError as error:
            failures += 1
            print(f"[launch] {job.id}: REFUSED - {error}")
    return 1 if failures else 0


def cmd_status(args: argparse.Namespace) -> int:
    plan = load_plan(args)
    while True:
        rows = collect_status(plan, live_gpu=not args.no_gpu)
        if args.json:
            print(json.dumps(rows, indent=1, default=str))
        else:
            print(f"{utc_now()}  jobs dir {plan.defaults.jobs_dir}")
            print(format_status(rows))
        if not args.watch:
            return 0
        time.sleep(float(args.watch))


def cmd_stop(args: argparse.Namespace) -> int:
    plan = load_plan(args)
    state = read_state(job_dir(plan, args.id))
    if state is None:
        raise ScheduleError(f"job {args.id} has no state")
    pid = state.get("pid")
    if not pid_alive(pid):
        print(f"job {args.id}: pid {pid} is not running ({job_status_word(state)})")
        return 1
    sig = getattr(signal, f"SIG{args.signal.upper()}", signal.SIGTERM)
    os.kill(int(pid), sig)
    print(f"sent {args.signal.upper()} to pid {pid} of job {args.id}; the runner resumes from its last checkpoint "
          "(checkpoint_every_steps) when relaunched with --force")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS_FILE), help="jobs.yaml (or .json)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("gpus", help="nvidia-smi inventory")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_gpus)
    p = sub.add_parser("plan", help="validate, check preregistration commits, show the GPU assignment")
    p.add_argument("--only", nargs="*", default=None)
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("launch", help="launch every enabled, not-yet-launched job")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="relaunch a finished/failed/lost job (the runner resumes)")
    p.set_defaults(func=cmd_launch)
    p = sub.add_parser("status", help="tail run_state.json / status.jsonl of every job")
    p.add_argument("--json", action="store_true")
    p.add_argument("--watch", type=float, default=None, help="repeat every N seconds")
    p.add_argument("--no-gpu", action="store_true", help="skip the live nvidia-smi columns")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("stop", help="signal a running job (default TERM)")
    p.add_argument("id")
    p.add_argument("--signal", default="TERM")
    p.set_defaults(func=cmd_stop)
    p = sub.add_parser("_wrap", help=argparse.SUPPRESS)
    p.add_argument("directory")
    p.set_defaults(func=lambda a: wrap_main(Path(a.directory)))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ScheduleError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
