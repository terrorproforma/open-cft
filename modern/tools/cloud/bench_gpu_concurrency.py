"""Per-GPU concurrency benchmark for the PIC-MCC step loop (Lambda 8x H100 sizing).

Question answered: on ONE GPU, how many PIC processes should share it?  The step is partly
launch-bound (~2 (n_r + 1) sequential block-Thomas launches per Poisson solve, ~480 per step on the
plume grid) and partly bandwidth-bound (inverse-block reads, particle push), so several processes per
GPU can raise the aggregate steps/s.  For N = 1, 2, 4 concurrent processes pinned to the same GPU
(``CUDA_VISIBLE_DEVICES``) the benchmark runs the SAME construction path as the production runner
(``experiments.pic2d_cft_steady_state_v1.run``: ``build_config`` -> ``load_inputs`` -> ``Simulation``)
on a frozen protocol, synchronises the processes on a file barrier and then times a fixed number of
production steps with the preregistered v4 preflight's own timing function
(``experiments.pic2d_cft_steady_state_v4.run._time_steps``: warm-up on the live simulation, window
accumulation on from the first step, wall ms/step over the timed steps) - the function that produced
the recorded 2.54 / 4.36 ms/step 5090 numbers, so the comparison is like-for-like.

Reported per (configuration, N): ms/step per process, aggregate steps/s on the GPU, GPU memory per
process (nvidia-smi compute-apps sample maximum and Warp's mempool high-water mark), host
factorisation / setup seconds, and the speedup against the local RTX 5090 anchors recorded in this
repository (see ``CONFIGS[...].anchor``).  ``factorise`` measures the host
Poisson factorisation (``WarpBlockThomas.__init__`` on the CPU device: ``np.linalg.inv`` per radial
row block, BLAS-bound) for N concurrent processes with the BLAS thread count pinned per process -
the local lesson was that two unpinned factorisations oversubscribed the threads and one took 20 min.

Nothing here modifies a protocol file or the ``cft_revival`` package: configuration variants
(33 / 25 um, production particle load) are in-memory overrides of the frozen protocol dictionaries.

Usage (from ``modern/`` with the venv active; see ``bench.sh``)::

    python -m tools.cloud.bench_gpu_concurrency run --gpu 0 --concurrency 1 2 4 \
        --configs channel-50um plume-v2.0-50um --out /work/bench/h100.json
    python -m tools.cloud.bench_gpu_concurrency factorise --concurrency 1 4 8 --configs plume-v2.1-50um
    python -m tools.cloud.bench_gpu_concurrency report /work/bench/h100.json --markdown /work/bench/h100.md
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODERN = Path(__file__).resolve().parents[2]
REPOSITORY = MODERN.parent
BLAS_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
SCHEMA = "cft-revival.tools.cloud.bench-gpu-concurrency/0.1.0"
LOCAL_ANCHOR_GPU = "NVIDIA GeForce RTX 5090 (driver 595.97, Warp 1.14.0 CUDA 12.9 build, Windows 11)"


# --------------------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class Anchor:
    """A recorded (or predicted) RTX 5090 ms/step for one configuration, with its provenance."""

    seed_load_ms_per_step: float | None
    seed_load_note: str
    production_ms_per_step: float | None
    production_note: str
    predicted: bool = False


@dataclass(frozen=True)
class BenchConfig:
    key: str
    protocol: str                       # protocol.json relative to modern/ (frozen; never written)
    description: str
    overrides: Mapping[str, Any]        # dotted-path in-memory overrides of the protocol dictionary
    production_seed_density_per_m3: float  # seed density giving ~the production macro-particle count
    anchor: Anchor
    default: bool = True


CHANNEL_PROTOCOL = "experiments/pic2d_cft_steady_state_v2/protocol.json"
CHANNEL_V4_PROTOCOL = "experiments/pic2d_cft_steady_state_v4/protocol.json"
PLUME_V20_PROTOCOL = "experiments/pic2d_cft_plume_v1/protocol.json"
PLUME_V21_PROTOCOL = "experiments/pic2d_cft_plume_v2_1/protocol.json"
# the v4 preflight's synthetic plateau load: 2 x 1.75e17 x 3.44e-7 m^3 / W macro-particles in the channel
PLATEAU_SEED_DENSITY = 1.75e17

CONFIGS: dict[str, BenchConfig] = {
    "channel-50um": BenchConfig(
        key="channel-50um", protocol=CHANNEL_PROTOCOL,
        description="accepted channel-only plateau configuration (steady-state v2 base, model v1.3, "
                    "3 x 24 mm, 60 x 480 = 50 um, dt 1.5 ps, W 6e4)",
        overrides={},
        production_seed_density_per_m3=PLATEAU_SEED_DENSITY,   # -> 2.0 M macro-particles = the base plateau load
        anchor=Anchor(
            seed_load_ms_per_step=2.26,
            seed_load_note="steady_state_v2/results/status.jsonl, median of steps 400-2400 (~0.26 M e-)",
            production_ms_per_step=1.98,
            production_note="steady_state_v2/results/summary.json ms_per_step_this_session (5.12 M steps, "
                            "~1.0 M e- at the plateau; model v1.3 without the CUDA-graph step)",
        ),
    ),
    "channel-33um": BenchConfig(
        key="channel-33um", protocol=CHANNEL_V4_PROTOCOL,
        description="the preregistered steady-state v4 grid refinement (392129e5): channel-only 90 x 720 = "
                    "33.3 um, dt 1.4 ps, W 2.667e4 = 6e4/2.25 (particles per cell as the base), v2.0.3 gates",
        overrides={},
        production_seed_density_per_m3=PLATEAU_SEED_DENSITY,   # 4.5 M macro-particles (the v4 preflight load)
        anchor=Anchor(
            seed_load_ms_per_step=2.54,
            seed_load_note="steady_state_v4/preflight.json timing_seed_load: 2.539 ms/step over 2000 steps after "
                           "200 warm-up (1.3 M seed macro-particles; measured 2026-09-04)",
            production_ms_per_step=4.36,
            production_note="steady_state_v4/preflight.json timing_plateau_load: 4.360 ms/step at 4.5 M "
                            "macro-particles (0.565 ms per M) -> 6.2 h to 3 transits; measured",
        ),
    ),
    "channel-25um": BenchConfig(
        key="channel-25um", protocol=CHANNEL_V4_PROTOCOL,
        description="channel-only 120 x 960 = 25 um, dt 1.0 ps, W 1.5e4 = 6e4/4 (particles per cell as the base; "
                    "8 M at the plateau) - the third point of the 50/33/25 Delta-convergence ladder (v4 gates)",
        overrides={"case.radial_cells": 120, "case.axial_cells": 960, "numerics.dt_s": 1.0e-12,
                   "case.macro_weight": 60000.0 / 4.0},
        production_seed_density_per_m3=PLATEAU_SEED_DENSITY,   # ~8 M macro-particles
        anchor=Anchor(
            seed_load_ms_per_step=None, seed_load_note="not recorded",
            production_ms_per_step=17.3,
            production_note="plume v1 README resolution decision: 17.3 ms/step at 25 um (23.1 h to 3 transits at "
                            "1.5 ps); the v4-calibrated model (fixed ~2.9 ms + 0.565 ms/M x 8 M) gives ~7.4 ms; "
                            "predicted, not measured",
            predicted=True,
        ),
        default=False,
    ),
    "plume-v2.0-50um": BenchConfig(
        key="plume-v2.0-50um", protocol=PLUME_V20_PROTOCOL,
        description="plume v2.0.3 configuration (channel + 12 x 12 mm box, 240 x 720 = 50 um, dt 1.5 ps, "
                    "W 6e4, flux-tube cathode; the step cost is that of attempts 7-8)",
        overrides={},
        production_seed_density_per_m3=7.6e17,   # channel seed -> ~4.4 M e- = the attempt-8 load
        anchor=Anchor(
            seed_load_ms_per_step=4.26,
            seed_load_note="plume_v1/results-attempt7-wall-budget-no-plateau/status.jsonl, median of steps "
                           "400-2400 (~0.26 M e-)",
            production_ms_per_step=7.08,
            production_note="attempt 8 resume: 7.0-7.15 ms/step at ~4.4 M particles (README launch log)",
        ),
    ),
    "plume-v2.1-50um": BenchConfig(
        key="plume-v2.1-50um", protocol=PLUME_V21_PROTOCOL,
        description="plume v2.1 configuration (channel + 24 x 12 mm box, 240 x 960 = 50 um, dt 1.5 ps)",
        overrides={},
        production_seed_density_per_m3=7.6e17,
        anchor=Anchor(
            seed_load_ms_per_step=None, seed_load_note="not recorded (never launched)",
            production_ms_per_step=8.2,
            production_note="v2.1 README cost table: 8.2 ms/step (+16 % over v2.0), 17.4 h to 3 transits; "
                            "predicted, not measured",
            predicted=True,
        ),
        default=False,
    ),
    "plume-v2.1-33um": BenchConfig(
        key="plume-v2.1-33um", protocol=PLUME_V21_PROTOCOL,
        description="plume v2.1 box at 360 x 1440 = 33.3 um, dt 1.4 ps (the resolved plume run; ~61 min host "
                    "factorisation, 6.0 GB of inverse blocks)",
        overrides={"case.radial_cells": 360, "case.axial_cells": 1440, "numerics.dt_s": 1.4e-12,
                   "case.macro_weight": 60000.0 / 2.25},
        production_seed_density_per_m3=7.6e17,
        anchor=Anchor(
            seed_load_ms_per_step=None, seed_load_note="not recorded",
            production_ms_per_step=22.4,
            production_note="plume v1 README resolution decision: 22.4 ms/step -> 47.5 h (50.9 h at 1.4 ps); "
                            "predicted, not measured",
            predicted=True,
        ),
        default=False,
    ),
}


def default_config_keys() -> list[str]:
    return [key for key, config in CONFIGS.items() if config.default]


def apply_overrides(protocol: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy ``protocol`` and set every ``a.b.c`` key of ``overrides`` (in memory only)."""

    result = copy.deepcopy(dict(protocol))
    for dotted, value in overrides.items():
        node = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def parse_override(text: str) -> tuple[str, Any]:
    """``--override case.radial_cells=90`` -> ("case.radial_cells", 90) (JSON value, string fallback)."""

    key, _, raw = text.partition("=")
    if not key or not raw:
        raise argparse.ArgumentTypeError(f"override must be KEY=VALUE, got {text!r}")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def blas_threads_per_process(process_count: int, cpu_count: int | None = None, cap: int = 16) -> int:
    """Threads each concurrent process may use so that N processes never oversubscribe the cores."""

    cores = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    return max(1, min(cap, cores // max(int(process_count), 1)))


def blas_environment(threads: int) -> dict[str, str]:
    return {name: str(int(threads)) for name in BLAS_THREAD_VARIABLES}


# --------------------------------------------------------------------------------------- nvidia-smi
def nvidia_smi(args: list[str], timeout_s: float = 10.0) -> list[list[str]]:
    """Rows of a ``--format=csv,noheader,nounits`` query, or ``[]`` when nvidia-smi is unavailable."""

    try:
        completed = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, check=True, timeout=timeout_s)
    except Exception:  # noqa: BLE001 - absent binary, timeout, non-zero exit: telemetry is optional
        return []
    return [[cell.strip() for cell in line.split(",")] for line in completed.stdout.splitlines() if line.strip()]


def gpu_inventory() -> list[dict[str, Any]]:
    rows = nvidia_smi(["--query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id"])
    inventory = []
    for row in rows:
        if len(row) < 6:
            continue
        inventory.append({"index": int(row[0]), "name": row[1], "uuid": row[2], "driver_version": row[3],
                          "memory_total_mib": float(row[4]), "pci_bus_id": row[5]})
    return inventory


def sample_compute_apps(gpu_index: int) -> tuple[dict[int, float], float | None]:
    """(used MiB per pid on ``gpu_index``, GPU utilisation percent) from one nvidia-smi call each."""

    used: dict[int, float] = {}
    for row in nvidia_smi(["--query-compute-apps=pid,used_memory", f"--id={gpu_index}"]):
        if len(row) >= 2:
            try:
                used[int(row[0])] = float(row[1])
            except ValueError:
                continue
    utilisation = None
    rows = nvidia_smi(["--query-gpu=utilization.gpu", f"--id={gpu_index}"])
    if rows and rows[0]:
        try:
            utilisation = float(rows[0][0])
        except ValueError:
            utilisation = None
    return used, utilisation


# --------------------------------------------------------------------------------------- barrier
def wait_for(path: Path, timeout_s: float, poll_s: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(poll_s)
    return False


# --------------------------------------------------------------------------------------- worker (GPU)
def _import_runner():
    for entry in (str(MODERN / "src"), str(MODERN)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from experiments.pic2d_cft_steady_state_v1 import run as runner

    return runner


def _time_steps_fallback(sim: Any, steps: int, *, warmup: int) -> dict[str, float]:
    """Same body as ``pic2d_cft_steady_state_v4.run._time_steps`` (used only if that module is absent)."""

    start = sim.backend.step_index
    sim.run(warmup, accumulate_from_step=start)
    t0 = time.perf_counter()
    sim.run(steps, accumulate_from_step=start)
    elapsed = time.perf_counter() - t0
    return {"steps": steps, "seconds": elapsed, "ms_per_step": 1e3 * elapsed / steps, "accumulation": True}


def timing_function() -> tuple[Callable[..., dict[str, float]], str]:
    """The preregistered v4 preflight's ``_time_steps`` (the function behind the 5090 anchors), else the fallback."""

    try:
        from experiments.pic2d_cft_steady_state_v4.run import _time_steps

        return _time_steps, "experiments.pic2d_cft_steady_state_v4.run._time_steps"
    except ImportError:
        return _time_steps_fallback, "tools.cloud.bench_gpu_concurrency._time_steps_fallback"


def load_bench_protocol(config: BenchConfig, extra_overrides: Mapping[str, Any] | None = None,
                        load: str = "seed") -> dict[str, Any]:
    protocol = json.loads((MODERN / config.protocol).read_text(encoding="utf-8"))
    overrides = dict(config.overrides)
    if load == "production":
        overrides["operating_point.seed_plasma_density_per_m3"] = config.production_seed_density_per_m3
    if extra_overrides:
        overrides.update(extra_overrides)
    return apply_overrides(protocol, overrides)


def worker_main(args: argparse.Namespace) -> int:
    """One benchmark process: construct like the runner, warm up, barrier, time ``args.steps`` steps."""

    config = CONFIGS[args.config]
    barrier = Path(args.barrier)
    out = Path(args.out)
    record: dict[str, Any] = {
        "config": config.key, "index": int(args.index), "count": int(args.count), "pid": os.getpid(),
        "load": args.load, "backend": args.backend, "warmup_steps": int(args.warmup), "steps": int(args.steps),
        "env": {name: os.environ.get(name) for name in (*BLAS_THREAD_VARIABLES, "CUDA_VISIBLE_DEVICES",
                                                          "CUDA_DEVICE_ORDER")},
    }
    try:
        import numpy as np

        runner = _import_runner()
        from cft_revival.pic2d.simulation import Simulation

        protocol = load_bench_protocol(config, dict(args.override or []), load=args.load)
        record["protocol_case"] = protocol["case"]
        record["dt_s"] = protocol["numerics"]["dt_s"]
        t0 = time.perf_counter()
        pic_config = runner.build_config(protocol, backend=args.backend)
        field_map, cross_sections = runner.load_inputs(pic_config, None, None, protocol=protocol)
        record["inputs_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        sim = Simulation(pic_config, field_map, cross_sections=cross_sections, backend=args.backend,
                         step_graph=runner.step_graph_flag(protocol))
        record["simulation_construct_s"] = time.perf_counter() - t1   # host factorisation + device upload
        record["cells"] = list(pic_config.grid.cell_shape)
        record["unknowns"] = int(sim.masks.unknown_count)
        record["plasma_cells"] = int(sim.masks.to_dict()["plasma_cells"])
        state = sim.state
        record["particles_initial"] = {"electrons": int(state.electrons.count), "ions": int(state.ions.count)}
        record["backend_name"] = sim.to_provenance().get("backend")
        time_steps, record["timing_function"] = timing_function()
        # pre-warm OUTSIDE the timed window so the barrier is reached with the Warp module loaded and the
        # production-variant CUDA graph captured (accumulation on, as in every production step); the timing
        # function then runs its own warm-up + timed steps exactly as the v4 preflight did on the 5090
        t2 = time.perf_counter()
        sim.run(int(args.warmup), accumulate_from_step=sim.backend.step_index)
        record["prewarm_s"] = time.perf_counter() - t2
        record["step_graph"] = sim.step_graph_state()
        # barrier: every process of this round must be stepping at the same time
        (barrier / f"ready-{args.index}").write_text(str(os.getpid()), encoding="utf-8")
        if not wait_for(barrier / "go", timeout_s=float(args.barrier_timeout)):
            raise TimeoutError("barrier 'go' never appeared")
        (barrier / f"measuring-{args.index}").write_text(str(os.getpid()), encoding="utf-8")
        timing = time_steps(sim, int(args.steps), warmup=200)
        record["timing"] = timing
        record["measure_s"] = float(timing["seconds"])
        record["ms_per_step_wall"] = float(timing["ms_per_step"])
        state = sim.state
        record["particles_final"] = {"electrons": int(state.electrons.count), "ions": int(state.ions.count)}
        record["gpu_memory"] = _warp_memory_report(sim)
        record["numpy"] = np.__version__
        record["ok"] = True
    except Exception as error:  # noqa: BLE001 - the parent reports the failure per process
        record["ok"] = False
        record["error"] = f"{type(error).__name__}: {error}"
    out.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return 0 if record["ok"] else 1


def _warp_memory_report(sim: Any) -> dict[str, Any]:
    report: dict[str, Any] = {}
    try:
        import warp as wp

        device = getattr(sim.backend, "device", None)
        if device is None or not getattr(device, "is_cuda", False):
            return report
        wp.synchronize_device(device)
        for name in ("get_mempool_used_mem_current", "get_mempool_used_mem_high"):
            fn = getattr(wp, name, None)
            if fn is not None:
                report[name.replace("get_", "")] = int(fn(device))
        report["device_total_bytes"] = int(device.total_memory)
        report["device_free_bytes"] = int(device.free_memory)      # device-wide (all processes)
        report["device_name"] = device.name
        report["device_uuid"] = getattr(device, "uuid", None)
    except Exception as error:  # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def quiet_warp(wp: Any) -> None:
    """Suppress the Warp init banner (``config.quiet`` is deprecated since 1.14 in favour of ``log_level``)."""

    if hasattr(wp, "LOG_WARNING"):
        wp.config.log_level = wp.LOG_WARNING
    else:  # pragma: no cover - older Warp
        wp.config.quiet = True


# --------------------------------------------------------------------------------------- worker (CPU)
def factorise_worker_main(args: argparse.Namespace) -> int:
    """Time the host block-Thomas factorisation (``WarpBlockThomas.__init__`` on the CPU device)."""

    config = CONFIGS[args.config]
    out = Path(args.out)
    record: dict[str, Any] = {"config": config.key, "index": int(args.index), "count": int(args.count),
                              "pid": os.getpid(),
                              "env": {name: os.environ.get(name) for name in BLAS_THREAD_VARIABLES}}
    try:
        import numpy as np

        runner = _import_runner()
        import warp as wp

        from cft_revival.pic2d.mesh import build_mesh_masks
        from cft_revival.pic2d.warp_backend import WarpBlockThomas

        quiet_warp(wp)
        wp.init()
        protocol = load_bench_protocol(config, dict(args.override or []))
        pic_config = runner.build_config(protocol, backend="warp-cuda")   # device-direct = the row-block path
        t0 = time.perf_counter()
        masks = build_mesh_masks(pic_config.grid)
        record["masks_s"] = time.perf_counter() - t0
        record["cells"] = list(pic_config.grid.cell_shape)
        barrier = Path(args.barrier) if args.barrier else None
        if barrier is not None:
            (barrier / f"ready-{args.index}").write_text(str(os.getpid()), encoding="utf-8")
            if not wait_for(barrier / "go", timeout_s=float(args.barrier_timeout)):
                raise TimeoutError("barrier 'go' never appeared")
        t1 = time.perf_counter()
        solver = WarpBlockThomas(masks, pic_config.potentials, pic_config.poisson, wp.get_device("cpu"),
                                 use_graph=False)
        record["factorisation_s"] = time.perf_counter() - t1
        record["inverse_blocks_bytes"] = int(solver.host_memory_bytes)
        record["block_size"] = int(solver.m)
        record["blocks"] = int(solver.nr + 1)
        record["numpy"] = np.__version__
        record["ok"] = True
    except Exception as error:  # noqa: BLE001
        record["ok"] = False
        record["error"] = f"{type(error).__name__}: {error}"
    out.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return 0 if record["ok"] else 1


# --------------------------------------------------------------------------------------- orchestration
def _spawn(command: list[str], env: Mapping[str, str], log_path: Path) -> subprocess.Popen:
    log = open(log_path, "w", encoding="utf-8")   # noqa: SIM115 - closed by the OS with the child
    return subprocess.Popen(command, cwd=str(MODERN), env=dict(env), stdout=log, stderr=subprocess.STDOUT)


def run_round(config: BenchConfig, count: int, *, gpu: int, warmup: int, steps: int, load: str,
              overrides: list[tuple[str, Any]], scratch: Path, python: str, barrier_timeout: float,
              threads_cap: int, sample_interval_s: float = 2.0, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Spawn ``count`` workers on ``gpu``, release the barrier when all are warm, sample nvidia-smi."""

    round_dir = scratch / f"{config.key}-N{count}"
    round_dir.mkdir(parents=True, exist_ok=True)
    threads = blas_threads_per_process(count, cap=threads_cap)
    env = dict(os.environ)
    env.update(blas_environment(threads))
    env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": f"{MODERN / 'src'}{os.pathsep}{MODERN}"})
    procs: list[subprocess.Popen] = []
    for k in range(count):
        command = [python, "-m", "tools.cloud.bench_gpu_concurrency", "worker", "--config", config.key,
                   "--index", str(k), "--count", str(count), "--warmup", str(warmup), "--steps", str(steps),
                   "--load", load, "--barrier", str(round_dir), "--out", str(round_dir / f"worker-{k}.json"),
                   "--barrier-timeout", str(barrier_timeout)]
        for key, value in overrides:
            command += ["--override", f"{key}={json.dumps(value)}"]
        procs.append(_spawn(command, env, round_dir / f"worker-{k}.log"))
    log(f"[bench] {config.key} N={count}: spawned {count} worker(s) on GPU {gpu}, {threads} BLAS threads each")
    # wait for every worker to finish its warm-up (or die)
    t_start = time.monotonic()
    while True:
        ready = sum((round_dir / f"ready-{k}").exists() for k in range(count))
        alive = [p for p in procs if p.poll() is None]
        if ready == count or not alive:
            break
        if time.monotonic() - t_start > barrier_timeout:
            break
        time.sleep(0.5)
    (round_dir / "go").write_text("go", encoding="utf-8")
    # sample nvidia-smi while the measurement runs
    memory_max: dict[int, float] = {}
    utilisation: list[float] = []
    while any(p.poll() is None for p in procs):
        used, util = sample_compute_apps(gpu)
        for pid, mib in used.items():
            memory_max[pid] = max(memory_max.get(pid, 0.0), mib)
        if util is not None:
            utilisation.append(util)
        time.sleep(sample_interval_s)
    workers = []
    for k, proc in enumerate(procs):
        path = round_dir / f"worker-{k}.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            record = {"config": config.key, "index": k, "count": count, "ok": False,
                      "error": f"no result file (exit code {proc.returncode})"}
        record["exit_code"] = proc.returncode
        record["nvidia_smi_used_mib_max"] = memory_max.get(int(record.get("pid", -1)))
        workers.append(record)
    return summarise_round(config, count, gpu, threads, workers, utilisation)


def summarise_round(config: BenchConfig, count: int, gpu: int, threads: int, workers: list[dict[str, Any]],
                    utilisation: list[float]) -> dict[str, Any]:
    ok = [w for w in workers if w.get("ok")]
    ms_wall = [float(w["ms_per_step_wall"]) for w in ok]
    aggregate = sum(1e3 / ms for ms in ms_wall) if ms_wall else None
    measure = [float(w["measure_s"]) for w in ok]
    return {
        "config": config.key, "concurrency": count, "gpu_index": gpu, "blas_threads_per_process": threads,
        "processes_ok": len(ok), "processes_failed": len(workers) - len(ok),
        "ms_per_step_wall_per_process": ms_wall,
        "ms_per_step_wall_mean": (sum(ms_wall) / len(ms_wall)) if ms_wall else None,
        "aggregate_steps_per_s": aggregate,
        "overlap_fraction": (min(measure) / max(measure)) if measure and max(measure) > 0 else None,
        "timing_function": sorted({str(w.get("timing_function")) for w in ok}),
        "setup_s_per_process": [float(w.get("simulation_construct_s", float("nan"))) for w in ok],
        "inputs_s_per_process": [float(w.get("inputs_s", float("nan"))) for w in ok],
        "prewarm_s_per_process": [float(w.get("prewarm_s", float("nan"))) for w in ok],
        "gpu_memory_mib_per_process_nvidia_smi": [w.get("nvidia_smi_used_mib_max") for w in ok],
        "gpu_memory_mib_per_process_warp_high": [
            (w.get("gpu_memory", {}).get("mempool_used_mem_high") or 0) / 2**20 if w.get("gpu_memory") else None
            for w in ok],
        "gpu_utilisation_percent_samples": utilisation,
        "particles_initial": [w.get("particles_initial") for w in ok],
        "step_graph": [w.get("step_graph") for w in ok],
        "workers": workers,
    }


def attach_speedups(rounds: list[dict[str, Any]]) -> None:
    """Add per-round speedups: vs the N=1 round of the same configuration and vs the 5090 anchors."""

    single: dict[str, float] = {}
    for r in rounds:
        if r["concurrency"] == 1 and r.get("aggregate_steps_per_s"):
            single[r["config"]] = float(r["aggregate_steps_per_s"])
    for r in rounds:
        agg = r.get("aggregate_steps_per_s")
        base = single.get(r["config"])
        r["aggregate_speedup_vs_single_process"] = (agg / base) if agg and base else None
        anchor = CONFIGS[r["config"]].anchor
        anchor_ms = anchor.seed_load_ms_per_step if r.get("load", "seed") == "seed" else anchor.production_ms_per_step
        mean = r.get("ms_per_step_wall_mean")
        r["anchor_5090_ms_per_step"] = anchor_ms
        r["anchor_5090_predicted"] = anchor.predicted if anchor_ms is not None else None
        # per-process speedup: how much faster ONE process runs here than on the 5090 (N=1 is the fair one)
        r["per_process_speedup_vs_5090"] = (anchor_ms / mean) if anchor_ms and mean else None
        # throughput speedup: aggregate steps/s of the GPU relative to one 5090 process
        r["gpu_throughput_speedup_vs_5090"] = (agg * anchor_ms / 1e3) if anchor_ms and agg else None


def run_command(args: argparse.Namespace) -> int:
    python = args.python or sys.executable
    keys = args.configs or default_config_keys()
    unknown = [k for k in keys if k not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configuration(s) {unknown}; known: {sorted(CONFIGS)}")
    scratch = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix="pic-bench-"))
    scratch.mkdir(parents=True, exist_ok=True)
    inventory = gpu_inventory()
    pinned = next((g for g in inventory if g["index"] == int(args.gpu)), None)
    report: dict[str, Any] = {
        "schema": SCHEMA, "kind": "gpu-concurrency", "host": socket.gethostname(), "platform": platform.platform(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_head": git_head(),
        "gpu_index": int(args.gpu), "gpu": pinned, "gpu_inventory": inventory, "cpu_count": os.cpu_count(),
        "warmup_steps": int(args.warmup), "steps": int(args.steps), "load": args.load,
        "overrides": [list(item) for item in (args.override or [])],
        "anchor_gpu": LOCAL_ANCHOR_GPU,
        "configs": {k: {"description": CONFIGS[k].description, "protocol": CONFIGS[k].protocol,
                        "overrides": dict(CONFIGS[k].overrides), "anchor": asdict(CONFIGS[k].anchor)} for k in keys},
        "rounds": [],
    }
    if args.dry_run:
        for key in keys:
            for count in args.concurrency:
                print(f"would run {key} with N={count} on GPU {args.gpu} "
                      f"({blas_threads_per_process(count, cap=args.threads_cap)} BLAS threads/process)")
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for key in keys:
        for count in args.concurrency:
            round_record = run_round(CONFIGS[key], int(count), gpu=int(args.gpu), warmup=int(args.warmup),
                                     steps=int(args.steps), load=args.load, overrides=list(args.override or []),
                                     scratch=scratch, python=python, barrier_timeout=float(args.barrier_timeout),
                                     threads_cap=int(args.threads_cap))
            round_record["load"] = args.load
            report["rounds"].append(round_record)
            attach_speedups(report["rounds"])
            out.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")   # partial results survive
            print(format_round_line(round_record))
    md = format_markdown(report)
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
    print(md)
    if not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


def factorise_command(args: argparse.Namespace) -> int:
    python = args.python or sys.executable
    keys = args.configs or ["channel-50um", "plume-v2.0-50um"]
    scratch = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix="pic-factorise-"))
    scratch.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": SCHEMA, "kind": "host-factorisation", "host": socket.gethostname(),
        "platform": platform.platform(), "cpu_count": os.cpu_count(), "git_head": git_head(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rounds": [],
    }
    for key in keys:
        for count in args.concurrency:
            count = int(count)
            threads = blas_threads_per_process(count, cap=int(args.threads_cap))
            round_dir = scratch / f"{key}-N{count}"
            round_dir.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env.update(blas_environment(threads))
            # CPU only: "-1" hides every CUDA device (an empty string is dropped by some shells, e.g. PowerShell)
            env.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": f"{MODERN / 'src'}{os.pathsep}{MODERN}"})
            procs = []
            for k in range(count):
                command = [python, "-m", "tools.cloud.bench_gpu_concurrency", "factorise-worker", "--config", key,
                           "--index", str(k), "--count", str(count), "--barrier", str(round_dir),
                           "--out", str(round_dir / f"worker-{k}.json"), "--barrier-timeout", str(args.barrier_timeout)]
                for okey, value in (args.override or []):
                    command += ["--override", f"{okey}={json.dumps(value)}"]
                procs.append(_spawn(command, env, round_dir / f"worker-{k}.log"))
            t_start = time.monotonic()
            while sum((round_dir / f"ready-{k}").exists() for k in range(count)) < count:
                if not any(p.poll() is None for p in procs) or time.monotonic() - t_start > float(args.barrier_timeout):
                    break
                time.sleep(0.2)
            (round_dir / "go").write_text("go", encoding="utf-8")
            for p in procs:
                p.wait()
            workers = []
            for k, proc in enumerate(procs):
                path = round_dir / f"worker-{k}.json"
                record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"ok": False, "index": k}
                record["exit_code"] = proc.returncode
                workers.append(record)
            ok = [w for w in workers if w.get("ok")]
            times = [float(w["factorisation_s"]) for w in ok]
            round_record = {
                "config": key, "concurrency": count, "blas_threads_per_process": threads,
                "processes_ok": len(ok), "processes_failed": count - len(ok),
                "factorisation_s_per_process": times,
                "factorisation_s_max": max(times) if times else None,
                "inverse_blocks_gib": (ok[0]["inverse_blocks_bytes"] / 2**30) if ok else None,
                "block_size": ok[0].get("block_size") if ok else None, "blocks": ok[0].get("blocks") if ok else None,
                "workers": workers,
            }
            report["rounds"].append(round_record)
            print(f"[factorise] {key} N={count} ({threads} threads/process): "
                  + (f"{min(times):.1f}-{max(times):.1f} s" if times else "FAILED"))
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    md = format_factorisation_markdown(report)
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
    print(md)
    if not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPOSITORY), capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------------------- formatting
def _fmt(value: Any, spec: str = ".2f", missing: str = "-") -> str:
    if value is None:
        return missing
    if isinstance(value, float) and math.isnan(value):
        return missing
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def _fmt_list(values: list[Any] | None, spec: str = ".2f") -> str:
    if not values:
        return "-"
    return " / ".join(_fmt(v, spec) for v in values)


def format_round_line(r: Mapping[str, Any]) -> str:
    return (f"[bench] {r['config']} N={r['concurrency']}: ms/step {_fmt_list(r.get('ms_per_step_wall_per_process'))}"
            f" aggregate {_fmt(r.get('aggregate_steps_per_s'), '.1f')} steps/s"
            f" (x{_fmt(r.get('aggregate_speedup_vs_single_process'))} vs N=1;"
            f" per-process x{_fmt(r.get('per_process_speedup_vs_5090'))} vs 5090)")


def format_markdown(report: Mapping[str, Any]) -> str:
    """Markdown table of a ``gpu-concurrency`` report (pure; unit-tested with synthetic timings)."""

    gpu = report.get("gpu") or {}
    head = (report.get("git_head") or "?")[:8]
    intro = (f"Host `{report.get('host')}`, driver {gpu.get('driver_version', '?')}, git `{head}`, "
             f"{report.get('utc')}; {report.get('warmup_steps')} warm-up + {report.get('steps')} timed steps per "
             f"process, particle load `{report.get('load')}`. Anchors: {report.get('anchor_gpu')}.")
    columns = ("| config | N | ms/step per process | aggregate steps/s | x vs N=1 | "
               "GPU MiB / process (nvidia-smi) | Warp mempool high MiB | setup s (factorisation + upload) | "
               "5090 anchor ms/step | per-process x vs 5090 | GPU throughput x vs 5090 |")
    lines = [
        f"# PIC step-loop concurrency on one GPU - {gpu.get('name', 'unknown GPU')} (index {report.get('gpu_index')})",
        "",
        intro,
        "",
        columns,
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.get("rounds", []):
        anchor = _fmt(r.get("anchor_5090_ms_per_step"))
        if r.get("anchor_5090_predicted"):
            anchor += " (pred.)"
        failed = f" ({r['processes_failed']} failed)" if r.get("processes_failed") else ""
        lines.append(
            f"| {r['config']} | {r['concurrency']}{failed} | {_fmt_list(r.get('ms_per_step_wall_per_process'))} | "
            f"{_fmt(r.get('aggregate_steps_per_s'), '.1f')} | {_fmt(r.get('aggregate_speedup_vs_single_process'))} | "
            f"{_fmt_list(r.get('gpu_memory_mib_per_process_nvidia_smi'), '.0f')} | "
            f"{_fmt_list(r.get('gpu_memory_mib_per_process_warp_high'), '.0f')} | "
            f"{_fmt_list(r.get('setup_s_per_process'), '.1f')} | {anchor} | "
            f"{_fmt(r.get('per_process_speedup_vs_5090'))} | {_fmt(r.get('gpu_throughput_speedup_vs_5090'))} |"
        )
    reading = ("Reading: ms/step is the v4 preflight's `_time_steps` measure (200 warm-up + the timed steps, window "
               "accumulation on); `x vs N=1` is the aggregate-throughput gain from sharing the GPU (the number that "
               "sets `slots_per_gpu` in `jobs.yaml`); `per-process x vs 5090` is the single-run speed-up (N=1 is the "
               "fair row; larger N slows every process); a `(pred.)` anchor is a cost-model prediction, not a "
               "measurement. The `seed` load times the launch-bound floor at the protocol seed; the `production` "
               "load re-seeds to the recorded plateau particle counts so the anchors are like-for-like.")
    lines += ["", reading]
    rounds = report.get("rounds", [])
    if any(r.get("overlap_fraction") is not None and r["overlap_fraction"] < 0.8 for r in rounds):
        warning = ("Warning: at least one round had < 80 % overlap of the timed windows (uneven process speeds): "
                   "the slower processes ran partly alone, so their ms/step is optimistic for that row.")
        lines += ["", warning]
    return "\n".join(lines) + "\n"


def format_factorisation_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Host Poisson factorisation (block-Thomas row blocks) - {report.get('host')} ({report.get('cpu_count')} CPUs)",
        "",
        "| config | N concurrent | BLAS threads / process | factorisation s per process | max s | block size | blocks | inverse blocks GiB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report.get("rounds", []):
        failed = f" ({r['processes_failed']} failed)" if r.get("processes_failed") else ""
        lines.append(f"| {r['config']} | {r['concurrency']}{failed} | {r.get('blas_threads_per_process')} | "
                     f"{_fmt_list(r.get('factorisation_s_per_process'), '.1f')} | {_fmt(r.get('factorisation_s_max'), '.1f')} | "
                     f"{_fmt(r.get('block_size'), 'd')} | {_fmt(r.get('blocks'), 'd')} | {_fmt(r.get('inverse_blocks_gib'))} |")
    note = ("Every process pins OMP/OPENBLAS/MKL/NUMEXPR threads to floor(CPUs / N) (cap 16): the local lesson "
            "was that two unpinned factorisations oversubscribed the BLAS threads (20 min without finishing).")
    lines += ["", note]
    return "\n".join(lines) + "\n"


def report_command(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if report.get("kind") == "host-factorisation":
        md = format_factorisation_markdown(report)
    else:
        attach_speedups(report.get("rounds", []))
        md = format_markdown(report)
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
    print(md)
    return 0


# --------------------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="GPU concurrency rounds on one GPU")
    run.add_argument("--gpu", type=int, default=0, help="nvidia-smi GPU index to pin (CUDA_DEVICE_ORDER=PCI_BUS_ID)")
    run.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    run.add_argument("--configs", nargs="*", default=None, help=f"subset of {sorted(CONFIGS)}")
    run.add_argument("--warmup", type=int, default=400,
                     help="pre-warm steps before the barrier (module load, graph capture); the timing function adds "
                          "its own 200 warm-up steps as in the v4 preflight")
    run.add_argument("--steps", type=int, default=2000, help="timed steps per process")
    run.add_argument("--load", choices=("seed", "production"), default="seed",
                     help="seed = protocol seed density (~0.26 M particles); production = re-seed to the plateau load")
    run.add_argument("--override", type=parse_override, action="append", default=None,
                     help="extra in-memory protocol override KEY=JSON (e.g. case.macro_weight=30000)")
    run.add_argument("--out", default=str(MODERN / "results" / "bench-gpu-concurrency.json"))
    run.add_argument("--markdown", default=None)
    run.add_argument("--scratch", default=None, help="barrier / worker files (default: a temp dir)")
    run.add_argument("--keep-scratch", action="store_true")
    run.add_argument("--python", default=None, help="interpreter for the workers (default: this one)")
    run.add_argument("--barrier-timeout", type=float, default=7200.0, help="seconds to wait for every worker to warm up")
    run.add_argument("--threads-cap", type=int, default=16)
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=run_command)

    fact = sub.add_parser("factorise", help="CPU-only: host block-Thomas factorisation under concurrency")
    fact.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    fact.add_argument("--configs", nargs="*", default=None)
    fact.add_argument("--override", type=parse_override, action="append", default=None)
    fact.add_argument("--out", default=str(MODERN / "results" / "bench-host-factorisation.json"))
    fact.add_argument("--markdown", default=None)
    fact.add_argument("--scratch", default=None)
    fact.add_argument("--keep-scratch", action="store_true")
    fact.add_argument("--python", default=None)
    fact.add_argument("--barrier-timeout", type=float, default=3600.0)
    fact.add_argument("--threads-cap", type=int, default=16)
    fact.set_defaults(func=factorise_command)

    rep = sub.add_parser("report", help="re-render the markdown table from a JSON report")
    rep.add_argument("report")
    rep.add_argument("--markdown", default=None)
    rep.set_defaults(func=report_command)

    worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", required=True, choices=sorted(CONFIGS))
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--count", type=int, required=True)
    worker.add_argument("--warmup", type=int, required=True)
    worker.add_argument("--steps", type=int, required=True)
    worker.add_argument("--load", choices=("seed", "production"), default="seed")
    worker.add_argument("--override", type=parse_override, action="append", default=None)
    worker.add_argument("--barrier", required=True)
    worker.add_argument("--out", required=True)
    worker.add_argument("--backend", default="warp-cuda")
    worker.add_argument("--barrier-timeout", type=float, default=7200.0)
    worker.set_defaults(func=worker_main)

    fworker = sub.add_parser("factorise-worker", help=argparse.SUPPRESS)
    fworker.add_argument("--config", required=True, choices=sorted(CONFIGS))
    fworker.add_argument("--index", type=int, required=True)
    fworker.add_argument("--count", type=int, required=True)
    fworker.add_argument("--override", type=parse_override, action="append", default=None)
    fworker.add_argument("--barrier", default=None)
    fworker.add_argument("--out", required=True)
    fworker.add_argument("--barrier-timeout", type=float, default=3600.0)
    fworker.set_defaults(func=factorise_worker_main)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
