"""v2.0.2 runner tooling: the background nvidia-smi sampler never blocks the stepping thread.

Plume attempt 7 (2026-09-04) spent 3.9 % of its wall budget in synchronous ``nvidia-smi`` calls
(17 of 238 hit the 5 s timeout under GPU contention).  The sampler runs on a daemon thread at a
configurable cadence; the step loop only reads the shared last value.
"""

from __future__ import annotations

import math
import threading
import time

import pytest

from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_cft_steady_state_v1.gpu_sampler import (
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    GpuUtilisationSampler,
    query_gpu_utilisation,
)


def test_a_hung_sampler_never_blocks_the_step_loop():
    """The query blocks forever: ``latest()`` and ``stop()`` return immediately, the thread is a daemon."""

    release = threading.Event()
    entered = threading.Event()

    def hung_query(timeout_s: float) -> float | None:
        entered.set()
        release.wait()          # never released while the "step loop" runs
        return 100.0

    sampler = GpuUtilisationSampler(interval_s=0.01, timeout_s=DEFAULT_TIMEOUT_SECONDS, query=hung_query).start()
    assert entered.wait(2.0), "the sampler thread did not start"
    # the step loop: many reads while the sampler is stuck inside nvidia-smi; each read is sub-millisecond
    worst = 0.0
    for _ in range(200):
        t0 = time.perf_counter()
        value = sampler.latest()
        worst = max(worst, time.perf_counter() - t0)
        assert value is None                         # nothing completed yet
    assert worst < 0.05
    assert sampler.calls == 0 and sampler.snapshot() == [] and sampler.thread_alive
    t0 = time.perf_counter()
    still_alive = sampler.stop(join_timeout_s=0.2)      # bounded: the join times out, the daemon thread is abandoned
    assert time.perf_counter() - t0 < 1.0 and still_alive is True and sampler._thread.daemon
    summary = sampler.summary()
    assert summary["calls"] == 0 and summary["samples"] == 0 and summary["thread_alive_at_stop"] is True
    release.set()                                        # let the thread finish so the test process stays clean
    sampler._thread.join(timeout=2.0)
    assert not sampler.thread_alive


def test_sampler_keeps_the_none_safe_float_contract_and_the_cadence():
    """Failures, exceptions and non-finite readings are ``None`` (canonical JSON, the attempt-7 lesson); values are floats."""

    readings = iter([37.0, float("nan"), RuntimeError("nvidia-smi missing"), 12, None, float("inf"), 55.5])

    def query(timeout_s: float) -> float | None:
        assert timeout_s == 0.5
        value = next(readings)
        if isinstance(value, Exception):
            raise value
        return value

    sampler = GpuUtilisationSampler(interval_s=0.001, timeout_s=0.5, query=query)
    for _ in range(7):
        sampler.sample_once()
    assert sampler.snapshot() == [37.0, None, None, 12.0, None, None, 55.5]
    assert all(isinstance(v, float) for v in sampler.snapshot() if v is not None)
    assert sampler.calls == 7 and sampler.failures == 4 and sampler.latest() == 55.5
    assert len(sampler.sample_times_s) == 7 and sampler.summary()["failures_or_timeouts"] == 4
    # background cadence: several samples land within a short run at a 5 ms interval, none in the stepping thread
    ticks = GpuUtilisationSampler(interval_s=0.005, timeout_s=0.5, query=lambda t: 1.0).start()
    deadline = time.perf_counter() + 2.0
    while len(ticks.snapshot()) < 3 and time.perf_counter() < deadline:
        time.sleep(0.005)
    assert ticks.stop() is False and len(ticks.snapshot()) >= 3 and ticks.latest() == 1.0
    # configuration guards and defaults: 5 min cadence, 5 s timeout
    assert DEFAULT_INTERVAL_SECONDS == 300.0 and DEFAULT_TIMEOUT_SECONDS == 5.0
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            GpuUtilisationSampler(interval_s=bad)
        with pytest.raises(ValueError):
            GpuUtilisationSampler(timeout_s=bad)


def test_query_gpu_utilisation_is_none_safe_when_nvidia_smi_is_unavailable(monkeypatch):
    """The one-shot query returns ``None`` (never NaN, never raises) when the tool is missing, slow or garbled."""

    import subprocess

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)
    assert query_gpu_utilisation(0.1) is None

    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", slow)
    assert query_gpu_utilisation(0.1) is None

    class Garbled:
        stdout = "N/A\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Garbled())
    assert query_gpu_utilisation(0.1) is None

    class Nan:
        stdout = "nan\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Nan())
    assert query_gpu_utilisation(0.1) is None

    class Fine:
        stdout = " 87 \n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Fine())
    value = query_gpu_utilisation(0.1)
    assert value == 87.0 and math.isfinite(value)


def test_runner_cli_exposes_the_sampler_cadence(monkeypatch, tmp_path):
    """``run --gpu-sample-interval-seconds`` reaches ``run_steady_state`` (default 300 s)."""

    captured: dict = {}

    def fake_run(protocol, results, **kwargs):
        captured.update(kwargs)
        return tmp_path / "summary.json"

    monkeypatch.setattr(runner, "run_steady_state", fake_run)
    monkeypatch.setattr(runner, "load_variants", lambda path: {})
    monkeypatch.setattr(runner, "apply_case", lambda protocol, case, variants: (protocol, "results"))
    monkeypatch.setattr(runner, "load_protocol", lambda path: {})
    assert runner.main(["run", "--backend", "cpu"], results=tmp_path / "results") == 0
    assert captured["gpu_sample_interval_seconds"] == 300.0
    assert runner.main(["run", "--backend", "cpu", "--gpu-sample-interval-seconds", "90"], results=tmp_path / "results") == 0
    assert captured["gpu_sample_interval_seconds"] == 90.0
