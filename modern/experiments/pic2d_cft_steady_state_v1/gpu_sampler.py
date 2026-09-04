"""Background GPU-utilisation sampler for the steady-state runner (v2.0.2 runner tooling).

Plume attempt 7 (2026-09-04) called ``nvidia-smi`` synchronously from the stepping thread once per
logged minute (every ~200 steps): 17 of 238 calls hit the 5 s timeout and the median call cost
2.3 s under GPU contention, 557 s = 3.9 % of the 4 h wall budget spent waiting on a diagnostic.
Here the call runs on a daemon thread at a configurable cadence (default 5 min) and the stepping
thread only reads a shared last value, so a slow or hung ``nvidia-smi`` (or a hung sampler thread)
can never block the step loop.  Every sample is ``float | None`` (``None`` for a failure, a
timeout or a non-finite reading - canonical JSON has no NaN; the attempt-7 lesson, ``3b8b577a``).
"""

from __future__ import annotations

import math
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

DEFAULT_INTERVAL_SECONDS = 300.0
DEFAULT_TIMEOUT_SECONDS = 5.0


def query_gpu_utilisation(timeout_s: float = DEFAULT_TIMEOUT_SECONDS) -> float | None:
    """One ``nvidia-smi`` utilisation reading in percent, or ``None`` on failure / timeout / non-finite output."""

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=timeout_s,
        )
        value = float(completed.stdout.strip().splitlines()[0])
        return value if math.isfinite(value) else None
    except Exception:
        return None


@dataclass
class GpuUtilisationSampler:
    """Samples ``query`` every ``interval_s`` on a daemon thread; ``latest()`` and ``stop()`` never block on the query.

    ``samples`` holds every completed reading in order (``None`` for failed ones) with its wall time;
    ``calls`` / ``failures`` count the query outcomes.  A ``query`` that never returns leaves the
    thread stuck in it: ``latest()`` still returns immediately (the last completed value or ``None``),
    ``stop()`` returns after ``join_timeout_s`` and reports ``thread_alive``; the daemon thread cannot
    hold the process open.
    """

    interval_s: float = DEFAULT_INTERVAL_SECONDS
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS
    query: Callable[[float], float | None] = query_gpu_utilisation
    clock: Callable[[], float] = time.monotonic
    samples: list[float | None] = field(default_factory=list)
    sample_times_s: list[float] = field(default_factory=list)
    calls: int = 0
    failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _latest: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.interval_s) or self.interval_s <= 0.0:
            raise ValueError("interval_s must be positive")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")

    # -- lifecycle -------------------------------------------------------------------
    def start(self) -> "GpuUtilisationSampler":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="gpu-utilisation-sampler", daemon=True)
            self._thread.start()
        return self

    def stop(self, join_timeout_s: float = 1.0) -> bool:
        """Ask the thread to stop and wait at most ``join_timeout_s``; returns whether it is still alive."""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
        return self.thread_alive

    @property
    def thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- reading (stepping thread) ------------------------------------------------------
    def latest(self) -> float | None:
        """Last completed sample (never blocks; ``None`` before the first completed call or after a failed one)."""

        with self._lock:
            return self._latest

    def snapshot(self) -> list[float | None]:
        with self._lock:
            return list(self.samples)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "interval_seconds": self.interval_s, "timeout_seconds": self.timeout_s, "calls": self.calls,
                "failures_or_timeouts": self.failures, "samples": len(self.samples), "thread_alive_at_stop": self.thread_alive,
                "note": "background daemon thread; the stepping thread reads the last value only (v2.0.2 runner: attempt 7 spent "
                        "3.9 % of its wall budget in synchronous nvidia-smi calls, 17 of 238 timed out)",
            }

    # -- thread body ---------------------------------------------------------------------
    def sample_once(self) -> float | None:
        """One query, recorded (also usable synchronously, e.g. in tests)."""

        value: float | None
        try:
            value = self.query(self.timeout_s)
        except Exception:
            value = None
        if value is not None and not (isinstance(value, (int, float)) and math.isfinite(value)):
            value = None
        with self._lock:
            self.calls += 1
            if value is None:
                self.failures += 1
            else:
                value = float(value)
            self.samples.append(value)
            self.sample_times_s.append(float(self.clock()))
            self._latest = value
        return value

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            if self._stop.wait(self.interval_s):
                break


__all__ = ["DEFAULT_INTERVAL_SECONDS", "DEFAULT_TIMEOUT_SECONDS", "GpuUtilisationSampler", "query_gpu_utilisation"]
