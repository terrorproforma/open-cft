"""Command-line entry points for validation and compatibility inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .backends import FemmExportBackend
from .kernels import (
    cusp_arrival_probabilities,
    cusp_arrival_probability,
    cusp_arrival_probability_python,
    legacy_cusp_fields,
)
from .models import AppConfig, DesignPoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cft-revival")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-config", help="validate a JSON configuration")
    validate.add_argument("path", type=Path)

    cusp = commands.add_parser("cusp-probability", help="evaluate the loss-cone kernel")
    cusp.add_argument("--low-t", type=float, required=True)
    cusp.add_argument("--high-t", type=float, required=True)

    benchmark = commands.add_parser(
        "benchmark-cusp", help="smoke-test a Warp batch device; timing is diagnostic"
    )
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--batch-size", type=int, default=65_536)
    benchmark.add_argument("--warmup", type=int, default=2)
    benchmark.add_argument("--repeat", type=int, default=5)
    benchmark.add_argument("--seed", type=int, default=20_170_032)
    benchmark.add_argument(
        "--gpu-busy",
        action="store_true",
        help="record that other work occupied the GPU during this run",
    )

    inspect = commands.add_parser(
        "inspect-femm-export", help="read one pair of existing FEMM export files"
    )
    inspect.add_argument("directory", type=Path)
    inspect.add_argument("--generation", type=int, required=True)
    inspect.add_argument("--individual", type=int, required=True)
    inspect.add_argument(
        "--design",
        type=float,
        nargs=8,
        metavar=("U", "I", "FLOW", "R1", "R2", "R3", "R4", "R5"),
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate-config":
        path: Path = arguments.path
        raw = json.loads(path.read_text(encoding="utf-8"))
        config = AppConfig.from_mapping(raw, path.parent)
        print(f"valid configuration; outputs={config.output_directory}")
        return 0

    if arguments.command == "cusp-probability":
        print(f"{cusp_arrival_probability(arguments.low_t, arguments.high_t):.17g}")
        return 0

    if arguments.command == "benchmark-cusp":
        return _benchmark_cusp(arguments)

    design = DesignPoint.from_sequence(arguments.design)
    fields = FemmExportBackend(arguments.directory).solve(
        design, arguments.generation, arguments.individual
    )
    low, high = legacy_cusp_fields(fields.centreline, fields.wall)
    probabilities = cusp_arrival_probabilities(low, high)
    print(
        json.dumps(
            {
                "low_field_t": low,
                "high_field_t": high,
                "p1_to_p4": probabilities.as_tuple(),
                "provenance": fields.provenance,
            },
            indent=2,
        )
    )
    return 0


def _benchmark_cusp(arguments: argparse.Namespace) -> int:
    from random import Random
    from statistics import median
    from time import perf_counter

    from .warp_backend import cusp_arrival_probabilities_warp

    if arguments.batch_size < 2:
        raise ValueError("batch-size must be at least 2 to include both edge cases")
    if arguments.warmup < 0 or arguments.repeat < 1:
        raise ValueError("warmup must be >= 0 and repeat must be >= 1")

    random = Random(arguments.seed)
    high = [1.0, 1.0]
    low = [0.0, 1.0]
    for _ in range(arguments.batch_size - 2):
        high_value = random.uniform(1.0e-6, 2.0)
        high.append(high_value)
        low.append(high_value * random.random())

    reference = [
        cusp_arrival_probability_python(low_value, high_value)
        for low_value, high_value in zip(low, high, strict=True)
    ]
    result = None
    for _ in range(arguments.warmup):
        result = cusp_arrival_probabilities_warp(low, high, device=arguments.device)

    timings_ms: list[float] = []
    for _ in range(arguments.repeat):
        started = perf_counter()
        result = cusp_arrival_probabilities_warp(low, high, device=arguments.device)
        timings_ms.append((perf_counter() - started) * 1000.0)
    assert result is not None
    max_error = max(
        abs(actual - expected)
        for actual, expected in zip(result.probabilities, reference, strict=True)
    )
    busy_note = (
        "GPU was reported busy; timing is noisy and non-authoritative."
        if arguments.gpu_busy
        else "Smoke timing is uncontrolled and non-authoritative; do not infer speedup."
    )
    print(
        json.dumps(
            {
                "backend": "nvidia-warp",
                "device": result.device,
                "batch_size": arguments.batch_size,
                "warmup_runs": arguments.warmup,
                "measured_runs": arguments.repeat,
                "timing_scope": "end-to-end validation, transfers, kernel, synchronization",
                "median_ms": median(timings_ms),
                "minimum_ms": min(timings_ms),
                "max_abs_error": max_error,
                "first_outputs": result.probabilities[:5],
                "timing_authoritative": False,
                "timing_note": busy_note,
            },
            indent=2,
        )
    )
    return 0
