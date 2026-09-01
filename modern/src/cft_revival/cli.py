"""Command-line entry points for validation and compatibility inspection."""

from __future__ import annotations

import argparse
import json
from math import isfinite
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

    l0_point = commands.add_parser(
        "l0-evaluate",
        help="evaluate one checked hypothetical L0 operating point",
    )
    l0_point.add_argument("config", type=Path)
    l0_point.add_argument("--output", type=Path)

    l0_sweep = commands.add_parser(
        "l0-sweep",
        help="run a deterministic checked L0 batch with CPU-reference parity",
    )
    l0_sweep.add_argument("config", type=Path)
    l0_sweep.add_argument(
        "--device",
        default="python",
        help="'python', 'cpu', 'cuda', or 'cuda:N'",
    )
    l0_sweep.add_argument("--output", type=Path)

    validate_campaign = commands.add_parser(
        "validate-campaign-spec",
        help="strictly validate optimization campaign spec v1.4 without BoTorch",
    )
    validate_campaign.add_argument("spec", type=Path)

    initial_design = commands.add_parser(
        "generate-initial-design",
        help="validate a campaign spec and emit deterministic initial designs",
    )
    initial_design.add_argument("spec", type=Path)
    initial_design.add_argument("--count", type=int, required=True)
    initial_design.add_argument("--seed", type=int, default=0)
    initial_design.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        return _run(arguments)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        parser.exit(
            2,
            f"{parser.prog}: error [{type(error).__name__}]: {error}\n",
        )
    return 2


def _run(arguments: argparse.Namespace) -> int:
    if arguments.command == "validate-config":
        path: Path = arguments.path
        raw = _read_json(path)
        config = AppConfig.from_mapping(raw, path.parent)
        print(f"valid configuration; outputs={config.output_directory}")
        return 0

    if arguments.command == "cusp-probability":
        print(f"{cusp_arrival_probability(arguments.low_t, arguments.high_t):.17g}")
        return 0

    if arguments.command == "benchmark-cusp":
        return _benchmark_cusp(arguments)

    if arguments.command == "l0-evaluate":
        from .physics.workflows import evaluate_operating_point_artifact

        artifact = evaluate_operating_point_artifact(_read_json(arguments.config))
        _emit_json(artifact, arguments.output)
        return 0

    if arguments.command == "l0-sweep":
        from .physics.workflows import evaluate_sweep_artifact

        artifact = evaluate_sweep_artifact(
            _read_json(arguments.config),
            device=arguments.device,
        )
        _emit_json(artifact, arguments.output)
        return 0

    if arguments.command in {
        "validate-campaign-spec",
        "generate-initial-design",
    }:
        from .optimization.spec import (
            campaign_spec_artifact,
            campaign_validation_artifact,
            load_json_strict,
        )

        spec = load_json_strict(arguments.spec)
        if arguments.command == "validate-campaign-spec":
            artifact = campaign_validation_artifact(spec)
            output = None
        else:
            artifact = campaign_spec_artifact(
                spec,
                initial_design_count=arguments.count,
                seed=arguments.seed,
            )
            output = arguments.output
        _emit_json(artifact, output)
        return 0

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


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant {value}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _check_finite_json(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"JSON contains non-finite number at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite_json(item, f"{path}.{key}")
        return
    raise ValueError(f"JSON contains unsupported value at {path}")


def _read_json(path: Path) -> dict[str, object]:
    raw = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    _check_finite_json(raw)
    return raw


def _emit_json(artifact: dict[str, object], output: Path | None) -> None:
    encoded = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
    if output is None:
        print(encoded)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{encoded}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "document_type": artifact.get("document_type"),
                "output": str(output),
            },
            indent=2,
        )
    )


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
