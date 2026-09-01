r"""Generate the three deterministic hypothetical L1a design artifacts.

Run from ``modern`` with:
    $env:PYTHONPATH="$PWD\src"
    python examples/axisymmetric/generate_designs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from cft_revival.fields import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    AzimuthalCurrentBand,
    SolverConfig,
    design_manifest,
    field_artifact,
    manifest_entry,
    solve_problem_cpu,
    solve_problem_warp,
    validate_design_manifest_file,
    write_design_manifest,
    write_field_artifact,
)


def hypothetical_designs() -> tuple[AxisymmetricProblem, ...]:
    domain = AxisymmetricDomain(
        radius_m=0.14,
        z_min_m=-0.16,
        z_max_m=0.16,
        radial_intervals=64,
        axial_intervals=128,
    )
    return (
        AxisymmetricProblem(
            "hypothetical-compact-mirror",
            domain,
            (
                AzimuthalCurrentBand("upstream", 0.042, 0.057, -0.0825, -0.0525, 2200, 1),
                AzimuthalCurrentBand("downstream", 0.042, 0.057, 0.0525, 0.0825, 2200, 1),
            ),
        ),
        AxisymmetricProblem(
            "hypothetical-opposed-cusp",
            domain,
            (
                AzimuthalCurrentBand("upstream", 0.047, 0.067, -0.090, -0.045, 2600, 1),
                AzimuthalCurrentBand("downstream", 0.047, 0.067, 0.045, 0.090, 2600, -1),
            ),
        ),
        AxisymmetricProblem(
            "hypothetical-thick-outer-triplet",
            domain,
            (
                AzimuthalCurrentBand("upstream-outer", 0.060, 0.090, -0.105, -0.055, 3200, 1),
                AzimuthalCurrentBand("central-inner", 0.032, 0.052, -0.025, 0.025, 1800, -1),
                AzimuthalCurrentBand("downstream-outer", 0.060, 0.090, 0.055, 0.105, 3200, 1),
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="python", choices=("python", "cpu", "cuda:0"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/axisymmetric/results"),
    )
    arguments = parser.parse_args()
    config = SolverConfig(
        relative_tolerance=1.0e-10,
        absolute_tolerance=1.0e-13,
        max_iterations=20_000,
        residual_history_stride=10,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for problem in hypothetical_designs():
        started = perf_counter()
        if arguments.backend == "python":
            field = solve_problem_cpu(problem, config)
        else:
            field = solve_problem_warp(problem, device=arguments.backend, config=config)
        elapsed = perf_counter() - started
        artifact = field_artifact(problem, config, field, map_stride=4, wall_radius_m=0.10)
        filename = f"{problem.name}-l1a-v1.json"
        artifact_path = arguments.output / filename
        file_sha256 = write_field_artifact(artifact_path, artifact)
        summary = artifact["summary"]
        manifest.append(
            manifest_entry(artifact_path, artifact, file_sha256)
        )
        print(
            f"{problem.name}: {elapsed:.6f} s diagnostic-only, "
            f"{field.diagnostics.iterations} iterations, "
            f"|B|max={summary['b_magnitude_max_t']:.6g} T"
        )
    manifest_path = arguments.output / "manifest-l1a-v1.json"
    write_design_manifest(manifest_path, design_manifest(manifest))
    validate_design_manifest_file(manifest_path)


if __name__ == "__main__":
    main()
