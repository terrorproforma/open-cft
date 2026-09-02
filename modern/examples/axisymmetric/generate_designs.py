r"""Generate the three deterministic hypothetical L1a design artifacts.

Run from ``modern`` with:
    $env:PYTHONPATH="$PWD\src"
    python examples/axisymmetric/generate_designs.py
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

from cft_revival.fields import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    AzimuthalCurrentBand,
    SolverConfig,
    canonical_field_artifact_bytes,
    canonical_payload_sha256,
    design_manifest,
    field_artifact,
    manifest_entry,
    solve_problem_cpu,
    solve_problem_warp,
    validate_design_manifest_file,
    write_design_manifest,
    write_field_artifact,
)

LEGACY_V11 = {
    "manifest_file_sha256": "8444389efc87f89495e34d46ccf2deedcc44ee65614dfdd660beecf84cedc3b4",
    "manifest_payload_sha256": "2c912b847702e14223170917850d1ecd5fbdfb45899d96ddb222b5577531d7a6",
    "artifacts": {
        "hypothetical-compact-mirror": {
            "file_sha256": "6510f6ea687022f358103bba99456e7bf651686e3add29205c0560c933981afb",
            "payload_sha256": "92e5535af0492e1697dad2540d8f6e837ba11f28f7a81626673a1c0004183348",
        },
        "hypothetical-opposed-cusp": {
            "file_sha256": "dbf05208dc77e694bb40bb3ca82e4ee3e7126bb3036156f7fa1a726eab06b5c6",
            "payload_sha256": "c4c7c3dc45466bfa4ba187e925b8e41a1c979b3700a59e185d24897501f97263",
        },
        "hypothetical-thick-outer-triplet": {
            "file_sha256": "ac5420d9276d3db03adffe548a459706de95593d74c4181af4034ddbd1ce4b7a",
            "payload_sha256": "d6ef0a42b0a73cfafc7cad1a3fdca8ca59fff4c13a444f5fe5ee8fae9ebf690b",
        },
    },
}


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
    current_manifest = design_manifest(manifest)
    manifest_file_hash = write_design_manifest(manifest_path, current_manifest)
    validate_design_manifest_file(manifest_path)

    migration_payload = {
        "schema_version": "cft-axisymmetric-serialization-migration/1.0.0",
        "from": {
            "artifact_schema": "cft-axisymmetric-field-map/1.1.0",
            "manifest_schema": "cft-axisymmetric-design-manifest/1.1.0",
            **LEGACY_V11,
        },
        "to": {
            "artifact_schema": "cft-axisymmetric-field-map/1.2.0",
            "manifest_schema": "cft-axisymmetric-design-manifest/1.2.0",
            "manifest_file_sha256": manifest_file_hash,
            "manifest_payload_sha256": current_manifest["integrity"]["payload_sha256"],
            "artifacts": {
                entry["name"]: {
                    "file_sha256": entry["artifact_file_sha256"],
                    "payload_sha256": entry["artifact_payload_sha256"],
                }
                for entry in manifest
            },
        },
        "policy": (
            "v1.1 is read-only historical serialization; new outputs use v1.2 "
            "signed-zero normalization; no experiment artifact is migrated in place"
        ),
    }
    migration = {
        **migration_payload,
        "integrity": {
            "algorithm": "sha256",
            "canonicalization": "field-json-sorted-utf8-signed-zero-v2",
            "payload_sha256": canonical_payload_sha256(migration_payload),
        },
    }
    migration_path = (
        arguments.output / "serialization-migration-v1.1-to-v1.2.json"
    )
    migration_bytes = canonical_field_artifact_bytes(
        migration, representation="file"
    )
    migration_path.write_bytes(migration_bytes)
    migration_hash = hashlib.sha256(migration_bytes).hexdigest()
    migration_path.with_name(migration_path.name + ".sha256").write_text(
        f"{migration_hash}  {migration_path.name}\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
