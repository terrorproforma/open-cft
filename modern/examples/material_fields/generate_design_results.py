"""Regenerate three gated hypothetical L1b design artifacts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from math import ceil
from pathlib import Path

from cft_revival.fields import AxisymmetricDomain
from cft_revival.geometry import (
    PermanentMagnetAuthority,
    PermanentMagnetRepresentationPlan,
    reference_variants,
)
from cft_revival.material_fields import (
    adapt_geometry,
    assess_publication,
    design_domain,
    material_field_artifact,
    MaterialSolveConfig,
    raw_run_observation,
    raster_memory_preflight,
    device_available,
    solve_material_problem_cpu,
    solve_material_problem_warp,
    validate_artifact,
    viewer_contract,
    write_json,
)
from cft_revival.material_fields.acceptance import (
    RawRunObservation,
    _evidence_implementation_sha256,
)
from cft_revival.material_fields.numerics import _implementation_sha256


def _peak(result) -> float:
    return max(
        (br * br + bz * bz) ** 0.5
        for br_row, bz_row in zip(result.field.b_r_t, result.field.b_z_t)
        for br, bz in zip(br_row, bz_row)
    )


def _axis_peak(result) -> float:
    return max(abs(value) for value in result.field.b_z_t[0])


def _relative(left: float, right: float) -> float:
    return abs(right - left) / max(abs(left), abs(right), 1.0e-300)


def _resolved_domain(geometry, padding: float, reference: AxisymmetricDomain) -> AxisymmetricDomain:
    provisional = design_domain(
        geometry, radial_intervals=4, axial_intervals=4, padding_factor=padding
    )
    radial_intervals = ceil(provisional.radius_m / reference.dr_m)
    lower_extension = ceil(
        max(0.0, reference.z_min_m - provisional.z_min_m) / reference.dz_m
    )
    upper_extension = ceil(
        max(0.0, provisional.z_max_m - reference.z_max_m) / reference.dz_m
    )
    return AxisymmetricDomain(
        radial_intervals * reference.dr_m,
        reference.z_min_m - lower_extension * reference.dz_m,
        reference.z_max_m + upper_extension * reference.dz_m,
        radial_intervals,
        reference.axial_intervals + lower_extension + upper_extension,
    )


def _minimum_resolved_domain(geometry, padding: float) -> AxisymmetricDomain:
    provisional = design_domain(
        geometry, radial_intervals=4, axial_intervals=4, padding_factor=padding
    )
    active_regions = tuple(
        region
        for region in geometry.regions
        if geometry.material_by_id(region.material_id).relative_permeability != 1.0
    )
    minimum_radial = min(
        min(
            region.r_outer_start_m - region.r_inner_start_m,
            region.r_outer_end_m - region.r_inner_end_m,
        )
        for region in active_regions
    )
    minimum_axial = min(region.z_max_m - region.z_min_m for region in active_regions)
    return AxisymmetricDomain(
        provisional.radius_m,
        provisional.z_min_m,
        provisional.z_max_m,
        ceil(12.05 * provisional.radius_m / minimum_radial),
        ceil(12.05 * provisional.z_span_m / minimum_axial),
    )


def _equivalent_geometry(geometry):
    authority = PermanentMagnetAuthority.EQUIVALENT_BOUND_CURRENT
    return replace(
        geometry,
        permanent_magnet_plan=PermanentMagnetRepresentationPlan(
            f"{geometry.config_id}-{authority.value}-v1", authority
        ),
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(
    result, *, study_id: str, role: str, directory: Path
) -> Path:
    """Atomically persist one closed raw run before releasing solver arrays."""

    observation = raw_run_observation(result, study_id=study_id, role=role)
    path = directory / f"{study_id}.raw-run.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            observation.to_dict(),
            stream,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        stream.write("\n")
    os.replace(temporary, path)
    digest = _file_sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return path


def _load_checkpoint(path: Path) -> RawRunObservation:
    declared, name = path.with_name(path.name + ".sha256").read_text(
        encoding="ascii"
    ).split()
    if name != path.name or declared != _file_sha256(path):
        raise RuntimeError(f"checkpoint sidecar mismatch: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    observation = RawRunObservation(**value)
    if observation.run_sha256 != hashlib.sha256(
        _canonical(
            {
                key: item
                for key, item in observation.to_dict().items()
                if key != "run_sha256"
            }
        )
    ).hexdigest():
        raise RuntimeError(f"checkpoint inner hash mismatch: {path.name}")
    return observation


def _cleanup_cuda() -> None:
    """Release dead host objects and return cached CUDA blocks to the driver."""

    gc.collect()
    import warp as wp

    wp.set_mempool_release_threshold("cuda:0", 0)
    wp.synchronize_device("cuda:0")


def _frozen_identity(geometry) -> dict[str, str]:
    return {
        "geometry_sha256": geometry.canonical_sha256,
        "evidence_implementation_sha256": _evidence_implementation_sha256(),
        "python_implementation_sha256": _implementation_sha256(
            "adapters.py", "models.py", "numerics.py"
        ),
        "warp_implementation_sha256": _implementation_sha256(
            "adapters.py", "models.py", "numerics.py", "warp_solver.py"
        ),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main(
    selected_config_ids: set[str] | None = None,
    *,
    memory_limited: bool = False,
    output_dir: Path | None = None,
) -> None:
    output = output_dir or Path(__file__).resolve().parent / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    entries_by_config: dict[str, dict[str, object]] = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries_by_config = {
            entry["config_id"]: entry for entry in existing.get("designs", ())
        }
    use_cuda = device_available("cuda")
    if not use_cuda:
        raise RuntimeError("formal publication evidence requires an available CUDA device")

    solve_config = MaterialSolveConfig(
        allow_underresolved_screening=memory_limited
    )

    def solve(problem):
        return solve_material_problem_warp(
            problem, device="cuda", config=solve_config
        )

    for geometry in reference_variants():
        if selected_config_ids is not None and geometry.config_id not in selected_config_ids:
            continue
        requested_base_domain = _minimum_resolved_domain(geometry, 3.0)
        requested_preflight = raster_memory_preflight(
            requested_base_domain, enforce=False
        )
        if memory_limited:
            if requested_preflight["fits"]:
                raise RuntimeError(
                    "memory-limited mode requires a failed high-resolution preflight"
                )
            # All reduced roles remain below the unconditional 64 MiB
            # conservative raster bound despite volatile free memory.
            base_domain = design_domain(
                geometry,
                radial_intervals=60,
                axial_intervals=120,
                padding_factor=3.0,
            )
        else:
            base_domain = requested_base_domain
        fine_domain = AxisymmetricDomain(
            base_domain.radius_m,
            base_domain.z_min_m,
            base_domain.z_max_m,
            ceil(1.25 * base_domain.radial_intervals),
            ceil(1.25 * base_domain.axial_intervals),
        )
        third_grids = (
            {
                "compact-high-gradient-stack-v1": (83, 166),
                "divergent-exit-stack-v1": (83, 166),
                "historical-envelope-baseline-v1": (83, 166),
            }
            if memory_limited
            else {
                "compact-high-gradient-stack-v1": (1047, 1802),
                "divergent-exit-stack-v1": (807, 1616),
                "historical-envelope-baseline-v1": (730, 2835),
            }
        )
        third_radial, third_axial = third_grids[geometry.config_id]
        third_domain = AxisymmetricDomain(
            base_domain.radius_m,
            base_domain.z_min_m,
            base_domain.z_max_m,
            third_radial,
            third_axial,
        )
        alignment_domain = AxisymmetricDomain(
            base_domain.radius_m,
            base_domain.z_min_m + 0.25 * base_domain.dz_m,
            base_domain.z_max_m + 0.25 * base_domain.dz_m,
            base_domain.radial_intervals,
            base_domain.axial_intervals,
        )
        expansion_1_domain = _resolved_domain(geometry, 4.5, base_domain)
        expansion_2_domain = _resolved_domain(geometry, 6.75, base_domain)
        parity_domain = design_domain(
            geometry,
            radial_intervals=80 if memory_limited else 192,
            axial_intervals=160 if memory_limited else 384,
            padding_factor=1.0,
        )
        parity_config = MaterialSolveConfig(allow_underresolved_screening=True)
        if memory_limited:
            reduced_domains = (
                base_domain,
                fine_domain,
                third_domain,
                alignment_domain,
                expansion_1_domain,
                expansion_2_domain,
                parity_domain,
            )
            if any(
                int(
                    raster_memory_preflight(domain, enforce=False)[
                        "estimated_raster_bytes"
                    ]
                )
                >= 64 * 1024**2
                for domain in reduced_domains
            ):
                raise RuntimeError(
                    "reduced study exceeds the unconditional 64 MiB raster bound"
                )
        qualification = (
            {
                "schema_version": "cft_revival.material_fields.qualification/1.4.0",
                "study_scope": "MEMORY_LIMITED_REDUCED_SCREENING",
                "status": "NOT_EVALUATED",
                "reason_code": "HOST_MEMORY_LIMIT",
                "required_role_count": 10,
                "completed_role_count": 10,
                "not_evaluated_roles": [],
                "requested_base_grid": [
                    requested_base_domain.radial_intervals,
                    requested_base_domain.axial_intervals,
                ],
                "executed_base_grid": [
                    base_domain.radial_intervals,
                    base_domain.axial_intervals,
                ],
                "estimated_requested_raster_bytes": int(
                    requested_preflight["estimated_raster_bytes"]
                ),
                "safe_raster_bytes": int(
                    requested_preflight["safe_raster_bytes"]
                ),
            }
            if memory_limited
            else None
        )

        plan = {
            "schema_version": "cft_revival.material_fields.raw_run_plan/1.4.0",
            "config_id": geometry.config_id,
            "identity": _frozen_identity(geometry),
            "qualification": qualification,
            "roles": [
                ["base", "base"],
                ["domain-1", "domain_expansion"],
                ["domain-2", "domain_expansion"],
                ["mesh-fine", "mesh_refinement"],
                ["mesh-third", "mesh_refinement_2"],
                ["alignment-1", "grid_alignment"],
                ["equivalent-base", "model_form"],
                ["equivalent-fine", "model_form"],
                ["parity-cpu", "backend_parity_cpu"],
                ["parity-cuda", "backend_parity_cuda"],
            ],
        }
        plan_sha = hashlib.sha256(_canonical(plan)).hexdigest()
        checkpoint_dir = (
            Path(tempfile.gettempdir())
            / "cft-revival-material-fields-v1.4"
            / geometry.config_id
            / plan_sha
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "plan.json").write_bytes(_canonical(plan) + b"\n")

        def assert_frozen() -> None:
            if _frozen_identity(geometry) != plan["identity"]:
                raise RuntimeError(
                    "material-field code or geometry changed during the frozen study"
                )

        def capture(
            study_id: str,
            role: str,
            problem,
            solver=solve,
            *,
            retain_result: bool = False,
        ):
            assert_frozen()
            print(f"{geometry.config_id}: {study_id}", flush=True)
            result = solver(problem)
            path = _checkpoint(
                result,
                study_id=study_id,
                role=role,
                directory=checkpoint_dir,
            )
            del problem
            _cleanup_cuda()
            assert_frozen()
            if retain_result:
                return path, result
            del result
            return path

        base_path, base = capture(
            "base",
            "base",
            adapt_geometry(geometry, base_domain),
            retain_result=True,
        )
        paths = {
            "base": base_path,
            "mesh-fine": capture(
                "mesh-fine",
                "mesh_refinement",
                adapt_geometry(geometry, fine_domain),
            ),
            "mesh-third": capture(
                "mesh-third",
                "mesh_refinement_2",
                adapt_geometry(geometry, third_domain),
            ),
            "alignment-1": capture(
                "alignment-1",
                "grid_alignment",
                adapt_geometry(geometry, alignment_domain),
            ),
            "domain-1": capture(
                "domain-1",
                "domain_expansion",
                adapt_geometry(geometry, expansion_1_domain),
            ),
            "domain-2": capture(
                "domain-2",
                "domain_expansion",
                adapt_geometry(geometry, expansion_2_domain),
            ),
            "equivalent-base": capture(
                "equivalent-base",
                "model_form",
                adapt_geometry(_equivalent_geometry(geometry), base_domain),
            ),
            "equivalent-fine": capture(
                "equivalent-fine",
                "model_form",
                adapt_geometry(_equivalent_geometry(geometry), fine_domain),
            ),
            "parity-cpu": capture(
                "parity-cpu",
                "backend_parity_cpu",
                adapt_geometry(geometry, parity_domain),
                solver=lambda problem: solve_material_problem_cpu(
                    problem, parity_config
                ),
            ),
            "parity-cuda": capture(
                "parity-cuda",
                "backend_parity_cuda",
                adapt_geometry(geometry, parity_domain),
                solver=lambda problem: solve_material_problem_warp(
                    problem, device="cuda", config=parity_config
                ),
            ),
        }
        assert_frozen()
        runs = {study_id: _load_checkpoint(path) for study_id, path in paths.items()}
        print(f"{geometry.config_id}: strict replay", flush=True)
        evidence = assess_publication(
            runs["base"],
            domain_expansions=(runs["domain-1"], runs["domain-2"]),
            mesh_fine=runs["mesh-fine"],
            mesh_third=runs["mesh-third"],
            alignment_sweeps=(runs["alignment-1"],),
            equivalent_base=runs["equivalent-base"],
            equivalent_fine=runs["equivalent-fine"],
            parity_cpu=runs["parity-cpu"],
            parity_cuda=runs["parity-cuda"],
            qualification=qualification,
        )
        print(f"{geometry.config_id}: artifact", flush=True)
        artifact = material_field_artifact(
            base,
            domain_expansions=(),
            mesh_fine=base,
            mesh_third=base,
            alignment_sweeps=(),
            equivalent_base=base,
            equivalent_fine=base,
            parity_cpu=base,
            parity_cuda=base,
            downsample_stride=4,
            precomputed_evidence=evidence,
            qualification=qualification,
            validate_value=False,
        )
        name = geometry.config_id.removesuffix("-v1")
        artifact_path = output / f"{name}.material-field.json"
        # The full map uses the exact result that produced the checkpointed
        # base run; no second solve may diverge from the replay-bound bytes.
        file_sha = write_json(artifact_path, artifact, validate_value=False)
        viewer = viewer_contract(artifact, validate_source=False)
        write_json(
            output / f"{name}.viewer.json",
            viewer,
            referenced_artifact=artifact,
            validate_value=False,
        )
        entries_by_config[geometry.config_id] = {
            "config_id": geometry.config_id,
            "artifact": artifact_path.name,
            "artifact_file_sha256": file_sha,
            "artifact_payload_sha256": artifact["integrity"]["payload_sha256"],
            "geometry_sha256": geometry.canonical_sha256,
            "acceptance_status": artifact["acceptance"]["status"],
            "sampled_cell_peak_t": artifact["summary"]["sampled_cell_peak"]["value_t"],
            "axis_bz_peak_t": artifact["summary"]["axis_bz_peak"]["value_t"],
            "relative_true_residual_l2": artifact["diagnostics"][
                "relative_true_residual_l2"
            ],
            "energy_balance_relative": artifact["diagnostics"][
                "energy_balance_relative"
            ],
            "boundary_to_peak_ratios": [
                item["boundary_to_peak_ratio"]
                for item in artifact["acceptance"]["studies"]
                if item["role"] in {"base", "domain_expansion"}
            ],
            "fixed_qois_bz_t": artifact["summary"]["fixed_qois_bz_t"],
            "pm_model_form_comparison": artifact["summary"]["pm_model_form_comparison"],
            "classification": artifact["classification"],
            "qualification": artifact["acceptance"]["qualification"],
        }
        del base, artifact, viewer, evidence, runs
        _cleanup_cuda()
    ordered_ids = [geometry.config_id for geometry in reference_variants()]
    missing = [config_id for config_id in ordered_ids if config_id not in entries_by_config]
    if missing:
        raise RuntimeError(f"manifest is missing design entries: {missing}")
    for config_id in ordered_ids:
        entry = entries_by_config[config_id]
        artifact_path = output / str(entry["artifact"])
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        validate_artifact(artifact, require_accepted=False)
        if artifact["acceptance"]["status"] != "SCREENING_NOT_ACCEPTED":
            raise RuntimeError(
                f"{config_id} unexpectedly left screening-only status"
            )
        if (
            _file_sha256(artifact_path) != entry["artifact_file_sha256"]
            or artifact["integrity"]["payload_sha256"]
            != entry["artifact_payload_sha256"]
        ):
            raise RuntimeError(f"{config_id} outer artifact hash mismatch")
        del artifact
        _cleanup_cuda()
    payload = {
        "schema_version": "cft_revival.material_fields.design_manifest/1.4.0",
        "model_level": "L1b",
        "designs": [entries_by_config[config_id] for config_id in ordered_ids],
        "limitations": [
            "Only artifacts with ACCEPTED_PUBLICATION_EVIDENCE pass strict validation.",
            "Linear recoil/iron simulations are hypothetical, not hardware predictions.",
            "Memory-limited reduced studies do not evaluate high-resolution qualification.",
        ],
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    manifest = {
        **payload,
        "integrity": {
            "algorithm": "sha256",
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "payload_sha256": hashlib.sha256(canonical).hexdigest(),
        },
    }
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design",
        action="append",
        dest="designs",
        help="regenerate only this exact geometry config ID (repeatable)",
    )
    parser.add_argument(
        "--memory-limited",
        action="store_true",
        help="run bounded ten-role screening after failed high-resolution preflight",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write the complete bundle to a staging directory",
    )
    arguments = parser.parse_args()
    main(
        set(arguments.designs) if arguments.designs else None,
        memory_limited=arguments.memory_limited,
        output_dir=arguments.output_dir,
    )
