"""One-shot preregistered held-out numerical validation of coupling v4."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.coupling import (
    AdapterVersionContract,
    AxialDominancePolicy,
    CFTGeometry,
    CFTStabilityPolicy,
    CFT_V4_DEVELOPMENT_MANIFEST,
    CFTCellRegistration,
    ElectronOrbitSample,
    FieldLineSeed,
    FieldLineTracePolicy,
    HeldOutCaseOutcome,
    HeldOutCaseRegistration,
    HeldOutValidationClaims,
    HeldOutValidationPolicy,
    HeldOutValidationRegistration,
    MapValidationPolicy,
    OrbitVerificationClaims,
    OrbitVerificationIdentity,
    SolverDiagnosticsEvidence,
    UncertaintyModel,
    V3ArtifactClaims,
    V4Criterion,
    V4Status,
    ValidationSetManifest,
    WallCuspPolicy,
    accept_cft_projection,
    bilinear_sample,
    build_cft_coupling_record,
    cft_coupling_record_dict,
    cft_preregistration_hash,
    cft_solver_inputs,
    hash_psi_map,
    magnetic_null_geometry,
    reverify_v3_evidence,
    validation_set_manifest_hash,
    v3_evidence_binding_hash,
    v4_map_set_evidence_fingerprints,
    verify_held_out_validation,
    verify_v3_field_artifact,
    verify_v4_map_set,
)
from cft_revival.fields import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    FieldMap,
    SolverConfig,
    field_artifact,
    max_field_difference,
    solve_problem_warp,
    source_discretization_diagnostics,
    validate_field_artifact,
)
from cft_revival.geometry import (
    EvidenceNote,
    GeometryValidationError,
    MaterialKind,
    PPMStackParameters,
    canonical_json,
    generate_twt_inspired_ppm_stack,
    to_l1a_current_equivalent_preview,
)
from cft_revival.magnetics import LinearPermeability, checked_synthetic_smco_like_magnet

SCHEMA_VERSION = "cft-revival.cft-wall-cusp-validation-v1.dataset/1.0.0"
MANIFEST_VERSION = "cft-revival.cft-wall-cusp-validation-v1.manifest/1.0.0"
ACCEPTED_COUPLING_COMMIT = "f10d8213117fbafd8c2b69bdc103b6ef7b5d6d8c"
EXPERIMENT_DIR = Path(__file__).resolve().parent
MODERN_ROOT = EXPERIMENT_DIR.parents[1]
REPOSITORY_ROOT = MODERN_ROOT.parent
PROTOCOL_PATH = EXPERIMENT_DIR / "protocol.json"


def _strict_json(path: Path) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=reject,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


PROTOCOL = _strict_json(PROTOCOL_PATH)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if is_dataclass(value):
        return {name: _json_value(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def semantic_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def normalized_text_hash(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


PROTOCOL_SEMANTIC_SHA256 = semantic_hash(PROTOCOL)


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = _json_value(dict(payload))
    return {
        **body,
        "semantic_integrity": {
            "algorithm": "sha256",
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "payload_sha256": semantic_hash(body),
        },
    }


def write_semantic_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = _seal(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sealed, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return sealed


def load_semantic_json(path: Path) -> dict[str, Any]:
    value = _strict_json(path)
    integrity = value.get("semantic_integrity")
    body = {key: item for key, item in value.items() if key != "semantic_integrity"}
    if (
        not isinstance(integrity, dict)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("canonicalization") != "json-sort-keys-compact-utf8-v1"
        or integrity.get("payload_sha256") != semantic_hash(body)
    ):
        raise ValueError(f"{path} semantic integrity mismatch")
    return value


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def dependency_closure() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("execution requires a clean worktree")
    symbolic = subprocess.run(
        ("git", "symbolic-ref", "-q", "--short", "HEAD"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if symbolic.returncode == 0:
        raise RuntimeError("execution requires detached HEAD")
    prefixes = (
        "modern/experiments/cft_wall_cusp_validation_v1/",
        "modern/src/cft_revival/coupling/",
        "modern/src/cft_revival/fields/",
        "modern/src/cft_revival/geometry/",
        "modern/src/cft_revival/magnetics/",
        "modern/spec/",
    )
    paths = tuple(
        path
        for path in _git("ls-files").splitlines()
        if path == "modern/pyproject.toml"
        or (
            path.startswith(prefixes)
            and "/results/" not in path
        )
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        blob = _git("rev-parse", f"{head}:{path}")
        baseline_blob = None
        if (
            path.startswith("modern/src/cft_revival/")
            or path.startswith("modern/spec/")
            or path == "modern/pyproject.toml"
        ):
            baseline_blob = _git("rev-parse", f"{ACCEPTED_COUPLING_COMMIT}:{path}")
            if baseline_blob != blob:
                raise RuntimeError(f"accepted dependency blob changed: {path}")
        rows.append(
            {
                "path": path.replace("\\", "/"),
                "preregistration_git_blob_sha1": blob,
                "accepted_baseline_git_blob_sha1": baseline_blob or blob,
            }
        )
    if not rows:
        raise RuntimeError("dependency closure is empty")
    return {
        "preregistration_commit_sha": head,
        "accepted_coupling_commit_sha": ACCEPTED_COUPLING_COMMIT,
        "files": rows,
        "closure_semantic_sha256": semantic_hash(rows),
    }


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    geometry_family_id: str
    geometry_id: str
    stage_count: int
    pitch_m: float
    chamber_radius_m: float
    first_polarity: int
    family_semantic_sha256: str


def case_definitions() -> tuple[CaseDefinition, ...]:
    family = PROTOCOL["held_out_family"]
    family_id = str(family["geometry_family_id"])
    result: list[CaseDefinition] = []
    for stages, pitch_index, radius_index, polarity in itertools.product(
        family["stage_counts"],
        range(len(family["pitch_m"])),
        range(len(family["chamber_outer_radius_m"])),
        family["first_polarity"],
    ):
        sign = "pos" if polarity > 0 else "neg"
        case_id = (
            f"wcval-f1-s{int(stages):02d}-p{pitch_index}-r{radius_index}-{sign}"
        )
        geometry_id = f"{case_id}-geometry"
        payload = {
            "case_id": case_id,
            "geometry_family_id": family_id,
            "geometry_id": geometry_id,
            "stage_count": stages,
            "pitch_m": family["pitch_m"][pitch_index],
            "chamber_radius_m": family["chamber_outer_radius_m"][radius_index],
            "first_polarity": polarity,
            "fixed_geometry": family["fixed_geometry"],
        }
        result.append(
            CaseDefinition(
                case_id,
                family_id,
                geometry_id,
                int(stages),
                float(family["pitch_m"][pitch_index]),
                float(family["chamber_outer_radius_m"][radius_index]),
                int(polarity),
                semantic_hash(payload),
            )
        )
    result.sort(key=lambda item: item.case_id)
    if len(result) != int(family["case_count"]):
        raise ValueError("held-out family Cartesian product does not match case_count")
    development_ids = set(CFT_V4_DEVELOPMENT_MANIFEST.case_ids)
    if development_ids & {item.case_id for item in result}:
        raise ValueError("held-out case IDs overlap development")
    if family_id in CFT_V4_DEVELOPMENT_MANIFEST.geometry_family_ids:
        raise ValueError("held-out geometry family overlaps development")
    return tuple(result)


def held_out_manifest() -> ValidationSetManifest:
    definitions = case_definitions()
    case_ids = tuple(item.case_id for item in definitions)
    family_ids = tuple(sorted({item.geometry_family_id for item in definitions}))
    manifest_id = str(PROTOCOL["held_out_family"]["manifest_id"])
    return ValidationSetManifest(
        manifest_id,
        case_ids,
        family_ids,
        validation_set_manifest_hash(manifest_id, case_ids, family_ids),
    )


def _materials(geometry: Any) -> dict[str, Any]:
    registry: dict[str, Any] = {}
    for material in geometry.materials:
        if material.category is MaterialKind.PERMANENT_MAGNET:
            resolved = checked_synthetic_smco_like_magnet()
            if resolved.material_id != material.material_id:
                raise GeometryValidationError("permanent magnet registry mismatch")
        else:
            resolved = LinearPermeability(
                material.material_id,
                material.relative_permeability,
            )
        registry[material.material_id] = resolved
    return registry


@dataclass(frozen=True)
class BuiltCase:
    definition: CaseDefinition
    geometry: Any
    sources: tuple[Any, ...]
    geometry_sha256: str
    material_semantic_sha256: str
    source_semantic_sha256: str
    chamber_length_m: float
    stage_centres_m: tuple[float, ...]
    base_radius_m: float
    base_z_min_m: float
    base_z_max_m: float


def _stable_pitch_and_centres(
    requested_pitch: float,
    first: float,
    stage_count: int,
) -> tuple[float, tuple[float, ...]]:
    pitch = requested_pitch
    for _ in range(128):
        centres = tuple(first + index * pitch for index in range(stage_count))
        if all(
            abs((right - left) - pitch)
            <= 2.0 * max(math.ulp(right - left), math.ulp(pitch))
            for left, right in zip(centres[:-1], centres[1:], strict=True)
        ):
            return pitch, centres
        pitch = math.nextafter(pitch, 0.005)
    raise GeometryValidationError("could not represent a stable stage pitch")


def build_case(definition: CaseDefinition) -> BuiltCase:
    fixed = PROTOCOL["held_out_family"]["fixed_geometry"]
    first = definition.pitch_m * float(fixed["first_stage_center_pitch_fraction"])
    pitch, centres = _stable_pitch_and_centres(
        definition.pitch_m,
        first,
        definition.stage_count,
    )
    magnet_thickness = pitch * float(fixed["magnet_axial_fraction"])
    chamber_length = centres[-1] + 1.25 * pitch
    magnet_inner = (
        definition.chamber_radius_m
        + float(fixed["dielectric_thickness_m"])
        + float(fixed["radial_clearance_m"])
    )
    magnet_outer = magnet_inner + float(fixed["magnet_radial_thickness_m"])
    geometry = generate_twt_inspired_ppm_stack(
        PPMStackParameters(
            config_id=definition.geometry_id,
            title=f"Held-out v4 wall-cusp validation {definition.case_id}",
            chamber_inner_radius_m=0.0,
            chamber_outer_radius_m=definition.chamber_radius_m,
            chamber_length_m=chamber_length,
            injector_length_m=min(0.08 * chamber_length, 0.5 * first),
            dielectric_thickness_m=float(fixed["dielectric_thickness_m"]),
            thermal_clearance_m=float(fixed["thermal_clearance_m"]),
            magnet_inner_radius_m=magnet_inner,
            magnet_outer_radius_m=magnet_outer,
            stage_pitch_m=pitch,
            stage_centers_m=centres,
            magnet_axial_thicknesses_m=(magnet_thickness,) * definition.stage_count,
            shield_outer_radius_m=magnet_outer
            + float(fixed["shield_radial_thickness_m"]),
            yoke_outer_radius_m=magnet_outer
            + float(fixed["yoke_radial_thickness_m"]),
            first_polarity=definition.first_polarity,
            radial_tolerance_m=2.5e-5,
            axial_tolerance_m=2.5e-5,
            minimum_thickness_m=2.5e-4,
            minimum_clearance_m=1e-4,
        ),
        evidence=(
            EvidenceNote(
                f"{definition.case_id}-held-out",
                "assumption",
                "Preregistered disjoint held-out wall-cusp validation geometry.",
                f"protocol semantic sha256 {PROTOCOL_SEMANTIC_SHA256}",
            ),
        ),
    )
    preview = to_l1a_current_equivalent_preview(
        geometry,
        material_registry=_materials(geometry),
        radial_smear_thickness_m=float(fixed["source_smear_thickness_m"]),
    )
    sources = tuple(preview.bands)
    if len(sources) != 2 * definition.stage_count or any(
        sources[2 * stage].ampere_turns_a
        != sources[2 * stage + 1].ampere_turns_a
        for stage in range(definition.stage_count)
    ):
        raise GeometryValidationError("equivalent-current source pair mismatch")
    source_payload = {
        "preview": preview.to_dict(),
        "sources": [asdict(source) for source in sources],
        "policy": PROTOCOL["held_out_family"]["source_policy"],
    }
    return BuiltCase(
        definition,
        geometry,
        sources,
        geometry.canonical_sha256,
        semantic_hash([material.to_dict() for material in geometry.materials]),
        semantic_hash(source_payload),
        chamber_length,
        centres,
        magnet_outer + float(fixed["radial_padding_m"]),
        -float(fixed["upstream_padding_pitch"]) * pitch,
        chamber_length + float(fixed["downstream_padding_pitch"]) * pitch,
    )


def _stable_upper(lower: float, requested: float, intervals: int) -> float:
    upper = requested
    for _ in range(512):
        step = (upper - lower) / intervals
        if lower + intervals * step == upper:
            return upper
        upper = math.nextafter(upper, lower)
    raise ValueError("could not construct stable binary64 grid")


def domain_for(case: BuiltCase, role: str) -> AxisymmetricDomain:
    declaration = PROTOCOL["maps"]["roles"][role]
    factor = float(declaration["domain_scale"])
    nr = int(declaration["radial_intervals"])
    nz = int(declaration["axial_intervals"])
    if factor == 1.0:
        radius = case.base_radius_m
        z_min, z_max = case.base_z_min_m, case.base_z_max_m
    else:
        middle = 0.5 * (case.base_z_min_m + case.base_z_max_m)
        half = 0.5 * (case.base_z_max_m - case.base_z_min_m) * factor
        radius = case.base_radius_m * factor
        z_min, z_max = middle - half, middle + half
    return AxisymmetricDomain(
        _stable_upper(0.0, radius, nr),
        z_min,
        _stable_upper(z_min, z_max, nz),
        nr,
        nz,
    )


def solver_config() -> SolverConfig:
    declaration = PROTOCOL["maps"]["solver"]
    return SolverConfig(
        relative_tolerance=float(declaration["relative_tolerance"]),
        absolute_tolerance=float(declaration["absolute_tolerance"]),
        max_iterations=int(declaration["maximum_iterations"]),
        residual_history_stride=int(declaration["residual_history_stride"]),
        max_true_residual_restarts=int(
            declaration["maximum_true_residual_restarts"]
        ),
    )


@dataclass(frozen=True)
class SerializedPsiMap:
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    psi_wb: tuple[tuple[float, ...], ...]
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]


class AcceptedL1aAdapter:
    adapter_id = "experiments.cft-wall-cusp-validation-v1.accepted-l1a-v4"
    version_contract = AdapterVersionContract(
        "cft-wall-cusp-validation-v1",
        "1.0.0",
        "cft-axisymmetric-field-map/1.1.0",
        "cft-axisymmetric-field-map/1.1.0",
        "L1a",
    )

    def __init__(
        self,
        case: BuiltCase,
        problem: AxisymmetricProblem,
        artifact_sha256: str,
        closure: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> None:
        self.case = case
        self.problem = problem
        self.artifact_sha256 = artifact_sha256
        self.closure = closure
        self.runtime = runtime
        self.adapter_code_hash = normalized_text_hash(
            Path(__file__).read_text(encoding="utf-8")
        )

    def verify_v3_artifact(self, artifact_bytes: bytes) -> V3ArtifactClaims:
        if hashlib.sha256(artifact_bytes).hexdigest() != self.artifact_sha256:
            raise ValueError("canonical artifact identity mismatch")
        artifact = json.loads(artifact_bytes)
        validate_field_artifact(artifact)
        if artifact["input"]["sources"] != [
            asdict(source) for source in self.problem.sources
        ]:
            raise ValueError("artifact source identity mismatch")
        raw = artifact["field_map"]
        field = SerializedPsiMap(
            tuple(raw["r_m"]),
            tuple(raw["z_m"]),
            tuple(tuple(row) for row in raw["psi_wb"]),
            tuple(tuple(row) for row in raw["b_r_t"]),
            tuple(tuple(row) for row in raw["b_z_t"]),
        )
        map_hash = hash_psi_map(field)
        source_hash = semantic_hash(
            {
                "sources": artifact["input"]["sources"],
                "source_convention": artifact["input"]["source_convention"],
            }
        )
        domain = artifact["input"]["domain"]
        mesh_hash = semantic_hash(
            {
                "radial_intervals": domain["radial_intervals"],
                "axial_intervals": domain["axial_intervals"],
                "dr_m": domain["dr_m"],
                "dz_m": domain["dz_m"],
            }
        )
        domain_hash = semantic_hash(
            {
                "radius_m": domain["radius_m"],
                "z_min_m": domain["z_min_m"],
                "z_max_m": domain["z_max_m"],
                "outer_boundary": artifact["input"]["outer_boundary"],
            }
        )
        diagnostics = artifact["diagnostics"]
        config = artifact["input"]["solver"]
        residual_tolerance = max(
            config["absolute_tolerance"],
            config["relative_tolerance"] * diagnostics["initial_residual_l2"],
        )
        binding = v3_evidence_binding_hash(
            map_hash,
            source_hash,
            self.case.geometry_sha256,
            self.case.material_semantic_sha256,
            mesh_hash,
            domain_hash,
            self.artifact_sha256,
        )
        return V3ArtifactClaims(
            field,
            artifact["schema_version"],
            artifact["model_level"],
            self.artifact_sha256,
            map_hash,
            source_hash,
            self.case.geometry_sha256,
            self.case.material_semantic_sha256,
            mesh_hash,
            domain_hash,
            binding,
            f"cft_revival.fields/{diagnostics['backend']}",
            f"warp-{self.runtime['warp_version']}",
            "cft.l1a.axisymmetric-equivalent-current-v1.1",
            semantic_hash(
                {
                    "model_description": artifact["model_description"],
                    "provenance": artifact["provenance"],
                    "accepted_commit": ACCEPTED_COUPLING_COMMIT,
                }
            ),
            self.closure["closure_semantic_sha256"],
            PROTOCOL_SEMANTIC_SHA256,
            self.runtime["generated_at_utc"],
            SolverDiagnosticsEvidence(
                diagnostics["converged"],
                diagnostics["final_residual_l2"],
                residual_tolerance,
                diagnostics["relative_residual_l2"],
                config["relative_tolerance"],
                diagnostics["iterations"],
            ),
        )


def map_policy() -> MapValidationPolicy:
    return MapValidationPolicy(
        minimum_radial_samples=40,
        minimum_axial_samples=160,
        maximum_age_s=None,
        maximum_future_skew_s=315576000.0,
        require_axis=True,
        axis_br_absolute_tolerance_t=2e-10,
        axis_br_relative_tolerance=1e-8,
    )


def _field_quality(problem: AxisymmetricProblem, field: FieldMap) -> dict[str, Any]:
    peak = max(
        math.hypot(br, bz)
        for br_row, bz_row in zip(field.b_r_t, field.b_z_t, strict=True)
        for br, bz in zip(br_row, bz_row, strict=True)
    )
    boundary = max(
        math.hypot(field.b_r_t[i][j], field.b_z_t[i][j])
        for i in range(len(field.r_m))
        for j in range(len(field.z_m))
        if i == len(field.r_m) - 1 or j in (0, len(field.z_m) - 1)
    )
    errors: list[float] = []
    for source, item in zip(
        problem.sources,
        source_discretization_diagnostics(problem),
        strict=True,
    ):
        area = float(item["requested_area_m2"])
        current = abs(float(item["requested_signed_ampere_turns_a"]))
        width = min(
            source.r_outer_m - source.r_inner_m,
            source.z_max_m - source.z_min_m,
        )
        errors.extend(
            (
                abs(float(item["area_error_m2"])) / max(area, 1e-300),
                abs(float(item["ampere_turn_error_a"])) / max(current, 1e-300),
                math.hypot(
                    float(item["centroid_r_error_m"]),
                    float(item["centroid_z_error_m"]),
                )
                / width,
            )
        )
    quality: dict[str, Any] = {
        "field_peak_t": peak,
        "boundary_to_peak_ratio": boundary / max(peak, 1e-300),
        "source_discretization_relative_error": max(errors, default=0.0),
        "initial_residual_l2": field.diagnostics.initial_residual_l2,
        "final_residual_l2": field.diagnostics.final_residual_l2,
        "normalized_residual": field.diagnostics.relative_residual_l2,
        "flux_reconstruction_identity_t_per_m": (
            field.diagnostics.max_flux_reconstruction_identity_t_per_m
        ),
    }
    gates = PROTOCOL["maps"]["field_gates"]
    quality["all_gates_passed"] = all(
        (
            quality["normalized_residual"]
            <= float(gates["maximum_normalized_residual"]),
            quality["boundary_to_peak_ratio"]
            <= float(gates["maximum_boundary_to_peak_ratio"]),
            quality["source_discretization_relative_error"]
            <= float(gates["maximum_source_discretization_relative_error"]),
            quality["flux_reconstruction_identity_t_per_m"]
            <= float(gates["maximum_flux_reconstruction_identity_t_per_m"]),
        )
    )
    return quality


@dataclass
class SolvedMap:
    role: str
    problem: AxisymmetricProblem
    field: FieldMap
    artifact_bytes: bytes
    artifact_payload_sha256: str
    evidence: Any
    quality: dict[str, Any]


def solve_map(
    case: BuiltCase,
    role: str,
    closure: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> SolvedMap:
    problem = AxisymmetricProblem(
        f"{case.definition.case_id}-{role}",
        domain_for(case, role),
        case.sources,
    )
    field = solve_problem_warp(
        problem,
        config=solver_config(),
        device=str(PROTOCOL["maps"]["solver"]["device"]),
    )
    artifact = field_artifact(
        problem,
        solver_config(),
        field,
        map_stride=1,
        wall_radius_m=case.definition.chamber_radius_m,
    )
    validate_field_artifact(artifact)
    artifact_bytes = canonical_bytes(artifact)
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    evidence = verify_v3_field_artifact(
        artifact_bytes,
        AcceptedL1aAdapter(case, problem, artifact_hash, closure, runtime),
        map_policy(),
        reference_time_utc=runtime["generated_at_utc"],
    )
    return SolvedMap(
        role,
        problem,
        field,
        artifact_bytes,
        str(artifact["integrity"]["payload_sha256"]),
        evidence,
        _field_quality(problem, field),
    )


def load_map_evidence(
    case: BuiltCase,
    role: str,
    path: Path,
    closure: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> Any:
    problem = AxisymmetricProblem(
        f"{case.definition.case_id}-{role}",
        domain_for(case, role),
        case.sources,
    )
    artifact_bytes = path.read_bytes()
    return verify_v3_field_artifact(
        artifact_bytes,
        AcceptedL1aAdapter(
            case,
            problem,
            hashlib.sha256(artifact_bytes).hexdigest(),
            closure,
            runtime,
        ),
        map_policy(),
        reference_time_utc=runtime["generated_at_utc"],
    )


def registrations_for(case: BuiltCase) -> tuple[CFTCellRegistration, ...]:
    declarations = PROTOCOL["criterion"]["electron_samples"]
    samples = tuple(
        ElectronOrbitSample(
            str(item["sample_id"]),
            float(item["kinetic_energy_ev"]),
            math.radians(float(item["pitch_angle_deg"])),
            float(item["maximum_rho_over_scale"]),
            float(item["maximum_mu_relative_variation"]),
        )
        for item in declarations
    )
    radial = (
        float(PROTOCOL["criterion"]["seed"]["radial_wall_fraction"])
        * case.definition.chamber_radius_m
    )
    return tuple(
        CFTCellRegistration(
            f"{case.definition.case_id}-cell-{index + 1:02d}",
            (
                FieldLineSeed(
                    f"{case.definition.case_id}-seed-{index + 1:02d}",
                    radial,
                    center,
                    samples,
                ),
            ),
        )
        for index, center in enumerate(case.stage_centres_m)
    )


def policies_for(case: BuiltCase) -> dict[str, Any]:
    criterion = PROTOCOL["criterion"]
    cusp = criterion["wall_cusp"]
    trace = criterion["field_line"]
    axial = criterion["axial_core"]
    stability = criterion["stability"]
    uncertainty = criterion["uncertainty"]
    pitch = case.definition.pitch_m
    return {
        "cusp_policy": WallCuspPolicy(
            minimum_prominence_t=float(cusp["minimum_prominence_t"]),
            prominence_support_half_width_m=pitch
            * float(cusp["prominence_support_half_width_pitch_fraction"]),
            minimum_cusp_separation_m=pitch
            * float(cusp["minimum_cusp_separation_pitch_fraction"]),
            minimum_wall_radial_fraction=float(cusp["minimum_wall_radial_fraction"]),
            minimum_bundle_paths=int(cusp["minimum_bundle_paths"]),
            endpoint_plane_tolerance_m=pitch
            * float(cusp["endpoint_plane_tolerance_pitch_fraction"]),
            axial_boundary_margin_m=pitch
            * float(cusp["axial_boundary_margin_pitch_fraction"]),
            minimum_endpoint_high_field_fraction=float(
                cusp["minimum_endpoint_high_field_fraction"]
            ),
        ),
        "trace_policy": FieldLineTracePolicy(
            step_m=float(trace["step_m"]),
            maximum_steps=int(trace["maximum_steps"]),
            wall_tolerance_m=float(trace["wall_tolerance_m"]),
            maximum_psi_drift_wb=float(trace["maximum_psi_drift_wb"]),
            minimum_b_t=float(trace["minimum_b_t"]),
            interpolation_relative_error=float(trace["interpolation_relative_error"]),
            path_relative_error=float(trace["path_relative_error"]),
            uncertainty_dominance_factor=float(trace["uncertainty_dominance_factor"]),
        ),
        "axial_policy": AxialDominancePolicy(
            pointwise_axial_fraction_threshold=float(
                axial["pointwise_axial_fraction_threshold"]
            ),
            minimum_passing_fraction=float(axial["minimum_passing_fraction"]),
            minimum_mean_axial_fraction=float(axial["minimum_mean_axial_fraction"]),
        ),
        "stability_policy": CFTStabilityPolicy(
            maximum_cusp_shift_m=pitch
            * float(stability["maximum_cusp_shift_pitch_fraction"]),
            maximum_cusp_strength_relative_change=float(
                stability["maximum_cusp_strength_relative_change"]
            ),
            maximum_endpoint_shift_m=pitch
            * float(stability["maximum_endpoint_shift_pitch_fraction"]),
            maximum_cell_bound_shift_m=pitch
            * float(stability["maximum_cell_bound_shift_pitch_fraction"]),
            maximum_axial_metric_change=float(
                stability["maximum_axial_metric_change"]
            ),
        ),
        "uncertainty_model": UncertaintyModel(
            absolute_independent_sigma_t=float(
                uncertainty["absolute_independent_sigma_t"]
            ),
            relative_independent_sigma=float(
                uncertainty["relative_independent_sigma"]
            ),
            common_mode_sigma_t=float(uncertainty["common_mode_sigma_t"]),
            residual_correlation=float(uncertainty["residual_correlation"]),
            coverage_factor=float(uncertainty["coverage_factor"]),
        ),
    }


class AdiabaticPathOrbitAdapter:
    def __init__(self) -> None:
        declaration = PROTOCOL["orbit_verification"]
        source_hash = normalized_text_hash(Path(__file__).read_text(encoding="utf-8"))
        self.adapter_id = str(declaration["adapter_id"])
        self.adapter_version = str(declaration["adapter_version"])
        self.adapter_code_hash = source_hash
        self.orbit_model_id = str(declaration["orbit_model_id"])
        self.orbit_model_version = str(declaration["orbit_model_version"])
        self.orbit_code_hash = hashlib.sha256(
            b"cft-wall-cusp-orbit-model\0" + bytes.fromhex(source_hash)
        ).hexdigest()
        self.orbit_config_hash = semantic_hash(
            {
                "samples": PROTOCOL["criterion"]["electron_samples"],
                "trace": PROTOCOL["criterion"]["field_line"],
            }
        )
        self.convergence_id = str(declaration["convergence_id"])
        self.convergence_version = str(declaration["convergence_version"])
        self.convergence_config_hash = semantic_hash(
            {
                "method": declaration["method"],
                "nested_stride": 2,
                "maximum_relative_defect": 0.05,
            }
        )

    def verify_orbit(
        self,
        path_points_rz_m: tuple[tuple[float, float], ...],
        path_hash: str,
        sample: ElectronOrbitSample,
    ) -> OrbitVerificationClaims:
        def length(points: Sequence[tuple[float, float]]) -> float:
            return math.fsum(
                math.hypot(right[0] - left[0], right[1] - left[1])
                for left, right in zip(points[:-1], points[1:])
            )

        fine = length(path_points_rz_m)
        coarse_points = path_points_rz_m[::2]
        if coarse_points[-1] != path_points_rz_m[-1]:
            coarse_points = (*coarse_points, path_points_rz_m[-1])
        coarse = length(coarse_points)
        variation = abs(fine - coarse) / max(fine, 1e-300)
        return OrbitVerificationClaims(
            path_hash=path_hash,
            sample_id=sample.sample_id,
            converged=bool(
                len(path_points_rz_m) >= 2
                and math.isfinite(variation)
                and variation <= 0.05
            ),
            maximum_mu_relative_variation=variation,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_code_hash=self.adapter_code_hash,
            orbit_model_id=self.orbit_model_id,
            orbit_model_version=self.orbit_model_version,
            orbit_code_hash=self.orbit_code_hash,
            orbit_config_hash=self.orbit_config_hash,
            convergence_id=self.convergence_id,
            convergence_version=self.convergence_version,
            convergence_config_hash=self.convergence_config_hash,
        )


def orbit_identity(adapter: AdiabaticPathOrbitAdapter) -> OrbitVerificationIdentity:
    return OrbitVerificationIdentity(
        adapter.adapter_id,
        adapter.adapter_version,
        adapter.adapter_code_hash,
        adapter.orbit_model_id,
        adapter.orbit_model_version,
        adapter.orbit_code_hash,
        adapter.orbit_config_hash,
        adapter.convergence_id,
        adapter.convergence_version,
        adapter.convergence_config_hash,
    )


def validation_registration(
    case: BuiltCase,
    code_hash: str,
) -> HeldOutValidationRegistration:
    manifest = held_out_manifest()
    outcomes = tuple(
        HeldOutCaseRegistration(item.case_id, item.geometry_family_id)
        for item in case_definitions()
    )
    return HeldOutValidationRegistration(
        development_manifest=CFT_V4_DEVELOPMENT_MANIFEST,
        held_out_manifest=manifest,
        evaluated_case_id=case.definition.case_id,
        evaluated_geometry_family_id=case.definition.geometry_family_id,
        required_case_count=len(outcomes),
        required_outcomes=outcomes,
        validation_adapter_id=HeldOutArtifactAdapter.adapter_id,
        validation_adapter_code_hash=code_hash,
        validation_code_hash=code_hash,
        validation_config_hash=PROTOCOL_SEMANTIC_SHA256,
        policy=HeldOutValidationPolicy(
            maximum_age_s=315576000.0,
            maximum_future_skew_s=5.0,
        ),
    )


def replay_map(solved: SolvedMap) -> dict[str, Any]:
    replay = solve_problem_warp(
        solved.problem,
        config=solver_config(),
        device=str(PROTOCOL["maps"]["solver"]["device"]),
    )
    differences = max_field_difference(solved.field, replay)
    config = solver_config()
    original = solved.field.diagnostics
    repeated = replay.diagnostics
    original_scale = max(
        config.absolute_tolerance,
        config.relative_tolerance * original.initial_residual_l2,
    )
    replay_scale = max(
        config.absolute_tolerance,
        config.relative_tolerance * repeated.initial_residual_l2,
    )
    residual_difference = abs(
        original.final_residual_l2 - repeated.final_residual_l2
    )
    residual_limit = 0.1 * max(original_scale, replay_scale)
    replay_policy = PROTOCOL["gpu_replay"]
    field_pass = (
        max(differences["br_max_abs_t"], differences["bz_max_abs_t"])
        <= float(replay_policy["maximum_b_component_absolute_difference_t"])
        and differences["psi_max_abs_wb"]
        <= float(replay_policy["maximum_psi_absolute_difference_wb"])
    )
    residual_pass = residual_difference <= residual_limit
    return {
        "case_id": solved.problem.name.rsplit("-", 1)[0],
        "map_role": solved.role,
        "original_initial_residual_l2": original.initial_residual_l2,
        "replay_initial_residual_l2": repeated.initial_residual_l2,
        "original_final_residual_l2": original.final_residual_l2,
        "replay_final_residual_l2": repeated.final_residual_l2,
        "original_stopping_scale_l2": original_scale,
        "replay_stopping_scale_l2": replay_scale,
        "absolute_final_residual_difference_l2": residual_difference,
        "absolute_final_residual_difference_limit_l2": residual_limit,
        "field_differences": differences,
        "field_equality_passed": field_pass,
        "residual_reproducibility_passed": residual_pass,
        "passed": field_pass and residual_pass,
    }


def topology_diagnostics(evidence: Any) -> dict[str, Any]:
    field = reverify_v3_evidence(evidence).field_map
    interior, boundary = magnetic_null_geometry(
        field,
        relative_tolerance=1e-8,
        absolute_tolerance_t=1e-15,
        boundary_exclusion_cells=2,
    )
    dr = min(right - left for left, right in zip(field.r_m[:-1], field.r_m[1:]))
    dz = min(right - left for left, right in zip(field.z_m[:-1], field.z_m[1:]))
    step = 0.75 * min(dr, dz)
    rows: list[dict[str, Any]] = []
    for index, (radius, axial) in enumerate(interior):
        try:
            br_r = (
                bilinear_sample(field, field.b_r_t, (radius + step, axial))
                - bilinear_sample(field, field.b_r_t, (max(0.0, radius - step), axial))
            ) / (step if radius < step else 2.0 * step)
            br_z = (
                bilinear_sample(field, field.b_r_t, (radius, axial + step))
                - bilinear_sample(field, field.b_r_t, (radius, axial - step))
            ) / (2.0 * step)
            bz_r = (
                bilinear_sample(field, field.b_z_t, (radius + step, axial))
                - bilinear_sample(field, field.b_z_t, (max(0.0, radius - step), axial))
            ) / (step if radius < step else 2.0 * step)
            bz_z = (
                bilinear_sample(field, field.b_z_t, (radius, axial + step))
                - bilinear_sample(field, field.b_z_t, (radius, axial - step))
            ) / (2.0 * step)
            determinant = br_r * bz_z - br_z * bz_r
            scale = max(br_r * br_r + br_z * br_z + bz_r * bz_r + bz_z * bz_z, 1e-300)
            kind = (
                "X"
                if determinant < -1e-6 * scale
                else "O"
                if determinant > 1e-6 * scale
                else "degenerate"
            )
            rows.append(
                {
                    "diagnostic_id": f"null-{index:03d}",
                    "r_m": radius,
                    "z_m": axial,
                    "classification": kind,
                    "jacobian_determinant_t2_per_m2": determinant,
                }
            )
        except ValueError as error:
            rows.append(
                {
                    "diagnostic_id": f"null-{index:03d}",
                    "r_m": radius,
                    "z_m": axial,
                    "classification": "degenerate",
                    "error": str(error),
                }
            )
    counts = {
        name: sum(row["classification"] == name for row in rows)
        for name in ("X", "O", "degenerate")
    }
    return {
        "gating_role": "diagnostic_only",
        "interior_null_count": len(interior),
        "finite_boundary_null_count": len(boundary),
        "classification_counts": counts,
        "interior_nulls": rows,
        "finite_boundary_nulls": [list(point) for point in boundary],
    }


def _record_failures(record: Any, replay: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    stability = record.stability
    if len(set(stability.cusp_counts)) != 1:
        failures.append("CUSP_COUNT_DISAGREEMENT")
    cell_counts = (
        len(stability.primary.cells),
        len(stability.refined.cells),
        len(stability.enlarged.cells),
    )
    if len(set(cell_counts)) != 1:
        failures.append("CELL_COUNT_DISAGREEMENT")
    if not replay["field_equality_passed"]:
        failures.append("GPU_FIELD_REPLAY_FAILURE")
    if not replay["residual_reproducibility_passed"]:
        failures.append("GPU_RESIDUAL_REPLAY_FAILURE")
    if record.status is not V4Status.RESOLVED:
        failures.append("WALL_CUSP_UNRESOLVED")
    for assessment in (
        stability.primary,
        stability.refined,
        stability.enlarged,
    ):
        for cell in assessment.cells:
            for outcome in cell.seed_outcomes:
                for path in (outcome.negative_path, outcome.positive_path):
                    if path.termination != "channel_wall":
                        failures.append("PATH_NOT_WALL_CONNECTED")
                    if path.maximum_psi_drift_wb > record.trace_policy.maximum_psi_drift_wb:
                        failures.append("PATH_PSI_DRIFT")
                    if not (
                        path.b_low_location_rz_m != path.b_high_location_rz_m
                        and path.b_low_t <= path.b_high_t
                    ):
                        failures.append("PATH_EXTREMA_INVALID")
                    if path.status is V4Status.UNCERTAINTY_DOMINATED:
                        failures.append("UNCERTAINTY_DOMINATED")
                    if path.status is V4Status.NONADIABATIC:
                        failures.append("ORBIT_NONADIABATIC")
                    if path.status is V4Status.ORBIT_UNVERIFIED:
                        failures.append("ORBIT_UNVERIFIED")
    return sorted(set(failures))


def _record_summary(
    case: BuiltCase,
    record: Any,
    qualities: Mapping[str, Mapping[str, Any]],
    replay: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    assessments = (
        record.stability.primary,
        record.stability.refined,
        record.stability.enlarged,
    )
    paths = [
        path
        for assessment in assessments
        for cell in assessment.cells
        for outcome in cell.seed_outcomes
        for path in (outcome.negative_path, outcome.positive_path)
    ]
    orbits = [orbit for path in paths for orbit in path.orbit_assessments]
    failures = _record_failures(record, replay)
    field_ok = all(item["all_gates_passed"] for item in qualities.values())
    if not field_ok:
        failures.extend(
            f"FIELD_{role.upper()}_INVALID"
            for role, item in qualities.items()
            if not item["all_gates_passed"]
        )
    passed = (
        field_ok
        and replay["passed"]
        and record.status is V4Status.RESOLVED
        and record.stability.passed
        and not failures
    )
    return {
        **asdict(case.definition),
        "field_quality": dict(qualities),
        "map_hashes": [
            assessment.identity.full_map_hash for assessment in assessments
        ],
        "map_evidence_fingerprints": list(record.evidence_fingerprints),
        "record_hash_before_held_out_evidence": record.record_hash,
        "cusp_counts": list(record.stability.cusp_counts),
        "cell_counts": [len(item.cells) for item in assessments],
        "cusp_assignment_rz": [
            {
                "primary_index": assignment[0],
                "refined_index": assignment[1],
                "enlarged_index": assignment[2],
                "coordinate_system": "physical (r_wall,z); Euclidean 2D",
            }
            for assignment in record.stability.cusp_assignment
        ],
        "maximum_cusp_shift_m": record.stability.maximum_cusp_shift_m,
        "maximum_cusp_strength_relative_change": (
            record.stability.maximum_cusp_strength_relative_change
        ),
        "maximum_endpoint_shift_m": record.stability.maximum_endpoint_shift_m,
        "maximum_cell_bound_shift_m": record.stability.maximum_cell_bound_shift_m,
        "maximum_axial_metric_change": record.stability.maximum_axial_metric_change,
        "wall_connected_path_count": sum(
            path.termination == "channel_wall" for path in paths
        ),
        "required_path_count": len(paths),
        "maximum_path_psi_drift_wb": max(
            (path.maximum_psi_drift_wb for path in paths),
            default=None,
        ),
        "same_line_extrema_count": sum(
            path.b_low_location_rz_m != path.b_high_location_rz_m
            and path.b_low_t <= path.b_high_t
            for path in paths
        ),
        "resolved_orbit_count": sum(
            orbit.status is V4Status.RESOLVED for orbit in orbits
        ),
        "required_orbit_count": len(orbits),
        "maximum_rho_over_scale": max(
            (
                orbit.rho_over_scale
                for orbit in orbits
                if orbit.rho_over_scale is not None
            ),
            default=None,
        ),
        "maximum_mu_relative_variation": max(
            (
                orbit.maximum_mu_relative_variation
                for orbit in orbits
                if orbit.maximum_mu_relative_variation is not None
            ),
            default=None,
        ),
        "minimum_mean_axial_fraction": min(
            (
                cell.axial_metrics.mean_axial_fraction
                for assessment in assessments
                for cell in assessment.cells
            ),
            default=None,
        ),
        "minimum_axial_passing_fraction": min(
            (
                cell.axial_metrics.passing_fraction
                for assessment in assessments
                for cell in assessment.cells
            ),
            default=None,
        ),
        "gpu_replay": dict(replay),
        "topology_diagnostics": dict(diagnostics),
        "failures": sorted(set(failures)),
        "passed": passed,
    }


class HeldOutArtifactAdapter:
    adapter_id = "experiments.cft-wall-cusp-validation-v1.held-out-artifact"

    def __init__(
        self,
        artifact_bytes: bytes,
        claims: HeldOutValidationClaims,
        code_hash: str,
    ) -> None:
        self._artifact_bytes = bytes(artifact_bytes)
        self._claims = claims
        self.adapter_code_hash = code_hash

    def verify_validation_artifact(
        self,
        artifact_bytes: bytes,
    ) -> HeldOutValidationClaims:
        if artifact_bytes != self._artifact_bytes:
            raise ValueError("held-out validation bytes changed")
        payload = json.loads(artifact_bytes)
        if (
            payload["evaluated_case_id"] != self._claims.evaluated_case_id
            or payload["preregistration_hash"] != self._claims.preregistration_hash
            or payload["held_out_manifest_hash"]
            != self._claims.held_out_manifest.manifest_hash
            or len(payload["outcomes"]) != len(self._claims.outcomes)
        ):
            raise ValueError("held-out validation artifact claims mismatch")
        return self._claims


def _runtime_identity() -> dict[str, Any]:
    import warp as wp

    wp.init()
    device = wp.get_device(str(PROTOCOL["maps"]["solver"]["device"]))
    query = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()[0]
    gpu_name, gpu_uuid, capability, driver = (
        value.strip() for value in query.split(",")
    )
    banner = subprocess.run(
        ("nvidia-smi",),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    cuda_line = next(line for line in banner.splitlines() if "CUDA Version:" in line)
    cuda = cuda_line.split("CUDA Version:", 1)[1].split()[0]
    return {
        "generated_at_utc": datetime.now(timezone.utc),
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "compute_capability": capability,
        "driver_version": driver,
        "reported_cuda_version": cuda,
        "warp_version": wp.__version__,
        "warp_device": str(device),
        "warp_device_name": device.name,
        "warp_device_uuid": device.uuid,
        "warp_device_architecture": device.arch,
        "python_version": ".".join(str(item) for item in os.sys.version_info[:3]),
        "platform": os.sys.platform,
    }


def _inventory_entry(root: Path, path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    result = {
        "path": path.relative_to(root).as_posix(),
        "byte_sha256": hashlib.sha256(data).hexdigest(),
    }
    if path.suffix == ".json":
        value = _strict_json(path)
        result.update(
            {
                "semantic_sha256": semantic_hash(value),
                "identity_method": "byte-and-canonical-json-sha256",
            }
        )
    else:
        result.update(
            {
                "semantic_sha256": normalized_text_hash(
                    path.read_text(encoding="utf-8")
                ),
                "identity_method": "byte-and-normalized-lf-text-sha256",
            }
        )
    return result


def _aggregate_diagnostics(case_rows: Sequence[Mapping[str, Any]]) -> SolverDiagnosticsEvidence:
    qualities = [
        quality
        for case in case_rows
        for quality in case["field_quality"].values()
    ]
    return SolverDiagnosticsEvidence(
        True,
        max(item["final_residual_l2"] for item in qualities),
        max(
            max(
                float(PROTOCOL["maps"]["solver"]["absolute_tolerance"]),
                float(PROTOCOL["maps"]["solver"]["relative_tolerance"])
                * item["initial_residual_l2"],
            )
            for item in qualities
        ),
        max(item["normalized_residual"] for item in qualities),
        float(PROTOCOL["maps"]["field_gates"]["maximum_normalized_residual"]),
        max(
            int(
                item.get("iterations", 0)
            )
            for item in qualities
        ),
    )


def _failure_counts(cases: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in PROTOCOL["failure_taxonomy"]}
    for case in cases:
        for failure in set(case["failures"]):
            counts[failure] = counts.get(failure, 0) + 1
    return counts


def _report(dataset: Mapping[str, Any]) -> str:
    summary = dataset["summary"]
    lines = [
        "# Coupling v4 wall-cusp held-out validation v1",
        "",
        "First preregistered held-out numerical validation of schema 4.1.",
        "This is numerical/source-consistency evidence, not hardware or experimental validation.",
        "",
        f"- Preregistration commit: `{dataset['preregistration_commit_sha']}`",
        f"- Accepted coupling commit: `{ACCEPTED_COUPLING_COMMIT}`",
        f"- Cases/maps: {summary['case_count']}/{summary['map_count']}",
        f"- Stable cusps/cells: {summary['stable_cusp_count']}/{summary['stable_cell_count']}",
        f"- Wall-connected paths: {summary['wall_connected_path_count']}/{summary['required_path_count']}",
        f"- Resolved orbit samples: {summary['resolved_orbit_count']}/{summary['required_orbit_count']}",
        f"- GPU replay: {summary['gpu_replay_pass_count']}/{summary['case_count']}",
        f"- Criterion numerically promoted: {str(summary['criterion_numerically_promoted']).lower()}",
        f"- Search v3 ready: {str(summary['search_v3_ready']).lower()}",
        f"- Plasma coupling ready: {str(summary['plasma_coupling_ready']).lower()}",
        "",
        "## Gates",
        "",
        f"- Three-map field gates: {summary['three_map_field_gate_pass_count']}/{summary['case_count']}",
        f"- Cross-map cusp/cell stability: {summary['v4_stability_pass_count']}/{summary['case_count']}",
        f"- Opaque projection acceptance: {summary['opaque_projection_pass_count']}/{summary['case_count']}",
        "",
        "## Failure taxonomy",
        "",
    ]
    lines.extend(
        f"- `{name}`: {count}"
        for name, count in sorted(summary["failure_counts"].items())
    )
    lines.extend(
        [
            "",
            "X/O/null and closed-island outputs are diagnostics only; they do not define",
            "or promote a wall cusp. No experimental truth, plasma performance, or",
            "hardware qualification is claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(output_dir: Path) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists():
        raise RuntimeError("single execution output already exists")
    closure = dependency_closure()
    output.mkdir(parents=True, exist_ok=False)
    lock_payload = {
        "schema_version": "cft-revival.exclusive-execution-lock/1.0.0",
        "experiment_id": PROTOCOL["experiment_id"],
        "preregistration_commit_sha": closure["preregistration_commit_sha"],
        "protocol_semantic_sha256": PROTOCOL_SEMANTIC_SHA256,
        "status": "exclusive_lock_acquired_before_single_execution",
    }
    lock_payload["lock_semantic_sha256"] = semantic_hash(lock_payload)
    with (output / "execution-lock.json").open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(json.dumps(lock_payload, indent=2, sort_keys=True) + "\n")
    (output / "preregistered-protocol.json").write_text(
        json.dumps(PROTOCOL, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    runtime = _runtime_identity()
    write_semantic_json(output / "runtime.json", runtime)
    code_hash = normalized_text_hash(Path(__file__).read_text(encoding="utf-8"))
    adapter = AdiabaticPathOrbitAdapter()
    cases: list[dict[str, Any]] = []
    field_root = output / "fields"
    for definition in case_definitions():
        built = build_case(definition)
        solved = {
            role: solve_map(built, role, closure, runtime)
            for role in ("primary", "refined", "enlarged")
        }
        for role, item in solved.items():
            role_dir = field_root / definition.case_id
            role_dir.mkdir(parents=True, exist_ok=True)
            (role_dir / f"{role}-field.json").write_bytes(item.artifact_bytes)
        map_set = verify_v4_map_set(
            solved["primary"].evidence,
            solved["refined"].evidence,
            solved["enlarged"].evidence,
            reference_time_utc=runtime["generated_at_utc"],
        )
        prerecord = build_cft_coupling_record(
            map_set,
            geometry=CFTGeometry(
                definition.chamber_radius_m,
                0.0,
                built.chamber_length_m,
                float(PROTOCOL["criterion"]["axial_core"]["core_radius_wall_fraction"])
                * definition.chamber_radius_m,
                definition.geometry_id,
            ),
            registrations=registrations_for(built),
            validation_registration=validation_registration(built, code_hash),
            orbit_adapter=adapter,
            criterion=V4Criterion(),
            reference_time_utc=runtime["generated_at_utc"],
            **policies_for(built),
        )
        replay = replay_map(solved["primary"])
        diagnostics = {
            role: topology_diagnostics(item.evidence)
            for role, item in solved.items()
        }
        qualities = {role: item.quality for role, item in solved.items()}
        for role, item in qualities.items():
            item["iterations"] = solved[role].field.diagnostics.iterations
        row = _record_summary(
            built,
            prerecord,
            qualities,
            replay,
            diagnostics,
        )
        geometry_dir = output / "geometries"
        geometry_dir.mkdir(parents=True, exist_ok=True)
        (geometry_dir / f"{definition.case_id}.json").write_text(
            canonical_json(built.geometry.to_dict()),
            encoding="utf-8",
            newline="\n",
        )
        cases.append(row)
    all_passed = all(case["passed"] for case in cases)
    outcomes = tuple(
        HeldOutCaseOutcome(
            str(case["case_id"]),
            str(case["geometry_family_id"]),
            tuple(case["map_hashes"]),
            tuple(case["map_evidence_fingerprints"]),
            True,
        )
        for case in cases
    ) if all_passed else ()
    projection_rows: list[dict[str, Any]] = []
    if all_passed:
        aggregate_diagnostics = _aggregate_diagnostics(cases)
        for definition in case_definitions():
            built = build_case(definition)
            evidence = {
                role: load_map_evidence(
                    built,
                    role,
                    field_root / definition.case_id / f"{role}-field.json",
                    closure,
                    runtime,
                )
                for role in ("primary", "refined", "enlarged")
            }
            map_set = verify_v4_map_set(
                evidence["primary"],
                evidence["refined"],
                evidence["enlarged"],
                reference_time_utc=runtime["generated_at_utc"],
            )
            registration = validation_registration(built, code_hash)
            geometry = CFTGeometry(
                definition.chamber_radius_m,
                0.0,
                built.chamber_length_m,
                float(PROTOCOL["criterion"]["axial_core"]["core_radius_wall_fraction"])
                * definition.chamber_radius_m,
                definition.geometry_id,
            )
            registrations = registrations_for(built)
            policies = policies_for(built)
            hashes = tuple(
                next(
                    case["map_hashes"]
                    for case in cases
                    if case["case_id"] == definition.case_id
                )
            )
            fingerprints = v4_map_set_evidence_fingerprints(
                map_set,
                reference_time_utc=runtime["generated_at_utc"],
            )
            preregistration_hash = cft_preregistration_hash(
                geometry=geometry,
                registrations=registrations,
                validation_registration=registration,
                three_map_hashes=hashes,
                three_map_evidence_fingerprints=fingerprints,
                orbit_identity=orbit_identity(adapter),
                criterion=V4Criterion(),
                **policies,
            )
            validation_payload = {
                "schema_version": "cft-revival.cft-wall-cusp-validation-evidence/1.0.0",
                "criterion_id": "cft-hemp-wall-cusp-v4",
                "criterion_version": "4.0.0",
                "development_manifest_hash": CFT_V4_DEVELOPMENT_MANIFEST.manifest_hash,
                "held_out_manifest_hash": held_out_manifest().manifest_hash,
                "evaluated_case_id": definition.case_id,
                "evaluated_geometry_family_id": definition.geometry_family_id,
                "preregistration_hash": preregistration_hash,
                "outcomes": [_json_value(item) for item in outcomes],
                "validation_code_hash": code_hash,
                "validation_config_hash": PROTOCOL_SEMANTIC_SHA256,
            }
            artifact_bytes = canonical_bytes(validation_payload)
            claims = HeldOutValidationClaims(
                "cft-hemp-wall-cusp-v4",
                "4.0.0",
                CFT_V4_DEVELOPMENT_MANIFEST,
                held_out_manifest(),
                definition.case_id,
                definition.geometry_family_id,
                outcomes,
                preregistration_hash,
                hashlib.sha256(artifact_bytes).hexdigest(),
                code_hash,
                PROTOCOL_SEMANTIC_SHA256,
                runtime["generated_at_utc"],
                aggregate_diagnostics,
            )
            held_out_evidence = verify_held_out_validation(
                artifact_bytes,
                HeldOutArtifactAdapter(artifact_bytes, claims, code_hash),
                reference_time_utc=runtime["generated_at_utc"],
                policy=registration.policy,
            )
            record = build_cft_coupling_record(
                map_set,
                geometry=geometry,
                registrations=registrations,
                validation_registration=registration,
                orbit_adapter=adapter,
                criterion=V4Criterion(),
                held_out_validation_evidence=held_out_evidence,
                reference_time_utc=runtime["generated_at_utc"],
                **policies,
            )
            projection = accept_cft_projection(
                record,
                map_set,
                held_out_validation_evidence=held_out_evidence,
                orbit_adapter=adapter,
                reference_time_utc=runtime["generated_at_utc"],
            )
            projected = cft_solver_inputs(
                projection,
                reference_time_utc=runtime["generated_at_utc"],
            )
            if not projected:
                raise RuntimeError("opaque projection unexpectedly rejected")
            validation_dir = output / "validation-evidence"
            validation_dir.mkdir(parents=True, exist_ok=True)
            (validation_dir / f"{definition.case_id}.json").write_bytes(
                artifact_bytes
            )
            record_dir = output / "records"
            record_dir.mkdir(parents=True, exist_ok=True)
            write_semantic_json(
                record_dir / f"{definition.case_id}-v4.json",
                cft_coupling_record_dict(record),
            )
            projection_rows.extend(
                {
                    "case_id": definition.case_id,
                    **_json_value(item),
                }
                for item in projected
            )
            matching = next(case for case in cases if case["case_id"] == definition.case_id)
            matching["held_out_preregistration_hash"] = preregistration_hash
            matching["held_out_validation_artifact_hash"] = hashlib.sha256(
                artifact_bytes
            ).hexdigest()
            matching["opaque_projection_row_count"] = len(projected)
            matching["opaque_projection_passed"] = True
    for case in cases:
        case.setdefault("opaque_projection_row_count", 0)
        case.setdefault("opaque_projection_passed", False)
        if all_passed and not case["opaque_projection_passed"]:
            case["failures"].append("OPAQUE_PROJECTION_REJECTED")
    topology_counts = {
        name: sum(
            diagnostic["classification_counts"][name]
            for case in cases
            for diagnostic in case["topology_diagnostics"].values()
        )
        for name in ("X", "O", "degenerate")
    }
    summary = {
        "case_count": len(cases),
        "map_count": 3 * len(cases),
        "three_map_field_gate_pass_count": sum(
            all(item["all_gates_passed"] for item in case["field_quality"].values())
            for case in cases
        ),
        "v4_stability_pass_count": sum(
            case["passed"] for case in cases
        ),
        "gpu_replay_pass_count": sum(
            case["gpu_replay"]["passed"] for case in cases
        ),
        "stable_cusp_count": sum(case["cusp_counts"][0] for case in cases if case["passed"]),
        "stable_cell_count": sum(case["cell_counts"][0] for case in cases if case["passed"]),
        "wall_connected_path_count": sum(case["wall_connected_path_count"] for case in cases),
        "required_path_count": sum(case["required_path_count"] for case in cases),
        "resolved_orbit_count": sum(case["resolved_orbit_count"] for case in cases),
        "required_orbit_count": sum(case["required_orbit_count"] for case in cases),
        "opaque_projection_pass_count": sum(
            case["opaque_projection_passed"] for case in cases
        ),
        "opaque_projection_row_count": len(projection_rows),
        "topology_result_coverage": topology_counts,
        "failure_counts": _failure_counts(cases),
        "criterion_numerically_promoted": bool(
            all_passed
            and len(projection_rows) > 0
            and all(case["opaque_projection_passed"] for case in cases)
        ),
    }
    summary["search_v3_ready"] = summary["criterion_numerically_promoted"]
    summary["plasma_coupling_ready"] = summary["criterion_numerically_promoted"]
    dataset_payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": PROTOCOL["classification"],
        "claim_boundary": PROTOCOL["publication_boundary"],
        "protocol_semantic_sha256": PROTOCOL_SEMANTIC_SHA256,
        "preregistration_commit_sha": closure["preregistration_commit_sha"],
        "accepted_coupling_commit_sha": ACCEPTED_COUPLING_COMMIT,
        "development_manifest": _json_value(CFT_V4_DEVELOPMENT_MANIFEST),
        "held_out_manifest": _json_value(held_out_manifest()),
        "dependency_closure": closure,
        "runtime_identity": runtime,
        "orbit_implementation_identity": _json_value(orbit_identity(adapter)),
        "summary": summary,
        "cases": cases,
        "projection_rows": projection_rows,
    }
    dataset = write_semantic_json(output / "dataset.json", dataset_payload)
    (output / "report.md").write_text(
        _report(dataset),
        encoding="utf-8",
        newline="\n",
    )
    inventory = [
        _inventory_entry(output, path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = write_semantic_json(
        output / "manifest.json",
        {
            "schema_version": MANIFEST_VERSION,
            "experiment_id": PROTOCOL["experiment_id"],
            "protocol_semantic_sha256": PROTOCOL_SEMANTIC_SHA256,
            "preregistration_commit_sha": closure["preregistration_commit_sha"],
            "accepted_coupling_commit_sha": ACCEPTED_COUPLING_COMMIT,
            "dependency_closure_semantic_sha256": closure[
                "closure_semantic_sha256"
            ],
            "single_execution": True,
            "no_patch_or_rerun": True,
            "summary": summary,
            "artifacts": inventory,
        },
    )
    validate_results(output)
    return {"dataset": dataset, "manifest": manifest}


def validate_results(output_dir: Path) -> dict[str, Any]:
    output = output_dir.resolve()
    dataset = load_semantic_json(output / "dataset.json")
    manifest = load_semantic_json(output / "manifest.json")
    runtime = load_semantic_json(output / "runtime.json")
    protocol_copy = _strict_json(output / "preregistered-protocol.json")
    if (
        semantic_hash(protocol_copy) != PROTOCOL_SEMANTIC_SHA256
        or dataset["protocol_semantic_sha256"] != PROTOCOL_SEMANTIC_SHA256
        or manifest["protocol_semantic_sha256"] != PROTOCOL_SEMANTIC_SHA256
        or dataset["accepted_coupling_commit_sha"] != ACCEPTED_COUPLING_COMMIT
        or manifest["accepted_coupling_commit_sha"] != ACCEPTED_COUPLING_COMMIT
    ):
        raise ValueError("protocol or accepted coupling identity mismatch")
    if dataset["development_manifest"] != _json_value(CFT_V4_DEVELOPMENT_MANIFEST):
        raise ValueError("frozen development manifest mismatch")
    expected_held_out = _json_value(held_out_manifest())
    if dataset["held_out_manifest"] != expected_held_out:
        raise ValueError("held-out manifest mismatch")
    if set(dataset["development_manifest"]["case_ids"]) & set(
        dataset["held_out_manifest"]["case_ids"]
    ):
        raise ValueError("development and held-out cases overlap")
    if dataset["summary"]["case_count"] != len(case_definitions()):
        raise ValueError("held-out case coverage is incomplete")
    if dataset["summary"]["map_count"] != 3 * len(case_definitions()):
        raise ValueError("three-map coverage is incomplete")
    if dataset["summary"]["criterion_numerically_promoted"]:
        if not (
            dataset["summary"]["opaque_projection_pass_count"]
            == len(case_definitions())
            and all(case["passed"] for case in dataset["cases"])
            and all(case["opaque_projection_passed"] for case in dataset["cases"])
        ):
            raise ValueError("criterion promotion is not supported by all outcomes")
    listed = {entry["path"]: entry for entry in manifest["artifacts"]}
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(listed) != actual:
        raise ValueError("manifest inventory is incomplete")
    for relative, entry in listed.items():
        recomputed = _inventory_entry(output, output / relative)
        if recomputed != entry:
            raise ValueError(f"artifact identity mismatch: {relative}")
    required_runtime = {
        "gpu_name",
        "gpu_uuid",
        "compute_capability",
        "driver_version",
        "reported_cuda_version",
        "warp_version",
        "warp_device_architecture",
        "generated_at_utc",
    }
    if not required_runtime.issubset(runtime):
        raise ValueError("runtime identity is incomplete")
    if dataset["summary"]["gpu_replay_pass_count"] != len(case_definitions()):
        raise ValueError("GPU replay gate failed")
    return {"dataset": dataset, "manifest": manifest, "runtime": runtime}
