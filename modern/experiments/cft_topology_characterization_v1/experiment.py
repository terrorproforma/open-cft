"""Preregistered developmental stage/null/cell topology characterization."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import shutil
import statistics
import subprocess
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.coupling import (
    AdapterVersionContract,
    FluxSurfacePolicy,
    MapValidationPolicy,
    SolverDiagnosticsEvidence,
    V3ArtifactClaims,
    bilinear_sample,
    magnetic_null_geometry,
    reverify_v3_evidence,
    trace_flux_contours,
    v3_evidence_binding_hash,
    verify_v3_field_artifact,
)
from cft_revival.coupling.v3_models import ValidatedPsiMap
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

SCHEMA_VERSION = "cft-revival.cft-topology-characterization-v1.dataset/1.0.0"
MANIFEST_VERSION = "cft-revival.cft-topology-characterization-v1.manifest/1.0.0"
ACCEPTED_COUPLING_COMMIT = "f80a360fd740a30017cdac1874cedbfa2806874a"
ACCEPTANCE_TIME_UTC = datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc)
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

    loaded = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=reject,
    )
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain one object")
    return loaded


PROTOCOL = _strict_json(PROTOCOL_PATH)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
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
        "modern/experiments/cft_topology_characterization_v1/",
        "modern/src/cft_revival/coupling/",
        "modern/src/cft_revival/fields/",
        "modern/src/cft_revival/geometry/",
        "modern/src/cft_revival/magnetics/",
        "modern/spec/",
    )
    paths = tuple(
        path
        for path in _git("ls-files").splitlines()
        if path == "modern/pyproject.toml" or path.startswith(prefixes)
    )
    rows: list[dict[str, str]] = []
    for path in paths:
        blob = _git("rev-parse", f"{head}:{path}")
        baseline_blob: str | None = None
        if path.startswith("modern/src/") or path.startswith("modern/spec/"):
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
    return {
        "preregistration_commit_sha": head,
        "accepted_coupling_v3_commit_sha": ACCEPTED_COUPLING_COMMIT,
        "files": rows,
        "closure_semantic_sha256": semantic_hash(rows),
    }


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    stage_count: int
    pitch_m: float
    chamber_radius_m: float
    first_polarity: int
    family_semantic_sha256: str


def case_definitions() -> tuple[CaseDefinition, ...]:
    family = PROTOCOL["families"]
    result: list[CaseDefinition] = []
    for stages, pitch_index, radius_index, polarity in itertools.product(
        family["stage_counts"],
        range(len(family["pitch_m"])),
        range(len(family["chamber_outer_radius_m"])),
        family["first_polarity"],
    ):
        pitch = float(family["pitch_m"][pitch_index])
        radius = float(family["chamber_outer_radius_m"][radius_index])
        sign = "pos" if polarity > 0 else "neg"
        case_id = f"topology-s{int(stages):02d}-p{pitch_index}-r{radius_index}-{sign}"
        payload = {
            "case_id": case_id,
            "stage_count": stages,
            "pitch_m": pitch,
            "chamber_radius_m": radius,
            "first_polarity": polarity,
            "fixed_geometry": family["fixed_geometry"],
        }
        result.append(
            CaseDefinition(
                case_id,
                int(stages),
                pitch,
                radius,
                int(polarity),
                semantic_hash(payload),
            )
        )
    result.sort(key=lambda item: item.case_id)
    if len(result) != int(family["case_count"]):
        raise ValueError("family Cartesian product does not match case_count")
    return tuple(result)


def _materials(geometry: Any) -> dict[str, Any]:
    registry: dict[str, Any] = {}
    for material in geometry.materials:
        if material.category is MaterialKind.PERMANENT_MAGNET:
            resolved = checked_synthetic_smco_like_magnet()
            if resolved.material_id != material.material_id:
                raise GeometryValidationError("permanent magnet registry mismatch")
        else:
            resolved = LinearPermeability(
                material.material_id, material.relative_permeability
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
    magnet_inner_radius_m: float
    magnet_outer_radius_m: float
    stage_centres_m: tuple[float, ...]
    magnet_axial_thickness_m: float
    base_radius_m: float
    base_z_min_m: float
    base_z_max_m: float


def _stable_pitch_and_centres(
    requested_pitch: float, first: float, stage_count: int
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
    fixed = PROTOCOL["families"]["fixed_geometry"]
    first = float(fixed["first_stage_center_m"])
    pitch, centres = _stable_pitch_and_centres(
        definition.pitch_m, first, definition.stage_count
    )
    magnet_thickness = (
        pitch * float(fixed["magnet_axial_fraction"])
    )
    chamber_length = centres[-1] + 1.25 * pitch
    magnet_inner = (
        definition.chamber_radius_m
        + float(fixed["dielectric_thickness_m"])
        + float(fixed["radial_clearance_m"])
    )
    magnet_outer = magnet_inner + float(fixed["magnet_radial_thickness_m"])
    geometry = generate_twt_inspired_ppm_stack(
        PPMStackParameters(
            config_id=f"{definition.case_id}-geometry",
            title=f"CFT topology characterization {definition.case_id}",
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
            magnet_axial_thicknesses_m=(magnet_thickness,)
            * definition.stage_count,
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
                f"{definition.case_id}-characterization",
                "assumption",
                "Preregistered developmental stage-to-topology characterization case.",
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
        "policy": PROTOCOL["families"]["source_policy"],
    }
    return BuiltCase(
        definition,
        geometry,
        sources,
        geometry.canonical_sha256,
        semantic_hash([material.to_dict() for material in geometry.materials]),
        semantic_hash(source_payload),
        chamber_length,
        magnet_inner,
        magnet_outer,
        centres,
        magnet_thickness,
        magnet_outer + float(fixed["radial_padding_m"]),
        -float(fixed["upstream_padding_pitch"]) * pitch,
        chamber_length
        + float(fixed["downstream_padding_pitch"]) * pitch,
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


class CharacterizationV3Adapter:
    adapter_id = "experiments.cft-topology-characterization-v1.direct-l1a-v3"
    version_contract = AdapterVersionContract(
        "cft-topology-characterization-v1",
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
        from cft_revival.coupling import hash_psi_map

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
                    "closure": self.closure["closure_semantic_sha256"],
                }
            ),
            self.closure["closure_semantic_sha256"],
            PROTOCOL_SEMANTIC_SHA256,
            ACCEPTANCE_TIME_UTC,
            SolverDiagnosticsEvidence(
                diagnostics["converged"],
                diagnostics["final_residual_l2"],
                residual_tolerance,
                diagnostics["relative_residual_l2"],
                config["relative_tolerance"],
                diagnostics["iterations"],
            ),
        )


def _map_policy() -> MapValidationPolicy:
    return MapValidationPolicy(
        minimum_radial_samples=40,
        minimum_axial_samples=120,
        maximum_age_s=None,
        maximum_future_skew_s=315576000.0,
        require_axis=True,
        axis_br_absolute_tolerance_t=2e-10,
        axis_br_relative_tolerance=1e-8,
    )


def _field_quality(problem: AxisymmetricProblem, field: FieldMap) -> dict[str, float]:
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
        problem.sources, source_discretization_diagnostics(problem), strict=True
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
    quality = {
        "field_peak_t": peak,
        "boundary_to_peak_ratio": boundary / max(peak, 1e-300),
        "source_discretization_relative_error": max(errors, default=0.0),
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
    validated: ValidatedPsiMap
    artifact_bytes: bytes
    artifact_semantic_sha256: str
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
        CharacterizationV3Adapter(
            case, problem, artifact_hash, closure, runtime
        ),
        _map_policy(),
        reference_time_utc=ACCEPTANCE_TIME_UTC,
    )
    validated = reverify_v3_evidence(
        evidence, reference_time_utc=ACCEPTANCE_TIME_UTC
    ).field_map
    return SolvedMap(
        role,
        problem,
        field,
        validated,
        artifact_bytes,
        str(artifact["integrity"]["payload_sha256"]),
        evidence,
        _field_quality(problem, field),
    )


def _sample_extended(
    field: ValidatedPsiMap,
    values: tuple[tuple[float, ...], ...],
    r_m: float,
    z_m: float,
    *,
    odd_radial: bool,
) -> float:
    reflected = r_m < 0.0
    result = bilinear_sample(field, values, (abs(r_m), z_m))
    return -result if reflected and odd_radial else result


def _field_vector(field: ValidatedPsiMap, point: tuple[float, float]) -> tuple[float, float]:
    r_m, z_m = point
    return (
        _sample_extended(field, field.b_r_t, r_m, z_m, odd_radial=True),
        _sample_extended(field, field.b_z_t, r_m, z_m, odd_radial=False),
    )


def _jacobian(
    field: ValidatedPsiMap, point: tuple[float, float], step: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    r_m, z_m = point
    if (
        abs(r_m) + step > field.r_m[-1]
        or z_m - step < field.z_m[0]
        or z_m + step > field.z_m[-1]
    ):
        raise ValueError("Jacobian stencil touches finite map boundary")
    br_plus_r, bz_plus_r = _field_vector(field, (r_m + step, z_m))
    br_minus_r, bz_minus_r = _field_vector(field, (r_m - step, z_m))
    br_plus_z, bz_plus_z = _field_vector(field, (r_m, z_m + step))
    br_minus_z, bz_minus_z = _field_vector(field, (r_m, z_m - step))
    scale = 0.5 / step
    return (
        (
            (br_plus_r - br_minus_r) * scale,
            (br_plus_z - br_minus_z) * scale,
        ),
        (
            (bz_plus_r - bz_minus_r) * scale,
            (bz_plus_z - bz_minus_z) * scale,
        ),
    )


def _frobenius(matrix: Sequence[Sequence[float]]) -> float:
    return math.sqrt(math.fsum(value * value for row in matrix for value in row))


def _topological_index(
    field: ValidatedPsiMap,
    point: tuple[float, float],
    radius: float,
    count: int,
) -> tuple[float, float]:
    angles: list[float] = []
    minimum = float("inf")
    for index in range(count + 1):
        angle = 2.0 * math.pi * index / count
        sample = (
            point[0] + radius * math.cos(angle),
            point[1] + radius * math.sin(angle),
        )
        br, bz = _field_vector(field, sample)
        minimum = min(minimum, math.hypot(br, bz))
        angles.append(math.atan2(bz, br))
    winding = 0.0
    for left, right in zip(angles[:-1], angles[1:], strict=True):
        delta = right - left
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        winding += delta
    return winding / (2.0 * math.pi), minimum


def local_topology(
    field: ValidatedPsiMap,
    point: tuple[float, float],
    mesh_scale_m: float,
) -> dict[str, Any]:
    policy = PROTOCOL["local_topology"]
    coarse_step = float(policy["coarse_step_mesh_factor"]) * mesh_scale_m
    refined_step = coarse_step * float(policy["refined_step_ratio"])
    try:
        coarse = _jacobian(field, point, coarse_step)
        refined = _jacobian(field, point, refined_step)
        difference = tuple(
            tuple(refined[i][j] - coarse[i][j] for j in range(2))
            for i in range(2)
        )
        relative_change = _frobenius(difference) / max(
            _frobenius(coarse), _frobenius(refined), 1e-300
        )
        determinant = (
            refined[0][0] * refined[1][1]
            - refined[0][1] * refined[1][0]
        )
        determinant_scale = max(_frobenius(refined) ** 2, 1e-300)
        determinant_threshold = float(
            policy["determinant_relative_tolerance"]
        ) * determinant_scale
        index, circle_minimum = _topological_index(
            field,
            point,
            float(policy["index_circle_radius_mesh_factor"]) * mesh_scale_m,
            int(policy["index_sample_count"]),
        )
        converged = relative_change <= float(
            policy["maximum_relative_jacobian_change"]
        )
        index_tolerance = float(policy["index_tolerance"])
        if converged and determinant < -determinant_threshold and abs(index + 1.0) <= index_tolerance:
            classification = "X"
        elif converged and determinant > determinant_threshold and abs(index - 1.0) <= index_tolerance:
            classification = "O"
        else:
            classification = "degenerate"
        return {
            "classification": classification,
            "coarse_jacobian_t_per_m": coarse,
            "refined_jacobian_t_per_m": refined,
            "relative_jacobian_change": relative_change,
            "jacobian_converged": converged,
            "determinant_t2_per_m2": determinant,
            "determinant_threshold_t2_per_m2": determinant_threshold,
            "topological_index": index,
            "index_circle_minimum_field_t": circle_minimum,
            "finite_difference_steps_m": [coarse_step, refined_step],
        }
    except Exception as error:
        return {
            "classification": "degenerate",
            "jacobian_converged": False,
            "error": f"{type(error).__name__}: {error}",
            "finite_difference_steps_m": [coarse_step, refined_step],
        }


def _detection_method(
    field: ValidatedPsiMap,
    point: tuple[float, float],
    tolerance_t: float,
) -> str:
    r_m, z_m = point
    if r_m != 0.0:
        return "bilinear_vector_root"
    nearest = min(range(len(field.z_m)), key=lambda index: abs(field.z_m[index] - z_m))
    magnitude = math.hypot(
        field.b_r_t[0][nearest], field.b_z_t[0][nearest]
    )
    return "axis_grid" if magnitude <= tolerance_t else "axis_sign_change"


def raw_detections(field: ValidatedPsiMap) -> tuple[dict[str, Any], ...]:
    policy = PROTOCOL["root_detection"]
    scale = max(
        math.hypot(br, bz)
        for br_row, bz_row in zip(field.b_r_t, field.b_z_t, strict=True)
        for br, bz in zip(br_row, bz_row, strict=True)
    )
    tolerance = max(
        float(policy["absolute_field_tolerance_t"]),
        float(policy["relative_field_tolerance"]) * scale,
    )
    interior, _ = magnetic_null_geometry(
        field,
        relative_tolerance=float(policy["relative_field_tolerance"]),
        absolute_tolerance_t=float(policy["absolute_field_tolerance_t"]),
        boundary_exclusion_cells=int(policy["finite_box_exclusion_cells"]),
    )
    rows = [
        {
            "r_m": point[0],
            "z_m": point[1],
            "method": _detection_method(field, point, tolerance),
            "finite_box_boundary": False,
            "field_magnitude_t": math.hypot(*_field_vector(field, point)),
        }
        for point in interior
    ]
    nr, nz = len(field.r_m), len(field.z_m)
    for i in range(nr):
        for j in range(nz):
            if i != nr - 1 and j not in (0, nz - 1):
                continue
            magnitude = math.hypot(field.b_r_t[i][j], field.b_z_t[i][j])
            if magnitude <= tolerance:
                rows.append(
                    {
                        "r_m": field.r_m[i],
                        "z_m": field.z_m[j],
                        "method": "finite_box_grid",
                        "finite_box_boundary": True,
                        "field_magnitude_t": magnitude,
                    }
                )
    rows.sort(key=lambda item: (item["z_m"], item["r_m"], item["method"]))
    return tuple(rows)


def cluster_detections(
    detections: Sequence[Mapping[str, Any]], mesh_scale_m: float
) -> tuple[dict[str, Any], ...]:
    tolerance = (
        float(
            PROTOCOL["root_detection"]["clustering"]["tolerance_mesh_factor"]
        )
        * mesh_scale_m
    )
    count = len(detections)
    adjacency = [set([index]) for index in range(count)]
    for left in range(count):
        for right in range(left + 1, count):
            distance = math.hypot(
                float(detections[left]["r_m"]) - float(detections[right]["r_m"]),
                float(detections[left]["z_m"]) - float(detections[right]["z_m"]),
            )
            if distance <= tolerance:
                adjacency[left].add(right)
                adjacency[right].add(left)
    unvisited = set(range(count))
    clusters: list[dict[str, Any]] = []
    while unvisited:
        seed = min(unvisited)
        component: set[int] = set()
        pending = [seed]
        while pending:
            index = pending.pop()
            if index in component:
                continue
            component.add(index)
            pending.extend(adjacency[index] - component)
        unvisited -= component
        members = tuple(detections[index] for index in sorted(component))
        clusters.append(
            {
                "r_m": math.fsum(float(item["r_m"]) for item in members)
                / len(members),
                "z_m": math.fsum(float(item["z_m"]) for item in members)
                / len(members),
                "member_count": len(members),
                "methods": sorted({str(item["method"]) for item in members}),
                "finite_box_boundary": any(
                    bool(item["finite_box_boundary"]) for item in members
                ),
                "members": [dict(item) for item in members],
                "cluster_tolerance_m": tolerance,
            }
        )
    clusters.sort(key=lambda item: (item["z_m"], item["r_m"]))
    for index, cluster in enumerate(clusters):
        cluster["root_id"] = f"root-{index:03d}"
    return tuple(clusters)


def geometry_association(
    case: BuiltCase,
    root: Mapping[str, Any],
    mesh_scale_m: float,
) -> dict[str, Any]:
    r_m, z_m = float(root["r_m"]), float(root["z_m"])
    margin = (
        float(PROTOCOL["physical_domains"]["strict_interior_margin_mesh_cells"])
        * mesh_scale_m
    )
    nearest_stage = min(
        range(case.definition.stage_count),
        key=lambda index: abs(case.stage_centres_m[index] - z_m),
    )
    stage_distance = abs(case.stage_centres_m[nearest_stage] - z_m)
    if root["finite_box_boundary"]:
        zone, reason = "finite_box_boundary", "finite_box_boundary"
    elif z_m <= 0.0 or z_m >= case.chamber_length_m:
        zone, reason = "outside_channel_axial", "outside_channel_axial"
    elif r_m >= case.magnet_inner_radius_m and r_m <= case.magnet_outer_radius_m:
        zone, reason = "magnet_or_current_sheet", "exterior_magnet_or_current_sheet"
    elif r_m > case.definition.chamber_radius_m:
        zone, reason = "yoke_or_material", "yoke_or_material"
    elif r_m >= case.definition.chamber_radius_m - margin:
        zone, reason = "channel_wall_margin", "wall_only"
    elif z_m <= margin or z_m >= case.chamber_length_m - margin:
        zone, reason = "channel_axial_margin", "outside_channel_axial"
    else:
        zone, reason = "plasma_channel", None
    return {
        "zone": zone,
        "nearest_stage_index": nearest_stage,
        "nearest_stage_center_z_m": case.stage_centres_m[nearest_stage],
        "stage_axial_distance_m": stage_distance,
        "inside_plasma_channel": zone == "plasma_channel",
        "geometric_exclusion_reason": reason,
        "channel_margin_m": margin,
    }


def separatrix_connectivity(
    field: ValidatedPsiMap,
    point: tuple[float, float],
    mesh_scale_m: float,
    chamber_radius_m: float,
    chamber_length_m: float,
) -> dict[str, Any]:
    r_m, z_m = point
    psi0 = bilinear_sample(field, field.psi_wb, point)
    radial_index = min(
        range(len(field.r_m)), key=lambda index: abs(field.r_m[index] - r_m)
    )
    axial_index = min(
        range(len(field.z_m)), key=lambda index: abs(field.z_m[index] - z_m)
    )
    stencil = tuple(
        field.psi_wb[i][j]
        for i in range(max(0, radial_index - 2), min(len(field.r_m), radial_index + 3))
        for j in range(max(0, axial_index - 2), min(len(field.z_m), axial_index + 3))
    )
    delta = max(1e-12, 0.02 * (max(stencil) - min(stencil)))
    near_distance = (
        float(PROTOCOL["separatrix"]["near_root_distance_mesh_factor"])
        * mesh_scale_m
    )
    surface_policy = FluxSurfacePolicy(
        psi_absolute_tolerance_wb=1e-12,
        psi_relative_tolerance=1e-8,
        connectivity_tolerance_m=float(
            PROTOCOL["separatrix"]["connectivity_tolerance_m"]
        ),
        boundary_exclusion_cells=2,
        minimum_contour_points=4,
        saddle_tie_policy="reject",
    )
    levels: list[dict[str, Any]] = []
    for label, level in (("minus", psi0 - delta), ("plus", psi0 + delta)):
        try:
            contours = trace_flux_contours(field, level, surface_policy)
        except Exception as error:
            levels.append(
                {
                    "side": label,
                    "psi_wb": level,
                    "error": f"{type(error).__name__}: {error}",
                    "nearby_closed_channel_count": 0,
                    "components": [],
                }
            )
            continue
        components: list[dict[str, Any]] = []
        for index, contour in enumerate(contours):
            distance = min(
                math.hypot(point_r - r_m, point_z - z_m)
                for point_r, point_z in contour.points_rz_m
            )
            channel_contained = all(
                0.0 <= point_r < chamber_radius_m
                and 0.0 < point_z < chamber_length_m
                for point_r, point_z in contour.points_rz_m
            )
            components.append(
                {
                    "component": index,
                    "closed": contour.closed,
                    "simple": contour.simple,
                    "touches_finite_boundary": contour.touches_boundary,
                    "channel_contained": channel_contained,
                    "minimum_root_distance_m": distance,
                    "near_root": distance <= near_distance,
                    "point_count": len(contour.points_rz_m),
                    "maximum_psi_residual_wb": contour.maximum_psi_residual_wb,
                    "topology_reason": contour.topology_reason,
                }
            )
        nearby_closed = sum(
            item["closed"]
            and item["simple"]
            and not item["touches_finite_boundary"]
            and item["channel_contained"]
            and item["near_root"]
            for item in components
        )
        levels.append(
            {
                "side": label,
                "psi_wb": level,
                "nearby_closed_channel_count": nearby_closed,
                "components": components,
            }
        )
    counts = tuple(item["nearby_closed_channel_count"] for item in levels)
    return {
        "root_psi_wb": psi0,
        "probe_delta_wb": delta,
        "levels": levels,
        "has_nearby_closed_channel_surface": any(value > 0 for value in counts),
        "closed_component_count_changes": len(set(counts)) > 1,
        "cell_bounding": any(value > 0 for value in counts)
        and len(set(counts)) > 1,
    }


def characterize_map(case: BuiltCase, solved: SolvedMap) -> dict[str, Any]:
    field = solved.validated
    mesh_scale = max(solved.problem.domain.dr_m, solved.problem.domain.dz_m)
    raw = raw_detections(field)
    clusters = cluster_detections(raw, mesh_scale)
    roots: list[dict[str, Any]] = []
    failures: list[str] = []
    for cluster in clusters:
        point = (float(cluster["r_m"]), float(cluster["z_m"]))
        association = geometry_association(case, cluster, mesh_scale)
        topology = local_topology(field, point, mesh_scale)
        if association["inside_plasma_channel"]:
            connectivity = separatrix_connectivity(
                field,
                point,
                mesh_scale,
                case.definition.chamber_radius_m,
                case.chamber_length_m,
            )
        else:
            connectivity = {
                "not_evaluated_reason": association["geometric_exclusion_reason"],
                "has_nearby_closed_channel_surface": False,
                "closed_component_count_changes": False,
                "cell_bounding": False,
            }
        classification = topology["classification"]
        eligible_cusp = (
            association["inside_plasma_channel"]
            and classification == "X"
            and connectivity["cell_bounding"]
        )
        eligible_cell = (
            association["inside_plasma_channel"]
            and classification == "O"
            and connectivity["has_nearby_closed_channel_surface"]
        )
        if association["geometric_exclusion_reason"] is not None:
            exclusion = association["geometric_exclusion_reason"]
        elif classification == "degenerate":
            exclusion = "jacobian_or_index_degenerate"
            failures.append("ROOT_CLASSIFICATION_DEGENERATE")
        elif classification == "X" and not connectivity["cell_bounding"]:
            exclusion = "no_cell_bounding_separatrix"
            failures.append("SEPARATRIX_UNRESOLVED")
        elif classification == "O" and not connectivity[
            "has_nearby_closed_channel_surface"
        ]:
            exclusion = "no_closed_cell_surface"
            failures.append("SEPARATRIX_UNRESOLVED")
        elif eligible_cusp:
            exclusion = None
        elif eligible_cell:
            exclusion = "cell_center_not_cusp"
        else:
            exclusion = "non_cusp_topology"
        if exclusion in ("outside_channel_axial",):
            failures.append("ROOT_OUTSIDE_CHANNEL")
        elif exclusion in ("exterior_magnet_or_current_sheet", "yoke_or_material"):
            failures.append("ROOT_EXCLUDED_HARDWARE")
        elif exclusion == "wall_only":
            failures.append("ROOT_WALL_ONLY")
        root = {
            **cluster,
            "field_vector_t": _field_vector(field, point),
            "field_magnitude_t": math.hypot(*_field_vector(field, point)),
            "local_topology": topology,
            "separatrix_connectivity": connectivity,
            "geometry_association": association,
            "eligible_cusp": eligible_cusp,
            "eligible_cell": eligible_cell,
            "exclusion_reason": exclusion,
        }
        root["root_semantic_sha256"] = semantic_hash(root)
        roots.append(root)
    return {
        "role": solved.role,
        "mesh_scale_m": mesh_scale,
        "cluster_tolerance_m": float(
            PROTOCOL["root_detection"]["clustering"]["tolerance_mesh_factor"]
        )
        * mesh_scale,
        "raw_detection_count": len(raw),
        "clustered_root_count": len(roots),
        "eligible_cusp_count": sum(root["eligible_cusp"] for root in roots),
        "eligible_cell_count": sum(root["eligible_cell"] for root in roots),
        "quality": solved.quality,
        "roots": roots,
        "failures": sorted(set(failures)),
    }


def _hungarian(cost: Sequence[Sequence[float]]) -> tuple[int, ...]:
    size = len(cost)
    if size == 0 or any(len(row) != size for row in cost):
        return ()
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j] = current
                    way[j] = j0
                if minimum[j] < delta:
                    delta, j1 = minimum[j], j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * size
    for j in range(1, size + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return tuple(assignment)


def assign_roots(
    primary: Mapping[str, Any], other: Mapping[str, Any]
) -> dict[str, Any]:
    left = primary["roots"]
    right = other["roots"]
    # A real row/column gets its own dummy alternatives even when the two
    # cardinalities are equal; forcing a remote real-real pairing before the
    # distance gate would not be a true assignment-with-unmatched solution.
    size = len(left) + len(right)
    threshold = (
        float(
            PROTOCOL["cross_map_correspondence"][
                "maximum_distance_mesh_factor"
            ]
        )
        * max(float(primary["mesh_scale_m"]), float(other["mesh_scale_m"]))
    )
    cost: list[list[float]] = []
    for i in range(size):
        row: list[float] = []
        for j in range(size):
            if i >= len(left) and j >= len(right):
                row.append(0.0)
            elif i >= len(left) or j >= len(right):
                row.append(threshold)
            else:
                row.append(
                    math.hypot(
                        float(left[i]["r_m"]) - float(right[j]["r_m"]),
                        float(left[i]["z_m"]) - float(right[j]["z_m"]),
                    )
                )
        cost.append(row)
    assignment = _hungarian(cost)
    matches: list[dict[str, Any]] = []
    used_right: set[int] = set()
    for i in range(len(left)):
        j = assignment[i]
        if j < len(right) and cost[i][j] <= threshold:
            used_right.add(j)
            matches.append(
                {
                    "primary_root_id": left[i]["root_id"],
                    "other_root_id": right[j]["root_id"],
                    "primary_index": i,
                    "other_index": j,
                    "shift_m": cost[i][j],
                    "classification_same": left[i]["local_topology"][
                        "classification"
                    ]
                    == right[j]["local_topology"]["classification"],
                    "cusp_eligibility_same": left[i]["eligible_cusp"]
                    == right[j]["eligible_cusp"],
                    "cell_eligibility_same": left[i]["eligible_cell"]
                    == right[j]["eligible_cell"],
                }
            )
    matched_left = {item["primary_index"] for item in matches}
    return {
        "maximum_assignment_distance_m": threshold,
        "matches": matches,
        "correspondence_count": len(matches),
        "unmatched_primary_root_ids": [
            left[index]["root_id"]
            for index in range(len(left))
            if index not in matched_left
        ],
        "unmatched_other_root_ids": [
            right[index]["root_id"]
            for index in range(len(right))
            if index not in used_right
        ],
        "maximum_shift_m": max(
            (item["shift_m"] for item in matches), default=None
        ),
    }


def cross_map_summary(characterized: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    primary = characterized["primary"]
    refined = assign_roots(primary, characterized["refined"])
    enlarged = assign_roots(primary, characterized["enlarged_domain"])
    refined_by_primary = {
        item["primary_index"]: item for item in refined["matches"]
    }
    enlarged_by_primary = {
        item["primary_index"]: item for item in enlarged["matches"]
    }
    stable_cusps: list[str] = []
    stable_cells: list[str] = []
    stable_roots: list[str] = []
    for index, root in enumerate(primary["roots"]):
        if index not in refined_by_primary or index not in enlarged_by_primary:
            continue
        pair = (refined_by_primary[index], enlarged_by_primary[index])
        stable = all(
            item["classification_same"]
            and item["cusp_eligibility_same"]
            and item["cell_eligibility_same"]
            for item in pair
        )
        if stable:
            stable_roots.append(root["root_id"])
            if root["eligible_cusp"]:
                stable_cusps.append(root["root_id"])
            if root["eligible_cell"]:
                stable_cells.append(root["root_id"])
    failures: list[str] = []
    if refined["unmatched_primary_root_ids"] or enlarged[
        "unmatched_primary_root_ids"
    ]:
        failures.append("CROSS_MAP_UNMATCHED")
    if any(
        not item["classification_same"]
        for study in (refined, enlarged)
        for item in study["matches"]
    ):
        failures.append("CROSS_MAP_CLASS_CHANGED")
    if any(
        not item["cusp_eligibility_same"] or not item["cell_eligibility_same"]
        for study in (refined, enlarged)
        for item in study["matches"]
    ):
        failures.append("CROSS_MAP_ELIGIBILITY_CHANGED")
    return {
        "primary_to_refined": refined,
        "primary_to_enlarged": enlarged,
        "stable_root_ids": stable_roots,
        "stable_root_count": len(stable_roots),
        "stable_eligible_cusp_ids": stable_cusps,
        "stable_eligible_cusp_count": len(stable_cusps),
        "stable_eligible_cell_ids": stable_cells,
        "stable_eligible_cell_count": len(stable_cells),
        "complete_primary_correspondence": len(stable_roots)
        == primary["clustered_root_count"],
        "failures": failures,
    }


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
    field_limits = PROTOCOL["gpu_replay"]["field_equality_tolerances"]
    field_pass = (
        max(differences["br_max_abs_t"], differences["bz_max_abs_t"])
        <= float(field_limits["maximum_b_component_absolute_difference_t"])
        and differences["psi_max_abs_wb"]
        <= float(field_limits["maximum_psi_absolute_difference_wb"])
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
        "residual_reproducibility_passed": residual_pass,
        "field_differences": differences,
        "field_equality_passed": field_pass,
        "passed": residual_pass and field_pass,
    }


def _mode(values: Sequence[int]) -> int:
    counts = {value: values.count(value) for value in set(values)}
    return min(counts, key=lambda value: (-counts[value], value))


def analyses(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stage_rows: list[dict[str, Any]] = []
    for stage_count in PROTOCOL["families"]["stage_counts"]:
        group = [case for case in cases if case["stage_count"] == stage_count]
        if not group:
            stage_rows.append(
                {
                    "stage_count": stage_count,
                    "case_count": 0,
                    "stable_eligible_cusp_counts": [],
                    "modal_stable_eligible_cusp_count": None,
                    "stable_eligible_cell_counts": [],
                    "modal_stable_eligible_cell_count": None,
                    "complete_correspondence_fraction": 0.0,
                    "median_maximum_shift_m": None,
                }
            )
            continue
        cusp_counts = [case["cross_map"]["stable_eligible_cusp_count"] for case in group]
        cell_counts = [case["cross_map"]["stable_eligible_cell_count"] for case in group]
        complete_fraction = sum(
            case["cross_map"]["complete_primary_correspondence"] for case in group
        ) / len(group)
        shifts = [
            shift
            for case in group
            for study_name in ("primary_to_refined", "primary_to_enlarged")
            for shift in (
                case["cross_map"][study_name]["maximum_shift_m"],
            )
            if shift is not None
        ]
        stage_rows.append(
            {
                "stage_count": stage_count,
                "case_count": len(group),
                "stable_eligible_cusp_counts": cusp_counts,
                "modal_stable_eligible_cusp_count": _mode(cusp_counts),
                "stable_eligible_cell_counts": cell_counts,
                "modal_stable_eligible_cell_count": _mode(cell_counts),
                "complete_correspondence_fraction": complete_fraction,
                "median_maximum_shift_m": statistics.median(shifts)
                if shifts
                else None,
            }
        )

    def factor_scores(name: str, levels: Sequence[Any]) -> list[dict[str, Any]]:
        rows = []
        for level in levels:
            group = [case for case in cases if case[name] == level]
            if not group:
                rows.append(
                    {
                        "level": level,
                        "mean_stable_correspondence_fraction": 0.0,
                        "median_matched_shift_m": None,
                    }
                )
                continue
            fractions = [
                case["cross_map"]["stable_root_count"]
                / max(1, case["maps"]["primary"]["clustered_root_count"])
                for case in group
            ]
            shifts = [
                item["shift_m"]
                for case in group
                for study in (
                    case["cross_map"]["primary_to_refined"],
                    case["cross_map"]["primary_to_enlarged"],
                )
                for item in study["matches"]
            ]
            rows.append(
                {
                    "level": level,
                    "mean_stable_correspondence_fraction": statistics.fmean(
                        fractions
                    ),
                    "median_matched_shift_m": statistics.median(shifts)
                    if shifts
                    else None,
                }
            )
        rows.sort(
            key=lambda item: (
                -item["mean_stable_correspondence_fraction"],
                float("inf")
                if item["median_matched_shift_m"] is None
                else item["median_matched_shift_m"],
                str(item["level"]),
            )
        )
        return rows

    factor_analysis = {
        "stage_count": factor_scores(
            "stage_count", PROTOCOL["families"]["stage_counts"]
        ),
        "pitch_m": factor_scores("pitch_m", PROTOCOL["families"]["pitch_m"]),
        "chamber_radius_m": factor_scores(
            "chamber_radius_m",
            PROTOCOL["families"]["chamber_outer_radius_m"],
        ),
        "first_polarity": factor_scores(
            "first_polarity", PROTOCOL["families"]["first_polarity"]
        ),
    }
    recommendation = {
        "classification": "descriptive_input_for_separate_search_v3_preregistration",
        "not_validated_or_optimal": True,
        "recommended_stage_counts": [
            item["level"] for item in factor_analysis["stage_count"][:2]
        ],
        "recommended_pitch_m": [
            factor_analysis["pitch_m"][0]["level"]
        ],
        "recommended_chamber_radius_m": [
            factor_analysis["chamber_radius_m"][0]["level"]
        ],
        "recommended_first_polarity": [
            factor_analysis["first_polarity"][0]["level"]
        ],
        "selection_rule": PROTOCOL["analyses"]["search_v3_recommendation"],
    }
    return {
        "stage_relation": stage_rows,
        "factor_scores": factor_analysis,
        "search_v3_recommendation": recommendation,
    }


def _failure_counts(cases: Sequence[Mapping[str, Any]], replay: Sequence[Mapping[str, Any]]):
    counts = {name: 0 for name in PROTOCOL["failure_taxonomy"]}
    for case in cases:
        for failure in set(case["failures"]):
            counts[failure] += 1
    counts["GPU_FIELD_REPLAY_FAILURE"] = sum(
        not item["field_equality_passed"] for item in replay
    )
    counts["GPU_RESIDUAL_REPLAY_FAILURE"] = sum(
        not item["residual_reproducibility_passed"] for item in replay
    )
    return counts


def _report(dataset: Mapping[str, Any]) -> str:
    summary = dataset["summary"]
    lines = [
        "# CFT topology characterization v1",
        "",
        "Developmental characterization only; not optimization or blind validation.",
        "",
        f"- Cases evaluated: {summary['evaluated_count']}",
        f"- Three-map field accepted: {summary['three_map_accepted_count']}",
        f"- Stable eligible cusps: {summary['stable_eligible_cusp_count']}",
        f"- Stable eligible cells: {summary['stable_eligible_cell_count']}",
        f"- GPU replay: {summary['gpu_replay_pass_count']}/{summary['gpu_replay_required_count']}",
        "- Mirror probabilities: not computed",
        "- Plasma state/power/performance: not computed",
        "",
        "## Empirical stage relation",
        "",
    ]
    for row in dataset["analyses"]["stage_relation"]:
        lines.append(
            f"- stages={row['stage_count']}: modal stable cusps="
            f"{row['modal_stable_eligible_cusp_count']}, modal stable cells="
            f"{row['modal_stable_eligible_cell_count']}, complete correspondence="
            f"{row['complete_correspondence_fraction']:.3f}"
        )
    recommendation = dataset["analyses"]["search_v3_recommendation"]
    lines.extend(
        [
            "",
            "## Descriptive search-v3 input",
            "",
            f"- Stage counts: {recommendation['recommended_stage_counts']}",
            f"- Pitch (m): {recommendation['recommended_pitch_m']}",
            f"- Chamber radius (m): {recommendation['recommended_chamber_radius_m']}",
            f"- First polarity: {recommendation['recommended_first_polarity']}",
            "",
            "These levels maximize the preregistered correspondence score in this",
            "developmental family. They are not validated or optimal and require a",
            "new search-v3 preregistration.",
            "",
            "## Failures",
            "",
        ]
    )
    lines.extend(
        f"- `{name}`: {count}"
        for name, count in sorted(summary["failure_counts"].items())
    )
    return "\n".join(lines) + "\n"


def _inventory_entry(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    if path.suffix == ".json":
        value = _strict_json(path)
        identity = semantic_hash(value)
        method = "canonical-json-sha256"
    else:
        identity = normalized_text_hash(path.read_text(encoding="utf-8"))
        method = "normalized-lf-text-sha256"
    return {
        "path": relative,
        "identity_method": method,
        "semantic_sha256": identity,
    }


def run_experiment(output_dir: Path) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists():
        raise RuntimeError("single execution output already exists")
    closure = dependency_closure()
    output.mkdir(parents=True, exist_ok=False)
    lock = {
        "schema_version": "cft-revival.exclusive-execution-lock/1.0.0",
        "experiment_id": PROTOCOL["experiment_id"],
        "preregistration_commit_sha": closure["preregistration_commit_sha"],
        "protocol_semantic_sha256": PROTOCOL_SEMANTIC_SHA256,
        "lock_semantic_sha256": semantic_hash(
            {
                "experiment_id": PROTOCOL["experiment_id"],
                "preregistration_commit_sha": closure[
                    "preregistration_commit_sha"
                ],
                "protocol_semantic_sha256": PROTOCOL_SEMANTIC_SHA256,
            }
        ),
        "status": "exclusive_lock_acquired_before_single_execution",
    }
    with (output / "execution-lock.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    (output / "preregistered-protocol.json").write_bytes(
        canonical_bytes(PROTOCOL)
    )
    runtime = _runtime_identity()
    write_semantic_json(output / "runtime.json", runtime)
    staging = output / ".staging"
    staging.mkdir()
    replay_ids = set(PROTOCOL["gpu_replay"]["case_ids"])
    replay_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for definition in case_definitions():
        failures: list[str] = []
        role_errors: dict[str, str] = {}
        solved_maps: dict[str, SolvedMap] = {}
        try:
            built = build_case(definition)
        except Exception as error:
            cases.append(
                {
                    **asdict(definition),
                    "geometry_valid": False,
                    "maps": {},
                    "cross_map": None,
                    "failures": ["GEOMETRY_INVALID"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        for role, failure_code in (
            ("primary", "FIELD_PRIMARY_INVALID"),
            ("refined", "FIELD_REFINED_INVALID"),
            ("enlarged_domain", "FIELD_ENLARGED_INVALID"),
        ):
            try:
                solved = solve_map(built, role, closure, runtime)
                solved_maps[role] = solved
                if not solved.quality["all_gates_passed"]:
                    failures.append(failure_code)
                role_dir = staging / definition.case_id
                role_dir.mkdir(parents=True, exist_ok=True)
                (role_dir / f"{role}-field.json").write_bytes(
                    solved.artifact_bytes
                )
            except Exception as error:
                role_errors[role] = f"{type(error).__name__}: {error}"
                failures.append(failure_code)
        if definition.case_id in replay_ids and "primary" in solved_maps:
            replay_rows.append(replay_map(solved_maps["primary"]))
        characterized = {
            role: characterize_map(built, solved)
            for role, solved in solved_maps.items()
        }
        for item in characterized.values():
            failures.extend(item["failures"])
        cross_map = (
            cross_map_summary(characterized)
            if len(characterized) == 3
            else None
        )
        if cross_map is not None:
            failures.extend(cross_map["failures"])
        maps_payload = {
            role: (
                {
                    "error": role_errors[role],
                }
                if role in role_errors
                else {
                    **characterized[role],
                    "artifact_semantic_sha256": solved_maps[
                        role
                    ].artifact_semantic_sha256,
                    "full_map_sha256": solved_maps[
                        role
                    ].validated.full_map_hash,
                    "domain": asdict(solved_maps[role].problem.domain),
                }
            )
            for role in ("primary", "refined", "enlarged_domain")
            if role in characterized or role in role_errors
        }
        cases.append(
            {
                **asdict(definition),
                "geometry_valid": True,
                "geometry_sha256": built.geometry_sha256,
                "material_semantic_sha256": built.material_semantic_sha256,
                "source_semantic_sha256": built.source_semantic_sha256,
                "geometry": {
                    "chamber_length_m": built.chamber_length_m,
                    "chamber_radius_m": definition.chamber_radius_m,
                    "stage_centres_m": built.stage_centres_m,
                    "magnet_inner_radius_m": built.magnet_inner_radius_m,
                    "magnet_outer_radius_m": built.magnet_outer_radius_m,
                },
                "maps": maps_payload,
                "cross_map": cross_map,
                "failures": sorted(set(failures)),
            }
        )
        geometry_dir = staging / definition.case_id
        geometry_dir.mkdir(parents=True, exist_ok=True)
        (geometry_dir / "geometry.json").write_text(
            canonical_json(built.geometry.to_dict()),
            encoding="utf-8",
            newline="\n",
        )
    valid_cases = [case for case in cases if case["cross_map"] is not None]
    result_analyses = analyses(valid_cases)
    transition_keys: set[tuple[int, int, int]] = set()
    representatives: list[str] = []
    for case in sorted(valid_cases, key=lambda item: item["case_id"]):
        key = (
            case["stage_count"],
            case["cross_map"]["stable_eligible_cusp_count"],
            case["cross_map"]["stable_eligible_cell_count"],
        )
        if key not in transition_keys and len(representatives) < 14:
            transition_keys.add(key)
            representatives.append(case["case_id"])
    representative_dir = output / "representatives"
    representative_entries: list[dict[str, Any]] = []
    for case_id in representatives:
        destination = representative_dir / case_id
        shutil.copytree(staging / case_id, destination)
        for path in sorted(destination.iterdir()):
            representative_entries.append(
                {
                    "case_id": case_id,
                    **_inventory_entry(output, path),
                }
            )
    shutil.rmtree(staging)
    failure_counts = _failure_counts(cases, replay_rows)
    summary = {
        "declared_case_count": len(case_definitions()),
        "evaluated_count": len(cases),
        "three_map_accepted_count": len(valid_cases),
        "stable_eligible_cusp_count": sum(
            case["cross_map"]["stable_eligible_cusp_count"]
            for case in valid_cases
        ),
        "stable_eligible_cell_count": sum(
            case["cross_map"]["stable_eligible_cell_count"]
            for case in valid_cases
        ),
        "gpu_replay_pass_count": sum(row["passed"] for row in replay_rows),
        "gpu_replay_required_count": len(replay_ids),
        "mirror_probability_count": 0,
        "plasma_publication_count": 0,
        "failure_counts": failure_counts,
    }
    dataset_payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": PROTOCOL["classification"],
        "purpose": PROTOCOL["purpose"],
        "protocol_semantic_sha256": PROTOCOL_SEMANTIC_SHA256,
        "preregistration_commit_sha": closure["preregistration_commit_sha"],
        "accepted_coupling_v3_commit_sha": ACCEPTED_COUPLING_COMMIT,
        "dependency_closure": closure,
        "runtime_identity": runtime,
        "summary": summary,
        "gpu_replay": replay_rows,
        "cases": cases,
        "analyses": result_analyses,
        "representative_case_ids": representatives,
        "publication": PROTOCOL["publication"],
    }
    dataset = write_semantic_json(output / "dataset.json", dataset_payload)
    report = _report(dataset)
    (output / "report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    inventory = [
        _inventory_entry(output, path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest_payload = {
        "schema_version": MANIFEST_VERSION,
        "experiment_id": PROTOCOL["experiment_id"],
        "protocol_semantic_sha256": PROTOCOL_SEMANTIC_SHA256,
        "preregistration_commit_sha": closure["preregistration_commit_sha"],
        "accepted_coupling_v3_commit_sha": ACCEPTED_COUPLING_COMMIT,
        "dependency_closure_semantic_sha256": closure[
            "closure_semantic_sha256"
        ],
        "single_execution": True,
        "summary": summary,
        "representatives": representative_entries,
        "artifacts": inventory,
    }
    manifest = write_semantic_json(output / "manifest.json", manifest_payload)
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
        or dataset["accepted_coupling_v3_commit_sha"]
        != ACCEPTED_COUPLING_COMMIT
    ):
        raise ValueError("protocol or baseline semantic identity mismatch")
    if dataset["summary"]["evaluated_count"] != int(
        PROTOCOL["families"]["case_count"]
    ):
        raise ValueError("not every family case was evaluated")
    if dataset["summary"]["mirror_probability_count"] != 0 or dataset[
        "summary"
    ]["plasma_publication_count"] != 0:
        raise ValueError("prohibited mirror/plasma output present")
    required_runtime = {
        "gpu_name",
        "gpu_uuid",
        "compute_capability",
        "driver_version",
        "reported_cuda_version",
        "warp_version",
        "warp_device_architecture",
    }
    if not required_runtime.issubset(runtime):
        raise ValueError("runtime identity incomplete")
    if len(dataset["gpu_replay"]) != len(PROTOCOL["gpu_replay"]["case_ids"]):
        raise ValueError("required replay cases missing")
    if not all(row["passed"] for row in dataset["gpu_replay"]):
        raise ValueError("GPU replay gate failed")
    listed = {entry["path"]: entry for entry in manifest["artifacts"]}
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(listed) != actual:
        raise ValueError("manifest inventory incomplete")
    for relative, entry in listed.items():
        recomputed = _inventory_entry(output, output / relative)
        if (
            recomputed["identity_method"] != entry["identity_method"]
            or recomputed["semantic_sha256"] != entry["semantic_sha256"]
        ):
            raise ValueError(f"semantic identity mismatch: {relative}")
    for case in dataset["cases"]:
        for map_result in case["maps"].values():
            if "roots" not in map_result:
                continue
            for root in map_result["roots"]:
                required = {
                    "r_m",
                    "z_m",
                    "local_topology",
                    "separatrix_connectivity",
                    "geometry_association",
                    "exclusion_reason",
                    "root_semantic_sha256",
                }
                if not required.issubset(root):
                    raise ValueError("root audit record incomplete")
    return {"dataset": dataset, "manifest": manifest, "runtime": runtime}
