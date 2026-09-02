"""One-shot preregistered held-out numerical validation of coupling v4."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
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
    reverify_v4_map_set,
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
from .canonical import (
    canonical_bytes,
    load_canonical,
    normalize,
    semantic_hash,
    write_canonical,
    write_raw,
)
from .control import (
    attempt_payload,
    clock_stamp,
    dependency_closure_payload,
    failure_payload,
    lock_finalized_payload,
    manifest_payload,
    stream_metadata_payload,
)

SCHEMA_VERSION = "cft-revival.cft-wall-cusp-validation-v3.dataset/1.0.0"
MANIFEST_VERSION = "cft-revival.cft-wall-cusp-validation-v3.manifest/1.0.0"
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


_json_value = normalize


def normalized_text_hash(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


PROTOCOL_SEMANTIC_SHA256 = semantic_hash(PROTOCOL)


def _dataclass_type_names(value: Any) -> tuple[str, ...]:
    names: set[str] = set()

    def visit(item: Any) -> None:
        if is_dataclass(item):
            names.add(f"{type(item).__module__}.{type(item).__name__}")
            for name in item.__dataclass_fields__:
                visit(getattr(item, name))
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(sorted(names))


def _structural_dataclass_probe(cls: type[Any]) -> Any:
    import collections.abc
    import types
    import typing
    from .canonical import TaggedSchema

    active: set[type[Any]] = set()

    def value_for(annotation: Any) -> Any:
        origin = typing.get_origin(annotation)
        arguments = typing.get_args(annotation)
        if annotation is Any:
            return "manufactured-any"
        if annotation is str:
            return "manufactured"
        if annotation is bool:
            return True
        if annotation is int:
            return 1
        if annotation is float:
            return 1.0
        if annotation is datetime:
            return datetime(2000, 1, 1, tzinfo=timezone.utc)
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return next(iter(annotation))
        if isinstance(annotation, type) and is_dataclass(annotation):
            return build(annotation)
        if origin in (list, set, frozenset, collections.abc.Sequence):
            return []
        if origin is tuple:
            if not arguments or arguments[-1:] == (Ellipsis,):
                return ()
            return tuple(value_for(item) for item in arguments)
        if origin in (dict, Mapping, collections.abc.Mapping):
            return {}
        if origin in (typing.Union, types.UnionType):
            selected = next((item for item in arguments if item is not type(None)), str)
            return value_for(selected)
        if origin is typing.Literal:
            return arguments[0]
        return TaggedSchema(
            "structural-type-placeholder",
            {"annotation": str(annotation)},
        )

    def build(target: type[Any]) -> Any:
        if target in active:
            return TaggedSchema(
                "recursive-dataclass-reference",
                {"type": f"{target.__module__}.{target.__name__}"},
            )
        active.add(target)
        instance = object.__new__(target)
        hints = typing.get_type_hints(target)
        for field in __import__("dataclasses").fields(target):
            object.__setattr__(
                instance,
                field.name,
                value_for(hints.get(field.name, Any)),
            )
        active.remove(target)
        return instance

    return build(cls)


def run_serialization_preflight() -> dict[str, Any]:
    """Exercise every public v4/v3 dataclass and a production record."""

    import inspect
    import tempfile
    from cft_revival.coupling.v3_models import BoundaryNullDiagnostic
    from cft_revival.coupling import v3_models, v4_models
    from tests.coupling import test_v4_cft_contract as manufactured

    maps = manufactured.map_set()
    adapter = MapGuidingCenterOrbitAdapter(maps, manufactured.REGISTRATIONS)
    record = manufactured.build(evidence=maps, orbit_adapter=adapter)
    matrix = {
        "finite_boundary_null": {
            "interior": (),
            "boundary": (
                BoundaryNullDiagnostic(-2.5, "z_min", 0.0),
                BoundaryNullDiagnostic(2.5, "z_max", 0.0),
            ),
        },
        "interior_null": {
            "interior": ((0.5, 0.0),),
            "boundary": (),
        },
        "empty_null": {"interior": (), "boundary": ()},
        "manufactured_production_v4_record": record,
        "v3_identity": record.stability.primary.identity,
        "v3_validation_policy": record.stability.primary.validation_policy,
        "orbit_diagnostics": tuple(adapter.diagnostics.values()),
    }
    public_types = tuple(
        sorted(
            (
                item
                for module in (v3_models, v4_models)
                for name, item in inspect.getmembers(module, inspect.isclass)
                if item.__module__ == module.__name__
                and not name.startswith("_")
                and is_dataclass(item)
            ),
            key=lambda item: f"{item.__module__}.{item.__name__}",
        )
    )
    probes = {
        f"{item.__module__}.{item.__name__}": _structural_dataclass_probe(item)
        for item in public_types
    }
    matrix["all_public_dataclass_type_probes"] = probes
    encoded = canonical_bytes(matrix)
    with tempfile.TemporaryDirectory(prefix="cft-v3-domain-preflight-") as tmp:
        path = Path(tmp) / "domain-matrix.canonical.json"
        stored = write_canonical(path, matrix, exclusive=True)
        if load_canonical(path) != stored:
            raise ValueError("domain matrix production write/load mismatch")
    serialized = set(_dataclass_type_names(matrix))
    declared = set(probes)
    missing = sorted(declared - serialized)
    if missing:
        raise ValueError(f"unserialized coupling dataclasses: {missing}")
    return {
        "schema_version": "cft-revival.cft-wall-cusp-validation-v3.serialization-preflight/1.0.0",
        "status": "passed",
        "prior_validation_held_out_map_access_count": 0,
        "matrix_case_ids": tuple(matrix),
        "declared_dataclass_types": tuple(sorted(declared)),
        "serialized_dataclass_types": tuple(sorted(serialized)),
        "missing_dataclass_types": missing,
        "matrix_semantic_sha256": hashlib.sha256(encoded).hexdigest(),
        "production_record_semantic_sha256": semantic_hash(record),
        "orbit_diagnostic_count": len(adapter.diagnostics),
    }


def write_semantic_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return write_canonical(path, payload, atomic=True)


def load_semantic_json(path: Path) -> dict[str, Any]:
    return load_canonical(path)


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
        "modern/experiments/cft_wall_cusp_validation_v3/",
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
    accepted_prefixes = (
        "modern/src/cft_revival/coupling/",
        "modern/src/cft_revival/fields/",
        "modern/src/cft_revival/geometry/",
        "modern/src/cft_revival/magnetics/",
        "modern/spec/coupling/",
        "modern/spec/fields/",
        "modern/spec/geometry/",
        "modern/spec/magnetics/",
    )
    for path in paths:
        blob = _git("rev-parse", f"{head}:{path}")
        baseline_blob = None
        if (
            path.startswith(accepted_prefixes)
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
    return dependency_closure_payload(
        preregistration_commit_sha=head,
        accepted_commit_sha=ACCEPTED_COUPLING_COMMIT,
        rows=rows,
    )


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
            f"wcval-v3-s{int(stages):02d}-p{pitch_index}-r{radius_index}-{sign}"
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
    excluded = family["excluded_accessed_evidence"]
    if set(excluded["v1_case_ids"]) & {item.case_id for item in result}:
        raise ValueError("held-out case IDs overlap v1 accessed cases")
    coordinate_tuples = {
        (
            item.stage_count,
            item.pitch_m,
            item.chamber_radius_m,
            item.first_polarity,
        )
        for item in result
    }
    if coordinate_tuples & {
        tuple(item) for item in excluded["v1_accessed_coordinate_tuples"]
    }:
        raise ValueError("held-out coordinates overlap v1 accessed coordinates")
    if (
        set(family["pitch_m"]) & set(excluded["development_pitch_m"])
        or set(family["chamber_outer_radius_m"])
        & set(excluded["development_chamber_outer_radius_m"])
    ):
        raise ValueError("held-out coordinates overlap development coordinates")
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
    adapter_id = "experiments.cft-wall-cusp-validation-v3.accepted-l1a-v4"
    version_contract = AdapterVersionContract(
        "cft-wall-cusp-validation-v3",
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


class MapGuidingCenterOrbitAdapter:
    """Map-bound nested-step guiding-centre and magnetic-moment verifier."""

    _ELECTRON_MASS_KG = 9.1093837139e-31
    _ELEMENTARY_CHARGE_C = 1.602176634e-19

    def __init__(self, map_set: Any, registrations: tuple[CFTCellRegistration, ...]) -> None:
        declaration = PROTOCOL["orbit_verification"]
        source_hash = normalized_text_hash(Path(__file__).read_text(encoding="utf-8"))
        self.adapter_id = str(declaration["adapter_id"])
        self.adapter_version = str(declaration["adapter_version"])
        self.adapter_code_hash = source_hash
        self.orbit_model_id = str(declaration["orbit_model_id"])
        self.orbit_model_version = str(declaration["orbit_model_version"])
        self.orbit_code_hash = hashlib.sha256(
            b"cft-wall-cusp-map-orbit-model-v3\0" + bytes.fromhex(source_hash)
        ).hexdigest()
        self._snapshots = reverify_v4_map_set(map_set)
        self._seed_ids = tuple(
            seed.seed_id
            for registration in registrations
            for seed in registration.seeds
        )
        map_hashes = tuple(item.field_map.full_map_hash for item in self._snapshots)
        self.orbit_config_hash = semantic_hash(
            {
                "samples": PROTOCOL["criterion"]["electron_samples"],
                "trace": PROTOCOL["criterion"]["field_line"],
                "map_hashes": map_hashes,
                "method": declaration["method"],
            }
        )
        self.convergence_id = str(declaration["convergence_id"])
        self.convergence_version = str(declaration["convergence_version"])
        self.convergence_config_hash = semantic_hash(
            {
                "timestep_refinement_factors": declaration[
                    "timestep_refinement_factors"
                ],
                "maximum_timestep_state_relative_difference": declaration[
                    "maximum_timestep_state_relative_difference"
                ],
                "maximum_energy_relative_drift": declaration[
                    "maximum_energy_relative_drift"
                ],
                "maximum_polyline_length_relative_defect": declaration[
                    "maximum_polyline_length_relative_defect"
                ],
            }
        )
        self.diagnostics: dict[tuple[str, str], dict[str, Any]] = {}

    @staticmethod
    def _path_hash(
        full_map_hash: str,
        seed_id: str,
        direction: int,
        psi_start_wb: float,
        points: tuple[tuple[float, float], ...],
    ) -> str:
        encoded = canonical_bytes(
            {
                "full_map_hash": full_map_hash,
                "seed_id": seed_id,
                "direction": direction,
                "psi_start_wb": psi_start_wb,
                "points": points,
            }
        )
        return hashlib.sha256(b"cft-v4-field-line-path\0" + encoded).hexdigest()

    def _bound_snapshot(
        self,
        path: tuple[tuple[float, float], ...],
        path_hash: str,
    ) -> tuple[Any, str, int]:
        matches: list[tuple[Any, str, int]] = []
        for snapshot in self._snapshots:
            psi = bilinear_sample(
                snapshot.field_map,
                snapshot.field_map.psi_wb,
                path[0],
            )
            for seed_id in self._seed_ids:
                for direction in (-1, 1):
                    if (
                        self._path_hash(
                            snapshot.field_map.full_map_hash,
                            seed_id,
                            direction,
                            psi,
                            path,
                        )
                        == path_hash
                    ):
                        matches.append((snapshot, seed_id, direction))
        if len(matches) != 1:
            raise ValueError("orbit path hash does not uniquely bind an accepted map")
        return matches[0]

    @staticmethod
    def _length(points: Sequence[tuple[float, float]]) -> float:
        return math.fsum(
            math.hypot(right[0] - left[0], right[1] - left[1])
            for left, right in zip(points[:-1], points[1:], strict=True)
        )

    def _evolution(
        self,
        snapshot: Any,
        path: tuple[tuple[float, float], ...],
        sample: ElectronOrbitSample,
        subdivisions: int,
    ) -> dict[str, Any]:
        field = snapshot.field_map
        energy_j = sample.kinetic_energy_ev * self._ELEMENTARY_CHARGE_C
        speed_squared = 2.0 * energy_j / self._ELECTRON_MASS_KG
        b0 = math.hypot(
            bilinear_sample(field, field.b_r_t, path[0]),
            bilinear_sample(field, field.b_z_t, path[0]),
        )
        mu0 = (
            self._ELECTRON_MASS_KG
            * speed_squared
            * math.sin(sample.pitch_angle_rad) ** 2
            / (2.0 * b0)
        )
        samples: list[dict[str, float]] = []
        distance = 0.0
        mirror_distance = None
        for left, right in zip(path[:-1], path[1:], strict=True):
            segment = math.hypot(right[0] - left[0], right[1] - left[1])
            for index in range(1, subdivisions + 1):
                fraction = index / subdivisions
                point = (
                    left[0] + fraction * (right[0] - left[0]),
                    left[1] + fraction * (right[1] - left[1]),
                )
                b_value = math.hypot(
                    bilinear_sample(field, field.b_r_t, point),
                    bilinear_sample(field, field.b_z_t, point),
                )
                perpendicular_squared = 2.0 * mu0 * b_value / self._ELECTRON_MASS_KG
                parallel_squared = speed_squared - perpendicular_squared
                step_distance = segment / subdivisions
                distance += step_distance
                if parallel_squared <= 0.0:
                    mirror_distance = distance
                    parallel_squared = 0.0
                observed_mu = (
                    self._ELECTRON_MASS_KG
                    * max(0.0, speed_squared - parallel_squared)
                    / (2.0 * b_value)
                )
                total_energy = (
                    0.5 * self._ELECTRON_MASS_KG * parallel_squared
                    + observed_mu * b_value
                )
                samples.append(
                    {
                        "s_m": distance,
                        "b_t": b_value,
                        "mu_j_per_t": observed_mu,
                        "energy_j": total_energy,
                        "pitch_angle_rad": math.atan2(
                            math.sqrt(max(0.0, perpendicular_squared)),
                            math.sqrt(max(0.0, parallel_squared)),
                        ),
                    }
                )
                if mirror_distance is not None:
                    break
            if mirror_distance is not None:
                break
        mu_variation = max(
            (
                abs(item["mu_j_per_t"] - mu0) / max(abs(mu0), 1e-300)
                for item in samples
            ),
            default=0.0,
        )
        energy_drift = max(
            (
                abs(item["energy_j"] - energy_j) / max(abs(energy_j), 1e-300)
                for item in samples
            ),
            default=0.0,
        )
        return {
            "subdivisions": subdivisions,
            "mu_initial_j_per_t": mu0,
            "maximum_mu_relative_variation": mu_variation,
            "maximum_energy_relative_drift": energy_drift,
            "mirror_distance_m": mirror_distance,
            "terminal_distance_m": distance,
            "terminal_pitch_angle_rad": (
                samples[-1]["pitch_angle_rad"]
                if samples
                else sample.pitch_angle_rad
            ),
            "sample_count": len(samples),
            "b_start_t": b0,
            "b_terminal_t": samples[-1]["b_t"] if samples else b0,
        }

    def verify_orbit(
        self,
        path_points_rz_m: tuple[tuple[float, float], ...],
        path_hash: str,
        sample: ElectronOrbitSample,
    ) -> OrbitVerificationClaims:
        snapshot, seed_id, direction = self._bound_snapshot(
            path_points_rz_m,
            path_hash,
        )
        refinements = tuple(
            self._evolution(snapshot, path_points_rz_m, sample, int(factor))
            for factor in PROTOCOL["orbit_verification"][
                "timestep_refinement_factors"
            ]
        )
        coarse, _, fine = refinements
        length = self._length(path_points_rz_m)
        coarse_points = path_points_rz_m[::2]
        if coarse_points[-1] != path_points_rz_m[-1]:
            coarse_points = (*coarse_points, path_points_rz_m[-1])
        polyline_defect = abs(length - self._length(coarse_points)) / max(
            length,
            1e-300,
        )
        if coarse["mirror_distance_m"] is None and fine["mirror_distance_m"] is None:
            state_difference = abs(
                coarse["terminal_pitch_angle_rad"]
                - fine["terminal_pitch_angle_rad"]
            ) / max(abs(fine["terminal_pitch_angle_rad"]), 1e-12)
        elif coarse["mirror_distance_m"] is not None and fine["mirror_distance_m"] is not None:
            state_difference = abs(
                coarse["mirror_distance_m"] - fine["mirror_distance_m"]
            ) / max(length, 1e-300)
        else:
            state_difference = float("inf")
        mu_variation = float(fine["maximum_mu_relative_variation"])
        energy_drift = float(fine["maximum_energy_relative_drift"])
        policy = PROTOCOL["orbit_verification"]
        converged = bool(
            math.isfinite(state_difference)
            and state_difference
            <= float(policy["maximum_timestep_state_relative_difference"])
            and energy_drift <= float(policy["maximum_energy_relative_drift"])
            and polyline_defect
            <= float(policy["maximum_polyline_length_relative_defect"])
            and mu_variation <= sample.maximum_mu_relative_variation
        )
        self.diagnostics[(path_hash, sample.sample_id)] = {
            "path_hash": path_hash,
            "sample_id": sample.sample_id,
            "seed_id": seed_id,
            "direction": direction,
            "full_map_hash": snapshot.field_map.full_map_hash,
            "kinetic_energy_ev": sample.kinetic_energy_ev,
            "initial_pitch_angle_rad": sample.pitch_angle_rad,
            "refinements": refinements,
            "timestep_state_relative_difference": state_difference,
            "polyline_length_relative_defect": polyline_defect,
            "converged": converged,
        }
        return OrbitVerificationClaims(
            path_hash=path_hash,
            sample_id=sample.sample_id,
            converged=converged,
            maximum_mu_relative_variation=mu_variation,
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


def orbit_identity(adapter: MapGuidingCenterOrbitAdapter) -> OrbitVerificationIdentity:
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
        "finite_boundary_nulls": _json_value(boundary),
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
    adapter_id = "experiments.cft-wall-cusp-validation-v3.held-out-artifact"

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
        "byte_count": len(data),
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
                "semantic_sha256": hashlib.sha256(data).hexdigest(),
                "identity_method": "exact-byte-sha256",
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
        "# Coupling v4 wall-cusp held-out validation v3",
        "",
        "Audit-corrected preregistered held-out numerical validation of schema 4.1.",
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


def _checkpoint(
    output: Path,
    phase: str,
    status: str,
    payload: Mapping[str, Any],
) -> None:
    path = output / "phase-status.json"
    events: list[dict[str, Any]] = []
    if path.exists():
        events = list(load_semantic_json(path)["events"])
    event = {
        "sequence": len(events) + 1,
        "clock": clock_stamp(),
        "phase": phase,
        "status": status,
        "payload": _json_value(payload),
    }
    events.append(event)
    write_semantic_json(
        path,
        {
            "schema_version": "cft-revival.cft-wall-cusp-validation-v3.phases/1.0.0",
            "attempt": 1,
            "events": events,
        },
    )


def _access(
    output: Path,
    definition: CaseDefinition,
    phase: str,
) -> None:
    access_dir = output / "access-log"
    access_dir.mkdir(parents=True, exist_ok=True)
    sequence = len(tuple(access_dir.glob("*.canonical.json"))) + 1
    row = {
        "schema_version": "cft-wall-cusp-v3.access-event/1.0.0",
        "sequence": sequence,
        "clock": clock_stamp(),
        "phase": phase,
        "case_id": definition.case_id,
        "geometry_family_id": definition.geometry_family_id,
        "geometry_id": definition.geometry_id,
        "coordinates": {
            "stage_count": definition.stage_count,
            "pitch_m": definition.pitch_m,
            "chamber_radius_m": definition.chamber_radius_m,
            "first_polarity": definition.first_polarity,
        },
    }
    write_canonical(
        access_dir / f"{sequence:06d}.canonical.json",
        row,
        exclusive=True,
    )


def _prerecord_checkpoint(record: Any, adapter: MapGuidingCenterOrbitAdapter) -> dict[str, Any]:
    assessments = (
        record.stability.primary,
        record.stability.refined,
        record.stability.enlarged,
    )
    cells = [cell for assessment in assessments for cell in assessment.cells]
    paths = [
        path
        for cell in cells
        for outcome in cell.seed_outcomes
        for path in (outcome.negative_path, outcome.positive_path)
    ]
    orbits = [item for path in paths for item in path.orbit_assessments]
    return {
        "record_hash": record.record_hash,
        "record_status": record.status.value,
        "stability_passed": record.stability.passed,
        "cusp_counts": record.stability.cusp_counts,
        "candidate_cell_count": len(cells),
        "resolved_cell_count": sum(cell.status is V4Status.RESOLVED for cell in cells),
        "candidate_path_count": len(paths),
        "resolved_path_count": sum(path.status is V4Status.RESOLVED for path in paths),
        "candidate_orbit_count": len(orbits),
        "resolved_orbit_count": sum(item.status is V4Status.RESOLVED for item in orbits),
        "evidence_fingerprints": record.evidence_fingerprints,
        "orbit_identity": orbit_identity(adapter),
        "orbit_diagnostics": tuple(adapter.diagnostics.values()),
    }


def run_experiment(output_dir: Path) -> dict[str, Any]:
    output = output_dir.resolve()
    if not output.is_dir():
        raise RuntimeError("launcher must initialize the results directory")
    acquired = load_canonical(output / "execution-lock-acquired.canonical.json")
    if (
        acquired["status"] != "exclusive_lock_acquired"
        or acquired["preregistration_commit_sha"] != _git("rev-parse", "HEAD")
    ):
        raise RuntimeError("valid preregistered exclusive lock is required")
    closure = load_canonical(output / "dependency-closure.canonical.json")
    runtime = load_canonical(output / "runtime-identity.canonical.json")
    _checkpoint(
        output,
        "worker_initialization",
        "complete",
        {
            "dependency_closure_semantic_sha256": closure[
                "closure_semantic_sha256"
            ],
            "held_out_access_count": 0,
        },
    )
    code_hash = normalized_text_hash(Path(__file__).read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    field_root = output / "fields"
    for definition in case_definitions():
        _access(output, definition, "held_out_case_access_started")
        _checkpoint(
            output,
            "held_out_case",
            "started",
            {"case_id": definition.case_id},
        )
        built = build_case(definition)
        solved: dict[str, SolvedMap] = {}
        for role in ("primary", "refined", "enlarged"):
            _access(output, definition, f"map_{role}_solve_started")
            item = solve_map(built, role, closure, runtime)
            solved[role] = item
            role_dir = field_root / definition.case_id
            role_dir.mkdir(parents=True, exist_ok=True)
            write_raw(
                role_dir / f"{role}-field.json",
                item.artifact_bytes,
                exclusive=True,
            )
            _checkpoint(
                output,
                f"map_{role}",
                "complete",
                {
                    "case_id": definition.case_id,
                    "artifact_sha256": hashlib.sha256(
                        item.artifact_bytes
                    ).hexdigest(),
                    "artifact_payload_sha256": item.artifact_payload_sha256,
                    "field_quality": item.quality,
                },
            )
        map_set = verify_v4_map_set(
            solved["primary"].evidence,
            solved["refined"].evidence,
            solved["enlarged"].evidence,
            reference_time_utc=runtime["generated_at_utc"],
        )
        registrations = registrations_for(built)
        adapter = MapGuidingCenterOrbitAdapter(map_set, registrations)
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
            registrations=registrations,
            validation_registration=validation_registration(built, code_hash),
            orbit_adapter=adapter,
            criterion=V4Criterion(),
            reference_time_utc=runtime["generated_at_utc"],
            **policies_for(built),
        )
        prerecord_dir = output / "prerecords"
        prerecord_dir.mkdir(parents=True, exist_ok=True)
        write_semantic_json(
            prerecord_dir / f"{definition.case_id}.json",
            {
                "record": cft_coupling_record_dict(prerecord),
                "summary": _prerecord_checkpoint(prerecord, adapter),
            },
        )
        _checkpoint(
            output,
            "prerecord",
            "complete",
            {
                "case_id": definition.case_id,
                **_prerecord_checkpoint(prerecord, adapter),
            },
        )
        replay = replay_map(solved["primary"])
        replay_dir = output / "gpu-replay"
        replay_dir.mkdir(parents=True, exist_ok=True)
        write_semantic_json(
            replay_dir / f"{definition.case_id}.json",
            replay,
        )
        _checkpoint(
            output,
            "gpu_replay",
            "complete",
            replay,
        )
        diagnostics = {
            role: topology_diagnostics(item.evidence)
            for role, item in solved.items()
        }
        diagnostic_dir = output / "topology-diagnostics"
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        write_semantic_json(
            diagnostic_dir / f"{definition.case_id}.json",
            diagnostics,
        )
        _checkpoint(
            output,
            "topology_diagnostics",
            "complete",
            {"case_id": definition.case_id, "diagnostics": diagnostics},
        )
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
        row["candidate_cell_count"] = sum(
            len(item.cells)
            for item in (
                prerecord.stability.primary,
                prerecord.stability.refined,
                prerecord.stability.enlarged,
            )
        )
        row["resolved_cell_count"] = sum(
            cell.status is V4Status.RESOLVED
            for item in (
                prerecord.stability.primary,
                prerecord.stability.refined,
                prerecord.stability.enlarged,
            )
            for cell in item.cells
        )
        row["orbit_implementation_identity"] = _json_value(
            orbit_identity(adapter)
        )
        row["orbit_diagnostics"] = _json_value(tuple(adapter.diagnostics.values()))
        geometry_dir = output / "geometries"
        geometry_dir.mkdir(parents=True, exist_ok=True)
        write_canonical(
            geometry_dir / f"{definition.case_id}.json",
            built.geometry.to_dict(),
            exclusive=True,
            sealed=False,
        )
        cases.append(row)
        _checkpoint(
            output,
            "held_out_case",
            "complete",
            {
                "case_id": definition.case_id,
                "passed": row["passed"],
                "candidate_cell_count": row["candidate_cell_count"],
                "resolved_cell_count": row["resolved_cell_count"],
            },
        )
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
            adapter = MapGuidingCenterOrbitAdapter(map_set, registrations)
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
            write_raw(
                validation_dir / f"{definition.case_id}.json",
                artifact_bytes,
                exclusive=True,
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
        "candidate_cell_count": sum(case["candidate_cell_count"] for case in cases),
        "resolved_cell_count": sum(case["resolved_cell_count"] for case in cases),
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
        "orbit_implementation_identities": [
            case["orbit_implementation_identity"] for case in cases
        ],
        "summary": summary,
        "cases": cases,
        "projection_rows": projection_rows,
    }
    dataset = write_semantic_json(output / "dataset.json", dataset_payload)
    write_raw(
        output / "report.md",
        _report(dataset).replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"),
        exclusive=True,
    )
    _checkpoint(output, "worker", "complete", summary)
    return {"dataset": dataset}


def finalize_attempt(
    output_dir: Path,
    *,
    exit_code: int,
    stdout_bytes: bytes,
    stderr_bytes: bytes,
    command: str,
) -> dict[str, Any]:
    """Finalize the sole worker without mutating the acquired lock."""

    output = output_dir.resolve()
    acquired = load_canonical(output / "execution-lock-acquired.canonical.json")
    write_raw(output / "stdout.bin", stdout_bytes, exclusive=True)
    write_raw(output / "stderr.bin", stderr_bytes, exclusive=True)
    streams = write_canonical(
        output / "process-streams.canonical.json",
        stream_metadata_payload(stdout=stdout_bytes, stderr=stderr_bytes),
        exclusive=True,
    )
    finalized = write_canonical(
        output / "execution-lock-finalized.canonical.json",
        lock_finalized_payload(
            acquired,
            exit_code=exit_code,
            stdout_sha256=streams["stdout"]["byte_sha256"],
            stderr_sha256=streams["stderr"]["byte_sha256"],
        ),
        exclusive=True,
    )
    write_canonical(
        output / "attempt-finalized.canonical.json",
        attempt_payload(
            acquired,
            state="completed" if exit_code == 0 else "failed",
            exit_code=exit_code,
        ),
        exclusive=True,
    )
    if exit_code == 0:
        if not (output / "dataset.json").exists():
            raise RuntimeError("successful worker omitted dataset")
        dataset = load_canonical(output / "dataset.json")
        summary = dataset["summary"]
        status = "success"
    else:
        phase = (
            load_canonical(output / "phase-status.json")
            if (output / "phase-status.json").exists()
            else {"events": []}
        )
        access_rows = [
            load_canonical(path)
            for path in sorted(
                (output / "access-log").glob("*.canonical.json")
            )
        ] if (output / "access-log").exists() else []
        prerecords = [
            load_canonical(path)
            for path in sorted((output / "prerecords").glob("*.json"))
        ] if (output / "prerecords").exists() else []
        prerecord_summaries = [item["summary"] for item in prerecords]
        summary = {
            "declared_case_count": len(case_definitions()),
            "declared_map_count": 3 * len(case_definitions()),
            "access_event_count": len(access_rows),
            "attempted_case_count": len(
                {
                    row["case_id"]
                    for row in access_rows
                    if row["phase"] == "held_out_case_access_started"
                }
            ),
            "materialized_map_count": len(
                tuple((output / "fields").rglob("*-field.json"))
            ) if (output / "fields").exists() else 0,
            "prerecord_count": len(prerecords),
            "gpu_replay_count": len(
                tuple((output / "gpu-replay").glob("*.json"))
            ) if (output / "gpu-replay").exists() else 0,
            "candidate_cell_count": sum(
                int(item["candidate_cell_count"])
                for item in prerecord_summaries
            ),
            "resolved_cell_count": sum(
                int(item["resolved_cell_count"])
                for item in prerecord_summaries
            ),
            "candidate_path_count": sum(
                int(item["candidate_path_count"])
                for item in prerecord_summaries
            ),
            "resolved_path_count": sum(
                int(item["resolved_path_count"])
                for item in prerecord_summaries
            ),
            "candidate_orbit_count": sum(
                int(item["candidate_orbit_count"])
                for item in prerecord_summaries
            ),
            "resolved_orbit_count": sum(
                int(item["resolved_orbit_count"])
                for item in prerecord_summaries
            ),
            "held_out_outcome_count": 0,
            "opaque_projection_count": 0,
            "criterion_numerically_promoted": False,
            "search_v3_ready": False,
            "plasma_coupling_ready": False,
        }
        traceback_hash = hashlib.sha256(stderr_bytes).hexdigest()
        failure = write_canonical(
            output / "failure.canonical.json",
            failure_payload(
                phase=(
                    str(phase["events"][-1]["phase"])
                    if phase["events"]
                    else "launcher_or_worker_initialization"
                ),
                exception_type="WorkerProcessFailure",
                message=stderr_bytes.decode("utf-8", errors="replace")[-2000:],
                traceback_sha256=traceback_hash,
                summary=summary,
            ),
            exclusive=True,
        )
        write_raw(
            output / "failure-report.md",
            (
                "# Coupling v4 wall-cusp validation v3 - immutable failure\n\n"
                "The sole detached attempt failed and was not patched or rerun.\n\n"
                f"- Exit code: {exit_code}\n"
                f"- Attempted cases/maps: {summary['attempted_case_count']}/"
                f"{summary['materialized_map_count']}\n"
                f"- Candidate/resolved cells: {summary['candidate_cell_count']}/"
                f"{summary['resolved_cell_count']}\n"
                f"- Candidate/resolved paths: {summary['candidate_path_count']}/"
                f"{summary['resolved_path_count']}\n"
                f"- Candidate/resolved orbits: {summary['candidate_orbit_count']}/"
                f"{summary['resolved_orbit_count']}\n"
                "- Criterion promoted: false\n"
            ).encode("utf-8"),
            exclusive=True,
        )
        status = "failed"
    inventory = [
        _inventory_entry(output, path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.canonical.json"
    ]
    write_canonical(
        output / "manifest.canonical.json",
        manifest_payload(
            experiment_id=str(PROTOCOL["experiment_id"]),
            preregistration_commit_sha=str(
                acquired["preregistration_commit_sha"]
            ),
            accepted_commit_sha=ACCEPTED_COUPLING_COMMIT,
            status=status,
            summary=summary,
            artifacts=inventory,
        ),
        exclusive=True,
    )
    return validate_results(output)


def validate_results(output_dir: Path) -> dict[str, Any]:
    output = output_dir.resolve()
    manifest = load_canonical(output / "manifest.canonical.json")
    runtime = load_canonical(output / "runtime-identity.canonical.json")
    snapshot = load_canonical(output / "protocol-snapshot.canonical.json")
    protocol_copy = snapshot["protocol"]
    acquired = load_canonical(output / "execution-lock-acquired.canonical.json")
    finalized = load_canonical(output / "execution-lock-finalized.canonical.json")
    streams = load_canonical(output / "process-streams.canonical.json")
    listed = {entry["path"]: entry for entry in manifest["artifacts"]}
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.canonical.json"
    }
    if set(listed) != actual:
        raise ValueError("manifest inventory is incomplete")
    for relative, entry in listed.items():
        recomputed = _inventory_entry(output, output / relative)
        if recomputed != entry:
            raise ValueError(f"artifact identity mismatch: {relative}")
    if hashlib.sha256((output / "stdout.bin").read_bytes()).hexdigest() != streams[
        "stdout"
    ]["byte_sha256"] or hashlib.sha256(
        (output / "stderr.bin").read_bytes()
    ).hexdigest() != streams["stderr"]["byte_sha256"]:
        raise ValueError("captured process stream identity mismatch")
    if finalized["acquired_lock_payload_sha256"] != acquired[
        "semantic_integrity"
    ]["payload_sha256"]:
        raise ValueError("finalized lock does not bind acquired lock")
    if manifest["status"] == "failed":
        failure = load_canonical(output / "failure.canonical.json")
        if (
            failure["status"] != "failed_immutable_no_patch_no_rerun"
            or acquired["attempt"] != 1
            or finalized["exit_code"] in (None, 0)
            or finalized["status"] != "failed_immutable"
            or semantic_hash(protocol_copy) != PROTOCOL_SEMANTIC_SHA256
            or failure["summary"]["criterion_numerically_promoted"]
            or failure["summary"]["opaque_projection_count"] != 0
        ):
            raise ValueError("typed failure bundle is inconsistent")
        return {
            "status": "failed",
            "failure": failure,
            "manifest": manifest,
            "runtime": runtime,
        }
    dataset = load_canonical(output / "dataset.json")
    if (
        semantic_hash(protocol_copy) != PROTOCOL_SEMANTIC_SHA256
        or dataset["protocol_semantic_sha256"] != PROTOCOL_SEMANTIC_SHA256
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
