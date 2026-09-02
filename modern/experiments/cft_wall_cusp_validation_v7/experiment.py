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
from dataclasses import asdict, dataclass, fields as dataclass_fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.coupling import (
    AxialDominancePolicy,
    CanonicalFieldV12Adapter,
    CanonicalFieldV12Binding,
    CFTGeometry,
    CFTStabilityPolicy,
    CFT_V4_DEVELOPMENT_MANIFEST,
    CFTCellRegistration,
    COUPLING_V4_SCHEMA_VERSION,
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
    V4Criterion,
    V4_FIELD_ARTIFACT_SCHEMA,
    V4_FIELD_CANONICALIZATION,
    V4Status,
    ValidationSetManifest,
    WallCuspPolicy,
    accept_cft_projection,
    bilinear_sample,
    build_cft_coupling_record,
    cft_coupling_record_dict,
    cft_preregistration_hash,
    cft_solver_inputs,
    magnetic_null_geometry,
    reverify_v3_evidence,
    reverify_v4_map_set,
    validation_set_manifest_hash,
    v4_map_set_evidence_fingerprints,
    verify_canonical_field_v12_artifact,
    verify_held_out_validation,
    verify_v4_map_set,
)
from cft_revival.fields import (
    ARTIFACT_SCHEMA_VERSION,
    AxisymmetricDomain,
    AxisymmetricProblem,
    FieldMap,
    SolverConfig,
    canonical_field_artifact_bytes,
    contains_negative_zero,
    field_artifact,
    field_artifact_canonical_bytes,
    reload_field_artifact_bytes,
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
from cft_revival.coupling.v3_models import BoundaryNullDiagnostic
from cft_revival.experiment_runtime import (
    Decision,
    ExecutionAttestation,
    ExperimentRuntime,
    RootPolicy,
    RunContext,
    RuntimeCallbacks,
    canonical_bytes,
    canonical_value as normalize,
    semantic_sha256 as semantic_hash,
    strict_json_loads,
    validate_bundle,
)

SCHEMA_VERSION = "cft-revival.cft-wall-cusp-validation-v7.dataset/1.0.0"
FOUNDATION_COMMIT = "b46e263950f91530ea61710b5dcc9354fc63cf6c"
ACCEPTED_COUPLING_COMMIT = FOUNDATION_COMMIT
EXPERIMENT_DIR = Path(__file__).resolve().parent
MODERN_ROOT = EXPERIMENT_DIR.parents[1]
REPOSITORY_ROOT = MODERN_ROOT.parent
PROTOCOL_PATH = EXPERIMENT_DIR / "protocol.json"
RESULT_ATTRIBUTES = b"* -text\n"


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


def serialize_boundary_null_diagnostic(
    diagnostic: BoundaryNullDiagnostic,
) -> dict[str, Any]:
    """Serialize one boundary null as plain domain JSON."""

    return {
        "z_m": diagnostic.z_m,
        "boundary": diagnostic.boundary,
        "b_magnitude_t": diagnostic.b_magnitude_t,
    }


def _plain_domain_json(value: Any) -> Any:
    """Convert callback payloads without reserved canonical envelopes."""

    if isinstance(value, BoundaryNullDiagnostic):
        return serialize_boundary_null_diagnostic(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain_domain_json(getattr(value, field.name))
            for field in dataclass_fields(value)
        }
    if isinstance(value, Enum):
        return _plain_domain_json(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("callback datetime must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or key == "__cft_type__":
                raise ValueError("callback mappings require non-reserved string keys")
            result[key] = _plain_domain_json(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_plain_domain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("callback payload floats must be finite")
        return 0.0 if value == 0.0 else value
    raise TypeError(f"unsupported callback payload type: {type(value).__name__}")


def _assert_plain_domain_json(value: Any) -> None:
    plain = _plain_domain_json(value)
    encoded = json.dumps(
        plain,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if b'"__cft_type__"' in encoded:
        raise ValueError("callback payload contains a reserved canonical envelope")


def _write_callback_json(
    context: RunContext,
    relative_path: str,
    payload: Any,
) -> Mapping[str, Any]:
    plain = _plain_domain_json(payload)
    _assert_plain_domain_json(plain)
    return context.write_json(relative_path, plain)


_json_value = _plain_domain_json


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
        return f"structural-placeholder:{annotation!s}"

    def build(target: type[Any]) -> Any:
        if target in active:
            return f"recursive-dataclass-reference:{target.__module__}.{target.__name__}"
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
    """Exercise field edge values and every public coupling diagnostic serializer."""

    import inspect
    from math import nextafter
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
        "field_numeric_edges": {
            "negative_zero": -0.0,
            "positive_zero": 0.0,
            "positive_subnormal": nextafter(0.0, 1.0),
            "negative_subnormal": nextafter(0.0, -1.0),
        },
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
    if strict_json_loads(encoded) != normalize(matrix):
        raise ValueError("runtime canonical serializer round-trip mismatch")
    field_edges = canonical_field_artifact_bytes(
        matrix["field_numeric_edges"], representation="payload"
    )
    parsed_edges = strict_json_loads(field_edges)
    if contains_negative_zero(parsed_edges):
        raise ValueError("field serializer retained signed negative zero")
    if parsed_edges["positive_subnormal"] == 0.0 or parsed_edges["negative_subnormal"] == 0.0:
        raise ValueError("field serializer flushed a binary64 subnormal")
    serialized = set(_dataclass_type_names(matrix))
    declared = set(probes)
    missing = sorted(declared - serialized)
    if missing:
        raise ValueError(f"unserialized coupling dataclasses: {missing}")
    return {
        "schema_version": "cft-revival.cft-wall-cusp-validation-v7.serialization-preflight/1.0.0",
        "status": "passed",
        "prior_validation_held_out_map_access_count": 0,
        "matrix_case_ids": tuple(matrix),
        "declared_dataclass_types": tuple(sorted(declared)),
        "serialized_dataclass_types": tuple(sorted(serialized)),
        "missing_dataclass_types": missing,
        "matrix_semantic_sha256": hashlib.sha256(encoded).hexdigest(),
        "production_record_semantic_sha256": semantic_hash(record),
        "orbit_diagnostic_count": len(adapter.diagnostics),
        "field_edge_bytes_sha256": hashlib.sha256(field_edges).hexdigest(),
        "field_signed_zero_normalized": True,
        "field_subnormals_preserved": True,
    }






def run_production_path_static_preflight() -> dict[str, Any]:
    """Reject implicit policy, legacy reload, or pre-canonicalized callback paths."""

    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    policy_fields = {
        "minimum_radial_samples",
        "minimum_axial_samples",
        "maximum_age_s",
        "maximum_future_skew_s",
        "require_axis",
        "axis_coordinate_tolerance_m",
        "axis_br_absolute_tolerance_t",
        "axis_br_relative_tolerance",
        "current_artifact_schema",
        "accepted_model_levels",
        "validated_migration_adapter_ids",
    }
    policy_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MapValidationPolicy"
    ]
    if len(policy_calls) != 2:
        raise ValueError("v7 must have two fully explicit MapValidationPolicy calls")
    for call in policy_calls:
        if call.args or {item.arg for item in call.keywords} != policy_fields:
            raise ValueError("every MapValidationPolicy call must set every field")
    reload_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "reload_field_artifact_bytes"
    ]
    if not reload_calls:
        raise ValueError("v7 production code must reload canonical field bytes")
    for call in reload_calls:
        legacy = next(
            (item.value for item in call.keywords if item.arg == "allow_legacy_v1_1"),
            None,
        )
        if not (isinstance(legacy, ast.Constant) and legacy.value is False):
            raise ValueError("every v7 field reload must explicitly disable legacy v1.1")
    direct_context_writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "context"
        and node.func.attr == "write_json"
    ]
    if len(direct_context_writes) != 1:
        raise ValueError("callbacks must use the single plain-domain write wrapper")
    wrapped_writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write_callback_json"
    ]
    for call in wrapped_writes:
        payload = call.args[2] if len(call.args) >= 3 else None
        if (
            isinstance(payload, ast.Call)
            and isinstance(payload.func, ast.Name)
            and payload.func.id in {"normalize", "canonical_value", "canonical_bytes"}
        ):
            raise ValueError("callback payloads must not be pre-canonicalized")
    policy = map_policy()
    expected = MapValidationPolicy(
        minimum_radial_samples=40,
        minimum_axial_samples=160,
        maximum_age_s=3600.0,
        maximum_future_skew_s=5.0,
        require_axis=True,
        axis_coordinate_tolerance_m=1e-12,
        axis_br_absolute_tolerance_t=2e-10,
        axis_br_relative_tolerance=1e-8,
        current_artifact_schema=ARTIFACT_SCHEMA_VERSION,
        accepted_model_levels=("L1a",),
        validated_migration_adapter_ids=(),
    )
    if policy != expected:
        raise ValueError("runtime map policy differs from the complete v7 policy")
    return {
        "status": "passed",
        "map_validation_policy_call_count": len(policy_calls),
        "explicit_policy_fields": tuple(sorted(policy_fields)),
        "legacy_disabled_reload_call_count": len(reload_calls),
        "production_policy": policy,
        "implicit_policy_defaults": False,
        "legacy_v1_1_reload_possible": False,
        "direct_context_write_json_call_count": len(direct_context_writes),
        "plain_callback_write_call_count": len(wrapped_writes),
        "precanonicalized_callback_payloads": False,
    }


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
    symbolic = subprocess.run(
        ("git", "symbolic-ref", "-q", "--short", "HEAD"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if symbolic.returncode == 0:
        raise RuntimeError("execution requires detached HEAD")
    paths = tuple(
        path
        for path in _git("ls-files").splitlines()
        if (
            path == "modern/pyproject.toml"
            or path.startswith("modern/src/cft_revival/")
            or path.startswith("modern/spec/")
            or path.startswith("modern/experiments/cft_wall_cusp_validation_v7/")
        )
        and "/results/" not in path
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
            baseline_blob = _git("rev-parse", f"{FOUNDATION_COMMIT}:{path}")
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
        "foundation_commit_sha": FOUNDATION_COMMIT,
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
            f"wcval-v7-s{int(stages):02d}-p{pitch_index}-r{radius_index}-{sign}"
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
    prior_ids = set().union(
        excluded["v1_accessed_case_ids"],
        excluded["v2_accessed_case_ids"],
        excluded["v3_accessed_case_ids"],
        excluded["v4_accessed_case_ids"],
        excluded["v5_accessed_case_ids"],
        excluded["v6_accessed_case_ids"],
    )
    if prior_ids & {item.case_id for item in result}:
        raise ValueError("held-out case IDs overlap prior accessed cases")
    coordinate_tuples = {
        (
            item.stage_count,
            item.pitch_m,
            item.chamber_radius_m,
            item.first_polarity,
        )
        for item in result
    }
    prior_coordinates = {
        tuple(item)
        for key in (
            "v1_accessed_coordinate_tuples",
            "v2_accessed_coordinate_tuples",
            "v3_accessed_coordinate_tuples",
            "v4_accessed_coordinate_tuples",
            "v5_accessed_coordinate_tuples",
            "v6_accessed_coordinate_tuples",
        )
        for item in excluded[key]
    }
    if coordinate_tuples & prior_coordinates:
        raise ValueError("held-out coordinates overlap prior accessed coordinates")
    prior_families = {
        excluded["v1_family_id"],
        excluded["v2_family_id"],
        excluded["v3_family_id"],
        excluded["v4_family_id"],
        excluded["v5_family_id"],
        excluded["v6_family_id"],
    }
    if family_id in prior_families:
        raise ValueError("held-out family overlaps a prior validation family")
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


def map_policy() -> MapValidationPolicy:
    """Return the complete v1.2 policy without inheriting any shared default."""

    return MapValidationPolicy(
        minimum_radial_samples=40,
        minimum_axial_samples=160,
        maximum_age_s=3600.0,
        maximum_future_skew_s=5.0,
        require_axis=True,
        axis_coordinate_tolerance_m=1e-12,
        axis_br_absolute_tolerance_t=2e-10,
        axis_br_relative_tolerance=1e-8,
        current_artifact_schema=ARTIFACT_SCHEMA_VERSION,
        accepted_model_levels=("L1a",),
        validated_migration_adapter_ids=(),
    )


def _field_quality(
    problem: AxisymmetricProblem,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    field = artifact["field_map"]
    diagnostics = artifact["diagnostics"]
    peak = max(
        math.hypot(br, bz)
        for br_row, bz_row in zip(field["b_r_t"], field["b_z_t"], strict=True)
        for br, bz in zip(br_row, bz_row, strict=True)
    )
    boundary = max(
        math.hypot(field["b_r_t"][i][j], field["b_z_t"][i][j])
        for i in range(len(field["r_m"]))
        for j in range(len(field["z_m"]))
        if i == len(field["r_m"]) - 1 or j in (0, len(field["z_m"]) - 1)
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
        "initial_residual_l2": diagnostics["initial_residual_l2"],
        "final_residual_l2": diagnostics["final_residual_l2"],
        "normalized_residual": diagnostics["relative_residual_l2"],
        "flux_reconstruction_identity_t_per_m": (
            diagnostics["max_flux_reconstruction_identity_t_per_m"]
        ),
        "iterations": diagnostics["iterations"],
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
    artifact_bytes: bytes
    artifact: dict[str, Any]
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
    artifact_bytes = field_artifact_canonical_bytes(artifact)
    reloaded = reload_field_artifact_bytes(
        artifact_bytes,
        source=f"{case.definition.case_id}-{role}",
        allow_legacy_v1_1=False,
    )
    if reloaded["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("solve did not produce a canonical field v1.2 artifact")
    evidence = verify_canonical_field_v12_artifact(
        artifact_bytes,
        CanonicalFieldV12Binding(
            geometry_hash=case.geometry_sha256,
            code_hash=closure["closure_semantic_sha256"],
            backend_version=f"warp-{runtime['warp_version']}",
            generated_at_utc=runtime["generated_at_utc"],
        ),
        map_policy(),
        reference_time_utc=runtime["generated_at_utc"],
    )
    return SolvedMap(
        role,
        problem,
        artifact_bytes,
        reloaded,
        str(artifact["integrity"]["payload_sha256"]),
        evidence,
        _field_quality(problem, reloaded),
    )


def load_map_evidence(
    case: BuiltCase,
    role: str,
    artifact_bytes: bytes,
    closure: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> Any:
    reloaded = reload_field_artifact_bytes(
        artifact_bytes,
        source=f"{case.definition.case_id}-{role}-runtime-store",
        allow_legacy_v1_1=False,
    )
    if reloaded["input"]["sources"] != [asdict(source) for source in case.sources]:
        raise ValueError("reloaded canonical artifact source identity mismatch")
    return verify_canonical_field_v12_artifact(
        artifact_bytes,
        CanonicalFieldV12Binding(
            geometry_hash=case.geometry_sha256,
            code_hash=closure["closure_semantic_sha256"],
            backend_version=f"warp-{runtime['warp_version']}",
            generated_at_utc=runtime["generated_at_utc"],
        ),
        map_policy(),
        reference_time_utc=runtime["generated_at_utc"],
    )


def _manufactured_development_case() -> BuiltCase:
    definition = CaseDefinition(
        case_id="development-manufactured-v7-field-pipeline",
        geometry_family_id="cft-topology-characterization-v1-cartesian-family",
        geometry_id="development-manufactured-v7-field-pipeline-geometry",
        stage_count=3,
        pitch_m=0.006,
        chamber_radius_m=0.009,
        first_polarity=1,
        family_semantic_sha256=semantic_hash(
            {
                "role": "non-held-out-manufactured-development-preflight",
                "stage_count": 3,
                "pitch_m": 0.006,
                "chamber_radius_m": 0.009,
                "first_polarity": 1,
            }
        ),
    )
    if definition.case_id in held_out_manifest().case_ids:
        raise ValueError("manufactured development case overlaps held-out manifest")
    return build_case(definition)


def _manufactured_validation_registration(
    case: BuiltCase,
    code_hash: str,
) -> HeldOutValidationRegistration:
    manifest_id = "manufactured-v7-non-held-out-registration"
    case_ids = (case.definition.case_id,)
    family_ids = ("manufactured-v7-registration-family",)
    manifest = ValidationSetManifest(
        manifest_id,
        case_ids,
        family_ids,
        validation_set_manifest_hash(manifest_id, case_ids, family_ids),
    )
    return HeldOutValidationRegistration(
        development_manifest=CFT_V4_DEVELOPMENT_MANIFEST,
        held_out_manifest=manifest,
        evaluated_case_id=case.definition.case_id,
        evaluated_geometry_family_id=family_ids[0],
        required_case_count=1,
        required_outcomes=(
            HeldOutCaseRegistration(case.definition.case_id, family_ids[0]),
        ),
        validation_adapter_id=HeldOutArtifactAdapter.adapter_id,
        validation_adapter_code_hash=code_hash,
        validation_code_hash=code_hash,
        validation_config_hash=PROTOCOL_SEMANTIC_SHA256,
        policy=HeldOutValidationPolicy(
            maximum_age_s=3600.0,
            maximum_future_skew_s=5.0,
        ),
    )


def run_production_field_pipeline_preflight(
    context: RunContext,
    closure: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the actual v7 v1.2 adapter path without held-out geometry access."""

    case = _manufactured_development_case()
    policy = map_policy()
    code_hash = normalized_text_hash(Path(__file__).read_text(encoding="utf-8"))
    solved: dict[str, SolvedMap] = {}
    claims_by_role: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    byte_equalities: dict[str, bool] = {}
    for role in ("primary", "refined", "enlarged"):
        context.before_expensive(
            f"v7-production-preflight-{role}",
            kind="solver",
            details={
                "role": role,
                "partition": "manufactured-development",
                "held_out": False,
            },
        )
        item = solve_map(case, role, closure, runtime)
        path = f"preflight/production-fields/{role}-field-v1.2.json"
        context.write_blob(path, item.artifact_bytes)
        stored = context.store.read_bytes(path)
        reloaded = reload_field_artifact_bytes(
            stored,
            source=f"v7-production-preflight-{role}",
            allow_legacy_v1_1=False,
        )
        binding = CanonicalFieldV12Binding(
            geometry_hash=case.geometry_sha256,
            code_hash=closure["closure_semantic_sha256"],
            backend_version=f"warp-{runtime['warp_version']}",
            generated_at_utc=runtime["generated_at_utc"],
        )
        adapter = CanonicalFieldV12Adapter(binding)
        claims = adapter.verify_v3_artifact(stored)
        accepted = verify_canonical_field_v12_artifact(
            stored,
            binding,
            policy,
            reference_time_utc=runtime["generated_at_utc"],
        )
        equality = (
            stored == item.artifact_bytes
            and stored == field_artifact_canonical_bytes(reloaded)
        )
        if not equality:
            raise ValueError("production preflight field bytes changed across reload")
        contract = adapter.version_contract
        if not (
            reloaded["schema_version"] == ARTIFACT_SCHEMA_VERSION
            == V4_FIELD_ARTIFACT_SCHEMA
            and reloaded["integrity"]["canonicalization"]
            == V4_FIELD_CANONICALIZATION
            and claims.artifact_schema_version == ARTIFACT_SCHEMA_VERSION
            and claims.model_level == "L1a"
            and contract.input_schema_version == ARTIFACT_SCHEMA_VERSION
            and contract.normalized_schema_version == ARTIFACT_SCHEMA_VERSION
            and contract.model_level == "L1a"
            and contract.is_migration is False
        ):
            raise ValueError("production v1.2 adapter contract preflight failed")
        snapshot = reverify_v3_evidence(
            accepted,
            reference_time_utc=runtime["generated_at_utc"],
        )
        if (
            snapshot.migration_manifest_bytes is not None
            or snapshot.migration_source_artifact_bytes is not None
        ):
            raise ValueError("direct v1.2 production evidence contains migration metadata")
        item.evidence = accepted
        solved[role] = item
        claims_by_role[role] = claims
        evidence[role] = accepted
        byte_equalities[role] = equality

    map_set = verify_v4_map_set(
        evidence["primary"],
        evidence["refined"],
        evidence["enlarged"],
        reference_time_utc=runtime["generated_at_utc"],
    )
    registrations = registrations_for(case)
    orbit_adapter = MapGuidingCenterOrbitAdapter(map_set, registrations)
    record = build_cft_coupling_record(
        map_set,
        geometry=CFTGeometry(
            case.definition.chamber_radius_m,
            0.0,
            case.chamber_length_m,
            float(
                PROTOCOL["criterion"]["axial_core"]["core_radius_wall_fraction"]
            )
            * case.definition.chamber_radius_m,
            case.definition.geometry_id,
        ),
        registrations=registrations,
        validation_registration=_manufactured_validation_registration(
            case, code_hash
        ),
        orbit_adapter=orbit_adapter,
        criterion=V4Criterion(),
        reference_time_utc=runtime["generated_at_utc"],
        **policies_for(case),
    )
    if not (
        record.schema_version == COUPLING_V4_SCHEMA_VERSION
        and record.criterion.criterion_version == "4.0.0"
        and tuple(record.field_migration_manifest_hashes) == (None, None, None)
        and tuple(record.field_migration_source_artifact_hashes)
        == (None, None, None)
    ):
        raise ValueError("production v4.2 record contract preflight failed")
    snapshots = reverify_v4_map_set(
        map_set,
        reference_time_utc=runtime["generated_at_utc"],
    )
    return {
        "status": "passed",
        "held_out_case_access_count": 0,
        "manufactured_case_id": case.definition.case_id,
        "field_schema_version": ARTIFACT_SCHEMA_VERSION,
        "field_canonicalization": V4_FIELD_CANONICALIZATION,
        "adapter_id": snapshots[0].adapter_id,
        "adapter_contract": snapshots[0].adapter_contract,
        "byte_equality_by_role": byte_equalities,
        "map_hashes": tuple(item.field_map.full_map_hash for item in snapshots),
        "evidence_fingerprints": v4_map_set_evidence_fingerprints(
            map_set,
            reference_time_utc=runtime["generated_at_utc"],
        ),
        "migration_manifest_hashes": record.field_migration_manifest_hashes,
        "migration_source_artifact_hashes": (
            record.field_migration_source_artifact_hashes
        ),
        "coupling_schema_version": record.schema_version,
        "criterion_version": record.criterion.criterion_version,
        "record_status": record.status.value,
        "orbit_diagnostic_count": len(orbit_adapter.diagnostics),
        "policy": policy,
    }


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
    """Psi-consistent nested midpoint-Boris numerical and physics verifier."""

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
            b"cft-wall-cusp-psi-midpoint-boris-v7\0" + bytes.fromhex(source_hash)
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
                key: declaration[key]
                for key in (
                    "nested_refinement_multipliers",
                    "base_steps_per_fastest_gyroperiod",
                    "maximum_declared_field_t",
                    "maximum_gyro_periods",
                    "maximum_phase_aligned_terminal_state_relative_difference",
                    "maximum_phase_aligned_trajectory_relative_difference",
                    "maximum_energy_relative_drift",
                    "maximum_cross_map_metric_relative_change",
                    "maximum_curvature_ordering_ratio",
                    "maximum_rho_over_lb",
                    "gyro_average_minimum_samples",
                    "maximum_polyline_length_relative_defect",
                )
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
        encoded = json.dumps(
            {
                "full_map_hash": full_map_hash,
                "seed_id": seed_id,
                "direction": direction,
                "psi_start_wb": psi_start_wb,
                "points": points,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(b"cft-v4-field-line-path\0" + encoded).hexdigest()

    def _bound_snapshot(
        self,
        path: tuple[tuple[float, float], ...],
        path_hash: str,
    ) -> tuple[Any, str, int]:
        matches: list[tuple[Any, str, int]] = []
        for snapshot in self._snapshots:
            psi = bilinear_sample(snapshot.field_map, snapshot.field_map.psi_wb, path[0])
            for seed_id in self._seed_ids:
                for direction in (-1, 1):
                    if self._path_hash(
                        snapshot.field_map.full_map_hash,
                        seed_id,
                        direction,
                        psi,
                        path,
                    ) == path_hash:
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

    @staticmethod
    def _cross(
        left: tuple[float, float, float],
        right: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    @staticmethod
    def _cell_index(values: Sequence[float], value: float) -> int:
        import bisect

        if value < values[0] or value > values[-1]:
            raise ValueError("orbit sample left accepted field domain")
        return min(len(values) - 2, max(0, bisect.bisect_right(values, value) - 1))

    @classmethod
    def _psi_field_sample(
        cls,
        field: Any,
        point: tuple[float, float],
    ) -> dict[str, float]:
        """Bilinear psi gradient with an explicit regular-axis limit."""

        radius, axial = point
        i = cls._cell_index(field.r_m, radius)
        j = cls._cell_index(field.z_m, axial)
        z0, z1 = field.z_m[j], field.z_m[j + 1]
        dz = z1 - z0
        v = (axial - z0) / dz
        start = min(max(i - 1, 0), len(field.r_m) - 3)
        radial_indices = (start, start + 1, start + 2)
        radial_nodes = tuple(field.r_m[index] for index in radial_indices)
        basis: list[float] = []
        derivatives: list[float] = []
        for k, xk in enumerate(radial_nodes):
            others = [index for index in range(3) if index != k]
            denominator = math.prod(xk - radial_nodes[index] for index in others)
            basis.append(
                math.prod(radius - radial_nodes[index] for index in others)
                / denominator
            )
            derivatives.append(
                math.fsum(
                    math.prod(
                        radius - radial_nodes[index]
                        for index in others
                        if index != differentiated
                    )
                    for differentiated in others
                )
                / denominator
            )
        axial_values = tuple(
            (1.0 - v) * field.psi_wb[index][j]
            + v * field.psi_wb[index][j + 1]
            for index in radial_indices
        )
        axial_differences = tuple(
            (field.psi_wb[index][j + 1] - field.psi_wb[index][j]) / dz
            for index in radial_indices
        )
        psi = math.fsum(
            weight * value for weight, value in zip(basis, axial_values, strict=True)
        )
        dpsi_dr = math.fsum(
            weight * value
            for weight, value in zip(derivatives, axial_values, strict=True)
        )
        dpsi_dz = math.fsum(
            weight * value
            for weight, value in zip(basis, axial_differences, strict=True)
        )
        axis_tolerance = max(1e-15, abs(field.r_m[-1]) * 1e-14)
        if abs(radius) <= axis_tolerance:
            if field.r_m[0] != 0.0 or len(field.r_m) < 3:
                raise ValueError("psi-gradient axis limit requires three radial nodes")
            psi1 = (1.0 - v) * field.psi_wb[1][j] + v * field.psi_wb[1][j + 1]
            psi2 = (1.0 - v) * field.psi_wb[2][j] + v * field.psi_wb[2][j + 1]
            axis_dr = field.r_m[1] - field.r_m[0]
            br, bz = 0.0, (16.0 * psi1 - psi2) / (6.0 * axis_dr * axis_dr)
            divergence_identity = 0.0
        else:
            br, bz = -dpsi_dz / radius, dpsi_dr / radius
            mixed = math.fsum(
                weight * value
                for weight, value in zip(
                    derivatives, axial_differences, strict=True
                )
            )
            divergence_identity = abs((-mixed + mixed) / radius)
        b = math.hypot(br, bz)
        if not all(math.isfinite(item) for item in (psi, br, bz, b)):
            raise ValueError("psi-gradient field sample is nonfinite")
        return {
            "psi_wb": psi,
            "br_t": br,
            "bz_t": bz,
            "b_t": b,
            "within_cell_divergence_identity_t_per_m": divergence_identity,
        }

    @classmethod
    def _cartesian_field(
        cls,
        field: Any,
        position: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], dict[str, float]]:
        radius = math.hypot(position[0], position[1])
        sample = cls._psi_field_sample(field, (radius, position[2]))
        radial_x = 1.0 if radius == 0.0 else position[0] / radius
        radial_y = 0.0 if radius == 0.0 else position[1] / radius
        magnetic = (
            sample["br_t"] * radial_x,
            sample["br_t"] * radial_y,
            sample["bz_t"],
        )
        return magnetic, sample

    @classmethod
    def _boris_rotate(
        cls,
        velocity: tuple[float, float, float],
        magnetic: tuple[float, float, float],
        dt: float,
    ) -> tuple[float, float, float]:
        factor = -cls._ELEMENTARY_CHARGE_C * dt / (2.0 * cls._ELECTRON_MASS_KG)
        t_vector = tuple(factor * item for item in magnetic)
        t_squared = math.fsum(item * item for item in t_vector)
        s_vector = tuple(2.0 * item / (1.0 + t_squared) for item in t_vector)
        v_prime_delta = cls._cross(velocity, t_vector)
        v_prime = tuple(velocity[index] + v_prime_delta[index] for index in range(3))
        v_plus_delta = cls._cross(v_prime, s_vector)
        return tuple(velocity[index] + v_plus_delta[index] for index in range(3))

    @classmethod
    def _midpoint_boris_step(
        cls,
        field: Any,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
        dt: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        midpoint = tuple(position[k] + 0.5 * dt * velocity[k] for k in range(3))
        updated = velocity
        endpoint = position
        for _ in range(3):
            magnetic, sample = cls._cartesian_field(field, midpoint)
            if sample["b_t"] <= 0.0:
                raise ValueError("midpoint orbit field magnitude must be positive")
            updated = cls._boris_rotate(velocity, magnetic, dt)
            endpoint = tuple(
                position[k] + 0.5 * dt * (velocity[k] + updated[k])
                for k in range(3)
            )
            midpoint = tuple(0.5 * (position[k] + endpoint[k]) for k in range(3))
        return endpoint, updated

    @classmethod
    def _ordering_metrics(
        cls,
        field: Any,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> dict[str, float]:
        magnetic, sample = cls._cartesian_field(field, position)
        b = sample["b_t"]
        if b <= 0.0:
            raise ValueError("ordering metrics require positive field")
        v_parallel = math.fsum(velocity[k] * magnetic[k] for k in range(3)) / b
        speed2 = math.fsum(item * item for item in velocity)
        v_perp = math.sqrt(max(0.0, speed2 - v_parallel * v_parallel))
        rho = cls._ELECTRON_MASS_KG * v_perp / (cls._ELEMENTARY_CHARGE_C * b)
        radius = math.hypot(position[0], position[1])
        h = 0.25 * min(
            field.r_m[1] - field.r_m[0],
            field.z_m[1] - field.z_m[0],
        )
        r_minus = max(field.r_m[0], radius - h)
        r_plus = min(field.r_m[-1], radius + h)
        z_minus = max(field.z_m[0], position[2] - h)
        z_plus = min(field.z_m[-1], position[2] + h)
        b_rm = cls._psi_field_sample(field, (r_minus, position[2]))["b_t"]
        b_rp = cls._psi_field_sample(field, (r_plus, position[2]))["b_t"]
        b_zm = cls._psi_field_sample(field, (radius, z_minus))["b_t"]
        b_zp = cls._psi_field_sample(field, (radius, z_plus))["b_t"]
        d_b_dr = 0.0 if r_plus == r_minus else (b_rp - b_rm) / (r_plus - r_minus)
        d_b_dz = 0.0 if z_plus == z_minus else (b_zp - b_zm) / (z_plus - z_minus)
        gradient_b = math.hypot(d_b_dr, d_b_dz)
        lb = math.inf if gradient_b == 0.0 else b / gradient_b
        unit_r, unit_z = sample["br_t"] / b, sample["bz_t"] / b
        minus_point = (
            min(field.r_m[-1], max(field.r_m[0], radius - h * unit_r)),
            min(field.z_m[-1], max(field.z_m[0], position[2] - h * unit_z)),
        )
        plus_point = (
            min(field.r_m[-1], max(field.r_m[0], radius + h * unit_r)),
            min(field.z_m[-1], max(field.z_m[0], position[2] + h * unit_z)),
        )
        minus = cls._psi_field_sample(field, minus_point)
        plus = cls._psi_field_sample(field, plus_point)
        distance = math.hypot(plus_point[0] - minus_point[0], plus_point[1] - minus_point[1])
        if distance == 0.0 or minus["b_t"] == 0.0 or plus["b_t"] == 0.0:
            curvature = 0.0
        else:
            curvature = math.hypot(
                plus["br_t"] / plus["b_t"] - minus["br_t"] / minus["b_t"],
                plus["bz_t"] / plus["b_t"] - minus["bz_t"] / minus["b_t"],
            ) / distance
        return {
            "b_t": b,
            "parallel_velocity_m_per_s": v_parallel,
            "rho_m": rho,
            "field_scale_length_m": lb,
            "rho_over_lb": 0.0 if math.isinf(lb) else rho / max(lb, 1e-300),
            "curvature_per_m": curvature,
            "rho_times_curvature": rho * curvature,
        }

    @classmethod
    def _state_sample(
        cls,
        field: Any,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
        energy_j: float,
    ) -> dict[str, Any]:
        magnetic, field_sample = cls._cartesian_field(field, position)
        ordering = cls._ordering_metrics(field, position, velocity)
        b = ordering["b_t"]
        parallel = ordering["parallel_velocity_m_per_s"]
        speed2 = math.fsum(item * item for item in velocity)
        perpendicular2 = max(0.0, speed2 - parallel * parallel)
        mu = cls._ELECTRON_MASS_KG * perpendicular2 / (2.0 * b)
        energy = 0.5 * cls._ELECTRON_MASS_KG * speed2
        cross_v_b = cls._cross(velocity, magnetic)
        signed_charge = -cls._ELEMENTARY_CHARGE_C
        guiding_centre = tuple(
            position[k] + cls._ELECTRON_MASS_KG * cross_v_b[k] / (signed_charge * b * b)
            for k in range(3)
        )
        return {
            "position_xyz_m": position,
            "velocity_xyz_m_per_s": velocity,
            "guiding_centre_xyz_m": guiding_centre,
            "mu_j_per_t": mu,
            "energy_j": energy,
            "energy_relative_drift": abs(energy - energy_j) / max(energy_j, 1e-300),
            "pitch_angle_rad": math.atan2(math.sqrt(perpendicular2), abs(parallel)),
            **ordering,
            "within_cell_divergence_identity_t_per_m": field_sample[
                "within_cell_divergence_identity_t_per_m"
            ],
        }

    @classmethod
    def _evolution(
        cls,
        snapshot: Any,
        path: tuple[tuple[float, float], ...],
        sample: ElectronOrbitSample,
        direction: int,
        duration_s: float,
        step_count: int,
        maximum_path_b_t: float,
    ) -> dict[str, Any]:
        field = snapshot.field_map
        energy_j = sample.kinetic_energy_ev * cls._ELEMENTARY_CHARGE_C
        speed = math.sqrt(2.0 * energy_j / cls._ELECTRON_MASS_KG)
        start = cls._psi_field_sample(field, path[0])
        if start["b_t"] <= 0.0:
            raise ValueError("orbit launch field magnitude must be positive")
        parallel_speed = direction * speed * math.cos(sample.pitch_angle_rad)
        perpendicular_speed = speed * math.sin(sample.pitch_angle_rad)
        position = (path[0][0], 0.0, path[0][1])
        velocity = (
            parallel_speed * start["br_t"] / start["b_t"],
            perpendicular_speed,
            parallel_speed * start["bz_t"] / start["b_t"],
        )
        dt = duration_s / step_count
        states = [cls._state_sample(field, position, velocity, energy_j)]
        completed = True
        initial_parallel = states[0]["parallel_velocity_m_per_s"]
        mirror_detected = False
        for _ in range(step_count):
            try:
                position, velocity = cls._midpoint_boris_step(field, position, velocity, dt)
                state = cls._state_sample(field, position, velocity, energy_j)
            except ValueError:
                completed = False
                break
            states.append(state)
            mirror_detected = mirror_detected or (
                initial_parallel * state["parallel_velocity_m_per_s"] < 0.0
            )
        mu0 = states[0]["mu_j_per_t"]
        instantaneous_mu_variation = max(
            abs(item["mu_j_per_t"] - mu0) / max(abs(mu0), 1e-300)
            for item in states
        )
        multiplier = max(1, step_count // max(1, len(states) - 1))
        fast_periods = max(
            duration_s * cls._ELEMENTARY_CHARGE_C * maximum_path_b_t
            / (2.0 * math.pi * cls._ELECTRON_MASS_KG),
            1e-12,
        )
        window = max(
            int(PROTOCOL["orbit_verification"]["gyro_average_minimum_samples"]),
            round(step_count / fast_periods),
        )
        window = min(window, len(states))
        averages = [
            math.fsum(item["mu_j_per_t"] for item in states[index:index + window]) / window
            for index in range(0, len(states) - window + 1, max(1, window // 2))
        ]
        if not averages:
            averages = [math.fsum(item["mu_j_per_t"] for item in states) / len(states)]
        average_reference = averages[0]
        gyro_mu_variation = max(
            abs(item - average_reference) / max(abs(average_reference), 1e-300)
            for item in averages
        )
        return {
            "step_count": step_count,
            "completed_step_count": len(states) - 1,
            "completed": completed and len(states) == step_count + 1,
            "duration_s": duration_s,
            "timestep_s": dt,
            "maximum_path_b_t": maximum_path_b_t,
            "states": states,
            "mu_initial_j_per_t": mu0,
            "instantaneous_mu_relative_variation": instantaneous_mu_variation,
            "gyro_averaged_mu_relative_variation": gyro_mu_variation,
            "gyro_averaged_mu_min_j_per_t": min(averages),
            "gyro_averaged_mu_max_j_per_t": max(averages),
            "maximum_energy_relative_drift": max(item["energy_relative_drift"] for item in states),
            "maximum_pitch_angle_change_rad": max(
                abs(item["pitch_angle_rad"] - sample.pitch_angle_rad) for item in states
            ),
            "maximum_rho_over_lb": max(item["rho_over_lb"] for item in states),
            "maximum_rho_times_curvature": max(
                item["rho_times_curvature"] for item in states
            ),
            "maximum_within_cell_divergence_identity_t_per_m": max(
                item["within_cell_divergence_identity_t_per_m"] for item in states
            ),
            "mirror_detected": mirror_detected,
        }

    @staticmethod
    def _phase_error(
        left: Mapping[str, Any],
        right: Mapping[str, Any],
        position_scale_m: float,
        speed_scale_m_per_s: float,
    ) -> float:
        centre = math.sqrt(math.fsum(
            (a - b) ** 2
            for a, b in zip(
                left["guiding_centre_xyz_m"],
                right["guiding_centre_xyz_m"],
                strict=True,
            )
        )) / max(position_scale_m, 1e-300)
        parallel = abs(
            left["parallel_velocity_m_per_s"] - right["parallel_velocity_m_per_s"]
        ) / max(speed_scale_m_per_s, 1e-300)
        energy = abs(left["energy_j"] - right["energy_j"]) / max(right["energy_j"], 1e-300)
        return max(centre, parallel, energy)

    @classmethod
    def _refinement_error(
        cls,
        coarse: Mapping[str, Any],
        fine: Mapping[str, Any],
        position_scale_m: float,
        speed_scale_m_per_s: float,
    ) -> dict[str, float]:
        ratio = fine["step_count"] // coarse["step_count"]
        common = min(len(coarse["states"]), (len(fine["states"]) - 1) // ratio + 1)
        errors = [
            cls._phase_error(
                coarse["states"][index],
                fine["states"][index * ratio],
                position_scale_m,
                speed_scale_m_per_s,
            )
            for index in range(common)
        ]
        return {
            "phase_aligned_terminal_state_relative_difference": errors[-1],
            "phase_aligned_trajectory_relative_difference": max(errors),
        }

    @classmethod
    def _nested_evolution(
        cls,
        snapshot: Any,
        path: tuple[tuple[float, float], ...],
        sample: ElectronOrbitSample,
        direction: int,
    ) -> dict[str, Any]:
        field = snapshot.field_map
        path_samples = [
            cls._psi_field_sample(field, point)
            for pair in zip(path[:-1], path[1:], strict=True)
            for point in (pair[0], ((pair[0][0] + pair[1][0]) / 2.0, (pair[0][1] + pair[1][1]) / 2.0))
        ]
        path_samples.append(cls._psi_field_sample(field, path[-1]))
        declared = float(PROTOCOL["orbit_verification"]["maximum_declared_field_t"])
        maximum_path_b = max(declared, *(item["b_t"] for item in path_samples))
        fastest_period = 2.0 * math.pi * cls._ELECTRON_MASS_KG / (
            cls._ELEMENTARY_CHARGE_C * maximum_path_b
        )
        energy_j = sample.kinetic_energy_ev * cls._ELEMENTARY_CHARGE_C
        speed = math.sqrt(2.0 * energy_j / cls._ELECTRON_MASS_KG)
        parallel_speed = speed * max(abs(math.cos(sample.pitch_angle_rad)), 0.1)
        transit_time = cls._length(path) / parallel_speed
        duration = min(
            transit_time,
            float(PROTOCOL["orbit_verification"]["maximum_gyro_periods"]) * fastest_period,
        )
        base_per_period = int(PROTOCOL["orbit_verification"]["base_steps_per_fastest_gyroperiod"])
        base_steps = max(4, math.ceil(duration / fastest_period * base_per_period))
        multipliers = tuple(int(item) for item in PROTOCOL["orbit_verification"]["nested_refinement_multipliers"])
        refinements = tuple(
            cls._evolution(
                snapshot,
                path,
                sample,
                direction,
                duration,
                base_steps * multiplier,
                maximum_path_b,
            )
            for multiplier in multipliers
        )
        coarse_medium = cls._refinement_error(refinements[0], refinements[1], cls._length(path), speed)
        medium_fine = cls._refinement_error(refinements[1], refinements[2], cls._length(path), speed)
        return {
            "duration_s": duration,
            "fastest_gyro_period_s": fastest_period,
            "maximum_declared_or_path_b_t": maximum_path_b,
            "base_step_count": base_steps,
            "nested_step_counts": tuple(item["step_count"] for item in refinements),
            "exact_terminal_times_s": tuple(item["duration_s"] for item in refinements),
            "refinements": refinements,
            "coarse_to_medium": coarse_medium,
            "medium_to_fine": medium_fine,
        }

    def verify_orbit(
        self,
        path_points_rz_m: tuple[tuple[float, float], ...],
        path_hash: str,
        sample: ElectronOrbitSample,
    ) -> OrbitVerificationClaims:
        snapshot, seed_id, direction = self._bound_snapshot(path_points_rz_m, path_hash)
        nested = self._nested_evolution(snapshot, path_points_rz_m, sample, direction)
        fine = nested["refinements"][-1]
        policy = PROTOCOL["orbit_verification"]
        terminal_error = nested["medium_to_fine"][
            "phase_aligned_terminal_state_relative_difference"
        ]
        trajectory_error = nested["medium_to_fine"][
            "phase_aligned_trajectory_relative_difference"
        ]
        numerical_reasons: list[str] = []
        if not all(item["completed"] for item in nested["refinements"]):
            numerical_reasons.append("trajectory-left-domain")
        if terminal_error > float(policy["maximum_phase_aligned_terminal_state_relative_difference"]):
            numerical_reasons.append("terminal-state-refinement")
        if trajectory_error > float(policy["maximum_phase_aligned_trajectory_relative_difference"]):
            numerical_reasons.append("trajectory-refinement")
        if fine["maximum_energy_relative_drift"] > float(policy["maximum_energy_relative_drift"]):
            numerical_reasons.append("energy-drift")
        numerical_converged = not numerical_reasons
        physical_reasons: list[str] = []
        mu_variation = float(fine["gyro_averaged_mu_relative_variation"])
        if mu_variation > sample.maximum_mu_relative_variation:
            physical_reasons.append("gyro-averaged-mu-variation")
        if fine["maximum_rho_over_lb"] > float(policy["maximum_rho_over_lb"]):
            physical_reasons.append("rho-over-LB-ordering")
        if fine["maximum_rho_times_curvature"] > float(policy["maximum_curvature_ordering_ratio"]):
            physical_reasons.append("field-line-curvature-ordering")
        physical_adiabatic = not physical_reasons
        classification = (
            "ORBIT_UNVERIFIED"
            if not numerical_converged
            else "RESOLVED"
            if physical_adiabatic
            else "NONADIABATIC"
        )
        length = self._length(path_points_rz_m)
        coarse_points = path_points_rz_m[::2]
        if coarse_points[-1] != path_points_rz_m[-1]:
            coarse_points = (*coarse_points, path_points_rz_m[-1])
        polyline_defect = abs(length - self._length(coarse_points)) / max(length, 1e-300)
        nested_evidence = {
            key: value
            for key, value in nested.items()
            if key != "refinements"
        }
        nested_evidence["refinements"] = tuple(
            {
                **{
                    key: value
                    for key, value in refinement.items()
                    if key != "states"
                },
                "state_sample_count": len(refinement["states"]),
            }
            for refinement in nested["refinements"]
        )
        self.diagnostics[(path_hash, sample.sample_id)] = {
            "path_hash": path_hash,
            "sample_id": sample.sample_id,
            "seed_id": seed_id,
            "direction": direction,
            "full_map_hash": snapshot.field_map.full_map_hash,
            "kinetic_energy_ev": sample.kinetic_energy_ev,
            "initial_pitch_angle_rad": sample.pitch_angle_rad,
            "classification": classification,
            "numerical_converged": numerical_converged,
            "physical_adiabatic": physical_adiabatic,
            "numerical_reasons": numerical_reasons,
            "physical_reasons": physical_reasons,
            "nested": nested_evidence,
            "gyro_averaged_mu_relative_variation": mu_variation,
            "instantaneous_mu_relative_variation": fine["instantaneous_mu_relative_variation"],
            "maximum_energy_relative_drift": fine["maximum_energy_relative_drift"],
            "maximum_rho_over_lb": fine["maximum_rho_over_lb"],
            "maximum_rho_times_curvature": fine["maximum_rho_times_curvature"],
            "mirror_detected": fine["mirror_detected"],
            "phase_aligned_terminal_state_relative_difference": terminal_error,
            "phase_aligned_trajectory_relative_difference": trajectory_error,
            "polyline_length_relative_defect": polyline_defect,
            "path_length_gate_passed": polyline_defect <= float(
                policy["maximum_polyline_length_relative_defect"]
            ),
        }
        return OrbitVerificationClaims(
            path_hash=path_hash,
            sample_id=sample.sample_id,
            converged=numerical_converged,
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


def run_orbit_manufactured_preflight() -> dict[str, Any]:
    """Verify uniform, gradient, mirror, axis, and divergence behavior."""

    from types import SimpleNamespace

    radii = tuple(0.001 * index / 16 for index in range(17))
    axial = tuple(-0.001 + 0.002 * index / 64 for index in range(65))
    b0 = 0.2

    def make_field(name: str, psi_function: Any) -> Any:
        return SimpleNamespace(
            r_m=radii,
            z_m=axial,
            psi_wb=tuple(
                tuple(psi_function(radius, z) for z in axial)
                for radius in radii
            ),
            full_map_hash=hashlib.sha256(name.encode()).hexdigest(),
        )

    fields = {
        "uniform": make_field("uniform", lambda r, z: 0.5 * b0 * r * r),
        "gradient": make_field(
            "gradient",
            lambda r, z: 0.5 * b0 * (1.0 + 40.0 * z) * r * r,
        ),
        "mirror": make_field(
            "mirror",
            lambda r, z: 0.5 * b0 * (1.0 + 1.0e7 * z * z) * r * r,
        ),
    }
    axis = MapGuidingCenterOrbitAdapter._psi_field_sample(
        fields["uniform"], (0.0, 0.0)
    )
    interior = MapGuidingCenterOrbitAdapter._psi_field_sample(
        fields["gradient"], (0.0001, 0.0001)
    )
    if (
        abs(axis["br_t"]) > 1e-15
        or abs(axis["bz_t"] - b0) > 1e-12
        or interior["within_cell_divergence_identity_t_per_m"] > 1e-12
    ):
        raise ValueError("manufactured psi-gradient axis/divergence check failed")
    sample = ElectronOrbitSample(
        "manufactured-30eV-p70",
        30.0,
        math.radians(70.0),
        0.5,
        0.02,
    )
    path = ((0.0001, 0.0), (0.0001, 0.0003))
    rows: dict[str, Any] = {}
    for name, map_field in fields.items():
        nested = MapGuidingCenterOrbitAdapter._nested_evolution(
            SimpleNamespace(field_map=map_field),
            path,
            sample,
            1,
        )
        fine = nested["refinements"][-1]
        steps = nested["nested_step_counts"]
        times = nested["exact_terminal_times_s"]
        metric_names = (
            "instantaneous_mu_relative_variation",
            "gyro_averaged_mu_relative_variation",
            "maximum_energy_relative_drift",
            "maximum_rho_over_lb",
            "maximum_rho_times_curvature",
        )
        if (
            steps[1] != 2 * steps[0]
            or steps[2] != 4 * steps[0]
            or len(set(times)) != 1
            or not all(item["completed"] for item in nested["refinements"])
            or not all(math.isfinite(float(fine[key])) for key in metric_names)
        ):
            raise ValueError(f"manufactured {name} orbit verification failed")
        rows[name] = {
            "nested_step_counts": steps,
            "exact_terminal_time_s": times[0],
            **{key: fine[key] for key in metric_names},
            "mirror_detected": fine["mirror_detected"],
            "phase_aligned_terminal_state_relative_difference": nested[
                "medium_to_fine"
            ]["phase_aligned_terminal_state_relative_difference"],
            "phase_aligned_trajectory_relative_difference": nested[
                "medium_to_fine"
            ]["phase_aligned_trajectory_relative_difference"],
        }
    if (
        rows["uniform"]["maximum_energy_relative_drift"]
        > float(PROTOCOL["orbit_verification"]["maximum_energy_relative_drift"])
        or rows["uniform"]["gyro_averaged_mu_relative_variation"] > 0.02
        or not rows["mirror"]["mirror_detected"]
    ):
        raise ValueError("manufactured invariant or mirror preflight failed")
    return {
        "status": "passed",
        "held_out_case_access_count": 0,
        "field_source": "manufactured divergence-free psi functions",
        "axis_bz_t": axis["bz_t"],
        "axis_br_t": axis["br_t"],
        "within_cell_divergence_identity_t_per_m": interior[
            "within_cell_divergence_identity_t_per_m"
        ],
        "fields": rows,
    }


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
    original_field = solved.artifact["field_map"]
    differences = {
        "psi_max_abs_wb": max(
            abs(float(left) - right)
            for left_row, right_row in zip(
                original_field["psi_wb"], replay.psi_wb, strict=True
            )
            for left, right in zip(left_row, right_row, strict=True)
        ),
        "br_max_abs_t": max(
            abs(float(left) - right)
            for left_row, right_row in zip(
                original_field["b_r_t"], replay.b_r_t, strict=True
            )
            for left, right in zip(left_row, right_row, strict=True)
        ),
        "bz_max_abs_t": max(
            abs(float(left) - right)
            for left_row, right_row in zip(
                original_field["b_z_t"], replay.b_z_t, strict=True
            )
            for left, right in zip(left_row, right_row, strict=True)
        ),
    }
    config = solver_config()
    original = solved.artifact["diagnostics"]
    repeated = replay.diagnostics
    original_scale = max(
        config.absolute_tolerance,
        config.relative_tolerance * original["initial_residual_l2"],
    )
    replay_scale = max(
        config.absolute_tolerance,
        config.relative_tolerance * repeated.initial_residual_l2,
    )
    residual_difference = abs(
        original["final_residual_l2"] - repeated.final_residual_l2
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
        "original_initial_residual_l2": original["initial_residual_l2"],
        "replay_initial_residual_l2": repeated.initial_residual_l2,
        "original_final_residual_l2": original["final_residual_l2"],
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
        "finite_boundary_nulls": [
            serialize_boundary_null_diagnostic(item) for item in boundary
        ],
    }


def _orbit_gate_summary(
    record: Any,
    adapter: MapGuidingCenterOrbitAdapter,
) -> dict[str, Any]:
    assessments = (
        record.stability.primary,
        record.stability.refined,
        record.stability.enlarged,
    )
    coupling_orbits = [
        orbit
        for assessment in assessments
        for cell in assessment.cells
        for seed in cell.seed_outcomes
        for path in (seed.negative_path, seed.positive_path)
        for orbit in path.orbit_assessments
    ]
    diagnostics = list(adapter.diagnostics.values())
    numerical_reason_counts: dict[str, int] = {}
    physical_reason_counts: dict[str, int] = {}
    classification_counts: dict[str, int] = {}
    for item in diagnostics:
        classification = str(item["classification"])
        classification_counts[classification] = (
            classification_counts.get(classification, 0) + 1
        )
        for reason in item["numerical_reasons"]:
            numerical_reason_counts[reason] = numerical_reason_counts.get(reason, 0) + 1
        for reason in item["physical_reasons"]:
            physical_reason_counts[reason] = physical_reason_counts.get(reason, 0) + 1
    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for item in diagnostics:
        key = (str(item["seed_id"]), int(item["direction"]), str(item["sample_id"]))
        groups.setdefault(key, []).append(item)
    cross_map_limit = float(
        PROTOCOL["orbit_verification"]["maximum_cross_map_metric_relative_change"]
    )
    cross_map_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        metrics = {
            name: [float(row[name]) for row in rows]
            for name in (
                "gyro_averaged_mu_relative_variation",
                "instantaneous_mu_relative_variation",
                "maximum_rho_over_lb",
                "maximum_rho_times_curvature",
            )
        }
        changes = {
            name: (
                max(values) - min(values)
            ) / max(max(abs(value) for value in values), 1e-300)
            for name, values in metrics.items()
        }
        passed = (
            len(rows) == 3
            and len({row["full_map_hash"] for row in rows}) == 3
            and all(bool(row["numerical_converged"]) for row in rows)
            and all(value <= cross_map_limit for value in changes.values())
        )
        cross_map_rows.append(
            {
                "seed_id": key[0],
                "direction": key[1],
                "sample_id": key[2],
                "map_count": len(rows),
                "relative_metric_changes": changes,
                "passed": passed,
            }
        )
    evaluated = len(diagnostics)
    coupling_reason_counts: dict[str, int] = {}
    for orbit in coupling_orbits:
        coupling_reason_counts[orbit.reason] = coupling_reason_counts.get(orbit.reason, 0) + 1
    finite_values = lambda name: [
        float(item[name])
        for item in diagnostics
        if math.isfinite(float(item[name]))
    ]

    def value_range(name: str) -> list[float | None]:
        values = finite_values(name)
        return [min(values), max(values)] if values else [None, None]

    return {
        "required_sample_count": len(coupling_orbits),
        "adapter_evaluated_sample_count": evaluated,
        "pre_adapter_not_evaluable_count": len(coupling_orbits) - evaluated,
        "numerically_converged_sample_count": sum(
            bool(item["numerical_converged"]) for item in diagnostics
        ),
        "physically_adiabatic_sample_count": sum(
            bool(item["numerical_converged"]) and bool(item["physical_adiabatic"])
            for item in diagnostics
        ),
        "classification_counts": classification_counts,
        "numerical_reason_counts": numerical_reason_counts,
        "physical_reason_counts": physical_reason_counts,
        "coupling_reason_counts": coupling_reason_counts,
        "cross_map_groups": cross_map_rows,
        "cross_map_converged": bool(cross_map_rows)
        and all(item["passed"] for item in cross_map_rows),
        "all_numerically_converged": (
            evaluated == len(coupling_orbits)
            and all(bool(item["numerical_converged"]) for item in diagnostics)
        ),
        "all_physically_adiabatic": (
            evaluated == len(coupling_orbits)
            and all(bool(item["physical_adiabatic"]) for item in diagnostics)
        ),
        "gyro_averaged_mu_relative_variation_range": value_range(
            "gyro_averaged_mu_relative_variation"
        ),
        "instantaneous_mu_relative_variation_range": value_range(
            "instantaneous_mu_relative_variation"
        ),
        "energy_relative_drift_range": value_range(
            "maximum_energy_relative_drift"
        ),
        "rho_over_lb_range": value_range("maximum_rho_over_lb"),
        "rho_times_curvature_range": value_range(
            "maximum_rho_times_curvature"
        ),
        "phase_aligned_terminal_error_range": value_range(
            "phase_aligned_terminal_state_relative_difference"
        ),
        "phase_aligned_trajectory_error_range": value_range(
            "phase_aligned_trajectory_relative_difference"
        ),
        "mirror_detected_count": sum(bool(item["mirror_detected"]) for item in diagnostics),
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
    cusps = [cusp for assessment in assessments for cusp in assessment.cusps]
    uncertainty_evaluable = bool(paths) and all(
        path.mirror_probability is not None
        and path.probability_lower is not None
        and path.probability_upper is not None
        and math.isfinite(path.mirror_probability)
        and math.isfinite(path.probability_lower)
        and math.isfinite(path.probability_upper)
        for path in paths
    )
    uncertainty_ordered = uncertainty_evaluable and all(
        0.0 <= path.probability_lower
        <= path.mirror_probability
        <= path.probability_upper
        <= 1.0
        for path in paths
    )
    failures = _record_failures(record, replay)
    if not uncertainty_evaluable:
        failures.append("UNCERTAINTY_NOT_EVALUABLE")
    elif not uncertainty_ordered:
        failures.append("UNCERTAINTY_BOUNDS_INVALID")
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
        "inter_cusp_cell_count": len(assessments[0].cells),
        "minimum_physical_cusp_prominence_t": min(
            (cusp.prominence_t for cusp in cusps), default=None
        ),
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
        "uncertainty_evaluable": uncertainty_evaluable,
        "uncertainty_bounds_finite_ordered": uncertainty_ordered,
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
        "axial_core_failure_count": sum(
            not cell.axial_metrics.passed
            for assessment in assessments
            for cell in assessment.cells
        ),
        "axial_core_evaluation_count": sum(
            1 for assessment in assessments for cell in assessment.cells
        ),
        "gpu_replay": dict(replay),
        "topology_diagnostics": dict(diagnostics),
        "failures": sorted(set(failures)),
        "passed": passed,
    }


class HeldOutArtifactAdapter:
    adapter_id = "experiments.cft-wall-cusp-validation-v7.held-out-artifact"

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
    import importlib.metadata
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
    dependencies: dict[str, str | None] = {}
    for distribution in ("cft-revival", "numpy", "warp-lang", "pytest"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = None
    return {
        "generated_at_utc": datetime.now(timezone.utc),
        "host": platform.node(),
        "host_machine": platform.machine(),
        "host_processor": platform.processor(),
        "host_release": platform.release(),
        "host_version": platform.version(),
        "operating_system": os.name,
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
        "python_implementation": platform.python_implementation(),
        "python_executable": os.sys.executable,
        "platform": os.sys.platform,
        "dependency_versions": dependencies,
        "nvidia_smi_query": query,
        "nvidia_smi_banner_sha256": hashlib.sha256(
            banner.encode("utf-8")
        ).hexdigest(),
    }




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
        "# Coupling v4.2 wall-cusp held-out validation v7",
        "",
        "Preregistered held-out numerical validation of coupling schema 4.2.",
        "This is numerical/source-consistency evidence, not hardware or experimental validation.",
        "",
        f"- Preregistration commit: `{dataset['preregistration_commit_sha']}`",
        f"- Accepted coupling commit: `{ACCEPTED_COUPLING_COMMIT}`",
        f"- Cases/maps: {summary['case_count']}/{summary['map_count']}",
        f"- Stable cusps/cells: {summary['stable_cusp_count']}/{summary['stable_cell_count']}",
        f"- Wall-connected paths: {summary['wall_connected_path_count']}/{summary['required_path_count']}",
        f"- Resolved orbit samples: {summary['resolved_orbit_count']}/{summary['required_orbit_count']}",
        f"- Numerically converged orbit samples: {summary['numerically_converged_orbit_count']}/{summary['required_orbit_count']}",
        f"- Physically adiabatic orbit samples: {summary['physically_adiabatic_orbit_count']}/{summary['required_orbit_count']}",
        f"- Axial-core failures: {summary['axial_core_failure_count']}/{summary['axial_core_evaluation_count']}",
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






def serialize_map_assessment_checkpoint(assessment: Any) -> dict[str, Any]:
    """Serialize resolved or ambiguous assessments, including empty collections."""

    return {
        "role": assessment.role,
        "identity": _plain_domain_json(assessment.identity),
        "validation_policy": _plain_domain_json(assessment.validation_policy),
        "status": assessment.status.value,
        "reason": assessment.reason,
        "detected_cusp_count": assessment.detected_cusp_count,
        "expected_cusp_count": assessment.expected_cusp_count,
        "cusps": [_plain_domain_json(item) for item in assessment.cusps],
        "cells": [_plain_domain_json(item) for item in assessment.cells],
    }


def _prerecord_checkpoint(record: Any, adapter: Any) -> dict[str, Any]:
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
        "cusp_counts": list(record.stability.cusp_counts),
        "candidate_cell_count": len(cells),
        "resolved_cell_count": sum(cell.status is V4Status.RESOLVED for cell in cells),
        "candidate_path_count": len(paths),
        "resolved_path_count": sum(path.status is V4Status.RESOLVED for path in paths),
        "candidate_orbit_count": len(orbits),
        "resolved_orbit_count": sum(item.status is V4Status.RESOLVED for item in orbits),
        "evidence_fingerprints": list(record.evidence_fingerprints),
        "orbit_identity": _plain_domain_json(orbit_identity(adapter)),
        "orbit_diagnostics": [
            _plain_domain_json(item) for item in adapter.diagnostics.values()
        ],
        "assessments": [
            serialize_map_assessment_checkpoint(item) for item in assessments
        ],
    }


def run_callback_summary_preflight(context: RunContext) -> dict[str, Any]:
    """Exercise actual callback serialization for every assessment shape."""

    from dataclasses import replace
    from tests.coupling import test_v4_cft_contract as manufactured

    maps = manufactured.map_set()

    class ResolvedSummaryAdapter(manufactured.OrbitAdapter):
        diagnostics: dict[str, Any] = {}

    resolved_adapter = ResolvedSummaryAdapter()
    resolved = manufactured.build(
        evidence=maps,
        orbit_adapter=resolved_adapter,
    )
    original = manufactured.REGISTRATIONS[0]
    extra_seed = replace(
        original.seeds[0],
        seed_id="manufactured-ambiguous-extra-seed",
    )
    extra_registration = replace(
        original,
        cell_id="manufactured-ambiguous-extra-cell",
        seeds=(extra_seed,),
    )
    ambiguous_registrations = (
        original,
        extra_registration,
    )
    ambiguous_adapter = MapGuidingCenterOrbitAdapter(
        maps, ambiguous_registrations
    )
    ambiguous = manufactured.build(
        evidence=maps,
        registrations=ambiguous_registrations,
        orbit_adapter=ambiguous_adapter,
    )
    if ambiguous.status is not V4Status.AMBIGUOUS or any(
        assessment.cells
        for assessment in (
            ambiguous.stability.primary,
            ambiguous.stability.refined,
            ambiguous.stability.enlarged,
        )
    ):
        raise ValueError("manufactured ambiguous zero-cell assessment changed")
    boundary = [
        serialize_boundary_null_diagnostic(
            BoundaryNullDiagnostic(-2.5, "z_min", 0.0)
        ),
        serialize_boundary_null_diagnostic(
            BoundaryNullDiagnostic(2.5, "z_max", 0.0)
        ),
    ]
    qualities = {
        role: {"all_gates_passed": True}
        for role in ("primary", "refined", "enlarged")
    }
    replay = {
        "field_equality_passed": True,
        "residual_reproducibility_passed": True,
        "passed": True,
    }
    topology = {
        role: {
            "gating_role": "diagnostic_only",
            "interior_null_count": 0,
            "finite_boundary_null_count": len(boundary),
            "classification_counts": {"X": 0, "O": 0, "degenerate": 0},
            "interior_nulls": [],
            "finite_boundary_nulls": boundary,
        }
        for role in ("primary", "refined", "enlarged")
    }
    rejection = _record_summary(
        _manufactured_development_case(),
        ambiguous,
        qualities,
        replay,
        topology,
    )
    rejection.update(
        {
            "path_length_convergence_passed": True,
            "source_consistency_passed": True,
            "complete_three_map_fingerprints": True,
            "numerical_gates_passed": False,
            "opaque_projection_passed": False,
            "passed": False,
        }
    )
    payload = {
        "resolved": _prerecord_checkpoint(resolved, resolved_adapter),
        "ambiguous_zero_cell_orbit": _prerecord_checkpoint(
            ambiguous, ambiguous_adapter
        ),
        "boundary_nonempty": topology,
        "assessment_rejection": rejection,
        "decision": Decision(
            False,
            {
                "promotion": False,
                "cases": [rejection],
                "reason": "manufactured-assessment-rejection",
            },
        ),
    }
    _assert_plain_domain_json(payload)
    path = "preflight/callback-summary-matrix.json"
    _write_callback_json(context, path, payload)
    stored = strict_json_loads(context.store.read_bytes(path))
    _assert_plain_domain_json(stored)
    if "__cft_type__" in json.dumps(stored, sort_keys=True):
        raise ValueError("callback preflight wrote a reserved envelope")
    return {
        "status": "passed",
        "held_out_case_access_count": 0,
        "resolved_status": resolved.status.value,
        "ambiguous_status": ambiguous.status.value,
        "ambiguous_cell_count": 0,
        "ambiguous_orbit_count": 0,
        "boundary_diagnostic_count": len(boundary),
        "assessment_rejection_passed": not rejection["passed"],
        "reserved_envelope_count": 0,
        "artifact_byte_sha256": hashlib.sha256(
            context.store.read_bytes(path)
        ).hexdigest(),
    }


@dataclass
class ValidationCallbacks:
    """Protocol-specific work behind the shared experiment runtime."""

    attestation: ExecutionAttestation
    closure: dict[str, Any] | None = None
    runtime_identity: dict[str, Any] | None = None

    def prebundle(self, context: RunContext) -> Mapping[str, Any]:
        self.closure = dependency_closure()
        if self.closure["preregistration_commit_sha"] != self.attestation.commit:
            raise RuntimeError("dependency closure and execution attestation differ")
        self.runtime_identity = _runtime_identity()
        static_preflight = run_production_path_static_preflight()
        preflight = run_serialization_preflight()
        orbit_preflight = run_orbit_manufactured_preflight()
        definitions = case_definitions()
        excluded = PROTOCOL["held_out_family"]["excluded_accessed_evidence"]
        prior_ids = set().union(
            excluded["v1_accessed_case_ids"],
            excluded["v2_accessed_case_ids"],
            excluded["v3_accessed_case_ids"],
            excluded["v4_accessed_case_ids"],
            excluded["v5_accessed_case_ids"],
            excluded["v6_accessed_case_ids"],
        )
        prior_coordinates = {
            tuple(item)
            for key in (
                "v1_accessed_coordinate_tuples",
                "v2_accessed_coordinate_tuples",
                "v3_accessed_coordinate_tuples",
                "v4_accessed_coordinate_tuples",
                "v5_accessed_coordinate_tuples",
                "v6_accessed_coordinate_tuples",
            )
            for item in excluded[key]
        }
        current_coordinates = {
            (
                item.stage_count,
                item.pitch_m,
                item.chamber_radius_m,
                item.first_polarity,
            )
            for item in definitions
        }
        disjointness = {
            "held_out_case_count": len(definitions),
            "prior_accessed_case_ids": sorted(prior_ids),
            "case_id_intersection": sorted(
                prior_ids & {item.case_id for item in definitions}
            ),
            "coordinate_intersection": sorted(
                current_coordinates & prior_coordinates
            ),
            "development_case_intersection": sorted(
                set(held_out_manifest().case_ids)
                & set(CFT_V4_DEVELOPMENT_MANIFEST.case_ids)
            ),
            "development_family_intersection": sorted(
                set(held_out_manifest().geometry_family_ids)
                & set(CFT_V4_DEVELOPMENT_MANIFEST.geometry_family_ids)
            ),
            "prior_validation_failures_preserved": {
                "v1": "preserved",
                "v2": "preserved",
                "v3": "preserved",
                "v4": "preserved",
                "v5": "preserved",
                "v6": "preserved",
            },
            "held_out_map_access_count": 0,
        }
        if any(
            disjointness[key]
            for key in (
                "case_id_intersection",
                "coordinate_intersection",
                "development_case_intersection",
                "development_family_intersection",
            )
        ):
            raise RuntimeError("held-out family is not disjoint")
        production_preflight = run_production_field_pipeline_preflight(
            context,
            self.closure,
            self.runtime_identity,
        )
        callback_preflight = run_callback_summary_preflight(context)
        _write_callback_json(context, "protocol/preregistered.json", PROTOCOL)
        _write_callback_json(
            context, "provenance/dependency-closure.json", self.closure
        )
        _write_callback_json(
            context, "provenance/runtime-identity.json", self.runtime_identity
        )
        _write_callback_json(context, "preflight/serialization.json", preflight)
        _write_callback_json(
            context, "preflight/orbit-manufactured.json", orbit_preflight
        )
        _write_callback_json(
            context, "preflight/production-path-static.json", static_preflight
        )
        _write_callback_json(
            context,
            "preflight/production-field-pipeline.json",
            production_preflight,
        )
        _write_callback_json(
            context,
            "preflight/callback-summary-report.json",
            callback_preflight,
        )
        _write_callback_json(context, "preflight/disjointness.json", disjointness)
        return {
            "protocol_semantic_sha256": PROTOCOL_SEMANTIC_SHA256,
            "dependency_closure_semantic_sha256": self.closure[
                "closure_semantic_sha256"
            ],
            "serialization_preflight_sha256": semantic_hash(preflight),
            "orbit_manufactured_preflight_sha256": semantic_hash(
                orbit_preflight
            ),
            "production_path_static_sha256": semantic_hash(static_preflight),
            "production_field_pipeline_sha256": semantic_hash(
                production_preflight
            ),
            "callback_summary_preflight_sha256": semantic_hash(
                callback_preflight
            ),
            "held_out_map_access_count": 0,
        }

    def development(self, context: RunContext) -> Decision:
        freeze = {
            "schema_version": "cft-wall-cusp-v7.threshold-freeze/1.0.0",
            "frozen_before_preregistration": True,
            "threshold_source": "manufactured and 56-case development evidence only",
            "prior_held_out_outcomes_used_for_tuning": False,
            "v6_disclosed_as_development_evidence_not_tuning": PROTOCOL[
                "disclosed_v6_evidence"
            ],
            "criterion": PROTOCOL["criterion"],
            "map_gates": PROTOCOL["maps"]["field_gates"],
            "gpu_replay": PROTOCOL["gpu_replay"],
            "orbit_verification": PROTOCOL["orbit_verification"],
            "development_manifest_hash": CFT_V4_DEVELOPMENT_MANIFEST.manifest_hash,
            "protocol_semantic_sha256": PROTOCOL_SEMANTIC_SHA256,
        }
        accepted = (
            freeze["development_manifest_hash"]
            == PROTOCOL["development_evidence"]["manifest_hash"]
            and not freeze["prior_held_out_outcomes_used_for_tuning"]
        )
        _write_callback_json(context, "development/threshold-freeze.json", freeze)
        return Decision(
            accepted,
            {
                "threshold_freeze_sha256": semantic_hash(freeze),
                "held_out_map_access_count": 0,
            },
        )

    def assessment(self, context: RunContext) -> Decision:
        if self.closure is None or self.runtime_identity is None:
            raise RuntimeError("prebundle state is unavailable")
        closure = self.closure
        runtime = self.runtime_identity
        code_hash = normalized_text_hash(Path(__file__).read_text(encoding="utf-8"))
        cases: list[dict[str, Any]] = []
        solved_by_case: dict[str, dict[str, SolvedMap]] = {}
        built_by_case: dict[str, BuiltCase] = {}

        for definition in case_definitions():
            context.before_expensive(
                definition.case_id,
                kind="backend",
                details={
                    "phase": "held-out-case",
                    "geometry_family_id": definition.geometry_family_id,
                },
            )
            built = build_case(definition)
            built_by_case[definition.case_id] = built
            _write_callback_json(context,
                f"geometries/{definition.case_id}.json",
                built.geometry.to_dict(),
            )
            solved: dict[str, SolvedMap] = {}
            for role in ("primary", "refined", "enlarged"):
                context.before_expensive(
                    f"{definition.case_id}-{role}",
                    kind="solver",
                    details={
                        "case_id": definition.case_id,
                        "map_role": role,
                        "device": PROTOCOL["maps"]["solver"]["device"],
                    },
                )
                item = solve_map(built, role, closure, runtime)
                artifact_path = f"fields/{definition.case_id}/{role}-field.json"
                context.write_blob(artifact_path, item.artifact_bytes)
                stored_bytes = context.store.read_bytes(artifact_path)
                if stored_bytes != item.artifact_bytes:
                    raise RuntimeError("atomic field artifact bytes changed")
                item.evidence = load_map_evidence(
                    built,
                    role,
                    stored_bytes,
                    closure,
                    runtime,
                )
                solved[role] = item
                _write_callback_json(context,
                    f"map-records/{definition.case_id}/{role}.json",
                    {
                        "artifact_byte_sha256": hashlib.sha256(
                            stored_bytes
                        ).hexdigest(),
                        "artifact_payload_sha256": item.artifact_payload_sha256,
                        "field_quality": item.quality,
                        "schema_version": item.artifact["schema_version"],
                    },
                )
            solved_by_case[definition.case_id] = solved
            map_set = verify_v4_map_set(
                solved["primary"].evidence,
                solved["refined"].evidence,
                solved["enlarged"].evidence,
                reference_time_utc=runtime["generated_at_utc"],
            )
            snapshots = reverify_v4_map_set(map_set)
            source_consistency = (
                len({item.claims.source_hash for item in snapshots}) == 1
                and len({item.claims.geometry_hash for item in snapshots}) == 1
                and len({item.claims.material_hash for item in snapshots}) == 1
            )
            registrations = registrations_for(built)
            adapter = MapGuidingCenterOrbitAdapter(map_set, registrations)
            prerecord = build_cft_coupling_record(
                map_set,
                geometry=CFTGeometry(
                    definition.chamber_radius_m,
                    0.0,
                    built.chamber_length_m,
                    float(
                        PROTOCOL["criterion"]["axial_core"][
                            "core_radius_wall_fraction"
                        ]
                    )
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
            _write_callback_json(context,
                f"prerecords/{definition.case_id}.json",
                {
                    "record": cft_coupling_record_dict(prerecord),
                    "summary": _prerecord_checkpoint(prerecord, adapter),
                },
            )
            context.before_expensive(
                f"{definition.case_id}-primary-replay",
                kind="solver",
                details={"case_id": definition.case_id, "map_role": "primary-replay"},
            )
            replay = replay_map(solved["primary"])
            _write_callback_json(context, f"replay/{definition.case_id}.json", replay)
            topology = {
                role: topology_diagnostics(item.evidence)
                for role, item in solved.items()
            }
            _write_callback_json(context,
                f"topology/{definition.case_id}.json",
                topology,
            )
            qualities = {role: item.quality for role, item in solved.items()}
            row = _record_summary(
                built,
                prerecord,
                qualities,
                replay,
                topology,
            )
            orbit_diagnostics = tuple(adapter.diagnostics.values())
            orbit_gates = _orbit_gate_summary(prerecord, adapter)
            if not (
                orbit_gates["all_numerically_converged"]
                and orbit_gates["cross_map_converged"]
            ):
                row["failures"].append("ORBIT_UNVERIFIED")
            elif not orbit_gates["all_physically_adiabatic"]:
                row["failures"].append("ORBIT_NONADIABATIC")
            if row["axial_core_failure_count"]:
                row["failures"].append("AXIAL_CORE_FAILURE")
            path_length_passed = all(
                item["path_length_gate_passed"] for item in orbit_diagnostics
            )
            if not path_length_passed:
                row["failures"].append("PATH_LENGTH_CONVERGENCE_FAILURE")
            if not source_consistency:
                row["failures"].append("SOURCE_CONSISTENCY_FAILURE")
            fingerprints = tuple(prerecord.evidence_fingerprints)
            fingerprints_complete = (
                len(fingerprints) == 3
                and len(set(fingerprints)) == 3
                and all(len(item) == 64 for item in fingerprints)
            )
            if not fingerprints_complete:
                row["failures"].append("THREE_MAP_FINGERPRINT_FAILURE")
            row.update(
                {
                    "numerical_gates_passed": bool(
                        row["passed"]
                        and orbit_gates["all_numerically_converged"]
                        and orbit_gates["all_physically_adiabatic"]
                        and orbit_gates["cross_map_converged"]
                        and path_length_passed
                        and source_consistency
                        and fingerprints_complete
                    ),
                    "path_length_convergence_passed": path_length_passed,
                    "source_consistency_passed": source_consistency,
                    "three_map_fingerprints_complete": fingerprints_complete,
                    "orbit_diagnostics": orbit_diagnostics,
                    "orbit_verification": orbit_gates,
                    "opaque_projection_passed": False,
                    "opaque_projection_row_count": 0,
                }
            )
            row["passed"] = row["numerical_gates_passed"]
            row["assessment_complete"] = True
            if not row["numerical_gates_passed"]:
                row["failures"] = sorted(set(row["failures"]))
                _write_callback_json(
                    context,
                    f"outcomes/{definition.case_id}.json",
                    row,
                )
            cases.append(row)

        all_numerical = all(case["numerical_gates_passed"] for case in cases)
        outcomes = tuple(
            HeldOutCaseOutcome(
                str(case["case_id"]),
                str(case["geometry_family_id"]),
                tuple(case["map_hashes"]),
                tuple(case["map_evidence_fingerprints"]),
                True,
            )
            for case in cases
        ) if all_numerical else ()
        projection_rows: list[dict[str, Any]] = []
        if all_numerical:
            aggregate_diagnostics = _aggregate_diagnostics(cases)
            for case in cases:
                definition = next(
                    item
                    for item in case_definitions()
                    if item.case_id == case["case_id"]
                )
                built = built_by_case[definition.case_id]
                context.before_expensive(
                    f"{definition.case_id}-opaque-projection",
                    kind="backend",
                    details={"case_id": definition.case_id},
                )
                evidence = {
                    role: load_map_evidence(
                        built,
                        role,
                        context.store.read_bytes(
                            f"fields/{definition.case_id}/{role}-field.json"
                        ),
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
                registrations = registrations_for(built)
                geometry = CFTGeometry(
                    definition.chamber_radius_m,
                    0.0,
                    built.chamber_length_m,
                    float(
                        PROTOCOL["criterion"]["axial_core"][
                            "core_radius_wall_fraction"
                        ]
                    )
                    * definition.chamber_radius_m,
                    definition.geometry_id,
                )
                adapter = MapGuidingCenterOrbitAdapter(map_set, registrations)
                policies = policies_for(built)
                fingerprints = v4_map_set_evidence_fingerprints(
                    map_set,
                    reference_time_utc=runtime["generated_at_utc"],
                )
                preregistration_hash = cft_preregistration_hash(
                    geometry=geometry,
                    registrations=registrations,
                    validation_registration=validation_registration(
                        built, code_hash
                    ),
                    three_map_hashes=tuple(case["map_hashes"]),
                    three_map_evidence_fingerprints=fingerprints,
                    orbit_identity=orbit_identity(adapter),
                    criterion=V4Criterion(),
                    **policies,
                )
                validation_payload = {
                    "schema_version": (
                        "cft-revival.cft-wall-cusp-validation-evidence/1.0.0"
                    ),
                    "criterion_id": "cft-hemp-wall-cusp-v4",
                    "criterion_version": "4.0.0",
                    "development_manifest_hash": (
                        CFT_V4_DEVELOPMENT_MANIFEST.manifest_hash
                    ),
                    "held_out_manifest_hash": held_out_manifest().manifest_hash,
                    "evaluated_case_id": definition.case_id,
                    "evaluated_geometry_family_id": definition.geometry_family_id,
                    "preregistration_hash": preregistration_hash,
                    "outcomes": [asdict(item) for item in outcomes],
                    "validation_code_hash": code_hash,
                    "validation_config_hash": PROTOCOL_SEMANTIC_SHA256,
                }
                artifact_bytes = json.dumps(
                    validation_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
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
                    policy=validation_registration(built, code_hash).policy,
                )
                record = build_cft_coupling_record(
                    map_set,
                    geometry=geometry,
                    registrations=registrations,
                    validation_registration=validation_registration(
                        built, code_hash
                    ),
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
                context.write_blob(
                    f"validation-evidence/{definition.case_id}.json",
                    artifact_bytes,
                )
                _write_callback_json(context,
                    f"records/{definition.case_id}-v4.2.json",
                    cft_coupling_record_dict(record),
                )
                _write_callback_json(context,
                    f"projections/{definition.case_id}.json",
                    {"rows": projected},
                )
                projection_rows.extend(
                    {"case_id": definition.case_id, **_json_value(item)}
                    for item in projected
                )
                case["opaque_projection_row_count"] = len(projected)
                case["opaque_projection_passed"] = bool(projected)
                case["held_out_preregistration_hash"] = preregistration_hash
                case["held_out_validation_artifact_hash"] = hashlib.sha256(
                    artifact_bytes
                ).hexdigest()

        for case in cases:
            case["passed"] = bool(
                all_numerical and case["opaque_projection_passed"]
            )
            if all_numerical and not case["opaque_projection_passed"]:
                case["failures"].append("OPAQUE_PROJECTION_REJECTED")
            case["failures"] = sorted(set(case["failures"]))
            if case["numerical_gates_passed"]:
                _write_callback_json(
                    context,
                    f"outcomes/{case['case_id']}.json",
                    case,
                )

        promotion = bool(
            all_numerical
            and cases
            and all(case["opaque_projection_passed"] for case in cases)
            and len(projection_rows) > 0
        )

        def aggregate_orbit_range(name: str) -> list[float | None]:
            values = [
                float(value)
                for case in cases
                for value in case["orbit_verification"][name]
                if value is not None
            ]
            return [min(values), max(values)] if values else [None, None]

        summary = {
            "case_count": len(cases),
            "map_count": 3 * len(cases),
            "three_map_field_gate_pass_count": sum(
                all(
                    item["all_gates_passed"]
                    for item in case["field_quality"].values()
                )
                for case in cases
            ),
            "three_map_fingerprint_pass_count": sum(
                case["three_map_fingerprints_complete"] for case in cases
            ),
            "source_consistency_pass_count": sum(
                case["source_consistency_passed"] for case in cases
            ),
            "v4_stability_pass_count": sum(
                case["numerical_gates_passed"] for case in cases
            ),
            "gpu_replay_pass_count": sum(
                case["gpu_replay"]["passed"] for case in cases
            ),
            "path_length_convergence_pass_count": sum(
                case["path_length_convergence_passed"] for case in cases
            ),
            "stable_cusp_count": sum(
                case["cusp_counts"][0]
                for case in cases
                if case["numerical_gates_passed"]
            ),
            "stable_cell_count": sum(
                case["cell_counts"][0]
                for case in cases
                if case["numerical_gates_passed"]
            ),
            "primary_detected_cusp_count": sum(
                case["cusp_counts"][0] for case in cases
            ),
            "primary_detected_cell_count": sum(
                case["cell_counts"][0] for case in cases
            ),
            "inter_cusp_cell_count": sum(
                case["inter_cusp_cell_count"] for case in cases
            ),
            "wall_connected_path_count": sum(
                case["wall_connected_path_count"] for case in cases
            ),
            "required_path_count": sum(
                case["required_path_count"] for case in cases
            ),
            "same_line_extrema_count": sum(
                case["same_line_extrema_count"] for case in cases
            ),
            "resolved_orbit_count": sum(
                case["resolved_orbit_count"] for case in cases
            ),
            "required_orbit_count": sum(
                case["required_orbit_count"] for case in cases
            ),
            "adapter_evaluated_orbit_count": sum(
                case["orbit_verification"]["adapter_evaluated_sample_count"]
                for case in cases
            ),
            "numerically_converged_orbit_count": sum(
                case["orbit_verification"]["numerically_converged_sample_count"]
                for case in cases
            ),
            "physically_adiabatic_orbit_count": sum(
                case["orbit_verification"]["physically_adiabatic_sample_count"]
                for case in cases
            ),
            "uncertainty_not_evaluable_case_count": sum(
                not case["uncertainty_evaluable"] for case in cases
            ),
            "axial_core_failure_count": sum(
                case["axial_core_failure_count"] for case in cases
            ),
            "axial_core_evaluation_count": sum(
                case["axial_core_evaluation_count"] for case in cases
            ),
            "gyro_averaged_mu_relative_variation_range": aggregate_orbit_range(
                "gyro_averaged_mu_relative_variation_range"
            ),
            "instantaneous_mu_relative_variation_range": aggregate_orbit_range(
                "instantaneous_mu_relative_variation_range"
            ),
            "energy_relative_drift_range": aggregate_orbit_range(
                "energy_relative_drift_range"
            ),
            "rho_over_lb_range": aggregate_orbit_range("rho_over_lb_range"),
            "rho_times_curvature_range": aggregate_orbit_range(
                "rho_times_curvature_range"
            ),
            "opaque_projection_pass_count": sum(
                case["opaque_projection_passed"] for case in cases
            ),
            "opaque_projection_row_count": len(projection_rows),
            "failure_counts": _failure_counts(cases),
            "criterion_numerically_promoted": promotion,
            "search_v3_ready": promotion,
            "plasma_coupling_ready": promotion,
            "hardware_validation_claim": False,
        }
        dataset = {
            "schema_version": SCHEMA_VERSION,
            "classification": PROTOCOL["classification"],
            "claim_boundary": PROTOCOL["publication_boundary"],
            "protocol_semantic_sha256": PROTOCOL_SEMANTIC_SHA256,
            "preregistration_commit_sha": closure["preregistration_commit_sha"],
            "foundation_commit_sha": FOUNDATION_COMMIT,
            "development_manifest": CFT_V4_DEVELOPMENT_MANIFEST,
            "held_out_manifest": held_out_manifest(),
            "dependency_closure": closure,
            "runtime_identity": runtime,
            "summary": summary,
            "cases": cases,
            "projection_rows": projection_rows,
        }
        _write_callback_json(context, "dataset.json", dataset)
        context.write_blob(
            "report.md",
            (_report(_plain_domain_json(dataset)) + "\n").encode("utf-8"),
        )
        return Decision(
            promotion,
            {
                "summary": summary,
                "claim": (
                    "numerically stable and source-consistent; "
                    "not hardware validated"
                    if promotion
                    else "not promoted; one or more preregistered gates failed"
                ),
            },
        )


def execute_validation(
    *,
    result_root: Path,
    cache_root: Path,
    attestation: ExecutionAttestation,
) -> Any:
    """Execute the sole attempt through the shared lifecycle runtime."""

    callbacks = ValidationCallbacks(attestation)
    runtime = ExperimentRuntime(
        experiment_id=PROTOCOL["experiment_id"],
        result_root=result_root,
        cache_root=cache_root,
        attestation=attestation,
        producer=execute_validation,
        source_root=MODERN_ROOT,
        root_policy=RootPolicy(
            approved_placeholders={".gitattributes": RESULT_ATTRIBUTES},
        ),
    )
    outcome = runtime.run(
        RuntimeCallbacks(
            callbacks.prebundle,
            callbacks.development,
            callbacks.assessment,
        )
    )
    return outcome


def validate_results(result_root: Path | None = None) -> Mapping[str, Any]:
    return validate_bundle(
        result_root or EXPERIMENT_DIR / "results",
        approved_placeholders={".gitattributes": RESULT_ATTRIBUTES},
    )
