"""Single-run preregistered four-cell L1a/coupling-v3/plasma-network search.

The protocol is data, not an adjustable argument.  This module deliberately has
no v2 same-z coupling import and publishes no plasma state, power, or performance
object.  A candidate can reach plasma_network only through an accepted v3 record
whose four preregistered cells pass every connected-surface quantile atomically.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.coupling import (
    AdapterVersionContract,
    CellRegistration,
    ElectronAdiabaticInputs,
    FluxSurfacePolicy,
    MapValidationPolicy,
    SolverDiagnosticsEvidence,
    SurfaceStatus,
    TopologyStatus,
    UncertaintyModel,
    V3ArtifactClaims,
    build_coupling_record,
    coupling_record_dict,
    global_solver_inputs,
    magnetic_null_geometry,
    reverify_v3_evidence,
    v3_evidence_binding_hash,
    verify_v3_field_artifact,
    verify_v3_topology_stability,
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
from cft_revival.plasma import SolverOptions
from cft_revival.plasma_network import (
    GeometryCell,
    GeometryNull,
    GeometryTopologySnapshot,
    NetworkInputs,
    NetworkSolverOptions,
    NullClassification,
    PublicationPolicy,
    SemanticHashes,
    SnapshotAdapter,
    TerminalBoundary,
    TerminalKind,
    UncertainScalar,
    build_chain_topology,
    provenance_hash,
    solve_network_multistart,
)

SCHEMA_VERSION = "cft-revival.four-cell-topology-search-v2.dataset/1.0.0"
MANIFEST_VERSION = "cft-revival.four-cell-topology-search-v2.manifest/1.0.0"
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
ACCEPTANCE_TIME_UTC = datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc)
EXPERIMENT_DIR = Path(__file__).resolve().parent
MODERN_ROOT = EXPERIMENT_DIR.parents[1]
REPOSITORY_ROOT = MODERN_ROOT.parent
PROTOCOL_PATH = EXPERIMENT_DIR / "protocol.json"
ACCEPTED_COUPLING_COMMIT = "f80a360fd740a30017cdac1874cedbfa2806874a"


def _strict_json(path: Path) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value}")

    loaded = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=reject,
    )
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return loaded


PROTOCOL = _strict_json(PROTOCOL_PATH)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if is_dataclass(value):
        return {name: _json_value(item) for name, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(name): _json_value(item) for name, item in value.items()}
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


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


PROTOCOL_SHA256 = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
PROTOCOL_PAYLOAD_SHA256 = stable_hash(PROTOCOL)


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = _json_value(dict(payload))
    return {
        **body,
        "integrity": {
            "algorithm": "sha256",
            "canonicalization": CANONICALIZATION,
            "payload_sha256": stable_hash(body),
        },
    }


def _write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def write_sealed_json(
    path: Path, payload: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    sealed = _seal(payload)
    data = (
        json.dumps(sealed, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return sealed, _write_bytes(path, data)


def _verify_sidecar(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = f"{digest}  {path.name}\n"
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii") != expected:
        raise ValueError(f"invalid SHA-256 sidecar for {path}")
    return digest


def load_sealed_json(path: Path) -> dict[str, Any]:
    _verify_sidecar(path)
    value = _strict_json(path)
    integrity = value.get("integrity")
    body = {name: item for name, item in value.items() if name != "integrity"}
    if (
        not isinstance(integrity, dict)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("canonicalization") != CANONICALIZATION
        or integrity.get("payload_sha256") != stable_hash(body)
    ):
        raise ValueError(f"{path} payload integrity mismatch")
    return value


def _git(*arguments: str, cwd: Path = REPOSITORY_ROOT) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def dependency_closure() -> dict[str, Any]:
    """Return a conservative, blob-exact transitive execution closure."""

    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("single execution requires a clean detached worktree")
    symbolic = subprocess.run(
        ("git", "symbolic-ref", "-q", "--short", "HEAD"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if symbolic.returncode == 0 and symbolic.stdout.strip():
        raise RuntimeError("single execution requires detached HEAD")
    prefixes = (
        "modern/experiments/four_cell_topology_search_v2/",
        "modern/src/cft_revival/coupling/",
        "modern/src/cft_revival/fields/",
        "modern/src/cft_revival/geometry/",
        "modern/src/cft_revival/magnetics/",
        "modern/src/cft_revival/optimization/",
        "modern/src/cft_revival/plasma/",
        "modern/src/cft_revival/plasma_network/",
        "modern/spec/",
    )
    tracked = tuple(
        line
        for line in _git("ls-files").splitlines()
        if line == "modern/pyproject.toml" or line.startswith(prefixes)
    )
    if not tracked or str(PROTOCOL["accepted_dependency_baseline"]["coupling_v3_commit"]) != ACCEPTED_COUPLING_COMMIT:
        raise RuntimeError("accepted dependency declaration is incomplete")
    rows: list[dict[str, str]] = []
    baseline_mismatches: list[str] = []
    for relative in tracked:
        path = REPOSITORY_ROOT / relative
        blob = _git("rev-parse", f"{head}:{relative}")
        rows.append(
            {
                "path": relative.replace("\\", "/"),
                "git_blob_sha1": blob,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        if relative.startswith("modern/src/cft_revival/") or relative.startswith(
            "modern/spec/"
        ):
            try:
                baseline_blob = _git(
                    "rev-parse", f"{ACCEPTED_COUPLING_COMMIT}:{relative}"
                )
            except subprocess.CalledProcessError:
                baseline_mismatches.append(relative)
            else:
                if baseline_blob != blob:
                    baseline_mismatches.append(relative)
    if baseline_mismatches:
        raise RuntimeError(
            "accepted dependency blobs differ from coupling-v3 baseline: "
            + ", ".join(baseline_mismatches)
        )
    return {
        "preregistration_commit_sha": head,
        "accepted_coupling_v3_commit_sha": ACCEPTED_COUPLING_COMMIT,
        "accepted_dependency_blobs_match": True,
        "path_policy": list(prefixes) + ["modern/pyproject.toml"],
        "files": rows,
        "closure_sha256": stable_hash(rows),
    }


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    while index:
        index, digit = divmod(index, base)
        denominator *= base
        value += digit / denominator
    return value


def _round_value(name: str, value: float) -> float:
    quantum = (
        float(PROTOCOL["sampling"]["manufacturing_rounding_m"])
        if name.endswith("_m")
        else float(PROTOCOL["sampling"]["manufacturing_rounding_ratio"])
    )
    return round(value / quantum) * quantum


def sample_candidates() -> tuple[dict[str, Any], ...]:
    sampling = PROTOCOL["sampling"]
    ranges = sampling["variables"]
    names = tuple(ranges)
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    if len(names) != len(bases):
        raise ValueError("protocol variable/base count mismatch")
    candidates: list[dict[str, Any]] = []
    for candidate_index in range(int(sampling["candidate_count"])):
        sequence_index = int(sampling["sequence_offset"]) + candidate_index
        values: dict[str, float] = {}
        for dimension, (name, base) in enumerate(zip(names, bases, strict=True)):
            low, high = (float(item) for item in ranges[name])
            shift = int(
                hashlib.sha256(f"four-cell-v2:{dimension}".encode()).hexdigest()[:13],
                16,
            ) / float(16**13)
            unit = (_radical_inverse(sequence_index, base) + shift) % 1.0
            values[name] = _round_value(name, low + unit * (high - low))
        identity = stable_hash(
            {
                "algorithm": sampling["algorithm"],
                "sequence_index": sequence_index,
                "values": values,
            }
        )
        candidates.append(
            {
                "candidate_id": f"v2-{candidate_index:03d}",
                "sequence_index": sequence_index,
                "sampling_identity_sha256": identity,
                "values": values,
            }
        )
    if len({item["sampling_identity_sha256"] for item in candidates}) != len(
        candidates
    ):
        raise ValueError("candidate sampling identities are not unique")
    return tuple(candidates)


def _stable_pitch_and_centres(
    pitch: float, first: float, count: int
) -> tuple[float, tuple[float, ...]]:
    target = 0.005
    for _ in range(128):
        centres = tuple(first + index * pitch for index in range(count))
        if all(
            abs((right - left) - pitch)
            <= 2.0 * max(math.ulp(right - left), math.ulp(pitch))
            for left, right in zip(centres, centres[1:])
        ):
            return pitch, centres
        pitch = math.nextafter(pitch, target)
    raise GeometryValidationError("could not represent a contract-stable pitch")


def _stable_grid_upper(lower: float, requested: float, intervals: int) -> float:
    upper = requested
    for _ in range(512):
        spacing = (upper - lower) / intervals
        if lower + intervals * spacing == upper:
            return upper
        upper = math.nextafter(upper, lower)
    raise GeometryValidationError("could not represent a contract-stable domain")


def _materials(geometry: Any) -> dict[str, Any]:
    registry: dict[str, Any] = {}
    for material in geometry.materials:
        if material.category is MaterialKind.PERMANENT_MAGNET:
            resolved = checked_synthetic_smco_like_magnet()
            if resolved.material_id != material.material_id:
                raise GeometryValidationError("permanent-magnet registry ID mismatch")
        else:
            resolved = LinearPermeability(
                material.material_id, material.relative_permeability
            )
        registry[material.material_id] = resolved
    return registry


@dataclass(frozen=True)
class BuiltCandidate:
    candidate_id: str
    declaration: Mapping[str, Any]
    geometry: Any
    sources: tuple[Any, ...]
    geometry_sha256: str
    material_sha256: str
    source_sha256: str
    chamber_radius_m: float
    wall_radius_m: float
    pitch_m: float
    stage_centres_m: tuple[float, ...]
    cusp_targets_m: tuple[float, ...]
    base_radius_m: float
    base_z_min_m: float
    base_z_max_m: float


def build_candidate(declaration: Mapping[str, Any]) -> BuiltCandidate:
    values = declaration["values"]
    candidate_id = str(declaration["candidate_id"])
    pitch = float(values["axial_pitch_m"])
    first = float(values["stack_start_m"]) + float(
        values["stack_offset_fraction"]
    ) * pitch
    pitch, centres = _stable_pitch_and_centres(pitch, first, 4)
    chamber_radius = float(values["chamber_outer_radius_m"])
    dielectric = float(values["dielectric_thickness_m"])
    magnet_inner = chamber_radius + dielectric + float(
        values["radial_clearance_m"]
    )
    magnet_outer = magnet_inner + float(values["magnet_radial_thickness_m"])
    magnet_thickness = pitch * float(values["magnet_axial_fraction"])
    chamber_length = centres[-1] + 1.25 * pitch
    geometry = generate_twt_inspired_ppm_stack(
        PPMStackParameters(
            config_id=f"{candidate_id}-geometry",
            title=f"Preregistered four-cell search v2 {candidate_id}",
            chamber_inner_radius_m=0.0,
            chamber_outer_radius_m=chamber_radius,
            chamber_length_m=chamber_length,
            injector_length_m=min(0.08 * chamber_length, 0.5 * centres[0]),
            dielectric_thickness_m=dielectric,
            thermal_clearance_m=2.5e-4,
            magnet_inner_radius_m=magnet_inner,
            magnet_outer_radius_m=magnet_outer,
            stage_pitch_m=pitch,
            stage_centers_m=centres,
            magnet_axial_thicknesses_m=(magnet_thickness,) * 4,
            shield_outer_radius_m=magnet_outer + 7.5e-4,
            yoke_outer_radius_m=magnet_outer + 1.75e-3,
            first_polarity=1,
            radial_tolerance_m=2.5e-5,
            axial_tolerance_m=2.5e-5,
            minimum_thickness_m=2.5e-4,
            minimum_clearance_m=1.0e-4,
        ),
        evidence=(
            EvidenceNote(
                f"{candidate_id}-preregistered-v2",
                "assumption",
                "Deterministic topology-targeted L1a candidate fixed before execution.",
                f"protocol sha256 {PROTOCOL_SHA256}",
            ),
        ),
    )
    smear = float(PROTOCOL["sampling"]["source_policy"]["source_smear_thickness_m"])
    preview = to_l1a_current_equivalent_preview(
        geometry,
        material_registry=_materials(geometry),
        radial_smear_thickness_m=smear,
    )
    alternating = float(values["alternating_strength_ratio"])
    sources = tuple(
        replace(
            band,
            ampere_turns_a=band.ampere_turns_a
            * (alternating if (band_index // 2) % 2 else 1.0),
        )
        for band_index, band in enumerate(preview.bands)
    )
    if len(sources) != 8 or any(
        sources[2 * index].ampere_turns_a != sources[2 * index + 1].ampere_turns_a
        for index in range(4)
    ):
        raise GeometryValidationError("equivalent-current stage pairing was not preserved")
    z_min = -float(values["upstream_padding_pitch"]) * pitch
    z_max = chamber_length + float(values["downstream_padding_pitch"]) * pitch
    base_radius = magnet_outer + float(values["radial_padding_m"])
    source_payload = {
        "preview": preview.to_dict(),
        "scaled_sources": [asdict(source) for source in sources],
        "pairing": "both sheets receive one stage-level multiplier",
        "alternating_strength_ratio": alternating,
    }
    targets = (centres[0] - 0.75 * pitch,) + tuple(
        0.5 * (left + right)
        for left, right in zip(centres[:-1], centres[1:], strict=True)
    )
    return BuiltCandidate(
        candidate_id,
        declaration,
        geometry,
        sources,
        geometry.canonical_sha256,
        stable_hash([material.to_dict() for material in geometry.materials]),
        stable_hash(source_payload),
        chamber_radius,
        chamber_radius - float(values["wall_sample_gap_m"]),
        pitch,
        centres,
        targets,
        base_radius,
        z_min,
        z_max,
    )


def _domain(candidate: BuiltCandidate, role: str) -> AxisymmetricDomain:
    declaration = PROTOCOL["maps"]["roles"][role]
    factor = float(declaration["domain_scale"])
    radial_intervals = int(declaration["radial_intervals"])
    axial_intervals = int(declaration["axial_intervals"])
    if factor == 1.0:
        z_min, z_max = candidate.base_z_min_m, candidate.base_z_max_m
        radius = candidate.base_radius_m
    else:
        z_mid = 0.5 * (candidate.base_z_min_m + candidate.base_z_max_m)
        half = 0.5 * (candidate.base_z_max_m - candidate.base_z_min_m) * factor
        z_min, z_max = z_mid - half, z_mid + half
        radius = candidate.base_radius_m * factor
    radius = _stable_grid_upper(0.0, radius, radial_intervals)
    z_max = _stable_grid_upper(z_min, z_max, axial_intervals)
    return AxisymmetricDomain(
        radius,
        z_min,
        z_max,
        radial_intervals,
        axial_intervals,
    )


def _solver_config() -> SolverConfig:
    policy = PROTOCOL["maps"]["solver"]
    return SolverConfig(
        relative_tolerance=float(policy["relative_tolerance"]),
        absolute_tolerance=float(policy["absolute_tolerance"]),
        max_iterations=int(policy["maximum_iterations"]),
        residual_history_stride=int(policy["residual_history_stride"]),
        max_true_residual_restarts=int(
            policy["maximum_true_residual_restarts"]
        ),
    )


def _artifact_bytes(artifact: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            artifact, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class SerializedPsiMap:
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    psi_wb: tuple[tuple[float, ...], ...]
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]


class GeneratedV3L1aAdapter:
    adapter_id = "experiments.four-cell-topology-search-v2.direct-l1a-v3"
    version_contract = AdapterVersionContract(
        "four-cell-search-v2-direct-v3",
        "1.0.0",
        "cft-axisymmetric-field-map/1.1.0",
        "cft-axisymmetric-field-map/1.1.0",
        "L1a",
    )

    def __init__(
        self,
        candidate: BuiltCandidate,
        role: str,
        problem: AxisymmetricProblem,
        artifact_hash: str,
        closure: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> None:
        self.candidate = candidate
        self.role = role
        self.problem = problem
        self.artifact_hash = artifact_hash
        self.closure = closure
        self.runtime = runtime
        self.adapter_code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def verify_v3_artifact(self, artifact_bytes: bytes) -> V3ArtifactClaims:
        if hashlib.sha256(artifact_bytes).hexdigest() != self.artifact_hash:
            raise ValueError("artifact bytes differ from accepted role bytes")
        artifact = json.loads(artifact_bytes)
        validate_field_artifact(artifact)
        expected_sources = [asdict(source) for source in self.problem.sources]
        if (
            artifact["input"]["name"] != self.problem.name
            or artifact["input"]["sources"] != expected_sources
        ):
            raise ValueError("artifact input identity differs from candidate role")
        raw = artifact["field_map"]
        field = SerializedPsiMap(
            tuple(raw["r_m"]),
            tuple(raw["z_m"]),
            tuple(tuple(row) for row in raw["psi_wb"]),
            tuple(tuple(row) for row in raw["b_r_t"]),
            tuple(tuple(row) for row in raw["b_z_t"]),
        )
        from cft_revival.coupling import hash_psi_map

        full_map_hash = hash_psi_map(field)
        source_hash = stable_hash(
            {
                "sources": expected_sources,
                "source_convention": artifact["input"]["source_convention"],
            }
        )
        domain = artifact["input"]["domain"]
        mesh_hash = stable_hash(
            {
                "radial_intervals": domain["radial_intervals"],
                "axial_intervals": domain["axial_intervals"],
                "dr_m": domain["dr_m"],
                "dz_m": domain["dz_m"],
            }
        )
        domain_hash = stable_hash(
            {
                "radius_m": domain["radius_m"],
                "z_min_m": domain["z_min_m"],
                "z_max_m": domain["z_max_m"],
                "outer_boundary": artifact["input"]["outer_boundary"],
            }
        )
        diagnostics = artifact["diagnostics"]
        solver = artifact["input"]["solver"]
        residual_tolerance = max(
            solver["absolute_tolerance"],
            solver["relative_tolerance"] * diagnostics["initial_residual_l2"],
        )
        artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
        binding = v3_evidence_binding_hash(
            full_map_hash,
            source_hash,
            self.candidate.geometry_sha256,
            self.candidate.material_sha256,
            mesh_hash,
            domain_hash,
            artifact_hash,
        )
        return V3ArtifactClaims(
            field_map=field,
            artifact_schema_version=artifact["schema_version"],
            model_level=artifact["model_level"],
            artifact_hash=artifact_hash,
            full_map_hash=full_map_hash,
            source_hash=source_hash,
            geometry_hash=self.candidate.geometry_sha256,
            material_hash=self.candidate.material_sha256,
            mesh_hash=mesh_hash,
            domain_hash=domain_hash,
            evidence_binding_hash=binding,
            backend_id=f"cft_revival.fields/{diagnostics['backend']}",
            backend_version=f"warp-{self.runtime['warp_version']}",
            field_model_id="cft.l1a.axisymmetric-equivalent-current-v1.1",
            field_model_hash=stable_hash(
                {
                    "model_description": artifact["model_description"],
                    "provenance": artifact["provenance"],
                    "accepted_dependency_closure": self.closure["closure_sha256"],
                }
            ),
            code_hash=str(self.closure["closure_sha256"]),
            config_hash=PROTOCOL_PAYLOAD_SHA256,
            generated_at_utc=ACCEPTANCE_TIME_UTC,
            diagnostics=SolverDiagnosticsEvidence(
                converged=diagnostics["converged"],
                residual_norm=diagnostics["final_residual_l2"],
                residual_tolerance=residual_tolerance,
                relative_residual=diagnostics["relative_residual_l2"],
                relative_tolerance=solver["relative_tolerance"],
                iterations=diagnostics["iterations"],
            ),
        )


def _map_validation_policy() -> MapValidationPolicy:
    validation = PROTOCOL["maps"]["validation"]
    return MapValidationPolicy(
        minimum_radial_samples=int(validation["minimum_radial_samples"]),
        minimum_axial_samples=int(validation["minimum_axial_samples"]),
        maximum_age_s=validation["maximum_age_s"],
        maximum_future_skew_s=float(validation["maximum_future_skew_s"]),
        require_axis=bool(validation["require_axis"]),
        axis_br_absolute_tolerance_t=float(
            validation["axis_br_absolute_tolerance_t"]
        ),
        axis_br_relative_tolerance=float(
            validation["axis_br_relative_tolerance"]
        ),
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
        thickness = min(
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
                / thickness,
            )
        )
    return {
        "field_peak_t": peak,
        "boundary_to_peak_ratio": boundary / max(peak, 1e-300),
        "source_discretization_relative_error": max(errors, default=0.0),
        "normalized_residual": field.diagnostics.relative_residual_l2,
        "flux_reconstruction_identity_t_per_m": (
            field.diagnostics.max_flux_reconstruction_identity_t_per_m
        ),
    }


def _field_gates(quality: Mapping[str, float]) -> dict[str, bool]:
    policy = PROTOCOL["maps"]["validation"]
    return {
        "residual": quality["normalized_residual"]
        <= float(policy["maximum_normalized_residual"]),
        "boundary": quality["boundary_to_peak_ratio"]
        <= float(policy["maximum_boundary_to_peak_ratio"]),
        "source": quality["source_discretization_relative_error"]
        <= float(policy["maximum_source_discretization_relative_error"]),
        "flux_identity": quality["flux_reconstruction_identity_t_per_m"]
        <= float(policy["maximum_flux_reconstruction_identity_t_per_m"]),
    }


@dataclass
class SolvedRole:
    role: str
    problem: AxisymmetricProblem
    field: FieldMap
    artifact_bytes: bytes
    artifact_hash: str
    evidence: Any
    quality: dict[str, float]
    gates: dict[str, bool]
    cusps_m: tuple[float, ...]
    boundary_null_count: int


def _solve_role(
    candidate: BuiltCandidate,
    role: str,
    closure: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> SolvedRole:
    problem = AxisymmetricProblem(
        f"{candidate.candidate_id}-{role}",
        _domain(candidate, role),
        candidate.sources,
    )
    field = solve_problem_warp(
        problem,
        config=_solver_config(),
        device=str(PROTOCOL["maps"]["solver"]["device"]),
    )
    artifact = field_artifact(
        problem,
        _solver_config(),
        field,
        map_stride=1,
        wall_radius_m=candidate.wall_radius_m,
    )
    validate_field_artifact(artifact)
    artifact_bytes = _artifact_bytes(artifact)
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    adapter = GeneratedV3L1aAdapter(
        candidate, role, problem, artifact_hash, closure, runtime
    )
    evidence = verify_v3_field_artifact(
        artifact_bytes,
        adapter,
        _map_validation_policy(),
        reference_time_utc=ACCEPTANCE_TIME_UTC,
    )
    accepted = reverify_v3_evidence(
        evidence, reference_time_utc=ACCEPTANCE_TIME_UTC
    ).field_map
    cusps, boundary = magnetic_null_geometry(
        accepted,
        boundary_exclusion_cells=int(
            PROTOCOL["topology"]["endpoint_exclusion_cells"]
        ),
    )
    cusp_z = tuple(point[1] for point in cusps)
    quality = _field_quality(problem, field)
    return SolvedRole(
        role,
        problem,
        field,
        artifact_bytes,
        artifact_hash,
        evidence,
        quality,
        _field_gates(quality),
        cusp_z,
        len(boundary),
    )


def topology_gates(
    candidate: BuiltCandidate, roles: Mapping[str, SolvedRole]
) -> dict[str, Any]:
    required = int(PROTOCOL["topology"]["required_stable_cell_count"])
    slot_fraction = float(
        PROTOCOL["topology"]["maximum_geometry_slot_error_pitch_fraction"]
    )
    maximum_shift = float(PROTOCOL["topology"]["maximum_cross_map_cusp_shift_m"])
    role_order = ("primary", "downsampled", "enlarged_domain")
    count_by_role = {role: len(roles[role].cusps_m) for role in role_order}
    exact_count = all(value == required for value in count_by_role.values())
    geometry_registered = exact_count and all(
        all(
            abs(observed - target) <= slot_fraction * candidate.pitch_m
            for observed, target in zip(
                roles[role].cusps_m, candidate.cusp_targets_m, strict=True
            )
        )
        for role in role_order
    )
    shifts: list[float] = []
    if exact_count:
        primary = roles["primary"].cusps_m
        shifts = [
            abs(reference - observed)
            for role in ("downsampled", "enlarged_domain")
            for reference, observed in zip(
                primary, roles[role].cusps_m, strict=True
            )
        ]
    stable_shift = exact_count and max(shifts, default=0.0) <= maximum_shift
    endpoint = True
    exclusion = int(PROTOCOL["topology"]["endpoint_exclusion_cells"])
    for role in role_order:
        solved = roles[role]
        dz = solved.problem.domain.dz_m
        endpoint = endpoint and all(
            solved.problem.domain.z_min_m + exclusion * dz
            < cusp
            < solved.problem.domain.z_max_m - exclusion * dz
            for cusp in solved.cusps_m
        )
    enlarged_ratio = roles["enlarged_domain"].quality[
        "boundary_to_peak_ratio"
    ]
    primary_ratio = roles["primary"].quality["boundary_to_peak_ratio"]
    domain_stable = enlarged_ratio <= primary_ratio * float(
        PROTOCOL["maps"]["validation"]["enlarged_boundary_ratio_growth_factor"]
    )
    field_accepted = all(
        all(solved.gates.values()) for solved in roles.values()
    )
    return {
        "count_by_role": count_by_role,
        "exact_count": exact_count,
        "geometry_registered": geometry_registered,
        "stable_shift": stable_shift,
        "maximum_observed_cusp_shift_m": max(shifts, default=None),
        "endpoint_exclusion": endpoint,
        "enlarged_domain_boundary_comparison": domain_stable,
        "all_field_gates": field_accepted,
        "stable": all(
            (
                exact_count,
                geometry_registered,
                stable_shift,
                endpoint,
                domain_stable,
                field_accepted,
            )
        ),
    }


def _surface_policy() -> FluxSurfacePolicy:
    return FluxSurfacePolicy(**PROTOCOL["coupling_v3"]["surface_policy"])


def _uncertainty() -> UncertaintyModel:
    return UncertaintyModel(**PROTOCOL["coupling_v3"]["field_uncertainty"])


def _electron_inputs() -> ElectronAdiabaticInputs:
    declaration = PROTOCOL["coupling_v3"]["electron_distribution"]
    return ElectronAdiabaticInputs(
        float(declaration["kinetic_energy_ev"]),
        float(declaration["perpendicular_energy_fraction"]),
        float(declaration["maximum_gyroradius_to_scale_length"]),
    )


def _classify_surface_failure(record: Any) -> tuple[str, ...]:
    mapping = {
        SurfaceStatus.OPEN_BOUNDARY: "CONTOUR_OPEN",
        SurfaceStatus.DISCONNECTED: "CONTOUR_DISCONNECTED",
        SurfaceStatus.EXACT_NULL: "CONTOUR_NULL",
        SurfaceStatus.NONADIABATIC: "NONADIABATIC",
        SurfaceStatus.UNCERTAINTY_DOMINATED: "COUPLING_UNCERTAINTY",
        SurfaceStatus.MISSING_ADIABATIC_INPUTS: "NONADIABATIC",
        SurfaceStatus.NUMERICALLY_INVALID: "COUPLING_INVALID",
        SurfaceStatus.PHYSICALLY_INVALID: "COUPLING_INVALID",
    }
    failures = {
        mapping[outcome.status]
        for cell in record.cells
        for outcome in cell.quantile_outcomes
        if outcome.status is not SurfaceStatus.VALID and outcome.status in mapping
    }
    return tuple(sorted(failures))


def _cell_distributions(record: Any) -> tuple[dict[str, float], ...]:
    distributions: list[dict[str, float]] = []
    coverage = float(record.uncertainty_model.coverage_factor)
    for cell in record.cells:
        surfaces = tuple(
            surface
            for surface in cell.surfaces
            if surface.probability.status is SurfaceStatus.VALID
        )
        if (
            cell.status is not SurfaceStatus.VALID
            or not surfaces
            or len(cell.quantile_outcomes)
            != len(PROTOCOL["topology"]["flux_quantiles"])
        ):
            raise ValueError("cell has no atomic all-quantile distribution")
        nominal_values = tuple(
            float(surface.probability.nominal_probability) for surface in surfaces
        )
        lower_values = tuple(
            float(surface.probability.probability_lower) for surface in surfaces
        )
        upper_values = tuple(
            float(surface.probability.probability_upper) for surface in surfaces
        )
        mirror_lowers = tuple(
            float(surface.probability.mirror_ratio_lower) for surface in surfaces
        )
        mirror_uppers = tuple(
            float(surface.probability.mirror_ratio_upper) for surface in surfaces
        )
        nominal = math.fsum(nominal_values) / len(nominal_values)
        lower, upper = min(lower_values), max(upper_values)
        if not 0.0 <= lower <= nominal <= upper < 1.0:
            raise ValueError("cell probability interval cannot enter plasma_network")
        mirror_lower, mirror_upper = min(mirror_lowers), max(mirror_uppers)
        standard = max(nominal - lower, upper - nominal) / coverage
        distributions.append(
            {
                "nominal_probability": nominal,
                "probability_lower": lower,
                "probability_upper": upper,
                "standard_uncertainty": standard,
                "mirror_ratio_nominal": 0.5 * (mirror_lower + mirror_upper),
                "mirror_ratio_lower": mirror_lower,
                "mirror_ratio_upper": mirror_upper,
                "mirror_standard_uncertainty": 0.5
                * (mirror_upper - mirror_lower)
                / coverage,
                "surface_count": len(surfaces),
            }
        )
    if len(distributions) != 4:
        raise ValueError("exactly four cell distributions are required")
    return tuple(distributions)


def _network_topology(
    candidate: BuiltCandidate,
    record: Any,
    distributions: tuple[dict[str, float], ...],
    probabilities: tuple[float, ...],
    closure_sha256: str,
):
    if len(record.interior_cusp_z_m) != 4 or len(probabilities) != 4:
        raise ValueError("N=4 projection requires exactly four stable v3 cells")
    positions = tuple(float(value) for value in record.interior_cusp_z_m)
    confidence = tuple(
        max(
            0.0,
            1.0
            - (
                item["probability_upper"] - item["probability_lower"]
            ),
        )
        for item in distributions
    )
    cells = tuple(
        GeometryCell(
            cell_id=f"cell-{index + 1}",
            axial_order=index,
            axial_position_m=position,
            volume_m3=math.pi
            * candidate.chamber_radius_m**2
            * max(candidate.pitch_m, record.cells[index].z_end_m - record.cells[index].z_start_m),
            confidence=confidence[index],
            provenance_sha256=provenance_hash(
                f"{record.record_hash}:geometry-cell:{index}"
            ),
        )
        for index, position in enumerate(positions)
    )
    nulls: list[GeometryNull] = []
    for index in range(3):
        distribution = distributions[index]
        nulls.append(
            GeometryNull(
                null_id=f"interior-cusp-{index + 1}",
                axial_position_m=0.5 * (positions[index] + positions[index + 1]),
                upstream_cell_id=cells[index].cell_id,
                downstream_cell_id=cells[index + 1].cell_id,
                classification=NullClassification.INTERIOR_CUSP,
                loss_probability=UncertainScalar(
                    probabilities[index],
                    distribution["standard_uncertainty"],
                    "1",
                    provenance_hash(
                        f"{record.record_hash}:probability:{index}"
                    ),
                ),
                mirror_ratio=UncertainScalar(
                    distribution["mirror_ratio_nominal"],
                    distribution["mirror_standard_uncertainty"],
                    "1",
                    provenance_hash(f"{record.record_hash}:mirror:{index}"),
                ),
                exclusion_reason=None,
                confidence=confidence[index],
                provenance_sha256=provenance_hash(
                    f"{record.record_hash}:interior-cusp:{index}"
                ),
            )
        )
    for index, boundary in enumerate(record.boundary_nulls):
        nulls.append(
            GeometryNull(
                null_id=f"finite-boundary-null-{index}",
                axial_position_m=boundary.z_m,
                upstream_cell_id=None,
                downstream_cell_id=None,
                classification=NullClassification.FINITE_BOUNDARY_NULL,
                loss_probability=None,
                mirror_ratio=None,
                exclusion_reason=(
                    f"finite computational boundary diagnostic: {boundary.boundary}"
                ),
                confidence=1.0,
                provenance_sha256=provenance_hash(
                    f"{record.record_hash}:boundary:{index}"
                ),
            )
        )
    terminals = (
        TerminalBoundary(
            "cathode",
            TerminalKind.CATHODE,
            cells[0].cell_id,
            record.cells[0].z_start_m,
            1.0,
            provenance_hash(f"{record.record_hash}:cathode"),
        ),
        TerminalBoundary(
            "anode",
            TerminalKind.ANODE,
            cells[-1].cell_id,
            record.cells[-1].z_end_m,
            1.0,
            provenance_hash(f"{record.record_hash}:anode"),
        ),
    )
    sigmas = tuple(item["standard_uncertainty"] for item in distributions[:3])
    covariance = tuple(
        tuple(sigmas[i] * sigmas[j] for j in range(3)) for i in range(3)
    )
    hashes = SemanticHashes(
        geometry_sha256=record.identity.geometry_hash,
        material_sha256=record.identity.material_hash,
        source_sha256=record.identity.source_hash,
        artifact_sha256=record.identity.artifact_hash,
        model_sha256=stable_hash(
            {
                "coupling_record": record.record_hash,
                "plasma_model": PROTOCOL["accepted_dependency_baseline"][
                    "plasma_model"
                ],
            }
        ),
        code_sha256=closure_sha256,
        schema_sha256=stable_hash(
            {
                "coupling": record.schema_version,
                "plasma": "plasma-chain-topology-2.0.0",
                "experiment": SCHEMA_VERSION,
            }
        ),
    )
    return build_chain_topology(
        SnapshotAdapter(
            GeometryTopologySnapshot(
                cells,
                tuple(nulls),
                terminals,
                covariance,
                hashes,
            )
        )
    )


def _network_options() -> NetworkSolverOptions:
    declaration = PROTOCOL["plasma_network"]
    return NetworkSolverOptions(
        least_squares=SolverOptions(
            max_iterations=int(declaration["maximum_iterations"]),
            residual_tolerance=float(declaration["residual_tolerance"]),
            gradient_tolerance=1e-10,
            step_tolerance=1e-12,
            initial_damping=1e-3,
        ),
        publication_policy=PublicationPolicy.REQUIRE_FULL_RANK,
        conservation_tolerance=float(declaration["conservation_tolerance"]),
    )


def _plasma_scenarios(distributions: tuple[dict[str, float], ...]):
    nominal = tuple(item["nominal_probability"] for item in distributions)
    yield "nominal", nominal
    for bits in itertools.product((0, 1), repeat=4):
        values = tuple(
            distributions[index][
                "probability_upper" if bit else "probability_lower"
            ]
            for index, bit in enumerate(bits)
        )
        yield "box-" + "".join(str(bit) for bit in bits), values


def _solve_plasma(
    candidate: BuiltCandidate,
    record: Any,
    distributions: tuple[dict[str, float], ...],
    closure_sha256: str,
) -> dict[str, Any]:
    outcomes: list[dict[str, Any]] = []
    for point_index, point in enumerate(
        PROTOCOL["plasma_network"]["operating_points"]
    ):
        for scenario_id, probabilities in _plasma_scenarios(distributions):
            topology = _network_topology(
                candidate,
                record,
                distributions,
                probabilities,
                closure_sha256,
            )
            anode_distribution = distributions[3]
            inputs = NetworkInputs(
                topology=topology,
                anode_voltage_v=float(point["anode_voltage_v"]),
                anode_current_a=float(point["anode_current_a"]),
                anode_arrival_probability=probabilities[3],
                anode_arrival_standard_uncertainty=anode_distribution[
                    "standard_uncertainty"
                ],
                anode_arrival_provenance_sha256=provenance_hash(
                    f"{record.record_hash}:anode:{scenario_id}"
                ),
            )
            result = solve_network_multistart(
                inputs,
                start_count=int(
                    PROTOCOL["plasma_network"]["deterministic_start_count"]
                ),
                options=_network_options(),
            )
            best = result.best.diagnostics
            identifiability = best.identifiability
            outcomes.append(
                {
                    "operating_point_id": f"point-{point_index + 1}",
                    "anode_voltage_v": point["anode_voltage_v"],
                    "anode_current_a": point["anode_current_a"],
                    "scenario_id": scenario_id,
                    "probabilities": list(probabilities),
                    "topology_identity_sha256": topology.identity_sha256,
                    "residual_root_found": best.numerical_converged,
                    "state_published": False,
                    "power_or_performance_published": False,
                    "residual_inf_norm": best.residual_inf_norm,
                    "conservation_inf_norm": best.conservation_inf_norm,
                    "selected_start_index": result.selected_start_index,
                    "residual_floor": result.residual_floor,
                    "rank": (
                        None
                        if identifiability is None
                        else {
                            "numerical_rank": identifiability.numerical_rank,
                            "state_size": identifiability.state_size,
                            "nullity": identifiability.nullity,
                            "structural_rank": identifiability.structural_rank,
                            "basis_valid": identifiability.basis_valid,
                        }
                    ),
                    "publication_reason": best.reason,
                    "identifiable_observables": [],
                    "attempts": [
                        {
                            "start_index": attempt.diagnostics.deterministic_start_index,
                            "residual_root_found": attempt.diagnostics.numerical_converged,
                            "published": False,
                            "reason": attempt.diagnostics.reason,
                            "residual_inf_norm": attempt.diagnostics.residual_inf_norm,
                            "conservation_inf_norm": attempt.diagnostics.conservation_inf_norm,
                            "rank": (
                                None
                                if attempt.diagnostics.identifiability is None
                                else attempt.diagnostics.identifiability.numerical_rank
                            ),
                        }
                        for attempt in result.attempts
                    ],
                }
            )
    roots = sum(item["residual_root_found"] for item in outcomes)
    return {
        "scenario_count": len(outcomes),
        "residual_root_count": roots,
        "all_interval_scenarios_have_root": roots == len(outcomes),
        "identifiable_observables": [],
        "unique_state_count": 0,
        "power_or_performance_publication_count": 0,
        "outcomes": outcomes,
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
    ).stdout.strip()
    parts = [item.strip() for item in query.splitlines()[0].split(",")]
    banner = subprocess.run(
        ("nvidia-smi",),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    cuda_line = next(
        line for line in banner.splitlines() if "CUDA Version:" in line
    )
    cuda_version = cuda_line.split("CUDA Version:", 1)[1].split()[0]
    return {
        "gpu_name": parts[0],
        "gpu_uuid": parts[1],
        "compute_capability": parts[2],
        "driver_version": parts[3],
        "reported_cuda_version": cuda_version,
        "warp_version": wp.__version__,
        "warp_device": str(device),
        "warp_device_name": device.name,
        "warp_device_uuid": device.uuid,
        "warp_device_architecture": device.arch,
        "warp_device_is_cuda": device.is_cuda,
        "python_implementation": os.sys.implementation.name,
        "python_version": ".".join(str(item) for item in os.sys.version_info[:3]),
        "platform": os.sys.platform,
    }


def _replay(
    candidate: BuiltCandidate, primary: SolvedRole
) -> dict[str, Any]:
    repeated = solve_problem_warp(
        primary.problem,
        config=_solver_config(),
        device=str(PROTOCOL["maps"]["solver"]["device"]),
    )
    differences = max_field_difference(primary.field, repeated)
    first = primary.field.diagnostics
    second = repeated.diagnostics
    diagnostic_relative = max(
        abs(first.relative_residual_l2 - second.relative_residual_l2)
        / max(
            abs(first.relative_residual_l2),
            abs(second.relative_residual_l2),
            1e-300,
        ),
        abs(first.final_residual_l2 - second.final_residual_l2)
        / max(abs(first.final_residual_l2), abs(second.final_residual_l2), 1e-300),
    )
    declaration = PROTOCOL["replay"]
    passed = (
        max(differences["br_max_abs_t"], differences["bz_max_abs_t"])
        <= float(declaration["maximum_b_component_absolute_difference_t"])
        and differences["psi_max_abs_wb"]
        <= float(declaration["maximum_psi_absolute_difference_wb"])
        and diagnostic_relative
        <= float(declaration["maximum_diagnostic_relative_difference"])
    )
    return {
        "candidate_id": candidate.candidate_id,
        "comparison": declaration["comparison"],
        "differences": differences,
        "diagnostic_relative_difference": diagnostic_relative,
        "passed": passed,
    }


def _candidate_failures(
    role_errors: Mapping[str, str],
    roles: Mapping[str, SolvedRole],
    topology: Mapping[str, Any] | None,
) -> list[str]:
    failures: list[str] = []
    role_code = {
        "primary": "FIELD_PRIMARY_INVALID",
        "downsampled": "FIELD_DOWNSAMPLED_INVALID",
        "enlarged_domain": "FIELD_ENLARGED_INVALID",
    }
    failures.extend(role_code[role] for role in role_errors)
    failures.extend(
        role_code[role]
        for role, solved in roles.items()
        if not all(solved.gates.values())
    )
    if topology is not None:
        if not topology["exact_count"]:
            failures.append("TOPOLOGY_COUNT")
        if topology["exact_count"] and not topology["geometry_registered"]:
            failures.append("TOPOLOGY_GEOMETRY_REGISTRATION")
        if not topology["endpoint_exclusion"]:
            failures.append("TOPOLOGY_ENDPOINT")
        if not topology["stable_shift"]:
            failures.append("TOPOLOGY_UNSTABLE")
        if not topology["enlarged_domain_boundary_comparison"]:
            failures.append("FIELD_DOMAIN_UNSTABLE")
    return sorted(set(failures))


def _ranking_key(case: Mapping[str, Any]) -> tuple[Any, ...]:
    quality = [
        item["quality"]
        for item in case["maps"].values()
        if isinstance(item, dict) and "quality" in item
    ]
    max_residual = max(
        (item["normalized_residual"] for item in quality), default=float("inf")
    )
    max_boundary = max(
        (item["boundary_to_peak_ratio"] for item in quality), default=float("inf")
    )
    plasma = case.get("plasma") or {}
    return (
        -int(bool(plasma.get("all_interval_scenarios_have_root"))),
        -int(bool(case.get("adiabatic"))),
        -int(bool(case.get("stable"))),
        max_residual,
        max_boundary,
        case["candidate_id"],
    )


def _report(dataset: Mapping[str, Any]) -> str:
    summary = dataset["summary"]
    failures = summary["failure_counts"]
    lines = [
        "# Preregistered four-cell topology search v2",
        "",
        f"- Preregistration commit: `{dataset['preregistration_commit_sha']}`",
        f"- Accepted coupling v3 baseline: `{ACCEPTED_COUPLING_COMMIT}`",
        f"- Evaluated: {summary['evaluated_count']}",
        f"- Three-map stable: {summary['stable_count']}",
        f"- All-quantile adiabatic v3: {summary['adiabatic_count']}",
        f"- Coupled into exact N=4 plasma_network: {summary['coupled_count']}",
        f"- Plasma residual-root scenarios: {summary['plasma_residual_root_scenario_count']}",
        f"- Identifiable observables: {summary['identifiable_observables']}",
        f"- Unique states published: {summary['unique_state_count']}",
        f"- Power/performance publications: {summary['power_or_performance_publication_count']}",
        "",
        "## Claim boundary",
        "",
        "Only connected constant-psi, same-contour v3 distributions that passed every",
        "preregistered quantile, covered field uncertainty, and rho_e/L_B gate are",
        "reported as physical mirror evidence. Residual roots are not performance.",
        "Rank-deficient plasma results contain diagnostics only: no state vector, power",
        "object, or performance claim is present.",
        "",
        "## Failure taxonomy counts",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in sorted(failures.items()))
    lines.extend(
        [
            "",
            "## Ranked candidates",
            "",
        ]
    )
    lines.extend(
        f"{index}. `{item['candidate_id']}` — stable={item['stable']}, "
        f"adiabatic={item['adiabatic']}, coupled={item['coupled']}, "
        f"failures={','.join(item['failures']) or 'none'}"
        for index, item in enumerate(dataset["ranking"][:10], 1)
    )
    return "\n".join(lines) + "\n"


def run_experiment(output_dir: Path) -> dict[str, Any]:
    """Execute the preregistered protocol exactly once in a clean detached worktree."""

    output = output_dir.resolve()
    if output.exists():
        raise RuntimeError("single-run results path already exists; rerun prohibited")
    closure = dependency_closure()
    output.mkdir(parents=True, exist_ok=False)
    lock_payload = {
        "schema_version": "cft-revival.exclusive-execution-lock/1.0.0",
        "experiment_id": PROTOCOL["experiment_id"],
        "preregistration_commit_sha": closure["preregistration_commit_sha"],
        "protocol_sha256": PROTOCOL_SHA256,
        "lock_identity_sha256": stable_hash(
            {
                "experiment_id": PROTOCOL["experiment_id"],
                "preregistration_commit_sha": closure["preregistration_commit_sha"],
                "protocol_sha256": PROTOCOL_SHA256,
            }
        ),
        "status": "exclusive_lock_acquired_before_single_execution",
    }
    lock_path = output / "execution-lock.json"
    with lock_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(lock_payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
    _write_bytes(lock_path, lock_path.read_bytes())
    protocol_copy_hash = _write_bytes(
        output / "preregistered-protocol.json", PROTOCOL_PATH.read_bytes()
    )
    runtime = _runtime_identity()
    _, runtime_hash = write_sealed_json(output / "runtime.json", runtime)
    declarations = sample_candidates()
    cases: list[dict[str, Any]] = []
    retained: list[
        tuple[
            tuple[Any, ...],
            BuiltCandidate,
            dict[str, SolvedRole],
        ]
    ] = []
    replay_rows: list[dict[str, Any]] = []
    replay_ids = set(PROTOCOL["replay"]["gpu_replay_candidate_ids"])
    for declaration in declarations:
        candidate_id = str(declaration["candidate_id"])
        role_errors: dict[str, str] = {}
        roles: dict[str, SolvedRole] = {}
        try:
            candidate = build_candidate(declaration)
        except Exception as error:
            cases.append(
                {
                    "candidate_id": candidate_id,
                    "sampling": declaration,
                    "geometry_valid": False,
                    "maps": {},
                    "stable": False,
                    "adiabatic": False,
                    "coupled": False,
                    "failures": ["GEOMETRY_INVALID"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        for role in ("primary", "downsampled", "enlarged_domain"):
            try:
                roles[role] = _solve_role(candidate, role, closure, runtime)
            except Exception as error:
                role_errors[role] = f"{type(error).__name__}: {error}"
        if candidate_id in replay_ids and "primary" in roles:
            try:
                replay_rows.append(_replay(candidate, roles["primary"]))
            except Exception as error:
                replay_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "passed": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        topology: dict[str, Any] | None = None
        if len(roles) == 3:
            topology = topology_gates(candidate, roles)
        failures = _candidate_failures(role_errors, roles, topology)
        coupling_payload: dict[str, Any] | None = None
        distributions: tuple[dict[str, float], ...] = ()
        plasma: dict[str, Any] | None = None
        stable = bool(topology and topology["stable"])
        stability_verified = False
        adiabatic = False
        coupled = False
        if stable:
            try:
                stability = verify_v3_topology_stability(
                    roles["primary"].evidence,
                    roles["downsampled"].evidence,
                    roles["enlarged_domain"].evidence,
                    maximum_cusp_shift_m=float(
                        PROTOCOL["topology"]["maximum_cross_map_cusp_shift_m"]
                    ),
                    reference_time_utc=ACCEPTANCE_TIME_UTC,
                )
                stability_verified = True
                registrations = tuple(
                    CellRegistration(
                        f"cell-{index + 1}",
                        tuple(
                            float(value)
                            for value in PROTOCOL["topology"]["flux_quantiles"]
                        ),
                    )
                    for index in range(4)
                )
                record = build_coupling_record(
                    roles["primary"].evidence,
                    stability_evidence=stability,
                    cell_registrations=registrations,
                    electron_inputs=_electron_inputs(),
                    surface_policy=_surface_policy(),
                    uncertainty_model=_uncertainty(),
                    reference_time_utc=ACCEPTANCE_TIME_UTC,
                )
                coupling_payload = coupling_record_dict(record)
                failures.extend(_classify_surface_failure(record))
                adiabatic = (
                    record.topology_status is TopologyStatus.RESOLVED
                    and len(global_solver_inputs(record)) > 0
                    and all(
                        surface.adiabatic_valid
                        for cell in record.cells
                        for surface in cell.surfaces
                    )
                )
                if adiabatic:
                    distributions = _cell_distributions(record)
                    coupled = len(distributions) == 4
                    if coupled:
                        try:
                            plasma = _solve_plasma(
                                candidate,
                                record,
                                distributions,
                                str(closure["closure_sha256"]),
                            )
                        except Exception as error:
                            coupled = False
                            failures.append("PLASMA_INPUT_INVALID")
                            coupling_payload["plasma_input_error"] = (
                                f"{type(error).__name__}: {error}"
                            )
                        else:
                            if not plasma["all_interval_scenarios_have_root"]:
                                failures.append("PLASMA_NONCONVERGENCE")
                            if any(
                                outcome["residual_root_found"]
                                and outcome["rank"] is not None
                                and outcome["rank"]["numerical_rank"]
                                < outcome["rank"]["state_size"]
                                for outcome in plasma["outcomes"]
                            ):
                                failures.append("PLASMA_RANK_DEFICIENT")
            except Exception as error:
                failures.append("COUPLING_INVALID")
                coupling_payload = {
                    "build_error": f"{type(error).__name__}: {error}"
                }
                if not stability_verified:
                    stable = False
                    failures.append("TOPOLOGY_UNSTABLE")
        case = {
            "candidate_id": candidate_id,
            "sampling": declaration,
            "geometry_valid": True,
            "geometry_sha256": candidate.geometry_sha256,
            "material_sha256": candidate.material_sha256,
            "source_sha256": candidate.source_sha256,
            "derived_geometry": {
                "pitch_m": candidate.pitch_m,
                "stage_centres_m": list(candidate.stage_centres_m),
                "cusp_targets_m": list(candidate.cusp_targets_m),
                "chamber_radius_m": candidate.chamber_radius_m,
                "wall_radius_m": candidate.wall_radius_m,
            },
            "maps": {
                role: (
                    {
                        "error": role_errors[role],
                    }
                    if role in role_errors
                    else {
                        "artifact_sha256": roles[role].artifact_hash,
                        "full_map_sha256": reverify_v3_evidence(
                            roles[role].evidence,
                            reference_time_utc=ACCEPTANCE_TIME_UTC,
                        ).field_map.full_map_hash,
                        "domain": asdict(roles[role].problem.domain),
                        "quality": roles[role].quality,
                        "gates": roles[role].gates,
                        "interior_cusp_z_m": list(roles[role].cusps_m),
                        "boundary_null_count": roles[role].boundary_null_count,
                    }
                )
                for role in ("primary", "downsampled", "enlarged_domain")
                if role in roles or role in role_errors
            },
            "topology": topology,
            "stable": stable,
            "adiabatic": adiabatic,
            "coupled": coupled,
            "cell_distributions": list(distributions),
            "coupling_v3": coupling_payload,
            "plasma": plasma,
            "failures": sorted(set(failures)),
        }
        cases.append(case)
        retained.append((_ranking_key(case), candidate, roles))
        retained.sort(key=lambda item: item[0])
        retained = retained[:2]
    ranking = sorted(cases, key=_ranking_key)
    failure_counts = {
        name: sum(name in case["failures"] for case in cases)
        for name in PROTOCOL["failure_taxonomy"]
    }
    summary = {
        "declared_candidate_count": len(declarations),
        "evaluated_count": len(cases),
        "three_map_accepted_count": sum(
            len(case["maps"]) == 3
            and all("error" not in value for value in case["maps"].values())
            for case in cases
        ),
        "stable_count": sum(case["stable"] for case in cases),
        "adiabatic_count": sum(case["adiabatic"] for case in cases),
        "coupled_count": sum(case["coupled"] for case in cases),
        "plasma_residual_root_scenario_count": sum(
            0
            if case["plasma"] is None
            else case["plasma"]["residual_root_count"]
            for case in cases
        ),
        "identifiable_observables": [],
        "unique_state_count": 0,
        "power_or_performance_publication_count": 0,
        "gpu_replay_pass_count": sum(item["passed"] for item in replay_rows),
        "gpu_replay_required_count": len(replay_ids),
        "failure_counts": failure_counts,
    }
    dataset_payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": "PREREGISTERED_PHYSICS_GATED_V2",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "preregistration_commit_sha": closure["preregistration_commit_sha"],
        "accepted_coupling_v3_commit_sha": ACCEPTED_COUPLING_COMMIT,
        "dependency_closure": closure,
        "runtime_identity": runtime,
        "claim_boundary": PROTOCOL["claim_boundary"],
        "summary": summary,
        "gpu_replay": replay_rows,
        "cases": cases,
        "ranking": [
            {
                "candidate_id": item["candidate_id"],
                "stable": item["stable"],
                "adiabatic": item["adiabatic"],
                "coupled": item["coupled"],
                "cell_distributions": item["cell_distributions"],
                "plasma": (
                    None
                    if item["plasma"] is None
                    else {
                        "scenario_count": item["plasma"]["scenario_count"],
                        "residual_root_count": item["plasma"][
                            "residual_root_count"
                        ],
                        "identifiable_observables": [],
                    }
                ),
                "failures": item["failures"],
            }
            for item in ranking
        ],
    }
    dataset, dataset_hash = write_sealed_json(
        output / "dataset.json", dataset_payload
    )
    representative_entries: list[dict[str, Any]] = []
    representative_dir = output / "representatives"
    for _, candidate, roles in retained:
        geometry_name = f"{candidate.candidate_id}-geometry.json"
        geometry_bytes = canonical_json(candidate.geometry.to_dict()).encode(
            "utf-8"
        ) + b"\n"
        geometry_hash = _write_bytes(
            representative_dir / geometry_name, geometry_bytes
        )
        representative_entries.append(
            {
                "candidate_id": candidate.candidate_id,
                "role": "geometry",
                "path": f"representatives/{geometry_name}",
                "sha256": geometry_hash,
            }
        )
        for role, solved in sorted(roles.items()):
            filename = f"{candidate.candidate_id}-{role}-field.json"
            digest = _write_bytes(
                representative_dir / filename, solved.artifact_bytes
            )
            representative_entries.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "role": role,
                    "path": f"representatives/{filename}",
                    "sha256": digest,
                }
            )
    report = _report(dataset)
    report_hash = _write_bytes(output / "report.md", report.encode("utf-8"))
    artifact_entries = [
        {
            "path": "dataset.json",
            "sha256": dataset_hash,
            "bytes": (output / "dataset.json").stat().st_size,
        },
        {
            "path": "runtime.json",
            "sha256": runtime_hash,
            "bytes": (output / "runtime.json").stat().st_size,
        },
        {
            "path": "report.md",
            "sha256": report_hash,
            "bytes": (output / "report.md").stat().st_size,
        },
        {
            "path": "execution-lock.json",
            "sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            "bytes": lock_path.stat().st_size,
        },
        {
            "path": "preregistered-protocol.json",
            "sha256": protocol_copy_hash,
            "bytes": (output / "preregistered-protocol.json").stat().st_size,
        },
    ]
    artifact_entries.extend(
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "bytes": (output / item["path"]).stat().st_size,
        }
        for item in representative_entries
    )
    manifest_payload = {
        "schema_version": MANIFEST_VERSION,
        "experiment_id": PROTOCOL["experiment_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "preregistration_commit_sha": closure["preregistration_commit_sha"],
        "accepted_coupling_v3_commit_sha": ACCEPTED_COUPLING_COMMIT,
        "dependency_closure_sha256": closure["closure_sha256"],
        "single_execution": True,
        "execution_lock_identity_sha256": lock_payload["lock_identity_sha256"],
        "summary": summary,
        "representatives": representative_entries,
        "artifacts": artifact_entries,
    }
    manifest, _ = write_sealed_json(output / "manifest.json", manifest_payload)
    validate_results(output)
    return {"dataset": dataset, "manifest": manifest}


def validate_results(output_dir: Path) -> dict[str, Any]:
    output = output_dir.resolve()
    dataset = load_sealed_json(output / "dataset.json")
    manifest = load_sealed_json(output / "manifest.json")
    runtime = load_sealed_json(output / "runtime.json")
    if (
        dataset["schema_version"] != SCHEMA_VERSION
        or manifest["schema_version"] != MANIFEST_VERSION
        or dataset["protocol_sha256"] != PROTOCOL_SHA256
        or manifest["protocol_sha256"] != PROTOCOL_SHA256
        or manifest["accepted_coupling_v3_commit_sha"] != ACCEPTED_COUPLING_COMMIT
        or dataset["accepted_coupling_v3_commit_sha"] != ACCEPTED_COUPLING_COMMIT
    ):
        raise ValueError("result protocol/schema/baseline identity mismatch")
    if hashlib.sha256(
        (output / "preregistered-protocol.json").read_bytes()
    ).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("result protocol copy differs from preregistration")
    if dataset["summary"]["evaluated_count"] != int(
        PROTOCOL["sampling"]["candidate_count"]
    ):
        raise ValueError("not every preregistered candidate was evaluated")
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
        raise ValueError("runtime identity is incomplete")
    if dataset["summary"]["unique_state_count"] != 0 or dataset["summary"][
        "power_or_performance_publication_count"
    ] != 0:
        raise ValueError("publication policy was violated")
    prohibited = {
        "state",
        "state_vector",
        "powers",
        "power_balance",
        "performance",
        "screening_performance",
    }
    for case in dataset["cases"]:
        plasma = case.get("plasma")
        if plasma is None:
            continue
        for outcome in plasma["outcomes"]:
            if prohibited.intersection(outcome):
                raise ValueError("plasma outcome contains prohibited publication data")
            if outcome["state_published"] or outcome[
                "power_or_performance_published"
            ]:
                raise ValueError("rank policy allowed a prohibited publication")
    if len(dataset["gpu_replay"]) != len(
        PROTOCOL["replay"]["gpu_replay_candidate_ids"]
    ) or not all(item["passed"] for item in dataset["gpu_replay"]):
        raise ValueError("required tolerance-based GPU replay did not pass")
    listed = {entry["path"]: entry for entry in manifest["artifacts"]}
    for relative, entry in listed.items():
        path = output / relative
        if not path.is_file():
            raise ValueError(f"manifest artifact missing: {relative}")
        if _verify_sidecar(path) != entry["sha256"]:
            raise ValueError(f"manifest digest mismatch: {relative}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"manifest byte count mismatch: {relative}")
    if set(listed) != {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
        and not path.name.endswith(".sha256")
        and path.name != "manifest.json"
    }:
        raise ValueError("manifest artifact inventory is not complete")
    return {"dataset": dataset, "manifest": manifest, "runtime": runtime}


def remove_failed_output(output_dir: Path) -> None:
    """Development helper used only before preregistration, never after execution."""

    if output_dir.exists():
        shutil.rmtree(output_dir)
