"""Deterministic replay of hash-bound L1b evidence."""

from __future__ import annotations

import hashlib
import json
import base64
import struct
import zlib
from dataclasses import asdict, dataclass
from math import fsum, hypot, pi, sqrt

from cft_revival.fields import AxisymmetricDomain
from cft_revival.fields.numerics import finalize_field_map
from cft_revival.geometry import deserialize_geometry
from cft_revival.magnetics import content_sha256, deserialize_handoff

from .adapters import rasterize_handoff
from .models import MaterialFieldValidationError
from .numerics import (
    _implementation_sha256,
    _apply_outer_boundary_values,
    apply_material_operator,
    assemble_rhs,
)


@dataclass(frozen=True, slots=True)
class ReplayReport:
    passed: bool
    coefficient_max_absolute_error: float
    source_max_absolute_error: float
    field_max_absolute_error_t: float
    true_residual_l2: float
    energy_balance_relative: float
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]
    problem: object


def _flat(rows: list[list[float]]) -> tuple[float, ...]:
    return tuple(value for row in rows for value in row)


def _max_error(left, right) -> float:
    if len(left) != len(right):
        return float("inf")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def _array_sha(*arrays) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        digest.update(struct.pack("<Q", len(values)))
        digest.update(struct.pack(f"<{len(values)}d", *values))
    return digest.hexdigest()


def _decode_solution(value: dict[str, object]) -> tuple[float, ...]:
    if set(value) != {
        "codec", "dtype", "layout", "count", "uncompressed_sha256", "data_base64"
    } or value["codec"] != "zlib-base64" or value["dtype"] != "float64-little-endian" or value["layout"] != "radial-major":
        raise MaterialFieldValidationError("raw solution encoding is unsupported")
    try:
        binary = zlib.decompress(base64.b64decode(value["data_base64"], validate=True))
    except (TypeError, ValueError, zlib.error) as error:
        raise MaterialFieldValidationError("raw solution decoding failed") from error
    if hashlib.sha256(binary).hexdigest() != value["uncompressed_sha256"]:
        raise MaterialFieldValidationError("raw solution binary hash mismatch")
    count = int(value["count"])
    if len(binary) != 8 * count:
        raise MaterialFieldValidationError("raw solution byte count mismatch")
    return struct.unpack(f"<{count}d", binary)


def replay_raw_run(raw: dict[str, object], *, backend: str) -> ReplayReport:
    """Rebuild a run from accepted bundles and verify its embedded solution."""
    problem_payload = raw["problem"]
    diagnostics = raw["diagnostics"]
    domain_payload = raw["domain"]
    geometry = deserialize_geometry(problem_payload["geometry_bundle_json"])
    contract = deserialize_handoff(problem_payload["magnetics_bundle_json"])
    if (
        geometry.canonical_sha256 != problem_payload["geometry_sha256"]
        or content_sha256(contract.to_dict()) != problem_payload["magnetics_sha256"]
    ):
        raise MaterialFieldValidationError("replay bundle identity mismatch")
    domain = AxisymmetricDomain(
        domain_payload["radius_m"],
        domain_payload["z_min_m"],
        domain_payload["z_max_m"],
        domain_payload["radial_intervals"],
        domain_payload["axial_intervals"],
    )
    rebuilt = rasterize_handoff(contract, domain, geometry=geometry)
    if (
        problem_payload["raster_diagnostics"]
        != [asdict(item) for item in rebuilt.raster_diagnostics]
        or problem_payload["weak_action_diagnostics"]
        != [asdict(item) for item in rebuilt.weak_action_diagnostics]
        or problem_payload["feature_effective_cells"]
        != [list(item) for item in rebuilt.feature_effective_cells]
        or problem_payload["qoi_locations_rz_m"]
        != [list(item) for item in rebuilt.qoi_locations_rz_m]
        or problem_payload["qoi_bore_windows_m"]
        != [list(item) for item in rebuilt.qoi_bore_windows_m]
        or problem_payload["open_boundary_policy"]
        != dict(rebuilt.open_boundary_policy)
    ):
        raise MaterialFieldValidationError(
            "replay diagnostics, policy, or physical QoI definitions differ"
        )
    coefficient_error = (
        0.0
        if _array_sha(
            rebuilt.radial_face_reluctivity_per_m_h,
            rebuilt.axial_face_reluctivity_per_m_h,
            rebuilt.remanence_g_r_face_a_per_m,
            rebuilt.remanence_g_z_face_a_per_m,
        )
        == problem_payload["coefficient_sha256"]
        else float("inf")
    )
    source = assemble_rhs(rebuilt)
    source_error = (
        0.0 if _array_sha(source) == problem_payload["source_sha256"] else float("inf")
    )
    solution = _decode_solution(raw["solution"])
    applied = apply_material_operator(rebuilt, solution)
    residual = tuple(source[index] - applied[index] for index in range(len(source)))
    true_residual = sqrt(
        fsum(
            residual[i * domain.shape[1] + j] ** 2
            for i in range(1, domain.shape[0] - 1)
            for j in range(1, domain.shape[1] - 1)
        )
    )
    field_solution = _apply_outer_boundary_values(rebuilt, solution)
    replayed = finalize_field_map(
        domain,
        field_solution,
        converged=True,
        iterations=0,
        initial_residual=0.0,
        final_residual=true_residual,
        residual_history=(true_residual,),
        backend="material_fields:replay",
    )
    field_error = 0.0
    magnetic = pi * domain.dr_m * domain.dz_m * fsum(
        solution[i * domain.shape[1] + j] * applied[i * domain.shape[1] + j]
        for i in range(1, domain.shape[0] - 1)
        for j in range(1, domain.shape[1] - 1)
    )
    coenergy = pi * domain.dr_m * domain.dz_m * fsum(
        solution[i * domain.shape[1] + j] * source[i * domain.shape[1] + j]
        for i in range(1, domain.shape[0] - 1)
        for j in range(1, domain.shape[1] - 1)
    )
    energy_error = abs(magnetic - coenergy) / max(
        abs(magnetic), abs(coenergy), 1.0e-300
    )
    config_json = diagnostics["run_config_json"]
    if (
        hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        != diagnostics["run_config_sha256"]
        or json.loads(config_json)["backend"] != backend
    ):
        raise MaterialFieldValidationError("replay solver configuration binding failed")
    expected_implementation = _implementation_sha256(
        *(
            ("adapters.py", "models.py", "numerics.py", "warp_solver.py")
            if backend.startswith("material_fields:warp:")
            else ("adapters.py", "models.py", "numerics.py")
        )
    )
    if diagnostics["implementation_sha256"] != expected_implementation:
        raise MaterialFieldValidationError("replay implementation identity mismatch")
    config_payload = json.loads(config_json)["config"]
    if float(config_payload["minimum_effective_feature_cells"]) < 12.0:
        raise MaterialFieldValidationError(
            "publication replay requires a twelve-cell minimum declaration"
        )
    residual_limit = max(
        float(config_payload["linear"]["absolute_tolerance"]),
        float(config_payload["linear"]["relative_tolerance"])
        * float(diagnostics["initial_residual_l2"]),
    )
    initial_residual = float(diagnostics["initial_residual_l2"])
    stored_final = float(diagnostics["final_true_residual_l2"])
    stored_relative = float(diagnostics["relative_true_residual_l2"])
    expected_stored_relative = (
        0.0 if initial_residual == 0.0 else stored_final / initial_residual
    )
    replay_relative = 0.0 if initial_residual == 0.0 else true_residual / initial_residual
    synchronization_count = int(diagnostics["host_synchronization_count"])
    check_interval = int(diagnostics["convergence_check_interval"])
    bounded_synchronization = (
        synchronization_count == 0
        if backend == "material_fields:python"
        else (
            check_interval >= 1
            and 3 <= synchronization_count
            <= int(diagnostics["iterations"]) // check_interval + 8
        )
    )
    passed = (
        coefficient_error == 0.0
        and source_error == 0.0
        and field_error == 0.0
        and true_residual <= residual_limit
        and abs(stored_relative - expected_stored_relative)
        <= max(1.0e-18, 1.0e-14 * expected_stored_relative)
        and abs(replay_relative - stored_relative)
        <= max(1.0e-15, 1.0e-6 * stored_relative)
        and abs(true_residual - stored_final)
        <= max(1.0e-12, 1.0e-6 * true_residual)
        and abs(energy_error - float(diagnostics["energy_balance_relative"]))
        <= 1.0e-12
        and bounded_synchronization
    )
    return ReplayReport(
        passed,
        coefficient_error,
        source_error,
        field_error,
        true_residual,
        energy_error,
        replayed.b_r_t,
        replayed.b_z_t,
        rebuilt,
    )
