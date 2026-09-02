"""Strict deterministic artifact and manifest contracts for L1a field maps."""

from __future__ import annotations

import builtins
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from math import fsum, hypot, isfinite, ulp
from pathlib import Path
from typing import Any, Iterable

from .models import (
    AxisymmetricProblem,
    FieldArtifactValidationError,
    FieldMap,
    SolverConfig,
    span_meets_minimum_grid_spacings,
)
from .numerics import current_density_grid, source_discretization_diagnostics
from .serialization import (
    CANONICALIZATION_V2,
    canonical_field_artifact_bytes,
    normalize_field_artifact_value,
    parse_field_json_bytes,
)

ARTIFACT_SCHEMA_VERSION = "cft-axisymmetric-field-map/1.2.0"
MANIFEST_SCHEMA_VERSION = "cft-axisymmetric-design-manifest/1.2.0"
LEGACY_ARTIFACT_SCHEMA_VERSION = "cft-axisymmetric-field-map/1.1.0"
LEGACY_MANIFEST_SCHEMA_VERSION = "cft-axisymmetric-design-manifest/1.1.0"
LEGACY_CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
CANONICALIZATION = CANONICALIZATION_V2
DEGENERATE_FIELD_ABS_T = 1.0e-14

# Every explicit contract failure in this module uses the documented typed
# ValueError subclass while preserving compatibility with callers catching
# builtins.ValueError.
ValueError = FieldArtifactValidationError


def _legacy_canonical_payload_bytes(payload: dict[str, object]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (builtins.ValueError, TypeError, OverflowError) as error:
        raise FieldArtifactValidationError(
            "canonical payload contains an unsupported or nonfinite value"
        ) from error


def canonical_payload_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        canonical_field_artifact_bytes(payload, representation="payload")
    ).hexdigest()


def _pretty_file_bytes(value: dict[str, object]) -> bytes:
    return canonical_field_artifact_bytes(value, representation="file")


def _seal(payload: dict[str, object]) -> dict[str, object]:
    normalized_payload = normalize_field_artifact_value(payload)
    sealed = dict(normalized_payload)
    sealed["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": CANONICALIZATION,
        "payload_sha256": canonical_payload_sha256(normalized_payload),
    }
    return normalize_field_artifact_value(sealed)


def _payload_without_integrity(value: dict[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if key != "integrity"}


def _indices(length: int, stride: int) -> tuple[int, ...]:
    selected = list(range(0, length, stride))
    if selected[-1] != length - 1:
        selected.append(length - 1)
    return tuple(selected)


def _axis_topology(field: FieldMap) -> dict[str, object]:
    scale = max(
        hypot(radial, axial)
        for radial_row, axial_row in zip(field.b_r_t, field.b_z_t, strict=True)
        for radial, axial in zip(radial_row, axial_row, strict=True)
    )
    return _axis_topology_values(field.z_m, field.b_r_t[0], field.b_z_t[0], scale)


def _axis_topology_values(
    z_values, br_values, bz_values, scale: float
) -> dict[str, object]:
    bz = bz_values
    br = br_values
    tolerance = max(1.0e-15, 1.0e-10 * scale, 32.0 * ulp(scale))
    if scale <= DEGENERATE_FIELD_ABS_T:
        return {
            "status": "degenerate_near_zero_field",
            "field_scale_t": scale,
            "null_tolerance_t": tolerance,
            "axis_nulls": [],
            "axis_plateaus": [],
        }

    low = [
        hypot(br[index], bz[index]) <= tolerance
        for index in range(len(z_values))
    ]
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(1, len(low) - 1):
        if low[index] and start is None:
            start = index
        if start is not None and (not low[index] or index == len(low) - 2):
            end = index if low[index] else index - 1
            runs.append((start, end))
            start = None

    nulls: list[dict[str, object]] = []
    plateaus: list[dict[str, object]] = []
    consumed_crossings: set[int] = set()
    for first, last in runs:
        left = first - 1
        right = last + 1
        if first == last and bz[left] * bz[right] < 0.0:
            nulls.append(
                {
                    "kind": "sign_changing_sample",
                    "r_m": 0.0,
                    "z_m": z_values[first],
                    "b_magnitude_t": hypot(br[first], bz[first]),
                }
            )
            consumed_crossings.update((left, first))
        elif (
            first == last
            and hypot(br[first], bz[first])
            < min(hypot(br[left], bz[left]), hypot(br[right], bz[right]))
            and min(abs(bz[left]), abs(bz[right])) > 10.0 * tolerance
        ):
            nulls.append(
                {
                    "kind": "isolated_sample",
                    "r_m": 0.0,
                    "z_m": z_values[first],
                    "b_magnitude_t": hypot(br[first], bz[first]),
                }
            )
        else:
            plateaus.append(
                {
                    "z_start_m": z_values[first],
                    "z_end_m": z_values[last],
                    "sample_count": last - first + 1,
                }
            )

    for index in range(1, len(bz) - 2):
        if index in consumed_crossings or low[index] or low[index + 1]:
            continue
        if bz[index] * bz[index + 1] < 0.0:
            fraction = abs(bz[index]) / (abs(bz[index]) + abs(bz[index + 1]))
            nulls.append(
                {
                    "kind": "sign_changing_interpolated",
                    "r_m": 0.0,
                    "z_m": z_values[index]
                    + fraction * (z_values[index + 1] - z_values[index]),
                    "b_magnitude_t": 0.0,
                }
            )
    nulls.sort(key=lambda item: float(item["z_m"]))
    status = (
        "resolved_axis_nulls"
        if nulls
        else "near_zero_axis_plateau"
        if plateaus
        else "no_resolved_axis_null"
    )
    return {
        "status": status,
        "field_scale_t": scale,
        "null_tolerance_t": tolerance,
        "axis_nulls": nulls,
        "axis_plateaus": plateaus,
    }


def field_artifact(
    problem: AxisymmetricProblem,
    config: SolverConfig,
    field: FieldMap,
    *,
    map_stride: int = 2,
    wall_radius_m: float | None = None,
) -> dict[str, object]:
    if isinstance(map_stride, bool) or not isinstance(map_stride, int) or map_stride < 1:
        raise ValueError("map_stride must be an integer >= 1")
    radial = _indices(len(field.r_m), map_stride)
    axial = _indices(len(field.z_m), map_stride)
    try:
        wall_radius = (
            0.8 * problem.domain.radius_m
            if wall_radius_m is None
            else float(wall_radius_m)
        )
    except (builtins.ValueError, TypeError, OverflowError) as error:
        raise FieldArtifactValidationError(
            "wall_radius_m must fit a finite binary64 number"
        ) from error
    if not isfinite(wall_radius) or not 0.0 <= wall_radius <= problem.domain.radius_m:
        raise ValueError("wall_radius_m must be finite and inside the domain")
    wall_i = min(range(len(field.r_m)), key=lambda i: abs(field.r_m[i] - wall_radius))
    magnitudes = tuple(
        hypot(br, bz)
        for br_row, bz_row in zip(field.b_r_t, field.b_z_t, strict=True)
        for br, bz in zip(br_row, bz_row, strict=True)
    )
    nr_nodes, nz_nodes = problem.domain.shape
    boundary_magnitudes = tuple(
        hypot(field.b_r_t[i][j], field.b_z_t[i][j])
        for i in range(nr_nodes)
        for j in range(nz_nodes)
        if i == nr_nodes - 1 or j in (0, nz_nodes - 1)
    )
    source_requested = fsum(
        source.polarity * source.ampere_turns_a for source in problem.sources
    )
    source_sampled = (
        fsum(current_density_grid(problem)) * problem.domain.dr_m * problem.domain.dz_m
    )
    diagnostics = asdict(field.diagnostics)
    diagnostics["residual_history_l2"] = list(field.diagnostics.residual_history_l2)
    diagnostics["requested_signed_ampere_turns_a"] = source_requested
    diagnostics["sampled_signed_ampere_turns_a"] = source_sampled
    diagnostics["ampere_turn_balance_error_a"] = source_sampled - source_requested
    diagnostics["source_discretization"] = list(
        source_discretization_diagnostics(problem)
    )
    domain = problem.domain
    payload: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_level": "L1a",
        "model_description": (
            "linear constant-permeability axisymmetric equivalent-current FDM; "
            "not permanent-magnet/nonlinear-iron FEM and not a plasma solve"
        ),
        "input": {
            "name": problem.name,
            "domain": {
                "radius_m": domain.radius_m,
                "z_min_m": domain.z_min_m,
                "z_max_m": domain.z_max_m,
                "radial_intervals": domain.radial_intervals,
                "axial_intervals": domain.axial_intervals,
                "dr_m": domain.dr_m,
                "dz_m": domain.dz_m,
            },
            "outer_boundary": problem.outer_boundary,
            "permeability_h_per_m": problem.permeability_h_per_m,
            "sources": [asdict(source) for source in problem.sources],
            "source_convention": (
                "continuous J_phi=polarity*ampere_turns/band_area; source must "
                "remain inside resolved interior dual-cell support; nodal values "
                "are overlap averages without boundary-volume compression"
            ),
            "solver": asdict(config),
        },
        "provenance": {
            "implementation": "cft_revival.fields structured-grid matrix-free FDM",
            "scalar": "IEEE-754 binary64",
            "backend": field.diagnostics.backend,
            "equation_ledger": "modern/spec/fields/equation-solver-ledger-v1.json",
        },
        "summary": {
            "b_magnitude_min_t": min(magnitudes),
            "b_magnitude_max_t": max(magnitudes),
            "outer_boundary_b_magnitude_min_t": min(boundary_magnitudes),
            "signed_requested_ampere_turns_a": source_requested,
            "topology": _axis_topology(field),
        },
        "diagnostics": diagnostics,
        "profiles": {
            "centreline": {
                "r_m": 0.0,
                "z_m": list(field.z_m),
                "b_r_t": list(field.b_r_t[0]),
                "b_z_t": list(field.b_z_t[0]),
            },
            "wall": {
                "requested_r_m": wall_radius,
                "sampled_r_m": field.r_m[wall_i],
                "z_m": list(field.z_m),
                "b_r_t": list(field.b_r_t[wall_i]),
                "b_z_t": list(field.b_z_t[wall_i]),
            },
        },
        "field_map": {
            "layout": "radial-major; values[field_r_index][field_z_index]",
            "downsample_stride": map_stride,
            "r_m": [field.r_m[i] for i in radial],
            "z_m": [field.z_m[j] for j in axial],
            "psi_wb": [[field.psi_wb[i][j] for j in axial] for i in radial],
            "b_r_t": [[field.b_r_t[i][j] for j in axial] for i in radial],
            "b_z_t": [[field.b_z_t[i][j] for j in axial] for i in radial],
            "b_magnitude_t": [
                [hypot(field.b_r_t[i][j], field.b_z_t[i][j]) for j in axial]
                for i in radial
            ],
        },
        "limitations": [
            "Hypothetical equivalent azimuthal currents only; no magnet grade or magnetization.",
            "Constant scalar permeability; no material interfaces or nonlinear B-H curve.",
            "Finite homogeneous-Dirichlet truncation boundary; no infinite/open boundary.",
            "Outer-boundary sensitivity requires a separate domain-size convergence study.",
            "Flux reconstruction identity is not independent solution validation.",
            "Structured-grid second-order FDM; this result is not FEM.",
            "No plasma response, discharge closure, thermal model, or experimental calibration.",
        ],
    }
    artifact = _seal(payload)
    validate_field_artifact(artifact)
    return artifact


def _closed(value: object, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{name} keys must be exactly {sorted(keys)}; "
            f"missing={sorted(keys-actual)}, unknown={sorted(actual-keys)}"
        )
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _number(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    try:
        converted = float(value)
    except (builtins.ValueError, OverflowError) as error:
        raise ValueError(f"{name} must fit a finite binary64 number") from error
    if not isfinite(converted) or (nonnegative and converted < 0.0):
        raise ValueError(f"{name} must be finite" + (" and non-negative" if nonnegative else ""))
    return converted


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _numeric_list(value: object, name: str, *, minimum_length: int = 1) -> list[float]:
    if not isinstance(value, list) or len(value) < minimum_length:
        raise ValueError(f"{name} must be a list with at least {minimum_length} entries")
    return [_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _monotonic(values: list[float], name: str) -> None:
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be finite and strictly increasing")


def _matrix(
    value: object, name: str, rows: int, columns: int
) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{name} shape does not match coordinates")
    result: list[list[float]] = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError(f"{name} shape does not match coordinates")
        numeric = _numeric_list(row, f"{name}[{index}]", minimum_length=columns)
        result.append(numeric)
    return result


def _validate_integrity(value: dict[str, object], kind: str) -> None:
    integrity = _closed(
        value.get("integrity"),
        f"{kind}.integrity",
        {"algorithm", "canonicalization", "payload_sha256"},
    )
    schema = value.get("schema_version")
    legacy = schema in {
        LEGACY_ARTIFACT_SCHEMA_VERSION,
        LEGACY_MANIFEST_SCHEMA_VERSION,
    }
    expected_canonicalization = (
        LEGACY_CANONICALIZATION if legacy else CANONICALIZATION
    )
    if (
        integrity["algorithm"] != "sha256"
        or integrity["canonicalization"] != expected_canonicalization
    ):
        raise ValueError(f"{kind} integrity algorithm/canonicalization is unsupported")
    digest = _string(integrity["payload_sha256"], f"{kind}.integrity.payload_sha256")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{kind} payload SHA-256 must be lowercase hexadecimal")
    payload = _payload_without_integrity(value)
    expected = (
        hashlib.sha256(_legacy_canonical_payload_bytes(payload)).hexdigest()
        if legacy
        else canonical_payload_sha256(payload)
    )
    if digest != expected:
        raise ValueError(f"{kind} canonical payload SHA-256 mismatch")


def _validate_topology_object(value: object, name: str) -> dict[str, Any]:
    topology = _closed(
        value,
        name,
        {"status", "field_scale_t", "null_tolerance_t", "axis_nulls", "axis_plateaus"},
    )
    status = _string(topology["status"], f"{name}.status")
    if status not in {
        "degenerate_near_zero_field",
        "resolved_axis_nulls",
        "near_zero_axis_plateau",
        "no_resolved_axis_null",
    }:
        raise ValueError(f"{name}.status is unsupported")
    field_scale = _number(topology["field_scale_t"], f"{name}.field_scale_t", nonnegative=True)
    _number(topology["null_tolerance_t"], f"{name}.null_tolerance_t", nonnegative=True)
    nulls = topology["axis_nulls"]
    plateaus = topology["axis_plateaus"]
    if not isinstance(nulls, list) or not isinstance(plateaus, list):
        raise ValueError(f"{name} nulls/plateaus must be lists")
    if status == "degenerate_near_zero_field":
        if field_scale > DEGENERATE_FIELD_ABS_T or nulls or plateaus:
            raise ValueError("degenerate topology must not fabricate nulls or plateaus")
    for index, value in enumerate(nulls):
        null = _closed(
            value,
            f"{name}.axis_nulls[{index}]",
            {"kind", "r_m", "z_m", "b_magnitude_t"},
        )
        if null["kind"] not in {
            "sign_changing_sample",
            "sign_changing_interpolated",
            "isolated_sample",
        }:
            raise ValueError("axis null kind is unsupported")
        if _number(null["r_m"], f"{name}.axis_nulls[{index}].r_m") != 0.0:
            raise ValueError("axis null radius must be zero")
        _number(null["z_m"], f"{name}.axis_nulls[{index}].z_m")
        _number(
            null["b_magnitude_t"],
            f"{name}.axis_nulls[{index}].b_magnitude_t",
            nonnegative=True,
        )
    for index, value in enumerate(plateaus):
        plateau = _closed(
            value,
            f"{name}.axis_plateaus[{index}]",
            {"z_start_m", "z_end_m", "sample_count"},
        )
        start = _number(
            plateau["z_start_m"], f"{name}.axis_plateaus[{index}].z_start_m"
        )
        end = _number(
            plateau["z_end_m"], f"{name}.axis_plateaus[{index}].z_end_m"
        )
        if end < start:
            raise ValueError("axis plateau range is inverted")
        _integer(
            plateau["sample_count"],
            f"{name}.axis_plateaus[{index}].sample_count",
            minimum=1,
        )
    return topology


def validate_field_artifact(artifact: dict[str, object]) -> None:
    top = _closed(
        artifact,
        "artifact",
        {
            "schema_version",
            "model_level",
            "model_description",
            "input",
            "provenance",
            "summary",
            "diagnostics",
            "profiles",
            "field_map",
            "limitations",
            "integrity",
        },
    )
    if top["schema_version"] not in {
        ARTIFACT_SCHEMA_VERSION,
        LEGACY_ARTIFACT_SCHEMA_VERSION,
    } or top["model_level"] != "L1a":
        raise ValueError("unsupported artifact schema or model level")
    _string(top["model_description"], "artifact.model_description")
    input_data = _closed(
        top["input"],
        "artifact.input",
        {
            "name",
            "domain",
            "outer_boundary",
            "permeability_h_per_m",
            "sources",
            "source_convention",
            "solver",
        },
    )
    _string(input_data["name"], "artifact.input.name")
    if input_data["outer_boundary"] != "homogeneous_dirichlet_psi":
        raise ValueError("artifact outer boundary is unsupported")
    _number(input_data["permeability_h_per_m"], "artifact.input.permeability_h_per_m")
    _string(input_data["source_convention"], "artifact.input.source_convention")
    domain = _closed(
        input_data["domain"],
        "artifact.input.domain",
        {
            "radius_m",
            "z_min_m",
            "z_max_m",
            "radial_intervals",
            "axial_intervals",
            "dr_m",
            "dz_m",
        },
    )
    radius = _number(domain["radius_m"], "domain.radius_m")
    z_min = _number(domain["z_min_m"], "domain.z_min_m")
    z_max = _number(domain["z_max_m"], "domain.z_max_m")
    radial_intervals = _integer(domain["radial_intervals"], "domain.radial_intervals", minimum=4)
    axial_intervals = _integer(domain["axial_intervals"], "domain.axial_intervals", minimum=4)
    dr = _number(domain["dr_m"], "domain.dr_m")
    dz = _number(domain["dz_m"], "domain.dz_m")
    if radius <= 0.0 or z_max <= z_min or dr <= 0.0 or dz <= 0.0:
        raise ValueError("artifact domain extents/spacings must be positive")
    if dr != radius / radial_intervals or dz != (z_max - z_min) / axial_intervals:
        raise ValueError("artifact derived grid spacing is inconsistent")

    sources = input_data["sources"]
    if not isinstance(sources, list):
        raise ValueError("artifact.input.sources must be a list")
    for index, source_value in enumerate(sources):
        source = _closed(
            source_value,
            f"sources[{index}]",
            {
                "name",
                "r_inner_m",
                "r_outer_m",
                "z_min_m",
                "z_max_m",
                "ampere_turns_a",
                "polarity",
            },
        )
        _string(source["name"], f"sources[{index}].name")
        source_numbers = {
            key: _number(source[key], f"sources[{index}].{key}")
            for key in (
                "r_inner_m",
                "r_outer_m",
                "z_min_m",
                "z_max_m",
                "ampere_turns_a",
            )
        }
        if (
            source_numbers["r_inner_m"] < 0.5 * dr
            or source_numbers["r_outer_m"] > radius - 0.5 * dr
            or source_numbers["z_min_m"] < z_min + 0.5 * dz
            or source_numbers["z_max_m"] > z_max - 0.5 * dz
        ):
            raise ValueError(f"sources[{index}] lies outside interior dual-cell support")
        if not span_meets_minimum_grid_spacings(
            source_numbers["r_inner_m"],
            source_numbers["r_outer_m"],
            dr,
        ) or not span_meets_minimum_grid_spacings(
            source_numbers["z_min_m"],
            source_numbers["z_max_m"],
            dz,
        ):
            raise ValueError(f"sources[{index}] is underresolved")
        if source_numbers["ampere_turns_a"] < 0.0:
            raise ValueError(f"sources[{index}].ampere_turns_a must be non-negative")
        if source["polarity"] not in (-1, 1) or isinstance(source["polarity"], bool):
            raise ValueError(f"sources[{index}].polarity must be -1 or +1")

    solver = _closed(
        input_data["solver"],
        "artifact.input.solver",
        {
            "relative_tolerance",
            "absolute_tolerance",
            "max_iterations",
            "residual_history_stride",
            "max_true_residual_restarts",
        },
    )
    relative_tolerance = _number(solver["relative_tolerance"], "solver.relative_tolerance")
    absolute_tolerance = _number(solver["absolute_tolerance"], "solver.absolute_tolerance")
    if relative_tolerance <= 0.0 or absolute_tolerance < 0.0:
        raise ValueError("artifact solver tolerances are invalid")
    max_iterations = _integer(solver["max_iterations"], "solver.max_iterations", minimum=1)
    _integer(solver["residual_history_stride"], "solver.residual_history_stride", minimum=1)
    _integer(
        solver["max_true_residual_restarts"],
        "solver.max_true_residual_restarts",
        minimum=0,
    )

    provenance = _closed(
        top["provenance"],
        "artifact.provenance",
        {"implementation", "scalar", "backend", "equation_ledger"},
    )
    for key, value in provenance.items():
        _string(value, f"artifact.provenance.{key}")

    diagnostics = _closed(
        top["diagnostics"],
        "artifact.diagnostics",
        {
            "converged",
            "iterations",
            "initial_residual_l2",
            "final_residual_l2",
            "relative_residual_l2",
            "residual_history_l2",
            "max_flux_reconstruction_identity_t_per_m",
            "true_residual_restarts",
            "stagnation_detected",
            "backend",
            "requested_signed_ampere_turns_a",
            "sampled_signed_ampere_turns_a",
            "ampere_turn_balance_error_a",
            "source_discretization",
        },
    )
    if diagnostics["converged"] is not True or diagnostics["stagnation_detected"] is not False:
        raise ValueError("artifact solver status is not converged/non-stagnant")
    iterations = _integer(diagnostics["iterations"], "diagnostics.iterations")
    if iterations > max_iterations:
        raise ValueError("artifact iteration count exceeds solver maximum")
    initial_residual = _number(
        diagnostics["initial_residual_l2"], "diagnostics.initial_residual_l2", nonnegative=True
    )
    final_residual = _number(
        diagnostics["final_residual_l2"], "diagnostics.final_residual_l2", nonnegative=True
    )
    relative_residual = _number(
        diagnostics["relative_residual_l2"],
        "diagnostics.relative_residual_l2",
        nonnegative=True,
    )
    expected_relative = 0.0 if initial_residual == 0.0 else final_residual / initial_residual
    if relative_residual != expected_relative:
        raise ValueError("artifact relative residual is inconsistent")
    if final_residual > max(absolute_tolerance, relative_tolerance * initial_residual):
        raise ValueError("artifact true residual exceeds solver tolerance")
    history = _numeric_list(
        diagnostics["residual_history_l2"],
        "diagnostics.residual_history_l2",
    )
    if any(value < 0.0 for value in history):
        raise ValueError("artifact residual history must be non-negative")
    _number(
        diagnostics["max_flux_reconstruction_identity_t_per_m"],
        "diagnostics.max_flux_reconstruction_identity_t_per_m",
        nonnegative=True,
    )
    _integer(diagnostics["true_residual_restarts"], "diagnostics.true_residual_restarts")
    _string(diagnostics["backend"], "diagnostics.backend")
    if diagnostics["backend"] != provenance["backend"]:
        raise ValueError("diagnostic and provenance backends differ")
    for key in (
        "requested_signed_ampere_turns_a",
        "sampled_signed_ampere_turns_a",
        "ampere_turn_balance_error_a",
    ):
        _number(diagnostics[key], f"diagnostics.{key}")
    if diagnostics["ampere_turn_balance_error_a"] != (
        diagnostics["sampled_signed_ampere_turns_a"]
        - diagnostics["requested_signed_ampere_turns_a"]
    ):
        raise ValueError("artifact ampere-turn balance is inconsistent")
    if not isinstance(diagnostics["source_discretization"], list):
        raise ValueError("diagnostics.source_discretization must be a list")
    for index, item in enumerate(diagnostics["source_discretization"]):
        source_diagnostic = _closed(
            item,
            f"source_discretization[{index}]",
            {
                "name",
                "requested_area_m2",
                "represented_overlap_area_m2",
                "area_error_m2",
                "requested_centroid_r_m",
                "represented_centroid_r_m",
                "centroid_r_error_m",
                "requested_centroid_z_m",
                "represented_centroid_z_m",
                "centroid_z_error_m",
                "requested_signed_ampere_turns_a",
                "represented_signed_ampere_turns_a",
                "ampere_turn_error_a",
                "radial_nodes_touched",
                "axial_nodes_touched",
                "dual_cells_touched",
            },
        )
        _string(source_diagnostic["name"], f"source_discretization[{index}].name")
        for key in source_diagnostic.keys() - {
            "name",
            "radial_nodes_touched",
            "axial_nodes_touched",
            "dual_cells_touched",
        }:
            _number(source_diagnostic[key], f"source_discretization[{index}].{key}")
        for key in ("radial_nodes_touched", "axial_nodes_touched", "dual_cells_touched"):
            _integer(source_diagnostic[key], f"source_discretization[{index}].{key}", minimum=1)
        if source_diagnostic["name"] != sources[index]["name"]:
            raise ValueError("source discretization order/name does not match sources")
        expected_area = (
            sources[index]["r_outer_m"] - sources[index]["r_inner_m"]
        ) * (sources[index]["z_max_m"] - sources[index]["z_min_m"])
        if source_diagnostic["requested_area_m2"] != expected_area:
            raise ValueError("source discretization requested area is inconsistent")
        expected_current = sources[index]["polarity"] * sources[index]["ampere_turns_a"]
        if source_diagnostic["requested_signed_ampere_turns_a"] != expected_current:
            raise ValueError("source discretization requested current is inconsistent")
        if source_diagnostic["area_error_m2"] != (
            source_diagnostic["represented_overlap_area_m2"]
            - source_diagnostic["requested_area_m2"]
        ):
            raise ValueError("source discretization area error is inconsistent")
        if source_diagnostic["ampere_turn_error_a"] != (
            source_diagnostic["represented_signed_ampere_turns_a"]
            - source_diagnostic["requested_signed_ampere_turns_a"]
        ):
            raise ValueError("source discretization ampere-turn error is inconsistent")
    if len(diagnostics["source_discretization"]) != len(sources):
        raise ValueError("source discretization count does not match sources")

    profiles = _closed(top["profiles"], "artifact.profiles", {"centreline", "wall"})
    centreline = _closed(
        profiles["centreline"],
        "profiles.centreline",
        {"r_m", "z_m", "b_r_t", "b_z_t"},
    )
    if _number(centreline["r_m"], "profiles.centreline.r_m") != 0.0:
        raise ValueError("centreline radius must be zero")
    centre_z = _numeric_list(
        centreline["z_m"], "profiles.centreline.z_m", minimum_length=2
    )
    _monotonic(centre_z, "profiles.centreline.z_m")
    if len(centre_z) != axial_intervals + 1:
        raise ValueError("centreline length does not match axial grid")
    centre_br = _numeric_list(centreline["b_r_t"], "profiles.centreline.b_r_t")
    centre_bz = _numeric_list(centreline["b_z_t"], "profiles.centreline.b_z_t")
    if len(centre_br) != len(centre_z) or len(centre_bz) != len(centre_z):
        raise ValueError("centreline field shape mismatch")
    expected_centre_z = [
        z_min + index * dz for index in range(axial_intervals + 1)
    ]
    if centre_z != expected_centre_z:
        raise ValueError("centreline coordinates are inconsistent with the grid")
    wall = _closed(
        profiles["wall"],
        "profiles.wall",
        {"requested_r_m", "sampled_r_m", "z_m", "b_r_t", "b_z_t"},
    )
    for key in ("requested_r_m", "sampled_r_m"):
        value = _number(wall[key], f"profiles.wall.{key}")
        if not 0.0 <= value <= radius:
            raise ValueError(f"profiles.wall.{key} lies outside the domain")
    wall_z = _numeric_list(wall["z_m"], "profiles.wall.z_m", minimum_length=2)
    _monotonic(wall_z, "profiles.wall.z_m")
    if wall_z != centre_z:
        raise ValueError("wall and centreline coordinates differ")
    wall_br = _numeric_list(wall["b_r_t"], "profiles.wall.b_r_t")
    wall_bz = _numeric_list(wall["b_z_t"], "profiles.wall.b_z_t")
    if len(wall_br) != len(wall_z) or len(wall_bz) != len(wall_z):
        raise ValueError("wall field shape mismatch")

    field_map = _closed(
        top["field_map"],
        "artifact.field_map",
        {
            "layout",
            "downsample_stride",
            "r_m",
            "z_m",
            "psi_wb",
            "b_r_t",
            "b_z_t",
            "b_magnitude_t",
        },
    )
    if field_map["layout"] != "radial-major; values[field_r_index][field_z_index]":
        raise ValueError("artifact field-map layout is unsupported")
    stride = _integer(
        field_map["downsample_stride"], "field_map.downsample_stride", minimum=1
    )
    r_values = _numeric_list(field_map["r_m"], "field_map.r_m", minimum_length=2)
    z_values = _numeric_list(field_map["z_m"], "field_map.z_m", minimum_length=2)
    _monotonic(r_values, "field_map.r_m")
    _monotonic(z_values, "field_map.z_m")
    if r_values[0] != 0.0 or r_values[-1] != radius:
        raise ValueError("field-map radial coordinates do not span the domain")
    if z_values[0] != z_min or z_values[-1] != z_max:
        raise ValueError("field-map axial coordinates do not span the domain")
    expected_radial_indices = _indices(radial_intervals + 1, stride)
    expected_axial_indices = _indices(axial_intervals + 1, stride)
    if r_values != [index * dr for index in expected_radial_indices]:
        raise ValueError("field-map radial coordinates are inconsistent with stride")
    if z_values != [z_min + index * dz for index in expected_axial_indices]:
        raise ValueError("field-map axial coordinates are inconsistent with stride")
    psi = _matrix(field_map["psi_wb"], "field_map.psi_wb", len(r_values), len(z_values))
    br = _matrix(field_map["b_r_t"], "field_map.b_r_t", len(r_values), len(z_values))
    bz = _matrix(field_map["b_z_t"], "field_map.b_z_t", len(r_values), len(z_values))
    magnitude = _matrix(
        field_map["b_magnitude_t"],
        "field_map.b_magnitude_t",
        len(r_values),
        len(z_values),
    )
    del psi
    for i in range(len(r_values)):
        for j in range(len(z_values)):
            expected = hypot(br[i][j], bz[i][j])
            tolerance = max(8.0 * ulp(expected), 1.0e-300)
            if abs(magnitude[i][j] - expected) > tolerance:
                raise ValueError("field-map |B| is inconsistent with Br/Bz")

    summary = _closed(
        top["summary"],
        "artifact.summary",
        {
            "b_magnitude_min_t",
            "b_magnitude_max_t",
            "outer_boundary_b_magnitude_min_t",
            "signed_requested_ampere_turns_a",
            "topology",
        },
    )
    minimum = _number(summary["b_magnitude_min_t"], "summary.b_magnitude_min_t", nonnegative=True)
    maximum = _number(summary["b_magnitude_max_t"], "summary.b_magnitude_max_t", nonnegative=True)
    if maximum < minimum:
        raise ValueError("summary field range is inverted")
    _number(
        summary["outer_boundary_b_magnitude_min_t"],
        "summary.outer_boundary_b_magnitude_min_t",
        nonnegative=True,
    )
    _number(summary["signed_requested_ampere_turns_a"], "summary.signed_requested_ampere_turns_a")
    expected_signed_current = fsum(
        source["polarity"] * source["ampere_turns_a"] for source in sources
    )
    if (
        summary["signed_requested_ampere_turns_a"] != expected_signed_current
        or diagnostics["requested_signed_ampere_turns_a"] != expected_signed_current
    ):
        raise ValueError("requested signed ampere-turn totals are inconsistent")
    topology = _validate_topology_object(summary["topology"], "summary.topology")
    if topology["field_scale_t"] != maximum:
        raise ValueError("summary topology field scale must equal |B| maximum")
    observed_magnitudes = [
        *[value for row in magnitude for value in row],
        *[hypot(radial, axial) for radial, axial in zip(centre_br, centre_bz)],
        *[hypot(radial, axial) for radial, axial in zip(wall_br, wall_bz)],
    ]
    if maximum < max(observed_magnitudes) or minimum > min(observed_magnitudes):
        raise ValueError("summary field range does not contain serialized fields")
    expected_topology = _axis_topology_values(
        centre_z, centre_br, centre_bz, maximum
    )
    if topology != expected_topology:
        raise ValueError("summary topology is inconsistent with centreline fields")

    limitations = top["limitations"]
    if not isinstance(limitations, list) or not limitations:
        raise ValueError("artifact limitations must be a non-empty list")
    for index, value in enumerate(limitations):
        _string(value, f"artifact.limitations[{index}]")
    _validate_integrity(artifact, "artifact")


def _write_canonical_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def field_artifact_canonical_bytes(artifact: dict[str, object]) -> bytes:
    """Validate one current artifact and return its sole persistent bytes."""

    validate_field_artifact(artifact)
    if artifact["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("legacy v1.1 artifacts are read-only and cannot be rewritten")
    return canonical_field_artifact_bytes(artifact, representation="file")


def write_field_artifact(path: str | Path, artifact: dict[str, object]) -> str:
    return _write_canonical_bytes(
        Path(path), field_artifact_canonical_bytes(artifact)
    )


def _validate_file_sidecar(path: Path, expected_digest: str | None = None) -> str:
    data_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_digest is not None and data_digest != expected_digest:
        raise ValueError(f"file SHA-256 mismatch for {path.name}")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing SHA-256 sidecar for {path.name}")
    expected_line = f"{data_digest}  {path.name}\n"
    if sidecar.read_text(encoding="ascii") != expected_line:
        raise ValueError(f"invalid SHA-256 sidecar for {path.name}")
    return data_digest


def _legacy_parse_json_bytes(data: bytes, source: str) -> dict[str, object]:
    def closed_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant {value!r} in {source}")

    try:
        loaded = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=closed_object,
            parse_constant=reject_constant,
        )
    except FieldArtifactValidationError:
        raise
    except (builtins.ValueError, OverflowError) as error:
        raise FieldArtifactValidationError(
            f"invalid JSON numeric value in {source}"
        ) from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{source} must contain one JSON object")
    return loaded


def _load_document_bytes(data: bytes, source: str) -> dict[str, object]:
    raw = _legacy_parse_json_bytes(data, source)
    schema = raw.get("schema_version")
    if schema in {ARTIFACT_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION}:
        return parse_field_json_bytes(
            data,
            source=source,
            require_canonical_file_bytes=True,
        )
    return raw


def reload_field_artifact_bytes(
    data: bytes,
    *,
    source: str = "<field-artifact-bytes>",
    allow_legacy_v1_1: bool = True,
) -> dict[str, object]:
    """Reload canonical current bytes or explicitly accepted legacy v1.1 bytes."""

    artifact = _load_document_bytes(data, source)
    if (
        artifact.get("schema_version") == LEGACY_ARTIFACT_SCHEMA_VERSION
        and not allow_legacy_v1_1
    ):
        raise ValueError("legacy v1.1 artifact reads are disabled")
    validate_field_artifact(artifact)
    return artifact


def validate_field_artifact_file(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_payload_sha256: str | None = None,
) -> dict[str, object]:
    artifact_path = Path(path)
    _validate_file_sidecar(artifact_path, expected_file_sha256)
    loaded = reload_field_artifact_bytes(
        artifact_path.read_bytes(), source=artifact_path.name
    )
    if expected_payload_sha256 is not None:
        actual = loaded["integrity"]["payload_sha256"]
        if actual != expected_payload_sha256:
            raise ValueError(f"payload SHA-256 mismatch for {artifact_path.name}")
    return loaded


def manifest_entry(
    artifact_path: str | Path,
    artifact: dict[str, object],
    file_sha256: str,
) -> dict[str, object]:
    path = Path(artifact_path)
    return {
        "name": artifact["input"]["name"],
        "artifact": path.name,
        "artifact_file_sha256": file_sha256,
        "artifact_payload_sha256": artifact["integrity"]["payload_sha256"],
        "backend": artifact["diagnostics"]["backend"],
        "iterations": artifact["diagnostics"]["iterations"],
        "relative_residual_l2": artifact["diagnostics"]["relative_residual_l2"],
        "b_magnitude_min_t": artifact["summary"]["b_magnitude_min_t"],
        "b_magnitude_max_t": artifact["summary"]["b_magnitude_max_t"],
        "topology": artifact["summary"]["topology"],
    }


def design_manifest(entries: Iterable[dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_level": "L1a",
        "runtime_policy": "timings are diagnostic only and are not benchmark evidence",
        "designs": deepcopy(list(entries)),
    }
    manifest = _seal(payload)
    validate_design_manifest(manifest)
    return manifest


def validate_design_manifest(manifest: dict[str, object]) -> None:
    top = _closed(
        manifest,
        "manifest",
        {"schema_version", "model_level", "runtime_policy", "designs", "integrity"},
    )
    if top["schema_version"] not in {
        MANIFEST_SCHEMA_VERSION,
        LEGACY_MANIFEST_SCHEMA_VERSION,
    } or top["model_level"] != "L1a":
        raise ValueError("unsupported manifest schema or model level")
    _string(top["runtime_policy"], "manifest.runtime_policy")
    designs = top["designs"]
    if not isinstance(designs, list) or not designs:
        raise ValueError("manifest.designs must be a non-empty list")
    names: set[str] = set()
    artifacts: set[str] = set()
    for index, value in enumerate(designs):
        entry = _closed(
            value,
            f"manifest.designs[{index}]",
            {
                "name",
                "artifact",
                "artifact_file_sha256",
                "artifact_payload_sha256",
                "backend",
                "iterations",
                "relative_residual_l2",
                "b_magnitude_min_t",
                "b_magnitude_max_t",
                "topology",
            },
        )
        name = _string(entry["name"], f"manifest.designs[{index}].name")
        artifact_name = _string(
            entry["artifact"], f"manifest.designs[{index}].artifact"
        )
        if Path(artifact_name).name != artifact_name or artifact_name in {".", ".."}:
            raise ValueError("manifest artifact path must be a plain filename")
        if name in names or artifact_name in artifacts:
            raise ValueError("manifest design names and artifact filenames must be unique")
        names.add(name)
        artifacts.add(artifact_name)
        for key in ("artifact_file_sha256", "artifact_payload_sha256"):
            digest = _string(entry[key], f"manifest.designs[{index}].{key}")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"manifest.designs[{index}].{key} is not SHA-256")
        _string(entry["backend"], f"manifest.designs[{index}].backend")
        _integer(entry["iterations"], f"manifest.designs[{index}].iterations")
        _number(
            entry["relative_residual_l2"],
            f"manifest.designs[{index}].relative_residual_l2",
            nonnegative=True,
        )
        minimum = _number(
            entry["b_magnitude_min_t"],
            f"manifest.designs[{index}].b_magnitude_min_t",
            nonnegative=True,
        )
        maximum = _number(
            entry["b_magnitude_max_t"],
            f"manifest.designs[{index}].b_magnitude_max_t",
            nonnegative=True,
        )
        if maximum < minimum:
            raise ValueError("manifest field range is inverted")
        topology = _validate_topology_object(
            entry["topology"], f"manifest.designs[{index}].topology"
        )
        if topology["field_scale_t"] != maximum:
            raise ValueError("manifest topology field scale must equal |B| maximum")
    _validate_integrity(manifest, "manifest")


def write_design_manifest(path: str | Path, manifest: dict[str, object]) -> str:
    validate_design_manifest(manifest)
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("legacy v1.1 manifests are read-only and cannot be rewritten")
    data = canonical_field_artifact_bytes(manifest, representation="file")
    return _write_canonical_bytes(Path(path), data)


def validate_design_manifest_file(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    _validate_file_sidecar(manifest_path)
    loaded = _load_document_bytes(
        manifest_path.read_bytes(), manifest_path.name
    )
    validate_design_manifest(loaded)
    root = manifest_path.resolve().parent
    for entry in loaded["designs"]:
        artifact_path = (root / entry["artifact"]).resolve()
        if artifact_path.parent != root:
            raise ValueError("manifest artifact path escapes its directory")
        artifact = validate_field_artifact_file(
            artifact_path,
            expected_file_sha256=entry["artifact_file_sha256"],
            expected_payload_sha256=entry["artifact_payload_sha256"],
        )
        if artifact["input"]["name"] != entry["name"]:
            raise ValueError("manifest design name does not match artifact payload")
        for key in (
            "backend",
            "iterations",
            "relative_residual_l2",
        ):
            if artifact["diagnostics"][key] != entry[key]:
                raise ValueError(f"manifest {key} does not match artifact")
        for key in ("b_magnitude_min_t", "b_magnitude_max_t", "topology"):
            if artifact["summary"][key] != entry[key]:
                raise ValueError(f"manifest {key} does not match artifact")
    return loaded
