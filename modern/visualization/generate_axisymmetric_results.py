"""Generate the standalone L1a axisymmetric field-results visualization.

Only checked solver artifacts are embedded. The generator intentionally omits
timestamps and runtime measurements so identical inputs produce identical
bytes.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import fsum, hypot, isclose, isfinite, ulp
from pathlib import Path
from typing import Any, Mapping, Sequence

MODERN = Path(__file__).resolve().parents[1]
RESULTS = MODERN / "examples" / "axisymmetric" / "results"
DEFAULT_MANIFEST = RESULTS / "manifest-l1a-v1.json"
DEFAULT_OUTPUT = Path(__file__).with_name("axisymmetric-results.html")

EXPECTED_DESIGNS = (
    (
        "hypothetical-compact-mirror",
        "hypothetical-compact-mirror-l1a-v1.json",
        "6510f6ea687022f358103bba99456e7bf651686e3add29205c0560c933981afb",
        "92e5535af0492e1697dad2540d8f6e837ba11f28f7a81626673a1c0004183348",
        "Compact mirror",
    ),
    (
        "hypothetical-opposed-cusp",
        "hypothetical-opposed-cusp-l1a-v1.json",
        "dbf05208dc77e694bb40bb3ca82e4ee3e7126bb3036156f7fa1a726eab06b5c6",
        "c4c7c3dc45466bfa4ba187e925b8e41a1c979b3700a59e185d24897501f97263",
        "Opposed cusp",
    ),
    (
        "hypothetical-thick-outer-triplet",
        "hypothetical-thick-outer-triplet-l1a-v1.json",
        "ac5420d9276d3db03adffe548a459706de95593d74c4181af4034ddbd1ce4b7a",
        "d6ef0a42b0a73cfafc7cad1a3fdca8ca59fff4c13a444f5fe5ee8fae9ebf690b",
        "Thick outer triplet",
    ),
)
EXPECTED_MANIFEST_FILE_SHA256 = (
    "8444389efc87f89495e34d46ccf2deedcc44ee65614dfdd660beecf84cedc3b4"
)
EXPECTED_MANIFEST_PAYLOAD_SHA256 = (
    "2c912b847702e14223170917850d1ecd5fbdfb45899d96ddb222b5577531d7a6"
)
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
DEGENERATE_FIELD_ABS_T = 1.0e-14


def _load_object(path: Path, label: str) -> dict[str, Any]:
    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path.name}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r} in {path.name}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=closed_object,
            parse_constant=reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid readable JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _closed(value: Any, label: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys do not match the closed schema")
    return value


def _canonical_payload_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _verify_integrity(value: dict[str, Any], label: str) -> str:
    integrity = _closed(
        value.get("integrity"),
        f"{label}.integrity",
        {"algorithm", "canonicalization", "payload_sha256"},
    )
    if (
        integrity["algorithm"] != "sha256"
        or integrity["canonicalization"] != CANONICALIZATION
    ):
        raise ValueError(f"{label} integrity algorithm/canonicalization is unsupported")
    digest = integrity["payload_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} payload SHA-256 must be lowercase hexadecimal")
    payload = {key: item for key, item in value.items() if key != "integrity"}
    if digest != _canonical_payload_sha256(payload):
        raise ValueError(f"{label} canonical payload SHA-256 mismatch")
    return digest


def _verify_file(path: Path, label: str, expected_digest: str | None = None) -> str:
    try:
        digest = sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"{label} file is not readable: {path.name}") from exc
    if expected_digest is not None and digest != expected_digest:
        raise ValueError(f"{label} file SHA-256 mismatch")
    sidecar = path.with_name(path.name + ".sha256")
    expected_line = f"{digest}  {path.name}\n"
    try:
        actual_line = sidecar.read_text(encoding="ascii")
    except OSError as exc:
        raise ValueError(f"{label} is missing its SHA-256 sidecar") from exc
    if actual_line != expected_line:
        raise ValueError(f"{label} SHA-256 sidecar is invalid")
    return digest


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _number_vector(value: Any, label: str, *, minimum: int = 1) -> list[float]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} values")
    return [_finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _strictly_increasing(values: Sequence[float], label: str) -> None:
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{label} must be strictly increasing")


def _matrix(
    value: Any, label: str, radial_count: int, axial_count: int
) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != radial_count:
        raise ValueError(f"{label} radial dimension does not match r_m")
    rows: list[list[float]] = []
    for index, row in enumerate(value):
        parsed = _number_vector(row, f"{label}[{index}]", minimum=axial_count)
        if len(parsed) != axial_count:
            raise ValueError(f"{label}[{index}] axial dimension does not match z_m")
        rows.append(parsed)
    return rows


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _indices(length: int, stride: int) -> list[int]:
    selected = list(range(0, length, stride))
    if selected[-1] != length - 1:
        selected.append(length - 1)
    return selected


def _validate_topology(value: Any, label: str) -> Mapping[str, Any]:
    topology = _closed(
        value,
        label,
        {"status", "field_scale_t", "null_tolerance_t", "axis_nulls", "axis_plateaus"},
    )
    statuses = {
        "degenerate_near_zero_field",
        "resolved_axis_nulls",
        "near_zero_axis_plateau",
        "no_resolved_axis_null",
    }
    status = topology["status"]
    if status not in statuses:
        raise ValueError(f"{label}.status is unsupported")
    field_scale = _finite_number(topology["field_scale_t"], f"{label}.field_scale_t")
    tolerance = _finite_number(topology["null_tolerance_t"], f"{label}.null_tolerance_t")
    if field_scale < 0 or tolerance < 0:
        raise ValueError(f"{label} scales must be non-negative")
    nulls = topology["axis_nulls"]
    plateaus = topology["axis_plateaus"]
    if not isinstance(nulls, list) or not isinstance(plateaus, list):
        raise ValueError(f"{label} nulls/plateaus must be arrays")
    for index, item in enumerate(nulls):
        null = _closed(
            item,
            f"{label}.axis_nulls[{index}]",
            {"kind", "r_m", "z_m", "b_magnitude_t"},
        )
        if null["kind"] not in {
            "sign_changing_sample", "sign_changing_interpolated", "isolated_sample"
        }:
            raise ValueError(f"{label} axis null kind is unsupported")
        if _finite_number(null["r_m"], "axis null r") != 0.0:
            raise ValueError(f"{label} axis null must lie on r=0")
        _finite_number(null["z_m"], "axis null z")
        if _finite_number(null["b_magnitude_t"], "axis null |B|") < 0:
            raise ValueError(f"{label} axis null magnitude must be non-negative")
    for index, item in enumerate(plateaus):
        plateau = _closed(
            item,
            f"{label}.axis_plateaus[{index}]",
            {"z_start_m", "z_end_m", "sample_count"},
        )
        start = _finite_number(plateau["z_start_m"], "plateau z_start")
        end = _finite_number(plateau["z_end_m"], "plateau z_end")
        _integer(plateau["sample_count"], "plateau sample_count", 1)
        if end < start:
            raise ValueError(f"{label} axis plateau range is inverted")
    if status == "degenerate_near_zero_field":
        if field_scale > DEGENERATE_FIELD_ABS_T or nulls or plateaus:
            raise ValueError("degenerate topology must not fabricate nulls or plateaus")
    elif status == "resolved_axis_nulls":
        if not nulls or plateaus:
            raise ValueError("resolved-null topology must contain only classified nulls")
    elif status == "near_zero_axis_plateau":
        if not plateaus or nulls:
            raise ValueError("plateau topology must contain only classified plateaus")
    elif nulls or plateaus:
        raise ValueError("no-null topology must not contain nulls or plateaus")
    return topology


def _axis_topology_values(
    z_values: Sequence[float],
    br_values: Sequence[float],
    bz_values: Sequence[float],
    scale: float,
) -> dict[str, Any]:
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
        hypot(br_values[index], bz_values[index]) <= tolerance
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
    nulls: list[dict[str, Any]] = []
    plateaus: list[dict[str, Any]] = []
    consumed_crossings: set[int] = set()
    for first, last in runs:
        left, right = first - 1, last + 1
        if first == last and bz_values[left] * bz_values[right] < 0.0:
            nulls.append(
                {
                    "kind": "sign_changing_sample",
                    "r_m": 0.0,
                    "z_m": z_values[first],
                    "b_magnitude_t": hypot(br_values[first], bz_values[first]),
                }
            )
            consumed_crossings.update((left, first))
        elif (
            first == last
            and hypot(br_values[first], bz_values[first])
            < min(
                hypot(br_values[left], bz_values[left]),
                hypot(br_values[right], bz_values[right]),
            )
            and min(abs(bz_values[left]), abs(bz_values[right])) > 10.0 * tolerance
        ):
            nulls.append(
                {
                    "kind": "isolated_sample",
                    "r_m": 0.0,
                    "z_m": z_values[first],
                    "b_magnitude_t": hypot(br_values[first], bz_values[first]),
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
    for index in range(1, len(bz_values) - 2):
        if index in consumed_crossings or low[index] or low[index + 1]:
            continue
        if bz_values[index] * bz_values[index + 1] < 0.0:
            fraction = abs(bz_values[index]) / (
                abs(bz_values[index]) + abs(bz_values[index + 1])
            )
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


def _validate_profile(
    profile: Any,
    label: str,
    *,
    expected_keys: set[str],
    expected_z: Sequence[float],
) -> None:
    parsed = _closed(profile, label, expected_keys)
    z = _number_vector(parsed["z_m"], f"{label}.z_m", minimum=2)
    if z != list(expected_z):
        raise ValueError(f"{label}.z_m does not match the configured grid")
    for key in ("b_r_t", "b_z_t"):
        values = _number_vector(parsed[key], f"{label}.{key}", minimum=len(z))
        if len(values) != len(z):
            raise ValueError(f"{label}.{key} length does not match z_m")


def _validate_artifact(
    artifact: dict[str, Any],
    manifest_entry: Mapping[str, Any],
    expected_name: str,
) -> None:
    required = {
        "diagnostics", "field_map", "input", "limitations", "model_description",
        "integrity", "model_level", "profiles", "provenance", "schema_version", "summary",
    }
    if set(artifact) != required:
        raise ValueError(f"{expected_name} artifact top-level keys do not match the contract")
    if artifact["schema_version"] != "cft-axisymmetric-field-map/1.1.0":
        raise ValueError(f"{expected_name} artifact schema_version is unsupported")
    if artifact["model_level"] != "L1a":
        raise ValueError(f"{expected_name} artifact model_level must be L1a")
    _text(artifact["model_description"], f"{expected_name}.model_description")

    inputs = _closed(
        artifact["input"],
        f"{expected_name}.input",
        {
            "name", "domain", "outer_boundary", "permeability_h_per_m",
            "solver", "source_convention", "sources",
        },
    )
    if inputs["name"] != expected_name:
        raise ValueError(f"{expected_name} artifact input identity does not match")
    if inputs["outer_boundary"] != "homogeneous_dirichlet_psi":
        raise ValueError(f"{expected_name} outer boundary is unsupported")
    if _finite_number(inputs["permeability_h_per_m"], "permeability") <= 0:
        raise ValueError(f"{expected_name} permeability must be positive")
    _text(inputs["source_convention"], "source convention")
    domain = _closed(
        inputs["domain"],
        f"{expected_name}.domain",
        {"radius_m", "z_min_m", "z_max_m", "radial_intervals", "axial_intervals",
         "dr_m", "dz_m"},
    )
    radius = _finite_number(domain["radius_m"], f"{expected_name} radius")
    z_min = _finite_number(domain["z_min_m"], f"{expected_name} z_min")
    z_max = _finite_number(domain["z_max_m"], f"{expected_name} z_max")
    radial_intervals = _integer(domain["radial_intervals"], "radial_intervals", 4)
    axial_intervals = _integer(domain["axial_intervals"], "axial_intervals", 4)
    dr = _finite_number(domain["dr_m"], "dr_m")
    dz = _finite_number(domain["dz_m"], "dz_m")
    if not (radius > 0 and z_min < z_max and dr > 0 and dz > 0):
        raise ValueError(f"{expected_name} domain bounds are invalid")
    if dr != radius / radial_intervals or dz != (z_max - z_min) / axial_intervals:
        raise ValueError(f"{expected_name} derived grid spacing is inconsistent")

    sources = inputs["sources"]
    if not isinstance(sources, list) or not 2 <= len(sources) <= 3:
        raise ValueError(f"{expected_name} must contain two or three source bands")
    for index, source in enumerate(sources):
        source = _closed(
            source,
            f"{expected_name}.sources[{index}]",
            {"name", "r_inner_m", "r_outer_m", "z_min_m", "z_max_m",
             "ampere_turns_a", "polarity"},
        )
        _text(source["name"], "source name")
        polarity = source.get("polarity")
        if polarity not in (-1, 1) or isinstance(polarity, bool):
            raise ValueError(f"{expected_name} source {index} polarity must be -1 or +1")
        ri = _finite_number(source.get("r_inner_m"), "source r_inner_m")
        ro = _finite_number(source.get("r_outer_m"), "source r_outer_m")
        za = _finite_number(source.get("z_min_m"), "source z_min_m")
        zb = _finite_number(source.get("z_max_m"), "source z_max_m")
        turns = _finite_number(source.get("ampere_turns_a"), "source ampere_turns_a")
        if not (
            ri >= 0.5 * dr and ro <= radius - 0.5 * dr
            and za >= z_min + 0.5 * dz and zb <= z_max - 0.5 * dz
            and ro - ri >= 2 * dr and zb - za >= 2 * dz and turns >= 0
        ):
            raise ValueError(f"{expected_name} source {index} geometry is outside the domain")

    solver = _closed(
        inputs["solver"],
        f"{expected_name}.solver",
        {"relative_tolerance", "absolute_tolerance", "max_iterations",
         "residual_history_stride", "max_true_residual_restarts"},
    )
    relative_tolerance = _finite_number(solver["relative_tolerance"], "relative tolerance")
    absolute_tolerance = _finite_number(solver["absolute_tolerance"], "absolute tolerance")
    max_iterations = _integer(solver["max_iterations"], "max_iterations", 1)
    _integer(solver["residual_history_stride"], "residual_history_stride", 1)
    max_restarts = _integer(
        solver["max_true_residual_restarts"], "max_true_residual_restarts"
    )
    if relative_tolerance <= 0 or absolute_tolerance < 0:
        raise ValueError(f"{expected_name} solver tolerances are invalid")

    field = artifact["field_map"]
    expected_field_keys = {
        "b_magnitude_t", "b_r_t", "b_z_t", "downsample_stride",
        "layout", "psi_wb", "r_m", "z_m",
    }
    if not isinstance(field, Mapping) or set(field) != expected_field_keys:
        raise ValueError(f"{expected_name} field_map keys do not match the contract")
    if field["layout"] != "radial-major; values[field_r_index][field_z_index]":
        raise ValueError(f"{expected_name} field_map layout is unsupported")
    stride = field["downsample_stride"]
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"{expected_name} downsample_stride is invalid")
    r = _number_vector(field["r_m"], f"{expected_name}.r_m", minimum=3)
    z = _number_vector(field["z_m"], f"{expected_name}.z_m", minimum=3)
    _strictly_increasing(r, f"{expected_name}.r_m")
    _strictly_increasing(z, f"{expected_name}.z_m")
    if not (isclose(r[0], 0.0, abs_tol=1e-15) and isclose(r[-1], radius)):
        raise ValueError(f"{expected_name} field radial bounds do not match the domain")
    if not (isclose(z[0], z_min) and isclose(z[-1], z_max)):
        raise ValueError(f"{expected_name} field axial bounds do not match the domain")
    if r != [index * dr for index in _indices(radial_intervals + 1, stride)]:
        raise ValueError(f"{expected_name} field radial coordinates do not match stride")
    if z != [z_min + index * dz for index in _indices(axial_intervals + 1, stride)]:
        raise ValueError(f"{expected_name} field axial coordinates do not match stride")
    matrices = {
        key: _matrix(field[key], f"{expected_name}.{key}", len(r), len(z))
        for key in ("psi_wb", "b_r_t", "b_z_t", "b_magnitude_t")
    }
    for i in range(len(r)):
        for j in range(len(z)):
            expected_b = hypot(matrices["b_r_t"][i][j], matrices["b_z_t"][i][j])
            tolerance = max(8.0 * ulp(expected_b), 1.0e-300)
            if abs(matrices["b_magnitude_t"][i][j] - expected_b) > tolerance:
                raise ValueError(f"{expected_name} |B| does not equal hypot(Br,Bz)")

    provenance = _closed(
        artifact["provenance"],
        f"{expected_name}.provenance",
        {"implementation", "scalar", "backend", "equation_ledger"},
    )
    for key, value in provenance.items():
        _text(value, f"{expected_name}.provenance.{key}")
    diagnostics = _closed(
        artifact["diagnostics"],
        f"{expected_name}.diagnostics",
        {
            "converged", "iterations", "initial_residual_l2", "final_residual_l2",
            "relative_residual_l2", "residual_history_l2",
            "max_flux_reconstruction_identity_t_per_m", "true_residual_restarts",
            "stagnation_detected", "backend", "requested_signed_ampere_turns_a",
            "sampled_signed_ampere_turns_a", "ampere_turn_balance_error_a",
            "source_discretization",
        },
    )
    if provenance.get("backend") != diagnostics.get("backend"):
        raise ValueError(f"{expected_name} backend provenance does not match diagnostics")
    if manifest_entry.get("backend") != diagnostics.get("backend"):
        raise ValueError(f"{expected_name} manifest backend does not match artifact")
    residuals = _number_vector(
        diagnostics.get("residual_history_l2"), f"{expected_name} residual history", minimum=2
    )
    if any(value < 0 for value in residuals):
        raise ValueError(f"{expected_name} residual history must be non-negative")
    if diagnostics["converged"] is not True or diagnostics["stagnation_detected"] is not False:
        raise ValueError(f"{expected_name} solver status is not converged/non-stagnant")
    initial = _finite_number(diagnostics["initial_residual_l2"], "initial residual")
    final = _finite_number(diagnostics["final_residual_l2"], "final residual")
    relative = _finite_number(diagnostics["relative_residual_l2"], "relative residual")
    if min(initial, final, relative) < 0:
        raise ValueError(f"{expected_name} residuals must be non-negative")
    expected_relative = 0.0 if initial == 0.0 else final / initial
    if relative != expected_relative:
        raise ValueError(f"{expected_name} relative residual is inconsistent")
    if final > max(absolute_tolerance, relative_tolerance * initial):
        raise ValueError(f"{expected_name} true residual exceeds solver tolerance")
    if residuals[-1] != final:
        raise ValueError(f"{expected_name} convergence diagnostics are inconsistent")
    iterations = _integer(diagnostics["iterations"], "iterations")
    if iterations > max_iterations:
        raise ValueError(f"{expected_name} iteration count exceeds solver maximum")
    if _integer(diagnostics["true_residual_restarts"], "true residual restarts") > max_restarts:
        raise ValueError(f"{expected_name} true residual restart count exceeds policy")
    if _finite_number(
        diagnostics["max_flux_reconstruction_identity_t_per_m"],
        "max flux reconstruction identity",
    ) < 0:
        raise ValueError(f"{expected_name} flux reconstruction identity must be non-negative")
    requested = _finite_number(diagnostics["requested_signed_ampere_turns_a"], "requested current")
    sampled = _finite_number(diagnostics["sampled_signed_ampere_turns_a"], "sampled current")
    balance = _finite_number(diagnostics["ampere_turn_balance_error_a"], "current balance")
    if balance != sampled - requested:
        raise ValueError(f"{expected_name} ampere-turn balance is inconsistent")
    for artifact_key, manifest_key in (
        ("iterations", "iterations"),
        ("relative_residual_l2", "relative_residual_l2"),
    ):
        if diagnostics.get(artifact_key) != manifest_entry.get(manifest_key):
            raise ValueError(f"{expected_name} manifest {manifest_key} does not match artifact")
    source_diagnostics = diagnostics["source_discretization"]
    if not isinstance(source_diagnostics, list) or len(source_diagnostics) != len(sources):
        raise ValueError(f"{expected_name} source discretization count is inconsistent")
    source_diag_keys = {
        "name", "requested_area_m2", "represented_overlap_area_m2", "area_error_m2",
        "requested_centroid_r_m", "represented_centroid_r_m", "centroid_r_error_m",
        "requested_centroid_z_m", "represented_centroid_z_m", "centroid_z_error_m",
        "requested_signed_ampere_turns_a", "represented_signed_ampere_turns_a",
        "ampere_turn_error_a", "radial_nodes_touched", "axial_nodes_touched",
        "dual_cells_touched",
    }
    for index, item in enumerate(source_diagnostics):
        item = _closed(item, f"source_discretization[{index}]", source_diag_keys)
        if item["name"] != sources[index]["name"]:
            raise ValueError(f"{expected_name} source discretization order/name is inconsistent")
        for key in source_diag_keys - {
            "name", "radial_nodes_touched", "axial_nodes_touched", "dual_cells_touched"
        }:
            _finite_number(item[key], f"source diagnostic {key}")
        for key in ("radial_nodes_touched", "axial_nodes_touched", "dual_cells_touched"):
            _integer(item[key], f"source diagnostic {key}", 1)
        expected_area = (sources[index]["r_outer_m"] - sources[index]["r_inner_m"]) * (
            sources[index]["z_max_m"] - sources[index]["z_min_m"]
        )
        expected_current = sources[index]["polarity"] * sources[index]["ampere_turns_a"]
        if (
            item["requested_area_m2"] != expected_area
            or item["requested_signed_ampere_turns_a"] != expected_current
            or item["area_error_m2"]
            != item["represented_overlap_area_m2"] - item["requested_area_m2"]
            or item["ampere_turn_error_a"]
            != item["represented_signed_ampere_turns_a"]
            - item["requested_signed_ampere_turns_a"]
        ):
            raise ValueError(f"{expected_name} source discretization diagnostics are inconsistent")

    profiles = _closed(
        artifact["profiles"], f"{expected_name}.profiles", {"centreline", "wall"}
    )
    expected_profile_z = [z_min + index * dz for index in range(axial_intervals + 1)]
    _validate_profile(
        profiles["centreline"],
        f"{expected_name}.centreline",
        expected_keys={"r_m", "z_m", "b_r_t", "b_z_t"},
        expected_z=expected_profile_z,
    )
    if profiles["centreline"]["r_m"] != 0.0:
        raise ValueError(f"{expected_name} centreline radius must be zero")
    _validate_profile(
        profiles["wall"],
        f"{expected_name}.wall",
        expected_keys={"requested_r_m", "sampled_r_m", "z_m", "b_r_t", "b_z_t"},
        expected_z=expected_profile_z,
    )
    for key in ("requested_r_m", "sampled_r_m"):
        wall_radius = _finite_number(profiles["wall"][key], f"wall {key}")
        if not 0 <= wall_radius <= radius:
            raise ValueError(f"{expected_name} wall radius lies outside the domain")

    summary = _closed(
        artifact["summary"],
        f"{expected_name}.summary",
        {
            "b_magnitude_max_t", "b_magnitude_min_t",
            "outer_boundary_b_magnitude_min_t",
            "signed_requested_ampere_turns_a", "topology",
        },
    )
    for key in ("b_magnitude_max_t", "b_magnitude_min_t", "topology"):
        if summary[key] != manifest_entry.get(key):
            raise ValueError(f"{expected_name} manifest {key} does not match artifact")
    minimum = _finite_number(summary["b_magnitude_min_t"], "field minimum")
    maximum = _finite_number(summary["b_magnitude_max_t"], "field maximum")
    boundary_minimum = _finite_number(
        summary["outer_boundary_b_magnitude_min_t"], "outer boundary field minimum"
    )
    if not 0 <= minimum <= maximum or boundary_minimum < 0:
        raise ValueError(f"{expected_name} summary field range is invalid")
    expected_signed_current = fsum(
        source["polarity"] * source["ampere_turns_a"] for source in sources
    )
    if (
        summary["signed_requested_ampere_turns_a"] != expected_signed_current
        or requested != expected_signed_current
    ):
        raise ValueError(f"{expected_name} requested current is inconsistent")
    centreline = profiles["centreline"]
    wall = profiles["wall"]
    topology = _validate_topology(summary["topology"], f"{expected_name}.topology")
    if topology["field_scale_t"] != maximum:
        raise ValueError(f"{expected_name} topology field scale must equal |B| maximum")
    if topology != _axis_topology_values(
        centreline["z_m"], centreline["b_r_t"], centreline["b_z_t"], maximum
    ):
        raise ValueError(f"{expected_name} topology is inconsistent with centreline fields")
    flattened = [value for row in matrices["b_magnitude_t"] for value in row]
    observed = flattened + [
        hypot(br, bz)
        for profile in (centreline, wall)
        for br, bz in zip(profile["b_r_t"], profile["b_z_t"], strict=True)
    ]
    if max(observed) > maximum:
        raise ValueError(f"{expected_name} shown field exceeds the full-grid summary maximum")
    if min(observed) < minimum:
        raise ValueError(f"{expected_name} shown field is below the full-grid summary minimum")
    limitations = artifact["limitations"]
    if not isinstance(limitations, list) or not limitations:
        raise ValueError(f"{expected_name} limitations must be a non-empty array")
    for index, limitation in enumerate(limitations):
        _text(limitation, f"{expected_name}.limitations[{index}]")


def build_payload(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load, identity-check, and validate the manifest and all three artifacts."""

    manifest_path = manifest_path.resolve()
    default_manifest = manifest_path == DEFAULT_MANIFEST.resolve()
    manifest_digest = _verify_file(
        manifest_path,
        "manifest",
        EXPECTED_MANIFEST_FILE_SHA256 if default_manifest else None,
    )
    manifest = _load_object(manifest_path, "manifest")
    manifest_payload_digest = _verify_integrity(manifest, "manifest")
    if default_manifest and manifest_payload_digest != EXPECTED_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("manifest payload SHA-256 does not match the reviewed solver manifest")
    if set(manifest) != {
        "designs", "integrity", "model_level", "runtime_policy", "schema_version"
    }:
        raise ValueError("manifest top-level keys do not match the contract")
    if manifest["schema_version"] != "cft-axisymmetric-design-manifest/1.1.0":
        raise ValueError("manifest schema_version is unsupported or superseded")
    if manifest["model_level"] != "L1a":
        raise ValueError("manifest model_level must be L1a")
    _text(manifest["runtime_policy"], "manifest.runtime_policy")
    entries = manifest["designs"]
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_DESIGNS):
        raise ValueError("manifest must contain exactly the three reviewed designs")

    payload_designs: list[dict[str, Any]] = []
    base = manifest_path.parent.resolve()
    for entry, expected in zip(entries, EXPECTED_DESIGNS, strict=True):
        name, filename, expected_file_digest, expected_payload_digest, label = expected
        entry = _closed(
            entry,
            f"manifest entry {name}",
            {
                "artifact", "artifact_file_sha256", "artifact_payload_sha256",
                "b_magnitude_max_t", "b_magnitude_min_t", "backend", "iterations",
                "name", "relative_residual_l2", "topology",
            },
        )
        if entry.get("name") != name or entry.get("artifact") != filename:
            raise ValueError("manifest design identity/order does not match reviewed artifacts")
        for key in ("artifact_file_sha256", "artifact_payload_sha256"):
            digest = entry[key]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} manifest {key} is not a SHA-256 digest")
        if default_manifest and (
            entry["artifact_file_sha256"] != expected_file_digest
            or entry["artifact_payload_sha256"] != expected_payload_digest
        ):
            raise ValueError(f"{name} manifest artifact anchors are superseded")
        _text(entry["backend"], f"{name} manifest backend")
        _integer(entry["iterations"], f"{name} manifest iterations")
        relative = _finite_number(
            entry["relative_residual_l2"], f"{name} manifest relative residual"
        )
        minimum = _finite_number(
            entry["b_magnitude_min_t"], f"{name} manifest field minimum"
        )
        maximum = _finite_number(
            entry["b_magnitude_max_t"], f"{name} manifest field maximum"
        )
        if relative < 0 or not 0 <= minimum <= maximum:
            raise ValueError(f"{name} manifest numerical range is invalid")
        topology = _validate_topology(entry["topology"], f"{name} manifest topology")
        if topology["field_scale_t"] != maximum:
            raise ValueError(f"{name} manifest topology scale does not match maximum")
        artifact_path = (base / filename).resolve()
        if Path(filename).name != filename or artifact_path.parent != base:
            raise ValueError("artifact path escapes the manifest directory")
        file_digest = _verify_file(
            artifact_path, f"{name} artifact", entry["artifact_file_sha256"]
        )
        artifact = _load_object(artifact_path, f"{name} artifact")
        payload_digest = _verify_integrity(artifact, f"{name} artifact")
        if payload_digest != entry["artifact_payload_sha256"]:
            raise ValueError(f"{name} artifact payload SHA-256 does not match manifest anchor")
        _validate_artifact(artifact, entry, name)
        field = artifact["field_map"]
        maximum = max(value for row in field["b_magnitude_t"] for value in row)
        max_locations: list[dict[str, float]] = []
        for i, row in enumerate(field["b_magnitude_t"]):
            for j, value in enumerate(row):
                if isclose(float(value), maximum, rel_tol=2e-14, abs_tol=1e-16):
                    max_locations.append({"r_m": field["r_m"][i], "z_m": field["z_m"][j]})
        payload_designs.append(
            {
                "id": name,
                "label": label,
                "artifact": filename,
                "file_sha256": file_digest,
                "payload_sha256": payload_digest,
                "input": artifact["input"],
                "summary": artifact["summary"],
                "diagnostics": artifact["diagnostics"],
                "profiles": artifact["profiles"],
                "field": field,
                "provenance": artifact["provenance"],
                "limitations": artifact["limitations"],
                "model_description": artifact["model_description"],
                "max_locations": max_locations,
                "shown_grid_max_t": maximum,
                "parity": {
                    "accepted_artifact_evidence": False,
                    "artifact_statement": (
                        "Manifest records Python artifact provenance and no accepted "
                        "CPU/CUDA artifact-parity evidence."
                    ),
                    "runtime_statement": (
                        "Runtime parity tests are separate verification-suite evidence; "
                        "they do not change this artifact's Python provenance."
                    ),
                },
            }
        )
    payload = {
        "schema": "cft-axisymmetric-visualization/1.1.0",
        "manifest": {
            "file": manifest_path.name,
            "file_sha256": manifest_digest,
            "payload_sha256": manifest_payload_digest,
            "schema_version": manifest["schema_version"],
            "runtime_policy": manifest["runtime_policy"],
        },
        "warning": (
            "Linear-vacuum equivalent-current L1a evidence only—not permanent magnets, "
            "nonlinear iron, plasma, experimental validation, or a validated design."
        ),
        "designs": payload_designs,
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Validate the visualization-specific envelope and embedded identities."""

    _closed(payload, "visualization payload", {"schema", "manifest", "warning", "designs"})
    if payload.get("schema") != "cft-axisymmetric-visualization/1.1.0":
        raise ValueError("visualization payload schema is unsupported")
    manifest = _closed(
        payload.get("manifest"),
        "visualization manifest identity",
        {"file", "file_sha256", "payload_sha256", "schema_version", "runtime_policy"},
    )
    if (
        manifest["file_sha256"] != EXPECTED_MANIFEST_FILE_SHA256
        or manifest["payload_sha256"] != EXPECTED_MANIFEST_PAYLOAD_SHA256
        or manifest["schema_version"] != "cft-axisymmetric-design-manifest/1.1.0"
    ):
        raise ValueError("embedded manifest identity is invalid or superseded")
    designs = payload.get("designs")
    if not isinstance(designs, list) or [item.get("id") for item in designs] != [
        item[0] for item in EXPECTED_DESIGNS
    ]:
        raise ValueError("visualization payload design identities/order are invalid")
    for design, expected in zip(designs, EXPECTED_DESIGNS, strict=True):
        _closed(
            design,
            f"visualization design {expected[0]}",
            {
                "id", "label", "artifact", "file_sha256", "payload_sha256", "input",
                "summary", "diagnostics", "profiles", "field", "provenance",
                "limitations", "model_description", "max_locations",
                "shown_grid_max_t", "parity",
            },
        )
        if (
            design.get("file_sha256") != expected[2]
            or design.get("payload_sha256") != expected[3]
        ):
            raise ValueError(f"{expected[0]} embedded artifact identity is invalid")
        parity = _closed(
            design.get("parity"),
            f"{expected[0]} parity",
            {"accepted_artifact_evidence", "artifact_statement", "runtime_statement"},
        )
        if parity["accepted_artifact_evidence"] is not False:
            raise ValueError(f"{expected[0]} must not claim unrecorded artifact parity")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>L1a Axisymmetric Field Results</title>
<style>
:root{color-scheme:dark;--bg:#071018;--panel:#101c27;--panel2:#142433;--text:#eef6fb;--muted:#9bb0bf;--line:#2c4353;--accent:#55d6be;--warn:#ffcf67;--red:#ff6b6b;--blue:#55a8ff;--shadow:#0008}
[data-theme=light]{color-scheme:light;--bg:#edf4f7;--panel:#fff;--panel2:#f3f8fa;--text:#10222c;--muted:#536b78;--line:#bfd0d8;--accent:#087f72;--warn:#805b00;--red:#b83232;--blue:#176db5;--shadow:#3452}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#15344b 0,transparent 34rem),var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
button,select{font:inherit;color:var(--text);background:var(--panel2);border:1px solid var(--line);border-radius:.55rem;padding:.48rem .7rem}button:hover,select:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,canvas:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
header,main,footer{width:min(1500px,calc(100% - 2rem));margin:auto}header{padding:2.2rem 0 1rem}.eyebrow{color:var(--accent);font-weight:750;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(2rem,5vw,4.4rem);line-height:.96;margin:.2rem 0 1rem;max-width:900px}h2{margin:.1rem 0 1rem;font-size:1.12rem}p{margin:.35rem 0}.warning{border:1px solid #8b681c;background:#513d1438;color:var(--warn);padding:.8rem 1rem;border-radius:.65rem;font-weight:650}.controls{display:flex;gap:.6rem;flex-wrap:wrap;align-items:end;margin:1rem 0}.control{display:grid;gap:.25rem}.control label{color:var(--muted);font-size:.8rem}
.grid{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(280px,.8fr);gap:1rem;margin:1rem 0}.panel{background:linear-gradient(145deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:.9rem;padding:1rem;box-shadow:0 12px 30px var(--shadow);min-width:0}.canvas-wrap{position:relative;min-height:360px}.canvas-wrap canvas{width:100%;height:clamp(360px,52vw,690px);display:block}.tip{position:absolute;pointer-events:none;background:#071018ee;color:#fff;border:1px solid #7f98aa;border-radius:.35rem;padding:.35rem .5rem;display:none;white-space:nowrap}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:.65rem}.metric-card{border:1px solid var(--line);border-radius:.7rem;padding:.75rem;background:var(--panel);min-width:0}.metric-card.active{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}.metric-card h3{font-size:.95rem;margin:0 0 .55rem}.kv{display:grid;grid-template-columns:1fr auto;gap:.22rem .6rem}.kv span:nth-child(odd){color:var(--muted)}.kv span:nth-child(even){font-variant-numeric:tabular-nums;text-align:right}.source{border-left:4px solid var(--blue);padding:.35rem .55rem;margin:.45rem 0;background:#55a8ff12}.source.neg{border-color:var(--red)}.topology{border:1px solid var(--line);border-radius:.55rem;padding:.6rem;background:#ffcf670d}.limits{border-top:1px solid var(--line);padding-top:.55rem;margin-top:.7rem}code{font-size:.78em;overflow-wrap:anywhere}.small{font-size:.82rem;color:var(--muted)}.profiles{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.plot{width:100%;height:280px;display:block}.wide{grid-column:1/-1}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:.1rem .5rem;margin:.1rem .2rem .1rem 0;color:var(--muted)}footer{padding:1rem 0 2.5rem;color:var(--muted)}
@media(max-width:900px){.grid,.profiles{grid-template-columns:1fr}.metrics{grid-template-columns:1fr}.canvas-wrap canvas{height:520px}}@media(max-width:520px){header,main,footer{width:min(100% - 1rem,1500px)}.canvas-wrap canvas{height:420px}.panel{padding:.7rem}}@media(prefers-reduced-motion:no-preference){.panel{transition:border-color .15s}}
</style>
</head>
<body>
<header><div class="eyebrow">Solver artifact viewer · L1a</div><h1>Axisymmetric field evidence</h1><p id="warning" class="warning"></p>
<div class="controls">
<div class="control"><label for="design">Field design</label><select id="design"></select></div>
<div class="control"><label for="component">Heatmap component</label><select id="component"><option value="b_magnitude_t">|B| magnitude</option><option value="b_r_t">Br signed</option><option value="b_z_t">Bz signed</option></select></div>
<button id="reset" type="button">Reset view</button><button id="theme" type="button" aria-pressed="false">Light theme</button>
</div><p class="small">Keyboard: 1–3 select designs; arrow keys move the map cursor; Home resets the cursor.</p></header>
<main>
<section class="metrics" id="metrics" aria-label="Side-by-side design metrics"></section>
<section class="grid">
<div class="panel"><h2>Field map with ψ-derived flux contours</h2><div class="canvas-wrap"><canvas id="field" tabindex="0" role="img" aria-label="Interactive axisymmetric field heatmap"></canvas><div id="tip" class="tip" role="status" aria-live="polite"></div></div><p class="small">Canvas raster uses the artifact’s radial-major grid. Lines are marching-squares isolines computed directly from ψ (Wb). Source rectangles show actual (z,r) band bounds; blue is + polarity and red is −.</p><p id="mapLimits" class="small limits"></p></div>
<aside class="panel"><h2 id="detailTitle">Design details</h2><div id="details"></div></aside>
</section>
<section class="profiles">
<div class="panel"><h2>Centreline profile</h2><canvas class="plot" id="centre" role="img" aria-label="Centreline magnetic field profile"></canvas><p id="centreLimits" class="small limits"></p></div>
<div class="panel"><h2>Wall profile</h2><canvas class="plot" id="wall" role="img" aria-label="Wall magnetic field profile"></canvas><p id="wallLimits" class="small limits"></p></div>
<div class="panel wide"><h2>Convergence history</h2><canvas class="plot" id="residual" role="img" aria-label="Log-scale solver residual history"></canvas><p class="small">History samples use the declared residual stride; the final point is placed at the reported final iteration. Flux-reconstruction identity is an internal reconstruction check, not independent divergence or PDE validation.</p><p id="residualLimits" class="small limits"></p></div>
</section>
<section class="panel" style="margin:1rem 0"><h2>Artifact identity and parity status</h2><div id="identity"></div></section>
</main><footer>Self-contained offline visualization. Timings are intentionally not presented as benchmark evidence.</footer>
<script id="axisymmetric-data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA=JSON.parse(document.getElementById("axisymmetric-data").textContent);
const $=id=>document.getElementById(id);let selected=0,component="b_magnitude_t",cursor=null,raf=0;
const designSelect=$("design");DATA.designs.forEach((d,i)=>{const o=document.createElement("option");o.value=i;o.textContent=d.label;designSelect.append(o)});
$("warning").textContent=DATA.warning;
const fmt=(v,n=4)=>Number(v).toLocaleString(undefined,{maximumSignificantDigits:n});
function fieldMaxLocation(d){return d.max_locations.map(p=>`r ${fmt(p.r_m)} m, z ${fmt(p.z_m)} m`).join("; ")}
const topologyNames={degenerate_near_zero_field:"Degenerate near-zero field",resolved_axis_nulls:"Resolved axis nulls",near_zero_axis_plateau:"Near-zero axis plateau",no_resolved_axis_null:"No resolved interior axis null"};
const nullNames={sign_changing_sample:"sign-changing sampled null",sign_changing_interpolated:"sign-changing interpolated null",isolated_sample:"isolated sampled minimum"};
function renderMetrics(){const root=$("metrics");root.textContent="";DATA.designs.forEach((d,i)=>{const c=document.createElement("article");c.className="metric-card"+(i===selected?" active":"");c.tabIndex=0;c.setAttribute("role","button");c.setAttribute("aria-pressed",i===selected);c.setAttribute("aria-label",`Select ${d.label}`);c.innerHTML=`<h3>${d.label}</h3><div class="kv"><span>max |B|</span><span>${fmt(d.summary.b_magnitude_max_t*1e3)} mT</span><span>topology</span><span>${topologyNames[d.summary.topology.status]}</span><span>axis nulls</span><span>${d.summary.topology.axis_nulls.length}</span><span>iterations</span><span>${d.diagnostics.iterations}</span><span>rel. residual</span><span>${d.diagnostics.relative_residual_l2.toExponential(2)}</span></div>`;c.onclick=()=>select(i);c.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();select(i)}};root.append(c)})}
function renderDetails(){const d=DATA.designs[selected],dom=d.input.domain,top=d.summary.topology;let html=`<p>${d.model_description}</p><div class="kv"><span>Domain r</span><span>0–${fmt(dom.radius_m)} m</span><span>Domain z</span><span>${fmt(dom.z_min_m)}–${fmt(dom.z_max_m)} m</span><span>Solver grid</span><span>${dom.radial_intervals+1} × ${dom.axial_intervals+1}</span><span>Grid spacing</span><span>${fmt(dom.dr_m)} × ${fmt(dom.dz_m)} m</span><span>Shown grid</span><span>${d.field.r_m.length} × ${d.field.z_m.length}</span><span>Downsample stride</span><span>${d.field.downsample_stride}</span><span>Artifact backend</span><span>${d.provenance.backend}</span><span>Full-grid max |B|</span><span>${fmt(d.summary.b_magnitude_max_t*1e3)} mT</span><span>Shown-grid max</span><span>${fmt(d.shown_grid_max_t*1e3)} mT at ${fieldMaxLocation(d)}</span><span>Outer-boundary min</span><span>${fmt(d.summary.outer_boundary_b_magnitude_min_t*1e3)} mT</span><span>Flux reconstruction identity</span><span>${d.diagnostics.max_flux_reconstruction_identity_t_per_m.toExponential(2)} T/m</span><span>True-residual restarts</span><span>${d.diagnostics.true_residual_restarts}</span></div><h2 style="margin-top:1rem">Equivalent-current source bands</h2>`;
d.input.sources.forEach(s=>{html+=`<div class="source ${s.polarity<0?"neg":""}"><strong>${s.name}: ${s.polarity>0?"+":"−"} polarity</strong><br>${fmt(s.ampere_turns_a)} A-turn · r ${fmt(s.r_inner_m)}–${fmt(s.r_outer_m)} m · z ${fmt(s.z_min_m)}–${fmt(s.z_max_m)} m</div>`});
const nulls=top.axis_nulls.length?top.axis_nulls.map(n=>`${nullNames[n.kind]} at z ${fmt(n.z_m)} m`).join("<br>"):"none classified";const plateaus=top.axis_plateaus.length?top.axis_plateaus.map(p=>`z ${fmt(p.z_start_m)}–${fmt(p.z_end_m)} m (${p.sample_count} samples)`).join("<br>"):"none classified";
html+=`<h2 style="margin-top:1rem">Topology classification</h2><div class="topology"><strong>${topologyNames[top.status]}</strong><br>Axis nulls: ${nulls}<br>Near-zero plateaus: ${plateaus}<br>Null tolerance: ${top.null_tolerance_t.toExponential(2)} T</div><p class="small">The reported ${fmt(d.summary.outer_boundary_b_magnitude_min_t*1e3)} mT outer-boundary minimum is a finite-box boundary sample, not an interior physical null.</p><p class="small"><strong>Artifact provenance:</strong> ${d.parity.artifact_statement}<br><strong>Runtime tests:</strong> ${d.parity.runtime_statement}</p>`;$("detailTitle").textContent=d.label;$("details").innerHTML=html;
const limitations="Limitations: "+d.limitations.join(" · ");for(const id of ["mapLimits","centreLimits","wallLimits","residualLimits"])$(id).textContent=limitations;
$("identity").innerHTML=`<p><span class="badge">manifest file SHA-256</span> <code>${DATA.manifest.file_sha256}</code></p><p><span class="badge">manifest payload SHA-256</span> <code>${DATA.manifest.payload_sha256}</code></p><p><span class="badge">artifact file SHA-256</span> <code>${d.file_sha256}</code></p><p><span class="badge">artifact payload SHA-256</span> <code>${d.payload_sha256}</code></p><p><span class="badge">schema</span> ${DATA.manifest.schema_version}</p><p><span class="badge">implementation</span> ${d.provenance.implementation}</p><p><span class="badge">scalar</span> ${d.provenance.scalar}</p><p><span class="badge">Accepted CPU/CUDA artifact parity</span> <strong>not recorded</strong> — ${d.parity.artifact_statement}</p><p><span class="badge">Runtime parity tests</span> ${d.parity.runtime_statement}</p>`}
function setup(canvas){const r=canvas.getBoundingClientRect(),dpr=Math.max(1,window.devicePixelRatio||1),w=Math.max(1,Math.round(r.width*dpr)),h=Math.max(1,Math.round(r.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}const c=canvas.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);return {c,w:r.width,h:r.height,dpr}}
const themeColor=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function color(v,lo,hi,signed){let t=(v-lo)/(hi-lo||1);t=Math.max(0,Math.min(1,t));if(signed){if(t<.5){const q=t*2;return `rgb(${Math.round(35+220*q)},${Math.round(92+163*q)},255)`}const q=(t-.5)*2;return `rgb(255,${Math.round(255-210*q)},${Math.round(255-215*q)})`}return `rgb(${Math.round(12+240*t)},${Math.round(28+190*Math.sqrt(t))},${Math.round(90+100*(1-t))})`}
function bounds(w,h){return {l:58,t:22,r:w-72,b:h-48}}
function mapPoint(z,r,d,b){const zs=d.field.z_m,rs=d.field.r_m;return [b.l+(z-zs[0])/(zs.at(-1)-zs[0])*(b.r-b.l),b.b-(r-rs[0])/(rs.at(-1)-rs[0])*(b.b-b.t)]}
function contourSegments(psi,rs,zs,level){const seg=[];for(let i=0;i<rs.length-1;i++)for(let j=0;j<zs.length-1;j++){const p=[psi[i][j],psi[i][j+1],psi[i+1][j+1],psi[i+1][j]],pts=[[zs[j],rs[i]],[zs[j+1],rs[i]],[zs[j+1],rs[i+1]],[zs[j],rs[i+1]]],hits=[];for(let e=0;e<4;e++){const a=p[e],bb=p[(e+1)%4];if((a<level&&bb>=level)||(bb<level&&a>=level)){const q=(level-a)/(bb-a);hits.push([pts[e][0]+q*(pts[(e+1)%4][0]-pts[e][0]),pts[e][1]+q*(pts[(e+1)%4][1]-pts[e][1])])}}if(hits.length===2)seg.push([hits[0],hits[1]]);else if(hits.length===4){seg.push([hits[0],hits[1]],[hits[2],hits[3]])}}return seg}
function drawField(){const d=DATA.designs[selected],s=setup($("field")),c=s.c,b=bounds(s.w,s.h),m=d.field[component],flat=m.flat(),signed=component!=="b_magnitude_t",ma=Math.max(...flat.map(Math.abs)),lo=signed?-ma:0,hi=ma;c.clearRect(0,0,s.w,s.h);c.fillStyle=themeColor("--panel");c.fillRect(0,0,s.w,s.h);
const off=document.createElement("canvas");off.width=d.field.z_m.length;off.height=d.field.r_m.length;const oc=off.getContext("2d"),img=oc.createImageData(off.width,off.height);for(let i=0;i<off.height;i++)for(let j=0;j<off.width;j++){const rgb=color(m[off.height-1-i][j],lo,hi,signed).match(/\d+/g).map(Number),k=(i*off.width+j)*4;img.data[k]=rgb[0];img.data[k+1]=rgb[1];img.data[k+2]=rgb[2];img.data[k+3]=255}oc.putImageData(img,0,0);c.imageSmoothingEnabled=true;c.drawImage(off,b.l,b.t,b.r-b.l,b.b-b.t);
const psi=d.field.psi_wb,ps=psi.flat(),pmin=Math.min(...ps),pmax=Math.max(...ps);c.strokeStyle="#ffffffb8";c.lineWidth=1;for(let n=1;n<=11;n++){const lev=pmin+(pmax-pmin)*n/12;c.beginPath();for(const line of contourSegments(psi,d.field.r_m,d.field.z_m,lev)){const a=mapPoint(...line[0],d,b),z=mapPoint(...line[1],d,b);c.moveTo(...a);c.lineTo(...z)}c.stroke()}
d.input.sources.forEach(src=>{const a=mapPoint(src.z_min_m,src.r_inner_m,d,b),q=mapPoint(src.z_max_m,src.r_outer_m,d,b);c.fillStyle=src.polarity>0?"#55a8ff66":"#ff6b6b66";c.strokeStyle=src.polarity>0?"#8ec5ff":"#ff9a9a";c.lineWidth=2;c.fillRect(a[0],q[1],q[0]-a[0],a[1]-q[1]);c.strokeRect(a[0],q[1],q[0]-a[0],a[1]-q[1])});
d.summary.topology.axis_plateaus.forEach(p=>{const a=mapPoint(p.z_start_m,0,d,b),q=mapPoint(p.z_end_m,0,d,b);c.fillStyle="#ffcf6755";c.fillRect(a[0],b.b-8,Math.max(2,q[0]-a[0]),8)});
d.summary.topology.axis_nulls.forEach(n=>{const p=mapPoint(n.z_m,n.r_m,d,b);c.strokeStyle="#ffcf67";c.lineWidth=2;c.beginPath();if(n.kind==="isolated_sample"){c.moveTo(p[0],p[1]-7);c.lineTo(p[0]+7,p[1]);c.lineTo(p[0],p[1]+7);c.lineTo(p[0]-7,p[1]);c.closePath()}else{c.arc(p[0],p[1],6,0,Math.PI*2);c.moveTo(p[0]-9,p[1]);c.lineTo(p[0]+9,p[1])}c.stroke()});
axes(c,b,s.w,s.h,"z (m)","r (m)",d.field.z_m[0],d.field.z_m.at(-1),d.field.r_m[0],d.field.r_m.at(-1));const x=s.w-48;c.font="11px system-ui";for(let k=0;k<80;k++){c.fillStyle=color(lo+(hi-lo)*(1-k/79),lo,hi,signed);c.fillRect(x,b.t+k*(b.b-b.t)/80,15,(b.b-b.t)/80+1)}c.fillStyle=themeColor("--text");c.fillText(`${fmt(hi*1e3)} mT`,x-10,b.t-7);c.fillText(`${fmt(lo*1e3)} mT`,x-10,b.b+15);
if(cursor){const p=mapPoint(cursor.z,cursor.r,d,b);c.strokeStyle="#fff";c.lineWidth=1;c.beginPath();c.moveTo(p[0]-8,p[1]);c.lineTo(p[0]+8,p[1]);c.moveTo(p[0],p[1]-8);c.lineTo(p[0],p[1]+8);c.stroke()}}
function axes(c,b,w,h,xlabel,ylabel,xmin,xmax,ymin,ymax){c.strokeStyle=themeColor("--line");c.fillStyle=themeColor("--muted");c.lineWidth=1;c.font="12px system-ui";c.strokeRect(b.l,b.t,b.r-b.l,b.b-b.t);c.textAlign="center";for(let i=0;i<=4;i++){const x=b.l+(b.r-b.l)*i/4;c.fillText(fmt(xmin+(xmax-xmin)*i/4,3),x,b.b+19)}c.fillText(xlabel,(b.l+b.r)/2,h-8);c.save();c.translate(14,(b.t+b.b)/2);c.rotate(-Math.PI/2);c.fillText(ylabel,0,0);c.restore();c.textAlign="right";for(let i=0;i<=4;i++)c.fillText(fmt(ymax-(ymax-ymin)*i/4,3),b.l-7,b.t+(b.b-b.t)*i/4+4);c.textAlign="left"}
function nearest(values,v){let k=0;for(let i=1;i<values.length;i++)if(Math.abs(values[i]-v)<Math.abs(values[k]-v))k=i;return k}
function updateCursor(clientX,clientY){const canvas=$("field"),rect=canvas.getBoundingClientRect(),b=bounds(rect.width,rect.height),d=DATA.designs[selected],x=Math.max(b.l,Math.min(b.r,clientX-rect.left)),y=Math.max(b.t,Math.min(b.b,clientY-rect.top)),zs=d.field.z_m,rs=d.field.r_m,zi=nearest(zs,zs[0]+(x-b.l)/(b.r-b.l)*(zs.at(-1)-zs[0])),ri=nearest(rs,rs[0]+(b.b-y)/(b.b-b.t)*(rs.at(-1)-rs[0]));cursor={zi,ri,z:zs[zi],r:rs[ri]};showTip(clientX-rect.left,clientY-rect.top);schedule(false)}
function showTip(x,y){const d=DATA.designs[selected],t=$("tip");if(!cursor){t.style.display="none";return}const f=d.field;t.textContent=`z ${fmt(cursor.z)} m · r ${fmt(cursor.r)} m · |B| ${fmt(f.b_magnitude_t[cursor.ri][cursor.zi]*1e3)} mT · Br ${fmt(f.b_r_t[cursor.ri][cursor.zi]*1e3)} mT · Bz ${fmt(f.b_z_t[cursor.ri][cursor.zi]*1e3)} mT`;t.style.display="block";t.style.left=Math.min(x+12,t.parentElement.clientWidth-t.offsetWidth-5)+"px";t.style.top=Math.max(4,y-36)+"px"}
function drawPlot(id,series,markers,yLabel,log=false){const s=setup($(id)),c=s.c,b={l:62,t:18,r:s.w-18,b:s.h-42},all=series.flatMap(q=>q.y),xmin=Math.min(...series.flatMap(q=>q.x)),xmax=Math.max(...series.flatMap(q=>q.x));let ymin=Math.min(...all),ymax=Math.max(...all);if(log){ymin=Math.log10(Math.max(ymin,1e-300));ymax=Math.log10(ymax)}else{const pad=(ymax-ymin||1)*.08;ymin-=pad;ymax+=pad}c.clearRect(0,0,s.w,s.h);c.fillStyle=themeColor("--panel");c.fillRect(0,0,s.w,s.h);axes(c,b,s.w,s.h,id==="residual"?"iteration":"z (m)",yLabel,xmin,xmax,ymin,ymax);series.forEach((q,k)=>{c.strokeStyle=q.color;c.lineWidth=2;c.beginPath();q.x.forEach((x,i)=>{const yy=log?Math.log10(Math.max(q.y[i],1e-300)):q.y[i],px=b.l+(x-xmin)/(xmax-xmin||1)*(b.r-b.l),py=b.b-(yy-ymin)/(ymax-ymin||1)*(b.b-b.t);i?c.lineTo(px,py):c.moveTo(px,py)});c.stroke();c.fillStyle=q.color;c.fillText(q.name,b.l+8,b.t+14+k*15)});markers.forEach(z=>{const x=b.l+(z-xmin)/(xmax-xmin||1)*(b.r-b.l);c.strokeStyle="#ffcf67";c.setLineDash([4,3]);c.beginPath();c.moveTo(x,b.t);c.lineTo(x,b.b);c.stroke();c.setLineDash([])})}
function drawProfiles(){const d=DATA.designs[selected],p=d.profiles,cz=p.centreline.z_m,wz=p.wall.z_m,mag=p.wall.b_r_t.map((v,i)=>Math.hypot(v,p.wall.b_z_t[i])),top=d.summary.topology,marks=top.axis_nulls.map(n=>n.z_m).concat(top.axis_plateaus.flatMap(p=>[p.z_start_m,p.z_end_m]));drawPlot("centre",[{x:cz,y:p.centreline.b_z_t.map(v=>v*1e3),name:"Bz",color:"#55d6be"}],marks,"Bz (mT)");drawPlot("wall",[{x:wz,y:p.wall.b_r_t.map(v=>v*1e3),name:"Br",color:"#55a8ff"},{x:wz,y:p.wall.b_z_t.map(v=>v*1e3),name:"Bz",color:"#ff6b6b"},{x:wz,y:mag.map(v=>v*1e3),name:"|B|",color:"#55d6be"}],marks,"field (mT)");const h=d.diagnostics.residual_history_l2,stride=d.input.solver.residual_history_stride,x=h.map((_,i)=>i===h.length-1?d.diagnostics.iterations:i*stride);drawPlot("residual",[{x,y:h,name:"L2 residual",color:"#55d6be"}],[],"log10 residual",true)}
function drawAll(){renderMetrics();renderDetails();drawField();drawProfiles()}
function schedule(full=true){cancelAnimationFrame(raf);raf=requestAnimationFrame(full?drawAll:drawField)}
function select(i){selected=i;designSelect.value=i;cursor=null;showTip();schedule()}
designSelect.onchange=()=>select(Number(designSelect.value));$("component").onchange=e=>{component=e.target.value;schedule()};$("reset").onclick=()=>{component="b_magnitude_t";$("component").value=component;select(0);$("field").focus()};
$("theme").onclick=()=>{const light=document.documentElement.dataset.theme!=="light";document.documentElement.dataset.theme=light?"light":"dark";$("theme").textContent=light?"Dark theme":"Light theme";$("theme").setAttribute("aria-pressed",light);schedule()};
$("field").addEventListener("pointermove",e=>updateCursor(e.clientX,e.clientY));$("field").addEventListener("pointerleave",()=>{cursor=null;showTip();schedule(false)});$("field").addEventListener("keydown",e=>{const d=DATA.designs[selected],zs=d.field.z_m,rs=d.field.r_m;if(e.key==="Home"){cursor={zi:Math.floor(zs.length/2),ri:0,z:zs[Math.floor(zs.length/2)],r:rs[0]}}else{if(!cursor)cursor={zi:Math.floor(zs.length/2),ri:0,z:zs[Math.floor(zs.length/2)],r:rs[0]};if(e.key==="ArrowLeft")cursor.zi=Math.max(0,cursor.zi-1);else if(e.key==="ArrowRight")cursor.zi=Math.min(zs.length-1,cursor.zi+1);else if(e.key==="ArrowDown")cursor.ri=Math.max(0,cursor.ri-1);else if(e.key==="ArrowUp")cursor.ri=Math.min(rs.length-1,cursor.ri+1);else return;cursor.z=zs[cursor.zi];cursor.r=rs[cursor.ri]}e.preventDefault();showTip(70,30);schedule(false)});
window.addEventListener("keydown",e=>{if(["INPUT","SELECT","BUTTON"].includes(e.target.tagName))return;if(["1","2","3"].includes(e.key))select(Number(e.key)-1)});new ResizeObserver(schedule).observe(document.querySelector("main"));window.addEventListener("pageshow",schedule);drawAll();
</script></body></html>
"""


def render_html(payload: Mapping[str, Any]) -> str:
    validate_payload(payload)
    encoded = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA__", encoded)


def generate(
    output_path: Path = DEFAULT_OUTPUT, manifest_path: Path = DEFAULT_MANIFEST
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(build_payload(manifest_path)), encoding="utf-8", newline="\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = generate(args.output, args.manifest)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
