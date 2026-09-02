"""Generate the deterministic offline P2 FEM qualification dashboard."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
FEM_ROOT = HERE.parent
THIRD_LEVEL = FEM_ROOT / "artifacts" / "third-level"
TEMPLATE_PATH = HERE / "dashboard.template.html"
DEFAULT_OUTPUT = HERE / "fem-reference-p2-qualification.html"
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
CLASSIFICATION = "independent_numerical_reference_not_hardware_validation"
ZERO_HASH = "0" * 64
GRID_WIDTH = 144
GRID_HEIGHT = 92
PROFILE_SAMPLES = 160
MESH_TRIANGLE_BUDGET = 1800

EVIDENCE = (
    (
        "historical-envelope-baseline",
        "Historical envelope",
        "a33e689bd437c55b473bd6eae1c3b4ef2d7843a2808108e54e805e1c4be1f71d",
        False,
    ),
    (
        "compact-high-gradient-stack",
        "Compact high-gradient",
        "a17aa3215a51f2b260c8113b105ca950884554390abb497f1faa9e9c5e5d2286",
        False,
    ),
    (
        "divergent-exit-stack",
        "Divergent exit",
        "0defabb5bf2aa7750bc4a39ce3392fcd6b23ef22470b4560bc0e37d37bb03da1",
        True,
    ),
)


class _HashWriter:
    def __init__(self) -> None:
        self.digest = sha256()

    def write(self, value: str) -> int:
        encoded = value.encode("utf-8")
        self.digest.update(encoded)
        return len(value)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError(f"{label} contains nonfinite constant {value!r}")

    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(
                source,
                object_pairs_hook=unique,
                parse_constant=reject,
            )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _stream_hash(path: Path, label: str, expected: str | None = None) -> str:
    digest = sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024**2), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"{label} is not readable") from error
    actual = digest.hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f"{label} file SHA-256 mismatch")
    return actual


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase hexadecimal SHA-256")
    return value


def _canonical_hash(value: Mapping[str, Any]) -> str:
    writer = _HashWriter()
    json.dump(
        value,
        writer,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return writer.digest.hexdigest()


def _verify_integrity(value: dict[str, Any], label: str) -> str:
    integrity = value.get("integrity")
    if not isinstance(integrity, dict) or set(integrity) != {
        "algorithm",
        "canonicalization",
        "payload_sha256",
    }:
        raise ValueError(f"{label} integrity contract differs")
    if (
        integrity["algorithm"] != "sha256"
        or integrity["canonicalization"] != CANONICALIZATION
    ):
        raise ValueError(f"{label} integrity algorithm differs")
    expected = _digest(integrity["payload_sha256"], f"{label} payload SHA-256")
    payload = {key: item for key, item in value.items() if key != "integrity"}
    if _canonical_hash(payload) != expected:
        raise ValueError(f"{label} canonical payload SHA-256 mismatch")
    return expected


def _safe_path(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or ":" in raw:
        raise ValueError(f"{label} is not a relative evidence path")
    normalized = raw.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"{label} escapes the evidence directory")
    path = root.joinpath(*pure.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the evidence directory")
    return path


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _rounded(values: np.ndarray, digits: int = 9) -> list[float]:
    return [float(f"{float(value):.{digits}g}") for value in values]


def _mesh_projection(
    sidecar_path: Path,
    expected_hash: str,
    label: str,
) -> dict[str, Any]:
    _stream_hash(sidecar_path, f"{label} array sidecar", expected_hash)
    try:
        with np.load(sidecar_path, allow_pickle=False) as archive:
            points = np.asarray(archive["mesh.vertices_rz_m"], dtype=np.float64)
            triangles = np.asarray(archive["mesh.triangles"], dtype=np.int64)
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"{label} mesh arrays are invalid") from error
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or len(points) < 3
        or len(triangles) < 1
        or not np.isfinite(points).all()
        or int(triangles.min()) < 0
        or int(triangles.max()) >= len(points)
    ):
        raise ValueError(f"{label} mesh array shape or range differs")
    count = min(MESH_TRIANGLE_BUDGET, len(triangles))
    indices = np.unique(np.linspace(0, len(triangles) - 1, count, dtype=np.int64))
    selected = points[triangles[indices]]
    r_min, z_min = points.min(axis=0)
    r_max, z_max = points.max(axis=0)
    scale = np.array(
        [max(float(r_max - r_min), 1.0e-300), max(float(z_max - z_min), 1.0e-300)]
    )
    normalized = (selected - np.array([r_min, z_min])) / scale
    return {
        "sampled_triangles": int(len(indices)),
        "source_triangles": int(len(triangles)),
        "extent_rz_m": _rounded(np.array([r_min, r_max, z_min, z_max]), 12),
        "triangles": [
            _rounded(triangle.reshape(-1), 7) for triangle in normalized
        ],
    }


def _fill_grid(values: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    grid = values.reshape(GRID_HEIGHT, GRID_WIDTH).copy()
    mask = occupied.reshape(GRID_HEIGHT, GRID_WIDTH).copy()
    if mask.all():
        return grid
    for _ in range(max(GRID_WIDTH, GRID_HEIGHT)):
        if mask.all():
            break
        total = np.zeros_like(grid)
        count = np.zeros_like(grid, dtype=np.int16)
        for source, source_mask in (
            (np.roll(grid, 1, axis=0), np.roll(mask, 1, axis=0)),
            (np.roll(grid, -1, axis=0), np.roll(mask, -1, axis=0)),
            (np.roll(grid, 1, axis=1), np.roll(mask, 1, axis=1)),
            (np.roll(grid, -1, axis=1), np.roll(mask, -1, axis=1)),
        ):
            total += np.where(source_mask, source, 0.0)
            count += source_mask
        fill = ~mask & (count > 0)
        grid[fill] = total[fill] / count[fill]
        mask[fill] = True
    return grid


def _profile(
    coordinates: np.ndarray,
    fields: dict[str, np.ndarray],
    radial_target: float,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
) -> dict[str, Any]:
    z_span = max(z_max - z_min, 1.0e-300)
    bins = np.clip(
        ((coordinates[:, 1] - z_min) / z_span * PROFILE_SAMPLES).astype(np.int64),
        0,
        PROFILE_SAMPLES - 1,
    )
    distance = np.abs(coordinates[:, 0] - radial_target)
    order = np.lexsort((distance, bins))
    ordered_bins = bins[order]
    _, first = np.unique(ordered_bins, return_index=True)
    chosen = order[first]
    chosen_bins = bins[chosen]
    target_bins = np.arange(PROFILE_SAMPLES)
    result: dict[str, Any] = {
        "radial_target_m": float(f"{radial_target:.12g}"),
        "z_m": _rounded(
            z_min + (target_bins + 0.5) * (z_max - z_min) / PROFILE_SAMPLES,
            10,
        ),
    }
    for key, values in fields.items():
        sampled = values[chosen]
        interpolated = np.interp(target_bins, chosen_bins, sampled)
        result[key] = _rounded(interpolated, 9)
    return result


def _viewer_projection(
    viewer: dict[str, Any],
    design_id: str,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "classification",
        "artifact_payload_sha256",
        "coordinates_rz_m",
        "triangles",
        "triangle_region_ids",
        "vertex_fields",
        "qois_bz_t",
        "limitations",
        "integrity",
    }
    if set(viewer) != required:
        raise ValueError(f"{design_id} viewer top-level contract differs")
    if (
        viewer["schema_version"] != "cft_revival.fem_reference.viewer/1.1.0"
        or viewer["classification"] != CLASSIFICATION
    ):
        raise ValueError(f"{design_id} viewer identity differs")
    coordinates = np.asarray(viewer["coordinates_rz_m"], dtype=np.float64)
    triangles = np.asarray(viewer["triangles"], dtype=np.int64)
    region_ids = viewer["triangle_region_ids"]
    fields = {
        key: np.asarray(viewer["vertex_fields"][key], dtype=np.float64)
        for key in ("psi_wb_per_rad", "b_r_t", "b_z_t")
    }
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or len(region_ids) != len(triangles)
        or any(values.shape != (len(coordinates),) for values in fields.values())
        or not np.isfinite(coordinates).all()
        or any(not np.isfinite(values).all() for values in fields.values())
        or int(triangles.min()) < 0
        or int(triangles.max()) >= len(coordinates)
    ):
        raise ValueError(f"{design_id} viewer arrays differ")
    r_min, z_min = coordinates.min(axis=0)
    r_max, z_max = coordinates.max(axis=0)
    r_span = max(float(r_max - r_min), 1.0e-300)
    z_span = max(float(z_max - z_min), 1.0e-300)
    x = np.clip(
        ((coordinates[:, 0] - r_min) / r_span * GRID_WIDTH).astype(np.int64),
        0,
        GRID_WIDTH - 1,
    )
    y = np.clip(
        ((coordinates[:, 1] - z_min) / z_span * GRID_HEIGHT).astype(np.int64),
        0,
        GRID_HEIGHT - 1,
    )
    linear = y * GRID_WIDTH + x
    cell_count = GRID_WIDTH * GRID_HEIGHT
    occupied = np.bincount(linear, minlength=cell_count)
    raster_fields: dict[str, list[float]] = {}
    magnitude = np.hypot(fields["b_r_t"], fields["b_z_t"])
    for key, values in {**fields, "b_magnitude_t": magnitude}.items():
        summed = np.bincount(linear, weights=values, minlength=cell_count)
        means = np.divide(
            summed,
            occupied,
            out=np.zeros(cell_count, dtype=np.float64),
            where=occupied > 0,
        )
        raster_fields[key] = _rounded(
            _fill_grid(means, occupied > 0).reshape(-1),
            8,
        )

    names = sorted(set(region_ids))
    codes = np.fromiter(
        (names.index(name) for name in region_ids),
        dtype=np.int16,
        count=len(region_ids),
    )
    centroids = coordinates[triangles].mean(axis=1)
    cx = np.clip(
        ((centroids[:, 0] - r_min) / r_span * GRID_WIDTH).astype(np.int64),
        0,
        GRID_WIDTH - 1,
    )
    cy = np.clip(
        ((centroids[:, 1] - z_min) / z_span * GRID_HEIGHT).astype(np.int64),
        0,
        GRID_HEIGHT - 1,
    )
    centroid_linear = cy * GRID_WIDTH + cx
    region_votes = np.vstack(
        [
            np.bincount(
                centroid_linear[codes == code],
                minlength=cell_count,
            )
            for code in range(len(names))
        ]
    )
    region_raster = np.argmax(region_votes, axis=0).astype(np.int16)
    no_vote = region_votes.sum(axis=0) == 0
    default_code = names.index(Counter(region_ids).most_common(1)[0][0])
    region_raster[no_vote] = default_code
    profiles = {
        "axis": _profile(
            coordinates,
            fields,
            float(r_min),
            float(r_min),
            float(r_max),
            float(z_min),
            float(z_max),
        ),
        "radial_slice": _profile(
            coordinates,
            fields,
            float(r_min + 0.35 * r_span),
            float(r_min),
            float(r_max),
            float(z_min),
            float(z_max),
        ),
    }
    counts = Counter(region_ids)
    return {
        "grid": {"width": GRID_WIDTH, "height": GRID_HEIGHT},
        "extent_rz_m": _rounded(np.array([r_min, r_max, z_min, z_max]), 12),
        "fields": raster_fields,
        "profiles": profiles,
        "regions": [
            {"id": name, "triangle_count": counts[name]} for name in names
        ],
        "region_raster": [int(value) for value in region_raster],
        "source_vertices": int(len(coordinates)),
        "source_triangles": int(len(triangles)),
    }


def _verify_checkpoint(
    base: Path,
    anchor: Mapping[str, Any],
    design_id: str,
    chain: str,
    previous_file_hash: str,
    previous_mesh_hash: str,
    include_mesh: bool,
) -> tuple[dict[str, Any], str, str]:
    path = _safe_path(base, anchor.get("file"), f"{design_id} {chain} checkpoint")
    expected_file = _digest(
        anchor.get("file_sha256"),
        f"{design_id} {chain} checkpoint file SHA-256",
    )
    _stream_hash(path, f"{design_id} {chain} checkpoint", expected_file)
    checkpoint = _load_object(path, f"{design_id} {chain} checkpoint")
    payload_hash = _verify_integrity(checkpoint, f"{design_id} {chain} checkpoint")
    if payload_hash != anchor.get("payload_sha256"):
        raise ValueError(f"{design_id} {chain} checkpoint payload binding differs")
    for key in ("mesh_sha256", "parent_mesh_sha256", "run_sha256"):
        if checkpoint.get(key) != anchor.get(key):
            raise ValueError(f"{design_id} {chain} checkpoint {key} differs")
    if (
        anchor.get("previous_checkpoint_file_sha256") != previous_file_hash
        or checkpoint.get("previous_checkpoint_file_sha256") != previous_file_hash
    ):
        raise ValueError(f"{design_id} {chain} checkpoint file ancestry differs")
    level = int(anchor["level"])
    if chain == "adaptive" and level > 0:
        if anchor.get("parent_mesh_sha256") != previous_mesh_hash:
            raise ValueError(f"{design_id} adaptive mesh ancestry differs")
    sidecar = checkpoint.get("array_sidecar")
    if not isinstance(sidecar, dict):
        raise ValueError(f"{design_id} {chain} checkpoint lacks array sidecar")
    sidecar_path = _safe_path(path.parent, sidecar.get("file"), "checkpoint sidecar")
    sidecar_hash = _digest(
        sidecar.get("file_sha256"),
        f"{design_id} {chain} sidecar SHA-256",
    )
    projection = (
        _mesh_projection(sidecar_path, sidecar_hash, f"{design_id} level {level}")
        if include_mesh
        else None
    )
    if not include_mesh:
        _stream_hash(sidecar_path, f"{design_id} {chain} array sidecar", sidecar_hash)
    return (
        {
            "metadata_file_sha256": expected_file,
            "metadata_payload_sha256": payload_hash,
            "array_file_sha256": sidecar_hash,
            "mesh": projection,
        },
        expected_file,
        str(anchor["mesh_sha256"]),
    )


def _design_payload(
    design_id: str,
    label: str,
    manifest_hash: str,
    expected_qualified: bool,
) -> dict[str, Any]:
    base = THIRD_LEVEL / design_id
    manifest_path = base / "manifest.json"
    _stream_hash(manifest_path, f"{design_id} manifest", manifest_hash)
    manifest = _load_object(manifest_path, f"{design_id} manifest")
    manifest_payload_hash = _verify_integrity(manifest, f"{design_id} manifest")
    if (
        manifest.get("artifact_authority")
        != "schema_v1.3_recomputed_acceptance_with_bound_checkpoint_chain"
        or manifest.get("classification") != CLASSIFICATION
        or manifest.get("diagnostic_policy")
        != {
            "hardware_validation": False,
            "timing_and_memory": "DIAGNOSTIC_ONLY",
        }
        or manifest.get("domain_expansion_evidence", {}).get("status") != "completed"
        or len(manifest.get("designs", [])) != 1
    ):
        raise ValueError(f"{design_id} is not completed authoritative evidence")
    record = manifest["designs"][0]
    if record.get("config_id") != f"{design_id}-v1":
        raise ValueError(f"{design_id} config identity differs")

    artifact_path = _safe_path(base, record["artifact"], f"{design_id} artifact")
    artifact_file_hash = _digest(
        record["artifact_file_sha256"], f"{design_id} artifact file SHA-256"
    )
    _stream_hash(artifact_path, f"{design_id} artifact", artifact_file_hash)
    artifact_payload_hash = _digest(
        record["artifact_payload_sha256"], f"{design_id} artifact payload SHA-256"
    )

    viewer_path = _safe_path(base, record["viewer"], f"{design_id} viewer")
    viewer_file_hash = _digest(
        record["viewer_file_sha256"], f"{design_id} viewer file SHA-256"
    )
    _stream_hash(viewer_path, f"{design_id} viewer", viewer_file_hash)
    viewer = _load_object(viewer_path, f"{design_id} viewer")
    viewer_payload_hash = _verify_integrity(viewer, f"{design_id} viewer")
    if viewer.get("artifact_payload_sha256") != artifact_payload_hash:
        raise ValueError(f"{design_id} viewer-to-artifact payload binding differs")
    field = _viewer_projection(viewer, design_id)

    runs = record["runs"]
    checkpoints = record["checkpoints"]
    if len(runs) != 3 or len(checkpoints) != 3:
        raise ValueError(f"{design_id} does not contain three adaptive levels")
    level_integrity: list[dict[str, Any]] = []
    previous_file_hash = ZERO_HASH
    previous_mesh_hash = ZERO_HASH
    levels = []
    for run, anchor in zip(runs, checkpoints):
        level = int(run["level"])
        if level != int(anchor["level"]):
            raise ValueError(f"{design_id} run/checkpoint level differs")
        verified, previous_file_hash, previous_mesh_hash = _verify_checkpoint(
            base,
            anchor,
            design_id,
            "adaptive",
            previous_file_hash,
            previous_mesh_hash,
            True,
        )
        quality = run["mesh_quality"]
        if (
            int(quality["p2_dofs"]) != int(anchor["p2_dofs"])
            or int(quality["triangles"]) != int(anchor["triangles"])
            or run["mesh_sha256"] != anchor["mesh_sha256"]
            or run["parent_mesh_sha256"] != anchor["parent_mesh_sha256"]
        ):
            raise ValueError(f"{design_id} run/checkpoint topology differs")
        preflight = run["allocation_preflight"]
        levels.append(
            {
                "level": level,
                "p2_dofs": int(quality["p2_dofs"]),
                "triangles": int(quality["triangles"]),
                "vertices": int(quality["vertices"]),
                "minimum_angle_deg": _finite(
                    quality["minimum_angle_deg"], "minimum angle"
                ),
                "maximum_aspect_indicator": _finite(
                    quality["maximum_aspect_indicator"], "aspect indicator"
                ),
                "minimum_area_m2": _finite(
                    quality["minimum_area_m2"], "minimum area"
                ),
                "adjacent_area_size_growth": _finite(
                    run["adjacent_area_size_growth"], "area size growth"
                ),
                "relative_true_residual_l2": _finite(
                    run["relative_true_residual_l2"], "true residual"
                ),
                "iterations": int(run["iterations"]),
                "assembly_seconds": _finite(
                    run["assembly_seconds"], "assembly time"
                ),
                "solve_seconds": _finite(run["solve_seconds"], "solve time"),
                "peak_working_set_bytes": int(run["peak_working_set_bytes"]),
                "modeled_allocation_bytes": int(preflight["modeled_bytes"]),
                "required_free_ram_bytes": int(preflight["required_free_ram_bytes"]),
                "available_ram_bytes": int(preflight["available_ram_bytes"]),
                "mesh_sha256": run["mesh_sha256"],
                "parent_mesh_sha256": run["parent_mesh_sha256"],
                "checkpoint_file_sha256": verified["metadata_file_sha256"],
                "checkpoint_payload_sha256": verified["metadata_payload_sha256"],
                "checkpoint_array_sha256": verified["array_file_sha256"],
                "qois_bz_t": {
                    key: _finite(value, key)
                    for key, value in sorted(run["qois_bz_t"].items())
                    if key.endswith("-bore-average")
                },
                "qoi_h_m": {
                    key: _finite(value, key)
                    for key, value in sorted(run["resolution"]["qoi_h_m"].items())
                },
                "refinement": {
                    **{
                        key: value
                        for key, value in run.get("adaptivity", {}).items()
                        if key
                        in {
                            "dorfler_marked_elements",
                            "marked_elements_after_conformity_closure",
                            "refined_parents_after_gradation_closure",
                            "marked_indicator_fraction",
                            "theta",
                        }
                    },
                    "terminal_level": "adaptivity" not in run,
                },
                "mesh_projection": verified["mesh"],
            }
        )
        level_integrity.append(
            {key: value for key, value in verified.items() if key != "mesh"}
        )

    domain_integrity = []
    previous_file_hash = ZERO_HASH
    previous_mesh_hash = ZERO_HASH
    for anchor in record["domain_checkpoints"]:
        verified, previous_file_hash, previous_mesh_hash = _verify_checkpoint(
            base,
            anchor,
            design_id,
            "domain",
            previous_file_hash,
            previous_mesh_hash,
            False,
        )
        domain_integrity.append(
            {key: value for key, value in verified.items() if key != "mesh"}
        )
    convergence = record["convergence"]
    qualified = bool(convergence["less_than_one_percent_reached"])
    if qualified != expected_qualified:
        raise ValueError(f"{design_id} qualification status differs from accepted record")
    committed_status = (
        "NUMERICAL_P2_QUALIFIED" if qualified else "SCREENING_ONLY"
    )
    if record.get("qualification_status") != committed_status:
        raise ValueError(f"{design_id} explicit qualification status differs")
    if (
        not convergence["two_successive_less_than_one_percent"]
        or not convergence["phase_matched_domain_expansion_gate"]
        or not convergence["adjacent_size_growth_gate"]
    ):
        raise ValueError(f"{design_id} completed non-order gates unexpectedly differ")
    if qualified != bool(convergence["stable_positive_order"]):
        raise ValueError(f"{design_id} qualification/order gate binding differs")
    final_qois = {
        key: value
        for key, value in viewer["qois_bz_t"].items()
        if key.endswith("-bore-average")
    }
    if final_qois != levels[-1]["qois_bz_t"]:
        raise ValueError(f"{design_id} final viewer QoIs differ from level evidence")
    domain = convergence["domain_expansion"]
    return {
        "id": design_id,
        "label": label,
        "status": "NUMERICAL P2 QUALIFIED" if qualified else "SCREENING ONLY",
        "qualified": qualified,
        "status_reason": (
            "All two-change, positive-order, mesh-growth, and phase-matched domain gates passed."
            if qualified
            else "Two-change, mesh-growth, and domain gates passed; one or more observed orders are non-positive."
        ),
        "classification": record["classification"],
        "identity": {
            "manifest_file_sha256": manifest_hash,
            "manifest_payload_sha256": manifest_payload_hash,
            "artifact_file_sha256": artifact_file_hash,
            "artifact_payload_sha256": artifact_payload_hash,
            "viewer_file_sha256": viewer_file_hash,
            "viewer_payload_sha256": viewer_payload_hash,
            "checkpoint_chain": level_integrity,
            "domain_checkpoint_chain": domain_integrity,
        },
        "levels": levels,
        "qoi_names": list(convergence["acceptance_qois"]),
        "successive_changes": convergence[
            "successive_volume_qoi_relative_changes"
        ],
        "observed_orders": convergence["observed_orders_from_actual_qoi_h"],
        "gates": {
            "two_successive_changes_below_one_percent": convergence[
                "two_successive_less_than_one_percent"
            ],
            "stable_positive_order": convergence["stable_positive_order"],
            "adjacent_size_growth": convergence["adjacent_size_growth_gate"],
            "phase_matched_domain_expansion": convergence[
                "phase_matched_domain_expansion_gate"
            ],
        },
        "domain": {
            "phase_matched": domain["phase_matched"],
            "passed": domain["passed"],
            "maximum_qoi_relative_change": domain["maximum_qoi_relative_change"],
            "successive_qoi_relative_changes": domain[
                "successive_qoi_relative_changes"
            ],
            "runs": [
                {
                    "padding_factor": run["padding_factor"],
                    "p2_dofs": run["p2_dofs"],
                    "triangles": run["triangles"],
                    "assembly_seconds": run["assembly_seconds"],
                    "solve_seconds": run["solve_seconds"],
                    "peak_working_set_bytes": run["peak_working_set_bytes"],
                    "relative_true_residual_l2": run[
                        "relative_true_residual_l2"
                    ],
                    "mesh_sha256": run["mesh_sha256"],
                    "bound_local_h_m": run["bound_local_h_m"],
                    "qois_bz_t": {
                        key: value
                        for key, value in run["qois_bz_t"].items()
                        if key.endswith("-bore-average")
                    },
                }
                for run in record["domain_runs"]
            ],
        },
        "field": field,
        "limitations": sorted(
            set(manifest["limitations"]) | set(viewer["limitations"])
        ),
    }


def build_payload() -> dict[str, Any]:
    designs = [
        _design_payload(design_id, label, digest, qualified)
        for design_id, label, digest, qualified in EVIDENCE
    ]
    payload = {
        "schema": "cft-revival.fem-reference-p2-qualification-dashboard/1.0.0",
        "title": "Accepted P2 FEM qualification evidence",
        "warning": (
            "Independent numerical P2 reference only. This dashboard makes no "
            "hardware, experimental, material-plasma, thrust, efficiency, or "
            "device-performance claim."
        ),
        "source_policy": (
            "Inline projections are generated only after exact SHA-256 verification "
            "of pinned manifests, artifacts, viewers, checkpoints, and binary sidecars."
        ),
        "designs": designs,
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != {
        "schema",
        "title",
        "warning",
        "source_policy",
        "designs",
    }:
        raise ValueError("dashboard payload top-level contract differs")
    if (
        payload["schema"]
        != "cft-revival.fem-reference-p2-qualification-dashboard/1.0.0"
    ):
        raise ValueError("dashboard payload schema differs")
    designs = payload["designs"]
    if not isinstance(designs, list) or len(designs) != 3:
        raise ValueError("dashboard payload must contain three designs")
    expected = {
        design_id: qualified for design_id, _label, _digest_value, qualified in EVIDENCE
    }
    if {design["id"] for design in designs} != set(expected):
        raise ValueError("dashboard design identities differ")
    for design in designs:
        if (
            design["qualified"] != expected[design["id"]]
            or len(design["levels"]) != 3
            or len(design["domain"]["runs"]) != 3
            or design["classification"] != CLASSIFICATION
        ):
            raise ValueError("dashboard qualification evidence differs")
        expected_status = (
            "NUMERICAL P2 QUALIFIED" if design["qualified"] else "SCREENING ONLY"
        )
        if design["status"] != expected_status:
            raise ValueError("dashboard status label differs")
        for level, record in enumerate(design["levels"]):
            if (
                record["level"] != level
                or record["mesh_projection"]["source_triangles"]
                != record["triangles"]
                or len(record["mesh_sha256"]) != 64
                or len(record["parent_mesh_sha256"]) != 64
                or len(record["checkpoint_file_sha256"]) != 64
                or len(record["checkpoint_array_sha256"]) != 64
            ):
                raise ValueError("dashboard level ordering differs")


def render_html(payload: Mapping[str, Any]) -> str:
    validate_payload(payload)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if template.count("__DATA__") != 1:
        raise ValueError("dashboard template must contain one data placeholder")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).replace("</", "<\\/")
    return template.replace("__DATA__", encoded)


def generate(output: Path = DEFAULT_OUTPUT) -> str:
    html = render_html(build_payload())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8", newline="\n")
    return sha256(html.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(generate(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
