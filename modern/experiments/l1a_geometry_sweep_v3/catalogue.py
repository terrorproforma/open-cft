"""Consumer contract: the v3 cusp/cell catalogue (``artifacts/cusp-cell-catalogue-v3.json``).

Schema ``cft-revival.cusp-cell-catalogue/1.1.0`` = the v3.1 catalogue entry (same cusp and
cell keys, imported from ``experiments.cusp_topology_search_v3_1.catalogue``) plus, per
entry, the sweep-v3 descriptors a future screening needs to stratify by regime:
``design_values``, ``x_w``, ``wall_radius_over_pitch``, ``stage_count``, the PPM
prediction, the per-cusp Koch ratios (``rho``), ``hemp_like_all_cusps`` and the
``inside_sweep_v2_box`` flag. Every entry carries its label; consumers must carry it and
must never read rho or the mirror descriptors as probabilities.

``load_catalogue`` verifies the bytes against the sealed bundle manifest before returning.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.experiment_runtime import strict_json_file

from experiments.cusp_topology_search_v3_1 import catalogue as v31

CATALOGUE_SCHEMA = "cft-revival.cusp-cell-catalogue/1.1.0"
CATALOGUE_RELATIVE_PATH = "artifacts/cusp-cell-catalogue-v3.json"
LABEL = "SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

RHO_KEYS = frozenset(
    {
        "cusp_id",
        "z_c_m",
        "axis_null_z_m",
        "wall_b_t",
        "upstream_axis_peak_t",
        "downstream_axis_peak_t",
        "upstream_wall_max_b_t",
        "downstream_wall_max_b_t",
        "rho_downstream",
        "rho_upstream",
        "rho_conservative",
        "rho_permissive",
        "rho_wall",
        "hemp_like_conservative",
        "cusp_is_wall_maximum",
    }
)
ENTRY_KEYS = v31.ENTRY_KEYS | frozenset(
    {
        "design_values",
        "sampling_provenance",
        "stage_count",
        "x_w",
        "wall_radius_over_pitch",
        "inside_sweep_v2_box",
        "ppm_prediction",
        "rho",
        "min_rho_conservative",
        "hemp_like_all_cusps",
        "predicted_hemp_like_i1",
        "five_stage_four_cusp_hemp_like",
        "wall_harmonics_b3_over_b1",
        "wall_harmonics_b5_over_b1",
        "rho_resolution_sensitivity_max",
        "v2_gates_passed",
    }
)


def _rho_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in sorted(RHO_KEYS)}


def catalogue_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one resolved design record onto the v3 catalogue contract."""

    base = v31.catalogue_entry(record)
    descriptors = record["descriptors"]["accepted"]
    return {
        **base,
        "design_values": dict(record["evidence"]["design_values"]),
        "sampling_provenance": record["evidence"]["sampling_provenance"],
        "stage_count": int(descriptors["stage_count"]),
        "x_w": float(descriptors["x_w"]),
        "wall_radius_over_pitch": float(descriptors["wall_radius_over_pitch"]),
        "inside_sweep_v2_box": bool(record["evidence"]["derived_geometry"].get("inside_sweep_v2_box", record["set_id"] == "sweep_v2")),
        "ppm_prediction": dict(descriptors["ppm_prediction"]),
        "rho": [_rho_row(row) for row in descriptors["cusps"]],
        "min_rho_conservative": descriptors["min_rho_conservative"],
        "hemp_like_all_cusps": bool(descriptors["hemp_like_all_cusps"]),
        "predicted_hemp_like_i1": bool(descriptors["predicted_hemp_like_i1"]),
        "five_stage_four_cusp_hemp_like": bool(descriptors["five_stage_four_cusp_hemp_like"]),
        "wall_harmonics_b3_over_b1": descriptors["wall_harmonics"].get("b3_over_b1"),
        "wall_harmonics_b5_over_b1": descriptors["wall_harmonics"].get("b5_over_b1"),
        "rho_resolution_sensitivity_max": record["descriptors"]["resolution_sensitivity"]["max_relative_rho_difference"],
        "v2_gates_passed": bool(record["v2_gates"]["passed"]),
    }


def build_catalogue(value: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *, protocol_semantic_sha256: str) -> dict[str, Any]:
    entries = [catalogue_entry(record) for record in records]
    entries.sort(key=lambda entry: (entry["set_id"], entry["design_id"]))
    catalogue = {
        "schema_version": CATALOGUE_SCHEMA,
        "experiment_id": value["experiment_id"],
        "protocol_semantic_sha256": protocol_semantic_sha256,
        "labels": [LABEL],
        "definition": "wall cusp := intersection of the separatrix of an axis null with the straight dielectric wall (cusp topology search v3.1 definition, imported); rho := |B|(r_w, z_c) / adjacent axis |B_z| peak (Koch et al. IEPC-2007-110 design ratio; see protocol.json#descriptors_v3)",
        "consumer_contract": value["catalogue"]["consumer_contract"],
        "mirror_descriptor_statement": value["claim_boundary"]["mirror_ratios_are_field_descriptors_not_probabilities"],
        "hemp_like_rule": value["descriptors_v3"]["hemp_like_rule"],
        "design_count": len(entries),
        "stable_design_count": sum(entry["stable"] for entry in entries),
        "hemp_like_design_count": sum(entry["hemp_like_all_cusps"] for entry in entries),
        "entries": entries,
    }
    validate_catalogue(catalogue)
    return catalogue


def validate_catalogue(catalogue: Mapping[str, Any]) -> None:
    if catalogue.get("schema_version") != CATALOGUE_SCHEMA:
        raise ValueError("catalogue schema version differs from the v3 contract")
    if list(catalogue.get("labels", [])) != [LABEL]:
        raise ValueError("catalogue labels differ from the contract")
    if catalogue.get("mirror_descriptor_statement") is not True:
        raise ValueError("catalogue must state that mirror descriptors are not probabilities")
    entries = catalogue.get("entries")
    if not isinstance(entries, list) or len(entries) != catalogue.get("design_count"):
        raise ValueError("catalogue entries do not match design_count")
    # The v3.1 validator checks the inherited geometry/cusp/cell structure of every entry.
    v31.validate_catalogue(
        {
            **{key: catalogue[key] for key in ("design_count", "stable_design_count")},
            "schema_version": v31.CATALOGUE_SCHEMA,
            "labels": list(v31.LABELS),
            "mirror_descriptor_statement": True,
            "entries": [{key: entry[key] for key in v31.ENTRY_KEYS} for entry in entries],
        }
    )
    hemp_like = 0
    for entry in entries:
        if set(entry) != ENTRY_KEYS:
            raise ValueError(f"catalogue entry keys differ from the v3 contract: {sorted(set(entry) ^ ENTRY_KEYS)}")
        if entry["label"] != LABEL:
            raise ValueError(f"{entry['design_id']}: unknown label")
        if len(entry["rho"]) != entry["wall_cusp_count"]:
            raise ValueError(f"{entry['design_id']}: rho rows differ from the cusp count")
        for row, cusp in zip(entry["rho"], entry["wall_cusps"], strict=True):
            if set(row) != RHO_KEYS:
                raise ValueError(f"{entry['design_id']}: rho keys differ from the contract")
            if row["cusp_id"] != cusp["cusp_id"] or row["z_c_m"] != cusp["z_c_m"]:
                raise ValueError(f"{entry['design_id']}: rho rows are not aligned with the cusps")
            for key in ("rho_downstream", "rho_upstream", "rho_conservative", "rho_permissive", "rho_wall"):
                if row[key] is not None and not row[key] > 0.0:
                    raise ValueError(f"{entry['design_id']}: {key} must be positive or null")
        expected_hemp = bool(entry["rho"]) and all(row["hemp_like_conservative"] for row in entry["rho"])
        if entry["hemp_like_all_cusps"] != expected_hemp:
            raise ValueError(f"{entry['design_id']}: hemp_like_all_cusps does not reproduce from the rho rows")
        if not (entry["x_w"] > 0.0 and abs(entry["x_w"] / entry["wall_radius_over_pitch"] - 3.141592653589793) < 1.0e-9):
            raise ValueError(f"{entry['design_id']}: x_w is not pi times r_w / L")
        if entry["five_stage_four_cusp_hemp_like"] != (entry["stage_count"] == 5 and entry["wall_cusp_count"] == 4 and expected_hemp):
            raise ValueError(f"{entry['design_id']}: five_stage_four_cusp_hemp_like does not reproduce")
        hemp_like += expected_hemp
    if catalogue["hemp_like_design_count"] != hemp_like:
        raise ValueError("hemp_like_design_count does not reproduce")


def load_catalogue(results_root: Path) -> dict[str, Any]:
    """Load the v3 catalogue only if its bytes are the ones sealed in the bundle manifest."""

    manifest = strict_json_file(results_root / "manifest.json")
    if manifest.get("state") != "accepted_result":
        raise ValueError("the catalogue's bundle is not an accepted result")
    entry = next((item for item in manifest["artifacts"] if item.get("path") == CATALOGUE_RELATIVE_PATH), None)
    if entry is None:
        raise ValueError("catalogue is not listed in the bundle manifest")
    raw = (results_root / CATALOGUE_RELATIVE_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["byte_sha256"] or len(raw) != entry["bytes"]:
        raise ValueError("catalogue bytes differ from the sealed manifest entry")
    catalogue = strict_json_file(results_root / CATALOGUE_RELATIVE_PATH)
    validate_catalogue(catalogue)
    return catalogue


def hemp_like_entries(catalogue: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Entries whose every wall cusp has rho_conservative >= 1.5 (labelled descriptors)."""

    return [entry for entry in catalogue["entries"] if entry["hemp_like_all_cusps"]]
