"""Consumer contract: the cusp/cell catalogue emitted by cusp topology search v3.

The catalogue lists, for every design of every set, the axis nulls, the wall cusps
(separatrix-wall intersections inside the straight dielectric) and the cells they bound,
so that the wall-loss geometry screening's launch design and the MDO closures can define
cells by actual separatrix intersections instead of channel fractions. Every entry carries
its label; consumers must carry it too and must never read the mirror descriptors as
probabilities.

``load_catalogue`` verifies the catalogue bytes against the sealed bundle manifest before
returning it, so a consumer cannot ingest an edited copy.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.experiment_runtime import strict_json_file

CATALOGUE_SCHEMA = "cft-revival.cusp-cell-catalogue/1.0.0"
CATALOGUE_RELATIVE_PATH = "artifacts/cusp-cell-catalogue.json"
LABELS = ("SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY", "P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY")
CELL_KINDS = ("anode_partial", "interior", "exit_partial", "unbounded")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

ENTRY_KEYS = frozenset(
    {
        "set_id",
        "design_id",
        "label",
        "stable",
        "geometry",
        "axis_nulls",
        "wall_cusps",
        "outside_intersections",
        "cells",
        "wall_cusp_count",
        "cell_count",
        "four_wall_cusps",
        "four_cells",
        "accepted_field_identity_sha256",
        "refined_field_identity_sha256",
        "record_path",
    }
)
CUSP_KEYS = frozenset(
    {
        "cusp_id",
        "axis_null_z_m",
        "z_c_m",
        "z_c_over_length",
        "wall_b_t",
        "wall_b_r_t",
        "wall_b_z_t",
        "angle_to_wall_normal_deg",
        "boundary_ambiguous",
        "distance_to_nearest_stage_gap_m",
        "distance_to_nearest_stage_centre_m",
    }
)
CELL_KEYS = frozenset(
    {
        "cell_id",
        "kind",
        "z_start_m",
        "z_end_m",
        "length_m",
        "length_over_pitch",
        "start_cusp_id",
        "end_cusp_id",
        "wall_b_min_t",
        "wall_b_min_z_m",
        "cusp_wall_b_min_t",
        "wall_mirror_ratio",
        "wall_mirror_ratio_strong_end",
        "axis_bz_peak_t",
        "axis_bz_peak_z_m",
        "axis_mirror_ratio",
    }
)


def _catalogue_cusp(cusp: Mapping[str, Any]) -> dict[str, Any]:
    return {key: cusp[key] for key in sorted(CUSP_KEYS)}


def _catalogue_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {key: cell[key] for key in sorted(CELL_KEYS)}


def catalogue_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one accepted design record onto the catalogue contract."""

    accepted = record["accepted"]
    topology = accepted["topology"]
    return {
        "set_id": record["set_id"],
        "design_id": record["design_id"],
        "label": record["label"],
        "stable": bool(record["stability"]["stable"]),
        "geometry": dict(record["geometry"]),
        "axis_nulls": [
            {"null_id": null["null_id"], "z_m": null["z_m"], "zone": null["zone"], "classification": null["classification"]}
            for null in accepted["axis_nulls"]["nulls"]
        ],
        "wall_cusps": [_catalogue_cusp(cusp) for cusp in topology["wall_cusps"]],
        "outside_intersections": [
            {"cusp_id": row["cusp_id"], "z_c_m": row["z_c_m"], "zone": row["zone"], "wall_b_t": row["wall_b_t"], "axis_null_z_m": row["axis_null_z_m"]}
            for row in topology["outside_intersections"]
        ],
        "cells": [_catalogue_cell(cell) for cell in topology["cells"]],
        "wall_cusp_count": int(topology["wall_cusp_count"]),
        "cell_count": int(topology["cell_count"]),
        "four_wall_cusps": bool(topology["four_wall_cusps"]),
        "four_cells": bool(topology["four_cells"]),
        "accepted_field_identity_sha256": record["identity"]["accepted_field_identity_sha256"],
        "refined_field_identity_sha256": record["identity"]["refined_field_identity_sha256"],
        "record_path": record["record_path"],
    }


def build_catalogue(value: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *, protocol_semantic_sha256: str) -> dict[str, Any]:
    entries = [catalogue_entry(record) for record in records]
    entries.sort(key=lambda entry: (entry["set_id"], entry["design_id"]))
    catalogue = {
        "schema_version": CATALOGUE_SCHEMA,
        "experiment_id": value["experiment_id"],
        "protocol_semantic_sha256": protocol_semantic_sha256,
        "labels": list(LABELS),
        "definition": "wall cusp := intersection of the separatrix of an axis null with the straight dielectric wall; cell := wall interval between consecutive cusps (plus anode-side and exit-side partial cells); see protocol.json#definition_v3",
        "consumer_contract": value["catalogue"]["consumer_contract"],
        "mirror_descriptor_statement": value["claim_boundary"]["mirror_ratios_are_field_descriptors_not_probabilities"],
        "design_count": len(entries),
        "stable_design_count": sum(entry["stable"] for entry in entries),
        "entries": entries,
    }
    validate_catalogue(catalogue)
    return catalogue


def validate_catalogue(catalogue: Mapping[str, Any]) -> None:
    if catalogue.get("schema_version") != CATALOGUE_SCHEMA:
        raise ValueError("catalogue schema version differs from the contract")
    if list(catalogue.get("labels", [])) != list(LABELS):
        raise ValueError("catalogue labels differ from the contract")
    if catalogue.get("mirror_descriptor_statement") is not True:
        raise ValueError("catalogue must state that mirror descriptors are not probabilities")
    entries = catalogue.get("entries")
    if not isinstance(entries, list) or len(entries) != catalogue.get("design_count"):
        raise ValueError("catalogue entries do not match design_count")
    keys = set()
    for entry in entries:
        if set(entry) != ENTRY_KEYS:
            raise ValueError(f"catalogue entry keys differ from the contract: {sorted(set(entry) ^ ENTRY_KEYS)}")
        if entry["label"] not in LABELS:
            raise ValueError(f"{entry['design_id']}: unknown label")
        key = (entry["set_id"], entry["design_id"])
        if key in keys:
            raise ValueError(f"duplicate catalogue entry {key}")
        keys.add(key)
        if type(entry["stable"]) is not bool:
            raise ValueError(f"{entry['design_id']}: stable must be a bool")
        for hash_key in ("accepted_field_identity_sha256", "refined_field_identity_sha256"):
            if not _HEX64.match(str(entry[hash_key])):
                raise ValueError(f"{entry['design_id']}: {hash_key} is not a SHA-256")
        cusps = entry["wall_cusps"]
        if len(cusps) != entry["wall_cusp_count"]:
            raise ValueError(f"{entry['design_id']}: wall_cusp_count differs from the cusp list")
        z_values = [cusp["z_c_m"] for cusp in cusps]
        if z_values != sorted(z_values):
            raise ValueError(f"{entry['design_id']}: cusps are not sorted")
        geometry = entry["geometry"]
        for cusp in cusps:
            if set(cusp) != CUSP_KEYS:
                raise ValueError(f"{entry['design_id']}: cusp keys differ from the contract")
            if not geometry["straight_z_min_m"] <= cusp["z_c_m"] <= geometry["straight_z_max_m"]:
                raise ValueError(f"{entry['design_id']}: cusp outside the straight dielectric")
        cells = entry["cells"]
        if len(cells) != entry["cell_count"] or len(cells) != (len(cusps) + 1 if cusps else 1):
            raise ValueError(f"{entry['design_id']}: cell count is inconsistent with the cusps")
        previous_end = geometry["straight_z_min_m"]
        for cell in cells:
            if set(cell) != CELL_KEYS:
                raise ValueError(f"{entry['design_id']}: cell keys differ from the contract")
            if cell["kind"] not in CELL_KINDS:
                raise ValueError(f"{entry['design_id']}: unknown cell kind")
            if cell["z_start_m"] != previous_end or cell["z_end_m"] < cell["z_start_m"]:
                raise ValueError(f"{entry['design_id']}: cells do not tile the straight dielectric")
            previous_end = cell["z_end_m"]
            for ratio_key in ("wall_mirror_ratio", "axis_mirror_ratio"):
                ratio = cell[ratio_key]
                if ratio is not None and not (ratio > 0.0):
                    raise ValueError(f"{entry['design_id']}: {ratio_key} must be positive or null")
        if previous_end != geometry["straight_z_max_m"]:
            raise ValueError(f"{entry['design_id']}: cells do not end at the straight-section end")
        if entry["four_wall_cusps"] != (len(cusps) == 4) or entry["four_cells"] != (len(cells) == 4 and bool(cusps)):
            raise ValueError(f"{entry['design_id']}: legacy-target flags are inconsistent")
    if catalogue["stable_design_count"] != sum(entry["stable"] for entry in entries):
        raise ValueError("stable_design_count does not reproduce")


def load_catalogue(results_root: Path) -> dict[str, Any]:
    """Load the catalogue only if its bytes are the ones sealed in the bundle manifest."""

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


def cells_for_design(catalogue: Mapping[str, Any], set_id: str, design_id: str) -> dict[str, Any]:
    """Cells of one design with the label the consumer must carry."""

    for entry in catalogue["entries"]:
        if entry["set_id"] == set_id and entry["design_id"] == design_id:
            return {
                "label": entry["label"],
                "stable": entry["stable"],
                "wall_radius_m": entry["geometry"]["wall_radius_m"],
                "straight_z_min_m": entry["geometry"]["straight_z_min_m"],
                "straight_z_max_m": entry["geometry"]["straight_z_max_m"],
                "cells": [
                    {
                        "cell_id": cell["cell_id"],
                        "kind": cell["kind"],
                        "z_start_m": cell["z_start_m"],
                        "z_end_m": cell["z_end_m"],
                        "axial_centre_m": 0.5 * (cell["z_start_m"] + cell["z_end_m"]),
                    }
                    for cell in entry["cells"]
                ],
                "wall_cusp_z_m": [cusp["z_c_m"] for cusp in entry["wall_cusps"]],
            }
    raise KeyError(f"{set_id}:{design_id} is not in the catalogue")
