"""Consumer contract of the cusp/cell catalogue (schema, tamper refusal, manifest binding)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cft_revival.experiment_runtime.canonical import canonical_bytes

from experiments.cusp_topology_search_v3 import catalogue as C
from experiments.cusp_topology_search_v3 import experiment as E


def _cusp(cusp_id: str, z_c: float, null_z: float, length: float) -> dict:
    return {
        "cusp_id": cusp_id,
        "null_id": cusp_id.replace("wall-cusp", "axis-null"),
        "axis_null_z_m": null_z,
        "z_c_m": z_c,
        "z_c_over_length": z_c / length,
        "zone": "straight_dielectric",
        "inside_straight_dielectric": True,
        "boundary_ambiguous": False,
        "wall_b_t": 0.1,
        "wall_b_r_t": 0.1,
        "wall_b_z_t": 0.0,
        "wall_normal_component_t": 0.1,
        "angle_to_wall_normal_deg": 0.0,
        "axis_to_wall_shift_m": z_c - null_z,
        "distance_to_nearest_stage_gap_m": 0.0001,
        "distance_to_nearest_stage_centre_m": 0.003,
        "z_c_flux_root_m": z_c,
        "flux_root_consistent": True,
    }


def _cell(cell_id: str, kind: str, start: float, end: float, start_cusp: str | None, end_cusp: str | None) -> dict:
    return {
        "cell_id": cell_id,
        "kind": kind,
        "z_start_m": start,
        "z_end_m": end,
        "length_m": end - start,
        "length_over_pitch": (end - start) / 0.006,
        "start_cusp_id": start_cusp,
        "end_cusp_id": end_cusp,
        "axis_interval_m": [start, end],
        "wall_b_min_t": 0.05,
        "wall_b_min_z_m": 0.5 * (start + end),
        "cusp_wall_b_min_t": 0.1,
        "cusp_wall_b_max_t": 0.1,
        "wall_mirror_ratio": 2.0,
        "wall_mirror_ratio_strong_end": 2.0,
        "axis_bz_peak_t": 0.2,
        "axis_bz_peak_z_m": 0.5 * (start + end),
        "axis_mirror_ratio": 0.5,
        "sweep_axis_bz_peaks_inside": None,
        "stage_centres_inside": 1,
    }


def synthetic_record() -> dict:
    length = 0.024
    cusps = [_cusp("wall-cusp-01", 0.006, 0.0061, length), _cusp("wall-cusp-02", 0.012, 0.012, length), _cusp("wall-cusp-03", 0.018, 0.0179, length)]
    cells = [
        _cell("cell-01", "anode_partial", 0.001, 0.006, None, "wall-cusp-01"),
        _cell("cell-02", "interior", 0.006, 0.012, "wall-cusp-01", "wall-cusp-02"),
        _cell("cell-03", "interior", 0.012, 0.018, "wall-cusp-02", "wall-cusp-03"),
        _cell("cell-04", "exit_partial", 0.018, 0.023, "wall-cusp-03", None),
    ]
    return {
        "set_id": "p2_divergent_exit",
        "design_id": "divergent-exit-stack",
        "label": E.P2_CLASSIFICATION,
        "record_path": "artifacts/designs/p2_divergent_exit/divergent-exit-stack.json",
        "geometry": {
            "wall_radius_m": 0.002,
            "straight_z_min_m": 0.001,
            "straight_z_max_m": 0.023,
            "chamber_length_m": length,
            "stage_pitch_m": 0.006,
            "stage_centres_m": [0.003, 0.009, 0.015, 0.021],
            "injector_length_m": 0.0015,
        },
        "identity": {"accepted_field_identity_sha256": "a" * 64, "refined_field_identity_sha256": "b" * 64},
        "stability": {"stable": True},
        "accepted": {
            "axis_nulls": {"nulls": [{"null_id": f"axis-null-0{i}", "z_m": z, "zone": "channel", "classification": "X"} for i, z in enumerate((0.0061, 0.012, 0.0179), start=1)]},
            "topology": {
                "wall_cusps": cusps,
                "outside_intersections": [],
                "cells": cells,
                "wall_cusp_count": 3,
                "cell_count": 4,
                "four_wall_cusps": False,
                "four_cells": True,
            },
        },
    }


@pytest.fixture(scope="module")
def value() -> dict:
    return E.protocol()


def test_catalogue_builds_and_validates_from_a_record(value: dict) -> None:
    catalogue = C.build_catalogue(value, [synthetic_record()], protocol_semantic_sha256="c" * 64)
    assert catalogue["schema_version"] == C.CATALOGUE_SCHEMA and catalogue["design_count"] == 1
    entry = catalogue["entries"][0]
    assert set(entry) == C.ENTRY_KEYS and entry["label"] == E.P2_CLASSIFICATION and entry["four_cells"] is True
    assert set(entry["wall_cusps"][0]) == C.CUSP_KEYS and set(entry["cells"][0]) == C.CELL_KEYS
    cells = C.cells_for_design(catalogue, "p2_divergent_exit", "divergent-exit-stack")
    assert cells["label"] == E.P2_CLASSIFICATION and len(cells["cells"]) == 4 and cells["wall_cusp_z_m"] == [0.006, 0.012, 0.018]
    assert abs(cells["cells"][1]["axial_centre_m"] - 0.009) <= 1.0e-15
    with pytest.raises(KeyError):
        C.cells_for_design(catalogue, "sweep_v2", "missing")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda e: e.__setitem__("label", "PLASMA_EVIDENCE"), "unknown label"),
        (lambda e: e["wall_cusps"].reverse(), "not sorted"),
        (lambda e: e["cells"][1].__setitem__("z_start_m", 0.0065), "do not tile"),
        (lambda e: e["cells"][-1].__setitem__("z_end_m", 0.0229), "do not end"),
        (lambda e: e.__setitem__("four_cells", False), "legacy-target flags"),
        (lambda e: e["cells"][1].__setitem__("wall_mirror_ratio", -1.0), "must be positive"),
        (lambda e: e.__setitem__("stable", 1), "stable must be a bool"),
        (lambda e: e["wall_cusps"][0].__setitem__("z_c_m", 0.0005), "not sorted|outside the straight"),
        (lambda e: e.pop("record_path"), "keys differ"),
    ],
)
def test_validate_catalogue_refuses_broken_entries(value: dict, mutate, message: str) -> None:
    catalogue = C.build_catalogue(value, [synthetic_record()], protocol_semantic_sha256="c" * 64)
    broken = copy.deepcopy(catalogue)
    mutate(broken["entries"][0])
    with pytest.raises(ValueError, match=message):
        C.validate_catalogue(broken)


def test_load_catalogue_requires_the_sealed_manifest_bytes(tmp_path: Path, value: dict) -> None:
    catalogue = C.build_catalogue(value, [synthetic_record()], protocol_semantic_sha256="c" * 64)
    root = tmp_path / "results"
    (root / "artifacts").mkdir(parents=True)
    raw = canonical_bytes(catalogue)
    (root / C.CATALOGUE_RELATIVE_PATH).write_bytes(raw)
    manifest = {
        "state": "accepted_result",
        "artifacts": [{"path": C.CATALOGUE_RELATIVE_PATH, "type": "file", "byte_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}],
    }
    (root / "manifest.json").write_bytes(canonical_bytes(manifest))
    loaded = C.load_catalogue(root)
    assert loaded["design_count"] == 1
    (root / C.CATALOGUE_RELATIVE_PATH).write_bytes(raw.replace(b'"stable":true', b'"stable":false'))
    with pytest.raises(ValueError, match="differ from the sealed manifest"):
        C.load_catalogue(root)
    manifest["state"] = "assessment_rejection"
    (root / "manifest.json").write_bytes(canonical_bytes(manifest))
    with pytest.raises(ValueError, match="not an accepted result"):
        C.load_catalogue(root)


def test_recorded_catalogue_is_schema_valid_but_refused_by_the_loader() -> None:
    """The v3 bundle is a recorded assessment_rejection; consumers must not ingest its catalogue."""

    if not (E.RESULTS_ROOT / "manifest.json").is_file():
        pytest.skip("not executed yet")
    with pytest.raises(ValueError, match="not an accepted result"):
        C.load_catalogue(E.RESULTS_ROOT)
    catalogue = json.loads((E.RESULTS_ROOT / "artifacts" / "cusp-cell-catalogue.json").read_text(encoding="utf-8"))
    C.validate_catalogue(catalogue)
    dataset = json.loads((E.RESULTS_ROOT / "artifacts" / "topology-dataset.json").read_text(encoding="utf-8"))
    assert catalogue["design_count"] == dataset["design_count"] == 281
    by_key = {(row["set_id"], row["design_id"]): row for row in dataset["designs"]}
    for entry in catalogue["entries"]:
        row = by_key[(entry["set_id"], entry["design_id"])]
        assert entry["wall_cusp_count"] == row["wall_cusp_count"] and entry["stable"] == row["stability"]["stable"]
        assert [cusp["z_c_m"] for cusp in entry["wall_cusps"]] == [cusp["z_c_m"] for cusp in row["wall_cusps"]]
        assert entry["label"] == row["label"]
