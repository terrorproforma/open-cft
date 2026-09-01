from __future__ import annotations

import ast
import hashlib
import math
import subprocess
from pathlib import Path

import pytest

from cft_revival.coupling.v3_models import ValidatedPsiMap
from cft_revival.fields import AxisymmetricProblem
from experiments.cft_topology_characterization_v1.experiment import (
    ACCEPTED_COUPLING_COMMIT,
    EXPERIMENT_DIR,
    PROTOCOL,
    PROTOCOL_SEMANTIC_SHA256,
    _strict_json,
    assign_roots,
    build_case,
    case_definitions,
    cluster_detections,
    domain_for,
    local_topology,
    normalized_text_hash,
    semantic_hash,
    separatrix_connectivity,
    validate_results,
)


def _double_well_map(points: int = 121) -> ValidatedPsiMap:
    radial = tuple(3.0 * index / (points - 1) for index in range(points))
    axial = tuple(3.0 * index / (points - 1) for index in range(points))
    center = 1.5
    axial_center = 1.5
    a = 0.25

    def values(radius: float, z_m: float) -> tuple[float, float, float]:
        x = radius - center
        y = z_m - axial_center
        psi = 0.25 * x**4 - 0.5 * a * x * x + 0.5 * y * y
        # Hamiltonian vector: determinant/index distinguish X and O roots.
        br = -y
        bz = x**3 - a * x
        return psi, br, bz

    rows = tuple(
        tuple(values(radius, z_m) for z_m in axial) for radius in radial
    )
    return ValidatedPsiMap(
        radial,
        axial,
        tuple(tuple(item[0] for item in row) for row in rows),
        tuple(tuple(item[1] for item in row) for row in rows),
        tuple(tuple(item[2] for item in row) for row in rows),
        hashlib.sha256(repr(rows).encode()).hexdigest(),
    )


def _root(root_id: str, r_m: float, z_m: float, kind: str = "X"):
    return {
        "root_id": root_id,
        "r_m": r_m,
        "z_m": z_m,
        "local_topology": {"classification": kind},
        "eligible_cusp": kind == "X",
        "eligible_cell": kind == "O",
    }


def _map_summary(roots, mesh=0.1):
    return {"roots": list(roots), "mesh_scale_m": mesh}


def test_protocol_is_developmental_and_has_no_retrospective_target() -> None:
    assert PROTOCOL["classification"] == "developmental_topology_characterization"
    assert PROTOCOL["not_a_design_optimization"]
    assert PROTOCOL["not_a_blind_validation"]
    assert not PROTOCOL["analyses"]["retrospective_pass_target_allowed"]
    assert not PROTOCOL["publication"]["mirror_probability"]
    assert not PROTOCOL["publication"]["plasma_state_power_or_performance"]
    assert PROTOCOL["families"]["stage_counts"] == list(range(2, 9))
    assert PROTOCOL["execution"]["single_execution"]
    assert semantic_hash(_strict_json(EXPERIMENT_DIR / "protocol.json")) == (
        PROTOCOL_SEMANTIC_SHA256
    )


def test_all_56_geometry_and_three_role_domains_are_strict() -> None:
    definitions = case_definitions()
    assert len(definitions) == 56
    assert {item.stage_count for item in definitions} == set(range(2, 9))
    assert len({item.family_semantic_sha256 for item in definitions}) == 56
    for definition in definitions:
        case = build_case(definition)
        assert len(case.sources) == 2 * definition.stage_count
        for stage in range(definition.stage_count):
            assert (
                case.sources[2 * stage].ampere_turns_a
                == case.sources[2 * stage + 1].ampere_turns_a
            )
        for role in ("primary", "refined", "enlarged_domain"):
            AxisymmetricProblem(
                f"{definition.case_id}-{role}",
                domain_for(case, role),
                case.sources,
            )
        primary = domain_for(case, "primary")
        refined = domain_for(case, "refined")
        enlarged = domain_for(case, "enlarged_domain")
        assert refined.radial_intervals > primary.radial_intervals
        assert refined.axial_intervals > primary.axial_intervals
        assert enlarged.radius_m > primary.radius_m
        assert enlarged.z_min_m < primary.z_min_m
        assert enlarged.z_max_m > primary.z_max_m


def test_mesh_scaled_clustering_combines_axis_grid_and_bilinear_detections() -> None:
    detections = (
        {
            "r_m": 0.0,
            "z_m": 0.0,
            "method": "axis_grid",
            "finite_box_boundary": False,
            "field_magnitude_t": 0.0,
        },
        {
            "r_m": 0.02,
            "z_m": 0.01,
            "method": "bilinear_vector_root",
            "finite_box_boundary": False,
            "field_magnitude_t": 1e-12,
        },
        {
            "r_m": 0.3,
            "z_m": 0.0,
            "method": "axis_sign_change",
            "finite_box_boundary": False,
            "field_magnitude_t": 0.0,
        },
    )
    clusters = cluster_detections(detections, 0.04)
    assert len(clusters) == 2
    combined = next(item for item in clusters if item["member_count"] == 2)
    assert combined["methods"] == ["axis_grid", "bilinear_vector_root"]
    assert combined["cluster_tolerance_m"] == pytest.approx(0.03)


def test_manufactured_jacobian_index_classifies_x_and_o() -> None:
    field = _double_well_map()
    mesh = max(field.r_m[1] - field.r_m[0], field.z_m[1] - field.z_m[0])
    x_point = local_topology(field, (1.5, 1.5), mesh)
    left_o = local_topology(field, (1.0, 1.5), mesh)
    right_o = local_topology(field, (2.0, 1.5), mesh)
    assert x_point["classification"] == "X"
    assert x_point["topological_index"] == pytest.approx(-1.0, abs=0.05)
    assert left_o["classification"] == "O"
    assert right_o["classification"] == "O"
    assert left_o["topological_index"] == pytest.approx(1.0, abs=0.05)


def test_manufactured_offset_surfaces_establish_cell_bounding_separatrix() -> None:
    field = _double_well_map()
    mesh = max(field.r_m[1] - field.r_m[0], field.z_m[1] - field.z_m[0])
    result = separatrix_connectivity(
        field,
        (1.5, 1.5),
        mesh,
        chamber_radius_m=2.9,
        chamber_length_m=2.9,
    )
    assert len(result["levels"]) == 2
    assert result["probe_delta_wb"] > 0.0
    assert all("components" in level for level in result["levels"])
    assert result["has_nearby_closed_channel_surface"]
    assert result["cell_bounding"]


def test_global_2d_assignment_allows_unmatched_equal_cardinality() -> None:
    primary = _map_summary((_root("p0", 0.0, 0.0), _root("p1", 10.0, 0.0)))
    other = _map_summary((_root("q0", 0.01, 0.0), _root("q1", 20.0, 0.0)))
    result = assign_roots(primary, other)
    assert result["correspondence_count"] == 1
    assert result["matches"][0]["primary_root_id"] == "p0"
    assert result["unmatched_primary_root_ids"] == ["p1"]
    assert result["unmatched_other_root_ids"] == ["q1"]


def test_semantic_identity_is_newline_independent() -> None:
    assert normalized_text_hash("a\nb\n") == normalized_text_hash("a\r\nb\r\n")
    assert semantic_hash({"b": 2, "a": 1}) == semantic_hash({"a": 1, "b": 2})


def test_no_prior_experiment_or_plasma_mirror_publication_imports() -> None:
    tree = ast.parse((EXPERIMENT_DIR / "experiment.py").read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("four_cell_topology_search" in module for module in imports)
    assert "cft_revival.plasma" not in imports
    assert "cft_revival.plasma_network" not in imports
    assert "cft_revival.coupling.losses" not in imports


def test_accepted_dependencies_match_coupling_v3_commit() -> None:
    root = EXPERIMENT_DIR.parents[2]
    result = subprocess.run(
        (
            "git",
            "diff",
            "--quiet",
            ACCEPTED_COUPLING_COMMIT,
            "--",
            "modern/src/cft_revival/coupling",
            "modern/src/cft_revival/fields",
            "modern/src/cft_revival/geometry",
            "modern/src/cft_revival/magnetics",
            "modern/spec",
        ),
        cwd=root,
    )
    assert result.returncode == 0


def test_result_lifecycle_before_or_after_run() -> None:
    results = EXPERIMENT_DIR / "results"
    if not results.exists():
        assert PROTOCOL["result_lifecycle"]["pre_run"].startswith("tests pass")
    else:
        result = validate_results(results)
        assert result["dataset"]["summary"]["evaluated_count"] == 56
        assert result["dataset"]["summary"]["mirror_probability_count"] == 0
        assert result["dataset"]["summary"]["plasma_publication_count"] == 0
