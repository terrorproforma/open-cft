from __future__ import annotations

import ast
import hashlib
import subprocess
from dataclasses import replace
from math import exp, sqrt
from pathlib import Path

import pytest

from cft_revival.coupling import (
    FluxSurfacePolicy,
    TopologyResolutionError,
    require_same_flux_surface,
    trace_flux_contours,
)
from cft_revival.coupling.v3_models import ValidatedPsiMap
from cft_revival.fields import AxisymmetricProblem
from experiments.four_cell_topology_search_v2.experiment import (
    ACCEPTED_COUPLING_COMMIT,
    EXPERIMENT_DIR,
    PROTOCOL,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    _domain,
    _plasma_scenarios,
    build_candidate,
    sample_candidates,
    validate_results,
)


def _island_map() -> ValidatedPsiMap:
    radial = tuple(3.0 * index / 40 for index in range(41))
    axial = tuple(-2.0 + 4.0 * index / 40 for index in range(41))

    def values(radius: float, z: float) -> tuple[float, float, float]:
        envelope = exp(-((radius - 1.0) ** 2 + z * z))
        return (
            radius * radius * envelope,
            2.0 * z * radius * envelope,
            2.0 * envelope * (1.0 - radius * (radius - 1.0)),
        )

    rows = tuple(
        tuple(values(radius, z) for z in axial) for radius in radial
    )
    provisional = ValidatedPsiMap(
        radial,
        axial,
        tuple(tuple(value[0] for value in row) for row in rows),
        tuple(tuple(value[1] for value in row) for row in rows),
        tuple(tuple(value[2] for value in row) for row in rows),
        "0" * 64,
    )
    return replace(
        provisional,
        full_map_hash=hashlib.sha256(
            repr((radial, axial, provisional.psi_wb)).encode()
        ).hexdigest(),
    )


def test_protocol_is_exact_v3_preregistration() -> None:
    assert hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert PROTOCOL["status"] == "preregistered_pending_single_execution"
    assert (
        PROTOCOL["accepted_dependency_baseline"]["coupling_v3_commit"]
        == ACCEPTED_COUPLING_COMMIT
    )
    assert PROTOCOL["sampling"]["candidate_count"] >= 128
    assert PROTOCOL["maps"]["independent_solves_per_candidate"] == 3
    assert PROTOCOL["topology"]["required_stable_cell_count"] == 4
    assert PROTOCOL["topology"]["saddle_tie_policy"] == "reject"
    assert not PROTOCOL["claim_boundary"]["same_z_proxy_allowed"]
    assert not PROTOCOL["claim_boundary"]["null_residue_fallback_allowed"]
    assert (
        PROTOCOL["plasma_network"]["publication_policy"] == "require_full_rank"
    )


def test_all_candidates_are_unique_strict_geometries_with_paired_sources() -> None:
    candidates = sample_candidates()
    assert len(candidates) == 128
    assert len({item["sampling_identity_sha256"] for item in candidates}) == 128
    built = tuple(build_candidate(item) for item in candidates)
    assert len({item.geometry_sha256 for item in built}) == 128
    for candidate in built:
        assert len(candidate.sources) == 8
        assert len(candidate.cusp_targets_m) == 4
        assert all(
            candidate.sources[2 * stage].ampere_turns_a
            == candidate.sources[2 * stage + 1].ampere_turns_a
            for stage in range(4)
        )
        assert tuple(sorted(candidate.cusp_targets_m)) == candidate.cusp_targets_m
        for role in ("primary", "downsampled", "enlarged_domain"):
            AxisymmetricProblem(
                f"{candidate.candidate_id}-{role}",
                _domain(candidate, role),
                candidate.sources,
            )


def test_synthetic_v3_contours_are_closed_flux_bound_not_same_z() -> None:
    field = _island_map()
    target = 0.8 * max(value for row in field.psi_wb for value in row)
    contours = trace_flux_contours(field, target)
    assert contours
    assert any(
        contour.closed
        and contour.simple
        and not contour.touches_boundary
        and contour.maximum_psi_residual_wb < 1e-10
        for contour in contours
    )
    with pytest.raises(TopologyResolutionError, match="different psi"):
        require_same_flux_surface(field, ((0.0, 0.0), (1.0, 0.0)))


def test_synthetic_exact_saddle_is_fail_closed() -> None:
    from cft_revival.coupling.surfaces import _cell_segments

    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    with pytest.raises(TopologyResolutionError, match="tie policy"):
        _cell_segments(
            corners,
            (1.0, -1.0, 1.0, -1.0),
            0.0,
            0.0,
            1e-12,
            FluxSurfacePolicy().saddle_tie_policy,
        )


def test_probability_box_propagation_has_all_vertices_and_nominal() -> None:
    distributions = tuple(
        {
            "nominal_probability": 0.1 + 0.01 * index,
            "probability_lower": 0.08 + 0.01 * index,
            "probability_upper": 0.12 + 0.01 * index,
        }
        for index in range(4)
    )
    scenarios = tuple(_plasma_scenarios(distributions))
    assert len(scenarios) == 17
    assert scenarios[0][0] == "nominal"
    assert len({scenario_id for scenario_id, _ in scenarios}) == 17
    assert {
        values for scenario_id, values in scenarios if scenario_id.startswith("box-")
    } == {
        tuple(
            distributions[index][
                "probability_upper" if bit else "probability_lower"
            ]
            for index, bit in enumerate(bits)
        )
        for bits in __import__("itertools").product((0, 1), repeat=4)
    }


def test_source_has_no_deprecated_coupling_or_prior_experiment_import() -> None:
    tree = ast.parse((EXPERIMENT_DIR / "experiment.py").read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "experiments.four_cell_topology_search.experiment" not in imported
    assert "cft_revival.coupling.screening_proxy" not in imported
    assert "cft_revival.coupling.records" not in imported


def test_accepted_package_blobs_match_exact_coupling_commit() -> None:
    root = EXPERIMENT_DIR.parents[2]
    changed = subprocess.run(
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
            "modern/src/cft_revival/optimization",
            "modern/src/cft_revival/plasma",
            "modern/src/cft_revival/plasma_network",
            "modern/spec",
        ),
        cwd=root,
    )
    assert changed.returncode == 0


def test_result_lifecycle_before_or_after_single_run() -> None:
    results = EXPERIMENT_DIR / "results"
    if not results.exists():
        assert PROTOCOL["result_lifecycle"]["pre_run"].startswith("tests pass")
        return
    validated = validate_results(results)
    assert validated["manifest"]["single_execution"]
    assert validated["dataset"]["summary"]["evaluated_count"] == 128
    assert validated["dataset"]["summary"]["unique_state_count"] == 0
    assert (
        validated["dataset"]["summary"][
            "power_or_performance_publication_count"
        ]
        == 0
    )
