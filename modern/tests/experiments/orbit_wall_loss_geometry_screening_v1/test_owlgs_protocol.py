"""Protocol, design binding and launch-design invariants of the geometry screening campaign."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict

import pytest

import cft_revival.orbit_mc as orbit_mc
from cft_revival.experiment_runtime import canonical_bytes
from cft_revival.orbit_mc import OrbitConfig, Termination

from experiments.orbit_wall_loss_geometry_screening_v1 import designs as D
from experiments.orbit_wall_loss_geometry_screening_v1 import experiment as E

CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"


@pytest.fixture(scope="module")
def value() -> dict:
    return E.protocol()


@pytest.fixture(scope="module")
def binding(value: dict) -> D.SweepBinding:
    return D.load_sweep_binding(value["field_source"])


def test_classification_label_is_screening_everywhere(value: dict) -> None:
    assert value["classification"] == CLASSIFICATION
    assert E.CLASSIFICATION == CLASSIFICATION
    assert "not P2-qualified" in value["classification_statement"] or "not P2" in value["classification_statement"]
    boundary = value["claim_boundary"]
    assert boundary["not_accepted_physical_orbit_evidence"] is True
    assert boundary["not_p2_qualified"] is True
    assert boundary["forbid_plasma_performance_publication"] is True
    assert boundary["forbid_pic_or_self_consistent_claim"] is True
    assert boundary["hardware_or_experimental_validation"] is False
    assert value["field_source"]["field_status"] == "accepted_L1a_screening_not_P2_qualified"
    assert value["shakedown"]["evidentiary"] is False
    assert value["shakedown"]["outcomes_enter_estimand"] is False


def test_design_batches_cover_the_96_accepted_designs(value: dict, binding: D.SweepBinding) -> None:
    primary = value["designs"]["primary_case_ids"]
    extension = value["designs"]["extension_case_ids"]
    assert len(primary) == 25
    assert len(extension) == 71
    assert not set(primary) & set(extension)
    assert set(primary) == set(binding.summary["nondominated_case_ids"])
    assert set(primary) | set(extension) == set(binding.cases_by_id)
    assert set(value["designs"]["representative_case_ids"]) == set(D.representative_case_ids(binding))
    assert set(value["designs"]["representative_case_ids"]) <= set(primary)
    ids = E.design_case_ids(value)
    assert len(ids) == (96 if value["designs"]["extension_batch_included"] else 25)
    assert ids == tuple(sorted(ids))
    for case_id in value["shakedown"]["design_case_ids"]:
        assert case_id in primary


def test_field_source_authority_matches_the_sealed_sweep_bundle(value: dict, binding: D.SweepBinding) -> None:
    declaration = value["field_source"]
    assert binding.manifest_file_sha256 == declaration["manifest_file_sha256"]
    assert binding.raw_file_sha256 == declaration["raw_results_file_sha256"]
    assert binding.summary_file_sha256 == declaration["summary_file_sha256"]
    assert binding.manifest["terminal_status"] == "ACCEPTED"
    assert declaration["resolve"]["domain"] == {
        "radius_m": 0.03, "z_min_m": -0.015, "z_max_m": 0.05, "radial_intervals": 80, "axial_intervals": 144,
    }
    assert asdict(D.sweep.SOLVER) == declaration["resolve"]["solver_config"]
    with pytest.raises(ValueError, match="authority differs"):
        D.load_sweep_binding({**declaration, "raw_results_file_sha256": "0" * 64})


def test_orbit_mc_contract_and_source_hashes_are_bound(value: dict) -> None:
    report = E.orbit_mc_contract_report(value)
    assert report["matches"], report
    assert orbit_mc.__version__ == "1.7.0" == value["orbit_mc_contract"]["package_version"]
    assert report["source_sha256"] == E.orbit_mc_source_sha256()
    binding = E.source_binding_report(value)
    assert binding["field_pipeline_source_sha256"] == D.field_pipeline_source_sha256()
    assert binding["experiment_code_sha256"] == E.experiment_code_sha256()
    assert set(binding["experiment_code_files"]) == {"consumer.py", "designs.py", "experiment.py", "run.py", "__init__.py"}
    files = binding["field_pipeline_source_files"]
    assert any(item.startswith("src/cft_revival/fields/") for item in files)
    assert any(item.startswith("src/cft_revival/geometry/") for item in files)
    assert any(item.startswith("src/cft_revival/magnetics/") for item in files)
    assert any(item.startswith("src/cft_revival/optimization/") for item in files)
    assert "experiments/l1a_geometry_sweep_v2/experiment.py" in files
    assert "experiments/l1a_geometry_sweep_v2/protocol.json" in files
    for path in D.field_pipeline_source_files():
        assert b"\r" not in path.read_bytes(), path


def test_rebuilt_designs_reproduce_the_sealed_identities(value: dict, binding: D.SweepBinding) -> None:
    for case_id in value["shakedown"]["design_case_ids"]:
        case = D.rebuild_case(binding, case_id)
        recorded = binding.cases_by_id[case_id]
        assert case.case_sha256 == recorded["case_sha256"]
        assert case.geometry_sha256 == recorded["geometry_sha256"]
        geometry = D.design_geometry(case)
        assert geometry.wall_radius_m == case.geometry.chamber.outer_radius_m
        assert geometry.exit_start_m == case.geometry.chamber.exit_start_m
        assert geometry.has_divergent_exit == (case.geometry.chamber.exit_length_m > 0.0)
        assert geometry.first_polarity in (-1, 1)
    with pytest.raises(ValueError, match="not a sweep-v2 case id"):
        D.case_index("nope")


def test_geometry_rule_gives_valid_design_dependent_orbit_configs(value: dict, binding: D.SweepBinding) -> None:
    bound = E.bind_designs(value, binding, E.design_case_ids(value))
    assert len(bound) == len(E.design_case_ids(value))
    slowest = math.sqrt(2.0 * 5.0 * D.EV_J / D.ELECTRON_MASS_KG)
    divergent = 0
    for item in bound.values():
        geometry = item.geometry
        for timestep in E.TIMESTEPS:
            config = E.orbit_config(value, geometry, timestep)
            assert isinstance(config, OrbitConfig)
            assert config.wall_radius_m == config.domain_radius_m == geometry.wall_radius_m
            assert config.wall_z_min_m == config.domain_z_min_m == 0.0
            assert config.wall_z_max_m == geometry.exit_start_m
            assert config.domain_z_max_m == geometry.chamber_length_m
            assert config.wall_z_max_m <= config.domain_z_max_m
            assert config.max_path_m == 2.0 * geometry.chamber_length_m
            assert config.max_time_s == pytest.approx(2.0 * config.max_path_m / slowest)
            assert config.max_rotation_rad == value["orbit_geometry_rule"]["timestep_policies"][timestep]["max_rotation_rad"]
        cells = D.launch_cells(geometry, value["launches"])
        assert [round(c["fraction"], 6) for c in cells] == [0.125, 0.375, 0.625, 0.875]
        for cell in cells:
            assert geometry.injector_length_m < cell["axial_center_m"] < geometry.exit_start_m
        positions = D.launch_positions(geometry, value["launches"])
        assert len(positions) == 8
        assert len({surface for surface, _ in positions}) == 8
        for surface, (x, y, z) in positions:
            assert y == 0.0
            assert 0.0 < x < geometry.wall_radius_m
            assert surface.split("-r", 1)[0].startswith("gs1-cell-")
        divergent += geometry.has_divergent_exit
    assert divergent == 90 if len(bound) == 96 else divergent >= 1


def test_every_case_has_512_launches_32_strata_and_globally_unique_seeds(value: dict, binding: D.SweepBinding) -> None:
    plan = E.evidentiary_plan(value)
    bound = E.bind_designs(value, binding, plan.case_ids)
    matrix = E.case_matrix(value, plan)
    expected_cases = 2 * len(plan.case_ids) + sum(
        1 for case_id in plan.case_ids if case_id in value["designs"]["representative_case_ids"]
    )
    assert len(matrix) == expected_cases
    launch_ids: set[str] = set()
    seeds: set[int] = set()
    for case_id, role, timestep, campaign, key in matrix:
        assert key == f"{case_id}--{role}-{timestep}"
        assert campaign == f"owlgs-v1:{case_id}:{role}:{timestep}"
        launches = E.build_case_launches(value, plan, bound[case_id].geometry, role, timestep)
        assert len(launches) == 512
        assert all(item.launch_id.startswith(campaign + ":") for item in launches)
        strata = Counter(
            (item.flux_surface_id.split("-r", 1)[0], item.kinetic_energy_ev, item.pitch_angle_rad, item.parallel_direction)
            for item in launches
        )
        assert len(strata) == 32
        assert set(strata.values()) == {16}
        assert len(E.batch_records(plan, launches)) == 8
        launch_ids.update(item.launch_id for item in launches)
        seeds.update(item.seed_id for item in launches)
    assert len(launch_ids) == 512 * expected_cases
    assert len(seeds) == 512 * expected_cases


def test_design_authorities_are_deterministic_and_closed(value: dict, binding: D.SweepBinding) -> None:
    plan = E.shakedown_plan(value)
    bound = E.bind_designs(value, binding, plan.case_ids)
    first = E.build_design_authorities(value, plan, bound)
    second = E.build_design_authorities(value, plan, bound)
    assert canonical_bytes(first) == canonical_bytes(second)
    assert first["case_count"] == 7
    assert first["total_launches"] == 7 * 64
    for case in first["cases"]:
        launches = E.build_case_launches(value, plan, bound[case["case_id"]].geometry, case["role"], case["timestep"])
        payload = canonical_bytes(E.runtime_launch_payload(case["campaign_id"], launches))
        assert E.load_runtime_launch_payload(payload, case["campaign_id"]) == tuple(sorted(launches, key=lambda i: i.launch_id))
        with pytest.raises(ValueError):
            E.load_runtime_launch_payload(payload, case["campaign_id"] + "x")


def test_runtime_launch_payload_rejects_tampering(value: dict, binding: D.SweepBinding) -> None:
    plan = E.shakedown_plan(value)
    bound = E.bind_designs(value, binding, plan.case_ids)
    case_id = plan.case_ids[0]
    campaign = E.campaign_id(plan, case_id, "accepted", "N")
    launches = E.build_case_launches(value, plan, bound[case_id].geometry, "accepted", "N")
    payload = E.runtime_launch_payload(campaign, launches)
    payload["launches"][0]["seed_id"] = "-1"
    with pytest.raises(ValueError, match="unsigned decimal"):
        E.load_runtime_launch_payload(canonical_bytes(payload), campaign)
    payload = E.runtime_launch_payload(campaign, launches)
    payload["extra"] = 1
    with pytest.raises(ValueError, match="not closed"):
        E.load_runtime_launch_payload(canonical_bytes(payload), campaign)


def test_gyrophase_grids_are_disjoint_from_every_prior_campaign(value: dict) -> None:
    report = E.gyrophase_grid_disjointness(value)
    assert report["disjoint"] is True
    assert set(report["offsets_rad"]) == {"v1_v2", "v3", "v4", "v4_shakedown", "evidentiary", "shakedown"}
    assert report["offsets_rad"]["evidentiary"] == pytest.approx(11 * math.pi / 96)
    assert report["offsets_rad"]["shakedown"] == pytest.approx(5 * math.pi / 96)
    assert report["minimum_separation_mod_period_rad"] > 0.03


def test_escape_subclasses_follow_the_divergent_exit_policy() -> None:
    config = OrbitConfig(
        wall_radius_m=0.002, wall_z_min_m=0.0, wall_z_max_m=0.018,
        domain_radius_m=0.002, domain_z_min_m=0.0, domain_z_max_m=0.024,
        max_time_s=1e-8, max_path_m=0.03,
    )

    class Result:
        def __init__(self, termination: Termination, position: tuple[float, float, float]) -> None:
            self.termination = termination
            self.final_position_m = position

    assert E.escape_subclass(Result(Termination.DOMAIN_ESCAPE, (0.001, 0.0, 0.0)), config) == "upstream_anode_plane"
    assert E.escape_subclass(Result(Termination.DOMAIN_ESCAPE, (0.001, 0.0, 0.024)), config) == "exit_plane"
    assert E.escape_subclass(Result(Termination.DOMAIN_ESCAPE, (0.002, 0.0, 0.020)), config) == "divergent_section_radial"
    assert E.escape_subclass(Result(Termination.DOMAIN_ESCAPE, (0.001, 0.0, 0.010)), config) == "unclassified"
    assert E.escape_subclass(Result(Termination.WALL_HIT, (0.002, 0.0, 0.010)), config) is None


def test_gates_declare_screening_convergence_and_zero_numerical_failures(value: dict) -> None:
    gates = value["gates"]
    assert gates["maximum_successive_probability_change"] == 0.02
    assert gates["require_zero_numerical_failures"] is True
    assert set(gates["numerical_failure_terminations"]) == {t.value for t in E.NUMERICAL_FAILURES}
    assert gates["maximum_relative_energy_error"] == 1e-10
    assert gates["maximum_wall_endpoint_error_m"] == 1e-8
    assert gates["require_final_velocity_equals_event_velocity"] is True
    assert gates["require_exact_authority_replay"] is True
    assert "cpu parity only" in gates["backend_parity_scope"]
    assert value["diagnostics"]["magnetic_moment_variation"]["binding"] is False
    assert not any("mu" in key.lower().split("_") or "magnetic_moment" in key.lower() for key in gates)


def test_csv_columns_are_unique_and_carry_the_label() -> None:
    assert len(set(E.CSV_COLUMNS)) == len(E.CSV_COLUMNS)
    assert "classification" in E.CSV_COLUMNS
    assert "p_wall_2N" in E.CSV_COLUMNS and "converged" in E.CSV_COLUMNS
