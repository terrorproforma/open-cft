"""Design re-solve: identity proof, QoI replay, stored-map agreement, bore interpolant."""

from __future__ import annotations

import numpy as np
import pytest

from cft_revival.orbit_mc import OrbitConfig, preflight_campaign

from experiments.orbit_wall_loss_geometry_screening_v1 import designs as D
from experiments.orbit_wall_loss_geometry_screening_v1 import experiment as E

REPRESENTATIVE = "l1a-gs-v2-000-48d2ccedd5"


@pytest.fixture(scope="module")
def value() -> dict:
    return E.protocol()


@pytest.fixture(scope="module")
def binding(value: dict) -> D.SweepBinding:
    return D.load_sweep_binding(value["field_source"])


@pytest.fixture(scope="module")
def resolved(value: dict, binding: D.SweepBinding) -> D.ResolvedDesign:
    return D.resolve_design(binding, REPRESENTATIVE, value, include_refined=False)


def test_resolved_representative_reproduces_the_stored_map_and_qois(resolved: D.ResolvedDesign) -> None:
    evidence = resolved.evidence
    assert evidence["passed"] is True
    assert evidence["resolve"]["backend"] == "python"
    assert evidence["resolve"]["qoi_replay"]["passed"] is True
    stored = evidence["resolve"]["stored_representative"]
    assert stored is not None and stored["passed"] is True
    assert stored["psi_max_abs_difference_wb"] <= 1e-15
    assert stored["b_max_abs_difference_t"] <= 1e-9
    assert evidence["accepted_bore_field"]["interpolation_error_report"]["b_relative_rms"] <= 0.05
    assert evidence["accepted_bore_field"]["bore_grid"]["r_max_m"] >= resolved.geometry.wall_radius_m
    assert evidence["accepted_bore_field"]["bore_grid"]["z_min_m"] <= 0.0
    assert evidence["accepted_bore_field"]["bore_grid"]["z_max_m"] >= resolved.geometry.chamber_length_m
    assert evidence["refined_bore_field"] is None and evidence["cross_resolution"] is None
    assert evidence["accepted_bore_field"]["source_identity_sha256"] == D.field_identity(
        resolved.case, E.protocol()["field_source"], "accepted"
    )


def test_bore_field_is_traversable_and_preflights_the_evidentiary_launches(value: dict, resolved: D.ResolvedDesign) -> None:
    field = resolved.accepted.field
    assert bool(np.all(field.traversable_cells))
    assert field.plasma_material_id == D.PLASMA_MATERIAL_ID
    plan = E.evidentiary_plan(value)
    launches = E.build_case_launches(value, plan, resolved.geometry, "accepted", "N")
    config = E.orbit_config(value, resolved.geometry, "N")
    report = preflight_campaign(launches, field, config)
    assert report["status"] == "passed"
    assert report["launch_count"] == 512
    assert report["maximum_launch_b_t"] <= report["maximum_declared_b_t"]


def test_field_identity_distinguishes_roles_and_designs(value: dict, binding: D.SweepBinding) -> None:
    declaration = value["field_source"]
    first = D.rebuild_case(binding, REPRESENTATIVE)
    second = D.rebuild_case(binding, "l1a-gs-v2-095-27f6ceb96c")
    assert D.field_identity(first, declaration, "accepted") != D.field_identity(first, declaration, "refined")
    assert D.field_identity(first, declaration, "accepted") != D.field_identity(second, declaration, "accepted")
    assert D.field_identity(first, declaration, "accepted") == D.field_identity(first, declaration, "accepted")


def test_rebuild_refuses_a_tampered_sealed_record(value: dict, binding: D.SweepBinding) -> None:
    tampered = dict(binding.cases_by_id)
    record = dict(tampered[REPRESENTATIVE])
    record["case_sha256"] = "0" * 64
    tampered[REPRESENTATIVE] = record
    forged = D.SweepBinding(
        binding.manifest, binding.raw, binding.summary, tampered,
        binding.manifest_file_sha256, binding.raw_file_sha256, binding.summary_file_sha256,
    )
    with pytest.raises(ValueError, match="case_sha256 differs"):
        D.rebuild_case(forged, REPRESENTATIVE)


def test_orbit_config_rule_rejects_impossible_geometry(value: dict) -> None:
    rule = value["orbit_geometry_rule"]
    geometry = D.DesignGeometry(
        case_id="x", design_id="y", wall_radius_m=0.002, chamber_length_m=0.02, injector_length_m=0.0016,
        exit_start_m=0.0016, exit_length_m=0.0184, exit_outer_radius_m=0.003, dielectric_thickness_m=0.0007,
        stage_count=3, stage_pitch_m=0.006, stage_centers_m=(0.003, 0.009, 0.015), magnet_axial_thickness_m=0.003,
        magnet_inner_radius_m=0.004, magnet_outer_radius_m=0.007, first_polarity=1, has_divergent_exit=True,
        straight_wall_scope="test",
    )
    with pytest.raises(ValueError, match="no straight channel span"):
        D.launch_cells(geometry, value["launches"])
    config = D.orbit_config_for(
        D.DesignGeometry(**{**geometry.__dict__, "exit_start_m": 0.015}), rule, rule["timestep_policies"]["N"]
    )
    assert isinstance(config, OrbitConfig)
    assert config.wall_z_max_m == 0.015
