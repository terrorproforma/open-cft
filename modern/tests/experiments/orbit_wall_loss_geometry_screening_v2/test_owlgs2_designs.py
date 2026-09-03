"""Design binding: sweep designs through the v1 pipeline (imported), the P2 row through the v4 adapter."""

from __future__ import annotations

import math

import pytest

from cft_revival.experiment_runtime.canonical import strict_json_file
from cft_revival.orbit_mc import OrbitConfig

from experiments.orbit_wall_loss_geometry_screening_v1 import designs as v1_designs
from experiments.orbit_wall_loss_geometry_screening_v2 import designs as D
from experiments.orbit_wall_loss_geometry_screening_v2 import experiment as E

REPRESENTATIVE = "l1a-gs-v2-000-48d2ccedd5"


@pytest.fixture(scope="module")
def value() -> dict:
    return E.protocol()


def test_sweep_design_binding_reuses_v1_identities(value: dict) -> None:
    sweep = D.load_sweep_binding(value["field_source"])
    design = D.bind_sweep_design(sweep, REPRESENTATIVE, value["field_source"], representative=True)
    assert design.set_id == D.SET_SWEEP and design.label == D.LABEL_SWEEP and design.representative is True
    assert design.sweep_index == 0 and design.design_key == REPRESENTATIVE
    v1_authorities = strict_json_file(v1_designs.EXPERIMENT / "design-authorities.json")
    v1_row = next(row for row in v1_authorities["designs"] if row["case_id"] == REPRESENTATIVE)
    assert design.accepted_field_identity == v1_row["accepted_field_identity_sha256"]
    assert design.refined_field_identity == v1_row["refined_field_identity_sha256"]
    assert design.identities["case_sha256"] == v1_row["case_sha256"]
    assert design.geometry == v1_row["geometry"]
    assert design.straight_z_min_m == 0.0 and design.straight_z_max_m == design.geometry["exit_start_m"]
    assert design.domain_z_max_m == design.geometry["chamber_length_m"]


def test_orbit_config_follows_the_v1_rule_on_the_design_authority(value: dict) -> None:
    sweep = D.load_sweep_binding(value["field_source"])
    design = D.bind_sweep_design(sweep, REPRESENTATIVE, value["field_source"], representative=True)
    rule = value["orbit_geometry_rule"]
    config = E.orbit_config(value, design, "N")
    assert isinstance(config, OrbitConfig)
    assert config.max_rotation_rad == 0.16 and E.orbit_config(value, design, "2N").max_rotation_rad == 0.08
    assert config.max_path_m == 2.0 * design.chamber_length_m
    slowest = math.sqrt(2.0 * 5.0 * D.EV_J / D.ELECTRON_MASS_KG)
    assert config.max_time_s == pytest.approx(2.0 * config.max_path_m / slowest)
    assert config.wall_z_max_m == design.geometry["exit_start_m"] and config.domain_z_max_m == design.chamber_length_m
    assert config.max_steps == rule["max_steps"] and config.event_tolerance_m == rule["event_tolerance_m"]
    # identical to v1's config for the same design and time step
    v1_config = v1_designs.orbit_config_for(v1_designs.design_geometry(v1_designs.rebuild_case(sweep, REPRESENTATIVE)), rule, rule["timestep_policies"]["N"])
    assert config == v1_config


def test_p2_design_is_bound_to_the_v4_authority_by_hash(value: dict) -> None:
    declaration = value["designs"]["p2_design"]
    design = D.bind_p2_design(declaration)
    assert design.set_id == D.SET_P2 and design.label == D.LABEL_P2 and design.representative is True
    assert design.wall_radius_m == 0.002 and design.straight_z_min_m == 0.001 and design.straight_z_max_m == 0.018
    assert design.domain_z_min_m == 0.001 and design.domain_z_max_m == 0.023 and design.chamber_length_m == 0.024
    assert design.identities["geometry_sha256"] == declaration["geometry_sha256"]
    assert design.geometry["stage_count"] == 4 and design.geometry["has_divergent_exit"] is True
    config = E.orbit_config(value, design, "N")
    assert config.wall_z_min_m == 0.001 and config.wall_z_max_m == 0.018 and config.domain_z_max_m == 0.023
    assert config.max_path_m == 0.048  # 2 L, not v4's 0.03 m
    v4 = D.v4_protocol(declaration)
    assert v4["orbit"]["max_path_m"] == 0.03 and v4["orbit"]["max_time_s"] == 1e-8
    assert design.accepted_field_identity != design.refined_field_identity
    bad = dict(declaration)
    bad["v4_protocol_file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="v4 protocol bytes"):
        D.bind_p2_design(bad)
    worse = {**declaration, "maps": {**declaration["maps"], "primary": {**declaration["maps"]["primary"], "mesh_sha256": "1" * 64}}}
    with pytest.raises(ValueError, match="differs from the declared authority"):
        D.bind_p2_design(worse)


def test_design_keys_and_sets(value: dict) -> None:
    keys = E.design_keys(value)
    assert len(keys) == 97 and keys == tuple(sorted(keys))
    assert E.design_set(value, REPRESENTATIVE) == D.SET_SWEEP and E.design_set(value, D.P2_DESIGN_ID) == D.SET_P2
    with pytest.raises(ValueError):
        E.design_set(value, "not-a-design")
