"""v3 builder: every Sobol design builds, the v2 rules are reproduced on v2 values, identities are stable."""

from __future__ import annotations

import math

import pytest

from experiments.l1a_geometry_sweep_v2 import experiment as sweep
from experiments.l1a_geometry_sweep_v3 import descriptors as DS
from experiments.l1a_geometry_sweep_v3 import designs as D
from experiments.l1a_geometry_sweep_v3 import experiment as E


@pytest.fixture(scope="module")
def value() -> dict:
    return E.protocol()


@pytest.fixture(scope="module")
def designs(value: dict):
    return D.sobol_design_list(value)


def test_every_sobol_design_builds_feasibly_inside_the_declared_regime(value: dict, designs) -> None:
    assert len(designs) == 128
    coverage = value["sampling"]["regime_coverage"]
    inside = predicted = five_stage = 0
    for index, design in enumerate(designs):
        case = D.build_case(design, index, value)
        derived = case.derived
        assert derived["feasibility"]["feasible"], (index, derived["feasibility"]["checks"])
        assert derived["feasibility"]["checks"]["radial_sums_exact"]
        assert coverage["wall_radius_over_pitch"][0] - 1e-9 <= derived["wall_radius_over_pitch"] <= coverage["wall_radius_over_pitch"][1] + 1e-9
        assert abs(derived["x_w"] - math.pi * derived["wall_radius_over_pitch"]) < 1e-12
        assert derived["stage_count"] in (3, 4, 5) and len(case.geometry.stages) == derived["stage_count"]
        assert case.case_id.startswith(f"l1a-gs-v3-{index:03d}-")
        assert abs(derived["represented_chamber_outer_radius_m"] - derived["requested_chamber_outer_radius_m"]) <= 0.5 * D.LENGTH_QUANTUM_M
        assert derived["magnet_remanence_t"] == 1.05
        inside += derived["inside_sweep_v2_box"]
        predicted += derived["x_w"] >= DS.X_STAR_HEMP_LIKE
        five_stage += derived["stage_count"] == 5 and derived["x_w"] >= DS.X_STAR_HEMP_LIKE
    assert inside == 6 and predicted == 51 and five_stage == 17


def test_v3_builder_reproduces_the_v2_rules_on_v2_values(value: dict) -> None:
    """On sweep-v2 design values the v3 builder gives the v2 geometry up to identifiers and the 2**-40 m quantum."""

    v2_designs = sweep.sample_designs()
    v3_variables = D.variables_from_protocol(value)
    for index in (0, 19, 38, 57, 76, 95):
        v2_case = sweep.build_case(v2_designs[index], index)
        remapped = D.Design(v2_designs[index].values, v3_variables, provenance="v2-values")
        v3_case = D.build_case(remapped, index, value)
        v2_geometry, v3_geometry = v2_case.geometry, v3_case.geometry
        assert len(v2_geometry.regions) == len(v3_geometry.regions)
        for left, right in zip(v2_geometry.regions, v3_geometry.regions, strict=True):
            assert left.region_id == right.region_id and left.role == right.role and left.polarity == right.polarity
            for name in ("r_inner_start_m", "r_outer_start_m", "r_inner_end_m", "r_outer_end_m", "z_min_m", "z_max_m"):
                assert abs(getattr(left, name) - getattr(right, name)) <= 4.0 * D.LENGTH_QUANTUM_M, (index, left.region_id, name)
        assert [s.magnetization for s in v2_geometry.stages] == [s.magnetization for s in v3_geometry.stages]
        assert len(v2_case.problem.sources) == len(v3_case.problem.sources)
        for left, right in zip(v2_case.problem.sources, v3_case.problem.sources, strict=True):
            assert left.polarity == right.polarity and abs(left.ampere_turns_a - right.ampere_turns_a) <= 1e-9 * abs(left.ampere_turns_a)
        assert v3_case.derived["inside_sweep_v2_box"] is True


def test_specs_and_identities_are_consistent(value: dict) -> None:
    specs = E.all_specs(value)
    assert len(specs) == 224 and len({spec.key for spec in specs}) == 224
    sobol = [spec for spec in specs if spec.set_id == D.SET_SOBOL]
    assert [spec.ordinal for spec in sobol] == list(range(128))
    assert sum(spec.representative for spec in sobol) == 4
    identity = D.design_identity_without_solving(sobol[0], value)
    case = D.sobol_case(sobol[0], value)
    assert identity["case_sha256"] == case.case_sha256 and identity["geometry_sha256"] == case.geometry_sha256
    assert identity["x_w"] == pytest.approx(math.pi * case.geometry.chamber.outer_radius_m / case.derived["represented_stage_pitch_m"])
    accepted = D.field_identity(case, value, "accepted", D.SET_SOBOL)
    refined = D.field_identity(case, value, "refined", D.SET_SOBOL)
    assert accepted != refined and len(accepted) == 64


def test_sweep_v2_held_out_specs_bind_to_the_sealed_bundle(value: dict) -> None:
    specs = [spec for spec in E.all_specs(value) if spec.set_id == D.SET_SWEEP]
    assert len(specs) == 96 and sum(spec.representative for spec in specs) == 4
    binding = D.sweep_binding()
    assert binding.manifest["terminal_status"] == "ACCEPTED"
    identity = D.design_identity_without_solving(specs[0], value)
    recorded = binding.cases_by_id[specs[0].design_id]
    assert identity["case_sha256"] == recorded["case_sha256"] and identity["geometry_sha256"] == recorded["geometry_sha256"]
    assert identity["inside_sweep_v2_box"] is True


def test_quantize_length_makes_radial_sums_exact() -> None:
    r = D.quantize_length(0.003088898111414164)
    d = D.quantize_length(0.0007785227762535214)
    assert (r + d) - r == d
    assert abs(r - 0.003088898111414164) <= 0.5 * D.LENGTH_QUANTUM_M
    assert D.quantize_length(0.0038) == pytest.approx(0.0038, abs=D.LENGTH_QUANTUM_M)
