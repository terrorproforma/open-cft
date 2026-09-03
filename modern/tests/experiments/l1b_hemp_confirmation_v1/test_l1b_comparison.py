"""Synthetic tests of the L1a-vs-P2 comparison, the tolerance rule and the confirmation verdict logic."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments.cusp_topology_search_v3_1.topology import (
    ChannelGeometry,
    TopologyPolicy,
    tracing_grid,
)
from experiments.l1b_hemp_confirmation_v1 import experiment as E

GEOMETRY = ChannelGeometry(wall_radius_m=0.003, straight_z_min_m=0.0, straight_z_max_m=0.009, chamber_length_m=0.010, stage_pitch_m=0.0035, stage_centres_m=(0.00175, 0.00525, 0.00875), injector_length_m=0.0008)


@pytest.fixture(scope="module")
def value() -> dict:
    return E.protocol()


def _cusp(cusp_id: str, z: float, b: float, *, ambiguous: bool = False) -> dict:
    return {"cusp_id": cusp_id, "null_id": cusp_id.replace("wall-cusp", "axis-null"), "axis_null_z_m": z, "z_c_m": z, "wall_b_t": b, "wall_b_r_t": b, "wall_b_z_t": 0.0, "angle_to_wall_normal_deg": 0.5, "boundary_ambiguous": ambiguous}


def _rho(cusp_id: str, z: float, b: float, axis: float, wall_max: float) -> dict:
    return {"cusp_id": cusp_id, "z_c_m": z, "wall_b_t": b, "upstream_axis_peak_t": axis, "downstream_axis_peak_t": axis * 0.9, "upstream_wall_max_b_t": wall_max, "downstream_wall_max_b_t": wall_max * 0.95, "rho_conservative": b / axis, "rho_downstream": b / (0.9 * axis), "rho_upstream": b / axis, "rho_wall": b / wall_max, "hemp_like_conservative": b / axis >= 1.5}


def _reference(zs: tuple[float, ...], b: float = 0.06, axis: float = 0.035) -> dict:
    cusps = [_cusp(f"wall-cusp-{i + 1:02d}", z, b) for i, z in enumerate(zs)]
    rho = [_rho(c["cusp_id"], c["z_c_m"], b, axis, 0.075) for c in cusps]
    return {
        "wall_cusps": cusps,
        "wall_cusp_count": len(cusps),
        "cell_count": len(cusps) + 1,
        "rho": rho,
        "min_rho_conservative": min(r["rho_conservative"] for r in rho),
        "hemp_like_all_cusps": all(r["hemp_like_conservative"] for r in rho),
        "axis_window_m": [-0.0035, 0.0135],
        "axis_nulls": [{"null_id": f"axis-null-{i + 1:02d}", "z_m": z, "zone": "channel", "classification": "X"} for i, z in enumerate(zs)] + [{"null_id": "axis-null-99", "z_m": -0.0032, "zone": "anode_side", "classification": "X"}],
    }


def _accepted(zs: tuple[float, ...], b: float = 0.072, *, window=(-0.0035, 0.0135), ambiguous_last: bool = False) -> tuple[dict, dict]:
    cusps = [_cusp(f"wall-cusp-{i + 1:02d}", z, b, ambiguous=(ambiguous_last and i == len(zs) - 1)) for i, z in enumerate(zs)]
    rho = [_rho(c["cusp_id"], c["z_c_m"], b, 0.038, 0.09) for c in cusps]
    accepted = {
        "topology": {"wall_cusps": cusps, "wall_cusp_count": len(cusps), "cell_count": len(cusps) + 1},
        "axis_nulls": {"window_m": list(window), "nulls": [{"null_id": f"axis-null-{i + 1:02d}", "z_m": z, "zone": "channel", "classification": "X"} for i, z in enumerate(zs)] + [{"null_id": "axis-null-99", "z_m": -0.0018, "zone": "anode_side", "classification": "X"}]},
    }
    descriptors = {"cusps": rho, "min_rho_conservative": min(r["rho_conservative"] for r in rho), "hemp_like_all_cusps": all(r["hemp_like_conservative"] for r in rho)}
    return accepted, descriptors


def test_tolerance_rule_is_one_level0_bore_element_or_the_l1a_step(value: dict) -> None:
    assert E.cusp_position_tolerance_m(0.003, value) == pytest.approx(0.0004513888888888889)
    assert E.cusp_position_tolerance_m(0.0042, value) == pytest.approx(0.0042 / 8)
    assert value["comparison"]["l1a_dz_m"] == pytest.approx(0.065 / 144)


def test_agreeing_maps_confirm_and_report_ratios(value: dict) -> None:
    reference = _reference((0.0034, 0.0072))
    accepted, descriptors = _accepted((0.00355, 0.00715))
    result = E.compare_to_l1a(reference, accepted, descriptors, GEOMETRY, value, source_strength_scale=0.8)
    assert result["count_agreement_strict"] and result["count_agreement_boundary_tolerant"] and result["cell_count_agreement"]
    assert result["cusp_match"]["bijection"] and result["matched_cusp_count"] == 2
    assert result["max_cusp_shift_m"] == pytest.approx(0.00015) and result["position_gate_passed"]
    assert result["max_cusp_shift_over_tolerance"] == pytest.approx(0.00015 / result["cusp_position_tolerance_m"])
    pair = result["matched_cusps"][0]
    assert pair["wall_b_ratio_p2_over_l1a"] == pytest.approx(0.072 / 0.06) and pair["p2_wall_b_unscaled_t"] == pytest.approx(0.072 / 0.8)
    assert result["peak_wall_b_ratio_p2_over_l1a"] == pytest.approx(0.09 / 0.075) and result["peak_wall_b_ratio_unscaled"] == pytest.approx(0.09 / 0.8 / 0.075)
    assert result["axis_peak_b_ratio_p2_over_l1a"] == pytest.approx(0.038 / 0.035)
    assert result["hemp_like_preserved"] and result["p2_hemp_like_all_cusps"]
    assert result["channel_axis_null_match"]["bijection"] and not result["axis_null_match"]["bijection"]
    assert result["outside_channel_axis_nulls"]["shifts_m"] == [pytest.approx(0.0014)]
    assert result["channel_axis_nulls"]["sorted_shifts_m"] == [pytest.approx(0.00015), pytest.approx(0.00005)]
    assert result["separatrix_lean_m"]["l1a_max"] == 0.0 and result["separatrix_lean_m"]["p2_max"] == 0.0


def test_count_change_is_boundary_tolerant_only_near_the_straight_section_ends(value: dict) -> None:
    reference = _reference((0.0034, 0.0072))
    accepted, descriptors = _accepted((0.0034, 0.0072, 0.00885), ambiguous_last=True)
    result = E.compare_to_l1a(reference, accepted, descriptors, GEOMETRY, value, source_strength_scale=1.0)
    assert not result["count_agreement_strict"] and result["count_agreement_boundary_tolerant"]
    assert not result["cusp_match"]["bijection"] and not result["position_gate_passed"]
    assert result["unmatched_cusps"] == [{"side": "p2", "z_c_m": 0.00885, "near_straight_section_end": True}]
    interior, descriptors_interior = _accepted((0.0034, 0.0053, 0.0072))
    result = E.compare_to_l1a(reference, interior, descriptors_interior, GEOMETRY, value, source_strength_scale=1.0)
    assert not result["count_agreement_strict"] and not result["count_agreement_boundary_tolerant"]
    assert result["unmatched_cusps"][0]["near_straight_section_end"] is False


def test_large_shift_fails_the_position_gate_but_keeps_the_count(value: dict) -> None:
    reference = _reference((0.0034, 0.0072))
    accepted, descriptors = _accepted((0.0034, 0.0080))
    result = E.compare_to_l1a(reference, accepted, descriptors, GEOMETRY, value, source_strength_scale=1.0)
    assert result["count_agreement_strict"] and not result["cusp_match"]["bijection"] and result["matched_cusp_count"] == 1
    assert not result["position_gate_passed"]


def _records(comparisons: list[dict]) -> list[dict]:
    return [{"design_id": f"d{i}", "comparison": item} for i, item in enumerate(comparisons)]


def _comparison(*, bt: bool, strict: bool, bijection: bool, shifts: tuple[float, ...], hemp: bool = True) -> dict:
    tolerance = 0.0005
    return {
        "count_agreement_boundary_tolerant": bt,
        "count_agreement_strict": strict,
        "cusp_match": {"bijection": bijection},
        "matched_cusps": [{"shift_m": s, "shift_over_tolerance": s / tolerance, "rho_conservative_ratio_p2_over_l1a": 1.1, "wall_b_ratio_p2_over_l1a": 1.2} for s in shifts],
        "all_matched_within_tolerance": all(s <= tolerance for s in shifts),
        "cusp_position_tolerance_m": tolerance,
        "hemp_like_preserved": hemp,
        "p2_min_rho_conservative": 1.7,
        "l1a_min_rho_conservative": 1.6,
        "peak_wall_b_ratio_p2_over_l1a": 1.2,
        "peak_wall_b_ratio_unscaled": 1.4,
        "axis_peak_b_ratio_p2_over_l1a": 1.1,
        "peak_wall_b_ratio_in_band": True,
    }


def test_confirmation_verdicts_follow_the_predeclared_rule(value: dict) -> None:
    confirmed = E.confirmation_gates(_records([_comparison(bt=True, strict=True, bijection=True, shifts=(0.0001, 0.0002))] * 3), value)
    assert confirmed["cusp_count_unchanged"]["passed"] and confirmed["cusp_position_shift"]["passed"] and confirmed["verdict"] == "CONFIRMED"
    assert confirmed["cusp_position_shift"]["max_shift_over_tolerance"] == pytest.approx(0.4) and confirmed["cusp_position_shift"]["matched_cusp_count"] == 6
    assert confirmed["hemp_like_preserved"]["fraction"] == 1.0 and confirmed["hemp_like_preserved"]["passed"] is None
    partial = E.confirmation_gates(_records([_comparison(bt=True, strict=True, bijection=True, shifts=(0.0001,)), _comparison(bt=True, strict=True, bijection=True, shifts=(0.0009,))]), value)
    assert partial["cusp_count_unchanged"]["passed"] and not partial["cusp_position_shift"]["passed"] and partial["verdict"] == "PARTIALLY_CONFIRMED"
    assert partial["cusp_position_shift"]["designs_exceeding_tolerance"] == ["d1"]
    disconfirmed = E.confirmation_gates(_records([_comparison(bt=False, strict=False, bijection=False, shifts=(0.0001,), hemp=False)]), value)
    assert not disconfirmed["cusp_count_unchanged"]["passed"] and not disconfirmed["cusp_position_shift"]["passed"] and disconfirmed["verdict"] == "DISCONFIRMED"
    assert disconfirmed["hemp_like_preserved"]["lost_designs"] == ["d0"] and disconfirmed["cusp_count_unchanged"]["disagreeing_designs"] == ["d0"]
    boundary = E.confirmation_gates(_records([_comparison(bt=True, strict=False, bijection=False, shifts=(0.0001,))]), value)
    assert boundary["cusp_count_unchanged"]["passed"] and boundary["cusp_count_unchanged"]["fraction_strict"] == 0.0 and not boundary["cusp_position_shift"]["passed"]


def test_window_containment_uses_the_v3_1_margin() -> None:
    policy = TopologyPolicy()
    r = np.linspace(0.0, 0.003, 9)
    z = np.linspace(-0.01, 0.02, 61)
    grid = tracing_grid(r, z, np.zeros((9, 61)), np.zeros((9, 61)), np.ones((9, 61)), 0.003)
    margin = policy.axis_window_margin_mesh_factor * grid.mesh_scale_m
    assert E.window_contained(grid, (-0.01 + margin, 0.02 - margin), policy)
    assert not E.window_contained(grid, (-0.01 + 0.5 * margin, 0.02 - margin), policy)


def test_design_gate_checks_read_every_map_and_the_budget() -> None:
    clean_map = {"axis_nulls": {"all_converged": True, "all_x_type": True, "all_classifications_agree": True, "nulls": [{"v1_local_topology": {"jacobian_converged": True}}]}, "all_traces_terminate_cleanly": True, "all_wall_traces_flux_consistent": True}
    record = {
        "evidence": {"identity_proven": True, "p2": {"all_levels_converged": True, "level_count": 2, "levels": [{"allocation_preflight": {"passed": True}}, {"allocation_preflight": {"passed": True}}], "peak_rss_bytes": 100}, "ram_budget": {"budget_bytes": 1000}},
        "coarse": clean_map,
        "accepted": clean_map,
        "refined": clean_map,
        "sampling_stability": {"stable": True},
        "axis_window_reproduced": True,
    }
    assert all(E.design_gate_checks(record).values())
    broken = copy.deepcopy(record)
    broken["refined"]["all_wall_traces_flux_consistent"] = False
    broken["evidence"]["p2"]["peak_rss_bytes"] = 5000
    checks = E.design_gate_checks(broken)
    assert not checks["every_wall_trace_flux_consistent"] and not checks["ram_policy_respected"] and checks["solver_converged_all_levels"]
    assert set(checks) == set(E.GATE_NAMES)
