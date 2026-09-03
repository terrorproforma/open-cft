"""Protocol, catalogue binding, cell launch design, allocation rule, estimators (v2)."""

from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import asdict

import pytest

from cft_revival.orbit_mc import wilson_interval
from cft_revival.orbit_mc.artifacts import content_hash

from experiments.orbit_wall_loss_geometry_screening_v2 import cells as C
from experiments.orbit_wall_loss_geometry_screening_v2 import designs as D
from experiments.orbit_wall_loss_geometry_screening_v2 import experiment as E

CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
DESIGN_000 = "l1a-gs-v2-000-48d2ccedd5"
DESIGN_088 = "l1a-gs-v2-088-54d047707b"


@pytest.fixture(scope="module")
def value() -> dict:
    return E.protocol()


@pytest.fixture(scope="module")
def catalogue(value: dict) -> C.CatalogueBinding:
    return C.load_bound_catalogue(value["cusp_cell_catalogue"])


@pytest.fixture(scope="module")
def sweep(value: dict) -> D.SweepBinding:
    return D.load_sweep_binding(value["field_source"])


@pytest.fixture(scope="module")
def bound(value: dict, sweep: D.SweepBinding, catalogue: C.CatalogueBinding) -> dict:
    return E.bind_designs(value, sweep, catalogue, (DESIGN_000, DESIGN_088, D.P2_DESIGN_ID))


# --------------------------------------------------------------------------
# protocol structure
# --------------------------------------------------------------------------


def test_protocol_declares_the_screening_classification_and_boundary(value: dict) -> None:
    assert value["classification"] == CLASSIFICATION
    assert value["experiment_id"] == "orbit-wall-loss-geometry-screening-v2"
    boundary = value["claim_boundary"]
    assert boundary["not_accepted_physical_orbit_evidence"] is True
    assert boundary["not_p2_qualified"] is True
    assert boundary["p2_row_is_not_v4_replication"] is True
    assert boundary["forbid_plasma_performance_publication"] is True
    assert boundary["shakedown_outcomes_are_not_evidence"] is True
    assert "geometric wall-access fraction" in boundary["estimand"]
    assert value["orbit_mc_contract"]["package_version"] == "1.7.0"
    assert "known_defect_v1_7" in value["orbit_mc_contract"]
    assert value["allocation"]["wilson_width_threshold"] == 0.10
    assert value["allocation"]["stage1_launches_per_cell"] == 128 and value["allocation"]["stage2_launches_per_cell"] == 512
    assert "REJECTED" in value["allocation"]["rejected_alternative"]
    assert value["control"]["fraction_per_cell"] == 0.125 and value["control"]["maximum_paired_probability_change"] == 0.02
    assert value["estimators"]["surrogate_readiness_floor"] == 0.02
    assert value["field_source"]["refined_diagnostic"]["coverage"].startswith("EVERY sweep-v2 design (96/96)")
    assert len(value["designs"]["sweep_case_ids"]) == 96 == value["designs"]["sweep_design_count"]
    assert value["designs"]["p2_design"]["included"] is True
    assert value["execution"]["max_case_workers"] == 12
    assert value["execution"]["git_common_lock"] == "orbit-wall-loss-geometry-screening-v2.execution.lock"
    assert value["cases"]["case_sizes"] == {"block": 128, "control_of_stage1_cell": 16, "control_of_topped_up_cell": 64}


def test_plans_have_wilson_exact_case_sizes_and_unsafe_plans_are_refused(value: dict) -> None:
    plan = E.evidentiary_plan(value)
    assert plan.case_sizes(value) == value["cases"]["case_sizes"]
    assert plan.block_count == 4 and plan.block_range(0) == (0, 16) and plan.block_range(3) == (48, 64)
    shakedown = E.shakedown_plan(value)
    assert shakedown.case_sizes(value) == value["shakedown"]["case_sizes"]
    assert shakedown.block_count == 2
    assert len(plan.design_keys) == 97 and D.P2_DESIGN_ID in plan.design_keys
    assert len(shakedown.design_keys) == 4 and D.P2_DESIGN_ID in shakedown.design_keys
    for n in (128, 64, 16, 512 // 8, 2, 4):
        assert E.wilson_exact_at_ends(n)
    for n in (3, 6, 12, 13, 384, 512, 640, 768, 1536):
        assert not E.wilson_exact_at_ends(n)
    unsafe = copy.deepcopy(value)
    unsafe["launches"]["stage1_points_per_stratum"] = 48  # 384-launch blocks: k = 0 inexact
    unsafe["launches"]["stage2_points_per_stratum"] = 96
    with pytest.raises(ValueError, match="not Wilson-exact"):
        E.evidentiary_plan(unsafe)
    with pytest.raises(ValueError, match="whole number"):
        E.CampaignPlan(**{**asdict(plan), "stage2_points_per_stratum": 40})
    with pytest.raises(ValueError, match="own namespace"):
        E.CampaignPlan(**{**asdict(plan), "control_seed_namespace": plan.seed_namespace})


def test_allocation_rule_tops_up_exactly_the_wide_stage_one_cells(value: dict) -> None:
    rule = E.evidentiary_plan(value).allocation_rule(value)
    topped = [k for k in range(129) if C.allocation_decision({"cell-01": {"wall_hit": k, "trials": 128}}, rule)["cells"]["cell-01"]["topped_up"]]
    assert topped == list(range(12, 117))
    for k in (0, 11, 117, 128):
        assert C.wilson_width(k, 128) <= 0.10
    for k in (12, 64, 116):
        assert C.wilson_width(k, 128) > 0.10
    decision = C.allocation_decision({"cell-02": {"wall_hit": 64, "trials": 128}, "cell-01": {"wall_hit": 128, "trials": 128}}, rule)
    assert decision["topped_up_cell_ids"] == ["cell-02"] and decision["saturated_cell_count"] == 1
    assert decision["stage2_launch_count"] == 384
    with pytest.raises(ValueError, match="stage-1 trial count"):
        C.allocation_decision({"cell-01": {"wall_hit": 1, "trials": 64}}, rule)


def test_binomial_floors_and_readiness_rule(value: dict) -> None:
    assert C.binomial_floor(0, 512) == 0.0
    assert 0.0 < C.jeffreys_floor(0, 512) < 0.002
    assert abs(C.binomial_floor(256, 512) - math.sqrt(0.25 / 512)) < 1e-15
    assert C.jeffreys_floor(256, 512) > 0.02  # p ~ 0.5 at n = 512 is NOT ready by the frozen rule
    assert C.jeffreys_floor(30, 512) <= 0.02
    assert C.jeffreys_floor(64, 128) > 0.02


# --------------------------------------------------------------------------
# catalogue binding and cells
# --------------------------------------------------------------------------


def test_catalogue_is_bound_by_hash_and_tampered_declarations_are_refused(value: dict, catalogue: C.CatalogueBinding) -> None:
    declaration = value["cusp_cell_catalogue"]
    assert catalogue.file_sha256 == declaration["catalogue_file_sha256"]
    assert catalogue.manifest_file_sha256 == declaration["manifest_file_sha256"]
    assert catalogue.catalogue["design_count"] == 281 == catalogue.catalogue["stable_design_count"]
    for key in ("catalogue_file_sha256", "manifest_file_sha256", "protocol_semantic_sha256", "experiment_id"):
        bad = dict(declaration)
        bad[key] = "0" * 64 if key.endswith("sha256") else "cusp-topology-search-v3"
        with pytest.raises(ValueError, match="catalogue"):
            C.load_bound_catalogue(bad)
    sweep_entries = [entry for entry in catalogue.catalogue["entries"] if entry["set_id"] == "sweep_v2"]
    assert len(sweep_entries) == 96
    assert Counter(entry["cell_count"] for entry in sweep_entries) == {3: 30, 4: 47, 5: 19}
    assert sum(entry["cell_count"] for entry in sweep_entries) == 373
    assert Counter(cell["kind"] for entry in sweep_entries for cell in entry["cells"]) == {"anode_partial": 96, "interior": 181, "exit_partial": 96}


def test_cells_are_catalogue_cells_with_midpoint_launch_planes_and_flags(value: dict, bound: dict, catalogue: C.CatalogueBinding) -> None:
    design = bound[DESIGN_000]
    entry = C.catalogue_entry(catalogue, "sweep_v2", DESIGN_000)
    assert [cell.cell_id for cell in design.cells] == [cell["cell_id"] for cell in entry["cells"]] == ["cell-01", "cell-02", "cell-03"]
    assert [cell.kind for cell in design.cells] == ["anode_partial", "interior", "exit_partial"]
    assert [cell.position_class for cell in design.cells] == ["anode_side", "interior", "exit_side"]
    for cell, raw in zip(design.cells, entry["cells"], strict=True):
        assert cell.z_start_m == raw["z_start_m"] and cell.z_end_m == raw["z_end_m"]
        assert cell.launch_z_m == 0.5 * (raw["z_start_m"] + raw["z_end_m"])
        assert abs(cell.wall_area_m2 - 2.0 * math.pi * entry["geometry"]["wall_radius_m"] * raw["length_m"]) < 1e-15
        assert cell.launch_plane_inside_injector_zone is False and cell.short_cell is False
    interior = design.cells[1]
    assert interior.start_cusp_z_m == entry["wall_cusps"][0]["z_c_m"] and interior.end_cusp_z_m == entry["wall_cusps"][1]["z_c_m"]
    assert interior.launch_z_m == 0.5 * (interior.start_cusp_z_m + interior.end_cusp_z_m)
    small = bound[DESIGN_088]
    assert small.cells[0].launch_plane_inside_injector_zone is True and small.cells[0].short_cell is True
    assert small.cells[0].launch_z_m > small.design.domain_z_min_m
    p2 = bound[D.P2_DESIGN_ID]
    assert p2.design.label == D.LABEL_P2 and p2.design.set_id == D.SET_P2
    assert [cell.kind for cell in p2.cells] == ["anode_partial", "interior", "interior", "exit_partial"]
    assert p2.cells[3].short_cell is True and p2.cells[3].boundary_ambiguous is True
    assert p2.cells[3].length_m < 3e-5
    assert p2.design.straight_z_min_m == 0.001 and p2.design.straight_z_max_m == 0.018 and p2.design.domain_z_max_m == 0.023
    with pytest.raises(ValueError, match="launch plane rule"):
        C.design_cells(entry, injector_length_m=0.0, rule={**value["launches"], "launch_plane_rule": "other"})


# --------------------------------------------------------------------------
# launches
# --------------------------------------------------------------------------


def test_stage_one_block_is_stratified_and_scrambled_sobol_inside_the_bands(value: dict, bound: dict) -> None:
    plan = E.evidentiary_plan(value)
    design = bound[DESIGN_000]
    cell = design.cells[1]
    launches = E.block_launches(value, plan, design, cell, 0)
    assert len(launches) == 128
    campaign = E.campaign_id(plan, DESIGN_000, cell.cell_id, "stage1")
    assert campaign == f"owlgs-v2:{DESIGN_000}:cell-02:stage1:N"
    strata = Counter((item.kinetic_energy_ev, round(math.degrees(item.pitch_angle_rad), 9), item.parallel_direction) for item in launches)
    assert len(strata) == 8 and set(strata.values()) == {16}
    bands = Counter(item.flux_surface_id for item in launches)
    assert bands == {"cell-02-r0.675": 64, "cell-02-r0.800": 64}
    per_stratum_band = Counter((item.kinetic_energy_ev, item.pitch_angle_rad, item.parallel_direction, item.flux_surface_id) for item in launches)
    assert set(per_stratum_band.values()) == {8}
    wall = design.design.wall_radius_m
    for item in launches:
        assert item.launch_id.startswith(campaign + ":")
        key = C.key_of_launch(item)
        e, p, x, d, g = key.split(":")
        assert e in ("E0", "E1") and p in ("P0", "P1") and x == f"X{cell.index}" and d in ("D-1", "D+1") and g.startswith("G")
        assert 0 <= C.key_index(key) < 16 and C.key_cell_index(key) == cell.index
        assert item.seed_id == int.from_bytes(__import__("hashlib").sha256(item.launch_id.encode()).digest()[:8], "big")
        fraction = item.position_m[0] / wall
        assert item.position_m[1] == 0.0 and item.position_m[2] == cell.launch_z_m
        if item.flux_surface_id.endswith("r0.675"):
            assert 0.65 <= fraction < 0.70
        else:
            assert 0.775 <= fraction < 0.825
        assert 0.0 <= item.gyrophase_rad < 2.0 * math.pi
    assert len({item.gyrophase_rad for item in launches}) == 128
    assert len({item.position_m for item in launches}) == 128


def test_stage_two_blocks_extend_the_same_sequences_and_candidate_hash_is_stable(value: dict, bound: dict) -> None:
    plan = E.evidentiary_plan(value)
    design = bound[DESIGN_000]
    cell = design.cells[0]
    blocks = [E.block_launches(value, plan, design, cell, block) for block in range(4)]
    keys = [sorted(C.key_of_launch(item) for item in block) for block in blocks]
    for block, block_keys in enumerate(keys):
        assert all(16 * block <= C.key_index(key) < 16 * (block + 1) for key in block_keys)
    assert len(set().union(*map(set, keys))) == 512
    candidates = C.candidate_records(namespace=plan.seed_namespace, design_key=DESIGN_000, cells=design.cells, rule=plan.launch_rule(value), wall_radius_m=design.design.wall_radius_m)
    by_key = {row["key"]: row for row in candidates}
    assert len(by_key) == 3 * 512
    for block in blocks:
        for item in block:
            row = by_key[C.key_of_launch(item)]
            assert row["position_m"] == list(item.position_m) and row["gyrophase_rad"] == item.gyrophase_rad and row["flux_surface_id"] == item.flux_surface_id
    assert content_hash(candidates) == C.candidate_sha256(namespace=plan.seed_namespace, design_key=DESIGN_000, cells=design.cells, rule=plan.launch_rule(value), wall_radius_m=design.design.wall_radius_m)
    other = C.candidate_sha256(namespace=plan.seed_namespace, design_key=DESIGN_088, cells=bound[DESIGN_088].cells, rule=plan.launch_rule(value), wall_radius_m=bound[DESIGN_088].design.wall_radius_m)
    assert other != content_hash(candidates)


def test_control_selection_is_frozen_seeded_one_eighth_and_control_launches_rebuild(value: dict, bound: dict) -> None:
    plan = E.evidentiary_plan(value)
    design = bound[DESIGN_000]
    cell = design.cells[2]
    keys = E.final_keys(value, plan, cell, topped_up=False)
    assert len(keys) == 128
    selection = E.cell_control_selection(plan, DESIGN_000, cell, keys, "ceil")
    assert len(selection) == 16 and selection == sorted(selection) and set(selection) <= set(keys)
    assert selection == E.cell_control_selection(plan, DESIGN_000, cell, keys, "ceil")
    topped_keys = E.final_keys(value, plan, cell, topped_up=True)
    assert len(topped_keys) == 512
    topped_selection = E.cell_control_selection(plan, DESIGN_000, cell, topped_keys, "ceil")
    assert len(topped_selection) == 64 and set(topped_selection) <= set(topped_keys)
    other_cell = E.cell_control_selection(plan, DESIGN_000, design.cells[0], E.final_keys(value, plan, design.cells[0], False), "ceil")
    assert other_cell != selection
    control = E.control_launches(value, plan, design, cell, selection)
    assert sorted(C.key_of_launch(item) for item in control) == selection
    assert all(item.launch_id.startswith(f"owlgs-v2:{DESIGN_000}:cell-03:control:2N:") for item in control)
    stage1 = {C.key_of_launch(item): item for item in E.block_launches(value, plan, design, cell, 0)}
    for item in control:
        partner = stage1[C.key_of_launch(item)]
        assert partner.position_m == item.position_m and partner.gyrophase_rad == item.gyrophase_rad
        assert partner.launch_id != item.launch_id
    with pytest.raises(ValueError, match="rounding"):
        C.control_selection(namespace="x", design_key="d", cell_keys={"c": ["E0:P0:X0:D+1:G0"]}, fraction=0.125, rounding="floor")


def test_shakedown_launches_are_disjoint_from_the_evidentiary_launches(value: dict, bound: dict) -> None:
    report = E.shakedown_disjointness(value, {key: bound[key] for key in (DESIGN_000, D.P2_DESIGN_ID)} | {key: E.bind_designs(value, D.load_sweep_binding(value["field_source"]), C.load_bound_catalogue(value["cusp_cell_catalogue"]), (key,))[key] for key in E.shakedown_plan(value).design_keys if key not in (DESIGN_000, D.P2_DESIGN_ID)})
    assert report["proven"] is True
    assert set(report["reports"]["against_evidentiary_same_designs"]["overlap_counts"].values()) == {0}
    assert report["namespaces"]["distinct"] is True


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------


def _cell(index: int = 0, length: float = 0.004) -> C.LaunchCell:
    return C.LaunchCell(
        cell_id=f"cell-0{index + 1}", index=index, kind="interior", position_class="interior", z_start_m=0.0, z_end_m=length, length_m=length,
        launch_z_m=0.5 * length, launch_plane_inside_injector_zone=False, short_cell=False, wall_area_m2=2.0 * math.pi * 0.0014 * length,
        start_cusp_id="wall-cusp-01", end_cusp_id="wall-cusp-02", start_cusp_z_m=0.0, end_cusp_z_m=length, length_over_pitch=1.0,
        wall_mirror_ratio=1.0, axis_mirror_ratio=0.5, wall_b_min_t=0.05, cusp_wall_b_min_t=0.05, axis_bz_peak_t=0.1, boundary_ambiguous=False,
    )


def test_pooled_cell_row_and_design_pooling_arithmetic() -> None:
    cell = _cell()
    stage1 = {"trials": 128, "wall_hit": 64, "reflected": 20, "domain_escape": 44, "timeout": 0}
    stage2 = {"trials": 384, "wall_hit": 200, "reflected": 60, "domain_escape": 124, "timeout": 0}
    decision = {"stage1_wilson_width": C.wilson_width(64, 128), "topped_up": True, "saturated": False}
    control = {"n_control": 64, "wall_N": 33, "wall_2N": 33, "discordant": 0, "delta_p_wall": 0.0, "quantum": 1 / 64}
    row = C.pooled_cell_row(cell, stage1, stage2, decision, control, readiness_floor=0.02)
    assert row["final"]["trials"] == 512 and row["final"]["wall_hit"] == 264
    assert row["final"]["p_wall"] == asdict(wilson_interval(264, 512))
    assert row["final"]["p_reflected"]["successes"] == 80 and row["final"]["p_escape"]["successes"] == 168
    assert row["final"]["wilson_width"] == C.wilson_width(264, 512)
    assert row["final"]["surrogate_ready"] is (C.jeffreys_floor(264, 512) <= 0.02)
    assert row["stage2_only"]["p_wall"]["probability"] == 200 / 384
    saturated = C.pooled_cell_row(cell, {"trials": 128, "wall_hit": 128, "reflected": 0, "domain_escape": 0, "timeout": 0}, None, {"stage1_wilson_width": C.wilson_width(128, 128), "topped_up": False, "saturated": True}, None, readiness_floor=0.02)
    assert saturated["final"]["trials"] == 128 and saturated["final"]["binomial_floor"] == 0.0 and saturated["final"]["surrogate_ready"] is True
    with pytest.raises(ValueError, match="partition"):
        C.pooled_cell_row(cell, {"trials": 128, "wall_hit": 100, "reflected": 0, "domain_escape": 0, "timeout": 0}, None, decision, None, readiness_floor=0.02)
    rows = [row, C.pooled_cell_row(_cell(1, 0.002), {"trials": 128, "wall_hit": 128, "reflected": 0, "domain_escape": 0, "timeout": 0}, None, {"stage1_wilson_width": 0.0291, "topped_up": False, "saturated": True}, None, readiness_floor=0.02)]
    area = C.design_pooled(rows, weight="wall_area")
    launches = C.design_pooled(rows, weight="launches")
    p1, p2 = 264 / 512, 1.0
    assert abs(area["probability"] - (0.004 * p1 + 0.002 * p2) / 0.006) < 1e-15
    assert abs(launches["probability"] - (512 * p1 + 128 * p2) / 640) < 1e-15
    assert area["weights"] == [pytest.approx(2 / 3), pytest.approx(1 / 3)]
    assert 0.0 <= area["lower"] <= area["probability"] <= area["upper"] <= 1.0
    with pytest.raises(ValueError):
        C.design_pooled(rows, weight="equal")


def test_cell_counts_and_paired_control_from_terminations() -> None:
    cells = (_cell(0), _cell(1))
    strat = {"stratum_id": "E0:P0:D+1", "energy_index": 0, "pitch_index": 0, "parallel_direction": 1}
    terminations = {C.launch_key(0, strat, i): ("wall_hit" if i % 2 else "domain_escape") for i in range(8)}
    terminations.update({C.launch_key(1, strat, i): "reflected" for i in range(4)})
    counts = C.cell_counts_from_terminations(terminations, cells)
    assert counts["cell-01"] == {"trials": 8, "wall_hit": 4, "reflected": 0, "domain_escape": 4, "timeout": 0}
    assert counts["cell-02"] == {"trials": 4, "wall_hit": 0, "reflected": 4, "domain_escape": 0, "timeout": 0}
    control = {C.launch_key(0, strat, i): ("wall_hit" if i in (1, 3) else "domain_escape") for i in (0, 1, 3, 4)}
    paired = E.paired_control(terminations, control, 0.02, cells)
    assert paired["n_control"] == 4 and paired["discordant"] == 0 and paired["delta_p_wall"] == 0.0 and paired["passed"] is True
    flipped = dict(control)
    flipped[C.launch_key(0, strat, 0)] = "wall_hit"
    paired = E.paired_control(terminations, flipped, 0.02, cells)
    assert paired["discordant"] == 1 and paired["delta_p_wall"] == 0.25 and paired["passed"] is False
    with pytest.raises(ValueError, match="partner"):
        E.paired_control(terminations, {C.launch_key(0, strat, 99): "wall_hit"}, 0.02, cells)


def test_spearman_rank_correlation() -> None:
    assert E.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert E.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert E.spearman([1, 2], [1, 2]) is None
    assert E.spearman([1, 1, 1], [1, 2, 3]) is None
    assert E.spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) == pytest.approx(0.8207826816681233)


def test_stage_naming_and_case_keys() -> None:
    assert E.stage2_stage(1) == "stage2b1" and E.stage_block("stage2b3") == 3 and E.stage_block("stage1") == 0
    assert E.stage_timestep("stage1") == "N" and E.stage_timestep("stage2b2") == "N" and E.stage_timestep("control") == "2N"
    assert E.case_key("d", "cell-01", "control") == "d--cell-01--control-2N"
    with pytest.raises(ValueError):
        E.stage_block("control")
    with pytest.raises(ValueError):
        E.stage_timestep("bogus")
