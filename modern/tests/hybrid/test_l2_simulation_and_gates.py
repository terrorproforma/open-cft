"""L2 v2: the coupled simulation on a synthetic field, checkpoint v2 bindings and the GATE-L2 gate functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cft_revival.hybrid import gates
from cft_revival.hybrid.cells import synthetic_partition
from cft_revival.hybrid.checkpoint_v2 import load_checkpoint_v2, save_checkpoint_v2
from cft_revival.hybrid.l2 import (
    HybridL2Config,
    HybridL2Simulation,
    PlateauRule,
    evaluate_plateau,
    flux_function_from_field,
)
from cft_revival.hybrid.models import HybridValidationError
from cft_revival.hybrid.pb_solver import HybridConvergenceError
from cft_revival.pic2d.fields import linear_psi_field_map, uniform_field_map
from cft_revival.pic2d.mcc import XenonCrossSections
from cft_revival.pic2d.models import BoundaryPotentials, ChannelGeometry, Grid2D
from cft_revival.pic2d.neutrals import NeutralInventoryConfig

PIC_MAP_KEYS = {"n_e_per_m3", "n_i_per_m3", "phi_v", "t_e_ev", "ionization_rate_per_m3_s", "wall_electron_flux_per_m2_s", "wall_ion_flux_per_m2_s",
                "wall_electron_mean_energy_ev", "wall_ion_mean_energy_ev", "exit_ion_current_density_a_per_m2",
                "exit_electron_current_density_a_per_m2", "window_steps"}


def make_simulation(*, seed: int = 1, leak: tuple[float, ...] = (0.3e-3, 0.3e-3, 0.5e-3), injection: float = 3e-3, **overrides) -> HybridL2Simulation:
    grid = Grid2D(ChannelGeometry(0.002, 0.0, 0.024, 0.018, 0.003), 15, 120)
    field = linear_psi_field_map(grid, 30.0)
    xs = XenonCrossSections.synthetic_for_tests()
    part = synthetic_partition(0.0, 0.024, [0.006, 0.012, 0.018])
    kwargs = {"grid": grid, "potentials": BoundaryPotentials(300.0, 0.0), "dt_s": 1e-9, "macro_weight": 3e5, "seed": seed, "injection_current_a": injection,
                  "injection_temperature_ev": 2.0, "seed_density_per_m3": 5e16, "seed_electron_temperature_ev": 5.0, "neutral_ceiling_per_m3": 5.5e19,
                  "neutral_temperature_k": 300.0, "neutral_inventory": NeutralInventoryConfig(8.551102004120011e16, 3e-8),
                  "cusp_conductance_s": (3e-5, 5e-5, 1.5e-5), "leak_half_width_m": leak, "series_interval_steps": 5, "averaging_window_steps": 20,
                  "checkpoint_every_steps": 20, "residual_window_steps": 20, "max_steps": 10000}
    kwargs.update(overrides)
    return HybridL2Simulation(HybridL2Config(**kwargs), field, xs, part)


def test_config_invariants() -> None:
    with pytest.raises(HybridValidationError):
        make_simulation(leak=(0.3e-3,))
    with pytest.raises(HybridValidationError):
        make_simulation(averaging_window_steps=23)
    with pytest.raises(HybridValidationError):
        make_simulation(cusp_conductance_s=(3e-5, -1.0, 1e-5))


def test_flux_function_and_population() -> None:
    grid = Grid2D(ChannelGeometry(0.002, 0.0, 0.024, 0.018, 0.003), 15, 120)
    field = uniform_field_map(grid, 0.2)
    psi = flux_function_from_field(grid, field.b_z_t)
    assert np.allclose(psi[:, 0], 0.5 * 0.2 * grid.r_m**2)            # psi = B r^2 / 2 for uniform B_z
    sim_all = make_simulation(leak=())
    assert sim_all.populated_node.sum() == sim_all.masks.plasma_node.sum()
    sim = make_simulation()
    assert 0 < sim.populated_node.sum() < sim.masks.plasma_node.sum()
    assert np.all(np.isfinite(sim.population_threshold_wb))


def test_synthetic_run_conserves_charge_and_atoms_and_writes_pic_layout_maps() -> None:
    sim = make_simulation()
    for _ in range(40):
        sim.step()
    series = sim.series_arrays()
    assert series["step"].size == 8
    scale = 1.602176634e-19 * series["electrons"]
    assert np.all(np.abs(series["total_charge_identity_c"]) <= 1e-7 * scale)
    assert np.all(series["gauss_relative_residual"] <= 1e-7)
    assert np.all(series["constraint_residual_max"] <= 1e-7)
    ledger = sim.state.neutral.ledger
    v = sim.neutrals.volume_m3
    closure = ledger["fed"] - ledger["ionized"] - ledger["effused"] - ledger["artificial"] - v * (sim.state.neutral.density_per_m3 - sim.neutrals.initial_density)
    assert abs(closure) <= 1e-9 * v * sim.neutrals.initial_density
    maps, kind = sim.maps()
    assert kind == "window_average" and set(maps) == PIC_MAP_KEYS
    assert maps["phi_v"].shape == sim.config.grid.node_shape and maps["wall_ion_flux_per_m2_s"].shape == (120,)
    # the wall electron flux lives only on populated wall nodes (depleted flux tubes receive none)
    wall_flux = sim.last_field.wall_electron_flux_per_s
    depleted_wall = sim.wall_node & ~sim.populated_node
    assert depleted_wall.any() and np.all(wall_flux[depleted_wall] == 0.0)
    assert np.any(wall_flux[sim.wall_node & sim.populated_node] > 0.0)
    plateau = sim.plateau()
    assert plateau is not None and plateau["reached"] is False
    assert sim.windowed_residual_over_electrode_work()["complete"] is True


def test_energy_ledger_closes_after_the_initial_transient() -> None:
    sim = make_simulation()
    for _ in range(200):
        sim.step()
    series = sim.series_arrays()
    tail = slice(-10, None)
    residual = float(series["interval_residual_j"][tail].sum())
    work = float(series["interval_electrode_work_j"][tail].sum())
    assert work > 0.0
    assert abs(residual) <= 0.1 * work


def test_fail_closed_on_exhausted_electron_energy() -> None:
    sim = make_simulation()
    sim.step()
    sim.state.electron_energy_j[:] = 1e-30
    with pytest.raises(HybridConvergenceError):
        sim.step()


def test_checkpoint_v2_roundtrip_is_bitwise_and_bindings_fail_closed(tmp_path: Path) -> None:
    sim = make_simulation()
    for _ in range(12):
        sim.step()
    json_path, _ = save_checkpoint_v2(tmp_path, "ck", sim)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "hybrid-checkpoint-v2"
    assert metadata["field_source_sha256"] == sim.field.source_sha256 and (tmp_path / "ck.field.npz").is_file()
    twin = make_simulation()
    report = load_checkpoint_v2(json_path, twin)
    assert report["field"]["mode"] == "bitwise" and report["step"] == 12
    for _ in range(5):
        sim.step()
        twin.step()
    assert np.array_equal(sim.state.phi_v, twin.state.phi_v)
    assert np.array_equal(sim.state.ions.particles.r_m, twin.state.ions.particles.r_m)
    assert np.array_equal(sim.state.electron_count, twin.state.electron_count)
    # a different configuration, partition or field is refused
    with pytest.raises(HybridValidationError, match="configuration"):
        load_checkpoint_v2(json_path, make_simulation(seed=2))
    other_field = make_simulation()
    other_field.field = uniform_field_map(other_field.config.grid, 0.1)
    with pytest.raises(Exception, match="field"):
        load_checkpoint_v2(json_path, other_field)
    # tampered arrays are refused by the byte hash
    raw = (tmp_path / "ck.npz").read_bytes()
    (tmp_path / "ck.npz").write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(Exception, match="SHA-256"):
        load_checkpoint_v2(json_path, make_simulation())


def test_plateau_rule_matches_the_pic_definition() -> None:
    t = np.linspace(0.0, 8e-6, 400)
    flat = np.ones_like(t)
    rising = 1.0 + 0.5 * t / t[-1]
    rule = PlateauRule(3.0, 2.4e-6, 0.05, 0.2)
    assert evaluate_plateau(t, flat, flat, flat, rule)["reached"] is True
    assert evaluate_plateau(t, rising, flat, flat, rule)["reached"] is False
    assert evaluate_plateau(t[:100], flat[:100], flat[:100], flat[:100], rule)["reached"] is False   # < 3 transits


# -- gates ---------------------------------------------------------------------------------------------------------------------

def test_compare_and_code_comparison() -> None:
    assert gates.compare("a", 1.05, 1.0, 0.1).status == "within"
    assert gates.compare("a", 1.2, 1.0, 0.1).status == "outside"
    assert gates.compare("a", 1.2, 1.0, None).status == "not_compared"
    assert gates.compare("a", None, 1.0, 0.1).status == "not_compared"
    assert gates.compare("a", 1.0, 0.0, 0.1).status == "not_compared"
    block = gates.code_comparison([gates.compare("a", 1.05, 1.0, 0.1), gates.compare("b", 2.0, 1.0, 0.1), gates.compare("c", 1.0, 1.0, None)])
    assert block["passed"] is False and block["outside"] == ["b"] and block["not_compared"] == ["c"]
    assert gates.code_comparison([gates.compare("c", 1.0, 1.0, None)])["passed"] is False      # nothing compared -> not passed


def test_interface_conservation_fails_on_each_check() -> None:
    good = {"charge_identity_max_relative": 1e-9, "charge_identity_bound": 1e-7, "neutral_ledger_closure_relative": 1e-12, "neutral_ledger_bound": 1e-9,
                "windowed_energy_residual_ratio": 0.01, "energy_residual_bound": 0.05, "plateau_reached": True}
    assert gates.interface_conservation(**good)["passed"] is True
    for key, bad in (("charge_identity_max_relative", 1e-3), ("neutral_ledger_closure_relative", 1e-3), ("windowed_energy_residual_ratio", 0.2),
                     ("windowed_energy_residual_ratio", None), ("plateau_reached", False)):
        assert gates.interface_conservation(**{**good, key: bad})["passed"] is False


def test_levels_and_uncertainty_and_verdicts() -> None:
    rows = [{"label": "a", "finished": True, "quantities": {"x": 1.0}}, {"label": "b", "finished": True, "quantities": {"x": 1.1}},
            {"label": "c", "finished": False, "quantities": {}}]
    two = gates.levels_gate(rows, minimum=3, quantity_keys=["x"])
    assert two["passed"] is False and two["levels_completed"] == 2
    assert np.isclose(two["spread"]["x"]["max_relative_spread"], 0.1 / 1.1)
    three = gates.levels_gate(rows + [{"label": "d", "finished": True, "quantities": {"x": 1.2}}], minimum=3, quantity_keys=["x"])
    assert three["passed"] is True
    unc = gates.uncertainty_components(input_component={"value": 0.1}, numerical={"value": 0.05}, emulator={"statement": "none"},
                                       model_discrepancy={"value": 0.3})
    assert unc["reported"] is True and unc["names"] == ["emulator", "input", "model_discrepancy", "numerical"]
    assert gates.uncertainty_components(input_component={}, numerical={"value": 0.05}, emulator={"statement": "none"}, model_discrepancy={"value": 0.3})["reported"] is False
    conservation = {"passed": True}
    passed_levels = {"levels_completed": 3, "passed": True}
    failed_levels = {"levels_completed": 1, "passed": False}
    accepted = gates.evaluate_l2_gates(conservation=conservation, spatial=passed_levels, temporal=passed_levels, comparison={"passed": True},
                                       uncertainty=unc, failed_cases=0)
    assert accepted["verdict"] == "accepted" and accepted["metrics"]["spatial_levels"] == 3
    rejected = gates.evaluate_l2_gates(conservation=conservation, spatial=passed_levels, temporal=passed_levels, comparison={"passed": False},
                                       uncertainty=unc, failed_cases=0)
    assert rejected["verdict"] == "rejected_on_comparison"
    not_evaluable = gates.evaluate_l2_gates(conservation=conservation, spatial=failed_levels, temporal=passed_levels, comparison={"passed": True},
                                            uncertainty=unc, failed_cases=2)
    assert not_evaluable["verdict"] == "not_evaluable" and not_evaluable["metrics"]["failed_cases_count"] == 2
    assert gates.evaluate_l2_gates(conservation={"passed": False}, spatial=passed_levels, temporal=passed_levels, comparison={"passed": True},
                                   uncertainty=unc, failed_cases=0)["verdict"] == "not_evaluable"
