"""Warp backend parity against the numpy reference (Warp CPU and CUDA when available)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

warp = pytest.importorskip("warp")

from cft_revival.pic2d import artifacts, kernels  # noqa: E402
from cft_revival.pic2d.fields import linear_psi_field_map, uniform_field_map  # noqa: E402
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections  # noqa: E402
from cft_revival.pic2d.mesh import build_mesh_masks  # noqa: E402
from cft_revival.pic2d.models import (  # noqa: E402
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PoissonConfig2D,
    StabilityLimits,
)
from cft_revival.pic2d.poisson import Poisson2D  # noqa: E402
from cft_revival.pic2d.simulation import InjectionConfig, PIC2DConfig, SeedPlasmaConfig, Simulation  # noqa: E402
from cft_revival.pic2d.warp_backend import WarpPoisson, device_available, resolve_device  # noqa: E402

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
DEVICES = [device for device in ("cpu", "cuda:0") if device_available(device)]
if not DEVICES:
    pytest.skip("no Warp device available", allow_module_level=True)


def _backend_name(device: str) -> str:
    return "warp-cpu" if device == "cpu" else "warp-cuda"


def _config(grid: Grid2D, *, mcc: bool, injection: bool, poisson: PoissonConfig2D = PoissonConfig2D()) -> PIC2DConfig:
    return PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=3,
        injection=InjectionConfig(0.05, 2.0) if injection else None,
        seed_plasma=SeedPlasmaConfig(1e16, 5.0), mcc=MCCConfig(1e21) if mcc else None, poisson=poisson,
        reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=25,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_fixed_point_deposition_is_bitwise_identical(device: str):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = linear_psi_field_map(grid, 2.0)
    config = _config(grid, mcc=False, injection=False)
    cpu = Simulation(config, field, backend="cpu")
    gpu = Simulation(config, field, backend=_backend_name(device), device=device)
    state = cpu.state
    masks = build_mesh_masks(grid)
    reference = kernels.deposit_node_charge(masks, cpu.backend.electron, state.electrons, fixed_point=True)
    backend = gpu.backend
    species = backend.species["e"]
    backend._deposit(species, 0, backend.acc_e, backend.q_e, backend.electron.charge_c * config.macro_weight)
    warp.synchronize_device(backend.device)
    device_charge = backend.q_e.numpy().reshape(grid.node_shape)
    assert np.array_equal(device_charge, reference)
    assert device_charge.sum() == pytest.approx(backend.electron.charge_c * config.macro_weight * state.electrons.count, rel=1e-11)


@pytest.mark.parametrize("device", DEVICES)
def test_one_step_gather_push_parity_and_identical_field(device: str):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = linear_psi_field_map(grid, 2.0)
    config = _config(grid, mcc=False, injection=False)
    cpu = Simulation(config, field, backend="cpu")
    gpu = Simulation(config, field, backend=_backend_name(device), device=device)
    cpu.run(1)
    gpu.run(1)
    a, b = cpu.state, gpu.state
    # identical direct field solve on both backends
    assert np.array_equal(a.phi_v, b.phi_v)
    assert a.electrons.count == b.electrons.count and a.ions.count == b.ions.count
    for name, scale in (("r_m", 3e-3), ("z_m", 24e-3), ("vr_m_per_s", None), ("vt_m_per_s", None), ("vz_m_per_s", None)):
        x = getattr(a.electrons, name)
        y = getattr(b.electrons, name)
        reference = scale if scale is not None else float(np.max(np.abs(x)))
        assert np.max(np.abs(x - y)) <= 64.0 * np.finfo(float).eps * reference, name
    for key, value in a.cumulative.items():
        if key.startswith("ke_") or key == "field_work_j":
            assert value == pytest.approx(b.cumulative[key], rel=1e-9, abs=1e-300), key  # float64 atomics
        else:
            assert value == b.cumulative[key], key  # exact integer counts


@pytest.mark.parametrize("device", DEVICES)
def test_multi_step_collisionless_parity(device: str):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = linear_psi_field_map(grid, 2.0)
    config = _config(grid, mcc=False, injection=False)
    cpu = Simulation(config, field, backend="cpu")
    gpu = Simulation(config, field, backend=_backend_name(device), device=device)
    cpu.run(50)
    gpu.run(50)
    a, b = cpu.state, gpu.state
    assert a.electrons.count == b.electrons.count and a.ions.count == b.ions.count
    for key in ("anode_electrons", "exit_electrons", "wall_electrons", "anode_ions", "exit_ions", "wall_ions"):
        assert a.cumulative[key] == b.cumulative[key]
    assert np.max(np.abs(a.phi_v - b.phi_v)) < 1e-9 * 300.0
    assert np.max(np.abs(a.surface_charge_c - b.surface_charge_c)) <= 1e-9 * max(np.max(np.abs(a.surface_charge_c)), 1e-300)


@pytest.mark.parametrize("device", DEVICES)
def test_gpu_pcg_matches_cpu_direct_solve(device: str):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    masks = build_mesh_masks(grid)
    resolved = resolve_device(device)
    potentials = BoundaryPotentials(300.0, 0.0)
    solver = WarpPoisson(masks, potentials, PoissonConfig2D(method="pcg", relative_tolerance=1e-11), resolved, use_graph=(device != "cpu"))
    generator = np.random.default_rng(2)
    charge = np.zeros(grid.node_shape)
    charge[masks.unknown_node] = 1e-15 * generator.standard_normal(masks.unknown_count)
    zero = warp.zeros(solver.node_count, dtype=warp.float64, device=resolved)
    q_e = warp.array(charge.ravel(), dtype=warp.float64, device=resolved)
    phi = warp.zeros(solver.node_count, dtype=warp.float64, device=resolved)
    iterations, residual, tolerance = solver.solve(q_e, zero, zero, phi)
    assert residual <= tolerance and iterations > 0
    reference = Poisson2D(masks).solve(charge * masks.charge_to_source, potentials).phi_v
    assert np.max(np.abs(phi.numpy().reshape(grid.node_shape) - reference)) < 1e-8 * 300.0


@pytest.mark.parametrize("device", DEVICES)
def test_mcc_statistics_agree_in_distribution(device: str):
    """Different RNG streams: compare process counts with a chi-square test on one MCC step."""

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = uniform_field_map(grid, 0.05)
    xs = XenonCrossSections.from_file()
    config = PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(0.0, 0.0), dt_s=5e-12, macro_weight=4e5, seed=5,
        seed_plasma=SeedPlasmaConfig(4e17, 40.0), mcc=MCCConfig(5e21), reference_density_per_m3=4e17,
        reference_electron_temperature_ev=40.0, limits=StabilityLimits(max_cell_debye_ratio=4.0), series_interval_steps=100,
    )
    cpu = Simulation(config, field, cross_sections=xs, backend="cpu")
    gpu = Simulation(config, field, cross_sections=xs, backend=_backend_name(device), device=device)
    cpu.run(3)
    gpu.run(3)
    a, b = cpu.state.cumulative, gpu.state.cumulative
    observed = np.array([[a["elastic"], a["excitations"], a["ionizations"]], [b["elastic"], b["excitations"], b["ionizations"]]])
    assert observed.min() > 30, observed
    totals = observed.sum(axis=1, keepdims=True)
    expected = totals * observed.sum(axis=0, keepdims=True) / observed.sum()
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    # 2 degrees of freedom: 99.9% quantile is 13.8
    assert chi2 < 13.8, (observed, chi2)


@pytest.mark.parametrize("device", DEVICES)
def test_gpu_run_is_deterministic_and_checkpoint_resumable(device: str, tmp_path: Path):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = uniform_field_map(grid, 0.05)
    xs = XenonCrossSections.from_file()
    config = _config(grid, mcc=True, injection=True)
    first = Simulation(config, field, cross_sections=xs, backend=_backend_name(device), device=device)
    second = Simulation(config, field, cross_sections=xs, backend=_backend_name(device), device=device)
    first.run(45)
    second.run(25)
    json_path, _ = artifacts.save_checkpoint(
        tmp_path, "gpu", second.state, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256, backend=_backend_name(device)
    )
    loaded = artifacts.load_checkpoint(json_path, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256)
    third = Simulation(config, field, cross_sections=xs, backend=_backend_name(device), device=device)
    third.load_state(loaded)
    third.run(20)
    a, b = first.state, third.state
    assert a.step == b.step == 45
    for name in ("r_m", "z_m", "vr_m_per_s", "vt_m_per_s", "vz_m_per_s"):
        assert np.array_equal(getattr(a.electrons, name), getattr(b.electrons, name)), name
        assert np.array_equal(getattr(a.ions, name), getattr(b.ions, name)), name
    assert np.array_equal(a.surface_charge_c, b.surface_charge_c)
    assert np.array_equal(a.phi_v, b.phi_v)
    for key, value in a.cumulative.items():
        assert value == pytest.approx(b.cumulative[key], rel=1e-9, abs=1e-300), key
    assert a.cumulative["ionizations"] > 0 and a.cumulative["injected_electrons"] > 0


def _assert_same_state(a, b) -> None:
    for name in ("r_m", "z_m", "vr_m_per_s", "vt_m_per_s", "vz_m_per_s"):
        assert np.array_equal(getattr(a.electrons, name), getattr(b.electrons, name)), name
        assert np.array_equal(getattr(a.ions, name), getattr(b.ions, name)), name
    assert np.array_equal(a.surface_charge_c, b.surface_charge_c)
    assert np.array_equal(a.phi_v, b.phi_v)
    assert a.injection_carry == b.injection_carry
    for key, value in a.cumulative.items():
        # integer tallies are exact; the float sums are atomically reduced (round-off, as between two direct runs)
        assert value == pytest.approx(b.cumulative[key], rel=1e-9, abs=1e-300), key


@pytest.mark.skipif("cuda:0" not in DEVICES, reason="CUDA graphs need a CUDA device")
def test_cuda_graph_step_is_bitwise_identical_to_the_direct_launches_for_200_steps(tmp_path: Path):
    """v1.4 blocker-2 change: the whole step replayed as one CUDA graph equals the uncaptured step.

    Dynamical state (positions, velocities, surface charge, potential, injection carry) bitwise;
    integer tallies exact; ionisation, injection and ion sub-cycling all active; a resume through
    the graph path is also bitwise.
    """

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = uniform_field_map(grid, 0.05)
    xs = XenonCrossSections.from_file()
    config = PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=7,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e16, 5.0), mcc=MCCConfig(1e21),
        poisson=PoissonConfig2D(method="device-direct", relative_tolerance=1e-10),
        reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=25, ion_subcycle=4, device_sync_steps=25,
    )
    direct = Simulation(config, field, cross_sections=xs, backend="warp-cuda", step_graph=False)
    graph = Simulation(config, field, cross_sections=xs, backend="warp-cuda", step_graph=True)
    assert direct.backend.step_graph is False and graph.backend.step_graph is True
    direct.run(200)
    graph.run(200)
    assert graph.backend.step_graph_active and graph.backend.graph_captures >= 2      # ion-push and electron-only variants
    a, b = direct.state, graph.state
    assert a.step == b.step == 200 and a.cumulative["ionizations"] > 0 and a.cumulative["injected_electrons"] > 0
    _assert_same_state(a, b)
    assert [r.to_dict()["currents_a"] for r in direct.series] == pytest.approx([r.to_dict()["currents_a"] for r in graph.series], rel=1e-9)
    # a resume through the graph path from a mid-run checkpoint of the direct path is bitwise too
    half = Simulation(config, field, cross_sections=xs, backend="warp-cuda", step_graph=False)
    half.run(125)
    json_path, _ = artifacts.save_checkpoint(tmp_path, "g", half.state, config, field_sha256=field.sha256,
                                             cross_section_sha256=xs.payload_sha256, backend="warp-cuda")
    loaded = artifacts.load_checkpoint(json_path, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256)
    resumed = Simulation(config, field, cross_sections=xs, backend="warp-cuda", step_graph=True)
    resumed.load_state(loaded)
    resumed.run(75)
    _assert_same_state(a, resumed.state)
    assert graph.to_provenance()["v1_4_options"]["step_graph"] is True
