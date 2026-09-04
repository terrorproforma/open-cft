"""pic2d model v2.0.5 (performance, physics-bitwise): born-ledger fold, fused born deposit, sampled window moments.

* the born-ion kinetic energy and the born axial momentum are tile-reduced inside ``mcc_kernel`` (no strided sums / single-
  thread adds after the spawn): they equal the particle sums over the born ions / secondaries to round-off;
* the spawn deposits the born ion into the frozen ion charge itself: the int64 accumulator is bitwise the full redeposit;
* the step launches no ``energy_sum`` / ``momentum_sum`` / ``deferred_add`` / ``deposit_unit`` kernels any more;
* ``moment_sample_interval`` K: K = 1 leaves the configuration identity unchanged, K != 1 enters it; the dynamics and the
  per-step accumulators (n_e, n_i, phi, ionisation, fluxes) are bitwise independent of K on both backends; the moment sums
  are exactly the K = 1 per-step moments of the sampled steps; ``moment_samples`` counts them;
* the peak-Debye window statistic normalises the occupancy floor by the moment samples (fallback: steps) and records the
  sample count only for K-sampled windows; the window ring carries the sample count across accumulator resets;
* frames difference the sample count like the sums (pre-v2.0.5 snapshots fall back to steps);
* the runner reads ``numerics.performance.moment_sample_interval`` and requires it to divide the sync interval;
* CUDA only: the forked two-stream step graph replays the direct launches bitwise on the physics and to 1e-12 on the maps.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

warp = pytest.importorskip("warp")

from cft_revival.pic2d import artifacts, kernels  # noqa: E402
from cft_revival.pic2d import warp_backend as wb  # noqa: E402
from cft_revival.pic2d.fields import linear_psi_field_map  # noqa: E402
from cft_revival.pic2d.frames import interval_maps  # noqa: E402
from cft_revival.pic2d.kernels import FIXED_POINT_SCALE  # noqa: E402
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections  # noqa: E402
from cft_revival.pic2d.mesh import build_mesh_masks  # noqa: E402
from cft_revival.pic2d.models import (  # noqa: E402
    ELECTRON_MASS_KG,
    EV_J,
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PIC2DValidationError,
    PoissonConfig2D,
    StabilityLimits,
    electron_species,
    xenon_ion_species,
)
from cft_revival.pic2d.simulation import (  # noqa: E402
    PEAK_WINDOW_SUM_KEYS,
    DiagnosticAccumulator,
    InjectionConfig,
    PIC2DConfig,
    PeakDebyeWindow,
    SeedPlasmaConfig,
    Simulation,
    momentum_z_kg_m_s,
    window_peak_debye,
)
from cft_revival.pic2d.warp_backend import device_available  # noqa: E402
from experiments.pic2d_cft_steady_state_v1 import run as runner  # noqa: E402

MODERN = Path(__file__).resolve().parents[2]
V4_PROTOCOL = MODERN / "experiments" / "pic2d_cft_steady_state_v4" / "protocol.json"
# the v4 preregistration pin (tests/pic2d/test_pic2d_steady_state_v4.py): K = 1 must leave it untouched
V4_CONFIG_SHA256_CUDA = "f10772b25b030e2ba9d86cbd48b9972b9e143cce689e0b0ee8d4d6f10ccc685f"
CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
DEVICES = [device for device in ("cpu", "cuda:0") if device_available(device)]
CUDA = [device for device in DEVICES if device != "cpu"]
BACKENDS = ["cpu"] + (["warp-cpu"] if "cpu" in DEVICES else [])


def _backend_name(device: str) -> str:
    return "warp-cpu" if device == "cpu" else "warp-cuda"


def _config(grid: Grid2D, *, k: int = 1, injection: bool = True, frozen_ions: bool = False, dense: bool = False,
            series: int = 25) -> PIC2DConfig:
    return PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=3,
        injection=InjectionConfig(0.05, 2.0) if injection else None,
        seed_plasma=SeedPlasmaConfig(3e16 if dense else 1e16, 30.0 if dense else 5.0),
        mcc=MCCConfig(1e23 if dense else 1e22), poisson=PoissonConfig2D(method="direct"),
        reference_density_per_m3=3e16 if dense else 1e16, reference_electron_temperature_ev=30.0 if dense else 5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.0, max_collision_probability=0.9 if dense else 0.1),
        series_interval_steps=series, ion_subcycle=100_000 if frozen_ions else 1, moment_sample_interval=k,
    )


@pytest.fixture(scope="module")
def grid() -> Grid2D:
    return Grid2D(CFT_GEOMETRY, 12, 96)


@pytest.fixture(scope="module")
def field(grid):
    return linear_psi_field_map(grid, 2.0)


@pytest.fixture(scope="module")
def xs() -> XenonCrossSections:
    return XenonCrossSections.from_file()


# -- configuration identity ----------------------------------------------------------------------------------------------

def test_moment_sample_interval_enters_the_identity_only_when_it_is_not_one(grid):
    default = _config(grid)
    explicit = _config(grid, k=1)
    sampled = _config(grid, k=5)
    assert default.moment_sample_interval == 1 and "moment_sample_interval" not in default.to_dict()
    assert artifacts.config_identity(default) == artifacts.config_identity(explicit)
    assert sampled.to_dict()["moment_sample_interval"] == 5
    assert artifacts.config_identity(sampled) != artifacts.config_identity(default)
    for bad in (0, -1, True, 2.0):
        with pytest.raises(PIC2DValidationError):
            _config(grid, k=bad)     # type: ignore[arg-type]


def test_v4_production_identity_is_unchanged_and_the_performance_block_is_read_by_the_runner():
    protocol = runner.load_protocol(V4_PROTOCOL)
    config = runner.build_config(protocol, backend="warp-cuda")
    assert config.moment_sample_interval == 1 and artifacts.config_identity(config) == V4_CONFIG_SHA256_CUDA
    sampled = copy.deepcopy(protocol)
    sampled["numerics"]["performance"] = {"moment_sample_interval": 5}
    config_k5 = runner.build_config(sampled, backend="warp-cuda")
    assert config_k5.moment_sample_interval == 5 and artifacts.config_identity(config_k5) != V4_CONFIG_SHA256_CUDA
    # everything else of the physics configuration is untouched
    assert {k: v for k, v in config_k5.to_dict().items() if k != "moment_sample_interval"} == config.to_dict()
    sampled["numerics"]["performance"] = {"moment_sample_interval": 3}      # 200 % 3 != 0
    with pytest.raises(PIC2DValidationError):
        runner.build_config(sampled, backend="warp-cuda")


# -- born-ledger fold and fused born deposit (Warp backends) -------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_born_ledger_tallies_equal_the_particle_sums_over_the_ionisation_products(device, grid, field, xs):
    """One step with frozen ions and no injection: the born ions are the tail of the ion array and the secondaries the
    tail of the electron array, so the folded tallies can be checked against explicit particle sums (round-off only)."""

    config = _config(grid, injection=False, frozen_ions=True, dense=True, series=1)
    sim = Simulation(config, field, cross_sections=xs, backend=_backend_name(device), device=device)
    n_i0 = sim.state.ions.count
    sim.run(1)
    state = sim.state
    births = int(state.cumulative["ionizations"])
    assert births > 50 and state.ions.count == n_i0 + births
    ion, electron = xenon_ion_species(config.macro_weight), electron_species(config.macro_weight)
    born_ions = state.ions.select(np.arange(state.ions.count) >= n_i0)
    secondaries = state.electrons.select(np.arange(state.electrons.count) >= state.electrons.count - births)
    ke_ref = kernels.kinetic_energy_j(ion, born_ions)
    pz_ref = momentum_z_kg_m_s(ion, born_ions) + momentum_z_kg_m_s(electron, secondaries)
    assert state.cumulative["ke_born_ions_j"] == pytest.approx(ke_ref, rel=1e-12)
    assert state.cumulative["pz_born"] == pytest.approx(pz_ref, rel=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_born_deposit_keeps_the_ion_accumulator_bitwise_the_full_redeposit(device, grid, field, xs):
    config = _config(grid, injection=False, frozen_ions=True, dense=True, series=5)
    sim = Simulation(config, field, cross_sections=xs, backend=_backend_name(device), device=device)
    sim.run(10, accumulate_from_step=0)
    assert sim.state.cumulative["ionizations"] > 100
    backend = sim.backend
    ions = backend.species["i"]
    accumulator = warp.zeros(backend.node_count, dtype=warp.int64, device=backend.device)
    warp.launch(wb.deposit_fixed_kernel, dim=ions.bound,
                inputs=[ions.r, ions.z, ions.alive_flags, backend.slots, 1, grid.dr_m, grid.dz_m, grid.geometry.z_min_m,
                        backend.nr, backend.nz, FIXED_POINT_SCALE, accumulator], device=backend.device)
    warp.synchronize_device(backend.device)
    assert np.array_equal(backend.acc_i.numpy(), accumulator.numpy())
    # the ionisation-rate window map holds exactly one unit of weight per birth
    sums = sim.diagnostic_sums()
    assert float(sums["ionization"].sum()) == pytest.approx(sim.state.cumulative["ionizations"], rel=1e-12)


def test_the_step_launches_no_strided_born_reductions(grid, field, xs, monkeypatch):
    assert not hasattr(wb, "deferred_add_kernel") and not hasattr(wb, "deposit_unit_kernel")
    config = _config(grid, dense=True, series=5)
    sim = Simulation(config, field, cross_sections=xs, backend="warp-cpu", device="cpu")
    names: list[str] = []
    original = warp.launch

    def counting_launch(kernel, *args, **kwargs):
        names.append(getattr(kernel, "key", getattr(kernel, "__name__", str(kernel))))
        return original(kernel, *args, **kwargs)

    monkeypatch.setattr(warp, "launch", counting_launch)
    for _ in range(4):          # inside one sync interval: only step kernels (no series record, no sync)
        sim.backend.step(True)
    joined = " ".join(names)
    assert "spawn_kernel" in joined and "mcc_kernel" in joined and "deposit_moment_kernel" in joined
    for forbidden in ("energy_sum_kernel", "momentum_sum_kernel", "deferred_add", "deposit_unit"):
        assert forbidden not in joined, forbidden
    # born deposits fused into the spawn: exactly one fixed-point deposit per species per step (ions pushed every step here)
    assert sum("deposit_fixed_kernel" in name for name in names) == 2 * 4


# -- sampled window moments ----------------------------------------------------------------------------------------------

PER_STEP_KEYS = ("n_e", "n_i", "phi", "ionization", "wall_electrons", "wall_ions", "wall_electron_energy_j", "wall_ion_energy_j",
                 "exit_ions", "exit_electrons")
MOMENT_KEYS = ("e_weight", "e_vr", "e_vt", "e_vz", "e_v2")


@pytest.mark.parametrize("backend", BACKENDS)
def test_moment_sampling_leaves_the_dynamics_and_the_per_step_sums_bitwise(backend, grid, field, xs):
    device = "cpu"
    steps, k = 20, 3
    every = Simulation(_config(grid, k=1), field, cross_sections=xs, backend=backend, device=device)
    sampled = Simulation(_config(grid, k=k), field, cross_sections=xs, backend=backend, device=device)
    every.run(steps, accumulate_from_step=0)
    sampled.run(steps, accumulate_from_step=0)
    a, b = every.state, sampled.state
    for name in ("r_m", "z_m", "vr_m_per_s", "vt_m_per_s", "vz_m_per_s"):
        np.testing.assert_array_equal(getattr(a.electrons, name), getattr(b.electrons, name))
        np.testing.assert_array_equal(getattr(a.ions, name), getattr(b.ions, name))
    np.testing.assert_array_equal(a.phi_v, b.phi_v)
    assert a.cumulative == b.cumulative
    sa, sb = every.diagnostic_sums(), sampled.diagnostic_sums()
    for key in PER_STEP_KEYS:
        np.testing.assert_array_equal(sa[key], sb[key], err_msg=key)
    assert int(sa["steps"][0]) == int(sb["steps"][0]) == steps
    assert int(sa["moment_samples"][0]) == steps and int(sb["moment_samples"][0]) == -(-steps // k) == 7
    assert not np.array_equal(sa["e_weight"], sb["e_weight"]) and float(sb["e_weight"].sum()) < float(sa["e_weight"].sum())
    maps_a, maps_b = every.diagnostic_arrays(), sampled.diagnostic_arrays()
    for key in ("n_e_per_m3", "n_i_per_m3", "phi_v", "ionization_rate_per_m3_s", "wall_ion_flux_per_m2_s", "window_steps"):
        np.testing.assert_array_equal(maps_a[key], maps_b[key], err_msg=key)
    assert int(maps_b["moment_samples"][0]) == 7 and int(maps_a["moment_samples"][0]) == steps
    assert np.all(np.isfinite(maps_b["t_e_ev"])) and float(maps_b["sample_count_e"].sum()) < float(maps_a["sample_count_e"].sum())


def test_sampled_moment_sums_are_exactly_the_per_step_moments_of_the_sampled_steps(grid, field, xs):
    """CPU reference: accumulate the K = 1 per-step moment increments of the steps 0, K, 2K, ... in order and compare
    with the K-sampled run (bitwise: the same additions in the same order)."""

    steps, k = 12, 4
    every = Simulation(_config(grid, k=1), field, cross_sections=xs, backend="cpu")
    sampled = Simulation(_config(grid, k=k), field, cross_sections=xs, backend="cpu")
    expected = {key: np.zeros(grid.node_shape) for key in MOMENT_KEYS}
    for step in range(steps):
        every.run(1, accumulate_from_step=0)
        increment = every.diagnostic_sums()          # the accumulators were reset: exactly this step's deposit
        every.backend.reset_diagnostics()
        if step % k == 0:
            for key in MOMENT_KEYS:
                expected[key] = expected[key] + increment[key]
    sampled.run(steps, accumulate_from_step=0)
    sums = sampled.diagnostic_sums()
    assert int(sums["moment_samples"][0]) == 3
    for key in MOMENT_KEYS:
        np.testing.assert_array_equal(sums[key], expected[key], err_msg=key)


def test_window_reset_restarts_the_sampling_phase_and_the_sample_count(grid, field, xs):
    sim = Simulation(_config(grid, k=4), field, cross_sections=xs, backend="warp-cpu", device="cpu")
    sim.run(6, accumulate_from_step=0)
    assert int(sim.diagnostic_sums()["moment_samples"][0]) == 2          # steps 0 and 4
    sim.backend.reset_diagnostics()
    assert int(sim.diagnostic_sums()["moment_samples"][0]) == 0 and int(sim.diagnostic_sums()["steps"][0]) == 0
    sim.run(5, accumulate_from_step=0)
    assert int(sim.diagnostic_sums()["moment_samples"][0]) == 2          # accumulated steps 0 and 4 of the new window
    assert int(sim.diagnostic_sums()["steps"][0]) == 5


# -- window gate, window ring, frames ------------------------------------------------------------------------------------

def _synthetic_sums(grid: Grid2D, *, steps: int, moment_samples: int | None, occupancy: float, n_e: float = 1e17, t_e_ev: float = 5.0):
    masks = build_mesh_masks(grid)
    shape = grid.node_shape
    w = np.where(masks.plasma_node, occupancy * (moment_samples if moment_samples is not None else steps), 0.0)
    v_th2 = 3.0 * t_e_ev * EV_J / ELECTRON_MASS_KG
    sums = {"n_e": np.where(masks.plasma_node, n_e * steps, 0.0), "e_weight": w, "e_vr": np.zeros(shape), "e_vt": np.zeros(shape),
            "e_vz": np.zeros(shape), "e_v2": w * v_th2}
    if moment_samples is not None:
        sums["moment_samples"] = np.array([moment_samples])
    return masks, sums


def test_window_peak_debye_normalises_the_occupancy_floor_by_the_moment_samples(grid):
    config = _config(grid)
    masks, sums = _synthetic_sums(grid, steps=100, moment_samples=20, occupancy=25.0)
    out = window_peak_debye(masks, config, sums, 100, min_mean_occupancy=16.0)
    assert out["resolved"] and out["mean_macro_particles_at_peak"] == pytest.approx(25.0) and out["window_moment_samples"] == 20
    assert out["t_e_peak_ev"] == pytest.approx(5.0, rel=1e-9) and out["n_e_peak_per_m3"] == pytest.approx(1e17, rel=1e-12)
    # without the sample count the same sums are read as one sample per step (pre-v2.0.5 layout): 500 / 100 = 5 < 16
    legacy = {key: value for key, value in sums.items() if key != "moment_samples"}
    out_legacy = window_peak_debye(masks, config, legacy, 100, min_mean_occupancy=16.0)
    assert not out_legacy["resolved"] and out_legacy["mean_macro_particles_at_peak"] == pytest.approx(5.0)
    assert "window_moment_samples" not in out_legacy
    # K = 1 windows keep the v2.0.3 record layout
    _, every = _synthetic_sums(grid, steps=100, moment_samples=100, occupancy=25.0)
    assert "window_moment_samples" not in window_peak_debye(masks, config, every, 100, min_mean_occupancy=16.0)
    # zero samples in a non-empty window: unresolved, not a division by zero
    _, none = _synthetic_sums(grid, steps=3, moment_samples=0, occupancy=25.0)
    assert not window_peak_debye(masks, config, none, 3, min_mean_occupancy=16.0)["resolved"]


def test_peak_debye_window_carries_the_sample_count_across_accumulator_resets(grid):
    config = _config(grid)
    masks, _ = _synthetic_sums(grid, steps=1, moment_samples=1, occupancy=1.0)
    window = PeakDebyeWindow(masks, config, window_steps=40, snapshot_steps=10, min_mean_occupancy=16.0)

    def reading(steps: int, samples: int, generation: int):
        _, sums = _synthetic_sums(grid, steps=steps, moment_samples=samples, occupancy=20.0)
        return sums, steps, generation

    window.reset(reading(0, 0, 0), 0)
    out = window.update(reading(20, 4, 0), 20)
    assert out["window_steps"] == 20 and out["window_moment_samples"] == 4 and out["mean_macro_particles_at_peak"] == pytest.approx(20.0)
    out = window.update(reading(40, 8, 0), 40)
    assert out["window_complete"] and out["window_moment_samples"] == 8
    # the runner reset the accumulators (generation 1): the completed window is carried, the trailing window spans both
    out = window.update(reading(20, 4, 1), 60)
    assert out["window_steps"] >= 40 and out["window_moment_samples"] == out["window_steps"] // 5
    assert out["mean_macro_particles_at_peak"] == pytest.approx(20.0) and out["resolved"]
    # a pre-v2.0.5 reading without the count is treated as one sample per step
    sums, steps, generation = reading(30, 6, 1)
    legacy = ({key: value for key, value in sums.items() if key in PEAK_WINDOW_SUM_KEYS}, steps, generation)
    out = window.update(legacy, 70)
    assert out["window_steps"] > 0 and np.isfinite(out["mean_macro_particles_at_peak"])


def test_interval_maps_difference_the_sample_count_and_fall_back_for_old_snapshots(grid):
    config = _config(grid)
    masks = build_mesh_masks(grid)
    acc = DiagnosticAccumulator(masks)
    acc.steps, acc.moment_samples = 10, 2
    acc.n_e[masks.plasma_node] = 10.0 * 1e16
    acc.e_weight[masks.plasma_node] = 2.0 * 4.0
    acc.e_v2[masks.plasma_node] = acc.e_weight[masks.plasma_node] * 1e12
    start = acc.raw_sums()
    end = {key: 2.0 * value for key, value in start.items()}
    maps = interval_maps(end, start, masks, config.macro_weight, config.dt_s)
    assert int(maps["window_steps"][0]) == 10 and int(maps["moment_samples"][0]) == 2
    assert maps["n_e_per_m3"][masks.plasma_node] == pytest.approx(1e16)
    assert maps["t_e_ev"][masks.plasma_node] == pytest.approx(ELECTRON_MASS_KG * 1e12 / (3.0 * EV_J))
    # a snapshot pair recorded before v2.0.5 carries no sample count: one sample per step
    legacy_start = {key: value for key, value in start.items() if key != "moment_samples"}
    legacy_end = {key: value for key, value in end.items() if key != "moment_samples"}
    legacy = interval_maps(legacy_end, legacy_start, masks, config.macro_weight, config.dt_s)
    assert int(legacy["moment_samples"][0]) == 10
    # a frame without a moment sample (cadence shorter than the sampling interval, excluded by the runner's alignment rule)
    # reports zero samples and zero T_e / sample counts, and keeps the per-step maps
    no_sample = dict(end)
    no_sample["moment_samples"] = start["moment_samples"].copy()
    for key in ("e_weight", "e_vr", "e_vt", "e_vz", "e_v2"):
        no_sample[key] = start[key].copy()
    empty = interval_maps(no_sample, start, masks, config.macro_weight, config.dt_s)
    assert int(empty["moment_samples"][0]) == 0 and not np.any(empty["t_e_ev"]) and not np.any(empty["sample_count_e"])
    assert empty["n_e_per_m3"][masks.plasma_node] == pytest.approx(1e16)
    restored = DiagnosticAccumulator.from_sums(masks, start)
    assert restored.moment_samples == 2 and restored.steps == 10
    assert DiagnosticAccumulator.from_sums(masks, legacy_start).moment_samples == 10


# -- CUDA: two-stream step graph -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("device", CUDA)
@pytest.mark.parametrize("k", [1, 5])
def test_forked_step_graph_replays_the_direct_launches(device, k, grid, field, xs):
    """The window branch (density accumulators + sampled electron moments) runs on a second stream inside the captured
    step and joins before the push: physics bitwise the direct path, float-atomic maps to 1e-12, born ledger to 1e-12."""

    config = PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=3, injection=InjectionConfig(0.05, 2.0),
        seed_plasma=SeedPlasmaConfig(1e16, 5.0), mcc=MCCConfig(1e22), poisson=PoissonConfig2D(method="device-direct"),
        reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0, limits=StabilityLimits(max_cell_debye_ratio=2.0),
        series_interval_steps=25, device_sync_steps=25, ion_subcycle=4, moment_sample_interval=k,
    )
    direct = Simulation(config, field, cross_sections=xs, backend="warp-cuda", device=device, step_graph=False)
    graph = Simulation(config, field, cross_sections=xs, backend="warp-cuda", device=device, step_graph=True)
    direct.run(200, accumulate_from_step=0)
    graph.run(200, accumulate_from_step=0)
    assert graph.backend.step_graph_active and graph.backend.diagnostic_forks > 0 and direct.backend.diagnostic_forks == 0
    a, b = direct.state, graph.state
    for name in ("r_m", "z_m", "vr_m_per_s", "vt_m_per_s", "vz_m_per_s"):
        np.testing.assert_array_equal(getattr(a.electrons, name), getattr(b.electrons, name))
        np.testing.assert_array_equal(getattr(a.ions, name), getattr(b.ions, name))
    np.testing.assert_array_equal(a.phi_v, b.phi_v)
    np.testing.assert_array_equal(a.surface_charge_c, b.surface_charge_c)
    for key, value in a.cumulative.items():
        if key.startswith(("ke_", "pz_")) or key == "field_work_j":
            assert value == pytest.approx(b.cumulative[key], rel=1e-12, abs=1e-300), key      # float64 atomics (tile order)
        else:
            assert value == b.cumulative[key], key
    sa, sb = direct.diagnostic_sums(), graph.diagnostic_sums()
    assert int(sa["moment_samples"][0]) == int(sb["moment_samples"][0]) == -(-200 // k)
    for key in PER_STEP_KEYS:
        np.testing.assert_array_equal(sa[key], sb[key], err_msg=key)
    for key in MOMENT_KEYS:
        np.testing.assert_allclose(sa[key], sb[key], rtol=1e-12, atol=1e-12 * float(np.abs(sa[key]).max() or 1.0), err_msg=key)
