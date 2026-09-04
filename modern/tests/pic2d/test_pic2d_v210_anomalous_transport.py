"""Model v2.1.0: the anomalous cross-field transport closure (Bohm hook) audited and completed.

* the perpendicular-rotation event (Brandt et al. 2016) preserves |v| AND v_parallel to round-off, resets the gyro-phase
  uniformly (<v_perp' . v_perp> = 0) and shifts the guiding centre by <|dX|^2> = 2 r_L^2 - CPU reference and Warp kernel
  (warp-cpu, and CUDA when present), the two in agreement to round-off on the same input;
* the event rate is the exact Poisson probability 1 - exp(-alpha omega_ce dt) at the local |B| on both backends;
* DIFFUSION: test electrons gyrating in a uniform B with no E spread across the field at D_perp = (kT_e / eB) alpha / (1 + alpha^2)
  within statistics for alpha = 1/16 and 0.345, for BOTH event models (the isotropic v1.4 redirect and the rotation) - the closure
  reproduces the declared Bohm-type coefficient and the audit's exact factor (0.345 for Brandt's nu = 0.4 omega_ce);
* ENERGY: the kinetic energy of a scattered population is unchanged to round-off; in the simulation the interval ledger residual
  stays at the collisionless level with the hook on (no energy term is owed: the event is elastic);
* IDENTITY: alpha = 0 (hook absent) leaves ``config_sha256`` exactly the v2.0.6 value (the ss-v4 pin); alpha > 0 and the model
  name both enter the identity; the isotropic model's ``to_dict`` is the v1.4 record so every recorded protocol still resolves;
* the runner's ``build_config`` reads ``numerics.anomalous_collisions.model``; the Warp ``bohm_kernel`` in the simulation
  matches the CPU reference in distribution (event counts) and both tally ``cumulative["anomalous"]`` / ``pz_collisions``.
"""

from __future__ import annotations

import copy
import json
from math import pi, sqrt
from pathlib import Path

import numpy as np
import pytest
from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import uniform_field_map
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections
from cft_revival.pic2d.models import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    EV_J,
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PIC2DValidationError,
    PoissonConfig2D,
    StabilityLimits,
)
from cft_revival.pic2d.sensitivity import (
    ANOMALOUS_MODEL_ISOTROPIC,
    ANOMALOUS_MODEL_ROTATION,
    ANOMALOUS_MODELS,
    BOHM_ALPHA_SERIES,
    AnomalousCollisionConfig,
    apply_anomalous_scattering,
    apply_bohm_rotation,
    apply_bohm_scattering,
    bohm_collision_probability,
    bohm_diffusion_coefficient_m2_per_s,
    rotate_about_field,
)
from cft_revival.pic2d.simulation import InjectionConfig, PIC2DConfig, SeedPlasmaConfig, Simulation
from experiments.pic2d_cft_steady_state_v1 import run as runner

MODERN = Path(__file__).resolve().parents[2]
V4_PROTOCOL = MODERN / "experiments" / "pic2d_cft_steady_state_v4" / "protocol.json"
V4_CONFIG_SHA256 = "f10772b25b030e2ba9d86cbd48b9972b9e143cce689e0b0ee8d4d6f10ccc685f"     # the recorded ss-v4 identity (summary.json; warp-cuda build)
CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
OMEGA_PER_TESLA = ELEMENTARY_CHARGE_C / ELECTRON_MASS_KG


def _random_velocities_and_fields(rng: np.random.Generator, n: int, t_ev: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    v = rng.normal(0.0, sqrt(t_ev * EV_J / ELECTRON_MASS_KG), size=(3, n))
    angle = rng.uniform(0.0, 2.0 * pi, n)
    b_mag = rng.uniform(0.02, 0.5, n)
    b = np.vstack([b_mag * np.cos(angle), b_mag * np.sin(angle)])    # (b_r, b_z); B has no theta component in the axisymmetric field
    return v, b


def _parallel_and_perpendicular(v: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b_mag = np.hypot(b[0], b[1])
    nr, nz = b[0] / b_mag, b[1] / b_mag
    v_par = nr * v[0] + nz * v[2]
    perp = v - np.vstack([nr * v_par, np.zeros_like(v_par), nz * v_par])
    return v_par, perp


# ---------------------------------------------------------------------------------------------------------------- event model

def test_rotation_preserves_speed_and_parallel_velocity_and_turns_the_perpendicular_part_by_phi():
    rng = np.random.default_rng(11)
    v, b = _random_velocities_and_fields(rng, 100_000)
    phi = rng.uniform(0.0, 2.0 * pi, v.shape[1])
    new = np.vstack(rotate_about_field(v[0], v[1], v[2], b[0], b[1], phi))
    np.testing.assert_allclose(np.sqrt((new**2).sum(axis=0)), np.sqrt((v**2).sum(axis=0)), rtol=1e-12)
    par_before, perp_before = _parallel_and_perpendicular(v, b)
    par_after, perp_after = _parallel_and_perpendicular(new, b)
    np.testing.assert_allclose(par_after, par_before, rtol=1e-11, atol=1e-6 * np.abs(par_before).max())
    np.testing.assert_allclose(np.sqrt((perp_after**2).sum(axis=0)), np.sqrt((perp_before**2).sum(axis=0)), rtol=1e-11)
    # the perpendicular vector turned by exactly phi: cos(angle between) = cos(phi)
    cos_turn = (perp_after * perp_before).sum(axis=0) / (perp_before**2).sum(axis=0)
    np.testing.assert_allclose(cos_turn, np.cos(phi), atol=1e-9)
    # |B| = 0 is a no-op (the selection probability is zero there anyway)
    zero = np.vstack(rotate_about_field(v[0, :5], v[1, :5], v[2, :5], np.zeros(5), np.zeros(5), phi[:5]))
    assert np.array_equal(zero, v[:, :5])


def test_rotation_event_rate_gyro_phase_reset_and_guiding_centre_step():
    rng = np.random.default_rng(3)
    n = 400_000
    t_ev = 5.0
    v = rng.normal(0.0, sqrt(t_ev * EV_J / ELECTRON_MASS_KG), size=(3, n))
    b_r, b_z = np.full(n, 0.3), np.full(n, 0.4)     # |B| = 0.5 T at a 37 deg tilt (a cusp-wall field; P = 0.042 -> ~17 000 events)
    dt, alpha = 1.4e-12, 0.345
    p = float(bohm_collision_probability(alpha, 0.5, dt))
    assert p == pytest.approx(-np.expm1(-alpha * OMEGA_PER_TESLA * 0.5 * dt))
    vr, vt, vz, count = apply_bohm_rotation(alpha, v[0], v[1], v[2], b_r, b_z, dt, rng)
    assert abs(count - n * p) <= 5.0 * sqrt(n * p)
    new = np.vstack([vr, vt, vz])
    changed = np.any(new != v, axis=0)
    assert int(changed.sum()) == count
    b = np.vstack([b_r, b_z])
    _, perp_before = _parallel_and_perpendicular(v[:, changed], b[:, changed])
    _, perp_after = _parallel_and_perpendicular(new[:, changed], b[:, changed])
    # uniform gyro-phase reset: <v_perp' . v_perp> / <v_perp^2> = <cos phi> = 0
    correlation = float((perp_after * perp_before).sum() / (perp_before**2).sum())
    assert abs(correlation) < 5.0 / sqrt(count)
    # guiding-centre step |dX| = |dv_perp| / omega_ce: <|dX|^2> = 2 <r_L^2> = 2 <v_perp^2> / omega_ce^2 (the random-walk step of D);
    # the sample mean of 2 v_perp^2 (1 - cos phi) has a relative sd of sqrt(2 / count) ~ 1.1 %
    omega = OMEGA_PER_TESLA * 0.5
    step2 = ((perp_after - perp_before) ** 2).sum(axis=0) / omega**2
    r_l2 = (perp_before**2).sum(axis=0) / omega**2
    assert float(step2.mean()) == pytest.approx(2.0 * float(r_l2.mean()), rel=5.0 * sqrt(2.0 / count))
    # kinetic energy of the population unchanged to round-off (elastic event)
    assert float((new**2).sum()) == pytest.approx(float((v**2).sum()), rel=1e-13)


def test_isotropic_model_draws_exactly_as_v1_4_and_the_dispatcher_selects_by_model():
    v, b = _random_velocities_and_fields(np.random.default_rng(1), 20_000)
    rng_a, rng_b = np.random.default_rng(9), np.random.default_rng(9)
    iso = AnomalousCollisionConfig(1.0 / 16.0)
    assert iso.model == ANOMALOUS_MODEL_ISOTROPIC and not iso.rotation
    a = apply_anomalous_scattering(iso, v[0], v[1], v[2], b[0], b[1], 5e-12, rng_a)
    ref = apply_bohm_scattering(1.0 / 16.0, v[0], v[1], v[2], np.hypot(b[0], b[1]), 5e-12, rng_b)
    assert a[3] == ref[3] > 0
    for x, y in zip(a[:3], ref[:3], strict=True):
        assert np.array_equal(x, y)
    rot = AnomalousCollisionConfig(1.0 / 16.0, model=ANOMALOUS_MODEL_ROTATION)
    assert rot.rotation and rot.to_dict() == {"model": ANOMALOUS_MODEL_ROTATION, "alpha": 1.0 / 16.0}
    assert iso.to_dict() == {"model": "bohm_isotropic_scattering", "alpha": 1.0 / 16.0}     # the v1.4 record, byte for byte
    assert rot.diffusion_factor == pytest.approx((1.0 / 16.0) / (1.0 + 1.0 / 256.0))
    assert AnomalousCollisionConfig(0.4).diffusion_factor == pytest.approx(0.4 / 1.16) and 0.4 / 1.16 == pytest.approx(0.345, abs=5e-4)
    assert BOHM_ALPHA_SERIES == (1.0 / 64.0, 1.0 / 16.0, 0.345) and set(ANOMALOUS_MODELS) == {ANOMALOUS_MODEL_ISOTROPIC, ANOMALOUS_MODEL_ROTATION}
    with pytest.raises(PIC2DValidationError, match="model"):
        AnomalousCollisionConfig(0.1, model="bohm_random_walk")
    with pytest.raises(PIC2DValidationError):
        AnomalousCollisionConfig(0.0, model=ANOMALOUS_MODEL_ROTATION)


# ------------------------------------------------------------------------------------------------------------- diffusion test

def _measure_cross_field_diffusion(alpha: float, model: str, *, b_t: float = 0.05, t_ev: float = 5.0, n: int = 12_000,
                                   omega_dt: float = 0.1, collision_times: float = 30.0, seed: int = 21) -> float:
    """Test electrons gyrating in a uniform B = B z-hat with no E; the hook is applied per step; returns D from the 2-D MSD.

    Exact gyration per step (rotation of the perpendicular velocity by omega dt), positions advanced with the mid-step
    velocity; MSD_2D(t) = <x^2 + y^2> = 4 D t once t >> 1 / nu.  Fitted over the second half of the run.
    """

    rng = np.random.default_rng(seed)
    omega = OMEGA_PER_TESLA * b_t
    dt = omega_dt / omega
    nu = alpha * omega
    steps = int(collision_times / (nu * dt))
    v_th = sqrt(t_ev * EV_J / ELECTRON_MASS_KG)
    vx, vy, vz = (rng.normal(0.0, v_th, n) for _ in range(3))
    x = np.zeros(n)
    y = np.zeros(n)
    config = AnomalousCollisionConfig(alpha, model=model)
    b_r = np.zeros(n)
    b_z = np.full(n, b_t)
    times: list[float] = []
    msd: list[float] = []
    c2, s2 = np.cos(0.5 * omega * dt), np.sin(0.5 * omega * dt)
    for k in range(steps):
        # half rotation - drift - half rotation (velocity Verlet for a pure gyration; exact to round-off)
        vx, vy = c2 * vx + s2 * vy, -s2 * vx + c2 * vy
        x += vx * dt
        y += vy * dt
        vx, vy = c2 * vx + s2 * vy, -s2 * vx + c2 * vy
        # the perpendicular plane of B = B z-hat is (x, y) = (r, theta) of the hook's (v_r, v_theta, v_z) triad
        vx, vy, vz, _ = apply_anomalous_scattering(config, vx, vy, vz, b_r, b_z, dt, rng)
        if k >= steps // 2 and k % 20 == 0:
            times.append((k + 1) * dt)
            msd.append(float(np.mean(x * x + y * y)))
    slope = float(np.polyfit(times, msd, 1)[0])
    return slope / 4.0


@pytest.mark.parametrize("alpha", [1.0 / 16.0, 0.345])
@pytest.mark.parametrize("model", ANOMALOUS_MODELS)
def test_cross_field_diffusion_coefficient_is_alpha_over_one_plus_alpha_squared_times_kt_over_eb(alpha: float, model: str):
    b_t, t_ev = 0.05, 5.0
    expected = bohm_diffusion_coefficient_m2_per_s(alpha, t_ev, b_t)
    assert expected == pytest.approx(t_ev / b_t * alpha / (1.0 + alpha**2))
    # the walker count / run length are sized per alpha so that each case costs a few seconds: the MSD slope over
    # N independent walkers carries a relative sd of order sqrt(2 / N) ~ 0.6-1 %
    if alpha > 0.3:
        measured = _measure_cross_field_diffusion(alpha, model, b_t=b_t, t_ev=t_ev, n=60_000, omega_dt=0.05, collision_times=60.0)
    else:
        measured = _measure_cross_field_diffusion(alpha, model, b_t=b_t, t_ev=t_ev, n=24_000, omega_dt=0.1, collision_times=40.0)
    assert measured == pytest.approx(expected, rel=0.05), (alpha, model, measured, expected, measured / expected)
    # and it is NOT the naive alpha kT/eB once alpha is not small (the 0.345 factor of the audit for Brandt's nu = 0.4 omega_ce:
    # 0.345 / 1.119 = 0.308 of kT/eB, 10.6 % below the naive 0.345)
    if alpha > 0.3:
        assert measured < 0.95 * t_ev / b_t * alpha


# ------------------------------------------------------------------------------------------------------- simulation + Warp

def _config(grid: Grid2D, **overrides) -> PIC2DConfig:
    base = {
        "grid": grid, "potentials": BoundaryPotentials(300.0, 0.0), "dt_s": 5e-12, "macro_weight": 2e6, "seed": 3,
        "injection": InjectionConfig(0.05, 2.0), "seed_plasma": SeedPlasmaConfig(1e16, 5.0), "mcc": MCCConfig(1e21),
        "poisson": PoissonConfig2D(method="direct"), "reference_density_per_m3": 1e16, "reference_electron_temperature_ev": 5.0,
        "limits": StabilityLimits(max_cell_debye_ratio=4.0), "series_interval_steps": 20,
    }
    return PIC2DConfig(**(base | overrides))


def test_rotation_model_in_the_cpu_simulation_is_tallied_ledgered_and_checkpointed(tmp_path: Path):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    field = uniform_field_map(grid, 0.05)
    config = _config(grid, anomalous=AnomalousCollisionConfig(0.345, model=ANOMALOUS_MODEL_ROTATION))
    sim = Simulation(config, field, cross_sections=xs)
    sim.run(20)
    hits = sim.state.cumulative["anomalous"]
    n_e = sim.state.electrons.count
    expected = 20 * n_e * (1.0 - np.exp(-0.345 * OMEGA_PER_TESLA * 0.05 * config.dt_s))
    assert abs(hits - expected) <= 5.0 * sqrt(expected) + 0.05 * expected
    assert sim.series[-1].currents_a["anomalous_collision_rate_per_s"] == pytest.approx(hits * config.macro_weight / (20 * config.dt_s))
    assert config.to_dict()["anomalous"] == {"model": ANOMALOUS_MODEL_ROTATION, "alpha": 0.345}
    assert sim.to_provenance()["config"]["anomalous"]["model"] == ANOMALOUS_MODEL_ROTATION
    # elastic: no energy term; the interval residual stays at the collisionless level
    assert abs(sim.series[-1].ledger["interval_residual_j"]) < 1e-3 * abs(sim.series[-1].ledger["total_energy_j"])
    # the momentum handed to the turbulent field is tallied (nonzero; there are MCC elastic events too, so only presence is asserted)
    assert "pz_collisions" in sim.state.cumulative
    json_path, _ = artifacts.save_checkpoint(tmp_path, "rot", sim.state, config, field_sha256=field.sha256,
                                             cross_section_sha256=xs.payload_sha256, backend="cpu")
    loaded = artifacts.load_checkpoint(json_path, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256)
    assert loaded.cumulative["anomalous"] == hits


def _warp_devices() -> list[str]:
    warp = pytest.importorskip("warp")
    from cft_revival.pic2d.warp_backend import device_available
    del warp
    return [device for device in ("cpu", "cuda:0") if device_available(device)]


@pytest.mark.parametrize("device", _warp_devices() or [pytest.param("none", marks=pytest.mark.skip("no Warp device"))])
def test_warp_bohm_kernel_rotation_preserves_speed_and_parallel_velocity_and_matches_the_rate(device: str):
    import warp as wp
    from cft_revival.pic2d import warp_backend as wb

    rng = np.random.default_rng(17)
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    n = 200_000
    t_ev = 5.0
    v = rng.normal(0.0, sqrt(t_ev * EV_J / ELECTRON_MASS_KG), size=(3, n))
    r = rng.uniform(0.2e-3, 1.8e-3, n)
    z = rng.uniform(1e-3, 17e-3, n)
    b_r_node = np.full(grid.node_shape, 0.3).ravel()      # |B| = 0.5 T at a 37 deg tilt: P = 0.042 -> ~8 400 events
    b_z_node = np.full(grid.node_shape, 0.4).ravel()
    dt, alpha = 1.4e-12, 0.345
    p = float(bohm_collision_probability(alpha, 0.5, dt))
    dev = wp.get_device(device)
    arrays = {name: wp.array(x.copy(), dtype=wp.float64, device=dev) for name, x in (("r", r), ("z", z), ("vr", v[0]), ("vt", v[1]), ("vz", v[2]))}
    alive = wp.array(np.ones(n, dtype=np.int32), dtype=wp.int32, device=dev)
    slots = wp.array(np.array([n, 0], dtype=np.int32), dtype=wp.int32, device=dev)
    seed_table = wp.array(np.array([wb.stream_seed(3, 0, s) for s in range(wb.SEED_STREAMS)], dtype=np.int32), dtype=wp.int32, device=dev)
    counter = wp.zeros(1, dtype=wp.int32, device=dev)
    stats = wp.zeros(wb.STATS_SIZE, dtype=wp.float64, device=dev)
    mass_weight = ELECTRON_MASS_KG * 2e6
    wp.launch(wb.bohm_kernel, dim=wb.padded_dim(n, wb.PARTICLE_BLOCK), block_dim=wb.PARTICLE_BLOCK,
              inputs=[arrays["r"], arrays["z"], arrays["vr"], arrays["vt"], arrays["vz"], alive, slots, seed_table, counter,
                      wp.array(b_r_node, dtype=wp.float64, device=dev), wp.array(b_z_node, dtype=wp.float64, device=dev),
                      grid.dr_m, grid.dz_m, grid.geometry.z_min_m, grid.cell_shape[0], grid.cell_shape[1],
                      alpha * dt * OMEGA_PER_TESLA, stats, wb.STATS_ANOMALOUS, mass_weight, 1],
              device=dev)
    wp.synchronize_device(dev)
    new = np.vstack([arrays["vr"].numpy(), arrays["vt"].numpy(), arrays["vz"].numpy()])
    host_stats = stats.numpy()
    count = int(host_stats[wb.STATS_ANOMALOUS])
    assert abs(count - n * p) <= 5.0 * sqrt(n * p)
    changed = np.any(new != v, axis=0)
    assert int(changed.sum()) == count > 1000
    np.testing.assert_allclose(np.sqrt((new**2).sum(axis=0)), np.sqrt((v**2).sum(axis=0)), rtol=1e-12)
    b = np.vstack([np.full(n, 0.3), np.full(n, 0.4)])
    par_before, perp_before = _parallel_and_perpendicular(v, b)
    par_after, perp_after = _parallel_and_perpendicular(new, b)
    np.testing.assert_allclose(par_after, par_before, rtol=1e-11, atol=1e-6 * np.abs(par_before).max())
    correlation = float((perp_after[:, changed] * perp_before[:, changed]).sum() / (perp_before[:, changed] ** 2).sum())
    assert abs(correlation) < 5.0 / sqrt(count)
    # the axial momentum tally equals m W sum(dv_z)
    assert host_stats[wb.STATS_PZ_COLLISIONS] == pytest.approx(mass_weight * float((new[2] - v[2]).sum()), rel=1e-9, abs=1e-30)
    # the device rotation equals the CPU reference on the SAME angles: recover phi from the turned perpendicular vector and re-apply
    idx = np.flatnonzero(changed)[:5000]
    cos_phi = (perp_after[:, idx] * perp_before[:, idx]).sum(axis=0) / (perp_before[:, idx] ** 2).sum(axis=0)
    nr, nz = 0.3 / 0.5, 0.4 / 0.5
    cross = np.vstack([-nz * perp_before[1, idx], nz * perp_before[0, idx] - nr * perp_before[2, idx], nr * perp_before[1, idx]])
    sin_phi = (perp_after[:, idx] * cross).sum(axis=0) / (perp_before[:, idx] ** 2).sum(axis=0)
    phi = np.arctan2(sin_phi, cos_phi)
    ref = np.vstack(rotate_about_field(v[0, idx], v[1, idx], v[2, idx], b[0, idx], b[1, idx], phi))
    np.testing.assert_allclose(ref, new[:, idx], rtol=1e-9, atol=1e-9 * np.abs(v).max())


@pytest.mark.parametrize("device", _warp_devices() or [pytest.param("none", marks=pytest.mark.skip("no Warp device"))])
def test_simulation_rotation_model_cpu_and_warp_agree_in_distribution(device: str):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    field = uniform_field_map(grid, 0.05)
    config = _config(grid, anomalous=AnomalousCollisionConfig(0.345, model=ANOMALOUS_MODEL_ROTATION), device_sync_steps=20)
    cpu = Simulation(config, field, cross_sections=xs, backend="cpu")
    gpu = Simulation(config, field, cross_sections=xs, backend="warp-cpu" if device == "cpu" else "warp-cuda", device=device)
    cpu.run(20)
    gpu.run(20)
    a, b = cpu.state.cumulative["anomalous"], gpu.state.cumulative["anomalous"]
    assert a > 200 and b > 200        # ~2.9e5 electrons x 20 steps x P(0.05 T) = 0.0015 -> ~500 events each
    assert abs(a - b) <= 5.0 * sqrt(a + b) + 0.05 * (a + b)      # different RNG streams; counts drift with N_e over the 20 steps
    assert gpu.series[-1].currents_a["anomalous_collision_rate_per_s"] == pytest.approx(b * config.macro_weight / (20 * config.dt_s))
    assert abs(gpu.series[-1].ledger["interval_residual_j"]) < 1e-3 * abs(gpu.series[-1].ledger["total_energy_j"])
    assert gpu.to_provenance()["config"]["anomalous"] == {"model": ANOMALOUS_MODEL_ROTATION, "alpha": 0.345}


@pytest.mark.skipif("cuda:0" not in _warp_devices(), reason="CUDA graphs need a CUDA device")
def test_cuda_graph_step_with_the_rotation_hook_is_bitwise_identical_to_the_direct_launches():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = uniform_field_map(grid, 0.05)
    xs = XenonCrossSections.from_file()
    config = PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=7,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e16, 5.0), mcc=MCCConfig(1e21),
        poisson=PoissonConfig2D(method="device-direct", relative_tolerance=1e-10),
        reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=25, ion_subcycle=4, device_sync_steps=25,
        anomalous=AnomalousCollisionConfig(1.0 / 16.0, model=ANOMALOUS_MODEL_ROTATION),
    )
    direct = Simulation(config, field, cross_sections=xs, backend="warp-cuda", step_graph=False)
    graph = Simulation(config, field, cross_sections=xs, backend="warp-cuda", step_graph=True)
    direct.run(100)
    graph.run(100)
    a, b = direct.state, graph.state
    assert a.step == b.step == 100 and a.cumulative["anomalous"] > 0
    for name in ("r_m", "z_m", "vr_m_per_s", "vt_m_per_s", "vz_m_per_s"):
        assert np.array_equal(getattr(a.electrons, name), getattr(b.electrons, name)), name
    assert np.array_equal(a.phi_v, b.phi_v)
    assert a.cumulative["anomalous"] == b.cumulative["anomalous"]
    for key, value in a.cumulative.items():
        assert value == pytest.approx(b.cumulative[key], rel=1e-9, abs=1e-300), key


# ------------------------------------------------------------------------------------------------------------------ identity

def test_alpha_zero_keeps_the_v2_0_6_identity_and_alpha_or_model_enter_config_sha256():
    protocol = runner.load_protocol(V4_PROTOCOL)
    base = runner.build_config(protocol, backend="warp-cuda")        # the identity is backend-tagged through the Poisson method; the pin is the CUDA record
    assert base.anomalous is None and "anomalous" not in base.to_dict()
    assert artifacts.config_identity(base) == V4_CONFIG_SHA256          # alpha = 0 == the recorded ss-v4 configuration, byte for byte
    with_alpha = copy.deepcopy(protocol)
    with_alpha["numerics"]["anomalous_collisions"] = {"alpha": 1.0 / 16.0, "model": ANOMALOUS_MODEL_ROTATION, "alpha_note": "ignored by build_config"}
    rot = runner.build_config(with_alpha, backend="warp-cuda")
    assert rot.anomalous == AnomalousCollisionConfig(1.0 / 16.0, model=ANOMALOUS_MODEL_ROTATION)
    with_alpha["numerics"]["anomalous_collisions"] = {"alpha": 1.0 / 16.0}
    iso = runner.build_config(with_alpha, backend="warp-cuda")
    assert iso.anomalous == AnomalousCollisionConfig(1.0 / 16.0) and iso.anomalous.model == ANOMALOUS_MODEL_ISOTROPIC   # recorded protocols: v1.4 model
    identities = {artifacts.config_identity(c) for c in (base, rot, iso)}
    assert len(identities) == 3
    with_alpha["numerics"]["anomalous_collisions"] = {"alpha": 1.0 / 64.0, "model": ANOMALOUS_MODEL_ROTATION}
    assert artifacts.config_identity(runner.build_config(with_alpha, backend="warp-cuda")) not in identities
    assert json.loads(json.dumps(rot.to_dict()))["anomalous"] == {"model": ANOMALOUS_MODEL_ROTATION, "alpha": 1.0 / 16.0}
    # everything else of the physics configuration is untouched by the hook
    assert {k: v for k, v in rot.to_dict().items() if k != "anomalous"} == base.to_dict()
