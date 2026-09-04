"""Model v2.3.0 ``xe_collision_set_v2`` (R3 of the physics completeness audit, 2026-09-05).

* e-Xe: the four Biagi-v7.1 excitation levels (8.315 / 9.447 / 9.917 / 11.7 eV) replace the lumped 8.32 eV channel:
  bound by payload hash like the v1 set, byte-identical elastic / ionisation tables, per-level energy loss, per-level
  tallies, inelastic ledger ``W sum_k n_k E_k``; the total excitation cross section equals the lumped one.
* Xe+ - Xe: charge exchange (Miller 2002) + momentum transfer (Phelps isotropic) as a null-collision operator on the ion
  population against the 0-D inventory density with a sampled Maxwellian atom; CEX fast neutrals leave the channel
  (inventory sink + thrust term) or thermalise on the wall; energy / momentum ledgers close on both backends.
* Identity: the collision set enters ``config_sha256``; the legacy set stays selectable with an unchanged identity and
  the v2.0.6 arithmetic (the old-vs-new bitwise replay of the legacy set was done against the 0901138a tree on both the
  numpy and the Warp CPU backends before this module was committed; here the legacy paths are pinned structurally).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from itertools import pairwise
from math import exp, sqrt
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.cross_sections_xe import (
    XE_COLLISION_SET_V2_NAME,
    XE_ELECTRON_SET_V2_FILE,
    XE_ELECTRON_SET_V2_PAYLOAD_SHA256,
    XE_ION_NEUTRAL_SET_V1_FILE,
    XE_ION_NEUTRAL_SET_V1_PAYLOAD_SHA256,
    CollisionSetConfig,
    compare_excitation_sets,
    spec_path,
    total_excitation_m2,
)
from cft_revival.pic2d.fields import linear_psi_field_map, uniform_field_map
from cft_revival.pic2d.ion_mcc import (
    FATE_EXIT,
    FATE_WALL,
    ION_MCC_KEYS,
    IonNeutralCrossSections,
    IonNeutralMCCConfig,
    IonNullCollisionMCC,
    fast_neutral_fate,
    ion_maximum_collision_frequency,
)
from cft_revival.pic2d.mcc import (
    PROCESS_ORDER,
    MCCConfig,
    NullCollisionMCC,
    UniformSigmaTable,
    XenonCrossSections,
    canonical_payload_sha256,
    electron_energy_ev,
    electron_speed_from_energy,
    maxwellian_velocity,
)
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import (
    BOLTZMANN_J_PER_K,
    EV_J,
    XENON_MASS_KG,
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PIC2DValidationError,
    ParticleArrays,
    PoissonConfig2D,
    StabilityLimits,
    xenon_ion_species,
)
from cft_revival.pic2d.neutrals import (
    FAST_NEUTRAL_EXIT_KEY,
    NEUTRAL_LEDGER_KEYS,
    NEUTRAL_LEDGER_KEYS_V2,
    NeutralInventoryConfig,
    NeutralState,
    feed_for_density,
)
from cft_revival.pic2d.simulation import InjectionConfig, PIC2DConfig, SeedPlasmaConfig, Simulation
from experiments.pic2d_cft_steady_state_v1 import run as runner

SPEC_DIR = Path(__file__).resolve().parents[2] / "spec" / "pic2d"
CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
STRAIGHT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 8.0e-3, 8.0e-3, 2.0e-3)
PLUME_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 12.0e-3, 9.0e-3, 3.0e-3, plume_radius_m=6.0e-3, plume_length_m=6.0e-3, body_dielectric_radius_m=4.0e-3)
LEVELS_EV = (8.315, 9.447, 9.917, 11.7)


def _warp_backends() -> list[str]:
    try:
        from cft_revival.pic2d.warp_backend import device_available
    except ImportError:  # pragma: no cover
        return []
    return [name for name, device in (("warp-cpu", "cpu"), ("warp-cuda", "cuda:0")) if device_available(device)]


BACKENDS = ["cpu", *_warp_backends()]
CUDA = [b for b in BACKENDS if b == "warp-cuda"]


def _load_builder(name: str):
    path = SPEC_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module      # dataclasses / relative imports inside the builder need the module registered
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def v2_set() -> XenonCrossSections:
    return XenonCrossSections.from_file(spec_path(XE_ELECTRON_SET_V2_FILE))


@pytest.fixture(scope="module")
def v1_set() -> XenonCrossSections:
    return XenonCrossSections.from_file()


@pytest.fixture(scope="module")
def ion_set() -> IonNeutralCrossSections:
    return IonNeutralCrossSections.from_file(spec_path(XE_ION_NEUTRAL_SET_V1_FILE))


@pytest.fixture(scope="module")
def collision_set() -> CollisionSetConfig:
    return CollisionSetConfig.xe_collision_set_v2()


# ----------------------------------------------------------------------------------------------------------------------
# 1. the data: bound, reproducible, spot-checked against the sources
# ----------------------------------------------------------------------------------------------------------------------

def test_v2_electron_set_is_bound_and_matches_the_pinned_payload_and_sidecar(v2_set: XenonCrossSections, v1_set: XenonCrossSections):
    assert v2_set.payload_sha256 == XE_ELECTRON_SET_V2_PAYLOAD_SHA256
    assert not v2_set.is_legacy_set and v1_set.is_legacy_set
    assert [p.identifier for p in v2_set.processes] == ["elastic", "excitation_8p315", "excitation_9p447", "excitation_9p917", "excitation_11p7", "ionization"]
    assert [p.threshold_ev for p in v2_set.excitation_levels] == list(LEVELS_EV)
    document = json.loads(spec_path(XE_ELECTRON_SET_V2_FILE).read_text(encoding="utf-8"))
    assert document["schema"] == "cft.pic2d.xenon-cross-sections.v2"
    payload = {k: v for k, v in document.items() if k != "integrity"}
    assert canonical_payload_sha256(payload) == document["integrity"]["payload_sha256"] == v2_set.payload_sha256
    sidecar = (SPEC_DIR / f"{XE_ELECTRON_SET_V2_FILE}.sha256").read_text(encoding="ascii")
    assert sidecar.split()[0] == v2_set.file_sha256
    provenance = document["provenance"]
    assert provenance["database"] == "Biagi-v7.1" and provenance["retrieved_utc"].endswith("Z") and provenance["upstream_reverified_utc"].endswith("Z")
    assert provenance["upstream_url"].startswith("https://raw.githubusercontent.com/lanl/ThunderBoltz/") and len(provenance["upstream_sha256"]) == 64
    assert (SPEC_DIR / provenance["extract_file"]).is_file()
    # the other databases of the same export were inspected (none is a 4-level set); their sums are recorded
    checks = provenance["cross_checks_other_databases_same_export"]
    assert {c["excitation_levels"] for c in checks.values()} >= {33, 6}
    # elastic and ionisation are byte-identical to v1; each level is zero below its own threshold
    v1 = {p.identifier: p for p in v1_set.processes}
    for identifier in ("elastic", "ionization"):
        p2 = next(p for p in v2_set.processes if p.identifier == identifier)
        assert np.array_equal(p2.energy_ev, v1[identifier].energy_ev) and np.array_equal(p2.cross_section_m2, v1[identifier].cross_section_m2)
    for level in v2_set.excitation_levels:
        assert level.at(np.array([level.threshold_ev - 1e-6]))[0] == 0.0 and level.at(np.array([level.threshold_ev + 0.5]))[0] > 0.0


def test_v2_levels_reproduce_the_lxcat_extract_rows(v2_set: XenonCrossSections):
    """Spot values: the level tables interpolate the Biagi-v7.1 LXCat blocks (source rows read independently here)."""

    v1 = _load_builder("build_xenon_cross_sections")
    blocks = v1.parse_lxcat_blocks(v1.EXTRACT_PATH.read_bytes().decode("utf-8"))
    excitation = sorted((b for b in blocks if b["kind"] == "EXCITATION"), key=lambda b: float(b["param"]))
    assert [float(b["param"]) for b in excitation] == list(LEVELS_EV)
    for block, level in zip(excitation, v2_set.excitation_levels, strict=True):
        for k in (5, 20, 60, 120, 180):            # tabulated source rows away from the threshold
            e, s = float(block["E"][k]), float(block["sigma"][k])
            assert level.at(np.array([e]))[0] == pytest.approx(s, rel=2e-3), (level.identifier, e)
    # the summed levels at the published probe energies (the builder's sanity table)
    total = total_excitation_m2(v2_set, np.array([30.0, 100.0]))
    assert total[0] == pytest.approx(2.944e-20, rel=2e-3) and total[1] == pytest.approx(2.104e-20, rel=2e-3)


def test_v2_builder_is_reproducible_offline(v2_set: XenonCrossSections):
    builder = _load_builder("build_xenon_cross_sections_v2")
    recorded = json.loads(spec_path(XE_ELECTRON_SET_V2_FILE).read_text(encoding="utf-8"))["provenance"]
    payload = builder.build(reverified_utc=recorded["upstream_reverified_utc"], cross_checks=recorded["cross_checks_other_databases_same_export"])
    assert payload["integrity"]["payload_sha256"] == v2_set.payload_sha256


def test_total_excitation_of_the_resolved_set_equals_the_lumped_channel(v1_set: XenonCrossSections, v2_set: XenonCrossSections):
    """R3a attribution: the total excitation frequency is unchanged; only the per-event loss moves (8.32 -> 9.4-10.1 eV)."""

    report = compare_excitation_sets(v1_set, v2_set)
    assert report["max_relative_deviation_above_10_ev"] < 1e-2          # grid interpolation across the level onsets only
    assert report["max_absolute_deviation_m2"] < 1e-22                    # 0.3 % of the 2.95e-20 m2 peak
    assert report["lumped_loss_ev"] == 8.32
    for energy, loss in report["resolved_mean_loss_ev"].items():
        assert 9.3 < loss < 10.2, (energy, loss)
    recorded = json.loads(spec_path(XE_ELECTRON_SET_V2_FILE).read_text(encoding="utf-8"))["lumped_reference"]
    assert recorded["payload_sha256"] == v1_set.payload_sha256 and recorded["max_relative_deviation_above_10_ev"] < 5e-3


def test_ion_neutral_set_is_bound_and_spot_checked_against_its_sources(ion_set: IonNeutralCrossSections):
    assert ion_set.payload_sha256 == XE_ION_NEUTRAL_SET_V1_PAYLOAD_SHA256
    assert [p.identifier for p in ion_set.processes] == ["cex", "mex"]
    document = json.loads(spec_path(XE_ION_NEUTRAL_SET_V1_FILE).read_text(encoding="utf-8"))
    payload = {k: v for k, v in document.items() if k != "integrity"}
    assert canonical_payload_sha256(payload) == document["integrity"]["payload_sha256"]
    sidecar = (SPEC_DIR / f"{XE_ION_NEUTRAL_SET_V1_FILE}.sha256").read_text(encoding="ascii")
    assert sidecar.split()[0] == ion_set.file_sha256
    cex, mex = ion_set.processes
    # Miller 2002: sigma = (87.3 - 13.6 log10 E) A^2 (300 eV -> 53.61 A^2, the audit's 5.4e-19 m2); floor at 0.1 eV
    for e in (1.0, 10.0, 100.0, 300.0, 1000.0):
        assert cex.at(np.array([e]))[0] == pytest.approx((87.3 - 13.6 * np.log10(e)) * 1e-20, rel=1e-4), e
    assert cex.at(np.array([0.0]))[0] == pytest.approx(cex.at(np.array([0.1]))[0], rel=1e-9) == pytest.approx(100.9e-20, rel=1e-3)
    # Phelps isotropic component (via the WarpX mirror rows): 3.39e-19 E^-1/2 m2; rows of the extract reproduced
    for e in (1.0, 10.0, 100.0):
        assert mex.at(np.array([e]))[0] == pytest.approx(3.39e-19 / sqrt(e), rel=3e-3), e
    extract = (SPEC_DIR / "sources" / "warpx_phelps_xe_ion_extract.txt").read_text(encoding="utf-8")
    rows = [line.split() for line in extract.split("FILE ion_scattering.dat")[1].split("END ion_scattering.dat")[0].strip().splitlines()[1:]]
    for e_text, s_text in rows[5::10]:
        assert mex.at(np.array([float(e_text)]))[0] == pytest.approx(float(s_text), rel=2e-3)
    # the Phelps backscatter cross-check lies above Miller (17-41 % at 10-300 eV), recorded, not used
    ratios = document["cross_check"]["ratio_to_miller_cex"]
    assert 1.1 < ratios["10"] < 1.25 and 1.3 < ratios["300"] < 1.5
    provenance = document["provenance"]
    assert provenance["cex"]["doi"] == "10.1063/1.1426246" and provenance["mex"]["doi"] == "10.1103/PhysRevE.68.046408"
    assert "1/2 M_Xe |v_ion - v_atom|^2" in provenance["energy_convention"]


def test_ion_builder_is_reproducible_offline(ion_set: IonNeutralCrossSections):
    builder = _load_builder("build_xenon_ion_neutral_cross_sections")
    assert builder.build()["integrity"]["payload_sha256"] == ion_set.payload_sha256


def test_tampered_ion_payload_and_wrong_declarations_fail_closed(tmp_path: Path, ion_set: IonNeutralCrossSections, v1_set: XenonCrossSections):
    document = json.loads(spec_path(XE_ION_NEUTRAL_SET_V1_FILE).read_text(encoding="utf-8"))
    document["processes"][0]["cross_section_m2"][-1] *= 1.01
    target = tmp_path / "tampered.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PIC2DValidationError):
        IonNeutralCrossSections.from_file(target)
    with pytest.raises(PIC2DValidationError):
        IonNeutralMCCConfig(XE_ION_NEUTRAL_SET_V1_FILE, "0" * 64, (("cex", "charge_exchange"), ("mex", "momentum_transfer"))).load()
    with pytest.raises(PIC2DValidationError):
        CollisionSetConfig.from_protocol({"name": "xe_collision_set_v3"})
    # a collision set declaration refuses the legacy data object
    with pytest.raises(PIC2DValidationError, match="payload"):
        CollisionSetConfig.xe_collision_set_v2().check_electron(v1_set)


# ----------------------------------------------------------------------------------------------------------------------
# 2. the electron operator: per-level kinematics, tallies, legacy path unchanged
# ----------------------------------------------------------------------------------------------------------------------

def test_multilevel_mcc_rates_losses_and_ledger_match_the_null_collision_expectation(v2_set: XenonCrossSections):
    config = MCCConfig(neutral_density_per_m3=2.0e21)
    operator = NullCollisionMCC(v2_set, config, xenon_ion_species(1.0))
    assert operator.table.excitation_count == 4 and operator.table.ionization_row == 5 and operator.table.table_m2.shape[0] == 6
    count = 400_000
    energy_ev = 40.0
    speed = float(electron_speed_from_energy(np.array([energy_ev]))[0])
    electrons = ParticleArrays(np.full(count, 1.0e-3), np.full(count, 12.0e-3), np.zeros(count), np.zeros(count), np.full(count, speed))
    dt = 2.0e-11
    result = operator.apply(electrons, dt, np.random.default_rng(11))
    sigma = operator.table.lookup(np.array([energy_ev]))[:, 0]
    nu = config.neutral_density_per_m3 * sigma * speed
    p_total = 1.0 - exp(-operator.nu_max * dt)
    tally = result.tally
    assert len(tally.excitation_levels) == 4 and sum(tally.excitation_levels) == tally.excitation
    observed = (tally.elastic, *tally.excitation_levels, tally.ionization)
    for n_obs, rate in zip(observed, nu, strict=True):
        p = p_total * rate / operator.nu_max
        assert abs(n_obs - count * p) < 4.0 * sqrt(count * p * (1.0 - p)), (n_obs, count * p)
    assert tally.candidates == tally.elastic + tally.excitation + tally.ionization + tally.null
    # every excited electron sits exactly at E - E_k for one of the four levels, with the right multiplicity
    new_energy = electron_energy_ev(result.electrons.vr_m_per_s, result.electrons.vt_m_per_s, result.electrons.vz_m_per_s)
    for k, threshold in enumerate(LEVELS_EV):
        at_level = int(np.count_nonzero(np.abs(new_energy - (energy_ev - threshold)) < 1e-7))
        assert at_level == tally.excitation_levels[k], (k, at_level)
    expected_loss = (sum(n * e for n, e in zip(tally.excitation_levels, LEVELS_EV, strict=True)) + tally.ionization * 12.13) * EV_J
    assert tally.inelastic_energy_loss_j == pytest.approx(expected_loss, rel=1e-12)
    assert tally.to_dict()["excitation_levels"] == list(tally.excitation_levels)
    assert "per-level threshold loss" in operator.to_dict()["kinematics"]["excitation"]


def test_legacy_operator_and_config_identity_are_unchanged(v1_set: XenonCrossSections):
    operator = NullCollisionMCC(v1_set, MCCConfig(2.0e21), xenon_ion_species(1.0))
    assert operator.table.excitation_count == 1 and operator.table.thresholds_ev == (0.0, 8.32, 12.13)
    count = 50_000
    speed = float(electron_speed_from_energy(np.array([40.0]))[0])
    electrons = ParticleArrays(np.full(count, 1.0e-3), np.full(count, 12.0e-3), np.zeros(count), np.zeros(count), np.full(count, speed))
    result = operator.apply(electrons, 2.0e-11, np.random.default_rng(9))
    tally = result.tally
    assert tally.excitation_levels == (tally.excitation,) and "excitation_levels" not in tally.to_dict()
    assert tally.inelastic_energy_loss_j == (tally.excitation * 8.32 + tally.ionization * 12.13) * EV_J      # bitwise the v2.0.6 expression
    assert operator.to_dict()["kinematics"]["excitation"] == "lumped 8.32 eV loss, isotropic"
    assert tuple(p.identifier for p in v1_set.processes) == PROCESS_ORDER
    # MCCConfig.to_dict has no collision-set key -> every recorded config_sha256 is unchanged
    assert MCCConfig(1e21).to_dict() == {"neutral_density_per_m3": 1e21, "neutral_temperature_k": 300.0, "energy_step_ev": 0.05, "energy_max_ev": 2000.0}
    with pytest.raises(PIC2DValidationError):
        MCCConfig(1e21, collision_set="xe_collision_set_v2")     # not a CollisionSetConfig


def test_collision_set_enters_the_identity_and_binds_the_supplied_data(collision_set: CollisionSetConfig, v1_set: XenonCrossSections):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)

    def config(cs):
        return PIC2DConfig(grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=3,
                           mcc=MCCConfig(1e21, collision_set=cs), poisson=PoissonConfig2D(method="direct"),
                           reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0, limits=StabilityLimits(max_cell_debye_ratio=2.0))

    legacy, resolved = config(None), config(collision_set)
    resolved_no_ions = config(CollisionSetConfig.xe_collision_set_v2(ion_neutral=False))
    identities = {artifacts.config_identity(c) for c in (legacy, resolved, resolved_no_ions)}
    assert len(identities) == 3
    record = resolved.to_dict()["mcc"]["collision_set"]
    assert record["name"] == XE_COLLISION_SET_V2_NAME and record["electron_payload_sha256"] == XE_ELECTRON_SET_V2_PAYLOAD_SHA256
    assert [p["id"] for p in record["electron_processes"]] == [p.identifier for p in collision_set.load_electron_cross_sections().processes]
    assert record["ion_neutral"]["payload_sha256"] == XE_ION_NEUTRAL_SET_V1_PAYLOAD_SHA256 and record["ion_neutral"]["fast_neutral_speed_threshold_factor"] == 4.0
    assert "collision_set" not in legacy.to_dict()["mcc"]
    field = linear_psi_field_map(grid, 2.0)
    # the declared set refuses other data; an undeclared multi-level set refuses to run (identity)
    with pytest.raises(PIC2DValidationError, match="payload"):
        Simulation(resolved, field, backend="cpu", cross_sections=v1_set)
    with pytest.raises(PIC2DValidationError, match="collision_set"):
        Simulation(legacy, field, backend="cpu", cross_sections=collision_set.load_electron_cross_sections())
    # a protocol block names the set and the ion options; the hashes come from the files, never from the protocol
    from_protocol = CollisionSetConfig.from_protocol({"name": "xe_collision_set_v2", "ion_neutral": {"fast_neutral_speed_threshold_factor": 3.0}})
    assert from_protocol.electron_payload_sha256 == XE_ELECTRON_SET_V2_PAYLOAD_SHA256 and from_protocol.ion_neutral.fast_neutral_speed_threshold_factor == 3.0
    assert CollisionSetConfig.from_protocol({"name": "xe_collision_set_v2", "ion_neutral": False}).ion_neutral is None


# ----------------------------------------------------------------------------------------------------------------------
# 3. the ion operator: sampling, conservation, rates, fast-neutral fate
# ----------------------------------------------------------------------------------------------------------------------

def test_maxwellian_neutral_sampling_moments():
    rng = np.random.default_rng(5)
    n = 400_000
    for temperature in (300.0, 500.0):
        vx, vy, vz = maxwellian_velocity(XENON_MASS_KG, temperature, rng.random((4, n)))
        v_th2 = BOLTZMANN_J_PER_K * temperature / XENON_MASS_KG
        for component in (vx, vy, vz):
            assert abs(component.mean()) < 4.0 * sqrt(v_th2 / n)
            assert component.var() == pytest.approx(v_th2, rel=0.01)
        speed = np.sqrt(vx * vx + vy * vy + vz * vz)
        assert speed.mean() == pytest.approx(sqrt(8.0 * v_th2 / np.pi), rel=0.01)        # <v> = sqrt(8 k T / pi M)
        assert (speed**2).mean() == pytest.approx(3.0 * v_th2, rel=0.01)


def _ion_operator(ion_set: IonNeutralCrossSections, masks, n_g: float, **kwargs) -> IonNullCollisionMCC:
    ion_config = IonNeutralMCCConfig("synthetic" if ion_set.file_sha256 is None else XE_ION_NEUTRAL_SET_V1_FILE, ion_set.payload_sha256,
                                     tuple((p.identifier, p.kind) for p in ion_set.processes), **kwargs)
    return IonNullCollisionMCC(ion_set, MCCConfig(n_g), ion_config, xenon_ion_species(1.0e4), masks)


def test_cex_conserves_weight_momentum_and_energy_and_books_the_fast_neutrals():
    masks = build_mesh_masks(Grid2D(CFT_GEOMETRY, 12, 96))
    synthetic = IonNeutralCrossSections.synthetic_for_tests(cex_m2=5.0e-19, mex_m2=0.0)
    operator = _ion_operator(synthetic, masks, 1.0e22)
    mass, weight = XENON_MASS_KG, 1.0e4
    count = 200_000
    speed = 1.5e4                                          # a 250 eV beam along +z near the axis, 20 mm from the exit
    rng = np.random.default_rng(21)
    ions = ParticleArrays(rng.uniform(0.0, 1.0e-3, count), np.full(count, 4.0e-3), np.zeros(count), np.zeros(count), np.full(count, speed))
    dt = 5.0e-11
    result = operator.apply(ions, dt, np.random.default_rng(3))
    tally = result.tally
    assert result.ions.count == count                                            # weight conservation: no ion is created or lost
    assert tally.mex == 0 and tally.cex > 500 and tally.candidates == tally.cex + tally.null
    changed = result.ions.vz_m_per_s != speed
    assert int(changed.sum()) == tally.cex
    # the CEX ion continues with a thermal velocity: |v'| ~ v_th, never the beam speed
    v_th = sqrt(BOLTZMANN_J_PER_K * 300.0 / mass)
    new_speed = np.sqrt(result.ions.vr_m_per_s**2 + result.ions.vt_m_per_s**2 + result.ions.vz_m_per_s**2)[changed]
    assert new_speed.max() < 8.0 * v_th and np.sqrt((new_speed**2).mean()) == pytest.approx(sqrt(3.0) * v_th, rel=0.1)
    # momentum: the ions' change equals (thermal atoms taken up) - (fast neutrals released); the fast neutrals carry M v_z each
    dpz_ions = weight * mass * float((result.ions.vz_m_per_s[changed] - speed).sum())
    assert tally.pz_ions_kg_m_s == pytest.approx(dpz_ions, rel=1e-12)
    assert tally.pz_fast_neutral_exit_kg_m_s + tally.pz_fast_neutral_wall_kg_m_s == pytest.approx(tally.cex * weight * mass * speed, rel=1e-12)
    assert tally.fast_neutral_exit_channel + tally.fast_neutral_wall + tally.fast_neutral_thermal == tally.cex and tally.fast_neutral_thermal == 0
    # energy: the ions hand exactly (beam KE - thermal KE) per event to the neutrals; the exiting fast neutrals carry the beam KE
    ke_before = 0.5 * mass * speed**2 * tally.cex
    ke_after = 0.5 * mass * float((new_speed**2).sum())
    assert tally.energy_loss_j == pytest.approx(weight * (ke_before - ke_after), rel=1e-12)
    assert tally.ke_fast_neutral_exit_j == pytest.approx(tally.fast_neutral_exit_channel * weight * 0.5 * mass * speed**2, rel=1e-12)
    # an axial beam inside the aperture leaves through the exit (the cone widens): every fast neutral exits
    assert tally.fast_neutral_exit_channel == tally.cex and tally.fast_neutral_wall == 0
    # the Bernoulli rate: P_cex = P_total nu_cex / nu_max with nu_cex = n_g sigma g (thermal spread negligible at 15 km/s)
    p_total = operator.collision_probability(dt)
    nu_cex = 1.0e22 * 5.0e-19 * speed
    p = p_total * nu_cex / operator.nu_max
    assert abs(tally.cex - count * p) < 4.0 * sqrt(count * p * (1.0 - p))


def test_mex_is_isotropic_in_the_centre_of_mass():
    """Equal masses, target ~at rest, isotropic CM scattering: E'/E is uniform on [0, 1] -> mean 1/2, variance 1/12."""

    masks = build_mesh_masks(Grid2D(CFT_GEOMETRY, 12, 96))
    synthetic = IonNeutralCrossSections.synthetic_for_tests(cex_m2=0.0, mex_m2=5.0e-19)
    operator = _ion_operator(synthetic, masks, 1.0e22)
    count = 200_000
    speed = 2.0e4                                      # v_th / v = 0.7 %
    ions = ParticleArrays(np.full(count, 1.0e-3), np.full(count, 12.0e-3), np.zeros(count), np.zeros(count), np.full(count, speed))
    result = operator.apply(ions, 5.0e-11, np.random.default_rng(4))
    tally = result.tally
    assert tally.cex == 0 and tally.mex > 500
    changed = result.ions.vz_m_per_s != speed
    ratio = (result.ions.vr_m_per_s**2 + result.ions.vt_m_per_s**2 + result.ions.vz_m_per_s**2)[changed] / speed**2
    n = ratio.size
    assert ratio.mean() == pytest.approx(0.5, abs=4.0 * sqrt(1.0 / 12.0 / n) + 0.01)
    assert ratio.var() == pytest.approx(1.0 / 12.0, rel=0.1)
    assert ratio.min() > -0.02 and ratio.max() < 1.02
    # energy handed to the atoms = the ions' loss (elastic: the CM kinetic energy is conserved, so nothing is created)
    ke_loss = 1.0e4 * 0.5 * XENON_MASS_KG * speed**2 * float((1.0 - ratio).sum())
    assert tally.energy_loss_j == pytest.approx(ke_loss, rel=1e-9)
    assert tally.pz_ions_kg_m_s == pytest.approx(1.0e4 * XENON_MASS_KG * float((result.ions.vz_m_per_s[changed] - speed).sum()), rel=1e-12)


def test_collision_frequencies_in_a_uniform_box_match_n_sigma_v_with_the_real_tables(ion_set: IonNeutralCrossSections):
    masks = build_mesh_masks(Grid2D(CFT_GEOMETRY, 12, 96))
    operator = _ion_operator(ion_set, masks, 3.0e21)
    energy_ev = 100.0
    speed = sqrt(2.0 * energy_ev * EV_J / XENON_MASS_KG)
    count = 400_000
    ions = ParticleArrays(np.full(count, 1.0e-3), np.full(count, 12.0e-3), np.zeros(count), np.zeros(count), np.full(count, speed))
    dt = 4.0e-11
    result = operator.apply(ions, dt, np.random.default_rng(8))
    tally = result.tally
    sigma = operator.table.lookup(np.array([energy_ev]))[:, 0]
    assert sigma[0] == pytest.approx((87.3 - 13.6 * 2.0) * 1e-20, rel=1e-3) and sigma[1] == pytest.approx(3.39e-20, rel=3e-3)
    p_total = operator.collision_probability(dt)
    for observed, s in ((tally.cex, sigma[0]), (tally.mex, sigma[1])):
        p = p_total * (3.0e21 * s * speed) / operator.nu_max
        assert abs(observed - count * p) < 4.0 * sqrt(count * p * (1.0 - p)), (observed, count * p)
    assert tally.ceiling_violations == 0
    # the ceiling is the table maximum of sigma_total g (sigma_CEX falls only logarithmically): 2000 eV here
    assert operator.nu_max == pytest.approx(ion_maximum_collision_frequency(operator.table, 3.0e21, XENON_MASS_KG))
    e_top = operator.table.energy_max_ev
    assert operator.nu_max == pytest.approx(3.0e21 * operator.table.lookup(np.array([e_top]))[:, 0].sum() * sqrt(2.0 * e_top * EV_J / XENON_MASS_KG), rel=1e-6)
    # lambda_CEX at the audit's point
    assert 1.0 / (3.0e19 * float(ion_set.processes[0].at(np.array([300.0]))[0])) == pytest.approx(62.2e-3, rel=1e-2)


def test_fast_neutral_fate_follows_the_straight_line_through_the_cell_mask():
    masks = build_mesh_masks(Grid2D(CFT_GEOMETRY, 12, 96))
    assert fast_neutral_fate(masks, 0.5e-3, 20.0e-3, 0.0, 0.0, 1.0e4) == FATE_EXIT           # axial, inside the aperture
    assert fast_neutral_fate(masks, 1.0e-3, 20.0e-3, 5.0e3, 0.0, -1.0e4) == FATE_WALL         # towards the anode
    assert fast_neutral_fate(masks, 1.0e-3, 10.0e-3, 1.0e4, 0.0, 1.0e3) == FATE_WALL          # radial: the bore wall
    assert fast_neutral_fate(masks, 1.9e-3, 10.0e-3, 1.0e3, 0.0, 1.0e4) == FATE_WALL          # grazing the bore before the cone
    assert fast_neutral_fate(masks, 1.5e-3, 17.9e-3, 1.0e3, 0.0, 1.0e4) == FATE_EXIT          # the same slope from 1.5 mm clears the widening cone (r 2.1 mm at the exit)
    # the cone is the one-cell stair step every particle sees: a neutral hugging the TRUE cone at 2.01 mm hits the stair
    assert fast_neutral_fate(masks, 1.9e-3, 17.9e-3, 1.0e3, 0.0, 1.0e4) == FATE_WALL
    assert fast_neutral_fate(masks, 0.0, 1.0e-3, 0.0, 5.0e2, 1.0e4) == FATE_EXIT              # azimuthal component: r(t) = |v_theta| t = 1.15 mm at the exit
    assert fast_neutral_fate(masks, 0.0, 1.0e-3, 0.0, 2.0e3, 1.0e4) == FATE_WALL              # ... 4.6 mm would be: the bore wall first
    assert fast_neutral_fate(masks, 0.0, 30.0e-3, 0.0, 0.0, -1.0e4) == FATE_EXIT              # born past the exit plane: leaves
    assert fast_neutral_fate(masks, 0.5e-3, 20.0e-3, 0.0, 0.0, 0.0) == FATE_WALL              # at rest: never leaves


def test_slow_cex_neutrals_return_to_the_inventory_and_plume_born_ones_leave():
    masks = build_mesh_masks(Grid2D(PLUME_GEOMETRY, 12, 36))
    synthetic = IonNeutralCrossSections.synthetic_for_tests(cex_m2=5.0e-19, mex_m2=0.0)
    operator = _ion_operator(synthetic, masks, 1.0e22, fast_neutral_speed_threshold_factor=4.0)
    v_th = sqrt(BOLTZMANN_J_PER_K * 300.0 / XENON_MASS_KG)
    count = 100_000
    # (a) thermal ions in the channel: their "fast" neutrals are below 4 v_th -> thermal, no inventory change
    slow = ParticleArrays(np.full(count, 1.0e-3), np.full(count, 6.0e-3), np.zeros(count), np.zeros(count), np.full(count, 2.0 * v_th))
    tally = operator.apply(slow, 1.0e-8, np.random.default_rng(1)).tally
    assert tally.cex > 100 and tally.fast_neutral_thermal == tally.cex and tally.fast_neutral_exit_channel == 0 and tally.pz_fast_neutral_exit_kg_m_s == 0.0
    # (b) fast ions in the plume: plume-born CEX, the fast neutral leaves the box (not an inventory atom)
    plume = ParticleArrays(np.full(count, 1.0e-3), np.full(count, 15.0e-3), np.zeros(count), np.zeros(count), np.full(count, 1.5e4))
    tally = operator.apply(plume, 5.0e-11, np.random.default_rng(2)).tally
    assert tally.cex > 100 and tally.cex_plume == tally.cex == tally.fast_neutral_exit_plume and tally.fast_neutral_exit_channel == 0
    assert tally.pz_fast_neutral_exit_kg_m_s == pytest.approx(tally.cex * 1.0e4 * XENON_MASS_KG * 1.5e4, rel=1e-12)


# ----------------------------------------------------------------------------------------------------------------------
# 4. the simulation: both backends, ledgers, inventory, identity, checkpoints
# ----------------------------------------------------------------------------------------------------------------------

def _discharge_config(grid: Grid2D, collision_set, *, inventory: bool = True, ion_subcycle: int = 4, seed: int = 3) -> PIC2DConfig:
    n_g = 1.0e21
    neutral = None
    if inventory:
        feed = feed_for_density(0.8 * n_g, np.pi * grid.geometry.exit_radius_m**2, 300.0)
        neutral = NeutralInventoryConfig(feed, None, wall_recycling=True, wall_temperature_k=500.0, initial_density_per_m3=0.9 * n_g)
    return PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=seed,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e16, 5.0, 1.0), mcc=MCCConfig(n_g, collision_set=collision_set),
        poisson=PoissonConfig2D(method="direct"), reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=25, ion_subcycle=ion_subcycle, neutral_inventory=neutral,
    )


def _identity_terms(sim: Simulation) -> dict[str, np.ndarray]:
    out = {k: [] for k in ("dke", "rhs", "residual", "h", "ion_loss")}
    for a, b in pairwise(sim.series):
        ca, cb = a.ledger["cumulative"], b.ledger["cumulative"]
        d = lambda key: float(cb.get(key, 0.0) - ca.get(key, 0.0))  # noqa: E731
        out["dke"].append((b.kinetic_electron_j + b.kinetic_ion_j) - (a.kinetic_electron_j + a.kinetic_ion_j))
        out["rhs"].append(d("field_work_j") + d("ke_injected_j") - d("ke_absorbed_anode_j") - d("ke_absorbed_exit_j") - d("ke_absorbed_wall_j")
                          + d("ke_born_ions_j") - d("inelastic_loss_j") - d("ion_neutral_loss_j"))
        out["residual"].append(b.ledger["interval_residual_j"])
        out["h"].append(d("field_work_j") + (b.field_energy_j - a.field_energy_j) - b.ledger["interval_electrode_work_j"])
        out["ion_loss"].append(d("ion_neutral_loss_j"))
    return {k: np.asarray(v) for k, v in out.items()}


@pytest.mark.parametrize("backend", BACKENDS)
def test_energy_and_atom_ledgers_close_with_the_v2_set_on_every_backend(backend: str, collision_set: CollisionSetConfig):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = collision_set.load_electron_cross_sections()
    sim = Simulation(_discharge_config(grid, collision_set), linear_psi_field_map(grid, 2.0), backend=backend, cross_sections=xs)
    sim.run(400)
    cumulative = sim.series[-1].ledger["cumulative"]
    assert set(ION_MCC_KEYS) <= set(cumulative) and cumulative["ion_mcc_candidates"] > 0 and cumulative["cex"] + cumulative["mex"] > 0
    levels = [cumulative[f"excitations_level_{k}"] for k in range(1, 5)]
    assert sum(levels) == cumulative["excitations"] > 0 and cumulative["ionizations"] > 0
    per_weight = (sum(n * e for n, e in zip(levels, LEVELS_EV, strict=True)) + cumulative["ionizations"] * 12.13) * EV_J
    assert cumulative["inelastic_loss_j"] == pytest.approx(per_weight * 2e6, rel=1e-12)          # the ledger sums per level x W
    terms = _identity_terms(sim)
    scale = max(float(np.max(np.abs(terms["dke"]))), float(sim.series[-1].kinetic_electron_j))
    assert float(np.max(np.abs(terms["dke"] - terms["rhs"]))) <= 1e-6 * scale                  # particle identity incl. the ion-neutral sink
    assert np.allclose(terms["residual"], terms["h"], rtol=0.0, atol=1e-6 * scale)
    assert "interval_ion_neutral_loss_j" in sim.series[-1].ledger
    # the atom ledger: fed + recycled - ionised - effused - artificial - fast_exit = V dn to round-off; the sink key is carried
    for record in sim.series:
        assert abs(record.neutral["interval_ledger_residual_atoms"]) < 1e-9 * record.neutral["feed_atoms_per_s"] * 25 * 5e-12 + 1e-6
        assert FAST_NEUTRAL_EXIT_KEY in record.neutral["ledger"] and "fast_neutral_exit_rate_per_s" in record.neutral
    currents = sim.series[-1].currents_a
    assert {"cex_rate_per_s", "mex_rate_per_s", "fast_neutral_exit_rate_per_s", "fast_neutral_wall_rate_per_s"} <= set(currents)
    provenance = sim.to_provenance()
    assert provenance["ion_mcc"]["cross_sections"]["payload_sha256"] == XE_ION_NEUTRAL_SET_V1_PAYLOAD_SHA256
    assert provenance["config"]["mcc"]["collision_set"]["name"] == XE_COLLISION_SET_V2_NAME


@pytest.mark.parametrize("backend", BACKENDS)
def test_legacy_set_records_carry_no_new_keys_and_the_identity_is_the_v2_0_6_one(backend: str, v1_set: XenonCrossSections):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    config = _discharge_config(grid, None)
    assert "collision_set" not in config.to_dict()["mcc"] and artifacts.config_identity(config) == artifacts.config_identity(_discharge_config(grid, None))
    sim = Simulation(config, linear_psi_field_map(grid, 2.0), backend=backend, cross_sections=v1_set)
    sim.run(100)
    cumulative = sim.series[-1].ledger["cumulative"]
    assert not (set(ION_MCC_KEYS) & set(cumulative)) and not any(key.startswith("excitations_level") for key in cumulative)
    assert set(sim.series[-1].neutral["ledger"]) == set(NEUTRAL_LEDGER_KEYS) and "interval_ion_neutral_loss_j" not in sim.series[-1].ledger
    assert sim.series[-1].neutral["ledger"] == dict(sim.state.neutral.ledger) and sim.state.neutral.ledger_keys == NEUTRAL_LEDGER_KEYS
    assert "ion_mcc" not in sim.to_provenance()


def test_cpu_and_warp_ion_collision_statistics_agree(collision_set: CollisionSetConfig):
    """Distributional parity on a hot, dense ion population: the same rates within statistics (different random streams)."""

    if len(BACKENDS) < 2:
        pytest.skip("Warp unavailable")
    grid = Grid2D(STRAIGHT_GEOMETRY, 8, 32)
    xs = collision_set.load_electron_cross_sections()
    tallies = {}
    for backend in BACKENDS[:2]:
        config = PIC2DConfig(
            grid=grid, potentials=BoundaryPotentials(0.0, 0.0), dt_s=2e-12, macro_weight=2e4, seed=7,
            seed_plasma=SeedPlasmaConfig(3e15, 1.0, 40.0), mcc=MCCConfig(1e22, collision_set=collision_set),
            poisson=PoissonConfig2D(method="direct"), reference_density_per_m3=3e15, reference_electron_temperature_ev=1.0,
            max_electron_energy_ev=100.0, limits=StabilityLimits(max_cell_debye_ratio=4.0, max_omega_pe_dt=0.5), series_interval_steps=20, ion_subcycle=1,
        )
        sim = Simulation(config, uniform_field_map(grid, 0.01), backend=backend, cross_sections=xs)
        sim.run(200)
        tallies[backend] = sim.series[-1].ledger["cumulative"]
    a, b = (tallies[name] for name in BACKENDS[:2])
    for key, floor in (("cex", 30), ("mex", 10), ("ion_mcc_candidates", 100)):     # sigma_MEX(40 eV) is 8 % of sigma_CEX
        assert a[key] > floor and b[key] > floor, key
        assert abs(a[key] - b[key]) < 5.0 * sqrt(a[key] + b[key]), (key, a[key], b[key])
    for key in ("ion_neutral_loss_j", "pz_ion_collisions"):
        assert np.sign(a[key]) == np.sign(b[key]) and abs(a[key] - b[key]) < 0.5 * (abs(a[key]) + abs(b[key])), key


@pytest.mark.parametrize("backend", CUDA)
def test_cuda_graph_and_direct_launch_are_bitwise_with_the_ion_mcc(backend: str, collision_set: CollisionSetConfig):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = collision_set.load_electron_cross_sections()
    states = []
    tallies = []
    for step_graph in (True, False):
        config = _discharge_config(grid, collision_set)
        sim = Simulation(config, linear_psi_field_map(grid, 2.0), backend=backend, cross_sections=xs, step_graph=step_graph)
        sim.run(150)
        states.append(sim.state)
        tallies.append(sim.series[-1].ledger["cumulative"])
    assert sim.step_graph_state() is False and states[0].electrons.count == states[1].electrons.count
    for name in ("r_m", "z_m", "vr_m_per_s", "vt_m_per_s", "vz_m_per_s"):
        assert np.array_equal(getattr(states[0].electrons, name), getattr(states[1].electrons, name))
        assert np.array_equal(getattr(states[0].ions, name), getattr(states[1].ions, name))
    assert np.array_equal(states[0].phi_v, states[1].phi_v)
    for key in ("cex", "mex", "ion_mcc_candidates", "fast_neutral_exit_channel", "fast_neutral_wall", "excitations", *[f"excitations_level_{k}" for k in range(1, 5)]):
        assert tallies[0][key] == tallies[1][key], key
    assert tallies[0]["cex"] + tallies[0]["mex"] > 0


def test_checkpoint_round_trip_carries_the_v2_neutral_layout_and_resumes_bitwise(tmp_path: Path, collision_set: CollisionSetConfig):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = collision_set.load_electron_cross_sections()
    config = _discharge_config(grid, collision_set)
    field = linear_psi_field_map(grid, 2.0)
    sim = Simulation(config, field, backend="cpu", cross_sections=xs)
    sim.run(100)
    state = sim.state
    assert state.neutral.ledger_keys == NEUTRAL_LEDGER_KEYS_V2
    json_path, _ = artifacts.save_checkpoint(tmp_path, "ck", state, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256, backend="cpu")
    metadata = artifacts.read_canonical_json(json_path)
    assert metadata["neutral_keys"] == ["density_per_m3", *NEUTRAL_LEDGER_KEYS_V2] and "cex" in metadata["cumulative_extra_keys"]
    loaded = artifacts.load_checkpoint(json_path, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256)
    assert loaded.neutral.ledger == state.neutral.ledger and loaded.cumulative["cex"] == state.cumulative["cex"]
    # resume: bitwise the uninterrupted run
    sim.run(100)
    resumed = Simulation(config, field, backend="cpu", cross_sections=xs)
    resumed.load_state(loaded)
    resumed.run(100)
    assert np.array_equal(resumed.state.ions.vz_m_per_s, sim.state.ions.vz_m_per_s) and np.array_equal(resumed.state.phi_v, sim.state.phi_v)
    assert resumed.state.neutral.ledger[FAST_NEUTRAL_EXIT_KEY] == sim.state.neutral.ledger[FAST_NEUTRAL_EXIT_KEY]
    # legacy arrays (5 keys) and v2 arrays (6 keys) both load; the v2 array is one entry longer
    assert NeutralState.from_array(np.array([1e20, 1, 2, 3, 4, 5, 6.0])).ledger[FAST_NEUTRAL_EXIT_KEY] == 6.0
    assert NeutralState.from_array(np.array([1e20, 1, 2, 3, 4, 5.0])).ledger_keys == NEUTRAL_LEDGER_KEYS


def test_the_inventory_sink_lowers_the_fixed_point_and_closes_the_atom_balance():
    from cft_revival.pic2d.neutrals import NeutralInventory

    area = np.pi * 9e-6
    feed = feed_for_density(3.0e19, area, 300.0)
    config = NeutralInventoryConfig(feed, None, wall_recycling=True, wall_temperature_k=500.0)
    inventory = NeutralInventory(config, ceiling_density_per_m3=5e19, exit_area_m2=area, temperature_k=300.0, volume_m3=3e-7)
    state = NeutralState.initial(4e19, fast_neutral_sink=True)
    s, r, f = 0.2 * feed, 0.1 * feed, 0.05 * feed
    without = inventory.advance(state, s, 1e-7, r, fast_neutral_exit_rate_per_s=0.0)
    with_sink = inventory.advance(state, s, 1e-7, r, fast_neutral_exit_rate_per_s=f)
    assert with_sink.fixed_point_per_m3 < without.fixed_point_per_m3
    assert with_sink.fixed_point_per_m3 == pytest.approx(inventory.fixed_point(s, r, f))
    assert with_sink.state.ledger[FAST_NEUTRAL_EXIT_KEY] == pytest.approx(f * 1e-7) and abs(with_sink.ledger_residual_atoms) < 1e-6 * 3e-7 * 4e19
    assert with_sink.state.density_per_m3 < without.state.density_per_m3
    # the legacy call (no sink argument) keeps the v1.4 arithmetic and layout
    legacy = inventory.advance(NeutralState.initial(4e19), s, 1e-7, r)
    assert legacy.state.ledger_keys == NEUTRAL_LEDGER_KEYS and legacy.fixed_point_per_m3 == without.fixed_point_per_m3
    assert legacy.state.density_per_m3 == without.state.density_per_m3
    with pytest.raises(PIC2DValidationError):
        inventory.advance(state, s, 1e-7, r, fast_neutral_exit_rate_per_s=-1.0)


def test_plume_domain_fast_neutrals_and_momentum_ledger_terms():
    grid = Grid2D(PLUME_GEOMETRY, 12, 36)
    collision_set = CollisionSetConfig.xe_collision_set_v2()
    xs = collision_set.load_electron_cross_sections()
    n_g = 1.0e22
    feed = feed_for_density(0.8 * n_g, np.pi * grid.geometry.exit_radius_m**2, 300.0)
    config = PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(200.0, 0.0), dt_s=5e-12, macro_weight=1e4, seed=5,
        seed_plasma=SeedPlasmaConfig(3e15, 5.0, 20.0, region="all"), mcc=MCCConfig(n_g, collision_set=collision_set),
        poisson=PoissonConfig2D(method="direct"), reference_density_per_m3=3e15, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.5), series_interval_steps=25, ion_subcycle=2,
        neutral_inventory=NeutralInventoryConfig(feed, None, wall_recycling=True, wall_temperature_k=500.0, initial_density_per_m3=0.9 * n_g),
    )
    sim = Simulation(config, uniform_field_map(grid, 0.01), backend="cpu", cross_sections=xs)
    sim.run(200)
    momentum = sim.series[-1].momentum
    for key in ("ion_collision_momentum_rate_n", "fast_neutral_exit_momentum_rate_n", "fast_neutral_wall_momentum_rate_n", "gas_momentum_rate_n", "fast_neutral_thrust_n"):
        assert key in momentum
    assert momentum["thrust_total_n"] == pytest.approx(momentum["thrust_flux_n"] + momentum["cold_gas_thrust_n"] + momentum["fast_neutral_exit_momentum_rate_n"], rel=1e-12)
    # the plasma momentum ledger closes with the ion-collision term (round-off)
    for record in sim.series[1:]:
        assert abs(record.momentum["interval_ledger_residual_kg_m_s"]) <= 1e-9 * abs(record.momentum["momentum_z_kg_m_s"]) + 1e-30
    cumulative = sim.series[-1].ledger["cumulative"]
    assert cumulative["cex"] + cumulative["mex"] > 0
    assert cumulative["cex_plume"] <= cumulative["cex"] and cumulative["fast_neutral_exit_plume"] <= cumulative["cex_plume"]


# ----------------------------------------------------------------------------------------------------------------------
# 5. the runner and the model spec entry
# ----------------------------------------------------------------------------------------------------------------------

def test_runner_builds_the_collision_set_from_the_protocol_and_the_arrays_carry_the_new_terms():
    protocol = json.loads((Path(__file__).resolve().parents[2] / "experiments" / "pic2d_cft_steady_state_v4" / "protocol.json").read_text(encoding="utf-8"))
    base = runner.build_config(protocol, backend="cpu")
    assert base.mcc.collision_set is None
    with_set = json.loads(json.dumps(protocol))
    with_set["operating_point"]["collision_set"] = {"name": "xe_collision_set_v2", "ion_neutral": True}
    config = runner.build_config(with_set, backend="cpu")
    assert config.mcc.collision_set.name == XE_COLLISION_SET_V2_NAME and config.mcc.collision_set.ion_neutral is not None
    assert artifacts.config_identity(config) != artifacts.config_identity(base)
    _, xs = runner.load_inputs(config, uniform_field_map(config.grid, 0.01), None, protocol=with_set)
    assert xs.payload_sha256 == XE_ELECTRON_SET_V2_PAYLOAD_SHA256
    _, legacy_xs = runner.load_inputs(base, uniform_field_map(base.grid, 0.01), None, protocol=protocol)
    assert legacy_xs.is_legacy_set
    # series arrays: the v2.3.0 scalars exist (NaN on records without the ion MCC)
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    collision_set = CollisionSetConfig.xe_collision_set_v2()
    sim = Simulation(_discharge_config(grid, collision_set), linear_psi_field_map(grid, 2.0), backend="cpu", cross_sections=collision_set.load_electron_cross_sections())
    sim.run(100)
    arrays = runner.records_to_arrays([record.to_dict() for record in sim.series])
    assert "current_cex_rate_per_s" in arrays and "neutral_ledger_fast_neutral_exit" in arrays and np.all(np.isfinite(arrays["interval_ion_neutral_loss_j"]))
    legacy = Simulation(_discharge_config(grid, None), linear_psi_field_map(grid, 2.0), backend="cpu", cross_sections=XenonCrossSections.from_file())
    legacy.run(50)
    old = runner.records_to_arrays([record.to_dict() for record in legacy.series])
    assert np.all(np.isnan(old["interval_ion_neutral_loss_j"])) and np.all(old["neutral_ledger_fast_neutral_exit"] == 0.0) and "current_cex_rate_per_s" not in old


def test_model_spec_v2_3_entry_binds_the_hashes_and_declares_the_expectations():
    spec = json.loads((SPEC_DIR / "pic2d-model-v2.3.json").read_text(encoding="utf-8"))
    entry = spec["xe_collision_set_v2"]
    assert spec["version"] == "2.3.0" and entry["identity"]["enters_config_sha256"] is True
    assert entry["electron_set"]["payload_sha256"] == XE_ELECTRON_SET_V2_PAYLOAD_SHA256 and entry["electron_set"]["file"] == XE_ELECTRON_SET_V2_FILE
    assert entry["ion_neutral_set"]["payload_sha256"] == XE_ION_NEUTRAL_SET_V1_PAYLOAD_SHA256 and entry["ion_neutral_set"]["file"] == XE_ION_NEUTRAL_SET_V1_FILE
    assert [p["threshold_ev"] for p in entry["electron_set"]["processes"] if p["kind"] == "excitation"] == list(LEVELS_EV)
    assert {p["id"] for p in entry["ion_neutral_set"]["processes"]} == {"cex", "mex"}
    assert "10.1063/1.1426246" in json.dumps(entry) and "10.1103/PhysRevE.68.046408" in json.dumps(entry)
    expectations = entry["predeclared_expectations"]
    assert {"discharge_current", "ionization_rate", "iedf_low_energy_population", "ion_thrust_redistribution"} <= set(expectations)
    assert entry["fast_neutral_contract"]["inventory_sink"] == "fast_neutral_exit_channel"
    # the uniform tables the operators actually use are those the spec describes
    xs = XenonCrossSections.from_file(spec_path(XE_ELECTRON_SET_V2_FILE))
    table = UniformSigmaTable.build(xs)
    assert table.process_count == 6 and table.excitation_thresholds_ev == LEVELS_EV
