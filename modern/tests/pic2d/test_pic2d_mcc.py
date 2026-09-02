"""Xenon cross-section spec and null-collision MCC verification."""

from __future__ import annotations

import json
from math import exp
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d.mcc import (
    DEFAULT_CROSS_SECTION_PATH,
    PROCESS_ORDER,
    MCCConfig,
    NullCollisionMCC,
    UniformSigmaTable,
    XenonCrossSections,
    canonical_payload_sha256,
    electron_energy_ev,
    electron_speed_from_energy,
    maximum_collision_frequency,
)
from cft_revival.pic2d.models import EV_J, ELECTRON_MASS_KG, PIC2DValidationError, ParticleArrays, xenon_ion_species

SPEC_DIR = Path(__file__).resolve().parents[2] / "spec" / "pic2d"


def test_cross_section_spec_is_tabulated_lxcat_with_bound_hashes():
    xs = XenonCrossSections.from_file()
    assert tuple(p.identifier for p in xs.processes) == PROCESS_ORDER
    assert xs.provenance["status"] == "lxcat-tabulated"
    assert any("Biagi" in citation for citation in xs.provenance["citations"])
    thresholds = [p.threshold_ev for p in xs.processes]
    assert thresholds == [0.0, 8.32, 12.13]
    for process in xs.processes[1:]:
        assert process.at(np.array([process.threshold_ev - 0.01]))[0] == 0.0
        assert process.source_bytes_sha256 if hasattr(process, "source_bytes_sha256") else True
    document = json.loads(DEFAULT_CROSS_SECTION_PATH.read_text(encoding="utf-8"))
    for item in document["processes"]:
        assert len(item["source_bytes_sha256"]) == 64
        assert (SPEC_DIR / item["source_file"]).is_file()
    sidecar = (SPEC_DIR / "xenon-cross-sections-v1.json.sha256").read_text(encoding="ascii")
    assert sidecar.split()[0] == xs.file_sha256
    # physical magnitude sanity (m2): Ramsauer minimum, peak ionisation
    elastic, excitation, ionization = xs.processes
    assert 1e-21 < elastic.at(np.array([0.65]))[0] < 5e-20  # momentum-transfer Ramsauer minimum
    assert 1e-19 < elastic.at(np.array([5.0]))[0] < 5e-19
    assert 3e-20 < ionization.at(np.array([100.0]))[0] < 7e-20
    assert 1e-20 < excitation.at(np.array([30.0]))[0] < 5e-20


def test_cross_section_payload_tamper_is_rejected(tmp_path: Path):
    document = json.loads(DEFAULT_CROSS_SECTION_PATH.read_text(encoding="utf-8"))
    payload = {key: value for key, value in document.items() if key != "integrity"}
    assert document["integrity"]["payload_sha256"] == canonical_payload_sha256(payload)
    document["processes"][2]["cross_section_m2"][-1] *= 1.01
    target = tmp_path / "tampered.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PIC2DValidationError):
        XenonCrossSections.from_file(target)


def test_uniform_table_matches_interpolation_and_hashes():
    xs = XenonCrossSections.from_file()
    table = UniformSigmaTable.build(xs, energy_step_ev=0.05, energy_max_ev=2000.0)
    energies = np.array([0.0, 0.37, 8.32, 12.13, 12.15, 33.3, 250.0, 1999.9, 5000.0])
    looked = table.lookup(energies)
    for k, process in enumerate(xs.processes):
        expected = process.at(np.minimum(energies, table.energy_max_ev))
        assert np.allclose(looked[k], expected, rtol=2e-2, atol=1e-22)  # grid straddles thresholds by <= 0.05 eV
    assert len(table.sha256()) == 64
    nu_max = maximum_collision_frequency(table, 1.0e21)
    assert 1e8 < nu_max < 5e9


def test_collision_rates_match_null_collision_expectation():
    xs = XenonCrossSections.from_file()
    config = MCCConfig(neutral_density_per_m3=2.0e21)
    operator = NullCollisionMCC(xs, config, xenon_ion_species(1.0))
    count = 400_000
    energy_ev = 40.0
    speed = float(electron_speed_from_energy(np.array([energy_ev]))[0])
    electrons = ParticleArrays(
        np.full(count, 1.0e-3), np.full(count, 12.0e-3), np.zeros(count), np.zeros(count), np.full(count, speed)
    )
    dt = 2.0e-11
    result = operator.apply(electrons, dt, np.random.default_rng(9))
    sigma = operator.table.lookup(np.array([energy_ev]))[:, 0]
    nu = config.neutral_density_per_m3 * sigma * speed
    probability_total = 1.0 - exp(-operator.nu_max * dt)
    assert probability_total < 0.1
    # each process is Bernoulli with p_k = nu_k dt (exact for the null-collision method: P_null * nu_k/nu_max)
    tally = result.tally
    for observed, rate in ((tally.elastic, nu[0]), (tally.excitation, nu[1]), (tally.ionization, nu[2])):
        p = probability_total * rate / operator.nu_max
        expected = count * p
        sigma_count = np.sqrt(count * p * (1.0 - p))
        assert abs(observed - expected) < 4.0 * sigma_count, (observed, expected, sigma_count)
    assert tally.candidates == tally.elastic + tally.excitation + tally.ionization + tally.null
    # kinematics: elastic preserves speed; excitation loses 8.32 eV; ionisation shares E - 12.13 eV
    new_energy = electron_energy_ev(result.electrons.vr_m_per_s, result.electrons.vt_m_per_s, result.electrons.vz_m_per_s)
    assert result.new_ions.count == tally.ionization == result.new_electrons.count
    changed = np.abs(new_energy - energy_ev) > 1e-9
    assert int(changed.sum()) == tally.excitation + tally.ionization
    secondary = electron_energy_ev(result.new_electrons.vr_m_per_s, result.new_electrons.vt_m_per_s, result.new_electrons.vz_m_per_s)
    primaries = np.sort(new_energy[changed])
    # excitation products all sit exactly at E - 8.32; ionisation products below E - 12.13
    assert np.isclose(primaries[-1], energy_ev - 8.32, atol=1e-6)
    assert np.all(secondary <= energy_ev - 12.13 + 1e-6) and np.all(secondary >= 0.0)
    assert tally.inelastic_energy_loss_j == pytest.approx((tally.excitation * 8.32 + tally.ionization * 12.13) * EV_J)
    assert np.allclose(result.new_ions.r_m, result.ionization_r_m) and np.allclose(result.new_ions.z_m, result.ionization_z_m)
    ion_thermal = np.sqrt(1.380649e-23 * 300.0 / xenon_ion_species(1.0).mass_kg)
    assert np.std(result.new_ions.vz_m_per_s) == pytest.approx(ion_thermal, rel=0.1)


def test_mcc_is_deterministic_for_a_seed_and_rejects_bad_config():
    xs = XenonCrossSections.synthetic_for_tests()
    operator = NullCollisionMCC(xs, MCCConfig(1.0e21), xenon_ion_species(1.0))
    electrons = ParticleArrays(np.full(1000, 1e-3), np.full(1000, 1e-2), np.zeros(1000), np.zeros(1000), np.full(1000, 3.0e6))
    a = operator.apply(electrons, 1e-11, np.random.default_rng(3))
    b = operator.apply(electrons, 1e-11, np.random.default_rng(3))
    assert np.array_equal(a.electrons.vz_m_per_s, b.electrons.vz_m_per_s)
    assert a.tally == b.tally
    with pytest.raises(PIC2DValidationError):
        MCCConfig(-1.0)
    with pytest.raises(PIC2DValidationError):
        MCCConfig(1e21, neutral_temperature_k=0.0)
