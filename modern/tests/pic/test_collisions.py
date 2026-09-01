from __future__ import annotations

from math import exp, fsum, sqrt
import json

import pytest

from cft_revival.pic import (
    CrossSectionTable,
    ElasticMCC,
    PICValidationError,
    ParticleState,
    Species,
)


def _constant_table() -> CrossSectionTable:
    return CrossSectionTable(
        process="synthetic isotropic elastic",
        energy_ev=(0.0, 100.0),
        cross_section_m2=(2.0e-20, 2.0e-20),
        source="synthetic-verification:constant-cross-section-v1",
    )


def _particles(count: int, speed: float) -> ParticleState:
    return ParticleState(
        [0.0] * count,
        [speed] * count,
        [0.0] * count,
        [0.0] * count,
    )


def test_seeded_mcc_is_bitwise_deterministic_and_speed_conserving() -> None:
    species = Species("electron-like", charge_c=-1.0e-19, mass_kg=9.0e-31)
    left = _particles(500, 2.0e5)
    right = left.copy()
    first = ElasticMCC(_constant_table(), 1.0e20, seed=8128)
    second = ElasticMCC(_constant_table(), 1.0e20, seed=8128)
    first_result = first.apply(species, left, 1.0e-8)
    second_result = second.apply(species, right, 1.0e-8)
    assert first_result == second_result
    assert left == right
    for vx, vy, vz in zip(
        left.vx_m_per_s, left.vy_m_per_s, left.vz_m_per_s, strict=True
    ):
        assert sqrt(vx * vx + vy * vy + vz * vz) == pytest.approx(2.0e5, rel=2.0e-16)


def test_collision_count_matches_binomial_rate_statistics() -> None:
    count = 20_000
    speed = 1.0e5
    density = 1.0e20
    dt = 2.0e-8
    sigma = 2.0e-20
    probability = 1.0 - exp(-density * sigma * speed * dt)
    species = Species("electron-like", charge_c=-1.0e-19, mass_kg=9.0e-31)
    diagnostics = ElasticMCC(
        _constant_table(), density, seed=20260901
    ).apply(species, _particles(count, speed), dt)
    standard_deviation = sqrt(count * probability * (1.0 - probability))
    assert diagnostics.expected_collisions == pytest.approx(count * probability, rel=1.0e-12)
    assert abs(diagnostics.accepted_collisions - count * probability) < 5.0 * standard_deviation


def test_mcc_rejects_unresolved_collision_probability() -> None:
    species = Species("electron-like", charge_c=-1.0e-19, mass_kg=9.0e-31)
    operator = ElasticMCC(_constant_table(), 1.0e22, seed=1)
    with pytest.raises(PICValidationError, match="probability"):
        operator.apply(species, _particles(1, 1.0e6), 1.0e-6)


def test_late_particle_failure_leaves_particles_rng_and_counters_identical() -> None:
    species = Species("electron-like", charge_c=-1.0e-19, mass_kg=9.0e-31)
    particles = ParticleState(
        [0.0, 0.0],
        [1.0e4, 1.0e308],
        [0.0, 0.0],
        [0.0, 0.0],
    )
    operator = ElasticMCC(_constant_table(), 1.0e20, seed=99)
    particle_bytes = json.dumps(
        {
            "x": particles.x_m,
            "vx": particles.vx_m_per_s,
            "vy": particles.vy_m_per_s,
            "vz": particles.vz_m_per_s,
        },
        separators=(",", ":"),
    ).encode()
    rng_before = repr(operator.rng.getstate()).encode()
    counters_before = (operator.trial_count, operator.accepted_count)
    with pytest.raises(PICValidationError, match="particle 1"):
        operator.apply(species, particles, 1.0e-8)
    particle_after = json.dumps(
        {
            "x": particles.x_m,
            "vx": particles.vx_m_per_s,
            "vy": particles.vy_m_per_s,
            "vz": particles.vz_m_per_s,
        },
        separators=(",", ":"),
    ).encode()
    assert particle_after == particle_bytes
    assert repr(operator.rng.getstate()).encode() == rng_before
    assert (operator.trial_count, operator.accepted_count) == counters_before


def test_cross_section_interpolation_hash_and_external_provenance_contract() -> None:
    table = CrossSectionTable(
        "synthetic ramp",
        (0.0, 5.0, 10.0),
        (0.0, 2.0e-20, 4.0e-20),
        "synthetic-verification:linear-interpolation",
    )
    assert table.at_energy_ev(7.5) == pytest.approx(3.0e-20)
    assert len(table.table_sha256) == 64
    with pytest.raises(PICValidationError, match="SHA-256"):
        CrossSectionTable(
            "untraceable",
            (0.0, 1.0),
            (1.0e-20, 1.0e-20),
            "LXCat unknown export",
        )
    assert fsum(table.cross_section_m2) == pytest.approx(6.0e-20)
