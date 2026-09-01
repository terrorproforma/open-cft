from dataclasses import replace
from math import sqrt

import pytest

from cft_revival.hybrid import (
    CartesianGrid1D,
    CollisionCrossSection,
    HybridValidationError,
    IsothermalQuasineutralClosure,
    Particle,
    SourceExchange,
    VelocityTimeLevel,
    XE_DOUBLE_PLUS,
    XE_PLUS,
    collide_with_neutral_background,
    collision_frequency_per_s,
    collision_probability,
    conservative_electron_exchange,
    deposit_cic_periodic,
    particle_kinetic_energy,
    random_u64,
    random_uniform,
)


def test_counter_rng_is_repeatable_keyed_and_order_independent() -> None:
    first = random_uniform(1234, 99, 7, stream=2, draw=3)
    assert first == random_uniform(1234, 99, 7, stream=2, draw=3)
    assert first != random_uniform(1234, 99, 8, stream=2, draw=3)
    assert 0.0 <= first < 1.0


@pytest.mark.parametrize(
    "counter_name",
    ["seed", "particle_id", "step", "stream", "draw"],
)
@pytest.mark.parametrize("invalid_value", [True, False, 1.0, 1.5])
def test_counter_rng_rejects_bool_and_non_integral_values(
    counter_name: str, invalid_value: object
) -> None:
    counters = {
        "seed": 1,
        "particle_id": 2,
        "step": 3,
        "stream": 4,
        "draw": 5,
    }
    counters[counter_name] = invalid_value
    with pytest.raises(HybridValidationError, match="unsigned 64-bit"):
        random_u64(**counters)  # type: ignore[arg-type]


def test_collision_frequency_and_bernoulli_statistics() -> None:
    fixture = CollisionCrossSection("charge_exchange", 1.0e-19)
    prototype = Particle(0, XE_PLUS, (0.5, 0.0, 0.0), (10_000.0, 0.0, 0.0))
    frequency = collision_frequency_per_s(prototype, 1.0e19, fixture)
    assert frequency == pytest.approx(10_000.0)
    probability = collision_probability(frequency, 1.0e-5)
    assert probability == pytest.approx(0.09516258196404043)

    count = 4096
    particles = tuple(replace(prototype, particle_id=index) for index in range(count))
    result = collide_with_neutral_background(
        particles,
        neutral_density_per_m3=1.0e19,
        cross_section=fixture,
        dt_s=1.0e-5,
        seed=8080,
        step=4,
    )
    expected = count * probability
    standard_deviation = sqrt(count * probability * (1.0 - probability))
    assert result.expected_collision_count == pytest.approx(expected)
    assert abs(result.collision_count - expected) <= 6.0 * standard_deviation

    reversed_result = collide_with_neutral_background(
        tuple(reversed(particles)),
        neutral_density_per_m3=1.0e19,
        cross_section=fixture,
        dt_s=1.0e-5,
        seed=8080,
        step=4,
    )
    by_id = {particle.particle_id: particle.velocity_m_per_s for particle in result.particles}
    reversed_by_id = {
        particle.particle_id: particle.velocity_m_per_s
        for particle in reversed_result.particles
    }
    assert by_id == reversed_by_id
    assert result.expected_collision_count == reversed_result.expected_collision_count
    assert result.source_exchange == reversed_result.source_exchange
    assert result.collision_count == reversed_result.collision_count


def test_charge_exchange_records_exact_reservoir_exchange() -> None:
    particle = Particle(10, XE_PLUS, (0.5, 0.0, 0.0), (100.0, -20.0, 3.0), weight=2.0)
    result = collide_with_neutral_background(
        (particle,),
        neutral_density_per_m3=1.0e30,
        cross_section=CollisionCrossSection("charge_exchange", 1.0e-19),
        dt_s=1.0,
        seed=1,
        step=0,
    )
    assert result.collision_count == 1
    assert result.particles[0].velocity_m_per_s == (0.0, 0.0, 0.0)
    assert result.source_exchange.momentum_residual_kg_m_per_s == (0.0, 0.0, 0.0)
    assert result.source_exchange.energy_residual_j == 0.0
    assert result.source_exchange.ion_energy_delta_j == -particle_kinetic_energy(particle)
    assert result.particles[0].species == particle.species
    assert result.species_count_delta == ()
    assert result.represented_charge_delta_c == 0.0
    assert result.particles[0].represented_charge_c == particle.represented_charge_c


def test_elastic_scattering_preserves_speed_and_accounts_momentum() -> None:
    particle = Particle(11, XE_DOUBLE_PLUS, (0.5, 0.0, 0.0), (100.0, -20.0, 3.0))
    result = collide_with_neutral_background(
        (particle,),
        neutral_density_per_m3=1.0e30,
        cross_section=CollisionCrossSection("elastic", 1.0e-19),
        dt_s=1.0,
        seed=5,
        step=2,
    )
    assert result.collision_count == 1
    assert particle_kinetic_energy(result.particles[0]) == pytest.approx(
        particle_kinetic_energy(particle), rel=2.0e-16
    )
    assert result.source_exchange.momentum_residual_kg_m_per_s == pytest.approx(
        (0.0, 0.0, 0.0), abs=0.0
    )
    assert result.source_exchange.energy_residual_j == pytest.approx(0.0, abs=1.0e-36)


def test_fluid_electron_interface_is_quasineutral_and_transport_unresolved() -> None:
    particles = (
        Particle(0, XE_PLUS, (0.25, 0.0, 0.0), (0.0, 0.0, 0.0)),
        Particle(1, XE_DOUBLE_PLUS, (0.75, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    moments = deposit_cic_periodic(particles, CartesianGrid1D(0.0, 1.0, 4))
    exchange = conservative_electron_exchange((1.0e-24, -2.0e-24, 0.0), 3.0e-20)
    result = IsothermalQuasineutralClosure(15_000.0).close(moments, exchange)
    assert result.state.anomalous_mobility_m2_per_v_s is None
    assert result.electric_field_v_per_m is None
    assert result.source_exchange.energy_residual_j == 0.0
    assert all(value >= 0.0 for value in result.state.number_density_per_m3)
    assert all(value > 0.0 for value in result.state.pressure_pa if value != 0.0)


def test_collision_and_closure_failure_cases() -> None:
    particle = Particle(0, XE_PLUS, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    fixture = CollisionCrossSection("elastic", 1.0e-19)
    with pytest.raises(HybridValidationError, match="non-negative"):
        collision_frequency_per_s(particle, -1.0, fixture)
    with pytest.raises(HybridValidationError, match="non-negative"):
        collision_probability(1.0, -1.0)
    with pytest.raises(HybridValidationError, match="non-negative"):
        collide_with_neutral_background(
            (),
            neutral_density_per_m3=-1.0,
            cross_section=fixture,
            dt_s=0.0,
            seed=1,
            step=0,
        )
    moments = deposit_cic_periodic((particle,), CartesianGrid1D(0.0, 1.0, 4))
    nonconservative = SourceExchange(ion_energy_delta_j=1.0)
    with pytest.raises(HybridValidationError, match="conservative"):
        IsothermalQuasineutralClosure(10_000.0).close(moments, nonconservative)


def test_xe_double_plus_charge_exchange_is_explicitly_unsupported() -> None:
    particle = Particle(
        77,
        XE_DOUBLE_PLUS,
        (0.5, 0.0, 0.0),
        (100.0, 0.0, 0.0),
    )
    with pytest.raises(HybridValidationError, match="Xe2\\+ products"):
        collide_with_neutral_background(
            (particle,),
            neutral_density_per_m3=0.0,
            cross_section=CollisionCrossSection(
                "charge_exchange", 1.0e-19
            ),
            dt_s=0.0,
            seed=1,
            step=0,
        )


def test_duplicate_rng_identity_fails_before_collision() -> None:
    particles = (
        Particle(88, XE_PLUS, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
        Particle(88, XE_PLUS, (0.6, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    with pytest.raises(HybridValidationError, match="unique"):
        collide_with_neutral_background(
            particles,
            neutral_density_per_m3=1.0,
            cross_section=CollisionCrossSection("elastic", 1.0e-19),
            dt_s=1.0,
            seed=1,
            step=0,
        )


def test_synchronous_velocity_cannot_enter_collision_advance() -> None:
    particle = Particle(
        89,
        XE_PLUS,
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        velocity_time_level=VelocityTimeLevel.SYNCHRONOUS_N,
    )
    with pytest.raises(HybridValidationError, match="leapfrog"):
        collide_with_neutral_background(
            (particle,),
            neutral_density_per_m3=1.0,
            cross_section=CollisionCrossSection("elastic", 1.0e-19),
            dt_s=1.0,
            seed=1,
            step=0,
        )
