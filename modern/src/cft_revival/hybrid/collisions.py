"""Deterministic Monte Carlo heavy-species/background collision operators."""

from __future__ import annotations

from dataclasses import replace
from math import cos, expm1, fsum, pi, sin, sqrt
from typing import Iterable

from .models import (
    CollisionBatchResult,
    CollisionCrossSection,
    HybridValidationError,
    Particle,
    SourceExchange,
    Vec3,
    VelocityTimeLevel,
    finite_scalar,
    finite_vec3,
    validated_particle_batch,
)
from .reference import particle_kinetic_energy, particle_momentum
from .rng import random_uniform


def collision_frequency_per_s(
    particle: Particle,
    neutral_density_per_m3: float,
    cross_section: CollisionCrossSection,
    neutral_velocity_m_per_s: Vec3 = (0.0, 0.0, 0.0),
) -> float:
    if not isinstance(particle, Particle):
        raise HybridValidationError("particle must be a Particle")
    if not isinstance(cross_section, CollisionCrossSection):
        raise HybridValidationError("cross_section must be CollisionCrossSection")
    density = finite_scalar("neutral_density_per_m3", neutral_density_per_m3)
    if density < 0.0:
        raise HybridValidationError("neutral_density_per_m3 must be non-negative")
    neutral_velocity = finite_vec3("neutral_velocity_m_per_s", neutral_velocity_m_per_s)
    relative_speed = sqrt(
        fsum(
            (ion - neutral) ** 2
            for ion, neutral in zip(
                particle.velocity_m_per_s, neutral_velocity, strict=True
            )
        )
    )
    return finite_scalar(
        "collision_frequency_per_s",
        density * cross_section.sigma_m2 * relative_speed,
    )


def collision_probability(frequency_per_s: float, dt_s: float) -> float:
    frequency = finite_scalar("frequency_per_s", frequency_per_s)
    dt = finite_scalar("dt_s", dt_s)
    if frequency < 0.0 or dt < 0.0:
        raise HybridValidationError("frequency_per_s and dt_s must be non-negative")
    return -expm1(-frequency * dt)


def collide_with_neutral_background(
    particles: Iterable[Particle],
    *,
    neutral_density_per_m3: float,
    cross_section: CollisionCrossSection,
    dt_s: float,
    seed: int,
    step: int,
    neutral_velocity_m_per_s: Vec3 = (0.0, 0.0, 0.0),
) -> CollisionBatchResult:
    """Apply independent elastic or charge-exchange Bernoulli events.

    The fixture represents a prescribed neutral reservoir. Every ion momentum
    and kinetic-energy change is recorded with an equal and opposite reservoir
    source; no unresolved electron or wall sink is hidden.
    """

    dt = finite_scalar("dt_s", dt_s)
    if dt < 0.0:
        raise HybridValidationError("dt_s must be non-negative")
    density = finite_scalar("neutral_density_per_m3", neutral_density_per_m3)
    if density < 0.0:
        raise HybridValidationError("neutral_density_per_m3 must be non-negative")
    if not isinstance(cross_section, CollisionCrossSection):
        raise HybridValidationError("cross_section must be CollisionCrossSection")
    for name, value in (("seed", seed), ("step", step)):
        if (
            type(value) is not int
            or not 0 <= value < 1 << 64
        ):
            raise HybridValidationError(
                f"{name} must be an unsigned 64-bit integer"
            )
    neutral_velocity = finite_vec3("neutral_velocity_m_per_s", neutral_velocity_m_per_s)
    input_batch = validated_particle_batch(particles)
    canonical_batch = tuple(
        sorted(input_batch, key=lambda particle: particle.particle_id)
    )
    if any(
        particle.velocity_time_level
        is not VelocityTimeLevel.LEAPFROG_N_MINUS_HALF
        for particle in canonical_batch
    ):
        raise HybridValidationError(
            "collision advance requires leapfrog_n_minus_one_half velocity"
        )
    if cross_section.process == "charge_exchange" and any(
        particle.alive and particle.species.charge_state not in {0, 1}
        for particle in canonical_batch
    ):
        raise HybridValidationError(
            "charge_exchange currently supports Xe+ only; Xe2+ products "
            "and source accounting are not implemented"
        )
    output_by_id: dict[int, Particle] = {}
    collisions = 0
    probabilities: list[float] = []
    momentum_deltas: list[Vec3] = []
    energy_deltas: list[float] = []

    for particle in canonical_batch:
        if not particle.alive or particle.species.charge_state == 0:
            output_by_id[particle.particle_id] = particle
            continue
        frequency = collision_frequency_per_s(
            particle,
            density,
            cross_section,
            neutral_velocity,
        )
        probability = collision_probability(frequency, dt)
        probabilities.append(probability)
        if random_uniform(seed, particle.particle_id, step, stream=0) >= probability:
            output_by_id[particle.particle_id] = particle
            continue

        collisions += 1
        if cross_section.process == "charge_exchange":
            velocity_new = neutral_velocity
        else:
            relative = tuple(
                ion - neutral
                for ion, neutral in zip(
                    particle.velocity_m_per_s, neutral_velocity, strict=True
                )
            )
            speed = sqrt(fsum(component * component for component in relative))
            cosine = 2.0 * random_uniform(
                seed, particle.particle_id, step, stream=1, draw=0
            ) - 1.0
            azimuth = 2.0 * pi * random_uniform(
                seed, particle.particle_id, step, stream=1, draw=1
            )
            transverse = sqrt(max(0.0, 1.0 - cosine * cosine))
            scattered_relative = (
                speed * transverse * cos(azimuth),
                speed * transverse * sin(azimuth),
                speed * cosine,
            )
            velocity_new = tuple(
                neutral + scattered
                for neutral, scattered in zip(
                    neutral_velocity, scattered_relative, strict=True
                )
            )

        changed = replace(particle, velocity_m_per_s=velocity_new)
        old_momentum = particle_momentum(particle)
        new_momentum = particle_momentum(changed)
        momentum_deltas.append(
            tuple(  # type: ignore[arg-type]
                new_momentum[axis] - old_momentum[axis] for axis in range(3)
            )
        )
        energy_deltas.append(
            particle_kinetic_energy(changed) - particle_kinetic_energy(particle)
        )
        output_by_id[particle.particle_id] = changed

    ion_momentum_tuple = tuple(
        fsum(delta[axis] for delta in momentum_deltas) for axis in range(3)
    )
    ion_energy = fsum(energy_deltas)
    return CollisionBatchResult(
        particles=tuple(output_by_id[particle.particle_id] for particle in input_batch),
        collision_count=collisions,
        expected_collision_count=fsum(probabilities),
        source_exchange=SourceExchange(
            ion_momentum_delta_kg_m_per_s=ion_momentum_tuple,  # type: ignore[arg-type]
            background_momentum_delta_kg_m_per_s=tuple(  # type: ignore[arg-type]
                -component for component in ion_momentum_tuple
            ),
            ion_energy_delta_j=ion_energy,
            background_energy_delta_j=-ion_energy,
        ),
        species_count_delta=(),
        represented_charge_delta_c=0.0,
    )
