"""Dependency-free CPU reference algorithms for the hybrid first slice."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import floor, fsum
from typing import Iterable

from .models import (
    AxisAlignedBox,
    BorisStepResult,
    BoundaryPolicy,
    CartesianGrid1D,
    DepositedMoments,
    HybridValidationError,
    Particle,
    SourceExchange,
    UniformFields,
    Vec3,
    VelocityTimeLevel,
    finite_scalar,
    validated_particle_batch,
)


def _add(left: Vec3, right: Vec3) -> Vec3:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _scale(value: Vec3, factor: float) -> Vec3:
    return tuple(factor * entry for entry in value)  # type: ignore[return-value]


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm_squared(value: Vec3) -> float:
    return fsum(entry * entry for entry in value)


def particle_momentum(particle: Particle) -> Vec3:
    return _scale(particle.velocity_m_per_s, particle.represented_mass_kg)


def particle_kinetic_energy(particle: Particle) -> float:
    return 0.5 * particle.represented_mass_kg * _norm_squared(particle.velocity_m_per_s)


def _boris_velocity_stages(
    particle: Particle,
    fields: UniformFields,
    dt: float,
) -> tuple[Vec3, Vec3, Vec3]:
    charge_to_mass = particle.species.charge_c / particle.species.mass_kg
    half_acceleration = _scale(fields.electric_v_per_m, 0.5 * charge_to_mass * dt)
    velocity_minus = _add(particle.velocity_m_per_s, half_acceleration)
    t_vector = _scale(fields.magnetic_t, 0.5 * charge_to_mass * dt)
    t_squared = _norm_squared(t_vector)
    s_vector = _scale(t_vector, 2.0 / (1.0 + t_squared))
    velocity_prime = _add(velocity_minus, _cross(velocity_minus, t_vector))
    velocity_plus = _add(velocity_minus, _cross(velocity_prime, s_vector))
    velocity_new = _add(velocity_plus, half_acceleration)
    return velocity_minus, velocity_plus, velocity_new


def _require_time_level(
    particle: Particle,
    expected: VelocityTimeLevel,
    operation: str,
) -> None:
    if not isinstance(particle, Particle):
        raise HybridValidationError(f"{operation} requires a Particle")
    if particle.velocity_time_level is not expected:
        raise HybridValidationError(
            f"{operation} requires velocity_time_level={expected.value}"
        )


def initialize_leapfrog(
    particle_at_n: Particle,
    fields_at_n: UniformFields,
    dt_s: float,
) -> Particle:
    """Map synchronous ``(x^n, v^n)`` to ``(x^n, v^(n-1/2))``."""

    _require_time_level(
        particle_at_n, VelocityTimeLevel.SYNCHRONOUS_N, "initialize_leapfrog"
    )
    if not isinstance(fields_at_n, UniformFields):
        raise HybridValidationError("fields_at_n must be UniformFields")
    dt = finite_scalar("dt_s", dt_s)
    if dt < 0.0:
        raise HybridValidationError("dt_s must be non-negative")
    if not particle_at_n.alive:
        return replace(
            particle_at_n,
            velocity_time_level=VelocityTimeLevel.LEAPFROG_N_MINUS_HALF,
        )
    _, _, velocity_half = _boris_velocity_stages(
        particle_at_n, fields_at_n, -0.5 * dt
    )
    return replace(
        particle_at_n,
        velocity_m_per_s=velocity_half,
        velocity_time_level=VelocityTimeLevel.LEAPFROG_N_MINUS_HALF,
    )


def synchronize_velocity(
    particle_at_n: Particle,
    fields_at_n: UniformFields,
    dt_s: float,
) -> Particle:
    """Map ``(x^n, v^(n-1/2))`` to a diagnostic ``(x^n, v^n)``."""

    _require_time_level(
        particle_at_n,
        VelocityTimeLevel.LEAPFROG_N_MINUS_HALF,
        "synchronize_velocity",
    )
    if not isinstance(fields_at_n, UniformFields):
        raise HybridValidationError("fields_at_n must be UniformFields")
    dt = finite_scalar("dt_s", dt_s)
    if dt < 0.0:
        raise HybridValidationError("dt_s must be non-negative")
    if not particle_at_n.alive:
        return replace(
            particle_at_n,
            velocity_time_level=VelocityTimeLevel.SYNCHRONOUS_N,
        )
    _, _, velocity_n = _boris_velocity_stages(
        particle_at_n, fields_at_n, 0.5 * dt
    )
    return replace(
        particle_at_n,
        velocity_m_per_s=velocity_n,
        velocity_time_level=VelocityTimeLevel.SYNCHRONOUS_N,
    )


def boris_push_diagnosed(
    particle: Particle,
    fields: UniformFields,
    dt_s: float,
) -> BorisStepResult:
    """Advance ``x^n,v^(n-1/2)`` to ``x^(n+1),v^(n+1/2)``."""

    _require_time_level(
        particle, VelocityTimeLevel.LEAPFROG_N_MINUS_HALF, "boris_push"
    )
    if not isinstance(fields, UniformFields):
        raise HybridValidationError("fields must be UniformFields")
    dt = finite_scalar("dt_s", dt_s)
    if dt < 0.0:
        raise HybridValidationError("dt_s must be non-negative")
    if dt == 0.0 or not particle.alive:
        return BorisStepResult(particle, 0.0, 0.0)

    velocity_minus, velocity_plus, velocity_new = _boris_velocity_stages(
        particle, fields, dt
    )
    position_new = _add(particle.position_m, _scale(velocity_new, dt))
    advanced = replace(
        particle,
        position_m=position_new,
        velocity_m_per_s=velocity_new,
    )
    quarter_dt_charge = 0.25 * dt * particle.represented_charge_c
    electric_work = quarter_dt_charge * fsum(
        fields.electric_v_per_m[axis]
        * (
            particle.velocity_m_per_s[axis]
            + velocity_minus[axis]
            + velocity_plus[axis]
            + velocity_new[axis]
        )
        for axis in range(3)
    )
    kinetic_delta = particle_kinetic_energy(advanced) - particle_kinetic_energy(
        particle
    )
    return BorisStepResult(advanced, electric_work, kinetic_delta)


def boris_push(particle: Particle, fields: UniformFields, dt_s: float) -> Particle:
    """Return the particle from :func:`boris_push_diagnosed`."""

    return boris_push_diagnosed(particle, fields, dt_s).particle


@dataclass(frozen=True, slots=True)
class BoundaryResult:
    particle: Particle
    source_exchange: SourceExchange


def apply_boundary(particle: Particle, box: AxisAlignedBox) -> BoundaryResult:
    """Apply one box policy and account wall exchange without hidden sinks."""

    if not particle.alive:
        return BoundaryResult(particle, SourceExchange())
    outside = any(
        coordinate < low or coordinate > high
        for coordinate, low, high in zip(
            particle.position_m, box.lower_m, box.upper_m, strict=True
        )
    )
    if not outside:
        return BoundaryResult(particle, SourceExchange())

    if box.policy is BoundaryPolicy.ABSORBING:
        momentum = particle_momentum(particle)
        energy = particle_kinetic_energy(particle)
        removed = replace(particle, alive=False)
        return BoundaryResult(
            removed,
            SourceExchange(
                ion_momentum_delta_kg_m_per_s=_scale(momentum, -1.0),
                background_momentum_delta_kg_m_per_s=momentum,
                ion_energy_delta_j=-energy,
                background_energy_delta_j=energy,
            ),
        )

    position = list(particle.position_m)
    velocity = list(particle.velocity_m_per_s)
    for axis, (low, high) in enumerate(zip(box.lower_m, box.upper_m, strict=True)):
        length = high - low
        coordinate = position[axis]
        if low <= coordinate <= high:
            continue
        if box.policy is BoundaryPolicy.PERIODIC:
            position[axis] = low + ((coordinate - low) % length)
        else:
            phase = (coordinate - low) % (2.0 * length)
            if phase <= length:
                position[axis] = low + phase
            else:
                position[axis] = high - (phase - length)
                velocity[axis] = -velocity[axis]

    bounded = replace(
        particle,
        position_m=tuple(position),  # type: ignore[arg-type]
        velocity_m_per_s=tuple(velocity),  # type: ignore[arg-type]
    )
    if box.policy is BoundaryPolicy.PERIODIC:
        return BoundaryResult(bounded, SourceExchange())

    ion_delta = _add(particle_momentum(bounded), _scale(particle_momentum(particle), -1.0))
    return BoundaryResult(
        bounded,
        SourceExchange(
            ion_momentum_delta_kg_m_per_s=ion_delta,
            background_momentum_delta_kg_m_per_s=_scale(ion_delta, -1.0),
        ),
    )


def advance_particles(
    particles: Iterable[Particle],
    fields: UniformFields,
    dt_s: float,
    box: AxisAlignedBox | None = None,
) -> tuple[tuple[Particle, ...], SourceExchange]:
    if not isinstance(fields, UniformFields):
        raise HybridValidationError("fields must be UniformFields")
    if box is not None and not isinstance(box, AxisAlignedBox):
        raise HybridValidationError("box must be AxisAlignedBox or None")
    dt = finite_scalar("dt_s", dt_s)
    if dt < 0.0:
        raise HybridValidationError("dt_s must be non-negative")
    batch = validated_particle_batch(particles)
    advanced: list[Particle] = []
    ion_momentum = [0.0, 0.0, 0.0]
    background_momentum = [0.0, 0.0, 0.0]
    ion_energy = 0.0
    background_energy = 0.0
    for particle in batch:
        pushed = boris_push(particle, fields, dt)
        boundary = (
            BoundaryResult(pushed, SourceExchange())
            if box is None
            else apply_boundary(pushed, box)
        )
        advanced.append(boundary.particle)
        exchange = boundary.source_exchange
        for axis in range(3):
            ion_momentum[axis] += exchange.ion_momentum_delta_kg_m_per_s[axis]
            background_momentum[axis] += exchange.background_momentum_delta_kg_m_per_s[axis]
        ion_energy += exchange.ion_energy_delta_j
        background_energy += exchange.background_energy_delta_j
    return (
        tuple(advanced),
        SourceExchange(
            ion_momentum_delta_kg_m_per_s=tuple(ion_momentum),  # type: ignore[arg-type]
            background_momentum_delta_kg_m_per_s=tuple(  # type: ignore[arg-type]
                background_momentum
            ),
            ion_energy_delta_j=ion_energy,
            background_energy_delta_j=background_energy,
        ),
    )


def deposit_cic_periodic(
    particles: Iterable[Particle],
    grid: CartesianGrid1D,
) -> DepositedMoments:
    """Deposit cell-centred 1-D CIC moments with conservative normalization."""

    if not isinstance(grid, CartesianGrid1D):
        raise HybridValidationError("grid must be CartesianGrid1D")
    count = grid.cell_count
    number = [0.0] * count
    charge = [0.0] * count
    current = [[0.0, 0.0, 0.0] for _ in range(count)]
    momentum = [[0.0, 0.0, 0.0] for _ in range(count)]
    energy = [0.0] * count
    spacing = grid.spacing_m
    volume = grid.cell_volume_m3

    batch = validated_particle_batch(particles, canonical_order=True)
    if any(
        particle.velocity_time_level
        is not VelocityTimeLevel.LEAPFROG_N_MINUS_HALF
        for particle in batch
    ):
        raise HybridValidationError(
            "deposition requires leapfrog_n_minus_one_half velocity"
        )
    for particle in batch:
        if not particle.alive:
            continue
        x = particle.position_m[0]
        if x < grid.x_min_m or x > grid.x_max_m:
            raise HybridValidationError("particle x lies outside the deposition domain")
        normalized = ((x - grid.x_min_m) % (grid.x_max_m - grid.x_min_m)) / spacing - 0.5
        left_raw = floor(normalized)
        fraction_right = normalized - left_raw
        nodes = (left_raw % count, (left_raw + 1) % count)
        shape_weights = (1.0 - fraction_right, fraction_right)
        represented_mass = particle.represented_mass_kg
        represented_charge = particle.represented_charge_c
        kinetic_energy = particle_kinetic_energy(particle)
        for node, shape in zip(nodes, shape_weights, strict=True):
            density_factor = shape / volume
            number[node] += particle.weight * density_factor
            charge[node] += represented_charge * density_factor
            energy[node] += kinetic_energy * density_factor
            for axis in range(3):
                velocity = particle.velocity_m_per_s[axis]
                current[node][axis] += represented_charge * velocity * density_factor
                momentum[node][axis] += represented_mass * velocity * density_factor

    return DepositedMoments(
        number_per_m3=tuple(number),
        charge_c_per_m3=tuple(charge),
        current_a_per_m2=tuple(tuple(row) for row in current),  # type: ignore[arg-type]
        momentum_kg_per_m2_s=tuple(tuple(row) for row in momentum),  # type: ignore[arg-type]
        kinetic_energy_j_per_m3=tuple(energy),
    )
