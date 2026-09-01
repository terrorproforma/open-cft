"""Tiny manufactured prescribed-field run for integration verification."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum

from .electrons import IsothermalQuasineutralClosure
from .models import (
    CartesianGrid1D,
    DepositedMoments,
    ElectronClosureResult,
    HybridValidationError,
    Particle,
    UniformFields,
    VelocityTimeLevel,
    XE,
    XE_DOUBLE_PLUS,
    XE_PLUS,
    finite_scalar,
)
from .reference import (
    boris_push_diagnosed,
    deposit_cic_periodic,
    initialize_leapfrog,
    particle_kinetic_energy,
)


@dataclass(frozen=True, slots=True)
class ManufacturedRunResult:
    """Auditable result from a verification fixture, not a thruster prediction."""

    particles: tuple[Particle, ...]
    moments: DepositedMoments
    electrons: ElectronClosureResult
    initial_kinetic_energy_j: float
    final_kinetic_energy_j: float
    electric_work_j: float
    step_count: int
    dt_s: float
    claim: str = "manufactured prescribed-field verification fixture only"
    time_integration_contract: str = "x^n,v^(n-1/2);E^n,B^n"

    @property
    def work_energy_residual_j(self) -> float:
        return (
            self.final_kinetic_energy_j
            - self.initial_kinetic_energy_j
            - self.electric_work_j
        )


def run_tiny_manufactured_case(
    *,
    step_count: int = 4,
    dt_s: float = 1.0e-8,
) -> ManufacturedRunResult:
    if not isinstance(step_count, int) or step_count < 0 or step_count > 32:
        raise HybridValidationError("step_count must be an integer in [0, 32]")
    dt = finite_scalar("dt_s", dt_s)
    if dt < 0.0:
        raise HybridValidationError("dt_s must be non-negative")
    fields = UniformFields(electric_v_per_m=(25.0, 0.0, 0.0))
    synchronous_particles = (
        Particle(
            0,
            XE,
            (0.20, 0.0, 0.0),
            (8.0, 0.0, 0.0),
            weight=2.0,
            velocity_time_level=VelocityTimeLevel.SYNCHRONOUS_N,
        ),
        Particle(
            1,
            XE_PLUS,
            (0.45, 0.0, 0.0),
            (10.0, 1.0, 0.0),
            weight=3.0,
            velocity_time_level=VelocityTimeLevel.SYNCHRONOUS_N,
        ),
        Particle(
            2,
            XE_DOUBLE_PLUS,
            (0.70, 0.0, 0.0),
            (12.0, -1.0, 0.0),
            velocity_time_level=VelocityTimeLevel.SYNCHRONOUS_N,
        ),
    )
    particles = tuple(
        initialize_leapfrog(particle, fields, dt)
        for particle in synchronous_particles
    )
    initial_energy = fsum(particle_kinetic_energy(particle) for particle in particles)
    electric_work = 0.0
    for _ in range(step_count):
        diagnosed = tuple(
            boris_push_diagnosed(particle, fields, dt)
            for particle in particles
        )
        particles = tuple(result.particle for result in diagnosed)
        electric_work += fsum(result.electric_work_j for result in diagnosed)
    final_energy = fsum(particle_kinetic_energy(particle) for particle in particles)
    moments = deposit_cic_periodic(particles, CartesianGrid1D(0.0, 1.0, 8))
    electrons = IsothermalQuasineutralClosure(20_000.0).close(moments)
    return ManufacturedRunResult(
        particles=particles,
        moments=moments,
        electrons=electrons,
        initial_kinetic_energy_j=initial_energy,
        final_kinetic_energy_j=final_energy,
        electric_work_j=electric_work,
        step_count=step_count,
        dt_s=dt,
    )
