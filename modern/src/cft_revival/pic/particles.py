"""Particle gather and leapfrog/Boris pushers."""

from __future__ import annotations

from math import isfinite
from typing import Sequence

from .electrostatic import gather_face_cic
from .models import Grid1D, PICValidationError, ParticleState, Species


def push_electrostatic_leapfrog(
    grid: Grid1D,
    species: Species,
    particles: ParticleState,
    electric_field_v_per_m: Sequence[float],
    dt_s: float,
) -> None:
    """Advance half-step velocity then periodic position, in place."""

    dt = float(dt_s)
    if not isfinite(dt) or dt <= 0.0:
        raise PICValidationError("dt_s must be finite and positive")
    particles.validate()
    electric_at_particles = gather_face_cic(
        grid, electric_field_v_per_m, particles.x_m
    )
    acceleration_scale = species.charge_c * dt / species.mass_kg
    if not isfinite(acceleration_scale):
        raise PICValidationError("electrostatic acceleration scale is not representable")
    proposed_velocity: list[float] = []
    proposed_position: list[float] = []
    for index, electric in enumerate(electric_at_particles):
        velocity = particles.vx_m_per_s[index] + acceleration_scale * electric
        position = grid.wrap(particles.x_m[index] + velocity * dt)
        if not isfinite(velocity) or not isfinite(position):
            raise PICValidationError("electrostatic push produced a nonfinite state")
        proposed_velocity.append(velocity)
        proposed_position.append(position)
    particles.vx_m_per_s[:] = proposed_velocity
    particles.x_m[:] = proposed_position


def boris_push_uniform(
    species: Species,
    particles: ParticleState,
    electric_field_v_per_m: tuple[float, float, float],
    magnetic_field_t: tuple[float, float, float],
    dt_s: float,
) -> None:
    """Advance velocity with the standard non-relativistic Boris rotation.

    This standalone kernel does not move positions because the verified grid is
    one-dimensional; it establishes the 3-V pusher boundary for later 2-D/3-D
    and axisymmetric adapters.
    """

    dt = float(dt_s)
    values = (*electric_field_v_per_m, *magnetic_field_t, dt)
    if any(not isfinite(float(value)) for value in values) or dt <= 0.0:
        raise PICValidationError("Boris fields must be finite and dt_s positive")
    particles.validate()
    qmdt2 = species.charge_c * dt / (2.0 * species.mass_kg)
    if not isfinite(qmdt2):
        raise PICValidationError("Boris charge-to-mass timestep is not representable")
    ex, ey, ez = electric_field_v_per_m
    tx, ty, tz = (qmdt2 * value for value in magnetic_field_t)
    t_squared = tx * tx + ty * ty + tz * tz
    sx = 2.0 * tx / (1.0 + t_squared)
    sy = 2.0 * ty / (1.0 + t_squared)
    sz = 2.0 * tz / (1.0 + t_squared)

    proposed_x: list[float] = []
    proposed_y: list[float] = []
    proposed_z: list[float] = []
    for index in range(particles.count):
        vmx = particles.vx_m_per_s[index] + qmdt2 * ex
        vmy = particles.vy_m_per_s[index] + qmdt2 * ey
        vmz = particles.vz_m_per_s[index] + qmdt2 * ez
        vpx = vmx + (vmy * tz - vmz * ty)
        vpy = vmy + (vmz * tx - vmx * tz)
        vpz = vmz + (vmx * ty - vmy * tx)
        vx = vmx + (vpy * sz - vpz * sy) + qmdt2 * ex
        vy = vmy + (vpz * sx - vpx * sz) + qmdt2 * ey
        vz = vmz + (vpx * sy - vpy * sx) + qmdt2 * ez
        if not isfinite(vx) or not isfinite(vy) or not isfinite(vz):
            raise PICValidationError("Boris push produced a nonfinite state")
        proposed_x.append(vx)
        proposed_y.append(vy)
        proposed_z.append(vz)
    particles.vx_m_per_s[:] = proposed_x
    particles.vy_m_per_s[:] = proposed_y
    particles.vz_m_per_s[:] = proposed_z
