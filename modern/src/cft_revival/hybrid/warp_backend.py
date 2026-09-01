"""Optional NVIDIA Warp float64 kernels for the hybrid first slice."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Sequence

from .models import (
    CartesianGrid1D,
    DepositedMoments,
    HybridDeviceError,
    HybridOptionalDependencyError,
    HybridValidationError,
    Particle,
    UniformFields,
    VelocityTimeLevel,
    finite_scalar,
    validated_particle_batch,
)

try:
    import warp as wp
except ImportError:  # pragma: no cover - exercised by monkeypatch when installed.
    wp = None  # type: ignore[assignment]


if wp is not None:

    @wp.kernel
    def _boris_kernel(
        position: wp.array2d(dtype=wp.float64),
        velocity: wp.array2d(dtype=wp.float64),
        charge_to_mass: wp.array(dtype=wp.float64),
        alive: wp.array(dtype=wp.int32),
        electric: wp.array(dtype=wp.float64),
        magnetic: wp.array(dtype=wp.float64),
        dt: wp.float64,
        output_position: wp.array2d(dtype=wp.float64),
        output_velocity: wp.array2d(dtype=wp.float64),
    ):
        i = wp.tid()
        if alive[i] == 0:
            for axis in range(3):
                output_position[i, axis] = position[i, axis]
                output_velocity[i, axis] = velocity[i, axis]
        else:
            half = wp.float64(0.5) * charge_to_mass[i] * dt
            vmx = velocity[i, 0] + half * electric[0]
            vmy = velocity[i, 1] + half * electric[1]
            vmz = velocity[i, 2] + half * electric[2]
            tx = half * magnetic[0]
            ty = half * magnetic[1]
            tz = half * magnetic[2]
            denominator = wp.float64(1.0) + tx * tx + ty * ty + tz * tz
            sx = wp.float64(2.0) * tx / denominator
            sy = wp.float64(2.0) * ty / denominator
            sz = wp.float64(2.0) * tz / denominator
            vpx = vmx + vmy * tz - vmz * ty
            vpy = vmy + vmz * tx - vmx * tz
            vpz = vmz + vmx * ty - vmy * tx
            vnx = vmx + vpy * sz - vpz * sy + half * electric[0]
            vny = vmy + vpz * sx - vpx * sz + half * electric[1]
            vnz = vmz + vpx * sy - vpy * sx + half * electric[2]
            output_velocity[i, 0] = vnx
            output_velocity[i, 1] = vny
            output_velocity[i, 2] = vnz
            output_position[i, 0] = position[i, 0] + dt * vnx
            output_position[i, 1] = position[i, 1] + dt * vny
            output_position[i, 2] = position[i, 2] + dt * vnz

    @wp.kernel
    def _deposit_kernel(
        position_x: wp.array(dtype=wp.float64),
        velocity: wp.array2d(dtype=wp.float64),
        weight: wp.array(dtype=wp.float64),
        represented_mass: wp.array(dtype=wp.float64),
        represented_charge: wp.array(dtype=wp.float64),
        alive: wp.array(dtype=wp.int32),
        x_min: wp.float64,
        length: wp.float64,
        spacing: wp.float64,
        volume: wp.float64,
        cell_count: wp.int32,
        number: wp.array(dtype=wp.float64),
        charge: wp.array(dtype=wp.float64),
        current: wp.array2d(dtype=wp.float64),
        momentum: wp.array2d(dtype=wp.float64),
        energy: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        if alive[i] != 0:
            wrapped = (position_x[i] - x_min) % length
            normalized = wrapped / spacing - wp.float64(0.5)
            left_raw = wp.int32(wp.floor(normalized))
            left = ((left_raw % cell_count) + cell_count) % cell_count
            right = (left + 1) % cell_count
            fraction_right = normalized - wp.float64(left_raw)
            shape_left = wp.float64(1.0) - fraction_right
            speed_squared = (
                velocity[i, 0] * velocity[i, 0]
                + velocity[i, 1] * velocity[i, 1]
                + velocity[i, 2] * velocity[i, 2]
            )
            for slot in range(2):
                node = left
                shape = shape_left
                if slot == 1:
                    node = right
                    shape = fraction_right
                density_factor = shape / volume
                wp.atomic_add(number, node, weight[i] * density_factor)
                wp.atomic_add(charge, node, represented_charge[i] * density_factor)
                wp.atomic_add(
                    energy,
                    node,
                    wp.float64(0.5)
                    * represented_mass[i]
                    * speed_squared
                    * density_factor,
                )
                for axis in range(3):
                    wp.atomic_add(
                        current,
                        node,
                        axis,
                        represented_charge[i] * velocity[i, axis] * density_factor,
                    )
                    wp.atomic_add(
                        momentum,
                        node,
                        axis,
                        represented_mass[i] * velocity[i, axis] * density_factor,
                    )


def warp_available() -> bool:
    return wp is not None


def _resolve_device(device: str):
    if wp is None:
        raise HybridOptionalDependencyError(
            "NVIDIA Warp is unavailable; install the optional gpu dependency"
        )
    if not isinstance(device, str):
        raise HybridDeviceError("device must be 'cpu', 'cuda', or 'cuda:N'")
    wp.init()
    requested = device.strip().lower()
    if requested == "cuda":
        requested = "cuda:0"
    if requested != "cpu" and not requested.startswith("cuda:"):
        raise HybridDeviceError("device must be 'cpu', 'cuda', or 'cuda:N'")
    try:
        return wp.get_device(requested)
    except (RuntimeError, ValueError) as error:
        raise HybridDeviceError(f"Warp device {requested!r} is unavailable") from error


def device_available(device: str) -> bool:
    try:
        _resolve_device(device)
    except (HybridOptionalDependencyError, HybridDeviceError):
        return False
    return True


def _validate_particles(particles: Sequence[Particle]) -> tuple[Particle, ...]:
    if not isinstance(particles, Sequence) or isinstance(particles, (str, bytes)):
        raise HybridValidationError("particles must be a one-dimensional sequence")
    return validated_particle_batch(particles)


def push_boris_warp(
    particles: Sequence[Particle],
    fields: UniformFields,
    dt_s: float,
    *,
    device: str,
) -> tuple[Particle, ...]:
    """Run one prescribed uniform-field Boris step on a selected Warp device."""

    batch = _validate_particles(particles)
    if not isinstance(fields, UniformFields):
        raise HybridValidationError("fields must be UniformFields")
    dt = finite_scalar("dt_s", dt_s)
    if dt < 0.0:
        raise HybridValidationError("dt_s must be non-negative")
    resolved = _resolve_device(device)
    if wp is None:  # Static type narrowing after _resolve_device.
        raise HybridOptionalDependencyError("NVIDIA Warp is unavailable")
    if any(
        particle.velocity_time_level
        is not VelocityTimeLevel.LEAPFROG_N_MINUS_HALF
        for particle in batch
    ):
        raise HybridValidationError(
            "Warp Boris push requires leapfrog_n_minus_one_half velocity"
        )
    if dt == 0.0 or not batch:
        return batch

    def array(values, dtype=wp.float64):
        return wp.array(values, dtype=dtype, device=resolved)

    position = wp.array(
        [list(particle.position_m) for particle in batch],
        dtype=wp.float64,
        ndim=2,
        device=resolved,
    )
    velocity = wp.array(
        [list(particle.velocity_m_per_s) for particle in batch],
        dtype=wp.float64,
        ndim=2,
        device=resolved,
    )
    charge_to_mass = array(
        [particle.species.charge_c / particle.species.mass_kg for particle in batch]
    )
    alive = array([int(particle.alive) for particle in batch], wp.int32)
    electric = array(list(fields.electric_v_per_m))
    magnetic = array(list(fields.magnetic_t))
    output_position = wp.empty((len(batch), 3), dtype=wp.float64, device=resolved)
    output_velocity = wp.empty((len(batch), 3), dtype=wp.float64, device=resolved)
    wp.launch(
        _boris_kernel,
        dim=len(batch),
        inputs=[
            position,
            velocity,
            charge_to_mass,
            alive,
            electric,
            magnetic,
            dt,
            output_position,
            output_velocity,
        ],
        device=resolved,
    )
    wp.synchronize_device(resolved)
    positions = output_position.numpy()
    velocities = output_velocity.numpy()
    if any(
        not isfinite(float(value))
        for matrix in (positions, velocities)
        for row in matrix
        for value in row
    ):
        raise HybridValidationError("Warp pusher produced a nonfinite particle state")
    return tuple(
        replace(
            particle,
            position_m=tuple(float(value) for value in positions[index]),
            velocity_m_per_s=tuple(float(value) for value in velocities[index]),
        )
        for index, particle in enumerate(batch)
    )


def deposit_cic_periodic_warp(
    particles: Sequence[Particle],
    grid: CartesianGrid1D,
    *,
    device: str,
) -> DepositedMoments:
    """Run conservative cell-centred CIC deposition on a selected Warp device."""

    batch = _validate_particles(particles)
    if not isinstance(grid, CartesianGrid1D):
        raise HybridValidationError("grid must be CartesianGrid1D")
    if any(
        particle.position_m[0] < grid.x_min_m
        or particle.position_m[0] > grid.x_max_m
        for particle in batch
        if particle.alive
    ):
        raise HybridValidationError("particle x lies outside the deposition domain")
    if any(
        particle.velocity_time_level
        is not VelocityTimeLevel.LEAPFROG_N_MINUS_HALF
        for particle in batch
    ):
        raise HybridValidationError(
            "deposition requires leapfrog_n_minus_one_half velocity"
        )
    resolved = _resolve_device(device)
    if wp is None:
        raise HybridOptionalDependencyError("NVIDIA Warp is unavailable")
    if not batch:
        zeros = tuple(0.0 for _ in range(grid.cell_count))
        zero_vectors = tuple(
            (0.0, 0.0, 0.0) for _ in range(grid.cell_count)
        )
        return DepositedMoments(
            number_per_m3=zeros,
            charge_c_per_m3=zeros,
            current_a_per_m2=zero_vectors,
            momentum_kg_per_m2_s=zero_vectors,
            kinetic_energy_j_per_m3=zeros,
        )

    def array(values, dtype=wp.float64):
        return wp.array(values, dtype=dtype, device=resolved)

    position_x = array([particle.position_m[0] for particle in batch])
    velocity = wp.array(
        [list(particle.velocity_m_per_s) for particle in batch],
        dtype=wp.float64,
        ndim=2,
        device=resolved,
    )
    weight = array([particle.weight for particle in batch])
    represented_mass = array([particle.represented_mass_kg for particle in batch])
    represented_charge = array([particle.represented_charge_c for particle in batch])
    alive = array([int(particle.alive) for particle in batch], wp.int32)
    number = wp.zeros(grid.cell_count, dtype=wp.float64, device=resolved)
    charge = wp.zeros(grid.cell_count, dtype=wp.float64, device=resolved)
    current = wp.zeros((grid.cell_count, 3), dtype=wp.float64, device=resolved)
    momentum = wp.zeros((grid.cell_count, 3), dtype=wp.float64, device=resolved)
    energy = wp.zeros(grid.cell_count, dtype=wp.float64, device=resolved)
    wp.launch(
        _deposit_kernel,
        dim=len(batch),
        inputs=[
            position_x,
            velocity,
            weight,
            represented_mass,
            represented_charge,
            alive,
            grid.x_min_m,
            grid.x_max_m - grid.x_min_m,
            grid.spacing_m,
            grid.cell_volume_m3,
            grid.cell_count,
            number,
            charge,
            current,
            momentum,
            energy,
        ],
        device=resolved,
    )
    wp.synchronize_device(resolved)
    host_number = number.numpy()
    host_charge = charge.numpy()
    host_current = current.numpy()
    host_momentum = momentum.numpy()
    host_energy = energy.numpy()
    values = (
        list(host_number)
        + list(host_charge)
        + [value for row in host_current for value in row]
        + [value for row in host_momentum for value in row]
        + list(host_energy)
    )
    if any(not isfinite(float(value)) for value in values):
        raise HybridValidationError("Warp deposition produced a nonfinite moment")
    return DepositedMoments(
        number_per_m3=tuple(float(value) for value in host_number),
        charge_c_per_m3=tuple(float(value) for value in host_charge),
        current_a_per_m2=tuple(
            tuple(float(value) for value in row) for row in host_current
        ),
        momentum_kg_per_m2_s=tuple(
            tuple(float(value) for value in row) for row in host_momentum
        ),
        kinetic_energy_j_per_m3=tuple(float(value) for value in host_energy),
    )
