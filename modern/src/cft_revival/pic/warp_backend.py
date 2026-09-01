"""Optional genuine Warp float64 CIC and electrostatic-push kernels."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
import sys
from typing import Sequence

from .electrostatic import (
    charge_density_per_particle_c_per_m3,
    integrated_charge_c,
    represented_charge_c,
)
from .models import Grid1D, PICDeviceError, PICValidationError, ParticleState, Species

try:
    import warp as wp
except ImportError:  # pragma: no cover - exercised only without the optional extra.
    wp = None  # type: ignore[assignment]


if wp is not None:

    @wp.kernel
    def _deposit_cic_kernel(
        position: wp.array(dtype=wp.float64),
        density: wp.array(dtype=wp.float64),
        x_min: wp.float64,
        inverse_dx: wp.float64,
        charge_over_volume: wp.float64,
        cells: int,
    ):
        particle = wp.tid()
        coordinate = (position[particle] - x_min) * inverse_dx
        left = int(wp.floor(coordinate))
        fraction = coordinate - wp.float64(left)
        right = left + 1
        if right == cells:
            right = 0
        wp.atomic_add(
            density, left, charge_over_volume * (wp.float64(1.0) - fraction)
        )
        wp.atomic_add(density, right, charge_over_volume * fraction)

    @wp.kernel
    def _gather_push_kernel(
        position: wp.array(dtype=wp.float64),
        velocity_x: wp.array(dtype=wp.float64),
        field: wp.array(dtype=wp.float64),
        x_min: wp.float64,
        length: wp.float64,
        inverse_dx: wp.float64,
        acceleration_dt: wp.float64,
        dt: wp.float64,
        cells: int,
    ):
        particle = wp.tid()
        coordinate = (position[particle] - x_min) * inverse_dx
        left = int(wp.floor(coordinate))
        fraction = coordinate - wp.float64(left)
        previous = left - 1
        if previous < 0:
            previous = cells - 1
        right = left + 1
        if right == cells:
            right = 0
        electric_left = wp.float64(0.5) * (field[previous] + field[left])
        electric_right = wp.float64(0.5) * (field[left] + field[right])
        electric = (
            (wp.float64(1.0) - fraction) * electric_left
            + fraction * electric_right
        )
        velocity = velocity_x[particle] + acceleration_dt * electric
        advanced = position[particle] + velocity * dt
        advanced = x_min + (advanced - x_min) - wp.floor(
            (advanced - x_min) / length
        ) * length
        velocity_x[particle] = velocity
        position[particle] = advanced


@dataclass(frozen=True, slots=True)
class WarpKernelResult:
    deposited_charge_density_c_per_m3: tuple[float, ...]
    particles: ParticleState
    device: str


def _resolve_device(device: str):
    if wp is None:
        raise PICDeviceError("NVIDIA Warp is unavailable")
    wp.init()
    requested = device.strip().lower() if isinstance(device, str) else ""
    if requested == "cuda":
        requested = "cuda:0"
    if requested != "cpu" and not requested.startswith("cuda:"):
        raise PICDeviceError("device must be 'cpu', 'cuda', or 'cuda:N'")
    try:
        return wp.get_device(requested)
    except (RuntimeError, ValueError) as error:
        raise PICDeviceError(f"Warp device {requested!r} is unavailable") from error


def device_available(device: str) -> bool:
    try:
        _resolve_device(device)
    except PICDeviceError:
        return False
    return True


def deposit_and_push_warp(
    grid: Grid1D,
    species: Species,
    particles: ParticleState,
    electric_field_v_per_m: Sequence[float],
    dt_s: float,
    *,
    device: str,
) -> WarpKernelResult:
    """Run CIC deposition and a gathered leapfrog kick/drift on one Warp device."""

    particles.validate()
    if len(electric_field_v_per_m) != grid.cells:
        raise PICValidationError("node field length must equal grid.cells")
    field_values = tuple(float(value) for value in electric_field_v_per_m)
    dt = float(dt_s)
    if any(not isfinite(value) for value in field_values) or not isfinite(dt) or dt <= 0.0:
        raise PICValidationError("field must be finite and dt_s positive")
    if any(not grid.x_min_m <= x < grid.x_max_m for x in particles.x_m):
        raise PICValidationError("Warp particles must start inside the periodic grid")
    charge_over_volume = charge_density_per_particle_c_per_m3(grid, species)
    acceleration_dt = species.charge_c * dt / species.mass_kg
    if not isfinite(charge_over_volume) or not isfinite(acceleration_dt):
        raise PICValidationError("Warp PIC scales are not representable")
    resolved = _resolve_device(device)
    if wp is None:  # Static narrowing after device resolution.
        raise PICDeviceError("NVIDIA Warp is unavailable")

    position = wp.array(particles.x_m, dtype=wp.float64, device=resolved)
    velocity = wp.array(particles.vx_m_per_s, dtype=wp.float64, device=resolved)
    density = wp.zeros(grid.cells, dtype=wp.float64, device=resolved)
    field = wp.array(field_values, dtype=wp.float64, device=resolved)
    wp.launch(
        _deposit_cic_kernel,
        dim=particles.count,
        inputs=[
            position,
            density,
            grid.x_min_m,
            1.0 / grid.dx_m,
            charge_over_volume,
            grid.cells,
        ],
        device=resolved,
    )
    wp.launch(
        _gather_push_kernel,
        dim=particles.count,
        inputs=[
            position,
            velocity,
            field,
            grid.x_min_m,
            grid.length_m,
            1.0 / grid.dx_m,
            acceleration_dt,
            dt,
            grid.cells,
        ],
        device=resolved,
    )
    wp.synchronize_device(resolved)
    state = ParticleState(
        [float(value) for value in position.numpy()],
        [float(value) for value in velocity.numpy()],
        particles.vy_m_per_s.copy(),
        particles.vz_m_per_s.copy(),
    )
    deposited = tuple(float(value) for value in density.numpy())
    if any(not isfinite(value) for value in deposited):
        raise PICValidationError("Warp deposition produced a nonfinite density")
    represented = represented_charge_c(species, particles.count)
    integrated = integrated_charge_c(grid, deposited)
    if not isclose(
        represented,
        integrated,
        rel_tol=64.0 * sys.float_info.epsilon,
        abs_tol=0.0,
    ):
        raise PICValidationError("Warp deposition did not conserve represented charge")
    return WarpKernelResult(
        deposited,
        state,
        str(resolved),
    )
