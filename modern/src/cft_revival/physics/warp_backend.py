"""Optional NVIDIA Warp float64 backend for batched L0 performance evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from .models import (
    ConservationDiagnostics,
    IdealPerformanceResult,
    OptionalDependencyError,
    PhysicsDeviceError,
    PhysicsValidationError,
    ReportedPowerBudget,
    XenonOperatingPoint,
)
from .numerics import (
    PreparedOperatingPoint,
    prepare_operating_point,
    validate_point_batch,
)
from .reference import _warnings

try:
    import warp as wp
except ImportError:  # pragma: no cover - tested by monkeypatch where Warp is installed.
    wp = None  # type: ignore[assignment]


if wp is not None:

    @wp.kernel
    def _performance_kernel(
        mass_flow_kg_per_s: wp.array(dtype=wp.float64),
        xenon_mass_kg: wp.array(dtype=wp.float64),
        total_rate_per_s: wp.array(dtype=wp.float64),
        plus_speed_m_per_s: wp.array(dtype=wp.float64),
        neutral_fraction: wp.array(dtype=wp.float64),
        plus_fraction: wp.array(dtype=wp.float64),
        double_plus_fraction: wp.array(dtype=wp.float64),
        axial_momentum_factor: wp.array(dtype=wp.float64),
        beam_current_a: wp.array(dtype=wp.float64),
        anode_current_a: wp.array(dtype=wp.float64),
        beam_power_w: wp.array(dtype=wp.float64),
        particle_beam_power_w: wp.array(dtype=wp.float64),
        anode_power_w: wp.array(dtype=wp.float64),
        thruster_power_w: wp.array(dtype=wp.float64),
        specific_impulse_s: wp.array(dtype=wp.float64),
        ppu_input_power_w: wp.array(dtype=wp.float64),
        canonical_ppu_loss_w: wp.array(dtype=wp.float64),
        output: wp.array2d(dtype=wp.float64),
    ):
        i = wp.tid()
        elementary_charge = wp.float64(1.602176634e-19)
        two = wp.float64(2.0)
        zero = wp.float64(0.0)

        mass_flow = mass_flow_kg_per_s[i]
        total_rate = total_rate_per_s[i]
        neutral_rate = neutral_fraction[i] * total_rate
        plus_rate = plus_fraction[i] * total_rate
        double_rate = double_plus_fraction[i] * total_rate
        plus_speed = plus_speed_m_per_s[i]
        double_speed = wp.sqrt(two) * plus_speed
        momentum_velocity = (
            plus_fraction[i] * plus_speed
            + double_plus_fraction[i] * double_speed
        )
        undiverged_thrust = mass_flow * momentum_velocity
        axial_thrust = axial_momentum_factor[i] * undiverged_thrust

        beam_current = beam_current_a[i]
        anode_current = anode_current_a[i]
        particle_beam_power = particle_beam_power_w[i]
        electrical_beam_power = beam_power_w[i]
        anode_power = anode_power_w[i]
        thruster_power = thruster_power_w[i]
        anode_efficiency = zero
        thruster_efficiency = zero
        ppu_efficiency = zero
        if anode_power > zero:
            anode_efficiency = electrical_beam_power / anode_power
        if thruster_power > zero:
            thruster_efficiency = electrical_beam_power / thruster_power
        if ppu_input_power_w[i] > zero:
            ppu_efficiency = electrical_beam_power / ppu_input_power_w[i]
        else:
            ppu_efficiency = thruster_efficiency

        reconstructed_rate = neutral_rate + plus_rate + double_rate
        reconstructed_current = (
            elementary_charge * plus_rate
            + two * elementary_charge * double_rate
        )

        output[i, 0] = total_rate
        output[i, 1] = neutral_rate
        output[i, 2] = plus_rate
        output[i, 3] = double_rate
        output[i, 4] = plus_speed
        output[i, 5] = double_speed
        output[i, 6] = undiverged_thrust
        output[i, 7] = axial_thrust
        output[i, 8] = specific_impulse_s[i]
        output[i, 9] = beam_current
        output[i, 10] = anode_current
        output[i, 11] = electrical_beam_power
        output[i, 12] = anode_power
        output[i, 13] = thruster_power
        output[i, 14] = canonical_ppu_loss_w[i]
        output[i, 15] = anode_efficiency
        output[i, 16] = thruster_efficiency
        output[i, 17] = ppu_efficiency
        output[i, 18] = reconstructed_rate - total_rate
        output[i, 19] = xenon_mass_kg[i] * reconstructed_rate - mass_flow
        output[i, 20] = reconstructed_current - beam_current
        output[i, 21] = particle_beam_power - electrical_beam_power


@dataclass(frozen=True, slots=True)
class WarpPerformanceBatchResult:
    results: tuple[IdealPerformanceResult, ...]
    device: str


def warp_available() -> bool:
    return wp is not None


def available_devices() -> tuple[str, ...]:
    if wp is None:
        return ()
    wp.init()
    return tuple(str(device) for device in wp.get_devices())


def _resolve_device(device: str):
    if wp is None:
        raise OptionalDependencyError(
            "NVIDIA Warp is unavailable; install the optional gpu dependency"
        )
    wp.init()
    if not isinstance(device, str):
        raise PhysicsDeviceError("device must be 'cpu', 'cuda', or 'cuda:N'")
    requested = device.strip().lower()
    if requested == "cuda":
        requested = "cuda:0"
    if requested != "cpu" and not requested.startswith("cuda:"):
        raise PhysicsDeviceError("device must be 'cpu', 'cuda', or 'cuda:N'")
    try:
        return wp.get_device(requested)
    except (RuntimeError, ValueError) as error:
        raise PhysicsDeviceError(f"Warp device {requested!r} is unavailable") from error


def device_available(device: str) -> bool:
    try:
        _resolve_device(device)
    except (OptionalDependencyError, PhysicsDeviceError):
        return False
    return True


def _validate_points(
    points: Sequence[XenonOperatingPoint],
) -> tuple[
    tuple[XenonOperatingPoint, ...], tuple[PreparedOperatingPoint, ...]
]:
    validated = validate_point_batch(points)
    prepared = tuple(prepare_operating_point(point) for point in validated)
    return validated, prepared


def evaluate_performance_warp(
    points: Sequence[XenonOperatingPoint],
    *,
    device: str,
) -> WarpPerformanceBatchResult:
    """Evaluate a validated batch with one float64 Warp launch.

    Input validation and host/device transfers bracket the hot section. The
    launch itself performs no host synchronization or scalar round trips.
    """

    batch, prepared = _validate_points(points)
    resolved = _resolve_device(device)
    if wp is None:  # Keeps static type checkers aware after _resolve_device.
        raise OptionalDependencyError("NVIDIA Warp is unavailable")

    def array(values: list[float]):
        return wp.array(values, dtype=wp.float64, device=resolved)

    mass_flow = array([point.propellant_mass_flow.kg_per_s for point in batch])
    xenon_mass = array([point.xenon_atom_mass_kg for point in batch])
    total_rate = array([values.total_rate for values in prepared])
    plus_speed = array([values.plus_speed for values in prepared])
    neutral = array([point.charge_state_fractions.xe_neutral for point in batch])
    plus = array([point.charge_state_fractions.xe_plus for point in batch])
    double_plus = array(
        [point.charge_state_fractions.xe_double_plus for point in batch]
    )
    axial_factor = array(
        [
            point.beam_divergence_factors.axial_momentum_fraction_of_ion_momentum
            for point in batch
        ]
    )
    beam_current = array([values.beam_current for values in prepared])
    anode_current = array([values.anode_current for values in prepared])
    beam_power = array([values.beam_power for values in prepared])
    particle_power = array([values.particle_beam_power for values in prepared])
    anode_power = array([values.anode_power for values in prepared])
    thruster_power = array([values.thruster_power for values in prepared])
    specific_impulse = array([values.specific_impulse for values in prepared])
    ppu = array([values.effective_ppu_input for values in prepared])
    ppu_loss = array([values.ppu_loss for values in prepared])
    output = wp.empty((len(batch), 22), dtype=wp.float64, device=resolved)

    wp.launch(
        kernel=_performance_kernel,
        dim=len(batch),
        inputs=[
            mass_flow,
            xenon_mass,
            total_rate,
            plus_speed,
            neutral,
            plus,
            double_plus,
            axial_factor,
            beam_current,
            anode_current,
            beam_power,
            particle_power,
            anode_power,
            thruster_power,
            specific_impulse,
            ppu,
            ppu_loss,
            output,
        ],
        device=resolved,
    )
    wp.synchronize_device(resolved)
    host_output = output.numpy()
    if any(not isfinite(float(value)) for row in host_output for value in row):
        raise PhysicsValidationError(
            "Warp produced a nonfinite derived state; no result was published"
        )

    results: list[IdealPerformanceResult] = []
    for index, point in enumerate(batch):
        row = host_output[index]
        prepared_point = prepared[index]
        budget = ReportedPowerBudget(
            beam_current_a=float(row[9]),
            anode_current_a=float(row[10]),
            beam_kinetic_power_w=float(row[11]),
            anode_input_power_w=float(row[12]),
            cathode_input_power_w=point.power_boundaries.cathode_input_power_w,
            thruster_electrical_input_power_w=float(row[13]),
            requested_ppu_input_power_w=(
                point.power_boundaries.ppu_input_power_w
            ),
            ppu_input_power_w=prepared_point.effective_ppu_input,
            ppu_boundary_adjustment_w=(
                prepared_point.ppu_boundary_adjustment
            ),
            ppu_conversion_loss_w=float(row[14]),
            anode_to_beam_efficiency=(
                None
                if prepared_point.anode_efficiency is None
                else float(row[15])
            ),
            thruster_electrical_to_beam_efficiency=(
                None
                if prepared_point.thruster_efficiency is None
                else float(row[16])
            ),
            ppu_input_to_beam_efficiency=(
                None
                if prepared_point.ppu_efficiency is None
                else float(row[17])
            ),
        )
        diagnostics = ConservationDiagnostics(
            particle_rate_residual_particles_per_s=float(row[18]),
            mass_flow_residual_kg_per_s=float(row[19]),
            beam_current_residual_a=float(row[20]),
            beam_power_residual_w=float(row[21]),
            ppu_power_margin_w=float(row[14]),
        )
        results.append(
            IdealPerformanceResult(
                total_xenon_particle_rate_per_s=float(row[0]),
                neutral_particle_rate_per_s=float(row[1]),
                xe_plus_particle_rate_per_s=float(row[2]),
                xe_double_plus_particle_rate_per_s=float(row[3]),
                xe_plus_speed_m_per_s=float(row[4]),
                xe_double_plus_speed_m_per_s=float(row[5]),
                undiverged_ion_thrust_n=float(row[6]),
                axial_thrust_n=float(row[7]),
                specific_impulse_s=float(row[8]),
                power_budget=budget,
                diagnostics=diagnostics,
                applicability_warnings=_warnings(point),
            )
        )
    return WarpPerformanceBatchResult(tuple(results), str(resolved))
