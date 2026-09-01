"""Dynamic residual, batch, Jacobian, conservation, and inequality interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence

from cft_revival.plasma import AnodeIonEnergySign

from .ledger import generate_equation_ledger
from .models import (
    DynamicBounds,
    NetworkClosures,
    NetworkInputs,
    NetworkNumericsError,
    NetworkPowerBalance,
    NetworkResidualEvaluation,
    NetworkState,
    NetworkValidationError,
)


@dataclass(frozen=True, slots=True)
class _Dual:
    value: float
    derivative: tuple[float, ...]

    def __add__(self, other: float | _Dual) -> _Dual:
        right = _as_dual(other, len(self.derivative))
        return _Dual(
            self.value + right.value,
            tuple(a + b for a, b in zip(self.derivative, right.derivative, strict=True)),
        )

    __radd__ = __add__

    def __neg__(self) -> _Dual:
        return _Dual(-self.value, tuple(-value for value in self.derivative))

    def __sub__(self, other: float | _Dual) -> _Dual:
        return self + (-_as_dual(other, len(self.derivative)))

    def __rsub__(self, other: float | _Dual) -> _Dual:
        return _as_dual(other, len(self.derivative)) - self

    def __mul__(self, other: float | _Dual) -> _Dual:
        right = _as_dual(other, len(self.derivative))
        return _Dual(
            self.value * right.value,
            tuple(
                self.value * b + right.value * a
                for a, b in zip(self.derivative, right.derivative, strict=True)
            ),
        )

    __rmul__ = __mul__

    def __truediv__(self, other: float | _Dual) -> _Dual:
        right = _as_dual(other, len(self.derivative))
        denominator = right.value * right.value
        return _Dual(
            self.value / right.value,
            tuple(
                (a * right.value - self.value * b) / denominator
                for a, b in zip(self.derivative, right.derivative, strict=True)
            ),
        )

    def power_three_halves(self) -> _Dual:
        root = sqrt(self.value)
        return _Dual(
            self.value * root,
            tuple(1.5 * root * value for value in self.derivative),
        )


def _as_dual(value: float | _Dual, size: int) -> _Dual:
    return value if isinstance(value, _Dual) else _Dual(float(value), (0.0,) * size)


def _layout(
    vector: Sequence[float | _Dual], n: int
) -> tuple[
    Sequence[float | _Dual],
    Sequence[float | _Dual],
    Sequence[float | _Dual],
    Sequence[float | _Dual],
    Sequence[float | _Dual],
    Sequence[float | _Dual],
]:
    return (
        vector[0:n],
        vector[n : 2 * n],
        vector[2 * n : 3 * n],
        vector[3 * n : 4 * n + 1],
        vector[4 * n + 1 : 5 * n + 2],
        vector[5 * n + 2 :],
    )


def default_bounds(inputs: NetworkInputs) -> DynamicBounds:
    n = inputs.dimensions.cell_count
    voltage = inputs.anode_voltage_v
    current = inputs.anode_current_a
    positive_temperature = voltage * 1.0e-14
    lower = (
        *(inputs.cathode_potential_v for _ in range(n)),
        *(positive_temperature for _ in range(n)),
        *(0.0 for _ in range(n)),
        *(0.0 for _ in range(n + 1)),
        *(0.0 for _ in range(n)),
        -2.0 * current,
        *(0.0 for _ in range(n - 1)),
    )
    upper = (
        *(1.5 * voltage for _ in range(n)),
        *(2.0 * voltage for _ in range(n)),
        *(2.0 * current for _ in range(n)),
        *(2.0 * current for _ in range(n + 1)),
        *(2.0 * current for _ in range(n)),
        0.0,
        *(2.0 * current for _ in range(n - 1)),
    )
    if len(lower) != inputs.dimensions.state_size:
        raise AssertionError("dynamic bound layout invariant failed")
    return DynamicBounds(lower, upper)


def _gains(
    vector: Sequence[float | _Dual], inputs: NetworkInputs
) -> tuple[float | _Dual, ...]:
    n = inputs.dimensions.cell_count
    phi, temperature, *_ = _layout(vector, n)
    gains: list[float | _Dual] = [
        phi[0] - inputs.cathode_potential_v + inputs.cathode_electron_temperature_ev
    ]
    gains.extend(
        phi[cell] - phi[cell - 1] + temperature[cell - 1]
        for cell in range(1, n)
    )
    return tuple(gains)


def _value(item: float | _Dual) -> float:
    return item.value if isinstance(item, _Dual) else float(item)


def _raw_residual(
    vector: Sequence[float | _Dual], inputs: NetworkInputs
) -> tuple[float | _Dual, ...]:
    n = inputs.dimensions.cell_count
    phi, temperature, source, electron, ion, cusp_ion = _layout(vector, n)
    probability = inputs.arrival_probabilities
    gain = _gains(vector, inputs)
    first_drop = phi[0] - inputs.cathode_potential_v
    if _value(first_drop) < 0.0:
        raise NetworkValidationError("first plasma potential is below the cathode")
    if any(_value(item) < 0.0 for item in gain):
        raise NetworkValidationError("every cell electron-energy gain must be non-negative")
    emitted = (
        inputs.cathode_perveance_a_per_v_3_2 * first_drop.power_three_halves()
        if isinstance(first_drop, _Dual)
        else inputs.cathode_perveance_a_per_v_3_2 * first_drop**1.5
    )

    residual: list[float | _Dual] = [electron[0] - emitted]
    residual.extend(
        electron[cell + 1]
        - electron[cell] * (1.0 - probability[cell])
        - source[cell]
        for cell in range(n - 1)
    )
    residual.extend(
        source[cell]
        - electron[cell]
        * (1.0 - probability[cell])
        * inputs.ionization_fraction
        * gain[cell]
        / inputs.xenon_ionization_energy_ev
        for cell in range(n)
    )
    residual.extend(
        ion[cell] - ion[cell + 1] - source[cell] + cusp_ion[cell]
        for cell in range(n - 1)
    )
    residual.append(ion[n - 1] - source[n - 1] - ion[n])
    residual.extend(
        temperature[cell]
        * (electron[cell] * (1.0 - probability[cell]) + source[cell])
        - inputs.thermalization_fraction
        * electron[cell]
        * (1.0 - probability[cell])
        * gain[cell]
        for cell in range(1, n)
    )
    residual.extend(
        electron[interface] + ion[interface] - inputs.anode_current_a
        for interface in range(n + 1)
    )
    residual.extend(
        cusp_ion[cusp] - electron[cusp] * probability[cusp]
        for cusp in range(n - 1)
    )

    transmitted: list[float | _Dual] = []
    for cell in range(n):
        incoming = electron[cell] * (1.0 - probability[cell]) * gain[cell]
        transmitted.append(incoming)
        outgoing = electron[cell] * (1.0 - probability[cell]) + source[cell]
        residual.append(
            (1.0 - inputs.excitation_fraction) * incoming
            - outgoing * temperature[cell]
            - source[cell] * inputs.xenon_ionization_energy_ev
        )

    beam: float | _Dual = ion[n - 1] * phi[n - 1]
    for cell in range(n - 2, -1, -1):
        beam = beam + (ion[cell] - ion[cell + 1]) * phi[cell]
    ionization_loss = inputs.xenon_ionization_energy_ev * sum(source)
    excitation_loss = inputs.excitation_fraction * sum(transmitted)
    cusp_loss: float | _Dual = electron[0] * probability[0] * (
        phi[0] - inputs.cathode_potential_v + inputs.xenon_ionization_energy_ev
    )
    # For N=1 p[0] is terminal, so no dielectric cusp term exists.
    if n == 1:
        cusp_loss = 0.0
    else:
        for cusp in range(1, n - 1):
            cusp_loss = cusp_loss + electron[cusp] * probability[cusp] * (
                phi[cusp]
                - phi[cusp - 1]
                + inputs.xenon_ionization_energy_ev
                + temperature[cusp - 1]
            )

    previous_phi: float | _Dual = (
        inputs.cathode_potential_v if n == 1 else phi[n - 2]
    )
    previous_temperature: float | _Dual = (
        inputs.cathode_electron_temperature_ev if n == 1 else temperature[n - 2]
    )
    anode_delta = phi[n - 1] - inputs.anode_voltage_v
    anode_electron = electron[n - 1] * probability[n - 1] * (
        inputs.anode_voltage_v - previous_phi + previous_temperature
    ) + (source[n - 1] + electron[n - 1] * (1.0 - probability[n - 1])) * (
        anode_delta + temperature[n - 1]
    )
    anode_ion = (
        -ion[n] * anode_delta
        if inputs.anode_ion_energy_sign is AnodeIonEnergySign.SOURCE_MINUS_SIGN
        else ion[n] * anode_delta
    )
    residual.append(
        beam
        + ionization_loss
        + excitation_loss
        + cusp_loss
        + anode_electron
        + anode_ion
        - inputs.anode_voltage_v * inputs.anode_current_a
    )
    if len(residual) != inputs.dimensions.residual_size:
        raise AssertionError("dynamic residual dimension invariant failed")
    return tuple(residual)


def residual_scales(inputs: NetworkInputs) -> tuple[float, ...]:
    ledger = generate_equation_ledger(inputs.topology)  # type: ignore[arg-type]
    current = inputs.anode_current_a
    power = inputs.anode_voltage_v * current
    scales = tuple(current if row.unit == "A" else power for row in ledger)
    if any(not isfinite(value) or value <= 0.0 for value in scales):
        raise NetworkValidationError("residual scales must be finite and positive")
    return scales


def _finite_tuple(name: str, values: Sequence[float]) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if any(not isfinite(value) for value in converted):
        raise NetworkNumericsError(f"{name} produced a non-finite value")
    return converted


def power_balance(state: NetworkState, inputs: NetworkInputs) -> NetworkPowerBalance:
    if state.dimensions != inputs.dimensions:
        raise NetworkValidationError("state and topology dimensions differ")
    evaluation_raw = _raw_residual(state.to_vector(), inputs)
    n = inputs.dimensions.cell_count
    phi = state.plasma_potential_v
    temperature = state.electron_temperature_ev
    source = state.ionization_source_current_a
    electron = state.electron_current_a
    ion = state.ion_current_a
    probability = inputs.arrival_probabilities
    gain = tuple(float(value) for value in _gains(state.to_vector(), inputs))
    beam = ion[n - 1] * phi[n - 1]
    for cell in range(n - 2, -1, -1):
        beam += (ion[cell] - ion[cell + 1]) * phi[cell]
    ionization = inputs.xenon_ionization_energy_ev * sum(source)
    excitation = inputs.excitation_fraction * sum(
        electron[cell] * (1.0 - probability[cell]) * gain[cell]
        for cell in range(n)
    )
    cusp = 0.0
    if n > 1:
        cusp = electron[0] * probability[0] * (
            phi[0] - inputs.cathode_potential_v + inputs.xenon_ionization_energy_ev
        )
        cusp += sum(
            electron[cell]
            * probability[cell]
            * (
                phi[cell]
                - phi[cell - 1]
                + inputs.xenon_ionization_energy_ev
                + temperature[cell - 1]
            )
            for cell in range(1, n - 1)
        )
    previous_phi = inputs.cathode_potential_v if n == 1 else phi[n - 2]
    previous_temperature = (
        inputs.cathode_electron_temperature_ev if n == 1 else temperature[n - 2]
    )
    delta = phi[n - 1] - inputs.anode_voltage_v
    anode_electron = electron[n - 1] * probability[n - 1] * (
        inputs.anode_voltage_v - previous_phi + previous_temperature
    ) + (source[n - 1] + electron[n - 1] * (1.0 - probability[n - 1])) * (
        delta + temperature[n - 1]
    )
    anode_ion = (
        -ion[n] * delta
        if inputs.anode_ion_energy_sign is AnodeIonEnergySign.SOURCE_MINUS_SIGN
        else ion[n] * delta
    )
    values = _finite_tuple(
        "power balance",
        (
            beam,
            ionization,
            excitation,
            cusp,
            anode_electron,
            anode_ion,
            anode_electron + anode_ion,
            inputs.anode_voltage_v * inputs.anode_current_a,
            float(evaluation_raw[-1]),
        ),
    )
    return NetworkPowerBalance(*values)


def evaluate_residual(
    state: NetworkState, inputs: NetworkInputs
) -> NetworkResidualEvaluation:
    if state.dimensions != inputs.dimensions:
        raise NetworkValidationError("state and topology dimensions differ")
    raw = _finite_tuple("residual", _raw_residual(state.to_vector(), inputs))  # type: ignore[arg-type]
    scales = residual_scales(inputs)
    normalized = _finite_tuple(
        "normalized residual",
        tuple(value / scale for value, scale in zip(raw, scales, strict=True)),
    )
    ledger = generate_equation_ledger(inputs.topology)  # type: ignore[arg-type]
    ids = tuple(row.row_id for row in ledger)
    n = inputs.dimensions.cell_count
    cursor = 0
    cursor += 1
    electron_continuity = raw[cursor : cursor + n - 1]
    cursor += n - 1
    cursor += n
    ion_continuity = raw[cursor : cursor + n]
    cursor += n
    cursor += n - 1
    interface = raw[cursor : cursor + n + 1]
    cursor += n + 1
    cusp = raw[cursor : cursor + n - 1]
    cursor += n - 1
    cell_energy = raw[cursor : cursor + n]
    return NetworkResidualEvaluation(
        raw=raw,
        normalized=normalized,
        equation_ids=ids,
        scales=scales,
        powers=power_balance(state, inputs),
        closures=NetworkClosures(
            electron_continuity_a=electron_continuity,
            ion_continuity_a=ion_continuity,
            interface_current_a=interface,
            cusp_current_a=cusp,
            cell_energy_w=cell_energy,
            global_energy_w=raw[-1],
        ),
    )


def evaluate_residual_batch(
    states: Sequence[NetworkState], inputs: Sequence[NetworkInputs]
) -> tuple[NetworkResidualEvaluation, ...]:
    """Ragged topology batches are rejected; homogeneous layouts are GPU-suitable."""

    if len(states) == 0 or len(states) != len(inputs):
        raise NetworkValidationError("states and inputs must be non-empty and equal length")
    dimensions = inputs[0].dimensions
    if any(item.dimensions != dimensions for item in inputs):
        raise NetworkValidationError("a batch must have one homogeneous topology dimension")
    if any(state.dimensions != dimensions for state in states):
        raise NetworkValidationError("every batch state must match the batch topology")
    return tuple(
        evaluate_residual(state, point)
        for state, point in zip(states, inputs, strict=True)
    )


def analytic_jacobian(
    state: NetworkState, inputs: NetworkInputs
) -> tuple[tuple[float, ...], ...]:
    if state.dimensions != inputs.dimensions:
        raise NetworkValidationError("state and topology dimensions differ")
    vector = state.to_vector()
    size = len(vector)
    dual_vector = tuple(
        _Dual(value, tuple(1.0 if row == column else 0.0 for row in range(size)))
        for column, value in enumerate(vector)
    )
    residual = _raw_residual(dual_vector, inputs)
    scales = residual_scales(inputs)
    matrix = tuple(
        tuple(value / scales[row] for value in item.derivative)
        for row, item in enumerate(residual)
        if isinstance(item, _Dual)
    )
    if (
        len(matrix) != inputs.dimensions.residual_size
        or any(len(row) != size for row in matrix)
        or any(not isfinite(value) for row in matrix for value in row)
    ):
        raise NetworkNumericsError("analytic Jacobian has invalid shape or values")
    return matrix


def analytic_jacobian_batch(
    states: Sequence[NetworkState], inputs: Sequence[NetworkInputs]
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Homogeneous fixed-layout Jacobian batch for a later accelerator backend."""

    if len(states) == 0 or len(states) != len(inputs):
        raise NetworkValidationError("states and inputs must be non-empty and equal length")
    dimensions = inputs[0].dimensions
    if any(item.dimensions != dimensions for item in inputs):
        raise NetworkValidationError("a Jacobian batch must have one homogeneous topology dimension")
    if any(state.dimensions != dimensions for state in states):
        raise NetworkValidationError("every Jacobian state must match the batch topology")
    return tuple(
        analytic_jacobian(state, point)
        for state, point in zip(states, inputs, strict=True)
    )


def finite_difference_jacobian(
    state: NetworkState,
    inputs: NetworkInputs,
    *,
    relative_step: float = 1.0e-6,
) -> tuple[tuple[float, ...], ...]:
    if not isfinite(relative_step) or relative_step <= 0.0:
        raise NetworkValidationError("relative_step must be finite and positive")
    vector = list(state.to_vector())
    bounds = default_bounds(inputs)
    base = evaluate_residual(state, inputs).normalized
    columns: list[tuple[float, ...]] = []
    for column, value in enumerate(vector):
        nominal = relative_step * max(1.0, abs(value))
        low_room = value - bounds.lower[column]
        high_room = bounds.upper[column] - value
        if low_room >= nominal and high_room >= nominal:
            left = vector.copy()
            right = vector.copy()
            left[column] -= nominal
            right[column] += nominal
            left_residual = evaluate_residual(
                NetworkState.from_vector(left, inputs.dimensions.cell_count), inputs
            ).normalized
            right_residual = evaluate_residual(
                NetworkState.from_vector(right, inputs.dimensions.cell_count), inputs
            ).normalized
            columns.append(
                tuple(
                    (high - low) / (2.0 * nominal)
                    for low, high in zip(left_residual, right_residual, strict=True)
                )
            )
        elif high_room > 0.0:
            step = min(nominal, high_room)
            right = vector.copy()
            right[column] += step
            right_residual = evaluate_residual(
                NetworkState.from_vector(right, inputs.dimensions.cell_count), inputs
            ).normalized
            columns.append(
                tuple((high - low) / step for low, high in zip(base, right_residual, strict=True))
            )
        elif low_room > 0.0:
            step = min(nominal, low_room)
            left = vector.copy()
            left[column] -= step
            left_residual = evaluate_residual(
                NetworkState.from_vector(left, inputs.dimensions.cell_count), inputs
            ).normalized
            columns.append(
                tuple((high - low) / step for low, high in zip(left_residual, base, strict=True))
            )
        else:
            raise NetworkValidationError(f"state variable {column} has no difference room")
    return tuple(
        tuple(columns[column][row] for column in range(len(vector)))
        for row in range(inputs.dimensions.residual_size)
    )


def constraint_margins(state: NetworkState, inputs: NetworkInputs) -> tuple[float, ...]:
    if state.dimensions != inputs.dimensions:
        raise NetworkValidationError("state and topology dimensions differ")
    phi = state.plasma_potential_v
    temperature = state.electron_temperature_ev
    ion = state.ion_current_a
    powers = power_balance(state, inputs)
    return (
        phi[0] - inputs.cathode_potential_v,
        *(phi[index] - phi[index - 1] for index in range(1, len(phi))),
        phi[-1] - inputs.anode_voltage_v,
        *temperature,
        ion[-2] + ion[-1],
        powers.beam_power_w,
        powers.ionization_loss_w,
        powers.excitation_loss_w,
        powers.cusp_loss_w,
        powers.anode_electron_loss_w,
    )


def is_feasible(
    state: NetworkState,
    inputs: NetworkInputs,
    bounds: DynamicBounds | None = None,
) -> bool:
    selected = default_bounds(inputs) if bounds is None else bounds
    vector = state.to_vector()
    if len(selected.lower) != len(vector):
        raise NetworkValidationError("bounds and state dimensions differ")
    return all(
        low <= value <= high
        for value, low, high in zip(vector, selected.lower, selected.upper, strict=True)
    ) and all(margin >= 0.0 for margin in constraint_margins(state, inputs))
