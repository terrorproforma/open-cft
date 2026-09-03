"""Fixed-layout residual and conservation API for CPU and future batch backends."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from sys import float_info
from typing import Sequence

from .models import (
    AnodeIonEnergySign,
    ConservationClosures,
    PlasmaNumericsError,
    PlasmaState,
    PlasmaValidationError,
    PowerBalance,
    ResidualEvaluation,
    StateBounds,
    XenonGlobalInputs,
)

CURRENT_ROWS = tuple(range(0, 12)) + tuple(range(15, 23))
POWER_ROWS = tuple(range(12, 15)) + tuple(range(23, 28))


@dataclass(frozen=True, slots=True)
class _Dual:
    value: float
    derivative: tuple[float, ...]

    def __add__(self, other: float | _Dual) -> _Dual:
        right = _dual(other, len(self.derivative))
        return _Dual(
            self.value + right.value,
            tuple(a + b for a, b in zip(self.derivative, right.derivative, strict=True)),
        )

    __radd__ = __add__

    def __neg__(self) -> _Dual:
        return _Dual(-self.value, tuple(-value for value in self.derivative))

    def __sub__(self, other: float | _Dual) -> _Dual:
        return self + (-_dual(other, len(self.derivative)))

    def __rsub__(self, other: float | _Dual) -> _Dual:
        return _dual(other, len(self.derivative)) - self

    def __mul__(self, other: float | _Dual) -> _Dual:
        right = _dual(other, len(self.derivative))
        return _Dual(
            self.value * right.value,
            tuple(
                self.value * b + right.value * a
                for a, b in zip(self.derivative, right.derivative, strict=True)
            ),
        )

    __rmul__ = __mul__

    def __truediv__(self, other: float | _Dual) -> _Dual:
        right = _dual(other, len(self.derivative))
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


def _dual(value: float | _Dual, size: int) -> _Dual:
    if isinstance(value, _Dual):
        return value
    return _Dual(float(value), (0.0,) * size)


def default_state_bounds(inputs: XenonGlobalInputs) -> StateBounds:
    """Return finite box bounds; ordering restrictions are separate inequalities."""

    voltage = inputs.anode_voltage_v
    current = inputs.anode_current_a
    positive_temperature = voltage * 1.0e-14
    lower = (
        *(inputs.cathode_potential_v for _ in range(4)),
        *(positive_temperature for _ in range(4)),
        *(0.0 for _ in range(4)),
        *(0.0 for _ in range(5)),
        *(0.0 for _ in range(4)),
        -2.0 * current,
        *(0.0 for _ in range(3)),
    )
    upper = (
        *(1.5 * voltage for _ in range(4)),
        *(2.0 * voltage for _ in range(4)),
        *(2.0 * current for _ in range(4)),
        *(2.0 * current for _ in range(5)),
        *(2.0 * current for _ in range(4)),
        0.0,
        *(2.0 * current for _ in range(3)),
    )
    return StateBounds(lower=lower, upper=upper)


def _gains(
    vector: Sequence[float | _Dual], inputs: XenonGlobalInputs
) -> tuple[float | _Dual, ...]:
    phi = vector[0:4]
    temperature = vector[4:8]
    return (
        phi[0] - inputs.cathode_potential_v + inputs.cathode_electron_temperature_ev,
        phi[1] - phi[0] + temperature[0],
        phi[2] - phi[1] + temperature[1],
        phi[3] - phi[2] + temperature[2],
    )


def _raw_residual(
    vector: Sequence[float | _Dual], inputs: XenonGlobalInputs
) -> tuple[float | _Dual, ...]:
    phi = vector[0:4]
    temperature = vector[4:8]
    source = vector[8:12]
    electron = vector[12:17]
    ion = vector[17:22]
    cusp_ion = vector[22:25]
    probability = inputs.cusp_arrival_probabilities
    gain = _gains(vector, inputs)
    ionization_energy = inputs.xenon_ionization_energy_ev
    excitation_fraction = inputs.excitation_fraction
    ionization_fraction = inputs.ionization_fraction
    thermalization_fraction = inputs.thermalization_fraction

    first_drop = phi[0] - inputs.cathode_potential_v
    first_drop_value = first_drop.value if isinstance(first_drop, _Dual) else first_drop
    if first_drop_value < 0.0:
        raise PlasmaValidationError(
            "first plasma potential must not be below the cathode potential"
        )
    if any(
        (value.value if isinstance(value, _Dual) else value) < 0.0
        for value in gain
    ):
        raise PlasmaValidationError("every cell electron-energy gain must be non-negative")
    if isinstance(first_drop, _Dual):
        emitted = inputs.cathode_perveance_a_per_v_3_2 * first_drop.power_three_halves()
    else:
        emitted = inputs.cathode_perveance_a_per_v_3_2 * first_drop**1.5

    residual: list[float | _Dual] = [electron[0] - emitted]
    for cell in range(3):
        residual.append(
            electron[cell + 1]
            - electron[cell] * (1.0 - probability[cell])
            - source[cell]
        )
    for cell in range(4):
        residual.append(
            source[cell]
            - electron[cell]
            * (1.0 - probability[cell])
            * ionization_fraction
            * gain[cell]
            / ionization_energy
        )
    for cell in range(3):
        residual.append(ion[cell] - ion[cell + 1] - source[cell] + cusp_ion[cell])
    residual.append(ion[3] - source[3] - ion[4])
    for cell in range(1, 4):
        transported_current = (
            electron[cell] * (1.0 - probability[cell]) + source[cell]
        )
        residual.append(
            temperature[cell] * transported_current
            - thermalization_fraction
            * electron[cell]
            * (1.0 - probability[cell])
            * gain[cell]
        )
    for interface in range(5):
        residual.append(electron[interface] + ion[interface] - inputs.anode_current_a)
    for cusp in range(3):
        residual.append(cusp_ion[cusp] - electron[cusp] * probability[cusp])

    transmitted: list[float | _Dual] = []
    for cell in range(4):
        incoming_power = electron[cell] * (1.0 - probability[cell]) * gain[cell]
        transmitted.append(incoming_power)
        outgoing_current = electron[cell] * (1.0 - probability[cell]) + source[cell]
        residual.append(
            (1.0 - excitation_fraction) * incoming_power
            - outgoing_current * temperature[cell]
            - source[cell] * ionization_energy
        )

    beam = (
        ion[3] * phi[3]
        + (ion[2] - ion[3]) * phi[2]
        + (ion[1] - ion[2]) * phi[1]
        + (ion[0] - ion[1]) * phi[0]
    )
    ionization_loss = ionization_energy * sum(source)
    excitation_loss = excitation_fraction * sum(transmitted)
    cusp_loss = (
        electron[0]
        * probability[0]
        * (phi[0] - inputs.cathode_potential_v + ionization_energy)
        + electron[1]
        * probability[1]
        * (phi[1] - phi[0] + ionization_energy + temperature[0])
        + electron[2]
        * probability[2]
        * (phi[2] - phi[1] + ionization_energy + temperature[1])
    )
    anode_delta = phi[3] - inputs.anode_voltage_v
    anode_electron_loss = (
        electron[3]
        * probability[3]
        * (inputs.anode_voltage_v - phi[2] + temperature[2])
        + (source[3] + electron[3] * (1.0 - probability[3]))
        * (anode_delta + temperature[3])
    )
    if inputs.anode_ion_energy_sign is AnodeIonEnergySign.SOURCE_MINUS_SIGN:
        anode_ion_exchange = -ion[4] * anode_delta
    else:
        anode_ion_exchange = ion[4] * anode_delta
    anode_net_power = anode_electron_loss + anode_ion_exchange
    residual.append(
        beam
        + ionization_loss
        + excitation_loss
        + cusp_loss
        + anode_net_power
        - inputs.anode_voltage_v * inputs.anode_current_a
    )
    return tuple(residual)


def _require_finite_vector(name: str, values: Sequence[float]) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if any(not isfinite(value) for value in converted):
        raise PlasmaNumericsError(f"{name} produced a non-finite value")
    return converted


def residual_scales(inputs: XenonGlobalInputs) -> tuple[float, ...]:
    current_scale = inputs.anode_current_a
    power_scale = inputs.anode_voltage_v * inputs.anode_current_a
    if (
        not isfinite(current_scale)
        or not isfinite(power_scale)
        or current_scale <= 0.0
        or power_scale < float_info.min
    ):
        raise PlasmaValidationError(
            "residual current and power scales must be finite, positive, "
            "and representable"
        )
    return tuple(power_scale if index in POWER_ROWS else current_scale for index in range(28))


def power_balance(state: PlasmaState, inputs: XenonGlobalInputs) -> PowerBalance:
    vector = state.to_vector()
    raw = _raw_residual(vector, inputs)
    global_closure = float(raw[-1])
    phi = state.plasma_potential_v
    temperature = state.electron_temperature_ev
    source = state.ionization_source_current_a
    electron = state.electron_current_a
    ion = state.ion_current_a
    probability = inputs.cusp_arrival_probabilities
    gain = _gains(vector, inputs)
    beam = (
        ion[3] * phi[3]
        + (ion[2] - ion[3]) * phi[2]
        + (ion[1] - ion[2]) * phi[1]
        + (ion[0] - ion[1]) * phi[0]
    )
    ionization = inputs.xenon_ionization_energy_ev * sum(source)
    excitation = inputs.excitation_fraction * sum(
        electron[cell] * (1.0 - probability[cell]) * float(gain[cell])
        for cell in range(4)
    )
    cusp = (
        electron[0]
        * probability[0]
        * (phi[0] - inputs.cathode_potential_v + inputs.xenon_ionization_energy_ev)
        + electron[1]
        * probability[1]
        * (phi[1] - phi[0] + inputs.xenon_ionization_energy_ev + temperature[0])
        + electron[2]
        * probability[2]
        * (phi[2] - phi[1] + inputs.xenon_ionization_energy_ev + temperature[1])
    )
    anode_delta = phi[3] - inputs.anode_voltage_v
    anode_electron = (
        electron[3]
        * probability[3]
        * (inputs.anode_voltage_v - phi[2] + temperature[2])
        + (source[3] + electron[3] * (1.0 - probability[3]))
        * (anode_delta + temperature[3])
    )
    anode_ion_exchange = (
        -ion[4] * anode_delta
        if inputs.anode_ion_energy_sign is AnodeIonEnergySign.SOURCE_MINUS_SIGN
        else ion[4] * anode_delta
    )
    anode_net = anode_electron + anode_ion_exchange
    values = _require_finite_vector(
        "power balance",
        (
            beam,
            ionization,
            excitation,
            cusp,
            anode_electron,
            anode_ion_exchange,
            anode_net,
            inputs.anode_voltage_v * inputs.anode_current_a,
            global_closure,
        ),
    )
    return PowerBalance(*values)


def evaluate_residual(state: PlasmaState, inputs: XenonGlobalInputs) -> ResidualEvaluation:
    """Evaluate raw dimensional and normalized equations on the CPU."""

    raw = _require_finite_vector(
        "residual",
        _raw_residual(state.to_vector(), inputs),  # type: ignore[arg-type]
    )
    scales = residual_scales(inputs)
    normalized = _require_finite_vector(
        "normalized residual",
        tuple(value / scale for value, scale in zip(raw, scales, strict=True)),
    )
    powers = power_balance(state, inputs)
    closures = ConservationClosures(
        interface_current_residual_a=raw[15:20],  # type: ignore[arg-type]
        cusp_current_residual_a=raw[20:23],  # type: ignore[arg-type]
        cell_energy_residual_w=raw[23:27],  # type: ignore[arg-type]
        global_energy_residual_w=raw[27],
    )
    return ResidualEvaluation(raw=raw, normalized=normalized, powers=powers, closures=closures)


def evaluate_residual_batch(
    states: Sequence[PlasmaState], inputs: Sequence[XenonGlobalInputs]
) -> tuple[ResidualEvaluation, ...]:
    """Batchable fixed-shape CPU API; no object-array or ragged semantics."""

    if len(states) == 0 or len(states) != len(inputs):
        raise PlasmaValidationError(
            "states and inputs must be non-empty equal-length sequences"
        )
    if any(not isinstance(state, PlasmaState) for state in states):
        raise PlasmaValidationError("every states entry must be PlasmaState")
    if any(not isinstance(point, XenonGlobalInputs) for point in inputs):
        raise PlasmaValidationError("every inputs entry must be XenonGlobalInputs")
    return tuple(
        evaluate_residual(state, point)
        for state, point in zip(states, inputs, strict=True)
    )


def analytic_jacobian(
    state: PlasmaState, inputs: XenonGlobalInputs
) -> tuple[tuple[float, ...], ...]:
    """Exact forward-mode chain-rule Jacobian of the normalized residual."""

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
    if len(matrix) != 28 or any(len(row) != 25 for row in matrix):
        raise PlasmaNumericsError("analytic Jacobian has an invalid shape")
    if any(not isfinite(value) for row in matrix for value in row):
        raise PlasmaNumericsError("analytic Jacobian produced a non-finite value")
    return matrix


def finite_difference_jacobian(
    state: PlasmaState,
    inputs: XenonGlobalInputs,
    *,
    relative_step: float = 1.0e-6,
) -> tuple[tuple[float, ...], ...]:
    """Bound-aware central finite-difference Jacobian used only as a check."""

    if not isfinite(relative_step) or relative_step <= 0.0:
        raise PlasmaValidationError("relative_step must be finite and positive")
    vector = list(state.to_vector())
    bounds = default_state_bounds(inputs)
    columns: list[tuple[float, ...]] = []
    for column, value in enumerate(vector):
        step = relative_step * max(1.0, abs(value))
        low_room = value - bounds.lower[column]
        high_room = bounds.upper[column] - value
        if low_room >= step and high_room >= step:
            minus = vector.copy()
            plus = vector.copy()
            minus[column] -= step
            plus[column] += step
            left = evaluate_residual(PlasmaState.from_vector(minus), inputs).normalized
            right = evaluate_residual(PlasmaState.from_vector(plus), inputs).normalized
            columns.append(
                tuple(
                    (b - a) / (2.0 * step)
                    for a, b in zip(left, right, strict=True)
                )
            )
        elif high_room > 0.0:
            actual = min(step, high_room)
            plus = vector.copy()
            plus[column] += actual
            base = evaluate_residual(state, inputs).normalized
            right = evaluate_residual(PlasmaState.from_vector(plus), inputs).normalized
            columns.append(tuple((b - a) / actual for a, b in zip(base, right, strict=True)))
        elif low_room > 0.0:
            actual = min(step, low_room)
            minus = vector.copy()
            minus[column] -= actual
            left = evaluate_residual(PlasmaState.from_vector(minus), inputs).normalized
            base = evaluate_residual(state, inputs).normalized
            columns.append(tuple((b - a) / actual for a, b in zip(left, base, strict=True)))
        else:
            raise PlasmaValidationError(f"state variable {column} has no finite-difference room")
    return tuple(tuple(columns[column][row] for column in range(25)) for row in range(28))


def potential_parametrized_state(
    inputs: XenonGlobalInputs, potentials: Sequence[float]
) -> PlasmaState:
    """Return the unique state that satisfies rows R00-R26 for given potentials.

    Rows R00-R26 determine every other state value from the four cell
    potentials by an explicit recursion (emission -> ionization -> cell energy
    -> continuity -> interface currents -> cusp currents).  The result is a
    diagnostic parametrization of the 27-row solution manifold; it does not
    evaluate the global power row R27 and it is not a published solution.
    """

    phi = _tuple_of_four("potentials", potentials)
    probability = inputs.cusp_arrival_probabilities
    cathode = inputs.cathode_potential_v
    energy = inputs.xenon_ionization_energy_ev
    if phi[0] < cathode:
        raise PlasmaValidationError("potentials[0] must not be below the cathode potential")
    electron: list[float] = [
        inputs.cathode_perveance_a_per_v_3_2 * (phi[0] - cathode) ** 1.5
    ]
    source: list[float] = []
    temperature: list[float] = []
    for cell in range(4):
        if cell == 0:
            gain = phi[0] - cathode + inputs.cathode_electron_temperature_ev
        else:
            gain = phi[cell] - phi[cell - 1] + temperature[cell - 1]
        if gain < 0.0:
            raise PlasmaValidationError("every cell electron-energy gain must be non-negative")
        transmitted = electron[cell] * (1.0 - probability[cell])
        ionization = transmitted * inputs.ionization_fraction * gain / energy
        source.append(ionization)
        outgoing = transmitted + ionization
        if outgoing <= 0.0:
            raise PlasmaValidationError("transported electron current must be positive")
        temperature.append(
            ((1.0 - inputs.excitation_fraction) * transmitted * gain - ionization * energy)
            / outgoing
        )
        if cell < 3:
            electron.append(outgoing)
    electron.append(electron[3] + source[3])
    ion = tuple(inputs.anode_current_a - value for value in electron)
    cusp_ion = tuple(electron[cell] * probability[cell] for cell in range(3))
    return PlasmaState(
        plasma_potential_v=phi,
        electron_temperature_ev=tuple(temperature),  # type: ignore[arg-type]
        ionization_source_current_a=tuple(source),  # type: ignore[arg-type]
        electron_current_a=tuple(electron),  # type: ignore[arg-type]
        ion_current_a=ion,  # type: ignore[arg-type]
        cusp_ion_current_a=cusp_ion,  # type: ignore[arg-type]
    )


def global_row_closed_form(state: PlasmaState, inputs: XenonGlobalInputs) -> float:
    """Closed form of the raw global power row R27 on the R00-R26 manifold [W].

    Substituting rows R00-R26 into R27 leaves

    ``2*(j_e3*(1-p4)+I4)*(phi_4-Ua) + EI*(p1*j_e0 + p2*j_e1 + p3*j_e2)``.

    The first term is the anode-fall sign inconsistency (zero only at
    ``phi_4 = Ua``); the second is the recombination energy of cusp-lost ions,
    which ``Pcusp`` books a second time after ``PI``.  Under the admissible
    region ``phi_4 >= Ua`` both terms are non-negative, so no admissible root
    exists when any interior cusp probability is positive.  Documented in
    ``docs/workstreams/global-plasma-closure-analysis.md``; the correction is
    ``PROPOSED_NOT_ACCEPTED`` in ``spec/plasma/equation-ledger.json``.
    """

    probability = inputs.cusp_arrival_probabilities
    electron = state.electron_current_a
    transported = electron[3] * (1.0 - probability[3]) + state.ionization_source_current_a[3]
    anode_fall_term = 2.0 * transported * (state.plasma_potential_v[3] - inputs.anode_voltage_v)
    recombination_term = inputs.xenon_ionization_energy_ev * fsum(
        electron[cell] * probability[cell] for cell in range(3)
    )
    value = anode_fall_term + recombination_term
    if not isfinite(value):
        raise PlasmaNumericsError("global row closed form produced a non-finite value")
    return value


def _tuple_of_four(name: str, values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise PlasmaValidationError(f"{name} must contain exactly four values")
    converted = tuple(float(value) for value in values)
    if any(not isfinite(value) for value in converted):
        raise PlasmaValidationError(f"{name} must be finite")
    return converted  # type: ignore[return-value]


def constraint_margins(state: PlasmaState, inputs: XenonGlobalInputs) -> tuple[float, ...]:
    """Return true inequality margins; feasibility means every value is >= zero."""

    phi = state.plasma_potential_v
    temperature = state.electron_temperature_ev
    ion = state.ion_current_a
    powers = power_balance(state, inputs)
    return (
        phi[0] - inputs.cathode_potential_v,
        phi[1] - phi[0],
        phi[2] - phi[1],
        phi[3] - phi[2],
        phi[3] - inputs.anode_voltage_v,
        *temperature,
        ion[3] + ion[4],
        powers.beam_power_w,
        powers.ionization_loss_w,
        powers.excitation_loss_w,
        powers.cusp_loss_w,
        powers.anode_electron_loss_w,
    )


def is_feasible(
    state: PlasmaState,
    inputs: XenonGlobalInputs,
    bounds: StateBounds | None = None,
) -> bool:
    selected = default_state_bounds(inputs) if bounds is None else bounds
    vector = state.to_vector()
    in_box = all(
        low <= value <= high
        for value, low, high in zip(vector, selected.lower, selected.upper, strict=True)
    )
    return in_box and all(margin >= 0.0 for margin in constraint_margins(state, inputs))
