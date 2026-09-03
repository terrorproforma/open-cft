"""Rows R00-R37 of the sheath-closed four-cell power balance (v2).

Rows R00-R26 are the v1 rows verbatim (the parity test in
``tests/plasma_v2`` compares them against ``cft_revival.plasma`` to
round-off).  R27 carries the two corrections of
``spec/plasma/equation-ledger.json#global_row_consistency`` and is an
identity on the R00-R26 manifold.  R28-R37 are new.

Forward-mode duals give the exact Jacobian; nothing here depends on numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, isfinite, log, pi, sqrt
from sys import float_info
from typing import Sequence

from cft_revival.plasma import PlasmaNumericsError, PlasmaValidationError

from .constants import ELECTRON_MASS_KG, ELEMENTARY_CHARGE_C, MASS_FLUX_RATIO, XENON_MASS_KG
from .models import (
    CUSP_PROBABILITY_ROWS,
    POWER_ROWS,
    RESIDUAL_SIZE,
    STATE_SIZE,
    VOLTAGE_ROWS,
    AnodeRow,
    CuspEnergySplit,
    CuspLossClosure,
    FourthPotentialRow,
    PowerBalanceV2,
    ResidualEvaluationV2,
    SheathClosureInputs,
    SheathClosureState,
    require_numerics,
)

@dataclass(frozen=True, slots=True)
class _Dual:
    value: float
    derivative: tuple[float, ...]

    def __add__(self, other: object) -> _Dual:
        right = _dual(other, len(self.derivative))
        return _Dual(
            self.value + right.value,
            tuple(a + b for a, b in zip(self.derivative, right.derivative, strict=True)),
        )

    __radd__ = __add__

    def __neg__(self) -> _Dual:
        return _Dual(-self.value, tuple(-value for value in self.derivative))

    def __sub__(self, other: object) -> _Dual:
        return self + (-_dual(other, len(self.derivative)))

    def __rsub__(self, other: object) -> _Dual:
        return _dual(other, len(self.derivative)) - self

    def __mul__(self, other: object) -> _Dual:
        right = _dual(other, len(self.derivative))
        return _Dual(
            self.value * right.value,
            tuple(
                self.value * b + right.value * a
                for a, b in zip(self.derivative, right.derivative, strict=True)
            ),
        )

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> _Dual:
        right = _dual(other, len(self.derivative))
        denominator = right.value * right.value
        return _Dual(
            self.value / right.value,
            tuple(
                (a * right.value - self.value * b) / denominator
                for a, b in zip(self.derivative, right.derivative, strict=True)
            ),
        )

    def __rtruediv__(self, other: object) -> _Dual:
        return _dual(other, len(self.derivative)) / self


def _dual(value: object, size: int) -> _Dual:
    if isinstance(value, _Dual):
        return value
    return _Dual(float(value), (0.0,) * size)  # type: ignore[arg-type]


def _val(value: object) -> float:
    return value.value if isinstance(value, _Dual) else float(value)  # type: ignore[arg-type]


def _pow_three_halves(value: object) -> object:
    if isinstance(value, _Dual):
        root = sqrt(value.value)
        return _Dual(value.value * root, tuple(1.5 * root * d for d in value.derivative))
    return float(value) ** 1.5  # type: ignore[arg-type]


def _sqrt(value: object) -> object:
    if isinstance(value, _Dual):
        root = sqrt(value.value)
        if root == 0.0:
            return _Dual(0.0, tuple(0.0 for _ in value.derivative))
        return _Dual(root, tuple(0.5 * d / root for d in value.derivative))
    return sqrt(float(value))  # type: ignore[arg-type]


def _exp(value: object) -> object:
    if isinstance(value, _Dual):
        result = exp(value.value)
        return _Dual(result, tuple(result * d for d in value.derivative))
    return exp(float(value))  # type: ignore[arg-type]


def _log(value: object) -> object:
    if isinstance(value, _Dual):
        if value.value <= 0.0:
            raise PlasmaValidationError("logarithm argument must be positive")
        return _Dual(log(value.value), tuple(d / value.value for d in value.derivative))
    number = float(value)  # type: ignore[arg-type]
    if number <= 0.0:
        raise PlasmaValidationError("logarithm argument must be positive")
    return log(number)


def _exp_neg_ratio(numerator: object, denominator: object) -> object:
    """``exp(-numerator/denominator)`` for ``denominator > 0`` without 0*inf."""

    ratio = numerator / denominator
    if _val(ratio) > 700.0:
        size = len(ratio.derivative) if isinstance(ratio, _Dual) else 0
        return _Dual(0.0, (0.0,) * size) if size else 0.0
    return _exp(-ratio)


def electron_mean_speed_m_per_s(temperature_ev: object) -> object:
    """Maxwellian mean speed ``sqrt(8 e T / (pi m_e))`` (Lieberman & Lichtenberg 2.4.9)."""

    return _sqrt(temperature_ev * (8.0 * ELEMENTARY_CHARGE_C / (pi * ELECTRON_MASS_KG)))


def hybrid_leak_width_m(temperature_ev: object, wall_field_t: float) -> object:
    """Hybrid gyroradius ``sqrt(r_e r_i)`` at the cusp wall field (Goebel & Katz Ch. 4).

    Declared characteristic speeds: electrons at ``v_perp = sqrt(2 e T/m_e)``
    (``r_e = sqrt(2 m_e e T)/(e B)``), ions at the Bohm speed
    (``r_i = sqrt(M e T)/(e B)``).  The literature does not fix these choices;
    the leak-width prefactor swept over [1, 4] absorbs them.
    """

    r_e = _sqrt(temperature_ev * (2.0 * ELECTRON_MASS_KG * ELEMENTARY_CHARGE_C)) / (
        ELEMENTARY_CHARGE_C * wall_field_t
    )
    r_i = _sqrt(temperature_ev * (XENON_MASS_KG * ELEMENTARY_CHARGE_C)) / (
        ELEMENTARY_CHARGE_C * wall_field_t
    )
    return _sqrt(r_e * r_i)


def default_state_bounds(inputs: SheathClosureInputs) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Finite box bounds for the 31-variable state (v1 box extended)."""

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
        *(0.0 for _ in range(3)),
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
        *(2.0 * voltage for _ in range(3)),
        *(1.0 - 1.0e-9 for _ in range(3)),
    )
    return lower, upper


def residual_scales(inputs: SheathClosureInputs) -> tuple[float, ...]:
    """Rows in A divide by Ia, rows in W by Ua*Ia, rows in V by Ua, dimensionless rows by 1."""

    current = inputs.anode_current_a
    voltage = inputs.anode_voltage_v
    power = voltage * current
    if power < float_info.min:
        raise PlasmaValidationError("Ua*Ia must be a normal positive power")
    scales: list[float] = []
    for row in range(RESIDUAL_SIZE):
        if row in POWER_ROWS:
            scales.append(power)
        elif row in VOLTAGE_ROWS:
            scales.append(voltage)
        elif row in CUSP_PROBABILITY_ROWS:
            scales.append(
                current if inputs.cusp_loss_closure is CuspLossClosure.CL4_HYBRID_AREA else 1.0
            )
        else:
            scales.append(current)
    return tuple(scales)


def _cusp_loss_row(
    cusp_index: int,
    probability: object,
    sheath_drop: object,
    temperature: object,
    entering_current: object,
    inputs: SheathClosureInputs,
) -> object:
    spec = inputs.cusps[cusp_index]
    closure = inputs.cusp_loss_closure
    if closure is CuspLossClosure.CL1_DECLARED:
        return probability - inputs.declared_cusp_probabilities[cusp_index]
    boltzmann = _exp_neg_ratio(sheath_drop, temperature)
    if closure is CuspLossClosure.CL3_SHEATH_LIMITED:
        return probability - spec.access_fraction * boltzmann
    density = spec.electron_density_per_m3
    field = spec.wall_field_t
    radius = inputs.wall_radius_m
    assert density is not None and field is not None and radius is not None
    leak_area = inputs.leak_width_prefactor * hybrid_leak_width_m(temperature, field) * (
        2.0 * pi * radius
    )
    thermal_current = (
        0.25 * ELEMENTARY_CHARGE_C * density * electron_mean_speed_m_per_s(temperature) * leak_area
    )
    return probability * entering_current - thermal_current * boltzmann


def raw_residual(vector: Sequence[object], inputs: SheathClosureInputs) -> tuple[object, ...]:
    """Evaluate the 38 dimensional rows on floats or duals."""

    phi = vector[0:4]
    temperature = vector[4:8]
    source = vector[8:12]
    electron = vector[12:17]
    ion = vector[17:22]
    cusp_ion = vector[22:25]
    sheath_drop = vector[25:28]
    probability = (vector[28], vector[29], vector[30], inputs.anode_cusp_probability)
    cathode = inputs.cathode_potential_v
    voltage = inputs.anode_voltage_v
    energy = inputs.xenon_ionization_energy_ev
    excitation_fraction = inputs.excitation_fraction
    ionization_fraction = inputs.ionization_fraction
    thermalization_fraction = inputs.thermalization_fraction

    gain = (
        phi[0] - cathode + inputs.cathode_electron_temperature_ev,
        phi[1] - phi[0] + temperature[0],
        phi[2] - phi[1] + temperature[1],
        phi[3] - phi[2] + temperature[2],
    )
    first_drop = phi[0] - cathode
    if _val(first_drop) < 0.0:
        raise PlasmaValidationError("first plasma potential must not be below the cathode potential")
    if any(_val(value) < 0.0 for value in gain):
        raise PlasmaValidationError("every cell electron-energy gain must be non-negative")
    for cell in range(4):
        if _val(temperature[cell]) <= 0.0:
            raise PlasmaValidationError("electron temperatures must be positive")

    survivors = tuple(electron[cell] * (1.0 - probability[cell]) for cell in range(4))
    lost = tuple(electron[cell] * probability[cell] for cell in range(4))
    emitted = inputs.cathode_perveance_a_per_v_3_2 * _pow_three_halves(first_drop)

    residual: list[object] = [electron[0] - emitted]
    for cell in range(3):
        residual.append(electron[cell + 1] - survivors[cell] - source[cell])
    for cell in range(4):
        residual.append(source[cell] - survivors[cell] * ionization_fraction * gain[cell] / energy)
    for cell in range(3):
        residual.append(ion[cell] - ion[cell + 1] - source[cell] + cusp_ion[cell])
    residual.append(ion[3] - source[3] - ion[4])
    for cell in range(1, 4):
        residual.append(
            temperature[cell] * (survivors[cell] + source[cell])
            - thermalization_fraction * survivors[cell] * gain[cell]
        )
    for interface in range(5):
        residual.append(electron[interface] + ion[interface] - inputs.anode_current_a)
    for cusp in range(3):
        residual.append(cusp_ion[cusp] - lost[cusp])
    transmitted: list[object] = []
    for cell in range(4):
        incoming_power = survivors[cell] * gain[cell]
        transmitted.append(incoming_power)
        residual.append(
            (1.0 - excitation_fraction) * incoming_power
            - (survivors[cell] + source[cell]) * temperature[cell]
            - source[cell] * energy
        )

    # R27 with the two corrections: Pcusp without +EI, anode electron term with
    # the electron sign.  On the R00-R26 manifold this row is identically zero.
    beam = (
        ion[3] * phi[3]
        + (ion[2] - ion[3]) * phi[2]
        + (ion[1] - ion[2]) * phi[1]
        + (ion[0] - ion[1]) * phi[0]
    )
    ionization_loss = energy * (source[0] + source[1] + source[2] + source[3])
    excitation_loss = excitation_fraction * (
        transmitted[0] + transmitted[1] + transmitted[2] + transmitted[3]
    )
    cusp_loss = lost[0] * gain[0] + lost[1] * gain[1] + lost[2] * gain[2]
    anode_electron = lost[3] * (voltage - phi[2] + temperature[2]) + (
        survivors[3] + source[3]
    ) * (voltage - phi[3] + temperature[3])
    anode_ion = -ion[4] * (phi[3] - voltage)
    residual.append(
        beam
        + ionization_loss
        + excitation_loss
        + cusp_loss
        + anode_electron
        + anode_ion
        - voltage * inputs.anode_current_a
    )

    # R28-R30: floating-dielectric sheath drop, linear in the cell temperature.
    coefficients = inputs.sheath_coefficients()
    for cusp in range(3):
        residual.append(sheath_drop[cusp] - coefficients[cusp] * temperature[cusp])

    # R31: anode row.
    closure = inputs.potentials
    anode_fall = phi[3] - voltage
    if closure.anode_row is AnodeRow.SHEATH:
        anode_ion_current = -ion[4]
        if _val(anode_ion_current) <= 0.0 or _val(electron[4]) <= 0.0:
            raise PlasmaValidationError(
                "the anode sheath row requires a positive ion current to the anode"
            )
        residual.append(
            anode_fall - temperature[3] * _log(MASS_FLUX_RATIO * anode_ion_current / electron[4])
        )
    else:
        residual.append(anode_fall - closure.anode_fall_v)

    # R32-R33: declared interior steps (CL-3-potentials).
    residual.append(phi[2] - phi[1] - closure.interior_step_3_v)
    residual.append(phi[3] - phi[2] - closure.interior_step_4_v)

    # R34: fourth potential relation.
    if closure.fourth_row is FourthPotentialRow.ANODE_FALL_DECLARED:
        residual.append(anode_fall - closure.anode_fall_v)
    else:
        assert closure.cathode_coupling_v is not None
        residual.append(first_drop - closure.cathode_coupling_v)

    # R35-R37: cusp loss probabilities under the declared closure.
    for cusp in range(3):
        residual.append(
            _cusp_loss_row(
                cusp,
                probability[cusp],
                sheath_drop[cusp],
                temperature[cusp],
                electron[cusp],
                inputs,
            )
        )
    if len(residual) != RESIDUAL_SIZE:
        raise PlasmaNumericsError("v2 residual has an invalid length")
    return tuple(residual)


def _finite_vector(name: str, values: Sequence[object]) -> tuple[float, ...]:
    converted = tuple(_val(value) for value in values)
    if any(not isfinite(value) for value in converted):
        raise PlasmaNumericsError(f"{name} produced a non-finite value")
    return converted


def power_balance(state: SheathClosureState, inputs: SheathClosureInputs) -> PowerBalanceV2:
    """Corrected power components and the per-cusp energy split."""

    vector = state.to_vector()
    raw = raw_residual(vector, inputs)
    core = state.core
    phi = core.plasma_potential_v
    temperature = core.electron_temperature_ev
    source = core.ionization_source_current_a
    electron = core.electron_current_a
    ion = core.ion_current_a
    probability = (*state.cusp_probability, inputs.anode_cusp_probability)
    cathode = inputs.cathode_potential_v
    voltage = inputs.anode_voltage_v
    gain = (
        phi[0] - cathode + inputs.cathode_electron_temperature_ev,
        phi[1] - phi[0] + temperature[0],
        phi[2] - phi[1] + temperature[1],
        phi[3] - phi[2] + temperature[2],
    )
    survivors = tuple(electron[cell] * (1.0 - probability[cell]) for cell in range(4))
    lost = tuple(electron[cell] * probability[cell] for cell in range(4))
    beam = (
        ion[3] * phi[3]
        + (ion[2] - ion[3]) * phi[2]
        + (ion[1] - ion[2]) * phi[1]
        + (ion[0] - ion[1]) * phi[0]
    )
    ionization = inputs.xenon_ionization_energy_ev * fsum(source)
    excitation = inputs.excitation_fraction * fsum(
        survivors[cell] * gain[cell] for cell in range(4)
    )
    splits: list[CuspEnergySplit] = []
    for cusp in range(3):
        drop = state.sheath_drop_v[cusp]
        total = lost[cusp] * gain[cusp]
        electron_wall = lost[cusp] * (gain[cusp] - drop)
        ion_wall = lost[cusp] * drop
        splits.append(
            CuspEnergySplit(
                lost_electron_current_a=lost[cusp],
                entering_energy_ev=gain[cusp],
                sheath_drop_v=drop,
                total_w=require_numerics("cusp total", total),
                electron_wall_w=require_numerics("cusp electron", electron_wall),
                ion_wall_w=require_numerics("cusp ion", ion_wall),
                electron_wall_energy_margin_ev=gain[cusp] - drop,
                maxwellian_electron_estimate_w=lost[cusp] * (2.0 * temperature[cusp] + drop),
                maxwellian_ion_estimate_w=lost[cusp] * (drop + 0.5 * temperature[cusp]),
            )
        )
    cusp_total = fsum(split.total_w for split in splits)
    anode_electron = lost[3] * (voltage - phi[2] + temperature[2]) + (
        survivors[3] + source[3]
    ) * (voltage - phi[3] + temperature[3])
    anode_ion = -ion[4] * (phi[3] - voltage)
    values = _finite_vector(
        "power balance",
        (
            beam,
            ionization,
            excitation,
            cusp_total,
            fsum(split.electron_wall_w for split in splits),
            fsum(split.ion_wall_w for split in splits),
            anode_electron,
            anode_ion,
            voltage * inputs.anode_current_a,
            _val(raw[27]),
        ),
    )
    return PowerBalanceV2(*values, cusps=tuple(splits))  # type: ignore[arg-type]


MARGIN_NAMES: tuple[str, ...] = (
    "phi_1 - phi_0",
    "phi_2 - phi_1",
    "phi_3 - phi_2",
    "phi_4 - phi_3",
    "phi_4 - Ua",
    "T_1",
    "T_2",
    "T_3",
    "T_4",
    "j_i3 + j_i4",
    "beam_power",
    "ionization_loss",
    "excitation_loss",
    "cusp_loss",
    "anode_electron_loss",
    "anode_ion_loss",
    "sheath_drop_1",
    "sheath_drop_2",
    "sheath_drop_3",
    "p_1",
    "p_2",
    "p_3",
    "1 - p_1",
    "1 - p_2",
    "1 - p_3",
    "anode_ion_current",
    "cusp_1_electron_wall_energy",
    "cusp_2_electron_wall_energy",
    "cusp_3_electron_wall_energy",
)
CUSP_ENERGY_MARGIN_INDICES: tuple[int, ...] = (26, 27, 28)


def constraint_margins(
    state: SheathClosureState, inputs: SheathClosureInputs
) -> tuple[float, ...]:
    """True inequality margins; feasibility means every enforced value is >= 0."""

    core = state.core
    phi = core.plasma_potential_v
    powers = power_balance(state, inputs)
    ion = core.ion_current_a
    # Strict positivity of the anode ion current is enforced by the logarithm
    # guard in raw_residual; the margin reports the value itself.
    anode_ion_current = -ion[4] if inputs.potentials.anode_row is AnodeRow.SHEATH else 1.0
    return (
        phi[0] - inputs.cathode_potential_v,
        phi[1] - phi[0],
        phi[2] - phi[1],
        phi[3] - phi[2],
        phi[3] - inputs.anode_voltage_v,
        *core.electron_temperature_ev,
        ion[3] + ion[4],
        powers.beam_power_w,
        powers.ionization_loss_w,
        powers.excitation_loss_w,
        powers.cusp_loss_w,
        powers.anode_electron_loss_w,
        powers.anode_ion_loss_w,
        *state.sheath_drop_v,
        *state.cusp_probability,
        *(1.0 - value for value in state.cusp_probability),
        anode_ion_current,
        *(split.electron_wall_energy_margin_ev for split in powers.cusps),
    )


def is_feasible(
    state: SheathClosureState,
    inputs: SheathClosureInputs,
    bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    *,
    enforce_cusp_energy_margin: bool = True,
) -> bool:
    lower, upper = default_state_bounds(inputs) if bounds is None else bounds
    vector = state.to_vector()
    if not all(low <= value <= high for value, low, high in zip(vector, lower, upper, strict=True)):
        return False
    margins = constraint_margins(state, inputs)
    for index, margin in enumerate(margins):
        if index in CUSP_ENERGY_MARGIN_INDICES and not enforce_cusp_energy_margin:
            continue
        if not margin >= 0.0:
            return False
    return True


def evaluate_residual(
    state: SheathClosureState, inputs: SheathClosureInputs
) -> ResidualEvaluationV2:
    raw = _finite_vector("residual", raw_residual(state.to_vector(), inputs))
    scales = residual_scales(inputs)
    normalized = _finite_vector(
        "normalized residual",
        tuple(value / scale for value, scale in zip(raw, scales, strict=True)),
    )
    powers = power_balance(state, inputs)
    margins = constraint_margins(state, inputs)
    return ResidualEvaluationV2(
        raw=raw,
        normalized=normalized,
        powers=powers,
        margins=margins,
        margin_names=MARGIN_NAMES,
        cusp_energy_margins_ev=tuple(  # type: ignore[arg-type]
            split.electron_wall_energy_margin_ev for split in powers.cusps
        ),
    )


def analytic_jacobian(
    state: SheathClosureState, inputs: SheathClosureInputs
) -> tuple[tuple[float, ...], ...]:
    """Exact forward-mode Jacobian of the normalized 38-row residual (38 x 31)."""

    vector = state.to_vector()
    size = len(vector)
    duals = tuple(
        _Dual(value, tuple(1.0 if row == column else 0.0 for row in range(size)))
        for column, value in enumerate(vector)
    )
    residual = raw_residual(duals, inputs)
    scales = residual_scales(inputs)
    matrix: list[tuple[float, ...]] = []
    for row, item in enumerate(residual):
        if isinstance(item, _Dual):
            matrix.append(tuple(value / scales[row] for value in item.derivative))
        else:
            matrix.append((0.0,) * size)
    if len(matrix) != RESIDUAL_SIZE or any(len(row) != STATE_SIZE for row in matrix):
        raise PlasmaNumericsError("v2 analytic Jacobian has an invalid shape")
    if any(not isfinite(value) for row in matrix for value in row):
        raise PlasmaNumericsError("v2 analytic Jacobian produced a non-finite value")
    return tuple(matrix)


def finite_difference_jacobian(
    state: SheathClosureState,
    inputs: SheathClosureInputs,
    *,
    relative_step: float = 1.0e-6,
) -> tuple[tuple[float, ...], ...]:
    """Central finite differences for checks only (interior points)."""

    vector = list(state.to_vector())
    columns: list[tuple[float, ...]] = []
    for column, value in enumerate(vector):
        step = relative_step * max(1.0, abs(value))
        minus = vector.copy()
        plus = vector.copy()
        minus[column] -= step
        plus[column] += step
        left = evaluate_residual(SheathClosureState.from_vector(minus), inputs).normalized
        right = evaluate_residual(SheathClosureState.from_vector(plus), inputs).normalized
        columns.append(tuple((b - a) / (2.0 * step) for a, b in zip(left, right, strict=True)))
    return tuple(
        tuple(columns[column][row] for column in range(STATE_SIZE)) for row in range(RESIDUAL_SIZE)
    )


def matrix_rank(
    matrix: Sequence[Sequence[float]], *, tolerance: float | None = None
) -> tuple[int, float]:
    """Numerical rank and condition estimate by pivoted modified Gram-Schmidt."""

    row_count = len(matrix)
    if row_count == 0:
        return 0, float_info.max
    column_count = len(matrix[0])
    columns = [[float(matrix[row][column]) for row in range(row_count)] for column in range(column_count)]

    def dot(left: Sequence[float], right: Sequence[float]) -> float:
        return fsum(a * b for a, b in zip(left, right, strict=True))

    scale = max(sqrt(dot(column, column)) for column in columns)
    selected_tolerance = (
        max(row_count, column_count) * float_info.epsilon * max(scale, 1.0)
        if tolerance is None
        else tolerance
    )
    rank = 0
    diagonal: list[float] = []
    for pivot in range(column_count):
        best = max(range(pivot, column_count), key=lambda index: dot(columns[index], columns[index]))
        if best != pivot:
            columns[pivot], columns[best] = columns[best], columns[pivot]
        norm = sqrt(dot(columns[pivot], columns[pivot]))
        if norm <= selected_tolerance:
            break
        rank += 1
        diagonal.append(norm)
        basis = [value / norm for value in columns[pivot]]
        for column in range(pivot + 1, column_count):
            coefficient = dot(basis, columns[column])
            columns[column] = [
                value - coefficient * direction
                for value, direction in zip(columns[column], basis, strict=True)
            ]
    condition = max(diagonal) / min(diagonal) if diagonal else float_info.max
    return rank, condition
