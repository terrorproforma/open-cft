"""SI constitutive laws for linear, nonlinear, and permanent-magnet materials.

The nonlinear law is a single-valued, odd-symmetric first-quadrant B-H curve.
It deliberately does not model hysteresis, rate dependence, or minor loops.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import Enum
from math import isfinite, nextafter

from .common import (
    MagneticsValidationError,
    VectorRZ,
    finite_float,
    nonempty_identifier,
)

MU0_H_PER_M = 1.2566370614359173e-6


class ExtrapolationPolicy(str, Enum):
    """Behaviour beyond the largest tabulated field magnitude."""

    ERROR = "error"
    LINEAR_TANGENT = "linear_tangent"


@dataclass(frozen=True, slots=True)
class LinearPermeability:
    """Isotropic constant permeability, represented by dimensionless ``mu_r``."""

    material_id: str
    relative_permeability: float

    def __post_init__(self) -> None:
        nonempty_identifier("material_id", self.material_id)
        relative = finite_float("relative_permeability", self.relative_permeability)
        if relative <= 0.0:
            raise MagneticsValidationError("relative_permeability must be positive")
        object.__setattr__(self, "relative_permeability", relative)

    @property
    def permeability_h_per_m(self) -> float:
        return finite_float(
            "permeability_h_per_m",
            MU0_H_PER_M * self.relative_permeability,
        )

    def b_from_h_t(self, magnetic_field_a_per_m: float) -> float:
        field = finite_float("magnetic_field_a_per_m", magnetic_field_a_per_m)
        return finite_float(
            "computed flux density",
            self.permeability_h_per_m * field,
        )

    def h_from_b_a_per_m(self, flux_density_t: float) -> float:
        flux = finite_float("flux_density_t", flux_density_t)
        return finite_float(
            "computed magnetic field",
            flux / self.permeability_h_per_m,
        )

    def differential_permeability_h_per_m(self, magnetic_field_a_per_m: float) -> float:
        finite_float("magnetic_field_a_per_m", magnetic_field_a_per_m)
        return self.permeability_h_per_m

    def secant_permeability_h_per_m(self, magnetic_field_a_per_m: float) -> float:
        finite_float("magnetic_field_a_per_m", magnetic_field_a_per_m)
        return self.permeability_h_per_m

    def coenergy_density_j_per_m3(self, magnetic_field_a_per_m: float) -> float:
        field = finite_float("magnetic_field_a_per_m", magnetic_field_a_per_m)
        return _finite_decimal_float(
            "computed coenergy density",
            _decimal(self.permeability_h_per_m) * _decimal(field) ** 2 / 2,
            allow_underflow_zero=True,
        )

    def energy_density_j_per_m3(self, flux_density_t: float) -> float:
        flux = finite_float("flux_density_t", flux_density_t)
        return _finite_decimal_float(
            "computed energy density",
            _decimal(flux) ** 2 / (2 * _decimal(self.permeability_h_per_m)),
            allow_underflow_zero=True,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "linear_isotropic",
            "material_id": self.material_id,
            "relative_permeability": self.relative_permeability,
            "permeability_h_per_m": self.permeability_h_per_m,
        }


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(value)


def _finite_decimal_float(
    name: str,
    value: Decimal,
    *,
    require_positive: bool = False,
    allow_underflow_zero: bool = False,
) -> float:
    converted = float(value)
    if not isfinite(converted):
        raise MagneticsValidationError(f"{name} is outside the representable float range")
    if value != 0 and converted == 0.0 and not allow_underflow_zero:
        raise MagneticsValidationError(f"{name} is below the representable float range")
    if require_positive and converted <= 0.0:
        raise MagneticsValidationError(f"{name} must be representably positive")
    return converted


def _pchip_derivatives_decimal(
    x: tuple[float, ...], y: tuple[float, ...]
) -> tuple[Decimal, ...]:
    """Positive shape-preserving knot derivatives without binary64 ratios."""

    with localcontext() as context:
        context.prec = 80
        xd = tuple(_decimal(value) for value in x)
        yd = tuple(_decimal(value) for value in y)
        widths = tuple(xd[index + 1] - xd[index] for index in range(len(x) - 1))
        rises = tuple(yd[index + 1] - yd[index] for index in range(len(y) - 1))
        secants = tuple(rise / width for rise, width in zip(rises, widths))
        if len(x) == 2:
            return (secants[0], secants[0])

        derivatives = [Decimal(0)] * len(x)
        for index in range(1, len(x) - 1):
            left_weight = 2 * widths[index] + widths[index - 1]
            right_weight = widths[index] + 2 * widths[index - 1]
            derivatives[index] = (left_weight + right_weight) / (
                left_weight / secants[index - 1]
                + right_weight / secants[index]
            )

        first = (
            (2 * widths[0] + widths[1]) * secants[0]
            - widths[0] * secants[1]
        ) / (widths[0] + widths[1])
        last = (
            (2 * widths[-1] + widths[-2]) * secants[-1]
            - widths[-1] * secants[-2]
        ) / (widths[-1] + widths[-2])
        positive_floor = Decimal(2) ** -52
        derivatives[0] = (
            min(secants[0], secants[1]) * positive_floor
            if first <= 0
            else min(first, 3 * secants[0])
        )
        derivatives[-1] = (
            min(secants[-1], secants[-2]) * positive_floor
            if last <= 0
            else min(last, 3 * secants[-1])
        )

        # A monotone cubic Bézier interval has positive derivative when its
        # normalized endpoint slopes are positive and sum to at most three.
        # Scaling only decreases shared knot slopes, so one forward pass also
        # preserves every interval already visited.
        for index, secant in enumerate(secants):
            alpha = derivatives[index] / secant
            beta = derivatives[index + 1] / secant
            total = alpha + beta
            if total > 3:
                scale = Decimal(3) / total
                derivatives[index] *= scale
                derivatives[index + 1] *= scale
        return tuple(+value for value in derivatives)


@dataclass(frozen=True, slots=True)
class TabulatedBHCurve:
    """Monotone PCHIP interpolation of an odd-symmetric, single-valued B-H law.

    Inputs contain the first quadrant including the origin. Strictly increasing
    ``H`` and ``B`` make inversion deterministic. Interval-normalized Hermite
    evaluation avoids raw dimensional cubic coefficients and cancellation.
    """

    material_id: str
    h_a_per_m: tuple[float, ...]
    b_t: tuple[float, ...]
    extrapolation: ExtrapolationPolicy = ExtrapolationPolicy.ERROR
    provenance: str = "unspecified"
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        nonempty_identifier("material_id", self.material_id)
        nonempty_identifier("provenance", self.provenance)
        if not isinstance(self.h_a_per_m, tuple) or not isinstance(self.b_t, tuple):
            raise MagneticsValidationError("B-H samples must be immutable tuples")
        if len(self.h_a_per_m) != len(self.b_t) or len(self.h_a_per_m) < 2:
            raise MagneticsValidationError("B-H tuples must have the same length >= 2")
        h_values = tuple(
            finite_float(f"h_a_per_m[{index}]", value)
            for index, value in enumerate(self.h_a_per_m)
        )
        b_values = tuple(
            finite_float(f"b_t[{index}]", value) for index, value in enumerate(self.b_t)
        )
        if h_values[0] != 0.0 or b_values[0] != 0.0:
            raise MagneticsValidationError("first-quadrant B-H data must start at (0, 0)")
        if any(right <= left for left, right in zip(h_values, h_values[1:])):
            raise MagneticsValidationError("H samples must be strictly increasing")
        if any(right <= left for left, right in zip(b_values, b_values[1:])):
            raise MagneticsValidationError("B samples must be strictly increasing")
        try:
            policy = ExtrapolationPolicy(self.extrapolation)
        except ValueError as error:
            raise MagneticsValidationError("unsupported B-H extrapolation policy") from error
        object.__setattr__(self, "h_a_per_m", h_values)
        object.__setattr__(self, "b_t", b_values)
        object.__setattr__(self, "extrapolation", policy)
        if not isinstance(self.is_synthetic, bool):
            raise MagneticsValidationError("is_synthetic must be boolean")
        _pchip_derivatives_decimal(h_values, b_values)

    @property
    def knot_derivatives_h_per_m(self) -> tuple[float, ...]:
        return tuple(
            _finite_decimal_float(
                f"PCHIP derivative[{index}]", derivative, require_positive=True
            )
            for index, derivative in enumerate(
                _pchip_derivatives_decimal(self.h_a_per_m, self.b_t)
            )
        )

    def _interval_tangents(self, index: int) -> tuple[Decimal, Decimal]:
        with localcontext() as context:
            context.prec = 80
            width = _decimal(self.h_a_per_m[index + 1]) - _decimal(
                self.h_a_per_m[index]
            )
            rise = _decimal(self.b_t[index + 1]) - _decimal(self.b_t[index])
            derivatives = _pchip_derivatives_decimal(self.h_a_per_m, self.b_t)
            return (
                +(derivatives[index] * width / rise),
                +(derivatives[index + 1] * width / rise),
            )

    @staticmethod
    def _normalized_value_and_derivative(
        normalized: Decimal, alpha: Decimal, beta: Decimal
    ) -> tuple[Decimal, Decimal]:
        """Evaluate monotone cubic Hermite value and derivative on ``[0,1]``."""

        with localcontext() as context:
            context.prec = 80
            one = Decimal(1)
            complement = one - normalized
            # Cubic Bézier/de Casteljau evaluation is stable near either end.
            p0 = Decimal(0)
            p1 = alpha / 3
            p2 = one - beta / 3
            p3 = one
            q0 = complement * p0 + normalized * p1
            q1 = complement * p1 + normalized * p2
            q2 = complement * p2 + normalized * p3
            r0 = complement * q0 + normalized * q1
            r1 = complement * q1 + normalized * q2
            value = complement * r0 + normalized * r1
            derivative = (
                alpha * complement * complement
                + 2 * (3 - alpha - beta) * complement * normalized
                + beta * normalized * normalized
            )
            return +value, +derivative

    def _locate_h(self, magnitude_h: float) -> tuple[int, Decimal]:
        index = min(
            bisect_right(self.h_a_per_m, magnitude_h) - 1,
            len(self.h_a_per_m) - 2,
        )
        with localcontext() as context:
            context.prec = 80
            lower = _decimal(self.h_a_per_m[index])
            width = _decimal(self.h_a_per_m[index + 1]) - lower
            return index, +((_decimal(magnitude_h) - lower) / width)

    def _segment_b_decimal(self, index: int, normalized: Decimal) -> Decimal:
        with localcontext() as context:
            context.prec = 80
            alpha, beta = self._interval_tangents(index)
            value, _ = self._normalized_value_and_derivative(
                normalized, alpha, beta
            )
            lower = _decimal(self.b_t[index])
            rise = _decimal(self.b_t[index + 1]) - lower
            return +(lower + rise * value)

    def _terminal_derivative_decimal(self) -> Decimal:
        return _pchip_derivatives_decimal(self.h_a_per_m, self.b_t)[-1]

    def _check_range(self, magnitude_h: float) -> None:
        if (
            magnitude_h > self.h_a_per_m[-1]
            and self.extrapolation is ExtrapolationPolicy.ERROR
        ):
            raise MagneticsValidationError(
                f"|H|={magnitude_h} A/m exceeds tabulated maximum "
                f"{self.h_a_per_m[-1]} A/m"
            )

    def b_from_h_t(self, magnetic_field_a_per_m: float) -> float:
        field = finite_float("magnetic_field_a_per_m", magnetic_field_a_per_m)
        magnitude = abs(field)
        self._check_range(magnitude)
        if magnitude > self.h_a_per_m[-1]:
            with localcontext() as context:
                context.prec = 80
                result_decimal = _decimal(
                    self.b_t[-1]
                ) + self._terminal_derivative_decimal() * (
                    _decimal(magnitude) - _decimal(self.h_a_per_m[-1])
                )
        else:
            index, normalized = self._locate_h(magnitude)
            result_decimal = self._segment_b_decimal(index, normalized)
        result = _finite_decimal_float("interpolated flux density", result_decimal)
        signed_result = result if field >= 0.0 else -result
        return finite_float("interpolated flux density", signed_result)

    def differential_permeability_h_per_m(self, magnetic_field_a_per_m: float) -> float:
        field = finite_float("magnetic_field_a_per_m", magnetic_field_a_per_m)
        magnitude = abs(field)
        self._check_range(magnitude)
        if magnitude > self.h_a_per_m[-1]:
            derivative = self._terminal_derivative_decimal()
        else:
            index, normalized = self._locate_h(magnitude)
            alpha, beta = self._interval_tangents(index)
            _, normalized_derivative = self._normalized_value_and_derivative(
                normalized, alpha, beta
            )
            with localcontext() as context:
                context.prec = 80
                rise = _decimal(self.b_t[index + 1]) - _decimal(self.b_t[index])
                width = _decimal(self.h_a_per_m[index + 1]) - _decimal(
                    self.h_a_per_m[index]
                )
                derivative = +(rise * normalized_derivative / width)
        return _finite_decimal_float(
            "differential_permeability_h_per_m",
            derivative,
            require_positive=True,
        )

    def secant_permeability_h_per_m(self, magnetic_field_a_per_m: float) -> float:
        field = finite_float("magnetic_field_a_per_m", magnetic_field_a_per_m)
        if field == 0.0:
            return _finite_decimal_float(
                "secant_permeability_h_per_m",
                _pchip_derivatives_decimal(self.h_a_per_m, self.b_t)[0],
                require_positive=True,
            )
        with localcontext() as context:
            context.prec = 80
            ratio = _decimal(abs(self.b_from_h_t(field))) / _decimal(abs(field))
        return _finite_decimal_float(
            "secant_permeability_h_per_m",
            ratio,
            require_positive=True,
        )

    def h_from_b_a_per_m(self, flux_density_t: float) -> float:
        flux = finite_float("flux_density_t", flux_density_t)
        magnitude = abs(flux)
        if magnitude > self.b_t[-1]:
            if self.extrapolation is ExtrapolationPolicy.ERROR:
                raise MagneticsValidationError(
                    f"|B|={magnitude} T exceeds tabulated maximum {self.b_t[-1]} T"
                )
            with localcontext() as context:
                context.prec = 80
                field_decimal = _decimal(
                    self.h_a_per_m[-1]
                ) + (_decimal(magnitude) - _decimal(self.b_t[-1])) / (
                    self._terminal_derivative_decimal()
                )
            field = _finite_decimal_float("extrapolated magnetic field", field_decimal)
            signed_field = field if flux >= 0.0 else -field
            return finite_float("extrapolated magnetic field", signed_field)
        if magnitude == 0.0:
            return 0.0
        if magnitude in self.b_t:
            field = self.h_a_per_m[self.b_t.index(magnitude)]
            return field if flux >= 0.0 else -field

        index = bisect_right(self.b_t, magnitude) - 1
        with localcontext() as context:
            context.prec = 160
            b_lower = _decimal(self.b_t[index])
            b_rise = _decimal(self.b_t[index + 1]) - b_lower
            alpha, beta = self._interval_tangents(index)
            h_lower = _decimal(self.h_a_per_m[index])
            h_width = _decimal(self.h_a_per_m[index + 1]) - h_lower
            flux_offset = _decimal(magnitude) - b_lower
            if alpha == 1 and beta == 1:
                # Exact local linear inversion avoids forming a normalized
                # target that lies below binary64's minimum subnormal.
                field_decimal = h_lower + h_width * flux_offset / b_rise
            else:
                target = flux_offset / b_rise
                lower = Decimal(0)
                upper = Decimal(1)
                normalized = target
                best = normalized
                best_residual = Decimal("Infinity")
                for _ in range(256):
                    value, derivative = self._normalized_value_and_derivative(
                        normalized, alpha, beta
                    )
                    residual = value - target
                    residual_magnitude = abs(residual)
                    if residual_magnitude < best_residual:
                        best = normalized
                        best_residual = residual_magnitude
                    if residual == 0:
                        best = normalized
                        break
                    if residual < 0:
                        lower = normalized
                    else:
                        upper = normalized
                    physical_lower = float(h_lower + h_width * lower)
                    physical_upper = float(h_lower + h_width * upper)
                    if (
                        physical_lower == physical_upper
                        or nextafter(physical_lower, physical_upper)
                        >= physical_upper
                    ):
                        break
                    candidate = normalized - residual / derivative
                    normalized = (
                        candidate
                        if lower < candidate < upper
                        else (lower + upper) / 2
                    )
                field_decimal = h_lower + h_width * best
        field = _finite_decimal_float("inverted magnetic field", field_decimal)
        signed_field = field if flux >= 0.0 else -field
        return finite_float("inverted magnetic field", signed_field)

    @staticmethod
    def _normalized_integral(
        normalized: Decimal, alpha: Decimal, beta: Decimal
    ) -> Decimal:
        with localcontext() as context:
            context.prec = 80
            cubic = -2 + alpha + beta
            quadratic = 3 - 2 * alpha - beta
            return +(
                cubic * normalized**4 / 4
                + quadratic * normalized**3 / 3
                + alpha * normalized**2 / 2
            )

    def _energies_at_h(self, magnitude: float) -> tuple[Decimal, Decimal]:
        """Integrate ``B dH`` and ``H dB`` directly as positive quantities."""

        self._check_range(magnitude)
        with localcontext() as context:
            context.prec = 80
            coenergy = Decimal(0)
            energy = Decimal(0)
            tabulated_limit = min(magnitude, self.h_a_per_m[-1])
            if tabulated_limit > 0.0:
                final_index, final_t = self._locate_h(tabulated_limit)
                for index in range(final_index + 1):
                    t = final_t if index == final_index else Decimal(1)
                    alpha, beta = self._interval_tangents(index)
                    q, _ = self._normalized_value_and_derivative(t, alpha, beta)
                    q_integral = self._normalized_integral(t, alpha, beta)
                    h0 = _decimal(self.h_a_per_m[index])
                    width = _decimal(self.h_a_per_m[index + 1]) - h0
                    b0 = _decimal(self.b_t[index])
                    rise = _decimal(self.b_t[index + 1]) - b0
                    coenergy += width * (b0 * t + rise * q_integral)
                    energy += h0 * rise * q + width * rise * (
                        t * q - q_integral
                    )
            if magnitude > self.h_a_per_m[-1]:
                h0 = _decimal(self.h_a_per_m[-1])
                delta = _decimal(magnitude) - h0
                tangent = self._terminal_derivative_decimal()
                coenergy += _decimal(self.b_t[-1]) * delta
                coenergy += tangent * delta * delta / 2
                energy += tangent * (2 * h0 * delta + delta * delta) / 2
            if coenergy < 0 or energy < 0:
                raise MagneticsValidationError(
                    "normalized constitutive integration produced negative energy"
                )
            return +energy, +coenergy

    def coenergy_density_j_per_m3(self, magnetic_field_a_per_m: float) -> float:
        field = finite_float("magnetic_field_a_per_m", magnetic_field_a_per_m)
        _, coenergy = self._energies_at_h(abs(field))
        return _finite_decimal_float(
            "computed coenergy density",
            coenergy,
            allow_underflow_zero=True,
        )

    def energy_density_j_per_m3(self, flux_density_t: float) -> float:
        flux = finite_float("flux_density_t", flux_density_t)
        field = self.h_from_b_a_per_m(flux)
        energy, _ = self._energies_at_h(abs(field))
        return _finite_decimal_float(
            "computed energy density",
            energy,
            allow_underflow_zero=True,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "tabulated_odd_symmetric_single_valued_b_h",
            "material_id": self.material_id,
            "h_a_per_m": self.h_a_per_m,
            "b_t": self.b_t,
            "interpolation": "monotone_pchip_fritsch_carlson",
            "extrapolation": self.extrapolation.value,
            "provenance": self.provenance,
            "is_synthetic": self.is_synthetic,
            "hysteresis": "out_of_scope",
        }


@dataclass(frozen=True, slots=True)
class SmCoPermanentMagnet:
    """SmCo-like recoil-line parameters with explicit temperature validity.

    Values identify a supplied dataset, not a grade name. Linear temperature
    coefficients and a constant recoil permeability are valid only over the
    declared closed interval.
    """

    material_id: str
    remanence_ref_t: float
    intrinsic_coercivity_ref_a_per_m: float
    recoil_relative_permeability: float
    reference_temperature_k: float
    remanence_temp_coefficient_per_k: float
    coercivity_temp_coefficient_per_k: float
    valid_temperature_min_k: float
    valid_temperature_max_k: float
    provenance: str
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        nonempty_identifier("material_id", self.material_id)
        nonempty_identifier("provenance", self.provenance)
        if not isinstance(self.is_synthetic, bool):
            raise MagneticsValidationError("is_synthetic must be boolean")
        names = (
            "remanence_ref_t",
            "intrinsic_coercivity_ref_a_per_m",
            "recoil_relative_permeability",
            "reference_temperature_k",
            "remanence_temp_coefficient_per_k",
            "coercivity_temp_coefficient_per_k",
            "valid_temperature_min_k",
            "valid_temperature_max_k",
        )
        values = {name: finite_float(name, getattr(self, name)) for name in names}
        if values["remanence_ref_t"] <= 0.0:
            raise MagneticsValidationError("remanence_ref_t must be positive")
        if values["intrinsic_coercivity_ref_a_per_m"] <= 0.0:
            raise MagneticsValidationError(
                "intrinsic_coercivity_ref_a_per_m must be positive"
            )
        if values["recoil_relative_permeability"] <= 0.0:
            raise MagneticsValidationError("recoil_relative_permeability must be positive")
        if values["valid_temperature_min_k"] <= 0.0:
            raise MagneticsValidationError("valid temperatures must be above absolute zero")
        if values["valid_temperature_max_k"] <= values["valid_temperature_min_k"]:
            raise MagneticsValidationError("temperature validity interval must be positive")
        if not (
            values["valid_temperature_min_k"]
            <= values["reference_temperature_k"]
            <= values["valid_temperature_max_k"]
        ):
            raise MagneticsValidationError(
                "reference_temperature_k must lie in the validity interval"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self._temperature_adjusted(
            values["valid_temperature_min_k"],
            "valid_temperature_min_k",
        )
        self._temperature_adjusted(
            values["valid_temperature_max_k"],
            "valid_temperature_max_k",
        )

    def _validated_temperature(self, temperature_k: float) -> float:
        temperature = finite_float("temperature_k", temperature_k)
        if not self.valid_temperature_min_k <= temperature <= self.valid_temperature_max_k:
            raise MagneticsValidationError(
                f"temperature {temperature} K is outside the declared validity interval "
                f"[{self.valid_temperature_min_k}, {self.valid_temperature_max_k}] K"
            )
        return temperature

    def _temperature_adjusted(self, temperature_k: float, name: str) -> tuple[float, float]:
        temperature = finite_float(name, temperature_k)
        delta = temperature - self.reference_temperature_k
        remanence = finite_float(
            "temperature-adjusted remanence",
            self.remanence_ref_t
            * (1.0 + self.remanence_temp_coefficient_per_k * delta),
        )
        coercivity = finite_float(
            "temperature-adjusted intrinsic coercivity",
            self.intrinsic_coercivity_ref_a_per_m
            * (1.0 + self.coercivity_temp_coefficient_per_k * delta),
        )
        if remanence <= 0.0 or coercivity <= 0.0:
            raise MagneticsValidationError(
                "temperature coefficients produce non-positive remanence/coercivity"
            )
        return remanence, coercivity

    def remanence_t(self, temperature_k: float) -> float:
        temperature = self._validated_temperature(temperature_k)
        return self._temperature_adjusted(temperature, "temperature_k")[0]

    def intrinsic_coercivity_a_per_m(self, temperature_k: float) -> float:
        temperature = self._validated_temperature(temperature_k)
        return self._temperature_adjusted(temperature, "temperature_k")[1]

    def magnetization_a_per_m(
        self, temperature_k: float, direction: VectorRZ
    ) -> VectorRZ:
        unit = direction.normalized()
        return unit.scaled(self.remanence_t(temperature_k) / MU0_H_PER_M)

    def recoil_b_parallel_t(
        self, magnetic_field_parallel_a_per_m: float, temperature_k: float
    ) -> float:
        field = finite_float(
            "magnetic_field_parallel_a_per_m", magnetic_field_parallel_a_per_m
        )
        return finite_float(
            "recoil flux density",
            self.remanence_t(temperature_k)
            + MU0_H_PER_M * self.recoil_relative_permeability * field,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "smco_like_linear_recoil_permanent_magnet",
            "material_id": self.material_id,
            "remanence_ref_t": self.remanence_ref_t,
            "intrinsic_coercivity_ref_a_per_m": self.intrinsic_coercivity_ref_a_per_m,
            "recoil_relative_permeability": self.recoil_relative_permeability,
            "reference_temperature_k": self.reference_temperature_k,
            "remanence_temp_coefficient_per_k": self.remanence_temp_coefficient_per_k,
            "coercivity_temp_coefficient_per_k": self.coercivity_temp_coefficient_per_k,
            "valid_temperature_min_k": self.valid_temperature_min_k,
            "valid_temperature_max_k": self.valid_temperature_max_k,
            "provenance": self.provenance,
            "is_synthetic": self.is_synthetic,
        }


def checked_synthetic_soft_magnetic_curve() -> TabulatedBHCurve:
    """Return a checked numerical example, explicitly not measured material data."""

    return TabulatedBHCurve(
        material_id="synthetic-soft-magnetic-example-v1",
        h_a_per_m=(0.0, 100.0, 300.0, 1_000.0, 3_000.0, 10_000.0, 30_000.0, 100_000.0),
        b_t=(0.0, 0.08, 0.22, 0.58, 0.92, 1.28, 1.40, 1.66),
        extrapolation=ExtrapolationPolicy.LINEAR_TANGENT,
        provenance=(
            "Synthetic checked example authored for algorithm verification; "
            "not measured and not representative of a named grade."
        ),
        is_synthetic=True,
    )


def checked_synthetic_smco_like_magnet() -> SmCoPermanentMagnet:
    """Return a plausible synthetic example, not a redistributable grade dataset."""

    return SmCoPermanentMagnet(
        material_id="synthetic-smco-like-example-v1",
        remanence_ref_t=1.05,
        intrinsic_coercivity_ref_a_per_m=1_600_000.0,
        recoil_relative_permeability=1.05,
        reference_temperature_k=293.15,
        remanence_temp_coefficient_per_k=-3.0e-4,
        coercivity_temp_coefficient_per_k=-2.0e-3,
        valid_temperature_min_k=253.15,
        valid_temperature_max_k=473.15,
        provenance=(
            "Synthetic SmCo-like checked example using plausible magnitudes only; "
            "not a vendor grade or qualification dataset."
        ),
        is_synthetic=True,
    )
