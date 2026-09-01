"""Conservative 1-D electrostatic PIC kernels and a verified Poisson interface."""

from __future__ import annotations

from dataclasses import dataclass
from math import frexp, fsum, isclose, isfinite, ldexp, sqrt
import sys
from typing import Protocol, Sequence, TypeVar

from .models import (
    EPSILON_0_F_PER_M,
    Grid1D,
    PICConvergenceError,
    PICValidationError,
    ParticleState,
    PoissonConfig,
    Species,
)


@dataclass(frozen=True, slots=True)
class PoissonDiagnostics:
    converged: bool
    iterations: int
    initial_residual_l2: float
    final_residual_l2: float
    required_residual_l2: float


@dataclass(frozen=True, slots=True)
class ElectrostaticField:
    potential_v: tuple[float, ...]
    electric_field_face_v_per_m: tuple[float, ...]
    diagnostics: PoissonDiagnostics
    removed_mean_charge_density_c_per_m3: float
    backend: str = "python"

    @property
    def electric_field_v_per_m(self) -> tuple[float, ...]:
        """Compatibility alias; values are face-centred, not nodal."""

        return self.electric_field_face_v_per_m


GridT = TypeVar("GridT")


class PoissonSolver(Protocol[GridT]):
    """Generic solver boundary for future axisymmetric and WarpX grids."""

    def solve(
        self,
        grid: GridT,
        charge_density_c_per_m3: Sequence[float],
        config: PoissonConfig,
        *,
        raise_on_nonconvergence: bool = True,
    ) -> ElectrostaticField: ...


def _finite_product_ratio(
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    context: str,
    require_nonzero: bool,
) -> float:
    """Evaluate a short product/ratio with one final binary64 exponent scaling."""

    mantissa = 1.0
    exponent = 0
    sign = 1.0
    for raw in numerators:
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise PICValidationError(f"{context} numerator must be numeric") from error
        if not isfinite(value):
            raise PICValidationError(f"{context} numerator must be finite")
        if value == 0.0:
            if require_nonzero:
                raise PICValidationError(f"{context} unexpectedly has zero numerator")
            return 0.0
        if value < 0.0:
            sign = -sign
        part, part_exponent = frexp(abs(value))
        mantissa *= part
        mantissa, normalization = frexp(mantissa)
        exponent += part_exponent + normalization
    for raw in denominators:
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise PICValidationError(f"{context} denominator must be numeric") from error
        if not isfinite(value) or value <= 0.0:
            raise PICValidationError(f"{context} denominator must be finite and positive")
        part, part_exponent = frexp(value)
        mantissa /= part
        mantissa, normalization = frexp(mantissa)
        exponent += normalization - part_exponent
    try:
        result = ldexp(sign * mantissa, exponent)
    except OverflowError as error:
        raise PICValidationError(f"{context} is not representable") from error
    if not isfinite(result) or (require_nonzero and result == 0.0):
        raise PICValidationError(f"{context} is not representable")
    return result


def represented_charge_c(species: Species, particle_count: int) -> float:
    if isinstance(particle_count, bool) or not isinstance(particle_count, int):
        raise PICValidationError("particle_count must be an integer")
    if particle_count < 1:
        raise PICValidationError("particle_count must be positive")
    try:
        count = float(particle_count)
    except OverflowError as error:
        raise PICValidationError("particle_count is not representable") from error
    return _finite_product_ratio(
        (species.charge_c, species.macro_weight, count),
        (),
        context="represented particle charge",
        require_nonzero=True,
    )


def charge_density_per_particle_c_per_m3(
    grid: Grid1D, species: Species
) -> float:
    return _finite_product_ratio(
        (species.charge_c, species.macro_weight),
        (grid.dx_m, grid.transverse_area_m2),
        context="volumetric particle charge density",
        require_nonzero=True,
    )


def physical_number_density_per_m3(
    grid: Grid1D, species: Species, particle_count: int
) -> float:
    if isinstance(particle_count, bool) or not isinstance(particle_count, int):
        raise PICValidationError("particle_count must be an integer")
    if particle_count < 1:
        raise PICValidationError("particle_count must be positive")
    try:
        count = float(particle_count)
    except OverflowError as error:
        raise PICValidationError("particle_count is not representable") from error
    return _finite_product_ratio(
        (species.macro_weight, count),
        (grid.length_m, grid.transverse_area_m2),
        context="physical particle number density",
        require_nonzero=True,
    )


def integrated_charge_c(grid: Grid1D, charge_density: Sequence[float]) -> float:
    if len(charge_density) != grid.cells:
        raise PICValidationError("charge-density length must equal grid.cells")
    try:
        density_sum = fsum(float(value) for value in charge_density)
    except (TypeError, ValueError, OverflowError) as error:
        raise PICValidationError("charge-density sum is not representable") from error
    return _finite_product_ratio(
        (density_sum, grid.dx_m, grid.transverse_area_m2),
        (),
        context="integrated deposited charge",
        require_nonzero=density_sum != 0.0,
    )


def cic_deposit_charge(
    grid: Grid1D,
    species: Species,
    particles: ParticleState,
) -> tuple[float, ...]:
    """Deposit macro-particle charge with periodic cloud-in-cell weighting.

    Node control volumes are ``dx * transverse_area_m2``. The volume integral
    of the returned density therefore equals represented particle charge.
    """

    particles.validate()
    density = [0.0] * grid.cells
    charge_over_volume = charge_density_per_particle_c_per_m3(grid, species)
    for position in particles.x_m:
        coordinate = (grid.wrap(position) - grid.x_min_m) / grid.dx_m
        left = int(coordinate) % grid.cells
        fraction = coordinate - int(coordinate)
        density[left] += charge_over_volume * (1.0 - fraction)
        density[(left + 1) % grid.cells] += charge_over_volume * fraction
        if not isfinite(density[left]) or not isfinite(
            density[(left + 1) % grid.cells]
        ):
            raise PICValidationError("deposited charge density overflowed")
    deposited = tuple(density)
    represented = represented_charge_c(species, particles.count)
    integrated = integrated_charge_c(grid, deposited)
    if not isclose(
        represented,
        integrated,
        rel_tol=32.0 * sys.float_info.epsilon,
        abs_tol=0.0,
    ):
        raise PICValidationError("CPU deposition did not conserve represented charge")
    return deposited


def add_uniform_background(
    charge_density_c_per_m3: Sequence[float],
    background_charge_density_c_per_m3: float,
) -> tuple[float, ...]:
    background = float(background_charge_density_c_per_m3)
    if not isfinite(background):
        raise PICValidationError("background charge density must be finite")
    values = tuple(float(value) + background for value in charge_density_c_per_m3)
    if not values or any(not isfinite(value) for value in values):
        raise PICValidationError("charge-density array must be nonempty and finite")
    return values


def gather_cic(
    grid: Grid1D,
    node_values: Sequence[float],
    positions_m: Sequence[float],
) -> tuple[float, ...]:
    """Interpolate periodic node values to particles using the deposition shape."""

    if len(node_values) != grid.cells:
        raise PICValidationError("node field length must equal grid.cells")
    if any(not isfinite(float(value)) for value in node_values):
        raise PICValidationError("node field must contain only finite values")
    gathered: list[float] = []
    for position in positions_m:
        converted = float(position)
        if not isfinite(converted):
            raise PICValidationError("particle positions must be finite")
        coordinate = (grid.wrap(converted) - grid.x_min_m) / grid.dx_m
        left = int(coordinate) % grid.cells
        fraction = coordinate - int(coordinate)
        value = (
            (1.0 - fraction) * float(node_values[left])
            + fraction * float(node_values[(left + 1) % grid.cells])
        )
        if not isfinite(value):
            raise PICValidationError("nodal gather produced a nonfinite value")
        gathered.append(value)
    return tuple(gathered)


def gather_face_cic(
    grid: Grid1D,
    face_values: Sequence[float],
    positions_m: Sequence[float],
) -> tuple[float, ...]:
    """Symmetrically average faces to nodes, then apply nodal CIC gather.

    The face field retains every Poisson mode for energy/Gauss diagnostics.
    Symmetric face-to-node reconstruction followed by the deposition shape
    preserves the standard periodic single-particle zero-self-force property.
    """

    if len(face_values) != grid.cells:
        raise PICValidationError("face field length must equal grid.cells")
    values = tuple(float(value) for value in face_values)
    if any(not isfinite(value) for value in values):
        raise PICValidationError("face field must contain only finite values")
    nodal = tuple(
        0.5 * (values[(index - 1) % grid.cells] + values[index])
        for index in range(grid.cells)
    )
    if any(not isfinite(value) for value in nodal):
        raise PICValidationError("face-to-node reconstruction became nonfinite")
    return gather_cic(grid, nodal, positions_m)


def _apply_negative_laplacian(grid: Grid1D, values: Sequence[float]) -> list[float]:
    output: list[float] = []
    for i in range(grid.cells):
        value = (
            (
                values[i] - values[(i - 1) % grid.cells]
            )
            - (
                values[(i + 1) % grid.cells] - values[i]
            )
        ) / grid.dx_m / grid.dx_m
        if not isfinite(value):
            raise PICConvergenceError("Poisson operator produced a nonfinite value")
        output.append(value)
    return output


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    value = fsum(a * b for a, b in zip(left, right, strict=True))
    if not isfinite(value):
        raise PICConvergenceError("Poisson inner product is not representable")
    return value


def scaled_l2(values: Sequence[float], *, context: str) -> float:
    """Overflow-safe binary64 L2 norm with a finite publication contract."""

    if not values:
        raise PICValidationError(f"{context} must not be empty")
    converted = tuple(float(value) for value in values)
    if any(not isfinite(value) for value in converted):
        raise PICConvergenceError(f"{context} contains a nonfinite value")
    scale = max(abs(value) for value in converted)
    if scale == 0.0:
        return 0.0
    norm = scale * sqrt(fsum((value / scale) ** 2 for value in converted))
    if not isfinite(norm):
        raise PICConvergenceError(f"{context} L2 norm is not representable")
    return norm


def _scaled_mean(values: Sequence[float], *, context: str) -> float:
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    mean = scale * fsum(value / scale for value in values) / len(values)
    if not isfinite(mean):
        raise PICConvergenceError(f"{context} mean is not representable")
    return mean


class PeriodicPoisson1D:
    """Mean-zero finite-difference periodic Poisson solve using conjugate gradients."""

    def solve(
        self,
        grid: Grid1D,
        charge_density_c_per_m3: Sequence[float],
        config: PoissonConfig = PoissonConfig(),
        *,
        raise_on_nonconvergence: bool = True,
    ) -> ElectrostaticField:
        if len(charge_density_c_per_m3) != grid.cells:
            raise PICValidationError("charge-density length must equal grid.cells")
        rho = tuple(float(value) for value in charge_density_c_per_m3)
        if any(not isfinite(value) for value in rho):
            raise PICValidationError("charge density must contain only finite values")

        mean_rho = _scaled_mean(rho, context="charge density")
        right_hand_side: list[float] = []
        for value in rho:
            source = value - mean_rho
            rhs_value = source / EPSILON_0_F_PER_M
            if not isfinite(source) or not isfinite(rhs_value):
                raise PICValidationError(
                    "charge source is not representable after neutralization"
                )
            right_hand_side.append(rhs_value)
        rhs_scale = max(abs(value) for value in right_hand_side)
        normalized_rhs = (
            [0.0] * grid.cells
            if rhs_scale == 0.0
            else [value / rhs_scale for value in right_hand_side]
        )
        potential_normalized = [0.0] * grid.cells
        residual = normalized_rhs.copy()
        direction = residual.copy()
        residual_squared = _dot(residual, residual)
        initial = scaled_l2(right_hand_side, context="initial Poisson residual")
        required = max(config.absolute_tolerance, config.relative_tolerance * initial)
        if not isfinite(required):
            raise PICValidationError("Poisson residual tolerance is not representable")
        iterations = 0
        converged = initial <= required

        for iteration in range(1, config.max_iterations + 1):
            if converged:
                break
            operator_direction = _apply_negative_laplacian(grid, direction)
            denominator = _dot(direction, operator_direction)
            if not isfinite(denominator) or denominator <= 0.0:
                raise PICConvergenceError("periodic Poisson CG lost positive definiteness")
            alpha = residual_squared / denominator
            if not isfinite(alpha):
                raise PICConvergenceError("Poisson CG alpha is not representable")
            for i in range(grid.cells):
                potential_normalized[i] += alpha * direction[i]
                residual[i] -= alpha * operator_direction[i]
                if not isfinite(potential_normalized[i]) or not isfinite(residual[i]):
                    raise PICConvergenceError("Poisson CG iterate became nonfinite")
            new_residual_squared = _dot(residual, residual)
            iterations = iteration
            normalized_final = scaled_l2(
                residual, context="recursive normalized Poisson residual"
            )
            physical_final = rhs_scale * normalized_final
            if not isfinite(physical_final):
                raise PICConvergenceError(
                    "recursive physical Poisson residual is not representable"
                )
            if physical_final <= required:
                residual_squared = new_residual_squared
                converged = True
                break
            beta = new_residual_squared / residual_squared
            if not isfinite(beta):
                raise PICConvergenceError("Poisson CG beta is not representable")
            for i in range(grid.cells):
                direction[i] = residual[i] + beta * direction[i]
                if not isfinite(direction[i]):
                    raise PICConvergenceError("Poisson CG direction became nonfinite")
            residual_squared = new_residual_squared

        potential = [value * rhs_scale for value in potential_normalized]
        if any(not isfinite(value) for value in potential):
            raise PICConvergenceError("Poisson potential is not representable")
        mean_potential = _scaled_mean(potential, context="Poisson potential")
        potential = [value - mean_potential for value in potential]
        if any(not isfinite(value) for value in potential):
            raise PICConvergenceError("gauge-corrected Poisson potential is nonfinite")
        true_residual = [
            right_hand_side[i] - value
            for i, value in enumerate(_apply_negative_laplacian(grid, potential))
        ]
        if any(not isfinite(value) for value in true_residual):
            raise PICConvergenceError("true Poisson residual contains a nonfinite value")
        final = scaled_l2(true_residual, context="true Poisson residual")
        converged = bool(
            converged
            and isfinite(initial)
            and isfinite(final)
            and isfinite(required)
            and final <= required
        )
        diagnostics = PoissonDiagnostics(converged, iterations, initial, final, required)
        if not converged and raise_on_nonconvergence:
            raise PICConvergenceError(
                f"periodic Poisson CG did not converge in {iterations} iterations: "
                f"true residual {final:.6e}, required <= {required:.6e}"
            )

        electric = tuple(
            -(potential[(i + 1) % grid.cells] - potential[i]) / grid.dx_m
            for i in range(grid.cells)
        )
        if any(not isfinite(value) for value in (*potential, *electric)):
            raise PICConvergenceError("Poisson solve produced a nonfinite field")
        return ElectrostaticField(
            tuple(potential),
            electric,
            diagnostics,
            mean_rho,
        )
