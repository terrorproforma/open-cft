"""Typed contracts for the L1a axisymmetric vacuum-field solver.

All public geometry values are SI.  The source bands represent an azimuthal
volume-current density obtained by smearing ampere-turns over an ``(r, z)``
cross-section; they are not permanent-magnet material models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, ulp

MU0_H_PER_M = 1.2566370614359173e-6


class FieldError(Exception):
    """Base error for the independent field-solver workstream."""


class FieldValidationError(FieldError, ValueError):
    """An input violates the documented L1a domain contract."""


class FieldArtifactValidationError(FieldValidationError):
    """A serialized L1a artifact or manifest violates its closed contract."""


class FieldConvergenceError(FieldError, RuntimeError):
    """The iterative solve did not meet its residual contract."""


class FieldDeviceError(FieldError, RuntimeError):
    """The requested execution backend is unavailable."""


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise FieldValidationError(f"{name} must be a finite real number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FieldValidationError(f"{name} must be a finite real number") from error
    if not isfinite(converted):
        raise FieldValidationError(f"{name} must be finite")
    return converted


def resolved_span_in_grid_spacings(
    lower: float, upper: float, spacing: float
) -> float:
    """Return the represented geometric span measured in grid spacings."""

    return (upper - lower) / spacing


def span_meets_minimum_grid_spacings(
    lower: float,
    upper: float,
    spacing: float,
    minimum_spacings: float = 2.0,
) -> bool:
    """Compare geometric width with a half-ULP endpoint uncertainty envelope.

    Decimal-looking endpoints such as 0.04 and 0.06 can subtract one binary64
    rounding step below 2*0.01.  The allowance is the summed half-ULP
    uncertainty of both endpoints and the represented target; moving either
    endpoint one meaningful representable step narrower falls outside it.
    """

    width = upper - lower
    target = minimum_spacings * spacing
    allowance = 0.5 * (ulp(lower) + ulp(upper) + ulp(target))
    return target - width <= allowance


@dataclass(frozen=True, slots=True)
class AxisymmetricDomain:
    """Closed meridional half-plane box, in metres."""

    radius_m: float
    z_min_m: float
    z_max_m: float
    radial_intervals: int
    axial_intervals: int
    _dr_m: float = field(init=False, repr=False)
    _dz_m: float = field(init=False, repr=False)
    _z_span_m: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        radius = _finite("radius_m", self.radius_m)
        z_min = _finite("z_min_m", self.z_min_m)
        z_max = _finite("z_max_m", self.z_max_m)
        for name, count in (
            ("radial_intervals", self.radial_intervals),
            ("axial_intervals", self.axial_intervals),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 4:
                raise FieldValidationError(f"{name} must be an integer >= 4")
            if count > 1_000_000:
                raise FieldValidationError(f"{name} exceeds the supported 1,000,000 limit")
        if radius <= 0.0 or z_max <= z_min:
            raise FieldValidationError("domain radius and axial span must be positive")
        z_span = z_max - z_min
        if not isfinite(z_span) or z_span <= 0.0:
            raise FieldValidationError("derived axial span must be finite and positive")
        dr = radius / self.radial_intervals
        dz = z_span / self.axial_intervals
        for name, spacing in (("dr_m", dr), ("dz_m", dz)):
            if not isfinite(spacing) or spacing <= 0.0:
                raise FieldValidationError(f"derived {name} must be finite and positive")
            spacing_squared = spacing * spacing
            if not isfinite(spacing_squared) or spacing_squared <= 0.0:
                raise FieldValidationError(
                    f"derived {name} squared must be finite and positive"
                )
            if not isfinite(1.0 / spacing):
                raise FieldValidationError(f"derived inverse {name} must be finite")
        for name, denominator in (
            ("dual-cell area", dr * dz),
            ("radial operator scale", dr * dr * dr),
            ("axial operator scale", dr * dz * dz),
        ):
            if (
                not isfinite(denominator)
                or denominator <= 0.0
                or not isfinite(1.0 / denominator)
            ):
                raise FieldValidationError(
                    f"derived {name} must remain finite and nonzero in binary64"
                )
        if z_min + dz <= z_min or z_max - dz >= z_max:
            raise FieldValidationError(
                "derived axial coordinates collapse at binary64 precision"
            )
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "z_min_m", z_min)
        object.__setattr__(self, "z_max_m", z_max)
        object.__setattr__(self, "_dr_m", dr)
        object.__setattr__(self, "_dz_m", dz)
        object.__setattr__(self, "_z_span_m", z_span)

    @property
    def dr_m(self) -> float:
        return self._dr_m

    @property
    def dz_m(self) -> float:
        return self._dz_m

    @property
    def z_span_m(self) -> float:
        return self._z_span_m

    @property
    def shape(self) -> tuple[int, int]:
        return (self.radial_intervals + 1, self.axial_intervals + 1)


@dataclass(frozen=True, slots=True)
class AzimuthalCurrentBand:
    """Rectangular equivalent coil cross-section.

    ``ampere_turns_a`` is signed only through ``polarity``.  The represented
    volume-current density is
    ``J_phi = polarity * ampere_turns_a / ((r1-r0)*(z1-z0))`` in A/m².
    """

    name: str
    r_inner_m: float
    r_outer_m: float
    z_min_m: float
    z_max_m: float
    ampere_turns_a: float
    polarity: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise FieldValidationError("source name must not be empty")
        values = {
            name: _finite(name, getattr(self, name))
            for name in (
                "r_inner_m",
                "r_outer_m",
                "z_min_m",
                "z_max_m",
                "ampere_turns_a",
            )
        }
        if values["r_inner_m"] <= 0.0:
            raise FieldValidationError("source bands must not touch the symmetry axis")
        if values["r_outer_m"] <= values["r_inner_m"]:
            raise FieldValidationError("source radial thickness must be positive")
        if values["z_max_m"] <= values["z_min_m"]:
            raise FieldValidationError("source axial thickness must be positive")
        if values["ampere_turns_a"] < 0.0:
            raise FieldValidationError("ampere_turns_a must be non-negative")
        if isinstance(self.polarity, bool) or self.polarity not in (-1, 1):
            raise FieldValidationError("polarity must be exactly -1 or +1")
        area = (values["r_outer_m"] - values["r_inner_m"]) * (
            values["z_max_m"] - values["z_min_m"]
        )
        if not isfinite(area) or area <= 0.0:
            raise FieldValidationError("source cross-sectional area must be finite and positive")
        density = values["ampere_turns_a"] / area
        if not isfinite(density):
            raise FieldValidationError("source current density must be finite")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def current_density_a_per_m2(self) -> float:
        area = (self.r_outer_m - self.r_inner_m) * (self.z_max_m - self.z_min_m)
        return self.polarity * self.ampere_turns_a / area


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Convergence and publication policy for matrix-free Jacobi PCG."""

    relative_tolerance: float = 1.0e-10
    absolute_tolerance: float = 1.0e-13
    max_iterations: int = 20_000
    residual_history_stride: int = 1
    max_true_residual_restarts: int = 2

    def __post_init__(self) -> None:
        relative = _finite("relative_tolerance", self.relative_tolerance)
        absolute = _finite("absolute_tolerance", self.absolute_tolerance)
        if relative <= 0.0 or absolute < 0.0:
            raise FieldValidationError("solver tolerances must be positive/non-negative")
        if (
            not isinstance(self.max_iterations, int)
            or isinstance(self.max_iterations, bool)
            or self.max_iterations < 1
        ):
            raise FieldValidationError("max_iterations must be an integer >= 1")
        if (
            not isinstance(self.residual_history_stride, int)
            or isinstance(self.residual_history_stride, bool)
            or self.residual_history_stride < 1
        ):
            raise FieldValidationError("residual_history_stride must be an integer >= 1")
        if (
            not isinstance(self.max_true_residual_restarts, int)
            or isinstance(self.max_true_residual_restarts, bool)
            or self.max_true_residual_restarts < 0
        ):
            raise FieldValidationError(
                "max_true_residual_restarts must be an integer >= 0"
            )
        object.__setattr__(self, "relative_tolerance", relative)
        object.__setattr__(self, "absolute_tolerance", absolute)


@dataclass(frozen=True, slots=True)
class AxisymmetricProblem:
    """One constant-permeability L1a vacuum solve."""

    name: str
    domain: AxisymmetricDomain
    sources: tuple[AzimuthalCurrentBand, ...] = ()
    permeability_h_per_m: float = MU0_H_PER_M
    outer_boundary: str = "homogeneous_dirichlet_psi"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise FieldValidationError("problem name must not be empty")
        if not isinstance(self.domain, AxisymmetricDomain):
            raise FieldValidationError("domain must be an AxisymmetricDomain")
        permeability = _finite("permeability_h_per_m", self.permeability_h_per_m)
        if permeability <= 0.0:
            raise FieldValidationError("permeability_h_per_m must be positive")
        if self.outer_boundary != "homogeneous_dirichlet_psi":
            raise FieldValidationError(
                "L1a supports only homogeneous_dirichlet_psi outer boundaries"
            )
        if not isinstance(self.sources, tuple):
            raise FieldValidationError("sources must be an immutable tuple")
        for source in self.sources:
            if not isinstance(source, AzimuthalCurrentBand):
                raise FieldValidationError("every source must be an AzimuthalCurrentBand")
            if (
                source.r_outer_m >= self.domain.radius_m
                or source.z_min_m <= self.domain.z_min_m
                or source.z_max_m >= self.domain.z_max_m
            ):
                raise FieldValidationError("source bands must lie strictly inside the domain")
            support_r_min = 0.5 * self.domain.dr_m
            support_r_max = self.domain.radius_m - 0.5 * self.domain.dr_m
            support_z_min = self.domain.z_min_m + 0.5 * self.domain.dz_m
            support_z_max = self.domain.z_max_m - 0.5 * self.domain.dz_m
            if (
                source.r_inner_m < support_r_min
                or source.r_outer_m > support_r_max
                or source.z_min_m < support_z_min
                or source.z_max_m > support_z_max
            ):
                raise FieldValidationError(
                    f"source {source.name!r} must lie inside the interior dual-cell "
                    f"support r=[{support_r_min:.17g},{support_r_max:.17g}], "
                    f"z=[{support_z_min:.17g},{support_z_max:.17g}] m"
                )
            if not span_meets_minimum_grid_spacings(
                source.r_inner_m,
                source.r_outer_m,
                self.domain.dr_m,
            ):
                raise FieldValidationError(
                    f"source {source.name!r} radial thickness must span at least "
                    "two grid spacings; refine the grid or enlarge the band"
                )
            if not span_meets_minimum_grid_spacings(
                source.z_min_m,
                source.z_max_m,
                self.domain.dz_m,
            ):
                raise FieldValidationError(
                    f"source {source.name!r} axial thickness must span at least "
                    "two grid spacings; refine the grid or enlarge the band"
                )
        object.__setattr__(self, "permeability_h_per_m", permeability)


@dataclass(frozen=True, slots=True)
class SolverDiagnostics:
    converged: bool
    iterations: int
    initial_residual_l2: float
    final_residual_l2: float
    relative_residual_l2: float
    residual_history_l2: tuple[float, ...]
    max_flux_reconstruction_identity_t_per_m: float
    true_residual_restarts: int
    stagnation_detected: bool
    backend: str


@dataclass(frozen=True, slots=True)
class FieldMap:
    """Node-centred deterministic field map; first index is radial."""

    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    psi_wb: tuple[tuple[float, ...], ...]
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]
    diagnostics: SolverDiagnostics
    level: str = "L1a"
