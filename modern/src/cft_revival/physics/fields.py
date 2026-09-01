"""Manufactured axisymmetric fields for future FEM verification gates.

These analytic fixtures are not a finite-element solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import frexp, isfinite, ldexp

from .models import PhysicsValidationError, _finite


def _scaled_half_product_t_m(field_t: float, radius_m: float) -> float:
    """Return ``field_t * radius_m / 2`` without premature range loss."""

    if field_t == 0.0 or radius_m == 0.0:
        return 0.0
    field_mantissa, field_exponent = frexp(field_t)
    radius_mantissa, radius_exponent = frexp(radius_m)
    try:
        result = ldexp(
            field_mantissa * radius_mantissa,
            field_exponent + radius_exponent - 1,
        )
    except OverflowError as error:
        raise PhysicsValidationError(
            "vector_potential_phi_t_m is not representable in binary64"
        ) from error
    if not isfinite(result):
        raise PhysicsValidationError(
            "vector_potential_phi_t_m is not representable in binary64"
        )
    return result


@dataclass(frozen=True, slots=True)
class AxisymmetricMagneticField:
    radial_t: float
    axial_t: float


@dataclass(frozen=True, slots=True)
class UniformAxialFieldFixture:
    """Uniform ``Bz=B0`` represented by regular ``A_phi=B0*r/2``."""

    b0_t: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "b0_t", _finite("b0_t", self.b0_t))

    def vector_potential_phi_t_m(self, radius_m: float, axial_m: float = 0.0) -> float:
        radius = _finite("radius_m", radius_m)
        _finite("axial_m", axial_m)
        if radius < 0.0:
            raise PhysicsValidationError("radius_m must be non-negative")
        return _scaled_half_product_t_m(self.b0_t, radius)

    def magnetic_field(
        self, radius_m: float, axial_m: float = 0.0
    ) -> AxisymmetricMagneticField:
        radius = _finite("radius_m", radius_m)
        _finite("axial_m", axial_m)
        if radius < 0.0:
            raise PhysicsValidationError("radius_m must be non-negative")
        return AxisymmetricMagneticField(radial_t=0.0, axial_t=self.b0_t)

    def axial_field_from_cylindrical_curl(
        self, radius_m: float, axial_m: float = 0.0
    ) -> float:
        """Evaluate ``Bz=(1/r)d(r*A_phi)/dr``, including its analytic axis limit."""

        radius = _finite("radius_m", radius_m)
        _finite("axial_m", axial_m)
        if radius < 0.0:
            raise PhysicsValidationError("radius_m must be non-negative")
        # r*A_phi = 0.5*B0*r^2, so its derivative is B0*r.
        # Return the simplified analytic result at every radius. Evaluating the
        # unsimplified quotient would add avoidable roundoff near the axis.
        return self.b0_t

    def axis_regularity_residual_t_m(self) -> float:
        """Return ``A_phi(0)``; regular axisymmetric fields require zero."""

        return self.vector_potential_phi_t_m(0.0)
