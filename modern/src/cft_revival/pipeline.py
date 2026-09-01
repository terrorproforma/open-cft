"""Top-level orchestration independent of concrete solver implementations."""

from __future__ import annotations

from .backends import MagneticFieldBackend, PlasmaBackend
from .kernels import (
    calculate_performance,
    cusp_arrival_probabilities,
    legacy_cusp_fields,
)
from .models import DesignPoint, LegacyPhysicsConstants, PerformanceResult


def evaluate_design(
    design: DesignPoint,
    generation: int,
    individual: int,
    magnetic_backend: MagneticFieldBackend,
    plasma_backend: PlasmaBackend,
    constants: LegacyPhysicsConstants = LegacyPhysicsConstants(),
) -> PerformanceResult:
    design.validate()
    fields = magnetic_backend.solve(design, generation, individual)
    low_field_t, high_field_t = legacy_cusp_fields(fields.centreline, fields.wall)
    probabilities = cusp_arrival_probabilities(low_field_t, high_field_t)
    plasma = plasma_backend.solve(design, probabilities)
    return calculate_performance(design, plasma, constants)
