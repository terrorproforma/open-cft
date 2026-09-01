"""Modern CFT/HEMP orchestration and validated numerical kernels."""

from .kernels import (
    calculate_performance,
    cusp_arrival_probabilities,
    cusp_arrival_probability,
    cusp_arrival_probability_python,
)
from .models import (
    CuspProbabilities,
    DesignPoint,
    LegacyPhysicsConstants,
    PerformanceResult,
)

__all__ = [
    "CuspProbabilities",
    "DesignPoint",
    "LegacyPhysicsConstants",
    "PerformanceResult",
    "calculate_performance",
    "cusp_arrival_probabilities",
    "cusp_arrival_probability",
    "cusp_arrival_probability_python",
]

__version__ = "0.1.0"
