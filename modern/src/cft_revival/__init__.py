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
from .optimization import Campaign as OptimizationCampaign
from .optimization import Design as OptimizationDesign
from .physics import XenonOperatingPoint as L0XenonOperatingPoint
from .physics import evaluate_performance as evaluate_l0_performance

__all__ = [
    "CuspProbabilities",
    "DesignPoint",
    "LegacyPhysicsConstants",
    "L0XenonOperatingPoint",
    "OptimizationCampaign",
    "OptimizationDesign",
    "PerformanceResult",
    "calculate_performance",
    "cusp_arrival_probabilities",
    "cusp_arrival_probability",
    "cusp_arrival_probability_python",
    "evaluate_l0_performance",
]

__version__ = "0.1.0"
