"""Modern CFT/HEMP orchestration and validated numerical kernels."""

from importlib import import_module
from types import ModuleType

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

_LAZY_PUBLIC_MODULES = frozenset(
    {
        "active_learning",
        "coupling",
        "fields",
        "hybrid",
        "magnetics",
        "pic",
        "plasma",
        "surrogates",
        "validation",
    }
)

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
    "active_learning",
    "coupling",
    "fields",
    "hybrid",
    "magnetics",
    "pic",
    "plasma",
    "surrogates",
    "validation",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> ModuleType:
    """Load accepted domain packages without importing optional stacks at startup."""

    if name not in _LAZY_PUBLIC_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module
