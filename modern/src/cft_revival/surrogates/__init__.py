"""Standalone surrogate-model runtime.

This package intentionally has no dependency on :mod:`cft_revival.optimization`.
"""

from ._linalg import numpy_available
from .benchmark import HeldoutBenchmarkReport, run_heldout_benchmark
from .gp import (
    DEFAULT_JITTER_POLICY,
    MODEL_SCHEMA_VERSION,
    ExactGP,
    FitDiagnostics,
    IndependentMultiOutputGP,
    Prediction,
    SurrogateSchema,
)
from .interop import (
    BoTorchTrainingData,
    OptionalInteropDependencyError,
    botorch_available,
)
from .multifidelity import (
    AR1Diagnostics,
    IndependentMultiOutputAR1,
    TwoFidelityAR1,
)
from .normalization import (
    InputNormalizer,
    OutputNormalizer,
    SurrogateError,
    SurrogateValidationError,
)
from .pod import (
    POD_SCHEMA_VERSION,
    FieldPrediction,
    PODBasis,
    PODFieldSurrogate,
    fixed_mesh_hash,
)
from .validation import (
    OODDetector,
    OODReport,
    RegressionMetrics,
    Split,
    VarianceCalibrator,
    grouped_spatial_split,
    regression_metrics,
)

__all__ = [
    "AR1Diagnostics",
    "BoTorchTrainingData",
    "DEFAULT_JITTER_POLICY",
    "ExactGP",
    "FieldPrediction",
    "FitDiagnostics",
    "HeldoutBenchmarkReport",
    "IndependentMultiOutputAR1",
    "IndependentMultiOutputGP",
    "InputNormalizer",
    "MODEL_SCHEMA_VERSION",
    "OODDetector",
    "OODReport",
    "OptionalInteropDependencyError",
    "OutputNormalizer",
    "POD_SCHEMA_VERSION",
    "PODBasis",
    "PODFieldSurrogate",
    "Prediction",
    "RegressionMetrics",
    "Split",
    "SurrogateSchema",
    "SurrogateError",
    "SurrogateValidationError",
    "TwoFidelityAR1",
    "VarianceCalibrator",
    "botorch_available",
    "fixed_mesh_hash",
    "grouped_spatial_split",
    "numpy_available",
    "regression_metrics",
    "run_heldout_benchmark",
]
