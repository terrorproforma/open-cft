"""L1b material-aware axisymmetric magnetic-field simulations."""

from .artifacts import (
    SCHEMA_VERSION,
    VIEWER_SCHEMA_VERSION,
    material_field_artifact,
    topology_descriptors,
    validate_artifact,
    validate_artifact_bundle,
    validate_viewer_contract,
    viewer_contract,
    write_json,
)
from .acceptance import assess_publication, raw_run_observation, study_metrics
from .models import (
    MaterialFieldConvergenceError,
    MaterialFieldError,
    MaterialFieldResult,
    MaterialFieldValidationError,
    MaterialSolveConfig,
    MaterialSolverDiagnostics,
    RasterDiagnostic,
    RasterizedMaterialProblem,
    WeakActionDiagnostic,
)
from .numerics import (
    apply_material_operator,
    assemble_rhs,
    material_operator_diagonal,
    minimum_operator_eigenvalue,
    solve_material_problem_cpu,
)
from .verification import interface_jump_residuals, max_result_difference, relative_field_l2
from .replay import ReplayReport, replay_raw_run
from .warp_solver import device_available, solve_material_problem_warp


def adapt_geometry(*args, **kwargs):
    """Lazily load geometry integration to keep the numerical core independent."""

    from .adapters import adapt_geometry as implementation

    return implementation(*args, **kwargs)


def design_domain(*args, **kwargs):
    """Lazily load the accepted geometry-domain adapter."""

    from .adapters import design_domain as implementation

    return implementation(*args, **kwargs)


def rasterize_handoff(*args, **kwargs):
    """Lazily load strict magnetics handoff rasterization."""

    from .adapters import rasterize_handoff as implementation

    return implementation(*args, **kwargs)


def raster_memory_preflight(*args, **kwargs):
    """Return the conservative host-memory rasterization bound."""

    from .adapters import raster_memory_preflight as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "SCHEMA_VERSION",
    "VIEWER_SCHEMA_VERSION",
    "MaterialFieldConvergenceError",
    "MaterialFieldError",
    "MaterialFieldResult",
    "MaterialFieldValidationError",
    "MaterialSolveConfig",
    "MaterialSolverDiagnostics",
    "RasterDiagnostic",
    "RasterizedMaterialProblem",
    "ReplayReport",
    "WeakActionDiagnostic",
    "adapt_geometry",
    "assess_publication",
    "raw_run_observation",
    "apply_material_operator",
    "assemble_rhs",
    "design_domain",
    "device_available",
    "interface_jump_residuals",
    "material_field_artifact",
    "material_operator_diagonal",
    "minimum_operator_eigenvalue",
    "max_result_difference",
    "rasterize_handoff",
    "raster_memory_preflight",
    "relative_field_l2",
    "replay_raw_run",
    "solve_material_problem_cpu",
    "solve_material_problem_warp",
    "study_metrics",
    "topology_descriptors",
    "validate_artifact",
    "validate_artifact_bundle",
    "validate_viewer_contract",
    "viewer_contract",
    "write_json",
]
